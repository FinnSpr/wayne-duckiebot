import time
import random
from enum import Enum
from typing import Tuple, Optional
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
import config

class State(Enum):
    DRIVE = 1
    STOP = 2
    FOLLOW = 3
    CROSS = 4
    TURN = 5
    BLOCKED = 6

class BehaviorPlanner:
    """
    Planning Module.
    Manages the Finite State Machine (FSM), state transitions,
    intersection logic, and trajectory waypoint generation.
    """
    def __init__(self):
        self.state = State.DRIVE
        self.state_entered_at = time.time()
        self.time_last_waypoint = time.time()
        self.prev_state = State.DRIVE
        self.blocked_state_last_time = None
        self.remained_in_blocked_state = 0.0

        self.crossing_decision = False
        self.decision_waypoint = None
        self.decision = None
        self.intersection_admitted = False
        self.crossing_vel_left = 0.0
        self.crossing_vel_right = 0.0

        self.left_ticks_before_relevant_state = 0
        self.right_ticks_before_relevant_state = 0
        self.left_ticks = 0
        self.right_ticks = 0

    def change_state(self, new_state: State):
        self.state = new_state
        self.state_entered_at = time.time()
        self.crossing_decision = False
        self.intersection_admitted = False
        self.remained_in_blocked_state = 0.0
        print(self.state)

    def time_passed(self, duration: float) -> bool:
        return time.time() - self.state_entered_at >= duration + self.remained_in_blocked_state

    def no_waypoint_passed(self, duration: float) -> bool:
        return time.time() - self.time_last_waypoint >= duration

    def distance_passed(self, distance: float):
        delta_ticks_left = self.left_ticks - self.left_ticks_before_relevant_state
        delta_ticks_right = self.right_ticks - self.right_ticks_before_relevant_state
        rotation_wheel_left = delta_ticks_left * config.ALPHA_WHEEL  # calculate total rotation of left wheel 
        rotation_wheel_right = delta_ticks_right * config.ALPHA_WHEEL # calculate total rotation of right wheel 
        d_left = config.WHEEL_RADIUS * rotation_wheel_left
        d_right = config.WHEEL_RADIUS * rotation_wheel_right
        d_A = (d_left + d_right) / 2
        return d_A >= distance

    def update_state(self, red_mask: np.ndarray):
        """Update FSM transitions based on perception input (red stop line mask)."""
        if self.state == State.DRIVE:
            if self.no_waypoint_passed(config.WAIT_UNTIL_TURN_TIME):
                self.change_state(State.TURN)
            if np.sum(red_mask[config.STOP_MARKER_Y:, :] > 0) >= config.MIN_AREA:
                self.change_state(State.STOP)
        if self.state == State.STOP:
            if self.time_passed(config.STOP_TIME):
                self.change_state(State.FOLLOW)
        if self.state == State.FOLLOW:
            follow_distance_map = {"left": config.FOLLOW_DISTANCE[0], "straight": config.FOLLOW_DISTANCE[1]}
            d = follow_distance_map.get(self.decision, config.FOLLOW_DISTANCE[2])
            follow_time_map = {"left": config.FOLLOW_TIME[0], "straight": config.FOLLOW_TIME[1]}
            t = follow_time_map.get(self.decision, config.FOLLOW_TIME[2])
            if config.USE_WHEEL_ODOMETRY:
                if self.distance_passed(d):
                    self.change_state(State.CROSS)
            else:
                if self.time_passed(t):
                    self.change_state(State.CROSS)
        if self.state == State.CROSS:
            if self.time_passed(config.CROSS_TIME):
                self.change_state(State.DRIVE)
        if self.state == State.TURN:
            if config.USE_WHEEL_ODOMETRY:
                if self.distance_passed(config.TURN_DISTANCE):
                    self.change_state(State.DRIVE)
            else:
                if self.time_passed(config.TURN_TIME):
                    self.change_state(State.DRIVE)

    def handle_blocking(self, is_blocked: bool):
        """Handle blocking/obstacle state transitions."""
        if self.state != State.BLOCKED:
            if is_blocked:
                self.prev_state = self.state
                self.state = State.BLOCKED
                print(self.state)
        elif self.state == State.BLOCKED:
            if self.blocked_state_last_time is not None:
                self.remained_in_blocked_state += time.time() - self.blocked_state_last_time
            if not is_blocked:
                self.state = self.prev_state
                print(self.state)

    def can_intersect(self, image: np.ndarray, red_lines: list) -> bool:
        """Determine whether the robot can proceed at the intersection (traffic rules/checks)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, config.BLUE_HSV_LOWER, config.BLUE_HSV_UPPER)
        h, w = mask.shape

        other_lines = [l for l in red_lines if l[1] < config.CUT_FRONT_STOP_LINE]
        if not other_lines:
            return True

        proximity_mask = np.zeros((h, w), dtype=np.uint8)
        for line in other_lines:
            cx, cy = line[0], line[1]
            cv2.circle(
                proximity_mask,
                (cx, cy),
                config.PROXIMITY_OTHER_VEHICLES_TO_RED_LINE,
                255,
                thickness=-1,
            )

        nearby_blue = cv2.bitwise_and(mask, proximity_mask)
        top_occupied = bool(np.any(nearby_blue[: h // 2, 50:400]))
        left_occupied = bool(np.any(nearby_blue[100:, : w // 3]))
        right_occupied = bool(np.any(nearby_blue[:300, w // 2 :]))

        if self.decision == "left":
            if right_occupied: print("Waiting for right duckiebot...")
            if top_occupied: print("Waiting for top duckiebot...")
            return not (right_occupied or top_occupied)
        if self.decision == "straight":
            if right_occupied: print("Waiting for right duckiebot...")
            return not right_occupied
        return True

    def _project_single_spline(
        self,
        spline: Tuple[np.ndarray, np.ndarray],
        image_width: int,
        image_height: int,
        is_white: bool
    ) -> np.ndarray:
        """Project waypoints using a single lane spline estimation."""
        sx, sy = spline
        if config.ENHANCED_LANE_DETECTION:
            if is_white:
                t_x = sx / image_width
                scale = config.SINGLE_LANE_SCALE_FACTOR_WHITE
                center_x = sx - sx * t_x * scale
            else:
                t_x = 1 - sx / image_width
                scale = config.SINGLE_LANE_SCALE_FACTOR_YELLOW
                center_x = sx + (image_width - sx) * t_x * scale

            t_y = 1 - sy / image_height
            center_y = sy + (image_height - sy) * t_y * scale
            return np.column_stack([center_x, center_y])
        else:
            factor = config.IMAGE_WIDTH_OFFSET_FACTOR_WHITE if is_white else config.IMAGE_WIDTH_OFFSET_FACTOR_YELLOW
            offset = image_width * factor
            return np.column_stack([sx - offset if is_white else sx + offset, sy])

    def compute_waypoints(
        self,
        white_spline: Optional[Tuple[np.ndarray, np.ndarray]],
        yellow_spline: Optional[Tuple[np.ndarray, np.ndarray]],
        red_lines: list,
        image_width: int,
        image_height: int,
    ) -> Optional[np.ndarray]:
        """Compute path waypoints based on lane splines or intersection stop lines."""
        point_to_stop_line = False

        if red_lines:
            point_to_stop_line = True
            if self.state == State.FOLLOW:
                other_lines = [l for l in red_lines if l[1] < config.CUT_FRONT_STOP_LINE]
                if not other_lines:
                    waypoints = None
                else:
                    v_left = [l for l in other_lines if l[2] == "vertical" and l[0] < config.LEFT_VS_RIGHT]
                    v_right = [l for l in other_lines if l[2] == "vertical" and l[0] >= config.LEFT_VS_RIGHT]
                    horiz = [l for l in other_lines if l[2] == "horizontal"]

                    # Check for exactly 1 matching line first, then fall back to >1 matching lines
                    choices = {}
                    for key, lst in [("straight", horiz), ("left", v_left), ("right", v_right)]:
                        if len(lst) == 1:
                            choices[key] = lst[0]
                    if not choices:
                        for key, lst in [("straight", horiz), ("left", v_left), ("right", v_right)]:
                            if len(lst) > 1:
                                choices[key] = lst[0]

                    print("\n------Possible Destinations------")
                    for key, value in choices.items():
                        print(f"{key}: {value}")
                    print("")

                    if not choices:
                        self.decision = "straight"
                        chosen = [image_width // 2, image_height // 2]
                    else:
                        self.decision, chosen = random.choice(list(choices.items()))

                    offsets = {
                        "straight": config.CROSSING_OFFSET_TOP,
                        "left": config.CROSSING_OFFSET_LEFT,
                        "right": config.CROSSING_OFFSET_RIGHT
                    }
                    target = np.array(chosen[:2]) + offsets[self.decision]

                    print(f"Decision: {self.decision}\n")
                    x = int(np.clip(target[0], 0, image_width))
                    y = int(np.clip(target[1], 0, image_height))

                    self.crossing_decision = True
                    self.decision_waypoint = np.array([[x, y]])
                    waypoints = self.decision_waypoint
            elif self.state == State.CROSS:
                point_to_stop_line = False
            else:
                current_line = max(
                    [l for l in red_lines if l[2] == "horizontal"],
                    key=lambda l: l[1],
                    default=(int(image_width / 2), 0),
                )
                waypoints = np.array([[current_line[0], current_line[1]]])

        if not point_to_stop_line:
            if white_spline is not None and yellow_spline is not None:
                wx, wy = white_spline
                yx, yy = yellow_spline
                center_x = (wx + yx) / 2.0
                center_y = (wy + yy) / 2.0

                if config.ENHANCED_LANE_DETECTION:
                    waypoints = np.column_stack([center_x, center_y])
                else:
                    try:
                        tck, _ = splprep([center_x, center_y], s=999999, k=3)
                        u_fine = np.linspace(0, 1, config.N_WAYPOINTS)
                        cx_spline, cy_spline = splev(u_fine, tck)
                        waypoints = np.column_stack([cx_spline, cy_spline])
                    except Exception:
                        waypoints = np.column_stack([center_x, center_y])
            elif white_spline is not None:
                waypoints = self._project_single_spline(white_spline, image_width, image_height, is_white=True)
            elif yellow_spline is not None:
                waypoints = self._project_single_spline(yellow_spline, image_width, image_height, is_white=False)
            else:
                waypoints = None

        return waypoints

    def set_ticks(self, ticks_left, ticks_right):
        if self.state == State.FOLLOW or self.state == State.TURN:
            self.left_ticks = ticks_left
            self.right_ticks = ticks_right
        else:
            self.left_ticks_before_relevant_state = ticks_left
            self.right_ticks_before_relevant_state = ticks_right
