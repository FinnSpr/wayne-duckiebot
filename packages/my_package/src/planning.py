import time
import random
from enum import Enum
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

    def update_state(self, red_mask: np.ndarray):
        """Update FSM transitions based on perception input (red stop line mask)."""
        if self.state == State.DRIVE:
            if self.no_waypoint_passed(config.WAIT_UNTIL_TURN_TIME):
                self.change_state(State.TURN)
            elif np.sum(red_mask[config.STOP_MARKER_Y:, :] > 0) >= config.MIN_AREA:
                self.change_state(State.STOP)
        elif self.state == State.STOP:
            if self.time_passed(config.STOP_TIME):
                self.change_state(State.FOLLOW)
        elif self.state == State.FOLLOW:
            t = 0
            if self.decision == "left":
                t = config.FOLLOW_TIME[0]
            elif self.decision == "straight":
                t = config.FOLLOW_TIME[1]
            else:
                t = config.FOLLOW_TIME[2]
            if self.time_passed(t):
                self.change_state(State.CROSS)
        elif self.state == State.CROSS:
            if self.time_passed(config.CROSS_TIME):
                self.change_state(State.DRIVE)
        elif self.state == State.TURN:
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
            r = True
            if right_occupied:
                print("Waiting for right duckiebot...")
                r = False
            if top_occupied:
                print("Waiting for top duckiebot...")
                r = False
            return r
        elif self.decision == "straight":
            if right_occupied:
                print("Waiting for right duckiebot...")
            return not right_occupied
        else:
            return True

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
                    vertical_other_lines_left = [
                        l for l in other_lines if l[2] == "vertical" and l[0] < config.LEFT_VS_RIGHT
                    ]
                    vertical_other_lines_right = [
                        l for l in other_lines if l[2] == "vertical" and l[0] >= config.LEFT_VS_RIGHT
                    ]
                    horizontal_other_lines = [l for l in other_lines if l[2] == "horizontal"]

                    choices = {}
                    if len(horizontal_other_lines) == 1:
                        choices["straight"] = horizontal_other_lines[0]
                    if len(vertical_other_lines_left) == 1:
                        choices["left"] = vertical_other_lines_left[0]
                    if len(vertical_other_lines_right) == 1:
                        choices["right"] = vertical_other_lines_right[0]

                    if not choices:
                        if len(horizontal_other_lines) > 1:
                            choices["straight"] = horizontal_other_lines[0]
                        if len(vertical_other_lines_left) > 1:
                            choices["left"] = vertical_other_lines_left[0]
                        if len(vertical_other_lines_right) > 1:
                            choices["right"] = vertical_other_lines_right[0]

                    print("\n------Possible Destinations------")
                    for key, value in choices.items():
                        print(f"{key}: {value}")
                    print("")

                    if not choices:
                        self.decision = "straight"
                        target = np.array([image_width // 2, image_height // 2]) + config.CROSSING_OFFSET_TOP
                    else:
                        self.decision, chosen = random.choice(list(choices.items()))
                        if self.decision == "straight":
                            target = np.array(chosen[:2]) + config.CROSSING_OFFSET_TOP
                        elif self.decision == "left":
                            target = np.array(chosen[:2]) + config.CROSSING_OFFSET_LEFT
                        elif self.decision == "right":
                            target = np.array(chosen[:2]) + config.CROSSING_OFFSET_RIGHT

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

                # Average the two splines pointwise to get center points
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
                wx, wy = white_spline
                if config.ENHANCED_LANE_DETECTION:
                    t_x = wx / image_width
                    t_y = 1 - wy / image_height
                    center_x = wx - wx * t_x * config.SINGLE_LANE_SCALE_FACTOR_WHITE
                    center_y = wy + (image_height - wy) * t_y * config.SINGLE_LANE_SCALE_FACTOR_WHITE
                    waypoints = np.column_stack([center_x, center_y])
                else:
                    offset = image_width * config.IMAGE_WIDTH_OFFSET_FACTOR_WHITE
                    waypoints = np.column_stack([wx - offset, wy])

            elif yellow_spline is not None:
                yx, yy = yellow_spline
                if config.ENHANCED_LANE_DETECTION:
                    t_x = 1 - yx / image_width
                    t_y = 1 - yy / image_height
                    center_x = yx + (image_width - yx) * t_x * config.SINGLE_LANE_SCALE_FACTOR_YELLOW
                    center_y = yy + (image_height - yy) * t_y * config.SINGLE_LANE_SCALE_FACTOR_YELLOW
                    waypoints = np.column_stack([center_x, center_y])
                else:
                    offset = image_width * config.IMAGE_WIDTH_OFFSET_FACTOR_YELLOW
                    waypoints = np.column_stack([yx + offset, yy])
            else:
                waypoints = None

        return waypoints
