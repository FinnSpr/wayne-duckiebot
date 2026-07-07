import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

import config


class State(Enum):
    DRIVE = 1
    STOP = 2
    FOLLOW = 3
    CROSS = 4
    TURN = 5
    BLOCKED = 6
    DUCKIE_AVOID = 7


@dataclass(frozen=True)
class Transition:
    from_state: State
    to_state: State
    condition: Callable[[], bool]
    action: Optional[Callable[[], None]] = None


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

        # Perception inputs (set by pipeline each frame before update_state)
        self.red_area = 0
        self.is_blocked = False
        self.duckie_nearby = False

        self.decision_waypoint = None
        self.decision = None
        self.intersection_admitted = False

        self.left_ticks_before_relevant_state = 0
        self.right_ticks_before_relevant_state = 0
        self.left_ticks = 0
        self.right_ticks = 0

        self._transitions = [
            Transition(State.DRIVE, State.STOP, condition=self._should_stop),
            Transition(State.DRIVE, State.TURN, condition=self._should_turn),
            # Transition(State.DRIVE, State.DUCKIE_AVOID, condition=self._duckie_too_close),
            Transition(State.STOP, State.FOLLOW, condition=self._stop_elapsed),
            Transition(State.FOLLOW, State.CROSS, condition=self._follow_finished),
            Transition(State.CROSS, State.DRIVE, condition=self._cross_elapsed),
            Transition(State.TURN, State.DRIVE, condition=self._turn_finished),
            # Transition(State.DUCKIE_AVOID, State.DRIVE, condition=self._duckie_clear),
        ]

    def update_state(self):
        """Evaluate FSM transitions — call once per frame after feeding perception inputs."""
        # ── BLOCKED (special: from any state, remembers prev_state) ──
        if self.is_blocked and self.state != State.BLOCKED:
            self.prev_state = self.state
            self.change_state(State.BLOCKED)
            return

        if self.state == State.BLOCKED:
            if self.blocked_state_last_time is not None:
                self.remained_in_blocked_state += (
                    time.time() - self.blocked_state_last_time
                )
            if not self.is_blocked:
                self.state = self.prev_state
                print(self.state)
            return

        # ── Normal transitions ──
        for t in self._transitions:
            if t.from_state is self.state and t.condition():
                self.change_state(t.to_state)
                if t.action:
                    t.action()
                break

    def change_state(self, new_state: State):
        self.state = new_state
        self.state_entered_at = time.time()
        self.intersection_admitted = False
        self.remained_in_blocked_state = 0.0
        if new_state == State.FOLLOW:
            self.decision_waypoint = None  # force fresh intersection choice
            self.decision = None
        if new_state == State.DUCKIE_AVOID:
            self._duckie_last_seen_at = time.time()
        print(self.state)

    def time_passed(self, duration: float) -> bool:
        return (
            time.time() - self.state_entered_at
            >= duration + self.remained_in_blocked_state
        )

    def no_waypoint_passed(self, duration: float) -> bool:
        return time.time() - self.time_last_waypoint >= duration

    def distance_passed(self, distance: float):
        delta_ticks_left = self.left_ticks - self.left_ticks_before_relevant_state
        delta_ticks_right = self.right_ticks - self.right_ticks_before_relevant_state
        rotation_wheel_left = (
            delta_ticks_left * config.ALPHA_WHEEL
        )  # calculate total rotation of left wheel
        rotation_wheel_right = (
            delta_ticks_right * config.ALPHA_WHEEL
        )  # calculate total rotation of right wheel
        d_left = config.WHEEL_RADIUS * rotation_wheel_left
        d_right = config.WHEEL_RADIUS * rotation_wheel_right
        d_A = (d_left + d_right) / 2
        return d_A >= distance

    # Transition conditions

    def _should_stop(self) -> bool:
        return self.red_area >= config.MIN_AREA

    def _should_turn(self) -> bool:
        return self.no_waypoint_passed(config.WAIT_UNTIL_TURN_TIME)

    def _stop_elapsed(self) -> bool:
        return self.time_passed(config.STOP_TIME)

    def _follow_finished(self) -> bool:
        follow_distance_map = {
            "left": config.FOLLOW_DISTANCE[0],
            "straight": config.FOLLOW_DISTANCE[1],
        }
        d = follow_distance_map.get(self.decision, config.FOLLOW_DISTANCE[2])
        follow_time_map = {
            "left": config.FOLLOW_TIME[0],
            "straight": config.FOLLOW_TIME[1],
        }
        t = follow_time_map.get(self.decision, config.FOLLOW_TIME[2])
        if config.USE_WHEEL_ODOMETRY:
            return self.distance_passed(d)
        return self.time_passed(t)

    def _cross_elapsed(self) -> bool:
        return self.time_passed(config.CROSS_TIME)

    def _turn_finished(self) -> bool:
        if config.USE_WHEEL_ODOMETRY:
            return self.distance_passed(config.TURN_DISTANCE)
        return self.time_passed(config.TURN_TIME)

    def _duckie_too_close(self) -> bool:
        return self.duckie_nearby

    def _duckie_clear(self) -> bool:
        """Exit DUCKIE_AVOID after no nearby duckie for a cooldown period."""
        if self.duckie_nearby:
            self._duckie_last_seen_at = time.time()
            return False
        return time.time() - self._duckie_last_seen_at >= getattr(
            config, "DUCKIE_AVOID_CLEAR_TIME", 2.0
        )

    def get_intersection_waypoint(
        self, red_lines: List[Tuple], image_width: int, image_height: int
    ) -> Optional[np.ndarray]:
        """Pick an intersection destination and return its waypoint.

        Called once when entering FOLLOW state. Analyses red stop lines to
        determine possible paths (straight / left / right), picks one at
        random, and returns the target waypoint in image coordinates.
        """
        other_lines = [l for l in red_lines if l[1] < config.CUT_FRONT_STOP_LINE]
        if not other_lines:
            return None

        v_left = [
            l for l in other_lines if l[2] == "vertical" and l[0] < config.LEFT_VS_RIGHT
        ]
        v_right = [
            l
            for l in other_lines
            if l[2] == "vertical" and l[0] >= config.LEFT_VS_RIGHT
        ]
        horiz = [l for l in other_lines if l[2] == "horizontal"]

        choices = {}
        for key, lst in [("straight", horiz), ("left", v_left), ("right", v_right)]:
            if len(lst) == 1:
                choices[key] = lst[0]
        if not choices:
            for key, lst in [
                ("straight", horiz),
                ("left", v_left),
                ("right", v_right),
            ]:
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
            "right": config.CROSSING_OFFSET_RIGHT,
        }
        target = np.array(chosen[:2]) + offsets[self.decision]

        print(f"Decision: {self.decision}\n")
        x = int(np.clip(target[0], 0, image_width))
        y = int(np.clip(target[1], 0, image_height))

        self.decision_waypoint = np.array([[x, y]])
        return self.decision_waypoint

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
            if right_occupied:
                print("Waiting for right duckiebot...")
            if top_occupied:
                print("Waiting for top duckiebot...")
            return not (right_occupied or top_occupied)
        if self.decision == "straight":
            if right_occupied:
                print("Waiting for right duckiebot...")
            return not right_occupied
        return True

    def set_ticks(self, ticks_left, ticks_right):
        if self.state in (State.FOLLOW, State.TURN, State.DUCKIE_AVOID):
            self.left_ticks = ticks_left
            self.right_ticks = ticks_right
        else:
            self.left_ticks_before_relevant_state = ticks_left
            self.right_ticks_before_relevant_state = ticks_right
