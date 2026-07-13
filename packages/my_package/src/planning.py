import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import config
import cv2
import numpy as np


class State(Enum):
    DRIVE = 1
    STOP = 2
    FOLLOW = 3
    CROSS = 4
    TURN = 5
    WAIT_FOR_INSTRUCTION = 6
    DUCKIE_AVOID = 7


@dataclass(frozen=True)
class Transition:
    from_state: State
    to_state: State
    condition: Callable[[], bool]


class BehaviorPlanner:
    """
    Planning Module.
    Manages the Finite State Machine (FSM), state transitions,
    intersection logic, and trajectory waypoint generation.
    """

    def __init__(self):
        self.state = State.DRIVE
        self.state_entered_at = None
        self.time_last_waypoint = None
        self.prev_state = State.DRIVE

        # Perception inputs (set by pipeline each frame before update_state)
        self.stop_line_area = 0
        self.duckie_in_roi = False

        self._intersection_decisions = deque()
        self._arrived = False

        self.decision_waypoint = None
        self.decision = None
        self.intersection_admitted = False
        self.intersection_speed = None

        self.left_ticks_before_relevant_state = 0
        self.right_ticks_before_relevant_state = 0
        self.left_ticks = 0
        self.right_ticks = 0

        self._transitions = [
            Transition(State.DRIVE, State.STOP, condition=self._should_stop),
            Transition(State.DRIVE, State.TURN, condition=self._should_turn),
            Transition(
                State.STOP, State.FOLLOW, condition=self._stop_elapsed_have_instructions
            ),
            Transition(State.FOLLOW, State.CROSS, condition=self._follow_finished),
            Transition(State.CROSS, State.DRIVE, condition=self._cross_elapsed),
            Transition(State.TURN, State.DRIVE, condition=self._turn_finished),
            Transition(
                State.DRIVE,
                State.WAIT_FOR_INSTRUCTION,
                condition=self._arrived_at_finish,
            ),
            Transition(
                State.WAIT_FOR_INSTRUCTION,
                State.DRIVE,
                condition=self._has_instructions,
            ),
            Transition(State.DRIVE, State.DUCKIE_AVOID, condition=self._duckie_in_roi),
            Transition(
                State.DUCKIE_AVOID, State.DRIVE, condition=self._no_duckies_in_roi
            ),
        ]

    def construct_timers_if_needed(self):
        """Constructs timers for the FSM if they haven't been initialized yet."""
        if self.state_entered_at is None:
            self.state_entered_at = time.time()
        if self.time_last_waypoint is None:
            self.time_last_waypoint = time.time()

    def set_intersection_decisions(self, decisions: list) -> None:
        self._intersection_decisions = deque(decisions)

    def set_arrived(self, arrived: bool) -> None:
        self._arrived = arrived

    def update_state(self):
        """Evaluate FSM transitions — call once per frame after feeding perception inputs."""
        # Transitions
        for t in self._transitions:
            if t.from_state is self.state and t.condition():
                self.change_state(t.to_state)
                break

    def change_state(self, new_state: State):
        self.prev_state = self.state
        self.state = new_state
        self.state_entered_at = time.time()
        self.intersection_admitted = False
        if new_state == State.FOLLOW:
            self.decision_waypoint = None  # force fresh intersection choice
            self.decision = None
        print(self.state)

    def time_passed(self, duration: float) -> bool:
        return time.time() - self.state_entered_at >= duration

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
    def _has_instructions(self) -> bool:
        return len(self._intersection_decisions) > 0

    def _arrived_at_finish(self) -> bool:
        return self._arrived

    def _should_stop(self) -> bool:
        return self.stop_line_area >= config.MIN_AREA

    def _should_turn(self) -> bool:
        return self.no_waypoint_passed(config.WAIT_UNTIL_TURN_TIME)

    def _stop_elapsed_have_instructions(self) -> bool:
        return self.time_passed(config.STOP_TIME) and self._has_instructions()

    def _follow_finished(self) -> bool:
        if not self.decision:
            return False
        d = config.FOLLOW_DISTANCE[self.decision]
        t = config.FOLLOW_TIME[self.decision]
        if config.USE_WHEEL_ODOMETRY:
            return self.distance_passed(d)
        return self.time_passed(t)

    def _cross_elapsed(self) -> bool:
        return self.time_passed(config.CROSS_TIME)

    def _turn_finished(self) -> bool:
        if config.USE_WHEEL_ODOMETRY:
            return self.distance_passed(config.TURN_DISTANCE)
        return self.time_passed(config.TURN_TIME)

    def _no_duckies_in_roi(self) -> bool:
        return not self.duckie_in_roi

    def _duckie_in_roi(self) -> bool:
        return self.duckie_in_roi

    def get_intersection_waypoint(self) -> Optional[np.ndarray]:
        """Get the waypoint for the current intersection decision."""
        if self.decision_waypoint is not None:
            return self.decision_waypoint

        if not self._intersection_decisions:
            print("No more intersection decisions available.")
            return None

        self.decision = self._intersection_decisions.popleft()
        self.decision_waypoint = config.CROSSING_OFFSET[self.decision]
        return self.decision_waypoint

    def can_intersect(self, image: np.ndarray, red_lines: list) -> bool:
        """Determine whether the robot can proceed at the intersection (traffic rules/checks)."""
        # TODO: better logic, not just blue color
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
