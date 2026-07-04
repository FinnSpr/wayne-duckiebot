#!/usr/bin/env python3
"""
Modular lane-following pipeline wrapper for Duckiebot.
Acts as backward-compatible entry point.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
import time

import config
from perception import PerceptionModule
from world_model import WorldModel
from planning import BehaviorPlanner, State
from control import Controller
from visualizer import Visualizer


def get_modes():
    return config.get_modes()


class SelfDrivingPipeline:
    def __init__(self):
        self.perception = PerceptionModule(use_object_detection=config.OBJECT_DETECTION)
        self.world_model = WorldModel()
        self.planner = BehaviorPlanner()
        self.controller = Controller()
        self.visualizer = Visualizer()

    def process(
        self, data
    ) -> Tuple[
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        # Determine image
        if config.ENHANCED_LANE_DETECTION:
            if hasattr(data, "_unwarped_image") and data._unwarped_image is not None:
                image = data._unwarped_image
            else:
                image = data._image
            image_height, image_width = image.shape[:2]
            white_lane_mask, yellow_mask, red_mask, edge_mask, white_color = (
                self.perception.filter_lane_colors_enhanced(image)
            )
        else:
            image = data._image
            image_height, image_width = image.shape[:2]
            white_lane_mask, yellow_mask, red_mask = (
                self.perception.filter_lane_colors_standard(image)
            )
            white_color = white_lane_mask
            edge_mask = None

        tof = data._tof if hasattr(data, "_tof") else None

        # Check for obstacles (Object Detection)
        object_detected = self.perception.check_obstacle(image)

        # Apply Region of Interest (ROI) mask in DRIVE or CROSS (with enhanced) state
        if self.planner.state == State.DRIVE or (
            self.planner.state == State.CROSS and config.ENHANCED_LANE_DETECTION
        ):
            white_lane_mask[: config.HIDE_TOP_OF_IMAGE, :] = 0
            yellow_mask[: config.HIDE_TOP_OF_IMAGE, :] = 0
            red_mask[: config.HIDE_TOP_OF_IMAGE, :] = 0

        self.planner.set_ticks(data._left_encoder, data._right_encoder)
        # Update FSM transitions
        self.planner.update_state(red_mask)

        if config.ENHANCED_LANE_DETECTION and self.planner.state == State.STOP:
            raw_hsv = cv2.cvtColor(data._image, cv2.COLOR_BGR2HSV)
            red_mask = self.perception.filter_red(raw_hsv)

        # Modeling lanes via Spline fitting
        white_spline = self.world_model.fit_spline(
            white_lane_mask, take_leftmost_pixels=False
        )
        yellow_spline = self.world_model.fit_spline(
            yellow_mask, take_leftmost_pixels=True
        )

        # Waypoint computation
        if not self.planner.crossing_decision:
            if self.planner.state == State.TURN:
                visualization = self.visualizer.visualize(
                    image,
                    white_lane_mask,
                    yellow_mask,
                    red_mask,
                    white_spline,
                    yellow_spline,
                    None,
                )
                self.planner.time_last_waypoint = time.time()
                if config.USE_TWIST:
                    return (
                        0.0,
                        config.TURN_OMEGA,
                        visualization,
                        edge_mask,
                        white_lane_mask,
                        yellow_mask,
                        red_mask,
                        white_color,
                    )
                else:
                    return (
                        0.0,
                        config.TURN_SPEED_RIGHT_WHEEL,
                        visualization,
                        edge_mask,
                        white_lane_mask,
                        yellow_mask,
                        red_mask,
                        white_color,
                    )

            # Extract stop lines as structured entities
            red_lines = self.world_model.extract_red_lines(red_mask)

            waypoints = self.planner.compute_waypoints(
                white_spline, yellow_spline, red_lines, image_width, image_height
            )

            if waypoints is None:
                visualization = self.visualizer.visualize(
                    image,
                    white_lane_mask,
                    yellow_mask,
                    red_mask,
                    white_spline,
                    yellow_spline,
                    None,
                )
                return (
                    0.0,
                    0.0,
                    visualization,
                    edge_mask,
                    white_lane_mask,
                    yellow_mask,
                    red_mask,
                    white_color,
                )

            # Control: Estimate heading error
            heading_error = self.controller.estimate_heading_error(
                waypoints, image_width, image_height
            )
            # Control: Calculate velocities
            is_stopped = self.planner.state == State.STOP
            if config.USE_TWIST:
                v, omega = self.controller.heading_to_twist(heading_error, is_stopped)
            else:
                vel_left, vel_right = self.controller.heading_to_wheel_commands(
                    heading_error, is_stopped
                )

            if config.USE_TWIST:
                # Store computed velocities in case we transition to crossing decision
                self.planner.crossing_vel = v
                self.planner.crossing_omega = omega
            else:
                self.planner.crossing_vel_left = vel_left
                self.planner.crossing_vel_right = vel_right

        # Handling Crossing Decision (from Behavior Planner FSM state: State.FOLLOW)
        if self.planner.crossing_decision:
            red_lines = self.world_model.extract_red_lines(red_mask)
            if not self.planner.intersection_admitted:
                self.planner.intersection_admitted = self.planner.can_intersect(
                    image, red_lines
                )

            if self.planner.intersection_admitted:
                if config.USE_TWIST:
                    v = self.planner.crossing_vel
                    omega = self.planner.crossing_omega
                else:
                    vel_left = self.planner.crossing_vel_left
                    vel_right = self.planner.crossing_vel_right
                waypoints = self.planner.decision_waypoint
            else:
                self.planner.state_entered_at = time.time()
                vel_left = 0.0
                vel_right = 0.0
                v = 0.0
                omega = 0.0
                waypoints = self.planner.decision_waypoint

        self.planner.time_last_waypoint = time.time()

        # Handle Blocking State (Collision Avoidance / Stop for obstacles)
        is_blocked = False
        if tof is not None and config.TOF_THRESHOLD > 0.0:
            if tof < config.TOF_THRESHOLD:
                is_blocked = True
        if object_detected:
            is_blocked = True

        self.planner.handle_blocking(is_blocked)

        if self.planner.state == State.BLOCKED:
            vel_left = 0.0
            vel_right = 0.0
            v = 0
            omega = 0

        # Visualization
        visualization = self.visualizer.visualize(
            image,
            white_lane_mask,
            yellow_mask,
            red_mask,
            white_spline,
            yellow_spline,
            waypoints,
        )

        self.planner.blocked_state_last_time = time.time()

        if config.USE_TWIST:
            return (
                v,
                omega,
                visualization,
                edge_mask,
                white_lane_mask,
                yellow_mask,
                red_mask,
                white_color,
            )
        else:
            return (
                vel_left,
                vel_right,
                visualization,
                edge_mask,
                white_lane_mask,
                yellow_mask,
                red_mask,
                white_color,
            )


# Persistent singleton instance for ROS callback persistence
_pipeline = SelfDrivingPipeline()


def process_all(
    data,
) -> Tuple[
    float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    return _pipeline.process(data)
