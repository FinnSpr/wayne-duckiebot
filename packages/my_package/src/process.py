#!/usr/bin/env python3
"""
Modular lane-following pipeline wrapper for Duckiebot.
Acts as backward-compatible entry point.
"""

import time

import config
import cv2
import numpy as np
import image_utils
from control import Controller
from perception import PerceptionModule
from planning import BehaviorPlanner, State
from visualizer import Visualizer
from world_model import WorldModel


class SelfDrivingPipeline:
    _COLOR_VIS_NAMES = ("visualization", "unwarped_image", "image")
    _BW_VIS_NAMES = (
        "edge_mask",
        "white_lane_mask",
        "yellow_mask",
        "red_mask",
        "white_color",
        "bev_mask",
    )

    def __init__(
        self,
        K: np.ndarray,
        D: np.ndarray,
        P: np.ndarray,
        H: np.ndarray,
        bev_config: image_utils.BEVConfig,
    ):
        self._K = K
        self._D = D
        self._P = P
        self._H = H
        self.bev_config = bev_config
        self.perception = PerceptionModule(use_object_detection=config.OBJECT_DETECTION)
        self.world_model = WorldModel()
        self.planner = BehaviorPlanner()
        self.controller = Controller()
        self.visualizer = Visualizer()

    @staticmethod
    def _build_vis_dicts(
        local_vars: dict,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        color = {
            n: local_vars[n]
            for n in SelfDrivingPipeline._COLOR_VIS_NAMES
            if n in local_vars and local_vars[n] is not None
        }
        bw = {
            n: local_vars[n]
            for n in SelfDrivingPipeline._BW_VIS_NAMES
            if n in local_vars and local_vars[n] is not None
        }
        return color, bw

    def process(
        self,
        image: np.ndarray,
    ) -> tuple[float, float, dict[str, np.ndarray], dict[str, np.ndarray]]:
        # Unwarp (undistort) the image
        unwarped_image = image_utils.unwarp_image(image, self._K, self._D, self._P)

        # Determine which image to use for processing
        if config.ENHANCED_LANE_DETECTION:
            proc_image = unwarped_image
            image_height, image_width = proc_image.shape[:2]
            white_lane_mask, yellow_mask, red_mask, edge_mask, white_color = (
                self.perception.filter_lane_colors_enhanced(proc_image)
            )
        else:
            proc_image = image
            image_height, image_width = proc_image.shape[:2]
            white_lane_mask, yellow_mask, red_mask = (
                self.perception.filter_lane_colors_standard(proc_image)
            )
            white_color = white_lane_mask
            edge_mask = None

        # # Check for obstacles (Object Detection)
        # object_detected = self.perception.check_obstacle(proc_image)
        bev_mask = image_utils.project_mask_to_bev(edge_mask, self._H, self.bev_config)

        # Apply Region of Interest (ROI) mask in DRIVE or CROSS (with enhanced) state
        if self.planner.state == State.DRIVE or (
            self.planner.state == State.CROSS and config.ENHANCED_LANE_DETECTION
        ):
            white_lane_mask[: config.HIDE_TOP_OF_IMAGE, :] = 0
            yellow_mask[: config.HIDE_TOP_OF_IMAGE, :] = 0
            red_mask[: config.HIDE_TOP_OF_IMAGE, :] = 0

        # Update FSM transitions
        self.planner.update_state(red_mask)

        if config.ENHANCED_LANE_DETECTION and self.planner.state == State.STOP:
            raw_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
                    proc_image,
                    white_lane_mask,
                    yellow_mask,
                    red_mask,
                    white_spline,
                    yellow_spline,
                    None,
                )
                self.planner.time_last_waypoint = time.time()
                color_vis, bw_vis = self._build_vis_dicts(locals())
                return (
                    0.0,
                    config.TURN_SPEED_RIGHT_WHEEL,
                    color_vis,
                    bw_vis,
                )

            # Extract stop lines as structured entities
            red_lines = self.world_model.extract_red_lines(red_mask)

            waypoints = self.planner.compute_waypoints(
                white_spline, yellow_spline, red_lines, image_width, image_height
            )

            if waypoints is None:
                visualization = self.visualizer.visualize(
                    proc_image,
                    white_lane_mask,
                    yellow_mask,
                    red_mask,
                    white_spline,
                    yellow_spline,
                    None,
                )
                color_vis, bw_vis = self._build_vis_dicts(locals())
                return (
                    0.0,
                    0.0,
                    color_vis,
                    bw_vis,
                )

            # Control: Estimate heading error
            heading_error = self.controller.estimate_heading_error(
                waypoints, image_width, image_height
            )
            # Control: Calculate velocities
            is_stopped = self.planner.state == State.STOP
            vel_left, vel_right = self.controller.heading_to_wheel_commands(
                heading_error, is_stopped
            )

            # Store computed velocities in case we transition to crossing decision
            self.planner.crossing_vel_left = vel_left
            self.planner.crossing_vel_right = vel_right

        # Handling Crossing Decision (from Behavior Planner FSM state: State.FOLLOW)
        if self.planner.crossing_decision:
            red_lines = self.world_model.extract_red_lines(red_mask)
            if not self.planner.intersection_admitted:
                self.planner.intersection_admitted = self.planner.can_intersect(
                    proc_image, red_lines
                )

            if self.planner.intersection_admitted:
                vel_left = self.planner.crossing_vel_left
                vel_right = self.planner.crossing_vel_right
                waypoints = self.planner.decision_waypoint
            else:
                self.planner.state_entered_at = time.time()
                vel_left = 0.0
                vel_right = 0.0
                waypoints = self.planner.decision_waypoint

        self.planner.time_last_waypoint = time.time()

        # Handle Blocking State (Collision Avoidance / Stop for obstacles)
        is_blocked = False
        self.planner.handle_blocking(is_blocked)

        if self.planner.state == State.BLOCKED:
            vel_left = 0.0
            vel_right = 0.0

        # Visualization
        visualization = self.visualizer.visualize(
            proc_image,
            white_lane_mask,
            yellow_mask,
            red_mask,
            white_spline,
            yellow_spline,
            waypoints,
        )

        self.planner.blocked_state_last_time = time.time()

        color_vis, bw_vis = self._build_vis_dicts(
            locals()
        )  # TODO: maybe fix this locals() hack, variables look like they are unused
        return (
            vel_left,
            vel_right,
            color_vis,
            bw_vis,
        )
