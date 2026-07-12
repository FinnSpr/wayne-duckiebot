"""
Modular lane-following pipeline wrapper for Duckiebot.
"""

import time
from typing import Dict, Optional, Tuple

import config
import cv2
import numpy as np
from control import Controller
import image_utils
from obstacle_avoidance import cem_planner, get_planning_cost_function

# from obstacle_avoidance import cem_planner, get_planning_cost_function
from perception import PerceptionModule
from planning import BehaviorPlanner, State
from visualizer import Visualizer
from world_model import WorldModel


class SelfDrivingPipeline:
    def __init__(
        self,
        K: np.ndarray,
        D: np.ndarray,
        P: np.ndarray,
        H: np.ndarray,
        bev_config: image_utils.BEVConfig,
    ):
        self.bev_config = bev_config
        self.avoidance_cost_fn = None

        self.perception = PerceptionModule(
            K=K, D=D, P=P, H=H, use_object_detection=config.OBJECT_DETECTION
        )
        self.world_model = WorldModel()
        self.planner = BehaviorPlanner()
        self.controller = Controller()
        self.visualizer = Visualizer()

        # Waypoint function by state
        self._waypoint_by_state = {
            State.DRIVE: self._waypoints_drive,
            State.CROSS: self._waypoints_cross,
            State.FOLLOW: self._waypoints_follow,
            State.DUCKIE_AVOID: self._waypoints_duckie_avoid,
        }

        # Speed calculation by state
        self._speed_by_state = {
            State.DRIVE: self._speed_drive,
            State.STOP: self._speed_stop,
            State.FOLLOW: self._speed_follow,
            State.CROSS: self._speed_drive,
            State.TURN: self._speed_turn,
            State.WAIT_FOR_INSTRUCTION: self._speed_stop,
            State.DUCKIE_AVOID: self._speed_drive,
        }

    def process(
        self,
        image: np.ndarray,
        left_encoder: int,
        right_encoder: int,
    ) -> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        self.planner.construct_timers_if_needed()

        # Perception
        self.perception.perceive(
            image,
            use_enhanced=config.ENHANCED_LANE_DETECTION,
            world_model=self.world_model,
        )

        # State update
        self.planner.set_ticks(left_encoder, right_encoder)
        self.planner.duckie_in_roi = self._check_duckie_in_roi()
        self.planner.stop_line_area = np.sum(
            self.perception.red_mask[
                int(config.STOP_MARKER_Y_RATIO * self.perception.image_height) :, :
            ]
            > 0
        )
        self.planner.update_state()

        # Waypoints for states that calculate them
        waypoint_function = self._waypoint_by_state.get(self.planner.state)
        waypoints = waypoint_function() if waypoint_function else None

        # Speed calculation
        speed_function = self._speed_by_state[self.planner.state]
        velocities = speed_function(waypoints)

        if waypoints is not None and len(waypoints) > 0:
            self.planner.time_last_waypoint = time.time()

        color_vis, bw_vis = self.get_visualizations(
            self.perception.proc_image, waypoints
        )
        if config.USE_TWIST:
            return velocities[0], velocities[1], color_vis, bw_vis
        return velocities[0], velocities[1], color_vis, bw_vis

    def _waypoints_drive(self) -> Optional[np.ndarray]:
        return self.world_model.get_drive_waypoints(
            self.perception.white_spline,
            self.perception.yellow_spline,
            self.perception.image_width,
            self.perception.image_height,
            red_mask=self.perception.red_mask,
        )

    def _waypoints_cross(self) -> Optional[np.ndarray]:
        return self.world_model.get_drive_waypoints(
            self.perception.white_spline,
            self.perception.yellow_spline,
            self.perception.image_width,
            self.perception.image_height,
            red_mask=None,
        )

    def _waypoints_follow(self) -> Optional[np.ndarray]:
        return np.array([self.planner.get_intersection_waypoint()])

    def _waypoints_duckie_avoid(self) -> Optional[np.ndarray]:
        """Drive waypoints offset sideways away from nearest duckie."""
        waypoints = self._waypoints_drive()
        if waypoints is None:
            return None
        target_waypoint = waypoints[0]
        self.avoidance_cost_fn = get_planning_cost_function(
            left_lane_mask=self.perception.left_white_lane,
            right_lane_mask=self.perception.right_white_lane,
            obstacle_bottom_image_coords=self.perception.duckies_bottom_centers,
            goal_position_image_coords=target_waypoint,
            H_image_to_metric=self.perception.H,
            bev_cfg=self.bev_config,
        )
        obstacle_avoidance_waypoints = cem_planner(cost_function=self.avoidance_cost_fn)
        waypoints_image = image_utils.world_to_image_coords(
            obstacle_avoidance_waypoints, self.perception.H
        )
        return waypoints_image

    def _speed_drive(self, waypoints: Optional[np.ndarray]) -> Tuple[float, float]:
        """Normal lane-following: heading error → wheel speeds."""
        if waypoints is None:
            return 0.0, 0.0
        heading_err = self.controller.estimate_heading_error(
            waypoints[0], self.perception.image_width, self.perception.image_height
        )
        if config.USE_TWIST:
            return self.controller.heading_to_twist(heading_err, False)
        return self.controller.heading_to_wheel_commands(heading_err, False)

    def _speed_stop(self, waypoints: Optional[np.ndarray]) -> Tuple[float, float]:
        return 0.0, 0.0

    def _speed_turn(self, waypoints: Optional[np.ndarray]) -> Tuple[float, float]:
        """Hardcoded in-place turn."""
        self.planner.time_last_waypoint = time.time()
        return (
            config.TURN_SPEED_LEFT_WHEEL,
            config.TURN_SPEED_RIGHT_WHEEL,
        )

    def _speed_follow(self, waypoints: Optional[np.ndarray]) -> Tuple[float, float]:
        """Intersection handling: wait for admission, then steer towards the
        global goal using heading error from the planner."""
        if not self.planner.intersection_admitted:
            red_lines = self.world_model.extract_red_lines(self.perception.red_mask)
            self.planner.intersection_admitted = self.planner.can_intersect(
                self.perception.proc_image, red_lines
            )
            if not self.planner.intersection_admitted:
                self.planner.state_entered_at = time.time()
                return 0.0, 0.0

        if config.USE_GLOBAL_POSE_FOLLOW:
            heading_err = self.planner.get_follow_heading_error()
        else:
            # Legacy: heading error from image-space waypoint
            if waypoints is None:
                return 0.0, 0.0
            heading_err = self.controller.estimate_heading_error(
                waypoints[0], self.perception.image_width, self.perception.image_height
            )

        if config.USE_TWIST:
            return self.controller.heading_to_twist(heading_err, False)
        return self.controller.heading_to_wheel_commands(heading_err, False)

    def _check_duckie_in_roi(self) -> bool:
        """Check if any detected duckies are in the ROI."""
        detections = self.perception.duckies_bottom_centers_world
        if detections.size == 0:
            return False
        roi = config.AVOIDANCE_START_ABSOLUTE_ROI
        in_roi = (np.abs(detections[:, 0]) < roi[0]) & (
            np.abs(detections[:, 1]) < roi[1]
        )
        return np.any(in_roi)

    def get_visualizations(
        self,
        image: np.ndarray,
        waypoints: Optional[np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Build the colour and black-and-white visualization dicts."""
        if not config.PUBLISH_VISUALIZATIONS and not config.LOCAL_TESTING:
            return {}, {}
        perception = self.perception

        white_combined = np.full(image.shape[:2], 0, dtype=np.uint8)
        if perception.left_white_lane is not None:
            white_combined = cv2.bitwise_or(white_combined, perception.left_white_lane)
        if perception.right_white_lane is not None:
            white_combined = cv2.bitwise_or(white_combined, perception.right_white_lane)

        visualization = self.visualizer.visualize(
            perception.proc_image,
            white_combined,
            perception.yellow_mask,
            perception.red_mask,
            perception.white_spline,
            perception.yellow_spline,
            waypoints,
        )

        color_vis: Dict[str, np.ndarray] = {
            "visualization": visualization,
            "unwarped_image": perception.proc_image,
            "image": image,
        }
        if self.avoidance_cost_fn is not None:
            color_vis["heatmap"] = image_utils.get_bev_heatmap_image(
                self.avoidance_cost_fn, self.bev_config
            )
        if config.USE_SEGMENTATION and perception.seg_model is not None:
            color_vis["segmentation"] = perception.seg_model.visualize()

        bw_vis: Dict[str, np.ndarray] = {}
        if perception.edge_mask is not None:
            bw_vis["edge_mask"] = perception.edge_mask
        if perception.right_white_lane is not None:
            bw_vis["white_lane_mask"] = white_combined
        if perception.yellow_mask is not None:
            bw_vis["yellow_lane_mask"] = perception.yellow_mask

        return color_vis, bw_vis
