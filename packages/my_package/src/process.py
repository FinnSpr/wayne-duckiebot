"""
Modular lane-following pipeline wrapper for Duckiebot.
"""

import time
from typing import Dict, Optional, Tuple

import config
import cv2
import numpy as np
from control import Controller
from image_utils import BEVConfig
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
        bev_config: BEVConfig,
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
            State.BLOCKED: self._speed_stop,
            State.DUCKIE_AVOID: self._speed_drive,
        }

    def process(
        self,
        image: np.ndarray,
        left_encoder: int,
        right_encoder: int,
    ) -> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        # Perception
        self.perception.perceive(
            image,
            use_enhanced=config.ENHANCED_LANE_DETECTION,
            world_model=self.world_model,
            K=self._K,
            D=self._D,
            P=self._P,
        )

        # State update
        # TODO: maybe forward to update_state instaed
        self.planner.set_ticks(left_encoder, right_encoder)
        self.planner.stop_line_area = np.sum(
            self.perception.red_mask[config.STOP_MARKER_Y :, :] > 0
        )
        self.planner.update_state()

        # Waypoints for states that calculate them
        waypoint_function = self._waypoint_by_state.get(self.planner.state)
        waypoints = waypoint_function() if waypoint_function else None

        # Speed calculation
        speed_function = self._speed_by_state[self.planner.state]
        velocities = speed_function(waypoints)

        self.planner.time_last_waypoint = time.time()
        self.planner.blocked_state_last_time = time.time()

        color_vis, bw_vis = self.get_visualizations(image, waypoints)
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
        return self.planner.get_intersection_waypoint()

    def _waypoints_duckie_avoid(self) -> Optional[np.ndarray]:
        """Drive waypoints offset sideways away from nearest duckie."""
        waypoints = self._waypoints_drive()
        target_waypoint = waypoints[0]
        planning_cost_fn = get_planning_cost_function(
            self.perception.left_white_lane,
            self.perception.right_white_lane,
            self.perception.detection_bottom_centers,
            target_waypoint,
            self._H,
            self.bev_config,
        )
        obstacle_avoidance_waypoints = cem_planner(cost_function=planning_cost_fn)
        return obstacle_avoidance_waypoints

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
        """Intersection handling: wait for admission, then drive."""
        if not self.planner.intersection_admitted:
            red_lines = self.world_model.extract_red_lines(self.perception.red_mask)
            self.planner.intersection_admitted = self.planner.can_intersect(
                self.perception.proc_image, red_lines
            )
            if not self.planner.intersection_admitted:
                self.planner.state_entered_at = time.time()
                return 0.0, 0.0
        if self.planner.intersection_speed is None:
            self.planner.intersection_speed = self._speed_drive(waypoints)
        return self.planner.intersection_speed

    def _check_duckie_nearby(self) -> bool:
        """Return True if any duckie is close enough to trigger avoidance."""
        p = self.perception
        if len(p.detection_bottom_centers) == 0:
            return False
        threshold_y = p.image_height * getattr(config, "DUCKIE_NEARBY_Y_RATIO", 0.6)
        return bool(np.any(p.detection_bottom_centers[:, 1] > threshold_y))

    def get_visualizations(
        self,
        image: np.ndarray,
        waypoints: Optional[np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Build the colour and black-and-white visualization dicts."""
        if not config.PUBLISH_VISUALIZATIONS and not config.LOCAL_TESTING:
            return {}, {}
        perception = self.perception

        white_combined = perception.right_white_lane.copy()
        if perception.left_white_lane is not None:
            white_combined = cv2.bitwise_or(
                perception.left_white_lane, perception.right_white_lane
            )

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
        bw_vis: Dict[str, np.ndarray] = {}
        if perception.edge_mask is not None:
            bw_vis["edge_mask"] = perception.edge_mask
        if perception.right_white_lane is not None:
            bw_vis["white_lane_mask"] = white_combined
        if perception.yellow_mask is not None:
            bw_vis["yellow_lane_mask"] = perception.yellow_mask

        return color_vis, bw_vis
