import numpy as np
from typing import Tuple
import config

class Controller:
    """
    Control Module.
    Responsible for estimating lateral errors and calculating differential wheel speeds.
    """
    def __init__(self):
        pass

    def estimate_heading_error(self, waypoints: np.ndarray, image_width: int, image_height: int) -> float:
        """Estimate heading error from waypoints."""
        farthest = waypoints[0]
        if config.ENHANCED_LANE_DETECTION:
            image_center_x = image_width / 2.0
            dx = farthest[0] - image_center_x
            dy = image_height - farthest[1]
            path_angle = np.arctan2(dx, dy)
            angle_error = path_angle / (np.pi / 2)
            return float(np.clip(angle_error, -1.0, 1.0))
        else:
            image_center_x = image_width / 2.0
            error_farthest = (farthest[0] - image_center_x) / image_center_x
            return float(np.clip(error_farthest, -1.0, 1.0))

    def heading_to_wheel_commands(self, heading_error: float, is_stopped: bool) -> Tuple[float, float]:
        """Convert a heading error to differential wheel commands."""
        if is_stopped:
            return 0.0, 0.0

        correction = config.STEERING_GAIN * heading_error
        correction = float(np.clip(correction, -config.MAX_SPEED_DIFF, config.MAX_SPEED_DIFF))

        vel_left = config.BASE_SPEED + correction
        vel_right = config.BASE_SPEED - correction

        # Clamp to valid range
        vel_left = float(np.clip(vel_left, -1.0, 1.0))
        vel_right = float(np.clip(vel_right, -1.0, 1.0))

        return vel_left, vel_right
