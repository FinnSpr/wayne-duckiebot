import numpy as np
from typing import Tuple
import config
from pid import PID

class Controller:
    """
    Control Module.
    Responsible for estimating lateral errors and calculating differential wheel speeds.
    """
    def __init__(self):
        self._pid = PID(config.KP, config.KI, config.KD)

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

    def heading_to_twist(self, heading_error, is_stopped):
        if is_stopped:
            self._pid.reset()
            return 0.0, 0.0
        
        omega = self._pid.update(heading_error)
        omega = float(np.clip(omega, -config.MAX_OMEGA, config.MAX_OMEGA))
        # v = float(config.BASE_SPEED)
        v = config.BASE_SPEED
        if getattr(config, "SLOW_DOWN_ON_TURN", False):
            v *= max(0.0, 1.0 - config.TURN_SLOWDOWN_GAIN * abs(heading_error))
        v = float(np.clip(v, 0.0, config.BASE_SPEED))

        return (v, omega)
