import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> float:
    """
    Args:
        ticks: Current tick count from the encoders.
        prev_ticks: Previous tick count from the encoders.
        resolution: Number of ticks per full wheel rotation returned by the encoder.
    Return:
        dphi: Rotation of the wheel in radians.
    """

    # TODO: these are random values, you have to implement your own solution in here
    delta_ticks = ticks - prev_ticks
    alpha = 2 * np.pi / resolution
    dphi = delta_ticks * alpha
    return dphi


def estimate_pose(
    R: float,
    baseline: float,
    x_prev: float,
    y_prev: float,
    theta_prev: float,
    delta_phi_left: float,
    delta_phi_right: float,
):
    """
    Calculate the current Duckiebot pose using the dead-reckoning model.

    Args:
        R:                  radius of wheel (both wheels are assumed to have the same size) - this is fixed in simulation,
                            and will be imported from your saved calibration for the real robot
        baseline:           distance from wheel to wheel; 2L of the theory
        x_prev:             previous x estimate - assume given
        y_prev:             previous y estimate - assume given
        theta_prev:         previous orientation estimate - assume given
        delta_phi_left:     left wheel rotation (rad)
        delta_phi_right:    right wheel rotation (rad)

    Return:
        x_curr:                  estimated x coordinate
        y_curr:                  estimated y coordinate
        theta_curr:              estimated heading
    """
    dist_left = R * delta_phi_left
    dist_right = R * delta_phi_right
    dist_center = (dist_left + dist_right) / 2
    delta_theta = (dist_right - dist_left) / baseline
    theta_curr = theta_prev + delta_theta
    theta_mid = theta_prev + delta_theta / 2

    x_curr = x_prev + dist_center * np.cos(theta_mid)
    y_curr = y_prev + dist_center * np.sin(theta_mid)

    return x_curr, y_curr, theta_curr
