import numpy as np


def diff_drive_step(
    pos: np.ndarray,
    angle: np.ndarray,
    v: np.ndarray,
    omega: np.ndarray,
    dt: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate one step of a differential-drive robot with constant commands.

    Uses the standard unicycle model.  Works on a single position or a batch.

    Args:
        pos: (..., 2) current positions (x, y).
        angle: (...,) current headings in radians.
        v: (...,) forward speeds (m/s).
        omega: (...,) angular velocities (rad/s).
        dt: Step duration (seconds). Scalar or broadcastable to v.

    Returns:
        (new_pos, new_angle) with the same shapes as inputs.
    """
    v = np.asarray(v, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)

    straight = np.abs(omega) < 1e-12

    r = np.divide(v, omega, where=~straight, out=np.zeros_like(v))
    new_angle = angle + omega * dt

    # Straight robots
    pos = pos.copy()
    pos[..., 0] += np.where(straight, v * np.cos(angle) * dt, 0.0)
    pos[..., 1] += np.where(straight, v * np.sin(angle) * dt, 0.0)

    # Turning robots
    pos[..., 0] += np.where(~straight, r * (np.sin(new_angle) - np.sin(angle)), 0.0)
    pos[..., 1] += np.where(~straight, -r * (np.cos(new_angle) - np.cos(angle)), 0.0)

    return pos, new_angle


def diff_drive_trajectory(
    start_pos: np.ndarray,
    start_angle: np.ndarray,
    v: np.ndarray,
    omega: np.ndarray,
    dt: float | np.ndarray = 0.1,
) -> np.ndarray:
    """
    Simulate differential-drive trajectories for a batch of robots.

    Args:
        start_pos: (N, 2) starting positions (x, y).
        start_angle: (N,) starting headings in radians.
        v: (N, L) forward speeds at each step (m/s).
        omega: (N, L) angular velocities at each step (rad/s).
        dt: Step duration in seconds. Scalar or (N, L).

    Returns:
        (N, L, 3) array where each step is [x, y, angle].
    """
    N, L = v.shape
    dt_arr = np.broadcast_to(np.asarray(dt, dtype=np.float64), (N, L))

    pos = np.asarray(start_pos, dtype=np.float64).reshape(N, 2)
    angle = np.asarray(start_angle, dtype=np.float64).ravel()

    trajectory = np.empty((N, L, 3), dtype=np.float64)

    for step in range(L):
        pos, angle = diff_drive_step(
            pos, angle, v[:, step], omega[:, step], dt_arr[:, step]
        )
        trajectory[:, step, :2] = pos
        trajectory[:, step, 2] = angle

    return trajectory
