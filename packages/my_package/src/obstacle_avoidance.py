from typing import Optional, Tuple, Union  # noqa: UP035

import config
import cv2
import image_utils
import kinematics
import numpy as np

# TODO: should maybe add cost for the heading of the bot


def cem_planner(
    cost_function: callable,
    start_pos: Tuple[float, float] = (0, 0),
    start_angle: float = 0.0,
    horizon: int = config.CEM_HORIZON,
    num_samples: int = config.CEM_NUM_SAMPLES,
    num_elites: int = config.CEM_NUM_ELITES,
    num_iterations: int = config.CEM_NUM_ITERATIONS,
    dt: float = config.CEM_DT,
    v_const: float = config.CEM_V_CONST,
    omega_mean: Union[float, np.ndarray] = config.CEM_OMEGA_MEAN,
    omega_std: Union[float, np.ndarray] = config.CEM_OMEGA_STD,
    temperature: float = config.CEM_TEMPERATURE,
    omega_clip: Tuple[float, float] = config.CEM_OMEGA_CLIP,
    action_repeat: int = config.CEM_ACTION_REPEAT,
    last_elite_actions: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cross-entropy method planner with MPPI-style elite weighting.

    Fixed speed, plans only omega.

    Args:
        cost_function: f(points) -> (M,) costs, where points is (M, 2).
        start_pos: (x, y) current robot position in world frame.
        start_angle: Current heading in radians.
        horizon: Number of time steps to plan ahead.
        num_samples: Number of trajectory samples per iteration.
        num_elites: Number of elites kept per iteration.
        num_iterations: Number of CEM iterations.
        dt: Time step duration (seconds).
        v_const: Fixed forward speed to use (not sampled).
        omega_mean: Initial mean angular velocity. Scalar or (horizon,).
        omega_std: Initial std of angular velocity. Scalar or (horizon,).
        temperature: MPPI temperature for softmax weighting.
        omega_clip: (min, max) bounds for angular velocity.
        action_repeat: Number of times each action is repeated with
            proportionally smaller dt for finer cost evaluation
            (1 = no repeat).
        last_elite_actions: (num_elites, horizon) array of the last iteration's elite omegas.

    Returns:
        (trajectory_positions, elite_actions) where
        positions is (horizon, 2) and elite_actions is (num_elites, horizon)
        omega values from the final iteration.
    """
    start_pos = np.asarray(start_pos, dtype=np.float64)
    start_angles = np.full(num_samples, start_angle)  # (N,)
    start_positions = np.tile(start_pos, (num_samples, 1))  # (N, 2)

    K = max(1, action_repeat)
    v_unrolled = np.full((num_samples, horizon), v_const, dtype=np.float64)
    v_repeated = np.repeat(v_unrolled, K, axis=1)  # (N, H*K)

    # Broadcast initial distribution parameters to (horizon,).
    omega_mean = np.broadcast_to(
        np.asarray(omega_mean, dtype=np.float64), (horizon,)
    ).copy()
    omega_std = np.broadcast_to(
        np.asarray(omega_std, dtype=np.float64), (horizon,)
    ).copy()

    best_cost = np.inf
    best_positions = None

    for _ in range(num_iterations):
        omega_samples = np.clip(
            np.random.normal(omega_mean, omega_std, size=(num_samples, horizon)),
            omega_clip[0],
            omega_clip[1],
        )
        omega_repeated = np.repeat(omega_samples, K, axis=1)  # (N, H*K)

        # Roll out with action repeat for fine-grained evaluation
        traj = kinematics.diff_drive_trajectory(
            start_positions,
            start_angles,
            v_repeated,
            omega_repeated,
            dt / K,
        )  # (N, H*K, 3)
        positions = traj[..., :2]  # (N, H*K, 2)
        angles = traj[..., 2]  # (N, H*K)

        # Evaluate cost at the front of the robot
        front_positions = _offset_to_front(
            positions, angles, offset=config.WHEEL_TO_FRONT_OFFSET
        )
        costs = get_trajectories_score(front_positions, cost_function)

        idx_min = np.argmin(costs)
        if costs[idx_min] < best_cost:
            best_cost = costs[idx_min]
            # Return just the horizon endpoints (every K-th position)
            best_positions = positions[idx_min, K - 1 :: K]

        # Elites
        elite_idx = np.argpartition(costs, num_elites)[:num_elites]
        elite_costs = costs[elite_idx]
        elite_omega = omega_samples[elite_idx]

        # MPPI-style weights within elites.
        c_min = elite_costs.min()
        w = np.exp(-(elite_costs - c_min) / temperature)
        w /= w.sum()

        omega_mean = np.clip(
            (w[:, None] * elite_omega).sum(axis=0), omega_clip[0], omega_clip[1]
        )
        omega_std = np.sqrt((w[:, None] * (elite_omega - omega_mean) ** 2).sum(axis=0))
        omega_std = np.maximum(omega_std, 1e-4)

    return best_positions, elite_omega


def get_trajectories_score(
    trajectories: np.ndarray,
    cost_function: callable,
    weight_final: float = config.PLANNING_WEIGHT_FINAL_POSITION,
) -> np.ndarray:
    """
    Score a batch of trajectories using a point-wise cost function.

    score_i = sum(cost(trajectories[i, :-1])) + weight_final * cost(trajectories[i, -1])

    Args:
        trajectories: (N, L, 2) array of waypoints in world coordinates.
        cost_function: f(points) -> (M,) costs, where points is (M, 2).
        weight_final: Multiplier for the final waypoint cost.

    Returns:
        (N,) float64 trajectory scores.
    """
    N, L, _ = trajectories.shape

    flat = trajectories.reshape(-1, 2)  # (N*L, 2)
    costs = np.asarray(cost_function(flat), dtype=np.float64).reshape(N, L)

    interior = costs[:, :-1].sum(axis=1)  # sum of all but last
    final = costs[:, -1] * weight_final  # weighted last
    return interior + final


def get_planning_cost_function(
    left_lane_mask: Optional[np.ndarray],
    right_lane_mask: Optional[np.ndarray],
    obstacle_bottom_image_coords: np.ndarray,
    goal_position_image_coords: np.ndarray,
    H_image_to_metric: np.ndarray,
    bev_cfg: image_utils.BEVConfig,
    obstacle_radius: float = config.DUCKIE_RADIUS,
    bot_width: float = config.BOT_WIDTH,
    avoidance_margin: float = config.AVOIDANCE_MARGIN,
    polyline_epsilon: float = config.LANE_POLY_EPSILON,
    lambda_obstacle_distance: float = config.LAMBDA_OBSTACLES,
    lambda_polygon_distance: float = config.LAMBDA_OBSTACLES,
    free_x_threshold: float = config.FREE_X_THRESHOLD,
):
    """
    Build a vectorized cost function for BEV planning.

    Lane masks are reduced to polylines in image space, projected to world
    coordinates, then combined into a single drivable-area polygon. Missing
    lanes fall back to the BEV ROI edges.

    cv2.pointPolygonTest computes the signed distance of each query point
    to the polygon boundary. These distances (and obstacle distances) are
    converted to soft penalties instead of hard inf barriers.

    Args:
        left_lane_mask: (H, W) uint8 binary mask of the left lane marking.
            Pass None to disable.
        right_lane_mask: (H, W) uint8 binary mask of the right lane marking.
            Pass None to disable.
        obstacle_bottom_image_coords: (M, 2) obstacle bottom-centre pixel
            locations (u, v).
        goal_position_image_coords: (2,) goal pixel location (u, v).
        H_image_to_metric: 3x3 homography (image pixels to metric ground).
        bev_cfg: BEVConfig describing the BEV region.
        obstacle_radius: Safety radius in metres.
        bot_width: Robot width in metres.
        avoidance_margin: Extra planning margin in metres.
        polyline_epsilon: approxPolyDP tolerance in pixels.
        lambda_obstacle_distance: Weight for obstacle proximity penalty.
        lambda_polygon_distance: Weight for polygon proximity penalty.
        free_x_threshold: Positions with |x_forward| below this skip
            all penalties (default 0 = disabled).

    Returns:
        Callable f(points) where points is (N, 2) world coordinates and
        returns an (N,) cost vector.
    """
    obstacle_positions_world = image_utils.image_to_world_coords(
        obstacle_bottom_image_coords, H_image_to_metric
    )
    # Shift from bottom-centre to centre of obstacle (approximate).
    obstacle_positions_world += np.array([[0.0, obstacle_radius]])

    goal_position_world = image_utils.image_to_world_coords(
        goal_position_image_coords, H_image_to_metric
    ).ravel()

    left_poly_world = _mask_to_world_polyline(
        left_lane_mask, H_image_to_metric, polyline_epsilon
    )
    right_poly_world = _mask_to_world_polyline(
        right_lane_mask, H_image_to_metric, polyline_epsilon
    )
    drivable_polygon = _build_drivable_polygon(
        left_poly_world, right_poly_world, bev_cfg
    )

    def cost_function(positions: np.ndarray) -> np.ndarray:
        return planning_cost_function(
            positions,
            obstacle_positions_world,
            drivable_polygon,
            goal_position_world,
            obstacle_radius,
            bot_width,
            avoidance_margin,
            lambda_obstacle_distance,
            lambda_polygon_distance,
            free_x_threshold,
        )

    return cost_function


def planning_cost_function(
    positions: np.ndarray,
    obstacle_positions: np.ndarray,
    drivable_polygon: np.ndarray,
    goal_position: np.ndarray,
    obstacle_radius: float,
    bot_width: float,
    avoidance_margin: float,
    lambda_obstacle_distance: float,
    lambda_polygon_distance: float,
    free_x_threshold: float,
) -> np.ndarray:
    """
    Soft navigation cost for each query position.

    If abs(x_forward) < free_x_threshold the cost is just distance to
    goal (all penalties skipped). Otherwise:

    cost = dist_to_goal
         + max(0, obs_threshold - min_obs_dist) * lambda_obstacle_distance
         + max(0, lane_margin - poly_dist) * lambda_polygon_distance

    Args:
        positions: (N, 2) world coordinates (x_forward, y_lateral).
        obstacle_positions: (M, 2) obstacle locations in world frame.
            May be empty (0, 2).
        drivable_polygon: (K, 1, 2) contour of the drivable area.
        goal_position: (2,) world coordinate of the goal.
        obstacle_radius: Safety radius around each obstacle (metres).
        bot_width: Robot width (metres).
        avoidance_margin: Extra margin (metres).
        lambda_obstacle_distance: Weight for obstacle proximity penalty.
        lambda_polygon_distance: Weight for polygon proximity penalty.
        free_x_threshold: Positions with |x_forward| below this skip
            all penalties (default 0 = disabled).

    Returns:
        (N,) float64 costs.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    obstacle_positions = np.atleast_2d(np.asarray(obstacle_positions, dtype=np.float64))
    goal_position = np.asarray(goal_position, dtype=np.float64).ravel()

    dist_to_goal = np.linalg.norm(positions - goal_position, axis=1)
    cost = dist_to_goal.copy()

    # Obstacle penalty
    obs_threshold = obstacle_radius + bot_width / 2.0 + avoidance_margin
    if obstacle_positions.size > 0:
        diff = positions[:, np.newaxis, :] - obstacle_positions[np.newaxis, :, :]
        dist_to_obs = np.linalg.norm(diff, axis=2)
        min_obs_dist = dist_to_obs.min(axis=1)
        penalty_obs = np.maximum(0.0, obs_threshold - min_obs_dist)
        cost += penalty_obs * lambda_obstacle_distance
    obstacle_plus_goal_cost = cost.copy()

    # Polygon penalty
    lane_margin = bot_width / 2.0 + avoidance_margin
    poly_dist = _polygon_distance(positions, drivable_polygon)
    penalty_poly = np.maximum(0.0, lane_margin - poly_dist)
    cost += penalty_poly * lambda_polygon_distance

    # skip penalties at bottom (cause polygon side is there)
    free = positions[:, 0] < free_x_threshold
    cost = np.where(free, obstacle_plus_goal_cost, cost)

    return cost


def _build_drivable_polygon(
    world_left: np.ndarray,
    world_right: np.ndarray,
    bev_cfg: image_utils.BEVConfig,
) -> np.ndarray:
    """
    Build a closed polygon for the drivable area between two lane polylines.

    Traces: near-right corner, right lane (near to far), far-right corner,
    far-left corner, left lane (far to near), near-left corner, back to
    near-right. Missing lanes fall back to the corresponding ROI edge.

    Args:
        world_left: (L, 2) left-lane polyline in world coords, or empty.
        world_right: (R, 2) right-lane polyline in world coords, or empty.
        bev_cfg: BEVConfig describing the ROI.

    Returns:
        (K, 1, 2) float32 contour for cv2.pointPolygonTest.
    """
    H = bev_cfg.bev_size[1]
    half_W = bev_cfg.bev_size[0] / 2.0

    near_right = np.array([[0.0, -half_W]], dtype=np.float64)
    far_right = np.array([[H, -half_W]], dtype=np.float64)
    far_left = np.array([[H, half_W]], dtype=np.float64)
    near_left = np.array([[0.0, half_W]], dtype=np.float64)

    parts = [near_right]

    if world_right.size > 0 and world_right.shape[0] >= 2:
        right_pts = np.atleast_2d(np.asarray(world_right, dtype=np.float64))
        order = np.argsort(right_pts[:, 0])
        parts.append(right_pts[order])
    parts.append(far_right)

    parts.append(far_left)
    if world_left.size > 0 and world_left.shape[0] >= 2:
        left_pts = np.atleast_2d(np.asarray(world_left, dtype=np.float64))
        order = np.argsort(left_pts[:, 0])
        parts.append(left_pts[order][::-1])
    parts.append(near_left)

    polygon = np.vstack(parts)
    return polygon.astype(np.float32).reshape(-1, 1, 2)


# TODO: maybe use shapely
def _polygon_distance(
    points: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """
    Signed distance from each point to the polygon boundary.

    Uses cv2.pointPolygonTest. Positive = inside, negative = outside.

    Args:
        points: (N, 2) query points.
        polygon: (K, 1, 2) contour.

    Returns:
        (N,) float64 signed distances.
    """
    return np.array(
        [cv2.pointPolygonTest(polygon, tuple(p), True) for p in points],
        dtype=np.float64,
    )


def _mask_to_world_polyline(
    mask: Optional[np.ndarray],
    H_image_to_metric: np.ndarray,
    epsilon: float = 2.0,
) -> np.ndarray:
    """
    Convert a binary lane mask to a simplified polyline in world coordinates.

    Fits the polyline in image space via _mask_to_polyline, then projects
    to the ground plane.

    Args:
        mask: Binary lane mask, or None.
        H_image_to_metric: 3x3 homography (image pixels to metric ground).
        epsilon: approxPolyDP tolerance in pixels.

    Returns:
        (K, 2) world-coordinate polyline (x_forward, y_lateral),
        or (0, 2) when mask is None or empty.
    """
    if mask is None:
        return np.empty((0, 2), dtype=np.float64)
    poly_img = _mask_to_polyline(mask, epsilon=epsilon)
    if poly_img.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    image_pts = poly_img[:, ::-1].astype(np.float64)
    return image_utils.image_to_world_coords(image_pts, H_image_to_metric)


def _mask_to_polyline(mask: np.ndarray, epsilon: float = 2.0) -> np.ndarray:
    """
    Extract a simplified polyline from a binary lane mask in image pixels.

    Scans the mask row by row; the rightmost foreground column in each row
    becomes a vertex. Simplifies with cv2.approxPolyDP.

    Args:
        mask: 2-D uint8 binary mask (nonzero = foreground).
        epsilon: approxPolyDP tolerance in pixels.

    Returns:
        (K, 2) float64 array of [row, col] coordinates, or (0, 2) if empty.
    """
    positions = image_utils.mask_to_positions(mask)
    if positions.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    rows = positions[:, 0]
    cols = positions[:, 1]
    unique_rows = np.unique(rows)
    mean_cols = np.array([cols[rows == r].max() for r in unique_rows])

    points = np.column_stack([unique_rows, mean_cols])

    contour = points.astype(np.int32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(contour, epsilon, closed=False)
    return approx[:, 0, :].astype(np.float64)


def _offset_to_front(
    positions: np.ndarray,
    angles: np.ndarray,
    offset: float,
) -> np.ndarray:
    """
    Shift positions forward by the robot's front overhang.

    The robot's origin is at the wheel midpoint.  The chassis extends
    ``offset`` metres ahead of that point.  This function projects each
    position forward along the corresponding heading.

    Args:
        positions: (..., 2) world coordinates (x_forward, y_lateral).
        angles: (...,) headings in radians.
        offset: Forward overhang in metres (default 0.06).

    Returns:
        (..., 2) shifted positions.
    """
    positions = np.asarray(positions, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    result = positions.copy()
    result[..., 0] += offset * np.cos(angles)
    result[..., 1] += offset * np.sin(angles)
    return result
