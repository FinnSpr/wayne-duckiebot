import config
import cv2
import image_utils
import kinematics
import numpy as np

# TODO: add offset from (0, 0) to front of bot
# TODO: should probably add cost for the heading of the bot
# TODO: x, y coordinate mismatch


def cem_planner(
    cost_function: callable,
    start_pos: tuple[float, float] = (0, 0),
    start_angle: float = 0.0,
    horizon: int = 10,
    num_samples: int = 200,
    num_elites: int = 20,
    num_iterations: int = 3,
    dt: float = 0.1,
    v_mean: float | np.ndarray = 0.2,
    v_std: float | np.ndarray = 0.1,
    omega_mean: float | np.ndarray = 0.0,
    omega_std: float | np.ndarray = 1.0,
    temperature: float = 0.1,
    v_clip: tuple[float, float] = (0.0, 1.0),
    omega_clip: tuple[float, float] = (-np.pi, np.pi),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-entropy method planner with MPPI-style elite weighting.

    Samples action sequences (v, omega) from independent Gaussians, rolls
    them out with a differential-drive model, evaluates trajectory costs,
    selects elites and refits the Gaussian means/variances using
    softmax-weighted statistics of the elites.

    Args:
        cost_function: f(points) -> (M,) costs, where points is (M, 2).
        start_pos: (x, y) current robot position in world frame.
        start_angle: Current heading in radians.
        horizon: Number of time steps to plan ahead.
        num_samples: Number of trajectory samples per iteration.
        num_elites: Number of elites kept per iteration.
        num_iterations: Number of CEM iterations.
        dt: Time step duration (seconds).
        v_mean: Initial mean forward speed. Scalar or (horizon,).
        v_std: Initial std of forward speed. Scalar or (horizon,).
        omega_mean: Initial mean angular velocity. Scalar or (horizon,).
        omega_std: Initial std of angular velocity. Scalar or (horizon,).
        temperature: MPPI temperature for softmax weighting.
        v_clip: (min, max) bounds for forward speed.
        omega_clip: (min, max) bounds for angular velocity.

    Returns:
        (trajectory_positions, trajectory_actions) where
        positions is (horizon, 2) and actions is (horizon, 2) [v, omega].
    """
    start_pos = np.asarray(start_pos, dtype=np.float64)

    # Broadcast initial distribution parameters to (horizon,).
    v_mean = np.broadcast_to(np.asarray(v_mean, dtype=np.float64), (horizon,)).copy()
    v_std = np.broadcast_to(np.asarray(v_std, dtype=np.float64), (horizon,)).copy()
    omega_mean = np.broadcast_to(
        np.asarray(omega_mean, dtype=np.float64), (horizon,)
    ).copy()
    omega_std = np.broadcast_to(
        np.asarray(omega_std, dtype=np.float64), (horizon,)
    ).copy()

    best_cost = np.inf
    best_actions = None
    best_positions = None

    for _ in range(num_iterations):
        # Sample action sequences and clip to bounds.
        v_samples = np.clip(
            np.random.normal(v_mean, v_std, size=(num_samples, horizon)),
            v_clip[0],
            v_clip[1],
        )
        omega_samples = np.clip(
            np.random.normal(omega_mean, omega_std, size=(num_samples, horizon)),
            omega_clip[0],
            omega_clip[1],
        )

        # Roll out all trajectories.
        start_positions = np.tile(start_pos, (num_samples, 1))  # (N, 2)
        start_angles = np.full(num_samples, start_angle)  # (N,)
        traj = kinematics.diff_drive_trajectory(
            start_positions, start_angles, v_samples, omega_samples, dt
        )  # (N, horizon, 3)
        positions = traj[..., :2]  # (N, horizon, 2)

        # Evaluate costs.
        costs = get_trajectories_score(positions, cost_function)

        # Track best overall.
        idx_min = np.argmin(costs)
        if costs[idx_min] < best_cost:
            best_cost = costs[idx_min]
            best_actions = np.column_stack([v_samples[idx_min], omega_samples[idx_min]])
            best_positions = positions[idx_min]

        # Select elites.
        elite_idx = np.argpartition(costs, num_elites)[:num_elites]
        elite_costs = costs[elite_idx]
        elite_v = v_samples[elite_idx]  # (K, horizon)
        elite_omega = omega_samples[elite_idx]

        # MPPI-style weights within elites.
        c_min = elite_costs.min()
        w = np.exp(-(elite_costs - c_min) / temperature)
        w /= w.sum()

        # Refit means (weighted) and clip to bounds.
        v_mean = np.clip((w[:, None] * elite_v).sum(axis=0), v_clip[0], v_clip[1])
        omega_mean = np.clip(
            (w[:, None] * elite_omega).sum(axis=0), omega_clip[0], omega_clip[1]
        )

        # Refit std (weighted).
        v_std = np.sqrt((w[:, None] * (elite_v - v_mean) ** 2).sum(axis=0))
        omega_std = np.sqrt((w[:, None] * (elite_omega - omega_mean) ** 2).sum(axis=0))

        # Prevent std collapse.
        v_std = np.maximum(v_std, 1e-4)
        omega_std = np.maximum(omega_std, 1e-4)

    return best_positions, best_actions


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
    left_lane_mask: np.ndarray | None,
    right_lane_mask: np.ndarray | None,
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
    free_y_threshold: float = config.FREE_Y_THRESHOLD,
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
        free_y_threshold: Positions with |y_lateral| below this skip
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
            free_y_threshold,
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
    free_y_threshold: float,
) -> np.ndarray:
    """
    Soft navigation cost for each query position.

    If abs(y_lateral) < free_y_threshold the cost is just distance to
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
        free_y_threshold: Positions with |y_lateral| below this skip
            all penalties (default 0 = disabled).

    Returns:
        (N,) float64 costs.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    obstacle_positions = np.atleast_2d(np.asarray(obstacle_positions, dtype=np.float64))
    goal_position = np.asarray(goal_position, dtype=np.float64).ravel()

    dist_to_goal = np.linalg.norm(positions - goal_position, axis=1)
    cost = dist_to_goal.copy()

    # --- soft obstacle penalty ---
    obs_threshold = obstacle_radius + bot_width / 2.0 + avoidance_margin
    if obstacle_positions.size > 0:
        diff = positions[:, np.newaxis, :] - obstacle_positions[np.newaxis, :, :]
        dist_to_obs = np.linalg.norm(diff, axis=2)
        min_obs_dist = dist_to_obs.min(axis=1)
        penalty_obs = np.maximum(0.0, obs_threshold - min_obs_dist)
        cost += penalty_obs * lambda_obstacle_distance
    obstacle_plus_goal_cost = cost.copy()

    # --- soft polygon penalty ---
    lane_margin = bot_width / 2.0 + avoidance_margin
    poly_dist = _polygon_distance(positions, drivable_polygon)
    penalty_poly = np.maximum(0.0, lane_margin - poly_dist)
    cost += penalty_poly * lambda_polygon_distance

    # --- free-y override: skip all penalties near centre-line ---
    free = positions[:, 0] < free_y_threshold
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
    mask: np.ndarray | None,
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
