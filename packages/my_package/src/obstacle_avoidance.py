import config
import cv2
import image_utils
import numpy as np

# TODO: add offset from (0, 0) to front of bot


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
):
    """
    Build a vectorized cost function for BEV planning.

    Lane masks are reduced to polylines in image space, projected to world
    coordinates, then combined into a single drivable-area polygon. Missing
    lanes fall back to the BEV ROI edges.

    cv2.pointPolygonTest computes the signed distance of each query point
    to the polygon boundary. Points outside the polygon or too close to its
    edge receive cost inf.

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
) -> np.ndarray:
    """
    Cost for each query position: inf if forbidden, else distance to goal.

    A position is forbidden if:
    - Too close to any obstacle (within obstacle_radius + bot_width/2
      + avoidance_margin).
    - Outside the drivable polygon.
    - Inside but closer than bot_width/2 + avoidance_margin to the
      polygon boundary.

    Args:
        positions: (N, 2) world coordinates (x_forward, y_lateral).
        obstacle_positions: (M, 2) obstacle locations in world frame.
            May be empty (0, 2).
        drivable_polygon: (K, 1, 2) contour of the drivable area in
            world coordinates.
        goal_position: (2,) world coordinate of the goal.
        obstacle_radius: Safety radius around each obstacle (metres).
        bot_width: Robot width (metres).
        avoidance_margin: Extra margin (metres).

    Returns:
        (N,) float64 costs: inf in forbidden regions, Euclidean distance
        to goal elsewhere.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    obstacle_positions = np.atleast_2d(np.asarray(obstacle_positions, dtype=np.float64))
    goal_position = np.asarray(goal_position, dtype=np.float64).ravel()

    dist_to_goal = np.linalg.norm(positions - goal_position, axis=1)
    forbidden = np.zeros(len(positions), dtype=bool)

    if obstacle_positions.size > 0:
        diff = positions[:, np.newaxis, :] - obstacle_positions[np.newaxis, :, :]
        dist_to_obs = np.linalg.norm(diff, axis=2)
        min_obs_dist = dist_to_obs.min(axis=1)
        forbidden |= min_obs_dist < (
            obstacle_radius + bot_width / 2.0 + avoidance_margin
        )

    lane_margin = bot_width / 2.0 + avoidance_margin
    poly_dist = _polygon_distance(positions, drivable_polygon)
    forbidden |= poly_dist < lane_margin

    cost = np.where(forbidden, np.inf, dist_to_goal)
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
