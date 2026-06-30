import config
import cv2
import image_utils
import numpy as np

# TODO: Is the camera at (0, 0) or the center of motors?


def get_planning_cost_function(
    left_lane_mask: np.ndarray | None,
    right_lane_mask: np.ndarray | None,
    obstacle_bottom_image_coords: np.ndarray,
    goal_position_image_coords: np.ndarray,
    H_image_to_metric: np.ndarray,
    obstacle_radius: float = config.DUCKIE_RADIUS,
    bot_width: float = config.BOT_WIDTH,
    avoidance_offset: float = config.AVOIDANCE_OFFSET,
    polyline_epsilon: float = 0.02,
):
    """
    Build a vectorized cost function for BEV planning with separate left
    and right lane masks.

    Converts obstacle coordinates, goal coordinates, and lane masks from
    image space to world coordinates.  Each lane mask is reduced to a
    polyline via :func:`_mask_to_world_polyline`, which projects pixels
    to the ground plane **first** and then fits the polyline in world
    space.

    The returned closure can be fed directly to
    :func:`image_utils.get_bev_heatmap_image` or any planner that evaluates
    costs over an ``(N, 2)`` array of world points.

    Args:
        left_lane_mask:
            ``(H_img, W_img)`` uint8 binary mask of the **left** lane
            marking.  Correct side = ``ccw ≥ 0``.  Pass ``None`` to disable.
        right_lane_mask:
            ``(H_img, W_img)`` uint8 binary mask of the **right** lane
            marking.  Correct side = ``ccw ≤ 0``.  Pass ``None`` to disable.
        obstacle_bottom_image_coords:
            ``(M, 2)`` obstacle bottom-centre locations in image pixel
            coordinates ``(u, v)``.
        goal_position_image_coords:
            ``(2,)`` goal location in image pixel coordinates ``(u, v)``.
        H_image_to_metric:
            3×3 homography (image pixels → metric ground plane).
        obstacle_radius:
            Safety radius in metres (default from config).
        bot_width:
            Robot width in metres (default from config).
        avoidance_offset:
            Extra planning offset in metres (default from config).
        polyline_epsilon:
            ``approxPolyDP`` simplification tolerance **in metres**
            (default 0.02 = 2 cm).

    Returns:
        Callable ``f(points: np.ndarray) -> np.ndarray`` where ``points``
        has shape ``(N, 2)`` (world coordinates) and returns an ``(N,)``
        cost vector.
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

    def cost_function(positions: np.ndarray) -> np.ndarray:
        return planning_cost_function(
            positions,
            obstacle_positions_world,
            left_poly_world,
            right_poly_world,
            goal_position_world,
            obstacle_radius,
            bot_width,
            avoidance_offset,
        )

    return cost_function


def planning_cost_function(
    positions: np.ndarray,
    obstacle_positions: np.ndarray,
    left_lane_polyline: np.ndarray,
    right_lane_polyline: np.ndarray,
    goal_position: np.ndarray,
    obstacle_radius: float,
    bot_width: float,
    avoidance_offset: float,
) -> np.ndarray:
    """
    Compute a navigation cost for each query position in world coordinates.

    The cost is ``+∞`` if the position is:

    * within ``obstacle_radius + bot_width/2 + avoidance_offset`` of any
      obstacle, **or**
    * within ``bot_width/2 + avoidance_offset`` of the left or right lane
      polyline, **or**
    * on the *wrong* side of either lane polyline:
      - **left lane**:  correct side is ``ccw ≥ 0`` (left of the segment).
      - **right lane**: correct side is ``ccw ≤ 0`` (right of the segment).

    Otherwise the cost is the Euclidean distance to ``goal_position``.

    Args:
        positions:
            ``(N, 2)`` array of world-coordinate query points
            ``(x_forward, y_lateral)``.
        obstacle_positions:
            ``(M, 2)`` array of obstacle locations in the same world frame.
            May be empty ``(0, 2)``.
        left_lane_polyline:
            ``(L, 2)`` left-lane polyline in world coordinates.
            May be empty ``(0, 2)`` to disable.
        right_lane_polyline:
            ``(R, 2)`` right-lane polyline in world coordinates.
            May be empty ``(0, 2)`` to disable.
        goal_position:
            ``(2,)`` world-coordinate of the navigation goal.
        obstacle_radius:
            Safety radius in metres around each obstacle.
        bot_width:
            Width of the robot in metres.
        avoidance_offset:
            Extra offset for path planning in metres.

    Returns:
        ``(N,)`` float64 array of scalar costs.  Elements are ``+∞`` in
        forbidden regions and ``‖position − goal‖`` elsewhere.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    obstacle_positions = np.atleast_2d(np.asarray(obstacle_positions, dtype=np.float64))
    goal_position = np.asarray(goal_position, dtype=np.float64).ravel()

    # --- baseline: distance to goal ---
    dist_to_goal = np.linalg.norm(positions - goal_position, axis=1)  # (N,)
    forbidden = np.zeros(len(positions), dtype=bool)

    # --- obstacle penalty ---
    if obstacle_positions.size > 0:
        diff = positions[:, np.newaxis, :] - obstacle_positions[np.newaxis, :, :]
        dist_to_obs = np.linalg.norm(diff, axis=2)  # (N, M)
        min_obs_dist = dist_to_obs.min(axis=1)  # (N,)
        forbidden |= min_obs_dist < (
            obstacle_radius + bot_width / 2.0 + avoidance_offset
        )

    lane_margin = bot_width / 2.0 + avoidance_offset

    # --- lane penalties ---
    forbidden |= _apply_lane_penalty(
        positions, left_lane_polyline, lane_margin, correct_ge=0.0
    )
    forbidden |= _apply_lane_penalty(
        positions, right_lane_polyline, lane_margin, correct_le=0.0
    )

    cost = np.where(forbidden, np.inf, dist_to_goal)
    return cost


def distance_to_polyline(
    positions: np.ndarray, polyline: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shortest distance from each query point to a polyline, and the index
    of the closest segment.

    For every position the function computes the minimum Euclidean distance
    across all segments of the polyline.  Works on a single point ``(2,)``
    as well as a batch of ``(N, 2)`` points.

    Args:
        positions:
            ``(N, 2)`` or ``(2,)`` array of query points.  Each row is
            ``(x, y)`` (or ``(u, v)`` in image space).
        polyline:
            ``(M, 2)`` array of polyline vertices.  Must have at least two
            rows (otherwise a ``ValueError`` is raised).

    Returns:
        ``(distances, segment_idx)`` tuple where:

        * **distances**   — ``(N,)`` float64 per-point shortest distances.
        * **segment_idx** — ``(N,)`` int64 index of the polyline segment
          (0 … M−2) that yielded the shortest distance for each point.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    polyline = np.asarray(polyline, dtype=np.float64)

    if polyline.ndim != 2 or polyline.shape[0] < 2 or polyline.shape[1] != 2:
        raise ValueError("polyline must have shape (M, 2) with M >= 2")

    # Segment endpoints  A → B  for each of the M-1 segments.
    A = polyline[:-1, :]  # (M-1, 2)
    B = polyline[1:, :]  # (M-1, 2)
    AB = B - A  # (M-1, 2)   segment vectors

    # Vector from each segment start to each query point.
    # positions  (N, 2) → (N, 1, 2)
    # A          (M-1, 2) → (1, M-1, 2)
    AP = positions[:, np.newaxis, :] - A[np.newaxis, :, :]  # (N, M-1, 2)

    # Projection scalar t along each segment (row-wise dot products).
    dot_AP_AB = np.sum(AP * AB[np.newaxis, :, :], axis=2)  # (N, M-1)
    dot_AB_AB = np.sum(AB * AB, axis=1)  # (M-1,)
    dot_AB_AB = np.where(dot_AB_AB == 0.0, 1e-8, dot_AB_AB)  # avoid /0

    t = dot_AP_AB / dot_AB_AB[np.newaxis, :]  # (N, M-1)
    t = np.clip(t, 0.0, 1.0)  # clamp to segment

    # Closest point on each segment for each query point.
    # A (1, M-1, 2) + t (N, M-1, 1) * AB (1, M-1, 2) → (N, M-1, 2)
    closest = A[np.newaxis, :, :] + t[:, :, np.newaxis] * AB[np.newaxis, :, :]

    # Per-segment distances.
    dists = np.linalg.norm(positions[:, np.newaxis, :] - closest, axis=2)

    # Minimum distance and which segment achieved it.
    segment_idx = np.argmin(dists, axis=1)  # (N,) int64
    min_dists = dists[np.arange(len(positions)), segment_idx]  # (N,)

    return min_dists, segment_idx


def _lane_ccw(
    positions: np.ndarray,
    polyline: np.ndarray,
    seg_idx: np.ndarray,
) -> np.ndarray:
    """
    Counter-clockwise cross product for each position against its closest
    polyline segment.

    ``ccw(P, A, B) = (Bx − Ax)·(Py − Ay) − (By − Ay)·(Px − Ax)``

    * ``ccw > 0`` → *P* is to the **left** of the directed segment *A→B*.
    * ``ccw < 0`` → *P* is to the **right**.
    * ``ccw = 0`` → *P* is collinear.

    Args:
        positions: ``(N, 2)`` query points.
        polyline:  ``(M, 2)`` polyline vertices.
        seg_idx:   ``(N,)`` index of the closest segment for each point
                   (as returned by :func:`distance_to_polyline`).

    Returns:
        ``(N,)`` float64 CCW values.
    """
    A = polyline[:-1, :]  # (M-1, 2)
    B = polyline[1:, :]  # (M-1, 2)
    A_closest = A[seg_idx]  # (N, 2)
    B_closest = B[seg_idx]  # (N, 2)
    AB = B_closest - A_closest  # (N, 2)
    AP = positions - A_closest  # (N, 2)
    return AB[:, 0] * AP[:, 1] - AB[:, 1] * AP[:, 0]  # (N,)


def _is_wrong_side(
    positions: np.ndarray,
    polyline: np.ndarray,
    seg_idx: np.ndarray,
    correct_ge: float | None = None,
    correct_le: float | None = None,
) -> np.ndarray:
    """
    Boolean mask indicating which positions are on the **wrong** side of a
    lane polyline.

    The “correct” side is defined by one or both of ``correct_ge`` /
    ``correct_le`` thresholds on the CCW cross product (see
    :func:`_lane_ccw`).  A position is *wrong* if its CCW value falls
    outside the specified interval.

    Args:
        positions:  ``(N, 2)`` query points.
        polyline:   ``(M, 2)`` lane polyline.
        seg_idx:    ``(N,)`` closest-segment indices.
        correct_ge: Lower bound for “correct” CCW (inclusive).  ``None``
                    means no lower bound.
        correct_le: Upper bound for “correct” CCW (inclusive).  ``None``
                    means no upper bound.

    Returns:
        ``(N,)`` bool array — ``True`` where the position is on the
        forbidden side.
    """
    ccw = _lane_ccw(positions, polyline, seg_idx)
    wrong = np.zeros(len(positions), dtype=bool)
    if correct_ge is not None:
        wrong |= ccw < correct_ge
    if correct_le is not None:
        wrong |= ccw > correct_le
    return wrong


def _apply_lane_penalty(
    positions: np.ndarray,
    polyline: np.ndarray,
    margin: float,
    correct_ge: float | None = None,
    correct_le: float | None = None,
) -> np.ndarray:
    """
    Forbidden mask for positions that are too close to *or* on the wrong
    side of a single lane polyline.

    Args:
        positions: ``(N, 2)`` query points.
        polyline:  ``(M, 2)`` lane polyline.  If ``M < 2`` the penalty is
                   a no-op (all ``False``).
        margin:    Minimum allowed distance to the polyline.
        correct_ge, correct_le:  CCW bounds forwarded to
                   :func:`_is_wrong_side`.

    Returns:
        ``(N,)`` bool array — ``True`` where the position is forbidden
        w.r.t. this lane.
    """
    polyline = np.atleast_2d(np.asarray(polyline, dtype=np.float64))
    if polyline.shape[0] < 2:
        return np.zeros(len(positions), dtype=bool)

    lane_dist, seg_idx = distance_to_polyline(positions, polyline)
    forbidden = lane_dist < margin
    forbidden |= _is_wrong_side(positions, polyline, seg_idx, correct_ge, correct_le)
    return forbidden


def _mask_to_world_polyline(
    mask: np.ndarray | None,
    H_image_to_metric: np.ndarray,
    epsilon: float = 0.02,
) -> np.ndarray:
    """
    Convert a binary lane mask to a simplified polyline **in world
    coordinates**.

    All foreground pixels are projected to the ground plane first; then
    points are binned by forward distance ``x``, the mean lateral
    coordinate ``y`` in each bin becomes a vertex, and finally
    ``cv2.approxPolyDP`` simplifies the sequence with tolerance
    ``epsilon`` **in metres**.

    Args:
        mask:              Binary lane mask, or ``None``.
        H_image_to_metric: 3×3 homography (image pixels → metric ground).
        epsilon:           ``approxPolyDP`` tolerance **in metres**
                           (default 2 cm).

    Returns:
        ``(K, 2)`` world-coordinate polyline ``(x_forward, y_lateral)``,
        or ``(0, 2)`` when ``mask`` is ``None`` or empty.
    """
    if mask is None:
        return np.empty((0, 2), dtype=np.float64)

    # 1.  All foreground pixels → image coords (u, v) = (col, row).
    positions = image_utils.mask_to_positions(mask)  # (N, 2) [row, col]
    if positions.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    image_pts = positions[:, ::-1].astype(np.float64)  # (N, 2) [u, v]

    # 2.  Project every pixel to the ground plane.
    world = image_utils.image_to_world_coords(image_pts, H_image_to_metric)

    # 3.  Bin by forward distance and take the mean lateral coordinate.
    x = world[:, 0]  # forward
    y = world[:, 1]  # lateral
    bin_width = max(epsilon / 2.0, 0.005)  # at least 5 mm
    bins = np.arange(x.min(), x.max() + bin_width, bin_width)
    bin_idx = np.digitize(x, bins)
    unique_bins = np.unique(bin_idx)

    x_binned = np.array([x[bin_idx == b].mean() for b in unique_bins])
    y_binned = np.array([y[bin_idx == b].mean() for b in unique_bins])

    # 4.  Sort by forward distance (should already be ordered, but be safe).
    order = np.argsort(x_binned)
    points = np.column_stack([x_binned[order], y_binned[order]])

    # 5.  Simplify with approxPolyDP (epsilon in metres).
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(contour, epsilon, closed=False)
    return approx[:, 0, :].astype(np.float64)
