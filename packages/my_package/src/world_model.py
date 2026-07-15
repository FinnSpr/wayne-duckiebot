from typing import Callable, List, Optional, Tuple

import config
import cv2
import image_utils
import numpy as np
from scipy.interpolate import splev, splprep


class WorldModel:
    """
    World Model / State Estimator Module.
    Responsible for spline fitting and structuring feature observations (such as red stop lines).
    """

    def __init__(self):
        # Cached splines from the last get_drive_waypoints call, exposed for visualization.
        # All are in **image pixel coordinates** (col, row).
        self.last_white_spline: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.last_yellow_spline: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.last_left_white_spline: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def get_drive_waypoints(
        self,
        right_white_boundary: Optional[np.ndarray],
        left_white_boundary: Optional[np.ndarray],
        yellow_boundary: Optional[np.ndarray],
        red_mask: Optional[np.ndarray],
        H: np.ndarray,
        image_width: int,
    ) -> Optional[np.ndarray]:
        """Compute lane-following waypoints for normal driving.

        Takes pre-extracted boundary point sets for image-space spline
        fitting.  Returns waypoints in image pixel coordinates.

        If red_mask is provided (not None), waypoints will point at the
        nearest horizontal red stop line instead of following the lane centre.
        Pass None to ignore red lines (used by CROSS state).

        Returns:
            (N, 2) waypoint array or None when no usable lane data exists.
        """
        self.last_white_spline, self.last_yellow_spline, self.last_left_white_spline = (
            None,
            None,
            None,
        )

        # Red stop line
        if red_mask is not None and np.any(red_mask > 0):
            red_lines = self.extract_red_lines(red_mask)
            if red_lines:
                return np.array([[red_lines[0][0], red_lines[0][1]]])

        # Right white + yellow lane boundaries visible
        if right_white_boundary is not None and yellow_boundary is not None:
            tck_white = self.fit_spline(
                right_white_boundary, sort_by="y", collapse_fn=np.min, smoothing=50000.0
            )
            tck_yellow = self.fit_spline(
                yellow_boundary, sort_by="y", collapse_fn=np.min, smoothing=50000.0
            )
            wx, wy = self.last_white_spline = _sample_spline(tck_white)
            yx, yy = self.last_yellow_spline = _sample_spline(tck_yellow)
            center_x = (wx + yx) / 2.0
            center_y = (wy + yy) / 2.0

            if config.ENHANCED_LANE_DETECTION:
                return np.column_stack([center_x, center_y])
            else:
                try:
                    tck, _ = splprep([center_x, center_y], s=999999, k=3)
                    u_fine = np.linspace(0, 1, config.N_WAYPOINTS)
                    cx_spline, cy_spline = splev(u_fine, tck)
                    return np.column_stack([cx_spline, cy_spline])
                except Exception:
                    return np.column_stack([center_x, center_y])

        # Only one lane boundary visible
        for boundary, lane_key in [
            (right_white_boundary, "right_white"),
            (yellow_boundary, "yellow"),
            (left_white_boundary, "left_white"),
        ]:
            if boundary is not None:
                waypoints = self._project_single_spline_from_boundary(
                    boundary, H, lane_key
                )
                if waypoints is not None:
                    return waypoints

        return None

    def fit_spline(
        self,
        points: np.ndarray,
        sort_by: str = "y",
        collapse_fn: Callable[[np.ndarray], float] = np.min,
        smoothing: float = 50000.0,
    ) -> Optional[Tuple]:
        """Fit a cubic smoothing spline to a set of 2-D points.

        The function is used for **both** image-space and world-space data;
        the only differences are the coordinate that serves as the
        independent axis and the amount of smoothing.

        Args:
            points:      (N, 2) array.  In image space columns are
                         (col, row); in world space columns are
                         (x_forward, y_lateral).
            sort_by:     Which coordinate to sort by and collapse
                         duplicates on.  "y" for image space (rows are
                         the independent axis), "x" for world space
                         (forward distance is the independent axis).
            collapse_fn: Function used to collapse points that share the
                         same independent-coordinate value (e.g. np.min
                         for image-space leftmost boundary, np.median
                         for noisy world projections).
            smoothing:   s parameter passed to :func:`splprep`.  Large
                         values give a smoother (more approximated) curve.
                         Typical values: 50 000 (pixel space), 0.005 (metre
                         space).

        Returns:
            The raw tck tuple from :func:`splprep`, or None if
            fitting fails.  Callers can sample it with
            :func:`_sample_spline` or evaluate derivatives via
            splev(u, tck, der=1).
        """
        if points is None or len(points) < 4:
            return None

        x = points[:, 0]
        y = points[:, 1]

        if sort_by == "x":
            indep, dep = x, y
        else:
            indep, dep = y, x

        sort_idx = np.argsort(indep)
        indep, dep = indep[sort_idx], dep[sort_idx]

        unique_indep = np.unique(indep)
        taken_dep = np.array([collapse_fn(dep[indep == v]) for v in unique_indep])
        indep, dep = unique_indep, taken_dep

        if len(indep) < 4:
            return None

        if sort_by == "x":
            xs, ys = indep, dep
        else:
            xs, ys = dep, indep

        try:
            tck, _ = splprep([xs, ys], s=smoothing, k=3)
            return tck
        except Exception:
            return None

    def _project_single_spline_from_boundary(
        self,
        boundary: np.ndarray,
        H: np.ndarray,
        lane_key: str,
    ) -> Optional[np.ndarray]:
        """Fit a spline on a single lane boundary in **world** coordinates,
        offset it toward the lane centre, and project back to image coords.

        The un-offset world spline is also projected back to image space
        and stored in the appropriate last_*_spline member so the
        visualiser can show it.

        Args:
            boundary: (N, 2) array of boundary points in image pixel coords.
            H:        3x3 homography  image pixels -> world (x_fwd, y_lat).
            lane_key: "right_white" | "yellow" | "left_white".

        Returns:
            (N, 2) waypoint array in image pixel coords, or None.
        """
        offset_world = config.SINGLE_SPLINE_OFFSET_WORLD[lane_key]

        if len(boundary) < config.MIN_LANE_BOUNDARY_POINTS:
            return None

        img_pts = boundary.astype(np.float64)
        world_pts = image_utils.image_to_world_coords(img_pts, H)

        # Fit spline in world coordinates
        tck = self.fit_spline(
            world_pts, sort_by="x", collapse_fn=np.median, smoothing=400
        )
        if tck is None:
            return None

        # Sample the world spline & project to image for visualisation
        sx, sy = _sample_spline(tck)
        world_spline_pts = np.column_stack([sx, sy])
        img_spline_pts = image_utils.world_to_image_coords(world_spline_pts, H)

        img_spline = (img_spline_pts[:, 0], img_spline_pts[:, 1])

        if lane_key == "right_white":
            self.last_white_spline = img_spline
        elif lane_key == "yellow":
            self.last_yellow_spline = img_spline
        else:
            self.last_left_white_spline = img_spline

        # 4. Compute tangents from spline derivative
        u_fine = np.linspace(0, 1, config.SINGLE_SPLINE_N_WAYPOINTS)
        sx, sy = splev(u_fine, tck)
        dx_du, dy_du = splev(u_fine, tck, der=1)

        tangent = np.column_stack([dx_du, dy_du])
        tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent_unit = tangent / np.maximum(tangent_norm, 1e-12)

        # Left normal = rotate tangent 90° CCW:  (-ty, tx)
        left_normal = np.column_stack([-tangent_unit[:, 1], tangent_unit[:, 0]])

        # Offset toward lane centre
        offset_x = sx + left_normal[:, 0] * offset_world
        offset_y = sy + left_normal[:, 1] * offset_world

        # 5. Project offset waypoints back to image
        world_waypoints = np.column_stack([offset_x, offset_y])
        image_waypoints = image_utils.world_to_image_coords(world_waypoints, H)

        return image_waypoints[::-1]  # Reverse order (farthest first)

    def extract_red_lines(
        self, red_mask: np.ndarray
    ) -> List[Tuple[int, int, str, float]]:
        """
        Detects red stop lines in a binary mask, fits bounding boxes,
        and returns list of lines with format: (cx, cy, orientation, angle).
        They are sorted by vertical position (cy) in descending order.
        """
        if np.all(red_mask == 0):
            return []

        binary = (red_mask > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        lines = []
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < config.MIN_AREA_STOP_LINE:
                continue

            component_pixels = np.column_stack(np.where(labels == label))
            points = component_pixels[:, ::-1].astype(np.float32)  # (x, y)
            _, _, angle = cv2.minAreaRect(points)
            orientation = (
                "vertical"
                if min(np.abs(angle), angle + 90) > config.ANGLE_THRESHOLD
                else "horizontal"
            )

            cx = int(centroids[label][0])
            cy = int(centroids[label][1])
            lines.append((cx, cy, orientation, angle, area))

        # Sort by angle
        lines = sorted(lines, key=lambda l: l[1], reverse=True)
        # Format as: (cx, cy, orientation, angle)
        return [(l[0], l[1], l[2], l[3]) for l in lines]


def _sample_spline(
    tck: Tuple, n: int = config.N_WAYPOINTS
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a splrep tck tuple at n uniformly-spaced parameter values."""
    u = np.linspace(0, 1, n)
    return splev(u, tck)
