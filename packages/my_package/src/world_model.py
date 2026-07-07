from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import splev, splprep

import config


class WorldModel:
    """
    World Model / State Estimator Module.
    Responsible for spline fitting and structuring feature observations (such as red stop lines).
    """

    def __init__(self):
        pass

    def fit_spline(
        self, mask: np.ndarray, take_leftmost_pixels: bool = True
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Fits a cubic spline to the non-zero pixels in a binary mask."""
        ys, xs = np.where(mask > 0)

        if len(xs) < config.MIN_LANE_PIXELS:
            return None

        # Sort by y (top to bottom in image)
        sort_idx = np.argsort(ys)
        xs, ys = xs[sort_idx], ys[sort_idx]

        # In standard mode, we downsample
        if not config.ENHANCED_LANE_DETECTION:
            step = max(1, len(xs) // 100)
            xs, ys = xs[::step], ys[::step]

        take_fn = np.min if take_leftmost_pixels else np.max

        # Remove duplicate y values which cause splprep to fail
        unique_ys = np.unique(ys)
        taken_xs = np.array([take_fn(xs[ys == y]) for y in unique_ys])
        xs, ys = taken_xs, unique_ys

        if len(xs) < 4:
            return None

        try:
            tck, _ = splprep([xs, ys], s=50000, k=3)
            u_fine = np.linspace(0, 1, config.N_WAYPOINTS)
            x_spline, y_spline = splev(u_fine, tck)
            return x_spline, y_spline
        except Exception:
            return None

    def get_drive_waypoints(
        self,
        white_spline: Optional[Tuple[np.ndarray, np.ndarray]],
        yellow_spline: Optional[Tuple[np.ndarray, np.ndarray]],
        image_width: int,
        image_height: int,
        red_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Compute lane-following waypoints for normal driving.

        If ``red_mask`` is provided (not None), waypoints will point at the
        nearest horizontal red stop line instead of following the lane centre.
        Pass ``None`` to ignore red lines (used by CROSS state).

        Returns:
            (N, 2) waypoint array or ``None`` when no usable lane data exists.
        """
        # Red stop line override (only when red_mask is given)
        # TODO: MASSIVE TODO, this is so so bad
        if red_mask is not None and np.any(red_mask > 0):
            red_lines = self.extract_red_lines(red_mask)
            if red_lines:
                horiz = [l for l in red_lines if l[2] == "horizontal"]
                if horiz:
                    current_line = max(
                        horiz,
                        key=lambda l: l[1],
                        default=(int(image_width / 2), 0),
                    )
                    return np.array([[current_line[0], current_line[1]]])

        # Two-lane case: centre of white + yellow splines
        if white_spline is not None and yellow_spline is not None:
            wx, wy = white_spline
            yx, yy = yellow_spline
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

        # Single-lane case: project from one spline
        if white_spline is not None:
            return self._project_single_spline(
                white_spline, image_width, image_height, is_white=True
            )
        if yellow_spline is not None:
            return self._project_single_spline(
                yellow_spline, image_width, image_height, is_white=False
            )

        return None

    def _project_single_spline(
        self,
        spline: Tuple[np.ndarray, np.ndarray],
        image_width: int,
        image_height: int,
        is_white: bool,
    ) -> np.ndarray:
        """Project waypoints using a single lane spline estimation."""
        sx, sy = spline
        if config.ENHANCED_LANE_DETECTION:
            if is_white:
                t_x = sx / image_width
                scale = config.SINGLE_LANE_SCALE_FACTOR_WHITE
                center_x = sx - sx * t_x * scale
            else:
                t_x = 1 - sx / image_width
                scale = config.SINGLE_LANE_SCALE_FACTOR_YELLOW
                center_x = sx + (image_width - sx) * t_x * scale

            t_y = 1 - sy / image_height
            center_y = sy + (image_height - sy) * t_y * scale
            return np.column_stack([center_x, center_y])
        else:
            factor = (
                config.IMAGE_WIDTH_OFFSET_FACTOR_WHITE
                if is_white
                else config.IMAGE_WIDTH_OFFSET_FACTOR_YELLOW
            )
            offset = image_width * factor
            return np.column_stack([sx - offset if is_white else sx + offset, sy])

    def extract_red_lines(
        self, red_mask: np.ndarray
    ) -> List[Tuple[int, int, str, float]]:
        """
        Detects red stop lines in a binary mask, fits bounding boxes,
        and returns list of lines with format: (cx, cy, orientation, angle).
        """
        # TODO: This is so bad
        if np.all(red_mask == 0):
            return []

        binary = (red_mask > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        lines = []
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < config.MIN_AREA:
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

        # Keep only the five largest horizontal/vertical markings
        lines = sorted(lines, key=lambda l: l[3], reverse=True)[:5]
        # Format as: (cx, cy, orientation, angle)
        return [(l[0], l[1], l[2], l[3]) for l in lines]
