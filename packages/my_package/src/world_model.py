import cv2
import numpy as np
from typing import Tuple, List, Optional
from scipy.interpolate import splprep, splev
import config

class WorldModel:
    """
    World Model / State Estimator Module.
    Responsible for spline fitting and structuring feature observations (such as red stop lines).
    """
    def __init__(self):
        pass

    def fit_spline(self, mask: np.ndarray, take_leftmost_pixels: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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

    def extract_red_lines(self, red_mask: np.ndarray) -> List[Tuple[int, int, str, float]]:
        """
        Detects red stop lines in a binary mask, fits bounding boxes,
        and returns list of lines with format: (cx, cy, orientation, angle).
        """
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
            points = component_pixels[:, ::-1].astype(np.float32) # (x, y)
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
