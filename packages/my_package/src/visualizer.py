from typing import Optional, Tuple

import cv2
import numpy as np


def draw_trajectory(
    image: np.ndarray,
    trajectory: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 0),
    radius: int = 3,
    thickness: int = 2,
) -> np.ndarray:
    """
    Overlay a trajectory on an RGB image.

    Draws dots at each waypoint connected by lines.

    Args:
        image: (H, W, 3) uint8 BGR/RGB image.
        trajectory: (N, 2) array of pixel coordinates (u, v) = (col, row).
        color: BGR tuple for the overlay (default black).
        radius: Radius of the waypoint dots.
        thickness: Thickness of the connecting lines.

    Returns:
        A new (H, W, 3) image with the trajectory drawn on it.
    """
    out = image.copy()
    pts = np.atleast_2d(np.asarray(trajectory, dtype=np.int32))
    if pts.shape[0] < 2:
        return out

    for i in range(len(pts) - 1):
        cv2.line(out, tuple(pts[i]), tuple(pts[i + 1]), color, thickness)
    for i in range(len(pts)):
        cv2.circle(out, tuple(pts[i]), radius, color, -1)

    return out


class Visualizer:
    """
    Visualizer Module.
    Generates debugging overlays showing detected lanes, splines, and planned waypoints.
    """

    def __init__(self):
        pass

    def visualize(
        self,
        image: np.ndarray,
        white_mask: np.ndarray,
        yellow_mask: np.ndarray,
        red_mask: np.ndarray,
        white_spline: Optional[Tuple[np.ndarray, np.ndarray]],
        yellow_spline: Optional[Tuple[np.ndarray, np.ndarray]],
        waypoints: Optional[np.ndarray],
    ) -> np.ndarray:
        # Dim everything to 15%, then restore lane pixels to full brightness
        vis = (image * 0.15).astype(np.uint8)
        if white_mask is not None:
            vis[white_mask > 0] = image[white_mask > 0]
        if yellow_mask is not None:
            vis[yellow_mask > 0] = image[yellow_mask > 0]
        if red_mask is not None:
            vis[red_mask > 0] = image[red_mask > 0]

        if white_spline is not None:
            wx, wy = white_spline
            for i in range(len(wx) - 1):
                cv2.line(
                    vis,
                    (int(wx[i]), int(wy[i])),
                    (int(wx[i + 1]), int(wy[i + 1])),
                    (255, 0, 0),
                    2,
                )

        if yellow_spline is not None:
            yx, yy = yellow_spline
            for i in range(len(yx) - 1):
                cv2.line(
                    vis,
                    (int(yx[i]), int(yy[i])),
                    (int(yx[i + 1]), int(yy[i + 1])),
                    (0, 255, 0),
                    2,
                )

        if waypoints is not None:
            for x, y in waypoints:
                cv2.circle(vis, (int(x), int(y)), 4, (255, 191, 0), -1)

        return vis
