from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import yaml


@dataclass
class BEVConfig:
    """
    Defines the physical extent and resolution of the BEV output image.

    bev_size:       (width_m, height_m) — physical region to observe, in meters.
                    width  = left/right extent, centered on robot x-axis.
                    height = forward extent, starting from robot position.
                    e.g. (0.6, 0.8) → 0.6m wide, 0.8m ahead.

    bev_resolution: meters per pixel in the BEV image.
                    e.g. 0.02 → each pixel = 2cm × 2cm on the ground.

    Derived:
        bev_w_px = int(bev_size[0] / bev_resolution)
        bev_h_px = int(bev_size[1] / bev_resolution)

    Pixel (u, v) corresponds to ground point:
        x_m = (u - bev_w_px/2) * bev_resolution   (+ = right,  - = left)
        y_m = (bev_h_px - v)   * bev_resolution   (+ = ahead,  v=0 = farthest row)
    """

    bev_size: Tuple[float, float]  # (width_m, height_m)
    bev_resolution: float  # meters per pixel

    @property
    def bev_w_px(self) -> int:
        return int(self.bev_size[0] / self.bev_resolution)

    @property
    def bev_h_px(self) -> int:
        return int(self.bev_size[1] / self.bev_resolution)

    @property
    def bev_shape(self) -> Tuple[int, int]:
        """(height_px, width_px) — numpy convention."""
        return (self.bev_h_px, self.bev_w_px)

    def pixel_to_metric(self, u: float, v: float) -> Tuple[float, float]:
        """BEV pixel (u, v) → ground plane (x_forward, y_right)."""
        x_m = (self.bev_h_px - v) * self.bev_resolution  # forward
        y_m = -(u - self.bev_w_px / 2) * self.bev_resolution  # left (positive = left)
        return x_m, y_m

    def metric_to_pixel(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """Ground plane (x_forward, y_right) → BEV pixel (u, v)."""
        u = -y_m / self.bev_resolution + self.bev_w_px / 2
        v = self.bev_h_px - x_m / self.bev_resolution
        return u, v


def load_calibrations(
    intrinsic_path: str,
    extrinsic_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load camera intrinsic and extrinsic calibration from YAML files.

    Args:
        intrinsic_path: Path to intrinsic.yaml (camera matrix, distortion, projection).
        extrinsic_path: Path to extrinsic.yaml (homography to ground plane).

    Returns:
        K: 3x3 camera intrinsic matrix.
        D: distortion coefficients (1-D array, length 5).
        P: 3x4 projection matrix.
        H: 3x3 homography matrix mapping image pixels → ground plane (metric, not pixels).
    """
    with open(intrinsic_path, "r") as f:
        intr = yaml.safe_load(f)

    K = np.array(intr["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(intr["distortion_coefficients"]["data"], dtype=np.float64)
    P = np.array(intr["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)

    with open(extrinsic_path, "r") as f:
        extr = yaml.safe_load(f)

    H_raw = np.array(extr["homography"], dtype=np.float64).reshape(3, 3)
    K_unwarped = P[:, :3]
    H = H_raw @ np.linalg.inv(K_unwarped)  # H: image pixels → metric ground plane

    return K, D, P, H


def unwarp_image(
    image: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    P: np.ndarray,
    reuse_maps: bool = True,
) -> np.ndarray:
    """
    Undistort (rectify) an image to remove lens distortion.

    Uses cv2.initUndistortRectifyMap + cv2.remap, following the same
    approach as visual_lane_servoing_node.py.

    Args:
        image: BGR image from the Duckiebot camera (H x W x 3).
        K:     3x3 camera intrinsic matrix.
        D:     distortion coefficients.
        P:     3x4 projection matrix.
        reuse_maps: If True, reuse the previously computed undistortion maps.

    Returns:
        unwarped: undistorted BGR image.
    """
    h, w = image.shape[:2]
    if not hasattr(unwarp_image, "_mapx") or not reuse_maps:
        unwarp_image._mapx, unwarp_image._mapy = cv2.initUndistortRectifyMap(
            K, D, None, P[:, :3], (w, h), cv2.CV_32FC1
        )
    unwarped = cv2.remap(
        image, unwarp_image._mapx, unwarp_image._mapy, cv2.INTER_NEAREST
    )

    return unwarped


def build_image_to_bev_homography(
    H_image_to_metric: np.ndarray, bev_cfg: BEVConfig
) -> np.ndarray:
    """
    Combine the extrinsic homography (image pixels → metric ground plane)
    with the BEV pixel scaling (metric → BEV image pixels).

    H_image_to_metric:  3×3 homography from camera calibration.
                        Maps image pixel (u_img, v_img, 1) →
                        metric ground point (x_m, y_m, w) via:
                            p_metric = H_image_to_metric @ p_img
                            x_m = p_metric[0] / p_metric[2]
                            y_m = p_metric[1] / p_metric[2]

    Returns H_image_to_bev: 3×3 homography mapping image pixels directly
                             to BEV image pixels.
    """
    res = bev_cfg.bev_resolution
    W = bev_cfg.bev_w_px
    H = bev_cfg.bev_h_px

    # TODO: Check this magic matrix
    S = np.array(
        [
            [0.0, -1.0 / res, W / 2.0],  # u from y_m
            [-1.0 / res, 0.0, float(H)],  # v from x_m
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return S @ H_image_to_metric


def mask_to_positions(mask: np.ndarray) -> np.ndarray:
    """
    Extract (row, col) pixel coordinates of all nonzero elements in a mask.

    Args:
        mask: 2-D binary mask (any nonzero value counts as foreground).

    Returns:
        (N, 2) int64 array where each row is [row, col] (y, x in image space).
        Returns an empty (0, 2) array if the mask is entirely zero.
    """
    return np.argwhere(mask)


def positions_to_mask(positions: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Reconstruct a binary mask from a set of (row, col) positions.

    Args:
        positions: (N, 2) int array of [row, col] coordinates.
        shape:      (height, width) of the output mask.

    Returns:
        uint8 binary mask of the given shape with 255 at the supplied positions.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if positions.size == 0:
        return mask
    rows = positions[:, 0].astype(int)
    cols = positions[:, 1].astype(int)
    mask[rows, cols] = 255
    return mask


def world_to_bev_coords(points: np.ndarray, bev_cfg: "BEVConfig") -> np.ndarray:
    """
    Project points from world/metric coordinates to BEV image pixel coordinates.

    Args:
        points:  (N, 2) array of world points (x_forward, y_lateral).
                 x_forward >= 0 is ahead of the robot.
                 y_lateral > 0 is to the left of the robot.
        bev_cfg: BEVConfig defining the BEV extent and resolution.

    Returns:
        (N, 2) float64 array of BEV pixel coordinates (u, v).
        u = column (0 … bev_w_px-1), v = row (0 … bev_h_px-1).
        Points outside the BEV region will map to out-of-bounds pixel values.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) points, got {points.shape}")

    u = -points[:, 1] / bev_cfg.bev_resolution + bev_cfg.bev_w_px / 2.0
    v = bev_cfg.bev_h_px - points[:, 0] / bev_cfg.bev_resolution
    return np.column_stack([u, v])


def bev_to_world_coords(points: np.ndarray, bev_cfg: "BEVConfig") -> np.ndarray:
    """
    Project points from BEV image pixel coordinates back to world/metric coordinates.

    Args:
        points:  (N, 2) array of BEV pixel coordinates (u, v).
                 u = column, v = row.
        bev_cfg: BEVConfig defining the BEV extent and resolution.

    Returns:
        (N, 2) float64 array of world points (x_forward, y_lateral).
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) points, got {points.shape}")

    x_forward = (bev_cfg.bev_h_px - points[:, 1]) * bev_cfg.bev_resolution
    y_lateral = -(points[:, 0] - bev_cfg.bev_w_px / 2.0) * bev_cfg.bev_resolution
    return np.column_stack([x_forward, y_lateral])


def image_to_world_coords(
    points: np.ndarray, H_image_to_metric: np.ndarray
) -> np.ndarray:
    """
    Project image pixel coordinates to world/metric ground-plane coordinates.

    Args:
        points:             (N, 2) array of image pixel coordinates (u, v).
                            u = column, v = row.
        H_image_to_metric:  3×3 homography from camera extrinsic calibration.
                            Maps image homogeneous → metric (x_forward, y_lateral).

    Returns:
        (N, 2) float64 array of world points (x_forward, y_lateral).
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) points, got {points.shape}")

    homogeneous = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    projected = homogeneous @ H_image_to_metric.T  # (N, 3)
    w = projected[:, 2:3]
    w_safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    world = projected[:, :2] / w_safe
    return world


def world_to_image_coords(
    points: np.ndarray, H_image_to_metric: np.ndarray
) -> np.ndarray:
    """
    Project world/metric ground-plane coordinates back to image pixel coordinates.

    Args:
        points:             (N, 2) array of world points (x_forward, y_lateral).
        H_image_to_metric:  3×3 homography from camera extrinsic calibration.

    Returns:
        (N, 2) float64 array of image pixel coordinates (u, v).
        Points behind the camera or at infinity are clipped at w=1e-12.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) points, got {points.shape}")

    H_metric_to_image = np.linalg.inv(H_image_to_metric)
    homogeneous = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    projected = homogeneous @ H_metric_to_image.T
    w = projected[:, 2:3]
    w_safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    image = projected[:, :2] / w_safe
    return image


def image_to_bev_coords(
    points: np.ndarray,
    H_image_to_metric: np.ndarray,
    bev_cfg: "BEVConfig",
) -> np.ndarray:
    """
    Project image pixel coordinates directly to BEV image pixel coordinates.

    This is a composition of image → world (homography) and world → BEV
    (metric scaling).  Equivalent to calling image_to_world_coords followed
    by world_to_bev_coords.

    Args:
        points:             (N, 2) array of image pixel coordinates (u, v).
        H_image_to_metric:  3×3 homography from camera extrinsic calibration.
        bev_cfg:            BEVConfig defining the BEV extent and resolution.

    Returns:
        (N, 2) float64 array of BEV pixel coordinates (u_bev, v_bev).
    """
    world = image_to_world_coords(points, H_image_to_metric)
    return world_to_bev_coords(world, bev_cfg)


def project_mask_to_bev(
    mask: np.ndarray,
    H_image_to_metric: np.ndarray,
    bev_cfg: BEVConfig,
) -> np.ndarray:
    """
    Project a segmentation mask into Bird's Eye View (BEV) image space.

    Args:
        mask:
            (H_img, W_img) uint8 binary mask from the segmentation model.
            Foreground pixels should be 255, background 0.

        H_image_to_metric:
            3×3 homography matrix from extrinsic camera calibration.
            Maps image-space homogeneous coordinates to metric ground-plane
            coordinates (robot-centric, x=right, y=forward, origin=robot).

        bev_cfg: BEVConfig.

    Returns:
        bev_mask: (bev_h_px, bev_w_px) uint8 binary BEV image.
    """
    H_image_to_bev = build_image_to_bev_homography(H_image_to_metric, bev_cfg)

    # warpPerspective maps each *output* pixel back through H^{-1} to find
    # its source — so we pass H_image_to_bev directly (not its inverse).
    # INTER_NEAREST: binary mask, no interpolation artifacts.
    # BORDER_CONSTANT with 0: pixels outside source image → background.
    bev_mask = cv2.warpPerspective(
        mask,
        H_image_to_bev,
        (bev_cfg.bev_w_px, bev_cfg.bev_h_px),  # (width, height) — OpenCV convention
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return bev_mask


def get_bev_heatmap_image(
    function_world_coords: callable,
    bev_cfg: BEVConfig,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Rasterize a vectorized scalar function over the BEV region into a
    colour-mapped BGR image suitable for ``cv2.imshow``.

    For every pixel ``(u, v)`` in the BEV image the corresponding world
    coordinate is computed via :func:`bev_to_world_coords` and
    ``function_world_coords`` is evaluated on the entire set of points at
    once (the function must be vectorized: ``(N, 2) → (N,)``).

    Finite values are normalised to ``[0, 255]`` and mapped through the
    chosen OpenCV colormap.  Pixels whose value is ``+∞``, ``-∞`` or
    ``NaN`` are rendered as **black** (0, 0, 0).

    Args:
        function_world_coords:
            Vectorized callable ``f(points)`` where ``points`` has shape
            ``(N, 2)`` (columns: ``x_forward, y_lateral``) and returns a
            1-D ``np.ndarray`` of ``N`` scalars.  Example::

                def gaussian(points):
                    return np.exp(-np.sum(points ** 2, axis=1))

        bev_cfg:
            BEVConfig defining the extent and resolution of the output.
        colormap:
            OpenCV colormap flag (default ``cv2.COLORMAP_JET``).

    Returns:
        ``np.ndarray`` of shape ``(bev_h_px, bev_w_px, 3)`` with dtype
        ``uint8``, ready for display or saving as an image.
    """
    W, H = bev_cfg.bev_w_px, bev_cfg.bev_h_px

    # Build the full grid of BEV pixel coordinates (u, v).
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))  # both (H, W)
    pixels = np.column_stack([u_grid.ravel(), v_grid.ravel()])  # (N, 2)

    # Convert every pixel to its world-coordinate equivalent.
    world = bev_to_world_coords(pixels, bev_cfg)

    # Evaluate the vectorized function on all points at once.
    values = np.asarray(function_world_coords(world), dtype=np.float64).reshape(H, W)

    # Identify pixels that should be black.
    inf_mask = ~np.isfinite(values)  # True for ±inf and NaN

    # Normalise the *finite* portion to [0, 255].
    finite = values[~inf_mask]
    if finite.size == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    vmin, vmax = finite.min(), finite.max()
    if vmax - vmin < 1e-12:
        norm = np.zeros_like(finite, dtype=np.uint8)
    else:
        norm = ((finite - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # Build a uint8 single-channel image for the colormap.
    gray = np.zeros((H, W), dtype=np.uint8)
    gray[~inf_mask] = norm

    # Apply colormap, then stamp black over inf pixels.
    colour = cv2.applyColorMap(gray, colormap)
    colour[inf_mask] = (0, 0, 0)

    return colour
