from dataclasses import dataclass

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

    bev_size: tuple[float, float]  # (width_m, height_m)
    bev_resolution: float  # meters per pixel

    @property
    def bev_w_px(self) -> int:
        return int(self.bev_size[0] / self.bev_resolution)

    @property
    def bev_h_px(self) -> int:
        return int(self.bev_size[1] / self.bev_resolution)

    @property
    def bev_shape(self) -> tuple[int, int]:
        """(height_px, width_px) — numpy convention."""
        return (self.bev_h_px, self.bev_w_px)

    def pixel_to_metric(self, u: float, v: float) -> tuple[float, float]:
        """BEV pixel (u, v) → ground plane (x_forward, y_right)."""
        x_m = (self.bev_h_px - v) * self.bev_resolution  # forward
        y_m = -(u - self.bev_w_px / 2) * self.bev_resolution  # left (positive = left)
        return x_m, y_m

    def metric_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Ground plane (x_forward, y_right) → BEV pixel (u, v)."""
        u = -y_m / self.bev_resolution + self.bev_w_px / 2
        v = self.bev_h_px - x_m / self.bev_resolution
        return u, v


def load_calibrations(
    intrinsic_path: str,
    extrinsic_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
