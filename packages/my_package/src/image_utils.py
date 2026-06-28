import cv2
import numpy as np
import yaml


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
        H: 3x3 homography matrix mapping image pixels → ground plane (BEV).
    """
    with open(intrinsic_path, "r") as f:
        intr = yaml.safe_load(f)

    K = np.array(intr["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(intr["distortion_coefficients"]["data"], dtype=np.float64)
    P = np.array(intr["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)

    with open(extrinsic_path, "r") as f:
        extr = yaml.safe_load(f)

    H = np.array(extr["homography"], dtype=np.float64).reshape(3, 3)

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
