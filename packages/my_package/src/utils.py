# Will be Mostly Replaced

import yaml
import numpy as np


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
    H = H_raw @ np.linalg.inv(K_unwarped)

    return K, D, P, H
