#!/usr/bin/env python3

"""
Camera calibration and image unwarping utilities for Duckiebot.

Provides:
    load_calibrations(camera_info_msg) -> K, D
    unwarp_image(image, K, D) -> unwarped_image
"""

import cv2
import numpy as np
from typing import Tuple


def load_calibrations(camera_info_msg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract camera intrinsic calibration from a ROS CameraInfo message.

    Args:
        camera_info_msg: sensor_msgs.msg.CameraInfo message

    Returns:
        K: 3x3 camera intrinsic matrix
        D: distortion coefficients (1-d array)
    """
    K = np.array(camera_info_msg.K, dtype=np.float64).reshape(3, 3)
    D = np.array(camera_info_msg.D, dtype=np.float64)
    return K, D


def unwarp_image(
    image: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """
    Undistort (rectify) an image to remove lens distortion.

    Uses cv2.initUndistortRectifyMap + cv2.remap, following the same
    approach as visual_lane_servoing_node.py.

    Args:
        image: BGR image from the Duckiebot camera (H x W x 3).
        K:     3x3 camera intrinsic matrix.
        D:     distortion coefficients.

    Returns:
        unwarped: undistorted BGR image.
    """
    h, w = image.shape[:2]

    # Compute the optimal new camera matrix (alpha=0 → keep all valid pixels)
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        K, D, (w, h), alpha=0.0
    )

    # Build the undistortion maps
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, D, None, new_camera_matrix, (w, h), cv2.CV_32FC1
    )

    # Undistort
    unwarped = cv2.remap(image, mapx, mapy, cv2.INTER_NEAREST)

    return unwarped
