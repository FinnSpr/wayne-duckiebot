#!/usr/bin/env python3
"""Local test harness for SelfDrivingPipeline.

Usage:
    python test_pipeline.py <path_to_image>
"""

import os
import sys

os.environ["LOCAL_TESTING"] = "True"

import cv2

import config
from image_utils import BEVConfig, load_calibrations
from process import SelfDrivingPipeline


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not load image from {image_path}")
        sys.exit(1)

    K, D, P, H = load_calibrations(
        config.INTRINSIC_CALIBRATION_FILE,
        config.EXTRINSIC_CALIBRATION_FILE,
    )
    bev_config = BEVConfig(
        bev_size=config.BEV_SIZE, bev_resolution=config.BEV_RESOLUTION
    )
    pipeline = SelfDrivingPipeline(K, D, P, H, bev_config)
    vel_left, vel_right, color_vis, bw_vis = pipeline.process(image)

    print(f"vel_left={vel_left:.3f}  vel_right={vel_right:.3f}")

    for vis_dict in [color_vis, bw_vis]:
        for name, img in vis_dict.items():
            cv2.imshow(name, img)

    print("Press any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
