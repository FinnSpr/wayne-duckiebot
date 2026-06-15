#!/usr/bin/env python3

"""
Test script for the lane-following pipeline.
Loads an image, runs process_all, and saves all output masks/visualizations
to the results/ folder.
"""

import sys
import os
import cv2
from process import process_all


class DummyData:
    """Minimal data wrapper compatible with process_all's expected interface."""

    def __init__(self, image):
        self._image = image
        self._unwarped_image = None


def main(image_path: str):
    # --- 1. Load image ---
    if not os.path.exists(image_path):
        print(f"Error: image not found at '{image_path}'")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: failed to read image at '{image_path}'")
        sys.exit(1)

    print(f"Loaded image: {image.shape}")

    # --- 2. Run the pipeline ---
    data = DummyData(image)
    result = process_all(data)

    # process_all returns a tuple of 7 values (or 3 on failure)
    if len(result) == 3:
        # No lanes detected — only vel_left, vel_right, visualization returned
        vel_left, vel_right, visualization = result
        edge_mask = None
        white_lane_mask = None
        yellow_mask = None
        red_mask = None
        white_color = None
    else:
        (
            vel_left,
            vel_right,
            visualization,
            edge_mask,
            white_lane_mask,
            yellow_mask,
            red_mask,
            white_color,
        ) = result

    print(f"Wheel commands: left={vel_left:.3f}, right={vel_right:.3f}")

    # --- 3. Save results ---
    # Results folder is relative to the package root (one level up from src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    def save_img(filename, img):
        path = os.path.join(results_dir, filename)
        cv2.imwrite(path, img)
        print(f"Saved: {path}")

    save_img("visualization.png", visualization)
    if edge_mask is not None:
        save_img("edge_mask.png", edge_mask)
    if white_lane_mask is not None:
        save_img("white_mask.png", white_lane_mask)
    if yellow_mask is not None:
        save_img("yellow_mask.png", yellow_mask)
    if white_color is not None:
        save_img("white_color.png", white_color)
    if red_mask is not None:
        save_img("red_mask.png", red_mask)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <image_path>")
        sys.exit(1)

    main(sys.argv[1])
