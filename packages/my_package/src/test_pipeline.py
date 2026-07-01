#!/usr/bin/env python3
"""Local test harness for SelfDrivingPipeline.

Usage:
    python test_pipeline.py <path_to_image>          # single-image mode
    python test_pipeline.py <path_to_directory>      # directory → video mode
"""

import os
import re
import sys

os.environ["LOCAL_TESTING"] = "True"

import config
import cv2
import numpy as np
from image_utils import BEVConfig, load_calibrations
from process import SelfDrivingPipeline


def _natural_sort_key(name: str) -> list:
    """Sort strings with embedded numbers naturally (e.g. frame_2 < frame_10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _load_image_paths(directory: str) -> list[str]:
    """Return sorted list of image-file paths inside *directory*."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    paths = []
    for f in os.listdir(directory):
        if os.path.splitext(f)[1].lower() in exts:
            paths.append(os.path.join(directory, f))
    paths.sort(key=lambda p: _natural_sort_key(os.path.basename(p)))
    return paths


def _build_composite_frame(
    color_vis: dict,
    bw_vis: dict,
    cell_size: tuple[int, int] | None = None,
    title: str = "",
) -> np.ndarray | None:
    """Stack every visualization into a single labelled grid image."""
    all_imgs: list[tuple[str, np.ndarray]] = []

    # colour windows
    for name in ("image", "unwarped_image", "visualization"):
        if name in color_vis and color_vis[name] is not None:
            all_imgs.append((name, color_vis[name]))

    # greyscale / false-colour windows
    for name in ("edge_mask", "white_lane_mask", "cost_heatmap", "bev_mask"):
        if name in bw_vis and bw_vis[name] is not None:
            img = bw_vis[name]
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            all_imgs.append((name, img))

    if not all_imgs:
        return None

    # decide grid dimensions (max 3 columns)
    n = len(all_imgs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    # target cell size – use first image if not given
    if cell_size is None:
        h0, w0 = all_imgs[0][1].shape[:2]
        cell_h, cell_w = h0, w0
    else:
        cell_h, cell_w = cell_size

    resized: list[np.ndarray] = []
    for label, img in all_imgs:
        r = cv2.resize(img, (cell_w, cell_h)).copy()
        cv2.putText(
            r,
            label,
            (5, cell_h - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        resized.append(r)

    # pad last row with black tiles if needed
    missing = cols * rows - n
    black = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    resized.extend([black] * missing)

    grid_rows = [np.hstack(resized[r * cols : (r + 1) * cols]) for r in range(rows)]
    composite = np.vstack(grid_rows)

    # overlay filename in top-left corner
    if title:
        cv2.putText(
            composite,
            title,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return composite


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_image_or_directory>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        image_paths = _load_image_paths(path)
        if not image_paths:
            print(f"Error: no images found in {path}")
            sys.exit(1)
        print(f"Found {len(image_paths)} images in {path}")
        is_directory = True
    else:
        image_paths = [path]
        is_directory = False

    # Initialise pipeline once
    K, D, P, H = load_calibrations(
        config.INTRINSIC_CALIBRATION_FILE,
        config.EXTRINSIC_CALIBRATION_FILE,
    )
    bev_config = BEVConfig(
        bev_size=config.BEV_SIZE, bev_resolution=config.BEV_RESOLUTION
    )
    pipeline = SelfDrivingPipeline(K, D, P, H, bev_config)

    # ------------------------------------------------------------------ single image
    if not is_directory:
        image = cv2.imread(image_paths[0])
        if image is None:
            print(f"Error: could not load image from {image_paths[0]}")
            sys.exit(1)

        vel_left, vel_right, color_vis, bw_vis = pipeline.process(image)
        print(f"vel_left={vel_left:.3f}  vel_right={vel_right:.3f}")

        for vis_dict in [color_vis, bw_vis]:
            for name, img in vis_dict.items():
                cv2.imshow(name, img)

        print("Press any key to close all windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ------------------------------------------------------------------ directory → video
    video_path = os.path.join(path, "pipeline_output.mp4")

    # figure out cell size from first image
    first_img = cv2.imread(image_paths[0])
    first_h, first_w = first_img.shape[:2]
    cell_h = min(first_h, 360)
    cell_w = min(first_w, 480)

    # dry-run first frame to obtain composite dimensions
    _, _, color_vis, bw_vis = pipeline.process(first_img)
    sample = _build_composite_frame(color_vis, bw_vis, cell_size=(cell_h, cell_w))
    if sample is None:
        print("Error: no visualisations produced by pipeline")
        sys.exit(1)

    out_h, out_w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 10.0, (out_w, out_h))

    print(f"Processing {len(image_paths)} images …")
    for i, img_path in enumerate(image_paths):
        image = cv2.imread(img_path)
        if image is None:
            print(f"  Warning: could not load {img_path}, skipping")
            continue

        vel_left, vel_right, color_vis, bw_vis = pipeline.process(image)
        composite = _build_composite_frame(
            color_vis,
            bw_vis,
            cell_size=(cell_h, cell_w),
            title=os.path.basename(img_path),
        )
        if composite is not None:
            writer.write(composite)

        print(
            f"  [{i + 1: >4}/{len(image_paths)}] {os.path.basename(img_path)}"
            f"  vel_left={vel_left:+.3f}  vel_right={vel_right:+.3f}"
        )

    writer.release()
    print(f"\nVideo saved → {video_path}")


if __name__ == "__main__":
    main()
