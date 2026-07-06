import argparse
import glob
import os
import csv as csvlib
import numpy as np

from visual_odometry import VisualOdometry


class OfflineVisualOdometry(VisualOdometry):
    """
    ROS-free re-implementation of VisualOdometryNode.cb_img / cb_imu logic.
    Inherits from the core VisualOdometry class with mask_top_ratio default.
    """

    def __init__(self, intrinsic_matrix, output_dir=None, save_matches=True):
        super().__init__(
            intrinsic_matrix=intrinsic_matrix,
            output_dir=output_dir,
            save_matches=save_matches,
            mask_top_ratio=0.45
        )


def load_sensor_csv(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        reader = csvlib.DictReader(f)
        return list(reader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="dir of seq images")
    parser.add_argument("--pattern", default="*.jpg", help="image glob pattern")
    parser.add_argument("--fx", type=float, default=307.73792946)
    parser.add_argument("--fy", type=float, default=314.98277734)
    parser.add_argument("--cx", type=float, default=329.69236795)
    parser.add_argument("--cy", type=float, default=244.46055889)
    parser.add_argument("--output_dir", default="out")
    parser.add_argument("--sensor_csv", default=None, help="Optional CSV with v,omega,dt per frame")
    args = parser.parse_args()

    image_paths = sorted(glob.glob(os.path.join(args.images_dir, args.pattern)))
    if not image_paths:
        raise SystemExit(f"No images found in {args.images_dir} matching {args.pattern}")

    K = np.array([
        [args.fx, 0, args.cx],
        [0, args.fy, args.cy],
        [0, 0, 1]
    ])

    sensor_rows = load_sensor_csv(args.sensor_csv)

    # vo = OfflineVisualOdometry(K, output_dir=args.output_dir)
    vo = OfflineVisualOdometry(K)
    trajectory = vo.run(image_paths, sensor_rows=sensor_rows)

    print("\nFinal EKF state [x, y, theta, bias]:", vo.ekf.get_state())
    print("Final R_total:\n", vo.R_total)
    print("Final t_total (unit scale unless overridden):\n", vo.t_total.ravel())

    traj_path = os.path.join(args.output_dir, "trajectory.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(traj_path, "w", newline="") as f:
        writer = csvlib.writer(f)
        writer.writerow(["frame", "x", "y", "theta"])
        for i, (x, y, theta) in enumerate(trajectory):
            writer.writerow([i, x, y, theta])
    print(f"Trajectory saved to {traj_path}")


if __name__ == "__main__":
    main()
