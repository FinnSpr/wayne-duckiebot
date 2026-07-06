import os
import cv2
import numpy as np


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


class EKF:
    """
    EKF with state x = [x, y, theta, bias] where bias is the gyro bias.

    Primary source:
      - predict_vo() uses the VO heading change (delta_theta_vo from rvec_y) and
        wheel-encoder velocity v to propagate the full state.
    Optional correction: IMU
      - update_imu() treats omega_meas as a noisy measurement of the heading
        change rate (omega = delta_theta/dt + bias) and refines theta + bias.
    Fallback: IMU-driven prediction
      - predict_imu() is used when VO fails (not enough features / bad geometry)
        so the filter never stalls between frames.
    """

    # Maximum angular velocity the robot can achieve (rad/s) plus room for noise.
    # Used to compute per-frame heading change gate: max_delta = MAX_OMEGA * dt.
    MAX_OMEGA = 20.0  # rad/s — set high enough to accommodate keypoint matching noise during sharp turns

    def __init__(self):
        self.x = np.zeros((4, 1))
        # state covariance
        self.P = np.diag([0.1, 0.1, 0.05, 0.01])
        # process noise for VO-primary prediction (higher position noise
        # since VO scale is uncertain)
        self.Q_vo = np.diag([0.005, 0.005, 0.05, 1e-5])
        # process noise for IMU-fallback prediction
        self.Q_imu = np.diag([0.005, 0.005, 0.002, 1e-5])
        # IMU measurement noise (rad/s)
        self.R_imu = np.array([[0.05]])
        # theta value at the start of the last predict step (needed by update_imu)
        self._theta_before_predict = 0.0

    def _wrap_theta(self):
        """Keep heading in (-pi, pi] — no absolute clamping since the
        robot can face any direction."""
        self.x[2, 0] = wrap_angle(self.x[2, 0])

    def predict_vo(self, v, delta_theta_vo, dt):
        """
        Primary prediction driven by VO heading change + wheel encoders.

        Uses midpoint (trapezoidal) integration: position is integrated at the
        heading halfway between theta_old and theta_new, which significantly
        reduces the discretisation error compared to Euler integration.

        Args:
            v:              linear velocity from wheel encoders (m/s)
            delta_theta_vo: heading change measured by VO this frame (rad)
            dt:             time-step (s)
        """
        theta = self.x[2, 0]
        self._theta_before_predict = theta

        theta_mid = wrap_angle(theta + delta_theta_vo / 2.0)
        theta_new = wrap_angle(theta + delta_theta_vo)

        self.x[0, 0] += v * np.cos(theta_mid) * dt
        self.x[1, 0] += v * np.sin(theta_mid) * dt
        self.x[2, 0] = theta_new
        # bias is a random-walk; no change in predict

        # Jacobian of the motion model w.r.t. [x, y, theta, bias]
        # Derivative of position update w.r.t. theta uses theta_mid
        F = np.array([
            [1, 0, -v * np.sin(theta_mid) * dt, 0],
            [0, 1,  v * np.cos(theta_mid) * dt, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.P = F @ self.P @ F.T + self.Q_vo
        self._wrap_theta()

    def predict_imu(self, v, omega_meas, dt):
        """
        Fallback prediction driven by IMU when VO is unavailable.

        Args:
            v:          linear velocity from wheel encoders (m/s)
            omega_meas: raw gyro angular velocity (rad/s)
            dt:         time-step (s)
        """
        theta = self.x[2, 0]
        bias = self.x[3, 0]
        omega = omega_meas - bias
        self._theta_before_predict = theta

        self.x[0, 0] += v * np.cos(theta) * dt
        self.x[1, 0] += v * np.sin(theta) * dt
        self.x[2, 0] = wrap_angle(theta + omega * dt)

        F = np.array([
            [1, 0, -v * np.sin(theta) * dt, 0],
            [0, 1,  v * np.cos(theta) * dt, 0],
            [0, 0, 1, -dt],
            [0, 0, 0, 1]
        ])
        self.P = F @ self.P @ F.T + self.Q_imu
        self._wrap_theta()

    def update_imu(self, omega_meas, dt):
        """
        IMU correction step (only called after a successful VO predict).

        Measurement model:
            omega_meas = delta_theta / dt + bias
                       = (theta_new - theta_old) / dt + bias
        which is linear in the state:
            H = [0, 0, 1/dt, 1]

        Args:
            omega_meas: raw gyro angular velocity (rad/s)
            dt:         time-step (s)
        """
        if dt <= 0:
            return

        delta_theta = wrap_angle(self.x[2, 0] - self._theta_before_predict)
        omega_predicted = delta_theta / dt + self.x[3, 0]

        H = np.array([[0.0, 0.0, 1.0 / dt, 1.0]])
        innovation = omega_meas - omega_predicted

        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ np.array([[innovation]])
        self.x[2, 0] = wrap_angle(self.x[2, 0])
        self.P = (np.eye(4) - K @ H) @ self.P
        self._wrap_theta()

    def get_state(self):
        return self.x.flatten()


class VisualOdometry:
    """
    ROS-free implementation of Visual Odometry processing.
    """

    def __init__(self, intrinsic_matrix, output_dir=None, save_matches=True,
                 mask_top_ratio=None):
        self.K = intrinsic_matrix
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.prev_image = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        self.ekf = EKF()
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))
        self.scale = 1.0

        self.R_latest = None
        self.t_latest = None
        self.latest_match_img = None

        self.output_dir = output_dir
        self.save_matches = save_matches
        self.mask_top_ratio = mask_top_ratio
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.trajectory = []  # (x, y, theta)

        self.camera_tilt_rad = np.pi / 12



    def _get_kp_desc(self, image):
        return self.orb.detectAndCompute(image, None)

    def process_frame(self, image, v=0.0, omega=0.0, dt=1.0, frame_idx=0,
                      scale_override=None, predict=True) -> (bool, str):
        """
        Process one frame.

        EKF update strategy:
          1. Attempt VO: compute delta_theta from the essential-matrix rotation.
          2. If VO succeeds  -> predict_vo(v, delta_theta_vo, dt)
                               + update_imu(omega, dt)  [only when omega != 0]
          3. If VO fails     -> predict_imu(v, omega, dt)  [IMU fallback]
        """
        self.latest_match_img = None

        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Apply top-masking hack if configured
        if self.mask_top_ratio is not None and self.mask_top_ratio > 0:
            mask_half_image = np.zeros(image_gray.shape, dtype=np.uint8)
            mask_half_image[
                    int(image_gray.shape[0] * self.mask_top_ratio):, :] = 255
            image_gray = cv2.bitwise_and(image_gray, image_gray,
                                         mask=mask_half_image)

        if self.prev_image is None:
            self.prev_image = image_gray
            self.prev_keypoints, self.prev_descriptors = self._get_kp_desc(
                    image_gray)
            self.trajectory.append(tuple(self.ekf.get_state()[:3]))
            return False, "First frame: no VO possible"

        # NOTE: EKF predict is now deferred until after VO computation so that
        # predict_vo() can consume the VO delta_theta directly.

        keypoints, descriptors = self._get_kp_desc(image_gray)
        matches = []
        if self.prev_descriptors is not None and descriptors is not None:
            matches = self.bf.match(self.prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda m: m.distance)
            # print("Distance [avg]:", np.mean([m.distance for m in matches]),
            #       "[min]:", np.min([m.distance for m in matches]),
            #       "[max]:", np.max([m.distance for m in matches]),
            #       "[std]:", np.std([m.distance for m in matches]))
            matches = [m for m in matches if m.distance < 50]

        if len(matches) < 8:
            # VO failed: fall back to IMU-driven prediction
            if predict:
                self.ekf.predict_imu(v, omega, dt)
            self.prev_image, self.prev_keypoints, self.prev_descriptors = image_gray, keypoints, descriptors
            self.trajectory.append(tuple(self.ekf.get_state()[:3]))
            return False, f"Frame {frame_idx}: Not enough matches ({len(matches)}) for VO"

        pts_prev = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts_prev, pts_curr, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is not None:
            mask = mask.ravel().astype(bool)
            pts_prev_inliers = pts_prev[mask]
            pts_curr_inliers = pts_curr[mask]

            if pts_prev_inliers.shape[0] < 8:
                if predict:
                    self.ekf.predict_imu(v, omega, dt)
                self.prev_image, self.prev_keypoints, self.prev_descriptors = image_gray, keypoints, descriptors
                self.trajectory.append(tuple(self.ekf.get_state()[:3]))
                return False, f"Frame {frame_idx}: Not enough inliers ({pts_prev_inliers.shape[0]}) after essential matrix estimation"

            _, R, t, mask_pose = cv2.recoverPose(
                E, pts_prev_inliers, pts_curr_inliers,
                self.K
            )

            mask_pose = mask_pose.ravel().astype(bool)
            pts_prev_final = pts_prev_inliers[mask_pose]

            if pts_prev_final.shape[0] < 8:
                if predict:
                    self.ekf.predict_imu(v, omega, dt)
                self.prev_image, self.prev_keypoints, self.prev_descriptors = image_gray, keypoints, descriptors
                self.trajectory.append(tuple(self.ekf.get_state()[:3]))
                return False, f"Frame {frame_idx}: Not enough inliers ({pts_prev_final.shape[0]}) after recoverPose"

            rvec, _ = cv2.Rodrigues(R)

            # Tilt correction: rotate R by Rx(-tilt) around camera X-axis
            ct = np.cos(self.camera_tilt_rad)
            st = np.sin(self.camera_tilt_rad)
            # Rotation matrix that un-tilts the camera: Rx(-tilt)
            Rx_inv = np.array([[1,  0,   0],
                               [0,  ct,  st],
                               [0, -st,  ct]])
            R_level = Rx_inv @ R @ Rx_inv.T   # bring into level-camera frame

            # Yaw = rotation about the camera Y-axis (pointing down = world-up)
            # For R_level, yaw is atan2(R[2,0], R[0,0])  (ZX plane)
            # equivalently atan2(-R[2,0], R[0,0]) for the standard form;
            # use the most numerically stable extraction:
            delta_theta_vo = float(np.arctan2(-R_level[2, 0], R_level[0, 0]))

            # Debug: print raw rotation components every frame to aid tuning
            # rvec_deg = np.degrees(rvec.flatten())
            # print(f"[VO] rvec(deg)=({rvec_deg[0]:.2f}, {rvec_deg[1]:.2f}, {rvec_deg[2]:.2f})  "
            #       f"delta_theta={np.degrees(delta_theta_vo):.2f}°")

            # Gate: reject VO if the single-frame heading change exceeds
            # what the robot can physically achieve at MAX_OMEGA.
            max_delta = EKF.MAX_OMEGA * dt if dt > 0 else 0.1
            vo_heading_valid = (abs(delta_theta_vo) <= max_delta)

            if predict:
                if vo_heading_valid:
                    self.ekf.predict_vo(v, delta_theta_vo, dt)
                    if omega != 0.0 and dt > 0:
                        self.ekf.update_imu(omega, dt)
                else:
                    # VO heading change too large — treat as outlier
                    self.ekf.predict_imu(v, omega, dt)

            frame_scale = scale_override if scale_override is not None else self.scale
            self.t_total = self.t_total + frame_scale * (self.R_total @ t)
            self.R_total = self.R_total @ R   # post-multiply: compose in order
            self.R_latest = R
            self.t_latest = t

            if self.save_matches:
                self.latest_match_img = cv2.drawMatches(
                    self.prev_image, self.prev_keypoints,
                    image_gray, keypoints,
                    matches[:50], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                # put text with current position
                cv2.putText(self.latest_match_img, f"Position: ({self.ekf.x[0, 0]:.2f}, {self.ekf.x[1, 0]:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                cv2.putText(self.latest_match_img, f"Heading : {np.degrees(self.ekf.x[2, 0]):.2f} deg",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                # draw arrow from center of image to the direction of the heading
                center = (self.latest_match_img.shape[1] // 4, 3 * self.latest_match_img.shape[0] // 4)
                heading_length = 50
                heading_angle = self.ekf.x[2, 0]
                heading_end = (int(center[0] - heading_length * np.sin(heading_angle)),
                               int(center[1] - heading_length * np.cos(heading_angle)))
                cv2.arrowedLine(self.latest_match_img, center, heading_end, (0, 0, 255), 2, tipLength=0.2)

                if self.output_dir:
                    out_path = os.path.join(self.output_dir, f"matches_{frame_idx:04d}.png")
                    cv2.imwrite(out_path, self.latest_match_img)

            self.prev_image, self.prev_keypoints, self.prev_descriptors = image_gray, keypoints, descriptors
            self.trajectory.append(tuple(self.ekf.get_state()[:3]))

            if False:
                cv2.imshow("Matches", self.latest_match_img)
                cv2.waitKey(0)
            return True, "Success"
        else:
            # Essential matrix estimation failed: fall back to IMU-driven prediction
            if predict:
                self.ekf.predict_imu(v, omega, dt)
            self.prev_image, self.prev_keypoints, self.prev_descriptors = image_gray, keypoints, descriptors
            self.trajectory.append(tuple(self.ekf.get_state()[:3]))
            return False, f"Frame {frame_idx}: Essential matrix estimation failed"

    def get_relative_pose(self):
        """
        Returns the delta pose between the last two EKF states as a tuple:
            (dx_world, dy_world, dtheta)   – world-frame displacement
            (ds_forward, ds_lateral, dtheta) – robot-body-frame displacement
        Returns None if fewer than 2 trajectory points exist.
        """
        if len(self.trajectory) < 2:
            return None

        x0, y0, th0 = self.trajectory[-2]
        x1, y1, th1 = self.trajectory[-1]

        dx_world = x1 - x0
        dy_world = y1 - y0
        dtheta = wrap_angle(th1 - th0)

        cos_th0, sin_th0 = np.cos(th0), np.sin(th0)
        ds_forward = cos_th0 * dx_world + sin_th0 * dy_world
        ds_lateral = -sin_th0 * dx_world + cos_th0 * dy_world

        return (dx_world, dy_world, dtheta), (ds_forward, ds_lateral, dtheta)

    def get_trajectory(self):
        return np.array(self.trajectory)

    def render_trajectory_cv2(self, canvas_size=500, margin=20):
        """
        Render the current trajectory as a BGR numpy array using only OpenCV.
        Thread-safe (no matplotlib). Suitable for publishing via ROS from any thread.

        Args:
            canvas_size: side length of the square output image in pixels.
            margin:      pixel margin around the drawn path.
        Returns:
            BGR numpy array of shape (canvas_size, canvas_size, 3), or None if
            fewer than 2 trajectory points exist.
        """
        traj = self.get_trajectory()
        if len(traj) < 2:
            return None

        xs, ys = traj[:, 0], traj[:, 1]

        # --- normalise world coordinates to pixel space ---
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_range = max(x_max - x_min, 1e-3)
        y_range = max(y_max - y_min, 1e-3)
        scale = (canvas_size - 2 * margin) / max(x_range, y_range)

        def to_px(x, y):
            px = int(margin + (x - x_min) * scale)
            # flip y so that "forward" is up on screen
            py = int(canvas_size - margin - (y - y_min) * scale)
            return (px, py)

        img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

        # draw grid lines
        for i in range(0, canvas_size, canvas_size // 4):
            cv2.line(img, (i, 0), (i, canvas_size), (40, 40, 40), 1)
            cv2.line(img, (0, i), (canvas_size, i), (40, 40, 40), 1)

        # draw path
        pts = [to_px(x, y) for x, y in zip(xs, ys)]
        for i in range(1, len(pts)):
            # colour shifts from blue (start) to green (end)
            t = i / max(len(pts) - 1, 1)
            colour = (int(255 * (1 - t)), int(255 * t), 0)
            cv2.line(img, pts[i - 1], pts[i], colour, 2)

        # draw heading arrows at every ~10th point
        step = max(1, len(traj) // 10)
        for i in range(0, len(traj), step):
            x, y, theta = traj[i]
            px, py = to_px(x, y)
            arrow_len = int(0.05 * scale)  # 5 cm in world → pixels
            ex = px + int(arrow_len * np.cos(theta))
            ey = py - int(arrow_len * np.sin(theta))  # y flipped
            cv2.arrowedLine(img, (px, py), (ex, ey), (0, 0, 255), 1,
                            tipLength=0.4)

        # start / end markers
        cv2.circle(img, pts[0],  5, (255, 255, 0), -1)   # cyan  = start
        cv2.circle(img, pts[-1], 5, (0,   255, 255), -1)  # yellow = end

        # text overlay
        state = self.ekf.get_state()
        cv2.putText(img,
                    f"Pos: ({state[0]:.2f}, {state[1]:.2f}) m",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img,
                    f"Hdg: {np.degrees(state[2]):.1f} deg",
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img,
                    f"Frames: {len(traj)}",
                    (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        return img

    def visualize_trajectory(self, trajectory=None, title="Trajectory"):
        """
        Generates an image of the trajectory to be used for publication via ROS or debugging
        otherwise, returns a figure with the trajectory plotted. Shows arrow indicating the direction of motion at the last point.
        """
        import matplotlib.pyplot as plt

        if trajectory is None:
            trajectory = self.get_trajectory()

        fig, ax = plt.subplots()
        ax.plot(trajectory[:, 0], trajectory[:, 1], marker='o', markersize=2)
        for i in range(0, len(trajectory), max(1, len(trajectory) // 20)):
            x, y, theta = trajectory[i]
            dx = 0.1 * np.cos(theta)
            dy = 0.1 * np.sin(theta)
            ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r')

        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_title(title)
        ax.axis('equal')
        ax.grid(True)
        return fig

    def run(self, image_paths, sensor_rows=None):
        """
        Limited ROS-free run loop for processing a sequence of images
        with optional sensor data - see offline_odometry.py
        """
        from tqdm import tqdm
        for idx, path in tqdm(enumerate(image_paths),
                              total=len(image_paths),
                              desc="Processing frames"):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Could not read image: {path}")
                continue

            v, omega, dt = 0.0, 0.0, 1.0
            if sensor_rows and idx < len(sensor_rows):
                v = float(sensor_rows[idx].get("v", 0.0))
                omega = float(sensor_rows[idx].get("omega", 0.0))
                dt = float(sensor_rows[idx].get("dt", 0.1))

            self.process_frame(img, v=v, omega=omega, dt=dt, frame_idx=idx, predict=True)

        return self.trajectory
