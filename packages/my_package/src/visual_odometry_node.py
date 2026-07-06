#!/usr/bin/env python3
import os
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Imu, CameraInfo
from duckietown_msgs.msg import WheelEncoderStamped

import config
from pinholecamera import PinholeCameraModel
from image_utils import load_img_from_msg
from visual_odometry import VisualOdometry


ENABLE_VISUAL_MATCHES = True


class VisualOdometryNode(DTROS):
    def __init__(self, node_name):
        super(VisualOdometryNode, self).__init__(
                node_name=node_name, node_type=NodeType.GENERIC)
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self.bridge = CvBridge()
        self.camera_model = None

        # VO after camera info is received
        self.vo = None

        # Wheel Encoder : scale estimation
        self.wheel_radius = config.WHEEL_RADIUS
        self.l_whl_res, self.r_whl_res = None, None
        self.left_ticks = None
        self.right_ticks = None

        self.prev_left_ticks_imu = None
        self.prev_right_ticks_imu = None
        self.prev_left_ticks_vo = None
        self.prev_right_ticks_vo = None

        self.wheel_dist = 0.0   # distance (m) traveled since last VO frame
        self.frame_idx = 0
        self.latest_omega = 0.0
        self.last_predict_time = None
        self.last_vo_time = None  # wall-clock time of last processed image

        # Subscribers
        self.camera_info_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/camera_info",
            CameraInfo,
            self.cb_cam_info
        )
        self.image_sub = None
        rospy.Subscriber(
            f"/{self._vehicle_name}/left_wheel_encoder_driver_node/tick",
            WheelEncoderStamped,
            self.cb_left_encoder,
        )
        rospy.Subscriber(
            f"/{self._vehicle_name}/right_wheel_encoder_driver_node/tick",
            WheelEncoderStamped,
            self.cb_right_encoder,
        )
        rospy.Subscriber(
            f"/{self._vehicle_name}/imu_node/raw",
            Imu,
            self.cb_imu
        )

        if ENABLE_VISUAL_MATCHES:
            self.match_pub = rospy.Publisher(
                f'/{self._vehicle_name}/vo_node/matched_image/compressed',
                CompressedImage,
                queue_size=1
            )

        self._last_process_time = rospy.Time(0)
        hz = 5
        self._process_interval = rospy.Duration(1.0 / hz)

        self.trajectory_pub = rospy.Publisher(
            f'/{self._vehicle_name}/vo_node/trajectory/image/compressed',
            CompressedImage,
            queue_size=1
        )
        # Throttle trajectory image to 0.5 Hz — the asyncio-based image viewer
        # cannot handle bursts at the full VO rate (5 Hz) and destroys pending tasks.
        # None = "never published yet" so the first frame always fires.
        self._last_traj_pub_time = None
        self._traj_pub_interval = rospy.Duration(2.0)  # publish at most every 2 s

    def cb_left_encoder(self, msg):
        if self.l_whl_res is None:
            self.l_whl_res = msg.resolution
        self.left_ticks = msg.data

    def cb_right_encoder(self, msg):
        if self.r_whl_res is None:
            self.r_whl_res = msg.resolution
        self.right_ticks = msg.data

    def _compute_velocity(self, dt):
        """Return average linear velocity (m/s) traveled since last call."""
        if None in (self.left_ticks, self.right_ticks, self.l_whl_res, self.r_whl_res) \
                or dt <= 0:
            return None

        if self.prev_left_ticks_imu is None or self.prev_right_ticks_imu is None:
            self.prev_left_ticks_imu = self.left_ticks
            self.prev_right_ticks_imu = self.right_ticks
            return 0.0

        dticks_left = self.left_ticks - self.prev_left_ticks_imu
        dticks_right = self.right_ticks - self.prev_right_ticks_imu

        l_dist_per_tick = (2 * np.pi * self.wheel_radius) / self.l_whl_res
        r_dist_per_tick = (2 * np.pi * self.wheel_radius) / self.r_whl_res
        dist_left = dticks_left * l_dist_per_tick
        dist_right = dticks_right * r_dist_per_tick

        self.prev_left_ticks_imu = self.left_ticks
        self.prev_right_ticks_imu = self.right_ticks

        return ((dist_left + dist_right) / 2.0) / dt

    def _compute_wheel_distance(self):
        """Return average distance (m) traveled since last call to this function."""
        if None in (self.left_ticks, self.right_ticks, self.l_whl_res, self.r_whl_res):
            return None

        if self.prev_left_ticks_vo is None or self.prev_right_ticks_vo is None:
            self.prev_left_ticks_vo = self.left_ticks
            self.prev_right_ticks_vo = self.right_ticks
            return 0.0

        dticks_left = self.left_ticks - self.prev_left_ticks_vo
        dticks_right = self.right_ticks - self.prev_right_ticks_vo

        l_dist_per_tick = (2 * np.pi * self.wheel_radius) / self.l_whl_res
        r_dist_per_tick = (2 * np.pi * self.wheel_radius) / self.r_whl_res
        dist_left = dticks_left * l_dist_per_tick
        dist_right = dticks_right * r_dist_per_tick

        self.prev_left_ticks_vo = self.left_ticks
        self.prev_right_ticks_vo = self.right_ticks

        return (dist_left + dist_right) / 2.0

    def cb_imu(self, msg):
        """IMU callback — used only on the real robot (not available in simulator).
        Runs a fallback EKF prediction between VO frames using raw gyro + wheel velocity."""
        if self.vo is None:
            return
        now = msg.header.stamp.to_sec()
        self.latest_omega = msg.angular_velocity.z

        if self.last_predict_time is None or now < self.last_predict_time:
            self.last_predict_time = now
            return

        dt = now - self.last_predict_time
        self.last_predict_time = now

        v = self._compute_velocity(dt)
        if v is None:
            v = 0.0
        # Use IMU fallback prediction (between VO frames on the real robot)
        self.vo.ekf.predict_imu(v, self.latest_omega, dt)

    def cb_cam_info(self, msg):
        """
        Callback function for camera info messages.
        Initializes the camera model with the received camera parameters.
        Subscribes to the image topic after receiving camera info.
        """
        self.camera_info_sub.unregister()
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)
        self.vo = VisualOdometry(
            intrinsic_matrix=self.camera_model.intrinsicMatrix(),
            save_matches=ENABLE_VISUAL_MATCHES,
            mask_top_ratio=None
        )

        buf_len = (msg.width * msg.height * 3) * 2
        self.image_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/image/compressed",
            CompressedImage,
            self.cb_img,
            queue_size=1,
            buff_size=buf_len
        )
        rospy.loginfo(f"{self.node_name}: Received camera info and initialized.")

    def cb_img(self, msg):
        now = msg.header.stamp
        if now.is_zero():
            now = rospy.Time.now()

        if (now - self._last_process_time) < self._process_interval:
            return
        self._last_process_time = now
        current_image = load_img_from_msg(msg)

        if self.prev_left_ticks_vo is None and self.left_ticks is not None:
            self.prev_left_ticks_vo = self.left_ticks
        if self.prev_right_ticks_vo is None and self.right_ticks is not None:
            self.prev_right_ticks_vo = self.right_ticks

        # Compute distance traveled since last VO frame (used as scale for t)
        wheel_dist = self._compute_wheel_distance()
        if wheel_dist is None:
            wheel_dist = 0.0

        # Estimate dt between consecutive VO frames for the EKF
        now_sec = now.to_sec()
        if self.last_vo_time is None:
            dt_vo = 1.0 / 5.0  # default to process rate
        else:
            dt_vo = max(now_sec - self.last_vo_time, 1e-3)
        self.last_vo_time = now_sec

        # Velocity = distance / dt (wheel encoders are primary source for position)
        v_vo = wheel_dist / dt_vo if dt_vo > 0 else 0.0

        success, reason = self.vo.process_frame(
            current_image,
            scale_override=wheel_dist,   # VO t-vector scaled to wheel distance
            v=v_vo,                      # wheel-encoder velocity drives EKF position
            omega=self.latest_omega,     # IMU omega for update step (0 in simulator)
            dt=dt_vo,
            predict=True,                # EKF is updated here (primary path in simulator)
            frame_idx=self.frame_idx
        )
        self.frame_idx += 1

        if ENABLE_VISUAL_MATCHES and self.vo.latest_match_img is not None:
            self.match_pub.publish(
                self.bridge.cv2_to_compressed_imgmsg(self.vo.latest_match_img)
            )

        if success:
            rospy.loginfo_throttle(
                1.0,
                f"Rel. translation (unit scale): {self.vo.t_latest.ravel()}"
            )
            # Throttle trajectory publish so the async image viewer is not overwhelmed
            time_since_last = (
                rospy.Duration(999) if self._last_traj_pub_time is None
                else (now - self._last_traj_pub_time)
            )
            if time_since_last >= self._traj_pub_interval:
                traj_img = self.vo.render_trajectory_cv2()
                if traj_img is not None:
                    self.trajectory_pub.publish(
                        self.bridge.cv2_to_compressed_imgmsg(traj_img, dst_format='jpg')
                    )
                    self._last_traj_pub_time = now
                else:
                    rospy.logwarn("[VO] render_trajectory_cv2() returned None")
        else:
            rospy.logwarn_throttle(2.0, f"[VO] {reason}")


if __name__ == '__main__':
    birdseye_node = VisualOdometryNode(node_name='vo_node')
    rospy.spin()
