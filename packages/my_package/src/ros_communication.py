#!/usr/bin/env python3
import cv2
import numpy as np
import os
import rospy
from dataclasses import dataclass

from dt_computer_vision.camera import CameraModel
from dt_computer_vision.ground_projection import GroundProjector
from dt_computer_vision.camera.homography import Homography, HomographyToolkit

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image, Range, CameraInfo
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
from cv_bridge import CvBridge
from turbojpeg import TurboJPEG

from process import process_all
from constants import DEFAULT_HOMOGRAPHY

class ROSCommunication(DTROS):
    def __init__(self, node_name):
        super(ROSCommunication, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC
        )

        self._last_process_time = rospy.Time(0)
        self._process_interval = rospy.Duration(1.0 / 5)  # 5 Hz

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._bridge = CvBridge()

        # Latest messages from each topic, initialized to None
        self._image = None
        self._tof = None
        self._left_encoder = None
        self._right_encoder = None

        # Subscribers
        rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed",
                         CompressedImage, self.cb_camera)
        rospy.Subscriber(f"/{self._vehicle_name}/front_center_tof_driver_node/range",
                         Range, self.cb_tof)
        rospy.Subscriber(f"/{self._vehicle_name}/left_wheel_encoder_driver_node/tick",
                         WheelEncoderStamped, self.cb_left_encoder)
        rospy.Subscriber(f"/{self._vehicle_name}/right_wheel_encoder_driver_node/tick",
                         WheelEncoderStamped, self.cb_right_encoder)
        self._sub_camera_info = rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/camera_info",
            CameraInfo,
            self.cb_info,
            queue_size=1,
        )

        # Publishers
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

        self._vis_publisher = rospy.Publisher(
            f"/{self._vehicle_name}/lane_detection/image/raw",
            Image, queue_size=1
        )
        self._undistorted_pub = rospy.Publisher(
            f"/{self._vehicle_name}/lane_detection/undistorted/image_raw",
            Image, queue_size=1
        )
        self._warped_pub = rospy.Publisher(
            f"/{self._vehicle_name}/lane_detection/warped/image_raw",
            Image, queue_size=1
        )
        self._lane_pub = rospy.Publisher(
            f"/{self._vehicle_name}/lane_detection/lane_vis/image_raw",
            Image, queue_size=1
        )
        self._mask_pub = rospy.Publisher(
            f"/{self._vehicle_name}/lane_detection/lane_vis/mask",
            Image, queue_size=1
        )

        # general things required
        self.camera_model = None
        self.mapx = None
        self.mapy = None
        self.jpeg = TurboJPEG()

    def load_extrinsics(self):
        """
        Loads the homography matrix from the extrinsic calibration file.

        Returns:
            :obj:`Homography`: the loaded homography matrix

        """
        # load extrinsic calibration
        cali_file_folder = "/data/config/calibrations/camera_extrinsic/"
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"
        # print(cali_file)

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log(
                f"Can't find calibration file: {cali_file}\n Using default calibration instead.",
                "warn",
            )
            cali_file = os.path.join(cali_file_folder, "default.yaml")

        # print(cali_file)
        # # Shutdown if no calibration file not found
        # if not os.path.isfile(cali_file):
        #     msg = "Found no calibration file ... aborting"
        #     self.logerr(msg)
        #     rospy.signal_shutdown(msg)

        try:
            # self.H = HomographyToolkit.load_from_disk(
            #     cali_file, return_date=False
            # )  # type: ignore
            # return self.H.reshape((3, 3))

            self.H = DEFAULT_HOMOGRAPHY
            return self.H
        except Exception as e:
            msg = f"Error in parsing calibration file {cali_file}:\n{e}"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

    def cb_info(self, msg):
        self.loginfo("Camera info message received. Unsubscribing from camera_info topic.")
        try:
            self._sub_camera_info.shutdown()
        except BaseException:
            pass
        H, W = msg.height, msg.width
        # create new camera info
        self.camera_model = CameraModel(
            width=W,
            height=H,
            K=np.reshape(msg.K, (3, 3)),
            D=np.reshape(msg.D, (5,)),
            P=np.reshape(msg.P, (3, 4)),
        )
        homography = self.load_extrinsics()
        self.camera_model.H = homography
        self.projector = GroundProjector(self.camera_model)

        rect_camera_K, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_model.K, self.camera_model.D, (W, H), alpha=0.0
        )
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.camera_model.K, self.camera_model.D, None, rect_camera_K, (W, H), cv2.CV_32FC1
        )

    
    # --- Callbacks: just store the latest message ---
    def cb_camera(self, msg):
        now = rospy.Time.now()
        if (now - self._last_process_time) < self._process_interval:
            return  # skip, too soon
        self._last_process_time = now
        self._image = self._bridge.compressed_imgmsg_to_cv2(msg)
        self.img = self.jpeg.decode(msg.data)
        if self.mapx is None or self.mapy is None:
            self.loginfo("Waiting for Camera Info")
            return
        self.process()

    def cb_tof(self, msg):
        self._tof = msg.range

    def cb_left_encoder(self, msg):
        self._left_encoder = msg.data

    def cb_right_encoder(self, msg):
        self._right_encoder = msg.data

    def process(self):
        vel_left, vel_right, undistorted, warped, lane_vis, overlay = process_all(self)

        self._undistorted_pub.publish(self._bridge.cv2_to_imgmsg(undistorted, encoding="bgr8"))
        self._warped_pub.publish(self._bridge.cv2_to_imgmsg(warped, encoding="mono8"))
        self._lane_pub.publish(self._bridge.cv2_to_imgmsg(lane_vis, encoding="bgr8"))
        self._vis_publisher.publish(self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8"))
        # self._mask_pub.publish(self._bridge.cv2_to_imgmsg(mask, encoding="mono8"))


        # self.loginfo(f"vel_left={vel_left:.3f}  vel_right={vel_right:.3f}")

        self._publisher.publish(WheelsCmdStamped(vel_left=vel_left, vel_right=vel_right))
        
    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)

if __name__ == '__main__':
    node = ROSCommunication(node_name='ros_communication')
    rospy.spin()