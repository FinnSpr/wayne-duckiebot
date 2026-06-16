#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image, Range, CameraInfo
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
from cv_bridge import CvBridge

from process import process_all
from image_utils import load_calibrations, unwarp_image

PUBLISH_TO_WHEELS = True
PUBLISH_MAIN_VISUALIZATION = True
PUBLISH_ALL_VISUALIZATIONS = True


class ROSCommunication(DTROS):
    def __init__(self, node_name):
        super(ROSCommunication, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )

        self._last_process_time = rospy.Time(0)
        self._process_interval = rospy.Duration(1.0 / 5)  # 5 Hz

        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._bridge = CvBridge()

        # Calibration data (loaded from CameraInfo topic)
        self._K = None
        self._D = None
        self._calib_loaded = False

        # Latest messages from each topic, initialized to None
        self._image = None
        self._unwarped_image = None
        self._tof = None
        self._left_encoder = None
        self._right_encoder = None

        self._unwarped_publisher = rospy.Publisher(
            f"/{self._vehicle_name}/lane_detection/image/unwarped",
            Image,
            queue_size=1,
        )

        self._vis_publisher = rospy.Publisher(
            f"/{self._vehicle_name}/lane_detection/image/raw", Image, queue_size=1
        )

        self._edge_publisher = rospy.Publisher(
            f"/{self._vehicle_name}/lane_detection/image/edges", Image, queue_size=1
        )

        self._white_lane_publisher = rospy.Publisher(
            f"/{self._vehicle_name}/lane_detection/image/white_edges",
            Image,
            queue_size=1,
        )

        self._white_color_publisher = rospy.Publisher(
            f"/{self._vehicle_name}/lane_detection/image/white_color",
            Image,
            queue_size=1,
        )

        self._yellow_publisher = rospy.Publisher(
            f"/{self._vehicle_name}/lane_detection/image/yellow", Image, queue_size=1
        )

        self._red_publisher = rospy.Publisher(
            f"/{self._vehicle_name}/lane_detection/image/red", Image, queue_size=1
        )

        # Subscribers
        self.sub_camera_info = rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/camera_info",
            CameraInfo,
            self.cb_camera_info,
        )

        rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/image/compressed",
            CompressedImage,
            self.cb_camera,
        )
        rospy.Subscriber(
            f"/{self._vehicle_name}/front_center_tof_driver_node/range",
            Range,
            self.cb_tof,
        )
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

        # Publisher
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

    # --- Callbacks ---
    def cb_camera_info(self, msg):
        """Receive camera intrinsic calibration from ROS CameraInfo topic."""
        if self._calib_loaded:
            return
        try:
            self.sub_camera_info.unregister()
        except BaseException:
            pass

        self._K, self._D = load_calibrations(msg)
        self._calib_loaded = True
        rospy.loginfo("[ros_communication] Camera intrinsics loaded from CameraInfo.")

    def cb_camera(self, msg):
        now = rospy.Time.now()
        if (now - self._last_process_time) < self._process_interval:
            return  # skip, too soon
        self._last_process_time = now
        self._image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # Unwarp (undistort) the image if calibrations are available
        if self._calib_loaded:
            self._unwarped_image = unwarp_image(self._image, self._K, self._D)
            # Publish the unwarped image
            unwarped_msg = self._bridge.cv2_to_imgmsg(
                self._unwarped_image, encoding="bgr8"
            )
            self._unwarped_publisher.publish(unwarped_msg)

        self.process()

    def cb_tof(self, msg):
        self._tof = msg.range

    def cb_left_encoder(self, msg):
        self._left_encoder = msg.data

    def cb_right_encoder(self, msg):
        self._right_encoder = msg.data

    def process(self):
        (
            vel_left,
            vel_right,
            visualization,
            edge_mask,
            white_lane_mask,
            yellow_mask,
            red_mask,
            white_color,
        ) = process_all(self)
        if PUBLISH_MAIN_VISUALIZATION:
            self._vis_publisher.publish(
                self._bridge.cv2_to_imgmsg(visualization, encoding="bgr8")
            )
        if PUBLISH_ALL_VISUALIZATIONS:
            if edge_mask is not None:
                self._edge_publisher.publish(
                    self._bridge.cv2_to_imgmsg(edge_mask, encoding="mono8")
                )
            self._white_lane_publisher.publish(
                self._bridge.cv2_to_imgmsg(white_lane_mask, encoding="mono8")
            )
            self._yellow_publisher.publish(
                self._bridge.cv2_to_imgmsg(yellow_mask, encoding="mono8")
            )
            self._white_color_publisher.publish(
                self._bridge.cv2_to_imgmsg(white_color, encoding="mono8")
            )
            self._red_publisher.publish(
                self._bridge.cv2_to_imgmsg(red_mask, encoding="mono8")
            )
        if PUBLISH_TO_WHEELS:
            self._publisher.publish(
                WheelsCmdStamped(vel_left=vel_left, vel_right=vel_right)
            )

    def on_shutdown(self):
        if PUBLISH_TO_WHEELS:
            stop = WheelsCmdStamped(vel_left=0, vel_right=0)
            self._publisher.publish(stop)


if __name__ == "__main__":
    node = ROSCommunication(node_name="ros_communication")
    rospy.spin()
