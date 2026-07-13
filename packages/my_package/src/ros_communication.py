#!/usr/bin/env python3
import os

import config
import cv2
import numpy as np
import ros_utils
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, WheelsCmdStamped
from image_utils import BEVConfig, load_calibrations
from process import SelfDrivingPipeline
from sensor_msgs.msg import CompressedImage, Image


class ROSCommunication(DTROS):
    def __init__(self, node_name):
        super().__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._last_process_time = rospy.Time(0)
        self._process_interval = rospy.Duration(1.0 / config.HZ)

        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._bridge = CvBridge()

        # Calibration data
        self._K, self._D, self._P, self._H = load_calibrations(
            config.INTRINSIC_CALIBRATION_FILE, config.EXTRINSIC_CALIBRATION_FILE
        )
        self.bev_cfg = BEVConfig(
            bev_size=config.BEV_SIZE, bev_resolution=config.BEV_RESOLUTION
        )

        # Latest messages from each topic, initialized to None
        self._image = None
        self._unwarped_image = None
        self._left_encoder = None
        self._right_encoder = None

        # Publisher
        self._visualization_publishers: dict = {}
        if config.PUBLISH_TO_WHEELS:
            if config.USE_TWIST:
                wheels_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
                self._wheel_publisher = rospy.Publisher(
                    wheels_topic, Twist2DStamped, queue_size=1
                )
            else:
                wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
                self._wheel_publisher = rospy.Publisher(
                    wheels_topic, WheelsCmdStamped, queue_size=1
                )
        ros_utils.wait_for_connection(self._wheel_publisher)

        # Self-driving pipeline
        self._pipeline = SelfDrivingPipeline(
            self._K, self._D, self._P, self._H, self.bev_cfg
        )
        self._pipeline.planner.set_intersection_decisions(config.INTERSECTION_DECISIONS)

        # Subscribers
        rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/image/compressed",
            CompressedImage,
            self.cb_camera,
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
        # rospy.Subscriber(
        #     f"/{self._vehicle_name}/navigation/turn_queue",
        #     String,
        #     self.cb_turn_queue,
        # )
        # rospy.Subscriber(
        #     f"/{self._vehicle_name}/navigation/arrived",
        #     Bool,
        #     self.cb_arrived,
        # )

    # --- Callbacks ---
    def cb_arrived(self, msg):
        self._pipeline.planner.set_arrived(msg.data)

    def cb_turn_queue(self, msg):
        decisions = list(msg.data)
        self._pipeline.planner.set_intersection_decisions(decisions)

    def cb_camera(self, msg):
        now = rospy.Time.now()
        if (now - self._last_process_time) < self._process_interval:
            return  # skip, too soon
        self._last_process_time = now
        self._image = self._bridge.compressed_imgmsg_to_cv2(msg)
        self.process()

    def cb_left_encoder(self, msg):
        self._left_encoder = msg.data

    def cb_right_encoder(self, msg):
        self._right_encoder = msg.data

    def _publish_vis(self, vis_name: str, vis_img: np.ndarray, encoding: str) -> None:
        if vis_name not in self._visualization_publishers:
            topic = f"/{self._vehicle_name}/lane_detection/image/{vis_name}"
            self._visualization_publishers[vis_name] = rospy.Publisher(
                topic, Image, queue_size=1
            )
        pub = self._visualization_publishers[vis_name]
        pub.publish(self._bridge.cv2_to_imgmsg(vis_img, encoding=encoding))

    def process(self):
        vel_left, vel_right, color_vis, bw_vis = self._pipeline.process(
            self._image, self._left_encoder, self._right_encoder
        )
        if config.PUBLISH_VISUALIZATIONS:
            for vis_name, vis_img in color_vis.items():
                self._publish_vis(vis_name, vis_img, "bgr8")
            for vis_name, vis_img in bw_vis.items():
                self._publish_vis(vis_name, vis_img, "mono8")
        if config.PUBLISH_TO_WHEELS:
            if config.USE_TWIST:
                v = vel_left
                omega = vel_right
                cmd = Twist2DStamped()
                cmd.header.stamp = rospy.Time.now()
                cmd.v = v
                cmd.omega = omega
                self._wheel_publisher.publish(cmd)
            else:
                self._wheel_publisher.publish(
                    WheelsCmdStamped(vel_left=vel_left, vel_right=vel_right)
                )

    def on_shutdown(self):
        if config.PUBLISH_TO_WHEELS:
            if config.USE_TWIST:
                stop = Twist2DStamped(v=0.0, omega=0.0)
            else:
                stop = WheelsCmdStamped(vel_left=0, vel_right=0)
            self._wheel_publisher.publish(stop)
            self._wheel_publisher.publish(stop)


if __name__ == "__main__":
    node = ROSCommunication(node_name="ros_communication")
    rospy.spin()
