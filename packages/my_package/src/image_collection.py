#!/usr/bin/env python3

import os
from threading import Lock

import config
import cv2
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage


class ImageCollection(DTROS):
    def __init__(self, node_name):
        super().__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._bridge = CvBridge()

        self._latest_image = None
        self._image_lock = Lock()
        self._frame_count = 0

        # Ensure output directory exists
        self._output_dir = "/data/images"
        os.makedirs(self._output_dir, exist_ok=True)

        rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/image/compressed",
            CompressedImage,
            self.cb_camera,
        )

        self._timer = rospy.Timer(rospy.Duration(1.0 / config.HZ), self._timer_callback)

        rospy.loginfo(
            f"ImageCollection node started. Saving images to {self._output_dir} at {config.HZ} Hz"
        )

    def cb_camera(self, msg):
        with self._image_lock:
            self._latest_image = self._bridge.compressed_imgmsg_to_cv2(msg)

    def _timer_callback(self, event):
        with self._image_lock:
            if self._latest_image is None:
                return
            image_to_save = self._latest_image

        filename = os.path.join(self._output_dir, f"frame_{self._frame_count:04d}.jpg")
        cv2.imwrite(filename, image_to_save)
        self._frame_count += 1

        rospy.loginfo_throttle(5, f"Saved {self._frame_count} images so far")


if __name__ == "__main__":
    node = ImageCollection(node_name="image_collection")
    rospy.spin()
