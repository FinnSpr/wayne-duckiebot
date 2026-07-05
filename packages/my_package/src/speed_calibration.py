#!/usr/bin/env python3
import os

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

HZ = 60.0
BASE_SPEED = 0.06


class SpeedCalibration(DTROS):
    """ROS node that sends a constant forward speed for calibration."""

    def __init__(self, node_name: str = "speed_calibration"):
        super().__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self._vehicle_name = os.environ["VEHICLE_NAME"]

        cmd_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self._cmd_publisher = rospy.Publisher(cmd_topic, Twist2DStamped, queue_size=1)

        self._timer = rospy.Timer(rospy.Duration(1.0 / HZ), self._timer_callback)

    def _timer_callback(self, event: rospy.timer.TimerEvent) -> None:
        cmd = Twist2DStamped(v=BASE_SPEED, omega=0.0)
        self._cmd_publisher.publish(cmd)

    def on_shutdown(self) -> None:
        self._cmd_publisher.publish(Twist2DStamped(v=0.0, omega=0.0))


if __name__ == "__main__":
    node = SpeedCalibration(node_name="speed_calibration")
    rospy.spin()
