#!/usr/bin/env python3
import csv
import math
import os
from multiprocessing import Lock

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped
from odometry import delta_phi, estimate_pose

HZ = 10.0
BASE_SPEED = 0.25
PERIOD = 20.0
CURVE_GAIN = 0.1

R = 0.0318
BASELINE = 0.1


class LoopTest(DTROS):
    """ROS node that commands a Duckiebot to drive a lemniscate (figure-8)."""

    def __init__(self, node_name: str = "loop_test"):
        super().__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self._vehicle_name = os.environ["VEHICLE_NAME"]

        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._wheel_publisher = rospy.Publisher(
            wheels_topic, WheelsCmdStamped, queue_size=1
        )
        # Wheel encoder subscriber:
        left_encoder_topic = (
            f"/{self._vehicle_name}/left_wheel_encoder_driver_node/tick"
        )
        rospy.Subscriber(left_encoder_topic, WheelEncoderStamped, self.cbLeftEncoder)
        self.left_wheel_mutex = Lock()

        # Wheel encoder subscriber:
        right_encoder_topic = (
            f"/{self._vehicle_name}/right_wheel_encoder_driver_node/tick"
        )
        rospy.Subscriber(right_encoder_topic, WheelEncoderStamped, self.cbRightEncoder)
        self.right_wheel_mutex = Lock()

        self.delta_phi_left = 0.0
        self.left_tick_prev = None
        self.delta_phi_right = 0.0
        self.right_tick_prev = None

        self.x_prev = 0.0
        self.y_prev = 0.0
        self.theta_prev = 0.0

        self._timer = rospy.Timer(rospy.Duration(1.0 / HZ), self._timer_callback)
        self._start_time = None

        self._period_counter = 0
        self._values = []

    def _timer_callback(self, event: rospy.timer.TimerEvent) -> None:
        self.left_wheel_mutex.acquire()
        self.right_wheel_mutex.acquire()

        if self._start_time is None:
            self._start_time = event.current_real.to_sec()
        t = event.current_real.to_sec() - self._start_time

        if t // PERIOD > self._period_counter:
            self.save_values_per_period()
            self._period_counter += 1
            self._values = []

        curvature = CURVE_GAIN * math.sin(2.0 * math.pi * t / PERIOD)

        vel_left = BASE_SPEED + curvature
        vel_right = BASE_SPEED - curvature

        vel_left = max(-1.0, min(1.0, vel_left))
        vel_right = max(-1.0, min(1.0, vel_right))
        print(f"vel_left={vel_left:.3f}, vel_right={vel_right:.3f}")

        # DEBUG: show accumulated encoder deltas before pose update
        rospy.loginfo_throttle(
            5,
            f"[TIMER] t={t:.2f}  acc_phi_L={self.delta_phi_left:.4f}  acc_phi_R={self.delta_phi_right:.4f}  x_before=({self.x_prev:.3f},{self.y_prev:.3f})",
        )

        self.x_prev, self.y_prev, self.theta_prev = estimate_pose(
            R,
            BASELINE,
            self.x_prev,
            self.y_prev,
            self.theta_prev,
            self.delta_phi_left,
            self.delta_phi_right,
        )
        self.delta_phi_left = 0.0
        self.delta_phi_right = 0.0

        # Save BOTH commanded velocities AND the encoder deltas that
        # produced this pose update. This allows fitting v_actual = f(v_cmd).
        # NOTE: the delta_phi values here were accumulated from commands
        # sent at the PREVIOUS timestep (due to the one-step lag).
        self._values.append(
            (
                t,
                vel_left,          # command sent NOW (will affect NEXT pose)
                vel_right,
                self.delta_phi_left,   # encoder rotation since last callback
                self.delta_phi_right,
                self.x_prev,       # pose AFTER applying encoder deltas
                self.y_prev,
                self.theta_prev,
            )
        )

        self._wheel_publisher.publish(
            WheelsCmdStamped(vel_left=vel_left, vel_right=vel_right)
        )

        self.left_wheel_mutex.release()
        self.right_wheel_mutex.release()

    def save_values_per_period(self):
        file_name = f"/values{self._period_counter}.csv"
        with open(file_name, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                [
                    "t",
                    "vel_left_cmd",
                    "vel_right_cmd",
                    "delta_phi_left",
                    "delta_phi_right",
                    "x_odo",
                    "y_odo",
                    "theta_odo",
                ]
            )
            csv_writer.writerows(self._values)

    def cbLeftEncoder(self, encoder_msg):
        """
        Wheel encoder callback
        Args:
            encoder_msg (:obj:`WheelEncoderStamped`) encoder ROS message.
        """
        with self.left_wheel_mutex:
            # initializing ticks to stored absolute value
            if self.left_tick_prev is None:
                self.left_tick_prev = encoder_msg.data
                return

            left_ticks_curr = encoder_msg.data

            # running the DeltaPhi() function copied from the notebooks to calculate rotations
            delta_phi_left = delta_phi(
                left_ticks_curr, self.left_tick_prev, encoder_msg.resolution
            )
            self.left_tick_prev = left_ticks_curr
            self.delta_phi_left += delta_phi_left
            # DEBUG: print encoder raw values
            rospy.loginfo_throttle(
                5,
                f"[LEFT]  ticks_curr={left_ticks_curr}  prev={self.left_tick_prev}  d_phi={delta_phi_left:.4f}  resolution={encoder_msg.resolution}",
            )

    def cbRightEncoder(self, encoder_msg):
        """
        Wheel encoder callback, the rotation of the wheel.
        Args:
            encoder_msg (:obj:`WheelEncoderStamped`) encoder ROS message.
        """

        with self.right_wheel_mutex:
            if self.right_tick_prev is None:
                self.right_tick_prev = encoder_msg.data
                return

            right_ticks_curr = encoder_msg.data

            # calculate rotation of right wheel
            delta_phi_right = delta_phi(
                right_ticks_curr, self.right_tick_prev, encoder_msg.resolution
            )
            self.right_tick_prev = right_ticks_curr
            self.delta_phi_right += delta_phi_right
            # DEBUG: print encoder raw values
            rospy.loginfo_throttle(
                5,
                f"[RIGHT] ticks_curr={right_ticks_curr}  prev={self.right_tick_prev}  d_phi={delta_phi_right:.4f}  resolution={encoder_msg.resolution}",
            )

    def on_shutdown(self) -> None:
        self._wheel_publisher.publish(WheelsCmdStamped(vel_left=0.0, vel_right=0.0))


if __name__ == "__main__":
    node = LoopTest(node_name="loop_test")
    rospy.spin()
