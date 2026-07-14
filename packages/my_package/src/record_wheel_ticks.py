#!/usr/bin/env python3
"""Node that records wheel encoder ticks and total distance traveled.

Prints the difference between the initial tick counts and the current tick counts
every 5 seconds, along with the total distance traveled computed from odometry.
"""

import os

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped

import config
import odometry


class RecordWheelTicks(DTROS):
    def __init__(self, node_name):
        super().__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # Initial ticks set on first reading, distances start at 0
        self._left_start_ticks = None
        self._right_start_ticks = None
        self._left_curr_ticks = None
        self._right_curr_ticks = None

        # Previous ticks for computing delta_phi on each update
        self._left_prev_ticks = None
        self._right_prev_ticks = None

        # Total distance traveled (meters)
        self._total_distance = 0.0

        # Encoder resolution: ticks per full wheel rotation
        # ALPHA_WHEEL = 2 * pi / 135, so resolution = (2 * pi) / ALPHA_WHEEL = 135
        self._resolution = int(2 * 3.1415926535 / config.ALPHA_WHEEL)

        # Wheel radius
        self._R = config.WHEEL_RADIUS

        # 5-second print timer
        self._last_print_time = rospy.Time.now()

        # Subscribers
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

    # --- Callbacks ---
    def cb_left_encoder(self, msg):
        tick = msg.data

        # Capture initial tick on first reading
        if self._left_start_ticks is None:
            self._left_start_ticks = tick
            self._left_prev_ticks = tick
        else:
            # Accumulate distance from left wheel
            dphi = odometry.delta_phi(tick, self._left_prev_ticks, self._resolution)
            self._total_distance += abs(self._R * dphi)
            self._left_prev_ticks = tick

        self._left_curr_ticks = tick
        self._maybe_print()

    def cb_right_encoder(self, msg):
        tick = msg.data

        # Capture initial tick on first reading
        if self._right_start_ticks is None:
            self._right_start_ticks = tick
            self._right_prev_ticks = tick
        else:
            # Accumulate distance from right wheel
            dphi = odometry.delta_phi(tick, self._right_prev_ticks, self._resolution)
            self._total_distance += abs(self._R * dphi)
            self._right_prev_ticks = tick

        self._right_curr_ticks = tick
        self._maybe_print()

    def _maybe_print(self):
        """Print stats if 5 seconds have elapsed since last print."""
        now = rospy.Time.now()
        elapsed = (now - self._last_print_time).to_sec()

        if elapsed >= 5.0:
            self._last_print_time = now

            # Tick differences from start
            if (self._left_start_ticks is not None
                    and self._right_start_ticks is not None
                    and self._left_curr_ticks is not None
                    and self._right_curr_ticks is not None):
                left_diff = self._left_curr_ticks - self._left_start_ticks
                right_diff = self._right_curr_ticks - self._right_start_ticks

                rospy.loginfo(
                    f"Wheel ticks diff - Left: {left_diff:>8d} ticks  |  "
                    f"Right: {right_diff:>8d} ticks  |  "
                    f"Total distance: {self._total_distance:.4f} m"
                )
            else:
                rospy.loginfo("Waiting for encoder data...")


if __name__ == "__main__":
    node = RecordWheelTicks(node_name="record_wheel_ticks")
    rospy.spin()
