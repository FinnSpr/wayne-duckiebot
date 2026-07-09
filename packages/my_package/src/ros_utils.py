#!/usr/bin/env python3
"""
ROS utility functions for my_package.

Contains helpers for ROS connections and custom message construction.
"""

from typing import Union

import numpy as np
import rospy
from my_package.msg import Trajectory, TrajectoryPoint


def wait_for_connection(
    obj: Union[rospy.Subscriber, rospy.Publisher], timeout: float = None
) -> None:
    """Block until *obj* has at least one connection.

    Works with both :class:`rospy.Subscriber` (waits for a publisher to connect)
    and :class:`rospy.Publisher` (waits for a subscriber to connect).

    Args:
        obj:     The subscriber or publisher to check.
        timeout: Maximum time (seconds) to wait; ``None`` means wait forever.

    Raises:
        rospy.ROSInterruptException: If ROS is shutting down.
        TimeoutError:                If *timeout* is reached without a connection.
    """
    rospy.loginfo("Waiting for connection on '%s' ...", obj.name)
    start = rospy.Time.now()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        if obj.get_num_connections() > 0:
            rospy.loginfo("Connected on '%s'.", obj.name)
            return

        if timeout is not None and (rospy.Time.now() - start).to_sec() > timeout:
            raise TimeoutError(
                f"Timed out waiting for connection on '{obj.name}' after {timeout:.1f} s."
            )

        rate.sleep()


# ---------------------------------------------------------------------------
# Trajectory message helpers
# ---------------------------------------------------------------------------


def trajectory_from_waypoints(
    waypoints: np.ndarray, speed: float, stamp=None
) -> Trajectory:
    """
    Convenience helper to build a Trajectory message.

    Args:
        waypoints: list of (x, y) tuples or (N, 2) np.ndarray
        speed: fixed forward speed (m/s)
        stamp: rospy.Time for the trajectory start (default: now)

    Returns:
        Trajectory message
    """
    traj = Trajectory()
    if stamp is None:
        stamp = rospy.Time.now()
    traj.header.stamp = stamp
    traj.speed = speed
    traj.waypoints = []
    for wp in waypoints:
        pt = TrajectoryPoint()
        pt.x = float(wp[0])
        pt.y = float(wp[1])
        traj.waypoints.append(pt)
    return traj


def waypoints_to_array(trajectory_msg: Trajectory) -> np.ndarray:
    """
    Extract waypoints from a Trajectory message as a (N, 2) numpy array.

    Args:
        trajectory_msg: Trajectory ROS message

    Returns:
        np.ndarray of shape (N, 2) with columns [x, y]
    """
    pts = trajectory_msg.waypoints
    arr = np.empty((len(pts), 2), dtype=np.float64)
    for i, pt in enumerate(pts):
        arr[i, 0] = pt.x
        arr[i, 1] = pt.y
    return arr
