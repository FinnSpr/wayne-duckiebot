#!/bin/bash

source /environment.sh

dt-launchfile-init
# rosrun my_package ros_communication.py
rosrun my_package visual_odometry_node.py
dt-launchfile-join
