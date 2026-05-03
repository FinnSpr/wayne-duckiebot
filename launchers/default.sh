#!/bin/bash

source /environment.sh

dt-launchfile-init
rosrun my_package ros_communication.py
dt-launchfile-join
