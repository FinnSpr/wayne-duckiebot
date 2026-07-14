#!/bin/bash

source /environment.sh

dt-launchfile-init
rosrun my_package record_wheel_ticks.py
dt-launchfile-join
