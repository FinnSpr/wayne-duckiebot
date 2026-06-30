#!/bin/bash

source /environment.sh

dt-launchfile-init
rosrun my_package test_trajectory.py
dt-launchfile-join
