#!/bin/bash

source /environment.sh

dt-launchfile-init
rosrun my_package debugging_node.py
dt-launchfile-join
