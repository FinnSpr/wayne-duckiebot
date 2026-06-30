#!/bin/bash

source /environment.sh

dt-launchfile-init
rosrun my_package image_collection.py
dt-launchfile-join
