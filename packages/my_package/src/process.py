#!/usr/bin/env python3
import cv2

def process_all(data):
    return image_green_check(data._image)

def image_green_check(image):
    # Convert to HSV for reliable color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green color range in HSV
    lower_green = (35, 50, 50)
    upper_green = (85, 255, 255)

    # Create mask and calculate percentage of green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = cv2.countNonZero(mask) / mask.size

    if green_ratio > 0.5:  # more than 50% of image is green → stop
        return 0.0, 0.0
    else:                  # not enough green → drive straight
        return 0.5, 0.5