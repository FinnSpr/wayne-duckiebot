#!/usr/bin/env python3
import unittest
import numpy as np
import cv2
import sys
import os

# Adjust path so we can import from the same directory when executing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from perception import PerceptionModule
from world_model import WorldModel
from planning import BehaviorPlanner, State
from control import Controller
from process import SelfDrivingPipeline

class DummyData:
    def __init__(self, image, tof=1.0):
        self._image = image
        self._tof = tof

class TestModularPipeline(unittest.TestCase):
    def setUp(self):
        # Generate dummy 480x640x3 image
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create standard lines to test perception/spline fitting
        # White lane on the right
        cv2.line(self.image, (480, 480), (480, 200), (255, 255, 255), 5)
        # Yellow lane on the left
        cv2.line(self.image, (160, 480), (160, 200), (0, 255, 255), 5)

    def test_perception_masking(self):
        pm = PerceptionModule(use_object_detection=False)
        white_mask, yellow_mask, red_mask = pm.filter_lane_colors_standard(self.image)
        
        # Verify right half has some white lane pixels
        self.assertTrue(np.any(white_mask[:, 320:] > 0))
        # Verify left half has yellow lane pixels
        self.assertTrue(np.any(yellow_mask > 0))

    def test_world_model_spline(self):
        wm = WorldModel()
        # Create a basic straight mask
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(mask, (300, 400), (300, 100), 255, 2)
        
        spline = wm.fit_spline(mask, take_leftmost_pixels=True)
        self.assertIsNotNone(spline)
        xs, ys = spline
        self.assertEqual(len(xs), config.N_WAYPOINTS)
        self.assertEqual(len(ys), config.N_WAYPOINTS)

    def test_behavior_planner_transitions(self):
        bp = BehaviorPlanner()
        self.assertEqual(bp.state, State.DRIVE)
        
        # Test transition to STOP when red stop marker is seen
        red_mask = np.zeros((480, 640), dtype=np.uint8)
        # Add stop line in the stop marker vertical range (y >= 350)
        cv2.rectangle(red_mask, (100, 380), (540, 420), 255, -1)
        bp.update_state(red_mask)
        self.assertEqual(bp.state, State.STOP)

    def test_controller_steering(self):
        ctrl = Controller()
        waypoints = np.array([[340, 200], [330, 250], [320, 300]])
        heading_err = ctrl.estimate_heading_error(waypoints, 640, 480)
        
        # Target x=340 is slightly to the right of center (320), so error should be positive
        self.assertGreater(heading_err, 0.0)
        
        vel_left, vel_right = ctrl.heading_to_wheel_commands(heading_err, is_stopped=False)
        # Since heading error is positive, we slow down right wheel to steer right
        self.assertGreater(vel_left, vel_right)

    def test_pipeline_integration(self):
        pipeline = SelfDrivingPipeline()
        data = DummyData(self.image, tof=1.0)
        
        res = pipeline.process(data)
        self.assertEqual(len(res), 8)
        vel_left, vel_right = res[0], res[1]
        self.assertTrue(-1.0 <= vel_left <= 1.0)
        self.assertTrue(-1.0 <= vel_right <= 1.0)

if __name__ == "__main__":
    unittest.main()
