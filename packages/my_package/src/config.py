import os
from pathlib import Path

import numpy as np


# Modes configuration
VIRTUAL = True
ENHANCED_LANE_DETECTION = True
OBJECT_DETECTION = False

HZ = 5 if VIRTUAL else 15

# Testing locally, no ROS
LOCAL_TESTING = os.environ.get("LOCAL_TESTING", "false").lower() == "true"
TEST_DATA_ROOT = Path(__file__).parent / "test_data"
TEST_DATA_DIR = TEST_DATA_ROOT / "virtual" if VIRTUAL else TEST_DATA_ROOT / "physical"


# ROS publishing configuration
PUBLISH_TO_WHEELS = True
# Debug only, Turn this off, it needs lot of resources
PUBLISH_VISUALIZATIONS = False

# Calibration files
EXTRINSIC_CALIBRATION_FILE = Path(
    "/data/config/calibrations/camera_extrinsic/default.yaml"
    if VIRTUAL
    else "/data/config/calibrations/camera_extrinsic/wayne.yaml"  # TODO: fix this path
)
INTRINSIC_CALIBRATION_FILE = Path(
    "/data/config/calibrations/camera_intrinsic/default.yaml"
    if VIRTUAL
    else "/data/config/calibrations/camera_intrinsic/wayne.yaml"  # TODO: fix this path
)
if LOCAL_TESTING:
    EXTRINSIC_CALIBRATION_FILE = TEST_DATA_DIR / "extrinsic.yaml"
    INTRINSIC_CALIBRATION_FILE = TEST_DATA_DIR / "intrinsic.yaml"

# OD Model
if LOCAL_TESTING:
    OD_MODEL_PATH = TEST_DATA_DIR / "od_model.onnx"
else:
    # TODO: set correct paths
    import rospkg

    rospack = rospkg.RosPack()
    PKG_ROOT = Path(rospack.get_path("my_package"))
    MODEL_PATH = PKG_ROOT / "object_detection.onnx"
OD_CONF_THRESHOLD = 0.4

# Duckietown constants
LANE_WIDTH = 0.21  # 21 cm
TILE_WIDTH = 0.61  # 61 cm

# BEV ROI for obstacle avoidance
BEV_SIZE = (TILE_WIDTH * 2, TILE_WIDTH * 2)  # meters (width, height/ahead)
BEV_RESOLUTION = 0.002  # 0.2 cm per pixel

# Hyperparameters dependent on VIRTUAL
if VIRTUAL:
    MIN_LANE_PIXELS = 30
    HIDE_TOP_OF_IMAGE = 250
    CROSSING_OFFSET_LEFT = np.array([160, -350])
    CROSSING_OFFSET_RIGHT = np.array([200, -140])
    MIN_AREA = 200
    TOF_THRESHOLD = 0.2

    WHITE_HSV_LOWER = np.array([0, 0, 180])
    WHITE_HSV_UPPER = np.array([180, 40, 255])
    YELLOW_HSV_LOWER = np.array([18, 80, 100])
    YELLOW_HSV_UPPER = np.array([35, 255, 255])

    STOP_TIME = 1
    FOLLOW_TIME = [4, 3, 2]  # left, top, right
    CROSS_TIME = 1.5
    TURN_TIME = 1.8
else:
    MIN_LANE_PIXELS = 100
    HIDE_TOP_OF_IMAGE = 200
    CROSSING_OFFSET_LEFT = np.array([160, -300])
    CROSSING_OFFSET_RIGHT = np.array([150, -140])
    MIN_AREA = 200
    TOF_THRESHOLD = 0.0

    WHITE_HSV_LOWER = np.array([0, 0, 140])
    WHITE_HSV_UPPER = np.array([180, 90, 255])
    YELLOW_HSV_LOWER = np.array([15, 55, 60])
    YELLOW_HSV_UPPER = np.array([40, 255, 255])

    STOP_TIME = 2
    FOLLOW_TIME = [4, 4, 3]  # left, top, right
    CROSS_TIME = 2
    TURN_TIME = 3

if VIRTUAL and not ENHANCED_LANE_DETECTION:
    BASE_SPEED = 0.25
    STEERING_GAIN = 0.1
    IMAGE_WIDTH_OFFSET_FACTOR_YELLOW = 0.25
    IMAGE_WIDTH_OFFSET_FACTOR_WHITE = 0.25
elif VIRTUAL and ENHANCED_LANE_DETECTION:
    BASE_SPEED = 0.2
    STEERING_GAIN = 0.1
elif not VIRTUAL and not ENHANCED_LANE_DETECTION:
    BASE_SPEED = 0.1
    STEERING_GAIN = 0.25
    IMAGE_WIDTH_OFFSET_FACTOR_YELLOW = 0.15
    IMAGE_WIDTH_OFFSET_FACTOR_WHITE = 0.25
else:
    BASE_SPEED = 0.2
    STEERING_GAIN = 0.25
    CROSSING_OFFSET_LEFT = np.array([40, -100])
    CROSSING_OFFSET_RIGHT = np.array([80, 100])

CROSSING_OFFSET_TOP = np.array([110, 0])

# Global parameters
MAX_SPEED_DIFF = 0.2
WHITE_LANE_ONLY_BIGGEST_COMPONENT = False
N_WAYPOINTS = 6
SINGLE_LANE_SCALE_FACTOR_WHITE = 0.65
SINGLE_LANE_SCALE_FACTOR_YELLOW = 0.6
STOP_MARKER_Y = 350
X_TOLERANCE = 5
Y_TOLERANCE = 5
ANGLE_THRESHOLD = 5
CUT_FRONT_STOP_LINE = 400
LEFT_VS_RIGHT = 320
WAIT_UNTIL_TURN_TIME = 3
TURN_SPEED_RIGHT_WHEEL = 0.4
PROXIMITY_OTHER_VEHICLES_TO_RED_LINE = 50

# Color filtering limits
SIGMA = 2.0
SOBEL_THRESHOLD = 50
RED_HSV_LOWER_1 = np.array([0, 80, 100])
RED_HSV_UPPER_1 = np.array([10, 255, 255])
RED_HSV_LOWER_2 = np.array([160, 80, 100])
RED_HSV_UPPER_2 = np.array([180, 255, 255])

BLUE_HSV_LOWER = np.array([110, 200, 20])
BLUE_HSV_UPPER = np.array([130, 255, 255])
GREEN_HSV_LOWER = np.array([35, 50, 50])
GREEN_HSV_UPPER = np.array([85, 255, 255])
