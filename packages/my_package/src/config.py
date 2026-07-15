import os
from pathlib import Path

import numpy as np

# Modes configuration
VIRTUAL = False
ENHANCED_LANE_DETECTION = True
OBJECT_DETECTION = False
USE_SEGMENTATION = True
SEGMENTATION_EDGE_DETECTION = False
USE_WHEEL_ODOMETRY = True
USE_TWIST = True
INTERSECTION_DECISIONS = ["straight"]


ORIGINAL_IMAGE_SIZE = (640, 480)  # (width, height)

HZ = 5 if VIRTUAL else 15

# Testing locally, no ROS
LOCAL_TESTING = os.environ.get("LOCAL_TESTING", "false").lower() == "true"
DATA_ROOT = Path(__file__).parent / "data"
DATA_DIR = DATA_ROOT / "virtual" if VIRTUAL else DATA_ROOT / "physical"


# ROS publishing configuration
PUBLISH_TO_WHEELS = True
# Debug only, Turn this off, it needs lot of resources
PUBLISH_VISUALIZATIONS = False

# Calibration files
EXTRINSIC_CALIBRATION_FILE = DATA_DIR / "extrinsic.yaml"
INTRINSIC_CALIBRATION_FILE = DATA_DIR / "intrinsic.yaml"

# OD Model
OD_MODEL_PATH = DATA_DIR / "od_model.onnx"
OD_CONF_THRESHOLD = 0.4

# Segmentation Model
SEG_MODEL_PATH = DATA_DIR / "seg_small_image.onnx"
MODEL_INPUT_SIZE = (320, 256)  # (width, height)
SEG_CONF_THRESHOLD = 0.4

# Duckietown constants
LANE_WIDTH = 0.28  # 28 cm
TILE_WIDTH = 0.61  # 61 cm
DUCKIE_RADIUS = 0.025  # 2.5 cm
BOT_WIDTH = 0.12  # 12 cm

# Obstacle avoidance
AVOIDANCE_START_ABSOLUTE_ROI = (TILE_WIDTH, BOT_WIDTH)  # (ahead, each side)
WHEEL_TO_FRONT_OFFSET = 0.06  # 6 cm
LANE_POLY_EPSILON = 5.0  # pixels
AVOIDANCE_MARGIN = 0.03  # 3 cm, margin from obstacles for path planning
LAMBDA_OBSTACLES = 1000.0
FREE_X_THRESHOLD = BOT_WIDTH / 2 + AVOIDANCE_MARGIN + 0.001
PLANNING_WEIGHT_FINAL_POSITION = 5.0

# CEM planning
CEM_HORIZON = 5
CEM_NUM_SAMPLES = 100
CEM_NUM_ELITES = 10
CEM_NUM_ITERATIONS = 3
CEM_DT = 0.5  # seconds per one planning (big) step
CEM_V_CONST = 0.2
CEM_OMEGA_MEAN = 0.0
CEM_OMEGA_STD = 1.0
CEM_TEMPERATURE = 1.0  # for MPPI weighing
CEM_OMEGA_CLIP = (-1.0, 1.0)
CEM_ACTION_REPEAT = 4  # how many small steps are there in one planning big step

# BEV ROI for obstacle avoidance
BEV_SIZE = (TILE_WIDTH, TILE_WIDTH)  # meters (width, height/ahead)
BEV_RESOLUTION = 0.001  # 0.1 cm per pixel

# Hyperparameters dependent on VIRTUAL
if VIRTUAL:
    MIN_LANE_BOUNDARY_POINTS = 30
    HIDE_TOP_OF_IMAGE = 250
    CROSSING_OFFSET_LEFT = np.array([160, -350])
    CROSSING_OFFSET_RIGHT = np.array([200, -140])
    MIN_AREA_STOP_LINE = 200
    TOF_THRESHOLD = 0.2

    WHITE_HSV_LOWER = np.array([0, 0, 180])
    WHITE_HSV_UPPER = np.array([180, 40, 255])
    YELLOW_HSV_LOWER = np.array([18, 80, 100])
    YELLOW_HSV_UPPER = np.array([35, 255, 255])

    STOP_TIME = 1
    FOLLOW_TIME = {
        "left": 4,
        "straight": 3,
        "right": 2,
    }
    FOLLOW_DISTANCE = {
        "left": 0.45,
        "straight": 0.35,
        "right": 0.25,
    }
    CROSS_TIME = 1.5

    TURN_SPEED_LEFT_WHEEL = 0.0
    TURN_SPEED_RIGHT_WHEEL = 0.4
    TURN_TIME = 1.8
    TURN_DISTANCE = 0.07
    TURN_TIME = 3.6
    TURN_OMEGA = 1.5

    # PID VALUES:
    KP = 1.2
    KI = 0.0
    KD = 0.0
    MAX_OMEGA = 2.0
    INTEGRAL_LIMIT = 1.0  # anti-windup clamp on the PID's integral term — tune to taste
    PID_MAX_DT = 0.5  # caps a single update's dt so a stale timestamp (after TURN/STOP) can't spike the integral
else:
    MIN_LANE_BOUNDARY_POINTS = 30
    HIDE_TOP_OF_IMAGE = 160
    CROSSING_OFFSET_LEFT = np.array([160, -300])
    CROSSING_OFFSET_RIGHT = np.array([150, -140])
    MIN_AREA_STOP_LINE = 100
    TOF_THRESHOLD = 0.0

    WHITE_HSV_LOWER = np.array([0, 0, 140])
    WHITE_HSV_UPPER = np.array([180, 90, 255])
    YELLOW_HSV_LOWER = np.array([15, 55, 60])
    YELLOW_HSV_UPPER = np.array([40, 255, 255])

    STOP_TIME = 2
    FOLLOW_TIME = {
        "left": 1,
        "straight": 1.7,
        "right": 1.2,
    }
    FOLLOW_DISTANCE = {
        "left": 0.45,
        "straight": 0.35,
        "right": 0.25,
    }
    CROSS_TIME = 1.5

    TURN_SPEED_LEFT_WHEEL = 0.0
    TURN_SPEED_RIGHT_WHEEL = 0.7
    TURN_TIME = 0.8
    TURN_DISTANCE = 0.07

    # PID VALUES:
    KP = 6.0
    KI = 0.0
    KD = 0.1
    MAX_OMEGA = 3.0
    INTEGRAL_LIMIT = 1.0
    PID_MAX_DT = 0.5

CROSSING_OFFSET_TOP = np.array([110, 0])

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
    STEERING_GAIN = 0.2
    CROSSING_OFFSET = {
        "left": np.array([0.55, 0.37]),
        "straight": np.array([0.62, 0]),
        "right": np.array([0.18, -0.12]),
    }
    CROSSING_OFFSET_LEFT = np.array([40, -230])
    CROSSING_OFFSET_RIGHT = np.array([80, 150])
    CROSSING_OFFSET_TOP = np.array([100, 0])

# Global parameters
MAX_SPEED_DIFF = 0.2
WHITE_LANE_MIN_AREA = 30
N_WAYPOINTS = 6
STOP_MARKER_Y_RATIO = 0.73
X_TOLERANCE = 5
Y_TOLERANCE = 5
ANGLE_THRESHOLD = 5
CUT_FRONT_STOP_LINE = 400
LEFT_VS_RIGHT = 320
WAIT_UNTIL_TURN_TIME = 3
PROXIMITY_OTHER_VEHICLES_TO_RED_LINE = 50
SINGLE_SPLINE_N_WAYPOINTS = 15  # because derivatives need more points
SINGLE_SPLINE_OFFSET_WORLD = {
    "right_white": LANE_WIDTH / 2,
    "yellow": -LANE_WIDTH / 2,
    "left_white": -3 * LANE_WIDTH / 2,
}

# Odometry
WHEEL_RADIUS = 0.0318
ALPHA_WHEEL = 2 * np.pi / 135

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
