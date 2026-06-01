#!/usr/bin/env python3

"""
Modular lane-following pipeline for Duckiebot.
Entry point: process_all(image) -> (vel_left, vel_right)
"""

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from typing import Tuple, Optional
import random
from enum import Enum
import time


# Finite State Machine

class State(Enum):
    DRIVE = 1
    STOP = 2
    FOLLOW = 3
    CROSS = 4
    TURN = 5

state = State.DRIVE

def print_state():
    print(state)
print_state()

state_entered_at = time.time()
time_last_waypoint = time.time()

crossing_decision = False
crossing_vel_left = 0.0
crossing_vel_right = 0.0
decision_waypoint = None
decision = None


# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────

# Base speed for both wheels (0.0 – 1.0)
BASE_SPEED = 0.25

# Steering gain: how strongly heading error affects wheel differential
# Higher = more aggressive turning
STEERING_GAIN = 0.1

# Maximum allowed wheel speed difference (clamps hard turns)
MAX_SPEED_DIFF = 0.2

# Minimum number of lane marking pixels required to attempt spline fitting
MIN_LANE_PIXELS = 30

# How much of the top of the image is hidden for the spline fitting
HIDE_TOP_OF_IMAGE = 250

# Number of waypoints to sample along each fitted spline
N_WAYPOINTS = 6

# Minimum y value for red intersection marker to stop the vehicle
STOP_MARKER_Y = 350

# Pixel tolerance in x or y direction to detect red stop lines
X_TOLERANCE = 5
Y_TOLERANCE = 5

# Minimum area for a red stop line to be treated as such
MIN_AREA = 200  # ignore small noise blobs

# Horizontal vs. vertical angle threshold
ANGLE_THRESHOLD = 5

# y threshold for cutting the front stop line(s)
CUT_FRONT_STOP_LINE = 400

# left vs right x thresholds for assigning detected stop lines
LEFT_VS_RIGHT = 320

# Offset between stop marking and target point (lane) when crossing an intersection
CROSSING_OFFSET_TOP = np.array([110, 0])
CROSSING_OFFSET_LEFT = np.array([160, -350])
CROSSING_OFFSET_RIGHT = np.array([200, -140])

# Times for the state transition (in s)
STOP_TIME = 1
FOLLOW_TIME = [4, 3, 2] # left, top, right
CROSS_TIME = 1.5

WAIT_UNTIL_TURN_TIME = 3
TURN_TIME = 1.8
TURN_SPEED_RIGHT_WHEEL = 0.4

# ─────────────────────────────────────────────
# COLOR FILTERING
# ─────────────────────────────────────────────

# HSV range for white lane markings
WHITE_HSV_LOWER = np.array([0,   0,   180])
WHITE_HSV_UPPER = np.array([180, 40,  255])

# HSV range for yellow lane markings
YELLOW_HSV_LOWER = np.array([18,  80,  100])
YELLOW_HSV_UPPER = np.array([35,  255, 255])

# Red wraps around the hue boundary in OpenCV (0–180 scale)
RED_HSV_LOWER_1 = np.array([  0,  80, 100])
RED_HSV_UPPER_1 = np.array([ 10, 255, 255])
RED_HSV_LOWER_2 = np.array([160,  80, 100])
RED_HSV_UPPER_2 = np.array([180, 255, 255])


def filter_lane_colors(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter the image for white and yellow lane markings using HSV thresholding.
    Only the right half of the image is considered for white markings,
    to avoid picking up the left white lane marking.
    Args:
        image: BGR image from the camera (480x640x3)
    Returns:
        white_mask: binary mask of right white lane pixels only
        yellow_mask: binary mask of yellow lane pixels
    """
    _, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    yellow_mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)
    
    mask1 = cv2.inRange(hsv, RED_HSV_LOWER_1, RED_HSV_UPPER_1)
    mask2 = cv2.inRange(hsv, RED_HSV_LOWER_2, RED_HSV_UPPER_2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Mask out left half for white to keep only right lane marking
    white_mask[:, :width // 2] = 0

    return white_mask, yellow_mask, red_mask


def fit_spline(mask: np.ndarray, take_leftmost_pixels=True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Fit a cubic spline to the non-zero pixels in a binary mask.
    Pixels are sorted top-to-bottom (by y coordinate) for a natural lane curve.
    Args:
        mask: binary mask of lane marking pixels
    Returns:
        (xs, ys): arrays of sampled spline points, or None if fitting failed
    """
    ys, xs = np.where(mask > 0)

    if len(xs) < MIN_LANE_PIXELS:
        return None

    # Sort by y (top to bottom in image)
    sort_idx = np.argsort(ys)
    xs, ys = xs[sort_idx], ys[sort_idx]

    # Downsample to speed up fitting and reduce noise influence
    step = max(1, len(xs) // 100)
    xs, ys = xs[::step], ys[::step]

    if take_leftmost_pixels:
        take_fn = np.min
    else:
        take_fn = np.max

    # Remove duplicate y values which cause splprep to fail based on take_leftmost_pixels
    unique_ys = np.unique(ys)
    taken_xs = np.array([take_fn(xs[ys == y]) for y in unique_ys])
    xs, ys = taken_xs, unique_ys

    if len(xs) < 4:
        return None

    try:
        tck, _ = splprep([xs, ys], s=50000, k=3)
        u_fine = np.linspace(0, 1, N_WAYPOINTS)
        x_spline, y_spline = splev(u_fine, tck)
        return x_spline, y_spline
    except Exception as e:
        return None


# ─────────────────────────────────────────────
# WAYPOINT PREDICTION
# ─────────────────────────────────────────────

def compute_waypoints(
    white_spline: Optional[Tuple[np.ndarray, np.ndarray]],
    yellow_spline: Optional[Tuple[np.ndarray, np.ndarray]],
    red_mask,
    image_height: int,
    image_width: int
) -> Optional[np.ndarray]:
    """
    Compute center waypoints between the white right lane marking and
    the yellow center lane marking. Falls back to a single spline if
    only one is detected.

    Args:
        white_spline:  (xs, ys) of the white lane spline, or None
        yellow_spline: (xs, ys) of the yellow lane spline, or None
        image_width:   width of the camera image in pixels

    Returns:
        waypoints: array of shape (N, 2) with (x, y) center points, or None
    """

    point_to_stop_line = False
    if not np.all(red_mask == 0):
        # ── Step 1: Find connected components (individual stop lines) ────────────
        binary = (red_mask > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        # stats columns: LEFT, TOP, WIDTH, HEIGHT, AREA

        lines = []
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < MIN_AREA:
                continue

            # Get all pixels belonging to this component
            component_pixels = np.column_stack(np.where(labels == label))  # (row, col) = (y, x)
            points = component_pixels[:, ::-1].astype(np.float32)          # flip to (x, y)
            # Fit a rotated bounding box → gives angle
            _, _, angle = cv2.minAreaRect(points)
            # angle is in [-90, 0), if close to -90 or 0, it is horizontal
            orientation = "vertical" if min(np.abs(angle), angle + 90) > ANGLE_THRESHOLD else "horizontal"
            
            cx = int(centroids[label][0])
            cy = int(centroids[label][1])
            lines.append([cx, cy, orientation, angle, area])

        # Keep only the five largest (the front line may be separated into two)
        lines = sorted(lines, key=lambda l: l[3], reverse=True)[:5]

        # Drop the area column, no longer needed
        lines = [l[:4] for l in lines]

        if len(lines) > 0:
            point_to_stop_line = True

            if state == State.FOLLOW:
                waypoints = compute_crossing_waypoint(lines, image_height, image_width)
            elif state == State.CROSS:
                point_to_stop_line = False
            else:
                waypoints = compute_stop_line_waypoint(lines, image_height, image_width)
    
    if not point_to_stop_line:
        if white_spline is not None and yellow_spline is not None:
            wx, wy = white_spline
            yx, yy = yellow_spline

            # Average the two splines pointwise to get center points
            center_x = (wx + yx) / 2.0
            center_y = (wy + yy) / 2.0

            # Fit a new spline through the center points
            try:
                tck, _ = splprep([center_x, center_y], s=999999, k=3)
                u_fine = np.linspace(0, 1, N_WAYPOINTS)
                cx_spline, cy_spline = splev(u_fine, tck)
                waypoints = np.column_stack([cx_spline, cy_spline])
            except Exception as e:
                waypoints = np.column_stack([center_x, center_y])

        elif white_spline is not None:
            # Only white lane visible: offset left by 25% of image width
            wx, wy = white_spline
            offset = image_width * 0.25
            waypoints = np.column_stack([wx - offset, wy])

        elif yellow_spline is not None:
            # Only yellow lane visible: offset right by 25% of image width
            yx, yy = yellow_spline
            offset = image_width * 0.25
            waypoints = np.column_stack([yx + offset, yy])

        else:
            waypoints = None

    return waypoints

def compute_crossing_waypoint(lines, image_height, image_width):
    """
    Detects all red stop lines at an intersection, excludes the one the car
    is currently halting at (closest, highest y), and returns the center
    waypoint of a randomly chosen other stop line.

    Args:
        lines
    Returns:
        (cx, cy) center of a randomly chosen far stop line, or None if not found
    """
    global crossing_decision, decision_waypoint, decision

    other_lines = [l for l in lines if l[1] < CUT_FRONT_STOP_LINE]

    if not other_lines:
        return None

    # Use orientation and minimum/maximum info to get the right direction of the other lines
    vertical_other_lines_left = [l for l in other_lines if l[2] == "vertical" and l[0] < LEFT_VS_RIGHT]
    vertical_other_lines_right = [l for l in other_lines if l[2] == "vertical" and l[0] >= LEFT_VS_RIGHT]
    horizontal_other_lines = [l for l in other_lines if l[2]  == "horizontal"]

    choices = {}
    if len(horizontal_other_lines) == 1:
        choices["straight"] = horizontal_other_lines[0]
    if len(vertical_other_lines_left) == 1:
        choices["left"] = vertical_other_lines_left[0]
    if len(vertical_other_lines_right) == 1:
        choices["right"] = vertical_other_lines_right[0]
    # first try assigning the safe lines which are likely correct (orientation, number, position)
    # only if none is safe, take more risky detections
    if not choices:
        if len(horizontal_other_lines) > 1:
            choices["straight"] = horizontal_other_lines[0]
        if len(vertical_other_lines_left) > 1:
            choices["left"] = vertical_other_lines_left[0]
        if len(vertical_other_lines_right) > 1:
            choices["right"] = vertical_other_lines_right[0]
    
    print("\n------Possible Destinations------")
    for key, value in choices.items():
        print(f"{key}: {value}")
    print("")

    decision, chosen = random.choice(list(choices.items()))

    if decision == "straight":
        target = np.array(chosen[:2]) + CROSSING_OFFSET_TOP
    elif decision == "left":
        target = np.array(chosen[:2]) + CROSSING_OFFSET_LEFT
    elif decision == "right":
        target = np.array(chosen[:2]) + CROSSING_OFFSET_RIGHT
    
    print(f"Decision: {decision}\n")

    x = int(np.clip(target[0], 0, image_width))
    y = int(np.clip(target[1], 0, image_height))

    crossing_decision = True
    decision_waypoint = np.array([[x, y]])
    return decision_waypoint

def compute_stop_line_waypoint(lines, image_height, image_width):
    """
    Finds the horizontal red halting line closest to the car (highest y)
    and returns the center point of that line.
    
    Args:
        lines
    Returns:
        (cx, cy) center of the halting line, or None if not found
    """
    # ── Step 2: Identify the closest stop line ──
    current_line = max([l for l in lines if l[2] == "horizontal"], key=lambda l: l[1], default=[int(image_width/2), 0])  # max cy
    return np.array([[current_line[0], current_line[1]]])


# ─────────────────────────────────────────────
# HEADING ESTIMATION
# ─────────────────────────────────────────────

def estimate_heading_error(waypoints: np.ndarray, image_width: int) -> float:
    """
    Estimate the lateral heading error from the center waypoints.
    Uses a lookahead point and compares its x position to the image center.

    A negative error means the path curves left  → steer left (slow left wheel).
    A positive error means the path curves right → steer right (slow right wheel).

    Args:
        waypoints:   array of shape (N, 2) with (x, y) waypoints
        image_width: width of the camera image in pixels

    Returns:
        heading_error: normalized lateral error in [-1, 1]
    """
    # Take farthest waypoint
    farthest = waypoints[0]

    # Normalize to [-1, 1] by half the image width
    image_center_x = image_width / 2.0
    error_farthest = (farthest[0] - image_center_x) / image_center_x
    return float(np.clip(error_farthest, -1.0, 1.0))


# ─────────────────────────────────────────────
# WHEEL COMMAND COMPUTATION
# ─────────────────────────────────────────────

def heading_to_wheel_commands(heading_error: float) -> Tuple[float, float]:
    """
    Convert a heading error to differential wheel speeds.

    Positive heading_error → car must turn right → slow down right wheel.
    Negative heading_error → car must turn left  → slow down left wheel.

    Args:
        heading_error: normalized lateral error in [-1, 1]

    Returns:
        (vel_left, vel_right): wheel velocities in [0, 1]
    """
    global crossing_vel_left, crossing_vel_right

    if state == State.STOP:
        vel_left = 0
        vel_right = 0
    else:
        correction = STEERING_GAIN * heading_error
        correction = float(np.clip(correction, -MAX_SPEED_DIFF, MAX_SPEED_DIFF))

        vel_left  = BASE_SPEED + correction
        vel_right = BASE_SPEED - correction

    # Clamp to valid range
    vel_left  = float(np.clip(vel_left,  -1.0, 1.0))
    vel_right = float(np.clip(vel_right, -1.0, 1.0))

    crossing_vel_left = vel_left
    crossing_vel_right = vel_right

    return vel_left, vel_right


# ─────────────────────────────────────────────
# MAIN PIPELINE ENTRY POINT
# ─────────────────────────────────────────────

def visualize(image, white_mask, yellow_mask, red_mask, white_spline, yellow_spline, waypoints):

    # Dim everything to 15%, then restore lane pixels to full brightness
    vis = (image * 0.15).astype(np.uint8)
    vis[white_mask > 0] = image[white_mask > 0]
    vis[yellow_mask > 0] = image[yellow_mask > 0]
    vis[red_mask > 0] = image[red_mask > 0]

    if white_spline is not None:
        wx, wy = white_spline
        for i in range(len(wx) - 1):
            cv2.line(vis, (int(wx[i]), int(wy[i])), (int(wx[i+1]), int(wy[i+1])), (255, 0, 0), 2)

    if yellow_spline is not None:
        yx, yy = yellow_spline
        for i in range(len(yx) - 1):
            cv2.line(vis, (int(yx[i]), int(yy[i])), (int(yx[i+1]), int(yy[i+1])), (0, 255, 0), 2)

    if waypoints is not None:
        for (x, y) in waypoints:
            cv2.circle(vis, (int(x), int(y)), 4, (255, 191, 0), -1)

    return vis

def state_transition(red_mask):
    
    def change_state(s: State):
        global state, state_entered_at, crossing_decision
        state = s
        state_entered_at = time.time()
        crossing_decision = False
        print_state()
    
    def time_passed(t):
        return time.time() - state_entered_at >= t
    
    def no_waypoint_passed(t):
        global time_last_waypoint
        return time.time() - time_last_waypoint >= t

    if state == State.DRIVE:
        if no_waypoint_passed(WAIT_UNTIL_TURN_TIME):
            change_state(State.TURN)
        elif not np.all(red_mask[STOP_MARKER_Y:, :] == 0):
            change_state(State.STOP)
    if state == State.STOP:
        if time_passed(STOP_TIME):
            change_state(State.FOLLOW)
    if state == State.FOLLOW:
        t = 0
        if decision == "left":
            t = FOLLOW_TIME[0]
        elif decision == "straight":
            t = FOLLOW_TIME[1]
        else:
            t = FOLLOW_TIME[2]
        if time_passed(t):
            change_state(State.CROSS)
    if state == State.CROSS:
        if time_passed(CROSS_TIME):
            change_state(State.DRIVE)
    if state == State.TURN:
        if time_passed(TURN_TIME):
            change_state(State.DRIVE)

def process_all(data) -> Tuple[float, float]:
    """
    Full lane-following pipeline.

    Steps:
        1. Filter image for white and yellow lane markings
        2. Fit cubic splines to each lane marking
        3. Compute center waypoints between the two markings
        4. Estimate heading error from lookahead waypoint
        5. Convert heading error to differential wheel speeds

    Args:
        image: BGR camera image (numpy array)

    Returns:
        (vel_left, vel_right): wheel velocities in [0, 1]
    """
    global time_last_waypoint

    image = data._image
    image_height, image_width = image.shape[:2]

    white_mask, yellow_mask, red_mask = filter_lane_colors(image)

    if state == State.DRIVE:
        white_mask[:HIDE_TOP_OF_IMAGE, :] = 0
        yellow_mask[:HIDE_TOP_OF_IMAGE, :] = 0
        red_mask[:HIDE_TOP_OF_IMAGE, :] = 0

    state_transition(red_mask)

    white_spline  = fit_spline(white_mask, take_leftmost_pixels=True)
    yellow_spline = fit_spline(yellow_mask, take_leftmost_pixels=False)

    if not crossing_decision:
        waypoints = compute_waypoints(white_spline, yellow_spline, red_mask, image_height, image_width)

        if waypoints is None:
            # No lane detected at all — stop safely
            visualization = visualize(image, white_mask, yellow_mask, red_mask, white_spline, yellow_spline, None)
            if state == State.TURN:
                time_last_waypoint = time.time()
                return 0.0, TURN_SPEED_RIGHT_WHEEL, visualization
            return 0.0, 0.0, visualization 

        # ── Step 4: Heading error ────────────────────
        heading_error = estimate_heading_error(waypoints, image_width)

        # ── Step 5: Wheel commands ───────────────────
        vel_left, vel_right = heading_to_wheel_commands(heading_error)
    else:
        vel_left = crossing_vel_left
        vel_right = crossing_vel_right
        waypoints = decision_waypoint

    time_last_waypoint = time.time()

    # ── Visualization ────────────────────────────
    visualization = visualize(image, white_mask, yellow_mask, red_mask, white_spline, yellow_spline, waypoints)

    return vel_left, vel_right, visualization




# import cv2


# def process_all(data):
#     return image_green_check(data._image)

# def image_green_check(image):
#     # Convert to HSV for reliable color detection
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Green color range in HSV
#     lower_green = (35, 50, 50)
#     upper_green = (85, 255, 255)

#     # Create mask and calculate percentage of green pixels
#     mask = cv2.inRange(hsv, lower_green, upper_green)
#     green_ratio = cv2.countNonZero(mask) / mask.size

#     if green_ratio > 0.5:  # more than 50% of image is green → stop
#         return 0.0, 0.0
#     else:                  # not enough green → drive straight
#         return 0.5, 0.5