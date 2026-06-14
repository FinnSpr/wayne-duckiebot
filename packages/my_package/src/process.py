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
from pid_controller import PIDController


# Finite State Machine


class State(Enum):
    DRIVE = 1
    STOP = 2
    FOLLOW = 3
    CROSS = 4


state = State.DRIVE


def print_state():
    print(state)


print_state()

state_entered_at = time.time()

crossing_decision = False
crossing_vel_left = 0.0
crossing_vel_right = 0.0
decision_waypoint = None
decision = None


# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────

VIRTUAL = True

# Base speed for both wheels (0.0 – 1.0)
BASE_SPEED = 0.15

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

# Scale factor for single-lane fallback: when only one lane marking is visible,
# project waypoints toward the opposite bottom corner with a perspective-like
# scaling.  0.0 = no shift, 1.0 = opposite image edge at the bottom row.
SINGLE_LANE_SCALE_FACTOR_WHITE = 0.65  # white-only → shift left
SINGLE_LANE_SCALE_FACTOR_YELLOW = 0.6  # yellow-only → shift right

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
FOLLOW_TIME = [4, 3, 2]  # left, top, right
CROSS_TIME = 1.5

# ─────────────────────────────────────────────
# COLOR FILTERING
# ─────────────────────────────────────────────

# Gaussian blur standard deviation for pre-processing
SIGMA = 2.0

# Sobel edge magnitude threshold
SOBEL_THRESHOLD = 50

# HSV range for white lane markings
WHITE_HSV_LOWER = np.array([0, 0, 200])
WHITE_HSV_UPPER = np.array([180, 30, 255])

# HSV range for yellow lane markings
YELLOW_HSV_LOWER = np.array([18, 80, 100])
YELLOW_HSV_UPPER = np.array([35, 255, 255])

# Red wraps around the hue boundary in OpenCV (0–180 scale)
RED_HSV_LOWER_1 = np.array([0, 80, 100])
RED_HSV_UPPER_1 = np.array([10, 255, 255])
RED_HSV_LOWER_2 = np.array([160, 80, 100])
RED_HSV_UPPER_2 = np.array([180, 255, 255])

# HSV range for green
GREEN_HSV_LOWER = np.array([35, 50, 50])
GREEN_HSV_UPPER = np.array([85, 255, 255])


def filter_lane_colors(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter the image for white and yellow lane markings.
    Steps:
        1. Gaussian blur to suppress noise
        2. Sobel edge detection with directional masks
        3. HSV color thresholding on the blurred image
        4. White lane mask: find rightmost right-edge pixel per row, then
           its corresponding left-edge pixel to the left
        5. Yellow/red: intersect color with edge mask
    Args:
        image: BGR image from the camera (480x640x3)
    Returns:
        white_lane_mask: binary mask of right white lane edges
        yellow_mask:     binary mask of yellow lane pixels
        red_mask:        binary mask of red stop-line pixels
        edge_mask:       binary mask of Sobel edges (before color filtering)
        white_color:     raw white color mask (before edge intersection)
    """
    # ── Step 1: Gaussian blur ──
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=SIGMA)

    # ── Step 2: Sobel edge detection ──
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_mask = (magnitude > SOBEL_THRESHOLD).astype(np.uint8) * 255

    mask_sobelx_pos = ((sobel_x > 0).astype(np.uint8)) * 255
    mask_sobelx_neg = ((sobel_x < 0).astype(np.uint8)) * 255
    mask_sobely_pos = ((sobel_y > 0).astype(np.uint8)) * 255

    # ── Step 3: HSV color thresholding on the blurred image ──
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    white_color = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    yellow_color = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)

    if VIRTUAL:
        green_mask = cv2.inRange(hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)
        white_color = cv2.bitwise_or(white_color, green_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white_color = cv2.morphologyEx(white_color, cv2.MORPH_OPEN, kernel)

    mask1 = cv2.inRange(hsv, RED_HSV_LOWER_1, RED_HSV_UPPER_1)
    mask2 = cv2.inRange(hsv, RED_HSV_LOWER_2, RED_HSV_UPPER_2)
    red_color = cv2.bitwise_or(mask1, mask2)

    white_edges = cv2.bitwise_and(white_color, edge_mask)
    right_edge_white_lane = cv2.bitwise_and(
        white_edges, cv2.bitwise_and(mask_sobelx_neg, mask_sobely_pos)
    )

    yellow_mask = cv2.bitwise_and(yellow_color, edge_mask)
    yellow_mask = cv2.bitwise_and(
        yellow_mask, cv2.bitwise_and(mask_sobelx_pos, mask_sobely_pos)
    )

    red_mask = cv2.bitwise_and(red_color, edge_mask)

    return right_edge_white_lane, yellow_mask, red_mask, edge_mask, white_color


def fit_spline(
    mask: np.ndarray, take_leftmost_pixels=True
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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
    image_width: int,
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
            component_pixels = np.column_stack(
                np.where(labels == label)
            )  # (row, col) = (y, x)
            points = component_pixels[:, ::-1].astype(np.float32)  # flip to (x, y)
            # Fit a rotated bounding box → gives angle
            _, _, angle = cv2.minAreaRect(points)
            # angle is in [-90, 0), if close to -90 or 0, it is horizontal
            orientation = (
                "vertical"
                if min(np.abs(angle), angle + 90) > ANGLE_THRESHOLD
                else "horizontal"
            )

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

            waypoints = np.column_stack([center_x, center_y])

        elif white_spline is not None:
            # Only white (right) lane visible — project toward bottom-left corner
            # t_x based on x (progress toward left edge), t_y based on y (progress toward bottom)
            wx, wy = white_spline
            t_x = wx / image_width  # 0 at left edge, 1 at right edge
            t_y = 1 - wy / image_height  # 0 at top, 1 at bottom
            center_x = wx - wx * t_x * SINGLE_LANE_SCALE_FACTOR_WHITE
            center_y = wy + (image_height - wy) * t_y * SINGLE_LANE_SCALE_FACTOR_WHITE
            waypoints = np.column_stack([center_x, center_y])

        elif yellow_spline is not None:
            # Only yellow (left) lane visible — project toward bottom-right corner
            yx, yy = yellow_spline
            t_x = 1 - yx / image_width
            t_y = 1 - yy / image_height
            center_x = yx + (image_width - yx) * t_x * SINGLE_LANE_SCALE_FACTOR_YELLOW
            center_y = yy + (image_height - yy) * t_y * SINGLE_LANE_SCALE_FACTOR_YELLOW
            waypoints = np.column_stack([center_x, center_y])

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
    vertical_other_lines_left = [
        l for l in other_lines if l[2] == "vertical" and l[0] < LEFT_VS_RIGHT
    ]
    vertical_other_lines_right = [
        l for l in other_lines if l[2] == "vertical" and l[0] >= LEFT_VS_RIGHT
    ]
    horizontal_other_lines = [l for l in other_lines if l[2] == "horizontal"]

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
    current_line = max(
        [l for l in lines if l[2] == "horizontal"],
        key=lambda l: l[1],
        default=[int(image_width / 2), 0],
    )  # max cy
    return np.array([[current_line[0], current_line[1]]])


# ─────────────────────────────────────────────
# HEADING ESTIMATION
# ─────────────────────────────────────────────


def estimate_heading_error(
    waypoints: np.ndarray, image_width: int, image_height: int
) -> float:
    """
    Estimate the lateral heading error from the center waypoints.
    Uses a lookahead point and compares its x position to the image center.

    A negative error means the path curves left  → steer left (slow left wheel).
    A positive error means the path curves right → steer right (slow right wheel).

    Args:
        waypoints:   array of shape (N, 2) with (x, y) waypoints
        image_width: width of the camera image in pixels
        image_height: height of the camera image in pixels
    Returns:
        heading_error: normalized lateral error in [-1, 1]
    """
    # Take farthest waypoint
    farthest = waypoints[0]
    image_center_x = image_width / 2.0

    dx = farthest[0] - image_center_x
    dy = image_height - farthest[1]
    path_angle = np.arctan2(dx, dy)

    angle_error = path_angle / (np.pi / 2)
    return float(np.clip(angle_error, -1.0, 1.0))


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

        vel_left = BASE_SPEED + correction
        vel_right = BASE_SPEED - correction

    # Clamp to valid range
    vel_left = float(np.clip(vel_left, -1.0, 1.0))
    vel_right = float(np.clip(vel_right, -1.0, 1.0))

    crossing_vel_left = vel_left
    crossing_vel_right = vel_right

    return vel_left, vel_right


# ─────────────────────────────────────────────
# MAIN PIPELINE ENTRY POINT
# ─────────────────────────────────────────────


def visualize(
    image,
    white_lane_mask,
    yellow_mask,
    red_mask,
    white_spline,
    yellow_spline,
    waypoints,
):

    # Dim everything to 15%, then restore lane pixels to full brightness
    vis = (image * 0.15).astype(np.uint8)
    vis[white_lane_mask > 0] = image[white_lane_mask > 0]
    vis[yellow_mask > 0] = image[yellow_mask > 0]
    vis[red_mask > 0] = image[red_mask > 0]

    if white_spline is not None:
        wx, wy = white_spline
        for i in range(len(wx) - 1):
            cv2.line(
                vis,
                (int(wx[i]), int(wy[i])),
                (int(wx[i + 1]), int(wy[i + 1])),
                (255, 0, 0),
                2,
            )

    if yellow_spline is not None:
        yx, yy = yellow_spline
        for i in range(len(yx) - 1):
            cv2.line(
                vis,
                (int(yx[i]), int(yy[i])),
                (int(yx[i + 1]), int(yy[i + 1])),
                (0, 255, 0),
                2,
            )

    if waypoints is not None:
        for x, y in waypoints:
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

    if state == State.DRIVE:
        if not np.all(red_mask[STOP_MARKER_Y:, :] == 0):
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
    # Use unwarped (bird's-eye) image if available, otherwise fall back to raw image
    if hasattr(data, "_unwarped_image") and data._unwarped_image is not None:
        image = data._unwarped_image
    else:
        image = data._image
    image_height, image_width = image.shape[:2]

    white_lane_mask, yellow_mask, red_mask, edge_mask, white_color = filter_lane_colors(
        image
    )

    if state == State.DRIVE:
        white_lane_mask[:HIDE_TOP_OF_IMAGE, :] = 0
        yellow_mask[:HIDE_TOP_OF_IMAGE, :] = 0
        red_mask[:HIDE_TOP_OF_IMAGE, :] = 0

    state_transition(red_mask)

    white_spline = fit_spline(white_lane_mask, take_leftmost_pixels=False)
    yellow_spline = fit_spline(yellow_mask, take_leftmost_pixels=False)

    if not crossing_decision:
        waypoints = compute_waypoints(
            white_spline, yellow_spline, red_mask, image_height, image_width
        )

        if waypoints is None:
            # No lane detected at all — stop safely
            visualization = visualize(
                image,
                white_lane_mask,
                yellow_mask,
                red_mask,
                white_spline,
                yellow_spline,
                None,
            )
            return (
                0.0,
                0.0,
                visualization,
                edge_mask,
                white_lane_mask,
                yellow_mask,
                white_color,
            )

        # ── Step 4: Heading error ────────────────────
        heading_error = estimate_heading_error(waypoints, image_width, image_height)

        # ── Step 5: Wheel commands ───────────────────
        vel_left, vel_right = heading_to_wheel_commands(heading_error)
    else:
        vel_left = crossing_vel_left
        vel_right = crossing_vel_right
        waypoints = decision_waypoint

    # ── Visualization ────────────────────────────
    visualization = visualize(
        image,
        white_lane_mask,
        yellow_mask,
        red_mask,
        white_spline,
        yellow_spline,
        waypoints,
    )

    return (
        vel_left,
        vel_right,
        visualization,
        edge_mask,
        white_lane_mask,
        yellow_mask,
        white_color,
    )
