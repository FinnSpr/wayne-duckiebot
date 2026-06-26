import cv2
import numpy as np
from typing import Tuple, Optional
import time, random
from enum import Enum

class State(Enum):
    DRIVE  = 1   # normal lane following
    STOP   = 2   # saw red, waiting STOP_TIME
    FOLLOW = 3   # decided direction, driving toward intersection target
    CROSS  = 4   # past the line, driving blind for CROSS_TIME


BASE_SPEED = 0.25
STEERING_GAIN = 0.1
MAX_SPEED_DIFF = 0.2
state            = State.DRIVE
state_entered_at = time.time()
crossing_decision    = False   # True once a direction has been picked
crossing_vel_left    = BASE_SPEED
crossing_vel_right   = BASE_SPEED
decision_waypoint    = None   # (x, y) in camera space
decision             = None   # "left" | "straight" | "right"

# Red HSV thresholds (hue wraps at 180 in OpenCV)
RED_HSV_LOWER_1 = np.array([  0,  80, 100])
RED_HSV_UPPER_1 = np.array([ 10, 255, 255])
RED_HSV_LOWER_2 = np.array([160,  80, 100])
RED_HSV_UPPER_2 = np.array([180, 255, 255])

MIN_AREA         = 200    # ignore tiny noise blobs
ANGLE_THRESHOLD  = 5     # degrees — below this → horizontal line
STOP_MARKER_Y    = 350   # only red below this row triggers STOP
CUT_FRONT_STOP_LINE = 400  # ignore lines below this (the car's own stop line during FOLLOW)
LEFT_VS_RIGHT    = 320   # x split for vertical left vs right lines

CROSSING_OFFSET_TOP   = np.array([ 110,    0])
CROSSING_OFFSET_LEFT  = np.array([ 160, -350])
CROSSING_OFFSET_RIGHT = np.array([ 200, -140])

STOP_TIME   = 1.0
FOLLOW_TIME = {"left": 4.0, "straight": 3.0, "right": 2.0}
CROSS_TIME  = 1.5

## CONSTANTS:

# HOMOGRAPHY
HOM = np.array([
    -0.024035492889439958,  0.19876388357863825,  0.5190570282632594,
    -0.682009467271606,    -0.004337319642967827, 0.0017543970533221737,
    -0.054766168586224295,  5.494719535914916,     0.9999999999999999
]).reshape((3, 3))

# INCOMING IMAGE DIMS
H = 480 
W = 640 
# CAMERA PARAMS (copied from the cfg)
K = np.array([307.7379294605756, 0.0, 329.692367951685, 0.0, 314.9827773443905, 244.4605588877848, 0.0, 0.0, 1.0]).reshape(3,3) 
D = np.array([-0.2565888993516047, 0.04481160508242147, -0.00505275149956019, 0.001308569367976665, 0.0]).reshape(5,) 
P = np.array([210.1107940673828, 0.0, 327.2577820024981, 0.0, 0.0, 253.8408660888672, 239.9969353923052, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3, 4)


def undistort_image(image):
    rect_camera_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), alpha=0.0)
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, rect_camera_K, (W, H), cv2.CV_32FC1)
    undistorted = cv2.remap(image, mapx, mapy, cv2.INTER_NEAREST)
    h, w = undistorted.shape[:2]
    if h != w:
        undistorted = cv2.resize(undistorted, (w, w), interpolation=cv2.INTER_NEAREST)
    return undistorted


# ─────────────────────────────────────────────
# PERSPECTIVE WARP
# ─────────────────────────────────────────────

# Source points: trapezoid on the road, found via click_points.py.
# PLACEHOLDERS — replace with values from your own captured frame.
WARP_SRC = np.float32( [[199, 296], [479, 298], [600, 476], [66, 476]] )

# Destination rectangle WARP_SRC gets mapped to. Tune margins to taste.
WARP_DST = np.float32([
    [100, 0],
    [540, 0],
    [600, 480],
    [100, 480],
])

WARP_MATRIX = cv2.getPerspectiveTransform(WARP_SRC, WARP_DST)
WARP_MATRIX_INV = cv2.getPerspectiveTransform(WARP_DST, WARP_SRC)


def warp_perspective(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, WARP_MATRIX, (w, h), flags=cv2.INTER_LINEAR)


def unwarp_perspective(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, WARP_MATRIX_INV, (w, h), flags=cv2.INTER_LINEAR)



def warp_perspective(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, WARP_MATRIX, (w, h), flags=cv2.INTER_LINEAR)


def unwarp_perspective(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, WARP_MATRIX_INV, (w, h), flags=cv2.INTER_LINEAR)

def get_warp_validity_mask(h: int, w: int) -> np.ndarray:
    """
    Returns a binary mask of pixels that are inside WARP_SRC in the
    original image (i.e. valid after perspective warp).  Apply this to
    the warped binary to kill the sharp artificial border that Sobel picks up.
    """
    src_mask = np.ones((h, w), dtype=np.uint8) * 255
    return cv2.warpPerspective(src_mask, WARP_MATRIX, (w, h),
                               flags=cv2.INTER_NEAREST)


# ─────────────────────────────────────────────
# SOBEL FILTERING
# ─────────────────────────────────────────────

SOBEL_KERNEL = 3
SOBEL_THRESH_MIN = 30
SOBEL_THRESH_MAX = 150
# Gaussian blur standard deviation for pre-processing
SIGMA = 2.0

# Sobel edge magnitude threshold
SOBEL_THRESHOLD = 60

def edge_detection(image):
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
    # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # white_color = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    # yellow_color = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)
    # white_color = cv2.bitwise_and(white_color, cv2.bitwise_not(yellow_color))

    # if VIRTUAL:
    #     green_mask = cv2.inRange(hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)
    #     white_color = cv2.bitwise_or(white_color, green_mask)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # white_color = cv2.morphologyEx(white_color, cv2.MORPH_OPEN, kernel)

    # red_color = filter_red(hsv)

    # white_edges = cv2.bitwise_and(white_color, edge_mask)
    # edges = cv2.bitwise_and(
    #     edge_mask, cv2.bitwise_and(mask_sobelx_neg, mask_sobely_pos)
    # )
    edges = cv2.bitwise_and(
        edge_mask, mask_sobelx_neg
    )

    return edges
# ─────────────────────────────────────────────
# HISTOGRAM PEAK DETECTION
# ─────────────────────────────────────────────

def histogram_peaks(binary: np.ndarray):
    bottom_half = binary[int(binary.shape[0] * 0.60):, :]
    histogram = np.sum(bottom_half, axis=0).astype(np.float32)

    # Smooth so we find the lane's general band, not individual pixel spikes
    histogram = cv2.GaussianBlur(
        histogram.reshape(1, -1), (31, 1), 0
    ).flatten()

    midpoint = histogram.shape[0] // 2
    leftx_base  = int(np.argmax(histogram[:midpoint]))
    rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint)
    return leftx_base, rightx_base, histogram

def histogram_peaks(binary: np.ndarray):
    bottom_half = binary[int(binary.shape[0] * 0.60):, :]
    histogram = np.sum(bottom_half, axis=0).astype(np.float32)
    histogram = cv2.GaussianBlur(histogram.reshape(1, -1), (31, 1), 0).flatten()

    midpoint = histogram.shape[0] // 2
    leftx_base  = int(np.argmax(histogram[:midpoint]))
    rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint)

    # Guard: if peaks are suspiciously close, they're on the same lane
    if abs(rightx_base - leftx_base) < int(LANE_WIDTH_PX * 0.4):
        if histogram[rightx_base] >= histogram[leftx_base]:
            leftx_base = -1   # invalid → sliding window will find nothing
        else:
            rightx_base = binary.shape[1]  # out of bounds → same effect

    return leftx_base, rightx_base, histogram


# ─────────────────────────────────────────────
# SLIDING WINDOW SEARCH + CURVE FITTING
# ─────────────────────────────────────────────

def track_single_lane(binary, start_x, start_y, window_margin, n_windows, min_pix):
    h, w = binary.shape
    step_size = h // n_windows

    nonzeroy, nonzerox = np.array(binary.nonzero()[0]), np.array(binary.nonzero()[1])

    lane_inds, windows = [], []

    curr_x, curr_y = start_x, start_y
    prev_cx, prev_cy = start_x, start_y + step_size   # fake "below" start
    vx, vy = 0.0, -1.0                                 # initial direction: straight up
    y_margin = step_size // 2

    for _ in range(n_windows):
        if not (0 <= curr_x < w and 0 <= curr_y < h):
            break

        win_x_low  = int(curr_x - window_margin)
        win_x_high = int(curr_x + window_margin)
        win_y_low  = int(curr_y - y_margin)
        win_y_high = int(curr_y + y_margin)
        windows.append((win_x_low, win_x_high, win_y_low, win_y_high))

        good_inds = (
            (nonzeroy >= win_y_low)  & (nonzeroy < win_y_high) &
            (nonzerox >= win_x_low) & (nonzerox < win_x_high)
        ).nonzero()[0]
        lane_inds.append(good_inds)

        # if np.sum(binary[nonzeroy[good_inds], nonzerox[good_inds]]) < 2:
        #     break
            # return np.concatenate(lane_inds) if lane_inds else np.array([], dtype=int), windows

        if len(good_inds) > min_pix:
            mean_x = float(np.mean(nonzerox[good_inds]))
            mean_y = float(np.mean(nonzeroy[good_inds]))

            dx = mean_x - prev_cx
            dy = mean_y - prev_cy
            mag = np.hypot(dx, dy)
            if mag > 0:
                vx, vy = dx / mag, dy / mag

            # ── THE FIX: clamp lateral step so one bad centroid can't
            #    flip the staircase direction ─────────────────────────
            raw_next_x = mean_x + vx * step_size
            raw_next_y = mean_y + vy * step_size
            curr_x = float(np.clip(raw_next_x,
                                   mean_x - window_margin,
                                   mean_x + window_margin))
            curr_y = float(raw_next_y)

            prev_cx, prev_cy = mean_x, mean_y

        else:
            # Coast — but clamp lateral drift here too
            old_cx = curr_x
            curr_x += vx * step_size
            curr_y += vy * step_size
            curr_x = float(np.clip(curr_x,
                                   old_cx - window_margin * 0.5,
                                   old_cx + window_margin * 0.5))

    lane_inds = np.concatenate(lane_inds) if lane_inds else np.array([], dtype=int)
    return lane_inds, windows

def sliding_window_search(binary: np.ndarray, leftx_base: int, rightx_base: int):
    N_WINDOWS = 9
    WINDOW_MARGIN = 60  # Lowered from 100. 40 gives an 80px wide search area.
    MIN_PIX = 30
    # MIN_FIT_POINTS = 50
    MIN_FIT_POINTS = 2500
    
    h = binary.shape[0]
    step_size = h // N_WINDOWS
    
    # Start exactly half a step-size from the bottom so the first box sits perfectly on the bottom edge
    start_y = h - (step_size // 2)

    left_inds, left_windows = track_single_lane(
        binary, leftx_base, start_y, WINDOW_MARGIN, N_WINDOWS, MIN_PIX
    )
    right_inds, right_windows = track_single_lane(
        binary, rightx_base, start_y, WINDOW_MARGIN, N_WINDOWS, MIN_PIX
    )
    
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]
    # print(len(lefty), len(righty))

    left_fit = np.polyfit(lefty, leftx, 2) if len(lefty) >= MIN_FIT_POINTS else None
    right_fit = np.polyfit(righty, rightx, 2) if len(righty) >= MIN_FIT_POINTS else None
    # right_fit = None

    return left_fit, right_fit, left_windows + right_windows
# ─────────────────────────────────────────────
# LANE CENTER + HEADING ERROR
# ─────────────────────────────────────────────

LOOKAHEAD_Y_FRAC = 0.6
LANE_HALF_WIDTH_PX = 150

LANE_WIDTH_PX = 380   # pixels in bird's-eye view — calibrate this once

# def compute_centre_fit(left_fit, right_fit, h: int):
#     """
#     Returns a 2nd-degree polynomial x = f(y) for the lane centre.
#     Works with both, left-only, or right-only detections.
#     Returns None if neither fit exists.
#     """
#     ploty = np.linspace(0, h - 1, h)

#     if left_fit is not None and right_fit is not None:
#         leftx  = np.polyval(left_fit,  ploty)
#         rightx = np.polyval(right_fit, ploty)

#         # Sanity-check: reject if the two curves cross or are absurdly wide
#         mean_width = float(np.mean(rightx - leftx))
#         if not (LANE_WIDTH_PX * 0.4 < mean_width < LANE_WIDTH_PX * 2.0):
#             # Implausible — fall through to single-lane logic
#             left_fit = left_fit if abs(mean_width) < LANE_WIDTH_PX * 2 else None
#             right_fit = None

#     if left_fit is not None and right_fit is not None:
#         centerx = (np.polyval(left_fit, ploty) + np.polyval(right_fit, ploty)) / 2.0

#     elif left_fit is not None:
#         # Only yellow (left) lane — centre is half a lane to the right
#         centerx = np.polyval(left_fit, ploty) + LANE_WIDTH_PX / 2.0

#     elif right_fit is not None:
#         # Only white (right) lane — centre is half a lane to the left
#         centerx = np.polyval(right_fit, ploty) - LANE_WIDTH_PX / 2.0

#     else:
#         return None

#     return np.polyfit(ploty, centerx, 2)

def compute_centre_fit(left_fit, right_fit, h: int, w: int):
    ploty = np.linspace(0, h - 1, h)

    if left_fit is not None and right_fit is not None:
        leftx  = np.polyval(left_fit,  ploty)
        rightx = np.polyval(right_fit, ploty)
        mean_width = float(np.mean(rightx - leftx))

        if not (LANE_WIDTH_PX * 0.4 < mean_width < LANE_WIDTH_PX * 2.0):
            # Both fits landed on the same line — figure out WHICH lane it is
            # by checking where the line actually sits in the image
            detected_x = float(np.mean((leftx + rightx) / 2.0))
            if detected_x > w / 2.0:
                left_fit = None   # line is right-of-centre → it's the right lane
            else:
                right_fit = None  # line is left-of-centre → it's the left lane

    if left_fit is not None and right_fit is not None:
        centerx = (np.polyval(left_fit, ploty) + np.polyval(right_fit, ploty)) / 2.0
    elif left_fit is not None:
        centerx = np.polyval(left_fit, ploty) + LANE_WIDTH_PX / 2.0
    elif right_fit is not None:
        centerx = np.polyval(right_fit, ploty) - LANE_WIDTH_PX / 2.0
    else:
        return None

    return np.polyfit(ploty, centerx, 2)


def compute_waypoint(centre_fit, h: int, lookahead_frac: float = 0.35):
    """
    Sample one lookahead point from the centre polynomial.
    lookahead_frac: 0 = top of image (far), 1 = bottom (near car).
    """
    if centre_fit is None:
        return None
    wy = int(h * lookahead_frac)
    wx = float(np.polyval(centre_fit, wy))
    return wx, wy


def estimate_heading_error(target_x: float, image_width: int) -> float:
    image_center_x = image_width / 2.0
    error = (target_x - image_center_x) / image_center_x
    return float(np.clip(error, -1.0, 1.0))


def heading_to_wheel_commands(heading_error: float) -> Tuple[float, float]:
    correction = STEERING_GAIN * heading_error
    correction = float(np.clip(correction, -MAX_SPEED_DIFF, MAX_SPEED_DIFF))
    vel_left = float(np.clip(BASE_SPEED + correction, -1.0, 1.0))
    vel_right = float(np.clip(BASE_SPEED - correction, -1.0, 1.0))
    return vel_left, vel_right


# ─────────────────────────────────────────────
# DEBUG VISUALIZATION (warped binary + sliding windows)
# ─────────────────────────────────────────────

def visualize_lane_detection(binary, left_fit, right_fit, windows) -> np.ndarray:
    """
    Draws the sliding windows and fitted polynomials on top of the
    binary mask, scaled to a viewable BGR image.
    """
    out = (np.dstack((binary, binary, binary)) * 255).astype(np.uint8)

    # for (xl_low, xl_high, xr_low, xr_high, y_low, y_high) in windows:
    #     cv2.rectangle(out, (xl_low, y_low), (xl_high, y_high), (0, 255, 0), 2)
    #     cv2.rectangle(out, (xr_low, y_low), (xr_high, y_high), (0, 255, 0), 2)

    for (xlow, xhigh, ylow, yhigh) in windows:
        cv2.rectangle(out, (xlow, ylow), (xhigh, yhigh), (0, 255, 0), 2)

    h = binary.shape[0]
    ploty = np.linspace(0, h - 1, h)

    if left_fit is not None:
        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        pts = np.array([np.transpose(np.vstack([leftx, ploty]))], dtype=np.int32)
        cv2.polylines(out, pts, False, (255, 0, 0), 3)

    if right_fit is not None:
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        pts = np.array([np.transpose(np.vstack([rightx, ploty]))], dtype=np.int32)
        cv2.polylines(out, pts, False, (0, 0, 255), 3)

    return out


LANE_FILL_COLOR = (230, 216, 173)   # light blue, BGR
WAYPOINT_COLOR = (240, 32, 160)     # purple, BGR
WAYPOINT_RADIUS = 10
OVERLAY_ALPHA = 0.45


def draw_lane_overlay(undistorted, left_fit, right_fit, waypoint=None) -> np.ndarray:
    """
    Fills the area between the two fitted lane curves and draws the
    lookahead waypoint, both in warped (bird's-eye) space, then warps
    everything back onto the undistorted image in a single pass via
    the inverse perspective transform.
    """
    h, w = undistorted.shape[:2]
    overlay_warp = np.zeros((h, w, 3), dtype=np.uint8)

    if left_fit is not None and right_fit is not None:
        ploty = np.linspace(0, h - 1, h)
        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        left_pts = np.array([np.transpose(np.vstack([leftx, ploty]))])
        right_pts = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
        lane_pts = np.hstack((left_pts, right_pts))

        cv2.fillPoly(overlay_warp, np.int_([lane_pts]), LANE_FILL_COLOR)

    if waypoint is not None:
        wx, wy = waypoint
        cv2.circle(overlay_warp, (int(wx), int(wy)), WAYPOINT_RADIUS, WAYPOINT_COLOR, -1)

    overlay = unwarp_perspective(overlay_warp)
    return cv2.addWeighted(undistorted, 1.0, overlay, OVERLAY_ALPHA, 0)

def visualize_lane_detection(binary, left_fit, right_fit, windows) -> np.ndarray:
    """
    Draws the sliding windows and fitted polynomials on top of the
    binary mask, scaled to a viewable BGR image.
    """
    out = (np.dstack((binary, binary, binary)) * 255).astype(np.uint8)

    # for (xl_low, xl_high, xr_low, xr_high, y_low, y_high) in windows:
    #     cv2.rectangle(out, (xl_low, y_low), (xl_high, y_high), (0, 255, 0), 2)
    #     cv2.rectangle(out, (xr_low, y_low), (xr_high, y_high), (0, 255, 0), 2)

    for (xlow, xhigh, ylow, yhigh) in windows:
        cv2.rectangle(out, (xlow, ylow), (xhigh, yhigh), (0, 255, 0), 2)

    h = binary.shape[0]
    ploty = np.linspace(0, h - 1, h)

    if left_fit is not None:
        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        pts = np.array([np.transpose(np.vstack([leftx, ploty]))], dtype=np.int32)
        cv2.polylines(out, pts, False, (255, 0, 0), 3)

    if right_fit is not None:
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        pts = np.array([np.transpose(np.vstack([rightx, ploty]))], dtype=np.int32)
        cv2.polylines(out, pts, False, (0, 0, 255), 3)

    return out

def apply_roi_mask(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    
    # Define a polygon covering ONLY the bottom portion (the road)
    polygon = np.array([[
        (0, h),             # Bottom-left
        (0, int(h * 0.55)), # Mid-left (cuts off top 55%)
        (w, int(h * 0.55)), # Mid-right
        (w, h)              # Bottom-right
    ]], np.int32)
    
    # Fill the polygon with white and apply it via bitwise_and
    cv2.fillPoly(mask, polygon, 255)
    masked_img = cv2.bitwise_and(img, mask)
    
    return masked_img

def get_warp_validity_mask(h: int, w: int) -> np.ndarray:
    """
    Returns a binary mask of pixels that are inside WARP_SRC in the
    original image (i.e. valid after perspective warp).  Apply this to
    the warped binary to kill the sharp artificial border that Sobel picks up.
    """
    src_mask = np.ones((h, w), dtype=np.uint8) * 255
    return cv2.warpPerspective(src_mask, WARP_MATRIX, (w, h),
                               flags=cv2.INTER_NEAREST)


# ─────────────────────────────────────────────
# RED DETECTION
# ─────────────────────────────────────────────

def filter_red(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, RED_HSV_LOWER_1, RED_HSV_UPPER_1)
    m2 = cv2.inRange(hsv, RED_HSV_LOWER_2, RED_HSV_UPPER_2)
    return cv2.bitwise_or(m1, m2)
    # mask = cv2.bitwise_or(m1, m2)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def detect_red_lines(red_mask: np.ndarray) -> list:
    """
    Returns a list of [cx, cy, orientation, angle] for each
    sufficiently large red blob. At most 5 returned (largest by area).
    """
    binary = (red_mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    lines = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            continue
        pixels = np.column_stack(np.where(labels == label))[:, ::-1].astype(np.float32)
        _, _, angle = cv2.minAreaRect(pixels)
        orientation = "vertical" if min(abs(angle), angle + 90) > ANGLE_THRESHOLD else "horizontal"
        cx, cy = int(centroids[label][0]), int(centroids[label][1])
        lines.append([cx, cy, orientation, angle, area])

    lines.sort(key=lambda l: l[4], reverse=True)
    return [l[:4] for l in lines[:5]]   # drop area column


# ─────────────────────────────────────────────
# INTERSECTION WAYPOINTS  (camera / undistorted space)
# ─────────────────────────────────────────────

def compute_stop_line_waypoint(lines: list, image_h: int, image_w: int):
    """
    Point toward the nearest horizontal red line (highest cy).
    Returns (x, y) in camera space, or None.
    """
    horizontals = [l for l in lines if l[2] == "horizontal"]
    if not horizontals:
        return None
    closest = max(horizontals, key=lambda l: l[1])
    cx, cy = closest[0], closest[1]
    return (int(np.clip(cx, 0, image_w)), int(np.clip(cy, 0, image_h)))


def compute_crossing_waypoint(lines: list, image_h: int, image_w: int):
    """
    Look at the OTHER stop lines (not the one the car is halting at),
    pick a random reachable direction, and store the decision globally.
    Returns (x, y) in camera space, or None.
    """
    global crossing_decision, decision_waypoint, decision

    other = [l for l in lines if l[1] < CUT_FRONT_STOP_LINE]
    if not other:
        return None

    v_left  = [l for l in other if l[2] == "vertical"   and l[0] <  LEFT_VS_RIGHT]
    v_right = [l for l in other if l[2] == "vertical"   and l[0] >= LEFT_VS_RIGHT]
    horiz   = [l for l in other if l[2] == "horizontal"]

    choices = {}
    if len(horiz)   == 1: choices["straight"] = horiz[0]
    if len(v_left)  == 1: choices["left"]     = v_left[0]
    if len(v_right) == 1: choices["right"]    = v_right[0]
    if not choices:       # looser fallback
        if horiz:   choices["straight"] = horiz[0]
        if v_left:  choices["left"]     = v_left[0]
        if v_right: choices["right"]    = v_right[0]

    if not choices:
        return None

    # decision, chosen = random.choice(list(choices.items()))
    if choices.get('straight'):
        decision, chosen = 'straight', choices['straight']
    elif choices.get('left'):
        decision, chosen = 'left', choices['left']
    elif choices.get('right'):
        decision, chosen = 'right', choices['right'] 
    offsets = {"straight": CROSSING_OFFSET_TOP,
               "left":     CROSSING_OFFSET_LEFT,
               "right":    CROSSING_OFFSET_RIGHT}
    target = np.array(chosen[:2]) + offsets[decision]

    x = int(np.clip(target[0], 0, image_w))
    y = int(np.clip(target[1], 0, image_h))

    crossing_decision = True
    decision_waypoint = (x, y)
    print("DECISION:", decision)
    return decision_waypoint


# ─────────────────────────────────────────────
# STATE TRANSITIONS
# ─────────────────────────────────────────────

def state_transition(red_mask: np.ndarray):
    global state, state_entered_at, crossing_decision

    def change(s: State):
        global state, state_entered_at, crossing_decision
        state = s
        state_entered_at = time.time()
        crossing_decision = False
        print(f"[FSM] → {s.name}")

    elapsed = time.time() - state_entered_at

    if state == State.DRIVE:
        if np.sum(red_mask[STOP_MARKER_Y:, :] > 0) >= MIN_AREA:
            change(State.STOP)

    elif state == State.STOP:
        if elapsed >= STOP_TIME:
            change(State.FOLLOW)

    elif state == State.FOLLOW:
        t = FOLLOW_TIME.get(decision, 3.0)
        if elapsed >= t:
            change(State.CROSS)

    elif state == State.CROSS:
        if elapsed >= CROSS_TIME:
            change(State.DRIVE)

# ─────────────────────────────────────────────
# MAIN PIPELINE ENTRY POINT
# ─────────────────────────────────────────────

# def process_all(data):
#     """
#     Returns:
#         vel_left, vel_right: computed wheel velocities
#         undistorted: rectified camera image
#         warped:       bird's-eye perspective warp of undistorted
#         lane_vis:     binary lane mask with sliding windows + fit overlaid
#         overlay:      detected lane (light blue) + waypoint (purple)
#                       projected back onto the undistorted image
#     """
#     # dist_img = data.img
#     # undistorted = cv2.remap(dist_img, data.mapx, data.mapy, cv2.INTER_NEAREST)
#     # h, w = undistorted.shape[:2]
#     # if h != w:
#     #     undistorted = cv2.resize(undistorted, (w, w), interpolation=cv2.INTER_NEAREST)
#     # image_width = undistorted.shape[1]

#     # warped = warp_perspective(undistorted)
#     # binary = sobel_threshold(warped)
#     # leftx_base, rightx_base = histogram_peaks(binary)
#     # left_fit, right_fit, windows = sliding_window_search(binary, leftx_base, rightx_base)
#     # # left_fit, right_fit, windows = sliding_window_search(binary, None, None)

#     # lane_vis = visualize_lane_detection(binary, left_fit, right_fit, windows)

#     # lookahead_y = binary.shape[0] * LOOKAHEAD_Y_FRAC
#     # center_x = compute_center_x(left_fit, right_fit, lookahead_y)

#     # if center_x is None:
#     #     vel_left, vel_right = 0.0, 0.0
#     #     waypoint = None
#     # else:
#     #     heading_error = estimate_heading_error(center_x, image_width)
#     #     vel_left, vel_right = heading_to_wheel_commands(heading_error)
#     #     waypoint = (center_x, lookahead_y)

#     # overlay = draw_lane_overlay(undistorted, left_fit, right_fit, waypoint)
#     ud = undistort_image(data.img)
#     # ud = cv2.remap(data.img, data.mapx, data.mapy, cv2.INTER_NEAREST)
#     image_width = ud.shape[1]
#     # ed= edge_detection(ud)
#     # persp = warp_perspective(ed)
#     ed = edge_detection(ud)
#     ed = apply_roi_mask(ed)
#     persp = warp_perspective(ed)

#     validity = get_warp_validity_mask(*ed.shape[:2])
#     persp = cv2.bitwise_and(persp, validity)
#     h = persp.shape[0]
#     left, right, hist = histogram_peaks(persp)
#     left_fit, right_fit, windows = sliding_window_search(persp, left, right)
#     centre_fit = compute_centre_fit(left_fit, right_fit, h, persp.shape[1])
#     waypoint   = compute_waypoint(centre_fit, h)

#     if waypoint is None:
#         vel_left = 0.0
#         vel_right = 0.0
#     else:
#         wx, wy = waypoint

#         heading_error = estimate_heading_error(wx, image_width)

#         vel_left, vel_right = heading_to_wheel_commands(heading_error)
#     lane_vis = visualize_lane_detection(persp, left_fit, right_fit, windows)
#     overlay = draw_lane_overlay(ud, left_fit, right_fit, waypoint)

#     return vel_left, vel_right, ud, persp, lane_vis, overlay

def process_all(data):
    global crossing_vel_left, crossing_vel_right

    ud = undistort_image(data.img)
    image_h, image_w = ud.shape[:2]

    # ── Red detection + state machine ──────────────────────────────
    red_mask = filter_red(ud)
    state_transition(red_mask)

    # ── Always run the lane pipeline (needed for DRIVE + vis) ──────
    ed   = edge_detection(ud)
    ed   = apply_roi_mask(ed)
    persp = warp_perspective(ed)
    validity = get_warp_validity_mask(*ed.shape[:2])
    persp = cv2.bitwise_and(persp, validity)
    h = persp.shape[0]

    left_base, right_base, _ = histogram_peaks(persp)
    left_fit, right_fit, windows = sliding_window_search(persp, left_base, right_base)
    centre_fit = compute_centre_fit(left_fit, right_fit, h, persp.shape[1])
    lane_waypoint = compute_waypoint(centre_fit, h)

    # ── Choose waypoint + velocities based on state ────────────────
    waypoint = None

    if state == State.DRIVE:
        waypoint = lane_waypoint
        if waypoint is None:
            vel_left = vel_right = 0.0
        else:
            wx, _ = waypoint
            heading_error = estimate_heading_error(wx, image_w)
            vel_left, vel_right = heading_to_wheel_commands(heading_error)
            crossing_vel_left, crossing_vel_right = vel_left, vel_right

    elif state == State.STOP:
        # Point toward stop line so we stop in a good position, but hold still
        lines = detect_red_lines(red_mask)
        waypoint = compute_stop_line_waypoint(lines, image_h, image_w) if lines else None
        vel_left = vel_right = 0.0

    elif state == State.FOLLOW:
        if not crossing_decision:
            lines = detect_red_lines(red_mask)
            if lines:
                waypoint = compute_crossing_waypoint(lines, image_h, image_w)
        else:
            waypoint = decision_waypoint

        if waypoint is not None:
            wx, _ = waypoint
            heading_error = estimate_heading_error(wx, image_w)
            vel_left, vel_right = heading_to_wheel_commands(heading_error)
            crossing_vel_left, crossing_vel_right = vel_left, vel_right
        else:
            vel_left, vel_right = crossing_vel_left, crossing_vel_right

    elif state == State.CROSS:
        # Blind drive — use the velocities frozen at the end of FOLLOW
        vel_left, vel_right = crossing_vel_left, crossing_vel_right
        waypoint = decision_waypoint

    # ── Visualisation ──────────────────────────────────────────────
    lane_vis = visualize_lane_detection(persp, left_fit, right_fit, windows)

    # Lane overlay works in warped space; for non-DRIVE states draw
    # the target point directly on the camera image instead
    if state == State.DRIVE:
        overlay = draw_lane_overlay(ud, left_fit, right_fit, lane_waypoint)
    else:
        overlay = ud.copy()
        if waypoint is not None:
            cv2.circle(overlay, (int(waypoint[0]), int(waypoint[1])),
                       WAYPOINT_RADIUS, WAYPOINT_COLOR, -1)
        # Tint the overlay to show which state we're in
        tint = {State.STOP:   (0,   0, 60),
                State.FOLLOW: (0,  60,  0),
                State.CROSS:  (60,  0,  0)}
        overlay = cv2.addWeighted(overlay, 0.85,
                                  np.full_like(overlay, tint[state]), 0.15, 0)

    return vel_left, vel_right, ud, persp, lane_vis, overlay