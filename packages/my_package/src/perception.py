from typing import List, Tuple

import config
import cv2
import numpy as np
from image_utils import unwarp_image
from object_detection import ODModel, get_bottom_center_detections
from world_model import WorldModel


class PerceptionModule:
    """
    Perception Module.
    Handles image processing, color filtering, edge detection, and obstacle detection.
    """

    def __init__(
        self,
        use_object_detection: bool = False,
        min_lane_pixels: int = config.MIN_LANE_PIXELS,
    ):
        self.use_object_detection = use_object_detection
        self.min_lane_pixels = min_lane_pixels
        if self.use_object_detection:
            self.od_model = ODModel()
        else:
            self.od_model = None

        # Cached perception results
        self.proc_image: np.ndarray = None
        self.right_white_lane: np.ndarray = None
        self.left_white_lane: np.ndarray = None
        self.yellow_mask: np.ndarray = None
        self.red_mask: np.ndarray = None
        self.edge_mask: np.ndarray = None
        self.white_color: np.ndarray = None
        self.detections: np.ndarray = None  # raw OD detections [x1,y1,x2,y2,score,cls]
        self.detection_bottom_centers: np.ndarray = None  # (N, 2) bottom-center points
        self.image_width: int = 0
        self.image_height: int = 0

    def filter_red(self, hsv: np.ndarray) -> np.ndarray:
        """Extract red color mask from HSV image."""
        mask1 = cv2.inRange(hsv, config.RED_HSV_LOWER_1, config.RED_HSV_UPPER_1)
        mask2 = cv2.inRange(hsv, config.RED_HSV_LOWER_2, config.RED_HSV_UPPER_2)
        return cv2.bitwise_or(mask1, mask2)

    def filter_lane_colors_enhanced(
        self, image: np.ndarray, obstacle_negative_mask: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Enhanced lane filtering using Sobel edge magnitudes and HSV color masks."""
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=config.SIGMA)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_mask = (magnitude > config.SOBEL_THRESHOLD).astype(np.uint8) * 255

        mask_sobelx_pos = ((sobel_x > 0).astype(np.uint8)) * 255
        mask_sobelx_neg = ((sobel_x < 0).astype(np.uint8)) * 255
        mask_sobely_pos = ((sobel_y > 0).astype(np.uint8)) * 255

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        white_color = cv2.inRange(hsv, config.WHITE_HSV_LOWER, config.WHITE_HSV_UPPER)
        yellow_color = cv2.inRange(
            hsv, config.YELLOW_HSV_LOWER, config.YELLOW_HSV_UPPER
        )
        white_color = cv2.bitwise_and(white_color, cv2.bitwise_not(yellow_color))
        if obstacle_negative_mask is not None:
            yellow_color = cv2.bitwise_and(yellow_color, obstacle_negative_mask)

        if config.VIRTUAL:
            green_mask = cv2.inRange(
                hsv, config.GREEN_HSV_LOWER, config.GREEN_HSV_UPPER
            )
            white_color = cv2.bitwise_or(white_color, green_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        white_color = cv2.morphologyEx(white_color, cv2.MORPH_OPEN, kernel)

        red_color = self.filter_red(hsv)
        white_edges = cv2.bitwise_and(white_color, edge_mask)
        right_edge_right_white_lane = cv2.bitwise_and(
            white_edges, cv2.bitwise_and(mask_sobelx_neg, mask_sobely_pos)
        )
        left_edge_left_white_lane = cv2.bitwise_and(
            white_edges, cv2.bitwise_and(mask_sobelx_pos, mask_sobely_pos)
        )

        if config.WHITE_LANE_ONLY_BIGGEST_COMPONENT:
            right_edge_right_white_lane = self._get_biggest_component(
                right_edge_right_white_lane
            )
            left_edge_left_white_lane = self._get_biggest_component(
                left_edge_left_white_lane
            )

        yellow_mask = cv2.bitwise_and(yellow_color, edge_mask)
        yellow_mask = cv2.bitwise_and(
            yellow_mask, cv2.bitwise_and(mask_sobelx_pos, mask_sobely_pos)
        )

        return (
            self._ensure_min_pixels(right_edge_right_white_lane),
            self._ensure_min_pixels(left_edge_left_white_lane),
            self._ensure_min_pixels(yellow_mask),
            red_color,
            edge_mask,
            white_color,
        )

    def filter_lane_colors_standard(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standard lane filtering using simple HSV ranges."""
        _, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv, config.WHITE_HSV_LOWER, config.WHITE_HSV_UPPER)
        yellow_mask = cv2.inRange(hsv, config.YELLOW_HSV_LOWER, config.YELLOW_HSV_UPPER)

        red_mask = self.filter_red(hsv)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        white_mask[:, : width // 2] = 0

        return white_mask, yellow_mask, red_mask

    def perceive(
        self,
        image: np.ndarray,
        use_enhanced: bool = True,
        world_model: WorldModel = None,
        K: np.ndarray = None,
        D: np.ndarray = None,
        P: np.ndarray = None,
    ) -> None:
        """Run the full perception pipeline and cache results as member variables.

        After calling this, the following attributes are populated:
            proc_image, right_white_lane, left_white_lane, yellow_mask,
            red_mask, edge_mask, white_color, detections,
            detection_bottom_centers, image_width, image_height,
            white_spline, yellow_spline (when world_model given).
        """
        if use_enhanced and K is not None:
            self.proc_image = unwarp_image(image, K, D, P)
        else:
            self.proc_image = image
        self.image_height, self.image_width = self.proc_image.shape[:2]

        if use_enhanced:
            (
                self.right_white_lane,
                self.left_white_lane,
                self.yellow_mask,
                self.red_mask,
                self.edge_mask,
                self.white_color,
            ) = self.filter_lane_colors_enhanced(self.proc_image)
        else:
            self.right_white_lane, self.yellow_mask, self.red_mask = (
                self.filter_lane_colors_standard(self.proc_image)
            )
            self.left_white_lane = None
            self.edge_mask = None
            self.white_color = self.right_white_lane

        # Object detection
        if self.use_object_detection:
            self.detections = self.od_model.get_detections(self.proc_image)
            self.detection_bottom_centers = get_bottom_center_detections(
                self.detections
            )
        else:
            self.detections = np.empty((0, 6))
            self.detection_bottom_centers = np.empty((0, 2))

        # Spline fitting
        if world_model is not None:
            self.white_spline = world_model.fit_spline(
                self.right_white_lane, take_leftmost_pixels=False
            )
            self.yellow_spline = world_model.fit_spline(
                self.yellow_mask, take_leftmost_pixels=True
            )
        else:
            self.white_spline = None
            self.yellow_spline = None

    def _ensure_min_pixels(self, mask: np.ndarray) -> np.ndarray:
        """Ensure that the mask has at least min_pixels non-zero pixels."""
        if np.count_nonzero(mask) < self.min_lane_pixels:
            return np.zeros_like(mask)
        return mask

    def _get_biggest_component(self, mask: np.ndarray) -> np.ndarray:
        """Return a binary mask of the biggest connected component."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return np.zeros_like(mask)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == largest_label).astype(np.uint8) * 255
