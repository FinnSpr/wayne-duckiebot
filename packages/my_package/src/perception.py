from typing import List

import config
import cv2
import numpy as np
import image_utils
from object_detection import ODModel, get_bottom_center_detections, get_negative_mask
from segmentation import SEGModel
from world_model import WorldModel


class PerceptionModule:
    """
    Perception Module.
    Handles image processing, color filtering, edge detection, and obstacle detection.
    """

    def __init__(
        self,
        K: np.ndarray,
        D: np.ndarray,
        P: np.ndarray,
        H: np.ndarray,
        use_object_detection: bool = False,
    ):
        self.use_object_detection = use_object_detection
        if self.use_object_detection:
            self.od_model = ODModel()
        else:
            self.od_model = None

        if config.USE_SEGMENTATION:
            self.seg_model = SEGModel()
        else:
            self.seg_model = None

        self.K = K
        self.D = D
        self.P = P
        self.H = image_utils.adjust_homography(H)

        # Cached perception results
        self.proc_image: np.ndarray = None
        self.right_white_lane: np.ndarray = None
        self.left_white_lane: np.ndarray = None
        self.yellow_mask: np.ndarray = None
        self.red_mask: np.ndarray = None
        self.edge_mask: np.ndarray = None
        self.white_color: np.ndarray = None
        self.duckie_detections: np.ndarray = (
            None  # raw OD detections [x1,y1,x2,y2,score,cls]
        )
        self.duckies_bottom_centers: np.ndarray = None  # (N, 2) bottom-center points
        self.duckies_bottom_centers_world: np.ndarray = None
        self.detections_negative_mask: np.ndarray = None
        self.image_width: int = 0
        self.image_height: int = 0

    def perceive(
        self,
        image: np.ndarray,
        use_enhanced: bool = True,
        world_model: WorldModel = None,
    ) -> None:
        """Run the full perception pipeline and cache results as member variables.

        After calling this, the following attributes are populated:
            proc_image, right_white_lane, left_white_lane, yellow_mask,
            red_mask, edge_mask, white_color, duckie_detections,
            duckies_bottom_centers, image_width, image_height,
            white_spline, yellow_spline (when world_model given).
        """
        if use_enhanced and self.K is not None:
            self.proc_image = image_utils.unwarp_image(image, self.K, self.D, self.P)
        else:
            self.proc_image = image
        self.proc_image = image_utils.adjust_image(self.proc_image)
        self.image_height, self.image_width = self.proc_image.shape[:2]

        self._detect_objects(self.proc_image)

        if use_enhanced:
            self.filter_lane_colors_enhanced(
                self.proc_image, self.detections_negative_mask
            )
        else:
            self.filter_lane_colors_standard(self.proc_image)
            self.left_white_lane = None
            self.edge_mask = None
            self.white_color = self.right_white_lane

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

    def get_raw_lane_colors(self, image: np.ndarray):
        """Returns (white_color, yellow_color, red_color) masks (uint8 0/255).

        Uses segmentation model if USE_SEGMENTATION is True, otherwise HSV filters.
        """
        if config.USE_SEGMENTATION and self.seg_model is not None:
            yellow_mask, white_mask, red_mask = self.seg_model.get_lane_masks(
                self._seg_detections
            )
            white_color = white_mask
            yellow_color = yellow_mask
            red_color = red_mask
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            white_color = cv2.inRange(
                hsv, config.WHITE_HSV_LOWER, config.WHITE_HSV_UPPER
            )
            yellow_color = cv2.inRange(
                hsv, config.YELLOW_HSV_LOWER, config.YELLOW_HSV_UPPER
            )
            white_color = cv2.bitwise_and(white_color, cv2.bitwise_not(yellow_color))
            red_color = self.filter_red(hsv)
        return white_color, yellow_color, red_color

    def _detect_objects(self, image: np.ndarray) -> None:
        """Run object detection and populate detection-related member variables.

        Sets self.duckie_detections, self.duckies_bottom_centers,
        self.duckies_bottom_centers_world, and self.detections_negative_mask.
        """
        if not self.use_object_detection and not config.USE_SEGMENTATION:
            self.duckie_detections = np.empty((0, 6))
            self.duckies_bottom_centers = np.empty((0, 2))
            self.duckies_bottom_centers_world = np.empty((0, 2))
            self.detections_negative_mask = None
            return

        if config.USE_SEGMENTATION and self.seg_model is not None:
            self._seg_detections = self.seg_model.get_detections(image)
            self.seg_model._last_detections = self._seg_detections
            self.duckie_detections = self.seg_model.get_duckie_detections(
                self._seg_detections
            )
            self.detections_negative_mask = None
        else:
            self.duckie_detections = self.od_model.get_detections(image)
            self.detections_negative_mask = get_negative_mask(
                self.image_height, self.image_width, self.duckie_detections
            )

        self.duckies_bottom_centers = get_bottom_center_detections(
            self.duckie_detections
        )
        self.duckies_bottom_centers_world = image_utils.image_to_world_coords(
            self.duckies_bottom_centers, self.H
        )

    def filter_lane_colors_enhanced(
        self, image: np.ndarray, obstacle_negative_mask: np.ndarray = None
    ) -> None:
        """Enhanced lane filtering using Sobel edge magnitudes and HSV color masks.
        Saves results directly to self."""
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=config.SIGMA)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_mask = (magnitude > config.SOBEL_THRESHOLD).astype(np.uint8) * 255

        mask_sobelx_pos = ((sobel_x > 0).astype(np.uint8)) * 255
        mask_sobelx_neg = ((sobel_x < 0).astype(np.uint8)) * 255
        mask_sobely_pos = ((sobel_y > 0).astype(np.uint8)) * 255

        white_color, yellow_color, red_color = self.get_raw_lane_colors(blurred)

        if obstacle_negative_mask is not None and not config.USE_SEGMENTATION:
            yellow_color = cv2.bitwise_and(yellow_color, obstacle_negative_mask)
            red_color = cv2.bitwise_and(red_color, obstacle_negative_mask)

        if config.VIRTUAL:
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(
                hsv, config.GREEN_HSV_LOWER, config.GREEN_HSV_UPPER
            )
            white_color = cv2.bitwise_or(white_color, green_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        white_color = cv2.morphologyEx(white_color, cv2.MORPH_OPEN, kernel)

        white_edges = cv2.bitwise_and(white_color, edge_mask)
        self.right_white_lane = cv2.bitwise_and(
            white_edges, cv2.bitwise_and(mask_sobelx_neg, mask_sobely_pos)
        )
        self.left_white_lane = cv2.bitwise_and(
            white_edges, cv2.bitwise_and(mask_sobelx_pos, mask_sobely_pos)
        )

        if config.WHITE_LANE_MIN_AREA > 0:
            self.right_white_lane = self.filter_components_over_threshold(
                self.right_white_lane
            )
            self.left_white_lane = self.filter_components_over_threshold(
                self.left_white_lane
            )

        self.yellow_mask = cv2.bitwise_and(yellow_color, edge_mask)
        self.yellow_mask = cv2.bitwise_and(
            self.yellow_mask, cv2.bitwise_and(mask_sobelx_pos, mask_sobely_pos)
        )

        self.right_white_lane = self._ensure_min_pixels(self.right_white_lane)
        self.left_white_lane = self._ensure_min_pixels(self.left_white_lane)
        self.yellow_mask = self._ensure_min_pixels(self.yellow_mask)
        self.red_mask = red_color
        self.edge_mask = edge_mask
        self.white_color = white_color

    def filter_red(self, hsv: np.ndarray) -> np.ndarray:
        """Extract red color mask from HSV image."""
        mask1 = cv2.inRange(hsv, config.RED_HSV_LOWER_1, config.RED_HSV_UPPER_1)
        mask2 = cv2.inRange(hsv, config.RED_HSV_LOWER_2, config.RED_HSV_UPPER_2)
        return cv2.bitwise_or(mask1, mask2)

    def filter_lane_colors_standard(self, image: np.ndarray) -> None:
        """Standard lane filtering using simple HSV ranges.
        Saves results directly to self."""
        _, width = image.shape[:2]

        white_mask, yellow_mask, red_mask = self.get_raw_lane_colors(image)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        white_mask[:, : width // 2] = 0

        self.right_white_lane = white_mask
        self.yellow_mask = yellow_mask
        self.red_mask = red_mask

    def _apply_lane_roi(self) -> None:
        roi = config.HIDE_TOP_OF_IMAGE
        for attr in ("left_white_lane", "right_white_lane", "yellow_mask", "red_mask"):
            mask = getattr(self, attr, None)
            if mask is not None:
                mask[:roi, :] = 0

    def _ensure_min_pixels(self, mask: np.ndarray) -> np.ndarray:
        """Ensure that the mask has at least min_pixels non-zero pixels."""
        if np.count_nonzero(mask) < config.MIN_LANE_PIXELS:
            return None
        return mask

    def filter_components_over_threshold(
        self, mask: np.ndarray, min_area: float = config.WHITE_LANE_MIN_AREA
    ) -> np.ndarray:
        """Filter connected components by area threshold.

        Returns a binary mask keeping only components whose area is at least
        min_area. If min_area is None, uses config.WHITE_LANE_MIN_AREA.
        """

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return np.zeros_like(mask)

        filtered = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == i] = 255
        return filtered
