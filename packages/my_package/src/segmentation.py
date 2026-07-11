#!/usr/bin/env python3

from pathlib import Path

import config
import cv2
import numpy as np
import onnxruntime as ort

print("AVAILABLE PROVIDERS: ", ort.get_available_providers())

# Threshold configurations
STOP_AREA = 2000  # mask pixel-count considered "too close"
CENTROID_INTERVAL = [160, 550, 100, 480]  # x1, x2, y1, y2 (net-input pixel space)
MASK_ALPHA = 0.5  # overlay transparency for visualization

# --- Custom Duckietown Class Color Map (BGR) ---
COLORS = {
    -1: (50, 50, 50),  # background → dark grey
    0: (0, 200, 255),  # yellow lane marking → gold
    1: (200, 200, 200),  # white lane marking  → light grey
    2: (0, 0, 255),  # red stop line
    3: (0, 180, 255),  # yellow rubber duck  → amber
    4: (255, 0, 0),  # blue duckiebot
}

"""
TODO: 
1. change model
2. use smaller image size
"""


class SEGModel:
    def __init__(self, num_classes=5):
        print("Initializing End-to-End SEGModel with Custom Colors", flush=True)

        if not config.SEG_MODEL_PATH.exists():
            raise FileNotFoundError(
                "ONNX model not found (did you download your trained model?):",
                config.SEG_MODEL_PATH,
            )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            str(config.SEG_MODEL_PATH),
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
        self.net_h = inp.shape[2]
        self.net_w = inp.shape[3]

        outs = self.session.get_outputs()
        self.pred_name = outs[0].name
        self.proto_name = outs[1].name
        self.mask_dim = outs[1].shape[1]  # e.g. 32

        self.num_classes = num_classes
        self.class_colors = COLORS
        print(
            f"Model Output Shape: {outs[0].shape} | Explicit Classes: {self.num_classes}",
            flush=True,
        )

        self._last_img = None
        self._last_detections = []

    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]

        if h != self.net_h or w != self.net_w:
            img_bgr = cv2.resize(
                img_bgr,
                (self.net_w, self.net_h),
                interpolation=cv2.INTER_LINEAR,
            )

        img = img_bgr[:, :, ::-1].astype(self.in_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def _run_detector(self, img_bgr):
        x = self._preprocess(img_bgr)
        preds, proto = self.session.run(
            [self.pred_name, self.proto_name], {self.input_name: x}
        )
        preds = preds[0]  # Shape: (300, 38)
        proto = proto[0]  # Shape: (32, mask_h, mask_w)
        return preds, proto

    def _decode(self, preds, proto):
        """Parses the pre-filtered NMS output array directly."""

        boxes_xyxy = preds[:, :4]
        scores = preds[:, 4]
        class_ids = preds[:, 5].astype(np.int32)
        mask_coeffs = preds[:, 6:]

        keep = scores >= config.SEG_CONF_THRESHOLD
        if not np.any(keep):
            return []

        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        mask_coeffs = mask_coeffs[keep]

        mask_h, mask_w = proto.shape[1:]
        proto_flat = proto.reshape(self.mask_dim, -1)
        detections = []

        for i in range(len(scores)):
            box, score, cls_id = boxes_xyxy[i], float(scores[i]), int(class_ids[i])
            coeff = mask_coeffs[i]

            # Prevent FP16 Math Underflow/Overflow
            coeff_f32 = coeff.astype(np.float32)
            proto_flat_f32 = proto_flat.astype(np.float32)

            # Blend prototypes -> single-channel soft mask at proto resolution
            mask = 1 / (1 + np.exp(-(coeff_f32 @ proto_flat_f32)))
            mask = mask.reshape(mask_h, mask_w)

            # Bilinear Upsample Soft Mask FIRST, Then Threshold and Crop
            mask_large = cv2.resize(
                mask,
                (self.net_w, self.net_h),
                interpolation=cv2.INTER_LINEAR,
            )

            bx1, by1, bx2, by2 = box
            mx1, my1 = max(int(bx1), 0), max(int(by1), 0)
            mx2, my2 = (
                min(int(np.ceil(bx2)), self.net_w),
                min(int(np.ceil(by2)), self.net_h),
            )

            if mx2 <= mx1 or my2 <= my1:
                continue

            crop = np.zeros_like(mask_large, dtype=bool)
            crop[my1:my2, mx1:mx2] = True

            mask_full = (mask_large > 0.5) & crop

            detections.append(
                {"box": box, "score": score, "cls_id": cls_id, "mask": mask_full}
            )

        return detections

    def _should_stop(self, detections):
        for det in detections:
            mask = det["mask"]
            area = int(mask.sum())
            if area == 0:
                continue

            ys, xs = np.nonzero(mask)
            centroid = [float(xs.mean()), float(ys.mean())]

            if area >= STOP_AREA:
                if (
                    centroid[0] >= CENTROID_INTERVAL[0]
                    and centroid[0] <= CENTROID_INTERVAL[1]
                    and centroid[1] >= CENTROID_INTERVAL[2]
                    and centroid[1] <= CENTROID_INTERVAL[3]
                ):
                    print(
                        f"Blocking object detected: x={centroid[0]:.1f}, y={centroid[1]:.1f}, "
                        f"area={area}px, confidence={det['score']:.2f}",
                        flush=True,
                    )
                    return True

        return False

    def get_lane_masks(self, detections=None):
        """Returns (yellow_mask, white_mask, red_mask) each uint8 (0/255) of shape (H, W)."""
        detections = self._last_detections if detections is None else detections
        masks = np.zeros((3, self.net_h, self.net_w), dtype=np.uint8)
        for det in detections:
            cls_id = det["cls_id"]
            if 0 <= cls_id <= 2:
                masks[cls_id][det["mask"]] = 255
        return masks[0], masks[1], masks[2]

    def get_duckie_detections(self, detections=None):
        """Returns class-3 (duckie) detections as [x1, y1, x2, y2, score, 3] rows."""
        detections = self._last_detections if detections is None else detections
        duckie_bboxes = []
        for det in detections:
            if det["cls_id"] == 3:
                x1, y1, x2, y2 = det["box"]
                duckie_bboxes.append([x1, y1, x2, y2, det["score"], 3])
        return np.array(duckie_bboxes) if duckie_bboxes else np.empty((0, 6))

    def get_detections(self, img):
        preds, proto = self._run_detector(img)
        return self._decode(preds, proto)

    def stop_for_object(self, img: np.ndarray):
        try:
            preds, proto = self._run_detector(img)
            detections = self._decode(preds, proto)
        except Exception as e:
            print(f"ONNX inference error {e}", flush=True)
            return True

        self._last_img = img
        self._last_detections = detections  # cached for visualization
        return self._should_stop(detections)

    def visualize(self, detections=None):
        """Colored mask overlay on black background, without blending onto the input frame."""
        detections = self._last_detections if detections is None else detections

        overlay = np.zeros((self.net_h, self.net_w, 3), dtype=np.uint8)
        for det in detections:
            color = self.class_colors.get(det["cls_id"], (0, 255, 0))
            overlay[det["mask"]] = color

        return overlay

    def class_map(self, detections=None, shape=None):
        """Raw per-pixel class-id map (background = -1)."""
        detections = self._last_detections if detections is None else detections
        shape = (self.net_h, self.net_w) if shape is None else shape

        cmap = -np.ones(shape, dtype=np.int32)
        for det in sorted(detections, key=lambda d: d["score"]):
            cmap[det["mask"]] = det["cls_id"]
        return cmap
