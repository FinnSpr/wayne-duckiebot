#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import onnxruntime as ort
import rospkg
import cv2

rospack = rospkg.RosPack()
PKG_ROOT = Path(rospack.get_path("my_package"))
SEG_MODEL_PATH = PKG_ROOT / "segmentation.onnx"  # yolov11n-seg export

CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45          # NMS IoU threshold (seg output isn't pre-NMS'd)
STOP_AREA = 2000              # mask pixel-count considered "too close"
CENTROID_INTERVAL = [160, 550, 100, 480]  # x1, x2, y1, y2 (net-input pixel space)
MASK_ALPHA = 0.5               # overlay transparency for visualization


class SEGModel:
    def __init__(self):
        print("Initializing SEGModel", flush=True)

        if not SEG_MODEL_PATH.exists():
            raise FileNotFoundError(
                "ONNX model not found (did you download your trained model?):",
                SEG_MODEL_PATH,
            )

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(SEG_MODEL_PATH),
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
        self.net_h = inp.shape[2]
        self.net_w = inp.shape[3]

        # -seg models have two outputs: raw predictions + mask prototypes
        outs = self.session.get_outputs()
        self.pred_name = outs[0].name
        self.proto_name = outs[1].name
        self.mask_dim = outs[1].shape[1]                       # e.g. 32
        self.num_classes = outs[0].shape[1] - 4 - self.mask_dim

        self._last_img = None
        self._last_detections = []

        # stable-ish colors per class id, for visualization
        self.class_colors = np.random.default_rng(0).integers(
            0, 255, size=(max(self.num_classes, 1), 3), dtype=np.uint8
        )

    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]

        if h != self.net_h or w != self.net_w:
            raise ValueError(
                f"Image size {h}x{w} does not match ONNX! Expected {self.net_h}x{self.net_w}"
            )

        img = img_bgr[:, :, ::-1].astype(self.in_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def _run_detector(self, img_bgr):
        x = self._preprocess(img_bgr)
        preds, proto = self.session.run(
            [self.pred_name, self.proto_name], {self.input_name: x}
        )
        preds = preds[0].T   # (num_predictions, 4 + num_classes + mask_dim)
        proto = proto[0]     # (mask_dim, mask_h, mask_w)
        return preds, proto

    def _decode(self, preds, proto):
        """Confidence filter -> box decode -> NMS -> per-instance mask assembly."""

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:4 + self.num_classes]
        mask_coeffs = preds[:, 4 + self.num_classes:]

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_scores)), class_ids]

        keep = scores >= CONF_THRESHOLD
        if not np.any(keep):
            return []

        boxes_cxcywh, scores = boxes_cxcywh[keep], scores[keep]
        class_ids, mask_coeffs = class_ids[keep], mask_coeffs[keep]

        cx, cy, bw, bh = boxes_cxcywh.T
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        nms_boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()

        mask_h, mask_w = proto.shape[1:]
        proto_flat = proto.reshape(self.mask_dim, -1)
        detections = []

        for i in idxs:
            box, score, cls_id = boxes_xyxy[i], float(scores[i]), int(class_ids[i])
            coeff = mask_coeffs[i]

            # blend prototypes -> single-channel soft mask at proto resolution
            mask = 1 / (1 + np.exp(-(coeff @ proto_flat)))
            mask = mask.reshape(mask_h, mask_w)

            # crop to this instance's box (scaled into proto-resolution space)
            sx, sy = mask_w / self.net_w, mask_h / self.net_h
            mx1, my1 = max(int(box[0] * sx), 0), max(int(box[1] * sy), 0)
            mx2 = min(int(np.ceil(box[2] * sx)), mask_w)
            my2 = min(int(np.ceil(box[3] * sy)), mask_h)
            crop = np.zeros_like(mask, dtype=bool)
            crop[my1:my2, mx1:mx2] = True
            mask = (mask > 0.5) & crop

            # upsample to net-input resolution so it lines up with the image
            mask_full = cv2.resize(
                mask.astype(np.uint8), (self.net_w, self.net_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            detections.append({"box": box, "score": score, "cls_id": cls_id, "mask": mask_full})

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

    def stop_for_object(self, img: np.ndarray):
        try:
            preds, proto = self._run_detector(img)
            detections = self._decode(preds, proto)
        except Exception as e:
            print(f"ONNX inference error {e}", flush=True)
            return True

        self._last_img = img
        self._last_detections = detections   # cached for visualize()/class_map()
        return self._should_stop(detections)

    # ------------------------------------------------------------------
    # Visualization — this is the "output image" you want to see
    # ------------------------------------------------------------------
    def visualize(self, img_bgr=None, detections=None, alpha=MASK_ALPHA):
        """Colored mask overlay + boxes, blended onto the input frame."""
        img_bgr = self._last_img if img_bgr is None else img_bgr
        detections = self._last_detections if detections is None else detections

        overlay = img_bgr.copy()
        for det in detections:
            color = tuple(int(c) for c in self.class_colors[det["cls_id"] % len(self.class_colors)])
            overlay[det["mask"]] = color

        out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

        for det in detections:
            x1, y1, x2, y2 = det["box"].astype(int)
            color = tuple(int(c) for c in self.class_colors[det["cls_id"] % len(self.class_colors)])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
            cv2.putText(out, f"{det['cls_id']}:{det['score']:.2f}", (x1, max(y1 - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        return out

    def class_map(self, detections=None, shape=None):
        """Raw per-pixel class-id map (background = -1), no color/blend —
        overlapping instances resolved in favor of the higher-confidence one."""
        detections = self._last_detections if detections is None else detections
        shape = (self.net_h, self.net_w) if shape is None else shape

        cmap = -np.ones(shape, dtype=np.int32)
        for det in sorted(detections, key=lambda d: d["score"]):
            cmap[det["mask"]] = det["cls_id"]
        return cmap