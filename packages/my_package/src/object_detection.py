#!/usr/bin/env python3
import config
import numpy as np
import onnxruntime as ort


class ODModel:
    def __init__(self):
        print("Initializing ODModel", flush=True)

        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                "ONNX model not found (did you download your trained model?):",
                config.MODEL_PATH,
            )

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(config.MODEL_PATH),
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32

        self.net_h = inp.shape[2]
        self.net_w = inp.shape[3]

    def _run_detector(self, img_bgr):
        x = self._preprocess(img_bgr)
        out = self.session.run(None, {self.input_name: x})[0]  # shape [1,N,6]
        return out[0]

    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]

        if h != self.net_h or w != self.net_w:
            raise ValueError(
                f"Image size {h}x{w} does not match ONNX! Expected {self.net_h}x{self.net_w}"
            )

        img = img_bgr[:, :, ::-1].astype(self.in_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def get_detections(self, img_bgr: np.ndarray) -> np.ndarray:
        """Runs OD model and returns detections with confidence over threshold.

        Args:
            img_bgr: BGR image from the Duckiebot camera (H x W x 3).

        Returns:
            thresholded_detections: Numpy array of shape (N, 6) where each row is
                [x1, y1, x2, y2, score, class_id] for a detected object with confidence
                above the threshold.
        """
        detections = self._run_detector(img_bgr)
        thresholded_detections = detections[detections[:, 4] >= config.CONF_THRESHOLD]
        return thresholded_detections

    def get_bottom_center_detections(self, img_bgr: np.ndarray) -> np.ndarray:
        """Runs OD model and returns bottom-center points of detections with confidence over threshold.

        Args:
            img_bgr: BGR image from the Duckiebot camera (H x W x 3).

        Returns:
            bottom_center_detections: Numpy array of shape (N, 2) where each row is
                [x, y] for the bottom-center point of a detected object with confidence
                above the threshold.
        """
        detections = self.get_detections(img_bgr)
        bottom_center_detections = []
        for det in detections:
            x1, _, x2, y2, _, _ = det
            bottom_center_x = (x1 + x2) / 2
            bottom_center_y = y2
            bottom_center_detections.append([bottom_center_x, bottom_center_y])
        return np.array(bottom_center_detections)
