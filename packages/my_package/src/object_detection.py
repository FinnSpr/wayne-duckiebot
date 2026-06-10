#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import onnxruntime as ort
import rospkg

rospack = rospkg.RosPack()
PROJECT_ROOT = Path(rospack.get_path("my_package"))
MODEL_PATH = PROJECT_ROOT / "matrix_object_detection.onnx"

CONF_THRESHOLD = 0.5
STOP_SIZE = 1
FORWARD_PWM = 0.2


class ODModel:
    def __init__(self):
        print("Initializing ODModel")

        if not MODEL_PATH.exists():
            raise FileNotFoundError("ONNX model not found (did you download your trained model?):", MODEL_PATH)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
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


    def _should_stop(self, detections: np.ndarray): 
        
        stop = False

        for x1, y1, x2, y2, score, _ in detections:

            #print(f"Detection: {x1}-{x2}, {y1}-{y2}, {score}")
            # TODO we don't want to consider detections with confidence (score) below CONF_THRESHOLD (a value you should set in config.py)


            # TODO we want to stop if there is a duckie closer than STOP_DISTANCE away
            # To calculate if the duckie is too close we need to convert the pixel coordinates to 
            # world coordinates. To do so you can use the `self.ground_projector` object which has
            # loaded the camera extrinsic calibration
            # Specifically, if you want to project an object of type `pix = Pixel(x=u, y=v)` to a ground plane
            # point, you can first convert it to a vector (`vec = self.ground_projector.camera.pixel2vector(pix)`) and
            # then you can intersect that vector with the ground plane (`self.ground_projector.vector2ground(vec)`). 
            # That will be the point on the ground plane corresponding to the input pixel. 
            if score < CONF_THRESHOLD:
                continue

            size = (x2-x1)*(y2-y1)
            #print(
                #f"Detection score={score:.2f}, "
                #f"size={size:.2f} m"
            #)

            if size >= STOP_SIZE:
                print(
                    f"STOP: object detected at "
                    f"{size:.2f} m"
                )
                return True

        return stop


    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]

        if h != self.net_h or w != self.net_w:
            raise ValueError(
                f"Image size {h}x{w} does not match ONNX! Expected {self.net_h}x{self.net_w}"
            )

        img = img_bgr[:, :, ::-1].astype(self.in_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img
        

    def stop_for_object(self, img: np.ndarray):
        try:
            detections = self._run_detector(img)
        except Exception as e:
            print(f"ONNX inference error {e}")
            return True
        if self._should_stop(detections):
            return True
        else:
            return False