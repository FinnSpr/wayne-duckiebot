# test.py
import ultralytics.utils.ops as _ops
_orig_process_mask = _ops.process_mask
def _process_mask_fp32(protos, masks_in, bboxes, shape, upsample=False):
    return _orig_process_mask(protos, masks_in.float(), bboxes, shape, upsample=upsample)
_ops.process_mask = _process_mask_fp32

import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

DATASET_DIR  = Path("./duckietown_dataset")
CLASSES_YAML = DATASET_DIR / "classes.yaml"
DEFAULT_MODEL = "best.onnx"
IMG_SIZE      = (480, 640)

# One colour per class + background last
COLORS = {
    -1: (50,  50,  50),    # background → dark grey
     0: (0,  200, 255),    # yellow lane marking → gold
     1: (200, 200, 200),   # white lane marking  → light grey
     2: (0,   0,  255),    # red stop line
     3: (0,  180, 255),    # yellow rubber duck  → amber
     4: (255, 0, 0),       # blue duckiebot
}


def load_classes(path: Path) -> list[str]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return [name for _, name in sorted(cfg["names"].items(), key=lambda kv: int(kv[0]))]


def build_segmap(frame: np.ndarray, result) -> np.ndarray:
    """Return an RGB image where every pixel is coloured by its class."""
    h, w   = frame.shape[:2]
    cls_map = np.full((h, w), fill_value=-1, dtype=np.int8)   # -1 = background
    conf_map = np.zeros((h, w), dtype=np.float32)              # track best conf per pixel

    if result.masks is not None and result.boxes is not None:
        masks  = result.masks.data.cpu().numpy()               # (N, H, W) uint8
        boxes  = result.boxes
        confs  = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for i, mask in enumerate(masks):
            bool_mask = mask > 0
            better    = bool_mask & (confs[i] > conf_map)     # only overwrite if higher conf
            cls_map[better]  = cls_ids[i]
            conf_map[better] = confs[i]

    # Build colour image
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in COLORS.items():
        out[cls_map == cls_id] = color

    # Blend with original so scene is still recognisable
    blended = cv2.addWeighted(frame, 0.35, out, 0.65, 0)
    return blended


def process_image(frame: np.ndarray, model, classes) -> np.ndarray:
    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=0.25,
        iou=0.3,
        verbose=False,
        half=True,
    )
    return build_segmap(frame, results[0])


def run(input_path: str, model_path: str, conf: float):
    import os
    classes = load_classes(CLASSES_YAML)
    model   = YOLO(model_path, task='segment')

    p = Path(input_path)

    # ── Single image ─────────────────────────────────────────────────────────
    if p.is_file():
        frame = cv2.imread(str(p))
        if frame is None:
            raise FileNotFoundError(f"Cannot read: {p}")
        annotated = process_image(frame, model, classes)
        out_path  = p.stem + "_result.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"Saved → {out_path}")
        return

    # ── Directory → video ────────────────────────────────────────────────────
    #k = lambda x : x.split('_')[1].split('.')[0]
    def k(x):
        try:
            x = os.path.basename(x)
            x = int(x.split('_')[1].split('.')[0])
            return x
        except: 
            return x
    if p.is_dir():
        images = sorted(p.glob("*.png"), key=k) + sorted(p.glob("*.jpg"), key=k) + sorted(p.glob("*.jpeg"), key=k)
        if not images:
            raise FileNotFoundError(f"No images found in {p}")

        print(f"Found {len(images)} images in {p}")

        # Read first frame to get dimensions
        first = cv2.imread(str(images[0]))
        h, w  = first.shape[:2]

        out_video = str(p.parent / (p.name + "_result.mp4"))
        writer    = cv2.VideoWriter(
            out_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,          # fps — adjust as needed
            (w, h),
        )

        for i, img_path in enumerate(images):
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  Skipping unreadable: {img_path.name}")
                continue

            annotated = process_image(frame, model, classes)
            writer.write(annotated)
            print(f"  [{i+1}/{len(images)}] {img_path.name}")

        writer.release()
        print(f"\nVideo saved → {out_video}")
        return

    raise ValueError(f"{input_path} is neither a file nor a directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input",  help="Image file or directory of images")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf",  default=0.25, type=float)
    args = parser.parse_args()
    run(args.input, args.model, args.conf)
