from huggingface_hub import login
TOKEN = input('hugging facetoken:').strip()
login(token=TOKEN)

import shutil
from pathlib import Path

import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from contextlib import nullcontext

from ultralytics import YOLO


import os
import random
from pathlib import Path
import shutil
import torch

def create_train_val_split(data_collection_dir, dataset_dir, split_percentage):
    data_collection_dir = Path(data_collection_dir)
    dataset_dir = Path(dataset_dir)

    train_img_dir = dataset_dir / "train" / "images"
    train_lbl_dir = dataset_dir / "train" / "labels"
    val_img_dir   = dataset_dir / "val" / "images"
    val_lbl_dir   = dataset_dir / "val" / "labels"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in data_collection_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    random.shuffle(images)
    split_idx = int(len(images) * split_percentage)

    train_images = images[:split_idx]
    val_images   = images[split_idx:]

    def copy_and_init_label(img_path, dest_img_dir, dest_lbl_dir=None):
        shutil.copy(img_path, dest_img_dir / img_path.name)
        # (dest_lbl_dir / (img_path.stem + ".txt")).touch()  # we can either create empty label files or ignore it

    for img in train_images:
        copy_and_init_label(img, train_img_dir, train_lbl_dir)

    for img in val_images:
        copy_and_init_label(img, val_img_dir, val_lbl_dir)

    print(f"Created train/val split: {len(train_images)} train, {len(val_images)} val")


    # Unzip the dataset
DATASET_DIR_NAME = "duckietown_dataset"
DATASET_ZIP_NAME = f"{DATASET_DIR_NAME}.zip"
DATASET_DIR_PATH = os.path.join('./', DATASET_DIR_NAME)


DATASET_DIR = Path(DATASET_DIR_PATH)
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
CLASSES_YAML = DATASET_DIR / "classes.yaml"


def show_info(base_path: str):
  TRAIN_DIR = "train"
  VALIDATION_DIR = "val"
  IMAGES_DIR = "images"
  LABELS_DIR = "labels"
  for l1 in [TRAIN_DIR, VALIDATION_DIR]:
    for l2 in [IMAGES_DIR, LABELS_DIR]:
      p = os.path.join(base_path, l1, l2)
      print(f"#Files in {l1}/{l2}: {len(os.listdir(p))}")


def unzip_dataset():
  # check zipped file
  zip_path = DATASET_ZIP_NAME
  assert os.path.exists(zip_path), f"No zipped dataset found at {zip_path}! Abort!"

  # unzip the data
  print("Unpacking zipped data...")
  shutil.unpack_archive(zip_path, DATASET_DIR_PATH)
  print(f"Zipped dataset unpacked to {DATASET_DIR_PATH}")

  create_train_val_split(
      data_collection_dir=DATASET_DIR_NAME,
      dataset_dir=DATASET_DIR_NAME,
      split_percentage=0.8,
  )

  # show some info
  show_info(DATASET_DIR_PATH)

import cv2
import numpy as np

def mask_to_yolo_seg_line(mask: np.ndarray, w: int, h: int, class_id: int, epsilon_factor: float = 0.005) -> str | None:
    """
    Convert a binary HxW mask to a YOLO segmentation line.
    Returns None if no valid contour is found.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Keep only the largest contour (avoids noise fragments)
    contour = max(contours, key=cv2.contourArea)

    # Simplify polygon to reduce file size without losing shape
    eps = epsilon_factor * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, eps, True)

    if len(contour) < 3:
        return None

    points = contour.reshape(-1, 2)
    coords = " ".join(f"{x/w:.6f} {y/h:.6f}" for x, y in points)
    return f"{class_id} {coords}"

def load_classes(classes_yaml):
  with open(classes_yaml, "r") as f:
    cfg = yaml.safe_load(f)
  #items = sorted(cfg["names"].items(), key=lambda kv: int(kv[0]))
  #return [name for _, name in items]
  return cfg['names']


def xyxy_to_yolo_line(bbox, w, h, class_id):
  x1, y1, x2, y2 = bbox
  cx = (x1 + x2) / 2.0 / w
  cy = (y1 + y2) / 2.0 / h
  bw = (x2 - x1) / w
  bh = (y2 - y1) / h
  return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


class Sam3AutoLabel:
    # --- prompts tuned for Duckietown visuals ---
    #CLASS_PROMPTS = {
    #    0: "yellow dashed lane line",
    #    1: "white lane line",
    #    2: "red stop line tape",
    #    3: "yellow rubber duck toy",
    #}

    def __init__(
        self,
        train_dir,
        val_dir,
        dataset_dir,
        classes_yaml,
        bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        confidence_threshold=0.3,
        device=None,
    ):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.dataset_dir = dataset_dir
        self.classes_yaml = classes_yaml

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.classes = load_classes(self.classes_yaml)
        print(self.classes)

        self.model = build_sam3_image_model(bpe_path=str(bpe_path))
        self.model.to(device)
        self.model.eval()

        self.processor = Sam3Processor(
            self.model,
            confidence_threshold=confidence_threshold,
            device=device,
        )

    def _detect(self, img_path: Path):
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext()
        )

        dets = []
        with torch.no_grad(), ctx:
            st = self.processor.set_image(image)

            for class_id, prompt in self.classes.items():
                out = self.processor.set_text_prompt(state=st, prompt=prompt)

                masks  = out.get("masks")   # shape: (N, H, W) boolean/float
                scores = out.get("scores")

                if masks is None or scores is None:
                    continue

                for mask, score in zip(masks, scores):
                    score = float(score)
                    if score < self.processor.confidence_threshold:
                        continue

                    # Convert to numpy bool array (H, W)
                    mask_np = mask.cpu().numpy()
                    if mask_np.ndim == 3:
                        mask_np = mask_np[0]          # drop channel dim if present
                    mask_np = mask_np.astype(bool)

                    dets.append({
                        "mask":     mask_np,
                        "score":    score,
                        "class_id": class_id,
                    })

        return dets, (w, h)

    def _label_split(self, split: str):
        split_dir  = self.train_dir if split == "train" else self.val_dir
        img_dir    = split_dir / "images"
        lbl_dir    = split_dir / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)

        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            img_paths.extend(img_dir.rglob(ext))

        for img_path in tqdm(img_paths, desc=f"Labeling {split}"):
            dets, (w, h) = self._detect(img_path)

            yolo_lines = []
            for d in dets:
                line = mask_to_yolo_seg_line(d["mask"], w, h, d["class_id"])
                if line is not None:
                    yolo_lines.append(line)

            label_path = lbl_dir / f"{img_path.stem}.txt"
            if yolo_lines:
                label_path.write_text("\n".join(yolo_lines))
            else:
                if label_path.exists():
                    label_path.unlink()

    def run(self):
        self._label_split("train")
        self._label_split("val")




if __name__ == '__main__':
    print("unzipping dataset")
    unzip_dataset()
    print("unzipeed dataset")

    #copy classes to yaml
    shutil.copy('./classes.yaml', str(CLASSES_YAML))
    print("copied classes.yaml")

    autolabel = Sam3AutoLabel(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        dataset_dir=DATASET_DIR,
        classes_yaml=CLASSES_YAML,
        confidence_threshold=0.3,
    )

    autolabel.run()

    model = YOLO("yolo11n-seg.yaml")   # <-- seg variant

    runs_root = DATASET_DIR / "runs"
    run_name = "duckietown_detection"
    print(runs_root)

    results = model.train(
        data=str(CLASSES_YAML),
        epochs=10,
        imgsz=(480, 640),
        batch=16,
        workers=2,
        project=str(runs_root),
        name="duckietown_segmentation",
    )

    run_dir = Path(results.save_dir)

    best = run_dir / "weights" / "best.pt"

    if not best.exists():
        raise FileNotFoundError(best)


    #all_exps = os.listdir(f"{DATASET_DIR}/runs")
    #latest_exp_index = np.argmax(all_exps)
    #latest_exp = all_exps[latest_exp_index]
    #print(f"Latest exp is {latest_exp}")

    #run_dir = DATASET_DIR / "runs" / f"{latest_exp}" # you may need to update this based on you run_id
    #best = run_dir / "weights" / "best.pt"
    model_path = DATASET_DIR / "weights" / "best.onnx"

    #if not best.exists():
    #    raise FileNotFoundError(f"Could not find best.pt at: {best}")

    print(f"Using checkpoint: {best}")
    model = YOLO(str(best))

    onnx_tmp = run_dir / "weights" / "best.onnx"
    print(f"Exporting ONNX to: {onnx_tmp}")

    model.export(
        format="onnx",
        opset=18,
        imgsz=(480,640), # (height, width)
        simplify=True,
        dynamic=False,
        nms=True,
        half=True, # use FP16 - make sure device is set accordingly
        batch=1,
        optimize=False,
        device=0 # required for FP16 export - default to CPU if not specified
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_path.write_bytes(onnx_tmp.read_bytes())

    print(f"Copied ONNX model to: {model_path}")
    print("Export finished successfully.")

