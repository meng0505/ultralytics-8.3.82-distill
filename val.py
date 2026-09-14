"""Simple validation/test launcher using the official Ultralytics API."""

from __future__ import annotations

from ultralytics import YOLO


# Edit this block and run `python val.py`.
LOCAL_VAL_DEFAULTS = {
    "weights": "/home/mengfanlong/ultralytics-8.3.82-distill/runs/train/sar-baseline-raw-nomosaic-yolov11s-ogsod-2.0-256/weights/best.pt",
    "data": "/home/mengfanlong/ultralytics-8.3.82-distill/dataset/data.yaml",
    "split": "test",  # "test" or "val"
    "imgsz": 256,
    "batch": 64,
    "device": 0,
    "half": False,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": True,
    "augment": False,
    "plots": True,
    "project": "runs/val",
    "name": "baseline-best-test",
}


def main():
    cfg = dict(LOCAL_VAL_DEFAULTS)
    print("\n==================== VAL CONFIG ====================")
    for key in (
        "weights",
        "data",
        "split",
        "imgsz",
        "batch",
        "device",
        "half",
        "conf",
        "iou",
        "max_det",
        "rect",
        "augment",
        "project",
        "name",
    ):
        print(f"{key}: {cfg[key]}")
    print("====================================================\n")

    model = YOLO(cfg.pop("weights"))
    metrics = model.val(**cfg)

    print("\n==================== VAL METRICS ===================")
    for key, value in metrics.results_dict.items():
        try:
            print(f"{key}: {float(value):.6f}")
        except Exception:
            print(f"{key}: {value}")
    print("====================================================")


if __name__ == "__main__":
    main()
