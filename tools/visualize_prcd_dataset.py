#!/usr/bin/env python3
"""Render synchronized paired augmentations and P3/P4/P5 support diagnostics from real OBB data."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.paired_obb_dataset import PairedOBBDataset  # noqa: E402
from ultralytics.nn.distillation.object_support import ObjectSupportSampler  # noqa: E402
from ultralytics.utils import DEFAULT_CFG  # noqa: E402
from ultralytics.utils.distill_visualization import save_paired_debug, save_support_masks  # noqa: E402


def resolve_entry(data: dict, key: str) -> str | None:
    """Resolve an Ultralytics data path while preserving absolute custom roots."""
    value = data.get(key)
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and data.get("path"):
        path = Path(str(data["path"])).expanduser() / path
    return str(path.resolve())


def tensor_to_bgr(image: torch.Tensor) -> np.ndarray:
    """Convert Ultralytics RGB CHW uint8 tensors to OpenCV BGR."""
    return np.ascontiguousarray(image.detach().cpu().permute(1, 2, 0).numpy()[..., ::-1])


def main() -> None:
    """Build the paired loader itself so the visualization audits the production transform path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--objects-per-sample", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--degrees", type=float, default=10.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=ROOT / "debug_visualizations")
    args = parser.parse_args()

    data = yaml.safe_load(args.data.read_text(encoding="utf-8")) or {}
    sar = resolve_entry(data, args.split)
    optical = resolve_entry(data, f"optical_{args.split}")
    labels = resolve_entry(data, f"labels_{args.split}")
    if not sar or not optical:
        raise KeyError(f"{args.data} must define {args.split} and optical_{args.split}")
    hyp = get_cfg(DEFAULT_CFG)
    hyp.mosaic = hyp.mixup = hyp.copy_paste = 0.0
    hyp.degrees, hyp.translate, hyp.scale = args.degrees, args.translate, args.scale
    hyp.fliplr, hyp.flipud = 0.5, 0.5
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = PairedOBBDataset(
        img_path=sar,
        optical_path=optical,
        label_path=labels,
        label_cache_path=Path(tempfile.gettempdir()) / f"prcd_visualize_{args.data.stem}_{args.split}.cache",
        pair_manifest=resolve_entry(data, "pair_manifest"),
        pair_manifest_split=args.split if data.get("pair_manifest") else None,
        filename_mapper=data.get("filename_mapper"),
        strict_pairs=True,
        imgsz=args.imgsz,
        batch_size=1,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=False,
        stride=32,
        pad=0,
        prefix="PRCD debug: ",
        task="obb",
        data={"names": data["names"], "nc": int(data["nc"])},
    )
    records = []
    for index in range(min(args.samples, len(dataset))):
        sample_seed = args.seed + index
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        sample = dataset[index]
        sample_id = sample["sample_id"]
        boxes = sample["bboxes"]
        classes = sample["cls"].view(-1)
        save_paired_debug(
            tensor_to_bgr(sample["img"]),
            tensor_to_bgr(sample["img_opt"]),
            boxes,
            classes,
            sample_id,
            args.output / f"{index:03d}_{sample_id}_synchronized_augmented.jpg",
        )
        object_records = []
        for object_index in range(min(args.objects_per_sample, len(boxes))):
            box = boxes[object_index : object_index + 1]
            class_id = int(classes[object_index])
            width_px = float(box[0, 2] * args.imgsz)
            height_px = float(box[0, 3] * args.imgsz)
            for stride in (8, 16, 32):
                level = f"P{stride.bit_length() - 1}"
                for mode in ("hard_gt_roi", "fixed_5", "adaptive"):
                    sampler = ObjectSupportSampler(mode=mode, output_grid=7, radius_min=1, radius_max=4)
                    geometry = sampler.build(box, torch.zeros(1, dtype=torch.long), (args.imgsz // stride,) * 2)
                    radius = float(geometry.radii[0])
                    title = (
                        f"id={sample_id} cls={class_id} gt={width_px:.1f}x{height_px:.1f}px "
                        f"{level} stride={stride} {mode} r={radius:.1f}"
                    )
                    save_support_masks(
                        geometry.core_mask[0],
                        geometry.context_mask[0],
                        args.output / f"{index:03d}_{sample_id}_obj{object_index}_{level}_{mode}.png",
                        title=title,
                    )
                    object_records.append(
                        {
                            "object_index": object_index,
                            "class_id": class_id,
                            "gt_width_px": width_px,
                            "gt_height_px": height_px,
                            "level": level,
                            "stride": stride,
                            "support_mode": mode,
                            "radius_cells": radius,
                            "valid": bool(geometry.valid[0]),
                        }
                    )
        records.append(
            {
                "sample_id": sample_id,
                "seed": sample_seed,
                "source_sample_ids": list(sample["source_sample_ids"]),
                "augmentation_policy": {
                    "mosaic": 0.0,
                    "mixup": 0.0,
                    "copy_paste": 0.0,
                    "degrees": args.degrees,
                    "translate": args.translate,
                    "scale": args.scale,
                    "fliplr": 0.5,
                    "flipud": 0.5,
                },
                "objects": object_records,
            }
        )
    (args.output / "visualization_manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(records)} paired diagnostics to {args.output}")


if __name__ == "__main__":
    main()
