#!/usr/bin/env python3
"""Measure unweighted PRCD magnitudes and student-neck gradient norms without optimization."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.data import build_dataloader  # noqa: E402
from ultralytics.models.yolo.obb.distill_model import (  # noqa: E402
    class_logit_map,
    pop_captured_features,
    stride_tensor_map,
)
from ultralytics.models.yolo.obb.distill_train import DistillOBBTrainer  # noqa: E402


def gradient_norm(value: torch.Tensor, features: tuple[torch.Tensor, ...], scale: float = 1.0) -> float:
    """Measure L2 gradient norm at captured student-neck outputs."""
    gradients = torch.autograd.grad(value * scale, features, retain_graph=True, allow_unused=True)
    squared = sum(float(gradient.detach().float().square().sum()) for gradient in gradients if gradient is not None)
    return math.sqrt(squared)


def summarize(values: list[float]) -> dict:
    """Summarize measured values."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()) if len(array) else None,
        "std": float(array.std()) if len(array) else None,
        "min": float(array.min()) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def main() -> None:
    """Run a no-update loss-scale probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--distill", required=True)
    parser.add_argument("--distill-dataset")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=ROOT / "loss_scale_report.json")
    args = parser.parse_args()
    overrides = {
        "model": args.model,
        "teacher": args.teacher,
        "data": args.data,
        "distill": args.distill,
        "distill_dataset": args.distill_dataset,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": 0,
        "amp": args.device != "cpu",
        "plots": False,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "project": "/tmp/prcd-loss-scale",
        "name": "probe",
        "exist_ok": True,
    }
    trainer = DistillOBBTrainer(overrides=overrides)
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device).train()
    trainer.set_model_attributes()
    trainer.model.criterion = trainer.model.init_criterion()
    trainer.epoch = int(trainer.distill_config["distill"]["optimization"]["distill_warmup_epochs"])
    dataset = trainer.build_dataset(trainer.trainset, mode="train", batch=args.batch)
    loader = build_dataloader(dataset, args.batch, workers=0, shuffle=True, rank=-1)

    magnitudes, norms = defaultdict(list), defaultdict(list)
    validity = defaultdict(list)
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.batches:
            break
        batch = trainer.preprocess_batch(batch)
        predictions = trainer.model.forward(batch["img"])
        student_features_by_stride = stride_tensor_map(trainer.model, pop_captured_features(trainer.model))
        feature_tuple = tuple(student_features_by_stride[stride] for stride in trainer.model.distiller.strides)
        detection_loss, detection_items = trainer.model.criterion(predictions, batch)
        output = trainer.model.distiller(
            student_features_by_stride,
            batch["_teacher_payload"]["features"],
            class_logit_map(trainer.model, predictions),
            batch["_teacher_payload"]["class_logits"],
            batch["bboxes"],
            batch["cls"],
            batch["batch_idx"],
            warmup_scale=1.0,
            step=batch_index,
        )
        batch_size = batch["img"].shape[0]
        components = {
            "detection": detection_loss,
            "crc": output.crc,
            "spd": output.spd,
            "icd": output.icd,
        }
        magnitudes["detection"].append(float(detection_items.sum()))
        for name in ("crc", "spd", "icd"):
            magnitudes[name].append(float(components[name].detach()))
        norms["detection"].append(gradient_norm(detection_loss, feature_tuple))
        for name in ("crc", "spd", "icd"):
            norms[name].append(gradient_norm(components[name], feature_tuple, scale=batch_size))
        validity["crc_instances"].append(output.valid_crc_instances)
        validity["spd_groups"].append(output.valid_spd_groups)
        validity["icd_groups"].append(output.valid_icd_groups)
        validity["prototype_coverage"].append(float(output.prototype_coverage))
        validity["prototype_variance"].append(float(output.prototype_variance))
        validity["icd_matrix_correlation"].append(float(output.icd_matrix_correlation))

    report = {
        "mode": "recommend_only_no_optimizer_step",
        "batches": min(args.batches, len(loader)),
        "batch_size": args.batch,
        "imgsz": args.imgsz,
        "magnitude": {key: summarize(value) for key, value in magnitudes.items()},
        "student_neck_gradient_norm": {key: summarize(value) for key, value in norms.items()},
        "validity": {key: summarize(value) for key, value in validity.items()},
        "weight_recommendation": {},
        "teacher_has_grad": any(parameter.grad is not None for parameter in trainer.teacher.parameters()),
    }
    detection_norm = report["student_neck_gradient_norm"]["detection"]["mean"]
    for name in ("crc", "spd", "icd"):
        component_norm = report["student_neck_gradient_norm"][name]["mean"]
        if component_norm and component_norm > 0:
            center = float(np.clip(0.1 * detection_norm / component_norm, 1e-4, 10.0))
            report["weight_recommendation"][name] = {
                "target": "approximately 10% of measured detection neck-gradient norm",
                "candidates": [center / 2, center, center * 2],
                "auto_applied": False,
            }
        else:
            report["weight_recommendation"][name] = {
                "candidates": [],
                "reason": "no valid non-zero component in probe batches",
                "auto_applied": False,
            }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
