#!/usr/bin/env python3
"""Run one real paired PRCD optimizer step and emit machine-readable invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.data import build_dataloader  # noqa: E402
from ultralytics.models.yolo.obb.distill_train import DistillOBBTrainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--distill", required=True)
    parser.add_argument("--distill-dataset")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=0.001)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "smoke_prcd_step.json")
    args = parser.parse_args()
    overrides = {
        "model": args.model,
        "teacher": args.teacher,
        "data": args.data,
        "distill": args.distill,
        "distill_dataset": args.distill_dataset,
        "batch": args.batch,
        "fraction": args.fraction,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": 0,
        "amp": False,
        "plots": False,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "project": "/tmp/prcd-smoke-current",
        "name": "one-step",
        "exist_ok": True,
    }
    trainer = DistillOBBTrainer(overrides=overrides)
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device).train()
    trainer.set_model_attributes()
    trainer.model.criterion = trainer.model.init_criterion()
    trainer.epoch = 0
    dataset = trainer.build_dataset(trainer.trainset, mode="train", batch=args.batch)
    loader = build_dataloader(dataset, args.batch, workers=0, shuffle=False, rank=-1)
    batch = trainer.preprocess_batch(next(iter(loader)))
    optimizer = trainer.build_optimizer(
        trainer.model, name="SGD", lr=1e-4, momentum=0.9, decay=float(trainer.args.weight_decay)
    )
    optimizer.zero_grad(set_to_none=True)
    loss, items = trainer.model(batch)
    if not torch.isfinite(loss) or not torch.isfinite(items).all():
        raise FloatingPointError("PRCD one-step loss is NaN/Inf")
    loss.backward()
    adapter_parameters = [
        parameter
        for name, parameter in trainer.model.named_parameters()
        if "distiller.representations" in name and parameter.requires_grad
    ]
    adapter_has_grad = any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in adapter_parameters)
    teacher_has_grad = any(parameter.grad is not None for parameter in trainer.teacher.parameters())
    if not adapter_has_grad or teacher_has_grad:
        raise RuntimeError(f"gradient invariant failed: {adapter_has_grad=}, {teacher_has_grad=}")
    optimizer.step()
    names = ("box", "cls", "dfl", *trainer.model.prcd_loss_names)
    report = {
        "status": "passed",
        "batch_size": int(batch["img"].shape[0]),
        "dataset_records_after_fraction": len(dataset),
        "full_pair_audit_records": dataset.pairing_report.paired_count,
        "loss_scalar": float(loss.detach()),
        "items": {name: float(value) for name, value in zip(names, items.detach().cpu())},
        "adapter_has_finite_grad": adapter_has_grad,
        "teacher_has_grad": teacher_has_grad,
        "teacher_registered_in_student": any("teacher" in name for name, _ in trainer.model.named_parameters()),
        "teacher_hash": getattr(trainer.args, "teacher_hash", None),
        "dataset_profile_hash": trainer.distill_config.get("_profile_hash"),
        "selected_dataset": trainer.distill_config.get("_selected_dataset"),
        "label_cache": str(dataset.explicit_label_cache_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
