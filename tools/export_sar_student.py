#!/usr/bin/env python3
"""Export a PRCD checkpoint as a pure SAR OBB detector and verify equivalence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import OBBModel, attempt_load_one_weight  # noqa: E402
from ultralytics.utils.torch_utils import de_parallel  # noqa: E402


def prediction_tensor(output):
    """Select decoded predictions from an OBB eval return."""
    return output[0] if isinstance(output, (tuple, list)) else output


def build_pure_student(model) -> OBBModel:
    """Copy only detector-compatible state into a plain OBBModel."""
    source = de_parallel(model).float()
    head = source.model[-1]
    student = OBBModel(source.yaml, ch=3, nc=head.nc, verbose=False)
    target_state = student.state_dict()
    source_state = source.state_dict()
    detector_state = {
        key: value.float()
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    missing, unexpected = student.load_state_dict(detector_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"pure detector state mismatch: missing={missing}, unexpected={unexpected}")
    for attribute in ("names", "stride", "args"):
        if hasattr(source, attribute):
            setattr(student, attribute, getattr(source, attribute))
    return student


def main() -> None:
    """Export and numerically compare a pure student."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    trained, checkpoint = attempt_load_one_weight(args.checkpoint, device="cpu", fuse=False)
    trained = de_parallel(trained).float().eval()
    student = build_pure_student(trained).float().eval()
    probe = torch.rand(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        expected = prediction_tensor(trained(probe))
        actual = prediction_tensor(student(probe))
    max_abs = float((expected - actual).abs().max())
    if not torch.allclose(expected, actual, atol=args.atol, rtol=1e-5):
        raise RuntimeError(f"student export changed inference: max_abs={max_abs}")
    forbidden = [key for key in student.state_dict() if any(token in key for token in ("teacher", "distiller", "prototype"))]
    if forbidden:
        raise RuntimeError(f"forbidden distillation state in export: {forbidden[:10]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    train_args = dict(checkpoint.get("train_args", {})) if isinstance(checkpoint, dict) else {}
    for key in (
        "teacher",
        "teacher_hash",
        "distill",
        "distill_dataset",
        "distill_profile_hash",
        "opt_root",
        "pair_manifest",
        "strict_pairs",
        "optical_shift",
        "optical_shift_direction",
    ):
        train_args.pop(key, None)
    torch.save(
        {
            "model": student.half(),
            "ema": None,
            "train_args": train_args,
            "prcd_export": {"source": str(args.checkpoint), "max_abs_error": max_abs},
        },
        args.output,
    )
    print(f"Exported pure SAR student to {args.output}; max_abs_error={max_abs:.3g}")


if __name__ == "__main__":
    main()
