#!/usr/bin/env python3
"""Verify pure SAR export complexity/equivalence and benchmark PyTorch inference latency."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.export_sar_student import build_pure_student, prediction_tensor  # noqa: E402
from ultralytics.nn.tasks import attempt_load_one_weight  # noqa: E402
from ultralytics.utils.torch_utils import de_parallel, get_flops, get_num_params  # noqa: E402


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model, image: torch.Tensor, warmup: int, repeats: int) -> list[float]:
    """Measure synchronized single-forward wall time."""
    with torch.no_grad():
        for _ in range(warmup):
            model(image)
        synchronize(image.device)
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(image)
            synchronize(image.device)
            timings.append(1000.0 * (time.perf_counter() - started))
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="PRCD training checkpoint.")
    parser.add_argument("--export", required=True, help="Pure SAR checkpoint from export_sar_student.py.")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    trained, _ = attempt_load_one_weight(args.checkpoint, device=device, fuse=False)
    exported, _ = attempt_load_one_weight(args.export, device=device, fuse=False)
    reference = build_pure_student(de_parallel(trained)).to(device).float().eval()
    exported = de_parallel(exported).to(device).float().eval()
    image = torch.rand(args.batch, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        expected = prediction_tensor(reference(image))
        actual = prediction_tensor(exported(image))
    max_abs_error = float((expected - actual).abs().max())
    if not torch.equal(expected, actual):
        raise RuntimeError(f"export is not bit-exact to detector-only checkpoint state: {max_abs_error=}")
    reference_params, exported_params = get_num_params(reference), get_num_params(exported)
    reference_flops, exported_flops = get_flops(reference, args.imgsz), get_flops(exported, args.imgsz)
    if reference_params != exported_params or reference_flops != exported_flops:
        raise RuntimeError("pure export changed detector parameter count or FLOPs")
    reference_times = benchmark(reference, image, args.warmup, args.repeats)
    exported_times = benchmark(exported, image, args.warmup, args.repeats)
    report = {
        "device": str(device),
        "batch": args.batch,
        "imgsz": args.imgsz,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "prediction_max_abs_error": max_abs_error,
        "reference_detector_parameters": reference_params,
        "exported_parameters": exported_params,
        "reference_detector_gflops": reference_flops,
        "exported_gflops": exported_flops,
        "reference_latency_ms": {
            "median": statistics.median(reference_times),
            "mean": statistics.mean(reference_times),
        },
        "exported_latency_ms": {
            "median": statistics.median(exported_times),
            "mean": statistics.mean(exported_times),
        },
        "distillation_in_export_state": any(
            token in key
            for key in exported.state_dict()
            for token in ("teacher", "distiller", "prototype")
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
