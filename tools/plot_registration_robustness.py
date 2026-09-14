#!/usr/bin/env python3
"""Plot AP versus controlled optical shift from an executed PRCD ablation plan."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def find_metric(metrics: dict, suffix: str) -> float | None:
    """Find a whitespace-normalized Ultralytics metric key by suffix."""
    for key, value in metrics.items():
        if key.replace(" ", "").endswith(suffix):
            return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = json.loads(args.plan.read_text(encoding="utf-8"))
    curves = defaultdict(list)
    for record in records:
        if record.get("status") != "completed" or "final_metrics" not in record:
            continue
        ap = find_metric(record["final_metrics"], "metrics/mAP50-95(B)")
        if ap is None:
            continue
        base_name = Path(record["config"]).stem
        curves[base_name].append((int(record.get("shift_px", 0)), ap))
    if not curves:
        raise ValueError("plan contains no completed runs with metrics/mAP50-95(B)")

    import matplotlib.pyplot as plt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5))
    for name, values in sorted(curves.items()):
        values.sort()
        axis.plot([item[0] for item in values], [item[1] for item in values], marker="o", label=name)
    axis.set_xlabel("optical shift (input pixels)")
    axis.set_ylabel("OBB mAP50-95")
    axis.set_title("registration robustness (equal training budget)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
