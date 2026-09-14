#!/usr/bin/env python3
"""Render the loss-magnitude and gradient-norm comparison from a PRCD probe JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils.distill_visualization import save_loss_scale_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "loss_scale_report.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "debug_visualizations/loss_magnitude_gradient_norm.png"
    )
    args = parser.parse_args()
    save_loss_scale_report(json.loads(args.report.read_text(encoding="utf-8")), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
