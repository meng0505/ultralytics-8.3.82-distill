#!/usr/bin/env python3
"""Build or execute equal-budget PRCD ablation commands."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = ROOT / "configs/prcd/ablations"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils.distill_config import load_distill_config  # noqa: E402


def deep_merge(base: dict, override: dict) -> dict:
    """Merge an ablation override onto a profiled base without mutating either input."""
    output = dict(base)
    for key, value in override.items():
        output[key] = (
            deep_merge(output[key], value)
            if isinstance(value, dict) and isinstance(output.get(key), dict)
            else value
        )
    return output


def main() -> None:
    """Generate controlled commands and optionally execute them sequentially."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-command",
        default=f"{sys.executable} -m ultralytics.models.yolo.obb.distill_train",
        help="Launcher before key=value overrides.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--teacher", help="Optical teacher checkpoint; required by every enabled distillation config.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--distill-dataset", help="Optional _per_dataset selection passed to the trainer.")
    parser.add_argument("--configs-dir", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--base-distill",
        type=Path,
        help="Profiled/recommended config onto which each small ablation YAML is overlaid.",
    )
    parser.add_argument("--include", nargs="*", help="Config stems; defaults to every YAML in configs-dir.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(ROOT / "runs/prcd_ablation"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "ablation_plan.json")
    parser.add_argument("--registration-shifts", type=int, nargs="*", default=[])
    args = parser.parse_args()

    configs = sorted(args.configs_dir.glob("*.yaml"))
    if args.include:
        requested = set(args.include)
        configs = [path for path in configs if path.stem in requested]
        missing = requested - {path.stem for path in configs}
        if missing:
            raise FileNotFoundError(f"unknown ablation configs: {sorted(missing)}")
    runs = [(path, 0) for path in configs]
    if args.registration_shifts:
        robustness = {"kd_pixel_response", "kd_hard_gt_roi", "prcd_crc_adaptive"}
        runs.extend(
            (path, shift)
            for path in configs
            if path.stem in robustness
            for shift in args.registration_shifts
            if shift != 0
        )

    records = []
    resolved_dir = args.output.parent / f"{args.output.stem}_resolved_configs"
    base_config = (
        load_distill_config(args.base_distill, dataset_name=args.distill_dataset) if args.base_distill else None
    )
    for config, shift in runs:
        ablation = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        if base_config is not None:
            config_data = load_distill_config(deep_merge(base_config, ablation))
            resolved_dir.mkdir(parents=True, exist_ok=True)
            resolved_config = resolved_dir / config.name
            resolved_config.write_text(
                yaml.safe_dump(config_data, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
        else:
            config_data = load_distill_config(ablation)
            resolved_config = config
        enabled = bool(config_data.get("distill", {}).get("enabled", True))
        if enabled and not args.teacher:
            raise ValueError(f"--teacher is required for enabled config {config.name}")
        name = config.stem + (f"_shift{shift}" if shift else "")
        command = [
            *shlex.split(args.base_command),
            f"model={args.model}",
            f"data={args.data}",
            f"distill={resolved_config}",
            f"epochs={args.epochs}",
            f"fraction={args.fraction}",
            f"batch={args.batch}",
            f"imgsz={args.imgsz}",
            f"seed={args.seed}",
            f"device={args.device}",
            "mosaic=0",
            "mixup=0",
            "copy_paste=0",
            f"optical_shift={shift}",
            f"project={args.project}",
            f"name={name}",
        ]
        if args.teacher:
            command.insert(len(shlex.split(args.base_command)) + 1, f"teacher={args.teacher}")
        if args.distill_dataset:
            command.append(f"distill_dataset={args.distill_dataset}")
        record = {
            "name": name,
            "config": str(config),
            "resolved_config": str(resolved_config),
            "shift_px": shift,
            "command": command,
            "status": "planned",
        }
        print(shlex.join(command))
        if args.execute:
            started = time.time()
            result = subprocess.run(command, check=False)
            record.update(
                status="completed" if result.returncode == 0 else "failed",
                returncode=result.returncode,
                elapsed_seconds=time.time() - started,
            )
            results_file = Path(args.project) / name / "results.csv"
            if result.returncode == 0 and results_file.exists():
                with results_file.open(encoding="utf-8-sig", newline="") as file:
                    rows = list(csv.DictReader(file))
                if rows:
                    record["final_metrics"] = {
                        key.strip(): float(value)
                        for key, value in rows[-1].items()
                        if value not in (None, "") and key.strip() != "epoch"
                    }
                    record["results_csv"] = str(results_file)
            if result.returncode:
                records.append(record)
                break
        records.append(record)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} ablation records to {args.output}")


if __name__ == "__main__":
    main()
