#!/usr/bin/env python3
"""Explicitly merge a measured loss-scale recommendation into a profiler-generated PRCD config."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml


def apply_recommendation(config: dict, report: dict, dataset: str | None, candidate_index: int) -> dict:
    """Return a copy with measured candidates recorded and one explicitly selected."""
    output = deepcopy(config)
    targets = []
    if dataset:
        try:
            targets.append(output["_per_dataset"][dataset])
        except KeyError as error:
            raise KeyError(f"dataset {dataset!r} is absent from config._per_dataset") from error
    else:
        targets.append(output)

    recommendations = report.get("weight_recommendation", {})
    for target in targets:
        distill = target["distill"]
        for component in ("crc", "spd", "icd"):
            candidates = recommendations.get(component, {}).get("candidates", [])
            if not candidates:
                raise ValueError(f"loss report has no usable {component} candidates")
            if not -len(candidates) <= candidate_index < len(candidates):
                raise IndexError(
                    f"candidate index {candidate_index} is outside {component} candidates (n={len(candidates)})"
                )
            distill[component]["weight_candidates"] = [float(value) for value in candidates]
            distill[component]["weight"] = float(candidates[candidate_index])
        evidence = target.setdefault("_evidence", {})
        evidence["loss_scale_probe"] = {
            "mode": report.get("mode"),
            "batch_size": report.get("batch_size"),
            "batches": report.get("batches"),
            "imgsz": report.get("imgsz"),
            "selected_candidate_index": candidate_index,
            "auto_applied_to_training": False,
            "teacher_has_grad": report.get("teacher_has_grad"),
        }

    # Keep the convenient top-level default consistent only when it describes
    # the selected dataset. Other dataset recommendations remain untouched.
    if dataset and output.get("_evidence", {}).get("profile_dataset") == dataset:
        output["distill"] = deepcopy(output["_per_dataset"][dataset]["distill"])
        output["_evidence"] = deepcopy(output["_per_dataset"][dataset].get("_evidence", {}))
    return output


def main() -> None:
    """Parse files and require an explicit output path; never mutate the input config in place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--dataset", help="Key under config._per_dataset; omit for a single-dataset config.")
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=0,
        help="Measured candidate to select (default 0, the conservative lower-gradient candidate).",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.config.resolve():
        raise ValueError("--output must differ from --config so recommendations are never silently applied")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    report = json.loads(args.report.read_text(encoding="utf-8"))
    merged = apply_recommendation(config, report, args.dataset, args.candidate_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote explicit recommendation merge to {args.output}")


if __name__ == "__main__":
    main()
