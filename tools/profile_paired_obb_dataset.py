#!/usr/bin/env python3
"""Profile paired optical/SAR OBB datasets and emit evidence-backed PRCD candidates.

The tool indexes every modality by an explicit sample ID. It never pairs files
by positional directory ordering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.data.paired_obb_dataset import build_pair_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML/JSON dataset-profile configuration.")
    parser.add_argument("--name", help="Single-dataset name when --config is not used.")
    parser.add_argument("--sar", type=Path, help="SAR root containing split subdirectories.")
    parser.add_argument("--optical", type=Path, help="Optical root containing split subdirectories.")
    parser.add_argument("--labels", type=Path, help="OBB label root containing split subdirectories.")
    parser.add_argument("--manifest", type=Path, help="Optional authoritative CSV/JSON pair manifest.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits for single-dataset mode.")
    parser.add_argument("--class-names", nargs="*", default=None)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--strides", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--batch-candidates", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-simulations", type=int, default=20)
    parser.add_argument("--skip-image-decode", action="store_true", help="Trust image headers/labels; skips corruption checks.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """Load multi-dataset configuration or construct a single-dataset equivalent."""
    if args.config:
        text = args.config.read_text(encoding="utf-8")
        config = json.loads(text) if args.config.suffix.lower() == ".json" else yaml.safe_load(text)
    else:
        missing = [key for key in ("name", "sar", "optical", "labels") if getattr(args, key) is None]
        if missing:
            raise ValueError(f"single-dataset mode requires: {', '.join(missing)}")
        config = {
            "datasets": [
                {
                    "name": args.name,
                    "sar": str(args.sar),
                    "optical": str(args.optical),
                    "labels": str(args.labels),
                    "manifest": str(args.manifest) if args.manifest else None,
                    "splits": args.splits,
                    "class_names": args.class_names,
                }
            ]
        }
    config.setdefault("imgsz", args.imgsz)
    config.setdefault("strides", args.strides)
    config.setdefault("batch_candidates", args.batch_candidates)
    return config


def split_path(source: str | Path, split: str) -> Path:
    """Resolve a split path from a template or root."""
    value = str(source)
    if "{split}" in value:
        return Path(value.format(split=split)).expanduser()
    root = Path(value).expanduser()
    child = root / split
    return child if child.is_dir() else root


def quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    """Return stable summary quantiles."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("min", "p10", "p25", "median", "p75", "p90", "p95", "max", "mean")}
    q = np.quantile(array, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1])
    return {
        "min": float(q[0]),
        "p10": float(q[1]),
        "p25": float(q[2]),
        "median": float(q[3]),
        "p75": float(q[4]),
        "p90": float(q[5]),
        "p95": float(q[6]),
        "max": float(q[7]),
        "mean": float(array.mean()),
    }


def finite_ratio(values: list[float], predicate) -> float | None:
    """Return the fraction satisfying a predicate."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(predicate(array))) if len(array) else None


def parse_obb_labels(path: str | Path, width: int, height: int) -> tuple[list[dict], list[str]]:
    """Parse normalized polygon or xywhr OBB labels into pixel-space geometry."""
    instances, errors = [], []
    label_path = Path(path)
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except Exception as error:
        return [], [f"read:{error}"]
    seen_annotations = set()
    for line_number, line in enumerate(lines, 1):
        fields = line.strip().split()
        if not fields:
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            errors.append(f"line {line_number}: non-numeric")
            continue
        annotation_key = tuple(values)
        if annotation_key in seen_annotations:
            errors.append(f"line {line_number}: duplicate annotation removed")
            continue
        seen_annotations.add(annotation_key)
        if len(values) == 9:
            class_id = int(values[0])
            points = np.asarray(values[1:], dtype=np.float32).reshape(4, 2)
            if np.nanmax(np.abs(points)) <= 2.0:
                points[:, 0] *= width
                points[:, 1] *= height
            (cx, cy), (box_width, box_height), angle_deg = cv2.minAreaRect(points)
            angle = math.radians(angle_deg)
        elif len(values) == 6:
            class_id = int(values[0])
            cx, cy, box_width, box_height, angle = values[1:]
            if max(abs(cx), abs(cy), abs(box_width), abs(box_height)) <= 2.0:
                cx, box_width = cx * width, box_width * width
                cy, box_height = cy * height, box_height * height
            if abs(angle) > 2 * math.pi:
                angle = math.radians(angle)
        else:
            errors.append(f"line {line_number}: expected 9 polygon or 6 xywhr fields, got {len(values)}")
            continue
        if not np.isfinite([cx, cy, box_width, box_height, angle]).all() or box_width <= 0 or box_height <= 0:
            errors.append(f"line {line_number}: non-finite or non-positive OBB")
            continue
        instances.append(
            {
                "class_id": class_id,
                "cx": float(cx),
                "cy": float(cy),
                "width": float(box_width),
                "height": float(box_height),
                "area": float(box_width * box_height),
                "equivalent_side": float(math.sqrt(box_width * box_height)),
                "aspect_ratio": float(max(box_width, box_height) / max(min(box_width, box_height), 1e-12)),
                "angle_rad": float(angle),
            }
        )
    return instances, errors


def add_density_statistics(instances: list[dict]) -> None:
    """Annotate nearest-neighbor distance and local-density indicators in place."""
    if not instances:
        return
    if len(instances) == 1:
        instances[0]["nearest_center_distance"] = None
        instances[0]["neighbor_within_2eq"] = False
        instances[0]["neighbors_within_64px"] = 0
        return
    centers = np.asarray([[item["cx"], item["cy"]] for item in instances], dtype=np.float64)
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    for index, item in enumerate(instances):
        item["nearest_center_distance"] = float(nearest[index])
        item["neighbor_within_2eq"] = bool(nearest[index] <= 2.0 * item["equivalent_side"])
        item["neighbors_within_64px"] = int(np.sum(distances[index] <= 64.0))


def support_statistics(instances: list[dict], image_size: tuple[int, int], imgsz: int, strides: list[int]) -> None:
    """Annotate actual letterboxed-training feature support for each stride."""
    height, width = image_size
    scale = imgsz / max(height, width)
    for item in instances:
        item["train_width"] = item["width"] * scale
        item["train_height"] = item["height"] * scale
        for stride in strides:
            mapped_width = item["train_width"] / stride
            mapped_height = item["train_height"] / stride
            item[f"s{stride}_width"] = mapped_width
            item[f"s{stride}_height"] = mapped_height
            item[f"s{stride}_area"] = mapped_width * mapped_height


def summarize_instances(instances: list[dict], strides: list[int]) -> dict:
    """Aggregate OBB and feature-support statistics."""
    result = {
        "instances": len(instances),
        "width_px": quantiles([item["width"] for item in instances]),
        "height_px": quantiles([item["height"] for item in instances]),
        "area_px2": quantiles([item["area"] for item in instances]),
        "equivalent_side_px": quantiles([item["equivalent_side"] for item in instances]),
        "aspect_ratio": quantiles([item["aspect_ratio"] for item in instances]),
        "angle_rad": quantiles([item["angle_rad"] for item in instances]),
        "nearest_center_distance_px": quantiles(
            [item["nearest_center_distance"] for item in instances if item["nearest_center_distance"] is not None]
        ),
        "local_density_neighbors_64px": quantiles([item["neighbors_within_64px"] for item in instances]),
        "probability_neighbor_within_2_equivalent_sides": (
            float(np.mean([item["neighbor_within_2eq"] for item in instances])) if instances else None
        ),
        "feature_support": {},
    }
    for stride in strides:
        widths = [item[f"s{stride}_width"] for item in instances]
        heights = [item[f"s{stride}_height"] for item in instances]
        areas = [item[f"s{stride}_area"] for item in instances]
        minimum_dimensions = [min(a, b) for a, b in zip(widths, heights)]
        result["feature_support"][f"P{int(math.log2(stride))}"] = {
            "stride": stride,
            "mapped_width_cells": quantiles(widths),
            "mapped_height_cells": quantiles(heights),
            "mapped_area_cells2": quantiles(areas),
            "area_lt_1_cell": finite_ratio(areas, lambda value: value < 1),
            "area_lt_2_cells": finite_ratio(areas, lambda value: value < 2),
            "area_lt_3_cells": finite_ratio(areas, lambda value: value < 3),
            "min_dimension_lt_1_cell": finite_ratio(minimum_dimensions, lambda value: value < 1),
            "min_dimension_lt_2_cells": finite_ratio(minimum_dimensions, lambda value: value < 2),
            "min_dimension_lt_3_cells": finite_ratio(minimum_dimensions, lambda value: value < 3),
        }
    return result


def batch_feasibility(
    per_image_counts: list[Counter], class_ids: list[int], candidates: list[int], seed: int, simulations: int
) -> dict:
    """Estimate class support by deterministic repeated random batch partitions."""
    if not per_image_counts:
        return {}
    count_matrix = np.asarray([[counts.get(class_id, 0) for class_id in class_ids] for counts in per_image_counts])
    output = {}
    for batch_size in candidates:
        aggregate = {class_id: {threshold: [] for threshold in (2, 3, 4, 8)} for class_id in class_ids}
        averages = {class_id: [] for class_id in class_ids}
        effective_groups = []
        for simulation in range(simulations):
            rng = np.random.default_rng(seed + simulation)
            order = rng.permutation(len(count_matrix))
            batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
            supported_group_counts = []
            for batch_indices in batches:
                counts = count_matrix[batch_indices].sum(axis=0)
                supported_group_counts.append(int(np.sum(counts >= 2)))
                for class_offset, class_id in enumerate(class_ids):
                    averages[class_id].append(float(counts[class_offset]))
                    for threshold in (2, 3, 4, 8):
                        aggregate[class_id][threshold].append(bool(counts[class_offset] >= threshold))
            effective_groups.extend(supported_group_counts)
        output[str(batch_size)] = {
            "per_class": {
                str(class_id): {
                    "mean_instances": float(np.mean(averages[class_id])),
                    **{
                        f"batch_fraction_ge_{threshold}": float(np.mean(aggregate[class_id][threshold]))
                        for threshold in (2, 3, 4, 8)
                    },
                }
                for class_id in class_ids
            },
            "mean_valid_icd_class_groups": float(np.mean(effective_groups)),
            "fraction_with_any_valid_icd_group": float(np.mean(np.asarray(effective_groups) > 0)),
        }
    return output


def profile_split(
    dataset: dict,
    split: str,
    imgsz: int,
    strides: list[int],
    batch_candidates: list[int],
    seed: int,
    simulations: int,
    decode_images: bool,
) -> tuple[dict, list[dict], list[Counter]]:
    """Profile one split and return aggregate report, instances and per-image class counts."""
    sar = split_path(dataset["sar"], split)
    optical = split_path(dataset["optical"], split)
    labels = split_path(dataset["labels"], split)
    manifest = dataset.get("manifest")
    records, pairing = build_pair_records(
        sar,
        optical,
        labels,
        manifest=manifest,
        split=split if manifest else None,
        sample_id_fn=dataset.get("filename_mapper"),
    )

    corrupt_sar, corrupt_optical, shape_mismatches = [], [], []
    empty_labels, label_errors = [], {}
    all_instances, per_image_counts = [], []
    dimensions = []
    for record in records:
        sar_image = cv2.imread(record.sar_path)
        optical_image = cv2.imread(record.optical_path)
        if sar_image is None:
            corrupt_sar.append(record.sar_path)
            continue
        if optical_image is None:
            corrupt_optical.append(record.optical_path)
            continue
        sar_shape, optical_shape = sar_image.shape[:2], optical_image.shape[:2]
        if sar_shape != optical_shape:
            shape_mismatches.append(
                {"sample_id": record.sample_id, "sar_shape": list(sar_shape), "optical_shape": list(optical_shape)}
            )
            continue
        dimensions.append(sar_shape)
        instances, errors = parse_obb_labels(record.label_path, sar_shape[1], sar_shape[0])
        if errors:
            label_errors[record.sample_id] = errors
        if not instances:
            empty_labels.append(record.sample_id)
        add_density_statistics(instances)
        support_statistics(instances, sar_shape, imgsz, strides)
        for item in instances:
            item.update(sample_id=record.sample_id, split=split, dataset=dataset["name"])
        all_instances.extend(instances)
        per_image_counts.append(Counter(item["class_id"] for item in instances))
        if not decode_images:
            # The flag is retained for CLI compatibility; dimensions are needed for correct
            # letterbox support and cv2 header-only decoding is not portable across formats.
            pass

    class_ids = sorted({item["class_id"] for item in all_instances})
    by_class = {
        str(class_id): summarize_instances([item for item in all_instances if item["class_id"] == class_id], strides)
        for class_id in class_ids
    }
    class_frequency = Counter(item["class_id"] for item in all_instances)
    declared_class_count = len(dataset.get("class_names", []))
    out_of_range_class_ids = [
        class_id for class_id in class_ids if class_id < 0 or class_id >= declared_class_count
    ]
    nonzero_frequencies = [class_frequency[class_id] for class_id in class_ids]
    imbalance = max(nonzero_frequencies) / min(nonzero_frequencies) if nonzero_frequencies else None
    result = {
        "paths": {"sar": str(sar), "optical": str(optical), "labels": str(labels), "manifest": manifest},
        "pairing": pairing.to_dict(),
        "integrity": {
            "corrupt_sar": corrupt_sar,
            "corrupt_optical": corrupt_optical,
            "shape_mismatch_count": len(shape_mismatches),
            "shape_mismatches": shape_mismatches,
            "empty_label_count": len(empty_labels),
            "empty_label_ids": empty_labels,
            "label_error_count": len(label_errors),
            "label_errors": label_errors,
            "image_dimensions": {
                f"{width}x{height}": count for (height, width), count in Counter(dimensions).most_common()
            },
        },
        "images": len(records),
        "instances_per_image": quantiles([sum(counter.values()) for counter in per_image_counts]),
        "class_frequency": {str(key): value for key, value in sorted(class_frequency.items())},
        "class_id_validation": {
            "declared_class_count": declared_class_count,
            "observed_class_ids": class_ids,
            "out_of_range_class_ids": out_of_range_class_ids,
            "valid": not out_of_range_class_ids,
        },
        "class_imbalance_max_over_min": imbalance,
        "overall": summarize_instances(all_instances, strides),
        "per_class": by_class,
    }
    if split == "train":
        result["batch_feasibility"] = batch_feasibility(
            per_image_counts, class_ids, batch_candidates, seed, simulations
        )
    return result, all_instances, per_image_counts


def recommend(profile: dict, imgsz: int, strides: list[int], batch_candidates: list[int]) -> dict:
    """Produce bounded initial candidates from measured support and batch co-occurrence."""
    train = profile["splits"].get("train") or next(iter(profile["splits"].values()))
    support = train["overall"]["feature_support"]
    p3_key = next((key for key, value in support.items() if value["stride"] == min(strides)), next(iter(support)))
    p3 = support[p3_key]
    next_level = next((value for value in support.values() if value["stride"] == min(strides) * 2), None)
    next_lt1 = next_level["area_lt_1_cell"] if next_level else 1.0
    levels = [p3_key]
    level_assignment = "single"
    if next_level and next_lt1 is not None and next_lt1 < 0.65:
        levels.append(next(key for key, value in support.items() if value is next_level))
        level_assignment = "scale_assigned"

    p90_width = p3["mapped_width_cells"]["p90"] or 1.0
    p90_height = p3["mapped_height_cells"]["p90"] or 1.0
    natural_half_extent = max(p90_width, p90_height) / 2
    radius_min = 1
    radius_max = int(np.clip(math.ceil(natural_half_extent) + 1, 2, 4))
    output_grid = 5 if radius_max <= 2 else 7

    feasibility = train.get("batch_feasibility", {})
    preferred_batch = str(max(batch_candidates))
    preferred = feasibility.get(preferred_batch, {})
    per_class = preferred.get("per_class", {})
    ge2 = [value["batch_fraction_ge_2"] for value in per_class.values()]
    icd_enabled = bool(ge2 and min(ge2) >= 0.5)
    max_instances = 32
    if per_class and max(value["mean_instances"] for value in per_class.values()) > 32:
        max_instances = 64
    elif per_class and max(value["mean_instances"] for value in per_class.values()) < 8:
        max_instances = 16

    batches_per_epoch = math.ceil(train["images"] / max(batch_candidates)) if train["images"] else 0
    prototype_momentum = 0.99 if batches_per_epoch >= 50 and (not ge2 or mean(ge2) >= 0.5) else 0.9
    imbalance = train["class_imbalance_max_over_min"] or 1.0
    class_balanced = imbalance >= 5.0

    evidence = {
        "profile_dataset": profile["name"],
        "input_size": imgsz,
        "p3_area_lt_1": p3["area_lt_1_cell"],
        "p3_min_dimension_lt_2": p3["min_dimension_lt_2_cells"],
        "next_level_area_lt_1": next_lt1,
        "p3_mapped_max_dimension_p90": max(p90_width, p90_height),
        "class_imbalance_max_over_min": imbalance,
        f"batch_{preferred_batch}_class_fraction_ge_2": ge2,
        "note": "Loss weights remain candidates until loss_scale_report.json measures magnitude and gradient norms.",
    }
    return {
        "distill": {
            "enabled": True,
            "levels": levels,
            "level_assignment": level_assignment,
            "support": {
                "mode": "adaptive",
                "output_grid": output_grid,
                "fixed_window": 5,
                "context_margin": 1,
                "radius_min": radius_min,
                "radius_max": radius_max,
                "exclude_neighbor_objects": True,
                "core_supersample": 4,
                "min_core_mass": 0.001,
            },
            "representation": {
                "adapter": "conv1x1_groupnorm",
                "embedding_dim": 128,
                "aggregation": "concat",
            },
            "crc": {
                "enabled": True,
                "regions": 2,
                "temperature": 1.0,
                "temperature_candidates": [0.5, 1.0, 2.0],
                "response_source": "class_logit",
                "weight": 0.1,
                "weight_candidates": [0.05, 0.1, 0.2],
            },
            "spd": {
                "enabled": True,
                "prototype_momentum": prototype_momentum,
                "momentum_candidates": [0.9, 0.99, 0.999],
                "prototype_relation_enabled": False,
                "weight": 0.1,
                "weight_candidates": [0.05, 0.1, 0.2],
            },
            "icd": {
                "enabled": icd_enabled,
                "min_instances": 2,
                "max_instances_per_class": max_instances,
                "max_instances_candidates": [16, 32, 64],
                "exclude_diagonal": True,
                "pair_direction": False,
                "memory_assisted": False,
                "cross_rank_gather": False,
                "weight": 0.05,
                "weight_candidates": [0.02, 0.05, 0.1],
            },
            "optimization": {
                "distill_warmup_epochs": 10,
                "loss_weight_mode": "magnitude_recommendation",
                "auto_apply_recommendation": False,
                "class_balanced_sampling_recommended": class_balanced,
            },
        },
        "_evidence": evidence,
    }


def profile_dataset(dataset: dict, config: dict, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """Profile all configured splits for one dataset."""
    result = {
        "name": dataset["name"],
        "class_names": dataset.get("class_names"),
        "imgsz": int(config["imgsz"]),
        "strides": [int(value) for value in config["strides"]],
        "splits": {},
    }
    all_instances = []
    ids_by_split = {}
    for split in dataset.get("splits", ["train", "val"]):
        split_result, instances, _ = profile_split(
            dataset,
            split,
            int(config["imgsz"]),
            [int(value) for value in config["strides"]],
            [int(value) for value in config["batch_candidates"]],
            args.seed,
            args.batch_simulations,
            not args.skip_image_decode,
        )
        result["splits"][split] = split_result
        all_instances.extend(instances)
        pairing = split_result["pairing"]
        ids_by_split[split] = set(
            item["sample_id"] for item in instances
        )  # Includes all non-corrupt labeled records with >=1 instance.
        # Empty images are valid and separately reported; pair identity leakage is recomputed from file indices below.
        ids_by_split[split].update(split_result["integrity"]["empty_label_ids"])
        if pairing["paired_count"] == 0:
            ids_by_split[split] = set()
    leakage = {}
    split_names = list(ids_by_split)
    for i, first in enumerate(split_names):
        for second in split_names[i + 1 :]:
            overlap = sorted(ids_by_split[first] & ids_by_split[second])
            leakage[f"{first}<->{second}"] = {"count": len(overlap), "examples": overlap[:100]}
    result["split_id_leakage"] = leakage
    result["recommendation"] = recommend(
        result, int(config["imgsz"]), [int(value) for value in config["strides"]], config["batch_candidates"]
    )
    return result, all_instances


def write_csv(profiles: list[dict], destination: Path) -> None:
    """Write one compact aggregate row per dataset/split/class."""
    fieldnames = [
        "dataset",
        "split",
        "class_id",
        "images",
        "instances",
        "instances_per_image_mean",
        "equivalent_side_median_px",
        "equivalent_side_p90_px",
        "aspect_ratio_median",
        "nearest_center_median_px",
        "neighbor_probability",
        "p3_area_lt_1",
        "p3_min_dim_lt_1",
        "p4_area_lt_1",
    ]
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for profile in profiles:
            for split, split_data in profile["splits"].items():
                groups = {"all": split_data["overall"], **split_data["per_class"]}
                for class_id, group in groups.items():
                    supports = list(group["feature_support"].values())
                    writer.writerow(
                        {
                            "dataset": profile["name"],
                            "split": split,
                            "class_id": class_id,
                            "images": split_data["images"],
                            "instances": group["instances"],
                            "instances_per_image_mean": split_data["instances_per_image"]["mean"],
                            "equivalent_side_median_px": group["equivalent_side_px"]["median"],
                            "equivalent_side_p90_px": group["equivalent_side_px"]["p90"],
                            "aspect_ratio_median": group["aspect_ratio"]["median"],
                            "nearest_center_median_px": group["nearest_center_distance_px"]["median"],
                            "neighbor_probability": group["probability_neighbor_within_2_equivalent_sides"],
                            "p3_area_lt_1": supports[0]["area_lt_1_cell"] if supports else None,
                            "p3_min_dim_lt_1": supports[0]["min_dimension_lt_1_cell"] if supports else None,
                            "p4_area_lt_1": supports[1]["area_lt_1_cell"] if len(supports) > 1 else None,
                        }
                    )


def percent(value: float | None) -> str:
    """Format an optional fraction."""
    return "n/a" if value is None else f"{100 * value:.1f}%"


def write_markdown(profiles: list[dict], destination: Path, profile_hash: str) -> None:
    """Write an evidence-oriented human report."""
    lines = [
        "# Paired OBB dataset profile",
        "",
        f"Profile hash: `{profile_hash}`.",
        "",
        "Pairing is based on explicit filename stems or the configured manifest; directory order is never used.",
        "",
    ]
    for profile in profiles:
        lines.extend([f"## {profile['name']}", ""])
        for split, data in profile["splits"].items():
            pairing = data["pairing"]
            overall = data["overall"]
            supports = list(overall["feature_support"].items())
            lines.extend(
                [
                    f"### {split}",
                    "",
                    f"- Complete pairs: {pairing['paired_count']} "
                    f"(SAR {pairing['sar_count']}, optical {pairing['optical_count']}, labels {pairing['label_count']}).",
                    f"- Missing IDs: SAR {len(pairing['missing_sar_ids'])}, optical "
                    f"{len(pairing['missing_optical_ids'])}, labels {len(pairing['missing_label_ids'])}.",
                    f"- Duplicate IDs: SAR {len(pairing['duplicate_sar_ids'])}, optical "
                    f"{len(pairing['duplicate_optical_ids'])}, labels {len(pairing['duplicate_label_ids'])}.",
                    f"- Corrupt images: SAR {len(data['integrity']['corrupt_sar'])}, optical "
                    f"{len(data['integrity']['corrupt_optical'])}; unequal pair dimensions "
                    f"{data['integrity']['shape_mismatch_count']}.",
                    f"- Label warning records: {data['integrity']['label_error_count']} "
                    f"(degenerate OBBs are skipped; exact duplicate annotations are counted once).",
                    f"- Class IDs: observed `{data['class_id_validation']['observed_class_ids']}` against "
                    f"{data['class_id_validation']['declared_class_count']} declared slots; out-of-range "
                    f"`{data['class_id_validation']['out_of_range_class_ids']}`.",
                    f"- Instances: {overall['instances']}; per image mean "
                    f"{data['instances_per_image']['mean']:.3f}; class frequencies `{data['class_frequency']}`.",
                    f"- Equivalent side: median {overall['equivalent_side_px']['median']:.2f}px, "
                    f"P90 {overall['equivalent_side_px']['p90']:.2f}px; aspect-ratio median "
                    f"{overall['aspect_ratio']['median']:.2f}.",
                    f"- Neighbor within two equivalent sides: "
                    f"{percent(overall['probability_neighbor_within_2_equivalent_sides'])}.",
                    "",
                ]
            )
            lines.append("| Level | Area <1 cell | Area <2 cells | min dimension <1 | min dimension <2 |")
            lines.append("|---|---:|---:|---:|---:|")
            for level, support in supports:
                lines.append(
                    f"| {level}/{support['stride']} | {percent(support['area_lt_1_cell'])} | "
                    f"{percent(support['area_lt_2_cells'])} | {percent(support['min_dimension_lt_1_cell'])} | "
                    f"{percent(support['min_dimension_lt_2_cells'])} |"
                )
            lines.append("")
        leakage = profile["split_id_leakage"]
        lines.append(f"Split identity leakage: `{leakage}`.")
        lines.extend(["", "### Recommended initial candidate", ""])
        recommendation = profile["recommendation"]
        distill = recommendation["distill"]
        evidence = recommendation["_evidence"]
        lines.extend(
            [
                f"- Levels `{distill['levels']}` with `{distill['level_assignment']}` assignment: "
                f"next-level area-under-one-cell ratio is {percent(evidence['next_level_area_lt_1'])}.",
                f"- Adaptive support radius {distill['support']['radius_min']}–"
                f"{distill['support']['radius_max']} and grid {distill['support']['output_grid']}: "
                f"P3 P90 max dimension is {evidence['p3_mapped_max_dimension_p90']:.2f} cells.",
                f"- ICD enabled `{distill['icd']['enabled']}`: largest candidate-batch per-class "
                f"fractions with at least two instances are "
                f"`{evidence[next(key for key in evidence if key.startswith('batch_'))]}`.",
                f"- Prototype momentum `{distill['spd']['prototype_momentum']}` and class-balanced sampling "
                f"recommendation `{distill['optimization']['class_balanced_sampling_recommended']}`.",
                "- CRC/SPD/ICD loss weights are explicitly provisional candidates. Run the loss-scale probe before "
                "choosing them; dataset geometry alone cannot determine gradient balance.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation constraints",
            "",
            "Feature-cell statistics describe spatial support after the configured letterbox scale. They do not imply "
            "that high-frequency SAR response or local context is noise. The recommendation favors region-level "
            "support because cross-modal pixel responses are not assumed to match.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def write_figures(profiles: list[dict], instances_by_dataset: dict[str, list[dict]], output: Path) -> None:
    """Write compact diagnostic figures; continue gracefully if matplotlib is unavailable."""
    try:
        import matplotlib.pyplot as plt
    except Exception as error:
        (output / "README.txt").write_text(f"Figures unavailable: {error}\n", encoding="utf-8")
        return
    for profile in profiles:
        name = profile["name"]
        instances = instances_by_dataset[name]
        if not instances:
            continue
        equivalent = [item["equivalent_side"] for item in instances]
        classes = Counter(item["class_id"] for item in instances)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(equivalent, bins=80)
        axes[0].set(title=f"{name}: equivalent OBB side", xlabel="pixels", ylabel="instances")
        axes[1].bar([str(key) for key in classes], [classes[key] for key in classes])
        axes[1].set(title=f"{name}: class frequency", xlabel="class", ylabel="instances")
        fig.tight_layout()
        fig.savefig(output / f"{name}_geometry.png", dpi=160)
        plt.close(fig)

        strides = profile["strides"]
        fig, axis = plt.subplots(figsize=(6, 4))
        for stride in strides:
            areas = np.sort([item[f"s{stride}_area"] for item in instances])
            cdf = np.linspace(0, 1, len(areas), endpoint=True)
            axis.plot(areas, cdf, label=f"stride {stride}")
        axis.axvline(1, color="black", linewidth=1, linestyle="--")
        axis.set(xscale="log", xlabel="mapped area (feature cells²)", ylabel="CDF", title=f"{name}: support CDF")
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / f"{name}_feature_support.png", dpi=160)
        plt.close(fig)


def main() -> None:
    """Run profiling and emit all requested artifacts."""
    args = parse_args()
    config = load_config(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "dataset_figures"
    figures.mkdir(parents=True, exist_ok=True)

    profiles, instances_by_dataset = [], {}
    for dataset in config["datasets"]:
        print(f"Profiling {dataset['name']}...", flush=True)
        profile, instances = profile_dataset(dataset, config, args)
        profiles.append(profile)
        instances_by_dataset[dataset["name"]] = instances

    payload = {
        "schema_version": 1,
        "imgsz": int(config["imgsz"]),
        "strides": [int(value) for value in config["strides"]],
        "batch_candidates": [int(value) for value in config["batch_candidates"]],
        "datasets": profiles,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    profile_hash = hashlib.sha256(canonical).hexdigest()
    payload["profile_hash"] = profile_hash
    (output / "dataset_profile.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(profiles, output / "dataset_profile.csv")
    write_markdown(profiles, output / "dataset_profile.md", profile_hash)
    write_figures(profiles, instances_by_dataset, figures)

    default_name = config.get("default_dataset", profiles[0]["name"])
    default_profile = next(profile for profile in profiles if profile["name"] == default_name)
    recommended = deepcopy_json(default_profile["recommendation"])
    recommended["_profile_hash"] = profile_hash
    recommended["_per_dataset"] = {
        profile["name"]: profile["recommendation"] for profile in profiles
    }
    yaml_text = yaml.safe_dump(recommended, sort_keys=False, allow_unicode=True)
    (output / "recommended_distill_config.yaml").write_text(yaml_text, encoding="utf-8")
    (output / "distill_recommended.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"Wrote profile {profile_hash} to {output}", flush=True)


def deepcopy_json(value: Any) -> Any:
    """Deep-copy a JSON-compatible structure without importing copy in the hot profiling path."""
    return json.loads(json.dumps(value))


if __name__ == "__main__":
    main()
