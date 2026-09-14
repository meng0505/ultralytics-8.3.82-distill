"""Configuration loading and validation for PRCD."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

DEFAULT_PRCD_CONFIG = {
    "distill": {
        "enabled": True,
        "objective": "prcd",
        "levels": ["P3"],
        "level_assignment": "single",
        "support": {
            "mode": "adaptive",
            "output_grid": 7,
            "fixed_window": 5,
            "context_margin": 1,
            "radius_min": 1,
            "radius_max": 4,
            "exclude_neighbor_objects": True,
            "core_supersample": 4,
            "min_core_mass": 1e-3,
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
            "response_source": "class_logit",
            "weight": 0.1,
        },
        "spd": {
            "enabled": True,
            "prototype_momentum": 0.99,
            "prototype_relation_enabled": False,
            "weight": 0.1,
        },
        "icd": {
            "enabled": True,
            "min_instances": 2,
            "max_instances_per_class": 32,
            "exclude_diagonal": True,
            "pair_direction": False,
            "memory_assisted": False,
            "cross_rank_gather": False,
            "weight": 0.05,
        },
        "optimization": {
            "distill_warmup_epochs": 10,
            "loss_weight_mode": "manual",
            "auto_apply_recommendation": False,
        },
        "baseline": {"weight": 0.1},
    }
}


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge mappings without mutating inputs."""
    output = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _merge(output[key], value)
        else:
            output[key] = deepcopy(value)
    return output


def level_to_stride(level: str | int) -> int:
    """Convert P3/P4/P5 or a numeric stride to its stride."""
    if isinstance(level, int):
        return level
    value = str(level).upper()
    if value.startswith("P") and value[1:].isdigit():
        return 2 ** int(value[1:])
    if value.isdigit():
        return int(value)
    raise ValueError(f"invalid feature level {level!r}")


def validate_distill_config(config: dict) -> dict:
    """Validate research-risk switches and normalize selected levels."""
    if "distill" not in config:
        raise KeyError("distillation config requires a top-level 'distill' mapping")
    distill = config["distill"]
    if distill["objective"] not in {"prcd", "direct_feature", "pixel_response", "hard_gt_roi"}:
        raise ValueError("distill.objective must be prcd, direct_feature, pixel_response, or hard_gt_roi")
    distill["strides"] = [level_to_stride(level) for level in distill["levels"]]
    if len(set(distill["strides"])) != len(distill["strides"]):
        raise ValueError("distillation levels resolve to duplicate strides")
    if distill["level_assignment"] not in {"single", "all", "scale_assigned"}:
        raise ValueError("level_assignment must be single, all, or scale_assigned")
    response_source = distill["crc"]["response_source"]
    if response_source not in {"class_logit", "class_probability", "class_times_quality"}:
        raise ValueError("CRC response_source must be class_logit, class_probability, or class_times_quality")
    if response_source == "class_times_quality":
        raise ValueError("class_times_quality is unavailable until localization quality is explicitly implemented")
    if distill["representation"]["aggregation"] not in {
        "concat",
        "student_self_weighted",
        "teacher_weighted",
        "simple_average",
    }:
        raise ValueError("unsupported object representation aggregation")
    if distill["icd"].get("pair_direction"):
        # The interface exists for ablation configuration, but the unverified risk
        # switch must not silently change the main objective.
        distill["icd"]["pair_direction_experimental"] = True
    if (
        distill["representation"]["aggregation"] in {"teacher_weighted", "student_self_weighted"}
        and not distill["crc"]["enabled"]
    ):
        raise ValueError("response-weighted object aggregation requires CRC to be enabled")
    for name in ("crc", "spd", "icd"):
        weight = float(distill[name]["weight"])
        if weight < 0:
            raise ValueError(f"{name} weight must be non-negative")
    return config


def load_distill_config(source: str | Path | dict | None, dataset_name: str | None = None) -> dict:
    """Load, merge and validate a PRCD config, optionally selecting a profiled dataset recommendation."""
    if source is None:
        loaded = {}
    elif isinstance(source, dict):
        loaded = deepcopy(source)
    else:
        loaded = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    if dataset_name and "_per_dataset" in loaded:
        if dataset_name not in loaded["_per_dataset"]:
            raise KeyError(f"dataset recommendation {dataset_name!r} not found")
        selected = loaded["_per_dataset"][dataset_name]
        loaded = {key: value for key, value in loaded.items() if key != "_per_dataset"}
        loaded["distill"] = selected["distill"]
        loaded["_evidence"] = selected.get("_evidence", {})
        loaded["_selected_dataset"] = dataset_name
    merged = _merge(DEFAULT_PRCD_CONFIG, loaded)
    return validate_distill_config(merged)


__all__ = ("DEFAULT_PRCD_CONFIG", "level_to_stride", "load_distill_config", "validate_distill_config")
