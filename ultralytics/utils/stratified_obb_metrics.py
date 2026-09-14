"""COCO-style ignored-range OBB AP accumulation for size and feature-support strata."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ultralytics.utils.metrics import ap_per_class


def obb_group_masks(
    boxes: np.ndarray,
    *,
    tiny_max: float = 8.0,
    small_max: float = 32.0,
    medium_max: float = 96.0,
    support_stride: int = 8,
) -> dict[str, np.ndarray]:
    """Assign native-pixel OBBs to non-overlapping size/support bins."""
    width = boxes[:, 2] if len(boxes) else np.zeros(0)
    height = boxes[:, 3] if len(boxes) else np.zeros(0)
    equivalent_side = np.sqrt(np.clip(width * height, 0, None))
    area = width * height
    area_cells = width * height / float(support_stride**2)
    min_dimension_cells = np.minimum(width, height) / float(support_stride)
    groups = {
        f"equiv_tiny_lt_{tiny_max:g}px": equivalent_side < tiny_max,
        f"equiv_small_{tiny_max:g}_{small_max:g}px": (equivalent_side >= tiny_max)
        & (equivalent_side < small_max),
        f"equiv_medium_{small_max:g}_{medium_max:g}px": (equivalent_side >= small_max)
        & (equivalent_side < medium_max),
        f"equiv_large_ge_{medium_max:g}px": equivalent_side >= medium_max,
        f"p{int(np.log2(support_stride))}_area_lt_1_cell": area_cells < 1,
        f"p{int(np.log2(support_stride))}_area_1_2_cells": (area_cells >= 1) & (area_cells < 2),
        f"p{int(np.log2(support_stride))}_area_ge_2_cells": area_cells >= 2,
        f"p{int(np.log2(support_stride))}_min_dim_lt_1_cell": min_dimension_cells < 1,
        f"p{int(np.log2(support_stride))}_min_dim_1_2_cells": (min_dimension_cells >= 1)
        & (min_dimension_cells < 2),
        f"p{int(np.log2(support_stride))}_min_dim_ge_2_cells": min_dimension_cells >= 2,
    }
    for axis_name, values in (("width", width), ("height", height)):
        groups.update(
            {
                f"{axis_name}_lt_{tiny_max:g}px": values < tiny_max,
                f"{axis_name}_{tiny_max:g}_{small_max:g}px": (values >= tiny_max) & (values < small_max),
                f"{axis_name}_{small_max:g}_{medium_max:g}px": (values >= small_max)
                & (values < medium_max),
                f"{axis_name}_ge_{medium_max:g}px": values >= medium_max,
            }
        )
    tiny_area, small_area, medium_area = tiny_max**2, small_max**2, medium_max**2
    groups.update(
        {
            f"pixel_area_lt_{tiny_area:g}": area < tiny_area,
            f"pixel_area_{tiny_area:g}_{small_area:g}": (area >= tiny_area) & (area < small_area),
            f"pixel_area_{small_area:g}_{medium_area:g}": (area >= small_area) & (area < medium_area),
            f"pixel_area_ge_{medium_area:g}": area >= medium_area,
        }
    )
    return groups


def greedy_correct(iou: np.ndarray, gt_classes: np.ndarray, pred_classes: np.ndarray, threshold: float) -> np.ndarray:
    """Match predictions to GT exactly once using the same descending-IoU policy as BaseValidator."""
    correct = np.zeros(len(pred_classes), dtype=bool)
    if not len(iou):
        return correct
    eligible = iou * (gt_classes[:, None] == pred_classes[None, :])
    matches = np.array(np.nonzero(eligible >= threshold)).T
    if not len(matches):
        return correct
    if len(matches) > 1:
        matches = matches[eligible[matches[:, 0], matches[:, 1]].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    correct[matches[:, 1].astype(int)] = True
    return correct


class StratifiedOBBAP:
    """Accumulate per-range OBB AP while ignoring out-of-range objects and detections."""

    def __init__(
        self,
        iou_thresholds: np.ndarray | None = None,
        *,
        tiny_max: float = 8.0,
        small_max: float = 32.0,
        medium_max: float = 96.0,
        support_stride: int = 8,
    ):
        self.iou_thresholds = (
            np.linspace(0.5, 0.95, 10) if iou_thresholds is None else np.asarray(iou_thresholds)
        )
        self.parameters = {
            "tiny_max_px": float(tiny_max),
            "small_max_px": float(small_max),
            "medium_max_px": float(medium_max),
            "support_stride": int(support_stride),
        }
        self._group_kwargs = {
            "tiny_max": tiny_max,
            "small_max": small_max,
            "medium_max": medium_max,
            "support_stride": support_stride,
        }
        self.records = defaultdict(
            lambda: [
                {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
                for _ in self.iou_thresholds
            ]
        )

    def add_image(
        self,
        pred_boxes: np.ndarray,
        pred_conf: np.ndarray,
        pred_classes: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
        iou: np.ndarray,
    ) -> None:
        """Add one native-coordinate image with an [N_gt, N_pred] probabilistic-IoU matrix."""
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 5)
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 5)
        pred_conf = np.asarray(pred_conf, dtype=np.float32).reshape(-1)
        pred_classes = np.asarray(pred_classes, dtype=np.int64).reshape(-1)
        gt_classes = np.asarray(gt_classes, dtype=np.int64).reshape(-1)
        iou = np.asarray(iou, dtype=np.float32).reshape(len(gt_boxes), len(pred_boxes))
        gt_groups = obb_group_masks(gt_boxes, **self._group_kwargs)
        pred_groups = obb_group_masks(pred_boxes, **self._group_kwargs)

        for group_name, gt_group in gt_groups.items():
            pred_group = pred_groups[group_name]
            for threshold_index, threshold in enumerate(self.iou_thresholds):
                record = self.records[group_name][threshold_index]
                group_iou = iou[gt_group]
                correct = greedy_correct(group_iou, gt_classes[gt_group], pred_classes, float(threshold))
                excluded = ~gt_group
                if excluded.any() and len(pred_boxes):
                    excluded_match = (
                        iou[excluded]
                        * (gt_classes[excluded, None] == pred_classes[None, :])
                        >= float(threshold)
                    ).any(axis=0)
                else:
                    excluded_match = np.zeros(len(pred_boxes), dtype=bool)
                # A correct in-range match wins. Otherwise use COCO-like range
                # ignore for detections or GT matches outside the current bin.
                ignored = ~correct & (excluded_match | ~pred_group)
                keep = ~ignored
                record["tp"].append(correct[keep, None])
                record["conf"].append(pred_conf[keep])
                record["pred_cls"].append(pred_classes[keep])
                record["target_cls"].append(gt_classes[gt_group])

    @staticmethod
    def _concatenate(values: list[np.ndarray], shape: tuple[int, ...], dtype) -> np.ndarray:
        return np.concatenate(values, axis=0) if values else np.zeros(shape, dtype=dtype)

    def compute(self) -> dict:
        """Compute AP50, AP50-95 and per-class AP for every populated group."""
        groups = {}
        for group_name, threshold_records in self.records.items():
            class_ap = defaultdict(lambda: np.full(len(self.iou_thresholds), np.nan, dtype=np.float64))
            target_counts = None
            for threshold_index, record in enumerate(threshold_records):
                tp = self._concatenate(record["tp"], (0, 1), bool)
                conf = self._concatenate(record["conf"], (0,), np.float32)
                pred_cls = self._concatenate(record["pred_cls"], (0,), np.int64)
                target_cls = self._concatenate(record["target_cls"], (0,), np.int64)
                if target_counts is None:
                    unique, counts = np.unique(target_cls, return_counts=True)
                    target_counts = {int(key): int(value) for key, value in zip(unique, counts)}
                if not len(target_cls):
                    continue
                result = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
                ap, class_ids = result[5], result[6]
                for row, class_id in enumerate(class_ids):
                    class_ap[int(class_id)][threshold_index] = float(ap[row, 0])
            per_class = {}
            for class_id, values in sorted(class_ap.items()):
                per_class[str(class_id)] = {
                    "targets": target_counts.get(class_id, 0),
                    "ap50": float(values[0]) if np.isfinite(values[0]) else None,
                    "ap50_95": float(np.nanmean(values)) if np.isfinite(values).any() else None,
                    "ap_by_iou": [
                        float(value) if np.isfinite(value) else None for value in values
                    ],
                }
            valid_ap50 = [value["ap50"] for value in per_class.values() if value["ap50"] is not None]
            valid_map = [value["ap50_95"] for value in per_class.values() if value["ap50_95"] is not None]
            groups[group_name] = {
                "targets": int(sum((target_counts or {}).values())),
                "targets_per_class": {str(key): value for key, value in sorted((target_counts or {}).items())},
                "map50": float(np.mean(valid_ap50)) if valid_ap50 else None,
                "map50_95": float(np.mean(valid_map)) if valid_map else None,
                "per_class": per_class,
            }
        return {
            "metric": "OBB AP with out-of-range GT/detections ignored",
            "iou_thresholds": self.iou_thresholds.tolist(),
            "group_parameters": self.parameters,
            "groups": groups,
        }

    @staticmethod
    def save(report: dict, json_path: str | Path, csv_path: str | Path | None = None) -> None:
        """Write nested JSON and a flat group/class CSV."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if csv_path is None:
            return
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=("group", "class_id", "targets", "ap50", "ap50_95")
            )
            writer.writeheader()
            for group_name, group in report["groups"].items():
                for class_id, metrics in group["per_class"].items():
                    writer.writerow(
                        {
                            "group": group_name,
                            "class_id": class_id,
                            "targets": metrics["targets"],
                            "ap50": metrics["ap50"],
                            "ap50_95": metrics["ap50_95"],
                        }
                    )


__all__ = ("StratifiedOBBAP", "greedy_correct", "obb_group_masks")
