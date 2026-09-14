"""Range-aware OBB AP grouping tests."""

import numpy as np

from ultralytics.utils.stratified_obb_metrics import StratifiedOBBAP, obb_group_masks


def test_size_and_feature_support_groups_are_non_overlapping():
    boxes = np.array([[10, 10, 4, 4, 0], [20, 20, 16, 16, 0], [30, 30, 128, 128, 0]], dtype=np.float32)
    groups = obb_group_masks(boxes, support_stride=8)
    size_groups = [value for key, value in groups.items() if key.startswith("equiv_")]
    area_groups = [value for key, value in groups.items() if key.startswith("p3_area_")]
    assert np.stack(size_groups).sum(axis=0).tolist() == [1, 1, 1]
    assert np.stack(area_groups).sum(axis=0).tolist() == [1, 1, 1]


def test_perfect_predictions_retain_perfect_stratified_ap():
    accumulator = StratifiedOBBAP(iou_thresholds=np.array([0.5, 0.75]))
    boxes = np.array([[20, 20, 4, 4, 0], [60, 60, 40, 40, 0]], dtype=np.float32)
    accumulator.add_image(
        pred_boxes=boxes,
        pred_conf=np.array([0.9, 0.8]),
        pred_classes=np.array([0, 1]),
        gt_boxes=boxes,
        gt_classes=np.array([0, 1]),
        iou=np.eye(2, dtype=np.float32),
    )
    report = accumulator.compute()
    tiny = report["groups"]["equiv_tiny_lt_8px"]
    medium = report["groups"]["equiv_medium_32_96px"]
    assert tiny["targets"] == medium["targets"] == 1
    assert tiny["map50"] > 0.99
    assert medium["map50_95"] > 0.99
