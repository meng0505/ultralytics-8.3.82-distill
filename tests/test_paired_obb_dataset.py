"""Tests for identity-safe paired OBB loading."""

from pathlib import Path

import cv2
import numpy as np

from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.paired_obb_dataset import PairedOBBDataset, build_pair_records
from ultralytics.utils import DEFAULT_CFG


def make_pair_tree(root: Path, count: int = 2):
    """Create a tiny normalized-polygon OBB pair tree."""
    sar, optical, labels = root / "images/train", root / "images-opt/train", root / "labels/train"
    for directory in (sar, optical, labels):
        directory.mkdir(parents=True)
    for index in range(count):
        image = np.full((64, 64, 3), 40 + index, dtype=np.uint8)
        cv2.imwrite(str(sar / f"id{index}.png"), image)
        cv2.imwrite(str(optical / f"id{index}.png"), image + 50)
        (labels / f"id{index}.txt").write_text(
            "0 0.25 0.35 0.65 0.25 0.72 0.55 0.32 0.65\n", encoding="utf-8"
        )
    return sar, optical, labels


def test_pair_id_matching_and_missing_detection(tmp_path):
    sar, optical, labels = make_pair_tree(tmp_path)
    records, report = build_pair_records(sar, optical, labels)
    assert [record.sample_id for record in records] == ["id0", "id1"]
    assert not report.has_errors
    (optical / "id1.png").unlink()
    records, report = build_pair_records(sar, optical, labels)
    assert [record.sample_id for record in records] == ["id0"]
    assert report.missing_optical_ids == ["id1"]


def test_nonstandard_sar_root_uses_explicit_labels(tmp_path):
    _, optical, labels = make_pair_tree(tmp_path, 1)
    sar = tmp_path / "imagesSAR/train"
    sar.mkdir(parents=True)
    cv2.imwrite(str(sar / "id0.png"), np.zeros((64, 64, 3), dtype=np.uint8))
    hyp = get_cfg(DEFAULT_CFG)
    hyp.mosaic = hyp.mixup = hyp.copy_paste = 0.0
    dataset = PairedOBBDataset(
        img_path=sar,
        optical_path=optical,
        label_path=labels,
        imgsz=64,
        batch_size=1,
        augment=False,
        hyp=hyp,
        task="obb",
        data={"names": {0: "object"}, "nc": 1},
    )
    assert dataset.label_files == [str(labels / "id0.txt")]
    sample = dataset[0]
    assert sample["bboxes"].shape == (1, 5)
    assert sample["img_sar"] is sample["img"]
    batch = dataset.collate_fn([sample])
    assert batch["img_sar"] is batch["img"]


def test_manifest_filters_by_explicit_id_and_split(tmp_path):
    sar, optical, labels = make_pair_tree(tmp_path, 2)
    manifest = tmp_path / "pairs.csv"
    manifest.write_text("sample_id,split\nid1,train\nid0,test\n", encoding="utf-8")
    records, report = build_pair_records(sar, optical, labels, manifest=manifest, split="train")
    assert [record.sample_id for record in records] == ["id1"]
    assert report.paired_count == 1
    assert not report.has_errors


def test_standard_validator_builder_honors_explicit_test_labels(tmp_path):
    _, _, labels = make_pair_tree(tmp_path, 1)
    sar = tmp_path / "imagesSAR/test"
    sar.mkdir(parents=True)
    cv2.imwrite(str(sar / "id0.png"), np.zeros((64, 64, 3), dtype=np.uint8))
    hyp = get_cfg(DEFAULT_CFG)
    hyp.task, hyp.imgsz, hyp.split = "obb", 64, "test"
    dataset = build_yolo_dataset(
        hyp,
        sar,
        batch=1,
        data={"names": {0: "object"}, "nc": 1, "labels_test": str(labels)},
        mode="val",
        stride=32,
    )
    assert dataset.label_files == [str(labels / "id0.txt")]
    assert dataset[0]["bboxes"].shape == (1, 5)


def test_duplicate_id_detection(tmp_path):
    sar, optical, labels = make_pair_tree(tmp_path, 1)
    cv2.imwrite(str(optical / "id0.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))
    _, report = build_pair_records(sar, optical, labels)
    assert "id0" in report.duplicate_optical_ids


def test_fraction_uses_full_pair_audit_and_isolated_label_cache(tmp_path):
    sar, optical, labels = make_pair_tree(tmp_path, 4)
    cache_path = tmp_path / "run_cache/labels.cache"
    hyp = get_cfg(DEFAULT_CFG)
    hyp.mosaic = hyp.mixup = hyp.copy_paste = 0.0
    dataset = PairedOBBDataset(
        img_path=sar,
        optical_path=optical,
        label_path=labels,
        label_cache_path=cache_path,
        fraction=0.5,
        imgsz=64,
        batch_size=1,
        augment=False,
        hyp=hyp,
        task="obb",
        data={"names": {0: "object"}, "nc": 1},
    )
    assert len(dataset) == 2
    assert dataset.pairing_report.paired_count == 4
    assert cache_path.exists()
    assert not (labels.parent / "train.cache").exists()


def test_synchronized_geometry_and_rotation(tmp_path):
    sar, optical, _ = make_pair_tree(tmp_path)
    hyp = get_cfg(DEFAULT_CFG)
    hyp.mosaic = hyp.mixup = hyp.copy_paste = 0.0
    hyp.degrees = 25.0
    hyp.translate = 0.2
    hyp.scale = 0.2
    hyp.fliplr = hyp.flipud = 0.5
    dataset = PairedOBBDataset(
        img_path=sar,
        optical_path=optical,
        imgsz=64,
        batch_size=2,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=False,
        stride=32,
        pad=0,
        prefix="test: ",
        task="obb",
        data={"names": {0: "object"}, "nc": 1},
    )
    sample = dataset[0]  # Internal allclose assertion validates replayed OBB geometry.
    assert sample["img"].shape == sample["img_opt"].shape == (3, 64, 64)
    assert sample["sample_id"] == "id0"
    assert sample["bboxes"].shape[1] == 5
    assert np.isfinite(sample["bboxes"].numpy()).all()


def test_paired_mosaic_replays_same_source_ids(tmp_path):
    sar, optical, _ = make_pair_tree(tmp_path, count=4)
    hyp = get_cfg(DEFAULT_CFG)
    hyp.mosaic = 1.0
    hyp.mixup = hyp.copy_paste = 0.0
    dataset = PairedOBBDataset(
        img_path=sar,
        optical_path=optical,
        imgsz=64,
        batch_size=2,
        augment=True,
        hyp=hyp,
        task="obb",
        data={"names": {0: "object"}, "nc": 1},
    )
    # Populate the stock mosaic buffer with more than one identity.
    for index in range(4):
        super(PairedOBBDataset, dataset).get_image_and_label(index)
    sample = dataset[0]
    assert len(sample["source_sample_ids"]) == 4
    assert sample["img"].shape == sample["img_opt"].shape
