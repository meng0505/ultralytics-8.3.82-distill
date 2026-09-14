# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Identity-safe paired optical/SAR datasets and indexing utilities."""

from __future__ import annotations

import csv
import importlib
import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import torch

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import IMG_FORMATS

SampleIdFn = Callable[[Path], str]


@dataclass(frozen=True)
class PairRecord:
    """One explicitly identified optical/SAR/label record."""

    sample_id: str
    sar_path: str | None
    optical_path: str | None
    label_path: str | None
    split: str | None = None


@dataclass
class PairingReport:
    """Pair-index diagnostics. Paths are strings to keep the report JSON serializable."""

    sar_count: int = 0
    optical_count: int = 0
    label_count: int = 0
    unique_ids: int = 0
    paired_count: int = 0
    duplicate_sar_ids: dict[str, list[str]] = field(default_factory=dict)
    duplicate_optical_ids: dict[str, list[str]] = field(default_factory=dict)
    duplicate_label_ids: dict[str, list[str]] = field(default_factory=dict)
    missing_sar_ids: list[str] = field(default_factory=list)
    missing_optical_ids: list[str] = field(default_factory=list)
    missing_label_ids: list[str] = field(default_factory=list)
    manifest_missing_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @property
    def has_errors(self) -> bool:
        """Whether identity resolution found ambiguous or incomplete records."""
        return any(
            (
                self.duplicate_sar_ids,
                self.duplicate_optical_ids,
                self.duplicate_label_ids,
                self.missing_sar_ids,
                self.missing_optical_ids,
                self.missing_label_ids,
                self.manifest_missing_ids,
            )
        )


def exact_stem_id(path: Path) -> str:
    """Default identity: the complete filename stem, without normalization or sorting."""
    return path.stem


def resolve_sample_id_fn(spec: str | SampleIdFn | None) -> SampleIdFn:
    """Resolve a user mapping callable from ``module:function`` or return exact-stem mapping."""
    if spec is None or spec == "exact_stem":
        return exact_stem_id
    if callable(spec):
        return spec
    if ":" not in spec:
        raise ValueError("filename mapper must be 'exact_stem', a callable, or 'module:function'")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"filename mapper {spec!r} is not callable")
    return function


def _iter_files(source: str | Path | Iterable[str | Path] | None, suffixes: set[str]) -> list[Path]:
    """Resolve a directory, explicit path collection, or newline-delimited file list."""
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if path.is_dir():
            return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in suffixes)
        if path.is_file() and path.suffix.lower() == ".txt" and "txt" not in suffixes:
            paths = []
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                item = line.strip()
                if item:
                    candidate = Path(item).expanduser()
                    paths.append(candidate if candidate.is_absolute() else (path.parent / candidate).resolve())
            return paths
        return [path]
    return [Path(p).expanduser() for p in source]


def _index_files(paths: Iterable[Path], sample_id_fn: SampleIdFn) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Index paths by identity while retaining duplicate diagnostics."""
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(str(sample_id_fn(path)), []).append(path)
    duplicates = {key: [str(p) for p in value] for key, value in grouped.items() if len(value) > 1}
    unique = {key: value[0] for key, value in grouped.items() if len(value) == 1}
    return unique, duplicates


def read_pair_manifest(path: str | Path, split: str | None = None) -> list[dict]:
    """Read CSV or JSON pair manifests without assuming one fixed column vocabulary."""
    manifest_path = Path(path).expanduser()
    if manifest_path.suffix.lower() == ".csv":
        with manifest_path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    elif manifest_path.suffix.lower() == ".json":
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = data.get("pairs", data.get("records", data)) if isinstance(data, dict) else data
    else:
        raise ValueError(f"unsupported pair manifest format: {manifest_path.suffix}")
    if not isinstance(rows, list):
        raise TypeError("pair manifest must contain a list of records")
    if split is not None:
        rows = [row for row in rows if not row.get("split") or str(row["split"]) == split]
    return rows


def _manifest_id(row: dict) -> str:
    """Extract a manifest identity using common explicit column names."""
    for key in ("sample_id", "stem", "id", "image_id"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    raise KeyError("manifest row has no sample identity column (sample_id/stem/id/image_id)")


def build_pair_records(
    sar_source: str | Path | Iterable[str | Path],
    optical_source: str | Path | Iterable[str | Path],
    label_source: str | Path | Iterable[str | Path] | None = None,
    *,
    manifest: str | Path | None = None,
    split: str | None = None,
    sample_id_fn: str | SampleIdFn | None = None,
    require_labels: bool = True,
) -> tuple[list[PairRecord], PairingReport]:
    """Build complete pair records from identity maps, never positional ordering."""
    id_fn = resolve_sample_id_fn(sample_id_fn)
    sar_paths = _iter_files(sar_source, set(IMG_FORMATS))
    optical_paths = _iter_files(optical_source, set(IMG_FORMATS))
    label_paths = _iter_files(label_source, {"txt"}) if label_source is not None else []
    sar, duplicate_sar = _index_files(sar_paths, id_fn)
    optical, duplicate_optical = _index_files(optical_paths, id_fn)
    labels, duplicate_labels = _index_files(label_paths, id_fn)

    report = PairingReport(
        sar_count=len(sar_paths),
        optical_count=len(optical_paths),
        label_count=len(label_paths),
        duplicate_sar_ids=duplicate_sar,
        duplicate_optical_ids=duplicate_optical,
        duplicate_label_ids=duplicate_labels,
    )
    manifest_rows = read_pair_manifest(manifest, split) if manifest else None
    if manifest_rows is not None:
        requested_ids = []
        seen = set()
        for row in manifest_rows:
            sample_id = _manifest_id(row)
            if sample_id in seen:
                report.manifest_missing_ids.append(f"duplicate:{sample_id}")
            else:
                requested_ids.append(sample_id)
                seen.add(sample_id)
    else:
        requested_ids = sorted(set(sar) | set(optical) | (set(labels) if require_labels else set()))

    records = []
    for sample_id in requested_ids:
        sar_path = sar.get(sample_id)
        optical_path = optical.get(sample_id)
        label_path = labels.get(sample_id) if label_source is not None else None
        if sar_path is None:
            report.missing_sar_ids.append(sample_id)
        if optical_path is None:
            report.missing_optical_ids.append(sample_id)
        if require_labels and label_path is None:
            report.missing_label_ids.append(sample_id)
        complete = sar_path is not None and optical_path is not None and (not require_labels or label_path is not None)
        if complete:
            records.append(
                PairRecord(
                    sample_id=sample_id,
                    sar_path=str(sar_path),
                    optical_path=str(optical_path),
                    label_path=str(label_path) if label_path else None,
                    split=split,
                )
            )
    report.unique_ids = len(requested_ids)
    report.paired_count = len(records)
    return records, report


def _capture_rng_state() -> tuple:
    """Capture worker-local RNG states used by the Ultralytics transform chain."""
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def _restore_rng_state(state: tuple) -> None:
    """Restore worker-local RNG states."""
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.random.set_rng_state(state[2])


class ExplicitLabelYOLODataset(YOLODataset):
    """YOLO dataset supporting an explicit label root for nonstandard image directory names."""

    def __init__(
        self,
        *args,
        label_path: str | Path | None = None,
        label_cache_path: str | Path | None = None,
        filename_mapper: str | SampleIdFn | None = None,
        **kwargs,
    ):
        self.explicit_label_path = str(label_path) if label_path else None
        self.explicit_label_cache_path = str(label_cache_path) if label_cache_path else None
        self.sample_id_fn = resolve_sample_id_fn(filename_mapper)
        super().__init__(*args, **kwargs)

    def label_files_for_images(self, image_files: list[str]) -> list[str]:
        """Resolve labels through sample identity rather than a directory-name substitution."""
        if not self.explicit_label_path:
            from ultralytics.data.utils import img2label_paths

            return img2label_paths(image_files)
        label_paths = _iter_files(self.explicit_label_path, {"txt"})
        labels, duplicates = _index_files(label_paths, self.sample_id_fn)
        if duplicates:
            raise ValueError(f"duplicate explicit label IDs: {list(duplicates)[:8]}")
        sample_ids = [str(self.sample_id_fn(Path(path))) for path in image_files]
        missing = [sample_id for sample_id in sample_ids if sample_id not in labels]
        if missing:
            raise ValueError(f"explicit label root is missing {len(missing)} IDs; examples={missing[:8]}")
        return [str(labels[sample_id]) for sample_id in sample_ids]

    def label_cache_file(self, image_files: list[str], label_files: list[str]) -> Path:
        """Keep run-specific/partial paired caches away from authoritative dataset caches."""
        if self.explicit_label_cache_path:
            path = Path(self.explicit_label_cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return Path(label_files[0]).parent.with_suffix(".cache")


class PairedOBBDataset(ExplicitLabelYOLODataset):
    """YOLO OBB dataset returning synchronized SAR and optical tensors."""

    def __init__(
        self,
        *args,
        optical_path: str | Path,
        label_path: str | Path | None = None,
        pair_manifest: str | Path | None = None,
        pair_manifest_split: str | None = None,
        filename_mapper: str | SampleIdFn | None = None,
        strict_pairs: bool = True,
        **kwargs,
    ):
        sar_source = kwargs.get("img_path", args[0] if args else None)
        if sar_source is None:
            raise ValueError("PairedOBBDataset requires img_path")
        self.optical_path = str(optical_path)
        self.pair_manifest = str(pair_manifest) if pair_manifest else None
        self.pair_manifest_split = pair_manifest_split
        self.sample_id_fn = resolve_sample_id_fn(filename_mapper)
        self.strict_pairs = strict_pairs
        self._pair_modality = "sar"
        self._trace_pair_ids = False
        self._source_sample_ids: list[str] = []
        super().__init__(*args, label_path=label_path, filename_mapper=filename_mapper, **kwargs)

        records, report = build_pair_records(
            sar_source,
            self.optical_path,
            manifest=self.pair_manifest,
            split=self.pair_manifest_split,
            sample_id_fn=self.sample_id_fn,
            require_labels=False,
        )
        self.pairing_report = report
        self._pairs = {record.sample_id: record for record in records}
        self.sample_ids = [str(self.sample_id_fn(Path(path))) for path in self.im_files]
        missing = [sample_id for sample_id in self.sample_ids if sample_id not in self._pairs]
        if strict_pairs and (missing or report.has_errors):
            raise ValueError(
                f"invalid optical/SAR pairing: selected_missing={len(missing)}, "
                f"source_errors={report.has_errors}; examples={missing[:8]}; report={report.to_dict()}"
            )

    def _load_optical(self, path: str) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load an optical image with the same pre-transform resize policy as BaseDataset."""
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Optical image not found or corrupt: {path}")
        h0, w0 = image.shape[:2]
        ratio = self.imgsz / max(h0, w0)
        if ratio != 1:
            width = min(int(np.ceil(w0 * ratio)), self.imgsz)
            height = min(int(np.ceil(h0 * ratio)), self.imgsz)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return image, (h0, w0), image.shape[:2]

    def _get_optical_image_and_label(self, index: int) -> dict:
        """Mirror BaseDataset.get_image_and_label using the indexed optical counterpart."""
        label = deepcopy(self.labels[index])
        label.pop("shape", None)
        sample_id = self.sample_ids[index]
        record = self._pairs.get(sample_id)
        if record is None:
            raise KeyError(f"no optical pair for SAR sample {sample_id!r}")
        image, original_shape, resized_shape = self._load_optical(record.optical_path)
        label["img"] = image
        label["im_file"] = record.optical_path
        label["ori_shape"] = original_shape
        label["resized_shape"] = resized_shape
        label["ratio_pad"] = (
            resized_shape[0] / original_shape[0],
            resized_shape[1] / original_shape[1],
        )
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def get_image_and_label(self, index):
        """Let stock mix transforms request the same identities from either modality."""
        if self._trace_pair_ids:
            self._source_sample_ids.append(self.sample_ids[index])
        return (
            self._get_optical_image_and_label(index)
            if self._pair_modality == "optical"
            else super().get_image_and_label(index)
        )

    def __getitem__(self, index):
        """Return one pair after replaying identical image-local geometry on both modalities."""
        self._pair_modality = "sar"
        sar_labels = super().get_image_and_label(index)
        sample_id = self.sample_ids[index]
        record = self._pairs.get(sample_id)
        if record is None:
            raise KeyError(f"no optical pair for SAR sample {sample_id!r}")

        optical_labels = self._get_optical_image_and_label(index)
        optical_shape = optical_labels["ori_shape"]
        if optical_shape != tuple(sar_labels["ori_shape"]):
            raise ValueError(
                f"pair {sample_id!r} has unequal source dimensions: "
                f"SAR={sar_labels['ori_shape']}, optical={optical_shape}"
            )
        state_before = _capture_rng_state()
        buffer_before = list(self.buffer)
        self._source_sample_ids = [sample_id]
        self._trace_pair_ids = True
        self._pair_modality = "sar"
        sar_output = self.transforms(sar_labels)
        sar_source_ids = tuple(self._source_sample_ids)
        state_after_sar = _capture_rng_state()
        buffer_after_sar = list(self.buffer)
        _restore_rng_state(state_before)
        self.buffer = list(buffer_before)
        self._source_sample_ids = [sample_id]
        self._pair_modality = "optical"
        optical_output = self.transforms(optical_labels)
        optical_source_ids = tuple(self._source_sample_ids)
        self._trace_pair_ids = False
        self._pair_modality = "sar"
        self.buffer = buffer_after_sar
        _restore_rng_state(state_after_sar)

        if sar_source_ids != optical_source_ids:
            raise RuntimeError(
                f"mix augmentation selected different pair identities for {sample_id!r}: "
                f"SAR={sar_source_ids}, optical={optical_source_ids}"
            )
        if sar_output["bboxes"].shape != optical_output["bboxes"].shape or not torch.allclose(
            sar_output["bboxes"], optical_output["bboxes"], atol=1e-6, rtol=0
        ):
            raise RuntimeError(f"geometric transform desynchronized OBBs for pair {sample_id!r}")
        sar_output["img_opt"] = optical_output["img"]
        sar_output["img_sar"] = sar_output["img"]
        sar_output["sar_path"] = sar_output["im_file"]
        sar_output["opt_path"] = record.optical_path
        sar_output["sample_id"] = sample_id
        sar_output["source_sample_ids"] = sar_source_ids
        return sar_output

    @staticmethod
    def collate_fn(batch):
        """Collate both image tensors while preserving standard YOLO OBB target semantics."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(item.values()) for item in batch]))
        for index, key in enumerate(keys):
            value = values[index]
            if key in {"img", "img_opt"}:
                value = torch.stack(value, 0)
            elif key == "img_sar":
                continue
            elif key in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[key] = value
        new_batch["img_sar"] = new_batch["img"]
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for index in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][index] += index
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


__all__ = (
    "PairRecord",
    "ExplicitLabelYOLODataset",
    "PairedOBBDataset",
    "PairingReport",
    "build_pair_records",
    "exact_stem_id",
    "read_pair_manifest",
    "resolve_sample_id_fn",
)
