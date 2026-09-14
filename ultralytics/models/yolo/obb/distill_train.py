"""Dedicated optical-teacher/SAR-student OBB trainer for PRCD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import warnings
from copy import copy
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ultralytics.data.paired_obb_dataset import ExplicitLabelYOLODataset, PairedOBBDataset
from ultralytics.models import yolo
from ultralytics.models.yolo.obb.distill_model import (
    DistillOBBModel,
    feature_channels_by_stride,
    install_head_capture,
    pop_captured_features,
    raw_obb_component_maps,
    stride_tensor_map,
)
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.distill_config import load_distill_config
from ultralytics.utils.torch_utils import de_parallel

# PyTorch has no deterministic CUDA backward for grid_sample, which PRCD uses
# for object-aligned support sampling. Keep deterministic mode and all other
# warnings, but avoid printing this known one-time limitation every run.
warnings.filterwarnings(
    "ignore",
    message=r"grid_sampler_2d_backward_cuda does not have a deterministic implementation.*",
    category=UserWarning,
)


def _reset_prcd_diagnostics(trainer) -> None:
    """Reset rank-zero diagnostic accumulators without widening the main loss table."""
    if RANK in {-1, 0}:
        trainer._prcd_diagnostic_sum = None
        trainer._prcd_diagnostic_batches = 0


def _accumulate_prcd_diagnostics(trainer) -> None:
    """Accumulate detached per-batch PRCD observability metrics on rank zero."""
    if RANK not in {-1, 0} or not trainer.distill_enabled:
        return
    model = de_parallel(trainer.model)
    values = getattr(model, "latest_prcd_diagnostics", None)
    if values is None:
        return
    values = values.detach().float().cpu()
    if trainer._prcd_diagnostic_sum is None:
        trainer._prcd_diagnostic_sum = values.clone()
    else:
        trainer._prcd_diagnostic_sum += values
    trainer._prcd_diagnostic_batches += 1
    model.latest_prcd_diagnostics = None


def _save_prcd_diagnostics(trainer) -> None:
    """Write epoch-mean diagnostic values to a dedicated CSV."""
    if (
        RANK not in {-1, 0}
        or not trainer.distill_enabled
        or trainer._prcd_diagnostic_sum is None
        or not trainer._prcd_diagnostic_batches
    ):
        return
    model = de_parallel(trainer.model)
    names = model.prcd_diagnostic_names
    values = (trainer._prcd_diagnostic_sum / trainer._prcd_diagnostic_batches).tolist()
    path = trainer.save_dir / "prcd_diagnostics.csv"
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(("epoch", *names))
        writer.writerow((trainer.epoch + 1, *values))


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a checkpoint without loading it twice."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class DistillOBBTrainer(yolo.obb.OBBTrainer):
    """Keep the frozen optical teacher trainer-owned and optimize only the SAR student."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = {} if overrides is None else dict(overrides)
        overrides["task"] = "obb"
        selected_config = load_distill_config(
            overrides.get("distill"), dataset_name=overrides.get("distill_dataset")
        )
        if selected_config["distill"]["enabled"]:
            overrides.setdefault("mosaic", 0.0)
            overrides.setdefault("mixup", 0.0)
            overrides.setdefault("copy_paste", 0.0)
        super().__init__(cfg, overrides, _callbacks)
        self.distill_config = selected_config
        self.teacher = None
        self._distill_step = 0
        self.add_callback("on_train_epoch_start", _reset_prcd_diagnostics)
        self.add_callback("on_train_batch_end", _accumulate_prcd_diagnostics)
        self.add_callback("on_train_epoch_end", _save_prcd_diagnostics)

    @property
    def distill_enabled(self) -> bool:
        """Whether the selected configuration enables distillation."""
        return bool(self.distill_config["distill"]["enabled"])

    def _load_teacher(self):
        """Load, freeze and hook an optical OBB teacher on the local rank."""
        if not self.args.teacher:
            raise ValueError("teacher=<optical best.pt> is required when distillation is enabled")
        teacher, _ = attempt_load_one_weight(self.args.teacher, device=self.device, inplace=True, fuse=False)
        teacher = teacher.float().to(self.device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        if teacher.model[-1].nc != self.data["nc"]:
            raise ValueError(
                f"teacher class count {teacher.model[-1].nc} does not match dataset nc={self.data['nc']}"
            )
        install_head_capture(teacher)
        self.args.teacher_hash = file_sha256(self.args.teacher)
        self.args.distill_profile_hash = self.distill_config.get("_profile_hash")
        return teacher

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build a normal OBB detector plus distillation-only student state."""
        model = DistillOBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if self.distill_enabled:
            self.teacher = self._load_teacher()
            model.configure_distillation(
                self.distill_config,
                teacher_channels=feature_channels_by_stride(self.teacher),
                student_channels=feature_channels_by_stride(model),
                seed=int(self.args.seed),
            )
            common = model.distiller.strides
            LOGGER.info(f"PRCD common teacher/student strides: {common}")
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """Use paired samples only for enabled training; validation and disabled runs are official paths."""
        label_path = self.data.get(f"labels_{mode}")
        if mode != "train" or not self.distill_enabled:
            if label_path:
                gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
                return ExplicitLabelYOLODataset(
                    img_path=img_path,
                    label_path=label_path,
                    label_cache_path=self.save_dir / f"labels_{mode}.cache",
                    filename_mapper=self.data.get("filename_mapper"),
                    imgsz=self.args.imgsz,
                    batch_size=batch,
                    augment=mode == "train",
                    hyp=self.args,
                    rect=self.args.rect or mode == "val",
                    cache=self.args.cache or None,
                    single_cls=self.args.single_cls or False,
                    stride=gs,
                    pad=0.0 if mode == "train" else 0.5,
                    prefix=colorstr(f"{mode}: "),
                    task="obb",
                    classes=self.args.classes,
                    data=self.data,
                    fraction=self.args.fraction if mode == "train" else 1.0,
                )
            return super().build_dataset(img_path, mode, batch)
        optical_path = self.data.get("optical_train") or getattr(self.args, "opt_root", None)
        if not optical_path:
            raise KeyError("paired data YAML needs optical_train, or pass opt_root=<path>")
        pair_manifest = self.data.get("pair_manifest") or getattr(self.args, "pair_manifest", None)
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return PairedOBBDataset(
            img_path=img_path,
            optical_path=optical_path,
            label_path=label_path,
            label_cache_path=self.save_dir / f"labels_{mode}.cache",
            pair_manifest=pair_manifest,
            pair_manifest_split="train" if pair_manifest else None,
            filename_mapper=self.data.get("filename_mapper"),
            strict_pairs=bool(getattr(self.args, "strict_pairs", True)),
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0,
            prefix=colorstr("train: "),
            task="obb",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
        )

    def preprocess_batch(self, batch):
        """Normalize paired tensors and create a detached teacher payload."""
        if not self.distill_enabled or "img_opt" not in batch:
            return super().preprocess_batch(batch)
        optical = batch["img_opt"].to(self.device, non_blocking=True).float() / 255
        shift = int(getattr(self.args, "optical_shift", 0))
        if shift:
            direction = str(getattr(self.args, "optical_shift_direction", "xy")).lower()
            dx, dy = (shift if "x" in direction else 0), (shift if "y" in direction else 0)
            optical = F.pad(optical, (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)))
            y0, x0 = max(-dy, 0), max(-dx, 0)
            optical = optical[..., y0 : y0 + batch["img_opt"].shape[-2], x0 : x0 + batch["img_opt"].shape[-1]]
        batch = super().preprocess_batch(batch)
        if optical.shape[-2:] != batch["img"].shape[-2:]:
            optical = F.interpolate(optical, size=batch["img"].shape[-2:], mode="bilinear", align_corners=False)
        batch["img_opt"] = optical
        self.teacher.eval()
        with torch.no_grad():
            predictions = self.teacher(optical)
            features = stride_tensor_map(self.teacher, pop_captured_features(self.teacher))
            components = raw_obb_component_maps(self.teacher, predictions)
            batch["_teacher_payload"] = {
                "features": {stride: value.detach() for stride, value in features.items()},
                "class_logits": {stride: value.detach() for stride, value in components["class_logits"].items()},
                "dfl": {stride: value.detach() for stride, value in components["dfl"].items()},
                "angle": {stride: value.detach() for stride, value in components["angle"].items()},
            }
        warmup_epochs = float(self.distill_config["distill"]["optimization"]["distill_warmup_epochs"])
        batch["_distill_scale"] = min(1.0, (self.epoch + 1) / warmup_epochs) if warmup_epochs > 0 else 1.0
        batch["_distill_step"] = self._distill_step
        self._distill_step += 1
        return batch

    def get_validator(self):
        """Return the official OBB validator while exposing train-only PRCD metrics."""
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
            *(
                de_parallel(self.model).prcd_loss_names
                if self.distill_enabled and hasattr(de_parallel(self.model), "prcd_loss_names")
                else ()
            ),
        )
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def save_model(self):
        """Save standard checkpoints plus exact (non-EMA-smoothed) PRCD state and provenance."""
        super().save_model()
        if not self.distill_enabled:
            return
        model = de_parallel(self.model)
        prcd_state = {key: value.detach().cpu() for key, value in model.distiller.state_dict().items()}
        metadata = {
            "config": self.distill_config,
            "teacher_path": str(self.args.teacher),
            "teacher_hash": getattr(self.args, "teacher_hash", None),
            "dataset_profile_hash": getattr(self.args, "distill_profile_hash", None),
        }
        targets = [self.last]
        if self.best_fitness == self.fitness:
            targets.append(self.best)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            targets.append(self.wdir / f"epoch{self.epoch}.pt")
        for target in targets:
            checkpoint = torch.load(target, map_location="cpu")
            checkpoint["prcd_state"] = prcd_state
            checkpoint["prcd_metadata"] = metadata
            temporary = target.with_suffix(target.suffix + ".prcd.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(target)

    def resume_training(self, checkpoint):
        """Restore optimizer/EMA through Ultralytics and exact prototype/adapter state through PRCD metadata."""
        super().resume_training(checkpoint)
        if not self.resume or not self.distill_enabled or checkpoint is None:
            return
        metadata = checkpoint.get("prcd_metadata", {})
        expected_config = metadata.get("config")
        if expected_config and expected_config != self.distill_config:
            raise ValueError("resume distillation config differs from the saved PRCD run")
        expected_teacher = metadata.get("teacher_hash")
        current_teacher = getattr(self.args, "teacher_hash", None)
        if expected_teacher and current_teacher and expected_teacher != current_teacher:
            raise ValueError("resume teacher checkpoint hash differs from the saved PRCD run")
        expected_profile = metadata.get("dataset_profile_hash")
        current_profile = getattr(self.args, "distill_profile_hash", None)
        if expected_profile and current_profile and expected_profile != current_profile:
            raise ValueError("resume dataset profile hash differs from the saved PRCD run")
        if "prcd_state" not in checkpoint:
            LOGGER.warning("PRCD resume checkpoint has no exact prcd_state; using EMA-carried state")
            return
        de_parallel(self.model).distiller.load_state_dict(checkpoint["prcd_state"], strict=True)


def main() -> None:
    """Small CLI that accepts standard Ultralytics key=value overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Ultralytics overrides, e.g. model=... data=... teacher=...")
    namespace = parser.parse_args()
    overrides = {}
    for item in namespace.overrides:
        if "=" not in item:
            raise ValueError(f"expected key=value override, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    DistillOBBTrainer(overrides=overrides).train()


if __name__ == "__main__":
    main()


__all__ = ("DistillOBBTrainer", "file_sha256")
