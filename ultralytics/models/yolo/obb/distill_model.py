"""Distillation-only OBB model wrapper preserving the detector inference path."""

from __future__ import annotations

import weakref

import torch
from torch import Tensor

from ultralytics.nn.distillation import PrototypeReferencedDistiller
from ultralytics.nn.tasks import OBBModel

_HEAD_FEATURES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _capture_head_input(module, args) -> None:
    """Capture immutable tensor references before Detect rewrites its input list."""
    _HEAD_FEATURES[module] = tuple(args[0])


def install_head_capture(model: OBBModel) -> None:
    """Install one forward-pre-hook on the final OBB head."""
    head = model.model[-1]
    if not getattr(head, "_prcd_capture_installed", False):
        head.register_forward_pre_hook(_capture_head_input)
        head._prcd_capture_installed = True


def pop_captured_features(model: OBBModel) -> tuple[Tensor, ...]:
    """Consume the most recent captured head-input tuple."""
    head = model.model[-1]
    features = _HEAD_FEATURES.pop(head, None)
    if features is None:
        raise RuntimeError("OBB head feature capture did not run")
    return features


def feature_channels_by_stride(model: OBBModel) -> dict[int, int]:
    """Read final-head input channels without hard-coding YAML layer indices."""
    head = model.model[-1]
    return {
        int(stride): int(head.cv2[index][0].conv.in_channels)
        for index, stride in enumerate(head.stride.detach().cpu().tolist())
    }


def parse_raw_obb_outputs(predictions) -> tuple[list[Tensor], Tensor]:
    """Extract raw maps and angle output from train or eval OBB return structures."""
    if (
        isinstance(predictions, (tuple, list))
        and len(predictions) == 2
        and isinstance(predictions[0], list)
    ):
        return predictions[0], predictions[1]
    if (
        isinstance(predictions, (tuple, list))
        and len(predictions) == 2
        and isinstance(predictions[1], (tuple, list))
        and len(predictions[1]) == 2
        and isinstance(predictions[1][0], list)
    ):
        return predictions[1][0], predictions[1][1]
    raise TypeError("unrecognized OBB head output structure")


def stride_tensor_map(model: OBBModel, values: tuple[Tensor, ...] | list[Tensor]) -> dict[int, Tensor]:
    """Map a head-ordered tensor collection by its actual numeric stride."""
    strides = [int(value) for value in model.model[-1].stride.detach().cpu().tolist()]
    if len(strides) != len(values):
        raise ValueError(f"head stride/tensor count mismatch: {len(strides)} != {len(values)}")
    return dict(zip(strides, values))


def class_logit_map(model: OBBModel, predictions) -> dict[int, Tensor]:
    """Extract raw class logits at every actual OBB head stride."""
    return raw_obb_component_maps(model, predictions)["class_logits"]


def raw_obb_component_maps(model: OBBModel, predictions) -> dict[str, dict[int, Tensor]]:
    """Expose DFL regression, class logits and angle outputs by actual stride."""
    raw_maps, angle = parse_raw_obb_outputs(predictions)
    head = model.model[-1]
    offset = int(head.reg_max * 4)
    angle_maps = []
    start = 0
    for raw in raw_maps:
        height, width = raw.shape[-2:]
        count = height * width
        angle_maps.append(angle[:, :, start : start + count].view(angle.shape[0], angle.shape[1], height, width))
        start += count
    return {
        "raw": stride_tensor_map(model, raw_maps),
        "dfl": stride_tensor_map(model, [raw[:, :offset] for raw in raw_maps]),
        "class_logits": stride_tensor_map(model, [raw[:, offset : offset + head.nc] for raw in raw_maps]),
        "angle": stride_tensor_map(model, angle_maps),
    }


class DistillOBBModel(OBBModel):
    """OBB detector whose training loss optionally includes PRCD."""

    def __init__(self, cfg="yolo11n-obb.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.distiller: PrototypeReferencedDistiller | None = None
        self.prcd_enabled = False
        self.prcd_config = None
        self.latest_prcd_diagnostics: Tensor | None = None

    def configure_distillation(
        self,
        config: dict,
        teacher_channels: dict[int, int],
        student_channels: dict[int, int],
        seed: int = 0,
    ) -> None:
        """Attach trainable student adapters and prototype buffers."""
        self.prcd_config = config
        self.prcd_enabled = bool(config["distill"]["enabled"])
        if self.prcd_enabled:
            self.distiller = PrototypeReferencedDistiller(
                config, teacher_channels, student_channels, self.model[-1].nc, seed=seed
            )
            install_head_capture(self)

    @property
    def prcd_loss_names(self) -> tuple[str, ...]:
        """Logging item names appended to official OBB losses."""
        return (
            "crc_loss",
            "spd_loss",
            "icd_loss",
        )

    @property
    def prcd_diagnostic_names(self) -> tuple[str, ...]:
        """Names written to the separate PRCD diagnostics CSV."""
        return (
            "distill_total",
            "valid_crc_instances",
            "valid_spd_groups",
            "valid_icd_groups",
            "prototype_coverage",
            "prototype_variance",
            "icd_matrix_correlation",
        )

    def loss(self, batch, preds=None):
        """Compute unchanged detection loss plus optional normalized PRCD loss."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        predictions = self.forward(batch["img"]) if preds is None else preds
        detection_loss, detection_items = self.criterion(predictions, batch)
        if not self.prcd_enabled:
            return detection_loss, detection_items

        zero_items = detection_items.new_zeros(len(self.prcd_loss_names))
        teacher_payload = batch.get("_teacher_payload")
        if teacher_payload is None or self.distiller is None:
            # Validation remains SAR-only but keeps a stable loss vector for BaseValidator.
            pop_captured_features(self)
            return detection_loss, torch.cat((detection_items, zero_items))

        student_features = stride_tensor_map(self, pop_captured_features(self))
        output = self.distiller(
            student_features=student_features,
            teacher_features=teacher_payload["features"],
            student_class_logits=class_logit_map(self, predictions),
            teacher_class_logits=teacher_payload["class_logits"],
            boxes=batch["bboxes"],
            classes=batch["cls"],
            batch_indices=batch["batch_idx"],
            warmup_scale=float(batch.get("_distill_scale", 1.0)),
            step=int(batch.get("_distill_step", 0)),
        )
        # Ultralytics' detection scalar is batch-size scaled while its displayed
        # items are per-batch normalized. Apply the same scalar convention.
        total_loss = detection_loss + output.total * batch["img"].shape[0]
        self.latest_prcd_diagnostics = output.diagnostic_items()
        return total_loss, torch.cat((detection_items, output.loss_items()))


__all__ = (
    "DistillOBBModel",
    "class_logit_map",
    "feature_channels_by_stride",
    "install_head_capture",
    "parse_raw_obb_outputs",
    "pop_captured_features",
    "raw_obb_component_maps",
    "stride_tensor_map",
)
