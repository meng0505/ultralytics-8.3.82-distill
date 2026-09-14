"""Prototype-Referenced Cross-Modal Distillation orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .crc import CrossModalResponseCalibration
from .icd import InstanceConfigurationDistillation
from .object_support import ObjectSupportSampler, SupportGeometry, masked_mean
from .prototype_bank import TeacherPrototypeBank
from .spd import SemanticPrototypeDistillation


class FixedProjection(nn.Module):
    """Deterministic non-trainable teacher projection."""

    def __init__(self, input_dim: int, output_dim: int, seed: int):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        matrix = torch.randn(output_dim, input_dim, generator=generator, dtype=torch.float32)
        matrix = F.normalize(matrix, dim=-1)
        self.register_buffer("weight", matrix)

    def forward(self, value: Tensor) -> Tensor:
        """Project in float32 without trainable teacher parameters."""
        return F.linear(value.float(), self.weight)


def _group_count(channels: int) -> int:
    """Choose the largest conventional GroupNorm divisor."""
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ObjectRepresentation(nn.Module):
    """Fixed teacher space and learnable student adapter for one feature stride."""

    def __init__(
        self,
        teacher_channels: int,
        student_channels: int,
        embedding_dim: int,
        adapter: str,
        aggregation: str,
        seed: int,
    ):
        super().__init__()
        self.aggregation = aggregation
        self.adapter_type = adapter
        pooled_factor = 2 if aggregation == "concat" else 1
        self.teacher_projection = FixedProjection(teacher_channels * pooled_factor, embedding_dim, seed)
        if adapter == "conv1x1_groupnorm":
            if aggregation == "concat" and embedding_dim % 2:
                raise ValueError("concat conv adapter requires an even embedding_dim")
            out_channels = embedding_dim // 2 if aggregation == "concat" else embedding_dim
            self.student_patch_adapter = nn.Sequential(
                nn.Conv2d(student_channels, out_channels, 1, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(),
            )
            self.student_vector_adapter = None
        else:
            self.student_patch_adapter = None
            input_dim = student_channels * pooled_factor
            if adapter == "identity":
                if input_dim != embedding_dim:
                    raise ValueError(f"identity adapter requires {input_dim=} == {embedding_dim=}")
                self.student_vector_adapter = nn.Identity()
            elif adapter in {"linear", "conv1x1"}:
                self.student_vector_adapter = nn.Linear(input_dim, embedding_dim)
            elif adapter in {"linear_layernorm", "conv1x1_layernorm"}:
                self.student_vector_adapter = nn.Sequential(
                    nn.Linear(input_dim, embedding_dim), nn.LayerNorm(embedding_dim)
                )
            elif adapter in {"mlp", "two_layer_mlp"}:
                self.student_vector_adapter = nn.Sequential(
                    nn.Linear(input_dim, embedding_dim),
                    nn.GELU(),
                    nn.LayerNorm(embedding_dim),
                    nn.Linear(embedding_dim, embedding_dim),
                )
            else:
                raise ValueError(f"unsupported student adapter {adapter!r}")

    def _aggregate(
        self,
        core: Tensor,
        context: Tensor,
        *,
        teacher_weights: Tensor | None,
        student_weights: Tensor | None,
        modality: str,
    ) -> Tensor:
        """Aggregate core/context vectors according to the configured ablation."""
        if self.aggregation == "concat":
            return torch.cat((core, context), dim=-1)
        if self.aggregation == "simple_average":
            return (core + context) / 2
        if self.aggregation == "teacher_weighted":
            if teacher_weights is None:
                raise RuntimeError("teacher_weighted aggregation requires CRC distributions")
            weights = teacher_weights
        else:
            weights = student_weights if modality == "student" else teacher_weights
            if weights is None:
                raise RuntimeError("self-weighted aggregation requires CRC distributions")
        return weights[:, :1] * core + weights[:, 1:2] * context

    def forward(
        self,
        teacher_patches: Tensor,
        student_patches: Tensor,
        geometry: SupportGeometry,
        *,
        teacher_weights: Tensor | None = None,
        student_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return normalized teacher/student object representations."""
        teacher_core = masked_mean(teacher_patches, geometry.core_mask)
        teacher_context = masked_mean(teacher_patches, geometry.context_mask)
        teacher_vector = self._aggregate(
            teacher_core,
            teacher_context,
            teacher_weights=teacher_weights,
            student_weights=student_weights,
            modality="teacher",
        )
        teacher_embedding = self.teacher_projection(teacher_vector)

        if self.student_patch_adapter is not None:
            student_patches = self.student_patch_adapter(student_patches)
        student_core = masked_mean(student_patches, geometry.core_mask)
        student_context = masked_mean(student_patches, geometry.context_mask)
        student_vector = self._aggregate(
            student_core,
            student_context,
            teacher_weights=teacher_weights,
            student_weights=student_weights,
            modality="student",
        )
        if self.student_vector_adapter is not None:
            student_vector = self.student_vector_adapter(student_vector.float())
        return (
            F.normalize(teacher_embedding.detach().float(), dim=-1, eps=1e-6),
            F.normalize(student_vector.float(), dim=-1, eps=1e-6),
        )


@dataclass
class DistillOutput:
    """Weighted distillation losses and observability metrics."""

    total: Tensor
    crc: Tensor
    spd: Tensor
    icd: Tensor
    valid_crc_instances: int
    valid_spd_groups: int
    valid_icd_groups: int
    prototype_coverage: Tensor
    prototype_variance: Tensor
    icd_matrix_correlation: Tensor

    def loss_items(self) -> Tensor:
        """Return only train-objective losses for the main Ultralytics progress table."""
        return torch.stack((self.crc.detach(), self.spd.detach(), self.icd.detach()))

    def diagnostic_items(self) -> Tensor:
        """Return observability metrics for the separate PRCD diagnostics log."""
        device = self.total.device
        return torch.stack(
            (
                self.total.detach(),
                torch.tensor(float(self.valid_crc_instances), device=device),
                torch.tensor(float(self.valid_spd_groups), device=device),
                torch.tensor(float(self.valid_icd_groups), device=device),
                self.prototype_coverage.detach(),
                self.prototype_variance.detach(),
                self.icd_matrix_correlation.detach(),
            )
        )


class PrototypeReferencedDistiller(nn.Module):
    """Coordinate support sampling, CRC, SPD and ICD across common feature strides."""

    def __init__(
        self,
        config: dict,
        teacher_channels: dict[int, int],
        student_channels: dict[int, int],
        num_classes: int,
        seed: int = 0,
    ):
        super().__init__()
        self.config = config
        distill = config["distill"]
        self.objective = distill.get("objective", "prcd")
        requested = [int(stride) for stride in distill["strides"]]
        self.strides = [
            stride for stride in requested if stride in teacher_channels and stride in student_channels
        ]
        if distill["enabled"] and not self.strides:
            raise ValueError(
                f"teacher/student have no requested common stride; requested={requested}, "
                f"teacher={sorted(teacher_channels)}, student={sorted(student_channels)}"
            )
        self.level_assignment = distill["level_assignment"]
        self.response_source = distill["crc"]["response_source"]
        self.support = ObjectSupportSampler(**distill["support"])
        representation = distill["representation"]
        self.representations = nn.ModuleDict(
            {
                str(stride): ObjectRepresentation(
                    teacher_channels[stride],
                    student_channels[stride],
                    int(representation["embedding_dim"]),
                    representation["adapter"],
                    representation["aggregation"],
                    seed + stride,
                )
                for stride in self.strides
            }
        )
        self.direct_adapters = nn.ModuleDict(
            {
                str(stride): nn.Conv2d(student_channels[stride], teacher_channels[stride], 1, bias=False)
                for stride in self.strides
            }
            if self.objective == "direct_feature"
            else {}
        )
        self.baseline_weight = float(distill.get("baseline", {}).get("weight", 0.1))
        self.crc_enabled = bool(distill["crc"]["enabled"])
        self.spd_enabled = bool(distill["spd"]["enabled"])
        self.icd_enabled = bool(distill["icd"]["enabled"])
        self.crc = CrossModalResponseCalibration(
            temperature=distill["crc"]["temperature"], regions=distill["crc"]["regions"]
        )
        self.bank = TeacherPrototypeBank(
            len(self.strides),
            num_classes,
            int(representation["embedding_dim"]),
            momentum=distill["spd"]["prototype_momentum"],
        )
        self.spd = SemanticPrototypeDistillation(
            prototype_relation_enabled=distill["spd"].get("prototype_relation_enabled", False)
        )
        self.icd = InstanceConfigurationDistillation(
            min_instances=distill["icd"]["min_instances"],
            max_instances_per_class=distill["icd"]["max_instances_per_class"],
            exclude_diagonal=distill["icd"]["exclude_diagonal"],
            pair_direction=distill["icd"].get("pair_direction", False),
            seed=seed,
        )
        self.weights = {
            "crc": float(distill["crc"]["weight"]),
            "spd": float(distill["spd"]["weight"]),
            "icd": float(distill["icd"]["weight"]),
        }

    def _assign_levels(self, boxes: Tensor, student_features: dict[int, Tensor]) -> Tensor | None:
        """Assign each instance to the coarsest adequately supported selected stride."""
        if self.level_assignment != "scale_assigned" or len(self.strides) == 1:
            return None
        assigned = torch.full((len(boxes),), self.strides[0], device=boxes.device, dtype=torch.long)
        for stride in sorted(self.strides):
            height, width = student_features[stride].shape[-2:]
            mapped_width = boxes[:, 2] * width
            mapped_height = boxes[:, 3] * height
            adequate = (mapped_width * mapped_height >= 1.0) & (torch.minimum(mapped_width, mapped_height) >= 0.5)
            assigned[adequate] = stride
        return assigned

    @staticmethod
    def _gt_response(logits: Tensor, classes: Tensor, geometry: SupportGeometry, support: ObjectSupportSampler) -> Tensor:
        """Sample sigmoid GT-class logits for each object."""
        patches = support.sample(logits, geometry)
        indices = torch.arange(len(patches), device=patches.device)
        return patches[indices, classes.long(), :, :][:, None].sigmoid()

    @staticmethod
    def _weighted_mean(losses: list[tuple[Tensor, int]], zero: Tensor) -> Tensor:
        """Average per-level losses by effective object/group count."""
        total_weight = sum(weight for _, weight in losses)
        return sum(loss * weight for loss, weight in losses) / total_weight if total_weight else zero

    def _baseline_forward(
        self,
        student_features: dict[int, Tensor],
        teacher_features: dict[int, Tensor],
        student_class_logits: dict[int, Tensor],
        teacher_class_logits: dict[int, Tensor],
        boxes: Tensor,
        classes: Tensor,
        batch_indices: Tensor,
        warmup_scale: float,
    ) -> DistillOutput:
        """Compute controlled direct-feature, pixel-response, or hard-GT baselines."""
        first_feature = next(iter(student_features.values()))
        zero = first_feature.float().sum() * 0.0
        losses: list[tuple[Tensor, int]] = []
        valid_instances = 0
        if self.objective in {"direct_feature", "pixel_response"}:
            for stride in self.strides:
                if self.objective == "direct_feature":
                    student = self.direct_adapters[str(stride)](student_features[stride]).float()
                    teacher = teacher_features[stride].detach().float()
                    if teacher.shape[-2:] != student.shape[-2:]:
                        teacher = F.interpolate(teacher, student.shape[-2:], mode="bilinear", align_corners=False)
                    loss = F.mse_loss(F.normalize(student, dim=1), F.normalize(teacher, dim=1))
                else:
                    student = student_class_logits[stride].float().sigmoid()
                    teacher = teacher_class_logits[stride].detach().float().sigmoid()
                    if teacher.shape[-2:] != student.shape[-2:]:
                        teacher = F.interpolate(teacher, student.shape[-2:], mode="bilinear", align_corners=False)
                    loss = F.mse_loss(student, teacher)
                losses.append((loss, 1))
        else:
            valid_labels = torch.isfinite(boxes).all(dim=-1) & (boxes[:, 2:4] > 0).all(dim=-1)
            valid_labels &= (classes >= 0) & (classes < self.bank.num_classes)
            for stride in self.strides:
                indices = valid_labels.nonzero(as_tuple=False).flatten()
                if not len(indices):
                    continue
                student = student_features[stride]
                teacher = teacher_features[stride].detach()
                if teacher.shape[-2:] != student.shape[-2:]:
                    teacher = F.interpolate(teacher.float(), student.shape[-2:], mode="bilinear", align_corners=False)
                geometry = self.support.build(
                    boxes[indices],
                    batch_indices[indices],
                    student.shape[-2:],
                    instance_indices=indices,
                    all_boxes=boxes,
                    all_batch_indices=batch_indices,
                )
                valid = geometry.valid
                if not valid.any():
                    continue
                teacher_repr, student_repr = self.representations[str(stride)](
                    self.support.sample(teacher, geometry),
                    self.support.sample(student, geometry),
                    geometry,
                )
                loss = (1.0 - F.cosine_similarity(student_repr[valid], teacher_repr[valid], dim=-1)).mean()
                count = int(valid.sum())
                losses.append((loss, count))
                valid_instances += count
        raw = self._weighted_mean(losses, zero)
        total = float(warmup_scale) * self.baseline_weight * raw
        return DistillOutput(
            total=total,
            crc=raw,
            spd=zero,
            icd=zero,
            valid_crc_instances=valid_instances,
            valid_spd_groups=0,
            valid_icd_groups=0,
            prototype_coverage=self.bank.coverage(),
            prototype_variance=zero.detach(),
            icd_matrix_correlation=zero.detach(),
        )

    def forward(
        self,
        student_features: dict[int, Tensor],
        teacher_features: dict[int, Tensor],
        student_class_logits: dict[int, Tensor],
        teacher_class_logits: dict[int, Tensor],
        boxes: Tensor,
        classes: Tensor,
        batch_indices: Tensor,
        *,
        warmup_scale: float = 1.0,
        step: int = 0,
    ) -> DistillOutput:
        """Compute PRCD on selected common strides."""
        first_feature = next(iter(student_features.values()))
        zero = first_feature.float().sum() * 0.0
        boxes = boxes.to(first_feature.device).float().view(-1, 5)
        classes = classes.to(first_feature.device).long().view(-1)
        batch_indices = batch_indices.to(first_feature.device).long().view(-1)
        if self.objective != "prcd":
            return self._baseline_forward(
                student_features,
                teacher_features,
                student_class_logits,
                teacher_class_logits,
                boxes,
                classes,
                batch_indices,
                warmup_scale,
            )
        valid_labels = torch.isfinite(boxes).all(dim=-1) & (boxes[:, 2:4] > 0).all(dim=-1)
        valid_labels &= (classes >= 0) & (classes < self.bank.num_classes)
        assigned = self._assign_levels(boxes, student_features)

        crc_losses: list[tuple[Tensor, int]] = []
        spd_losses: list[tuple[Tensor, int]] = []
        icd_losses: list[tuple[Tensor, int]] = []
        valid_crc = valid_spd = valid_icd = 0
        variances, correlations = [], []
        for level_index, stride in enumerate(self.strides):
            selected = valid_labels.clone()
            if assigned is not None:
                selected &= assigned == stride
            indices = selected.nonzero(as_tuple=False).flatten()
            if not len(indices):
                continue
            student_feature = student_features[stride]
            teacher_feature = teacher_features[stride]
            if teacher_feature.shape[-2:] != student_feature.shape[-2:]:
                teacher_feature = F.interpolate(
                    teacher_feature.float(), size=student_feature.shape[-2:], mode="bilinear", align_corners=False
                )
            geometry = self.support.build(
                boxes[indices],
                batch_indices[indices],
                student_feature.shape[-2:],
                instance_indices=indices,
                all_boxes=boxes,
                all_batch_indices=batch_indices,
            )
            student_patches = self.support.sample(student_feature, geometry)
            teacher_patches = self.support.sample(teacher_feature.detach(), geometry)
            geometry_valid = geometry.valid
            if not geometry_valid.any():
                continue

            teacher_distribution = student_distribution = None
            if self.crc_enabled:
                teacher_response = self._gt_response(
                    teacher_class_logits[stride].detach(), classes[indices], geometry, self.support
                )
                student_response = self._gt_response(
                    student_class_logits[stride], classes[indices], geometry, self.support
                )
                crc_output = self.crc(teacher_response, student_response, geometry)
                count = int(crc_output.valid.sum())
                crc_losses.append((crc_output.loss, count))
                valid_crc += count
                teacher_distribution = crc_output.teacher_distribution
                student_distribution = crc_output.student_distribution

            teacher_repr, student_repr = self.representations[str(stride)](
                teacher_patches,
                student_patches,
                geometry,
                teacher_weights=teacher_distribution,
                student_weights=student_distribution,
            )
            teacher_repr = teacher_repr[geometry_valid]
            student_repr = student_repr[geometry_valid]
            level_classes = classes[indices][geometry_valid]
            if self.spd_enabled:
                spd_output = self.spd(student_repr, teacher_repr, level_classes, level_index, self.bank)
                spd_losses.append((spd_output.loss, spd_output.valid_groups))
                valid_spd += spd_output.valid_groups
                variances.append(spd_output.teacher_variance)
            else:
                self.bank.update(teacher_repr, level_classes, level_index)
            if self.icd_enabled:
                icd_output = self.icd(
                    student_repr, teacher_repr, level_classes, level_index, self.bank, step=step
                )
                icd_losses.append((icd_output.loss, icd_output.valid_groups))
                valid_icd += icd_output.valid_groups
                correlations.append(icd_output.matrix_correlation)

        crc_loss = self._weighted_mean(crc_losses, zero)
        spd_loss = self._weighted_mean(spd_losses, zero)
        icd_loss = self._weighted_mean(icd_losses, zero)
        total = float(warmup_scale) * (
            self.weights["crc"] * crc_loss + self.weights["spd"] * spd_loss + self.weights["icd"] * icd_loss
        )
        variance = torch.stack(variances).mean() if variances else zero.detach()
        correlation = torch.stack(correlations).mean() if correlations else zero.detach()
        return DistillOutput(
            total=total,
            crc=crc_loss,
            spd=spd_loss,
            icd=icd_loss,
            valid_crc_instances=valid_crc,
            valid_spd_groups=valid_spd,
            valid_icd_groups=valid_icd,
            prototype_coverage=self.bank.coverage(),
            prototype_variance=variance,
            icd_matrix_correlation=correlation,
        )


__all__ = ("DistillOutput", "FixedProjection", "ObjectRepresentation", "PrototypeReferencedDistiller")
