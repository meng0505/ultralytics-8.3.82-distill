"""Continuous object-support sampling for cross-modal OBB distillation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class SupportGeometry:
    """Sampling geometry shared by teacher and student at one feature stride."""

    grid: Tensor
    core_mask: Tensor
    context_mask: Tensor
    near_context_mask: Tensor
    outer_context_mask: Tensor
    batch_indices: Tensor
    instance_indices: Tensor
    radii: Tensor
    valid: Tensor
    feature_shape: tuple[int, int]

    def to(self, device: torch.device | str) -> "SupportGeometry":
        """Move tensor fields to a device."""
        return SupportGeometry(
            grid=self.grid.to(device),
            core_mask=self.core_mask.to(device),
            context_mask=self.context_mask.to(device),
            near_context_mask=self.near_context_mask.to(device),
            outer_context_mask=self.outer_context_mask.to(device),
            batch_indices=self.batch_indices.to(device),
            instance_indices=self.instance_indices.to(device),
            radii=self.radii.to(device),
            valid=self.valid.to(device),
            feature_shape=self.feature_shape,
        )


def normalized_xywhr_to_feature(boxes: Tensor, height: int, width: int) -> Tensor:
    """Map normalized continuous xywhr boxes to feature coordinates without rounding centers."""
    scale = boxes.new_tensor((width, height, width, height, 1.0))
    return boxes * scale


def masked_mean(values: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Mask-normalized spatial mean in float32."""
    values32, mask32 = values.float(), mask.float()
    numerator = (values32 * mask32).sum(dim=(-2, -1))
    denominator = mask32.sum(dim=(-2, -1)).clamp_min(eps)
    return numerator / denominator


class ObjectSupportSampler(nn.Module):
    """Build fixed/adaptive rotated object supports and sample continuous feature patches."""

    def __init__(
        self,
        mode: str = "adaptive",
        output_grid: int = 7,
        fixed_window: int = 5,
        context_margin: int = 1,
        radius_min: int = 1,
        radius_max: int = 4,
        exclude_neighbor_objects: bool = True,
        core_supersample: int = 4,
        min_core_mass: float = 1e-3,
    ):
        super().__init__()
        aliases = {"fixed_3": ("fixed", 3), "fixed_5": ("fixed", 5), "fixed_7": ("fixed", 7)}
        if mode in aliases:
            mode, fixed_window = aliases[mode]
        if mode not in {"hard_gt_roi", "fixed", "adaptive"}:
            raise ValueError(f"unsupported support mode {mode!r}")
        if output_grid < 3 or output_grid % 2 == 0:
            raise ValueError("output_grid must be an odd integer >= 3")
        if fixed_window not in {3, 5, 7}:
            raise ValueError("fixed_window must be one of 3, 5, 7")
        if not 1 <= radius_min <= radius_max:
            raise ValueError("support radii must satisfy 1 <= radius_min <= radius_max")
        if core_supersample < 1:
            raise ValueError("core_supersample must be positive")
        self.mode = mode
        self.output_grid = int(output_grid)
        self.fixed_window = int(fixed_window)
        self.context_margin = int(context_margin)
        self.radius_min = int(radius_min)
        self.radius_max = int(radius_max)
        self.exclude_neighbor_objects = bool(exclude_neighbor_objects)
        self.core_supersample = int(core_supersample)
        self.min_core_mass = float(min_core_mass)

    def _radii(self, boxes_feature: Tensor) -> Tensor:
        """Compute support half extents in feature cells."""
        if self.mode == "fixed":
            return torch.full(
                (len(boxes_feature),),
                (self.fixed_window - 1) / 2,
                device=boxes_feature.device,
                dtype=torch.float32,
            )
        natural = boxes_feature[:, 2:4].amax(dim=1) / 2
        if self.mode == "hard_gt_roi":
            return natural.clamp_min(self.radius_min).float()
        margin = 0 if self.mode == "hard_gt_roi" else self.context_margin
        return (natural.ceil() + margin).clamp(self.radius_min, self.radius_max).float()

    @staticmethod
    def _rotated_occupancy(points_x: Tensor, points_y: Tensor, boxes: Tensor, expansion: float = 1.0) -> Tensor:
        """Evaluate hard rotated-box occupancy at arbitrary broadcastable point coordinates."""
        dx = points_x - boxes[:, None, None, 0]
        dy = points_y - boxes[:, None, None, 1]
        cos, sin = boxes[:, 4].cos()[:, None, None], boxes[:, 4].sin()[:, None, None]
        local_x = cos * dx + sin * dy
        local_y = -sin * dx + cos * dy
        half_width = boxes[:, None, None, 2] * (0.5 * expansion)
        half_height = boxes[:, None, None, 3] * (0.5 * expansion)
        return ((local_x.abs() <= half_width) & (local_y.abs() <= half_height)).float()

    def _soft_occupancy(
        self, sample_x: Tensor, sample_y: Tensor, boxes: Tensor, spacing: Tensor, expansion: float = 1.0
    ) -> Tensor:
        """Antialias rotated boxes by uniform sub-cell supersampling."""
        samples = self.core_supersample
        if samples == 1:
            return self._rotated_occupancy(sample_x, sample_y, boxes, expansion)
        offsets = (torch.arange(samples, device=boxes.device, dtype=torch.float32) + 0.5) / samples - 0.5
        occupancy = torch.zeros_like(sample_x, dtype=torch.float32)
        for offset_y in offsets:
            for offset_x in offsets:
                occupancy += self._rotated_occupancy(
                    sample_x + offset_x * spacing[:, None, None],
                    sample_y + offset_y * spacing[:, None, None],
                    boxes,
                    expansion,
                )
        return occupancy / float(samples * samples)

    def build(
        self,
        boxes: Tensor,
        batch_indices: Tensor,
        feature_shape: tuple[int, int],
        *,
        instance_indices: Tensor | None = None,
        all_boxes: Tensor | None = None,
        all_batch_indices: Tensor | None = None,
    ) -> SupportGeometry:
        """Construct a standard KxK support for selected normalized xywhr instances."""
        height, width = int(feature_shape[0]), int(feature_shape[1])
        device = boxes.device
        instance_indices = (
            torch.arange(len(boxes), device=device, dtype=torch.long)
            if instance_indices is None
            else instance_indices.to(device=device, dtype=torch.long)
        )
        batch_indices = batch_indices.to(device=device, dtype=torch.long).view(-1)
        if len(boxes) == 0:
            empty_grid = boxes.new_zeros((0, self.output_grid, self.output_grid, 2), dtype=torch.float32)
            empty_mask = boxes.new_zeros((0, 1, self.output_grid, self.output_grid), dtype=torch.float32)
            return SupportGeometry(
                empty_grid,
                empty_mask,
                empty_mask,
                empty_mask,
                empty_mask,
                batch_indices,
                instance_indices,
                boxes.new_zeros(0, dtype=torch.float32),
                torch.zeros(0, device=device, dtype=torch.bool),
                (height, width),
            )

        boxes_feature = normalized_xywhr_to_feature(boxes.float(), height, width)
        radii = self._radii(boxes_feature)
        base = torch.linspace(-1.0, 1.0, self.output_grid, device=device, dtype=torch.float32)
        offset_y, offset_x = torch.meshgrid(base, base, indexing="ij")
        sample_x = boxes_feature[:, None, None, 0] + offset_x * radii[:, None, None]
        sample_y = boxes_feature[:, None, None, 1] + offset_y * radii[:, None, None]
        # align_corners=False maps feature coordinate x to 2*x/W-1.
        grid_x = sample_x.mul(2.0 / width).sub(1.0)
        grid_y = sample_y.mul(2.0 / height).sub(1.0)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        spacing = (2 * radii / max(self.output_grid - 1, 1)).clamp_min(1e-6)

        core = self._soft_occupancy(sample_x, sample_y, boxes_feature, spacing)
        expanded = self._soft_occupancy(sample_x, sample_y, boxes_feature, spacing, expansion=2.0)
        in_bounds = ((sample_x >= 0) & (sample_x <= width) & (sample_y >= 0) & (sample_y <= height)).float()

        tiny = core.sum(dim=(-2, -1)) < self.min_core_mass
        if tiny.any():
            center = self.output_grid // 2
            core[tiny, center, center] = 1.0

        neighbor = torch.zeros_like(core)
        if self.exclude_neighbor_objects and all_boxes is not None and len(all_boxes):
            all_boxes_feature = normalized_xywhr_to_feature(all_boxes.float(), height, width)
            all_batch_indices = all_batch_indices.to(device=device, dtype=torch.long).view(-1)
            for local_index, source_index in enumerate(instance_indices):
                candidates = (all_batch_indices == batch_indices[local_index]) & (
                    torch.arange(len(all_boxes_feature), device=device) != source_index
                )
                if candidates.any():
                    sx = sample_x[local_index : local_index + 1].expand(int(candidates.sum()), -1, -1)
                    sy = sample_y[local_index : local_index + 1].expand(int(candidates.sum()), -1, -1)
                    other = self._soft_occupancy(
                        sx,
                        sy,
                        all_boxes_feature[candidates],
                        spacing[local_index : local_index + 1].expand(int(candidates.sum())),
                    )
                    neighbor[local_index] = other.amax(dim=0)

        context = in_bounds * (1.0 - core).clamp_min(0) * (1.0 - neighbor).clamp_min(0)
        near_context = context * (expanded - core).clamp(0, 1)
        outer_context = (context - near_context).clamp_min(0)
        valid = (core.sum(dim=(-2, -1)) >= self.min_core_mass) & (context.sum(dim=(-2, -1)) > 0)
        return SupportGeometry(
            grid=grid,
            core_mask=core[:, None],
            context_mask=context[:, None],
            near_context_mask=near_context[:, None],
            outer_context_mask=outer_context[:, None],
            batch_indices=batch_indices,
            instance_indices=instance_indices,
            radii=radii,
            valid=valid,
            feature_shape=(height, width),
        )

    @staticmethod
    def sample(feature: Tensor, geometry: SupportGeometry) -> Tensor:
        """Bilinearly sample per-instance feature maps with continuous centers."""
        if len(geometry.batch_indices) == 0:
            return feature.new_zeros((0, feature.shape[1], geometry.grid.shape[1], geometry.grid.shape[2]))
        selected = feature[geometry.batch_indices]
        return F.grid_sample(
            selected,
            geometry.grid.to(device=feature.device, dtype=torch.float32),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


__all__ = ("ObjectSupportSampler", "SupportGeometry", "masked_mean", "normalized_xywhr_to_feature")
