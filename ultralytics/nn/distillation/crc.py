"""Cross-Modal Response Calibration (CRC)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .object_support import SupportGeometry, masked_mean


@dataclass
class CRCOutput:
    """CRC loss, distributions and validity diagnostics."""

    loss: Tensor
    teacher_distribution: Tensor
    student_distribution: Tensor
    valid: Tensor


class CrossModalResponseCalibration(nn.Module):
    """Transfer core/context GT-class response distributions without pixel matching."""

    def __init__(self, temperature: float = 1.0, regions: int = 2, eps: float = 1e-6):
        super().__init__()
        if temperature <= 0:
            raise ValueError("CRC temperature must be positive")
        if regions not in {2, 3}:
            raise ValueError("CRC supports two or three regions")
        self.temperature = float(temperature)
        self.regions = int(regions)
        self.eps = float(eps)

    def _distribution(self, responses: Tensor, geometry: SupportGeometry) -> Tensor:
        masks = (
            (geometry.core_mask, geometry.context_mask)
            if self.regions == 2
            else (geometry.core_mask, geometry.near_context_mask, geometry.outer_context_mask)
        )
        energies = torch.stack([masked_mean(responses, mask, self.eps).squeeze(1) for mask in masks], dim=-1)
        return F.softmax(energies.float() / self.temperature, dim=-1)

    def forward(
        self, teacher_responses: Tensor, student_responses: Tensor, geometry: SupportGeometry
    ) -> CRCOutput:
        """Compute KL(teacher || student), normalized by valid instance count."""
        zero = student_responses.float().sum() * 0.0
        if not len(student_responses):
            empty = student_responses.new_zeros((0, self.regions), dtype=torch.float32)
            return CRCOutput(zero, empty, empty, geometry.valid)
        teacher_distribution = self._distribution(teacher_responses.detach(), geometry)
        student_distribution = self._distribution(student_responses, geometry)
        masks = (
            (geometry.core_mask, geometry.context_mask)
            if self.regions == 2
            else (geometry.core_mask, geometry.near_context_mask, geometry.outer_context_mask)
        )
        region_valid = torch.stack(
            [mask.float().sum(dim=(-3, -2, -1)) > self.eps for mask in masks], dim=-1
        ).all(dim=-1)
        valid = geometry.valid & region_valid
        if not valid.any():
            return CRCOutput(zero, teacher_distribution, student_distribution, valid)
        log_student = student_distribution[valid].clamp_min(self.eps).log()
        teacher = teacher_distribution[valid].clamp_min(self.eps)
        loss = F.kl_div(log_student, teacher, reduction="batchmean").clamp_min(0)
        return CRCOutput(loss, teacher_distribution, student_distribution, valid)


__all__ = ("CRCOutput", "CrossModalResponseCalibration")
