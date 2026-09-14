"""Instance Configuration Distillation (ICD)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .prototype_bank import TeacherPrototypeBank


@dataclass
class ICDOutput:
    """ICD loss and group diagnostics."""

    loss: Tensor
    valid_groups: int
    sampled_instances: int
    matrix_correlation: Tensor


class InstanceConfigurationDistillation(nn.Module):
    """Compare class-internal prototype-centered cosine configuration matrices."""

    def __init__(
        self,
        min_instances: int = 2,
        max_instances_per_class: int = 32,
        exclude_diagonal: bool = True,
        pair_direction: bool = False,
        seed: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if min_instances < 2:
            raise ValueError("ICD min_instances must be at least two")
        if max_instances_per_class < min_instances:
            raise ValueError("ICD max_instances_per_class must be >= min_instances")
        self.min_instances = int(min_instances)
        self.max_instances_per_class = int(max_instances_per_class)
        self.exclude_diagonal = bool(exclude_diagonal)
        self.pair_direction = bool(pair_direction)
        self.seed = int(seed)
        self.eps = float(eps)

    def _sample(self, indices: Tensor, class_id: int, level_index: int, step: int) -> Tensor:
        """Deterministically cap one class group using a CPU generator."""
        if len(indices) <= self.max_instances_per_class:
            return indices
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 1_000_003 * level_index + 10_007 * class_id + int(step))
        order = torch.randperm(len(indices), generator=generator)[: self.max_instances_per_class]
        return indices[order.to(indices.device)]

    def forward(
        self,
        student: Tensor,
        teacher: Tensor,
        class_ids: Tensor,
        level_index: int,
        bank: TeacherPrototypeBank,
        *,
        step: int = 0,
    ) -> ICDOutput:
        """Compute per-class Gram MSE after modality-specific prototype centering."""
        zero = student.float().sum() * 0.0
        class_ids = class_ids.long().view(-1)
        losses, correlations = [], []
        sampled_instances = 0
        for class_id in sorted(set(class_ids.detach().cpu().tolist())):
            indices = (class_ids == class_id).nonzero(as_tuple=False).flatten()
            if len(indices) < self.min_instances or not 0 <= class_id < bank.num_classes:
                continue
            indices = self._sample(indices, class_id, level_index, step)
            teacher_proto, initialized = bank.get(
                level_index, torch.tensor([class_id], device=student.device, dtype=torch.long)
            )
            if not bool(initialized.item()):
                continue
            student_proto = F.normalize(student[indices].float().mean(dim=0), dim=0, eps=self.eps)
            teacher_residual = teacher[indices].float() - teacher_proto[0]
            student_residual = student[indices].float() - student_proto
            teacher_norm = teacher_residual.norm(dim=-1)
            student_norm = student_residual.norm(dim=-1)
            residual_valid = (teacher_norm > self.eps) & (student_norm > self.eps)
            if int(residual_valid.sum()) < self.min_instances:
                continue
            teacher_residual = F.normalize(teacher_residual[residual_valid], dim=-1, eps=self.eps)
            student_residual = F.normalize(student_residual[residual_valid], dim=-1, eps=self.eps)
            teacher_matrix = teacher_residual @ teacher_residual.T
            student_matrix = student_residual @ student_residual.T
            if self.exclude_diagonal:
                matrix_mask = ~torch.eye(len(teacher_matrix), device=student.device, dtype=torch.bool)
            else:
                matrix_mask = torch.ones_like(teacher_matrix, dtype=torch.bool)
            teacher_values = teacher_matrix[matrix_mask].detach()
            student_values = student_matrix[matrix_mask]
            if not len(teacher_values):
                continue
            group_losses = [F.mse_loss(student_values, teacher_values)]
            if self.pair_direction:
                group_losses.append(
                    (1.0 - F.cosine_similarity(student_residual, teacher_residual.detach(), dim=-1)).mean()
                )
            losses.append(torch.stack(group_losses).mean())
            if len(teacher_values) > 1 and teacher_values.std() > self.eps and student_values.std() > self.eps:
                centered_teacher = teacher_values - teacher_values.mean()
                centered_student = student_values - student_values.mean()
                correlations.append(
                    (centered_teacher * centered_student).mean()
                    / (centered_teacher.std(unbiased=False) * centered_student.std(unbiased=False) + self.eps)
                )
            sampled_instances += len(indices)
        loss = torch.stack(losses).mean() if losses else zero
        correlation = torch.stack(correlations).mean() if correlations else zero.detach()
        return ICDOutput(loss, len(losses), sampled_instances, correlation)


__all__ = ("ICDOutput", "InstanceConfigurationDistillation")
