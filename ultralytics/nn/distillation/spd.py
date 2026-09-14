"""Semantic Prototype Distillation (SPD)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .prototype_bank import TeacherPrototypeBank


@dataclass
class SPDOutput:
    """SPD loss and class-level diagnostics."""

    loss: Tensor
    valid_groups: int
    teacher_variance: Tensor
    student_prototypes: dict[int, Tensor]


class SemanticPrototypeDistillation(nn.Module):
    """Align current student class prototypes to detached teacher EMA prototypes."""

    def __init__(self, eps: float = 1e-6, prototype_relation_enabled: bool = False):
        super().__init__()
        self.eps = float(eps)
        self.prototype_relation_enabled = bool(prototype_relation_enabled)

    def forward(
        self,
        student: Tensor,
        teacher: Tensor,
        class_ids: Tensor,
        level_index: int,
        bank: TeacherPrototypeBank,
    ) -> SPDOutput:
        """Update teacher memory and compute normalized cosine prototype loss."""
        zero = student.float().sum() * 0.0
        class_ids = class_ids.long().view(-1)
        bank.update(teacher, class_ids, level_index)
        losses, variances = [], []
        valid_groups = 0
        student_prototypes: dict[int, Tensor] = {}
        valid_class_ids = sorted(set(class_ids.detach().cpu().tolist()))
        teacher_group_prototypes, student_group_prototypes = [], []
        for class_id in valid_class_ids:
            if not 0 <= class_id < bank.num_classes:
                continue
            group = class_ids == class_id
            student_proto = F.normalize(student[group].float().mean(dim=0), dim=0, eps=self.eps)
            student_prototypes[class_id] = student_proto
            teacher_proto, initialized = bank.get(
                level_index, torch.tensor([class_id], device=student.device, dtype=torch.long)
            )
            if not bool(initialized.item()) or student_proto.norm() <= self.eps:
                continue
            teacher_proto = teacher_proto[0]
            losses.append(
                (1.0 - F.cosine_similarity(student_proto[None], teacher_proto[None], dim=-1)[0]).clamp_min(0)
            )
            valid_groups += 1
            teacher_batch_proto = F.normalize(teacher[group].float().mean(dim=0), dim=0, eps=self.eps)
            variances.append(
                (1.0 - F.cosine_similarity(teacher[group].float(), teacher_batch_proto[None], dim=-1))
                .clamp_min(0)
                .mean()
            )
            teacher_group_prototypes.append(teacher_proto)
            student_group_prototypes.append(student_proto)

        if self.prototype_relation_enabled and len(teacher_group_prototypes) >= 2:
            teacher_matrix = torch.stack(teacher_group_prototypes)
            student_matrix = torch.stack(student_group_prototypes)
            teacher_relation = teacher_matrix @ teacher_matrix.T
            student_relation = student_matrix @ student_matrix.T
            relation_mask = ~torch.eye(len(teacher_matrix), device=student.device, dtype=torch.bool)
            losses.append(F.mse_loss(student_relation[relation_mask], teacher_relation[relation_mask].detach()))
        loss = torch.stack(losses).mean() if losses else zero
        variance = torch.stack(variances).mean() if variances else zero.detach()
        return SPDOutput(loss, valid_groups, variance, student_prototypes)


__all__ = ("SPDOutput", "SemanticPrototypeDistillation")
