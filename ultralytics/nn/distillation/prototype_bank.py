"""Distributed teacher prototype memory for PRCD."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class PrototypeUpdate:
    """Diagnostics from one class-wise prototype update."""

    active_classes: Tensor
    batch_counts: Tensor


class TeacherPrototypeBank(nn.Module):
    """EMA teacher prototypes keyed by ``(feature_level, class_id)``."""

    def __init__(self, num_levels: int, num_classes: int, embedding_dim: int, momentum: float = 0.99, eps: float = 1e-6):
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError("prototype momentum must be in [0, 1)")
        self.num_levels = int(num_levels)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("prototypes", torch.zeros(num_levels, num_classes, embedding_dim, dtype=torch.float32))
        self.register_buffer("counts", torch.zeros(num_levels, num_classes, dtype=torch.long))

    @staticmethod
    def _distributed_sum(value: Tensor) -> Tensor:
        """All-reduce a detached tensor when DDP is initialized."""
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    @torch.no_grad()
    def update(self, representations: Tensor, class_ids: Tensor, level_index: int) -> PrototypeUpdate:
        """Synchronize class sums/counts and update normalized EMA prototypes."""
        if not 0 <= level_index < self.num_levels:
            raise IndexError(f"level index {level_index} outside prototype bank")
        representations = representations.detach().float()
        class_ids = class_ids.detach().long().view(-1)
        sums = torch.zeros(
            self.num_classes, self.embedding_dim, device=self.prototypes.device, dtype=torch.float32
        )
        batch_counts = torch.zeros(self.num_classes, device=self.prototypes.device, dtype=torch.float32)
        if len(representations):
            valid = (class_ids >= 0) & (class_ids < self.num_classes)
            sums.index_add_(0, class_ids[valid], representations[valid])
            batch_counts.index_add_(0, class_ids[valid], torch.ones_like(class_ids[valid], dtype=torch.float32))
        self._distributed_sum(sums)
        self._distributed_sum(batch_counts)
        active = batch_counts > 0
        if active.any():
            batch_prototypes = F.normalize(sums[active] / batch_counts[active, None], dim=-1, eps=self.eps)
            previous = self.prototypes[level_index, active]
            initialized = self.counts[level_index, active] > 0
            mixed = torch.where(
                initialized[:, None],
                self.momentum * previous + (1.0 - self.momentum) * batch_prototypes,
                batch_prototypes,
            )
            self.prototypes[level_index, active] = F.normalize(mixed, dim=-1, eps=self.eps)
            self.counts[level_index, active] += batch_counts[active].long()
        return PrototypeUpdate(active_classes=active.nonzero(as_tuple=False).flatten(), batch_counts=batch_counts)

    def get(self, level_index: int, class_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Fetch detached prototypes and an initialized mask."""
        class_ids = class_ids.long()
        valid_ids = (class_ids >= 0) & (class_ids < self.num_classes)
        safe_ids = class_ids.clamp(0, self.num_classes - 1)
        prototypes = self.prototypes[level_index, safe_ids].detach()
        initialized = valid_ids & (self.counts[level_index, safe_ids] > 0)
        initialized &= prototypes.norm(dim=-1) > self.eps
        return prototypes, initialized

    def coverage(self) -> Tensor:
        """Fraction of class-level slots initialized at least once."""
        return (self.counts > 0).float().mean()


__all__ = ("PrototypeUpdate", "TeacherPrototypeBank")
