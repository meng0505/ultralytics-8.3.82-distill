"""Prototype bank and SPD state tests."""

import torch

from ultralytics.nn.distillation.prototype_bank import TeacherPrototypeBank
from ultralytics.nn.distillation.spd import SemanticPrototypeDistillation


def test_spd_ema_update_and_absent_class():
    bank = TeacherPrototypeBank(1, 3, 4, momentum=0.9)
    spd = SemanticPrototypeDistillation()
    teacher = torch.tensor([[1.0, 0, 0, 0], [0.8, 0.2, 0, 0], [0, 1.0, 0, 0]])
    student = teacher.clone().requires_grad_()
    output = spd(student, teacher, torch.tensor([0, 0, 1]), 0, bank)
    assert output.valid_groups == 2
    assert bank.counts[0].tolist() == [2, 1, 0]
    assert bank.prototypes[0, 2].count_nonzero() == 0
    output.loss.backward()
    assert student.grad is not None


def test_prototype_resume_state_restoration():
    bank = TeacherPrototypeBank(2, 3, 4)
    bank.update(torch.randn(3, 4), torch.tensor([0, 1, 1]), 1)
    restored = TeacherPrototypeBank(2, 3, 4)
    restored.load_state_dict(bank.state_dict())
    assert torch.equal(restored.counts, bank.counts)
    assert torch.allclose(restored.prototypes, bank.prototypes)
