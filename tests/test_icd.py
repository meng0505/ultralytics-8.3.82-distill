"""ICD skip and deterministic sampling tests."""

import torch

from ultralytics.nn.distillation.icd import InstanceConfigurationDistillation
from ultralytics.nn.distillation.prototype_bank import TeacherPrototypeBank


def initialized_bank(dim=8):
    bank = TeacherPrototypeBank(1, 2, dim)
    bank.update(torch.randn(4, dim), torch.tensor([0, 0, 0, 0]), 0)
    return bank


def test_icd_single_instance_skips():
    student = torch.randn(1, 8, requires_grad=True)
    output = InstanceConfigurationDistillation()(student, torch.randn(1, 8), torch.tensor([0]), 0, initialized_bank())
    assert output.valid_groups == 0
    assert output.loss.item() == 0


def test_icd_sampling_is_deterministic():
    torch.manual_seed(5)
    teacher = torch.randn(20, 8)
    student = torch.randn(20, 8, requires_grad=True)
    classes = torch.zeros(20, dtype=torch.long)
    bank = initialized_bank()
    module = InstanceConfigurationDistillation(max_instances_per_class=8, seed=17)
    first = module(student, teacher, classes, 0, bank, step=4)
    second = module(student, teacher, classes, 0, bank, step=4)
    assert first.sampled_instances == second.sampled_instances == 8
    assert torch.allclose(first.loss, second.loss)


def test_pair_direction_is_explicit_ablation():
    teacher = torch.randn(4, 8)
    student = torch.randn(4, 8, requires_grad=True)
    output = InstanceConfigurationDistillation(pair_direction=True)(
        student, teacher, torch.zeros(4, dtype=torch.long), 0, initialized_bank()
    )
    assert output.valid_groups == 1
    assert torch.isfinite(output.loss)
