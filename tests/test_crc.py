"""CRC probability and numerical stability tests."""

import torch

from ultralytics.nn.distillation.crc import CrossModalResponseCalibration
from ultralytics.nn.distillation.object_support import ObjectSupportSampler


def test_crc_probability_normalization_and_gradient():
    sampler = ObjectSupportSampler(mode="fixed_5", output_grid=5)
    geometry = sampler.build(torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3]]), torch.tensor([0]), (16, 16))
    teacher = torch.randn(1, 1, 5, 5)
    student = torch.randn(1, 1, 5, 5, requires_grad=True)
    output = CrossModalResponseCalibration(temperature=1.0)(teacher, student, geometry)
    assert torch.allclose(output.teacher_distribution.sum(-1), torch.ones(1))
    assert torch.allclose(output.student_distribution.sum(-1), torch.ones(1))
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert torch.isfinite(student.grad).all()


def test_crc_empty_gt_is_differentiable_zero():
    sampler = ObjectSupportSampler(output_grid=7)
    geometry = sampler.build(torch.zeros(0, 5), torch.zeros(0, dtype=torch.long), (16, 16))
    student = torch.zeros(0, 1, 7, 7, requires_grad=True)
    output = CrossModalResponseCalibration()(student.detach(), student, geometry)
    assert output.loss.item() == 0
    output.loss.backward()
    assert student.grad is not None


def test_crc_three_regions_finite():
    sampler = ObjectSupportSampler(mode="fixed_7", output_grid=7)
    geometry = sampler.build(torch.tensor([[0.5, 0.5, 0.08, 0.08, 0.0]]), torch.tensor([0]), (32, 32))
    response = torch.rand(1, 1, 7, 7)
    output = CrossModalResponseCalibration(regions=3)(response, response.clone().requires_grad_(), geometry)
    assert torch.isfinite(output.loss)
