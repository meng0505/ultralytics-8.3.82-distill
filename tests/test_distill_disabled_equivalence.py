"""Strict baseline-equivalence checks for disabled PRCD."""

from copy import deepcopy
from types import SimpleNamespace

import torch

from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.obb.distill_model import DistillOBBModel
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.distill_config import load_distill_config


def make_models():
    torch.manual_seed(0)
    baseline = OBBModel("ultralytics/cfg/models/v8/yolov8-obb.yaml", nc=3, verbose=False)
    candidate = DistillOBBModel("ultralytics/cfg/models/v8/yolov8-obb.yaml", nc=3, verbose=False)
    candidate.load_state_dict(deepcopy(baseline.state_dict()))
    candidate.configure_distillation(load_distill_config({"distill": {"enabled": False}}), {}, {})
    hyp = get_cfg(DEFAULT_CFG)
    baseline.args = hyp
    candidate.args = hyp
    return baseline, candidate


def test_disabled_inference_and_parameter_equivalence():
    baseline, candidate = make_models()
    image = torch.randn(1, 3, 64, 64)
    baseline.eval()
    candidate.eval()
    with torch.no_grad():
        expected = baseline(image)[0]
        actual = candidate(image)[0]
    assert torch.equal(expected, actual)
    assert [name for name, _ in baseline.named_parameters()] == [name for name, _ in candidate.named_parameters()]
    assert sum(parameter.numel() for parameter in baseline.parameters()) == sum(
        parameter.numel() for parameter in candidate.parameters()
    )


def test_disabled_detection_loss_equivalence():
    baseline, candidate = make_models()
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3]]),
    }
    baseline.train()
    candidate.train()
    expected_loss, expected_items = baseline(deepcopy(batch))
    actual_loss, actual_items = candidate(deepcopy(batch))
    assert torch.equal(expected_loss, actual_loss)
    assert torch.equal(expected_items, actual_items)


def test_disabled_optimizer_group_equivalence():
    baseline, candidate = make_models()
    stub = SimpleNamespace(args=SimpleNamespace(lr0=0.01, momentum=0.9), data={"nc": 3})
    expected = BaseTrainer.build_optimizer(stub, baseline, name="SGD", lr=0.01, momentum=0.9, decay=5e-4)
    actual = BaseTrainer.build_optimizer(stub, candidate, name="SGD", lr=0.01, momentum=0.9, decay=5e-4)
    expected_groups = [
        (group["weight_decay"], [parameter.numel() for parameter in group["params"]])
        for group in expected.param_groups
    ]
    actual_groups = [
        (group["weight_decay"], [parameter.numel() for parameter in group["params"]])
        for group in actual.param_groups
    ]
    assert expected_groups == actual_groups
