"""Pure SAR export-state and teacher-gradient tests."""

import torch

from tools.export_sar_student import build_pure_student
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.obb.distill_model import (
    DistillOBBModel,
    feature_channels_by_stride,
    install_head_capture,
    pop_captured_features,
    raw_obb_component_maps,
)
from ultralytics.utils.distill_config import load_distill_config
from ultralytics.utils import DEFAULT_CFG


def test_pure_student_removes_distillation_state():
    model = DistillOBBModel("ultralytics/cfg/models/v8/yolov8-obb.yaml", nc=3, verbose=False)
    channels = feature_channels_by_stride(model)
    model.configure_distillation(load_distill_config(None), channels, channels)
    pure = build_pure_student(model)
    assert not any("distiller" in key or "prototype" in key for key in pure.state_dict())
    model.eval()
    pure.eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(model(image)[0], pure(image)[0])


def test_teacher_payload_has_no_gradient():
    config = load_distill_config(None)
    model = DistillOBBModel("ultralytics/cfg/models/v8/yolov8-obb.yaml", nc=3, verbose=False)
    channels = feature_channels_by_stride(model)
    model.configure_distillation(config, channels, channels)
    teacher_features = {8: torch.randn(1, channels[8], 8, 8, requires_grad=True)}
    teacher_logits = {8: torch.randn(1, 3, 8, 8, requires_grad=True)}
    student_features = {8: torch.randn(1, channels[8], 8, 8, requires_grad=True)}
    student_logits = {8: torch.randn(1, 3, 8, 8, requires_grad=True)}
    output = model.distiller(
        student_features,
        teacher_features,
        student_logits,
        teacher_logits,
        torch.tensor([[0.5, 0.5, 0.1, 0.1, 0.0]]),
        torch.tensor([0]),
        torch.tensor([0]),
    )
    output.total.backward()
    assert teacher_features[8].grad is None
    assert teacher_logits[8].grad is None


def test_raw_obb_components_are_exposed_by_actual_stride():
    model = DistillOBBModel("ultralytics/cfg/models/v8/yolov8-obb.yaml", nc=3, verbose=False).train()
    install_head_capture(model)
    predictions = model(torch.randn(2, 3, 64, 64))
    features = pop_captured_features(model)
    components = raw_obb_component_maps(model, predictions)
    strides = [int(value) for value in model.model[-1].stride.tolist()]
    assert list(components["class_logits"]) == strides
    assert len(features) == len(strides)
    for index, stride in enumerate(strides):
        assert components["dfl"][stride].shape[1] == 4 * model.model[-1].reg_max
        assert components["class_logits"][stride].shape[1] == 3
        assert components["angle"][stride].shape[1] == 1
        assert components["angle"][stride].shape[-2:] == features[index].shape[-2:]


def test_enabled_validation_without_teacher_payload_keeps_stable_loss_vector():
    model = DistillOBBModel("ultralytics/cfg/models/v8/yolov8-obb.yaml", nc=3, verbose=False)
    channels = feature_channels_by_stride(model)
    model.configure_distillation(load_distill_config(None), channels, channels)
    model.args = get_cfg(DEFAULT_CFG)
    model.eval()
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3]]),
    }
    with torch.no_grad():
        predictions = model(batch["img"])
        loss, items = model.loss(batch, predictions)
    assert torch.isfinite(loss)
    assert items.shape == (3 + len(model.prcd_loss_names),)
    assert torch.equal(items[3:], torch.zeros_like(items[3:]))
