"""PRCD configuration selection and risk-gate tests."""

from unittest.mock import patch

import pytest

from ultralytics.cfg import entrypoint
from ultralytics.utils.distill_config import load_distill_config


def test_dataset_selection_keeps_matching_evidence_only():
    config = load_distill_config("recommended_distill_config.yaml", dataset_name="ogsod-1.0")
    assert config["_selected_dataset"] == "ogsod-1.0"
    assert config["_evidence"]["profile_dataset"] == "ogsod-1.0"
    assert "_per_dataset" not in config


def test_response_weighted_pool_requires_crc():
    with pytest.raises(ValueError, match="requires CRC"):
        load_distill_config(
            {
                "distill": {
                    "representation": {"aggregation": "teacher_weighted"},
                    "crc": {"enabled": False},
                }
            }
        )


def test_yolo_obb_distill_train_cli_routes_to_dedicated_trainer():
    with patch("ultralytics.models.yolo.obb.distill_train.DistillOBBTrainer") as trainer:
        entrypoint(
            debug="yolo obb distill-train model=fake.pt data=fake.yaml "
            "distill=configs/prcd/ablations/baseline_sar.yaml device=cpu"
        )
    overrides = trainer.call_args.kwargs["overrides"]
    assert overrides["task"] == "obb"
    assert overrides["distill"].endswith("baseline_sar.yaml")
    trainer.return_value.train.assert_called_once_with()
