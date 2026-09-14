#!/usr/bin/env python3
"""Run official OBB validation plus native-size and P3-support stratified AP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.models.yolo.obb.val import OBBValidator  # noqa: E402
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.metrics import batch_probiou  # noqa: E402
from ultralytics.utils.stratified_obb_metrics import StratifiedOBBAP  # noqa: E402


class StratifiedOBBValidator(OBBValidator):
    """Collect range-aware stats before delegating unchanged official OBB metrics."""

    def __init__(self, *args, stratified_output: Path, group_parameters: dict, max_images: int | None = None, **kwargs):
        self.stratified_output = Path(stratified_output)
        self.group_parameters = group_parameters
        self.max_images = max_images
        self.stratified = None
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="val", batch=None):
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        if self.max_images is None or self.max_images >= len(dataset):
            return dataset

        class LimitedDataset(Subset):
            collate_fn = staticmethod(dataset.collate_fn)
            im_files = dataset.im_files[: self.max_images]

        return LimitedDataset(dataset, range(self.max_images))

    def init_metrics(self, model):
        super().init_metrics(model)
        self.stratified = StratifiedOBBAP(
            self.iouv.detach().cpu().numpy(), **self.group_parameters
        )

    def _native_batch_copy(self, si: int, batch: dict) -> dict:
        """Prepare native GT coordinates without mutating the official validator batch."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1).clone()
        bbox = batch["bboxes"][idx].clone()
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            pbatch = self._native_batch_copy(si, batch)
            gt_cls, gt_boxes = pbatch["cls"], pbatch["bbox"]
            predn = self._prepare_pred(pred, pbatch) if len(pred) else pred
            pred_boxes = (
                torch.cat((predn[:, :4], predn[:, -1:]), dim=-1)
                if len(predn)
                else gt_boxes.new_zeros((0, 5))
            )
            iou = (
                batch_probiou(gt_boxes, pred_boxes)
                if len(gt_boxes) and len(pred_boxes)
                else gt_boxes.new_zeros((len(gt_boxes), len(pred_boxes)))
            )
            self.stratified.add_image(
                pred_boxes.detach().cpu().numpy(),
                predn[:, 4].detach().cpu().numpy() if len(predn) else np.zeros(0),
                predn[:, 5].detach().cpu().numpy() if len(predn) else np.zeros(0),
                gt_boxes.detach().cpu().numpy(),
                gt_cls.detach().cpu().numpy(),
                iou.detach().cpu().numpy(),
            )
        super().update_metrics(preds, batch)

    def get_stats(self):
        official = super().get_stats()
        report = self.stratified.compute()
        report["official_obb_metrics"] = {key: float(value) for key, value in official.items()}
        report["partial_evaluation_max_images"] = self.max_images
        StratifiedOBBAP.save(
            report,
            self.stratified_output,
            self.stratified_output.with_suffix(".csv"),
        )
        return official


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tiny-max", type=float, default=8.0)
    parser.add_argument("--small-max", type=float, default=32.0)
    parser.add_argument("--medium-max", type=float, default=96.0)
    parser.add_argument("--support-stride", type=int, default=8)
    parser.add_argument(
        "--max-images",
        type=int,
        help="Smoke-test only: evaluate the first N records and mark the report partial.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validator = StratifiedOBBValidator(
        args={
            "model": args.model,
            "data": args.data,
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "plots": False,
            "task": "obb",
        },
        stratified_output=args.output,
        group_parameters={
            "tiny_max": args.tiny_max,
            "small_max": args.small_max,
            "medium_max": args.medium_max,
            "support_stride": args.support_stride,
        },
        max_images=args.max_images,
    )
    official = validator(model=args.model)
    print(json.dumps({"official": official, "stratified": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
