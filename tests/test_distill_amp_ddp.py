"""AMP finiteness and DDP prototype synchronization tests."""

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ultralytics.nn.distillation.distiller import PrototypeReferencedDistiller
from ultralytics.nn.distillation.prototype_bank import TeacherPrototypeBank
from ultralytics.utils.distill_config import load_distill_config


def test_amp_distillation_loss_is_finite():
    config = load_distill_config(None)
    module = PrototypeReferencedDistiller(config, {8: 16}, {8: 8}, num_classes=2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = module(
            {8: torch.randn(2, 8, 16, 16, requires_grad=True)},
            {8: torch.randn(2, 16, 16, 16)},
            {8: torch.randn(2, 2, 16, 16, requires_grad=True)},
            {8: torch.randn(2, 2, 16, 16)},
            torch.tensor([[0.5, 0.5, 0.02, 0.02, 0.0], [0.3, 0.4, 0.2, 0.1, 0.5]]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
        )
    assert output.total.dtype == torch.float32
    assert torch.isfinite(output.total)
    output.total.backward()


def _ddp_worker(rank: int, init_file: str, output_dir: str):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    bank = TeacherPrototypeBank(1, 2, 2, momentum=0.9)
    representation = torch.tensor([[1.0, 0.0]]) if rank == 0 else torch.tensor([[0.0, 1.0]])
    bank.update(representation, torch.tensor([0]), 0)
    torch.save(bank.state_dict(), Path(output_dir) / f"rank{rank}.pt")
    dist.destroy_process_group()


def test_ddp_prototype_sum_count_sync(tmp_path):
    init_file = tmp_path / "ddp_init"
    mp.spawn(_ddp_worker, args=(str(init_file), str(tmp_path)), nprocs=2, join=True)
    first = torch.load(tmp_path / "rank0.pt", map_location="cpu")
    second = torch.load(tmp_path / "rank1.pt", map_location="cpu")
    assert torch.equal(first["counts"], second["counts"])
    assert torch.allclose(first["prototypes"], second["prototypes"])
    assert first["counts"][0, 0] == 2
