"""Object support coordinate, mask and context tests."""

import torch

from ultralytics.nn.distillation.object_support import ObjectSupportSampler, normalized_xywhr_to_feature


def test_normalized_xywhr_keeps_continuous_center():
    box = torch.tensor([[0.333, 0.257, 0.1, 0.2, 0.5]])
    mapped = normalized_xywhr_to_feature(box, height=16, width=32)
    assert torch.allclose(mapped[0, :4], torch.tensor([10.656, 4.112, 3.2, 3.2]), atol=1e-6)


def test_fixed_and_adaptive_support_shapes():
    boxes = torch.tensor([[0.333, 0.257, 0.01, 0.02, 0.3], [0.7, 0.7, 0.4, 0.2, 1.0]])
    batch = torch.zeros(2, dtype=torch.long)
    fixed = ObjectSupportSampler(mode="fixed_5", output_grid=5)
    adaptive = ObjectSupportSampler(mode="adaptive", output_grid=7, radius_min=1, radius_max=4)
    fixed_geometry = fixed.build(boxes, batch, (16, 16))
    adaptive_geometry = adaptive.build(boxes, batch, (16, 16))
    assert fixed_geometry.grid.shape == (2, 5, 5, 2)
    assert adaptive_geometry.grid.shape == (2, 7, 7, 2)
    assert torch.allclose(fixed_geometry.radii, torch.tensor([2.0, 2.0]))
    assert adaptive_geometry.radii[1] > adaptive_geometry.radii[0]
    # Continuous center is the central sampling coordinate, not a rounded cell.
    assert torch.allclose(fixed_geometry.grid[0, 2, 2], torch.tensor([-0.334, -0.486]), atol=1e-6)


def test_tiny_obb_has_nonzero_soft_core_and_finite_sample():
    boxes = torch.tensor([[0.5, 0.5, 1e-5, 1e-5, 0.8]])
    sampler = ObjectSupportSampler(mode="adaptive", output_grid=7, core_supersample=4)
    geometry = sampler.build(boxes, torch.tensor([0]), (32, 32))
    patch = sampler.sample(torch.randn(1, 4, 32, 32), geometry)
    assert geometry.core_mask.sum() >= 1
    assert torch.isfinite(patch).all()


def test_neighbor_object_is_excluded_from_context():
    boxes = torch.tensor([[0.5, 0.5, 0.08, 0.08, 0.0], [0.58, 0.5, 0.08, 0.08, 0.0]])
    batch = torch.tensor([0, 0])
    with_exclusion = ObjectSupportSampler(
        mode="fixed_7", output_grid=7, exclude_neighbor_objects=True
    ).build(boxes[:1], batch[:1], (32, 32), instance_indices=torch.tensor([0]), all_boxes=boxes, all_batch_indices=batch)
    without_exclusion = ObjectSupportSampler(
        mode="fixed_7", output_grid=7, exclude_neighbor_objects=False
    ).build(boxes[:1], batch[:1], (32, 32), instance_indices=torch.tensor([0]), all_boxes=boxes, all_batch_indices=batch)
    assert with_exclusion.context_mask.sum() < without_exclusion.context_mask.sum()


def test_empty_gt_returns_empty_geometry():
    sampler = ObjectSupportSampler()
    geometry = sampler.build(torch.zeros(0, 5), torch.zeros(0, dtype=torch.long), (16, 16))
    assert geometry.grid.shape[0] == 0
    assert geometry.core_mask.shape == (0, 1, 7, 7)


def test_dense_obb_context_remains_finite():
    centers = torch.linspace(0.35, 0.65, 16)
    boxes = torch.stack(
        (centers, centers.flip(0), torch.full_like(centers, 0.04), torch.full_like(centers, 0.03), centers), dim=1
    )
    batch = torch.zeros(len(boxes), dtype=torch.long)
    sampler = ObjectSupportSampler(mode="adaptive", exclude_neighbor_objects=True)
    geometry = sampler.build(
        boxes,
        batch,
        (32, 32),
        instance_indices=torch.arange(len(boxes)),
        all_boxes=boxes,
        all_batch_indices=batch,
    )
    assert torch.isfinite(geometry.core_mask).all()
    assert torch.isfinite(geometry.context_mask).all()
