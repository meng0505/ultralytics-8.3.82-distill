"""Dependency-light PRCD debug visualization helpers with explicit experiment metadata."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.utils.ops import xywhr2xyxyxyxy


def _draw_obbs(image: np.ndarray, boxes: torch.Tensor, classes: torch.Tensor, color_seed: int = 0) -> np.ndarray:
    """Draw normalized xywhr OBBs with stable class colors."""
    output = image.copy()
    height, width = output.shape[:2]
    if not len(boxes):
        return output
    scaled = boxes.detach().cpu().float().clone()
    scaled[:, [0, 2]] *= width
    scaled[:, [1, 3]] *= height
    corners = xywhr2xyxyxyxy(scaled).round().int().numpy()
    for points, class_id in zip(corners, classes.detach().cpu().long().view(-1).tolist()):
        rng = np.random.default_rng(color_seed + class_id)
        color = tuple(int(value) for value in rng.integers(64, 256, 3))
        cv2.polylines(output, [points], True, color, 2, cv2.LINE_AA)
        center = tuple(points.mean(axis=0).astype(int))
        cv2.circle(output, center, 3, color, -1)
        cv2.putText(output, str(class_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return output


def save_paired_debug(
    sar: np.ndarray,
    optical: np.ndarray,
    boxes: torch.Tensor,
    classes: torch.Tensor,
    sample_id: str,
    destination: str | Path,
) -> None:
    """Save side-by-side aligned modalities with identical OBB colors."""
    sar_drawn = _draw_obbs(sar, boxes, classes)
    optical_drawn = _draw_obbs(optical, boxes, classes)
    canvas = np.concatenate((sar_drawn, optical_drawn), axis=1)
    cv2.putText(canvas, sample_id, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas)


def save_support_masks(core: torch.Tensor, context: torch.Tensor, destination: str | Path, title: str = "") -> None:
    """Save core/context masks as a color diagnostic."""
    core_np = core.detach().float().cpu().squeeze().numpy()
    context_np = context.detach().float().cpu().squeeze().numpy()
    image = np.stack((np.zeros_like(core_np), context_np, core_np), axis=-1)
    image = cv2.resize((255 * image).clip(0, 255).astype(np.uint8), (280, 280), interpolation=cv2.INTER_NEAREST)
    if title:
        cv2.putText(image, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image)


def save_matrix(matrix: torch.Tensor, destination: str | Path, title: str) -> None:
    """Save a cosine/configuration matrix with a color bar."""
    import matplotlib.pyplot as plt

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    handle = axis.imshow(matrix.detach().float().cpu().numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_title(title)
    figure.colorbar(handle, ax=axis)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def save_matrix_pair(
    teacher: torch.Tensor,
    student: torch.Tensor,
    destination: str | Path,
    *,
    matrix_kind: str,
    metadata: str,
) -> None:
    """Save teacher/student prototype-cosine or ICD matrices on one shared scale."""
    import matplotlib.pyplot as plt

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    arrays = (teacher.detach().float().cpu().numpy(), student.detach().float().cpu().numpy())
    for axis, array, modality in zip(axes, arrays, ("optical teacher", "SAR student")):
        handle = axis.imshow(array, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_title(f"{modality}: {matrix_kind}")
    figure.suptitle(metadata, fontsize=9)
    figure.colorbar(handle, ax=axes.ravel().tolist(), shrink=0.8)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_response_diagnostic(
    teacher_response: torch.Tensor,
    student_response: torch.Tensor,
    teacher_distribution: torch.Tensor,
    student_distribution: torch.Tensor,
    destination: str | Path,
    *,
    metadata: str,
    region_names: tuple[str, ...] = ("core", "context"),
) -> None:
    """Save GT-class response patches and their explicitly normalized region distributions."""
    import matplotlib.pyplot as plt

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    teacher = teacher_response.detach().float().cpu().squeeze().numpy()
    student = student_response.detach().float().cpu().squeeze().numpy()
    teacher_dist = teacher_distribution.detach().float().cpu().view(-1).numpy()
    student_dist = student_distribution.detach().float().cpu().view(-1).numpy()
    vmin, vmax = min(float(teacher.min()), float(student.min())), max(float(teacher.max()), float(student.max()))
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for axis, response, modality in zip(axes[:2], (teacher, student), ("optical teacher", "SAR student")):
        handle = axis.imshow(response, vmin=vmin, vmax=vmax, cmap="viridis")
        axis.set_title(f"{modality} GT-class response")
    x = np.arange(len(region_names))
    width = 0.36
    axes[2].bar(x - width / 2, teacher_dist, width, label="teacher")
    axes[2].bar(x + width / 2, student_dist, width, label="student")
    axes[2].set_xticks(x, region_names)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("region response distribution")
    axes[2].legend()
    figure.suptitle(metadata, fontsize=9)
    figure.colorbar(handle, ax=axes[:2], shrink=0.8)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_embedding_projection(
    teacher_embeddings: torch.Tensor,
    student_embeddings: torch.Tensor,
    classes: torch.Tensor,
    destination: str | Path,
    *,
    metadata: str,
    seed: int = 0,
) -> str:
    """Save a joint teacher/student t-SNE, falling back to deterministic PCA for very small inputs."""
    import matplotlib.pyplot as plt

    teacher = teacher_embeddings.detach().float().cpu()
    student = student_embeddings.detach().float().cpu()
    if teacher.shape != student.shape or teacher.ndim != 2:
        raise ValueError("teacher/student embeddings must have equal [N, D] shapes")
    values = torch.cat((teacher, student), dim=0).numpy()
    if len(teacher) >= 3:
        from sklearn.manifold import TSNE

        perplexity = min(30.0, max(2.0, float(len(values) - 1) / 3))
        projected = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(values)
        method = "t-SNE"
    else:
        centered = torch.from_numpy(values) - torch.from_numpy(values).mean(dim=0, keepdim=True)
        _, _, right = torch.linalg.svd(centered, full_matrices=False)
        projected = (centered @ right[:2].T).numpy()
        method = "PCA (too few points for t-SNE)"

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    class_ids = classes.detach().cpu().long().view(-1).numpy()
    figure, axis = plt.subplots(figsize=(6, 5))
    for offset, marker, modality in ((0, "o", "teacher"), (len(teacher), "x", "student")):
        points = projected[offset : offset + len(teacher)]
        handle = axis.scatter(points[:, 0], points[:, 1], c=class_ids, marker=marker, alpha=0.8, label=modality)
    axis.set_title(f"{method}: {metadata}")
    axis.legend()
    figure.colorbar(handle, ax=axis, label="class ID")
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return method


def save_loss_scale_report(report: dict, destination: str | Path) -> None:
    """Plot raw magnitude and student-neck gradient norms from a loss probe report."""
    import matplotlib.pyplot as plt

    names = ("detection", "crc", "spd", "icd")
    magnitudes = [report["magnitude"][name]["mean"] for name in names]
    gradients = [report["student_neck_gradient_norm"][name]["mean"] for name in names]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, values, title in zip(axes, (magnitudes, gradients), ("raw loss magnitude", "student-neck gradient norm")):
        axis.bar(names, values)
        axis.set_yscale("log")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"no-update probe: batch={report.get('batch_size')}, batches={report.get('batches')}, "
        f"imgsz={report.get('imgsz')}"
    )
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


__all__ = (
    "save_embedding_projection",
    "save_loss_scale_report",
    "save_matrix",
    "save_matrix_pair",
    "save_paired_debug",
    "save_response_diagnostic",
    "save_support_masks",
)
