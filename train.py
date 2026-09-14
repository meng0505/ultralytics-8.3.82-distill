"""PRCD optical-to-SAR OBB training launcher.

Right-click this file in the IDE to train. Normally only the three settings below
need to be changed:

    DATASET = "ogsod1"          # ogsod1 | ogsod2 | sfoc
    STAGE = "distill"           # optical_teacher | sar_baseline | distill
    DISTILL_VARIANT = "spd_icd" # full | spd_icd
    DISTILL_LEVELS = ("P3", "P4", "P5")

The dataset paths, teacher checkpoints, SAR initial weights and dataset-driven
PRCD profile selection are defined in this file so that no command line is
required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ultralytics.models.yolo.obb.distill_train import DistillOBBTrainer


# ------------------------------ 一键运行设置 ------------------------------
# OGSOD-1.0/2.0 已有 teacher 和 SAR baseline，可直接使用 distill。
# SFOC 尚无 teacher：先将 STAGE 改为 optical_teacher 训练，再训练
# sar_baseline（可选但推荐），最后改为 distill。
DATASET = "ogsod1"  # "ogsod1", "ogsod2", "sfoc"
STAGE = "distill"  # "optical_teacher", "sar_baseline", "distill"
# full=CRC+SPD+ICD；spd_icd=只启用 SPD+ICD（关闭 CRC）。
DISTILL_VARIANT = "spd_icd"  # "full", "spd_icd"
# P3/P4/P5 全部参与蒸馏；all 表示每个有效目标都在三个尺度计算蒸馏损失。
DISTILL_LEVELS = ("P3", "P4", "P5")
DISTILL_LEVEL_ASSIGNMENT = "all"
# 主蒸馏实验必须与 SAR baseline 从同一个通用预训练权重出发，保证公平。
# 仅做“baseline 后继续蒸馏”的附加实验时才改为 True。
DISTILL_FROM_SAR_BASELINE = False

DEVICE: int | str = 0
WORKERS = 8
SEED = 0
DETERMINISTIC = True
AMP = True
CACHE = False

# True 只检查数据、路径和最终参数，不启动训练。
PREFLIGHT_ONLY = False

# 在这里覆盖任意 Ultralytics 参数，例如 {"batch": 32, "epochs": 10}。
# 此处优先级最高，适合短实验；正式实验建议保持为空。
EXTRA_ARGS: dict[str, Any] = {}


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
RUNS_TRAIN = ROOT / "runs" / "train"
RUNS_PRCD = ROOT / "runs" / "prcd"
DISTILL_CONFIG = ROOT / "recommended_distill_config.yaml"
SFOC512_DISTILL_CONFIG = ROOT / "configs" / "prcd" / "sfoc512_three_scale.yaml"
BASELINE_CONFIG = ROOT / "configs" / "prcd" / "ablations" / "baseline_sar.yaml"


# imgsz=256 与现有 teacher/baseline 以及 dataset_profile.json 的统计输入一致。
# OGSOD-1.0 沿用现有 teacher/baseline 的 batch=128；其 256 输入下实测
# batch=64 仅占约 1.86 GB，扩大 batch 可减少 step 并提高 GPU 利用率。
# OGSOD-2.0 暂保留更保守的 batch=64。
DATASETS: dict[str, dict[str, Any]] = {
    "ogsod1": {
        "display_name": "OGSOD-1.0",
        "profile_key": "ogsod-1.0",
        "distill_config": DISTILL_CONFIG,
        "paired_yaml": DATA_DIR / "ogsod1_paired.yaml",
        "optical_yaml": DATA_DIR / "ogsod1_optical.yaml",
        "imgsz": 256,
        "batch": 512,
        "epochs": 400,
        "teacher_epochs": 400,
        "teacher_init": ROOT / "yolov8s-obb.pt",
        "student_init": ROOT / "yolov8n-obb.pt",
        "teacher": RUNS_TRAIN
        / "opt-teacher-yolov8s-ogsod-1.0-256-nomosaic"
        / "weights"
        / "best.pt",
        "sar_baseline": RUNS_TRAIN
        / "sar-baseline-raw-yolov8n-ogsod-1.0-256-nomosaic"
        / "weights"
        / "best.pt",
        "teacher_run": "opt-teacher-yolov8s-ogsod-1.0-256-nomosaic",
        "baseline_run": "sar-baseline-yolov8n-ogsod-1.0-256-nomosaic-prcd",
        "distill_run": "ogsod1-prcd-yolov8n",
    },
    "ogsod2": {
        "display_name": "OGSOD-2.0",
        "profile_key": "ogsod-2.0",
        "distill_config": DISTILL_CONFIG,
        "paired_yaml": DATA_DIR / "ogsod2_paired.yaml",
        "optical_yaml": DATA_DIR / "ogsod2_optical.yaml",
        "imgsz": 256,
        "batch": 64,
        # Student-side SAR baseline and distillation use the same budget for fair comparison.
        "epochs": 400,
        "teacher_epochs": 500,
        "teacher_init": ROOT / "yolov8s-obb.pt",
        "student_init": ROOT / "yolov8n-obb.pt",
        "teacher": RUNS_TRAIN
        / "opt-teacher-yolov8s-ogsod-2.0-256-nomosaic"
        / "weights"
        / "best.pt",
        "sar_baseline": RUNS_TRAIN
        / "sar-baseline-raw-nomosaic-yolov8n-ogsod-2.0-256"
        / "weights"
        / "best.pt",
        "teacher_run": "opt-teacher-yolov8s-ogsod-2.0-256-nomosaic",
        "baseline_run": "sar-baseline-yolov8n-ogsod-2.0-256-nomosaic-prcd",
        "distill_run": "ogsod2-prcd-yolov8n",
    },
    "sfoc": {
        "display_name": "SFOC-1.0-100K-SOA",
        "profile_key": "sfoc-1.0-100k-soa-512",
        "distill_config": SFOC512_DISTILL_CONFIG,
        "paired_yaml": DATA_DIR / "sfoc_paired.yaml",
        "optical_yaml": DATA_DIR / "sfoc_optical.yaml",
        # Native-resolution SFOC experiment. Distillation uses every object on
        # P3/P4/P5. Batch 16 is conservative for teacher+student on 16 GB VRAM.
        "imgsz": 512,
        "batch": 16,
        "teacher_batch": 32,
        "epochs": 400,
        "teacher_epochs": 500,
        "teacher_init": ROOT / "yolov8s-obb.pt",
        "student_init": ROOT / "yolov8n-obb.pt",
        "teacher": RUNS_TRAIN
        / "opt-teacher-yolov8s-sfoc-soa-512-nomosaic"
        / "weights"
        / "best.pt",
        "sar_baseline": RUNS_TRAIN
        / "sar-baseline-yolov8n-sfoc-soa-512-nomosaic"
        / "weights"
        / "best.pt",
        "teacher_run": "opt-teacher-yolov8s-sfoc-soa-512-nomosaic",
        "baseline_run": "sar-baseline-yolov8n-sfoc-soa-512-nomosaic",
        "distill_run": "sfoc-prcd-yolov8n-512",
    },
}

VALID_STAGES = {"optical_teacher", "sar_baseline", "distill"}
VALID_DISTILL_VARIANTS = {"full", "spd_icd"}


def _distill_config_for_variant(
    cfg: dict[str, Any],
    variant: str,
    levels: tuple[str, ...] = DISTILL_LEVELS,
    level_assignment: str = DISTILL_LEVEL_ASSIGNMENT,
) -> Path:
    """Return a full dataset-specific config, materializing safe ablation overrides when needed."""
    variant = variant.strip().lower()
    if variant not in VALID_DISTILL_VARIANTS:
        raise ValueError(
            f"DISTILL_VARIANT={variant!r} 无效，可选值：{', '.join(sorted(VALID_DISTILL_VARIANTS))}"
        )
    normalized_levels = tuple(str(level).strip().upper() for level in levels)
    if not normalized_levels:
        raise ValueError("DISTILL_LEVELS 不能为空")
    if len(set(normalized_levels)) != len(normalized_levels):
        raise ValueError(f"DISTILL_LEVELS 存在重复尺度：{normalized_levels}")
    if any(level not in {"P3", "P4", "P5"} for level in normalized_levels):
        raise ValueError(f"DISTILL_LEVELS 仅支持 P3/P4/P5：{normalized_levels}")
    level_assignment = level_assignment.strip().lower()
    if level_assignment not in {"single", "all", "scale_assigned"}:
        raise ValueError(f"DISTILL_LEVEL_ASSIGNMENT={level_assignment!r} 无效")

    source = Path(cfg["distill_config"])
    loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    profile_key = cfg["profile_key"]
    if "_per_dataset" in loaded:
        if profile_key not in loaded["_per_dataset"]:
            raise KeyError(f"{source} 中不存在数据集配置 {profile_key!r}")
        selected = loaded["_per_dataset"][profile_key]
        loaded = {
            "distill": selected["distill"],
            "_evidence": selected.get("_evidence", {}),
            "_profile_hash": loaded.get("_profile_hash"),
        }
    distill = loaded.setdefault("distill", {})
    distill["enabled"] = True
    distill["objective"] = "prcd"
    distill["levels"] = list(normalized_levels)
    distill["level_assignment"] = level_assignment
    distill.setdefault("crc", {})["enabled"] = variant == "full"
    if variant == "spd_icd":
        distill["crc"]["weight"] = 0.0
    distill.setdefault("spd", {})["enabled"] = True
    distill.setdefault("icd", {})["enabled"] = True
    loaded["_launcher_variant"] = variant
    loaded["_launcher_levels"] = list(normalized_levels)
    loaded["_launcher_level_assignment"] = level_assignment

    level_tag = "".join(level.lower() for level in normalized_levels)
    output = ROOT / ".tmp_config" / "prcd_launcher" / f"{profile_key}-{level_tag}-{variant}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(loaded, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(output)
    return output


def _resolved_yaml(path: Path) -> dict[str, Any]:
    """Load a dataset YAML and resolve the image/label paths used by preflight."""
    if not path.is_file():
        raise FileNotFoundError(f"数据配置不存在：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = Path(data.get("path") or path.parent)
    if not base.is_absolute():
        base = (path.parent / base).resolve()
    for key in (
        "train",
        "val",
        "test",
        "optical_train",
        "optical_val",
        "optical_test",
        "labels_train",
        "labels_val",
        "labels_test",
        "pair_manifest",
    ):
        value = data.get(key)
        if value and isinstance(value, str):
            candidate = Path(value).expanduser()
            data[key] = str(candidate if candidate.is_absolute() else (base / candidate).resolve())
    return data


def _require_paths(data_yaml: Path, *, paired: bool) -> None:
    """Fail before allocating a GPU when a configured dataset path is unavailable."""
    data = _resolved_yaml(data_yaml)
    keys = ["train", "val", "labels_train", "labels_val"]
    if paired:
        keys.append("optical_train")
    if data.get("pair_manifest"):
        keys.append("pair_manifest")
    missing = [(key, data.get(key)) for key in keys if not data.get(key) or not Path(data[key]).exists()]
    if missing:
        details = "\n".join(f"  - {key}: {value}" for key, value in missing)
        raise FileNotFoundError(f"{data_yaml.name} 存在缺失路径：\n{details}")


def build_train_overrides(
    dataset: str = DATASET,
    stage: str = STAGE,
    distill_variant: str = DISTILL_VARIANT,
) -> dict[str, Any]:
    """Build the complete trainer arguments for a right-click run."""
    dataset = dataset.strip().lower()
    stage = stage.strip().lower()
    distill_variant = distill_variant.strip().lower()
    if dataset not in DATASETS:
        raise ValueError(f"DATASET={dataset!r} 无效，可选值：{', '.join(DATASETS)}")
    if stage not in VALID_STAGES:
        raise ValueError(f"STAGE={stage!r} 无效，可选值：{', '.join(sorted(VALID_STAGES))}")

    cfg = DATASETS[dataset]
    stage_batch = cfg.get("teacher_batch", cfg["batch"]) if stage == "optical_teacher" else cfg["batch"]
    common: dict[str, Any] = {
        "imgsz": cfg["imgsz"],
        "batch": stage_batch,
        "workers": WORKERS,
        "device": DEVICE,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "amp": AMP,
        "cache": CACHE,
        "optimizer": "SGD",
        "patience": 50,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
        "val": True,
        "plots": True,
        "exist_ok": False,
    }

    if stage == "optical_teacher":
        overrides = {
            **common,
            "model": str(cfg["teacher_init"]),
            "data": str(cfg["optical_yaml"]),
            "distill": str(BASELINE_CONFIG),
            "epochs": cfg["teacher_epochs"],
            "project": str(RUNS_TRAIN),
            "name": cfg["teacher_run"],
        }
    elif stage == "sar_baseline":
        overrides = {
            **common,
            "model": str(cfg["student_init"]),
            "data": str(cfg["paired_yaml"]),
            "distill": str(BASELINE_CONFIG),
            "epochs": cfg["epochs"],
            "project": str(RUNS_TRAIN),
            "name": cfg["baseline_run"],
        }
    else:
        if DISTILL_FROM_SAR_BASELINE:
            if not cfg["sar_baseline"].is_file():
                raise FileNotFoundError(f"指定从 SAR baseline 初始化，但权重不存在：{cfg['sar_baseline']}")
            student_start = cfg["sar_baseline"]
        else:
            student_start = cfg["student_init"]
        selected_distill_config = _distill_config_for_variant(cfg, distill_variant)
        level_tag = "".join(str(level).strip().lower() for level in DISTILL_LEVELS)
        variant_suffix = "" if distill_variant == "full" else f"-{distill_variant.replace('_', '-')}"
        overrides = {
            **common,
            "model": str(student_start),
            "teacher": str(cfg["teacher"]),
            "data": str(cfg["paired_yaml"]),
            "distill": str(selected_distill_config),
            "distill_dataset": cfg["profile_key"],
            "epochs": cfg["epochs"],
            "project": str(RUNS_PRCD),
            "name": f"{cfg['distill_run']}-{level_tag}{variant_suffix}",
            "strict_pairs": True,
        }
    overrides.update(EXTRA_ARGS)
    return overrides


def preflight(
    dataset: str = DATASET,
    stage: str = STAGE,
    distill_variant: str = DISTILL_VARIANT,
) -> dict[str, Any]:
    """Validate the selected one-click experiment and print its exact inputs."""
    stage = stage.strip().lower()
    distill_variant = distill_variant.strip().lower()
    cfg = DATASETS.get(dataset.strip().lower())
    if cfg is None:
        raise ValueError(f"未知 DATASET={dataset!r}")
    overrides = build_train_overrides(dataset, stage, distill_variant)
    paired = stage != "optical_teacher"
    _require_paths(Path(overrides["data"]), paired=paired and stage == "distill")

    required_files = [Path(overrides["model"]), Path(overrides["distill"])]
    if stage == "distill":
        required_files.append(Path(overrides["teacher"]))
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        if stage == "distill" and Path(overrides["teacher"]) in missing_files:
            raise FileNotFoundError(
                f"{cfg['display_name']} 尚无 optical teacher：{overrides['teacher']}\n"
                '请先把 train.py 顶部 STAGE 改为 "optical_teacher" 并右键运行；'
                '训练完成后再改回 "distill"。'
            )
        raise FileNotFoundError("缺少训练文件：\n" + "\n".join(f"  - {path}" for path in missing_files))

    print("=" * 78)
    print("PRCD 一键训练预检通过")
    print(f"数据集：{cfg['display_name']}")
    print(f"阶段：  {stage}")
    print(f"数据：  {overrides['data']}")
    print(f"模型：  {overrides['model']}")
    if stage == "distill":
        print(f"教师：  {overrides['teacher']}")
        print(
            f"学生初始化：{'SAR baseline（附加微调实验）' if DISTILL_FROM_SAR_BASELINE else '通用 YOLOv8n-OBB 预训练权重（主实验）'}"
        )
        print(f"组合：  {distill_variant}（{'CRC+SPD+ICD' if distill_variant == 'full' else 'SPD+ICD'}）")
        print(f"尺度：  {'/'.join(DISTILL_LEVELS)}（每个有效目标在全部尺度蒸馏）")
        print(f"配置：  {overrides['distill']}")
    print(f"输出：  {Path(overrides['project']) / overrides['name']}")
    print(f"参数：  imgsz={overrides['imgsz']}, batch={overrides['batch']}, epochs={overrides['epochs']}")
    print("=" * 78)
    return overrides


def main() -> None:
    """Run the selected stage without requiring command-line arguments."""
    overrides = preflight()
    if PREFLIGHT_ONLY:
        print("PREFLIGHT_ONLY=True：仅检查配置，不启动训练。")
        return
    DistillOBBTrainer(overrides=overrides).train()


if __name__ == "__main__":
    main()


__all__ = ("DATASETS", "build_train_overrides", "preflight")
