# Optical-to-SAR PRCD for Ultralytics OBB

本仓库新增了一条独立的 optical-teacher / SAR-student 训练路径，实现
Prototype-Referenced Cross-Modal Distillation（PRCD）。主配置由三部分组成：

- CRC：在共同 OBB 对象支持上蒸馏 GT 类别的 core/context 区域分布；
- SPD：将当前 SAR 类别 prototype 对齐到按类别、特征层维护的 optical teacher EMA prototype；
- ICD：比较 prototype-centered 类内实例构型矩阵，样本不足时跳过。

它不假设 optical 与 SAR 像素响应相同，不把上下文或强散射直接视为噪声，也不把
P4/P5 对极小目标的硬池化当成默认可靠支持。`distill.enabled=false` 时保留官方
SAR OBB 检测路径；最终导出是无 teacher、adapter、prototype 或 distiller 的纯
SAR detector。

## 已由数据决定的初始配置

完整统计见 `dataset_profile.md/json/csv`，当前 profile hash 为
`18884c48b7f40f38d726f02eda884f2bfdc3f7e5f8ced3a9b01b4853a7a3102d`。

| 数据集 | 训练对 | 关键空间支持事实 | 初始候选 |
|---|---:|---|---|
| OGSOD-2.0 | 14,250 | P3 面积不足 1 cell 为 39.7%，P4 为 82.3% | P3、adaptive、半径 1–4、K=7 |
| OGSOD-1.0 | 14,664 | P4 面积不足 1 cell 为 83.2% | P3、adaptive、半径 1–4、K=7 |
| OSPRC | 2,640 | P3 面积不足 1 cell 为 68.7%，P4 为 89.2% | P3、adaptive、半径 1–3、K=7 |
| SFOC-1.0-100K-SOA（512） | 4,252 | P3/P4/P5 面积不足 1 cell 为 10.6%/24.8%/45.0% | P3+P4+P5 all、adaptive、半径 1–4 |

OGSOD-2.0 的 CPU no-update probe 给出的保守下界候选是 CRC `5.0`、SPD
`0.4076469`、ICD `0.4552757`。它们使各蒸馏项的实测 student-neck 梯度接近
检测梯度的约 10%，但不代表 AP 最优；正式 batch/GPU 环境仍需重新探测并做有限
候选消融。

## 环境与入口

本实现基于仓库内 Ultralytics `8.3.82`。推荐使用当前环境：

```bash
cd /home/mengfanlong/ultralytics-8.3.82-distill
export PYTHONPATH="$PWD:${PYTHONPATH}"
```

专用入口：

```bash
yolo obb distill-train key=value ...
```

也可直接运行：

```bash
python -m ultralytics.models.yolo.obb.distill_train key=value ...
```

### 根目录右键一键运行

不想填写命令行时，直接打开仓库根目录的 `train.py`，只修改顶部三个变量：

```python
DATASET = "ogsod2"  # ogsod1 | ogsod2 | sfoc
STAGE = "distill"  # optical_teacher | sar_baseline | distill
DISTILL_VARIANT = "spd_icd"  # full=CRC+SPD+ICD | spd_icd=SPD+ICD
DISTILL_LEVELS = ("P3", "P4", "P5")
DISTILL_LEVEL_ASSIGNMENT = "all"
DISTILL_FROM_SAR_BASELINE = False
```

只跑 SPD+ICD 时保持 `STAGE = "distill"`，并设置
`DISTILL_VARIANT = "spd_icd"`。启动器会继承所选数据集的完整 PRCD
profile，仅关闭 CRC（`enabled: false, weight: 0.0`），不会丢失该数据集的
尺度、loss weight、teacher 路径等设置。生成的完整配置位于
`.tmp_config/prcd_launcher/`，无需手工修改。

默认 `DISTILL_LEVELS = ("P3", "P4", "P5")` 且
`DISTILL_LEVEL_ASSIGNMENT = "all"`，表示每个有效目标都会分别在 P3、P4、P5
计算所启用的蒸馏损失；`spd_icd` 因而是在三个尺度上同时执行 SPD 和 ICD。

主蒸馏实验保持 `DISTILL_FROM_SAR_BASELINE = False`：SAR baseline 与蒸馏学生
都从同一个 `yolov8n-obb.pt` 出发，baseline 只作为无蒸馏对照，不参与蒸馏初始化。
仅在明确开展“已训练 baseline 再蒸馏”的附加微调实验时才设为 `True`。

然后在 IDE 中右键运行 `train.py`。OGSOD-1.0/2.0 会自动选择各自已有的
teacher 和 SAR baseline。SFOC 当前没有 teacher，应依次运行
`optical_teacher`、可选但推荐的 `sar_baseline`、最后 `distill`。三套 paired
数据 YAML 位于 `dataset/`；仅有 train/val 或 train/test 的数据集会明确复用
已有评估划分。

SFOC 使用原生 `512x512` 输入：teacher batch=32，SAR baseline/PRCD batch=16。
其主配置是 `configs/prcd/sfoc512_three_scale.yaml`，每个有效实例都在
P3/P4/P5 三个尺度执行 CRC、SPD 和 ICD；它与 OGSOD 的 256 profile 完全分开。

## 1. 重新统计与配对验证

```bash
python tools/profile_paired_obb_dataset.py \
  --config configs/prcd/datasets_profile.yaml \
  --output .
```

单独检查 OGSOD-2.0：

```bash
python tools/validate_pair_alignment.py \
  --sar /media/mengfanlong/111E18BA111E18BA/数据集/OGSOD-2.0/imagesSAR/train/images \
  --optical /media/mengfanlong/111E18BA111E18BA/数据集/OGSOD-2.0/images/train/images \
  --labels /media/mengfanlong/111E18BA111E18BA/数据集/OGSOD-2.0/labels/train \
  --strict \
  --output pair_validation_ogsod2_train.json
```

配对依据是 exact sample ID、manifest 或用户指定的 `module:function` mapper，
从不使用目录排序后 `zip`。训练期标签 cache 写入当前 run 目录，短比例 pilot
不会覆盖完整数据集的 `labels/train.cache`。

## 2. OGSOD-2.0 推荐起跑命令

```bash
yolo obb distill-train \
  model=runs/train/sar-baseline-raw-nomosaic-yolov8n-ogsod-2.0-256/weights/best.pt \
  teacher=runs/train/opt-teacher-yolov8s-ogsod-2.0-256-nomosaic/weights/best.pt \
  data=configs/prcd/data_ogsod2_paired.yaml \
  distill=recommended_distill_config.yaml \
  distill_dataset=ogsod-2.0 \
  imgsz=256 batch=128 epochs=500 seed=0 device=0 \
  mosaic=0 mixup=0 copy_paste=0 \
  project=runs/prcd name=ogsod2_prcd_p3
```

`batch=128` 来自原 teacher/baseline 训练条件和 batch feasibility 统计，并非对
任意 GPU 的硬编码。如果显存不足，应先降低 batch，再用下一节的 probe 重新生成
损失权重候选。

OGSOD-1.0：

```bash
yolo obb distill-train \
  model=runs/train/sar-baseline-raw-yolov8n-ogsod-1.0-256-nomosaic/weights/best.pt \
  teacher=runs/train/opt-teacher-yolov8s-ogsod-1.0-256-nomosaic/weights/best.pt \
  data=configs/prcd/data_ogsod1_paired.yaml \
  distill=recommended_distill_config.yaml distill_dataset=ogsod-1.0 \
  imgsz=256 batch=128 epochs=400 seed=0 device=0 \
  mosaic=0 mixup=0 copy_paste=0 \
  project=runs/prcd name=ogsod1_prcd_p3
```

OSPRC：

```bash
yolo obb distill-train \
  model=runs/train/sar-baseline-raw-nomosaic-yolov8n-osprc-256/weights/best.pt \
  teacher=runs/train/opt-teacher-yolov8s-osprc-256-nomosaic/weights/best.pt \
  data=configs/prcd/data_osprc_paired.yaml \
  distill=recommended_distill_config.yaml distill_dataset=osprc \
  imgsz=256 batch=128 epochs=500 seed=0 device=0 \
  mosaic=0 mixup=0 copy_paste=0 \
  project=runs/prcd name=osprc_prcd_p3
```

## 3. 在目标 batch 上重新探测损失权重

此命令不执行 optimizer step：

```bash
python tools/analyze_distill_loss_scale.py \
  --model runs/train/sar-baseline-raw-nomosaic-yolov8n-ogsod-2.0-256/weights/best.pt \
  --teacher runs/train/opt-teacher-yolov8s-ogsod-2.0-256-nomosaic/weights/best.pt \
  --data configs/prcd/data_ogsod2_paired.yaml \
  --distill recommended_distill_config.yaml \
  --distill-dataset ogsod-2.0 \
  --batch 128 --batches 8 --imgsz 256 --device 0 \
  --output loss_scale_report_gpu_b128.json
```

推荐值不会静默修改用户配置。显式选择第 0 个保守候选并写到新文件：

```bash
python tools/apply_loss_scale_recommendation.py \
  --config recommended_distill_config.yaml \
  --report loss_scale_report_gpu_b128.json \
  --dataset ogsod-2.0 \
  --candidate-index 0 \
  --output configs/prcd/ogsod2_gpu_b128.yaml
```

## 4. SFOC：先训练 teacher，再做蒸馏

该数据源当前是两类任务：类别 `0=plane`、`1=ship`，teacher 和 student 的
检测 head 均使用 `nc: 2`。源数据早期沿用六类母表的稀疏 ID `0/3`，现已显式
重映射为连续 `0/1`；映射记录位于数据集 `labels/remap_manifest.json`。

先训练 optical teacher：

```bash
yolo obb train \
  model=yolov8s-obb.pt \
  data=configs/prcd/data_sfoc_soa_optical.yaml \
  imgsz=512 batch=32 epochs=500 seed=0 device=0 \
  mosaic=0 mixup=0 copy_paste=0 \
  project=runs/train name=opt-teacher-yolov8s-sfoc-soa-512-nomosaic
```

教师尚未完成前，可先跑 SAR-only 基线。使用专用 trainer 是为了识别
`imagesSAR` 与显式 `labels_*`：

```bash
yolo obb distill-train \
  model=yolov8n-obb.pt \
  data=configs/prcd/data_sfoc_soa_paired.yaml \
  distill=configs/prcd/ablations/baseline_sar.yaml \
  imgsz=512 batch=16 epochs=500 seed=0 device=0 \
  mosaic=0 mixup=0 copy_paste=0 \
  project=runs/train name=sar-baseline-yolov8n-sfoc-soa-512-nomosaic
```

有 teacher 和 SAR baseline 后：

```bash
yolo obb distill-train \
  model=runs/train/sar-baseline-yolov8n-sfoc-soa-512-nomosaic/weights/best.pt \
  teacher=runs/train/opt-teacher-yolov8s-sfoc-soa-512-nomosaic/weights/best.pt \
  data=configs/prcd/data_sfoc_soa_paired.yaml \
  distill=configs/prcd/sfoc512_three_scale.yaml \
  distill_dataset=sfoc-1.0-100k-soa-512 \
  imgsz=512 batch=16 epochs=500 seed=0 device=0 \
  mosaic=0 mixup=0 copy_paste=0 \
  project=runs/prcd name=sfoc-prcd-yolov8n-512-p3p4p5
```

在正式运行前，必须先针对 SFOC teacher/student 做 loss-scale probe；当前 SFOC
权重仍只是几何统计产生的候选。SFOC-512 的完整统计位于
`configs/prcd/profiles/sfoc512/`。

## 5. 验证、恢复与纯 SAR 导出

训练 checkpoint 保存 teacher 路径/hash、profile hash、完整选中配置和精确
PRCD state；恢复时不一致会报错：

```bash
yolo obb distill-train resume=runs/prcd/ogsod2_prcd_p3/weights/last.pt
```

验证原 checkpoint：

```bash
yolo obb val \
  model=runs/prcd/ogsod2_prcd_p3/weights/best.pt \
  data=configs/prcd/data_ogsod2_paired.yaml \
  split=test imgsz=256 batch=128 device=0
```

导出纯 SAR student：

```bash
python tools/export_sar_student.py \
  --checkpoint runs/prcd/ogsod2_prcd_p3/weights/best.pt \
  --output runs/prcd/ogsod2_prcd_p3/weights/sar_student.pt \
  --imgsz 256 --atol 1e-5
```

再用标准 OBB 入口验证/推理：

```bash
yolo obb val \
  model=runs/prcd/ogsod2_prcd_p3/weights/sar_student.pt \
  data=configs/prcd/data_ogsod2_paired.yaml \
  split=test imgsz=256 batch=128 device=0

yolo obb predict \
  model=runs/prcd/ogsod2_prcd_p3/weights/sar_student.pt \
  source=dataset/images/test imgsz=256 device=0
```

官方 AP 加原始像素尺寸/P3 支持分组 AP：

```bash
python tools/evaluate_prcd_stratified.py \
  --model runs/prcd/ogsod2_prcd_p3/weights/sar_student.pt \
  --data configs/prcd/data_ogsod2_paired.yaml \
  --split test --imgsz 256 --batch 128 --device 0 \
  --output runs/prcd/ogsod2_prcd_p3/stratified_test_ap.json
```

分组 evaluator 对其它尺寸范围的 GT 和 detection 使用 ignore，而不是在每个组中
重复计为 false positive；同时输出 per-class AP、原始像素 width/height/area、
等效边长 tiny/small/medium/large、P3 area cells 和 P3 min-dimension cells
分组。

证明纯 SAR export 没有额外参数/FLOPs，并在目标设备测延迟：

```bash
python tools/benchmark_sar_export.py \
  --checkpoint runs/prcd/ogsod2_prcd_p3/weights/best.pt \
  --export runs/prcd/ogsod2_prcd_p3/weights/sar_student.pt \
  --imgsz 256 --batch 1 --device 0 --warmup 50 --repeats 500 \
  --output runs/prcd/ogsod2_prcd_p3/sar_export_benchmark.json
```

## 6. 等预算消融与配准鲁棒性

只生成计划，不执行：

```bash
python tools/run_distill_ablation.py \
  --model runs/train/sar-baseline-raw-nomosaic-yolov8n-ogsod-2.0-256/weights/best.pt \
  --teacher runs/train/opt-teacher-yolov8s-ogsod-2.0-256-nomosaic/weights/best.pt \
  --data configs/prcd/data_ogsod2_paired.yaml \
  --base-distill recommended_distill_config.yaml \
  --distill-dataset ogsod-2.0 \
  --epochs 20 --fraction 1.0 --batch 128 --imgsz 256 --seed 0 --device 0 \
  --registration-shifts 1 2 4 8 \
  --output ablation_plan.json
```

`--base-distill` 确保每个小型消融 YAML 只覆盖目标开关，同时继承该数据集的
support、momentum 和实测 loss weights。加 `--execute` 才会顺序执行。runner
会把最终 `results.csv` 指标回填到计划 JSON；
执行失败立即停止，避免后续配置在不同前提下继续。配准曲线：

```bash
python tools/plot_registration_robustness.py \
  --plan ablation_plan.json \
  --output debug_visualizations/registration_robustness.png
```

## 7. 可视化

生成真实配对同步增强、同色 OBB 和 P3/P4/P5 的 hard/fixed/adaptive 支持：

```bash
python tools/visualize_prcd_dataset.py \
  --data configs/prcd/data_ogsod2_paired.yaml \
  --samples 3 --objects-per-sample 3 --imgsz 256 \
  --output debug_visualizations
```

损失量级图：

```bash
python tools/plot_loss_scale_report.py \
  --report loss_scale_report.json \
  --output debug_visualizations/loss_magnitude_gradient_norm.png
```

`ultralytics/utils/distill_visualization.py` 还提供 response/distribution、
prototype t-SNE、prototype cosine matrix 和 teacher/student ICD matrix 的统一
绘图函数。图标题接口要求调用方传入 sample ID、class、GT 尺寸、level、support
和 response distribution 元数据。

## 8. 测试

```bash
python -m pytest -q \
  tests/test_paired_obb_dataset.py \
  tests/test_object_support.py \
  tests/test_crc.py tests/test_spd.py tests/test_icd.py \
  tests/test_distill_config.py \
  tests/test_distill_disabled_equivalence.py \
  tests/test_distill_inference_removal.py \
  tests/test_distill_amp_ddp.py \
  tests/test_stratified_obb_metrics.py
```

DDP 测试需要本机 loopback socket 权限。当前证据与未完成项分别记录在
`smoke_test_results.md`、`baseline_equivalence_report.md` 和
`known_limitations.md`。
