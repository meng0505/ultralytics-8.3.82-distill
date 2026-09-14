# Changed files summary

由于工作区没有有效 Git `HEAD`，本清单代替 commit/diff provenance。

## 对官方路径的最小修改

- `ultralytics/cfg/default.yaml`：声明 teacher、distill、pairing、profile hash 和
  optical shift 参数，使官方配置校验认识这些 key。
- `ultralytics/cfg/__init__.py`：增加 `yolo obb distill-train` 专用分派；普通
  `yolo obb train` 不变。
- `ultralytics/data/dataset.py`：增加两个可选 hook，供新增 paired dataset
  显式解析 labels 和选择 run-local label cache；无 hook 时仍执行原逻辑。
- `ultralytics/data/build.py`：仅当 data YAML 明确提供 `labels_<split>` 时构建
  显式标签 dataset，使非标准 SFOC 目录也可使用标准 `yolo obb val`。

未修改 `ultralytics/utils/loss.py`、OBB head、官方 `OBBTrainer`、
`OBBModel` 或主 optimization loop。

## 新增核心实现

- `ultralytics/data/paired_obb_dataset.py`
- `ultralytics/models/yolo/obb/distill_model.py`
- `ultralytics/models/yolo/obb/distill_train.py`
- `ultralytics/nn/distillation/{__init__,object_support,crc,prototype_bank,spd,icd,distiller}.py`
- `ultralytics/utils/distill_config.py`
- `ultralytics/utils/distill_visualization.py`
- `ultralytics/utils/stratified_obb_metrics.py`

## 数据与实验配置

- `configs/prcd/datasets_profile.yaml`
- `configs/prcd/data_{ogsod1,ogsod2,osprc,sfoc_soa}_paired.yaml`
- `configs/prcd/data_sfoc_soa_optical.yaml`
- `configs/prcd/ablations/*.yaml`：17 个 baseline/KD/PRCD 消融配置。
- `recommended_distill_config.yaml`：全数据 geometry profile 加 OGSOD-2.0
  loss probe 后的主候选。
- `distill_recommended.yaml`：同一次生成流程保留的兼容副本。

## 工具

- `tools/profile_paired_obb_dataset.py`
- `tools/validate_pair_alignment.py`
- `tools/analyze_distill_loss_scale.py`
- `tools/apply_loss_scale_recommendation.py`
- `tools/run_distill_ablation.py`
- `tools/export_sar_student.py`
- `tools/visualize_prcd_dataset.py`
- `tools/plot_loss_scale_report.py`
- `tools/plot_registration_robustness.py`
- `tools/smoke_prcd_step.py`
- `tools/evaluate_prcd_stratified.py`
- `tools/benchmark_sar_export.py`

## 测试

- `tests/test_paired_obb_dataset.py`
- `tests/test_object_support.py`
- `tests/test_crc.py`
- `tests/test_spd.py`
- `tests/test_icd.py`
- `tests/test_distill_config.py`
- `tests/test_distill_disabled_equivalence.py`
- `tests/test_distill_inference_removal.py`
- `tests/test_distill_amp_ddp.py`
- `tests/test_stratified_obb_metrics.py`

## 生成的证据与文档

- `implementation_plan.md`
- `repository_audit.md`
- `dataset_profile.{md,json,csv}`
- `dataset_figures/*.png`
- `loss_scale_report.json`
- `smoke_prcd_step.json`
- `debug_visualizations/*`
- `README_PRCD.md`
- `baseline_equivalence_report.md`
- `smoke_test_results.md`
- `parameter_selection.md`
- `teacher_weights_inventory.md`
- `known_limitations.md`
- 本文件。

源数据没有被重写。一次早期 fraction smoke 暴露了上游共享 label-cache 风险后，
OGSOD-2.0 `dataset/labels/train.cache` 已按完整 14,250 张重建；新增 paired
trainer 现在固定使用 run-local cache，回归测试覆盖该行为。
