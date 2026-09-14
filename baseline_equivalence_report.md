# Baseline equivalence report

验证日期：2026-07-29。环境：Ultralytics 8.3.82、PyTorch 2.2.2+cu121、
NumPy 1.26.4；当前进程 `torch.cuda.is_available() == False`。

## 严格边界

`distill.enabled: false` 时：

- `DistillOBBModel` 不安装 head capture，不创建 adapter、prototype 或 distiller；
- detection forward 和 `v8OBBLoss` 沿用官方实现；
- 普通 OGSOD 数据仍由官方 `OBBTrainer/YOLODataset` 路径处理；
- 仅 SFOC 这种非标准 `imagesSAR -> labels` 目录在 YAML 提供 `labels_*` 时使用
  显式标签 dataset。

## 自动测试结果

| 项目 | 比较条件 | 容差 | 结果 |
|---|---|---:|---|
| 参数名与参数量 | 同 YAML、同 state dict | 精确 | 通过 |
| inference decoded tensor | 同 batch、eval | `atol=0, rtol=0` (`torch.equal`) | 通过 |
| detection scalar loss | 同 batch、train | `atol=0, rtol=0` | 通过 |
| box/cls/DFL items | 同 batch、train | `atol=0, rtol=0` | 通过 |
| optimizer parameter groups | SGD；组顺序、weight decay、逐参数 numel | 精确 | 通过 |
| pure-student inference | distill model detector state复制到普通 `OBBModel` | `atol=0, rtol=0` | 通过 |

测试位置：`tests/test_distill_disabled_equivalence.py` 和
`tests/test_distill_inference_removal.py`。

真实 OGSOD-2.0 smoke checkpoint 的 pure-SAR 导出再次得到
`max_abs_error=0`，且导出 state 中不存在 `teacher`、`distiller` 或
`prototype` key。

## 解释

这里证明的是代码路径和数值等价，不是重新训练后的 AP 可重复性。完整训练 AP 还会
受到 CUDA/cuDNN、GPU 数量、worker 调度和数据增强随机性的影响。由于当前 NVIDIA
driver 不可用，本次没有伪造或宣称一轮新的 GPU baseline AP。
