# Parameter candidates and selection evidence

本文件区分三类结论：完整数据统计决定的初始值、受控候选、仍需真实 AP 决策的项。

## Geometry-driven initial values

规则由 `tools/profile_paired_obb_dataset.py::recommend` 固定实现：

- 仅当下一层面积不足 1 cell 的比例低于 65% 时，才把下一层加入初始候选并使用
  scale assignment；
- adaptive `radius_max = clip(ceil(P3 max(width,height) P90 / 2) + 1, 2, 4)`；
- `radius_max <= 2` 使用 K=5，否则 K=7；
- batch=128 的每类 `P(instances >= 2) >= 0.5` 才默认允许 ICD；
- 每类平均实例数决定 ICD cap 候选中心 16/32/64；
- 每 epoch 至少 50 个 batch 且 ICD 支持充分时，prototype momentum 初始值为
  0.99，否则为 0.9；
- train class `max/min >= 5` 时建议 class-balanced sampling，但框架不会静默
  改变 sampler。

由完整数据得到：

| 数据集 | levels/assignment | support | ICD cap | momentum | balanced建议 |
|---|---|---|---:|---:|---|
| OGSOD-2.0 | P3/single | adaptive, r=1–4, K=7 | 64 | 0.99 | 是（6.52x） |
| OGSOD-1.0 | P3/single | adaptive, r=1–4, K=7 | 64 | 0.99 | 是（7.72x） |
| OSPRC | P3/single | adaptive, r=1–3, K=7 | 64 | 0.9 | 是（5.84x） |
| SFOC-SOA-512 | P3+P4+P5/all（主实验约束） | adaptive, r=1–4, K=7 | 64 | 0.99 | 否（1.63x） |

P3/P4/P5 不是按 feature-list 位置匹配，而是按实际 stride 8/16/32 匹配。
OGSOD-1.0 teacher 的额外 P2 不会错配到 student P3。

## Controlled candidates

生成的有限候选为：

```text
levels: P3 | P3+P4 | P3+P4+P5/all | scale-assigned P3/P4
support: hard_gt_roi | fixed_3 | fixed_5 | fixed_7 | adaptive
CRC temperature: 0.5 | 1.0 | 2.0
prototype momentum: 0.9 | 0.99 | 0.999
ICD max instances/class: 16 | 32 | 64
```

此外提供 two/three regions、neighbor exclusion、teacher-weighted pooling、
prototype relation 和 pair direction 的独立 YAML。它们都是消融，不会叠加到
主配置后再冒充单一机制收益。

## Measured loss-scale candidates

OGSOD-2.0 使用 batch=8、4 batches、imgsz=256、无 optimizer step 的 CPU probe：

| 项 | raw loss mean | student-neck grad norm mean | candidates |
|---|---:|---:|---|
| detection | 3.76291 | 1.77044 | reference |
| CRC | 0.002549 | 0.003204 | 5, 10, 20 |
| SPD | 1.13047 | 0.217154 | 0.407647, 0.815294, 1.630588 |
| ICD | 0.115759 | 0.194436 | 0.455276, 0.910551, 1.821103 |

候选中心使用：

```text
clip(0.1 * detection_gradient_norm / component_gradient_norm, 1e-4, 10)
```

再取 `[center/2, center, center*2]`。主配置选择第 0 个作为保守起点。该 probe
平均有效 CRC 实例为 18，SPD/ICD 类别组均为 1.5，teacher 无梯度。原始报告在
`loss_scale_report.json`，图在
`debug_visualizations/loss_magnitude_gradient_norm.png`。

## AP gates

以下不能由几何或单步梯度决定，必须用相同 seed、epoch、batch、增强和初始化比较：

- ICD 是否优于 CRC+SPD；
- SFOC 三尺度主实验相对 P3、P3+P4 消融的实际收益；
- prototype relation 与 paired residual direction 是否有效；
- paired mosaic 是否有净收益；
- 三分区是否优于 core/context 两区；
- 各 loss weight 的最终选择；
- 0/1/2/4/8 px optical shift 下的鲁棒性排序。

当前没有 GPU AP 结果，因此没有把任何候选写成“最优”。
