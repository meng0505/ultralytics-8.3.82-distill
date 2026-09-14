# Known limitations and research gates

1. **没有正式 GPU/AP 结论。** 当前 PyTorch 是 CUDA build，但 NVIDIA driver
   不可用。已完成 CPU loss/gradient probe、真实单步优化和双进程 CPU DDP，
   未运行正式 epoch、mAP、显存或吞吐评估。

2. **OGSOD-2.0 之外的 loss 权重仍是候选。** OGSOD-1.0、OSPRC、SFOC 的
   support/level 参数来自完整几何统计，但 CRC/SPD/ICD 权重必须在各自
   teacher/student 和目标 batch 下重新 probe。

3. **SFOC teacher 尚不存在。** 已提供 optical teacher、SAR-only baseline 和
   后续蒸馏命令。源标签曾沿用六类母表 ID `0/3`，现已显式重映射为连续
   `0=plane, 1=ship`，正式 teacher/student head 均为 `nc=2`；旧标签保存在
   数据集同级 `labels_original_ids_0_3_backup/`。

4. **ICD、P4、prototype relation、paired residual direction 不是既定收益。**
   默认候选保留 ICD 是为了验证研究假设；是否进入最终方法必须由等预算 AP 消融
   决定。`pair_direction` 明确标记为实验开关且默认关闭。

5. **跨 rank ICD gather 未实现。** prototype 更新使用 sum/count all-reduce，
   ICD 只保证每个 rank 内相同实例顺序。`cross_rank_gather` 和
   `memory_assisted` 目前是保留且默认关闭的接口，不会被静默执行。

6. **paired mosaic 已测试但默认关闭。** 自动测试证明两模态选择相同四个 IDs、
   使用同一 RNG/buffer replay 且变换后 OBB 一致；正式主配置仍关闭 mosaic、
   mixup、copy-paste，以保证与数据 profile 和无 mosaic teacher/baseline 的
   比较前提一致。

7. **可视化分两层。** 数据工具已经生成真实 paired augmentation、OBB、
   P3/P4/P5 support 和 loss/gradient 图。response、prototype t-SNE/cosine、
   ICD matrix 的绘图 API 已实现，但需要训练期真实 tensor 才能生成；本次没有用
   随机张量伪造这些图。

8. **旧 checkpoint 的数据 provenance 不完整。** 旧 `args.yaml` 指向会变化的
   通用 data YAML，没有当时文件清单 hash。新 PRCD checkpoint 保存 teacher
   SHA256、完整选中 config 和 dataset profile hash；这不能追溯修复旧 run。

9. **Git provenance 不可用。** 工作区挂载的 `.git` 为空，父仓库也没有有效
   `HEAD`，因此无法可信报告 commit hash 或创建阶段 commits。所有改动以
   `changed_files_summary.md` 和测试报告追踪。

10. **同步增强参数记录的是策略与确定性 seed。** 输出 manifest 记录 sample ID、
    source IDs、seed 和增强范围，最终 OBB 由两模态 allclose 断言验证；上游
    `RandomPerspective` 不公开每次抽样的完整 affine/flip 参数结构，未为此侵入式
    改写官方增强器。
