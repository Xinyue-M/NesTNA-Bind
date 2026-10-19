# Optimized 单轮 GPU 测试

- PBS 作业：`11176483.node1`
- 提交脚本：`run_test.pbs`
- 队列／节点：`siais_gpu_4` / `SIAIS-Gnode021`
- GPU：RTX 3090，PBS 分配的 GPU 0
- 运行时间：2026-10-19 23:16:45 至 23:20:17 CST
- 总耗时：**3 分 32 秒**
- 作业状态：完成；退出码 **0**
- 范围：第 0 折、1 epoch，从头初始化模型，完整数据
- 样本数：528 训练、132 验证、150 测试

训练命令：

```bash
python -u train_cv.py --taskname RNA --taskbase site --num_sampling all \
  --num_neighbors 128 --split_ratio 0.8 --seed 0 --fold 0 --epochs 1 \
  --checkpoints_dir logs/test1_11176483.node1
```

## 自动检查结果

- `out.csv` 恰好包含 epoch 0 的一行指标。
- `split.json` 中的样本数正确，训练和验证样本不重叠。
- `Model_Epoch0.pth`、`Bestmodel_Epoch0.pth` 均能严格重新加载到 CPU 模型。
- `Predict_Epoch0.pth` 包含 35,686 个残基预测及等长二值标签，正标签 4,828 个。
- 所有预测概率有限，范围在 [0, 1]。
- 日志包含 `PBS_OPTIMIZED_TEST_SUCCESS` 和 `PBS_TEST_EXIT_STATUS=0`。

## 本轮指标

以下为训练损失所使用的 patch 中心映射及表面标签重建口径：

| 指标 | 结果 |
|---|---:|
| 训练 loss | 0.426348 |
| 验证 AUROC | 0.809234 |
| 测试 AUROC | 0.816513 |
| 测试 F1 | 0.507636 |
| 测试 AUPRC | 0.492490 |
| 验证集选择的阈值 | 0.18 |

这里只验证单轮完整流程的可运行性，不代表最终收敛性能。

## 文件位置

作业日志和自动检查结果：

- `logs/test1_11176483.node1/job.log`
- `logs/test1_11176483.node1/verification.json`

训练输出目录：

`logs/test1_11176483.node1/2026-10-19_23-16-49-749766_site_RNA/fold_0/`

包括模型、预测、CSV 指标、划分清单及运行代码备份。

再次提交：

```bash
cd /public/home/maxy2022/prorna/new_24/nestnabind/optimized
qsub run_test.pbs
```
