# NesTNA-Bind

A nested Transformer for nucleic-acid binding site prediction.

本仓库根目录包含重构后的训练和推理代码，兼容原有 `ViT_ESM` 权重。
`data_preparation/`、`esm_preparation/`、数据集清单及 MIT 许可证继续保留。

## 代码结构

| 文件 | 职责 |
|---|---|
| `Arguments.py` | 训练参数及兼容选项 |
| `dataloader.py` | 共用的数据准备、FPS 采样和 `Protein` 数据结构 |
| `vit_esm.py` | 局部／全局 Transformer 模型 |
| `utils.py` | 残基汇总、损失、指标、数据划分和早停 |
| `train_cv.py` | 训练、验证、测试和模型保存 |
| `inference.py` | 单蛋白或批量预测 |
| `run_test.pbs` | 完整数据、第 0 折、1 epoch GPU 测试 |
| `run_train_50.pbs` | 第 0 折、最多 50 epoch 的训练作业 |
| `verify_refactor.py` | 原工作区的 CPU 数值回归校验 |

训练／验证／测试共用 epoch 循环，训练与推理共用数据准备；各折独立保存结果。
ESM 字典按需加载并跨折缓存。`utils.AA_CODES` 保留给既有预处理脚本使用。
已移除未使用的旧数据集实现和运行分支；运行产物不纳入版本控制。

## 环境与数据

已验证的运行环境：Python 3.7.16、PyTorch 1.12.1+cu113、NumPy 1.21.5、
scikit-learn 1.0.2，以及 `torch_scatter`、`einops`、`tqdm`。
表面预处理与 ESM 生成另需各自脚本的依赖。

训练输入为预处理表面文件和预计算 ESM 特征，不直接读取原始 PDB。
这些大文件和训练权重未包含在 Git 仓库中。数据目录应包含：

```text
DATA_ROOT/
├── pt_all_128_12_1/
│   ├── RNA-663_Train/   # 实际处理得到 660 个文件
│   └── RNA-157_Test/    # 实际处理得到 150 个文件
└── esm/
    ├── RNA-663_Train    # pickle 字典，按输入文件名索引
    └── RNA-157_Test
```

默认 `DATA_ROOT` 是仓库目录的上一级；训练时也可通过 `--data-root` 显式指定。
项目工作区的数据位于 `/public/home/maxy2022/prorna/new_24/nestnabind`。

## 训练

在仓库根目录、已配置的 Python 环境中运行：

```bash
python train_cv.py --taskname RNA --taskbase site --num_sampling all \
  --num_neighbors 128 --split_ratio 0.8 --seed 0 --fold 0 --epochs 50 \
  --data-root /path/to/DATA_ROOT
```

不传 `--fold` 时运行五折。每折输出到 `logs/<时间>_site_RNA/fold_<编号>/`，
包含 `split.json`、`out.csv`、`log.txt`、普通／最佳模型、测试预测和代码快照。
`out.csv` 使用完整指标名称，如 `val_auroc`、`test_precision`。
原始残基标签口径的附加指标使用 `legacy_` 前缀。

为了保持与旧实验可比较，训练沿用固定的 Python shuffle 种子 123，
`--seed` 仍是旧版元数据，每折仍重新打乱文件再切片；这些验证集并非严格互斥。
训练最大采样 2048 patches，验证／测试最大采样 8000 patches。
早停根据验证 AUROC，连续 10 次未改善时停止；连续 3 次未改善时降低学习率。

## 推理

```bash
python inference.py --checkpoint /path/to/Bestmodel_Epoch0.pth \
  --data-dir /path/to/DATA_ROOT/pt_all_128_12_1/RNA-157_Test \
  --esm-file /path/to/DATA_ROOT/esm/RNA-157_Test \
  --protein 1b2m_A_C --threshold 0.18 --output-dir predictions/example
```

不传 `--protein` 即处理整个目录。`--device auto` 自动选择 CUDA 或 CPU。
`--evaluate` 计算有标签输入的指标；无标签预测不需要标签、序列或法向量。
阈值默认 0.5，应按所用 checkpoint 的验证结果指定，不在测试数据上调参。

输出 `residue_predictions.csv` 和 `summary.json`，包含原始残基编号、
结合概率、预测类别和覆盖情况；未覆盖残基的概率／类别留空。
按 patch 第一个点对应的残基做最大正类概率汇总，与训练损失的口径一致。

## PBS 测试与验证记录

从仓库根目录提交：

```bash
qsub run_test.pbs
# 或提交最多 50 epoch 的训练：
qsub run_train_50.pbs
```

PBS 文件使用本项目集群的队列、RTX 3090 和 Python 环境路径。
在其他环境使用前，应调整队列、资源与环境路径，并按上面的数据布局准备数据。
单轮测试结束时会检查权重重新加载和预测数据，再输出成功标记及 `verification.json`。

- [CPU 重构校验](VERIFICATION.md)：旧权重兼容、模型输出与梯度、数据及损失对比。
- [单轮 GPU 测试](GPU_TEST_RESULT.md)：完整的 528／132／150 划分，退出码 0。

`verify_refactor.py` 供原工作区回归使用：它依赖上级目录的旧脚本、真实数据、
一轮训练权重和推理基线。普通克隆没有这些大文件，应先准备相应基线。
原工作区的详细运行日志和权重保存在工作区，不包含于仓库。
