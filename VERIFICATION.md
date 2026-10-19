# 重构校验记录

校验环境：现有 Python 3.7.16 / PyTorch 1.12.1+cu113 环境，使用 CPU 和 4 个线程。
原工作区的校验日志保存在 `nestnabind/optimized/verification.log`，不纳入 Git 仓库。

## 代码规模

| 模块 | 原版行数 | 优化版行数 |
|---|---:|---:|
| Arguments.py | 127 | 31 |
| dataloader.py | 317 | 146 |
| vit_esm.py | 275 | 127 |
| utils.py | 297 | 131 |
| train_cv.py | 248 | 123 |
| inference.py | 263 | 138 |
| provider.py（没有被调用） | 251 | 删除 |
| 合计 | 1778 | 696 |

行数包含注释和空行；新校验脚本单独计算，不计入运行模块。
表格对应原工作区的优化版；本仓库的 `utils.py` 另保留既有预处理脚本依赖的 `AA_CODES` 常量。

## 通过的检查

1. 严格加载旧版 `Bestmodel_Epoch0.pth`，参数名称、顺序及形状兼容。
2. 固定随机状态后，模型在 eval 和 train 模式下的前向输出与旧版逐元素完全一致。
   所测参数梯度最大绝对差为 `5.960464477539063e-08`，
   通过 `rtol=1e-5, atol=1e-6` 的浮点比较。
3. 真实文件 `1b2m_A_C`（1455 patches）与 `1fje_B_A`（训练时 FPS 到 2048 patches）
   的特征、坐标、ESM、patch 索引和多数残基映射与旧版完全一致。
   两套指标对应的概率和标签、损失及损失对 logits 的梯度均完全一致。
4. 优化版推理 CLI 处理 `1b2m_A_C`、`1fje_B_A`：
   输出 94 + 158 = 252 行残基数据，与原版 `predictions/smoke_labeled/`
   的 CSV 每个字段完全一致，所有评估指标也一致。
   新输出保存在 `optimized/predictions/verification/`。
5. 删除标签、法向量和序列后仍可构造相同模型输入，支持无标签推理。
6. 真实蛋白质采样到 96 patches 后，完成 site 分支的一轮训练／验证／测试，
   输出 CSV、普通与最佳权重、测试预测，并成功重新加载权重。
7. 使用临时小数据集，从其他工作目录调用训练 CLI，验证 `--data-root`、
   3／1／2 个训练／验证／测试样本的划分、`split.json` 和独立 `fold_0` 目录。
8. patch 分类分支也完成了一轮训练／验证／测试，并保存最佳模型。
9. 最佳 epoch 为 0、随后三轮未改善时，学习率正常降到 `0.0001` 并重新加载 epoch 0。
10. 所有 Python 文件通过编译检查，PBS 脚本通过 `bash -n` 检查。

上述记录对应 CPU 重构校验。随后已通过完整数据的单轮 GPU 测试，见 `GPU_TEST_RESULT.md`。
没有进行完整 50 epoch 的性能或收敛比较；CPU 数值校验不代表所有设备上的位级一致性。
原目录代码及作业 `11176452.node1` 未修改。
