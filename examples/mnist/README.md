# P3：MNIST 训练示例

本目录是 P3 的最小训练闭环示例（`196 → 64 → 4`，14×14 下采样、4 类），
由 `train/src/eserNN_run.rs` / `runtime/build.rs`（`aot-train` 特性）读取。

## 数据需自行下载/生成

数据集切片 `Xte.bin`（测试集，约 1.94 MB）**不入库**，需要自己生成：

```bash
python examples/mnist/make_mnist.py
```

该脚本会：

1. 若装有 `torchvision`，下载 MNIST 到 `examples/mnist/mnist_data/`
   （`datasets.MNIST(..., download=True)`），做 2×2 均值下采样到 14×14，取前 4 类；
   否则退回**可分的合成数据**（保证仍能学会，便于离线复现）。
2. 划分 9:1 训练/测试，写出全部 `.bin` 与 `meta.txt`。
3. 用 PyTorch 跑 200 步参考训练，写出 `ref_loss.txt`。

## 目录内容

| 文件 | 作用 |
|---|---|
| `make_mnist.py` | 生成数据、初始权重与 PyTorch 参考 loss |
| `train_step.vml` | 训练步指令流（前向 + 反向 + AdamW） |
| `eval_step.vml` | 评测步指令流 |
| `meta.txt` | `训练样本数 测试样本数 IN H NCLS` |
| `X.bin` `Y.bin` | 训练集输入 / 标签（f32） |
| `W1.bin` `b1.bin` `W2.bin` `b2.bin` | 初始权重（He 初始化） |
| `x0.bin` `y0.bin` | 单样本输入 / 标签 |
| `Yte.bin` | 测试集标签（**保留**：`runtime/build.rs` 的 `VMLM_AOT_DATA` 默认表需要它） |
| `ref_loss.txt` | PyTorch 参考 loss 曲线（200 步） |
| `_vmlm_grads.bin` | 梯度 dump（`train/src/graddump.rs` 产出，`tools/gradcheck_vs_torch.py` 消费） |

> `Xte.bin` 已从仓库移除（体积换来的收益不值得）。它只被 `XTE=...` 这一条
> AOT 数据表项引用；跑 `make_mnist.py` 即可重新生成。
