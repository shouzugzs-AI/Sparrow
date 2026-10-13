# examples/minicpm_24L_ns —— MiniCPM5-1B 24 层 / 无 section 变体（**不含上游权重**）

本目录是 `examples/minicpm_full/` 的同构变体（`_ns` = 不带 section 的镜像），
用于对照 section 追加前后的行为与体积。

## 内容

| 文件 | 是什么 |
|---|---|
| [full.vml](full.vml) | 由 `tools/p12_build_minicpm.py` 生成的 24 层前向指令流 |
| [weights.args](weights.args) | 汇编器权重注入清单（相对路径，须先 `cd` 到本目录） |
| [meta.json](meta.json) | 架构超参与 prompt token |
| ~~`weights/`~~ | **未随本仓库分发** —— 原因同 `examples/minicpm_full/README.md` 第 2 节 |

## 为什么没有 `weights/`

`weights/` 里的张量是**从上游 MiniCPM 模型提取的权重**，受
**Apache-2.0 +《MiniCPM 模型社区许可协议》**约束（学术研究完全开放；商用需按上游
填写问卷登记），与本仓库的 **MPL-2.0** 不是同一套条款。
为避免许可混装，这些文件已移出公开仓库，仅保留在维护者的私有工作副本中。

复现方法（自备上游权重）见 [../minicpm_full/README.md](../minicpm_full/README.md) 第 3 节。

## 许可

- 本目录自有内容：[MPL-2.0](../../LICENSE)。
- 上游 MiniCPM 权重（**未分发**）：Apache-2.0 +《MiniCPM 模型社区许可协议》。
- `weights/` 已被 `.gitignore` 排除，防止同步时被误加回来。
