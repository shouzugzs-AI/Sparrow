# examples/minicpm_full —— MiniCPM5-1B 全模型对照（**不含上游权重**）

## 1. 本目录里有什么

| 文件 | 是什么 |
|---|---|
| [full.vml](full.vml) | 由 `tools/p12_build_minicpm.py` 生成的 24 层前向指令流（4,293 行） |
| [weights.args](weights.args) | 汇编器的权重注入清单（222 个常量，**相对路径**，须先 `cd` 到本目录） |
| [meta.json](meta.json) | 架构超参（`layers=24 hidden=1536 heads=16 kv_heads=2 head_dim=128 inter=4608 vocab=130560 theta=5e6`）与本次对照的 prompt token |
| [build_log.txt](build_log.txt) | 生成 + 汇编过程的原始日志（含产物体积、section read-back 校验） |
| ~~`weights/`~~ | **未随本仓库分发** —— 见下一节 |

## 2. 为什么这里没有 `weights/`

`weights/` 里那些张量（`L0_in_ln.f32` 等，6144 B = 1536 × f32）是**从上游 MiniCPM
模型提取出来的权重**，不是本项目的作品。

上游许可是 **Apache-2.0 +《MiniCPM 模型社区许可协议》**：面壁智能（ModelBest /
OpenBMB）在 2024-06 公告中说明，MiniCPM / MiniCPM-V 权重对**学术研究完全开放**，
企业、个人**填写问卷登记后**亦允许商业使用。
来源：<https://kw.beijing.gov.cn/xwdt/kcyx/xwdtyqqy/202406/t20240606_3821895.html>

这**与本仓库的 MPL-2.0 不是同一套条款**（而且它额外带一个"商用需登记"的条件）。
为了不在一个 MPL-2.0 仓库里混装另一套许可，这些权重已**移出公开仓库**，
仅保留在维护者的私有工作副本中。

> 本目录其余内容（`full.vml` / `weights.args` / `meta.json` / `build_log.txt`）
> 全部是**本项目自己的产物**，按 **MPL-2.0** 发布。
> 第三方清单见开源发布时随仓库提供的 `THIRD-PARTY-NOTICES.md`。

## 3. 想复现这个对照，需要自备输入

1. 从上游取得 MiniCPM5-1B 权重（按其社区许可；**商用请先按上游要求登记**）；
2. `python tools/p12_build_minicpm.py`（用法见脚本头部）→ 在本目录重建 `weights/`；
3. 汇编 `full.vml`：`weights.args` 是相对路径清单，须先 `cd examples/minicpm_full`；
4. 与 HuggingFace 对齐：`python tools/p12_hf_ref.py --full` 生成 `hf_logits.npy`，
   再跑 `tools/p12_logits_check.py`。
   （`hf_logits.npy` 是**本机测量产物**，体积大且同样是上游模型的输出，因此也不入库。）

## 4. 已验证的结论（方法都在本仓库，只是输入权重需自备）

在维护者机器上实测：24 层前向 + final norm + lm_head 与 HuggingFace
**argmax 4/4 全同**、相对误差 **1.88e-06**；产物体积与 `build_log.txt` 记录一致。

复现该结论所需的**全部工具链**（生成器、汇编器、校验脚本）都在本仓库；
缺的只有"上游权重 + 本机生成的 HF 参考"这两项**非本项目产物**的输入。

## 5. 许可

- 本目录自有内容：[MPL-2.0](../../LICENSE)（仓库根 `LICENSE`）。
- 上游 MiniCPM 权重（**未分发**）：Apache-2.0 +《MiniCPM 模型社区许可协议》。
- 本目录**不**包含任何上游权重文件；`weights/` 已被 `.gitignore` 排除，
  以防从私有工作副本同步时被误加回来。
