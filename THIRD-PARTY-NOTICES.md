# 第三方与上游材料通知（Third-Party Notices）

本仓库的 [LICENSE](LICENSE)（**MPL-2.0**）覆盖由著作权人 **金施哲 (Jin Shizhe)**
持有著作权的部分。下列材料另有其来源与条款，**不因本仓库的 MPL-2.0 而改变**。

---

## 1. 上游模型权重：**本仓库不分发**

`examples/minicpm_full/` 与 `examples/minicpm_24L_ns/` 只包含**本项目自己生成**的产物 ——
指令流 `full.vml`、汇编清单 `weights.args`、配置与日志 `meta.json` / `build_log.txt`。
这些内容按 MPL-2.0 发布。

**上游模型的权重张量（`weights/`）没有随本仓库分发。** 它们来自上游 MiniCPM 模型，
条款与本仓库不同（见下），因此仅保留在维护者的私有工作副本中。

### 上游信息

| 项 | 说明 |
|---|---|
| 上游模型 | **MiniCPM**（面壁智能 / ModelBest / OpenBMB）；本地构建使用 `MiniCPM5-1B` 配置 |
| 上游条款 | **Apache-2.0 +《MiniCPM 模型社区许可协议》** |
| 开源/商用 | 权重对**学术研究完全开放**；企业、个人**填写问卷登记后**允许商业使用 |
| 来源（公告） | <https://kw.beijing.gov.cn/xwdt/kcyx/xwdtyqqy/202406/t20240606_3821895.html> |

**若你要复现本仓库中"引擎 vs HuggingFace 数值对齐"的结论**：请自行从上游取得
MiniCPM 权重，并遵守其社区许可 —— 特别是**商业使用前需按上游要求完成登记**。
本仓库提供全部工具链（生成器、汇编器、校验脚本），见
[`examples/minicpm_full/README.md`](examples/minicpm_full/README.md)。

---

## 2. 自有但独立的作者项目：ESERNN

`examples/eserNN/`、`tools/gen_eserNN.py`、`tools/eserNN_*.py`、
`train/src/eserNN_*.rs` 涉及的 **ESERNN（Existential State Engine）** 是本项目作者
**金施哲** 自己的独立项目，著作权与 VMLM 同属一人。因此这些移植与对照产物
**随本仓库按 MPL-2.0 发布，不涉及任何第三方许可**。

---

## 3. Rust 依赖

本项目是 Rust workspace；依赖的许可证可在本地枚举：

```powershell
cargo metadata --format-version 1 --all-features |
  ConvertFrom-Json | Select-Object -ExpandProperty packages |
  Select-Object name, version, license | Sort-Object name |
  Export-Csv -NoTypeInformation -Encoding utf8 deps-licenses.csv
# 或（若已安装）： cargo deny check licenses
```

依赖均以其各自许可证分发；本仓库不对它们作任何额外授权或担保。

---

## 4. 若你认为有遗漏

如果你发现本仓库包含了应当在此声明的第三方材料，请开 issue 说明
（文件路径 + 上游来源 + 许可），我们会补上声明或在必要时移除该材料。

---

## 5. 上游材料的声明模板

若日后本仓库需要**重新纳入**第三方材料（例如按上游许可重新分发 MiniCPM 权重），
请把逐字保留的声明放在本节：

```
================================================================
<上游项目名> — <版本 / 提交>
<上游版权声明原文（逐字保留，不得改写）>
Licensed under <SPDX 标识>. Full text: <URL 或本仓库路径>
本仓库对它的使用方式：<移植 / 权重提取 / 参考对照>
================================================================
```

并同时在该材料所在目录的 `README.md` 顶部写明来源、版本、许可与衍生关系。
