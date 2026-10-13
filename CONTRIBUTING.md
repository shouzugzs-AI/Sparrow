# 贡献指南（CONTRIBUTING）

本文说明**如何构建、如何跑测试**，以及**当前的贡献政策**。

> **一句话版：本仓库当前不合并外部 Pull Request。** 欢迎开 issue 报告问题；
> fork 与自行修改完全允许（MPL-2.0）。见 [§4](#4-贡献政策当前暂不合并外部-pull-request)。

---

## 1. 先决条件

| 项目 | 要求 |
| --- | --- |
| 平台 | Windows x86_64（`x86_64-pc-windows-msvc`） |
| CPU | 支持 **AVX2 + FMA** |
| 工具链 | Rust（MSVC toolchain），`stable` 即可 |
| Target | `x86_64-pc-windows-msvc` |

> 本项目是**纯 CPU** 实现，不需要 GPU，也不需要 CUDA。

---

## 2. 如何构建

### 2.1 全 workspace（排除引擎）

```bat
cargo build --release --workspace --exclude vmlm-runtime
```

### 2.2 引擎（**必须显式指定 `--target`**）

```bat
cd runtime
cargo build --target x86_64-pc-windows-msvc --release
```

> ⚠️ **引擎那条必须带 `--target`**，否则会报 `E0152`。
> 原因是 `vmlm-runtime` 是 `no_std` + `/NODEFAULTLIB`，链接参数由
> `runtime/.cargo/config.toml` 提供，只有在该 target 下才生效。

### 2.3 产物体积不变量（重要）

修改**不得**改变引擎产物的体积。当前已验证的基线：

| 产物 | 体积 |
| --- | --- |
| `target\x86_64-pc-windows-msvc\release\vmlm-runtime.exe`（引擎） | **75,776 B** |
| `vmlm-tests.exe`（测试） | **187,392 B** |
| `vmlm-b1-full.exe`（B1 完整训练） | **627,712 B** |

如果你的改动让 `vmlm-runtime.exe` 的体积发生了变化，请先确认**这是有意的**，
并在 PR 中明确说明原因与新的体积数字。**体积变化不能默认当成"正常"。**

---

## 3. 如何跑测试

### 3.1 验收与冒烟测试套件

```bat
target\release\vmlm-tests.exe
```

### 3.2 逐 crate 单元测试

```bat
cargo test --release --workspace --exclude vmlm-runtime
```

### 3.3 快速自检清单

提交 PR 之前，请至少确认：

1. `cargo build --release --workspace --exclude vmlm-runtime` → **exit 0**
2. `cd runtime && cargo build --target x86_64-pc-windows-msvc --release` → **exit 0**
3. `target\release\vmlm-tests.exe` → 无 FAIL
4. `vmlm-runtime.exe` 体积仍为 **75,776 B**（或有 PR 中说明的理由）

---

## 4. 贡献政策：当前暂不合并外部 Pull Request

### 4.1 规则

> **本仓库当前不合并外部提交。** Pull Request 会被关闭，并引导到 issue。

原因很直接：**审阅并合并外部代码需要一套贡献者协议**（它决定那段代码的著作权归属
与再许可权），而维护者目前没有精力去起草、审阅并长期维护这套流程。
在没有精力之前，维护者选择**只维护自己写的代码**，而不是用一份未经审阅的协议去收贡献。

### 4.2 你仍然可以做什么

| 想做的事 | 是否可以 | 怎么做 |
|---|---|---|
| 使用 / 修改 / 分发 / 商用 | ✅ **允许** | MPL-2.0 已授予这些权利，见 [LICENSE](LICENSE) |
| fork 后自己继续开发 | ✅ **允许** | 无需任何人同意 |
| 报告缺陷、提改进建议 | ✅ **欢迎** | 开 issue |
| 提交 Pull Request | ❌ **暂不处理** | 会被关闭；请改为开 issue 说明问题 |

### 4.3 将来若开放外部贡献

维护者会**先在本仓库公布届时生效的贡献者协议与签署方式**，再合并第一份外部 PR。
**在此之前，本仓库不存在任何贡献者协议** —— 既不需要签 CAA，也不需要 DCO，
你提交的任何内容都不会被合并进本仓库。

> **历史说明（事实陈述）：** 本仓库曾起草过一份「贡献者著作权转让协议（CAA）」，
> 但它**从未生效、也从未被任何人签署**，因此现已**不再随本仓库提供**。
> 本仓库自始至终**没有**使用过其它贡献者协议（如 CLA）。
> 若将来需要，会先把它定稿并由执业律师审阅后，再重新引入。

---

## 5. 代码约定

1. **不要顺手重构。** PR 应聚焦于单一主题；无关的格式化、改名、移动文件
   请拆成独立的 PR。
2. **不要改动 `.rs` 的业务逻辑**去"顺带"修正别的东西；每处改动都应有
   明确理由。
3. **保持体积不变量**（见 §2.3）：`vmlm-runtime` 的体积是被**测量**的指标，
   不是被**断言**的。
4. **文档与实测一致**：文档中出现的数字应当是**实测**结果，而不是估算或
   预期值。若你更新了某个指标，请同时给出测量方式。
5. **许可证头**：本仓库**不**为每个源文件添加许可证头。许可证与授权条件的统一
   位置是 [LICENSE](LICENSE) 与 [LICENSE-NOTICE](LICENSE-NOTICE)（**MPL-2.0**，
   依 Exhibit A 集中通知）。新文件若确实需要头部声明，请使用
   [SOURCE-HEADER.txt](SOURCE-HEADER.txt) 的三行标准文本。
   请**不要**在源文件中插入与 LICENSE 冲突的权利声明（例如把本项目声明为专有软件）。
6. **不要提交权重 / 词表**：`.cpk` 等检查点不入库（见 `.gitignore`）。

---

## 6. Pull Request 流程（当前：不合并）

> **本节仅在将来重新开放外部贡献后适用。当前提交的 PR 会被关闭并引导到 issue**
> （原因见 §4.1）。

1. Fork 仓库并从 `main` 拉出特性分支。
2. 完成改动，跑通 §3 的自检清单。
3. 提交 PR，描述中写明改了什么、为什么，以及构建与测试结果（exit code、体积数字）。
4. 等待维护者评审。维护者**无义务**接受任何 Contribution。

---

## 7. 许可证

本项目以 **MPL-2.0** 发布（见 [LICENSE](LICENSE) 与 [LICENSE-NOTICE](LICENSE-NOTICE)）。

**本仓库当前不接收外部 Contribution**（§4），因此不存在贡献者协议。
任何人仍可依 MPL-2.0 使用、修改、分发本项目，或 fork 后自行继续开发 ——
这些都不需要维护者的额外许可。
