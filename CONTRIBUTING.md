# 贡献指南（CONTRIBUTING）

感谢你愿意为 VMLM 贡献代码。本文说明**如何构建 / 如何跑测试 / 以及为什么
必须签 CAA**。

> **一句话版：未签 CAA 的 PR 不会被合并。** 见 [§4](#4-caa必须签) 与
> [CAA.md](CAA.md)。

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

## 4. CAA：必须签

### 4.1 规则

> **未签 CAA 的 PR 不会被合并。**

这不是形式主义。本项目以 **MPL-2.0（Mozilla Public License 2.0，逐文件 copyleft）**
发布，同时维护者保留对**自有代码及已签 CAA 的贡献**进行**双许可 / 商业授权**的
权利。若无 CAA 中的**著作权转让**与**再许可（换证）授权**，维护者就**无法**
合法地提供商业许可——这会直接影响项目的可持续性。

> **唯一性：** 本项目**只**使用 [CAA.md](CAA.md)（贡献者著作权转让协议）这一份
> 贡献授权文件；本仓库从未使用过 Contributor License Agreement（CLA）或其它
> 贡献者协议。CAA 的生效状态见其顶部声明。

### 4.2 为什么不接受仅 DCO / `Signed-off-by`

`Signed-off-by:` 行（DCO）只证明**来源合规**——即"我有权提交这段代码"。
它**不构成著作权转让**，也**不授予**维护者任何**再许可权（sublicense）
或换证权（relicense）**。

因此本项目**两者都要**，但**CAA 才是转让权利的那一份**。详见
[CAA.md §11](CAA.md)。

### 4.3 怎么签

在 Pull Request 的描述或评论中**显式声明**（这是目前唯一的签署方式）：

**English version（推荐，须原文包含下列句子）：**

```
I assign the copyright in my Contribution to the VMLM Maintainer under CAA.md (including the license-back in §3).
Signed-off-by: Your Name <your@email.com>
```

**中文版：**

```
我依 CAA.md 将我的 Contribution 的著作财产权转让给 VMLM 维护者
（包括 CAA.md 第 3 条的回授许可）。
Signed-off-by: 你的姓名 <你的邮箱>
```

要点：

- **每个 PR 都要有**；多人贡献时**每位**贡献者各自签署。
- `Signed-off-by` 用真实姓名与真实邮箱，与 Git 提交 author 一致。
- 签署须**真实**：姓名、邮箱、日期；以法人 / 机构名义签署时另需**实体全称、
  职务与实体类型**（见 [CAA.md §11.2](CAA.md)）。
- 若贡献在**雇佣范围内**完成，需取得**雇主书面同意**，或由雇主作为转让方签署。
- **在正式对外收费 / 商业授权前，CAA 模板须由执业律师审阅**（见 CAA.md 顶部
  声明，特别提示 §7 精神权利与 §8 回退条款）。

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

## 6. Pull Request 流程

1. Fork 仓库并从 `main` 拉出特性分支。
2. 完成改动，跑通 §3 的自检清单。
3. 提交 PR，描述中写明：
   - 改了什么、为什么；
   - 构建与测试结果（exit code、体积数字）；
   - **CAA 签署声明 + `Signed-off-by` 行**（见 §4.3）。
4. 等待维护者评审。维护者**无义务**接受任何 Contribution，可自行决定
   接受、拒绝、修改或部分采用（见 [CAA.md §10](CAA.md)）。

---

## 7. 许可证

本项目以 **MPL-2.0** 发布（见 [LICENSE](LICENSE) 与 [LICENSE-NOTICE](LICENSE-NOTICE)）。

提交 Contribution 即表示你同意 [CAA.md](CAA.md) 的全部条款，并确认你的
Contribution 将以 **MPL-2.0** 向公众提供，同时授权维护者依 CAA 第 2 条取得
著作财产权、依第 4 条进行再许可 / 换证（含将本项目整体或部分改以其它许可发布，
以及提供商业许可）。
