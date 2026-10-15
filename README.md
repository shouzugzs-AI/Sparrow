# VMLM

> 项目代号 **VMLM**；GitHub 仓库名 **Sparrow**（`shouzugzs-AI/Sparrow`）。

**VMLM** 是一个用 Rust 写的、纯 CPU 的张量虚拟机与模型工程。它由「极简张量虚拟机 / 引擎」和「可训练的神经网络栈」两部分组成：底层是一个不依赖标准库、把权重当作程序来装载执行的 VM，上层是一套六层抽库的 Rust workspace，同时承载**稠密**与 **B1/B2 稀疏**两条建模路径。

本仓库是 VMLM 的**全量副本**（**开源，MPL-2.0**，见 [LICENSE](LICENSE)）：稠密路径与 B1/B2 稀疏路径的代码都在这里。

- 许可：**Mozilla Public License 2.0（MPL-2.0，开源）** —— 见 [LICENSE](LICENSE) 与 [LICENSE-NOTICE](LICENSE-NOTICE)。可自由使用、修改、分发（**包括商业使用**）；对 MPL 覆盖文件的**修改**须以 MPL 公开
- 贡献：**当前暂不合并外部 Pull Request**（欢迎开 issue）；详见 [CONTRIBUTING.md](CONTRIBUTING.md)
- 平台：Windows x86_64（`x86_64-pc-windows-msvc`），AVX2 + FMA
- 文档索引：[docs/文档索引.md](docs/文档索引.md)；架构总览见 [docs/VMLM架构白皮书.md](docs/VMLM架构白皮书.md)

---

## 1. 架构：六层抽库

依赖方向是单向的，从最底层的 `sys` 一路抽到最上层的 `models`，再由 `core` 组装成 VM：

```
L0  vmlm-sys        系统层（分配、I/O、平台原语）
L1  vmlm-tensor     张量与内存布局
L2a vmlm-ops        算子（稠密内核 + 稀疏连接算子）
L2b vmlm-trans      转译
L3  vmlm-blocks     网络构件（层/块）
L4  vmlm-models     模型组装（含 B1 稀疏模型）
    vmlm-core       VM（把上面几层组装成可执行的计算图）
```

围绕这六层，还有若干工程 crate：

| crate | 作用 |
| --- | --- |
| `sys` → `tensor` → `ops` / `trans` → `blocks` → `models` → `core` | 六层抽库主体 |
| `runtime` | 引擎（`vmlm-runtime.exe`），`no_std`，权重即程序 |
| `train` | 训练器（稠密 / B1 稀疏 / 单文件 chat 打包等） |
| `tests` | 验收与冒烟测试（`vmlm-tests.exe`） |
| `asm` | 汇编器 |
| `pt2vmb` | 外部格式 → `.vmb` 转换 |
| `server` | P17 起的外部 HTTP / CLI crate（独立 member，可用 std） |

`server` 被刻意拆成**独立的 workspace member**：`vmlm-runtime` 是 `no_std` + `/NODEFAULTLIB`，把需要 std、要链接 winsock 的 HTTP 侧混进同一个 crate graph 会破坏引擎的链接参数，因此两者分开。

---

## 2. 两条路径：稠密与 B1/B2 稀疏

### 稠密

常规的稠密张量计算，位于 `ops::kernels`：GEMM、attention、norm、loss 等。

### B1 / B2 稀疏

稀疏路径的代码分布：

| 位置 | 内容 |
| --- | --- |
| `ops::conn` | 连接（稀疏拓扑相关算子） |
| `ops::b1_pure` | B1 的纯稀疏算子 |
| `core::conn_sparse` | 连接稀疏 |
| `core::conn_train` | 连接稀疏的训练侧 |
| `core::b1_pure` | B1 纯稀疏核心 |
| `core::b2_topk` | B2 的 top-k 重选 |
| `core::sparse` | 稀疏公共部分 |
| `models::b1` / `models::head` / `models::attn_sparse` / `models::model` | B1 模型组装、head、稀疏 attention |

**稀疏机制**：每个神经元固定保留 `k0` 条连接；训练/演化过程中按连接强度做 **top-k 重选**，把弱连接换成强连接；连接表按**块对齐**组织，以便内核连续访问。也就是说，稀疏体现在**拓扑**上（谁连谁、连多少条），而不是靠把张量置零来省算力。

---

## 3. 构建

> 以下命令均已验证。**引擎那条必须带 `--target`**，否则会报 `E0152`。

全 workspace（排除引擎）：

```bat
cargo build --release --workspace --exclude vmlm-runtime
```

引擎（必须显式指定 target）：

```bat
cd runtime
cargo build --target x86_64-pc-windows-msvc --release
```

跑测试：

```bat
target\release\vmlm-tests.exe
```

### 已验证的产物体积

| 产物 | 体积 |
| --- | --- |
| `vmlm-runtime.exe`（引擎） | **75,776 B** |
| `vmlm-tests.exe`（测试） | **187,392 B** |
| `vmlm-b1-full.exe`（B1 完整训练） | **627,712 B** |
| 单文件 chat（内嵌 9.2 MB 权重时） | 约 **9.98 MB** |

---

## 4. 权重从哪来（重要）

**本仓库不含权重、也不含词表**：`.cpk` 检查点不在版本库内（见 [.gitignore](.gitignore)）。请按下面两种方式之一取得权重。

### 4.1 默认行为：占位权重，能编译但不能对话

`train/build.rs` 在找不到权重时会**生成占位文件**，这样整个 workspace **仍然能编译通过**。此时构建出来的 `vmlm-chat.exe` 只有约 **136 KB**，启动后会打印「**内嵌权重格式不对**」并退出——这是预期行为，不是构建失败。

### 4.2 构建可用的 chat

任选其一：

1. 自己训练出 `.cpk`，放进 `artifacts/`；
2. 用环境变量显式指定：

```bat
set VMLM_MODEL=<path\to\model.cpk>
set VMLM_VOCAB=<path\to\vocab>
```

---

## 5. Quick start（已验证的冒烟）

### 5.1 稠密对照

```bat
vmlm-sparseb-dense24.exe --d 512 --layers 8 --batch 16 --seq 64 --steps 100 --lr 3e-4
```

实测结果：`checks=6 fail=0`，train loss **4.160010 → 2.635886**。

### 5.2 B1 稀疏 100 步

```bat
vmlm-b1-full.exe --sparse-io 1 --dim 64 --k0 4 --k0-embed 16 --k0-head 16 --k0-attn 4 ^
  --layers 4 --no-residual 1 --head-skip 1 --attn-every 2 --attn-sparse 1 ^
  --attn-src-shared 1 --attn-calib 1 --attn-window 1 ^
  --seq 16 --batch 2 --steps 100 --threads 1 --data-dir <你的训练数据目录>
```

实测结果：100 步后 `loss_hash` 稳定为 **`0x5b9545a6ee554f51`**（可复现的确定性指纹）。注意需要 `--data-dir` 指向训练数据。

---

## 6. 仓库结构

```
vmlmd/
├─ sys/ tensor/ ops/ trans/ blocks/ models/ core/   六层抽库
├─ runtime/     引擎（no_std）
├─ train/       训练器（含 build.rs 的权重占位逻辑）
├─ tests/       验收与冒烟
├─ asm/         汇编器
├─ pt2vmb/      外部格式 → .vmb
├─ server/      HTTP / CLI
├─ docs/        设计与实测文档（见 docs/文档索引.md）
├─ tools/       辅助脚本
├─ artifacts/   训练产物（不入库）
├─ sparrow/     描述语言（DSL：源码 → .spar → .vml → .vmb；另有 Rust AOT 后端）
└─ Cargo.toml   workspace 定义
```

### 6.1 描述语言 Sparrow（`sparrow/`）

`sparrow/` 是本项目的**神经网络架构描述语言**（自带 `[workspace]`，**不是**引擎 workspace 的
member —— 加它不会进引擎的构建/链接面）：

```sparrow
模型 attncausal：
  线性 输入=4 输出=4 序列=4
  注意力 维度=4 头=2 序列=4 因果=是
  连接 线性 注意力
  前向 线性 注意力
```

* 两条后端：`.vmb`（交给本仓库的 `asm/` 与 `runtime/`）与 **Rust AOT**
  （`--backend lib --lang rust`，生成的 crate 静态链接本仓库的 `vmlm-ops`，所以与 `.vmb` 逐字符一致）；
* 训练路径：`损失 均方` + `反向`（按拓扑反序发射 `MSE_BWD` / `GEMM_BWD_W` / `GEMM_BWD_X` /
  `BIAS_BWD` / `ACT_BWD`，并把每个参数的梯度打印出来）；
* 零第三方依赖；`cargo test` **89 项**；一条命令跑完全部验收（11 个驱动 + "跑完工作区干净"不变量）：

```powershell
cd sparrow
cargo test --locked
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\verify_all.ps1   # 退出码 0 = 全绿
```

**想直接上手语言**：[`sparrow/docs/使用指南.md`](sparrow/docs/使用指南.md) —— 16 节：语法全表、
每种层与属性（含哪些是静默 no-op）、命令行、诊断码、9 条可跑配方、十个最常撞的坑。

入口在 [`sparrow/README.md`](sparrow/README.md)（语言速览 / 两个后端 / 已知边界），
变更记录见 [`sparrow/CHANGELOG.md`](sparrow/CHANGELOG.md)，实测踩过的坑见
[`sparrow/docs/TRAPS.md`](sparrow/docs/TRAPS.md)，逐项证据见
[`sparrow/REPORT_V015.md`](sparrow/REPORT_V015.md)。
工具会**自动**在"上一级就是引擎仓库根"的布局下推导 `--vmlm-root`，不需要额外参数。

---

## 7. 许可证与授权

### 7.1 许可证：Mozilla Public License 2.0（MPL-2.0）

本项目**自本版本起**以 **Mozilla Public License 2.0（MPL-2.0）**发布 —— 正文见
[LICENSE](LICENSE)，Exhibit A 通知见 [LICENSE-NOTICE](LICENSE-NOTICE)。
MPL-2.0 是 OSI 认可的开源许可证：你可以自由使用、修改、分发本项目，**包括商业使用**。

- `LICENSE` 为 MPL-2.0 **官方原文**（16,726 B，SHA-256
  `FAB3DD6BDAB226F1C08630B1DD917E11FCB4EC5E1E020E2C16F83A0A13863E85`），未作任何改写；
  官方地址 <https://mozilla.org/MPL/2.0/>。
- 著作权人：**金施哲**（GitHub 账号 `shouzugzs`）。版权声明：
  `Copyright (c) 2026 金施哲 (Jin Shizhe)`。
- **逐文件 copyleft**：对 MPL 覆盖文件的**修改**必须以 MPL 公开；与其它代码组成的
  Larger Work 可按你自己的条款分发（§3.3）。
- 本仓库**未附加** MPL-2.0 Exhibit B 通知，故**不属于** "Incompatible With Secondary
  Licenses"，可组合进 GPL 等 Secondary License。
- 本仓库**不**为每个源文件添加许可证头；依 Exhibit A，统一通知位于
  [LICENSE-NOTICE](LICENSE-NOTICE)。

### 7.2 是否需要商业许可

MPL-2.0 已允许商业使用，**通常不需要**另行取得商业许可。以下情况可联系维护者：

> **GitHub：[@shouzugzs](https://github.com/shouzugzs)** —— 通过仓库 issue 或 GitHub
> 私信联系；需要正式书面联系（合同、发票、OEM 授权书）时请在其中说明，维护者会提供
> 专用联系渠道。

- 希望以**非 MPL 条款**使用对 MPL 覆盖文件的修改（例如在闭源产品中修改 MPL 代码）；
- 希望获得**不含 copyleft 义务**的版本，或 OEM / 嵌入 / 白标授权；
- 需要以本项目名义提供的**官方支持、培训或认证**（商标使用另见商标政策）。

维护者可针对具体情况提供书面许可（商业许可、OEM/嵌入许可等）。

### 7.3 贡献政策：当前暂不合并外部 Pull Request

维护者精力有限，**本仓库当前不合并外部提交**。这不影响你使用本项目：

- **允许**：fork、修改、再分发、商用（MPL-2.0 已授予这些权利，见 §7.1）；
- **欢迎**：通过 issue 报告缺陷与改进建议；
- **暂不处理**：Pull Request（会被关闭并引导到 issue）。

因此本仓库**当前不使用任何贡献者协议**（既不需要签 CAA，也不需要 DCO）。
若将来开放合并外部贡献，维护者会**先公布届时生效**的贡献者协议与签署方式，
再合并第一份外部 PR。详见 [CONTRIBUTING.md §4](CONTRIBUTING.md)。

> MPL-2.0 已允许任何人 fork 并继续开发；"不合并 PR"只影响是否把外部代码并进
> **本仓库**，不影响你基于本项目做任何事。

### 7.4 关于许可证沿革（事实陈述）

- **首个版本**（git 历史起点 `a1ce136`，至本版本之前）：**保留所有权利（专有许可）**。
  **本项目在该阶段从未以任何开源许可证公开发布过任何版本。**
- 在整理阶段，工作区曾**临时**放置过 MIT OR Apache-2.0 与 MPL-2.0 的许可证文件，
  但那些文件**从未随任何发布分发**，也**从未**对任何第三方构成授权。
- **自本版本起**：以 **MPL-2.0** 发布（本节即该变更的公告）。

> MPL-2.0 是**不可收回**的：任何人依 MPL-2.0 获得的副本，其许可**永久有效**，
> 维护者**不能**将其撤回或改为专有。维护者可能对**未来版本**采用不同许可或附加条款，
> 但**不影响**本版本及任何已按 MPL-2.0 发布的版本。

---

> **另见**：[LICENSE-NOTICE](LICENSE-NOTICE)（Exhibit A 集中通知）、
> [TRADEMARK.md](TRADEMARK.md)（名称与标识使用规则）、
> [SECURITY.md](SECURITY.md)（漏洞报告）、
> [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)（第三方材料）。

---

## 8. 在国内网络下访问（实测）

`github.com` 网页与 `raw.githubusercontent.com` 在本机**不通**（SNI 阻断）。实测可用的入口：

| 用途 | 入口 |
| --- | --- |
| 读网页 / 单文件 | `https://gh-proxy.com/https://github.com/shouzugzs-AI/Sparrow` |
| 下载整包 zip | `https://gh-proxy.com/https://github.com/shouzugzs-AI/Sparrow/archive/refs/heads/main.zip`（3.3 MB / 8.2 s） |
| 元数据 / 单文件（程序化） | `https://api.github.com/repos/shouzugzs-AI/Sparrow`（HTTP 200，不依赖第三方） |
| git 推送 / 拉取 | **SSH over 443**：`~/.ssh/config` 里 `Host github.com` → `HostName ssh.github.com` + `Port 443` |

`gitee.com` / `gitcode.com` 可访问，适合做**镜像**（含核验命令）。

完整实测表、可抄的命令、镜像做法与注意事项见
[**docs/访问方式_国内网络.md**](docs/访问方式_国内网络.md)。
