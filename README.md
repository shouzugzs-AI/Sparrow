# VMLM

> 项目代号 **VMLM**；GitHub 仓库名 **Sparrow**（`shouzugzs-AI/Sparrow`）。

**VMLM** 是一个用 Rust 写的、纯 CPU 的张量虚拟机与模型工程。它由「极简张量虚拟机 / 引擎」和「可训练的神经网络栈」两部分组成：底层是一个不依赖标准库、把权重当作程序来装载执行的 VM，上层是一套六层抽库的 Rust workspace，同时承载**稠密**与 **B1/B2 稀疏**两条建模路径。

本仓库是 VMLM 的**全量副本**（**专有软件，保留所有权利**）：稠密路径与 B1/B2 稀疏路径的代码都在这里。

- 许可：**保留所有权利（All Rights Reserved，专有许可）** —— 见 [LICENSE](LICENSE)。**本仓库当前不是开源项目**，未授予任何复制、分发、修改或商业使用许可；如需授权请联系维护者
- 贡献：若获维护者邀请贡献，须签 [CAA.md](CAA.md)（著作权转让 + 回授；见 [CONTRIBUTING.md](CONTRIBUTING.md)）
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
└─ Cargo.toml   workspace 定义
```

---

## 7. 许可证与授权

### 7.1 本项目保留所有权利（专有许可）

本项目**当前**采用**保留所有权利（All Rights Reserved）**的专有许可，正文见 [LICENSE](LICENSE)。本项目的**著作权人**（权利人）为 **金施哲**（GitHub 账号 `shouzugzs`）；未经其事先书面许可，任何人不得使用、复制、分发、修改或商业化本项目。
**本仓库不是开源项目**，未授予任何开源许可证。未经维护者事先书面许可，任何人不得：

- 复制、发行、公开传播、公开提供或以其它方式向第三方提供本项目的全部或任何部分；
- 修改、改编、翻译、演绎或创作衍生作品；
- 反向工程、反编译、反汇编（法律强制性规定明确允许者除外）；
- 将本项目或其任何部分用于**商业目的**，包括但不限于内部生产使用、SaaS 托管/对外提供服务、
  二次分发、嵌入或捆绑至其它产品、或以本项目为基础提供有偿服务；
- 移除、遮蔽或更改本项目中的著作权、商标、专利或其它权利声明。

### 7.2 如何申请授权 / 商业许可

如需**使用授权、评估许可或商业许可**，请联系：

> **`<联系方式占位>`**

维护者可针对具体情况提供书面许可（如评估许可、商业许可、OEM/嵌入许可等）。
**任何未经书面许可的使用均属侵权。**

### 7.3 贡献须签 CAA

若维护者邀请你贡献代码，你必须签署 [CAA.md](CAA.md)（贡献者著作权转让协议）：你将 Contribution 的
**著作财产权转让**给维护者，同时获得一项**回授**许可（你仍可在自己的项目/产品中自由使用你自己的贡献）。

**仅 DCO / `Signed-off-by` 不被接受**——它只证明来源合规，**不构成著作权转让**。未签 CAA 的 PR 不会被合并。
详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

该「著作权转让 + 再许可」安排，使维护者保留对项目进行**双许可 / 商业授权**的能力。

### 7.4 关于历史许可证（事实陈述）

本项目在**早期版本**中曾以 **MIT OR Apache-2.0** 双许可发布。**自本版本起，本项目一律以
[LICENSE](LICENSE)（保留所有权利）为准**；`LICENSE-APACHE` 以及此前一段时间使用过的
MPL-2.0 相关文件（`LICENSE-NOTICE` 等）均已移除，不再随本仓库分发。

> 维护者保留在**将来**把本项目改为其它许可（含开源许可，例如 MPL-2.0）的权利；
> 但该变更**仅对此后发布的版本生效**，且**已发布版本的授权不可撤回**。
> 开源切换模板与步骤由维护者单独保管（`维护者本地保留的模板目录\`），**不随本仓库分发**。

---

> **注意**：[CAA.md](CAA.md) 是**模板，不构成法律意见**；在正式对外收费 /
> 商业授权之前，须由执业律师审阅。见 CAA.md 顶部声明（特别提示 §7 精神权利与
> §8 回退条款）。
