# 描述语言的仓库边界与迁移手册（V0.1.5；V0.1.6 已合并）

> **V0.1.6 状态：已按本文 §5 合并为引擎仓库的子目录 `sparrow/`。**
> §1–§4 保留为**合并前**的实测记录（那三条判据正是当时用来证明"合并可行且代价为零"的）。
> 合并后的验证与发布记录见 `REPORT_V015.md` §13。
>
> 起因：用户裁定「描述语言可以作为附加仓库，或者与 vmlm 合并」。
> 这份文档把这个架构选择变成**可复现的实测**，而不是一句偏好。

---

## 1. 结论（先说结果）

| 问题 | 实测答案 |
|---|---|
| 现在是什么？ | `D:\czsai\sparrow` 是**独立附加仓库**：自带 `.git`，**385 个文件**入库（基线提交 `34e6904` 时是 296），本版 HEAD 用 `git log -1` 查 |
| 「并入 vmlm 目录」可行吗？ | **可行**，而且**不需要动父仓库的 members 表**（判据 1a / 1b / 2，见 §4） |
| 「并成 vmlm 的 workspace member」呢？ | 不推荐：会把 vmlm 的 release profile 套到语言 crate 上，并让语言混进引擎的构建/链接面（§3） |
| 两者互斥吗？ | 不互斥。迁移 = 一次拷贝 + 一次验证，命令在 §5 |

---

## 2. 三个目录分别是什么

| 目录 | 是什么 | 版本控制 | 与本语言的关系 |
|---|---|---|---|
| `D:\czsai\vmlm` | **私有工作副本**：引擎 / 汇编器在这里构建（`target\release\vmlm-asm.exe`、`vmlm-runtime.exe`） | 无 git | 语言的 `--vmlm-root` 指向它；6 个 lib 后端验收用例都对着它跑 |
| `D:\czsai\vmlmd` | **上游开源树**：GitHub `shouzugzs-AI/Sparrow`（MPL-2.0，含 README / LICENSE / LICENSE-NOTICE / TRADEMARK / SECURITY / docs），由 `sync_vmlmd.ps1` 从 vmlm 同步。早先那份 `CAA.md` 已在开源切换中删除（改为**暂不合并外部 PR**，见其 CONTRIBUTING §4） | 有 git，工作区干净 | 本仓库**没有改动它任何一个文件** |
| `D:\czsai\sparrow` | **本仓库**：描述语言（前端 `源码 → .spar → .vml/.vmb` + lib 后端 `→ Rust AOT 源`） | V0.1.5 起有 git | — |

**名字冲突要提醒**：上游**仓库名**就叫 `Sparrow`（`shouzugzs-AI/Sparrow`），而本语言在 V0.1 也被命名为 Sparrow（当时并不知道上游仓库名）。读的时候别混：
- 上游 Sparrow = 整套 VMLM（虚拟机 + 训练栈）
- 本仓库 = **描述语言**（写架构的文本前端 + 两个后端）

---

## 3. 两条路的代价（为什么当前选独立仓库）

### 3.1 并入 vmlm **目录**（拷贝成 `vmlm\sparrow\`，保留自带的 `[workspace]`）
- 得到：一次 clone 拿到全部；lib 后端的 `--vmlm-root` 变成同仓相对路径。
- 付出：**实测为零** —— 探针照抄 vmlm 的 members 表与 release profile 造了父仓库，
  父仓库 `cargo metadata` 不含 sparrow、`cargo build --release` 正常（判据 1a/1b），
  嵌套的 sparrow `cargo test` 仍全绿（判据 2；**数字不写死**，与真实仓库逐项比对）。
- 机制：`sparrow/Cargo.toml` 第一行就是 `[workspace]`，它自己就是 workspace 根；
  父仓库的 members 是**显式列表**，不会把它吸进去。判据 3 证明这一行不是装饰。

### 3.2 并成 vmlm 的 workspace **member**（改 vmlm 的 `members` 表）
- 代价 1：vmlm 根 `[profile.release]` 是 `panic="abort"` + `lto=true` + `codegen-units=1` + `strip=true`，会整体套到语言 crate 上（语言是 std 程序，`panic=abort` 会让 `cargo test` 的 panic 捕获语义变化）。
- 代价 2：vmlm 自己的纪律写在它的 `Cargo.toml` 注释里 ——「加 member 必须让 `target\release\vmlm-runtime.exe` **逐字节不变**」。多一个 std member 就要重验这条，而语言对引擎零贡献。
- 代价 3：语言跑一次验收会写 561 个生成物到 `out/`、`examples/`，混进引擎的构建图没有任何收益。
- 结论：**不推荐**。

### 3.3 独立仓库（当前选择）
- 得到：与 vmlm 的构建/链接纪律完全隔离；语言自身**零第三方依赖**；上游开源树零改动。
- 付出：`--vmlm-root` 必须显式给（hard-fail，缺了就 `E109` 报清楚）；README / 许可 / CI 得自己带（目前**没有 remote**）。

---

## 4. 三条判据与实测输出

驱动：`tools/repo_boundary_probe.ps1`（只写 `out/`，不碰 vmlm / vmlmd 任何文件）

| 判据 | 内容 | 实测 |
|---|---|---|
| 1a | 父仓库包列表**不含** sparrow | PASS `packages=stub-sys,…,stub-server`（13 个占位） |
| 1b | 父仓库 `cargo build --release` 成功（带 vmlm 的 profile） | PASS `exit=0` |
| 2 | 嵌套 `sparrow\cargo test` | PASS（**与真实仓库同一数字**，本版 85 passed / 0 failed；脚本不写死，写死过一次 81 后来加测试就假失败） |
| 3 | **can-fail**：删掉拷贝里的 `[workspace]` 行 ⇒ 必须失败并点名 workspace | PASS `exit=101`，`error: current package believes it's in a workspace when it's not:` |

判据 3 是这张表的支点：没有它，判据 2 可能只是「怎么写都能过」。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File tools/repo_boundary_probe.ps1   # 退出码 0 = 全绿
```

复现时的**实测坑**（已记进 `docs/TRAPS.md` A9）：脚本第一版用
`($lines | Where-Object {...})[0]` 取 members 行，只匹配到一条时 PowerShell 把它解包成
string，`[0]` 取到的是**首字符 `m`** ⇒ 父仓库 `Cargo.toml` 写成 `members` 位置一个字母 `m`，
cargo `exit=101`，判据 1a/1b/3 全红。**假失败**，不是边界问题。

---

## 5. 迁移步骤（若决定并入 vmlm）

```powershell
# 1) 拷贝：排除 target/out/.git（.git 一定要排除，否则变成嵌套仓库）
robocopy D:\czsai\sparrow D:\czsai\vmlm\sparrow /E /XD target out .git

# 2) 语言自身仍然全绿（81 项）
cd D:\czsai\vmlm\sparrow ; cargo test

# 3) 引擎的字节纪律仍成立（这是 vmlm 自己的规矩，必须自己验一次）
cd D:\czsai\vmlm\runtime ; cargo build --target x86_64-pc-windows-msvc --release

# 4) 父仓库不认识语言 crate（应无输出）
cd D:\czsai\vmlm ; cargo metadata --format-version 1 --no-deps | Select-String sparrow
```

`git add` / `git commit` 前先确认 CWD 就是 `D:\czsai\vmlm`：本次实际踩过一次
「在 `D:\czsai` 跑 `git init` + `git add -A`」，那会把 20 GB 工作区整个暂存（见 TRAPS H1）。

---

## 6. 未验证 / 边界（不藏）

1. 探针里的父仓库是**忠实镜像**（members 表逐字照抄 + 13 个占位 crate + 同样的 release profile），**不是**真实 vmlm 树。真并进去以后，§5 第 3/4 步仍要跑一遍 —— 那时验的才是真的引擎字节。
2. 本仓库**没有 remote**，也没有 LICENSE / README / CI。要不要推到 GitHub（新仓库，或作为上游仓库的子目录）由你决定；本仓库对上游 `vmlmd` 零改动。
3. `.gitattributes` 只对齐了上游的 `* text=auto eol=lf` 与二进制清单，没有逐条照抄上游的忽略规则（上游那份是给 vmlm 的目录结构写的）。
4. `out/`、`target/` 不入库（`.gitignore`）；`examples/` 下的 `.spar/.vml/.vmb/.args/*_bins` **入库**，当作「生成物指纹」：因此 `tools/verify_all.ps1` 能在跑完**全部驱动**（本版 10 个，清单就是脚本里的 `$drivers`）后断言 `git status` 为空（已实测 PASS）。
5. **既有 12 个提交的作者是占位身份** `czsai <czsai@localhost>`（当时没配 `user.name/user.email`）。
   重写作者会改掉**全部哈希**，而本文与 `REPORT_V015.md` 引用着这些哈希（`34e6904` / `c5a598a` …），
   所以本次只把本仓库的 `user.*` 对齐成 `shouzugzs`（**只影响将来的提交**），历史保持不动。
   若推之前决定重写：`git rebase --root --exec "git commit --amend --no-edit --reset-author"`，
   改完必须把文档里的哈希一并更新（否则文档指向不存在的提交）。
6. **README / LICENSE / CHANGELOG / CI 已补**（V0.1.6）：许可 **MPL-2.0**（与上游引擎一致，
   `LICENSE` = 官方原文 SHA-256 `FAB3DD6B…63E85`），另见 `LICENSE-NOTICE` / `SOURCE-HEADER.txt` /
   `CHANGELOG.md` / `.github/workflows/ci.yml`（CI 只跑不需要引擎的部分）。
   **仍然没有 remote**：要不要推、推到哪，由维护者决定。
7. **布局结论（V0.1.6）：推荐"合并为引擎仓库的子目录"**，理由与本节的实测一致 ——
   本语言离开引擎不可用（`.vmb` 要汇编器/运行时，AOT 要 `vmlm-ops` 源码），
   而合并后 `--vmlm-root` 的默认推导（同级目录）**正好等于引擎仓库根**，不用额外参数；
   两边许可口径已统一成 MPL-2.0，不产生混许可问题；名字冲突也随之消失
   （"Sparrow 项目 = 引擎 + 它的描述语言"）。独立性不受影响：`Cargo.toml` 第一行仍是
   `[workspace]`，父仓库不会把本 crate 吸进它的构建/链接面（§4 判据 1a/1b/2/3 已实测）。
   反向操作（拆出来当独立仓库）只是拷一个目录，步骤见 §5 的同类命令。
