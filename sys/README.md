# vmlm-sys（L0 底层调用层）

## 职责边界

* **只**放：SIMD intrinsics、内存分配、线程原语、计时、CPU 探测。
* **不做**：region 切分 / 任务调度（那是 L2a `vmlm-ops` 的事）。
* 多后端并列：`src/x86_avx2/`（现有）、将来的 `x86_avx512.rs` / `arm_neon.rs` / `cuda.rs`，运行时 dispatch。
* 跨层接口形态：**C ABI 函数指针表**（见 `lib.rs` 的 `Backend`）或 opaque 句柄，**不要**用 Rust trait 当跨层接口。

## ⚠️ 引擎（vmlm-runtime）的正确构建方式 —— 别踩这个坑

`vmlm-runtime` 是 `no_std`，需要 `runtime/.cargo/config.toml` 里的链接参数：
`/ENTRY:mainCRTStartup`、`/SUBSYSTEM:CONSOLE`、`/NODEFAULTLIB`、`kernel32.lib`、`/OPT:REF,ICF`，
以及 `target-feature=+avx2,+fma`。

**Cargo 只读取「调用目录向上」的 config**，所以：

| 做法 | 结果 |
|---|---|
| 在 workspace 根跑 `cargo build --release -p vmlm-runtime` | ❌ LNK1561「必须定义入口点」（根目录的 config 没有那些 link-arg） |
| 在 `runtime/` 跑 `cargo build --release`（不带 `--target`） | ❌ LNK1120：无法解析 `mainCRTStartup`（`[target.x86_64-pc-windows-msvc]` 段落不生效） |
| **在 `runtime/` 跑带 `--target` 的命令** | ✅ 正确 |

canonical 命令（与 `tools/build_all.ps1` 的 `Step "vmlm-runtime (engine core)" $rt @("--target",$triple)` 一致）：

```powershell
cd runtime
cargo build --target x86_64-pc-windows-msvc --release
copy target\x86_64-pc-windows-msvc\release\vmlm-runtime.exe ..\target\release\vmlm-runtime.exe
```

验收：`target\release\vmlm-runtime.exe` **必须是 74752 字节**。
（`--target` 还会解锁 `runtime/build.rs`；这也是必须带它的原因之一。）

## 抽库纪律（本项目）

1. 只搬不改：kernel 签名、行为、性能、累加顺序、cfg 一律不动。
2. 每搬一块跑一遍：gradcheck PASS 40/0、100 步 loss 逐位不变、s/step 噪声内（±1%）、
   `vmlm-b1-full.exe` / `vmlm-tests.exe` / `vmlm-runtime.exe` 三个体积不变。
3. 搬到 core 之外的符号，core 侧用 `pub use vmlm_sys::…::*;` 重导出，
   保证 `vmlm_core::…` 的既有路径与**全部调用点零改动**。
4. 跨 crate 必需时允许的唯一可见性变化：`fn` → `pub fn`（如 `hsum256`、`round_f32`）。
5. 每块完成后快照到 `_opt/l0_<块名>_src/`；任一项不达标即回退该块并记录原因。
