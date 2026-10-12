# 单文件 EXE（`vmlm-chat`）：权重内嵌 + 空文件夹实测

> 按你的要求：先做 1-4（实现 / Cargo 配置 / 编译命令 / 实测体积）——**已全部完成**，
> 并且做了第 5 步的空文件夹实测（日志见下）。**没有改训练器行为**（哈希不变，见 §6）。

## 1. 实现（`train/src/bin/vmlm_chat.rs`）

| 要点 | 做法 |
|---|---|
| 内嵌权重 | `static MODEL: &[u8] = include_bytes!(env!("VMLM_MODEL_RESOLVED"));` → 进 `.rodata` |
| 内嵌词表 | `static VOCAB: &[u8] = include_bytes!(env!("VMLM_VOCAB_RESOLVED"));` |
| 路径解析 | `train/build.rs`：优先环境变量 `VMLM_MODEL`，否则 `artifacts/b1_C3_10k.cpk`，否则 artifacts 下最新 `*.cpk`（词表固定 `artifacts/b1_char/char_vocab_hex.txt`）|
| 复用生成实现 | `#[path = "../b1_full_train.rs"] mod core;` + 新增 `core::set_args()`：**在同一个进程里注入 argv 后调用 `core::run_main()`**，跑的就是 `--generate 1` 那条完整路径（同一套前向/采样/标定/检查点代码）。**没有第二份前向或采样实现。** |
| 自配置 | 从 `.cpk` 头部读 dim/k0/k0e/k0h/layers/n_attn/k0_attn/V/开关 → EXE 自己知道模型结构，命令行不用给 |
| 控制台编码 | 启动时 `SetConsoleOutputCP(65001)` + `SetConsoleCP(65001)`（GBK→UTF-8），中文输出/输入都正常 |
| 交互 | `> ` 提示符；`quit`/`exit`/空行退出；提示符里可临时改参数：`seed=2 temp=0.9 k=50 p=0.9 len=80 今天` |
| 默认参数 | `temp=0.7 采样top-k=30 top-p=0.85 gen_len=100 seq=64 threads=4`（按你的规格）|
| 数据集 | 训练器启动时会读 `<data-dir>/{train_ids,val_ids}.bin + meta.json`（即使只生成）。单文件不能带数据集 ⇒ chat 在临时目录**合成**一份确定性伪随机 id（8192 个）+ `meta.json`；并传 `--attn-calib 0`（检查点里的标定因子**已经乘进 Wq/Wk**，重标会改模型）|
| 每个 prompt 一个 one-shot 子进程 | 避免同进程反复 `run_main()` 造成的内存增长（每次加载 ~40-100 MB 不回收）|

**采样 top-k 与 B2 的 top-k**：仍然是两回事。chat 只传 `--top-k`（解码策略，默认 30）；
B2 的前向 top-k **没有实现、没有开关、没有任何调用**。

## 2. Cargo.toml 改动

```toml
[package]
name = "vmlm-train"
build = "build.rs"          # 编译期解析要内嵌的文件

[[bin]]
name = "vmlm-chat"          # 注意：显式命名带连字符（文件名 vmlm_chat.rs 只会得到 vmlm_chat）
path = "src/bin/vmlm_chat.rs"

[profile.release]
lto = "fat"
strip = "symbols"
panic = "abort"
opt-level = 3
codegen-units = 1
```
⚠️ `[profile.release]` 作用于**整个包**（含 `vmlm-b1-full`）。已验证：训练器行为不变（§6）。

## 3. 编译命令

```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-chat
```
换成别的权重：`set VMLM_MODEL=<repo>/artifacts/your.cpk` 后再编译（build.rs 会 `rerun-if-env-changed`）。

## 4. 实测体积

| 内嵌的权重 | 参数量 | EXE 体积 |
|---|---|---|
| **`artifacts/b1_C3_10k.cpk`（你指定的 40MB 权重）** | 9,945,344 | **38.58 MB** ✅ |
| `artifacts/b1_gen_model.cpk`（979k 参数，600 步训过） | 979,008 | **4.38 MB** |

（`b1_C3_10k.cpk` 本体 37.9 MB + 词表 63 KB + 代码 ≈ 38.6 MB —— 与"约 40MB"一致。）

## 5. 空文件夹实测（只放一个 EXE）

把 `vmlm-chat.exe` 单独拷到 `%TEMP%\vmlm_singlefile_test\`（**目录里只有这个 EXE**），
喂入 `今天` 后退出：

```
=== vmlm-chat（单文件 EXE：权重与词表已内嵌）===
模型 39.8 MB | dim=4096 k0=32 k0e=128 k0h=128 layers=32 attn_every=4 k0a=32 V=4054 | 词表 63644 字节
采样默认: temperature=0.7 采样top-k=30 top-p=0.85 gen_len=100 seq=64 threads=4 seed=1
输入 prompt 回车生成；quit / exit / 空行 退出
> 退惕我凛弓相年琶卡匝缀邮唾瘾绫崇我莞矩蛮角赔一效啀凫窣鹞苇嫌岁芷封困报液纹讦笨称柏明⑦咋肠饯烈一起茫欣小不捎抿暗
呜！
呜！…（随后大量重复「呜！」）
  （100 字符 / 7.8s = 12.8 字符/秒；seed=1 temp=0.7 k=30 p=0.85）
> bye
```

**如实说明**：`b1_C3_10k.cpk` 是我为"凑出 40MB 内嵌文件"而**只训了 40 步**的 C3 模型
（val 8.0955 ≈ 11.7 bits/char，还贴在均匀地板上），所以输出是"字表内乱码 + 高频口癖『呜！』"。
**打包本身完全正常**（自配置、单文件、能跑、速度 12.8 字符/秒）。

因此我另外用**训练了 600 步的 979k 字符模型**编了一个 4.38 MB 的单文件 EXE 做质量对照：

```
模型 3.9 MB | dim=1536 k0=8 k0e=32 k0h=32 layers=24 attn_every=4 k0a=8 V=4054 | 词表 63644 字节
> 小尖，咱们等这了，等不过得子！嗷呜~这啦！我等等那
呜~我
呜~
呜~
呜~
嗷呜~
呜！
嗷呜~嗷呜~
嗷呜~
呜~我小尖，我小尖不，我小尖的！
  （100 字符 / 2.4s = 41.1 字符/秒）
```
（两个 EXE 的完整日志存到 `artifacts/singlefile/test_C3_39MB.txt`、`test_char_4MB.txt`。）

## 6. 纪律核对

| 纪律 | 结果 |
|---|---|
| 不改训练器 | ✅ `vmlm-b1-full` 仍可用；`loss_hash = 0xc9c987c15d75d1d6` **逐位不变**（LTO/abort 之后复测） |
| 不改稀疏层内核 | ✅ 核心 `b1_forward`/`b1_backward*` 一个字节没动（引擎 core size 仍 74752 B） |
| 生成复用现有实现 | ✅ chat 走的就是 `--generate` 路径（`set_args` + `run_main`），无第二份实现 |
| 采样确定性 | ✅ 同 seed 同 prompt 同输出（前一轮已验：3 次 sha256 相同） |
| 不自己跑长测试 | ✅ 最长一次是 600 步（≈8 分钟），C3 只跑了 40 步用来产 40MB 检查点 |
| 套件 | ✅ **65/65**；`build_all` OK；`tests 0`；`ci_path_guard CLEAN` |

## 7. 性能与两个诚实提醒

1. **每个 prompt 会重新加载一次模型**（one-shot 子进程）：C3 40MB 模型约 1-2 s 额外开销；
   979k 模型几乎无感。若想更快，需要把引擎抽成 lib 做常驻循环（那次改动会动训练器结构，需要你点头）。
2. **"运行时从内存读"目前是"分布上单文件、读取路径经临时文件"**：
   `include_bytes!` 的数据确实在 EXE 的 `.rodata`（**分发只需一个文件** ✅），
   但为了让**现有** `--load-model` 代码路径原样复用，我把内嵌字节落到了 `%TEMP%\vmlm_chat_<pid>/model.cpk`
   再交给驱动读（39.8 MB 写盘 ~50-100 ms，之后走页缓存）。
   要做到**纯内存加载**，需要给驱动加一个 `--load-model @mem` 钩子（几行改动）——要的话我加。
3. 合成数据集只用于"让驱动跑起来"（标定/体检那一个 val 批），不参与生成语义；
   因为传了 `--attn-calib 0`，它**不会**改动载入的权重。
