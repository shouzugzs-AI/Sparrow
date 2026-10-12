# B1Model 设计书（L4 模型组装 · 待实施）

**状态**：设计完成，**未开工**（本轮预算不足，按"不硬上"纪律不启动半途重构）。
**依据**：`train/src/b1_full_train.rs`（4087 行）实测侦察，行号均为该文件当前行号。

---

## 1. 为什么值得做（实测证据：同一组装写了两遍）

`run_main` 从 620 行起到文件末 ≈ **3400 行单函数**，其中**两处**是同一套模型组装：

| 相位 | 行号 | 组装序列 |
|---|---|---|
| **训练：前向** | 1824 → 1860/1865 → 1921/1930 | `par_b1_forward`(逐层 B1) → `attn_forward_sparse`/`attn_forward`(每 attn_every 层) → `par_head_fwd_wt`/`par_b1_forward(head)` →（head_skip 时 head2 分支 1935） |
| **训练：反向** | 2084/2086/2088 → 2160/2177 → 2224 | `head_bwd_batched`/`par_head_bwd`/`par_b1_backward(head)` → `attn_backward_sparse_par`/`attn_backward` → 逐层 `par_b1_backward` |
| **推理/生成：前向** | 2796 → 2809/2814 → 2831/2838 → 2843 | `par_b1_forward` → `attn_forward_sparse`/`attn_forward` → head 前向 → `sample_next` |

⇒ **同一份前向被抄了两遍**（训练与生成），反向只存在于训练侧。这是 `B1Model` 的第一收益点。

## 2. 模型状态清单（`run_main` 的 alloc 块，1116-1175+）

| 类别 | 缓冲 |
|---|---|
| 参数 | `emb`(n_vocab×dim)、`lm_w`、`lm_b`、`scs[l]`（`SparseConnW`/层）、`head`、`head2`（head_skip） |
| 梯度 | `emb_g`、`lm_g`、`lm_bg`、`head_dw`、`dws[l]` |
| Adam 矩 | `emb_m/emb_v`、`lm_m/lm_v`、`lm_bm/lm_bv`、`head_m/head_v` |
| 工作区 | `etab`、`x/y/h/logits/dlogits/dh/dx/dy` 等按 `tok_total` 分配，另有**每 worker 私有 dw 分段** `dwp`（`nslot×nth×max_wlen`） |
| 块级 scratch | t8 整栈集群（已在 `models::b1`，thread_local） |

配置项（都来自 argv，须进 `B1Config`）：`dim, k0, k0_embed, k0_head, k0_attn, layers, attn_every, attn_sparse,
attn_src_shared, attn_calib(+target), attn_window, head_skip, no_residual, depth_scale, seq, batch,
vocab_mode, chars_per_token, parallel_mode, threads, pin, t8*, fusion, w-bf16 …`

## 3. 提议接口（只用已暴露的 API）

```rust
// crates: models (L4) —— 依赖 blocks(L3) / ops(L2a) / tensor(L1) / sys(L0)，不依赖 core、不依赖 trainer
pub struct B1Config { /* 上表全部配置项；含 parallel/t8 开关 */ }
pub struct B1Weights { emb: *mut f32, lm_w: *mut f32, lm_b: *mut f32,
                       layers: Vec<SparseConnW>, head: SparseConnW, head2: Option<SparseConnW> }
pub struct B1Scratch { /* 上面"工作区" + 每 worker dw 分段 + h/logits/dlogits/... */ }

impl B1Model {
    pub unsafe fn new(cfg: B1Config) -> Self;                  // 参数 + 矩 + 工作区一次分好
    pub unsafe fn forward(&mut self, tok: &[u32], tt: usize) -> *const f32;   // -> logits
    pub unsafe fn backward(&mut self, dlogits: *const f32);                   // 累加进 *_g / dws / head_dw
    pub unsafe fn sample_next(&mut self, last_pos: usize, rng: &mut Rng64, params: &SampleParams) -> u32;
    pub unsafe fn save(&self, path: &str);  pub unsafe fn load(&mut self, path: &str);   // ck_write/ck_read
}
```

**双向复用是硬约束**（用户判定）：`vmlm-b1-full` 用 `forward+backward+save/load`，
`vmlm-chat` 用 `forward+sample_next+load`。若某个方法只能在 trainer 里用，说明抽象错了 → 回退重设计。

## 4. 逐位一致的关键风险（必须逐条守住）

1. **缓冲别名**：1777 行注释记录的已修 bug —— 边界层 `par_b1_forward(xin=y, y)` 会读到被覆写的输入。
   搬移时缓冲的**别名关系**（`xin` 与 `y`、`h` 与 `h_ins[0]`）必须原样保留，不能"顺手整理"。
2. **归约顺序**：`par_b1_backward` 的每线程 dw 分段 + 固定线程号归约（`nt` 变化本就会报告差异，不假装一致）。
3. **head/head2 双路径**：`head_skip` 时 `h_ins[0]` 与 `logits2/d_h0` 的第二组调用顺序（1935/2248-2253）。
4. **t8 开关与 thread_local**：`T8_ON/T8_FWD/T8_BWD` 现在是 `models::b1` 的 static，由 CLI 写入；
   搬进 `B1Config` 后必须仍**在同一进程、同一时刻**对所有 worker 一致。
5. **标定与体检**：`attn_calib`（worst |q·k| → g 向量）与 `attn_qk_stats` 体检在训练/评估两侧都要用，
   属于模型级副作用，须随 `B1Model` 一起走，不能留在 CLI 分支里。

## 5. 分阶段计划（每阶段都跑五项验收，失败即回退）

| 阶段 | 内容 | 验收加强项 |
|---|---|---|
| **L4c-1** | `B1Config/B1Weights/B1Scratch` 落地；把**推理/生成**的前向（2796-2843）先接进来（不改训练侧） | `vmlm-chat` 生成输出**逐字节相同**；引擎 sha 不变 |
| **L4c-2** | 训练前向接入（1824-1935），训练侧只留 step 编排 | loss 逐位不变（`loss_hash`） |
| **L4c-3** | 反向接入（2084-2253），训练侧只留 clip/Adam/调度 | gradcheck 逐字节相同 + loss 逐位不变 |
| **L4c-4** | 删掉 trainer 里重复的组装；trainer 只剩数据加载/step 编排/eval/ckpt/CLI | 五项 + chat 逐字节 |

**五项验收（每阶段）**：gradcheck PASS ｜ 100 步 loss 逐位不变（`loss_hash=0x10c80251727e8126` 为当前基线）
｜ s/step ≤1%（空载 5 轮交替）｜ 三体积不增大（现为 engine 75776 / tests 187392 / b1-full 622592）
｜ `vmlm-tests` 全 PASS。**额外**：`vmlm-chat` 生成逐字节相同。

## 6. 前置（可顺带做掉）
`core/src/interp.rs` 的 34 行 VM 侧 JIT 驱动（`jit_entry`/`jit_compile`/`jit_run`）——L2b 的唯一残留。
它需要新定一个 `trans` 侧接口（VM 私有态边界），与 L4c 阶段本来就要动的 VM 边界一起做更省事。

---

## 附：L4c-1 验收基准（已捕获，2026 本轮）

**`vmlm-chat` 生成基线（逐字节）**：权重/词表内嵌的 `target/release/vmlm-chat.exe`（9958912 B，
`dim=4096 k0=8 k0e=16 k0h=16 layers=32 attn_every=4 k0a=8 V=4054`），prompt = `今天`：

```
prompt 文件 : _opt/chat_prompt.txt      ("今天\n", UTF-8 无 BOM)
命令        : Start-Process target\release\vmlm-chat.exe `
                -RedirectStandardInput _opt\chat_prompt.txt `
                -RedirectStandardOutput _opt\chat_base_N.txt -Wait
基线输出    : _opt/chat_base_1.txt   798 B
sha256(16)  : 899CC4AD6CA6D662       （连跑两次完全一致 ⇒ 可用作逐字节判据）
```

生成内容（首行）：`> 天边吃的时候我就先把最甜的事，我带你这份，我可以后一起去溪边吃甜甜的尾巴一定要一起来，小尖！嗷呜！`

⇒ **L4c-1 的"chat 生成逐字节相同"判据现在就绪**：改完代码后重跑同一条命令，比对
`chat_base_1.txt` 的 sha256 必须仍是 `899CC4AD…`。

**注意**：`vmlm-chat.exe` 与 trainer 共享同一份源码（`train/src/bin/vmlm_chat.rs` 用
`#[path = "../b1_full_train.rs"] mod core;`）⇒ 只要 `b1_full_train.rs` 的生成路径切到 `B1Model`，
chat 自动跟着走；无需在 chat 侧另写一份调用。
但 chat 内嵌权重（`include_bytes!` of 9.2 MB + 词表），改完 trainer 源码后**必须重建 `vmlm-chat`**才能验。

