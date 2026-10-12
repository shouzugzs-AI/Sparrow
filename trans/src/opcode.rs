//! L2b opcode 表：语义契约（从 core/src/format.rs 逐字节搬运）。

// ---- opcode（见白皮书 §3.4，共 34 条）----
pub const OP_HALT: u8 = 0;
// A. 搬运 / 形状
pub const OP_LOADK: u8 = 1;
pub const OP_COPY: u8 = 2;
pub const OP_RESHAPE: u8 = 3;
pub const OP_TRANSPOSE: u8 = 4;
pub const OP_CONCAT: u8 = 5;
pub const OP_SLICE: u8 = 6;
// B. 线性代数
pub const OP_GEMM: u8 = 10;
pub const OP_BMM: u8 = 11;
pub const OP_OUTER: u8 = 12;
pub const OP_GEMV: u8 = 13;
pub const OP_EMB: u8 = 14;
// C. 逐元素 / 激活
pub const OP_EWADD: u8 = 20;
pub const OP_EWMUL: u8 = 21;
pub const OP_EWSUB: u8 = 22;
pub const OP_EWDIV: u8 = 23;
pub const OP_SCALE: u8 = 24;
pub const OP_ACT: u8 = 25;
pub const OP_EXP: u8 = 26;
pub const OP_LOG: u8 = 27;
pub const OP_CLAMP: u8 = 28;
pub const OP_NEG: u8 = 29; // 逐元素取负
// D. 归约 / 归一化
pub const OP_LN: u8 = 30;
pub const OP_RMSNORM: u8 = 31;
pub const OP_SUM: u8 = 32;
pub const OP_MEAN: u8 = 33;
pub const OP_MAX: u8 = 34;
pub const OP_NORM: u8 = 35;
pub const OP_SOFTMAX: u8 = 36;
// E. 控制流 / 稀疏 / 特殊
pub const OP_MASK_MUL: u8 = 40;
pub const OP_TOPK_THRESH: u8 = 41;
pub const OP_TOPK_IDX: u8 = 42;
pub const OP_GATHER: u8 = 43;
pub const OP_SCATTER: u8 = 44;
pub const OP_BRANCH: u8 = 45;
pub const OP_JMP: u8 = 46;
pub const OP_CMPLE: u8 = 47;
pub const OP_OP_CALL: u8 = 48;
pub const OP_SAVE: u8 = 49;
pub const OP_LOAD: u8 = 50;
pub const OP_PRINT: u8 = 51; // 调试：打印槽位前 n 个 f32
pub const OP_SHAPE: u8 = 52; // 声明激活槽形状: dst=槽位, a=rows, b=cols（零开销元指令）
pub const OP_BIAS: u8 = 53; // 逐行加偏置: dst += broadcast(bias)  (M,N) += (N)
// ---- 反向指令（P3：反向也是指令流，由 .vml 脚本显式编写）----
// 梯度区与常量池同布局：BWD_* 的 dst 指向「梯度槽」（汇编器按 g<name> 约定分配）
pub const OP_GEMM_BWD_W: u8 = 60; // dw[N,K] += dy[M,N]^T · x[M,K]
pub const OP_GEMM_BWD_X: u8 = 61; // dx[M,K]  = dy[M,N] · w[N,K]
pub const OP_BIAS_BWD: u8 = 62; //   db[N]   += Σ_i dy[i][j]
pub const OP_ACT_BWD: u8 = 63; //    dx      = dy ⊙ act'(ypre)
pub const OP_CE_BWD: u8 = 64; //    交叉熵+softmax 反向，写 dlogits，返回 loss
pub const OP_MSE_BWD: u8 = 65; //   MSE 反向，写 dy，返回 loss
pub const OP_ZERO: u8 = 66; //      清零槽位（每步开始把梯度区清零）
pub const OP_ADAM: u8 = 67; //      Adam 就地对常量池做一步更新（需求 2 核心）
pub const OP_SGD: u8 = 68; //       SGD 就地更新
pub const OP_LOSS_ADD: u8 = 69; //  累加 loss 到标量槽
pub const OP_CLIP_GRAD: u8 = 70; // 梯度裁剪
pub const OP_WRITECOL: u8 = 71; // 写列切片: dst[.., off..off+C] = src[.., 0..C]
// ---- P6 动态稀疏 ----
pub const OP_EWMAX: u8 = 72; // 逐元素取较大: dst = max(a, b)
pub const OP_GEMASK: u8 = 73; // 阈值转掩码: dst = (a >= 0) ? 1 : 0
// ---- P7 反向算子扩展（编号已冻结，见 docs/P7_分工与接口契约.md §2.1）----
pub const OP_SOFTMAX_BWD: u8 = 80; // dst=dx, a=dy, b=y(softmax输出)
pub const OP_LN_BWD: u8 = 81; // dst=dx, a=dy, b=x；dg/db 写进紧邻槽
pub const OP_OUTER_BWD: u8 = 82; // dst=dv, a=ds, b=v；dk 写 ins.dt
pub const OP_EMB_BWD: u8 = 83; // dst=dtable, a=dy, b=row 索引槽
pub const OP_ADAMW: u8 = 84; // dst=参数, a=梯度, b=超参[lr,b1,b2,eps,wd]
pub const OP_LN_EPS: u8 = 85; // 设定后续 LN 的 eps：读 a 槽第 0 个元素（默认 1e-5）
// ---- P8 BPTT：按 step 保存/恢复激活 ----
// 问题：激活槽只有一份，step 之间被覆盖，反向拿不到历史激活。
// 解法：不引入"自动保存"（那要改动整个执行模型），而是给脚本两条显式指令，
//       由脚本在正向时把需要的历史 push 到环形缓冲，反向时 pop 回来。
//       这与"反向也是指令流"的既有设计一致 —— 时序由脚本掌控，引擎不猜。
pub const OP_PUSH: u8 = 86; // push 槽位到 step 栈：a=源槽, dst=栈槽, b=第几个 step
pub const OP_POP: u8 = 87; //  pop 第 b 个 step 的内容到 dst 槽：a=栈槽
pub const OP_STACK_INIT: u8 = 88; // 初始化 step 栈：a=栈槽, b=最大 step 数（b 槽 = 阈值广播源）
// ---- P10 运行时循环（把「逐层/逐 head 编译期展开」变成运行时循环）----
//
// 动机（白皮书 §13.5 实测）：ESERNN 的指令数 = 161 + 154×(layers-1)，
// 与 hidden 无关 —— 说明**逐层展开是随层数线性膨胀的那一项**。
// MiniCPM 是 24 层 × 16 头，展开约 4800 条指令 ≈ 77 KB 指令流。
// 有了运行时循环，24 层只需写**一份 layer 模板**。
//
// 设计：倒计数 + 落到 0 退出（可如此省掉「初始化比较」和独立自增：
// 循环体尾部一条指令即可）。计数器就是**普通激活槽**（SHAPE ctr,1,1），
// **不新增寄存器堆、不加变长指令、不改 .vmb 格式**。
pub const OP_LOOP_CNT: u8 = 89; // ctr = ctr - 1; if ctr > 0 { ip = dst }
// ---- P10-B: RoPE（旋转位置编码）----
//
// 为什么需要专门的指令（MiniCPM / Llama 全系都要）
// ----------------------------------------------
// RoPE 是「按位置旋转 q/k 的每一对分量」。没有它时，P10-B 的 layer-0
// 切片只能用 **numpy 从 HF 导出的 sin/cos 常量表** 绕过 —— 那样能验证
// 其余算子，但**位置一多就把表撑爆**：S=131072、hd=128 时 cos+sin 是
// 131072×128×4×2 = 134 MB 常量，而且每个新序列都要重算/重导。
//
// 有了这条指令，cos/sin 在**运行时按位置算**（还是用引擎自己的 exp_f，
// 与 P9 还清的精度债一致），常量池里只留 theta 一个标量。
//
// 语义（与 HF `rotate_half` 等价，即「前半/后半」配对）
// --------------------------------------------------
//   对每一行 r（位置 pos）、每一对 i in [0, hd/2)：
//       x1 = src[r, i]         x2 = src[r, i + hd/2]
//       a  = pos * theta^(-2i/hd)
//       dst[r, i]        = x1*cos(a) - x2*sin(a)
//       dst[r, i+hd/2]   = x1*sin(a) + x2*cos(a)
//
// 编码（16B 定长指令塞得下，不需要第 5 操作数）
// -------------------------------------------
//   dst = 输出槽（[S,hd]）
//   a   = 源槽（[S,hd]，通常是 q 或 k）
//   b   = theta 标量槽（只读它的第 0 个元素）
//   dt  = **起始位置** pos0（u16）：第 r 行用 pos0 + r
//
// 为什么位置放进 dt 而不是再加操作数：dt 是 u16 且这条指令不需要
// 「第 4 个槽」，所以正好空着。解码时按位置增量算，配合 LOOP_CNT
// 就能覆盖任意长的序列，而指令流长度与序列长度**无关**。
pub const OP_ROPE: u8 = 90;
// 逐元素 sin / cos（P10 新增，供 RoPE 与调试用）。
// 为什么单独暴露：`OP_ROPE` 内部要用，但脚本侧也需要能验证三角函数精度
// （P10 的 tools/p10_sincos_accuracy.py 就是这么测的 —— 直接调真引擎，
//  而不是 Python 复刻，因为复刻脚本在这个项目里错过两次）。
// 编码：dst=输出, a=源（逐元素）
pub const OP_SIN: u8 = 91;
pub const OP_COS: u8 = 92;
// ======================= P16: 连接稀疏（connection sparsity）=======================
//
// 两条**必须分开**的原则，本段把它们的编码钉死：
//   (1) 连接稀疏     —— 每个神经元只保留 top-k 条连接（k0=20）
//   (2) 连接稀疏激活 —— **只算存活的连接，零连接跳过**
// 二者共享同一个量：连接激活强度 s_i = w_i · x_i。
//
// 为什么不是「稠密算完再乘 0」：本项目上一代的 self-designed sparse 模型
// （前身项目 sparse_dyn）正是那么做的，README 自陈实测稠密 110ms vs 稀疏
// 148ms —— **不降反升**。乘 0 省的是 FLOPs 的账，不是时间的账。
// 所以这里的两条指令都按「跳过」实现，并且 `OP_SPARSE_MATVEC` 把
// **实际执行的乘加次数**作为可测量输出写出来，让「跳过」可以被证伪。
//
// ---- OP_TOPK_BY_STRENGTH (93) ----
//   dst = 输出槽（须为 [L,2] 的整数表：第 0 列 = 存活槽下标，第 1 列 = 该槽的强度）
//   a   = 强度槽（逐连接，[L,1] 或 [1,L]）
//   b   = k 标量槽（读它的第 0 个元素，四舍五入成整数）
//   dt  = 未使用（k 走槽位而不是 dt —— 见 interp.rs 里的理由）
//
//   设计决策（必须写下）：输出**索引对**，而不是「就地清零非 top-k」。
//   理由见 interp.rs 该分支的长注释 —— 一句话：清零版与 OP_SPARSE_MATVEC
//   无法 bit-exact（`acc + 0.0*w` 在 acc = -0.0 时翻转符号，IEEE 不保证）。
//
//   硬约束：k > L **硬报错**，不静默截断（截断会让「要求 k0=20 的模型」
//   在 L<20 时静默退化成稠密，而那正是本任务要消灭的静默降级）。
//
// ---- OP_SPARSE_MATVEC (94) ----
//   dst = 输出槽，形状 [n_neurons] 或 [1,n_neurons]
//   a   = CSR offsets 槽（u32-as-f32，长度 n_neurons+1，**前缀和**）
//   b   = CSR src 槽（u32-as-f32，长度 = 连接总数；每项是输入神经元下标）
//   fl  = 低 6 位：第 4 个操作数（连接权重槽）的槽号；bit6 = FL_NNZ 计数标志
//   dt  = 输入向量 x 的槽号
//
//   为什么权重槽走 fl 低位而不是再占一个字段：P16 的一条硬要求是
//   **不加宽指令**（16B 定长）。op:u8|fl:u8|dt:u16|dst:u32|a:u32|b:u32 已满，
//   `fl` 只用了 bit0..bit5，低 6 位正好放一个 ≤63 的槽号（P16 的层级
//   槽表很小；超过 63 由汇编器**硬报错**，不静默截断）。
//
//   输出第 n_neurons 个元素在 FL_NNZ 时是**实际执行的乘加次数**（f32），
//   所以 dst 的形状是 [n_neurons+1]。这不是调试打印（那要 core::fmt，
//   本项目为省 3KB 刻意不用），而是**留在激活区里的证据**：
//   工具把它与「稠密等价 n_neurons²」并排报出来，跳过与否立刻可判。
//
// ============================ P16-5 接口契约：两条指令如何组合 ============================
//
// 这是 93 与 94 之间**唯一**的耦合点，写在这里是因为它是接口契约，不是实现细节。
//
// ---- 93 在分段模式（`seg`，fl bit7）下的输出 ----
//   a   = 强度槽，形状 [n, m]      —— **一行一个神经元**
//   dst = 选择表，形状 [n, 2k]     —— 每神经元一段，段长 2k
//   第 j 段的第 q 项（q < k）：
//       dst[j, 2q]   = 该行内第 q 个存活连接的**行内相对下标** rel（升序）
//       dst[j, 2q+1] = 该连线（行内下标 rel）的强度
//   存活不足 k 个的尾巴**填 0**。
//
// ---- 94 在门控模式（`selgate`，fl bit7）下如何消费它 ----
//   读第 j 段，对每个 q：
//       绝对连接下标  t   = offsets[j] + rel        <-- **加行首偏移**
//       src（输入神经元） = src[t]
//       权重              = weights[t]
//       项                = weights[t] * x[src[t]]
//   升序累加（与未门控路径的 CSR 行走顺序**相同**，这是 bit-exact 的前提）。
//
// ⚠️ **为什么要加 o0**：93 的编号是「相对它拿到的那一行」的 0..m-1，
//    94 的编号是「相对整个 CSR」的 0..nnz-1。两者**只在第 0 行一致**。
//    第一版门控实现就是漏了这个 o0，被引擎自己的守卫当场抓住：
//        SPARSE_MATVEC: sel index outside its CSR row   j=0 idx=8 row=[0,6)
//    这正是「守卫发现设计错」而不是「守卫碍事」的一个例子，
//    所以那条守卫**保留**，它守的是这个契约。
//
// ---- 93 的扁平模式（不带 `seg`）----
//   仍然可用：dst 为 [L,2]，对**整个数组**取一次 top-k，下标相对数组首。
//   平坦模式**不能**驱动 94 的 CSR 行走（编号不同源），只能用于
//   「把整条候选表排个序」这类用途。task-8 的用例就是它。
//
// ---- 槽位约定 ----
//   sel 槽必须**紧跟 src 槽之后**（引擎按 `src + 1` 取）。
//   与 LN_BWD（gamma/dg/db）、OUTER_BWD（k/dk）同一套相邻约定；
//   同它们一样，**汇编器硬校验**（不是信任）：第 6 个操作数不是 src+1 就 panic。
pub const OP_TOPK_BY_STRENGTH: u8 = 93;
pub const OP_SPARSE_MATVEC: u8 = 94;
// ---- P13 task E: 运行时常量基址（把 24 层展开变成运行时循环）----
//
// 动机（P10 §16.8 未做项 5，也是任务书 E）：
//   P10 的 `LOOP_CNT` 已经能跑 24 次迭代，P13 的 KV cache 也复用了它。
//   但**权重寻址是静态的** —— 槽位到常量的绑定在装载时做一次，
//   所以循环体每轮读到的都是同一层的权重。要让「一份 layer 模板」跑 24 层，
//   循环需要一个**运行时可变**的常量基址。
//
// 为什么这样设计（先量了产物布局才定的）：
//   实测本产物每个「逐层常量」的步长是**均匀的** —— 16 个逐层常量
//   （in_ln/q/q_sc/k/k_sc/v/v_sc/o/o_sc/post_ln/gate/gate_sc/up/up_sc/down/down_sc）
//   在常量池里全部相隔 28,383,232 字节（tools 里记录了这次测量）。
//   所以    weight(L, name) = base(name) + L * 28,383,232
//   循环每轮只需把基址加一个固定步长。
//
// 语义：dst 未使用；a = 持有**字节增量**的标量槽（f32，四舍五入成整数）。
//   layer_base += delta
//
// ⚠️ 安全边界（必须有，否则是静默读到垃圾）：
//   基址默认 0 —— **未使用本指令的程序逐字节不受影响**。
//   每次取常量时校验 `const_off + layer_base` 仍落在常量池内，
//   越界直接 exit(9)。否则一个跑飞的 LOOP_CNT 会读到激活区，
//   得到**看起来像数字的垃圾**，这正是本项目一律改成硬失败的那一类。
pub const OP_LAYERB: u8 = 7;

// 指令标志位（同样属于 opcode 语义契约；从 core/src/format.rs 逐字节搬运）
/// `OP_SPARSE_MATVEC` 的 fl bit6：把**实际执行的乘加次数**写到 dst[n_neurons]。
///
/// 为什么用一位标志而不是无条件写：无条件写会把 dst 的形状改成 [n+1]，
/// 于是「只想拿 y」的程序必须为测量多分一格并记得忽略它 —— 那样
/// **生产路径在为验收仪器付代价**。加一位标志后，生产路径的 dst 就是
/// 干净的 [n]，测量路径显式打开它。两种用法各自自洽。
///
/// 注意 bit6 = 1<<6 = 64，与上面 6 位槽号掩码恰好不重叠。
pub const FL_NNZ: u8 = 1 << 6;
/// `OP_SPARSE_MATVEC` 的 fl bit7：**SEL-GATED 模式** —— 让 `k` 真正进入前向算术。
///
/// ============================ 为什么需要它（P16-5 补的缺口）============================
/// task-8 把两条指令各自做对了，但它们**不组合**：`OP_TOPK_BY_STRENGTH` 写
/// `sel[L,2]` 选择表，而 `OP_SPARSE_MATVEC` 只读 CSR 与 x，**从不回读 sel**。
/// 后果是实测出来的（不是推断）：固定 CSR 与 x，k=2/5/24 的 y
/// **逐位完全相同**，nnz 恒为稠密值 24。也就是说：
///   * 「连接稀疏（每神经元留 top-k）」这一半算了，但**算了没人用**；
///   * 「连接稀疏激活（只算存活连接）」这一半，省的只是 CSR 里本来就没有的项，
///     **与 k 无关**。
/// 于是 task-9 的稀疏率扫描在任何比例下都测不出拐点 —— 因为 k 根本不是
/// 前向的输入。这个缺口不补，「稀疏率 vs 精度」这条曲线不可能有信号。
///
/// ============================ 三条设计理由（P16-5 的核心决策）============================
/// 选的是 **A′：在既有 94 号指令上加一个 sel-gated 模式位**，而不是
/// brief 里的 A（新算子）或 B（构建期剪 CSR）。三条理由，逐条可验证：
///
/// 1. **不给指令加宽、不加新 opcode 号。**
///    16B 定长是 P16 的硬约束；`fl` 的 bit0..5 已给权重槽号、bit6 已给
///    FL_NNZ，bit7 是最后一个空闲位。复用既有指令意味着**不消耗新的号段**，
///    因此不可能与任何其他成员正在分配的 opcode 相撞。
///
/// 2. **未门控路径**逐位**不变（这一条是可验证的承诺，不是形容词）。**
///    bit7=0 时引擎走的分支、读的槽、做的浮点加法**一字未改** ——
///    门控只是在一个 `if` 里加了另一条分支，原分支的机器码路径不受影响。
///    所以 task-8 已经测过的两件事不可能回归：
///      * 16/16 条守卫（它们全部在门控检查**之前**，两条路径共用）；
///      * 单层 bit-exact 证明（未门控路径的求和序未变）。
///    验收方式不是「我说不变」，而是**重跑 task-8 的三个工具**（见
///    tools/p16_sparse_ops.py / p16_guardprobe.py / p16_asm_wiring.py）。
///
/// 3. **它在推理期真的使用 `sel`。**
///    brief 的 Design B 是**构建期**按 k 剪 CSR：那样 k 只是汇编器的一个
///    参数，产物里根本没有 k，引擎也无从知道 k 是多少 —— 等于没测到
///    「连接稀疏激活」这一半。A′ 让 k 经 `sel` 流进运行时求和，
///    这才是 P1 声称的机制。
///
/// ============================ 门控语义（与未门控模式的差异，共 3 处）============================
///   src 槽：未门控时是 CSR 的 src 数组（每项 = 输入神经元下标）；
///           门控时是 `sel[L,2]` 选择表（第 0 列 = 连接下标，第 1 列 = 强度）。
///   fl bit6 (FL_NNZ)：两种模式都可用；但门控模式下它数的**只是被选中的项**，
///           于是「省了多少」这件事本身也随 k 变化 —— 这正是它作为证据的价值。
///   offsets：两种模式含义相同（逐神经元的 CSR 前缀和）。
///
/// 门控模式的**新守卫**（每条都有自己的消息）：sel 表宽度必须恰好为 2；
/// sel 里的连接下标必须落在 `[offsets[j], offsets[j+1])` 这一段之内
/// —— 越界意味着 `sel` 与 CSR **互不一致**，那是「看似合理的垃圾数」的
/// 典型来源，一律 exit(9)。
///
/// bit7 = 1<<7 = 128，与低 6 位槽号掩码、bit6 的 FL_NNZ 三者互不重叠。
pub const FL_SELGATE: u8 = 1 << 7;
// ---- 指令标志位 fl ----
// P8：fl 从 u16 收窄到 u8（只用到 bit0~bit5），腾出的字节让 `dt` 加宽到 u16。
// 目的：4 操作数指令（LN / LN_BWD / OUTER_BWD）的第 4 操作数槽号不再被截断到 8 位。
pub const FL_ACC: u16 = 1 << 0; // C += A*B 而非 C = A*B
pub const FL_TRANS_B: u16 = 1 << 1; // B 转置
pub const FL_BIAS: u16 = 1 << 2; // 加偏置（b 槽为偏置）
pub const FL_BROADCAST: u16 = 1 << 3; // 逐元素广播
pub const FL_CAUSAL: u16 = 1 << 4; // softmax 因果掩码
pub const FL_NEG: u16 = 1 << 5; // 取负
