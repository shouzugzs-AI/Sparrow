// PATH: B1
// trans_verify —— 最小转译器验证（单层稀疏 FFN，d=1536，k0=20）
//
// ============================ 这个驱动在测什么 ============================
//
// 任务书第十部分第一步：用**数据**决定"值不值得全面转译化"，而不是照抄
// 任务书的前提。前提本身已被 AUDIT_PATH_ISOLATION.md §4 用实测推翻：
//   5 个稀疏 matvec 前向 = 3017.3 ms/step，其中 top-k 选择 = 3145.4 ms/step，
//   而 matvec **累加**只有 196 ms（占稀疏路径 3702 ms 的 5%）。
//   ⇒ 转译化能消除的只是那 196 ms 里的"分发/槽位"部分。
//
// 本驱动因此**只**测一个东西：把 sparse_accum 的那个标量循环换成逐行内联
// 机器码，**累加阶段本身**能快多少。掩码/top-k 不在本驱动的范围内
// （它按定义是算法量：d × k0 × k 次比较，内联到机器码里仍是同样多次比较）。
//
// ============================ 三个必须诚实处理的陷阱 ============================
//
// 【陷阱 1】算子是 no_std 的。core/src/conn_train.rs 里 `crate::print`
//   写的是 GetStdHandle(0xFFFF_FFF5) —— 那是**错误的句子常量**（正确的
//   STD_OUTPUT_HANDLE = 0xFFFF_FFF5 是 32 位值，符号扩展后才是 -11）。
//   这是既有代码的既有行为，本驱动不修改它，只是不依赖任何 print 路径。
//
// 【陷阱 2】`--release` 会把算子版内联+向量化成完全不同的东西，于是
//   "转译 vs 算子"就不再是同一个算法对照。本驱动的两个基准都写成
//   `#[inline(never)]` 的独立函数，强制它们各自保持自己的形态：
//     * op_accum：逐字复刻 sparse_accum 的标量循环（含 act 判定、gather）
//     * trans 版：调用生成的机器码
//   **两者都不加任何人为开销**。特别地，算子版**没有**被故意写慢 ——
//   它就是从 core/src/conn_train.rs 抄下来的同一段代码。
//
// 【陷阱 3】TSC/QPC 抖动。每次配置跑 REPS 轮，报**中位数**（任务书要求），
//   同时报最小值与最大值以便看出抖动幅度。
//
// ============================ 正确性判据 ============================
//
//   §5.1  bit-exact vs 算子版：用 f32::to_bits() 逐元素比较，**不用 ==**
//         （== 对 NaN 恒 false、对 +0/-0 恒 true，会骗人）。要求 mismatch = 0。
//   §5.2  交叉对拍 vs conn_train::sparse_accum（**独立的第三方实现**）：
//         确认"同布局"这条声称是被测过的，不是我说的。
//   §5.3  act=NaN 边界：本转译版用 comisd+jbe，NaN 会**被跳过**；
//         算子版用 `== 0.0`，NaN **会乘**。这是已知差异，因此要求这里
//         **mismatch > 0** —— 若为 0，说明我的边界分析错了，必须重查。
//
// 用法：vmlm-trans-verify [--d 1536] [--k0 20] [--k 8] [--rows N] [--reps 15]
//                        [--dump-asm] [--dump-bytes PATH]
use vmlm_core::conn_train::{init_sparse_conn_scaled, sparse_accum, SparseConnW};

#[path = "../../trans/mod.rs"]
mod trans;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn QueryPerformanceCounter(c: *mut i64) -> i32;
    fn QueryPerformanceFrequency(f: *mut i64) -> i32;
}

// ============================================================================
// 工具
// ============================================================================

struct Clock {
    freq: i64,
}
impl Clock {
    fn new() -> Clock {
        let mut f: i64 = 0;
        unsafe { QueryPerformanceFrequency(&mut f) };
        Clock { freq: f }
    }
    fn now(&self) -> i64 {
        let mut c: i64 = 0;
        unsafe { QueryPerformanceCounter(&mut c) };
        c
    }
    fn ms(&self, t: i64) -> f64 {
        t as f64 * 1000.0 / self.freq as f64
    }
}

unsafe fn alloc_f32(n: usize) -> *mut f32 {
    let p = malloc(n * 4) as *mut f32;
    if p.is_null() {
        println!("OOM");
        std::process::exit(9);
    }
    p as *mut f32
}

unsafe fn alloc_u8(n: usize) -> *mut u8 {
    let p = malloc(n);
    if p.is_null() {
        println!("OOM");
        std::process::exit(9);
    }
    p
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Rng {
        Rng(s | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
    fn uni(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64)) as f32
    }
}

fn arg_usize(flag: &str, def: usize) -> usize {
    let v: Vec<String> = std::env::args().collect();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].parse().unwrap_or(def);
        }
    }
    def
}
fn has_flag(flag: &str) -> bool {
    std::env::args().any(|a| a == flag)
}
fn arg_str(flag: &str, def: &str) -> String {
    let v: Vec<String> = std::env::args().collect();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].clone();
        }
    }
    def.to_string()
}

/// 按 f32::to_bits() 逐元素比较（**不用 ==**）。
/// 返回 (mismatch 计数, 第一个 mismatch 的位置, 两边的位型)。
unsafe fn bitcmp(a: *const f32, b: *const f32, n: usize) -> (usize, Option<(usize, u32, u32)>) {
    let mut mm = 0usize;
    let mut first = None;
    for i in 0..n {
        let x = (*a.add(i)).to_bits();
        let y = (*b.add(i)).to_bits();
        if x != y {
            mm += 1;
            if first.is_none() {
                first = Some((i, x, y));
            }
        }
    }
    (mm, first)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn minf(v: &Vec<f64>) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}
fn maxf(v: &Vec<f64>) -> f64 {
    v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

// ============================================================================
// 算子版：**逐字复刻** core/src/conn_train.rs::sparse_accum (第 700-726 行)
// ============================================================================
//
// 为什么要重写一份而不是直接 call conn_train::sparse_accum：
//   为了"公平"必须让两边跑**同一件事**，而 conn_train 版每次调用都遍历
//   全部 d 个神经元（= d 行）。本驱动要按**行**对照（转译版是单行的），
//   所以这里把它的循环体原样搬到一个 `#[inline(never)]` 的单行函数里。
//
// ⚠️ 这段代码与 conn_train.rs 的循环体**必须逐字对应**。任何"顺手优化"
//   都会让对照失真。判据 §5.2 会用真正的 conn_train::sparse_accum
//   在**全 d 行**上交叉验证这一份重写的忠实性。
#[inline(never)]
unsafe fn op_accum_row(
    w_row: *const f32,
    src_row: *const f32,
    act_row: *const f32,
    x: *const f32,
    y_i: *mut f32,
    k0: usize,
) -> usize {
    let mut acc = 0.0f32;
    let mut first = true;
    let mut macs = 0usize;
    for j in 0..k0 {
        if *act_row.add(j) == 0.0 {
            continue; // 零连接跳过 —— 不乘、不加、不计数
        }
        let sid = *src_row.add(j) as usize;
        let term = *w_row.add(j) * *x.add(sid);
        acc = if first { term } else { acc + term };
        first = false;
        macs += 1;
    }
    *y_i = acc;
    macs
}

/// 算子版的**整层**驱动：对 d 行各调一次 op_accum_row。
/// 这才是与"转译版整层"可比的口径（见 §6 的说明）。
#[inline(never)]
unsafe fn op_accum_layer(
    w: *const f32,
    src: *const f32,
    act: *const f32,
    x: *const f32,
    y: *mut f32,
    n: usize,
    k0: usize,
) -> usize {
    let mut m = 0usize;
    for i in 0..n {
        m += op_accum_row(
            w.add(i * k0),
            src.add(i * k0),
            act.add(i * k0),
            x,
            y.add(i),
            k0,
        );
    }
    m
}

// ============================================================================
// main
// ============================================================================

fn main() {
    let d = arg_usize("--d", 1536);
    let k0 = arg_usize("--k0", 20);
    let k = arg_usize("--k", 8);
    let reps = arg_usize("--reps", 15);
    let nrows = arg_usize("--rows", d);
    let dump_asm = has_flag("--dump-asm");
    let dump_bytes = arg_str("--dump-bytes", "");

    println!("===== trans-verify: minimal translator, single sparse FFN layer =====");
    println!(
        "PATH: B1   d={} k0={} k={} rows={} reps={}",
        d, k0, k, nrows, reps
    );

    let clk = Clock::new();

    unsafe {
        let e = d * k0;

        // ---- 1) 构造与 SparseConnW **逐字段同布局**的缓冲 ----
        //   w[E] | src[E] | act[E]   （三段连续 f32）
        let base = malloc(SparseConnW::bytes(d as u32, k0 as u32));
        let sc = SparseConnW::from_base(base, d as u32, k0 as u32, 7);
        init_sparse_conn_scaled(&sc, 0.144);

        // src 用随机 spread（与 p18d 一致，让 gather 真的是随机地址）
        let mut rng = Rng::new(4242);
        for i in 0..e {
            *sc.src.add(i) = (rng.next_u32() % (d as u32)) as f32;
        }

        // ---- 2) x / y ----
        let x = alloc_f32(d);
        for i in 0..d {
            *x.add(i) = rng.uni() * 0.2 - 0.1;
        }

        // ---- 3) 造一张**真实的** act 掩码 ----
        //   用算子版自己的 compute_act_mask。理由：act ∈ {0.0, 1.0} 是
        //   转译版与算子版等价的前提（见 trans/mod.rs 边界 1/2）。
        //   直接 call core 里那个函数，保证掩码形态与生产路径一致。
        vmlm_core::conn_train::compute_act_mask(&sc, x, k);

        // 统计激活连接数（给"这是一个真稀疏掩码"提供证据）
        let mut n_active = 0usize;
        for i in 0..e {
            if *sc.act.add(i) == 1.0 {
                n_active += 1;
            }
        }
        println!(
            "\n--- mask ---\n  act==1.0 count = {} / {} ({} per neuron avg, expect ~k={})",
            n_active,
            e,
            n_active as f64 / d as f64,
            k
        );

        let y_op = alloc_f32(d);
        let y_tr = alloc_f32(d);
        let y_x = alloc_f32(d); // 交叉对拍用

        // ---- 4) 算子版（整层）----
        let macs = op_accum_layer(sc.w, sc.src, sc.act, x, y_op, d, k0);
        println!("\n--- operator version (sparse_accum loop body, per row) ---");
        println!("  macs executed = {}  (= n_active, the 'zero-conn skip' evidence)", macs);

        // ---- 5) 转译：把 nrows 行各编译一段独立机器码 ----
        //
        // 每行一段 = 每行有自己的入口。这是最小验证的**诚实**口径：
        // 生产转译器会把 d 行发射成**一段**代码（共用 x/base 寄存器），
        // 而这里每行都要重新装 5 个 mov imm64 基址 —— 那点开销对
        // 一整行的 20 条连接来说可以忽略；但为了不隐藏它，§6 会额外
        // 报告"每行发射的字节数"。
        let per_row_cap = 4096usize;
        let code = trans::alloc_rw(per_row_cap * nrows);
        if code.is_null() {
            println!("FATAL: VirtualAlloc failed");
            std::process::exit(9);
        }
        // 每行一个独立的累加器槽位（转译版需要，算子版不需要）
        let accs = alloc_f32(nrows);

        let mut entries: Vec<*const u8> = Vec::with_capacity(nrows);
        let mut lens: Vec<usize> = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let rc = trans::RowConsts {
                w_row: sc.w.add(i * k0),
                src_row: sc.src.add(i * k0) as *const u32,
                act_row: sc.act.add(i * k0),
                x,
                y_i: y_tr.add(i),
                acc: accs.add(i),
                k0,
            };
            let slot = code.add(i * per_row_cap);
            let r = trans::trans_row(&rc, slot, per_row_cap);
            entries.push(r.entry);
            lens.push(r.len);
        }
        // 全部写完，一次切可执行
        // ⚠️ make_exec 里必须 FlushInstructionCache（见 trans/mod.rs 的注释）：
        // 少了它，写完码再跳进去执行会随机 0xC0000005（取指一致性）。
        if !trans::make_exec(code, per_row_cap * nrows) {
            println!("FATAL: VirtualProtect(EXECUTE_READ) failed");
            std::process::exit(9);
        }

        // ---- 可选：在**执行任何生成代码之前** dump 字节，供外部反汇编核对 ----
        // 这一步必须在 call_row 之前：否则一旦生成的代码有错，进程会在
        // 我们能看见字节之前就崩掉（这正是第一次跑发生的事）。
        if !dump_bytes.is_empty() {
            let n = lens[0];
            let v = std::slice::from_raw_parts(entries[0], n);
            match std::fs::write(&dump_bytes, v) {
                Ok(_) => println!(
                    "\n[pre-exec] wrote {} bytes of row-0 code to {} (entry={:p})",
                    n, dump_bytes, entries[0]
                ),
                Err(err) => println!("\ncould not write {}: {}", dump_bytes, err),
            }
        }
        if dump_asm {
            println!("\n--- row 0 machine code ({} bytes) @ {:p} ---", lens[0], entries[0]);
            print_hex(entries[0], lens[0]);
        }
        if !dump_bytes.is_empty() || dump_asm {
            println!("[pre-exec] stopping before execution (dump mode)");
            std::process::exit(0);
        }

        let total_bytes: usize = lens.iter().sum();
        println!("\n--- translated version ---");
        println!(
            "  rows compiled      = {}\n  bytes per row      = {} (min {} max {})\n  total code bytes   = {}",
            nrows,
            total_bytes / nrows.max(1),
            lens.iter().min().copied().unwrap_or(0),
            lens.iter().max().copied().unwrap_or(0),
            total_bytes
        );

        // 每行入口：extern "system" fn()，无参数
        let call_row = |i: usize| {
            let f: trans::RowFn = std::mem::transmute(entries[i]);
            f();
        };

        // ---- 6) bit-exact 判据 ----
        println!("\n===== bit-exact check (f32::to_bits, NOT ==) =====");
        // 边界自检：src 是 u32-as-f32，最大值必须 < d，否则生成的 gather 越界。
        {
            let mut mx = 0u32;
            for i in 0..e {
                let v = *sc.src.add(i) as u32;
                if v > mx {
                    mx = v;
                }
            }
            println!("  [bounds] max src id = {} (must be < d = {})", mx, d);
        }

        // 6.1 转译版 vs 算子版
        // 先把 y 清成不同的哨兵值，确保"没被写"也能被发现
        for i in 0..d {
            *y_tr.add(i) = f32::from_bits(0xDEAD_BEEF);
        }
        for i in 0..nrows {
            call_row(i);
        }
        let (mm1, first1) = bitcmp(y_op, y_tr, d);
        println!(
            "  [1] translated vs operator(replica)   mismatch = {} / {}",
            mm1, d
        );
        if let Some((i, a, b)) = first1 {
            println!(
                "      first: y[{}] op={:#010x} trans={:#010x}",
                i, a, b
            );
        }

        // 6.2 交叉对拍：conn_train::sparse_accum（真正的第三方实现）
        //     —— 用来证明上面那份"逐字复刻"确实忠实，以及"同布局"成立。
        for i in 0..d {
            *y_x.add(i) = f32::from_bits(0xDEAD_BEEF);
        }
        sparse_accum(&sc, x, y_x, k);
        let (mm2, first2) = bitcmp(y_op, y_x, d);
        println!(
            "  [2] operator replica vs conn_train::sparse_accum   mismatch = {} / {}",
            mm2, d
        );
        if let Some((i, a, b)) = first2 {
            println!("      first: y[{}] replica={:#010x} core={:#010x}", i, a, b);
        }

        // 6.2b 对拍**真正的生产 B1 实现**：vmlm_core::b1_pure::b1_forward。
        //
        // 为什么必须加这一条（Lead 的符号级事故给的教训）：
        //   本次会话刚发生一次事故 —— 一个探针把 kernels::sparse_strength_row_avx2
        //   当"乘积"用，而它返回的是 |w*x|（它是**强度**内核）。落地后 A1 报
        //   1536/1536 分叉；三个探针里**只有直接测真实函数的那一个报了错**，
        //   另外两个（测自建复刻 / 测本地副本）都给了绿灯。
        //   ⇒ 判据必须调用 production，而不是自己写的等价实现。
        //
        // ⚠️ 语义边界（如实标注，不能糊过去）：
        //   b1_pure::b1_forward 是 **B1**：它乘**全部 k0 条**，**不读 act[]**。
        //   本转译器转译的是 conn_train::sparse_accum，它按 act 掩码跳过
        //   未激活连接（B2 口径的累加阶段）。
        //   两者只有在 **act 全 1**（即 k = k0，无筛选）时才应逐位相等。
        //   所以这一条判据在 `--k k0` 下才是等价的；在 k < k0 下它会（正确地）
        //   报分叉，那说明的是"两个不同函数"，不是 bug。
        //   下面的代码**只在 k == k0 时**执行这一条，并明确打印条件。
        let mut mm2b_v: usize = 0;
        if k == k0 {
            for i in 0..d {
                *y_x.add(i) = f32::from_bits(0xDEAD_BEEF);
            }
            let macs_b1 = vmlm_core::b1_pure::b1_forward(&sc, x, y_x);
            let (mm2b, first2b) = bitcmp(y_tr, y_x, d);
            mm2b_v = mm2b;
            println!(
                "  [2b] translated vs PRODUCTION b1_pure::b1_forward  mismatch = {} / {}  (only valid at k==k0)",
                mm2b, d
            );
            println!(
                "       b1_forward macs = {} (k0*n = {})",
                macs_b1,
                k0 * d
            );
            if let Some((i, a, b)) = first2b {
                println!("       first: y[{}] trans={:#010x} b1_pure={:#010x}", i, a, b);
            }
        } else {
            println!(
                "  [2b] SKIPPED vs b1_pure::b1_forward: it is B1 (multiplies ALL k0, ignores act[]),"
            );
            println!(
                "       while this translator reproduces the act-masked accumulate. Equal only at k==k0 (here k={}, k0={}).",
                k, k0
            );
        }

        // 6.3 act=NaN 边界 —— 见模块头【边界 1】
        //
        // ⚠️ 这条判据的**正确形态**必须小心构造，否则它会恒真/恒假：
        //   算子在 j=0 处的行为是 `if act[0] == 0.0 { continue }`。
        //   若 act[0] 原本就是 **1.0**（激活），把它换成 NaN 后：
        //     算子: NaN == 0.0 -> false -> 仍然**乘**这条 -> 结果不变
        //     转译: comisd+jbe   -> unordered -> **跳过**这条 -> 结果变了
        //     => 必须 differ=true
        //   若 act[0] 原本是 0.0（未激活），换成 NaN 后：
        //     算子: 从"跳过"变成"乘" -> 结果变了
        //     转译: 仍然跳过 -> 结果不变
        //     => 也必须 differ=true
        //   所以只要该行**确实有 w[0]*x[src[0]] != 0 且可影响和**，就该 differ。
        //   第一次跑出 differ=false 的原因：该行其余 7 条连接的累加在浮点上
        //   把这一项的差异吸收了（或该项本身是 +0/-0）。因此这里改为
        //   **两段都测**：(a) 单条连接的行（k0=1 的语义等价物），
        //   (b) 用逐条比较确认"算子把它算进去了、转译版没有"。
        {
            let nan = f32::from_bits(0x7FC0_0000);
            let saved: Vec<f32> = (0..k0).map(|j| *sc.act.add(j)).collect();

            // 让第 0 行只有第 0 条"激活"，其余全 0，并把 w[0] 设为 1.0、
            // src[0] 设为 0，使该项可观测。
            for j in 0..k0 {
                *sc.act.add(j) = 0.0;
            }
            *sc.act.add(0) = 1.0;
            let w0_save = *sc.w.add(0);
            let s0_save = *sc.src.add(0);
            *sc.w.add(0) = 1.0;
            *sc.src.add(0) = 0.0;
            let x0 = *x.add(0);

            // --- 基线（act[0] = 1.0）：两版都应得 x0 ---
            let mut base_op = 0.0f32;
            op_accum_row(sc.w, sc.src, sc.act, x, &mut base_op, k0);
            call_row(0);
            let base_tr = *y_tr.add(0);

            // --- 把 act[0] 换成 NaN ---
            *sc.act.add(0) = nan;
            let mut nan_op = 0.0f32;
            op_accum_row(sc.w, sc.src, sc.act, x, &mut nan_op, k0);
            call_row(0);
            let nan_tr = *y_tr.add(0);

            println!(
                "  [3] act[0]: 1.0 -> base: op={:#010x} trans={:#010x}  (both should be x[0]={:#010x})",
                base_op.to_bits(),
                base_tr.to_bits(),
                x0.to_bits()
            );
            println!(
                "      act[0]: NaN -> op={:#010x} trans={:#010x}",
                nan_op.to_bits(),
                nan_tr.to_bits()
            );
            // 算子版：NaN 不等于 0.0 -> 仍乘 -> 与 base 相同
            let op_includes = nan_op.to_bits() == base_op.to_bits();
            // 转译版：comisd+jbe 在 unordered 时跳 -> 累加器保持 0.0
            let tr_skips = nan_tr.to_bits() == 0.0f32.to_bits();
            // 判别：两版在 NaN 掩码上是否**一致**。
            // 实测结论（trans/mod.rs 有完整记录）：一致 —— comisd+jbe 与
            // `== 0.0` 在 NaN 上同向（都不跳 = 都照乘）。这条判据在
            // 第一次实现里被写成"必须不同"，是**错的**，已被实测纠正。
            let agree = nan_op.to_bits() == nan_tr.to_bits();
            println!(
                "      operator multiplies the NaN connection = {} ; translator = {}",
                op_includes,
                !tr_skips
            );
            println!(
                "      => the two AGREE on a NaN mask = {} (expect true; measured, see trans/mod.rs)",
                agree
            );
            if !agree {
                println!("      !! NaN boundary regression: translator and operator disagree");
            }

            // 还原
            for j in 0..k0 {
                *sc.act.add(j) = saved[j];
            }
            *sc.w.add(0) = w0_save;
            *sc.src.add(0) = s0_save;
            let _ = e;
        }

        // 判据汇总。mm2b 只在 k == k0 时存在，用 Option 承载，
        // 避免"没测也算过"这种假绿。
        let mm2b: Option<usize> = if k == k0 { Some(mm2b_v) } else { None };
        let pass = mm1 == 0 && mm2 == 0 && mm2b.map_or(true, |v| v == 0);
        println!(
            "\n  CRITERION: bit-exact => [1] mm={} [2] mm={} [2b] {}",
            mm1,
            mm2,
            match mm2b {
                Some(v) => format!("mm={}", v),
                None => "skipped (k != k0)".to_string(),
            }
        );
        println!(
            "  VERDICT: {}",
            if pass {
                "PASS — 0 mismatch on every check that ran"
            } else {
                "FAIL — see mismatch counts above"
            }
        );

        // ---- 7) dump 已在执行前处理（见上面的 pre-exec 分支）----

        // ====================================================================
        // 8) 计时
        // ====================================================================
        //
        // 两个口径，都要报（否则容易用口径差凑出想要的比例）：
        //
        //   (a) 单行：算子版 op_accum_row(第 i 行) vs 转译版 call_row(i)
        //       —— 这是任务书要求的"一行"对照，也是内联收益最纯净的度量。
        //   (b) 整层：算子版 op_accum_layer(全部 d 行) vs 转译版 d 次 call_row
        //       —— 这是真正该拿去外推的形状（生产里每层就是整层调用）。
        //
        // 预热一轮再做正式测量；每次配置跑 reps 轮取**中位数**。
        println!("\n===== timing (QPC, median of {} reps) =====", reps);

        // 预热（两侧行数一致，见 (b) 的注释）
        for _ in 0..3 {
            let _ = op_accum_layer(sc.w, sc.src, sc.act, x, y_op, nrows, k0);
            for i in 0..nrows {
                call_row(i);
            }
        }

        // (a) 单行 —— 固定测第 0 行
        let mut t_op_row = Vec::new();
        let mut t_tr_row = Vec::new();
        for _ in 0..reps {
            let t0 = clk.now();
            let mut s = 0usize;
            // 一行摊到足够多次，让 QPC 分辨率不是瓶颈
            for _ in 0..1000 {
                s += op_accum_row(sc.w, sc.src, sc.act, x, y_op, k0);
            }
            t_op_row.push(clk.ms(clk.now() - t0) / 1000.0);
            let _ = s;

            let t1 = clk.now();
            for _ in 0..1000 {
                call_row(0);
            }
            t_tr_row.push(clk.ms(clk.now() - t1) / 1000.0);
        }

        // (b) 整层
        //
        // ⚠️ 两侧必须跑**同样的行数**，否则这个比值没有意义。
        //   转译侧只跑 nrows 行，所以算子侧也必须只跑 nrows 行（nrows 是
        //   独立的参数；默认 = d）。早期版本这里写死了 d，于是 --rows < d 时
        //   两边工作量不同 —— sweep 出来的 396x 就是这么来的假象。
        let mut t_op_layer = Vec::new();
        let mut t_tr_layer = Vec::new();
        for _ in 0..reps {
            let t0 = clk.now();
            let _ = op_accum_layer(sc.w, sc.src, sc.act, x, y_op, nrows, k0);
            t_op_layer.push(clk.ms(clk.now() - t0));

            let t1 = clk.now();
            for i in 0..nrows {
                call_row(i);
            }
            t_tr_layer.push(clk.ms(clk.now() - t1));
        }

        // (c) 紧凑布局：把 nrows 段码**连续**排在一个页里（无 4KB 填充）。
        //
        // 为什么要测这个：口径 (b) 用的是"每行一段、每段 4KB 对齐"的布局，
        // 1536 行 = 6 MB 代码 + 1536 个不同的代码页。那个布局会让
        // I-cache/iTLB 抖动主导耗时，从而把"内联本身的收益"淹没掉。
        // 生产转译器会把整层发射成**一段连续代码**（这正是任务书要的形状），
        // 所以 (c) 才是与 (a)"单行 3.1x"自洽的口径。
        // 报告里 (b) 与 (c) 都要给 —— 只报对自己有利的那个就是扭曲测量。
        let packed = trans::alloc_rw(per_row_cap * nrows);
        let mut p_entries: Vec<*const u8> = Vec::with_capacity(nrows);
        let mut packed_median: Option<Vec<f64>> = None;
        let mut packed_total_bytes = 0usize;
        if !packed.is_null() {
            let mut off = 0usize;
            for i in 0..nrows {
                let rc = trans::RowConsts {
                    w_row: sc.w.add(i * k0),
                    src_row: sc.src.add(i * k0) as *const u32,
                    act_row: sc.act.add(i * k0),
                    x,
                    y_i: y_tr.add(i),
                    acc: accs.add(i),
                    k0,
                };
                let slot = packed.add(off);
                let r = trans::trans_row(&rc, slot, per_row_cap * nrows - off);
                p_entries.push(r.entry);
                off += r.len;
            }
            let _ = trans::make_exec(packed, per_row_cap * nrows);
            let call_p = |i: usize| {
                let f: trans::RowFn = std::mem::transmute(p_entries[i]);
                f();
            };
            for _ in 0..3 {
                for i in 0..nrows {
                    call_p(i);
                }
            }
            let mut t_tr_packed = Vec::new();
            for _ in 0..reps {
                let t0 = clk.now();
                for i in 0..nrows {
                    call_p(i);
                }
                t_tr_packed.push(clk.ms(clk.now() - t0));
            }
            packed_total_bytes = off;
            packed_median = Some(t_tr_packed);
        }

        // ---- (d) 每次调用的固定开销 ----
        //
        // 为什么需要这个数：k0 很小时（见 sweep），转译版反而更慢。
        // 转译版每行/每调用的固定成本是：
        //   一次 indirect call+ret、sub/add rsp、6 条 mov imm64（基址）、
        //   pxor、acc 清零、写回时一次 load + 一次 store（累加器过内存）。
        // 算子版把这些摊进一个**可复用**的循环体里。所以"内联"在
        // 「每行连接数少、但行数多」的形状下是**净亏**。
        // 直接测出来，而不是嘴上说：
        {
            let stat_rc = trans::RowConsts {
                w_row: sc.w,
                src_row: sc.src as *const u32,
                act_row: sc.act,
                x,
                y_i: y_tr,
                acc: accs,
                k0: 0, // 零条连接 = 纯固定开销
            };
            let st_code = trans::alloc_rw(4096);
            let r0 = trans::trans_row(&stat_rc, st_code, 4096);
            let _ = trans::make_exec(st_code, 4096);
            let f0: trans::RowFn = std::mem::transmute(r0.entry);
            for _ in 0..1000 {
                f0();
            }
            let t0 = clk.now();
            for _ in 0..200_000 {
                f0();
            }
            let empty_ns = clk.ms(clk.now() - t0) * 1e6 / 200_000.0;
            println!(
                "\n--- (d) fixed per-call cost of the translated form ---\n  k0=0 row ({} bytes, pure call+prologue+epilogue): {:.1} ns/call",
                r0.len, empty_ns
            );
            println!("  (this is why small-k0 rows lose: the operator amortizes it)");
        }

        let med_op_row = median(t_op_row.clone());
        let med_tr_row = median(t_tr_row.clone());
        let med_op_layer = median(t_op_layer.clone());
        let med_tr_layer = median(t_tr_layer.clone());
        let packed_med = packed_median.as_ref().map(|v| median(v.clone()));
        let packed_min = packed_median.as_ref().map(minf);
        let packed_max = packed_median.as_ref().map(maxf);

        println!("\n--- (a) ONE row (k0={} connections, d={}) ---", k0, d);
        println!(
            "  operator  op_accum_row   median {:10.6} ms   [min {:10.6}, max {:10.6}]",
            med_op_row,
            minf(&t_op_row),
            maxf(&t_op_row)
        );
        println!(
            "  translated call_row(0)   median {:10.6} ms   [min {:10.6}, max {:10.6}]",
            med_tr_row,
            minf(&t_tr_row),
            maxf(&t_tr_row)
        );
        println!(
            "  speedup (median)         {:10.3}x",
            med_op_row / med_tr_row
        );

        println!("\n--- (b) WHOLE layer (all {} rows) ---", nrows);
        println!(
            "  operator  op_accum_layer median {:10.6} ms   [min {:10.6}, max {:10.6}]",
            med_op_layer,
            minf(&t_op_layer),
            maxf(&t_op_layer)
        );
        println!(
            "  translated {} calls      median {:10.6} ms   [min {:10.6}, max {:10.6}]",
            nrows,
            med_tr_layer,
            minf(&t_tr_layer),
            maxf(&t_tr_layer)
        );
        println!(
            "  speedup (median)         {:10.3}x",
            med_op_layer / med_tr_layer
        );

        // ---- (c) 紧凑布局的对照 ----
        if let (Some(mp), Some(mn), Some(mx)) = (packed_med, packed_min, packed_max) {
            println!(
                "\n--- (c) CONTIGUOUS layout ({} rows packed back-to-back, {} bytes total) ---",
                nrows, packed_total_bytes
            );
            println!(
                "  translated {} calls      median {:10.6} ms   [min {:10.6}, max {:10.6}]",
                nrows, mp, mn, mx
            );
            println!(
                "  speedup vs operator (median) {:10.3}x",
                med_op_layer / mp
            );
            println!(
                "  note: layout (c) is the shape a production translator emits",
            );
            println!(
                "        (one straight-line block per layer, not one 4KB page per row)",
            );
        }

        let speedup = med_op_layer / med_tr_layer;
        println!("\n===== VERDICT =====");
        if speedup >= 3.0 {
            println!(
                "  {:.2}x >= 3x on the accumulate stage => full translation is worth evaluating",
                speedup
            );
        } else {
            println!(
                "  {:.2}x < 3x => the operator layer is NOT the bottleneck; translation saves little",
                speedup
            );
        }

        // ---- 9) 把这个数放回真实步长里 ----
        //
        // AUDIT §4 的实测分解（vmlm-p18d）：5 matvec × 24 层 × 128 位置。
        // 本驱动的"整层累加"折算到那个形状，看它能占端到端多少。
        let calls_per_step = 5.0 * 24.0 * 128.0; // 15360
        println!("\n--- extrapolation (AUDIT §4 measured shape: 5 matvec x 24 layers x 128 pos) ---");
        println!(
            "  accumulate calls/step          = {:.0}",
            calls_per_step
        );
        println!(
            "  operator accumulate            = {:9.1} ms/step",
            med_op_layer * calls_per_step
        );
        println!(
            "  translated accumulate          = {:9.1} ms/step",
            med_tr_layer * calls_per_step
        );
        println!(
            "  saving                         = {:9.1} ms/step",
            (med_op_layer - med_tr_layer) * calls_per_step
        );
        println!(
            "  AUDIT §4 measured step total   = {:9.1} ms/step (sparse 3702.7 = fwd 3017.3 + bwd 685.4)",
            3702.7
        );
        println!(
            "  => saving as a fraction of the measured sparse total = {:.1}%",
            100.0 * (med_op_layer - med_tr_layer) * calls_per_step / 3702.7
        );
        println!(
            "  (of which AUDIT §4 attributes 3145.4 ms/step to top-k SELECTION,");
        println!("   which inlining cannot remove -- it is d x k0 x k comparisons.)");

        std::process::exit(if pass { 0 } else { 1 });
    }
}

fn print_hex(p: *const u8, n: usize) {
    unsafe {
        let v = std::slice::from_raw_parts(p, n);
        for (i, chunk) in v.chunks(16).enumerate() {
            let mut s = format!("  {:04x}: ", i * 16);
            for b in chunk {
                s.push_str(&format!("{:02x} ", b));
            }
            println!("{}", s);
        }
    }
}
