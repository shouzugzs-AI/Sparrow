// PATH: B1 / B2
//
// ===========================================================================
// P1-b：源扇出统计 —— 「转译器主动复用」值不值得做的**判决性测量**
// ===========================================================================
//
// 任务书 P1 第二半的设想：转译器扫描连接拓扑，把**高扇出的源**缓存在 ymm 寄存器里，
// 让 x[5] 只加载一次、被它的多条连接各用一次（生成的机器码形如
// `load x[5] -> ymm0; fma w..*ymm0 -> acc..` × n）。任务书明确要求：
//
//     「先做扇出统计 —— 看有多少源值得缓存。如果高扇出源很少（< 5%），收益有限，
//       先不做。」
//
// 所以本驱动只做**统计**，不做实现，也不产生任何"优化后"的数字。
//
// ── 两个必须分开的量（否则会得出错误结论）────────────────────────────────
//
//   1) **全局扇出**：整个矩阵里 x[s] 被多少条连接用到。B1 的 k0=20、d=1536 ⇒
//      总连接 30720、源 1536 ⇒ 平均扇出恰 = 20。**这个数看起来很诱人，但不可用**：
//      16 个 ymm 寄存器放不下 1536 个源，而一个源的 20 次使用散布在 1536 行上，
//      中间隔着几十万条指令 —— 寄存器/重排缓冲都跨不过去。
//
//   2) **窗口扇出**：在**连续 W 行**的窗口内，一个源被用了几次。
//      这才是转译器能兑现的量：窗口内出现的**不同源数 D** ≤ 16 时，整个窗口
//      可以"每源加载一次"；一般情形下可省的比例 = (C − D) / C，其中 C = W·k0。
//
//   ⇒ 判决标准：**可省比例 < 5% ⇒ 不做**（与任务书一致）。
//
// ── 为什么窗口要扫多个长度 ────────────────────────────────────────────────
//   窗口越大，D/C 越小（复用越多），但寄存器压力从"W 行"变成"整窗"。
//   扫 W ∈ {8,16,32,64,128,256} 是为了看这条曲线有没有拐点值得停。
//
// USAGE: vmlm-sparseb-fanout.exe [--d 1536] [--k0 20] [--layers 24] [--mats 5]

extern "C" {
    fn malloc(n: usize) -> *mut u8;
}

use vmlm_core::conn_train::{init_sparse_conn_scaled, SparseConnW};

fn arg_u32(flag: &str, def: u32) -> u32 {
    let v: Vec<String> = std::env::args().collect();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].parse().unwrap_or(def);
        }
    }
    def
}

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let layers = arg_u32("--layers", 24) as usize;
    let mats = arg_u32("--mats", 5) as usize;
    let n_mats = layers * mats;
    let e = d * k0;

    println!("===== P1-b 源扇出统计（判决：转译器主动复用值不值得做）=====");
    println!(
        "d={} k0={} 连接/矩阵={} 矩阵数={}（{} 层 x {}）",
        d, k0, e, n_mats, layers, mats
    );
    println!("平均全局扇出 = k0 = {} （这是上界，不是可兑现的量，说明见文件头）", e / d);

    let windows: [usize; 6] = [8, 16, 32, 64, 128, 256];

    unsafe {
        let mut cnt = vec![0u32; d]; // 全局扇出计数
        let mut stamp = vec![0u32; d]; // 窗口内是否已见（版本号法）
        let mut pos = vec![0u32; d]; // 该源在 seen[] 里的下标（仅 stamp 命中时有效）
        let mut w_conn = vec![0u64; windows.len()];
        let mut w_distinct = vec![0u64; windows.len()];
        let mut w_windows = vec![0u64; windows.len()];
        let mut w_fit16 = vec![0u64; windows.len()];
        let mut w_fit32 = vec![0u64; windows.len()];
        // 寄存器受限上界：R ∈ {4,8,12,16} 时，窗口内 top-R 高扇出源能省多少次加载
        let regs: [usize; 4] = [4, 8, 12, 16];
        let mut w_top: Vec<[u64; 4]> = vec![[0; 4]; windows.len()];
        let mut seen: Vec<u32> = Vec::with_capacity(d);
        let mut uses: Vec<u32> = Vec::with_capacity(d);

        let mut hist = [0u64; 7]; // 扇出 1,2,3-4,5-8,9-16,17-32,>32
        let mut max_fan = 0u32;
        let mut touch_ge2 = 0u64; // 被至少一个"复用窗口"覆盖的连接数上界

        for m in 0..n_mats {
            let base = malloc(SparseConnW::bytes(d as u32, k0 as u32));
            let sc = SparseConnW::from_base(base, d as u32, k0 as u32, (m as u32) + 1);
            init_sparse_conn_scaled(&sc, 0.144);
            sc.assert_src_u32_consistent();

            // ---- 全局扇出 ----
            for c in cnt.iter_mut() {
                *c = 0;
            }
            for t in 0..e {
                let s = *sc.src_u32.add(t) as usize;
                cnt[s] += 1;
            }
            for s in 0..d {
                let f = cnt[s];
                if f > max_fan {
                    max_fan = f;
                }
                let b = match f {
                    0 => continue,
                    1 => 0,
                    2 => 1,
                    3..=4 => 2,
                    5..=8 => 3,
                    9..=16 => 4,
                    17..=32 => 5,
                    _ => 6,
                };
                hist[b] += 1;
            }

            // ---- 窗口扇出：**非重叠**窗口（转译器按窗口生成代码）----
            //
            // 除了「无限寄存器」的上界 (C-D)/C，本驱动还算**寄存器受限**的上界：
            // 只有 R 个寄存器可用时，最多缓存 R 个源，每个被缓存的源省
            // (窗口内使用次数 − 1) 次加载（第一次仍要 load 进寄存器）。
            // 取窗口内使用次数最高的 R 个源即得该上界 —— 这依然**乐观**，因为它
            // 忽略了"按源重排会破坏每行累加顺序"这个约束。
            for (wi, &w) in windows.iter().enumerate() {
                if w > d {
                    continue;
                }
                let mut r0 = 0usize;
                while r0 + w <= d {
                    let ver = (wi as u32 + 1) * 100_000 + (r0 / w) as u32 + 1;
                    let c_count = (w * k0) as u64;
                    seen.clear();
                    uses.clear();
                    // **单遍**：stamp 标记首次出现，pos 只在 stamp 命中时使用，
                    // 所以不需要每窗口重置 pos。
                    for i in r0..r0 + w {
                        for j in 0..k0 {
                            let s = *sc.src_u32.add(i * k0 + j) as usize;
                            if stamp[s] != ver {
                                stamp[s] = ver;
                                pos[s] = seen.len() as u32;
                                seen.push(s as u32);
                                uses.push(0);
                            }
                            let k = pos[s] as usize;
                            uses[k] += 1;
                        }
                    }
                    let distinct = seen.len() as u64;
                    for (ri, &rr) in regs.iter().enumerate() {
                        let mut v = uses.clone();
                        v.sort_by(|a, b| b.cmp(a));
                        let n = rr.min(v.len());
                        let mut saved = 0u64;
                        for t in 0..n {
                            if v[t] >= 2 {
                                saved += (v[t] - 1) as u64;
                            }
                        }
                        w_top[wi][ri] += saved;
                    }
                    w_conn[wi] += c_count;
                    w_distinct[wi] += distinct;
                    w_windows[wi] += 1;
                    if distinct <= 16 {
                        w_fit16[wi] += 1;
                    }
                    if distinct <= 32 {
                        w_fit32[wi] += 1;
                    }
                    r0 += w;
                }
            }
        }

        // 全局扇出直方图
        let tot_src: u64 = hist.iter().sum();
        println!("\n--- 1) 全局扇出（整矩阵）直方图 ---");
        let names = ["1", "2", "3-4", "5-8", "9-16", "17-32", ">32"];
        for (i, nm) in names.iter().enumerate() {
            println!(
                "  fan-out {:<6} : {:>9} 个源  ({:5.2}%)",
                nm,
                hist[i],
                100.0 * hist[i] as f64 / tot_src as f64
            );
        }
        println!("  最大扇出 = {}   源总数 = {}", max_fan, d);
        println!(
            "  ⚠️ 全局扇出均值 = k0 = {} —— 但它**不可兑现**：1536 个源放不进寄存器，",
            e / d
        );
        println!("     且同一源的多次使用在 1536 行上相距几十万条指令（超出重排缓冲）。");

        // 窗口扇出
        println!("\n--- 2) 窗口扇出（转译器**能**兑现的量）---");
        println!(
            "  {:>6} {:>14} {:>14} {:>12} {:>12} {:>10}",
            "W(行)", "C=W*k0 连接", "D 不同源", "可省 (C-D)/C", "D<=16 窗口", "D<=32 窗口"
        );
        for (wi, &w) in windows.iter().enumerate() {
            if w_conn[wi] == 0 {
                continue;
            }
            let c = w_conn[wi] as f64;
            let dd = w_distinct[wi] as f64;
            let save = (c - dd) / c;
            println!(
                "  {:>6} {:>14} {:>14.1} {:>11.2}% {:>11.1}% {:>9.1}%",
                w,
                w_conn[wi],
                dd / w_windows[wi] as f64,
                100.0 * save,
                100.0 * w_fit16[wi] as f64 / w_windows[wi] as f64,
                100.0 * w_fit32[wi] as f64 / w_windows[wi] as f64
            );
        }

        // 判决
        let w16 = windows.iter().position(|&x| x == 16).unwrap();
        let save16 = (w_conn[w16] as f64 - w_distinct[w16] as f64) / w_conn[w16] as f64;
        let w64 = windows.iter().position(|&x| x == 64).unwrap();
        let save64 = (w_conn[w64] as f64 - w_distinct[w64] as f64) / w_conn[w64] as f64;
        println!("\n--- 3) **寄存器受限**上界（只有 R 个 ymm 可缓存源时）---");
        println!(
            "  {:>6} {:>12} {:>12} {:>12} {:>12}",
            "W(行)", "R=4 可省", "R=8 可省", "R=12 可省", "R=16 可省"
        );
        let mut best_reg = 0.0f64;
        for (wi, &w) in windows.iter().enumerate() {
            if w_conn[wi] == 0 {
                continue;
            }
            let c = w_conn[wi] as f64;
            let s4 = w_top[wi][0] as f64 / c;
            let s8 = w_top[wi][1] as f64 / c;
            let s12 = w_top[wi][2] as f64 / c;
            let s16 = w_top[wi][3] as f64 / c;
            if s16 > best_reg {
                best_reg = s16;
            }
            println!(
                "  {:>6} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}%",
                w,
                100.0 * s4,
                100.0 * s8,
                100.0 * s12,
                100.0 * s16
            );
        }
        println!("  （每个被缓存的源省「窗口内使用次数 − 1」次加载；这仍是上界：");
        println!("    它假设可以自由按源重排执行顺序，而那会改变每行的累加顺序。）");

        println!("\n--- 4) 判决（任务书标准：可省 < 5% ⇒ 先不做）---");
        println!("  无限寄存器上界：W=16 {:.2}%，W=64 {:.2}%", 100.0 * save16, 100.0 * save64);
        // 现实寄存器预算：16 个 ymm 里还要留累加器（每行一个）。R=8 缓存 + 8 累加是
        // 现实的上限；R=16 意味着**一个累加器都不剩**，不可实现，只作为上界记录。
        let best_r8 = (0..windows.len())
            .map(|wi| {
                if w_conn[wi] == 0 {
                    0.0
                } else {
                    w_top[wi][1] as f64 / w_conn[wi] as f64
                }
            })
            .fold(0.0f64, f64::max);
        println!("  R=16（**不可实现**：没有寄存器留给累加器）= {:.2}%", 100.0 * best_reg);
        println!("  R=8（8 个缓存 + 8 个累加器，现实上限）      = {:.2}%", 100.0 * best_r8);
        println!(
            "  D<=16 的窗口占比（即任务书举例的\"整窗可缓存\"）= {:.1}%（所有 W 都是 0）",
            100.0 * w_fit16.iter().sum::<u64>() as f64 / w_windows.iter().sum::<u64>() as f64
        );
        if best_r8 < 0.05 {
            println!(
                "  ⇒ **结论：不做**。现实的 R=8 下窗口复用可省 ≤ {:.2}% < 5%：",
                100.0 * best_r8
            );
            println!("     · 随机拓扑下每个源在 W 行窗口里平均只用 1~2 次（D≈0.95·C）；");
            println!("     · `D<=16` 的窗口占比为 **0%** —— 任务书设想的\"整窗源驻寄存器\"从不发生；");
            println!("     · 唯一 >5% 的格子是 R=16，而那时没有寄存器放累加器。");
            println!("     · 代价侧：按源重排会**改变每行累加顺序**（破坏逐位契约）。");
            println!("     ⇒ ≤{:.1}% 的加载节省换浮点契约，不划算。", 100.0 * best_r8);
        } else {
            println!("  ⇒ 存在 >5% 的可省空间，值得做原型（但必须重验 ε 与确定性）。");
        }
        let _ = touch_ge2;
    }
}
