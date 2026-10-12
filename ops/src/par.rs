//! L2a 调度：**region 切分与归约**（从 `train/src/b1_full_train.rs` 逐字节搬运）。
//!
//! 边界（用户契约 3）：调度属于 L2a；L0（`vmlm-sys`）只提供线程原语与常驻池。
//! 本模块需要 std（常驻池是 `std::sync::Mutex/Condvar`）⇒ 由 `thread-std` feature 门控，
//! 引擎路径不启用。

use vmlm_sys::thread::*;

/// 把"捕获了裸指针的闭包"标成可跨线程共享。
/// 安全性依据：inter-op 里各线程只碰自己 slot 的私有缓冲（slot 由参数给定，且已按 slot 分段），
/// 共享的 token 维缓冲按 token 段互不重叠；权重在反向期间只读，梯度各写自己的 slot。
pub struct SharedFn<F>(pub F);
unsafe impl<F> Send for SharedFn<F> {}
unsafe impl<F> Sync for SharedFn<F> {}

// =====================================================================
// 常驻线程池（--persistent-threads 1，默认开）
// =====================================================================
// 原来每次 par_for 都 spawn+join；inter-op 每步要 spawn nslot 个、每层还要 spawn 多次，
// 线程创建/同步吃掉了并行收益（实测 1.9x，环境天花板 ~2.9x）。改成常驻 worker 只做"派发+等待"。
// 数值不变性：分片与 spawn 路径逐字相同（chunk=ceil(n/nt)，worker i 取 [i*chunk, min(.,n))），
// 每个 worker 内部累加顺序不变，跨线程归约仍在屏障之后 ⇒ 应与 spawn **逐位相同**。

/// 类型擦除的任务指针：把 fat pointer 拆成 (data, vtable) 两半，避开 trait object 的 'static 约束。
/// 安全性：调用方在 run() 里阻塞到所有 worker 完成才返回，任务引用在此期间一直有效。




pub fn par_for<F: Fn(usize, usize) + Sync>(n: usize, nt: usize, f: F) {
    if nt <= 1 || n <= 1 {
        f(0, n);
        return;
    }
    let nt = nt.min(n);
    if !IN_WORKER.with(|c| c.get()) {
        if let Some(p) = POOL.get() {
            if p.nt == nt && !p.is_shutdown() {
                p.run(n, &f);
                return;
            }
        }
    }
    let chunk = (n + nt - 1) / nt;
    let npin = LOGICAL_CPUS.load(std::sync::atomic::Ordering::Relaxed);
    let pin = PIN_THREADS.load(std::sync::atomic::Ordering::Relaxed) == 1 && npin > 0;
    std::thread::scope(|sc| {
        for t in 0..nt {
            let lo = t * chunk;
            let hi = ((t + 1) * chunk).min(n);
            if lo >= hi {
                break;
            }
            let fr = &f;
            let cpu = if pin { t * npin / nt } else { usize::MAX };
            sc.spawn(move || {
                if cpu != usize::MAX {
                    pin_current_thread(cpu);
                }
                fr(lo, hi);
            });
        }
    });
}/// 逐元素并行（for i in 0..n）
pub unsafe fn par_elem<F: Fn(usize) + Sync>(n: usize, nt: usize, f: F) {
    par_for(n, nt, |lo, hi| {
        for i in lo..hi {
            f(i);
        }
    });
}
