//! L0 常驻线程池（std 版，`thread-std` feature 门控）。
//!
//! 从 `train/src/b1_full_train.rs` **逐字节搬运**：`ErasedTask` / `PoolState` / `Pool` /
//! `POOL` / `IN_WORKER` / `impl Pool` / `init_pool` / `shutdown_pool` / `PIN_THREADS` / `LOGICAL_CPUS`。
//! 唯一改动：可见性 `fn` → `pub fn`、`Pool.nt` → `pub`（trainer 的 `par_for` 要读）。
//! 边界：**pool 原语**属 L0；"region 怎么切分/归约" 仍属 L2a（留在 trainer 的 `par_for`）。
#![allow(dead_code)]

use super::*;

#[derive(Clone, Copy)]
struct ErasedTask {
    data: *const (),
    vtable: *const (),
}
unsafe impl Send for ErasedTask {}
unsafe impl Sync for ErasedTask {}

impl ErasedTask {
    fn from_fn(f: &(dyn Fn(usize, usize) + Sync)) -> Self {
        let fat: *const (dyn Fn(usize, usize) + Sync) = f;
        let (data, vtable): (*const (), *const ()) = unsafe { std::mem::transmute(fat) };
        ErasedTask { data, vtable }
    }
    fn call(&self, lo: usize, hi: usize) {
        let fat: *const (dyn Fn(usize, usize) + Sync) =
            unsafe { std::mem::transmute((self.data, self.vtable)) };
        let f: &(dyn Fn(usize, usize) + Sync) = unsafe { &*fat };
        f(lo, hi);
    }
}

pub static PIN_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
pub static LOGICAL_CPUS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct PoolState {
    gen: u64,
    n: usize,
    task: Option<ErasedTask>,
    served: usize,
    shutdown: bool,
}

pub struct Pool {
    pub nt: usize,
    m: std::sync::Mutex<PoolState>,
    cv_work: std::sync::Condvar,
    cv_done: std::sync::Condvar,
}

pub static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();
thread_local! {
    pub static IN_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl Pool {
    pub fn is_shutdown(&self) -> bool {
        self.m.lock().map(|g| g.shutdown).unwrap_or(true)
    }

    /// 派发一次任务并等待全部 worker 完成（阻塞到屏障）。
    pub fn run(&self, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        let mut st = match self.m.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        if st.shutdown {
            return;
        }
        st.n = n;
        st.served = 0;
        st.task = Some(ErasedTask::from_fn(f));
        st.gen += 1;
        let my_gen = st.gen;
        self.cv_work.notify_all();
        while st.served < self.nt && st.gen == my_gen {
            st = match self.cv_done.wait(st) {
                Ok(g) => g,
                Err(e) => e.into_inner(),
            };
        }
        st.task = None;
    }
}

/// 训练开始前调用一次：创建 nt 个常驻 worker（可选绑核）。
pub fn init_pool(nt: usize, pin: bool) {
    let nt = nt.max(1);
    if nt <= 1 {
        return;
    }
    let pool = Pool {
        nt,
        m: std::sync::Mutex::new(PoolState { gen: 0, n: 0, task: None, served: 0, shutdown: false }),
        cv_work: std::sync::Condvar::new(),
        cv_done: std::sync::Condvar::new(),
    };
    if POOL.set(pool).is_err() {
        return;
    }
    let npin = LOGICAL_CPUS.load(std::sync::atomic::Ordering::Relaxed);
    for t in 0..nt {
        let _ = std::thread::Builder::new().name(format!("vmlm-pool-{}", t)).spawn(move || {
            IN_WORKER.with(|c| c.set(true));
            #[cfg(windows)]
            if pin && npin > 0 {
                unsafe {
                    let h = GetCurrentThread();
                    SetThreadAffinityMask(h, 1usize << ((t + 1) % npin));
                }
            }
            let p = match POOL.get() {
                Some(p) => p,
                None => return,
            };
            let mut last_gen = 0u64;
            loop {
                let (n, task) = {
                    let mut st = match p.m.lock() {
                        Ok(g) => g,
                        Err(e) => e.into_inner(),
                    };
                    while st.gen == last_gen && !st.shutdown {
                        st = match p.cv_work.wait(st) {
                            Ok(g) => g,
                            Err(e) => e.into_inner(),
                        };
                    }
                    if st.shutdown {
                        return;
                    }
                    last_gen = st.gen;
                    (st.n, st.task)
                };
                if let Some(task) = task {
                    let chunk = (n + p.nt - 1) / p.nt;
                    let lo = t * chunk;
                    let hi = ((t + 1) * chunk).min(n);
                    if lo < hi {
                        task.call(lo, hi);
                    }
                }
                let mut st = match p.m.lock() {
                    Ok(g) => g,
                    Err(e) => e.into_inner(),
                };
                st.served += 1;
                if st.served >= p.nt {
                    p.cv_done.notify_all();
                }
            }
        });
    }
}

pub fn shutdown_pool() {
    if let Some(p) = POOL.get() {
        let mut st = match p.m.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        st.shutdown = true;
        p.cv_work.notify_all();
    }
}
