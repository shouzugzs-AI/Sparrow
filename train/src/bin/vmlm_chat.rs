//! vmlm-chat：**单文件** EXE —— 权重与词表用 `include_bytes!` 在编译期内嵌，运行时不需要任何外部文件。
//!
//! 交互：`> ` 提示符 → 读一行 → 生成 → 打印 → 继续；`quit` / `exit` / 空行 退出。
//! 提示符里可以临时改采样参数：`seed=2 temp=0.9 k=50 p=0.9 len=80 今天`
//!
//! 复用方式（不重写生成实现）：
//!   本 bin 与训练器**共享同一份源码**（`#[path = "../b1_full_train.rs"] mod core;`），
//!   并通过 `core::set_args()` 注入 argv 后调用 `core::run_main()` ——
//!   走的就是训练器的 `--generate 1` 那条完整路径（同一套前向/采样/标定/检查点逻辑）。
//!   为避免每次生成都泄漏内存，交互循环**为每个 prompt 起一个 one-shot 子进程**（子进程退出即回收）。

#[path = "../b1_full_train.rs"]
mod core;

static MODEL: &[u8] = include_bytes!(env!("VMLM_MODEL_RESOLVED"));
static VOCAB: &[u8] = include_bytes!(env!("VMLM_VOCAB_RESOLVED"));

#[link(name = "kernel32")]
extern "system" {
    fn SetConsoleOutputCP(wCodePageID: u32) -> i32;
    fn SetConsoleCP(wCodePageID: u32) -> i32;
}

fn main() {
    unsafe {
        // Windows 控制台默认 GBK：切成 UTF-8，否则中文输出是乱码、中文输入也会被转码
        SetConsoleOutputCP(65001);
        SetConsoleCP(65001);
    }

    let tmp = std::env::temp_dir().join(format!("vmlm_chat_{}", std::process::id()));
    let data = tmp.join("data");
    let _ = std::fs::create_dir_all(&data);
    let model_file = tmp.join("model.cpk");
    let vocab_file = data.join("char_vocab_hex.txt");
    std::fs::write(&model_file, MODEL).expect("write embedded model");
    std::fs::write(&vocab_file, VOCAB).expect("write embedded vocab");


    // ---- 从 .cpk 头部读出模型几何（EXE 自己知道模型长什么样）----
    let u = |i: usize| -> u32 {
        u32::from_le_bytes([MODEL[i * 4], MODEL[i * 4 + 1], MODEL[i * 4 + 2], MODEL[i * 4 + 3]])
    };
    if MODEL.len() < 60 || u(0) != 0xB1C0_0DE1 {
        eprintln!("内嵌权重格式不对（魔数 0x{:08x}，长度 {}）", if MODEL.len() >= 4 { u(0) } else { 0 }, MODEL.len());
        std::process::exit(2);
    }
    let dim = u(2);
    let k0 = u(3);
    let k0e = u(4);
    let k0h = u(5);
    let layers = u(6);
    let n_attn = u(7);
    let k0a = u(8);
    let n_vocab = u(9);
    let io_sparse = u(10) == 1;
    let head_skip = u(11) == 1;
    let attn_sparse = u(12) == 1;
    let attn_every = if n_attn > 0 { (layers / n_attn).max(1) } else { 0 };

    // 训练器启动时会读 <data-dir>/train_ids.bin、val_ids.bin、meta.json —— 即使 --generate 也读
    //（标定/体检要用一个 val 批）。单文件分发不能带数据集，所以这里**合成**一份：
    //   确定性伪随机 id ∈ [1, V)（V 从 .cpk 头部读），长度 8192 ≥ seq*batch。
    // 注意：合成的是随机 token，说明它只用于"让驱动跑起来"，不参与生成语义；生成只用 prompt。
    // 同时传 --attn-calib 0：检查点里的标定因子**已经乘进 Wq/Wk**，再标一次反而会改模型。
    {
        let mut st: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let mut buf: Vec<u8> = Vec::with_capacity(8192 * 4);
        for _ in 0..8192 {
            let id = 1 + (next() % (u(9).max(2) as u64 - 1)) as u32;
            buf.extend_from_slice(&id.to_le_bytes());
        }
        std::fs::write(data.join("train_ids.bin"), &buf).expect("write synthetic train ids");
        std::fs::write(data.join("val_ids.bin"), &buf).expect("write synthetic val ids");
        std::fs::write(
            data.join("meta.json"),
            format!("{{\"vocab_size\":{}}}", u(9)),
        )
        .expect("write meta.json");
    }

    // 默认采样参数（按任务规格）
    let mut seed: u32 = 1;
    let mut temp = 0.7f32;
    let mut topk = 30usize;
    let mut topp = 0.85f32;
    let mut gen_len = 100usize;
    let mut n_cand = 4usize; // best-of-K：多抽 K 条，用模型自评 bits/char 挑最好（K=1 关掉）
    // [生成质量] 重复抑制（计划书 docs/B1_rep_penalty_and_dataset_plan.md §1 的推荐值）：
    //   rp：历史里出现过的 token，正 logit 除以它、负 logit 乘它（只看最近 64 个）
    //   ng：禁止形成重复的 N-gram（N>=2）
    let mut rep_pen = 1.2f32;
    let mut ng = 4usize;
    let seq = 64usize;
    let threads = 4usize;

    #[allow(clippy::too_many_arguments)]
    let base_args = |seed: u32, temp: f32, topk: usize, topp: f32, gen_len: usize, n_cand: usize, rep_pen: f32, ng: usize, prompt_file: &std::path::Path, gen_out: &std::path::Path| -> Vec<String> {
        let mut a: Vec<String> = vec![
            "vmlm-chat".into(),
            "--sparse-io".into(), if io_sparse { "1".into() } else { "0".into() },
            "--dim".into(), dim.to_string(),
            "--k0".into(), k0.to_string(),
            "--layers".into(), layers.to_string(),
            "--no-residual".into(), "1".into(),
            "--head-skip".into(), if head_skip { "1".into() } else { "0".into() },
            "--seq".into(), seq.to_string(),
            "--batch".into(), "4".into(),
            "--threads".into(), threads.to_string(),
            "--seed".into(), seed.to_string(),
            "--steps".into(), "1".into(),
            "--warmup".into(), "1".into(),
            "--eval-every".into(), "999999".into(),
            "--log-every".into(), "999999".into(),
            "--plateau-patience".into(), "0".into(),
            "--data-dir".into(), data.display().to_string(),
            "--vocab-mode".into(), "full".into(),
            "--chars-per-token".into(), "1.0".into(),
            "--out-dir".into(), tmp.display().to_string(),
            "--generate".into(), "1".into(),
            "--prompt-file".into(), prompt_file.display().to_string(),
            "--gen-len".into(), gen_len.to_string(),
            "--n-candidates".into(), n_cand.to_string(),
            "--temperature".into(), temp.to_string(),
            "--top-k".into(), topk.to_string(),
            "--top-p".into(), topp.to_string(),
            "--rep-penalty".into(), rep_pen.to_string(),
            "--no-repeat-ngram".into(), ng.to_string(),
            "--gen-out".into(), gen_out.display().to_string(),
            "--load-model".into(), model_file.display().to_string(),
        ];
        if attn_every > 0 {
            a.push("--attn-every".into());
            a.push(attn_every.to_string());
            a.push("--attn-sparse".into());
            a.push(if attn_sparse { "1".into() } else { "0".into() });
            a.push("--attn-src-shared".into());
            a.push("1".into());
            // 检查点里的标定因子已经乘进 Wq/Wk ⇒ 这里**不再重标**（重标会改模型）
            a.push("--attn-calib".into());
            a.push("0".into());
            a.push("--attn-calib-target".into());
            a.push("qk".into());
            if attn_sparse {
                a.push("--k0-attn".into());
                a.push(k0a.to_string());
            }
        }
        if io_sparse {
            a.push("--k0-embed".into());
            a.push(k0e.to_string());
            a.push("--k0-head".into());
            a.push(k0h.to_string());
        }
        a
    };

    // ---- one-shot 子进程模式：直接跑一次 --generate ----
    if std::env::var("VMLM_CHAT_ONESHOT").is_ok() {
        let argv: Vec<String> = std::env::args().collect();
        core::set_args(argv);
        core::run_main();
        return;
    }

    // ---- 交互模式 ----
    println!("=== vmlm-chat（单文件 EXE：权重与词表已内嵌）===");
    println!(
        "模型 {:.1} MB | dim={} k0={} k0e={} k0h={} layers={} attn_every={} k0a={} V={} | 词表 {} 字节",
        MODEL.len() as f64 / 1e6, dim, k0, k0e, k0h, layers, attn_every, k0a, n_vocab, VOCAB.len()
    );
    println!(
        "采样默认: temperature={} 采样top-k={} top-p={} gen_len={} seq={} threads={} seed={} | 重复抑制 rp={} ng={}",
        temp, topk, topp, gen_len, seq, threads, seed, rep_pen, ng
    );
    println!("输入 prompt 回车生成；quit / exit / 空行 退出");
    println!("可临时改参数：seed=2 temp=0.9 k=50 p=0.9 len=80 n=4 rp=1.2 ng=4 今天（n=best-of-K，默认 4；n=1 关闭；rp=1.0 关重复惩罚，ng=0 关 N-gram 禁止）");

    let stdin = std::io::stdin();
    loop {
        use std::io::Write;
        print!("> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("读 stdin 失败: {}", e);
                break;
            }
        }
        let input = line.trim_end_matches(['\n', '\r']).to_string();
        if input.is_empty() || input == "quit" || input == "exit" {
            break;
        }
        // 解析可选的 key=value 参数
        let mut rest: Vec<&str> = Vec::new();
        for tok in input.split_whitespace() {
            let hit = ["seed=", "temp=", "k=", "p=", "len=", "n=", "rp=", "ng="]
                .iter()
                .find(|p| tok.starts_with(**p));
            match hit {
                Some(p) => {
                    let v = &tok[p.len()..];
                    match *p {
                        "seed=" => seed = v.parse().unwrap_or(seed),
                        "temp=" => temp = v.parse().unwrap_or(temp),
                        "k=" => topk = v.parse().unwrap_or(topk),
                        "p=" => topp = v.parse().unwrap_or(topp),
                        "n=" => n_cand = v.parse().unwrap_or(n_cand).max(1),
                        "rp=" => rep_pen = v.parse().unwrap_or(rep_pen),
                        "ng=" => ng = v.parse().unwrap_or(ng),
                        _ => gen_len = v.parse().unwrap_or(gen_len),
                    }
                }
                None => rest.push(tok),
            }
        }
        let prompt = if rest.is_empty() { "今天".to_string() } else { rest.join(" ") };
        let pf = tmp.join("prompt.txt");
        let go = tmp.join("gen_out.txt");
        std::fs::write(&pf, &prompt).expect("write prompt");

        let args = base_args(seed, temp, topk, topp, gen_len, n_cand, rep_pen, ng, &pf, &go);
        let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("vmlm-chat.exe"));
        let t0 = std::time::Instant::now();
        let out = std::process::Command::new(exe)
            .args(&args)
            .env("VMLM_CHAT_ONESHOT", "1")
            .output();
        match out {
            Ok(o) => {
                if !o.status.success() {
                    eprintln!("生成失败（退出码 {:?}）", o.status.code());
                    eprintln!("{}", String::from_utf8_lossy(&o.stderr));
                    continue;
                }
                // 以 gen_out 文件为准（UTF-8，不受控制台编码影响）
                let txt = std::fs::read_to_string(&go).unwrap_or_default();
                let gen_only = txt
                    .split("[gen only]")
                    .nth(1)
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();
                let secs = t0.elapsed().as_secs_f32();
                println!("{}", gen_only);
                println!(
                    "  （{} 字符 / {:.1}s = {:.1} 字符/秒；seed={} temp={} k={} p={}）",
                    gen_only.chars().count(),
                    secs,
                    gen_only.chars().count() as f32 / secs.max(1e-6),
                    seed, temp, topk, topp
                );
                let _ = n_cand;
            }
            Err(e) => eprintln!("无法启动子进程: {}", e),
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    println!("bye");
}
