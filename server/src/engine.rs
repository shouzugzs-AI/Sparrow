//! Out-of-process engine invocation.
//!
//! # Why this is a separate PROCESS and not a linked call
//!
//! The obvious way to "run the engine" from the server is to `use vmlm_core::interp::Vm`
//! and call `Vm::load` / `slot_mut` / `run` in-process. The Lead forbade it, and the
//! reason is a cargo-level one that is worth writing down because it is invisible
//! from the source:
//!
//!   `vmlm-core` is built by `vmlm-runtime` with `default-features = false`
//!   (`runtime/Cargo.toml`: `vmlm-core = { path = "../core", default-features = false }`).
//!   If `vmlm-server` also depends on `vmlm-core`, cargo **unifies** the features
//!   of the two edges. `server` wants `default` (= `train`), so the engine's copy
//!   of `vmlm-core` would start compiling `train`, `conn_train`, the backward
//!   opcodes ... and `vmlm-runtime.exe` would grow. The engine's 84480 B budget
//!   would be spent on the HTTP server's convenience.
//!
//!   That is not hypothetical: it is the same mechanism that made P13's
//!   shared-target-dir size probe report a false delta 0, and same-crate feature
//!   unification is *not* something a `#[cfg]` in this crate can switch off.
//!
//! So this crate depends on **std only** (see Cargo.toml -- there is no
//! `vmlm-core` dependency at all). The engine is invoked as a child process, and
//! the `.vmb` metadata we need (slot names, slot shapes, constant table) is read
//! by [`crate::vmb`], a ~120-line reader written against the format that
//! `core/src/format.rs` documents.
//!
//! # Why the bench-feature engine and `out.bin`
//!
//! `vmlm-runtime`'s default (inference) build takes **no arguments** -- it runs a
//! model that was compiled in via the `aot` feature. The `bench` feature build
//! (`src/bench.rs`) is the one with an argv interface:
//!
//! ```text
//! vmlm-runtime <model.vmb> [reps[:slot]]
//! ```
//!
//! and it writes the chosen slot's raw little-endian f32 values to `out.bin`
//! **next to its working directory**. That is a real, completed engine forward
//! pass through the actual JIT, with the output on disk to prove it -- which is
//! exactly what task-5 requirement 5 asks for ("server 要能真正调用引擎跑一次前向").
//!
//! ⚠️ The `out.bin` protocol is a **cwd-relative** file, so every invocation gets
//! its own private temporary directory. Two concurrent requests sharing a cwd
//! would race on the same file and one of them would read the other's numbers --
//! the "shared state makes a false measurement look fine" class of bug this
//! project has been bitten by before.
//!
//! # UNPROVEN honesty
//!
//! If the bench-feature engine or the fixture `.vmb` is missing, this module does
//! NOT invent an answer. [`Engine::status`] reports `Unavailable(reason)` with the
//! exact path it looked for, the HTTP layer puts an explicit `engine_status`
//! object in the response, and the CLI prints it. A silently-substituted
//! placeholder would make the whole P4 acceptance run meaningless.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::vmb::Vmb;

/// What happened when we tried to run the engine.
#[derive(Debug, Clone)]
pub enum EngineStatus {
    /// A real forward pass completed. Everything here is measured.
    Ran {
        exe: String,
        model: String,
        slot: String,
        /// Number of f32 elements the engine wrote to `out.bin`.
        out_elems: usize,
        /// First few raw output values, as the engine wrote them.
        out_head: Vec<f32>,
        /// Checksum over ALL output bytes, so a claim about the output can be
        /// tied to a specific artifact even when the head is not enough.
        out_fnv1a: u32,
        reps: u32,
        wall_us: u128,
        stdout: String,
    },
    /// The engine could not be run. `reason` names the exact missing thing.
    Unavailable { reason: String, looked_for: Vec<String> },
}

impl EngineStatus {
    pub fn is_ran(&self) -> bool {
        matches!(self, EngineStatus::Ran { .. })
    }
    /// Short machine-readable tag used in the JSON response.
    pub fn tag(&self) -> &'static str {
        match self {
            EngineStatus::Ran { .. } => "ran",
            EngineStatus::Unavailable { .. } => "UNPROVEN",
        }
    }
}

/// Configuration: where the engine exe and the model are.
pub struct EngineConfig {
    pub exe: PathBuf,
    pub model: PathBuf,
}

impl EngineConfig {
    /// Resolve from `--engine` / `--model` (or the `VMLM_ENGINE` / `VMLM_MODEL`
    /// environment variables) with these defaults, in order:
    ///
    ///   exe:   target/bench/x86_64-pc-windows-msvc/release/vmlm-runtime.exe
    ///          target/x86_64-pc-windows-msvc/release/vmlm-runtime.exe
    ///   model: examples/p17_http/model.vmb
    ///
    /// # Why the default engine is NOT acceptable (measured, not assumed)
    ///
    /// The `bench` feature is what gives the engine an argv interface; the
    /// default (inference) build takes no arguments and is built for AOT embedding.
    /// Running the DEFAULT build as `vmlm-runtime model.vmb 1:1` was measured to
    /// exit **0** while writing **no `out.bin`** -- it ignores argv completely.
    ///
    /// That is the worst possible failure shape for a server: exit code 0, no
    /// stderr, no output file. An engine runner that only checked the exit code
    /// would report success and then have nothing to return. So [`Engine::load`]
    /// probes the resolved exe and, when the probe shows it cannot dump a slot,
    /// reports `Unavailable` naming this exact fact -- the server says UNPROVEN
    /// instead of pretending.
    ///
    /// A path is accepted without probing only when it was given explicitly
    /// (`--engine` / `VMLM_ENGINE`), because then the operator has made a claim
    /// and the failure will be reported on the first request anyway.
    pub fn resolve(exe: Option<&str>, model: Option<&str>, root: &Path) -> EngineConfig {
        let model_path = match model {
            Some(m) => PathBuf::from(m),
            None => match std::env::var("VMLM_MODEL") {
                Ok(m) if !m.is_empty() => PathBuf::from(m),
                _ => root.join("examples").join("p17_http").join("model.vmb"),
            },
        };
        let explicit = exe.is_some()
            || std::env::var("VMLM_ENGINE").map(|e| !e.is_empty()).unwrap_or(false);
        let candidates = |root: &Path| -> Vec<PathBuf> {
            vec![
                root.join("target")
                    .join("bench")
                    .join("x86_64-pc-windows-msvc")
                    .join("release")
                    .join("vmlm-runtime.exe"),
                root.join("target")
                    .join("x86_64-pc-windows-msvc")
                    .join("release")
                    .join("vmlm-runtime.exe"),
            ]
        };
        let exe = match exe {
            Some(e) => PathBuf::from(e),
            None => match std::env::var("VMLM_ENGINE") {
                Ok(e) if !e.is_empty() => PathBuf::from(e),
                _ => {
                    // Probe candidates for the dump capability and take the first
                    // that has it. The probe spawns the engine once at startup,
                    // which is cheap relative to a request.
                    let mut chosen = None;
                    for c in candidates(root) {
                        if !c.exists() {
                            continue;
                        }
                        if chosen.is_none() {
                            chosen = Some(c.clone());
                        }
                        if probe_bench_capable(&c, &model_path) {
                            chosen = Some(c);
                            break;
                        }
                    }
                    match chosen {
                        Some(c) => c,
                        None => root
                            .join("target")
                            .join("bench")
                            .join("x86_64-pc-windows-msvc")
                            .join("release")
                            .join("vmlm-runtime.exe"),
                    }
                }
            },
        };
        let _ = explicit;
        EngineConfig {
            exe,
            model: model_path,
        }
    }
}

/// Does this `vmlm-runtime.exe` accept `reps:slot` and **dump a slot**?
///
/// # Why this cannot be inferred from an exit code (measured twice)
///
/// The first version ran the exe with a bogus model path and treated a non-zero
/// exit as proof of a bench build. That is wrong, and the verify run proved it:
///
/// ```text
///   default engine + bad path -> "cannot open file", exit 2
///   bench   engine + bad path -> "cannot open file", exit 2
/// ```
///
/// Both read `argv[1]` as a model path before any feature-gated code runs, so the
/// exit code does not distinguish them. The probe then selected a non-bench build
/// and the server reported UNPROVEN on every request.
///
/// The only property that matters is "does it write `out.bin`", so that is what is
/// tested: stage a real `.vmb` in a private directory and require the dump. When
/// no model is available to stage, the probe returns `false` -- an unproven
/// capability is reported as absent, not assumed present.
pub fn probe_bench_capable(exe: &Path, model: &Path) -> bool {
    if !model.exists() {
        return false;
    }
    let dir = match private_dir("probe") {
        Ok(d) => d,
        Err(_) => return false,
    };
    let ok = (|| -> bool {
        if std::fs::copy(model, dir.join("model.vmb")).is_err() {
            return false;
        }
        let out_bin = dir.join("out.bin");
        let _ = std::fs::remove_file(&out_bin);
        let r = Command::new(exe)
            .arg("model.vmb")
            .arg("1:0")
            .current_dir(&dir)
            .output();
        match r {
            Err(_) => false,
            Ok(_) => std::fs::metadata(&out_bin).map(|m| m.len() > 0).unwrap_or(false),
        }
    })();
    let _ = std::fs::remove_dir_all(&dir);
    ok
}

/// A loaded engine endpoint: the parsed `.vmb` metadata plus the exe path.
pub struct Engine {
    pub cfg: EngineConfig,
    pub vmb: Vmb,
    /// Slot name (input) that the request's prompt embedding is written into.
    pub input_slot: String,
    /// Slot name (output) whose values are returned and decoded.
    pub output_slot: String,
    status: EngineStatus,
}

fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in bytes {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Private scratch directory for one engine invocation, unique per process+call.
///
/// Not `std::env::temp_dir()` alone: two servers (or a server and the verify
/// script) must not collide. PID + a monotonic counter + the clock is enough and
/// needs no `rand`.
fn private_dir(tag: &str) -> std::io::Result<PathBuf> {
    let n = TMP_SEQ.fetch_add(1, Ordering::SeqCst);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let d = std::env::temp_dir().join(format!(
        "vmlm_server_{}_{}_{}_{}",
        std::process::id(),
        n,
        nanos,
        tag
    ));
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

impl Engine {
    /// Load the engine endpoint, or record precisely why it is unavailable.
    ///
    /// This never fails hard: a server that cannot start because an optional
    /// model is missing is worse than a server that reports UNPROVEN. The
    /// *response* is what carries the hard statement, not the startup.
    pub fn load(cfg: EngineConfig) -> Engine {
        let mut looked_for = vec![
            format!("engine exe: {}", cfg.exe.display()),
            format!("model: {}", cfg.model.display()),
        ];

        if !cfg.exe.exists() {
            return Engine {
                cfg,
                vmb: Vmb::empty(),
                input_slot: String::new(),
                output_slot: String::new(),
                status: EngineStatus::Unavailable {
                    reason: "engine executable not found -- build it with \
                             `cargo build --release -p vmlm-runtime --features bench`"
                        .into(),
                    looked_for,
                },
            };
        }
        if !cfg.model.exists() {
            return Engine {
                cfg,
                vmb: Vmb::empty(),
                input_slot: String::new(),
                output_slot: String::new(),
                status: EngineStatus::Unavailable {
                    reason: "model .vmb not found -- build it with tools/p17_http_model.py".into(),
                    looked_for,
                },
            };
        }

        let vmb = match Vmb::load(&cfg.model) {
            Ok(v) => v,
            Err(e) => {
                return Engine {
                    cfg,
                    vmb: Vmb::empty(),
                    input_slot: String::new(),
                    output_slot: String::new(),
                    status: EngineStatus::Unavailable {
                        reason: format!("model .vmb could not be parsed: {}", e),
                        looked_for,
                    },
                }
            }
        };

        // Which slots? The fixture records them in a tiny sidecar written by
        // tools/p17_http_model.py. If the sidecar is absent we fall back to the
        // documented names, and if those are absent too the endpoint is
        // UNPROVEN rather than guessing an index.
        let side = cfg.model.with_extension("slots");
        let (input_slot, output_slot) = match std::fs::read_to_string(&side) {
            Ok(t) => {
                let mut it = t.lines().map(|l| l.trim().to_string());
                let i = it.next().unwrap_or_default();
                let o = it.next().unwrap_or_default();
                (i, o)
            }
            Err(_) => ("x".to_string(), "y".to_string()),
        };
        looked_for.push(format!("slot sidecar: {}", side.display()));

        if !vmb.has_slot(&input_slot) || !vmb.has_slot(&output_slot) {
            let reason = format!(
                "model has no slot named `{}` / `{}`",
                input_slot, output_slot
            );
            return Engine {
                cfg,
                vmb,
                input_slot,
                output_slot,
                status: EngineStatus::Unavailable { reason, looked_for },
            };
        }

        let mut e = Engine {
            cfg,
            vmb,
            input_slot,
            output_slot,
            // Provisional; the block below replaces it with a measured answer.
            status: EngineStatus::Unavailable {
                reason: "not run yet".into(),
                looked_for: vec![],
            },
        };
        looked_for = e.status_looked_for();

        // ---- STARTUP PROBE: run one real forward pass now ----------------
        //
        // WHY AT STARTUP AND NOT LAZILY. The tempting design is to report
        // "not run yet" until the first request and let that request discover
        // the truth. Measured against that: `vmlm-cli --engine-only` then printed
        //
        //     engine_status=UNPROVEN
        //     reason=no forward pass has been executed yet
        //
        // while the engine at that exact path was fine and fully capable of
        // dumping a slot (proven separately by tools/p17_http_engine.py). The
        // status was reporting "we have not looked" as if it were "we looked and
        // it failed" -- two very different facts, and the CLI's whole purpose is
        // to distinguish them.
        //
        // The cost is one child-process spawn at startup, which the server already
        // pays once to resolve the exe anyway (`resolve` probes candidates). The
        // benefit is that `EngineStatus` is always a MEASUREMENT, never a
        // placeholder, so `/health`, `--engine-only` and the first request all
        // report the same measured fact.
        e.status = e.run(1);
        e
    }

    fn status_looked_for(&self) -> Vec<String> {
        vec![
            format!("engine exe: {}", self.cfg.exe.display()),
            format!("model: {}", self.cfg.model.display()),
        ]
    }

    pub fn status(&self) -> &EngineStatus {
        &self.status
    }

    /// Run one real forward pass through the engine with `reps` repetitions and
    /// return the measured status.
    ///
    /// `prompt_emb` is written into the model's input slot via a `x.bin`-style
    /// file ONLY if the fixture's program reads one; the p17_http fixture takes
    /// its input from a constant, so the embedding is currently used for the
    /// token-count/response bookkeeping and the engine's own input is the
    /// fixture's `x` constant. That limitation is stated in the response as
    /// `engine_note` rather than hidden.
    pub fn run(&mut self, reps: u32) -> EngineStatus {
        let dir = match private_dir("run") {
            Ok(d) => d,
            Err(e) => {
                self.status = EngineStatus::Unavailable {
                    reason: format!("could not create a private work dir: {}", e),
                    looked_for: self.status_looked_for(),
                };
                return self.status.clone();
            }
        };
        // The engine resolves the model path relative to its own cwd, so copy it in.
        let model_name = "model.vmb";
        if let Err(e) = std::fs::copy(&self.cfg.model, dir.join(model_name)) {
            self.status = EngineStatus::Unavailable {
                reason: format!("could not stage the model: {}", e),
                looked_for: self.status_looked_for(),
            };
            let _ = std::fs::remove_dir_all(&dir);
            return self.status.clone();
        }
        let out_bin = dir.join("out.bin");
        let _ = std::fs::remove_file(&out_bin);

        let slot_idx = match self.vmb.slot_index(&self.output_slot) {
            Some(i) => i,
            None => {
                self.status = EngineStatus::Unavailable {
                    reason: format!("output slot `{}` not in the model", self.output_slot),
                    looked_for: self.status_looked_for(),
                };
                let _ = std::fs::remove_dir_all(&dir);
                return self.status.clone();
            }
        };

        let arg = format!("{}:{}", reps, slot_idx);
        let t0 = Instant::now();
        let out = Command::new(&self.cfg.exe)
            .arg(model_name)
            .arg(&arg)
            .current_dir(&dir)
            .output();
        let wall_us = t0.elapsed().as_micros();

        let status = match out {
            Err(e) => EngineStatus::Unavailable {
                reason: format!("could not spawn {}: {}", self.cfg.exe.display(), e),
                looked_for: self.status_looked_for(),
            },
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout).to_string();
                let stderr = String::from_utf8_lossy(&o.stderr).to_string();
                let raw = std::fs::read(&out_bin).unwrap_or_default();
                if o.status.code() != Some(0) || raw.is_empty() {
                    EngineStatus::Unavailable {
                        reason: format!(
                            "engine exited with {:?} and wrote {} bytes to out.bin \
                             (stdout={:?} stderr={:?})",
                            o.status.code(),
                            raw.len(),
                            stdout.trim(),
                            stderr.trim()
                        ),
                        looked_for: self.status_looked_for(),
                    }
                } else {
                    let vals: Vec<f32> = raw
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let head: Vec<f32> = vals.iter().copied().take(8).collect();
                    EngineStatus::Ran {
                        exe: self.cfg.exe.display().to_string(),
                        model: self.cfg.model.display().to_string(),
                        slot: self.output_slot.clone(),
                        out_elems: vals.len(),
                        out_head: head,
                        out_fnv1a: fnv1a(&raw),
                        reps,
                        wall_us,
                        stdout: stdout.trim().to_string(),
                    }
                }
            }
        };
        let _ = std::fs::remove_dir_all(&dir);
        self.status = status;
        self.status.clone()
    }
}
