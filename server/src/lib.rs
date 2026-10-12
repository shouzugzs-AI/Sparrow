//! VMLM HTTP service library (P17-P4).
//!
//! This crate is the **standalone** HTTP/CLI face of VMLM. It is deliberately NOT
//! part of the engine:
//!
//! ```text
//!   runtime/  -> vmlm-runtime.exe   no_std, /NODEFAULTLIB, budget 84480 B
//!   server/   -> vmlm-server.exe    std, CRT, no budget, NOT linked into the engine
//!            -> vmlm-cli.exe
//! ```
//!
//! `server/Cargo.toml` depends on **std only** for the HTTP/JSON/auth layer --
//! in particular the OUT-OF-PROCESS engine driver (`src/engine.rs`) does not use
//! `vmlm-core`. It DOES link `vmlm-core` for two things that must not be
//! reimplemented:
//!
//!   * the `.vmb` reader used by the in-process catalog path, and
//!   * the P18-P1 BPE tokenizer ([`bpe_bridge`]).
//!
//! Why that is safe for the engine's size budget: `vmlm-runtime` depends on
//! `vmlm-core` with `default-features = false` and its own explicit feature
//! list, so features enabled HERE cannot reach the engine. That is a structural
//! argument, and `tools/build_all.ps1` re-measures the engine size so it does
//! not have to be taken on faith.
//!
//! Module map:
//!
//!   * [`json`]      -- hand-written strict JSON parser + writer (no serde)
//!   * [`http`]      -- raw Winsock HTTP/1.1 server and a tiny test client
//!   * [`auth`]      -- Bearer authentication
//!   * [`vmb`]       -- minimal read-only `.vmb` metadata reader
//!   * [`engine`]    -- out-of-process engine invocation, with UNPROVEN reporting
//!   * [`tokenizer`] -- a byte-level tokenizer (used for `usage` counters)
//!   * [`bpe_bridge`]-- P18-P1: the REAL BPE tokenizer (core/src/bpe.rs),
//!                      closing the "prompt -> BPE -> engine" gap of P17 §4.2
//!   * [`chat`]      -- `/v1/chat/completions`: validation + response building
//!   * [`session`]   -- conversation state for the CLI

pub mod auth;
pub mod bpe_bridge;
pub mod chat;
pub mod engine;
pub mod http;
pub mod json;
pub mod session;
pub mod tokenizer;
pub mod vmb;

/// Root of the VMLM checkout, used to resolve default paths for the engine exe
/// and the model. Walk up from the current dir looking for `Cargo.toml` +
/// `runtime/`, so both `cd server && cargo run` and `cargo run` from the root work.
pub fn find_root() -> std::path::PathBuf {
    if let Ok(r) = std::env::var("VMLM_ROOT") {
        if !r.is_empty() {
            return std::path::PathBuf::from(r);
        }
    }
    let mut d = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    for _ in 0..6 {
        if d.join("runtime").join("Cargo.toml").exists() && d.join("core").join("Cargo.toml").exists()
        {
            return d;
        }
        match d.parent() {
            Some(p) => d = p.to_path_buf(),
            None => break,
        }
    }
    std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
}
