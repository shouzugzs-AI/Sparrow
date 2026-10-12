//! `vmlm-cli` -- stdin/stdout interaction with the VMLM engine.
//!
//! Two modes:
//!
//! ```text
//!   vmlm-cli [--engine PATH] [--model PATH] [--gen N] [--once "prompt"]
//!   vmlm-cli --serve [--port N] --token T      # thin HTTP client, same CLI surface
//! ```
//!
//! # Why the CLI exists at all (and what it must NOT do)
//!
//! The point of the CLI is to be usable with **no server and no network**: read
//! stdin, run the engine, write stdout. That makes it the fallback path for any
//! environment where binding a socket is impossible, and it is also the only
//! place where the engine is driven with no HTTP layer in between -- so a bug in
//! the HTTP layer cannot hide behind CLI numbers.
//!
//! # Session commands
//!
//! ```text
//!   /system <text>   set the system prompt
//!   /clear           forget the conversation
//!   /gen [N]         run N generation steps (default from --gen)
//!   /engine          show the engine status WITHOUT running it
//!   /sess            show the session state
//!   /help            this list
//!   /quit            exit
//! ```
//!
//! Any other line is treated as a user message: it is appended, a forward pass is
//! run, and the assistant's reply is printed to stdout. When the engine is
//! UNPROVEN the CLI prints the reason to **stderr** and exits non-zero at the end
//! -- it never prints a fabricated reply. (Fabricating one would make `--once`
//! look successful in CI, which is precisely the silent degradation this project
//! bans.)
//!
//! Exit codes: 0 ok; 2 bad arguments; 5 the engine was UNPROVEN at least once.

use std::io::{BufRead, Write};
use std::sync::Mutex;

use vmlm_server::engine::{Engine, EngineConfig, EngineStatus};
use vmlm_server::session::Session;
use vmlm_server::tokenizer;

fn usage() -> &'static str {
    "\
vmlm-cli -- VMLM engine CLI (P17-P4)

USAGE:
    vmlm-cli [OPTIONS]                 interactive stdin/stdout REPL
    vmlm-cli --once \"prompt\"           one turn, then exit
    vmlm-cli --engine-only             report engine status and exit
    vmlm-cli --selftest                run the offline self-test (no server)

OPTIONS:
    --engine <PATH>   vmlm-runtime.exe (default: the bench build under target/)
    --model  <PATH>   .vmb to run (default examples/p17_http/model.vmb)
    --gen <N>         generation steps per turn (default 1, max 64)
    --system <TEXT>   initial system prompt
    --quiet           suppress the banner
    -h, --help        this text

EXIT CODES:
    0 ok   2 bad arguments   5 engine UNPROVEN at least once
"
}

struct Args {
    engine: Option<String>,
    model: Option<String>,
    gen: u32,
    once: Option<String>,
    engine_only: bool,
    selftest: bool,
    system: Option<String>,
    quiet: bool,
}

fn parse_args() -> Result<Args, (String, i32)> {
    let mut a = Args {
        engine: None,
        model: None,
        gen: 1,
        once: None,
        engine_only: false,
        selftest: false,
        system: None,
        quiet: false,
    };
    let v: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < v.len() {
        let k = v[i].as_str();
        macro_rules! next {
            ($what:expr) => {{
                if i + 1 >= v.len() {
                    return Err((format!("`{}` requires a value ({})", k, $what), 2));
                }
                i += 1;
                v[i].clone()
            }};
        }
        match k {
            "-h" | "--help" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "--engine" => a.engine = Some(next!("a path to vmlm-runtime.exe")),
            "--model" => a.model = Some(next!("a path to a .vmb file")),
            "--gen" => {
                let g = next!("a number in 1..=64");
                a.gen = g
                    .parse()
                    .map_err(|_| (format!("--gen `{}` is not an integer in 1..=64", g), 2))?;
                if a.gen == 0 || a.gen > 64 {
                    return Err((format!("--gen must be within 1..=64, got {}", a.gen), 2));
                }
            }
            "--once" => a.once = Some(next!("the prompt text")),
            "--system" => a.system = Some(next!("the system prompt")),
            "--engine-only" => a.engine_only = true,
            "--selftest" => a.selftest = true,
            "--quiet" => a.quiet = true,
            other => {
                return Err((
                    format!("unknown argument `{}`\n\n{}", other, usage()),
                    2,
                ))
            }
        }
        i += 1;
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err((m, c)) => {
            eprintln!("[vmlm-cli] error: {}\n", m);
            std::process::exit(c);
        }
    };

    if args.selftest {
        std::process::exit(selftest());
    }

    let root = vmlm_server::find_root();
    let cfg = EngineConfig::resolve(args.engine.as_deref(), args.model.as_deref(), &root);
    eprintln!("[vmlm-cli] engine: {}", cfg.exe.display());
    eprintln!("[vmlm-cli] model:  {}", cfg.model.display());
    let engine = Mutex::new(Engine::load(cfg));

    if args.engine_only {
        let e = engine.lock().unwrap();
        match e.status() {
            EngineStatus::Unavailable { reason, looked_for } => {
                println!("engine_status=UNPROVEN");
                println!("reason={}", reason);
                for l in looked_for {
                    println!("looked_for={}", l);
                }
                std::process::exit(5);
            }
            _ => {
                println!("engine_status=ready");
                println!("vmb_bytes={}", e.vmb.bytes);
                println!("vmb_slots={}", e.vmb.n_slots);
                println!("vmb_consts={}", e.vmb.n_const);
                std::process::exit(0);
            }
        }
    }

    let mut sess = Session::new("vmlm-sparse");
    if let Some(s) = args.system {
        sess.set_system(s);
    }

    if let Some(p) = args.once {
        let rc = turn(&engine, &mut sess, &p, args.gen);
        std::process::exit(rc);
    }

    if !args.quiet {
        eprintln!(
            "[vmlm-cli] interactive mode. Type /help for commands, /quit to exit."
        );
    }
    let stdin = std::io::stdin();
    let mut any_unproven = false;
    loop {
        eprint!("vmlm> ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("[vmlm-cli] read error: {}", e);
                break;
            }
        }
        let line = line.trim_end_matches(['\n', '\r']);
        if line.trim().is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('/') {
            let (cmd, arg) = match rest.split_once(' ') {
                Some((c, a)) => (c, a.trim()),
                None => (rest, ""),
            };
            match cmd {
                "quit" | "exit" | "q" => break,
                "help" | "h" => print!("{}", usage()),
                "system" => {
                    if arg.is_empty() {
                        eprintln!(
                            "[vmlm-cli] usage: /system <text>  (currently: {:?})",
                            sess.system
                        );
                    } else {
                        sess.set_system(arg);
                        eprintln!("[vmlm-cli] system prompt set ({} chars)", arg.chars().count());
                    }
                }
                "clear" => {
                    sess.clear();
                    eprintln!("[vmlm-cli] conversation cleared (system prompt kept)");
                }
                "sess" => {
                    eprintln!(
                        "[vmlm-cli] model={} messages={} chars={} system={:?}",
                        sess.model,
                        sess.wire_messages().len(),
                        sess.context_chars(),
                        sess.system
                    );
                }
                "engine" => {
                    let e = engine.lock().unwrap();
                    match e.status() {
                        EngineStatus::Unavailable { reason, .. } => {
                            eprintln!("[vmlm-cli] engine_status=UNPROVEN reason={}", reason)
                        }
                        EngineStatus::Ran { out_elems, wall_us, .. } => eprintln!(
                            "[vmlm-cli] engine_status=ran elems={} wall_us={}",
                            out_elems, wall_us
                        ),
                    }
                }
                "gen" => {
                    let n: u32 = if arg.is_empty() {
                        args.gen
                    } else {
                        match arg.parse() {
                            Ok(n) if n >= 1 && n <= 64 => n,
                            _ => {
                                eprintln!("[vmlm-cli] /gen takes a number in 1..=64");
                                continue;
                            }
                        }
                    };
                    let last = sess
                        .messages
                        .iter()
                        .rev()
                        .find(|m| m.role == "user")
                        .map(|m| m.content.clone());
                    match last {
                        None => eprintln!("[vmlm-cli] nothing to generate from; send a message first"),
                        Some(p) => {
                            // /gen re-runs the forward on the same prompt.
                            sess.messages.pop();
                            let rc = turn(&engine, &mut sess, &p, n);
                            any_unproven |= rc != 0;
                        }
                    }
                }
                other => eprintln!(
                    "[vmlm-cli] unknown command `/{other}`; /help lists the commands"
                ),
            }
            continue;
        }
        let rc = turn(&engine, &mut sess, line, args.gen);
        any_unproven |= rc != 0;
    }
    std::process::exit(if any_unproven { 5 } else { 0 });
}

/// One turn: append the user message, run the engine, print the reply.
/// Returns 0 on a real forward pass, 5 when the engine is UNPROVEN.
/// Resolve the BPE vocabulary for the CLI, same precedence order as the server
/// (env override, then the checked-in artifact, then the packaged flat layout).
fn bpe_vocab_path() -> String {
    if let Ok(p) = std::env::var("VMLM_BPE_VOCAB") {
        if !p.is_empty() {
            return p;
        }
    }
    let root = vmlm_server::find_root();
    let a = root.join("examples").join("p17_bpe").join("bpe_vocab.vmb");
    if a.exists() {
        return a.to_string_lossy().into_owned();
    }
    root.join("tokenizer.bin").to_string_lossy().into_owned()
}

fn turn(engine: &Mutex<Engine>, sess: &mut Session, prompt: &str, gen: u32) -> i32 {
    sess.push_user(prompt);
    let rendered = tokenizer::render_prompt(&sess.wire_messages());
    // P18-P1: use the REAL BPE tokenizer here too, and say which one ran.
    //
    // The old line printed "byte-level tokenizer, NOT BPE" unconditionally --
    // honest at the time, and now wrong. It is replaced by a report of what
    // actually happened, keeping the byte-level count as a labelled fallback so
    // a missing vocabulary cannot silently look like a successful encode.
    match vmlm_server::bpe_bridge::Bpe::load(&bpe_vocab_path()) {
        Ok(b) => {
            let ids = b.encode(&rendered);
            eprintln!(
                "[vmlm-cli] prompt: {} chars, {} BPE ids (vocab {} bytes, {} merges, {} pieces, fp={:08x}, roundtrip_exact={})",
                rendered.chars().count(),
                ids.len(),
                b.arena_bytes(),
                b.n_merges(),
                b.n_pieces(),
                b.vocab_fingerprint(),
                b.roundtrip_ok(&rendered)
            );
        }
        Err(e) => {
            let ntokens = tokenizer::tokenize(&rendered).len();
            eprintln!(
                "[vmlm-cli] prompt: {} chars, {} tokens (BPE UNAVAILABLE -> byte-level fallback; reason: {})",
                rendered.chars().count(),
                ntokens,
                e
            );
        }
    }

    let status = {
        let mut e = engine.lock().unwrap();
        e.run(gen)
    };

    match status {
        EngineStatus::Ran {
            out_elems,
            out_head,
            out_fnv1a,
            wall_us,
            reps,
            slot,
            ..
        } => {
            let text = tokenizer::decode_engine_output(&rendered, &out_head, out_fnv1a);
            println!("{}", text);
            eprintln!(
                "[vmlm-cli] engine ran: slot={} elems={} reps={} wall_us={} fnv1a={:08x}",
                slot, out_elems, reps, wall_us, out_fnv1a
            );
            sess.push_assistant(text);
            0
        }
        EngineStatus::Unavailable { reason, looked_for } => {
            // The user message stays in the session on purpose: nothing was
            // answered, so dropping it would lose the turn silently.
            sess.messages.pop();
            eprintln!("[vmlm-cli] engine_status=UNPROVEN -- no forward pass was run.");
            eprintln!("[vmlm-cli] reason: {}", reason);
            for l in looked_for {
                eprintln!("[vmlm-cli] looked for: {}", l);
            }
            5
        }
    }
}

/// Offline self-test: everything except the socket. Runs the same JSON and
/// tokenizer criteria the HTTP layer depends on, and prints PASS/FAIL with the
/// detail. `--selftest` is what the packaging check runs for the `cli` option.
fn selftest() -> i32 {
    let mut pass = 0;
    let mut fail = 0;
    let mut check = |name: &str, ok: bool, detail: String| {        println!("[{}] {}  {}", if ok { "PASS" } else { "FAIL" }, name, detail);
        if ok {
            pass += 1;
        } else {
            fail += 1;
        }
    };

    // JSON accept
    let good = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0.5}"#;
    match vmlm_server::json::parse(good) {
        Ok(_) => check("json accepts a valid chat request", true, "parsed".into()),
        Err(e) => check("json accepts a valid chat request", false, e.to_string()),
    }
    // JSON rejections -- each must fail, and say where.
    for (name, bad, want) in [
        ("json rejects a missing closing brace", r#"{"model":"m""#, "unterminated"),
        ("json rejects trailing garbage", r#"{"a":1} x"#, "trailing data"),
        ("json rejects a trailing comma", "[1,2,]", "trailing comma"),
        ("json rejects duplicate keys", r#"{"a":1,"a":2}"#, "duplicate"),
        ("json rejects a leading zero", "[01]", "leading zero"),
    ] {
        match vmlm_server::json::parse(bad) {
            Err(e) => check(
                name,
                e.message.contains(want) && e.offset > 0,
                format!("byte {} line {} col {}: {}", e.offset, e.line, e.column, e.message),
            ),
            Ok(v) => check(name, false, format!("ACCEPTED {:?} -- should not have", v)),
        }
    }
    // Validation rejections (the 400s the server must produce).
    for (name, bad, want) in [
        (
            "validate rejects a missing messages field",
            r#"{"model":"m"}"#,
            "`messages` is missing",
        ),
        (
            "validate rejects an empty messages array",
            r#"{"model":"m","messages":[]}"#,
            "must not be empty",
        ),
        (
            "validate rejects a numeric content",
            r#"{"model":"m","messages":[{"role":"user","content":42}]}"#,
            "must be a string",
        ),
        (
            "validate rejects an unknown role",
            r#"{"model":"m","messages":[{"role":"wizard","content":"x"}]}"#,
            "unknown role",
        ),
        (
            "validate rejects stream:true",
            r#"{"model":"m","messages":[{"role":"user","content":"x"}],"stream":true}"#,
            "not implemented",
        ),
    ] {
        match vmlm_server::chat::validate(bad) {
            Err(r) => check(name, r.message.contains(want), r.message),
            Ok(_) => check(name, false, "ACCEPTED -- should not have".into()),
        }
    }
    // Tokenizer round trip.
    let s = "hello \u{4e2d}\u{6587} 12345";
    let t = vmlm_server::tokenizer::tokenize(s);
    check(
        "tokenizer round-trips exactly",
        vmlm_server::tokenizer::detokenize(&t) == s,
        format!("{} tokens", t.len()),
    );
    // Auth.
    check(
        "auth rejects a missing header",
        vmlm_server::auth::check(None, Some("t")).is_err(),
        "None -> Err".into(),
    );
    check(
        "auth accepts the right token",
        vmlm_server::auth::check(Some("Bearer t"), Some("t")).is_ok(),
        "Bearer t -> Ok".into(),
    );

    println!("\nselftest: {} PASS, {} FAIL", pass, fail);
    if fail == 0 {
        0
    } else {
        1
    }
}
