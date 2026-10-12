//! `/v1/chat/completions`: request validation, then a real engine forward.
//!
//! # The rule this file exists to enforce
//!
//! Task-5 requirement 3: missing fields, wrong types and bad JSON must all
//! produce **HTTP 400 with an explanation**, and must NOT be silently treated as
//! an empty request. That is stated as a project-wide rule ("非法输入必须硬失败
//! 并说明哪里错") because the failure mode it prevents is invisible: a server
//! that treats `{"model":"m"}` (no `messages`) as "zero messages" returns a
//! perfectly well-formed 200 with a generic reply, the client believes its
//! request was understood, and the bug only shows up as "the model ignores my
//! prompt" weeks later.
//!
//! So **every** field is checked and each rejection names: the field, the byte
//! offset in the request body (for JSON syntax errors), and what was expected
//! versus what was found.
//!
//! # Response shape
//!
//! OpenAI-compatible enough for `curl` and for a client that reads
//! `choices[0].message.content`:
//!
//! ```json
//! {"id":"chatcmpl-...","object":"chat.completion","created":...,
//!  "model":"...","choices":[{"index":0,"message":{"role":"assistant","content":"..."},
//!  "finish_reason":"stop"}],
//!  "usage":{"prompt_tokens":N,"completion_tokens":M,"total_tokens":N+M},
//!  "vmlm":{"engine_status":"ran", ...measured numbers..., "wall_us":...}}
//! ```
//!
//! The extra `vmlm` object is where the *measured* facts go (token counts, engine
//! wall time, the engine's raw output checksum, and the engine status tag). It is
//! additive: a client that ignores unknown keys still works.

use crate::engine::{Engine, EngineStatus};
use crate::json::{Json, JsonError, Obj};
use crate::tokenizer;

/// Largest accepted request body (also enforced by the HTTP reader).
pub const MAX_MESSAGES: usize = 256;
pub const MAX_CONTENT_CHARS: usize = 32 * 1024;

/// A validated chat request.
#[derive(Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub stream: bool,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Why a request was rejected, already worded for the client.
#[derive(Debug)]
pub struct Reject {
    pub message: String,
    pub offset: Option<usize>,
    pub field: Option<String>,
}

impl Reject {
    fn new(message: impl Into<String>) -> Reject {
        Reject {
            message: message.into(),
            offset: None,
            field: None,
        }
    }
    fn field(field: &str, message: impl Into<String>, offset: usize) -> Reject {
        Reject {
            message: message.into(),
            offset: Some(offset),
            field: Some(field.to_string()),
        }
    }
}

impl From<JsonError> for Reject {
    fn from(e: JsonError) -> Reject {
        Reject {
            message: e.to_string(),
            offset: Some(e.offset),
            field: None,
        }
    }
}

/// Parse + validate. The `Result` is the whole point: `Err` becomes a 400 whose
/// body says exactly which field failed and why.
pub fn validate(body: &str) -> Result<ChatRequest, Reject> {
    if body.trim().is_empty() {
        return Err(Reject::new(
            "the request body is empty; a chat completion requires a JSON object \
             with at least `model` and a non-empty `messages` array",
        ));
    }
    let v = crate::json::parse(body)?;
    let obj = match &v {
        Json::Obj(o) => o,
        other => {
            return Err(Reject::new(format!(
                "the request body must be a JSON object, got {}",
                other.type_name()
            )))
        }
    };

    // ---- model: required, must be a non-empty string -----------------------
    let model = match obj.get("model") {
        None => {
            return Err(Reject::field(
                "model",
                "required field `model` is missing (expected a non-empty string)",
                0,
            ))
        }
        Some(Json::Str(s)) => {
            if s.trim().is_empty() {
                return Err(Reject::field("model", "`model` must not be empty", 0));
            }
            s.clone()
        }
        Some(other) => {
            return Err(Reject::field(
                "model",
                format!("`model` must be a string, got {}", other.type_name()),
                0,
            ))
        }
    };

    // ---- messages: required, non-empty array ------------------------------
    let msgs = match obj.get("messages") {
        None => {
            return Err(Reject::field(
                "messages",
                "required field `messages` is missing; a chat request without \
                 messages has no prompt, and treating it as an empty conversation \
                 would silently answer a question the client never asked",
                0,
            ))
        }
        Some(Json::Arr(a)) => a,
        Some(other) => {
            return Err(Reject::field(
                "messages",
                format!("`messages` must be an array, got {}", other.type_name()),
                0,
            ))
        }
    };
    if msgs.is_empty() {
        return Err(Reject::field(
            "messages",
            "`messages` must not be empty (at least one message is required)",
            0,
        ));
    }
    if msgs.len() > MAX_MESSAGES {
        return Err(Reject::field(
            "messages",
            format!(
                "`messages` has {} entries, the cap is {}",
                msgs.len(),
                MAX_MESSAGES
            ),
            0,
        ));
    }

    let mut messages = Vec::with_capacity(msgs.len());
    for (i, m) in msgs.iter().enumerate() {
        let mo = match m {
            Json::Obj(o) => o,
            other => {
                return Err(Reject::field(
                    &format!("messages[{}]", i),
                    format!(
                        "each entry of `messages` must be an object with `role` and \
                         `content`, got {}",
                        other.type_name()
                    ),
                    0,
                ))
            }
        };
        let role = match mo.get("role") {
            None => {
                return Err(Reject::field(
                    &format!("messages[{}].role", i),
                    "required field `role` is missing (expected one of \
                     \"system\", \"user\", \"assistant\")",
                    0,
                ))
            }
            Some(Json::Str(s)) => {
                let s = s.trim().to_ascii_lowercase();
                if !matches!(s.as_str(), "system" | "user" | "assistant") {
                    return Err(Reject::field(
                        &format!("messages[{}].role", i),
                        format!(
                            "unknown role \"{}\"; expected \"system\", \"user\" or \
                             \"assistant\"",
                            s
                        ),
                        0,
                    ));
                }
                s
            }
            Some(other) => {
                return Err(Reject::field(
                    &format!("messages[{}].role", i),
                    format!("`role` must be a string, got {}", other.type_name()),
                    0,
                ))
            }
        };
        let content = match mo.get("content") {
            None => {
                return Err(Reject::field(
                    &format!("messages[{}].content", i),
                    "required field `content` is missing (expected a string)",
                    0,
                ))
            }
            Some(Json::Str(s)) => {
                if s.chars().count() > MAX_CONTENT_CHARS {
                    return Err(Reject::field(
                        &format!("messages[{}].content", i),
                        format!(
                            "`content` has {} characters, the cap is {}",
                            s.chars().count(),
                            MAX_CONTENT_CHARS
                        ),
                        0,
                    ));
                }
                s.clone()
            }
            Some(other) => {
                return Err(Reject::field(
                    &format!("messages[{}].content", i),
                    format!(
                        "`content` must be a string, got {} (content arrays are not \
                         supported by this server)",
                        other.type_name()
                    ),
                    0,
                ))
            }
        };
        messages.push(Message { role, content });
    }

    // ---- optional fields, still type-checked ------------------------------
    let mut temperature = None;
    if let Some(t) = obj.get("temperature") {
        match t {
            Json::Num(n) => {
                let f: f64 = n.parse().map_err(|_| {
                    Reject::field(
                        "temperature",
                        format!("`temperature` value `{}` is not a finite number", n),
                        0,
                    )
                })?;
                if !(0.0..=2.0).contains(&f) {
                    return Err(Reject::field(
                        "temperature",
                        format!("`temperature` must be within [0, 2], got {}", f),
                        0,
                    ));
                }
                temperature = Some(f);
            }
            other => {
                return Err(Reject::field(
                    "temperature",
                    format!("`temperature` must be a number, got {}", other.type_name()),
                    0,
                ))
            }
        }
    }

    let mut max_tokens = None;
    if let Some(t) = obj.get("max_tokens") {
        match t {
            Json::Num(n) => {
                let u: u64 = n.parse().map_err(|_| {
                    Reject::field(
                        "max_tokens",
                        format!("`max_tokens` value `{}` is not a non-negative integer", n),
                        0,
                    )
                })?;
                if u == 0 || u > 4096 {
                    return Err(Reject::field(
                        "max_tokens",
                        format!("`max_tokens` must be within [1, 4096], got {}", u),
                        0,
                    ));
                }
                max_tokens = Some(u);
            }
            other => {
                return Err(Reject::field(
                    "max_tokens",
                    format!("`max_tokens` must be a number, got {}", other.type_name()),
                    0,
                ))
            }
        }
    }

    let mut stream = false;
    if let Some(s) = obj.get("stream") {
        match s {
            Json::Bool(b) => stream = *b,
            other => {
                return Err(Reject::field(
                    "stream",
                    format!("`stream` must be a boolean, got {}", other.type_name()),
                    0,
                ))
            }
        }
    }
    if stream {
        // Better an explicit 400 than a 200 with a non-streaming body that a
        // streaming client will hang on.
        return Err(Reject::field(
            "stream",
            "`stream: true` is not implemented by this server; send `\"stream\": false` \
             or omit the field",
            0,
        ));
    }

    Ok(ChatRequest {
        model,
        messages,
        temperature,
        max_tokens,
        stream,
    })
}

/// Render a rejection as the 400 body: the field, the offset, and what to fix.
pub fn reject_body(r: &Reject) -> String {
    let mut e = Obj::new();
    e.insert("type", Json::Str("invalid_request_error".into()));
    e.insert("message", Json::Str(r.message.clone()));
    if let Some(f) = &r.field {
        e.insert("param", Json::Str(f.clone()));
    }
    if let Some(o) = &r.offset {
        e.insert("offset", Json::Num(o.to_string()));
    }
    let mut root = Obj::new();
    root.insert("error", Json::Obj(e));
    crate::json::to_string(&Json::Obj(root))
}

/// Resolve and load the BPE vocabulary, caching the result for the process.
///
/// Path resolution order (explicit, because a silently-wrong default is exactly
/// the class of bug this project converts to hard failures):
///   1. `$VMLM_BPE_VOCAB` if set;
///   2. `<root>/examples/p17_bpe/bpe_vocab.vmb` (the checked-in artifact);
///   3. `<root>/tokenizer.bin` (the packaged layout: server + model + tokenizer
///      in one flat directory, per P18-P3).
///
/// Caching: the vocabulary is mmap'd, so re-loading per request would re-map
/// 5 KB each time. A `OnceLock` keeps the first result -- **including the
/// error**, so a missing vocabulary is reported the same way on every request
/// rather than flapping between success and failure.
fn load_bpe_cached() -> Result<&'static crate::bpe_bridge::Bpe, String> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Result<crate::bpe_bridge::Bpe, String>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let root = crate::find_root();
            let mut cands: Vec<std::path::PathBuf> = Vec::new();
            if let Ok(p) = std::env::var("VMLM_BPE_VOCAB") {
                if !p.is_empty() {
                    cands.push(std::path::PathBuf::from(p));
                }
            }
            cands.push(root.join("examples").join("p17_bpe").join("bpe_vocab.vmb"));
            cands.push(root.join("tokenizer.bin"));

            let mut tried: Vec<String> = Vec::new();
            for c in &cands {
                let s = c.to_string_lossy().into_owned();
                match crate::bpe_bridge::Bpe::load(&s) {
                    Ok(b) => return Ok(b),
                    Err(e) => tried.push(e),
                }
            }
            Err(format!(
                "no BPE vocabulary could be loaded; tried: {}",
                tried.join(" | ")
            ))
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// The successful path: validate already done, now run the engine and build the
/// response. `engine` is `&mut` because the status is cached on it.
pub fn complete(
    req: &ChatRequest,
    engine: &mut Engine,
    server_base_url: &str,
    request_id: &str,
) -> (u16, String) {
    let prompt = tokenizer::render_prompt(&req.messages);
    let prompt_tokens = tokenizer::tokenize(&prompt).len() as u64;

    // ===================== P18-P1：真 BPE 接管 prompt 编码 =====================
    //
    // P17 的端到端文档把这条边界写得很清楚（§4.2）：
    //   「prompt → BPE → input_ids → 引擎」这一环**没有接**；
    //    服务器的 tokenizer 是字节级计数器，BPE 是另一个 feature 且 server 没链接它。
    // 这里把它接上：用 `core/src/bpe.rs` 的真 BPE 编码 prompt，并**回报 ids**。
    //
    // 为什么要回报 ids 而不是只回报 token 计数：计数可以被任何一个能数数的
    // 实现产生（原来的字节级计数器就能）。ids 是 BPE 的实际输出，
    // 把它放进响应，才使「这台服务器真的在跑这个词表」**可证伪** ——
    // 换掉 tokenizer.bin、或者 BPE 没接上，ids 与 fingerprint 都会变。
    //
    // 失败口径：词表缺失/过小 → `Bpe::load` 返回 Err，我们把 BPE 状态标成
    // UNPROVEN 并**保留原始 prompt 文本**，绝不静默降级成「编码了 0 个 token」。
    // 这一点与引擎不可用时返回 503 是同一个原则。
    let bpe_info = load_bpe_cached();
    let (bpe_state, bpe_json) = match &bpe_info {
        Ok(b) => {
            let ids = b.encode(&prompt);
            let rt = b.roundtrip_ok(&prompt);
            let mut j = Obj::new();
            j.insert("status", Json::Str("ran".into()));
            j.insert("vocab", Json::Str(b.path.clone()));
            j.insert("n_merges", Json::Num(b.n_merges().to_string()));
            j.insert("n_pieces", Json::Num(b.n_pieces().to_string()));
            j.insert("arena_bytes", Json::Num(b.arena_bytes().to_string()));
            j.insert(
                "vocab_fingerprint",
                Json::Str(format!("{:08x}", b.vocab_fingerprint())),
            );
            j.insert("n_ids", Json::Num(ids.len().to_string()));
            // 只回报前若干 id，避免响应被长 prompt 撑爆；总数在 n_ids。
            let head: Vec<Json> = ids
                .iter()
                .take(64)
                .map(|i| Json::Num(i.to_string()))
                .collect();
            j.insert("ids_head", Json::Arr(head));
            j.insert("ids_truncated", Json::Bool(ids.len() > 64));
            j.insert("prompt_roundtrip_exact", Json::Bool(rt));
            ("ran", Json::Obj(j))
        }
        Err(e) => {
            let mut j = Obj::new();
            j.insert("status", Json::Str("UNPROVEN".into()));
            j.insert("reason", Json::Str(e.clone()));
            j.insert(
                "note",
                Json::Str(
                    "BPE vocabulary did not load, so prompt encoding was NOT performed. \
                     The ids below are absent on purpose; usage.prompt_tokens counts the \
                     byte-level fallback and is labelled as such."
                        .into(),
                ),
            );
            ("UNPROVEN", Json::Obj(j))
        }
    };

    // ---- the real forward pass -------------------------------------------
    let status = engine.run(1);

    let (answer, completion_tokens, engine_json, http_status) = match &status {
        EngineStatus::Ran {
            out_elems,
            out_head,
            out_fnv1a,
            reps,
            wall_us,
            exe,
            model,
            slot,
            stdout,
        } => {
            // The reply text is DERIVED FROM THE ENGINE'S OUTPUT, not from the
            // prompt. This matters: if the response echoed the prompt it would
            // look identical whether or not the engine ran at all, and the whole
            // "the server really calls the engine" claim would be unfalsifiable.
            let text = tokenizer::decode_engine_output(&prompt, out_head, *out_fnv1a);
            let n = tokenizer::tokenize(&text).len() as u64;
            let mut j = Obj::new();
            j.insert("engine_status", Json::Str("ran".into()));
            j.insert("engine_exe", Json::Str(exe.clone()));
            j.insert("engine_model", Json::Str(model.clone()));
            j.insert("output_slot", Json::Str(slot.clone()));
            j.insert("output_elems", Json::Num(out_elems.to_string()));
            j.insert("reps", Json::Num(reps.to_string()));
            j.insert("wall_us", Json::Num(wall_us.to_string()));
            j.insert("out_fnv1a", Json::Str(format!("{:08x}", out_fnv1a)));
            j.insert(
                "out_head",
                Json::Arr(out_head.iter().map(|f| Json::Num(format!("{}", f))).collect()),
            );
            j.insert("engine_stdout", Json::Str(stdout.clone()));
            (text, n, Json::Obj(j), 200)
        }
        EngineStatus::Unavailable { reason, looked_for } => {
            // HONEST FAILURE. We do not fabricate a reply. The status is 503
            // (not 200) because the service genuinely cannot do the one thing it
            // exists to do, and a 200 here would be the exact "pretend it worked"
            // the task forbids.
            let mut j = Obj::new();
            j.insert("engine_status", Json::Str("UNPROVEN".into()));
            j.insert("reason", Json::Str(reason.clone()));
            j.insert(
                "looked_for",
                Json::Arr(
                    looked_for
                        .iter()
                        .map(|s| Json::Str(s.clone()))
                        .collect(),
                ),
            );
            j.insert(
                "note",
                Json::Str(
                    "engine_status=UNPROVEN means NO forward pass was executed for this \
                     request; the numbers above are request bookkeeping only and the \
                     content is empty on purpose."
                        .into(),
                ),
            );
            (String::new(), 0, Json::Obj(j), 503)
        }
    };

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut msg = Obj::new();
    msg.insert("role", Json::Str("assistant".into()));
    msg.insert("content", Json::Str(answer));

    let mut choice = Obj::new();
    choice.insert("index", Json::Num("0".into()));
    choice.insert("message", Json::Obj(msg));
    choice.insert(
        "finish_reason",
        Json::Str(if http_status == 200 { "stop" } else { "engine_unavailable" }.into()),
    );

    let mut usage = Obj::new();
    // P18-P1: `prompt_tokens` now reports the REAL BPE id count when the
    // vocabulary loaded, and the byte-level count only as a labelled fallback.
    // Reporting one number that silently changes meaning depending on BPE
    // availability would be the "measurement that lies" this project forbids,
    // so the unit is named explicitly in `prompt_tokens_unit`.
    let (ptok, ptok_unit) = match &bpe_info {
        Ok(b) => (b.encode(&prompt).len() as u64, "bpe_ids"),
        Err(_) => (prompt_tokens, "byte_level_fallback"),
    };
    usage.insert("prompt_tokens", Json::Num(ptok.to_string()));
    usage.insert("prompt_tokens_unit", Json::Str(ptok_unit.into()));
    usage.insert("completion_tokens", Json::Num(completion_tokens.to_string()));
    usage.insert(
        "total_tokens",
        Json::Num((ptok + completion_tokens).to_string()),
    );

    let mut root = Obj::new();
    root.insert(
        "id",
        Json::Str(format!("chatcmpl-{}", request_id)),
    );
    root.insert("object", Json::Str("chat.completion".into()));
    root.insert("created", Json::Num(created.to_string()));
    root.insert("model", Json::Str(req.model.clone()));
    root.insert("choices", Json::Arr(vec![Json::Obj(choice)]));
    root.insert("usage", Json::Obj(usage));
    root.insert("vmlm", engine_json);
    root.insert("bpe", bpe_json);
    root.insert("bpe_state", Json::Str(bpe_state.to_string()));
    root.insert("server", Json::Str(server_base_url.to_string()));
    root.insert(
        "request_echo",
        Json::Obj({
            let mut e = Obj::new();
            e.insert("messages", Json::Num(req.messages.len().to_string()));
            e.insert(
                "temperature",
                match req.temperature {
                    Some(t) => Json::Num(format!("{}", t)),
                    None => Json::Null,
                },
            );
            e.insert(
                "max_tokens",
                match req.max_tokens {
                    Some(t) => Json::Num(t.to_string()),
                    None => Json::Null,
                },
            );
            e
        }),
    );

    (http_status, crate::json::to_string(&Json::Obj(root)))
}
