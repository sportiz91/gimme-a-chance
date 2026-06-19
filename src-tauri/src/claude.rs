//! Persistent Claude Code session.
//!
//! Instead of spawning `claude --print "<prompt>"` per question (which paid the
//! full CLI boot + MCP spin-up + giant-system-prompt TTFT on EVERY call, ~10s),
//! we keep ONE `claude` process alive for the whole app lifetime, fed over stdin
//! as stream-json and read back as stream-json. The expensive boot + prompt-cache
//! write is paid once at startup (via `warmup`) and kept warm by a heartbeat, so
//! real interview questions hit a warm cache and stream their first token in ~1s.
//!
//! Architecture: a single async "manager" task owns the child's stdin+stdout and
//! processes one turn at a time (the conversation is inherently serial). Callers
//! use `ask`/`warmup`, which post a `Request` over an mpsc channel and await a
//! oneshot reply. If the child dies, the manager respawns it on the next request.

use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde_json::Value;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{mpsc, oneshot};

/// Haiku 4.5 — fast model. The whole point of the persistent session is to make
/// this snappy; the model is the other half of the latency budget.
const MODEL: &str = "claude-haiku-4-5-20251001";

/// Guardrail injected via `--append-system-prompt`. Keeps answers terse AND tells
/// the model not to reach for tools — a tool-call mid-answer would blow up latency
/// with an unbounded agentic loop. We also watch for `tool_use` events as a tripwire.
const SYSTEM_PROMPT: &str = "You are a real-time interview copilot. Answer directly \
and concisely so the user can read your answer out loud. Keep it under 4 sentences \
unless it is a coding question, in which case give the code and explain the approach \
in 1-2 sentences first. Do NOT use any tools or skills — answer purely from your own \
knowledge, immediately.";

/// A turn that hangs longer than this is treated as a dead session → respawn.
const TURN_TIMEOUT: Duration = Duration::from_secs(20);

/// Result of one completed turn.
pub struct AskOutcome {
    pub answer: String,
    /// Time to first streamed token (ms). The number that matters for "feels fast".
    pub ttft_ms: u64,
    /// Total turn time (ms), from sending the prompt to the `result` event.
    pub total_ms: u64,
    /// Tokens read from the prompt cache. >0 means the cache was warm (cheap+fast).
    pub cache_read_tokens: u64,
    /// Tokens written to the cache. >0 means a cold prefix (warmup/heartbeat pays this).
    pub cache_creation_tokens: u64,
    /// True if the model emitted a `tool_use` block — the guardrail leaked.
    pub tool_use: bool,
}

/// Streamed token delta pushed to the frontend so the answer paints incrementally.
#[derive(Clone, serde::Serialize)]
struct AnswerDelta {
    trace_id: String,
    text: String,
}

struct Request {
    prompt: String,
    trace_id: String,
    /// When false (warmup/heartbeat) we don't emit deltas to the UI.
    stream: bool,
    responder: oneshot::Sender<Result<AskOutcome>>,
}

/// Handle to the persistent session. Cheap to clone (just an mpsc sender).
#[derive(Clone)]
pub struct ClaudeSession {
    tx: mpsc::UnboundedSender<Request>,
}

impl ClaudeSession {
    /// Spawn the manager task and return a handle. The child process is created
    /// lazily on the first request (so `warmup` triggers the actual boot).
    #[must_use]
    pub fn spawn(app: AppHandle) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        tauri::async_runtime::spawn(manager(rx, app));
        Self { tx }
    }

    /// Ask a real interview question. Streams deltas to the frontend.
    pub async fn ask(&self, question: &str, context: &str, trace_id: &str) -> Result<AskOutcome> {
        self.send(build_prompt(question, context), trace_id.to_string(), true)
            .await
    }

    /// Warm up (or keep warm) the session: boots the process on first call and
    /// writes/refreshes the prompt cache. Response is discarded, not streamed.
    pub async fn warmup(&self) -> Result<AskOutcome> {
        self.send(
            "Reply with exactly one word: ready".to_string(),
            "warmup".to_string(),
            false,
        )
        .await
    }

    async fn send(&self, prompt: String, trace_id: String, stream: bool) -> Result<AskOutcome> {
        let (responder, rx) = oneshot::channel();
        self.tx
            .send(Request {
                prompt,
                trace_id,
                stream,
                responder,
            })
            .map_err(|_| anyhow::anyhow!("claude session manager is gone"))?;
        rx.await.context("claude session dropped the request")?
    }
}

/// Build the user-facing prompt (mirrors the old one-shot behaviour). The terse
/// instructions also live in the system prompt; keeping a light version here makes
/// each message self-describing if the system prompt ever changes.
fn build_prompt(question: &str, context: &str) -> String {
    if context.is_empty() {
        question.to_string()
    } else {
        format!(
            "Interview context (recent transcription):\n\
             ---\n\
             {context}\n\
             ---\n\n\
             The interviewer just asked: \"{question}\"\n\n\
             Give a concise, direct answer I can say out loud."
        )
    }
}

/// The manager task: owns the child and processes one request at a time.
async fn manager(mut rx: mpsc::UnboundedReceiver<Request>, app: AppHandle) {
    let mut io: Option<ChildIo> = None;

    while let Some(req) = rx.recv().await {
        // (Re)spawn the child if we don't have a live one.
        if io.is_none() {
            match spawn_child() {
                Ok(child_io) => io = Some(child_io),
                Err(e) => {
                    _ = req.responder.send(Err(e));
                    continue;
                }
            }
        }
        let child_io = io.as_mut().expect("just ensured Some");

        match tokio::time::timeout(TURN_TIMEOUT, process_turn(child_io, &req, &app)).await {
            Ok(Ok(outcome)) => {
                _ = req.responder.send(Ok(outcome));
            }
            Ok(Err(e)) => {
                tracing::error!(error = %e, "claude turn failed; dropping session, will respawn");
                io = None; // kill_on_drop tears down the child
                _ = req.responder.send(Err(e));
            }
            Err(_elapsed) => {
                tracing::error!(
                    timeout_s = TURN_TIMEOUT.as_secs(),
                    "claude turn timed out; respawning"
                );
                io = None;
                _ = req
                    .responder
                    .send(Err(anyhow::anyhow!("claude turn timed out")));
            }
        }
    }

    tracing::info!("claude session manager shutting down (channel closed)");
}

struct ChildIo {
    // Held to keep the process alive (kill_on_drop tears it down when this drops).
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

/// Spawn the persistent `claude` process in stream-json I/O mode. The app runs as
/// a Windows .exe while `claude` lives in WSL, so we go through `wsl.exe`.
fn spawn_child() -> Result<ChildIo> {
    let mut cmd = Command::new("wsl.exe");
    cmd.arg("--")
        .arg("/home/lasantoneta/.local/bin/claude")
        .arg("--print")
        .arg("--input-format")
        .arg("stream-json")
        .arg("--output-format")
        .arg("stream-json")
        .arg("--include-partial-messages") // token-level streaming
        .arg("--verbose") // required alongside stream-json output in --print mode
        .arg("--model")
        .arg(MODEL)
        .arg("--append-system-prompt")
        .arg(SYSTEM_PROMPT)
        .arg("--no-session-persistence") // we manage the conversation in-memory
        .arg("--dangerously-skip-permissions")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = cmd
        .spawn()
        .context("Failed to spawn claude via wsl.exe. Is WSL running and claude installed?")?;

    let stdin = child.stdin.take().context("claude child had no stdin")?;
    let stdout = child.stdout.take().context("claude child had no stdout")?;

    // Drain stderr to logs so the pipe never fills (a full stderr pipe would block
    // the child mid-write and deadlock the turn).
    if let Some(stderr) = child.stderr.take() {
        tauri::async_runtime::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(target: "claude_stderr", "{line}");
            }
        });
    }

    tracing::info!(model = MODEL, "spawned persistent claude session");
    Ok(ChildIo {
        _child: child,
        stdin,
        stdout: BufReader::new(stdout),
    })
}

/// Write one user message and read stream-json events until the turn's `result`.
async fn process_turn(io: &mut ChildIo, req: &Request, app: &AppHandle) -> Result<AskOutcome> {
    let msg = serde_json::json!({
        "type": "user",
        "message": {
            "role": "user",
            "content": [{ "type": "text", "text": req.prompt }],
        },
    });
    let line = format!("{}\n", serde_json::to_string(&msg)?);
    io.stdin
        .write_all(line.as_bytes())
        .await
        .context("write to claude stdin")?;
    io.stdin.flush().await.context("flush claude stdin")?;

    let t_send = Instant::now();
    let mut ttft_ms: u64 = 0;
    let mut acc = String::new();
    let mut tool_use = false;
    let mut buf = String::new();

    loop {
        buf.clear();
        let n = io
            .stdout
            .read_line(&mut buf)
            .await
            .context("read claude stdout")?;
        if n == 0 {
            bail!("claude stdout closed (process exited)");
        }
        let Ok(v) = serde_json::from_str::<Value>(buf.trim()) else {
            continue; // skip any non-JSON noise
        };

        match v.get("type").and_then(Value::as_str).unwrap_or("") {
            // Partial token deltas — this is where streaming + TTFT come from.
            "stream_event" => {
                let event = v.get("event");
                let is_text_delta = event.and_then(|e| e.get("type")).and_then(Value::as_str)
                    == Some("content_block_delta");
                if is_text_delta {
                    if let Some(text) = event
                        .and_then(|e| e.get("delta"))
                        .and_then(|d| d.get("text"))
                        .and_then(Value::as_str)
                    {
                        if ttft_ms == 0 {
                            ttft_ms = elapsed_ms(t_send);
                        }
                        acc.push_str(text);
                        if req.stream {
                            _ = app.emit(
                                "answer-delta",
                                AnswerDelta {
                                    trace_id: req.trace_id.clone(),
                                    text: text.to_string(),
                                },
                            );
                        }
                    }
                }
            }
            // Full assistant message — inspect for a tool_use block (guardrail tripwire).
            "assistant" => {
                if let Some(content) = v
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(Value::as_array)
                {
                    if content
                        .iter()
                        .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
                    {
                        tool_use = true;
                    }
                }
            }
            // End of turn. Carries the full text + usage (cache token counts).
            "result" => {
                let answer = v
                    .get("result")
                    .and_then(Value::as_str)
                    .map_or_else(|| acc.clone(), ToString::to_string);
                let usage = v.get("usage");
                let cache_read = usage_token(usage, "cache_read_input_tokens");
                let cache_creation = usage_token(usage, "cache_creation_input_tokens");
                let total_ms = elapsed_ms(t_send);
                if ttft_ms == 0 {
                    ttft_ms = total_ms;
                }
                return Ok(AskOutcome {
                    answer: answer.trim().to_string(),
                    ttft_ms,
                    total_ms,
                    cache_read_tokens: cache_read,
                    cache_creation_tokens: cache_creation,
                    tool_use,
                });
            }
            _ => {}
        }
    }
}

fn usage_token(usage: Option<&Value>, key: &str) -> u64 {
    usage
        .and_then(|u| u.get(key))
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

fn elapsed_ms(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}
