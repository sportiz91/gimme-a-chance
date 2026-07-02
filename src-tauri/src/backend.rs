//! Answering + vision backends over `OpenAI`-compatible HTTP providers.
//!
//! Answering (the "brain") is a fallback chain ordered speed-first so the copilot
//! never goes silent mid-interview:
//!   1. Groq `llama-3.1-8b-instant`   — fastest (~150ms TTFT)
//!   2. Groq `llama-3.3-70b-versatile` — smarter fallback (still Groq)
//!   3. `OpenAI` `gpt-4o-mini`         — survives a full Groq/Cloudflare outage
//!
//! A provider is tried until the FIRST streamed token; a failure before that
//! (HTTP error, connect/first-token timeout) cascades to the next provider with
//! nothing emitted yet. Once tokens start flowing we commit to that provider — a
//! mid-stream failure finalizes the partial answer rather than restarting.
//!
//! The user can pin the brain to a single model (`BrainModel`), and picks a
//! separate vision model (`VisionModel`) that DESCRIBES a screenshot as text
//! (see [`ApiBackend::describe`]). Both share one streaming SSE reader
//! ([`stream_sse_content`]) and one request builder ([`chat_body`]).

use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use futures_util::StreamExt;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};

use crate::lang::Language;

/// Browser User-Agent. Groq sits behind Cloudflare, whose bot filter 403s
/// (error 1010) non-browser client signatures like the default reqwest UA.
const BROWSER_UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

const OPENAI_URL: &str = "https://api.openai.com/v1/chat/completions";
const GROQ_URL: &str = "https://api.groq.com/openai/v1/chat/completions";

const SYSTEM_PROMPT_EN: &str = "You are a real-time interview copilot. Answer directly and \
concisely so the user can read your answer out loud. Keep it under 4 sentences unless it's \
a coding question, in which case give the code and explain the approach in 1-2 sentences first.";
const SYSTEM_PROMPT_ES: &str = "Sos un copiloto de entrevistas en tiempo real. Respondé en \
español, de forma directa y concisa, para que el usuario pueda leer la respuesta en voz alta. \
Máximo 4 oraciones, salvo que sea una pregunta de código, en cuyo caso dá el código y explicá \
el enfoque en 1-2 oraciones primero.";

fn system_prompt(language: Language) -> &'static str {
    match language {
        Language::English => SYSTEM_PROMPT_EN,
        Language::Spanish => SYSTEM_PROMPT_ES,
    }
}

// The vision model transcribes/describes the screen as text; it must NOT solve
// anything (that's the brain's job) and must answer in the selected language.
// Code fidelity is non-negotiable: a paraphrased or elided statement poisons
// every answer built on it, so the prompt bans ellipsis/summarizing outright.
// Deliberately framed as a neutral TRANSCRIPTIONIST, not an "interview
// copilot": with multiple shots of a problem statement, the copilot framing
// sometimes pattern-matched as exam cheating and drew refusals ("I can't help
// with that"). Transcription itself is innocuous — keep the prompt about that.
// "The user's own screen" is stated (truthfully) because a full-monitor shot
// (tabs, logged-in avatars, taskbar) otherwise reads as surveilling someone
// else's screen — another refusal magnet.
const VISION_SYS_EN: &str = "You are a meticulous screen transcriptionist. The user is sharing \
screenshots of their own screen. Transcribe and describe, faithfully, everything visible: \
problem statements, code, diagrams, error messages, and UI text. Transcribe ALL visible code \
and prose VERBATIM and COMPLETE — never use ellipsis ('...'), never summarize, never skip \
lines. Reproduce code character-for-character as an exact code block. Describe non-text \
elements (diagrams, UI) concisely. Do NOT solve, answer, or comment on anything — only \
transcribe and describe. Write in ENGLISH. Plain text.";
const VISION_SYS_ES: &str = "Sos un transcriptor meticuloso de pantallas. El usuario te \
comparte capturas de su propia pantalla. Transcribí y describí, de forma fiel, todo lo \
visible: enunciados, código, diagramas, mensajes de error y texto de UI. Transcribí TODO el \
código y el texto visibles de forma TEXTUAL y COMPLETA — nunca uses puntos suspensivos \
('...'), nunca resumas, nunca saltees líneas. Reproducí el código carácter por carácter como \
un bloque de código exacto. Los elementos no textuales (diagramas, UI) describilos de forma \
concisa. NO resuelvas, respondas ni comentes nada — solamente transcribí y describí. Escribí \
en ESPAÑOL. Texto plano.";

fn vision_system(language: Language) -> &'static str {
    match language {
        Language::English => VISION_SYS_EN,
        Language::Spanish => VISION_SYS_ES,
    }
}

// Several shots are described PER IMAGE ("Image N: …"), with scroll-stitching
// as the conditional case. The old wording asserted the shots were "consecutive
// scrolls of the SAME page" — with unrelated shots that false premise both drew
// refusals and made the model merge/drop the later images (measured: ~1650
// chars merged vs ~2000+ per-image on the same two screenshots).
fn vision_instruction(language: Language, shots: usize) -> &'static str {
    match language {
        Language::English => {
            if shots <= 1 {
                "Transcribe and describe this screenshot."
            } else {
                "Transcribe and describe each screenshot SEPARATELY and in order, under the \
                 heading 'Image N:'. If two consecutive screenshots are scrolls of the same \
                 page, continue the transcription without repeating the overlap. Transcribe \
                 ALL visible code COMPLETE — do not cut anything."
            }
        }
        Language::Spanish => {
            if shots <= 1 {
                "Transcribí y describí esta captura de pantalla."
            } else {
                "Transcribí y describí cada captura POR SEPARADO y en orden, con el encabezado \
                 'Imagen N:'. Si dos capturas consecutivas son scrolls de la misma página, \
                 continuá la transcripción sin repetir lo solapado. Todo el código visible: \
                 transcribilo COMPLETO, sin cortar nada."
            }
        }
    }
}

/// Budget to first streamed token (answer path). Exceeding it cascades to the next provider.
const FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(4);
/// Max idle between chunks AFTER the first token (a mid-stream stall).
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(10);
/// Vision first-token budget. Reasoning models (gpt-5.x) can think a while before
/// the first visible token, so give more slack than the answer path.
const VISION_FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(25);
/// Vision output ceiling. A long `LeetCode` statement + examples + code template
/// runs 1500-2500 tokens; the old 700 silently truncated it, breaking the
/// "transcribe verbatim and complete" contract.
const VISION_MAX_OUT: u32 = 2500;

/// Result of one API answer turn.
pub struct ApiOutcome {
    pub answer: String,
    pub ttft_ms: u64,
    pub total_ms: u64,
    /// Which provider actually answered, e.g. `groq/llama-3.1-8b-instant`.
    pub provider: String,
}

/// Result of one vision describe turn.
pub struct VisionOutcome {
    pub text: String,
    pub ttft_ms: u64,
    pub total_ms: u64,
}

/// Streamed content delta emitted to the frontend (same payload for answer & vision;
/// only the event name differs).
#[derive(Clone, serde::Serialize)]
struct StreamDelta {
    trace_id: String,
    text: String,
}

struct Provider {
    /// Human label shown in metrics / logs.
    name: &'static str,
    url: &'static str,
    model: &'static str,
    /// Env/keyring name of the credential this provider uses.
    key_env: &'static str,
}

const OPENAI_GPT4O_MINI: Provider = Provider {
    name: "openai/gpt-4o-mini",
    url: OPENAI_URL,
    model: "gpt-4o-mini",
    key_env: "OPENAI_API_KEY",
};
const OPENAI_GPT55: Provider = Provider {
    name: "openai/gpt-5.5",
    url: OPENAI_URL,
    model: "gpt-5.5",
    key_env: "OPENAI_API_KEY",
};

/// The default brain: speed-first Groq chain, `OpenAI` as the outage backstop.
const CHAIN: &[Provider] = &[
    Provider {
        name: "groq/llama-3.1-8b-instant",
        url: GROQ_URL,
        model: "llama-3.1-8b-instant",
        key_env: "GROQ_API_KEY",
    },
    Provider {
        name: "groq/llama-3.3-70b-versatile",
        url: GROQ_URL,
        model: "llama-3.3-70b-versatile",
        key_env: "GROQ_API_KEY",
    },
    OPENAI_GPT4O_MINI,
];
const BRAIN_4O_MINI: &[Provider] = &[OPENAI_GPT4O_MINI];
const BRAIN_GPT55: &[Provider] = &[OPENAI_GPT55];

/// The model that DESCRIBES the screen (vision-capable). Selected in the UI.
#[derive(Clone, Copy, Debug, Default)]
pub enum VisionModel {
    #[default]
    Gpt4oMini,
    Gpt55,
}

impl VisionModel {
    #[must_use]
    pub fn model_id(self) -> &'static str {
        match self {
            Self::Gpt4oMini => "gpt-4o-mini",
            Self::Gpt55 => "gpt-5.5",
        }
    }

    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Gpt4oMini => "gpt_4o_mini",
            Self::Gpt55 => "gpt_5_5",
        }
    }

    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "gpt_4o_mini" | "gpt-4o-mini" => Some(Self::Gpt4oMini),
            "gpt_5_5" | "gpt-5.5" => Some(Self::Gpt55),
            _ => None,
        }
    }
}

/// The model that ANSWERS (later: a wired-in agent). Selected in the UI.
/// `Auto` keeps the historical Groq→OpenAI fallback chain.
#[derive(Clone, Copy, Debug, Default)]
pub enum BrainModel {
    #[default]
    Auto,
    Gpt4oMini,
    Gpt55,
}

impl BrainModel {
    fn providers(self) -> &'static [Provider] {
        match self {
            Self::Auto => CHAIN,
            Self::Gpt4oMini => BRAIN_4O_MINI,
            Self::Gpt55 => BRAIN_GPT55,
        }
    }

    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gpt4oMini => "gpt_4o_mini",
            Self::Gpt55 => "gpt_5_5",
        }
    }

    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "auto" => Some(Self::Auto),
            "gpt_4o_mini" | "gpt-4o-mini" => Some(Self::Gpt4oMini),
            "gpt_5_5" | "gpt-5.5" => Some(Self::Gpt55),
            _ => None,
        }
    }
}

pub struct ApiBackend {
    client: reqwest::Client,
    groq: Option<SecretString>,
    openai: Option<SecretString>,
}

impl Default for ApiBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiBackend {
    #[must_use]
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent(BROWSER_UA)
            .build()
            .expect("failed to build reqwest client");
        let groq = crate::secrets::load_key("GROQ_API_KEY");
        let openai = crate::secrets::load_key("OPENAI_API_KEY");
        if groq.is_none() && openai.is_none() {
            tracing::error!(
                "API mode has NO usable keys (GROQ_API_KEY / OPENAI_API_KEY) — \
                 every request will fail. Set them before the interview."
            );
        } else {
            tracing::info!(
                groq_key = groq.is_some(),
                openai_key = openai.is_some(),
                "API backend initialized"
            );
        }
        Self {
            client,
            groq,
            openai,
        }
    }

    fn key_for(&self, key_env: &str) -> Option<&SecretString> {
        match key_env {
            "GROQ_API_KEY" => self.groq.as_ref(),
            "OPENAI_API_KEY" => self.openai.as_ref(),
            _ => None,
        }
    }

    /// Ask the selected brain, cascading on pre-first-token failure. Streams
    /// `answer-delta` events. Errors only if EVERY candidate provider fails.
    pub async fn ask(
        &self,
        question: &str,
        context: &str,
        language: Language,
        brain: BrainModel,
        trace_id: &str,
        app: &AppHandle,
    ) -> Result<ApiOutcome> {
        let prompt = build_user(question, context, language);
        let system = system_prompt(language);
        let mut last_err = anyhow!("no providers available (no API keys?)");
        for p in brain.providers() {
            let Some(key) = self.key_for(p.key_env) else {
                tracing::warn!(provider = p.name, "skipping provider — no key");
                continue;
            };
            match self
                .try_provider(p, key, system, &prompt, trace_id, app)
                .await
            {
                Ok(outcome) => {
                    tracing::info!(
                        provider = p.name,
                        ttft_ms = outcome.ttft_ms,
                        total_ms = outcome.total_ms,
                        "API turn answered"
                    );
                    return Ok(outcome);
                }
                Err(e) => {
                    tracing::warn!(provider = p.name, error = %e, "provider failed; cascading to next");
                    last_err = e;
                }
            }
        }
        Err(last_err)
    }

    async fn try_provider(
        &self,
        p: &Provider,
        key: &SecretString,
        system: &str,
        prompt: &str,
        trace_id: &str,
        app: &AppHandle,
    ) -> Result<ApiOutcome> {
        let body = chat_body(p.model, system, json!(prompt), 500, 0.4);
        let t0 = Instant::now();
        let resp = tokio::time::timeout(
            FIRST_TOKEN_TIMEOUT,
            self.client
                .post(p.url)
                .bearer_auth(key.expose_secret())
                .json(&body)
                .send(),
        )
        .await
        .map_err(|_| anyhow!("connect timed out"))??;

        let (answer, ttft_ms, total_ms) =
            stream_sse_content(resp, "answer-delta", trace_id, app, t0, FIRST_TOKEN_TIMEOUT)
                .await?;
        Ok(ApiOutcome {
            answer,
            ttft_ms,
            total_ms,
            provider: p.name.to_string(),
        })
    }

    /// Describe screenshots (base64 JPEGs, no `data:` prefix) as text, streaming
    /// `vision-delta` events. Several shots go in ONE multimodal message — they
    /// are consecutive scrolls of the same page, reconstructed in order. Vision
    /// always goes through `OpenAI` — the only vision-capable provider wired in.
    pub async fn describe(
        &self,
        imgs: &[String],
        model: VisionModel,
        language: Language,
        trace_id: &str,
        app: &AppHandle,
    ) -> Result<VisionOutcome> {
        let key = self
            .openai
            .as_ref()
            .ok_or_else(|| anyhow!("vision needs OPENAI_API_KEY (not set)"))?;
        if imgs.is_empty() {
            bail!("no screenshots to describe");
        }
        let mut parts = vec![json!({
            "type": "text",
            "text": vision_instruction(language, imgs.len()),
        })];
        for img in imgs {
            parts.push(json!({"type": "image_url", "image_url": {
                "url": format!("data:image/jpeg;base64,{img}"),
                "detail": "high",
            }}));
        }
        let content = Value::from(parts);
        let body = chat_body(
            model.model_id(),
            vision_system(language),
            content,
            VISION_MAX_OUT,
            0.2,
        );
        let t0 = Instant::now();
        let resp = tokio::time::timeout(
            VISION_FIRST_TOKEN_TIMEOUT,
            self.client
                .post(OPENAI_URL)
                .bearer_auth(key.expose_secret())
                .json(&body)
                .send(),
        )
        .await
        .map_err(|_| anyhow!("connect timed out"))??;

        let (text, ttft_ms, total_ms) = stream_sse_content(
            resp,
            "vision-delta",
            trace_id,
            app,
            t0,
            VISION_FIRST_TOKEN_TIMEOUT,
        )
        .await?;
        // A refusal streams back as a normal, tiny completion — flag it loudly
        // so log analysis doesn't mistake it for a nearly-empty screen.
        if looks_like_refusal(&text) {
            tracing::warn!(chars = text.len(), text = %text, "vision describe looks like a REFUSAL");
        }
        tracing::info!(
            model = model.model_id(),
            shots = imgs.len(),
            ttft_ms,
            total_ms,
            chars = text.len(),
            "vision describe done"
        );
        Ok(VisionOutcome {
            text,
            ttft_ms,
            total_ms,
        })
    }
}

/// Heuristic: a tiny completion whose text apologizes is a refusal, not a
/// nearly-empty screen. Drives the WARN above and the gpt-5.5 retry in
/// `describe_queue`.
#[must_use]
pub fn looks_like_refusal(text: &str) -> bool {
    let lower = text.to_lowercase();
    text.len() < 200
        && ["no puedo", "lo siento", "i can't", "i cannot", "sorry"]
            .iter()
            .any(|m| lower.contains(m))
}

/// Build an `OpenAI`-style chat/completions body. Reasoning models (gpt-5.x) reject
/// `temperature` and cap output via `max_completion_tokens`; classic models use
/// `max_tokens` + `temperature`.
fn chat_body(
    model: &str,
    system: &str,
    user_content: Value,
    max_out: u32,
    temperature: f32,
) -> Value {
    let mut body = json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user"},
        ],
        "stream": true,
    });
    // Move the (possibly multimodal) user content in — consuming it.
    body["messages"][1]["content"] = user_content;
    if model.starts_with("gpt-5") {
        // Reasoning + visible tokens share this budget; leave headroom for reasoning.
        body["max_completion_tokens"] = json!(max_out + 1024);
    } else {
        body["max_tokens"] = json!(max_out);
        body["temperature"] = json!(temperature);
    }
    body
}

/// Drain an `OpenAI`-style SSE `chat/completions` stream, emitting each content
/// delta as `event_name` to the frontend. Returns `(full_text, ttft_ms, total_ms)`.
/// Errors before the first token so the caller can cascade; commits after it.
async fn stream_sse_content(
    resp: reqwest::Response,
    event_name: &'static str,
    trace_id: &str,
    app: &AppHandle,
    t0: Instant,
    first_token_timeout: Duration,
) -> Result<(String, u64, u64)> {
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let snippet: String = body.chars().take(200).collect();
        bail!("HTTP {status}: {snippet}");
    }

    let mut stream = resp.bytes_stream();
    let mut line_buf = String::new();
    let mut text = String::new();
    let mut ttft_ms = 0_u64;
    let mut got_first = false;

    loop {
        let timeout = if got_first {
            STREAM_IDLE_TIMEOUT
        } else {
            first_token_timeout
        };
        let next = tokio::time::timeout(timeout, stream.next()).await;

        let chunk = match next {
            Ok(Some(Ok(bytes))) => bytes,
            Ok(Some(Err(e))) => {
                // Network error: keep what we have if streaming; else cascade.
                if got_first {
                    break;
                }
                return Err(anyhow!("stream error before first token: {e}"));
            }
            Ok(None) => break, // stream ended cleanly
            Err(_) => {
                if got_first {
                    break;
                }
                bail!("no first token within {first_token_timeout:?}");
            }
        };

        line_buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(nl) = line_buf.find('\n') {
            let line: String = line_buf.drain(..=nl).collect();
            let line = line.trim();
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                let total_ms = elapsed_ms(t0);
                return Ok((
                    text.trim().to_string(),
                    if ttft_ms == 0 { total_ms } else { ttft_ms },
                    total_ms,
                ));
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if let Some(delta) = v
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
            {
                if delta.is_empty() {
                    continue;
                }
                if !got_first {
                    got_first = true;
                    ttft_ms = elapsed_ms(t0);
                }
                text.push_str(delta);
                _ = app.emit(
                    event_name,
                    StreamDelta {
                        trace_id: trace_id.to_string(),
                        text: delta.to_string(),
                    },
                );
            }
        }
    }

    if !got_first {
        bail!("stream produced no content");
    }
    let total_ms = elapsed_ms(t0);
    Ok((text.trim().to_string(), ttft_ms, total_ms))
}

fn build_user(question: &str, context: &str, language: Language) -> String {
    // The language directive lives in the system prompt (sent every turn here), so
    // an empty-context user prompt can stay bare — the model already knows to
    // answer in the chosen language.
    match language {
        Language::English => {
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
        Language::Spanish => {
            if context.is_empty() {
                question.to_string()
            } else {
                format!(
                    "Contexto de la entrevista (transcripción reciente):\n\
                     ---\n\
                     {context}\n\
                     ---\n\n\
                     El entrevistador acaba de preguntar: \"{question}\"\n\n\
                     Dame una respuesta concisa y directa para leer en voz alta."
                )
            }
        }
    }
}

fn elapsed_ms(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}
