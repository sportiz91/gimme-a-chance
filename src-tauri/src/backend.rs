//! Answering backend: a direct HTTP fallback chain over `OpenAI`-compatible providers.
//!
//! The API chain is ordered for speed-first with graceful degradation, so the
//! copilot never goes silent mid-interview:
//!   1. Groq `llama-3.1-8b-instant`   — fastest (~150ms TTFT)
//!   2. Groq `llama-3.3-70b-versatile` — smarter fallback (still Groq)
//!   3. `OpenAI` `gpt-4o-mini`         — survives a full Groq/Cloudflare outage
//!
//! A provider is tried until the FIRST streamed token; a failure before that
//! (HTTP error, connect/first-token timeout) cascades to the next provider with
//! nothing emitted yet. Once tokens start flowing we commit to that provider — a
//! mid-stream failure finalizes the partial answer rather than restarting (which
//! would garble the UI).

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

/// Budget to first streamed token. Exceeding it cascades to the next provider.
const FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(4);
/// Max idle between chunks AFTER the first token (a mid-stream stall).
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(10);

/// Result of one API turn.
pub struct ApiOutcome {
    pub answer: String,
    pub ttft_ms: u64,
    pub total_ms: u64,
    /// Which provider actually answered, e.g. `groq/llama-3.1-8b-instant`.
    pub provider: String,
}

#[derive(Clone, serde::Serialize)]
struct AnswerDelta {
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

const CHAIN: &[Provider] = &[
    Provider {
        name: "groq/llama-3.1-8b-instant",
        url: "https://api.groq.com/openai/v1/chat/completions",
        model: "llama-3.1-8b-instant",
        key_env: "GROQ_API_KEY",
    },
    Provider {
        name: "groq/llama-3.3-70b-versatile",
        url: "https://api.groq.com/openai/v1/chat/completions",
        model: "llama-3.3-70b-versatile",
        key_env: "GROQ_API_KEY",
    },
    Provider {
        name: "openai/gpt-4o-mini",
        url: "https://api.openai.com/v1/chat/completions",
        model: "gpt-4o-mini",
        key_env: "OPENAI_API_KEY",
    },
];

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

    /// Ask the chain, cascading on pre-first-token failure. Streams `answer-delta`
    /// events to the frontend. Errors only if EVERY provider fails.
    pub async fn ask(
        &self,
        question: &str,
        context: &str,
        language: Language,
        trace_id: &str,
        app: &AppHandle,
    ) -> Result<ApiOutcome> {
        let prompt = build_user(question, context, language);
        let system = system_prompt(language);
        let mut last_err = anyhow!("no providers available (no API keys?)");
        for p in CHAIN {
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

    // A cohesive streaming routine: send, race the first token, then drain the SSE
    // stream emitting deltas. Splitting it would just scatter the shared loop state.
    #[allow(clippy::too_many_lines)]
    async fn try_provider(
        &self,
        p: &Provider,
        key: &SecretString,
        system: &str,
        prompt: &str,
        trace_id: &str,
        app: &AppHandle,
    ) -> Result<ApiOutcome> {
        let body = json!({
            "model": p.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": true,
            "max_tokens": 500,
            "temperature": 0.4,
        });

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

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let snippet: String = body.chars().take(200).collect();
            bail!("HTTP {status}: {snippet}");
        }

        let mut stream = resp.bytes_stream();
        let mut line_buf = String::new();
        let mut answer = String::new();
        let mut ttft_ms = 0_u64;
        let mut got_first = false;

        loop {
            let timeout = if got_first {
                STREAM_IDLE_TIMEOUT
            } else {
                FIRST_TOKEN_TIMEOUT
            };
            let next = tokio::time::timeout(timeout, stream.next()).await;

            let chunk = match next {
                Ok(Some(Ok(bytes))) => bytes,
                Ok(Some(Err(e))) => {
                    // Network error: bail (cascade) if nothing sent yet, else keep what we have.
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
                    bail!("no first token within {FIRST_TOKEN_TIMEOUT:?}");
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
                    return Ok(ApiOutcome {
                        answer: answer.trim().to_string(),
                        ttft_ms: if ttft_ms == 0 { total_ms } else { ttft_ms },
                        total_ms,
                        provider: p.name.to_string(),
                    });
                }
                let Ok(v) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                if let Some(text) = v
                    .pointer("/choices/0/delta/content")
                    .and_then(Value::as_str)
                {
                    if text.is_empty() {
                        continue;
                    }
                    if !got_first {
                        got_first = true;
                        ttft_ms = elapsed_ms(t0);
                    }
                    answer.push_str(text);
                    _ = app.emit(
                        "answer-delta",
                        AnswerDelta {
                            trace_id: trace_id.to_string(),
                            text: text.to_string(),
                        },
                    );
                }
            }
        }

        if !got_first {
            bail!("stream produced no content");
        }
        let total_ms = elapsed_ms(t0);
        Ok(ApiOutcome {
            answer: answer.trim().to_string(),
            ttft_ms,
            total_ms,
            provider: p.name.to_string(),
        })
    }
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
