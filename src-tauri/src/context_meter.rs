//! Live context-token meter: "how many tokens is the brain's context RIGHT NOW?"
//!
//! That question has two answers. The EXACT one is `usage.prompt_tokens` of the
//! last agent request — computed by the server that bills it, fidelity perfect
//! by definition. But it only exists after a request, and the number the user
//! most wants to see is the one BEFORE any press: right after the manager
//! injects a past interview's context (💉).
//!
//! The scheme (same as codex/Cline, plus a local estimator for the gap):
//! every line entering the agent transcript is counted at push time with the
//! `o200k_base` tokenizer — the identical vocabulary `OpenAI`'s servers run for
//! the gpt-4o/gpt-5 family, so plain-text counts match theirs to within the
//! chat-format wrapper (~4 tokens/message). Whenever a real `usage` lands
//! (agent press or warm ping) it becomes the ANCHOR, and the local count only
//! covers lines added since — the meter's error is bounded by the un-anchored
//! delta, never the whole context.
//!
//! NOT covered on purpose: the auto-answer (`ask_brain`) rolling 20-line
//! context (a bounded window over these same lines — it can't grow), and the
//! agent's own answers (they are never re-injected into any context; the user
//! speaking them is what echoes them back, via STT).

use std::sync::atomic::Ordering;
use std::time::Instant;

use tiktoken_rs::o200k_base_singleton;

use crate::backend;
use crate::AppState;

/// Chat-format wrapper tokens the server adds around the 3 agent messages
/// (per-message framing + reply priming) — the only part of the prompt a
/// local count can't see. The anchor absorbs the residual after the first
/// real `usage` arrives.
const CHAT_FORMAT_OVERHEAD: u64 = 12;

/// `o200k_base` token count of `text`. The singleton builds the BPE ranks on
/// first use (~100ms) — [`warmup`] pays that at startup, off the hot paths.
pub fn count_tokens(text: &str) -> u64 {
    o200k_base_singleton()
        .encode_with_special_tokens(text)
        .len() as u64
}

/// Build the tokenizer off-thread at startup so the first transcript line
/// (pushed from an STT worker thread) never pays the init cost.
pub fn warmup() {
    tauri::async_runtime::spawn_blocking(|| {
        let t0 = Instant::now();
        let _ = count_tokens("warmup");
        tracing::info!(
            init_ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
            "o200k tokenizer ready"
        );
    });
}

/// One reading of the meter.
pub struct ContextGauge {
    /// Best current estimate of the agent context size in tokens.
    pub used: u64,
    /// The share of `used` that is a local estimate the API hasn't confirmed
    /// yet — 0 right after a press with nothing new ingested since. The UI
    /// shows `~` while it's non-zero.
    pub pending: u64,
    /// Context window of the model an agent press would use right now.
    pub window: u64,
}

/// Compute the meter from current app state. Cheap enough for the 2s metrics
/// tick: post-anchor it's pure atomics + one saturating sub; pre-anchor it
/// re-tokenizes only the (small) system prompt + Interview State tail.
pub fn gauge(state: &AppState) -> ContextGauge {
    let brain = state.brain_model.lock().map(|g| *g).unwrap_or_default();
    let window = backend::context_window(brain.agent_model_id());
    let line_tokens = state.agent.line_tokens_total();

    let anchor_prompt = state.metrics.agent_prompt_tokens.load(Ordering::Relaxed);
    if anchor_prompt > 0 {
        let anchor_lines = state
            .metrics
            .agent_anchor_line_tokens
            .load(Ordering::Relaxed);
        let pending = line_tokens.saturating_sub(anchor_lines);
        return ContextGauge {
            used: anchor_prompt + pending,
            pending,
            window,
        };
    }

    // No usage anchor yet this session: estimate the whole prompt the next
    // press would send — [styled system] + [transcript] + [volatile tail].
    let language = state
        .language
        .lock()
        .map(|g| *g)
        .unwrap_or(crate::lang::Language::English);
    let style = state.response_style.lock().map(|g| *g).unwrap_or_default();
    let base = backend::agent_prompt_base_tokens(language, style, &state.agent.state_block());
    let used = base + line_tokens + CHAT_FORMAT_OVERHEAD;
    ContextGauge {
        used,
        pending: used,
        window,
    }
}
