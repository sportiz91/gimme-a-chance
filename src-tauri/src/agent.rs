//! Agent mode: the rolling interview transcript + the Interview State block.
//!
//! The transcript is the agent's memory. Every final transcription line,
//! screen description, and clipboard ingest is appended here — append-only,
//! lines are never rewritten. That discipline is load-bearing: the agent
//! prompt is [static system] + [transcript] + [volatile tail], so an intact,
//! growing prefix lets `OpenAI`'s automatic prompt caching bill repeated
//! presses at the cached rate (~90% off input) and cut time-to-first-token.
//!
//! The Interview State is a small structured document (active question,
//! decisions made, candidate claims…) rewritten in the background by a cheap
//! model every ~[`REFRESH_DELTA_CHARS`] of new transcript. It rides at the
//! END of the agent prompt, where model attention is strongest — the antidote
//! to a decision from minute 10 sitting in the mid-context dead zone by
//! minute 90 ("lost in the middle"). The transcript itself is never truncated
//! or summarized: a 1-2h interview is ~2-5% of the answering model's context
//! window, so compaction-for-fitting never applies at this scale.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tauri::{AppHandle, Manager};

use crate::{storage, AppState};

/// New-transcript budget between state refreshes: ~10K chars ≈ 2.5K tokens
/// ≈ 10-12 minutes of speech. An absolute-delta trigger, deliberately NOT a
/// percent-of-context-window one (which would never fire at interview scale).
const REFRESH_DELTA_CHARS: u64 = 10_000;
/// An agent press refreshes a state older than this in the background (the
/// press signals the state should be current), provided enough new transcript
/// arrived to be worth a call.
const STALE_AFTER_SECS: u64 = 300;
/// Skip refreshing over a trickle — a couple of exchanged lines.
const MIN_DELTA_CHARS: u64 = 1_200;
/// Defensive clamp on the Interview State document. The prompt asks the
/// refresher for ~600 words, but the doc rides in EVERY press's volatile
/// tail — a runaway model must not grow it unbounded.
const STATE_MAX_CHARS: usize = 8_000;

/// One transcript entry. `t_s` is seconds since the session started — an
/// interview-relative clock the model can use ("at 12:30 you said…") that
/// stays byte-stable forever, unlike a wall clock.
pub struct TranscriptLine {
    pub speaker: &'static str,
    pub text: String,
    pub t_s: u64,
}

/// Prompt label per source — matches the labels the auto-answer context uses.
fn label(speaker: &str) -> &'static str {
    match speaker {
        "me" => "You",
        "screen" => "Screen",
        "clipboard" => "Clipboard",
        _ => "Interviewer",
    }
}

/// One transcript line exactly as it appears in the prompt: `[mm:ss] Label:
/// text` (mm is unbounded past the hour). Factored so the push-time token
/// count sees the same bytes `transcript_text` will send.
fn format_line(t_s: u64, speaker: &str, text: &str) -> String {
    format!(
        "[{:02}:{:02}] {}: {}",
        t_s / 60,
        t_s % 60,
        label(speaker),
        text
    )
}

pub struct AgentSession {
    started: Instant,
    lines: Mutex<Vec<TranscriptLine>>,
    /// Total transcript chars ever appended — the cheap token estimator
    /// (~4 chars/token) driving the refresh trigger and the debug meter.
    chars_total: AtomicU64,
    /// o200k tokens of every formatted line ever appended (+1/line for the
    /// newline join) — the context meter's live estimate of transcript size.
    /// Real, precise counts (unlike `chars_total`): counted at push time by
    /// the same tokenizer family the answering models use.
    line_tokens: AtomicU64,
    /// The Interview State document (markdown). Empty until the first refresh.
    state_block: Mutex<String>,
    /// `lines.len()` at the last successful refresh — the next refresh reads
    /// only the delta.
    state_covered_lines: AtomicU64,
    /// `chars_total` when the last refresh was scheduled — the delta trigger.
    chars_at_refresh: AtomicU64,
    /// Unix seconds of the last successful refresh (0 = never).
    state_epoch_s: AtomicU64,
    refresh_in_flight: AtomicBool,
}

impl Default for AgentSession {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            lines: Mutex::new(Vec::new()),
            chars_total: AtomicU64::new(0),
            line_tokens: AtomicU64::new(0),
            state_block: Mutex::new(String::new()),
            state_covered_lines: AtomicU64::new(0),
            chars_at_refresh: AtomicU64::new(0),
            state_epoch_s: AtomicU64::new(0),
            refresh_in_flight: AtomicBool::new(false),
        }
    }
}

impl AgentSession {
    /// Append a line; returns the line's session clock (`t_s`) plus whether
    /// enough new transcript accumulated that an Interview State refresh is
    /// due.
    fn push(&self, speaker: &'static str, text: &str) -> (u64, bool) {
        let t_s = self.elapsed_s();
        // Count the line as the prompt will contain it (label prefix + the
        // newline join). Sub-ms for speech lines; a 100k-char injection pays
        // a few ms on the command thread, not on any audio path.
        let toks = crate::context_meter::count_tokens(&format_line(t_s, speaker, text)) + 1;
        self.line_tokens.fetch_add(toks, Ordering::Relaxed);
        if let Ok(mut lines) = self.lines.lock() {
            lines.push(TranscriptLine {
                speaker,
                text: text.to_string(),
                t_s,
            });
        }
        let total = self
            .chars_total
            .fetch_add(text.len() as u64, Ordering::Relaxed)
            + text.len() as u64;
        let due = total.saturating_sub(self.chars_at_refresh.load(Ordering::Relaxed))
            >= REFRESH_DELTA_CHARS;
        (t_s, due)
    }

    /// Seconds since the session started — the transcript's `t_s` clock.
    /// Also stamps the non-transcript persisted events (questions, answers).
    pub fn elapsed_s(&self) -> u64 {
        self.started.elapsed().as_secs()
    }

    /// The full transcript as prompt text, `[mm:ss] Label: text` per line
    /// (mm is unbounded past the hour). Also returns the line count.
    pub fn transcript_text(&self) -> (String, usize) {
        let Ok(lines) = self.lines.lock() else {
            return (String::new(), 0);
        };
        let text = lines
            .iter()
            .map(|l| format_line(l.t_s, l.speaker, &l.text))
            .collect::<Vec<_>>()
            .join("\n");
        (text, lines.len())
    }

    /// Running o200k estimate of the transcript's prompt size in tokens —
    /// the context meter's live component (see `context_meter::gauge`).
    pub fn line_tokens_total(&self) -> u64 {
        self.line_tokens.load(Ordering::Relaxed)
    }

    /// Current Interview State document (empty until the first refresh lands).
    pub fn state_block(&self) -> String {
        self.state_block
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Inputs for a refresh: (previous state, new transcript lines since the
    /// last refresh, line index the refresh will have covered, delta chars).
    /// None when nothing new arrived.
    fn snapshot_delta(&self) -> Option<(String, String, u64, u64)> {
        let covered = usize::try_from(self.state_covered_lines.load(Ordering::Relaxed)).ok()?;
        let lines = self.lines.lock().ok()?;
        if lines.len() <= covered {
            return None;
        }
        let delta = lines[covered..]
            .iter()
            .map(|l| format!("{}: {}", label(l.speaker), l.text))
            .collect::<Vec<_>>()
            .join("\n");
        let delta_chars = delta.len() as u64;
        let upto = lines.len() as u64;
        drop(lines);
        Some((self.state_block(), delta, upto, delta_chars))
    }

    fn commit_state(&self, mut new_state: String, covered_upto: u64) {
        if new_state.len() > STATE_MAX_CHARS {
            let cut = (0..=STATE_MAX_CHARS)
                .rev()
                .find(|i| new_state.is_char_boundary(*i))
                .unwrap_or(0);
            tracing::warn!(
                chars = new_state.len(),
                clamped_to = cut,
                "state doc over budget — clamped"
            );
            new_state.truncate(cut);
        }
        if let Ok(mut s) = self.state_block.lock() {
            *s = new_state;
        }
        self.state_covered_lines
            .store(covered_upto, Ordering::Relaxed);
        self.state_epoch_s.store(now_epoch_s(), Ordering::Relaxed);
    }
}

fn now_epoch_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Append a transcript line from any source (audio pipelines, vision,
/// clipboard) and kick a background state refresh when one is due. Callable
/// from any thread — Tauri's async runtime handle is global.
pub fn push_line(app: &AppHandle, speaker: &'static str, text: &str) {
    let kind = match speaker {
        "screen" => "screen",
        "clipboard" => "clipboard",
        _ => "transcript",
    };
    push_line_tagged(app, speaker, kind, text);
}

/// [`push_line`] with an explicit persistence `kind`: the stealth selection
/// grab (Ctrl+Shift+V) tags its lines `clipboard_stealth` so the session log
/// tells the two clipboard paths apart, while the prompt label stays
/// "Clipboard" for both.
pub fn push_line_tagged(app: &AppHandle, speaker: &'static str, kind: &'static str, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    let state = app.state::<AppState>();
    let (t_s, due) = state.agent.push(speaker, text);
    storage::record(storage::Event {
        kind,
        speaker: (kind == "transcript").then_some(speaker),
        content: text.to_string(),
        t_s,
        meta: None,
    });
    state.metrics.transcript_chars.store(
        state.agent.chars_total.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    state
        .metrics
        .transcript_lines
        .fetch_add(1, Ordering::Relaxed);
    // A landed interviewer final is the wake-up for any deferred agent press
    // (the press waits on this when the question was still being spoken).
    if kind == "transcript" && speaker == "interviewer" {
        state.interviewer_final.notify_waiters();
    }
    if due {
        spawn_refresh(app);
    }
}

/// Background-refresh the state if it's stale and enough transcript arrived —
/// called on each agent press so a long quiet stretch ends with a catch-up.
pub fn refresh_if_stale(app: &AppHandle) {
    let state = app.state::<AppState>();
    let epoch = state.agent.state_epoch_s.load(Ordering::Relaxed);
    let age_s = now_epoch_s().saturating_sub(epoch);
    let delta_chars = state
        .agent
        .chars_total
        .load(Ordering::Relaxed)
        .saturating_sub(state.agent.chars_at_refresh.load(Ordering::Relaxed));
    if age_s >= STALE_AFTER_SECS && delta_chars >= MIN_DELTA_CHARS {
        spawn_refresh(app);
    }
}

/// Run one Interview State update on the cheap model, off the caller's
/// thread. At most one in flight; a failure retries only after another
/// transcript delta accumulates (no hot retry loop against a down API).
fn spawn_refresh(app: &AppHandle) {
    let state = app.state::<AppState>();
    let session = Arc::clone(&state.agent);
    let api = Arc::clone(&state.api);
    let metrics = Arc::clone(&state.metrics);
    if session.refresh_in_flight.swap(true, Ordering::SeqCst) {
        return;
    }
    let Some((prev_state, delta, covered_upto, delta_chars)) = session.snapshot_delta() else {
        session.refresh_in_flight.store(false, Ordering::SeqCst);
        return;
    };
    session.chars_at_refresh.store(
        session.chars_total.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    tauri::async_runtime::spawn(async move {
        let t0 = Instant::now();
        match api.refresh_interview_state(&prev_state, &delta).await {
            Ok((new_state, request_id, usage)) => {
                tracing::info!(
                    target: "agent",
                    delta_chars,
                    state_chars = new_state.len(),
                    total_ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
                    request_id = request_id.as_deref().unwrap_or("-"),
                    state = %new_state,
                    "interview state refreshed"
                );
                session.commit_state(new_state, covered_upto);
                metrics.add_spend(usage);
                metrics
                    .state_epoch_s
                    .store(now_epoch_s(), Ordering::Relaxed);
            }
            Err(e) => {
                tracing::warn!(
                    target: "agent",
                    error = %e,
                    delta_chars,
                    "interview state refresh failed (will retry after more transcript)"
                );
            }
        }
        session.refresh_in_flight.store(false, Ordering::SeqCst);
    });
}
