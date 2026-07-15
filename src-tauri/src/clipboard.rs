//! Clipboard ingestion — the exact-text context source.
//!
//! Vision OCR can mangle a character; the clipboard cannot. Anything the user
//! can select (editor code, a problem statement) is best ingested by copying
//! it. Two paths, both landing in the frontend's `clipboard-text` handling:
//!
//! - **Auto-clip**: an OS clipboard listener (clipboard-master — event-driven,
//!   no polling) fires on every copy. While the toggle is on, text that differs
//!   from the last ingested clip is emitted as a `clipboard-text` event.
//! - **Manual** (Ctrl+Shift+V → `copy_and_ingest` command): turns the current
//!   selection into context. Primary path: UI Automation `TextPattern` reads
//!   the focused app's selection directly — no keystroke, no clipboard write,
//!   nothing the target app (or its JS `copy` listeners) can observe. Fallback
//!   for apps without UIA text support: synthesize Ctrl+C and read the
//!   clipboard; the watcher is suppressed meanwhile so the clip isn't
//!   ingested twice.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use clipboard_master::{CallbackResult, ClipboardHandler, Master};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use tauri::{AppHandle, Emitter};

/// Ingest cap. Sized for whole prep documents (~25k tokens), not just
/// interview-sized code snippets: the answering models run with ~1M-token
/// context, so the ceiling only exists so a stray "copy a whole log file"
/// can't flood the answer context.
const MAX_CLIP_CHARS: usize = 100_000;

/// Payload of the `clipboard-text` event (the auto-clip path).
#[derive(Clone, serde::Serialize)]
struct ClipboardText {
    text: String,
}

/// Read the current clipboard text, trimmed and capped at [`MAX_CLIP_CHARS`].
/// Retries briefly: right after a copy, the source app may still hold the
/// clipboard open (a classic Windows race). Blocking — call off the runtime.
pub fn read_text() -> Result<String> {
    let mut last_err = anyhow!("clipboard read failed");
    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_millis(50));
        }
        match arboard::Clipboard::new().and_then(|mut c| c.get_text()) {
            Ok(text) => return Ok(cap(text.trim())),
            Err(e) => last_err = anyhow!("clipboard read: {e}"),
        }
    }
    Err(last_err)
}

/// Read the focused app's current selection via UI Automation (`TextPattern`).
/// No keystroke, no clipboard write — the target app sees no copy happen, so
/// copy-detection (e.g. a page's JS `copy` listener) never fires. Two attempts:
/// the first UIA query against a Chromium window switches its accessibility
/// tree on, which can take a beat to populate.
fn read_selection_uia() -> Result<String> {
    let automation = uiautomation::UIAutomation::new().map_err(|e| anyhow!("uia init: {e}"))?;
    let mut last_err = anyhow!("uia selection read failed");
    for attempt in 0..2 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_millis(150));
        }
        match try_selection_once(&automation) {
            Ok(text) => return Ok(text),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

fn try_selection_once(automation: &uiautomation::UIAutomation) -> Result<String> {
    let focused = automation
        .get_focused_element()
        .map_err(|e| anyhow!("uia focused element: {e}"))?;
    let pattern = focused
        .get_pattern::<uiautomation::patterns::UITextPattern>()
        .map_err(|e| anyhow!("focused element has no TextPattern: {e}"))?;
    let ranges = pattern
        .get_selection()
        .map_err(|e| anyhow!("uia get_selection: {e}"))?;
    let mut text = String::new();
    for range in &ranges {
        if let Ok(t) = range.get_text(-1) {
            text.push_str(&t);
        }
    }
    let text = text.trim();
    if text.is_empty() {
        bail!("empty selection via UIA");
    }
    Ok(cap(text))
}

/// Fallback: simulate Ctrl+C on the focused app, wait for it to publish the
/// copy, then read the clipboard. The hotkey chord (Ctrl+Shift+V) may still be
/// physically held when this runs — a leaked Shift would turn the copy into
/// Ctrl+Shift+C (`DevTools` in browsers), so Shift is synthetically released
/// first (a no-op if the user already let go).
fn copy_via_synthetic_ctrl_c() -> Result<String> {
    let mut enigo = Enigo::new(&Settings::default()).map_err(|e| anyhow!("enigo init: {e}"))?;
    // A beat for the user's fingers to leave the chord before we synthesize.
    std::thread::sleep(Duration::from_millis(120));
    (|| {
        enigo.key(Key::Shift, Direction::Release)?;
        enigo.key(Key::Control, Direction::Press)?;
        enigo.key(Key::Unicode('c'), Direction::Click)?;
        enigo.key(Key::Control, Direction::Release)
    })()
    .map_err(|e| anyhow!("synthetic Ctrl+C: {e}"))?;
    // Let the target app write the clipboard before reading it.
    std::thread::sleep(Duration::from_millis(150));
    read_text()
}

/// Grab the current selection: UIA first (invisible to the target app), then
/// the synthetic-copy fallback for apps without UIA text support. Blocking —
/// call off the runtime.
pub fn copy_selection_and_read() -> Result<String> {
    match read_selection_uia() {
        Ok(text) => {
            tracing::info!(chars = text.len(), "selection read via UIA (no copy event)");
            return Ok(text);
        }
        Err(e) => {
            tracing::warn!(error = %e, "UIA selection read failed; falling back to synthetic Ctrl+C");
        }
    }
    copy_via_synthetic_ctrl_c()
}

fn cap(s: &str) -> String {
    let mut out: String = s.chars().take(MAX_CLIP_CHARS).collect();
    if out.len() < s.len() {
        out.push_str("\n…[truncated]");
    }
    out
}

struct Watcher {
    app: AppHandle,
    enabled: Arc<AtomicBool>,
    /// True while a manual `copy_and_ingest` is in flight — its synthetic copy
    /// fires this watcher too, and the manual path already ingests the text.
    suppress: Arc<AtomicBool>,
    last_clip: Arc<Mutex<String>>,
}

impl ClipboardHandler for Watcher {
    fn on_clipboard_change(&mut self) -> CallbackResult {
        if self.suppress.load(Ordering::Relaxed) || !self.enabled.load(Ordering::Relaxed) {
            return CallbackResult::Next;
        }
        match read_text() {
            Ok(text) if !text.is_empty() => {
                // Dedup against the last ingested clip (manual ingests update it
                // too) so re-copying the same thing doesn't spam the context.
                if let Ok(mut last) = self.last_clip.lock() {
                    if *last == text {
                        return CallbackResult::Next;
                    }
                    last.clone_from(&text);
                }
                tracing::info!(chars = text.len(), "auto-clip ingested");
                crate::agent::push_line(&self.app, "clipboard", &text);
                _ = self.app.emit("clipboard-text", ClipboardText { text });
            }
            // Empty text, or a non-text copy (image/files) — nothing to ingest.
            Ok(_) => {}
            Err(e) => tracing::debug!(error = %e, "clipboard change without readable text"),
        }
        CallbackResult::Next
    }

    fn on_clipboard_error(&mut self, error: std::io::Error) -> CallbackResult {
        tracing::warn!(error = %error, "clipboard watcher error");
        CallbackResult::Next
    }
}

/// Spawn the OS clipboard listener on its own thread (clipboard-master runs a
/// blocking message loop). Lives for the whole app.
pub fn spawn_watcher(
    app: AppHandle,
    enabled: Arc<AtomicBool>,
    suppress: Arc<AtomicBool>,
    last_clip: Arc<Mutex<String>>,
) {
    std::thread::spawn(move || {
        let watcher = Watcher {
            app,
            enabled,
            suppress,
            last_clip,
        };
        match Master::new(watcher) {
            Ok(mut master) => {
                if let Err(e) = master.run() {
                    tracing::error!(error = %e, "clipboard watcher stopped");
                }
            }
            Err(e) => tracing::error!(error = %e, "clipboard watcher failed to start"),
        }
    });
}
