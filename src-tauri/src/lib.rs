#[cfg(feature = "sherpa")]
mod aec;
mod agent;
#[cfg(feature = "counting-alloc")]
mod alloc_counter;
mod audio;
mod backend;
mod capture;
mod clipboard;
mod cloud_stt;
mod commands;
mod crashlog;
#[cfg(feature = "sherpa")]
mod dtln;
mod error;
mod lang;
mod latency;
mod metrics;
mod secrets;
mod storage;
#[cfg(feature = "sherpa")]
mod stt;
mod telemetry;
mod transcriber;
mod tts;
mod vad;

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use tauri::{Emitter, Manager};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

// Only one `#[global_allocator]` is allowed per binary, so the active
// allocator is selected by feature flags in priority order:
// - `--features dhat-heap`:      dhat profiler (per-alloc call stacks)
// - `--features counting-alloc`: CountingAllocator — counts bytes + size
//                                histogram + peak. Wraps System OR MiMalloc
//                                depending on `--features mimalloc`.
// - `--features mimalloc` alone: MiMalloc directly, no instrumentation.
// - default (debug):             assert_no_alloc real-time guard.
// - default (release):           system allocator (HeapAlloc on Windows).

#[cfg(all(feature = "counting-alloc", feature = "dhat-heap"))]
compile_error!("features `counting-alloc` and `dhat-heap` are mutually exclusive — pick one");

#[cfg(all(feature = "dhat-heap", feature = "mimalloc"))]
compile_error!(
    "feature `dhat-heap` already replaces the global allocator; `mimalloc` cannot combine with it"
);

#[cfg(all(
    debug_assertions,
    not(feature = "counting-alloc"),
    not(feature = "dhat-heap"),
    not(feature = "mimalloc")
))]
#[global_allocator]
static ALLOCATOR: assert_no_alloc::AllocDisabler = assert_no_alloc::AllocDisabler;

#[cfg(feature = "counting-alloc")]
#[global_allocator]
static ALLOCATOR: alloc_counter::CountingAllocator = alloc_counter::CountingAllocator;

// `mimalloc` alone (no counting-alloc, no dhat) just replaces the global
// allocator with MiMalloc directly. Useful for production builds where you
// want mimalloc's performance without any instrumentation overhead.
#[cfg(all(
    feature = "mimalloc",
    not(feature = "counting-alloc"),
    not(feature = "dhat-heap")
))]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

// Stores the dhat profiler so we can drop it explicitly from Tauri's
// RunEvent::Exit handler (needed because winit calls process::exit()
// on the main thread, skipping destructors of locals in `run()`).
#[cfg(feature = "dhat-heap")]
static DHAT_PROFILER: std::sync::OnceLock<std::sync::Mutex<Option<dhat::Profiler>>> =
    std::sync::OnceLock::new();

/// Shared app state across Tauri commands
pub struct AppState {
    pub is_listening: Arc<Mutex<bool>>,
    /// Agent-mode session: the rolling interview transcript (all sources,
    /// append-only) + the background-refreshed Interview State document.
    pub agent: Arc<agent::AgentSession>,
    pub metrics: Arc<metrics::Metrics>,
    /// Direct-API backend (Groq → `OpenAI` fallback chain). Built at startup.
    pub api: Arc<backend::ApiBackend>,
    /// Vision model that describes screenshots. Toggled from the UI; default gpt-4o-mini.
    pub vision_model: Arc<Mutex<backend::VisionModel>>,
    /// Brain model that answers (later: a wired-in agent). Toggled from the UI; default Auto.
    pub brain_model: Arc<Mutex<backend::BrainModel>>,
    /// Latest screen description (for the future agent + debug). Overwritten each capture.
    pub last_description: Arc<Mutex<String>>,
    /// Screenshots queued for a multi-shot describe (base64 JPEGs, in capture
    /// order = the user's top-to-bottom scroll order).
    pub capture_queue: Arc<Mutex<Vec<String>>>,
    /// Auto-clip: when true (the default), every OS copy is ingested as context.
    /// The UI checkbox is the off switch for noisy copy-paste sessions.
    pub auto_clip: Arc<AtomicBool>,
    /// True while `copy_and_ingest`'s synthetic Ctrl+C is in flight — tells the
    /// clipboard watcher to skip that change (the manual path ingests it).
    pub manual_copy: Arc<AtomicBool>,
    /// Last ingested clip — dedup for the auto watcher (manual updates it too).
    pub last_clip: Arc<Mutex<String>>,
    /// Transcription + answer language. Toggled from the UI; default = English.
    /// Read at `start_listening` (STT engine) and `ask_brain` (prompt) time.
    pub language: Arc<Mutex<lang::Language>>,
    /// Text-to-speech engine for the "simulate interviewer" self-test.
    pub tts: Arc<tts::TtsEngine>,
    /// The local English whisper model (`base.en`), loaded once and shared across
    /// Listen sessions and (in dual mode) both capture pipelines — the offline STT
    /// fallback.
    pub whisper: Arc<OnceLock<Arc<transcriber::WhisperTranscriber>>>,
    /// The local Spanish whisper model (multilingual `base`), loaded lazily on the
    /// first Spanish Listen so English users never pay for it.
    pub whisper_es: Arc<OnceLock<Arc<transcriber::WhisperTranscriber>>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            is_listening: Arc::new(Mutex::new(false)),
            agent: Arc::new(agent::AgentSession::default()),
            metrics: Arc::new(metrics::Metrics::default()),
            api: Arc::new(backend::ApiBackend::new()),
            vision_model: Arc::new(Mutex::new(backend::VisionModel::default())),
            brain_model: Arc::new(Mutex::new(backend::BrainModel::default())),
            last_description: Arc::new(Mutex::new(String::new())),
            capture_queue: Arc::new(Mutex::new(Vec::new())),
            auto_clip: Arc::new(AtomicBool::new(true)),
            manual_copy: Arc::new(AtomicBool::new(false)),
            last_clip: Arc::new(Mutex::new(String::new())),
            language: Arc::new(Mutex::new(lang::Language::default())),
            tts: Arc::new(tts::TtsEngine::new()),
            whisper: Arc::new(OnceLock::new()),
            whisper_es: Arc::new(OnceLock::new()),
        }
    }
}

// Overlay hiding without the black-box bug.
//
// On Windows, `hide()`/`show()` degrades a window's `WDA_EXCLUDEFROMCAPTURE`
// display affinity to `WDA_MONITOR` — a solid BLACK rectangle in any screen
// share — and re-asserting the affinity afterward does NOT restore it (measured;
// tauri#14189). So we never `hide()`/`show()` the overlays. "Hiding" parks a
// window far off-screen; "showing" moves it back. Windows stay VISIBLE the whole
// time, so the exclusion holds and the overlay is truly invisible to
// Meet/Zoom/Teams across any number of hide/show cycles.

/// Off-screen parking coordinate (well outside any monitor).
const OFFSCREEN_XY: i32 = -32000;

/// Parked-windows map: label → the on-screen position to restore it to. `None`
/// means "no saved spot yet" (reveal centers it). Key present ⇒ window is hidden.
type ParkedWindows = HashMap<String, Option<(i32, i32)>>;

static PARKED: LazyLock<Mutex<ParkedWindows>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Labels parked by the last Ctrl+Shift+H press, restored on the next one.
static HIDDEN_WINDOWS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// True if the window is currently parked off-screen (our notion of "hidden").
pub(crate) fn is_hidden(label: &str) -> bool {
    PARKED
        .lock()
        .map(|p| p.contains_key(label))
        .unwrap_or(false)
}

/// Park a window off-screen, remembering where it was so `reveal_onscreen` can
/// put it back. No-op if already parked. Never touches visibility or the display
/// affinity, so the window stays excluded from capture.
pub(crate) fn park_offscreen(window: &tauri::WebviewWindow) {
    let label = window.label().to_string();
    {
        let mut parked = PARKED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if parked.contains_key(&label) {
            return;
        }
        let pos = window.outer_position().ok().map(|p| (p.x, p.y));
        tracing::debug!(label = window.label(), saved = ?pos, "park off-screen");
        parked.insert(label, pos);
    }
    if let Err(e) = window.set_position(tauri::PhysicalPosition::new(OFFSCREEN_XY, OFFSCREEN_XY)) {
        tracing::warn!(error = %e, label = window.label(), "park off-screen failed");
    }
}

/// Move a parked window back on-screen — to its saved spot, or centered on the
/// first reveal. Never calls `show()`: the window was already visible, just
/// off-screen, so its `WDA_EXCLUDEFROMCAPTURE` affinity was never disturbed.
pub(crate) fn reveal_onscreen(window: &tauri::WebviewWindow) {
    let label = window.label().to_string();
    let saved = PARKED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&label);
    match saved {
        Some(Some((x, y))) => {
            if let Err(e) = window.set_position(tauri::PhysicalPosition::new(x, y)) {
                tracing::warn!(error = %e, label = window.label(), "reveal set_position failed");
            }
        }
        _ => {
            if let Err(e) = window.center() {
                tracing::warn!(error = %e, label = window.label(), "reveal center failed");
            }
        }
    }
    // Cheap insurance — we never hide/show, so the affinity should already hold.
    _ = window.set_content_protected(true);
    if let Ok(pos) = window.outer_position() {
        tracing::debug!(
            label = window.label(),
            x = pos.x,
            y = pos.y,
            "revealed on-screen"
        );
    }
}

// `run` is a long-by-nature Tauri builder: shortcut registration, the async
// metrics emitter (with a sizeable counting-alloc branch), and app wiring all
// live here.
#[allow(clippy::too_many_lines)]
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Disable Chromium's native window occlusion tracking BEFORE any webview is
    // created. Our overlays get parked fully off-screen for "hidden"; with
    // occlusion tracking on, Chromium stops compositing them and they come back
    // BLANK when moved on-screen (a `focusable:false` window never gets the
    // activation event that would resume it). The env var APPENDS to wry's
    // default browser args (Chromium unions `--disable-features`), and being
    // process-global it applies identically to every webview — sidestepping the
    // blank/frozen-window trap of per-window arg mismatches (tauri#13092).
    #[cfg(windows)]
    std::env::set_var(
        "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
        "--disable-features=CalculateNativeWinOcclusion",
    );

    // The guards must live for the whole program lifetime, otherwise the non-blocking
    // writer thread is dropped and pending log lines may be lost on exit.
    let _telemetry_guards: &'static _ = Box::leak(Box::new(telemetry::init()));

    // On Windows, Tauri's underlying winit event loop calls `std::process::exit()`
    // when the last window closes, which SKIPS Rust destructors for locals in
    // this function. That means a locally-bound `dhat::Profiler` would never
    // drop → never write its JSON.
    //
    // Workaround: stash the profiler in a static `OnceLock<Mutex<Option<_>>>`
    // and drop it explicitly from Tauri's `RunEvent::Exit` callback, which
    // fires BEFORE the process-exit call.
    #[cfg(feature = "dhat-heap")]
    {
        let path = telemetry::logs_dir().join("dhat-heap.json");
        let profiler = dhat::Profiler::builder().file_name(&path).build();
        DHAT_PROFILER
            .set(std::sync::Mutex::new(Some(profiler)))
            .ok();
        tracing::info!(path = %path.display(), "dhat heap profiler started");
    }

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        allocator = metrics::active_allocator_name(),
        "gimme-a-chance starting"
    );

    // Meeting persistence: open sessions.sqlite, insert this run's session
    // row, spawn the writer thread. Before the builder so no early producer
    // (the clipboard watcher, a fast Listen) can beat it. A failure only
    // warns — a broken database must never keep the copilot from starting.
    if let Err(e) = storage::init() {
        tracing::warn!(error = %e, "session persistence unavailable");
    }

    // Ctrl+Shift+H toggles the overlay's visibility (same binding as screen-peek).
    let toggle_shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyH);
    let debug_shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyD);
    // Graceful quit: Tauri's run loop returns cleanly, letting local guards like
    // the dhat profiler drop properly and write their output. Killing via Ctrl+C
    // in the terminal would skip destructors.
    let quit_shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyQ);
    // Screenshot queue (same bindings as screen-peek): Ctrl+Shift+Enter queues a
    // capture of the current screen; Ctrl+Shift+1 describes the whole queue in
    // one vision call (or captures+describes in one go when the queue is empty).
    let queue_shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Enter);
    let describe_shortcut =
        Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Digit1);
    // Ctrl+Shift+V: re-ingest the current clipboard by hand (the auto-clip
    // watcher already ingests every copy while the checkbox is on).
    let clip_shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyV);
    // Ctrl+Shift+Space: the agent press — "read everything so far and help me
    // with whatever is needed RIGHT NOW" (no question heuristics involved).
    let agent_shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space);

    let app_state = AppState::default();
    let metrics_for_emitter = Arc::clone(&app_state.metrics);
    let whisper_cell = Arc::clone(&app_state.whisper);
    let auto_clip = Arc::clone(&app_state.auto_clip);
    let manual_copy = Arc::clone(&app_state.manual_copy);
    let last_clip = Arc::clone(&app_state.last_clip);

    // Preload the whisper model at startup (off the main thread) so the first
    // "Listen" doesn't pay the ~140MB load, and so it's loaded only once.
    std::thread::spawn(move || {
        match transcriber::WhisperTranscriber::new(lang::Language::English) {
            Ok(w) => {
                _ = whisper_cell.set(Arc::new(w));
                tracing::info!("whisper model preloaded at startup");
            }
            Err(e) => {
                tracing::warn!(error = %e, "whisper preload failed (will retry on first Listen)");
            }
        }
    });

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, shortcut, event| {
                    if event.state() != ShortcutState::Pressed {
                        return;
                    }
                    if shortcut == &toggle_shortcut {
                        // Panic/toggle key. If any overlay is on-screen, park them
                        // all off-screen (remembering which); otherwise bring back
                        // the set the last press parked (default: just main).
                        // Parking (never hide/show) keeps content protection intact,
                        // so nothing is ever captured as a black box — main AND the
                        // answer pop-out (tauri#14189).
                        let windows = app.webview_windows();
                        let on_screen: Vec<String> = windows
                            .keys()
                            .filter(|l| !is_hidden(l.as_str()))
                            .cloned()
                            .collect();
                        if on_screen.is_empty() {
                            let restore = {
                                let mut h =
                                    HIDDEN_WINDOWS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                                if h.is_empty() {
                                    vec!["main".to_string()]
                                } else {
                                    std::mem::take(&mut *h)
                                }
                            };
                            tracing::debug!(?restore, "window toggle: reveal");
                            for label in &restore {
                                if let Some(w) = windows.get(label) {
                                    reveal_onscreen(w);
                                }
                            }
                        } else {
                            tracing::debug!(?on_screen, "window toggle: park all");
                            for label in &on_screen {
                                if let Some(w) = windows.get(label) {
                                    park_offscreen(w);
                                }
                            }
                            if let Ok(mut h) = HIDDEN_WINDOWS.lock() {
                                *h = on_screen;
                            }
                        }
                    } else if shortcut == &debug_shortcut {
                        tracing::debug!("debug panel toggle");
                        _ = app.emit("toggle-debug-panel", ());
                    } else if shortcut == &quit_shortcut {
                        // Close the main window, which causes Tauri's run loop to
                        // return Ok(()) normally. `app.exit()` would call
                        // `std::process::exit()` which SKIPS local destructors,
                        // so the dhat profiler would never write its JSON.
                        if let Some(window) = app.get_webview_window("main") {
                            tracing::info!("quit shortcut pressed, closing main window");
                            if let Err(e) = window.close() {
                                tracing::error!(error = %e, "failed to close window");
                            }
                        }
                    } else if shortcut == &queue_shortcut {
                        tracing::debug!("queue-capture shortcut pressed");
                        _ = app.emit("trigger-queue-capture", ());
                    } else if shortcut == &describe_shortcut {
                        tracing::debug!("describe-queue shortcut pressed");
                        _ = app.emit("trigger-describe-queue", ());
                    } else if shortcut == &clip_shortcut {
                        tracing::debug!("clipboard-ingest shortcut pressed");
                        _ = app.emit("trigger-clipboard-ingest", ());
                    } else if shortcut == &agent_shortcut {
                        tracing::debug!("agent-query shortcut pressed");
                        _ = app.emit("trigger-agent-query", ());
                        // When the app is parked off-screen the press still runs —
                        // the hidden main webview drives `ask_agent` and mirrors the
                        // answer — but nothing is on screen to read it. Bring the
                        // answer pop-out on-screen (built non-focusable, so it never
                        // steals focus from the interview) carrying the streamed
                        // answer.
                        if is_hidden("main") {
                            if let Some(answer) = app.get_webview_window("answer") {
                                reveal_onscreen(&answer);
                            }
                        }
                    }
                })
                .build(),
        )
        .setup(move |app| {
            // Pre-create the pop-out answer overlay HIDDEN. Building webview
            // windows after the event loop is live hangs on Windows (wry left
            // the webview half-initialized: white window, build() never
            // returned) — setup-time creation is the reliable path, and it
            // makes the ⛶ button instant (show/hide from then on).
            tauri::WebviewWindowBuilder::new(
                app,
                "answer",
                tauri::WebviewUrl::App("answer.html".into()),
            )
            .title("gimme — answer")
            .inner_size(720.0, 560.0)
            // Created VISIBLE but parked off-screen — never hide()/show()n, so its
            // WDA_EXCLUDEFROMCAPTURE affinity never degrades to a black box
            // (tauri#14189). `reveal_onscreen` brings it in; `park_offscreen` sends
            // it back out.
            .position(f64::from(OFFSCREEN_XY), f64::from(OFFSCREEN_XY))
            .resizable(true)
            .transparent(true)
            .decorations(false)
            .always_on_top(true)
            .content_protected(true)
            .skip_taskbar(true)
            // Non-focusable (WS_EX_NOACTIVATE): appears without stealing keyboard
            // focus from the interview app. It's read-only — mouse drag/scroll/close
            // still work; it just never becomes the foreground window.
            .focusable(false)
            .build()?;
            // Mark it parked so Ctrl+Shift+H ignores it until it's first revealed;
            // `None` = no saved spot yet, so the first reveal centers it.
            PARKED
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert("answer".to_string(), None);
            tracing::info!("answer overlay pre-created (parked off-screen)");

            app.global_shortcut().register(toggle_shortcut)?;
            app.global_shortcut().register(debug_shortcut)?;
            app.global_shortcut().register(quit_shortcut)?;
            // The queue/describe bindings are shared with screen-peek; if that
            // overlay is running it owns them, so a failed registration must not
            // kill the app — warn and leave the UI buttons as the fallback.
            // Ctrl+Shift+Space rides along: some IMEs/apps hold it, and the 🤖
            // button covers a lost registration.
            for (label, sc) in [
                ("Ctrl+Shift+Enter", queue_shortcut),
                ("Ctrl+Shift+1", describe_shortcut),
                ("Ctrl+Shift+V", clip_shortcut),
                ("Ctrl+Shift+Space", agent_shortcut),
            ] {
                if let Err(e) = app.global_shortcut().register(sc) {
                    tracing::warn!(shortcut = label, error = %e, "shortcut registration failed (held by another app?)");
                }
            }
            tracing::info!(
                window_toggle = "Ctrl+Shift+H",
                debug_panel = "Ctrl+Shift+D",
                quit = "Ctrl+Shift+Q",
                queue_capture = "Ctrl+Shift+Enter",
                describe_queue = "Ctrl+Shift+1",
                clipboard_ingest = "Ctrl+Shift+V",
                agent_query = "Ctrl+Shift+Space",
                "global shortcuts registered"
            );

            // OS clipboard listener for auto-clip — its own thread; the message
            // loop blocks for the app's lifetime.
            clipboard::spawn_watcher(app.handle().clone(), auto_clip, manual_copy, last_clip);

            // Periodic metrics emitter: every 2s push a snapshot to the frontend
            // so the debug panel can refresh without polling. `setup` runs before
            // Tokio's reactor is available, so use Tauri's async runtime which
            // works regardless of context.
            //
            // When the `counting-alloc` feature is active, also refresh the heap
            // counters into the Metrics struct so the snapshot carries them.
            let emitter_app: tauri::AppHandle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                #[cfg(feature = "counting-alloc")]
                let mut ticks: u64 = 0;
                loop {
                    interval.tick().await;

                    #[cfg(feature = "counting-alloc")]
                    {
                        use std::sync::atomic::Ordering;
                        ticks += 1;

                        // Refresh scalar counters every tick (2s) so the debug
                        // panel stays snappy.
                        metrics_for_emitter
                            .heap_live_bytes
                            .store(alloc_counter::live_bytes(), Ordering::Relaxed);
                        metrics_for_emitter
                            .heap_total_allocated_bytes
                            .store(alloc_counter::total_allocated(), Ordering::Relaxed);
                        metrics_for_emitter
                            .heap_peak_live_bytes
                            .store(alloc_counter::peak_live_bytes(), Ordering::Relaxed);

                        // Refresh the 16-bucket histograms. Cheap — 32 atomic
                        // stores per tick. The frontend doesn't render these
                        // (too many columns) but the JSONL log picks them up
                        // on the 10s cadence below.
                        let live_buckets = alloc_counter::live_counts_by_bucket();
                        let total_buckets = alloc_counter::total_allocs_by_bucket();
                        for i in 0..live_buckets.len() {
                            metrics_for_emitter.heap_buckets_live[i]
                                .store(live_buckets[i], Ordering::Relaxed);
                            metrics_for_emitter.heap_buckets_total[i]
                                .store(total_buckets[i], Ordering::Relaxed);
                        }

                        // Log the full histogram to JSONL every 5th tick
                        // (~10s) so heap evolution is recorded even when the
                        // debug panel isn't open. Serializing the arrays as
                        // bracketed strings keeps the JSONL line readable.
                        if ticks.is_multiple_of(5) {
                            tracing::info!(
                                target: "heap",
                                live_bytes = alloc_counter::live_bytes(),
                                peak_live_bytes = alloc_counter::peak_live_bytes(),
                                total_allocated_bytes = alloc_counter::total_allocated(),
                                total_deallocated_bytes = alloc_counter::total_deallocated(),
                                live_buckets = ?live_buckets,
                                total_buckets = ?total_buckets,
                                buckets_labels = ?alloc_counter::BUCKET_LABELS,
                                allocator = metrics::active_allocator_name(),
                                "heap report"
                            );
                        }
                    }

                    let snapshot = metrics_for_emitter.snapshot();
                    _ = emitter_app.emit("metrics", snapshot);
                }
            });
            Ok(())
        })
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            commands::list_audio_devices,
            commands::list_output_devices,
            commands::start_listening,
            commands::stop_listening,
            commands::ask_brain,
            commands::ask_agent,
            commands::queue_capture,
            commands::describe_queue,
            commands::clear_capture_queue,
            commands::capture_queue_len,
            commands::copy_and_ingest,
            commands::clear_clipboard,
            commands::set_auto_clip,
            commands::set_vision_model,
            commands::get_vision_model,
            commands::set_brain_model,
            commands::get_brain_model,
            commands::set_language,
            commands::get_language,
            commands::simulate_interviewer,
            commands::log_from_frontend,
            commands::open_answer_window,
        ])
        .on_window_event(|window, event| {
            // The answer overlay is pre-created once and reused: closing it
            // (✕, Alt+F4) only hides it, so ⛶ can reveal it again.
            if window.label() == "answer" {
                if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    // Park off-screen (not hide()) so re-revealing it never shows a
                    // black box — same reason as the toggle key (tauri#14189).
                    if let Some(w) = window.app_handle().get_webview_window("answer") {
                        park_offscreen(&w);
                    }
                }
                return;
            }
            // The app's lifetime is the MAIN window's. Auxiliary windows must
            // not keep the process — and the audio pipelines and global
            // shortcuts with it — alive after it closes. destroy() (not
            // close()) so the hide-on-close handler above can't intercept.
            if window.label() == "main" && matches!(event, tauri::WindowEvent::Destroyed) {
                for (label, w) in window.app_handle().webview_windows() {
                    if label != "main" {
                        tracing::info!(%label, "destroying auxiliary window with main");
                        if let Err(e) = w.destroy() {
                            tracing::warn!(error = %e, %label, "failed to destroy auxiliary window");
                        }
                    }
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app_handle, event| {
            // This runs BEFORE winit/tao calls std::process::exit() (which
            // skips destructors) — the last chance to flush anything to disk.
            if matches!(event, tauri::RunEvent::Exit) {
                // Drain the session log and stamp the session's `ended_at`.
                storage::shutdown();
                // Drop the dhat profiler explicitly so its JSON gets written.
                #[cfg(feature = "dhat-heap")]
                if let Some(mu) = DHAT_PROFILER.get() {
                    if let Ok(mut guard) = mu.lock() {
                        if let Some(profiler) = guard.take() {
                            tracing::info!("dropping dhat profiler, writing JSON");
                            drop(profiler);
                        }
                    }
                }
            }
        });
}
