#[cfg(feature = "sherpa")]
mod aec;
#[cfg(feature = "counting-alloc")]
mod alloc_counter;
mod audio;
mod backend;
mod claude;
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
#[cfg(feature = "sherpa")]
mod stt;
mod telemetry;
mod transcriber;
mod tts;
mod vad;

use std::sync::{Arc, Mutex, OnceLock};

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
    pub transcript: Arc<Mutex<String>>,
    pub metrics: Arc<metrics::Metrics>,
    /// The persistent Claude session. Constructed in `setup` (needs the `AppHandle`),
    /// so it starts empty and is filled once at startup.
    pub claude: Arc<OnceLock<claude::ClaudeSession>>,
    /// Direct-API backend (Groq → `OpenAI` fallback chain). Built at startup.
    pub api: Arc<backend::ApiBackend>,
    /// Which backend answers `ask_claude`. Toggled from the UI; default = API.
    pub mode: Arc<Mutex<backend::Mode>>,
    /// Transcription + answer language. Toggled from the UI; default = English.
    /// Read at `start_listening` (STT engine) and `ask_claude` (prompt) time.
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
            transcript: Arc::new(Mutex::new(String::new())),
            metrics: Arc::new(metrics::Metrics::default()),
            claude: Arc::new(OnceLock::new()),
            api: Arc::new(backend::ApiBackend::new()),
            mode: Arc::new(Mutex::new(backend::Mode::Api)),
            language: Arc::new(Mutex::new(lang::Language::default())),
            tts: Arc::new(tts::TtsEngine::new()),
            whisper: Arc::new(OnceLock::new()),
            whisper_es: Arc::new(OnceLock::new()),
        }
    }
}

/// Boot the persistent Claude session at app startup (not lazily on the first
/// question). Spawning the process alone isn't enough — the prompt cache is only
/// written when we actually send a message — so we fire a discarded `warmup`
/// immediately. Claude Code caches its (huge) system+tools prefix with a 1-HOUR
/// ephemeral TTL that refreshes on each read, so a gentle heartbeat keeps it warm
/// for an arbitrarily long interview. Real questions then always hit a warm cache.
fn start_claude_session(app: &tauri::AppHandle, cell: &Arc<OnceLock<claude::ClaudeSession>>) {
    let session = claude::ClaudeSession::spawn(app.clone());
    _ = cell.set(session.clone());
    tauri::async_runtime::spawn(async move {
        match session.warmup().await {
            Ok(o) => tracing::info!(
                ttft_ms = o.ttft_ms,
                total_ms = o.total_ms,
                cache_creation_tokens = o.cache_creation_tokens,
                "claude warmup complete (session ready)"
            ),
            Err(e) => tracing::warn!(error = %e, "claude warmup failed"),
        }

        // Heartbeat: keep the 1-hour prompt cache warm. 30min < 60min TTL, and a
        // cache read refreshes the TTL, so this keeps it alive indefinitely while
        // barely touching the 5-hour subscription usage limit.
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(1800));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        interval.tick().await; // consume the immediate first tick (warmup already ran)
        loop {
            interval.tick().await;
            match session.warmup().await {
                Ok(o) => tracing::debug!(
                    cache_read_tokens = o.cache_read_tokens,
                    cache_creation_tokens = o.cache_creation_tokens,
                    total_ms = o.total_ms,
                    "claude heartbeat (cache_read>0 ⇒ cache stayed warm)"
                ),
                Err(e) => tracing::warn!(error = %e, "claude heartbeat failed"),
            }
        }
    });
}

// `run` is a long-by-nature Tauri builder: shortcut registration, the async
// metrics emitter (with a sizeable counting-alloc branch), and app wiring all
// live here. Session startup was already extracted to `start_claude_session`.
#[allow(clippy::too_many_lines)]
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
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

    // Ctrl+Shift+H toggles the overlay's visibility (same binding as screen-peek).
    let toggle_shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyH);
    let debug_shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyD);
    // Graceful quit: Tauri's run loop returns cleanly, letting local guards like
    // the dhat profiler drop properly and write their output. Killing via Ctrl+C
    // in the terminal would skip destructors.
    let quit_shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyQ);

    let app_state = AppState::default();
    let metrics_for_emitter = Arc::clone(&app_state.metrics);
    let claude_cell = Arc::clone(&app_state.claude);
    let whisper_cell = Arc::clone(&app_state.whisper);

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
                        if let Some(window) = app.get_webview_window("main") {
                            let is_visible = window.is_visible().unwrap_or(false);
                            tracing::debug!(
                                was_visible = is_visible,
                                action = if is_visible { "hide" } else { "show" },
                                "window toggle"
                            );
                            let result = if is_visible {
                                window.hide()
                            } else {
                                window.show().and_then(|()| window.set_focus())
                            };
                            if let Err(e) = result {
                                tracing::warn!(error = %e, "window toggle failed");
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
                    }
                })
                .build(),
        )
        .setup(move |app| {
            app.global_shortcut().register(toggle_shortcut)?;
            app.global_shortcut().register(debug_shortcut)?;
            app.global_shortcut().register(quit_shortcut)?;
            tracing::info!(
                window_toggle = "Ctrl+Shift+H",
                debug_panel = "Ctrl+Shift+D",
                quit = "Ctrl+Shift+Q",
                "global shortcuts registered"
            );

            // Boot the persistent Claude session NOW (app startup), not lazily on
            // the first question — see `start_claude_session` for the why.
            start_claude_session(app.handle(), &claude_cell);

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
            commands::ask_claude,
            commands::set_mode,
            commands::get_mode,
            commands::set_language,
            commands::get_language,
            commands::simulate_interviewer,
            commands::log_from_frontend,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app_handle, event| {
            // When Tauri signals the process is about to exit, drop the dhat
            // profiler explicitly so its JSON gets written. This runs BEFORE
            // winit/tao calls std::process::exit(), giving us our last chance.
            #[cfg(feature = "dhat-heap")]
            if matches!(event, tauri::RunEvent::Exit) {
                if let Some(mu) = DHAT_PROFILER.get() {
                    if let Ok(mut guard) = mu.lock() {
                        if let Some(profiler) = guard.take() {
                            tracing::info!("dropping dhat profiler, writing JSON");
                            drop(profiler);
                        }
                    }
                }
            }
            #[cfg(not(feature = "dhat-heap"))]
            let _ = event;
        });
}
