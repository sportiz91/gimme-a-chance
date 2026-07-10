// The `#[tauri::command]` macro expands into code that binds the returned
// `Result<(), _>` with `let _ = ...`, which trips `let_underscore_must_use`.
// The ignore is intentional on tauri's side (IPC ACK), so we allow it here.
#![allow(clippy::let_underscore_must_use)]

use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::Emitter;

use crate::audio;
use crate::error::AppError;
use crate::storage;
use crate::AppState;

#[tauri::command]
#[tracing::instrument]
pub fn list_audio_devices() -> Result<Vec<audio::DeviceInfo>, AppError> {
    audio::list_input_devices().map_err(|e| AppError::Audio(e.to_string()))
}

/// Output (render) devices — the loopback sources for capturing the interviewer.
#[tauri::command]
#[tracing::instrument]
pub fn list_output_devices() -> Result<Vec<audio::DeviceInfo>, AppError> {
    audio::list_output_devices().map_err(|e| AppError::Audio(e.to_string()))
}

#[tauri::command]
#[tracing::instrument(skip(state, app), fields(device = device_name.as_deref().unwrap_or("default"), source = source.as_deref().unwrap_or("mic")))]
pub async fn start_listening(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
    device_name: Option<String>,
    source: Option<String>,
) -> Result<(), AppError> {
    {
        let mut is_listening = state
            .is_listening
            .lock()
            .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
        if *is_listening {
            return Err(AppError::Audio("Already listening".into()));
        }
        *is_listening = true;
    }

    let is_listening = Arc::clone(&state.is_listening);
    let metrics = Arc::clone(&state.metrics);

    // The language the UI selected (default English). Drives the on-device model
    // set, the cloud/local Whisper language param, and (downstream) the answer
    // prompts. Read once per Listen, so switching language takes effect on the
    // next Listen.
    let language = state
        .language
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;

    // Local whisper model for this language — the offline STT fallback. English
    // (`base.en`) is preloaded at startup; Spanish (multilingual `base`) is built
    // lazily on the first Spanish Listen and cached. One instance serves both dual
    // pipelines.
    let whisper_cell = match language {
        crate::lang::Language::English => &state.whisper,
        crate::lang::Language::Spanish => &state.whisper_es,
    };
    let whisper = if let Some(w) = whisper_cell.get() {
        Arc::clone(w)
    } else {
        let w = Arc::new(
            crate::transcriber::WhisperTranscriber::new(language)
                .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?,
        );
        let _ = whisper_cell.set(Arc::clone(&w));
        w
    };

    // STT engine: GIMME_STT_ENGINE = "whisper" (force local) | "sherpa"
    // (on-device Parakeet, per chunk) | "streaming" (on-device hybrid with
    // live partials) | unset → Groq cloud. The on-device engines need
    // --features sherpa + fetched models. All resolve the model set for `language`.
    let stt_pref = std::env::var("GIMME_STT_ENGINE").unwrap_or_default();
    let engine = match stt_pref.as_str() {
        "whisper" => audio::SttEngine::LocalWhisper,
        "sherpa" | "parakeet" => sherpa_engine_or_default(language),
        "streaming" | "online" => streaming_engine_or_default(language),
        _ => default_stt_engine(language),
    };

    // "both" → dual capture: loopback (interviewer) + mic (you), both transcribed
    // and labeled. Otherwise a single source.
    match source.as_deref() {
        Some("both") => {
            // Shared bleed window (text-level dedup backstop) + AEC reference
            // channel (signal-level echo cancellation): the interviewer pipeline
            // feeds both, the mic pipeline consumes both to keep the
            // interviewer's headset bleed out of the [You] track. Single-source
            // capture below passes `None`.
            let bleed = audio::BleedWindow::default();
            let (aec_tx, aec_rx) = crossbeam_channel::bounded::<Vec<f32>>(64);
            spawn_pipeline(
                app.clone(),
                Arc::clone(&is_listening),
                Arc::clone(&metrics),
                None,
                audio::CaptureSource::Loopback,
                "interviewer",
                Arc::clone(&whisper),
                engine.clone(),
                language,
                Some(bleed.clone()),
                Some(audio::AecEnd::Reference(aec_tx)),
            );
            spawn_pipeline(
                app,
                is_listening,
                metrics,
                None,
                audio::CaptureSource::Microphone,
                "me",
                whisper,
                engine,
                language,
                Some(bleed),
                Some(audio::AecEnd::Canceller(aec_rx)),
            );
        }
        other => {
            let cs = audio::CaptureSource::from_opt(other);
            let speaker = match cs {
                audio::CaptureSource::Loopback => "interviewer",
                audio::CaptureSource::Microphone => "me",
            };
            spawn_pipeline(
                app,
                is_listening,
                metrics,
                device_name,
                cs,
                speaker,
                whisper,
                engine,
                language,
                None,
                None,
            );
        }
    }

    Ok(())
}

/// Default chain: Groq cloud Whisper when a key is present, else local whisper.
fn default_stt_engine(language: crate::lang::Language) -> audio::SttEngine {
    crate::cloud_stt::GroqStt::new(language)
        .map(Arc::new)
        .map_or(audio::SttEngine::LocalWhisper, audio::SttEngine::Groq)
}

/// `GIMME_STT_ENGINE=sherpa`: on-device Parakeet (for `language`) if this build
/// carries the `sherpa` feature and the models are fetched; otherwise warn and use
/// the default chain rather than dying mid-setup.
fn sherpa_engine_or_default(language: crate::lang::Language) -> audio::SttEngine {
    #[cfg(feature = "sherpa")]
    {
        if let Some(p) = crate::stt::parakeet(language) {
            return audio::SttEngine::Parakeet(p);
        }
        tracing::warn!(
            "GIMME_STT_ENGINE=sherpa but Parakeet could not load; using default STT engine"
        );
    }
    #[cfg(not(feature = "sherpa"))]
    tracing::warn!(
        "GIMME_STT_ENGINE=sherpa but this build lacks the `sherpa` feature; using default STT engine"
    );
    default_stt_engine(language)
}

/// `GIMME_STT_ENGINE=streaming`: hybrid on-device engine for `language` — live
/// partials from the light online model, finals re-decoded with Parakeet; same
/// degrade-to-default contract as the Parakeet path.
fn streaming_engine_or_default(language: crate::lang::Language) -> audio::SttEngine {
    #[cfg(feature = "sherpa")]
    {
        if let Some(s) = crate::stt::streaming(language) {
            // Warm Parakeet (for this language) now so the first endpoint doesn't
            // pay the model load; if it's missing, finals fall back to the online
            // hypothesis.
            if crate::stt::parakeet(language).is_none() {
                tracing::warn!(
                    "Parakeet unavailable — streaming finals will use the online hypothesis"
                );
            }
            return audio::SttEngine::Streaming(s);
        }
        tracing::warn!(
            "GIMME_STT_ENGINE=streaming but the streaming model could not load; using default STT engine"
        );
    }
    #[cfg(not(feature = "sherpa"))]
    tracing::warn!(
        "GIMME_STT_ENGINE=streaming but this build lacks the `sherpa` feature; using default STT engine"
    );
    default_stt_engine(language)
}

/// Show and focus the pop-out answer overlay (pre-created hidden at setup —
/// runtime `WebviewWindowBuilder::build` hangs on Windows, so the window is
/// never built here, only revealed). Async keeps it off the main thread.
#[allow(clippy::unused_async)]
#[tauri::command]
#[tracing::instrument(skip(app))]
pub async fn open_answer_window(app: tauri::AppHandle) -> Result<(), AppError> {
    use tauri::Manager;
    let w = app
        .get_webview_window("answer")
        .ok_or_else(|| AppError::Other(anyhow::anyhow!("answer window missing (setup failed?)")))?;
    w.show().map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
    let _ = w.set_focus();
    tracing::info!("answer overlay shown");
    Ok(())
}

/// Payload for the `stt-warmup` event, so the UI can show "warming models"
/// while heavy on-device models load in the background.
#[derive(Clone, Serialize)]
struct WarmupPayload {
    state: &'static str,
}

/// Pre-load every model `start_listening` would need for `language`, off the
/// calling thread, so the first Listen answers in milliseconds instead of
/// paying multi-second model loads. Safe to fire repeatedly: the sherpa
/// loaders' `OnceLock::get_or_init` serializes a concurrent Listen against
/// the warm-up, and the whisper cells at worst build a transient duplicate
/// that loses the `set` race.
fn warm_stt_models(app: tauri::AppHandle, state: &AppState, language: crate::lang::Language) {
    let whisper_cell = match language {
        crate::lang::Language::English => Arc::clone(&state.whisper),
        crate::lang::Language::Spanish => Arc::clone(&state.whisper_es),
    };
    std::thread::spawn(move || {
        app.emit("stt-warmup", WarmupPayload { state: "started" })
            .ok();
        let t0 = std::time::Instant::now();

        // On-device engines — with GIMME_STT_ENGINE=streaming these are what
        // the first Listen actually waits on.
        let stt_pref = std::env::var("GIMME_STT_ENGINE").unwrap_or_default();
        #[cfg(feature = "sherpa")]
        match stt_pref.as_str() {
            "sherpa" | "parakeet" => {
                let _ = crate::stt::parakeet(language);
            }
            "streaming" | "online" => {
                let _ = crate::stt::streaming(language);
                let _ = crate::stt::parakeet(language);
            }
            _ => {}
        }
        #[cfg(not(feature = "sherpa"))]
        drop(stt_pref);

        // Local whisper fallback for this language. English is preloaded at
        // startup already; Spanish builds here on the first Spanish session.
        if whisper_cell.get().is_none() {
            match crate::transcriber::WhisperTranscriber::new(language) {
                Ok(w) => {
                    let _ = whisper_cell.set(Arc::new(w));
                }
                Err(e) => tracing::warn!(error = %e, ?language, "whisper warm-up failed"),
            }
        }

        tracing::info!(?language, elapsed = ?t0.elapsed(), "stt warm-up done");
        app.emit("stt-warmup", WarmupPayload { state: "done" }).ok();
    });
}

/// Spawn one capture+transcribe pipeline on its own OS thread (`cpal::Stream` isn't
/// Send), with a single-threaded tokio runtime to drive the async STT calls.
#[allow(clippy::too_many_arguments)]
fn spawn_pipeline(
    app: tauri::AppHandle,
    is_listening: Arc<std::sync::Mutex<bool>>,
    metrics: Arc<crate::metrics::Metrics>,
    device_name: Option<String>,
    source: audio::CaptureSource,
    speaker: &'static str,
    whisper: Arc<crate::transcriber::WhisperTranscriber>,
    engine: audio::SttEngine,
    language: crate::lang::Language,
    bleed: Option<audio::BleedWindow>,
    aec: Option<audio::AecEnd>,
) {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime");
        rt.block_on(async {
            if let Err(e) = audio::capture_and_transcribe(
                app,
                is_listening,
                metrics,
                device_name,
                source,
                speaker,
                whisper,
                engine,
                language,
                bleed,
                aec,
            )
            .await
            {
                tracing::error!(error = %e, speaker, "audio pipeline error");
            }
        });
    });
}

#[tauri::command]
#[tracing::instrument(skip(state))]
pub async fn stop_listening(state: tauri::State<'_, AppState>) -> Result<(), AppError> {
    let mut is_listening = state
        .is_listening
        .lock()
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    *is_listening = false;
    Ok(())
}

/// Record which backend/provider answered the last turn (shown in the debug panel).
fn set_provider(state: &tauri::State<'_, AppState>, provider: &str) {
    if let Ok(mut p) = state.metrics.last_provider.lock() {
        provider.clone_into(&mut p);
    }
}

/// The Ask box: a question the user typed. The app never answers on its own —
/// this and `ask_agent` are the only two paths to an answer, both explicit.
#[tauri::command]
#[tracing::instrument(skip(state, app, context), fields(trace_id = trace_id.as_deref().unwrap_or("-"), question_len = question.len(), context_len = context.len()))]
pub async fn ask_brain(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
    trace_id: Option<String>,
    question: String,
    context: String,
) -> Result<String, AppError> {
    let metrics = Arc::clone(&state.metrics);
    let trace_id = trace_id.unwrap_or_else(|| "-".into());
    storage::record(storage::Event {
        kind: "question",
        speaker: None,
        content: question.clone(),
        t_s: state.agent.elapsed_s(),
        meta: Some(serde_json::json!({ "trace_id": trace_id })),
    });
    let language = state
        .language
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    let brain = state
        .brain_model
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;

    let outcome = state
        .api
        .ask(&question, &context, language, brain, &trace_id, &app)
        .await
        .map_err(|e| AppError::Llm(e.to_string()))?;
    metrics
        .last_llm_ms
        .store(outcome.total_ms, Ordering::Relaxed);
    metrics
        .last_llm_ttft_ms
        .store(outcome.ttft_ms, Ordering::Relaxed);
    set_provider(&state, &outcome.provider);
    storage::record(storage::Event {
        kind: "answer",
        speaker: None,
        content: outcome.answer.clone(),
        t_s: state.agent.elapsed_s(),
        meta: Some(serde_json::json!({
            "trace_id": trace_id,
            "trigger": "ask",
            "provider": outcome.provider,
            "question": question,
        })),
    });
    Ok(outcome.answer)
}

/// Agent press (Ctrl+Shift+Space): answer from the FULL rolling transcript +
/// Interview State, letting the model infer what help is needed — no question
/// heuristics involved. Streams `answer-delta` events like `ask_brain`.
#[tauri::command]
#[tracing::instrument(skip(state, app), fields(trace_id = trace_id.as_deref().unwrap_or("-")))]
pub async fn ask_agent(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
    trace_id: Option<String>,
) -> Result<String, AppError> {
    let trace_id = trace_id.unwrap_or_else(|| "-".into());
    let language = state
        .language
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    let brain = state
        .brain_model
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;

    let (transcript, transcript_lines) = state.agent.transcript_text();
    if transcript.is_empty() {
        return Err(AppError::Llm(
            "nothing transcribed yet — the agent has no context to work from".into(),
        ));
    }
    let state_block = state.agent.state_block();

    #[cfg(debug_assertions)]
    dump_agent_prompt(&trace_id, &transcript, &state_block);

    let outcome = state
        .api
        .ask_agent(&transcript, &state_block, language, brain, &trace_id, &app)
        .await
        .map_err(|e| AppError::Llm(e.to_string()))?;

    let m = &state.metrics;
    m.last_llm_ms.store(outcome.total_ms, Ordering::Relaxed);
    m.last_llm_ttft_ms.store(outcome.ttft_ms, Ordering::Relaxed);
    m.agent_prompt_tokens
        .store(outcome.usage.prompt, Ordering::Relaxed);
    m.agent_cached_tokens
        .store(outcome.usage.cached, Ordering::Relaxed);
    m.agent_completion_tokens
        .store(outcome.usage.completion, Ordering::Relaxed);
    set_provider(&state, &format!("agent/{}", outcome.model));

    // The one JSONL line that reconstructs a press: prompt shape, cache hit
    // rate (cached_tokens is THE health check that the append-only prompt
    // layout keeps paying), latency, and the full answer. Pair with the
    // debug-build prompt dump for byte-exact repro.
    tracing::info!(
        target: "agent",
        trace_id = %trace_id,
        model = outcome.model,
        transcript_lines,
        transcript_chars = transcript.len(),
        state_chars = state_block.len(),
        prompt_tokens = outcome.usage.prompt,
        cached_tokens = outcome.usage.cached,
        completion_tokens = outcome.usage.completion,
        ttft_ms = outcome.ttft_ms,
        total_ms = outcome.total_ms,
        request_id = outcome.request_id.as_deref().unwrap_or("-"),
        answer = %outcome.answer,
        "agent press answered"
    );

    storage::record(storage::Event {
        kind: "answer",
        speaker: None,
        content: outcome.answer.clone(),
        t_s: state.agent.elapsed_s(),
        meta: Some(serde_json::json!({
            "trace_id": trace_id,
            "trigger": "agent",
            "model": outcome.model,
            "prompt_tokens": outcome.usage.prompt,
            "cached_tokens": outcome.usage.cached,
            "completion_tokens": outcome.usage.completion,
        })),
    });

    #[cfg(debug_assertions)]
    append_agent_answer(&trace_id, &outcome.answer);

    // A press is a natural moment to catch a stale state up, in the background
    // — never blocking the answer (which already streamed).
    crate::agent::refresh_if_stale(&app);
    Ok(outcome.answer)
}

/// Debug builds only: persist the exact agent prompt (transcript + state) to
/// `logs/agent-prompts/`, so "why did it answer THIS?" can be answered offline
/// against the API with the very same input.
#[cfg(debug_assertions)]
fn dump_agent_prompt(trace_id: &str, transcript: &str, state_block: &str) {
    let dir = crate::telemetry::logs_dir().join("agent-prompts");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join(format!("{trace_id}.md"));
    let content = format!("# transcript\n\n{transcript}\n\n# interview state\n\n{state_block}\n");
    if std::fs::write(&path, content).is_ok() {
        tracing::info!(path = %path.display(), "agent prompt dumped (debug build)");
    }
}

/// Debug builds only: complete the dump with the answer, so each file under
/// `logs/agent-prompts/` holds the full input→output pair of one press.
#[cfg(debug_assertions)]
fn append_agent_answer(trace_id: &str, answer: &str) {
    use std::io::Write;
    let path = crate::telemetry::logs_dir()
        .join("agent-prompts")
        .join(format!("{trace_id}.md"));
    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&path) {
        _ = writeln!(f, "\n# answer\n\n{answer}");
    }
}

/// Generate an interviewer question with TTS (Kokoro→OpenAI), save it as a WAV,
/// log it to the JSONL, and play it through the default output device. Used by
/// the "Simulate interviewer" button to exercise the full capture→STT→LLM loop.
/// Returns the saved WAV path.
#[tauri::command]
#[tracing::instrument(skip(state), fields(chars = text.len()))]
pub async fn simulate_interviewer(
    state: tauri::State<'_, AppState>,
    text: String,
) -> Result<String, AppError> {
    let language = state
        .language
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    let outcome = state
        .tts
        .synthesize_and_save(&text, language)
        .await
        .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
    crate::tts::play_file(&outcome.wav_path);
    tracing::info!(provider = %outcome.provider, gen_ms = outcome.gen_ms, "simulating interviewer");
    Ok(outcome.wav_path.to_string_lossy().into_owned())
}

/// Switch the transcription + answer language at runtime (`"english"` or
/// `"spanish"`). Takes effect on the next Listen (STT engine rebuilt) and the next
/// answer (prompt rebuilt) — see `crate::lang`.
#[allow(clippy::needless_pass_by_value)]
#[tauri::command]
#[tracing::instrument(skip(state, app))]
pub fn set_language(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    language: String,
) -> Result<(), AppError> {
    let new_lang = crate::lang::Language::from_tag(&language)
        .ok_or_else(|| AppError::Other(anyhow::anyhow!("unknown language: {language}")))?;
    {
        let mut guard = state
            .language
            .lock()
            .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
        *guard = new_lang;
    }
    tracing::info!(?new_lang, "transcription/answer language switched");
    // Warm this language's models in the background so the next Listen is
    // instant. Fires on startup too — the UI pushes the persisted language
    // as soon as it loads (initLanguage).
    warm_stt_models(app, &state, new_lang);
    Ok(())
}

/// Current language as a string (for the UI to reflect on load).
#[allow(clippy::needless_pass_by_value)]
#[tauri::command]
pub fn get_language(state: tauri::State<'_, AppState>) -> Result<String, AppError> {
    let language = state
        .language
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    Ok(language.tag().to_string())
}

/// Cap on queued screenshots. Ten covers a very long page; past that the user
/// almost certainly forgot the queue was filling up.
const MAX_QUEUED_SHOTS: usize = 10;

/// Capture the primary monitor off the async runtime (xcap grabs synchronously).
async fn capture_screen_blocking() -> Result<String, AppError> {
    tauri::async_runtime::spawn_blocking(crate::capture::capture_primary_jpeg_base64)
        .await
        .map_err(|e| AppError::Vision(format!("capture task join: {e}")))?
        .map_err(|e| AppError::Vision(e.to_string()))
}

/// Capture with the overlays hidden. `contentProtected` makes DXGI duplication
/// render our windows as BLACK RECTANGLES over everything beneath them — they
/// covered most of every shot (the model couldn't see the page behind it, and
/// a large redacted region is itself a refusal magnet for gpt-4o-mini).
/// Hide every visible app window (main + answer pop-out) → capture → restore
/// exactly those. Costs a ~200ms flicker of the overlays.
async fn capture_screen_hiding_overlay(app: &tauri::AppHandle) -> Result<String, AppError> {
    use tauri::Manager;
    let hidden: Vec<_> = app
        .webview_windows()
        .into_values()
        .filter(|w| w.is_visible().unwrap_or(false))
        .collect();
    for w in &hidden {
        if let Err(e) = w.hide() {
            tracing::warn!(error = %e, label = w.label(), "could not hide overlay for capture");
        }
    }
    if !hidden.is_empty() {
        // A beat for the compositor to actually drop the windows.
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
    }
    let img = capture_screen_blocking().await;
    for w in &hidden {
        if let Err(e) = w.show() {
            tracing::warn!(error = %e, label = w.label(), "could not re-show overlay after capture");
        }
    }
    img
}

/// Capture the primary monitor onto the screenshot queue (Ctrl+Shift+Enter),
/// for a later multi-shot describe. Returns the queue length (the UI badge).
#[tauri::command]
#[tracing::instrument(skip(state, app))]
pub async fn queue_capture(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<usize, AppError> {
    {
        let q = state
            .capture_queue
            .lock()
            .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
        if q.len() >= MAX_QUEUED_SHOTS {
            return Err(AppError::Vision(format!(
                "screenshot queue is full ({MAX_QUEUED_SHOTS}); describe or clear it"
            )));
        }
    }
    let img = capture_screen_hiding_overlay(&app).await?;
    let mut q = state
        .capture_queue
        .lock()
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    q.push(img);
    tracing::info!(queued = q.len(), "screenshot queued");
    Ok(q.len())
}

/// Describe the queued screenshots (capture order = scroll order) in ONE vision
/// call (Ctrl+Shift+1). An empty queue is the one-key path: capture the screen
/// right now and describe just that. Streams `vision-delta` events; returns
/// (and stores) the full description text.
#[tauri::command]
#[tracing::instrument(skip(state, app), fields(trace_id = trace_id.as_deref().unwrap_or("-")))]
pub async fn describe_queue(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
    trace_id: Option<String>,
) -> Result<String, AppError> {
    let trace_id = trace_id.unwrap_or_else(|| "-".into());
    let language = state
        .language
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    let vision_model = state
        .vision_model
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;

    let mut imgs = {
        let mut q = state
            .capture_queue
            .lock()
            .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
        std::mem::take(&mut *q)
    };
    if imgs.is_empty() {
        imgs.push(capture_screen_hiding_overlay(&app).await?);
    }

    #[cfg(debug_assertions)]
    dump_captures(&trace_id, &imgs);

    let mut outcome = match state
        .api
        .describe(&imgs, vision_model, language, &trace_id, &app)
        .await
    {
        Ok(o) => o,
        Err(e) => {
            // Put the shots back so a transient API failure doesn't cost the
            // user their scrolled captures (the UI re-syncs the badge).
            if let Ok(mut q) = state.capture_queue.lock() {
                if q.is_empty() {
                    *q = imgs;
                }
            }
            return Err(AppError::Vision(e.to_string()));
        }
    };

    // gpt-4o-mini deterministically refused some multi-shot desktop captures
    // that gpt-5.5 handled fine (measured 2/2 vs 0/2 on dumped shots). Rather
    // than hand the user a refusal, burn the extra latency and retry once with
    // gpt-5.5; if the retry also fails, keep the original text.
    if crate::backend::looks_like_refusal(&outcome.text)
        && !matches!(vision_model, crate::backend::VisionModel::Gpt55)
    {
        tracing::warn!(text = %outcome.text, "describe refused; retrying once with gpt-5.5");
        match state
            .api
            .describe(
                &imgs,
                crate::backend::VisionModel::Gpt55,
                language,
                &trace_id,
                &app,
            )
            .await
        {
            Ok(o) => outcome = o,
            Err(e) => tracing::warn!(error = %e, "gpt-5.5 retry failed; keeping the refusal text"),
        }
    }

    if let Ok(mut d) = state.last_description.lock() {
        d.clone_from(&outcome.text);
    }
    crate::agent::push_line(&app, "screen", &outcome.text);
    state
        .metrics
        .last_vision_ms
        .store(outcome.total_ms, Ordering::Relaxed);
    Ok(outcome.text)
}

/// Debug builds only: persist the exact shots sent to a describe under
/// `logs/captures/`, so a refusal can be reproduced offline against the API
/// with the very same images.
#[cfg(debug_assertions)]
fn dump_captures(trace_id: &str, imgs: &[String]) {
    use base64::Engine;
    let dir = crate::telemetry::logs_dir().join("captures");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    for (i, b64) in imgs.iter().enumerate() {
        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
            let path = dir.join(format!("{trace_id}-{i}.jpg"));
            if std::fs::write(&path, bytes).is_ok() {
                tracing::info!(path = %path.display(), "capture dumped (debug build)");
            }
        }
    }
}

/// Drop all queued screenshots (the 🗑 button).
#[allow(clippy::needless_pass_by_value)]
#[tauri::command]
#[tracing::instrument(skip(state))]
pub fn clear_capture_queue(state: tauri::State<'_, AppState>) -> Result<(), AppError> {
    let mut q = state
        .capture_queue
        .lock()
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    let dropped = q.len();
    q.clear();
    tracing::info!(dropped, "screenshot queue cleared");
    Ok(())
}

/// Current queue length — lets the UI badge re-sync after a reload (the queue
/// lives in the backend and survives frontend hot reloads).
#[allow(clippy::needless_pass_by_value)]
#[tauri::command]
pub fn capture_queue_len(state: tauri::State<'_, AppState>) -> Result<usize, AppError> {
    let q = state
        .capture_queue
        .lock()
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    Ok(q.len())
}

/// Switch the vision (screen-describing) model (`"gpt_4o_mini"` | `"gpt_5_5"`).
#[allow(clippy::needless_pass_by_value)]
#[tauri::command]
#[tracing::instrument(skip(state))]
pub fn set_vision_model(state: tauri::State<'_, AppState>, model: String) -> Result<(), AppError> {
    let new_model = crate::backend::VisionModel::from_tag(&model)
        .ok_or_else(|| AppError::Other(anyhow::anyhow!("unknown vision model: {model}")))?;
    let mut guard = state
        .vision_model
        .lock()
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    *guard = new_model;
    tracing::info!(?new_model, "vision model switched");
    Ok(())
}

/// Current vision model tag (for the UI to reflect on load).
#[allow(clippy::needless_pass_by_value)]
#[tauri::command]
pub fn get_vision_model(state: tauri::State<'_, AppState>) -> Result<String, AppError> {
    let m = state
        .vision_model
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    Ok(m.tag().to_string())
}

/// Switch the brain (answering) model (`"auto"` | `"gpt_4o_mini"` | `"gpt_5_5"`).
#[allow(clippy::needless_pass_by_value)]
#[tauri::command]
#[tracing::instrument(skip(state))]
pub fn set_brain_model(state: tauri::State<'_, AppState>, model: String) -> Result<(), AppError> {
    let new_model = crate::backend::BrainModel::from_tag(&model)
        .ok_or_else(|| AppError::Other(anyhow::anyhow!("unknown brain model: {model}")))?;
    let mut guard = state
        .brain_model
        .lock()
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    *guard = new_model;
    tracing::info!(?new_model, "brain model switched");
    Ok(())
}

/// Current brain model tag (for the UI to reflect on load).
#[allow(clippy::needless_pass_by_value)]
#[tauri::command]
pub fn get_brain_model(state: tauri::State<'_, AppState>) -> Result<String, AppError> {
    let m = state
        .brain_model
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    Ok(m.tag().to_string())
}

/// Copy the current selection (synthetic Ctrl+C to the focused app) and ingest
/// the resulting clipboard text (Ctrl+Shift+V, the manual path). The watcher is
/// suppressed for the duration so the copy isn't ingested twice; no dedup —
/// re-ingesting the same clip on purpose must work — but the dedup state is
/// updated so a later auto-clip of the same text stays quiet.
#[tauri::command]
#[tracing::instrument(skip(state, app))]
pub async fn copy_and_ingest(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<String, AppError> {
    state.manual_copy.store(true, Ordering::SeqCst);
    let result =
        tauri::async_runtime::spawn_blocking(crate::clipboard::copy_selection_and_read).await;
    state.manual_copy.store(false, Ordering::SeqCst);
    let text = result
        .map_err(|e| AppError::Clipboard(format!("clipboard task join: {e}")))?
        .map_err(|e| AppError::Clipboard(e.to_string()))?;
    if text.is_empty() {
        return Err(AppError::Clipboard(
            "nothing selected (clipboard has no text)".into(),
        ));
    }
    if let Ok(mut last) = state.last_clip.lock() {
        last.clone_from(&text);
    }
    crate::agent::push_line_tagged(&app, "clipboard", "clipboard_stealth", &text);
    tracing::info!(chars = text.len(), "clipboard ingested (manual copy)");
    Ok(text)
}

/// Forget the last ingested clip (the Clipboard 🗑 button). The frontend drops
/// the clip from its box and context; resetting the dedup here lets the same
/// text be copied and ingested again afterwards.
#[allow(clippy::needless_pass_by_value)]
#[tauri::command]
#[tracing::instrument(skip(state))]
pub fn clear_clipboard(state: tauri::State<'_, AppState>) -> Result<(), AppError> {
    let mut last = state
        .last_clip
        .lock()
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    last.clear();
    tracing::info!("clipboard context cleared");
    Ok(())
}

/// Toggle the auto-clip monitor (ingest every OS copy) — the UI checkbox.
#[allow(clippy::needless_pass_by_value)]
#[tauri::command]
#[tracing::instrument(skip(state))]
pub fn set_auto_clip(state: tauri::State<'_, AppState>, enabled: bool) -> Result<(), AppError> {
    state.auto_clip.store(enabled, Ordering::Relaxed);
    tracing::info!(enabled, "auto-clip toggled");
    Ok(())
}

/// Structured log entry forwarded from the frontend so that JS timings
/// and events land in the same JSONL file as Rust logs.
#[derive(Deserialize)]
pub struct FrontendLogEntry {
    pub level: String,
    pub event: String,
    pub trace_id: Option<String>,
    pub elapsed_ms: Option<u64>,
    pub data: Option<serde_json::Value>,
}

#[tauri::command]
pub fn log_from_frontend(entry: FrontendLogEntry) {
    let trace_id = entry.trace_id.unwrap_or_else(|| "-".into());
    let elapsed_ms = entry.elapsed_ms.unwrap_or(0);
    let data = entry.data.map_or_else(|| "{}".into(), |v| v.to_string());

    match entry.level.as_str() {
        "error" => tracing::error!(
            target: "frontend",
            trace_id = %trace_id,
            elapsed_ms,
            data = %data,
            "{}",
            entry.event
        ),
        "warn" => tracing::warn!(
            target: "frontend",
            trace_id = %trace_id,
            elapsed_ms,
            data = %data,
            "{}",
            entry.event
        ),
        "debug" => tracing::debug!(
            target: "frontend",
            trace_id = %trace_id,
            elapsed_ms,
            data = %data,
            "{}",
            entry.event
        ),
        _ => tracing::info!(
            target: "frontend",
            trace_id = %trace_id,
            elapsed_ms,
            data = %data,
            "{}",
            entry.event
        ),
    }
}
