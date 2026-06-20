// The `#[tauri::command]` macro expands into code that binds the returned
// `Result<(), _>` with `let _ = ...`, which trips `let_underscore_must_use`.
// The ignore is intentional on tauri's side (IPC ACK), so we allow it here.
#![allow(clippy::let_underscore_must_use)]

use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde::Deserialize;

use crate::audio;
use crate::backend::Mode;
use crate::error::AppError;
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

    // Shared local whisper model (preloaded at startup; built lazily on first use
    // if the preload hasn't finished). One instance serves both dual pipelines.
    let whisper = if let Some(w) = state.whisper.get() {
        Arc::clone(w)
    } else {
        let w = Arc::new(
            crate::transcriber::WhisperTranscriber::new()
                .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?,
        );
        let _ = state.whisper.set(Arc::clone(&w));
        w
    };

    // STT engine: GIMME_STT_ENGINE = "whisper" (force local) | "sherpa"
    // (on-device Parakeet, per chunk) | "streaming" (on-device Nemotron with
    // live partials) | unset → Groq cloud. The on-device engines need
    // --features sherpa + fetched models.
    let stt_pref = std::env::var("GIMME_STT_ENGINE").unwrap_or_default();
    let engine = match stt_pref.as_str() {
        "whisper" => audio::SttEngine::LocalWhisper,
        "sherpa" | "parakeet" => sherpa_engine_or_default(),
        "streaming" | "online" => streaming_engine_or_default(),
        _ => default_stt_engine(),
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
                None,
                None,
            );
        }
    }

    Ok(())
}

/// Default chain: Groq cloud Whisper when a key is present, else local whisper.
fn default_stt_engine() -> audio::SttEngine {
    crate::cloud_stt::GroqStt::new()
        .map(Arc::new)
        .map_or(audio::SttEngine::LocalWhisper, audio::SttEngine::Groq)
}

/// `GIMME_STT_ENGINE=sherpa`: on-device Parakeet if this build carries the
/// `sherpa` feature and the models are fetched; otherwise warn and use the
/// default chain rather than dying mid-setup.
fn sherpa_engine_or_default() -> audio::SttEngine {
    #[cfg(feature = "sherpa")]
    {
        if let Some(p) = crate::stt::parakeet() {
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
    default_stt_engine()
}

/// `GIMME_STT_ENGINE=streaming`: hybrid on-device engine — live partials from
/// the light online model, finals re-decoded with Parakeet; same
/// degrade-to-default contract as the Parakeet path.
fn streaming_engine_or_default() -> audio::SttEngine {
    #[cfg(feature = "sherpa")]
    {
        if let Some(s) = crate::stt::streaming() {
            // Warm Parakeet now so the first endpoint doesn't pay the model
            // load; if it's missing, finals fall back to the online hypothesis.
            if crate::stt::parakeet().is_none() {
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
    default_stt_engine()
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

#[tauri::command]
#[tracing::instrument(skip(state, app, context), fields(trace_id = trace_id.as_deref().unwrap_or("-"), question_len = question.len(), context_len = context.len()))]
pub async fn ask_claude(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
    trace_id: Option<String>,
    question: String,
    context: String,
) -> Result<String, AppError> {
    let metrics = Arc::clone(&state.metrics);
    let trace_id = trace_id.unwrap_or_else(|| "-".into());
    let mode = state
        .mode
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;

    match mode {
        Mode::ClaudeCode => {
            let session = state
                .claude
                .get()
                .ok_or_else(|| AppError::Claude("claude session not ready yet".into()))?;
            let outcome = session
                .ask(&question, &context, &trace_id)
                .await
                .map_err(|e| AppError::Claude(e.to_string()))?;
            metrics
                .last_llm_ms
                .store(outcome.total_ms, Ordering::Relaxed);
            metrics
                .last_llm_ttft_ms
                .store(outcome.ttft_ms, Ordering::Relaxed);
            metrics
                .last_llm_cache_read_tokens
                .store(outcome.cache_read_tokens, Ordering::Relaxed);
            metrics
                .last_llm_cache_creation_tokens
                .store(outcome.cache_creation_tokens, Ordering::Relaxed);
            metrics.last_llm_spawn_ms.store(0, Ordering::Relaxed);
            if outcome.tool_use {
                metrics.llm_tool_use_count.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(%trace_id, "model leaked a tool_use despite the no-tools guardrail");
            }
            set_provider(&state, "claude-code");
            Ok(outcome.answer)
        }
        Mode::Api => {
            let outcome = state
                .api
                .ask(&question, &context, &trace_id, &app)
                .await
                .map_err(|e| AppError::Claude(e.to_string()))?;
            metrics
                .last_llm_ms
                .store(outcome.total_ms, Ordering::Relaxed);
            metrics
                .last_llm_ttft_ms
                .store(outcome.ttft_ms, Ordering::Relaxed);
            // API providers don't report prompt-cache tokens; reset the CLI-only stats.
            metrics
                .last_llm_cache_read_tokens
                .store(0, Ordering::Relaxed);
            metrics
                .last_llm_cache_creation_tokens
                .store(0, Ordering::Relaxed);
            metrics.last_llm_spawn_ms.store(0, Ordering::Relaxed);
            set_provider(&state, &outcome.provider);
            Ok(outcome.answer)
        }
    }
}

/// Switch the answering mode at runtime (`"api"` or `"claude_code"`).
// Tauri commands take `State` and deserialized args by value — that's the macro
// contract, not a needless move.
#[allow(clippy::needless_pass_by_value)]
#[tauri::command]
#[tracing::instrument(skip(state))]
pub fn set_mode(state: tauri::State<'_, AppState>, mode: String) -> Result<(), AppError> {
    let new_mode = match mode.as_str() {
        "api" => Mode::Api,
        "claude_code" => Mode::ClaudeCode,
        other => return Err(AppError::Other(anyhow::anyhow!("unknown mode: {other}"))),
    };
    let mut guard = state
        .mode
        .lock()
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    *guard = new_mode;
    tracing::info!(?new_mode, "answering mode switched");
    Ok(())
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
    let outcome = state
        .tts
        .synthesize_and_save(&text)
        .await
        .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
    crate::tts::play_file(&outcome.wav_path);
    tracing::info!(provider = %outcome.provider, gen_ms = outcome.gen_ms, "simulating interviewer");
    Ok(outcome.wav_path.to_string_lossy().into_owned())
}

/// Current answering mode as a string (for the UI to reflect on load).
#[allow(clippy::needless_pass_by_value)]
#[tauri::command]
pub fn get_mode(state: tauri::State<'_, AppState>) -> Result<String, AppError> {
    let mode = state
        .mode
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    Ok(match mode {
        Mode::Api => "api".into(),
        Mode::ClaudeCode => "claude_code".into(),
    })
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
