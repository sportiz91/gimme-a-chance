// The `#[tauri::command]` macro expands into code that binds the returned
// `Result<(), _>` with `let _ = ...`, which trips `let_underscore_must_use`.
// The ignore is intentional on tauri's side (IPC ACK), so we allow it here.
#![allow(clippy::let_underscore_must_use)]

use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde::Deserialize;

use crate::audio;
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
    let language = state
        .language
        .lock()
        .map(|g| *g)
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;

    let outcome = state
        .api
        .ask(&question, &context, language, &trace_id, &app)
        .await
        .map_err(|e| AppError::Llm(e.to_string()))?;
    metrics
        .last_llm_ms
        .store(outcome.total_ms, Ordering::Relaxed);
    metrics
        .last_llm_ttft_ms
        .store(outcome.ttft_ms, Ordering::Relaxed);
    set_provider(&state, &outcome.provider);
    Ok(outcome.answer)
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
#[tracing::instrument(skip(state))]
pub fn set_language(state: tauri::State<'_, AppState>, language: String) -> Result<(), AppError> {
    let new_lang = crate::lang::Language::from_tag(&language)
        .ok_or_else(|| AppError::Other(anyhow::anyhow!("unknown language: {language}")))?;
    let mut guard = state
        .language
        .lock()
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    *guard = new_lang;
    tracing::info!(?new_lang, "transcription/answer language switched");
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
