// The `#[tauri::command]` macro expands into code that binds the returned
// `Result<(), _>` with `let _ = ...`, which trips `let_underscore_must_use`.
// The ignore is intentional on tauri's side (IPC ACK), so we allow it here.
#![allow(clippy::let_underscore_must_use)]

use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde::Deserialize;

use crate::audio;
use crate::claude;
use crate::error::AppError;
use crate::AppState;

#[tauri::command]
#[tracing::instrument]
pub fn list_audio_devices() -> Result<Vec<audio::DeviceInfo>, AppError> {
    audio::list_input_devices().map_err(|e| AppError::Audio(e.to_string()))
}

#[tauri::command]
#[tracing::instrument(skip(state, app), fields(device = device_name.as_deref().unwrap_or("default")))]
pub async fn start_listening(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
    device_name: Option<String>,
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

    // Spawn a dedicated OS thread because cpal::Stream is not Send.
    // Inside, we create a single-threaded tokio runtime to await the audio pipeline.
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime");
        rt.block_on(async {
            if let Err(e) =
                audio::capture_and_transcribe(app, is_listening, metrics, device_name).await
            {
                tracing::error!(error = %e, "audio pipeline error");
            }
        });
    });

    Ok(())
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

#[tauri::command]
#[tracing::instrument(skip(state, context), fields(trace_id = trace_id.as_deref().unwrap_or("-"), question_len = question.len(), context_len = context.len()))]
pub async fn ask_claude(
    state: tauri::State<'_, AppState>,
    trace_id: Option<String>,
    question: String,
    context: String,
) -> Result<String, AppError> {
    let metrics = Arc::clone(&state.metrics);
    match claude::ask(&question, &context).await {
        Ok(result) => {
            metrics
                .last_llm_ms
                .store(result.spawn_ms + result.wait_ms, Ordering::Relaxed);
            metrics
                .last_llm_spawn_ms
                .store(result.spawn_ms, Ordering::Relaxed);
            Ok(result.answer)
        }
        Err(e) => Err(AppError::Claude(e.to_string())),
    }
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
    let data = entry
        .data
        .map_or_else(|| "{}".into(), |v| v.to_string());

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
