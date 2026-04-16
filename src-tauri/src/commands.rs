use std::sync::Arc;

use crate::audio;
use crate::claude;
use crate::error::AppError;
use crate::AppState;

#[tauri::command]
pub fn list_audio_devices() -> Result<Vec<audio::DeviceInfo>, AppError> {
    audio::list_input_devices().map_err(|e| AppError::Audio(e.to_string()))
}

#[tauri::command]
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

    // Spawn a dedicated OS thread because cpal::Stream is not Send.
    // Inside, we create a single-threaded tokio runtime to await the audio pipeline.
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime");
        rt.block_on(async {
            if let Err(e) = audio::capture_and_transcribe(app, is_listening, device_name).await {
                eprintln!("Audio pipeline error: {e}");
            }
        });
    });

    Ok(())
}

#[tauri::command]
pub async fn stop_listening(state: tauri::State<'_, AppState>) -> Result<(), AppError> {
    let mut is_listening = state
        .is_listening
        .lock()
        .map_err(|e| AppError::Other(anyhow::anyhow!("{e}")))?;
    *is_listening = false;
    Ok(())
}

#[tauri::command]
pub async fn ask_claude(question: String, context: String) -> Result<String, AppError> {
    claude::ask(&question, &context)
        .await
        .map_err(|e| AppError::Claude(e.to_string()))
}
