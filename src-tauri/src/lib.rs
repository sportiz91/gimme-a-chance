mod audio;
mod claude;
mod transcriber;

use std::sync::{Arc, Mutex};
use tauri::Manager;

/// Shared app state across Tauri commands
pub struct AppState {
    pub is_listening: Arc<Mutex<bool>>,
    pub transcript: Arc<Mutex<String>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            is_listening: Arc::new(Mutex::new(false)),
            transcript: Arc::new(Mutex::new(String::new())),
        }
    }
}

#[tauri::command]
fn list_audio_devices() -> Result<Vec<audio::DeviceInfo>, String> {
    audio::list_input_devices().map_err(|e| e.to_string())
}

#[tauri::command]
async fn start_listening(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
    device_name: Option<String>,
) -> Result<(), String> {
    let mut is_listening = state.is_listening.lock().map_err(|e| e.to_string())?;
    if *is_listening {
        return Err("Already listening".into());
    }
    *is_listening = true;
    drop(is_listening);

    let is_listening = Arc::clone(&state.is_listening);

    // Spawn audio capture + transcription pipeline in a blocking thread
    // (whisper-rs types are not Send, so we can't use tokio::spawn)
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            if let Err(e) = audio::capture_and_transcribe(app, is_listening, device_name).await {
                eprintln!("Audio pipeline error: {e}");
            }
        });
    });

    Ok(())
}

#[tauri::command]
async fn stop_listening(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut is_listening = state.is_listening.lock().map_err(|e| e.to_string())?;
    *is_listening = false;
    Ok(())
}

#[tauri::command]
async fn ask_claude(question: String, context: String) -> Result<String, String> {
    claude::ask(&question, &context)
        .await
        .map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            list_audio_devices,
            start_listening,
            stop_listening,
            ask_claude,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
