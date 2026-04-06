mod audio;
mod claude;
mod commands;
mod error;
mod transcriber;

use std::sync::{Arc, Mutex};

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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::list_audio_devices,
            commands::start_listening,
            commands::stop_listening,
            commands::ask_claude,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
