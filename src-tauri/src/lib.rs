mod audio;
mod claude;
mod commands;
mod error;
mod transcriber;

use std::sync::{Arc, Mutex};

use tauri::Manager;
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,gimme_a_chance_lib=debug".into()),
        )
        .init();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        "gimme-a-chance starting"
    );

    let toggle_shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space);

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, shortcut, event| {
                    if shortcut == &toggle_shortcut && event.state() == ShortcutState::Pressed {
                        if let Some(window) = app.get_webview_window("main") {
                            let is_visible = window.is_visible().unwrap_or(false);
                            tracing::debug!(
                                was_visible = is_visible,
                                action = if is_visible { "hide" } else { "show" },
                                "toggle triggered"
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
                    }
                })
                .build(),
        )
        .setup(move |app| {
            app.global_shortcut().register(toggle_shortcut)?;
            tracing::info!(shortcut = "Ctrl+Shift+Space", "global shortcut registered");
            Ok(())
        })
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
