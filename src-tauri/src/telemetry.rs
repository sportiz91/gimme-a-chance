use std::path::PathBuf;

use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};

use crate::crashlog;

/// Guards that must live for the program's lifetime so background writers keep flushing.
/// The `flame` guard only exists when the `flame` feature is enabled.
pub struct TelemetryGuards {
    #[allow(dead_code)]
    file: WorkerGuard,
    #[cfg(feature = "flame")]
    #[allow(dead_code)]
    flame: tracing_flame::FlushGuard<std::io::BufWriter<std::fs::File>>,
}

/// In debug builds we log into `<project_root>/logs/` so the user can tail files
/// from their editor. In release builds we write under the user's local data dir.
#[must_use]
pub fn logs_dir() -> PathBuf {
    if cfg!(debug_assertions) {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map_or_else(|| PathBuf::from("logs"), |p| p.join("logs"))
    } else {
        dirs_next::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("gimme-a-chance")
            .join("logs")
    }
}

/// Initialize tracing with:
/// - pretty formatter → stdout (human readable while developing)
/// - JSON Lines formatter → rotating file (machine readable for later analysis)
/// - optional flamegraph layer → `logs/flame.folded` (only with `--features flame`)
///
/// Returns guards that MUST be kept alive for the lifetime of the process, otherwise
/// pending log lines or flamegraph samples can be dropped on exit.
pub fn init() -> TelemetryGuards {
    let logs = logs_dir();
    if let Err(e) = std::fs::create_dir_all(&logs) {
        eprintln!("telemetry: could not create logs dir {}: {e}", logs.display());
    }

    let file_appender = RollingFileAppender::builder()
        .rotation(Rotation::WEEKLY)
        .filename_prefix("gimme-a-chance")
        .filename_suffix("jsonl")
        .build(&logs)
        .expect("telemetry: failed to build rolling file appender");

    let (file_writer, file_guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,gimme_a_chance_lib=debug"));

    let file_layer = fmt::layer()
        .json()
        .with_current_span(true)
        .with_span_list(true)
        .with_target(true)
        .with_thread_names(true)
        .with_writer(file_writer);

    let stdout_layer = fmt::layer()
        .pretty()
        .with_target(false)
        .with_writer(std::io::stdout);

    // Breadcrumb ring buffer — the panic hook dumps these when the sky falls.
    let (breadcrumb_layer, breadcrumb_buffer) = crashlog::BreadcrumbLayer::new();

    let registry = Registry::default()
        .with(env_filter)
        .with(breadcrumb_layer)
        .with(file_layer)
        .with(stdout_layer);

    #[cfg(feature = "flame")]
    let (registry, flame_guard) = {
        let flame_path = logs.join("flame.folded");
        let (flame_layer, flame_guard) = tracing_flame::FlameLayer::with_file(&flame_path)
            .expect("telemetry: failed to create flame layer");
        (registry.with(flame_layer), flame_guard)
    };

    // Tracy sees `#[tracing::instrument]` spans as zones automatically.
    // Run the Tracy GUI (`tracy-profiler`) before launching to capture.
    #[cfg(feature = "tracy")]
    let registry = registry.with(tracing_tracy::TracyLayer::default());

    registry.init();

    // Install the panic hook AFTER subscriber init so tracing::error! from the
    // hook actually reaches the subscribers.
    crashlog::install_panic_hook(breadcrumb_buffer);

    tracing::info!(
        logs_dir = %logs.display(),
        rotation = "weekly",
        flame = cfg!(feature = "flame"),
        tracy = cfg!(feature = "tracy"),
        "telemetry initialized"
    );

    TelemetryGuards {
        file: file_guard,
        #[cfg(feature = "flame")]
        flame: flame_guard,
    }
}
