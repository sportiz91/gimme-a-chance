//! Tier 1 + Tier 2 panic handling.
//!
//! - Tier 1: a global panic hook routes the panic info through `tracing::error!`
//!   so it lands in the JSONL file (in addition to Rust's default stderr output).
//! - Tier 2: a tracing `Layer` keeps the last N events in an in-memory ring buffer.
//!   When the panic hook fires, it snapshots the buffer and dumps the events as
//!   breadcrumbs, giving future-you the story of what happened just before the crash.
#![allow(clippy::module_name_repetitions)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer};

const BREADCRUMB_CAPACITY: usize = 50;

pub type BreadcrumbBuffer = Arc<Mutex<VecDeque<Breadcrumb>>>;

#[derive(Clone)]
pub struct Breadcrumb {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: String,
    pub target: String,
    pub message: String,
}

pub struct BreadcrumbLayer {
    buffer: BreadcrumbBuffer,
}

impl BreadcrumbLayer {
    #[must_use]
    pub fn new() -> (Self, BreadcrumbBuffer) {
        let buffer = Arc::new(Mutex::new(VecDeque::with_capacity(BREADCRUMB_CAPACITY)));
        (
            Self {
                buffer: Arc::clone(&buffer),
            },
            buffer,
        )
    }
}

impl<S: Subscriber> Layer<S> for BreadcrumbLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        // Skip events emitted by the panic hook itself so we don't pollute the
        // breadcrumbs with our own dump lines.
        let target = event.metadata().target();
        if target.starts_with("panic") {
            return;
        }

        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);

        let Ok(mut buf) = self.buffer.lock() else {
            return;
        };
        if buf.len() >= BREADCRUMB_CAPACITY {
            buf.pop_front();
        }
        buf.push_back(Breadcrumb {
            timestamp: chrono::Utc::now(),
            level: event.metadata().level().to_string(),
            target: target.to_string(),
            message: visitor.0,
        });
    }
}

/// Install a global panic hook that:
/// 1. Calls Rust's original panic hook (so stderr still gets the normal output).
/// 2. Snapshots the breadcrumb buffer and emits each event through `tracing::error!`.
/// 3. Logs the panic location and payload through `tracing::error!` so it lands in JSONL.
///
/// Safe to call once at startup.
pub fn install_panic_hook(buffer: BreadcrumbBuffer) {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Keep the default stderr output so terminal users always see the panic
        // even if the tracing subscriber is broken or not flushed.
        default_hook(info);

        let location = info.location().map_or_else(
            || "unknown".to_string(),
            |l| format!("{}:{}", l.file(), l.line()),
        );
        let payload = panic_payload_string(info);

        // `try_lock` avoids a self-deadlock if the panic somehow fired while
        // the breadcrumb layer was already holding the mutex.
        let crumbs: Vec<Breadcrumb> = match buffer.try_lock() {
            Ok(b) => b.iter().cloned().collect(),
            Err(_) => Vec::new(),
        };

        tracing::error!(
            target: "panic",
            location = %location,
            breadcrumbs = crumbs.len(),
            payload = %payload,
            "PANIC (breadcrumbs follow)"
        );
        for c in crumbs {
            tracing::error!(
                target: "panic.breadcrumb",
                at = %c.timestamp.to_rfc3339(),
                level = %c.level,
                origin = %c.target,
                "{}",
                c.message
            );
        }
    }));
}

fn panic_payload_string(info: &std::panic::PanicHookInfo<'_>) -> String {
    let p = info.payload();
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[derive(Default)]
struct MessageVisitor(String);

impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" && self.0.is_empty() {
            self.0 = format!("{value:?}");
        }
    }
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" && self.0.is_empty() {
            self.0 = value.to_string();
        }
    }
}
