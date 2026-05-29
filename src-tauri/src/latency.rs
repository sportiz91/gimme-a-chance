//! Real-time safe latency sampling.
//!
//! The audio callback pushes `u64` microsecond samples into an SPSC ring buffer
//! (lock-free, allocation-free). A dedicated reporter thread drains the ring,
//! feeds a `hdrhistogram::Histogram`, and logs p50/p95/p99 periodically.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#![allow(clippy::let_underscore_must_use)]

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use hdrhistogram::Histogram;
use ringbuf::{
    traits::{Consumer, Split},
    HeapRb,
};
use tracing::info;

use crate::metrics::Metrics;

pub type LatencyProducer = <HeapRb<u64> as Split>::Prod;
pub type LatencyConsumer = <HeapRb<u64> as Split>::Cons;

const RING_CAPACITY: usize = 4096;
// Percentiles are refreshed often (for the live debug panel) but logged
// less often (to keep the JSONL file from being flooded).
const METRICS_REFRESH_INTERVAL: Duration = Duration::from_secs(2);
const LOG_INTERVAL: Duration = Duration::from_secs(10);
// Histogram tracks values from 1µs to ~1 minute with 3 significant digits.
const HIST_MIN_US: u64 = 1;
const HIST_MAX_US: u64 = 60_000_000;
const HIST_PRECISION: u8 = 3;

/// Create an SPSC ring buffer tuned for microsecond latency samples.
/// The producer is safe to use inside real-time audio callbacks.
#[must_use]
pub fn channel() -> (LatencyProducer, LatencyConsumer) {
    HeapRb::<u64>::new(RING_CAPACITY).split()
}

/// Spawn a reporter thread that drains `consumer` into a histogram and logs
/// percentiles every [`REPORT_INTERVAL`]. The thread exits once `running`
/// becomes false AND the consumer is drained.
pub fn spawn_reporter(
    name: &'static str,
    mut consumer: LatencyConsumer,
    running: Arc<std::sync::atomic::AtomicBool>,
    metrics: Arc<Metrics>,
) -> thread::JoinHandle<()> {
    thread::Builder::new()
        .name(format!("latency-reporter-{name}"))
        .spawn(move || {
            let mut hist: Histogram<u64> =
                Histogram::new_with_bounds(HIST_MIN_US, HIST_MAX_US, HIST_PRECISION)
                    .expect("hdrhistogram bounds are valid");

            let mut last_refresh = std::time::Instant::now();
            let mut last_log = std::time::Instant::now();
            let mut scratch = [0u64; 256];

            while running.load(std::sync::atomic::Ordering::Relaxed) {
                let count = consumer.pop_slice(&mut scratch);
                if count > 0 {
                    for &sample in &scratch[..count] {
                        let clamped = sample.clamp(HIST_MIN_US, HIST_MAX_US);
                        let _ = hist.record(clamped);
                    }
                }

                if last_refresh.elapsed() >= METRICS_REFRESH_INTERVAL {
                    let should_log = last_log.elapsed() >= LOG_INTERVAL;
                    publish(&hist, &metrics, name, should_log);
                    last_refresh = std::time::Instant::now();
                    if should_log {
                        last_log = last_refresh;
                    }
                }

                if count == 0 {
                    thread::sleep(Duration::from_millis(50));
                }
            }

            while consumer.pop_slice(&mut scratch) > 0 {}
            publish(&hist, &metrics, name, true);
        })
        .expect("failed to spawn latency reporter thread")
}

fn publish(hist: &Histogram<u64>, metrics: &Metrics, name: &'static str, emit_log: bool) {
    if hist.is_empty() {
        return;
    }
    let p50 = hist.value_at_quantile(0.50);
    let p95 = hist.value_at_quantile(0.95);
    let p99 = hist.value_at_quantile(0.99);
    let p999 = hist.value_at_quantile(0.999);

    metrics
        .callback_samples
        .store(hist.len(), Ordering::Relaxed);
    metrics.callback_p50_us.store(p50, Ordering::Relaxed);
    metrics.callback_p95_us.store(p95, Ordering::Relaxed);
    metrics.callback_p99_us.store(p99, Ordering::Relaxed);
    metrics.callback_p999_us.store(p999, Ordering::Relaxed);

    if emit_log {
        info!(
            target: "latency",
            metric = name,
            samples = hist.len(),
            p50_us = p50,
            p95_us = p95,
            p99_us = p99,
            p999_us = p999,
            max_us = hist.max(),
            "latency report"
        );
    }
}
