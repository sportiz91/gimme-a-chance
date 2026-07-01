//! Shared metrics atomics read by the debug panel emitter and written by
//! various sources: the latency reporter (callback percentiles), the STT
//! pipeline (per-chunk transcription ms), the LLM command (per-call ms),
//! and — when the `counting-alloc` feature is active — the `CountingAllocator`
//! (live heap bytes, peak live bytes, and a 16-bucket size histogram).
#![allow(clippy::module_name_repetitions)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[cfg(feature = "counting-alloc")]
use crate::alloc_counter::BUCKET_COUNT;

// When the feature is off we still need the constant for the struct layout so
// that `MetricsSnapshot` has a stable size regardless of features. 16 matches
// `alloc_counter::BUCKET_COUNT` so the panel format stays identical.
#[cfg(not(feature = "counting-alloc"))]
const BUCKET_COUNT: usize = 16;

#[derive(Debug)]
pub struct Metrics {
    pub callback_samples: AtomicU64,
    pub callback_p50_us: AtomicU64,
    pub callback_p95_us: AtomicU64,
    pub callback_p99_us: AtomicU64,
    pub callback_p999_us: AtomicU64,
    pub last_stt_ms: AtomicU64,
    pub last_llm_ms: AtomicU64,
    /// Time-to-first-token of the last LLM turn (ms) — the "feels fast" metric.
    pub last_llm_ttft_ms: AtomicU64,
    /// Total ms of the last vision (screen describe) turn.
    pub last_vision_ms: AtomicU64,
    /// Which backend/provider answered the last turn (e.g. `groq/llama-3.1-8b-instant`).
    /// A `String`, so it lives behind a `Mutex` rather than an atomic.
    pub last_provider: Mutex<String>,
    // Heap fields are always present but only populated when `counting-alloc`
    // is active. When the feature is off they stay at 0 and the UI shows "—".
    pub heap_live_bytes: AtomicU64,
    pub heap_total_allocated_bytes: AtomicU64,
    pub heap_peak_live_bytes: AtomicU64,
    /// Count of currently-live allocations per size bucket.
    pub heap_buckets_live: [AtomicU64; BUCKET_COUNT],
    /// Count of total-ever allocations per size bucket (monotonic).
    pub heap_buckets_total: [AtomicU64; BUCKET_COUNT],
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            callback_samples: AtomicU64::new(0),
            callback_p50_us: AtomicU64::new(0),
            callback_p95_us: AtomicU64::new(0),
            callback_p99_us: AtomicU64::new(0),
            callback_p999_us: AtomicU64::new(0),
            last_stt_ms: AtomicU64::new(0),
            last_llm_ms: AtomicU64::new(0),
            last_llm_ttft_ms: AtomicU64::new(0),
            last_vision_ms: AtomicU64::new(0),
            last_provider: Mutex::new(String::new()),
            heap_live_bytes: AtomicU64::new(0),
            heap_total_allocated_bytes: AtomicU64::new(0),
            heap_peak_live_bytes: AtomicU64::new(0),
            // `std::array::from_fn` avoids the `Copy` requirement that a
            // `[AtomicU64::new(0); N]` initializer would impose.
            heap_buckets_live: std::array::from_fn(|_| AtomicU64::new(0)),
            heap_buckets_total: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl Metrics {
    pub fn snapshot(&self) -> MetricsSnapshot {
        let load_array = |arr: &[AtomicU64; BUCKET_COUNT]| -> [u64; BUCKET_COUNT] {
            let mut out = [0_u64; BUCKET_COUNT];
            for (i, slot) in arr.iter().enumerate() {
                out[i] = slot.load(Ordering::Relaxed);
            }
            out
        };

        MetricsSnapshot {
            callback_samples: self.callback_samples.load(Ordering::Relaxed),
            callback_p50_us: self.callback_p50_us.load(Ordering::Relaxed),
            callback_p95_us: self.callback_p95_us.load(Ordering::Relaxed),
            callback_p99_us: self.callback_p99_us.load(Ordering::Relaxed),
            callback_p999_us: self.callback_p999_us.load(Ordering::Relaxed),
            last_stt_ms: self.last_stt_ms.load(Ordering::Relaxed),
            last_llm_ms: self.last_llm_ms.load(Ordering::Relaxed),
            last_llm_ttft_ms: self.last_llm_ttft_ms.load(Ordering::Relaxed),
            last_vision_ms: self.last_vision_ms.load(Ordering::Relaxed),
            last_provider: self
                .last_provider
                .lock()
                .map(|g| g.clone())
                .unwrap_or_default(),
            heap_live_bytes: self.heap_live_bytes.load(Ordering::Relaxed),
            heap_total_allocated_bytes: self.heap_total_allocated_bytes.load(Ordering::Relaxed),
            heap_peak_live_bytes: self.heap_peak_live_bytes.load(Ordering::Relaxed),
            heap_buckets_live: load_array(&self.heap_buckets_live),
            heap_buckets_total: load_array(&self.heap_buckets_total),
            counting_alloc_enabled: cfg!(feature = "counting-alloc"),
            allocator_name: active_allocator_name(),
        }
    }
}

// ── active_allocator_name ──────────────────────────────────────────────────
//
// One `fn` definition per cfg slice. Only one matches any given build, so
// there's always exactly one symbol of this name. Multiple `#[cfg]`-gated
// copies is cleaner than a single function with interior cfg branches
// because the dhat / counting-alloc / mimalloc paths reference items that
// don't exist in feature-off builds.

/// Human-readable name of the active global allocator. Compile-time constant
/// in practice — decided entirely by feature flags and `debug_assertions`.
#[cfg(feature = "dhat-heap")]
#[must_use]
pub fn active_allocator_name() -> &'static str {
    "dhat"
}

#[cfg(all(feature = "counting-alloc", not(feature = "dhat-heap")))]
#[must_use]
pub fn active_allocator_name() -> &'static str {
    crate::alloc_counter::inner_allocator_name()
}

#[cfg(all(
    feature = "mimalloc",
    not(feature = "counting-alloc"),
    not(feature = "dhat-heap")
))]
#[must_use]
pub fn active_allocator_name() -> &'static str {
    "mimalloc"
}

#[cfg(all(
    not(feature = "mimalloc"),
    not(feature = "counting-alloc"),
    not(feature = "dhat-heap")
))]
#[must_use]
pub fn active_allocator_name() -> &'static str {
    if cfg!(debug_assertions) {
        "assert_no_alloc (debug)"
    } else if cfg!(windows) {
        "system (HeapAlloc)"
    } else {
        "system"
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSnapshot {
    pub callback_samples: u64,
    pub callback_p50_us: u64,
    pub callback_p95_us: u64,
    pub callback_p99_us: u64,
    pub callback_p999_us: u64,
    pub last_stt_ms: u64,
    pub last_llm_ms: u64,
    pub last_llm_ttft_ms: u64,
    pub last_vision_ms: u64,
    pub last_provider: String,
    pub heap_live_bytes: u64,
    pub heap_total_allocated_bytes: u64,
    pub heap_peak_live_bytes: u64,
    pub heap_buckets_live: [u64; BUCKET_COUNT],
    pub heap_buckets_total: [u64; BUCKET_COUNT],
    pub counting_alloc_enabled: bool,
    pub allocator_name: &'static str,
}
