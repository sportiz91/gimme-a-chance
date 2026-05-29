//! Heap profiler via a custom `#[global_allocator]` wrapping an inner allocator.
//!
//! When active (feature `counting-alloc`), every heap alloc/dealloc in the
//! process flows through this module. We track three things on every call:
//!
//! 1. Running totals of bytes allocated and deallocated.
//! 2. Peak live bytes ever observed (updated via `fetch_max`).
//! 3. A 16-bucket logarithmic histogram — count of allocs (total and currently
//!    live) bucketed by size in powers of two, from ≤16 B up to >1 MB.
//!
//! The inner allocator (what actually fulfills the request) is selected at
//! compile time:
//! - Default: `std::alloc::System` — on Windows that's `HeapAlloc`, on Linux
//!   typically glibc `malloc`.
//! - With `--features mimalloc`: Microsoft's mimalloc, designed for low
//!   fragmentation and high multi-thread throughput.
//!
//! Running `cargo run --features counting-alloc` then
//! `cargo run --features "counting-alloc mimalloc"` lets you A/B compare
//! two allocators with identical instrumentation on top.
//!
//! Mutually exclusive with `assert_no_alloc::AllocDisabler` and `dhat::Alloc` —
//! only one `#[global_allocator]` is allowed per binary.
//!
//! Overhead per alloc: roughly three atomic `fetch_add` operations plus one
//! `fetch_max` (~5 ns each on x86-64). Negligible for desktop workloads.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicU64, Ordering};

// ── Inner allocator selection ──────────────────────────────────────────────
//
// Picked at compile time. Both `System` and `MiMalloc` are unit structs that
// implement `GlobalAlloc`, so we can use them as `const` values and delegate
// without holding any state.

#[cfg(not(feature = "mimalloc"))]
const INNER: std::alloc::System = std::alloc::System;

#[cfg(feature = "mimalloc")]
const INNER: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Human-readable name of the inner allocator. Shown in the debug panel and
/// logged on startup so you always know which allocator is actually underneath.
#[must_use]
pub const fn inner_allocator_name() -> &'static str {
    if cfg!(feature = "mimalloc") {
        "mimalloc"
    } else if cfg!(windows) {
        "system (HeapAlloc)"
    } else {
        "system (libc malloc)"
    }
}

// ── Counters ────────────────────────────────────────────────────────────────

static ALLOCATED: AtomicU64 = AtomicU64::new(0);
static DEALLOCATED: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Number of distinct size buckets in the histogram. Powers of two from 16 B
/// up to 1 MB, plus one catch-all bucket for anything larger.
pub const BUCKET_COUNT: usize = 16;

/// Upper bounds (inclusive) of each bucket in bytes. The last bucket is
/// conceptually "greater than the previous bound" — any size above 1 MB.
/// Ordered so index 0 is the smallest.
pub const BUCKET_BOUNDS: [usize; BUCKET_COUNT] = [
    16,
    32,
    64,
    128,
    256,
    512,
    1_024,      // 1 KB
    2_048,      // 2 KB
    4_096,      // 4 KB
    8_192,      // 8 KB
    16_384,     // 16 KB
    32_768,     // 32 KB
    65_536,     // 64 KB
    262_144,    // 256 KB
    1_048_576,  // 1 MB
    usize::MAX, // > 1 MB
];

/// Human-readable labels for each bucket. Used in log output and optional
/// panel display.
pub const BUCKET_LABELS: [&str; BUCKET_COUNT] = [
    "≤16B", "≤32B", "≤64B", "≤128B", "≤256B", "≤512B", "≤1KB", "≤2KB", "≤4KB", "≤8KB", "≤16KB",
    "≤32KB", "≤64KB", "≤256KB", "≤1MB", ">1MB",
];

// `[const { ... }; N]` is the stable syntax for initializing a fixed-size
// array of non-Copy types at compile time.
static TOTAL_ALLOCS_BY_BUCKET: [AtomicU64; BUCKET_COUNT] =
    [const { AtomicU64::new(0) }; BUCKET_COUNT];
static LIVE_ALLOCS_BY_BUCKET: [AtomicU64; BUCKET_COUNT] =
    [const { AtomicU64::new(0) }; BUCKET_COUNT];

/// Maps a byte size to its bucket index (0..BUCKET_COUNT).
#[inline]
#[must_use]
pub fn bucket_for(size: usize) -> usize {
    // Linear scan over 16 small constants is faster than branching ladders
    // and keeps the implementation obvious. The compiler turns this into a
    // handful of compares.
    for (i, &bound) in BUCKET_BOUNDS.iter().enumerate() {
        if size <= bound {
            return i;
        }
    }
    BUCKET_COUNT - 1
}

// ── Allocator ───────────────────────────────────────────────────────────────

pub struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let size_u64 = size as u64;

        // 1. Running totals. Using `fetch_add` to get the previous value lets
        //    us compute the post-op live bytes without a second load.
        let prev_allocated = ALLOCATED.fetch_add(size_u64, Ordering::Relaxed);
        let live_now =
            (prev_allocated + size_u64).saturating_sub(DEALLOCATED.load(Ordering::Relaxed));

        // 2. Peak tracking. `fetch_max` atomically bumps the peak if live_now
        //    exceeds it. One instruction on modern x86-64 (LOCK CMPXCHG loop
        //    in hardware).
        PEAK_LIVE_BYTES.fetch_max(live_now, Ordering::Relaxed);

        // 3. Histogram. Two counters per bucket: total-ever (monotonic) and
        //    live-right-now (decreases on dealloc).
        let bucket = bucket_for(size);
        TOTAL_ALLOCS_BY_BUCKET[bucket].fetch_add(1, Ordering::Relaxed);
        LIVE_ALLOCS_BY_BUCKET[bucket].fetch_add(1, Ordering::Relaxed);

        // SAFETY: forwarding the unmodified layout to the inner allocator is
        // always valid. We don't touch `layout` in any way.
        unsafe { INNER.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size();
        DEALLOCATED.fetch_add(size as u64, Ordering::Relaxed);
        LIVE_ALLOCS_BY_BUCKET[bucket_for(size)].fetch_sub(1, Ordering::Relaxed);

        // SAFETY: caller guarantees `ptr` came from our `alloc` with the same
        // `layout`; we forward both untouched to the inner allocator.
        unsafe { INNER.dealloc(ptr, layout) }
    }
}

// ── Public readers ──────────────────────────────────────────────────────────

/// Total bytes ever requested via `alloc` since process start.
#[must_use]
pub fn total_allocated() -> u64 {
    ALLOCATED.load(Ordering::Relaxed)
}

/// Total bytes ever returned via `dealloc` since process start.
#[must_use]
pub fn total_deallocated() -> u64 {
    DEALLOCATED.load(Ordering::Relaxed)
}

/// Current live heap size in bytes = allocated − deallocated.
/// Grows monotonically if there is a leak.
#[must_use]
pub fn live_bytes() -> u64 {
    total_allocated().saturating_sub(total_deallocated())
}

/// Maximum live heap size ever observed during process lifetime.
#[must_use]
pub fn peak_live_bytes() -> u64 {
    PEAK_LIVE_BYTES.load(Ordering::Relaxed)
}

/// Snapshot of the live-allocations histogram: how many chunks are currently
/// live in each size bucket. A large count in a small bucket with a small
/// count in a large bucket suggests external fragmentation risk.
#[must_use]
pub fn live_counts_by_bucket() -> [u64; BUCKET_COUNT] {
    let mut out = [0_u64; BUCKET_COUNT];
    for (i, slot) in LIVE_ALLOCS_BY_BUCKET.iter().enumerate() {
        out[i] = slot.load(Ordering::Relaxed);
    }
    out
}

/// Snapshot of the total-ever allocations histogram: how many chunks of each
/// size were ever requested. Useful for understanding workload characteristics
/// (e.g. "90% of allocs are ≤64B" → maybe a pool allocator would help).
#[must_use]
pub fn total_allocs_by_bucket() -> [u64; BUCKET_COUNT] {
    let mut out = [0_u64; BUCKET_COUNT];
    for (i, slot) in TOTAL_ALLOCS_BY_BUCKET.iter().enumerate() {
        out[i] = slot.load(Ordering::Relaxed);
    }
    out
}
