//! Acoustic Echo Cancellation for dual ("both") capture.
//!
//! With a headset the interviewer's audio leaks from the earpiece into the mic,
//! so the mic pipeline hears the interviewer's voice as bleed and transcribes it
//! as ghost `[You]` lines. The fix is the textbook one: cancel the echo using
//! the signal we KNOW is playing — the loopback.
//!
//! Two engines, selected by `GIMME_AEC_ENGINE`:
//! - **`aec3`** (default) — pure-Rust port of WebRTC AEC3 (linear filter +
//!   non-linear suppressor). Great, but the suppressor erodes the user's voice
//!   during double-talk.
//! - **`dtln`** — DTLN-aec, a deep-learning AEC (Microsoft AEC Challenge) run on
//!   the pure-Rust `tract` ONNX runtime. Preserves the user's voice during
//!   double-talk far better. See [`crate::dtln`].
//!
//! Both consume the loopback as the reference and the mic as the capture,
//! returning the mic with the interviewer's voice removed. The whole pipeline
//! runs at 16 kHz mono.

use aec3::nodes::audio::AudioFormat;
use aec3::pipelines::linear::{self, LinearPipeline};
use anyhow::{anyhow, Result};

const SAMPLE_RATE: u32 = 16_000;

/// Echo-cancellation engine for the mic pipeline. Dispatches to whichever
/// backend `GIMME_AEC_ENGINE` selected; the streaming loop just calls
/// [`push_reference`](Self::push_reference) and [`cancel`](Self::cancel).
pub enum EchoCanceller {
    Aec3(Aec3Canceller),
    Dtln(crate::dtln::DtlnCanceller),
}

impl EchoCanceller {
    pub fn new() -> Result<Self> {
        if std::env::var("GIMME_AEC_ENGINE").ok().as_deref() == Some("dtln") {
            match crate::dtln::DtlnCanceller::new() {
                Ok(d) => {
                    tracing::info!("AEC engine: DTLN-aec (tract)");
                    return Ok(Self::Dtln(d));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "DTLN-aec init failed; falling back to AEC3");
                }
            }
        }
        tracing::info!("AEC engine: AEC3");
        Aec3Canceller::new().map(Self::Aec3)
    }

    /// Feed reference (loopback) audio — what's playing in the user's ear.
    pub fn push_reference(&mut self, samples: &[f32]) {
        match self {
            Self::Aec3(c) => c.push_reference(samples),
            Self::Dtln(c) => c.push_reference(samples),
        }
    }

    /// Cancel the interviewer's echo from mic `samples`, returning cleaned audio.
    pub fn cancel(&mut self, samples: &[f32]) -> Vec<f32> {
        match self {
            Self::Aec3(c) => c.cancel(samples),
            Self::Dtln(c) => c.cancel(samples),
        }
    }
}

/// WebRTC AEC3 (pure-Rust `aec3` crate). Works in 10 ms frames (160 samples),
/// so this buffers the variable-size drains into exact frames.
pub struct Aec3Canceller {
    pipeline: LinearPipeline,
    /// Samples per 10 ms AEC frame (160 at 16 kHz mono).
    frame_len: usize,
    /// Reference (loopback) samples not yet aligned to a frame boundary.
    render_buf: Vec<f32>,
    /// Capture (mic) samples not yet aligned to a frame boundary.
    capture_buf: Vec<f32>,
    /// Scratch for one frame of cancelled output (reused, no per-frame alloc).
    frame_out: Vec<f32>,
}

impl Aec3Canceller {
    pub fn new() -> Result<Self> {
        let format = AudioFormat::ten_ms(SAMPLE_RATE, 1);
        let frame_len = format.sample_count();
        // Initial guess for the loopback→mic delay. AEC3's delay estimator
        // refines it adaptively, so this is only a starting hint; tune via env
        // var if convergence is slow on a given headset.
        let delay_ms = std::env::var("GIMME_AEC_DELAY_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let pipeline = linear::builder(format, format)
            .initial_delay_ms(delay_ms)
            .enable_high_pass_filter(true)
            // Keep the signal as close to raw as possible for the STT models —
            // cancel the echo, but don't let noise suppression or auto-gain
            // reshape the user's voice (the recognizers are already noise-robust,
            // and AGC would fight the RMS gate downstream).
            .enable_noise_suppression(false)
            .enable_gain_controller2(false)
            .build()
            .map_err(|e| anyhow!("building AEC3 pipeline: {e:?}"))?;
        tracing::info!(delay_ms, frame_len, "AEC3 echo canceller initialized");
        Ok(Self {
            pipeline,
            frame_len,
            render_buf: Vec::with_capacity(frame_len * 4),
            capture_buf: Vec::with_capacity(frame_len * 4),
            frame_out: vec![0.0; frame_len],
        })
    }

    /// Feed reference (loopback) audio — what's playing in the user's ear.
    pub fn push_reference(&mut self, samples: &[f32]) {
        self.render_buf.extend_from_slice(samples);
        while self.render_buf.len() >= self.frame_len {
            if let Err(e) = self
                .pipeline
                .handle_render_frame(&self.render_buf[..self.frame_len])
            {
                tracing::warn!(error = ?e, "AEC render frame failed");
            }
            self.render_buf.drain(..self.frame_len);
        }
    }

    /// Cancel the interviewer's echo from `samples` (mic audio), returning the
    /// cleaned audio. May return slightly fewer samples than the input when the
    /// tail doesn't fill a whole frame (it's carried to the next call).
    pub fn cancel(&mut self, samples: &[f32]) -> Vec<f32> {
        self.capture_buf.extend_from_slice(samples);
        let mut out = Vec::with_capacity(self.capture_buf.len());
        while self.capture_buf.len() >= self.frame_len {
            let processed = match self
                .pipeline
                .process_capture_frame(&self.capture_buf[..self.frame_len], &mut self.frame_out)
            {
                Ok(true) => true,
                Ok(false) => false,
                Err(e) => {
                    tracing::warn!(error = ?e, "AEC capture frame failed");
                    false
                }
            };
            if processed {
                out.extend_from_slice(&self.frame_out);
            } else {
                // No cancelled output this frame — pass the raw mic through so
                // we never drop the user's audio.
                out.extend_from_slice(&self.capture_buf[..self.frame_len]);
            }
            self.capture_buf.drain(..self.frame_len);
        }
        out
    }
}
