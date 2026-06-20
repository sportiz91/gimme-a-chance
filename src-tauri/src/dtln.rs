//! DTLN-aec: deep-learning acoustic echo cancellation.
//!
//! A port of the DTLN-aec inference loop (Westhausen, Microsoft AEC Challenge
//! 2021, 3rd place) running the two ONNX models on `tract` — a pure-Rust ONNX
//! runtime, so no native onnxruntime to clash with sherpa's. Unlike the classic
//! AEC3 (linear filter + spectral suppressor), the learned model preserves the
//! user's voice during double-talk instead of suppressing it along with the
//! residual echo.
//!
//! Per 128-sample hop over a 512-sample window: take the magnitude spectra of
//! the mic and loopback windows; the first model turns them into a mask over
//! the 257 frequency bins; the inverse transform of the masked mic spectrum is
//! a first estimate; the second model refines it in the time domain using the
//! loopback; overlap-add yields 128 output samples. The recurrent states carry
//! between hops, so this is true streaming.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;
use tract_onnx::prelude::*;

const SAMPLE_RATE: usize = 16_000;
const BLOCK_LEN: usize = 512;
const BLOCK_SHIFT: usize = 128;
const FREQ_BINS: usize = BLOCK_LEN / 2 + 1; // 257
/// LSTM state tensor shape, shared by both models ([1, 2 layers, 128 units, h+c]).
const STATE_SHAPE: [usize; 4] = [1, 2, 128, 2];

type DtlnModel = Arc<TypedRunnableModel>;

fn models_dir() -> PathBuf {
    dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("gimme-a-chance")
        .join("models")
        .join("dtln")
}

fn load(path: &Path) -> Result<DtlnModel> {
    tract_onnx::onnx()
        .model_for_path(path)
        .with_context(|| format!("loading DTLN model {}", path.display()))?
        .into_optimized()
        .map_err(|e| anyhow!("optimizing {}: {e}", path.display()))?
        .into_runnable()
        .map_err(|e| anyhow!("making {} runnable: {e}", path.display()))
}

pub struct DtlnCanceller {
    m1: DtlnModel,
    m2: DtlnModel,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    // Sliding analysis windows (512), shifted by 128 each hop.
    in_buf: Vec<f32>,
    lpb_buf: Vec<f32>,
    // Overlap-add accumulator (512).
    out_buf: Vec<f32>,
    // Samples awaiting a full hop.
    mic_acc: Vec<f32>,
    lpb_acc: Vec<f32>,
    // LSTM states carried between hops.
    states1: Tensor,
    states2: Tensor,
}

impl DtlnCanceller {
    pub fn new() -> Result<Self> {
        let dir = models_dir();
        let t0 = Instant::now();
        let m1 = load(&dir.join("dtln_aec_128_1.onnx"))?;
        let m2 = load(&dir.join("dtln_aec_128_2.onnx"))?;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(BLOCK_LEN);
        let ifft = planner.plan_fft_inverse(BLOCK_LEN);
        let states1 = Tensor::zero::<f32>(&STATE_SHAPE)?;
        let states2 = Tensor::zero::<f32>(&STATE_SHAPE)?;
        tracing::info!(
            load_ms = t0.elapsed().as_millis() as u64,
            model = %dir.display(),
            "DTLN-aec loaded (tract)"
        );
        Ok(Self {
            m1,
            m2,
            fft,
            ifft,
            in_buf: vec![0.0; BLOCK_LEN],
            lpb_buf: vec![0.0; BLOCK_LEN],
            out_buf: vec![0.0; BLOCK_LEN],
            mic_acc: Vec::with_capacity(BLOCK_LEN),
            lpb_acc: Vec::with_capacity(BLOCK_LEN),
            states1,
            states2,
        })
    }

    /// Feed reference (loopback) audio.
    pub fn push_reference(&mut self, samples: &[f32]) {
        self.lpb_acc.extend_from_slice(samples);
    }

    /// Cancel the interviewer's echo from mic `samples`, returning cleaned audio.
    /// Processes as many aligned 128-sample hops as both buffers allow.
    pub fn cancel(&mut self, samples: &[f32]) -> Vec<f32> {
        self.mic_acc.extend_from_slice(samples);
        let mut out = Vec::with_capacity(self.mic_acc.len());
        let t0 = Instant::now();
        let mut hops = 0u32;
        while self.mic_acc.len() >= BLOCK_SHIFT && self.lpb_acc.len() >= BLOCK_SHIFT {
            // Snapshot the hop; drain only after a successful process so a
            // transient model error doesn't desync the two streams.
            let mut mic_hop = [0.0f32; BLOCK_SHIFT];
            let mut lpb_hop = [0.0f32; BLOCK_SHIFT];
            mic_hop.copy_from_slice(&self.mic_acc[..BLOCK_SHIFT]);
            lpb_hop.copy_from_slice(&self.lpb_acc[..BLOCK_SHIFT]);
            match self.process_hop(&mic_hop, &lpb_hop) {
                Ok(hop_out) => {
                    out.extend_from_slice(&hop_out);
                    self.mic_acc.drain(..BLOCK_SHIFT);
                    self.lpb_acc.drain(..BLOCK_SHIFT);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "DTLN hop failed; passing mic through");
                    out.extend_from_slice(&mic_hop);
                    self.mic_acc.drain(..BLOCK_SHIFT);
                    self.lpb_acc.drain(..BLOCK_SHIFT);
                }
            }
            hops += 1;
        }
        if hops > 0 {
            // A hop is 8 ms of audio (128 samples @ 16 kHz). If per-hop time
            // exceeds that, the canceller can't keep up and `backlog_ms` climbs.
            let per_hop_us = t0.elapsed().as_micros() as u64 / u64::from(hops);
            let backlog_ms = self.mic_acc.len() * 1000 / SAMPLE_RATE;
            tracing::debug!(hops, per_hop_us, backlog_ms, "dtln cancel");
        }
        out
    }

    fn process_hop(
        &mut self,
        mic: &[f32; BLOCK_SHIFT],
        lpb: &[f32; BLOCK_SHIFT],
    ) -> Result<[f32; BLOCK_SHIFT]> {
        // Slide the windows: drop the oldest hop, append the new one.
        self.in_buf.copy_within(BLOCK_SHIFT.., 0);
        self.in_buf[BLOCK_LEN - BLOCK_SHIFT..].copy_from_slice(mic);
        self.lpb_buf.copy_within(BLOCK_SHIFT.., 0);
        self.lpb_buf[BLOCK_LEN - BLOCK_SHIFT..].copy_from_slice(lpb);

        // FFT both windows (realfft mutates its input, so use scratch copies).
        let mut in_scratch = self.in_buf.clone();
        let mut in_spec = self.fft.make_output_vec();
        self.fft
            .process(&mut in_scratch, &mut in_spec)
            .map_err(|e| anyhow!("mic fft: {e}"))?;
        let in_mag: Vec<f32> = in_spec.iter().map(|c| c.norm()).collect();

        let mut lpb_scratch = self.lpb_buf.clone();
        let mut lpb_spec = self.fft.make_output_vec();
        self.fft
            .process(&mut lpb_scratch, &mut lpb_spec)
            .map_err(|e| anyhow!("lpb fft: {e}"))?;
        let lpb_mag: Vec<f32> = lpb_spec.iter().map(|c| c.norm()).collect();

        // model_1(mic_mag, state, lpb_mag) → mask (graph input order: mic, state, lpb).
        let mic_mag_t = Tensor::from_shape(&[1, 1, FREQ_BINS], &in_mag)?;
        let lpb_mag_t = Tensor::from_shape(&[1, 1, FREQ_BINS], &lpb_mag)?;
        let r1 = self
            .m1
            .run(tvec!(
                mic_mag_t.into(),
                self.states1.clone().into(),
                lpb_mag_t.into()
            ))
            .map_err(|e| anyhow!("model_1: {e}"))?;
        let mask_view = r1[0].view();
        let mask = mask_view
            .as_slice::<f32>()
            .map_err(|e| anyhow!("mask: {e}"))?;
        self.states1 = r1[1].clone().into_tensor();

        // estimated = iFFT(mic_spectrum · mask). realfft's inverse is unnormalized,
        // so scale by 1/N to match numpy's irfft.
        let mut masked: Vec<Complex<f32>> = in_spec
            .iter()
            .zip(mask.iter())
            .map(|(c, &m)| c * m)
            .collect();
        let mut estimated = self.ifft.make_output_vec();
        self.ifft
            .process(&mut masked, &mut estimated)
            .map_err(|e| anyhow!("ifft: {e}"))?;
        let scale = 1.0 / BLOCK_LEN as f32;
        for x in &mut estimated {
            *x *= scale;
        }

        // model_2(estimated, state, lpb_window) → cleaned time-domain block.
        let est_t = Tensor::from_shape(&[1, 1, BLOCK_LEN], &estimated)?;
        let lpb_win_t = Tensor::from_shape(&[1, 1, BLOCK_LEN], &self.lpb_buf)?;
        let r2 = self
            .m2
            .run(tvec!(
                est_t.into(),
                self.states2.clone().into(),
                lpb_win_t.into()
            ))
            .map_err(|e| anyhow!("model_2: {e}"))?;
        let out_view = r2[0].view();
        let out_block = out_view
            .as_slice::<f32>()
            .map_err(|e| anyhow!("out_block: {e}"))?;
        self.states2 = r2[1].clone().into_tensor();

        // Overlap-add: shift out by a hop, zero the tail, add the new block.
        self.out_buf.copy_within(BLOCK_SHIFT.., 0);
        self.out_buf[BLOCK_LEN - BLOCK_SHIFT..].fill(0.0);
        for (o, &v) in self.out_buf.iter_mut().zip(out_block.iter()) {
            *o += v;
        }
        let mut hop = [0.0f32; BLOCK_SHIFT];
        hop.copy_from_slice(&self.out_buf[..BLOCK_SHIFT]);
        Ok(hop)
    }
}
