#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::let_underscore_must_use,
    clippy::too_many_lines
)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use assert_no_alloc::assert_no_alloc;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{
    traits::{Consumer, Producer, Split},
    HeapRb,
};
use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::latency;
use crate::metrics::Metrics;
use crate::transcriber;
use crate::vad::{ChunkAction, VadChunker, FRAME_SAMPLES};

const WHISPER_SAMPLE_RATE: u32 = 16_000;
// Ring buffer: 30 seconds of audio
const RING_BUFFER_SIZE: usize = WHISPER_SAMPLE_RATE as usize * 30;
// Pre-allocated scratch for mono downmix. Sized for up to ~100ms at 192kHz stereo (very generous).
const MONO_SCRATCH_CAPACITY: usize = 48_000;

#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub name: String,
    pub is_default: bool,
}

#[allow(clippy::unnecessary_wraps)]
pub fn list_input_devices() -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let default_device = host.default_input_device();
    let default_name = default_device
        .as_ref()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();

    let mut devices = Vec::new();
    if let Ok(input_devices) = host.input_devices() {
        for device in input_devices {
            if let Ok(name) = device.name() {
                devices.push(DeviceInfo {
                    is_default: name == default_name,
                    name,
                });
            }
        }
    }
    Ok(devices)
}

/// Resample from `source_rate` to 16kHz mono f32
fn resample_to_16k(samples: &[f32], source_rate: u32) -> Vec<f32> {
    if source_rate == WHISPER_SAMPLE_RATE {
        return samples.to_vec();
    }
    let ratio = f64::from(source_rate) / f64::from(WHISPER_SAMPLE_RATE);
    let output_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;
        let sample = if idx + 1 < samples.len() {
            f64::from(samples[idx]) * (1.0 - frac) + f64::from(samples[idx + 1]) * frac
        } else if idx < samples.len() {
            f64::from(samples[idx])
        } else {
            0.0
        };
        output.push(sample as f32);
    }
    output
}

/// Whisper emits bracketed meta-tokens when it fails to detect speech in a chunk.
/// Common ones: `[BLANK_AUDIO]`, `[MUSIC]`, `[SOUND]`, `[INAUDIBLE]`. They look like
/// text to our filter but have no signal, so we drop them before bubbling to the UI.
fn is_whisper_non_speech_tag(text: &str) -> bool {
    let t = text.trim();
    t.starts_with('[') && t.ends_with(']') && !t.contains(' ')
}

/// Downmix multi-channel interleaved samples into `out` without allocating
/// (assuming `out` has enough capacity — it is pre-sized at stream init).
fn to_mono_into(samples: &[f32], channels: u16, out: &mut Vec<f32>) {
    out.clear();
    if channels == 1 {
        out.extend_from_slice(samples);
        return;
    }
    let ch = channels as usize;
    let inv = 1.0 / f32::from(channels);
    let frames = samples.len() / ch;
    for frame_idx in 0..frames {
        let base = frame_idx * ch;
        let mut sum = 0.0f32;
        for offset in 0..ch {
            sum += samples[base + offset];
        }
        out.push(sum * inv);
    }
}

/// Main audio capture + transcription loop
#[tracing::instrument(skip(app, is_listening, metrics), fields(device = device_name.as_deref().unwrap_or("default")))]
pub async fn capture_and_transcribe(
    app: AppHandle,
    is_listening: Arc<Mutex<bool>>,
    metrics: Arc<Metrics>,
    device_name: Option<String>,
) -> Result<()> {
    let host = cpal::default_host();

    let device = if let Some(name) = device_name {
        host.input_devices()?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .context(format!("Device '{name}' not found"))?
    } else {
        host.default_input_device()
            .context("No default input device")?
    };

    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels();

    tracing::info!(
        device = %device.name().unwrap_or_default(),
        sample_rate,
        channels,
        "audio capture starting"
    );

    // Lock-free ring buffer: producer in audio callback, consumer in processing thread
    let rb = HeapRb::<f32>::new(RING_BUFFER_SIZE);
    let (mut producer, mut consumer) = rb.split();

    // Latency pipeline: the callback is the producer, a reporter thread consumes.
    let (mut lat_producer, lat_consumer) = latency::channel();
    let lat_running = Arc::new(AtomicBool::new(true));
    let _lat_thread = latency::spawn_reporter(
        "audio_callback",
        lat_consumer,
        Arc::clone(&lat_running),
        Arc::clone(&metrics),
    );

    // Pre-allocated scratch buffer for mono downmix; moved into the callback so no alloc per call.
    let mut mono_scratch: Vec<f32> = Vec::with_capacity(MONO_SCRATCH_CAPACITY);

    // Build input stream
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let start = Instant::now();
            // The body must be allocation-free. In debug this will panic on any heap
            // allocation; in release the wrapper is a no-op.
            assert_no_alloc(|| {
                to_mono_into(data, channels, &mut mono_scratch);
                for &sample in &mono_scratch {
                    _ = producer.try_push(sample);
                }
            });
            let elapsed_us = start.elapsed().as_micros().min(u64::MAX as u128) as u64;
            _ = lat_producer.try_push(elapsed_us);
        },
        |err| tracing::warn!(error = %err, "audio stream error"),
        None,
    )?;

    stream.play()?;
    app.emit("listening-started", ()).ok();

    // Initialize whisper
    let whisper = transcriber::WhisperTranscriber::new()?;

    // VAD chunker: cuts on silence rather than wall time.
    let mut chunker = VadChunker::new();
    // Pending buffer of 16kHz mono samples waiting to be consumed in VAD-sized frames.
    let mut pending: Vec<f32> = Vec::with_capacity(FRAME_SAMPLES * 4);

    loop {
        if !*is_listening
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            break;
        }

        // Drain ring buffer into our processing buffer, resampling as we go.
        let mut temp = [0.0f32; 1024];
        loop {
            let count = consumer.pop_slice(&mut temp);
            if count == 0 {
                break;
            }
            let resampled = resample_to_16k(&temp[..count], sample_rate);
            pending.extend_from_slice(&resampled);
        }

        // Feed full VAD frames; anything short of a frame stays for next iteration.
        while pending.len() >= FRAME_SAMPLES {
            let frame: Vec<f32> = pending.drain(..FRAME_SAMPLES).collect();
            if let ChunkAction::Emit(chunk) = chunker.push_frame(&frame) {
                let transcribe_start = Instant::now();
                match whisper.transcribe(&chunk) {
                    Ok(text) => {
                        let elapsed_ms = transcribe_start.elapsed().as_millis();
                        metrics
                            .last_stt_ms
                            .store(elapsed_ms as u64, Ordering::Relaxed);
                        let trimmed = text.trim().to_string();
                        if trimmed.is_empty() || is_whisper_non_speech_tag(&trimmed) {
                            tracing::debug!(
                                text = %trimmed,
                                elapsed_ms,
                                chunk_ms = chunk.len() * 1000 / WHISPER_SAMPLE_RATE as usize,
                                "discarded non-speech chunk"
                            );
                        } else {
                            let trace_id = uuid::Uuid::new_v4().to_string();
                            tracing::info!(
                                trace_id = %trace_id,
                                text = %trimmed,
                                elapsed_ms,
                                chunk_samples = chunk.len(),
                                chunk_ms = chunk.len() * 1000 / WHISPER_SAMPLE_RATE as usize,
                                "transcription"
                            );
                            let payload = serde_json::json!({
                                "trace_id": trace_id,
                                "text": trimmed,
                            });
                            app.emit("transcription", payload).ok();
                        }
                    }
                    Err(e) => tracing::warn!(error = %e, "transcription failed"),
                }
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
    }

    drop(stream);
    lat_running.store(false, Ordering::Relaxed);
    app.emit("listening-stopped", ()).ok();
    tracing::info!("audio capture stopped");
    Ok(())
}
