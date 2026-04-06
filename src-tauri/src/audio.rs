#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{
    traits::{Consumer, Producer, Split},
    HeapRb,
};
use serde::Serialize;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};

use crate::transcriber;

const WHISPER_SAMPLE_RATE: u32 = 16_000;
const CHUNK_DURATION_SECS: f32 = 5.0;
const CHUNK_SAMPLES: usize = (WHISPER_SAMPLE_RATE as f32 * CHUNK_DURATION_SECS) as usize;
// Ring buffer: 30 seconds of audio
const RING_BUFFER_SIZE: usize = WHISPER_SAMPLE_RATE as usize * 30;

#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub name: String,
    pub is_default: bool,
}

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

/// Convert multi-channel to mono by averaging
fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels as usize)
        .map(|frame| frame.iter().sum::<f32>() / f32::from(channels))
        .collect()
}

/// Main audio capture + transcription loop
pub async fn capture_and_transcribe(
    app: AppHandle,
    is_listening: Arc<Mutex<bool>>,
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

    eprintln!(
        "Capturing from '{}' at {}Hz, {} channels",
        device.name().unwrap_or_default(),
        sample_rate,
        channels
    );

    // Lock-free ring buffer: producer in audio callback, consumer in processing thread
    let rb = HeapRb::<f32>::new(RING_BUFFER_SIZE);
    let (mut producer, mut consumer) = rb.split();

    // Build input stream
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Convert to mono in the callback (fast, no allocations for small frames)
            let mono = to_mono(data, channels);
            for &sample in &mono {
                let _ = producer.try_push(sample);
            }
        },
        |err| eprintln!("Audio stream error: {err}"),
        None,
    )?;

    stream.play()?;
    app.emit("listening-started", ()).ok();

    // Initialize whisper
    let whisper = transcriber::WhisperTranscriber::new()?;

    // Processing loop: read chunks from ring buffer, transcribe
    let chunk_size = (WHISPER_SAMPLE_RATE as f32 * CHUNK_DURATION_SECS) as usize;
    let mut audio_buffer: Vec<f32> = Vec::with_capacity(chunk_size * 2);

    loop {
        // Check if we should stop
        if !*is_listening
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            break;
        }

        // Drain ring buffer into our processing buffer
        let mut temp = [0.0f32; 1024];
        loop {
            let count = consumer.pop_slice(&mut temp);
            if count == 0 {
                break;
            }
            // Resample to 16kHz
            let resampled = resample_to_16k(&temp[..count], sample_rate);
            audio_buffer.extend_from_slice(&resampled);
        }

        // When we have enough audio, transcribe
        if audio_buffer.len() >= chunk_size {
            let chunk: Vec<f32> = audio_buffer.drain(..chunk_size).collect();

            // RMS energy check — skip silent chunks
            let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
            if rms < 0.01 {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                continue;
            }

            match whisper.transcribe(&chunk) {
                Ok(text) if !text.trim().is_empty() => {
                    eprintln!("Transcribed: {}", text.trim());
                    app.emit("transcription", text.trim()).ok();
                }
                Ok(_) => {} // empty transcription, skip
                Err(e) => eprintln!("Transcription error: {e}"),
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    drop(stream);
    app.emit("listening-stopped", ()).ok();
    Ok(())
}
