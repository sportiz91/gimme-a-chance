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

use crate::cloud_stt;
use crate::latency;
use crate::metrics::Metrics;
use crate::transcriber;
use crate::vad::{ChunkAction, VadChunker, FRAME_SAMPLES};

const WHISPER_SAMPLE_RATE: u32 = 16_000;
// Ring buffer: 30 seconds of audio
const RING_BUFFER_SIZE: usize = WHISPER_SAMPLE_RATE as usize * 30;
// Pre-allocated scratch for mono downmix. Sized for up to ~100ms at 192kHz stereo (very generous).
const MONO_SCRATCH_CAPACITY: usize = 48_000;
// Below this RMS amplitude a chunk is treated as silence/noise and skipped BEFORE
// whisper runs — a microsecond check that vetoes a ~2.5s inference. Tunable: speech
// sits well above ~0.02; raise if noise still leaks, lower if quiet speech is dropped.
const RMS_SILENCE_THRESHOLD: f32 = 0.01;

#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub name: String,
    pub is_default: bool,
}

/// Cross-pipeline acoustic-bleed filter for dual ("both") capture.
///
/// With a headset, the interviewer's audio leaks from the earpiece into the
/// mic, so the mic pipeline re-transcribes the interviewer's words as if the
/// user spoke them — ghost `[You]` lines. A fixed RMS gate can't catch this:
/// bleed loudness tracks the headset volume, so any threshold that survives a
/// loud session clips the user's own quiet speech. Instead the interviewer
/// pipeline publishes its rolling hypothesis here (partials included — the mic
/// often endpoints the attenuated bleed BEFORE the interviewer finalizes), and
/// the mic pipeline drops any final whose tokens mostly already appear in that
/// recent window. Robust to volume; only your own words (which never play back
/// through the system) survive.
/// Timestamped hypotheses (each already tokenized) the interviewer has spoken
/// recently, oldest first.
type BleedHistory = std::collections::VecDeque<(Instant, Vec<String>)>;

// The window is read only by the sherpa streaming path; on cloud/whisper builds
// it's constructed and threaded through but never queried.
#[cfg_attr(not(feature = "sherpa"), allow(dead_code))]
#[derive(Clone, Default)]
pub struct BleedWindow {
    recent: Arc<Mutex<BleedHistory>>,
}

#[cfg_attr(not(feature = "sherpa"), allow(dead_code))]
impl BleedWindow {
    /// How far back the interviewer's words count as a possible bleed source.
    const WINDOW: std::time::Duration = std::time::Duration::from_secs(12);
    /// Below this, an echo like "yeah"/"okay" is too ambiguous to filter on.
    const MIN_TOKENS: usize = 3;
    /// Fraction of a mic final's tokens that must appear in the interviewer's
    /// recent window to call it bleed.
    const BLEED_FRACTION: f32 = 0.6;

    fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(str::to_lowercase)
            .collect()
    }

    /// Record the interviewer's current hypothesis as a possible bleed source.
    fn publish(&self, text: &str) {
        let toks = Self::tokenize(text);
        if toks.is_empty() {
            return;
        }
        let now = Instant::now();
        let mut q = self
            .recent
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        q.push_back((now, toks));
        while q
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t) > Self::WINDOW)
        {
            q.pop_front();
        }
    }

    /// Is `text` mostly an echo of the interviewer's recent words?
    fn is_bleed(&self, text: &str) -> bool {
        let toks = Self::tokenize(text);
        if toks.len() < Self::MIN_TOKENS {
            return false;
        }
        let now = Instant::now();
        let q = self
            .recent
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut recent: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (t, words) in q.iter() {
            if now.duration_since(*t) <= Self::WINDOW {
                recent.extend(words.iter().map(String::as_str));
            }
        }
        let hits = toks.iter().filter(|w| recent.contains(w.as_str())).count();
        hits as f32 / toks.len() as f32 >= Self::BLEED_FRACTION
    }
}

/// One end of the dual-capture echo-cancellation link. In "both" capture the
/// interviewer (loopback) pipeline feeds its audio as the AEC reference; the mic
/// pipeline receives it and cancels the interviewer's bleed out of the mic.
/// The channel plumbing is always available; only the AEC3 engine that consumes
/// it (in the sherpa streaming path) is feature-gated — hence the fields read
/// as dead on cloud/whisper builds.
#[cfg_attr(not(feature = "sherpa"), allow(dead_code))]
pub enum AecEnd {
    /// Loopback pipeline: publishes its 16 kHz audio as the reference signal.
    Reference(crossbeam_channel::Sender<Vec<f32>>),
    /// Mic pipeline: receives the reference and runs echo cancellation.
    Canceller(crossbeam_channel::Receiver<Vec<f32>>),
}

/// Which engine transcribes VAD chunks. Local whisper-rs stays as the grace
/// fallback for the fallible engines (cloud request, native inference), so a
/// hiccup never leaves the user without transcription mid-interview.
#[derive(Clone)]
pub enum SttEngine {
    /// Groq cloud Whisper — default when a `GROQ_API_KEY` is present.
    Groq(std::sync::Arc<cloud_stt::GroqStt>),
    /// On-device Parakeet via sherpa-onnx (UI: Local on, partials off).
    #[cfg(feature = "sherpa")]
    Parakeet(&'static crate::stt::ParakeetStt),
    /// On-device hybrid streaming via sherpa-onnx
    /// (UI: Local on, partials on): a light online model emits live partial
    /// hypotheses while the speaker talks; on endpoint the utterance audio is
    /// re-decoded with offline Parakeet for a high-quality final.
    #[cfg(feature = "sherpa")]
    Streaming(&'static crate::stt::StreamingStt),
    /// Local whisper-rs only (last-resort fallback: no Groq key, no models).
    LocalWhisper,
}

impl SttEngine {
    fn describe(&self) -> &'static str {
        match self {
            Self::Groq(_) => "groq/whisper-large-v3-turbo (+ local fallback)",
            #[cfg(feature = "sherpa")]
            Self::Parakeet(_) => "parakeet-tdt-0.6b-v2 via sherpa-onnx (+ local fallback)",
            #[cfg(feature = "sherpa")]
            Self::Streaming(_) => "hybrid: streaming partials + parakeet finals (sherpa-onnx)",
            Self::LocalWhisper => "whisper-rs local (base.en)",
        }
    }
}

/// Where to capture audio from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureSource {
    /// A microphone (WASAPI capture device). Picks up the user's own voice.
    Microphone,
    /// System audio via WASAPI loopback — records whatever is *playing* on an
    /// output device (the interviewer's voice from Zoom/Meet/YouTube). Works even
    /// with headphones, since it taps the render stream, not a mic.
    Loopback,
}

impl CaptureSource {
    #[must_use]
    pub fn from_opt(s: Option<&str>) -> Self {
        match s {
            Some("loopback") => Self::Loopback,
            _ => Self::Microphone,
        }
    }
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

/// List output (render) devices — these are the loopback-capture sources for
/// hearing the interviewer (whatever is playing through that device).
#[allow(clippy::unnecessary_wraps)]
pub fn list_output_devices() -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let default_name = host
        .default_output_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();

    let mut devices = Vec::new();
    if let Ok(output_devices) = host.output_devices() {
        for device in output_devices {
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

/// Root-mean-square amplitude of a chunk — a cheap loudness proxy used to skip
/// near-silent chunks before paying for whisper inference.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
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
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip(app, is_listening, metrics, whisper, engine, language, bleed, aec), fields(device = device_name.as_deref().unwrap_or("default"), source = ?source, speaker, lang = language.tag()))]
pub async fn capture_and_transcribe(
    app: AppHandle,
    is_listening: Arc<Mutex<bool>>,
    metrics: Arc<Metrics>,
    device_name: Option<String>,
    source: CaptureSource,
    speaker: &'static str,
    whisper: std::sync::Arc<transcriber::WhisperTranscriber>,
    engine: SttEngine,
    language: crate::lang::Language,
    bleed: Option<BleedWindow>,
    aec: Option<AecEnd>,
) -> Result<()> {
    // `bleed` and `aec` are only consumed by the sherpa streaming path; bind
    // them to silence unused warnings on cloud/whisper builds and non-streaming
    // engines.
    let _ = &bleed;
    let _ = &aec;
    // `language` selects the on-device Parakeet finals in the streaming path only;
    // bind it on non-sherpa builds where that path is compiled out.
    #[cfg(not(feature = "sherpa"))]
    let _ = language;
    let host = cpal::default_host();

    // Resolve the capture device + format. For Loopback we pick an OUTPUT device
    // and build an *input* stream on it — cpal's WASAPI backend then captures its
    // render stream (loopback). For Microphone we use a normal input device.
    let (device, config) = match source {
        CaptureSource::Loopback => {
            let device = match &device_name {
                Some(name) => host
                    .output_devices()?
                    .find(|d| d.name().map(|n| &n == name).unwrap_or(false))
                    .context(format!("Output device '{name}' not found"))?,
                None => host
                    .default_output_device()
                    .context("No default output device")?,
            };
            let config = device.default_output_config()?;
            (device, config)
        }
        CaptureSource::Microphone => {
            let device = match &device_name {
                Some(name) => host
                    .input_devices()?
                    .find(|d| d.name().map(|n| &n == name).unwrap_or(false))
                    .context(format!("Input device '{name}' not found"))?,
                None => host
                    .default_input_device()
                    .context("No default input device")?,
            };
            let config = device.default_input_config()?;
            (device, config)
        }
    };

    let sample_rate = config.sample_rate().0;
    let channels = config.channels();

    tracing::info!(
        device = %device.name().unwrap_or_default(),
        sample_rate,
        channels,
        source = ?source,
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

    tracing::info!(
        speaker,
        stt_engine = engine.describe(),
        "capture pipeline started"
    );

    // Streaming engine: continuous decode with partial hypotheses — a different
    // loop shape (no VAD batching), so it takes its own branch and shares the
    // cleanup tail below via early return.
    #[cfg(feature = "sherpa")]
    if let SttEngine::Streaming(stt) = &engine {
        streaming_loop(
            &app,
            &is_listening,
            &metrics,
            speaker,
            language,
            stt,
            &mut consumer,
            sample_rate,
            bleed.as_ref(),
            aec,
        )
        .await;
        drop(stream);
        lat_running.store(false, Ordering::Relaxed);
        app.emit("listening-stopped", ()).ok();
        tracing::info!("audio capture stopped");
        return Ok(());
    }

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
                // Pre-whisper energy gate: a VAD false-positive (keyboard click,
                // breath) is near-silent — skip it without paying ~2.5s of inference.
                let level = rms(&chunk);
                if level < RMS_SILENCE_THRESHOLD {
                    tracing::debug!(
                        rms = level,
                        chunk_ms = chunk.len() * 1000 / WHISPER_SAMPLE_RATE as usize,
                        "skipped low-energy chunk before whisper (RMS gate)"
                    );
                    continue;
                }
                let transcribe_start = Instant::now();
                let result = match &engine {
                    SttEngine::Groq(g) => match cloud_stt::encode_wav_16k_mono(&chunk) {
                        Ok(wav) => match g.transcribe_wav(wav).await {
                            Ok(text) => Ok(text),
                            Err(e) => {
                                tracing::warn!(error = %e, "Groq STT failed; falling back to local whisper");
                                whisper.transcribe(&chunk)
                            }
                        },
                        Err(e) => Err(e),
                    },
                    #[cfg(feature = "sherpa")]
                    SttEngine::Parakeet(p) => match p.transcribe(&chunk) {
                        Ok(text) => Ok(text),
                        Err(e) => {
                            tracing::warn!(error = %e, "Parakeet STT failed; falling back to local whisper");
                            whisper.transcribe(&chunk)
                        }
                    },
                    // Streaming runs its own loop above and never reaches the
                    // chunked path; defensive fallback rather than a panic.
                    #[cfg(feature = "sherpa")]
                    SttEngine::Streaming(_) => whisper.transcribe(&chunk),
                    SttEngine::LocalWhisper => whisper.transcribe(&chunk),
                };
                match result {
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
                                speaker,
                                text = %trimmed,
                                elapsed_ms,
                                chunk_samples = chunk.len(),
                                chunk_ms = chunk.len() * 1000 / WHISPER_SAMPLE_RATE as usize,
                                "transcription"
                            );
                            let payload = serde_json::json!({
                                "trace_id": trace_id,
                                "text": trimmed,
                                "speaker": speaker,
                            });
                            app.emit("transcription", payload).ok();
                            crate::agent::push_line(&app, speaker, &trimmed);
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

/// Continuous-decode loop for the hybrid streaming engine.
///
/// The light online model only powers `transcription-partial` (ephemeral gray
/// text, typos acceptable). On endpoint the buffered utterance audio is
/// re-decoded with offline Parakeet — a far stronger model — and THAT becomes
/// the `transcription` final (same payload as the chunked path, so question
/// detection and the answer chain work unchanged).
///
/// Finals and partials are RMS-gated like the VAD path: without the gate, a
/// strong model happily transcribes faint speaker-bleed into the mic, which
/// shows up as duplicated `[You]` lines during dual capture.
#[cfg(feature = "sherpa")]
#[allow(clippy::too_many_arguments)]
async fn streaming_loop<C: Consumer<Item = f32>>(
    app: &AppHandle,
    is_listening: &Arc<Mutex<bool>>,
    metrics: &Arc<Metrics>,
    speaker: &'static str,
    language: crate::lang::Language,
    stt: &'static crate::stt::StreamingStt,
    consumer: &mut C,
    sample_rate: u32,
    bleed: Option<&BleedWindow>,
    aec: Option<AecEnd>,
) {
    // Cap on the buffered utterance audio re-decoded by Parakeet on endpoint.
    const UTTERANCE_CAP: usize = WHISPER_SAMPLE_RATE as usize * 60;
    // Spanish-only: utterances whose online hypothesis is this short (in words) skip
    // the Canary offline final and keep the Kroko online hypothesis. Canary
    // hallucinates a word on tiny isolated fragments ("funcionales" → "Mussales");
    // the monolingual Kroko streaming model doesn't.
    const SPANISH_SHORT_FINAL_WORDS: usize = 2;
    // Trailing no-token time after which the Parakeet second pass starts
    // speculatively — well before the endpointer's rule2 (0.9s), so the decode
    // runs INSIDE the confirmation window instead of after it. Measured from
    // the online model's token timestamps (same signal the endpointer uses),
    // NOT from audio energy: background music keeps RMS high through speech
    // pauses and starves an energy-based trigger.
    const SPECULATIVE_TRAILING_S: f32 = 0.35;
    // Buffer partition margin past the last token (covers the token's own
    // tail), and a cap on how much un-tokenized audio carries over.
    const CARRY_PADDING_S: f32 = 0.24;
    const CARRY_MAX: usize = WHISPER_SAMPLE_RATE as usize * 3;

    // Bleed filter roles (dual capture only): the interviewer pipeline FEEDS the
    // shared window; the mic pipeline READS it to drop ghost lines. Single-source
    // capture passes `None`, so both are no-ops.
    let publish_bleed = bleed.filter(|_| speaker == "interviewer");
    let filter_bleed = bleed.filter(|_| speaker == "me");
    // Echo-cancellation roles (dual capture only): the loopback pipeline forwards
    // its audio as the reference; the mic pipeline builds an EchoCanceller and
    // subtracts that reference from the mic before the STT ever sees it — the
    // real fix for headset bleed, robust to volume and double-talk. A failed AEC
    // init degrades to raw mic (the bleed dedup downstream still applies).
    let aec_reference = match &aec {
        Some(AecEnd::Reference(tx)) => Some(tx.clone()),
        _ => None,
    };
    let mut aec_canceller = match aec {
        Some(AecEnd::Canceller(rx)) => match crate::aec::EchoCanceller::new() {
            Ok(c) => Some((rx, c)),
            Err(e) => {
                tracing::warn!(error = %e, "AEC init failed; mic runs without echo cancellation");
                None
            }
        },
        _ => None,
    };
    // The mic pipeline needs a higher gate than the loopback one: with loud
    // playback, speaker-bleed into the mic lands just above the global 0.01
    // threshold (one ghost [You] line per session), while the user's own
    // voice at their own mic sits far higher.
    let rms_gate = if speaker == "me" {
        RMS_SILENCE_THRESHOLD * 2.5
    } else {
        RMS_SILENCE_THRESHOLD
    };

    let recognizer = &stt.recognizer;
    let stream = recognizer.create_stream();
    let mut last_partial = String::new();
    // Utterance audio since the last endpoint, kept for the Parakeet re-decode.
    // Silence-only segments self-clear: the endpointer fires on trailing
    // silence with an empty hypothesis, which resets the buffer below.
    let mut utterance: Vec<f32> = Vec::with_capacity(WHISPER_SAMPLE_RATE as usize * 30);
    let mut utterance_peak_rms = 0.0f32;
    // Samples at the buffer head carried over from the previous segment. The
    // online model's token timestamps are SEGMENT-relative (they restart at
    // reset), so every timestamp→buffer-position conversion needs this offset.
    let mut carry_len: usize = 0;
    // In-flight speculative Parakeet decode of the current utterance.
    let mut speculative: Option<tokio::task::JoinHandle<Option<String>>> = None;

    loop {
        if !*is_listening
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            break;
        }

        // Pull any reference audio the loopback pipeline has forwarded, and feed
        // it to the echo canceller BEFORE this iteration's mic capture, so the
        // reference leads the capture the way AEC3 expects.
        if let Some((rx, canceller)) = &mut aec_canceller {
            while let Ok(reference) = rx.try_recv() {
                canceller.push_reference(&reference);
            }
        }

        // Drain whatever the audio callback produced since last iteration.
        let mut temp = [0.0f32; 1024];
        loop {
            let count = consumer.pop_slice(&mut temp);
            if count == 0 {
                break;
            }
            let resampled = resample_to_16k(&temp[..count], sample_rate);
            // Loopback: forward this audio as the AEC reference for the mic.
            if let Some(tx) = &aec_reference {
                tx.try_send(resampled.clone()).ok();
            }
            // Mic: cancel the interviewer's echo before anything downstream.
            let processed = match &mut aec_canceller {
                Some((_, canceller)) => canceller.cancel(&resampled),
                None => resampled,
            };
            utterance_peak_rms = utterance_peak_rms.max(rms(&processed));
            if utterance.len() < UTTERANCE_CAP {
                utterance.extend_from_slice(&processed);
            }
            stream.accept_waveform(WHISPER_SAMPLE_RATE.cast_signed(), &processed);
        }

        let decode_start = Instant::now();
        let mut decoded = false;
        while recognizer.is_ready(&stream) {
            recognizer.decode(&stream);
            decoded = true;
        }
        if decoded {
            metrics
                .last_stt_ms
                .store(decode_start.elapsed().as_millis() as u64, Ordering::Relaxed);
        }

        let (text, last_token_s) = recognizer
            .get_result(&stream)
            .map(|r| {
                let last = r
                    .timestamps
                    .as_ref()
                    .and_then(|t| t.last().copied())
                    .unwrap_or(0.0);
                (r.text, last)
            })
            .unwrap_or_default();

        // Speculative second pass: the endpointer needs ~0.9s without new
        // tokens before confirming, but the speech already ended — use that
        // window to run Parakeet so the final is ready when the endpoint
        // fires. Trailing time = audio fed since reset minus the last token's
        // timestamp (both segment-relative).
        let segment_s =
            utterance.len().saturating_sub(carry_len) as f32 / WHISPER_SAMPLE_RATE as f32;
        if speculative.is_none()
            && !text.is_empty()
            && utterance_peak_rms >= rms_gate
            && segment_s - last_token_s >= SPECULATIVE_TRAILING_S
        {
            if let Some(parakeet) = crate::stt::parakeet(language) {
                let snapshot = utterance.clone();
                speculative = Some(tokio::task::spawn_blocking(move || {
                    parakeet
                        .transcribe(&snapshot)
                        .ok()
                        .map(|t| t.trim().to_string())
                        .filter(|t| !t.is_empty())
                }));
            }
        }

        if recognizer.is_endpoint(&stream) {
            let streaming_text = text.trim().to_string();
            let had_speech = !streaming_text.is_empty();
            // Partition the buffer just after the last decoded token: audio
            // beyond that produced no tokens yet, so it belongs to the NEXT
            // utterance (the endpointer lags the real boundary by up to
            // rule2; clearing that tail swallowed boundary words). The cut
            // lands in post-token silence, never inside a word. The online
            // stream needs nothing — its un-decoded features survive reset.
            let tail_start = (carry_len
                + ((last_token_s + CARRY_PADDING_S) * WHISPER_SAMPLE_RATE as f32) as usize)
                .min(utterance.len());
            if had_speech && utterance_peak_rms >= rms_gate {
                // Hybrid final. Fast path: the speculative decode launched at
                // silence onset is (nearly) done — awaiting it costs ~0-50ms.
                // Slow path (speculation never started or was invalidated):
                // decode synchronously like before. Either way, fall back to
                // the online hypothesis if Parakeet produced nothing.
                let final_start = Instant::now();
                let was_speculative = speculative.is_some();
                // On short Spanish fragments, skip the Canary offline final (it
                // hallucinates without context) and keep the Kroko online hypothesis.
                let short_fragment = language == crate::lang::Language::Spanish
                    && streaming_text.split_whitespace().count() <= SPANISH_SHORT_FINAL_WORDS;
                let second_pass = if short_fragment {
                    // Any in-flight offline decode is discarded at the `speculative`
                    // reset after this block; just don't await it here.
                    tracing::debug!(
                        speaker,
                        text = %streaming_text,
                        "short Spanish fragment — keeping Kroko online hypothesis over Canary"
                    );
                    None
                } else {
                    match speculative.take() {
                        Some(handle) => handle.await.ok().flatten(),
                        None => crate::stt::parakeet(language).and_then(|p| {
                            match p.transcribe(&utterance[..tail_start]) {
                                Ok(t) if !t.trim().is_empty() => Some(t.trim().to_string()),
                                Ok(_) => None,
                                Err(e) => {
                                    tracing::warn!(error = %e, "Parakeet final failed; using streaming text");
                                    None
                                }
                            }
                        }),
                    }
                };
                let final_text = second_pass.unwrap_or(streaming_text);
                if filter_bleed.is_some_and(|b| b.is_bleed(&final_text)) {
                    // Mic final that echoes the interviewer's recent words —
                    // acoustic headset bleed, not the user. Drop it.
                    tracing::debug!(
                        speaker,
                        text = %final_text,
                        "discarded mic final as interviewer bleed"
                    );
                } else {
                    // The interviewer's words feed the bleed window (so the mic
                    // pipeline can recognize their echo).
                    if let Some(b) = publish_bleed {
                        b.publish(&final_text);
                    }
                    let trace_id = uuid::Uuid::new_v4().to_string();
                    tracing::info!(
                        trace_id = %trace_id,
                        speaker,
                        text = %final_text,
                        final_ms = final_start.elapsed().as_millis() as u64,
                        speculative = was_speculative,
                        utterance_ms = utterance.len() * 1000 / WHISPER_SAMPLE_RATE as usize,
                        "transcription"
                    );
                    let payload = serde_json::json!({
                        "trace_id": trace_id,
                        "text": final_text,
                        "speaker": speaker,
                    });
                    app.emit("transcription", payload).ok();
                    crate::agent::push_line(app, speaker, &final_text);
                }
            } else if had_speech {
                tracing::debug!(
                    peak_rms = utterance_peak_rms,
                    "discarded low-energy streaming final (RMS gate)"
                );
            }
            recognizer.reset(&stream);
            last_partial.clear();
            // Carry the un-tokenized tail into the next utterance buffer so
            // boundary words reach the next Parakeet pass. Silence-only
            // segments clear fully (nothing worth carrying, and carrying
            // would grow the buffer across consecutive silent endpoints).
            if had_speech {
                let carry_end = utterance.len().min(tail_start + CARRY_MAX);
                utterance.copy_within(tail_start..carry_end, 0);
                utterance.truncate(carry_end - tail_start);
            } else {
                utterance.clear();
            }
            carry_len = utterance.len();
            utterance_peak_rms = rms(&utterance);
            speculative = None;
            // Clear the partial line in the UI — the final (if any) replaced it.
            app.emit(
                "transcription-partial",
                serde_json::json!({ "speaker": speaker, "text": "" }),
            )
            .ok();
        } else if !text.is_empty() && text != last_partial && utterance_peak_rms >= rms_gate {
            // The hypothesis grew: speech resumed after the snapshot, so an
            // in-flight speculative decode no longer covers the utterance.
            // Drop it (the blocking task finishes on its own, discarded).
            speculative = None;
            // Feed the interviewer's live hypothesis to the bleed window NOW,
            // not just on endpoint: the mic pipeline often finalizes the
            // attenuated bleed before the interviewer's own endpoint, so the
            // window must already hold these words for the echo test to match.
            if let Some(b) = publish_bleed {
                b.publish(&text);
            }
            app.emit(
                "transcription-partial",
                serde_json::json!({ "speaker": speaker, "text": text }),
            )
            .ok();
            last_partial = text;
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }
}
