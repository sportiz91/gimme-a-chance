//! Cloud speech-to-text via Groq's hosted Whisper (`whisper-large-v3-turbo`).
//!
//! Fast (~0.4s for a multi-second chunk) and far more accurate than the local
//! `base.en` model, with no native deps. The audio pipeline uses this as the
//! primary engine and gracefully falls back to local whisper-rs if a request
//! fails (rate limit, network, timeout) — so a cloud hiccup never leaves the
//! user without transcription mid-interview.
#![allow(clippy::cast_possible_truncation)]

use std::io::Cursor;

use anyhow::{bail, Context, Result};
use secrecy::{ExposeSecret, SecretString};

const URL: &str = "https://api.groq.com/openai/v1/audio/transcriptions";
const MODEL: &str = "whisper-large-v3-turbo";
/// Browser UA — Groq sits behind Cloudflare, which 403s (error 1010) the default
/// reqwest client signature.
const BROWSER_UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

pub struct GroqStt {
    client: reqwest::Client,
    key: SecretString,
}

impl GroqStt {
    /// Build the engine, or `None` if there's no `GROQ_API_KEY` (caller uses local).
    #[must_use]
    pub fn new() -> Option<Self> {
        let key = crate::secrets::load_key("GROQ_API_KEY")?;
        let client = reqwest::Client::builder()
            .user_agent(BROWSER_UA)
            .build()
            .ok()?;
        Some(Self { client, key })
    }

    /// Transcribe a WAV blob. Returns the recognized text (may be empty for silence).
    pub async fn transcribe_wav(&self, wav: Vec<u8>) -> Result<String> {
        let part = reqwest::multipart::Part::bytes(wav)
            .file_name("chunk.wav")
            .mime_str("audio/wav")?;
        let form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", MODEL)
            // Force English: interviews are in English and the local fallback model
            // is English-only (base.en). Without this, Whisper auto-detects and
            // transcribes (e.g.) Spanish speech in Spanish.
            .text("language", "en")
            .text("response_format", "text");

        let resp = self
            .client
            .post(URL)
            .bearer_auth(self.key.expose_secret())
            .multipart(form)
            .send()
            .await
            .context("Groq STT request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let snippet: String = resp
                .text()
                .await
                .unwrap_or_default()
                .chars()
                .take(160)
                .collect();
            bail!("Groq STT HTTP {status}: {snippet}");
        }
        Ok(resp
            .text()
            .await
            .context("reading Groq STT body")?
            .trim()
            .to_string())
    }
}

/// Encode 16 kHz mono f32 samples as a WAV blob in memory (for upload).
pub fn encode_wav_16k_mono(samples: &[f32]) -> Result<Vec<u8>> {
    encode_wav_mono(samples, 16_000)
}

/// Encode mono f32 samples at an arbitrary rate as a 16-bit PCM WAV blob
/// (Kokoro synthesizes at 24 kHz).
pub fn encode_wav_mono(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).context("creating WAV writer")?;
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
            writer.write_sample(v).context("writing WAV sample")?;
        }
        writer.finalize().context("finalizing WAV")?;
    }
    Ok(cursor.into_inner())
}
