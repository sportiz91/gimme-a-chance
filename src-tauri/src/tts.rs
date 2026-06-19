//! Text-to-speech for the "simulate interviewer" self-test loop.
//!
//! Provider chain (mirrors the LLM fallback pattern): **local Kokoro first**
//! (offline, free — gated behind the `sherpa` feature), falling back to
//! **`OpenAI` `gpt-4o-mini-tts`**. Every clip is saved to disk and logged to the
//! JSONL so a run can be inspected/debugged later (by Claude Code or a human).
//!
//! Generation, persistence, and playback are deliberately separate stages:
//! synthesize → save WAV + log → play through the default output device (so a
//! loopback capture can pick it up, closing the test loop).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use secrecy::{ExposeSecret, SecretString};

/// Neutral voice + interviewer tone for the `OpenAI` path.
const OPENAI_VOICE: &str = "alloy";
const OPENAI_INSTRUCTIONS: &str =
    "Speak like a calm, professional senior engineer conducting a technical \
     interview. Neutral American accent, clear and unhurried.";
const BROWSER_UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

pub struct TtsOutcome {
    pub provider: String,
    pub wav_path: PathBuf,
    pub gen_ms: u64,
}

pub struct TtsEngine {
    client: reqwest::Client,
    openai: Option<SecretString>,
    clips_dir: PathBuf,
}

impl Default for TtsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TtsEngine {
    #[must_use]
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent(BROWSER_UA)
            .build()
            .expect("failed to build reqwest client");
        let openai = crate::secrets::load_key("OPENAI_API_KEY");
        let clips_dir = dirs_next::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("gimme-a-chance")
            .join("tts-clips");
        if let Err(e) = std::fs::create_dir_all(&clips_dir) {
            tracing::warn!(error = %e, dir = %clips_dir.display(), "could not create tts-clips dir");
        }
        tracing::info!(
            openai_key = openai.is_some(),
            kokoro = cfg!(feature = "sherpa"),
            clips_dir = %clips_dir.display(),
            "TTS engine initialized"
        );
        Self {
            client,
            openai,
            clips_dir,
        }
    }

    /// Synthesize `text`, save it as a WAV, and log a structured JSONL line.
    /// Returns the outcome (provider + path) so the caller can play it.
    pub async fn synthesize_and_save(&self, text: &str) -> Result<TtsOutcome> {
        let t0 = Instant::now();
        let (bytes, provider) = self.synthesize(text).await?;

        let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let path = self.clips_dir.join(format!("{stamp}-{}.wav", slug(text)));
        std::fs::write(&path, &bytes)
            .with_context(|| format!("writing tts clip to {}", path.display()))?;

        let gen_ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX);
        tracing::info!(
            target: "tts",
            event = "tts_generated",
            provider = %provider,
            chars = text.len(),
            bytes = bytes.len(),
            gen_ms,
            wav_path = %path.display(),
            "generated interviewer clip"
        );
        Ok(TtsOutcome {
            provider,
            wav_path: path,
            gen_ms,
        })
    }

    /// Try local Kokoro (if built), else `OpenAI`. Returns `(wav_bytes, provider)`.
    async fn synthesize(&self, text: &str) -> Result<(Vec<u8>, String)> {
        #[cfg(feature = "sherpa")]
        {
            match crate::stt::kokoro_tts(text) {
                Ok(Some(bytes)) => return Ok((bytes, "kokoro".into())),
                Ok(None) => tracing::debug!("kokoro model not present; using OpenAI TTS"),
                Err(e) => tracing::warn!(error = %e, "kokoro TTS failed; falling back to OpenAI"),
            }
        }
        let bytes = self.openai_tts(text).await?;
        Ok((bytes, "openai/gpt-4o-mini-tts".into()))
    }

    async fn openai_tts(&self, text: &str) -> Result<Vec<u8>> {
        let key = self
            .openai
            .as_ref()
            .ok_or_else(|| anyhow!("no Kokoro and no OPENAI_API_KEY — cannot synthesize speech"))?;
        let body = serde_json::json!({
            "model": "gpt-4o-mini-tts",
            "voice": OPENAI_VOICE,
            "input": text,
            "response_format": "wav",
            "instructions": OPENAI_INSTRUCTIONS,
        });
        let resp = self
            .client
            .post("https://api.openai.com/v1/audio/speech")
            .bearer_auth(key.expose_secret())
            .json(&body)
            .send()
            .await
            .context("OpenAI TTS request failed")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let snippet: String = resp
                .text()
                .await
                .unwrap_or_default()
                .chars()
                .take(200)
                .collect();
            anyhow::bail!("OpenAI TTS HTTP {status}: {snippet}");
        }
        Ok(resp
            .bytes()
            .await
            .context("reading TTS audio bytes")?
            .to_vec())
    }
}

/// Play a WAV file through the default output device, on a dedicated thread.
/// Fire-and-forget: returns immediately while audio plays (so the capture
/// pipeline on another thread can transcribe it as it sounds).
pub fn play_file(path: &Path) {
    let path = path.to_path_buf();
    std::thread::spawn(move || {
        if let Err(e) = play_blocking(&path) {
            tracing::warn!(error = %e, path = %path.display(), "tts playback failed");
        }
    });
}

fn play_blocking(path: &Path) -> Result<()> {
    use std::io::BufReader;
    // rodio's OutputStream isn't Send, so it must be created and used on this thread.
    let (_stream, handle) =
        rodio::OutputStream::try_default().context("no default audio output device")?;
    let sink = rodio::Sink::try_new(&handle).context("could not create audio sink")?;
    let file = std::fs::File::open(path).context("opening tts clip")?;
    let source = rodio::Decoder::new(BufReader::new(file)).context("decoding tts clip")?;
    sink.append(source);
    sink.sleep_until_end();
    Ok(())
}

/// Filesystem-safe short slug from the prompt text, for debuggable filenames.
fn slug(text: &str) -> String {
    let s: String = text
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let s = s.trim_matches('-').to_lowercase();
    let truncated: String = s
        .split('-')
        .filter(|p| !p.is_empty())
        .take(6)
        .collect::<Vec<_>>()
        .join("-");
    if truncated.is_empty() {
        "clip".into()
    } else {
        truncated.chars().take(48).collect()
    }
}
