use anyhow::{Context, Result};
use std::fmt::Write;
use std::path::PathBuf;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const MODEL_FILENAME: &str = "ggml-base.en.bin";

pub struct WhisperTranscriber {
    ctx: WhisperContext,
}

impl WhisperTranscriber {
    pub fn new() -> Result<Self> {
        let model_path = Self::model_path();
        if !model_path.exists() {
            anyhow::bail!(
                "Whisper model not found at {}. Download it with:\n\
                 curl -L -o {} https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
                model_path.display(),
                model_path.display()
            );
        }

        let ctx = WhisperContext::new_with_params(
            model_path.to_str().unwrap(),
            WhisperContextParameters::default(),
        )
        .context("Failed to load Whisper model")?;

        eprintln!("Whisper model loaded: {}", model_path.display());
        Ok(Self { ctx })
    }

    /// Transcribe a chunk of 16kHz mono f32 audio
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let mut state = self
            .ctx
            .create_state()
            .context("Failed to create whisper state")?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_n_threads(4);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_temperature(0.0);
        // Single segment mode for real-time chunks
        params.set_single_segment(true);

        state
            .full(params, audio)
            .map_err(|e| anyhow::anyhow!("Whisper inference failed: {e:?}"))?;

        let mut text = String::new();
        for segment in state.as_iter() {
            let _ = write!(text, "{segment}");
        }

        Ok(text)
    }

    fn model_path() -> PathBuf {
        let data_dir = dirs_next::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("gimme-a-chance")
            .join("models");

        std::fs::create_dir_all(&data_dir).ok();
        data_dir.join(MODEL_FILENAME)
    }
}
