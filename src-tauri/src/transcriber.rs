use anyhow::{Context, Result};
use std::fmt::Write;
use std::path::PathBuf;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::lang::Language;

/// Domain vocabulary primer for a software-engineering interview. Steers whisper
/// toward correct spellings of jargon it otherwise garbles. The technical terms
/// stay in English (they're English loanwords in a Spanish interview too); only
/// the framing sentence is translated.
const INITIAL_PROMPT_EN: &str = "A technical software engineering interview. Topics include \
Rust, tokio, async/await, Next.js, React, the DOM, TypeScript, Kubernetes, Docker, \
PostgreSQL, Redis, gRPC, REST, GraphQL, WebSocket, OAuth, JWT, mutex, semaphore, \
latency, throughput, rate limiter, and system design.";
const INITIAL_PROMPT_ES: &str = "Una entrevista técnica de ingeniería de software. Temas: \
Rust, tokio, async/await, Next.js, React, el DOM, TypeScript, Kubernetes, Docker, \
PostgreSQL, Redis, gRPC, REST, GraphQL, WebSocket, OAuth, JWT, mutex, semáforo, \
latencia, throughput, rate limiter, y diseño de sistemas.";

/// Local whisper.cpp model file per language. English uses the English-only
/// `base.en` (slightly sharper on English); Spanish needs the multilingual `base`.
fn model_filename(lang: Language) -> &'static str {
    match lang {
        Language::English => "ggml-base.en.bin",
        Language::Spanish => "ggml-base.bin",
    }
}

fn initial_prompt(lang: Language) -> &'static str {
    match lang {
        Language::English => INITIAL_PROMPT_EN,
        Language::Spanish => INITIAL_PROMPT_ES,
    }
}

pub struct WhisperTranscriber {
    ctx: WhisperContext,
    lang: Language,
}

impl WhisperTranscriber {
    pub fn new(lang: Language) -> Result<Self> {
        let model_path = Self::model_path(lang);
        if !model_path.exists() {
            anyhow::bail!(
                "Whisper ({}) model not found at {}. Download it with:\n\
                 curl -L -o {} https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
                lang.whisper_code(),
                model_path.display(),
                model_path.display(),
                model_filename(lang)
            );
        }

        let ctx = WhisperContext::new_with_params(
            model_path.to_str().unwrap(),
            WhisperContextParameters::default(),
        )
        .context("Failed to load Whisper model")?;

        eprintln!("Whisper model loaded: {}", model_path.display());
        Ok(Self { ctx, lang })
    }

    /// Transcribe a chunk of 16kHz mono f32 audio
    #[tracing::instrument(skip(self, audio), fields(samples = audio.len()))]
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let mut state = self
            .ctx
            .create_state()
            .context("Failed to create whisper state")?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(self.lang.whisper_code()));
        // 6 threads on the i7-1165G7 (4 physical / 8 logical): whisper.cpp is
        // compute-bound so gains taper past physical cores, but 6 edges out 4.
        params.set_n_threads(6);
        // Bias decoding toward the technical vocabulary that base mangles
        // ("tokio"→"Tokyo", "the DOM"→"dumb", "Next.js"→"Next Shaiyes"). The prompt
        // primes whisper's context so these spellings become likely.
        params.set_initial_prompt(initial_prompt(self.lang));
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
            write!(text, "{segment}").expect("writing to String is infallible");
        }

        Ok(text)
    }

    fn model_path(lang: Language) -> PathBuf {
        let data_dir = dirs_next::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("gimme-a-chance")
            .join("models");

        std::fs::create_dir_all(&data_dir).ok();
        data_dir.join(model_filename(lang))
    }
}
