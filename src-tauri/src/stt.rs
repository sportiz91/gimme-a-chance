//! On-device models via the official `sherpa-onnx` crate (k2-fsa bindings),
//! gated behind the `sherpa` feature. Models are fetched by
//! `scripts/fetch-models.ps1`.
//!
//! Three engines live here:
//!
//! - **`ParakeetStt`** — offline `NeMo` Parakeet TDT, decodes one VAD chunk at a
//!   time (UI: Local on, partials off). Most accurate, ~200ms per chunk.
//! - **`StreamingStt`** — light online model with endpoint detection
//!   (UI: Local on, partials on). Powers live partial hypotheses ONLY; on
//!   endpoint the audio loop re-decodes the utterance with `ParakeetStt`, so
//!   finals get Parakeet quality. A heavier online model (Nemotron 0.6b) was
//!   tried here and saturated the CPU with dual capture — partials lagged
//!   behind real time. Small+fast wins for ephemeral text.
//! - **`kokoro_tts`** — local Kokoro TTS for the simulate-interviewer loop.
//!
//! Everything loads once per app run (`OnceLock`) and falls back cleanly when
//! absent or broken: STT → the default Groq/whisper chain, Kokoro → `OpenAI`
//! TTS. A missing model is a degraded mode, never an error mid-interview.
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};

use crate::lang::Language;
use sherpa_onnx::{
    GenerationConfig, OfflineCanaryModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineTransducerModelConfig, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig,
    OnlineRecognizer, OnlineRecognizerConfig,
};

/// Kokoro en-v0_19 speaker id. Voice order: `af`(0), `af_bella`, `af_nicole`,
/// `af_sarah`, `af_sky`, `am_adam`, `am_michael`(6), `bf_emma`, `bf_isabella`,
/// `bm_george`, `bm_lewis`(10). `am_michael` reads closest to a calm male
/// interviewer.
const KOKORO_SID: i32 = 6;
const KOKORO_SPEED: f32 = 1.0;
/// The whole pipeline runs at 16 kHz mono; both recognizers expect the same.
const SAMPLE_RATE: u32 = 16_000;

/// Where sherpa-onnx models live (Kokoro TTS, Parakeet STT, streaming Nemotron).
#[must_use]
pub fn models_dir() -> PathBuf {
    dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("gimme-a-chance")
        .join("models")
        .join("sherpa")
}

/// Locate the directory holding `marker` under `models_dir()/sub` — either the
/// directory itself or one level down (tar archives extract into a nested
/// folder like `parakeet/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/`). `sub` is a
/// relative path so callers can nest by language (e.g. `es/parakeet`).
fn find_model_root(sub: &Path, marker: &str) -> Option<PathBuf> {
    let base = models_dir().join(sub);
    if base.join(marker).exists() {
        return Some(base);
    }
    for entry in std::fs::read_dir(&base).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() && path.join(marker).exists() {
            return Some(path);
        }
    }
    None
}

/// Find the `.onnx` file in `dir` whose name starts with `prefix`, preferring
/// the int8-quantized variant when both exist (model archives name files like
/// `encoder.int8.onnx` or `encoder-epoch-99-avg-1-chunk-16-left-128.onnx`).
fn find_onnx(dir: &Path, prefix: &str) -> Result<String> {
    let entries = std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?;
    let mut fallback = None;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_onnx = Path::new(&name)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"));
        if name.starts_with(prefix) && is_onnx {
            if name.contains("int8") {
                return Ok(entry.path().to_string_lossy().into_owned());
            }
            fallback = Some(entry.path().to_string_lossy().into_owned());
        }
    }
    fallback.ok_or_else(|| anyhow!("no {prefix}*.onnx found in {}", dir.display()))
}

fn path_str(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

// ---------------------------------------------------------------------------
// Parakeet STT (offline, per VAD chunk)
// ---------------------------------------------------------------------------

/// Offline `NeMo` transducer for high-quality finals. The model differs by
/// language, but the sherpa-onnx API (encoder/decoder/joiner) does not — only the
/// model directory changes:
/// - **English** — Parakeet-TDT 0.6b v2 (int8), under `parakeet/`.
/// - **Spanish** — NVIDIA `Canary-180m-flash`, under `es/parakeet/`, PINNED to
///   Spanish via `src_lang`/`tgt_lang` = "es". The multilingual Parakeet-TDT v3 (a
///   transducer) is avoided: it auto-detects language per utterance and flipped
///   ~1/3 of Spanish utterances to English ("voy a usar la plataforma…" → "We use
///   the platform…"), and sherpa-onnx's offline TRANSDUCER config exposes no
///   language pin (verified in c-api.h). Canary is an attention enc-dec whose config
///   DOES take a source language, so it stays in Spanish AND keeps high accuracy. (A
///   monolingual `fast-conformer-es` was tried first but transcribed rioplatense
///   poorly — "Gull eschamp" for "voy a usar la plataforma".)
///
/// The `Mutex` is deliberate even though `OfflineRecognizer` is `Send + Sync`: it
/// serializes decodes so the dual-capture pipelines never run two passes
/// concurrently — on a 4-core laptop that halves each decode's speed and blows the
/// speculative window (measured: 8s of audio jumped from ~0.8s to ~2.1s under
/// contention).
pub struct ParakeetStt {
    recognizer: std::sync::Mutex<OfflineRecognizer>,
}

impl ParakeetStt {
    pub fn new(lang: Language) -> Result<Self> {
        let sub = lang.sherpa_subdir("parakeet");
        let root = find_model_root(&sub, "tokens.txt").ok_or_else(|| {
            anyhow!(
                "Parakeet ({}) model not found under {} — run scripts/fetch-models.ps1",
                lang.tag(),
                models_dir().join(&sub).display()
            )
        })?;
        let t0 = Instant::now();
        let mut config = OfflineRecognizerConfig::default();
        match lang {
            // English: Parakeet-TDT transducer (encoder/decoder/joiner).
            Language::English => {
                config.model_config.transducer = OfflineTransducerModelConfig {
                    encoder: Some(find_onnx(&root, "encoder")?),
                    decoder: Some(find_onnx(&root, "decoder")?),
                    joiner: Some(find_onnx(&root, "joiner")?),
                };
            }
            // Spanish: Canary attention enc-dec (encoder/decoder, no joiner), pinned
            // to Spanish so it can't auto-detect into English. `use_pnc` keeps the
            // punctuation+casing the LLM and the on-screen transcript expect.
            Language::Spanish => {
                config.model_config.canary = OfflineCanaryModelConfig {
                    encoder: Some(find_onnx(&root, "encoder")?),
                    decoder: Some(find_onnx(&root, "decoder")?),
                    src_lang: Some("es".into()),
                    tgt_lang: Some("es".into()),
                    use_pnc: true,
                };
            }
        }
        config.model_config.tokens = Some(path_str(&root.join("tokens.txt")));
        config.model_config.num_threads = 4;
        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| anyhow!("OfflineRecognizer::create failed (see sherpa-onnx logs)"))?;
        tracing::info!(
            load_ms = t0.elapsed().as_millis() as u64,
            model = %root.display(),
            "Parakeet (sherpa-onnx) loaded"
        );
        Ok(Self {
            recognizer: std::sync::Mutex::new(recognizer),
        })
    }

    /// Transcribe a chunk of 16 kHz mono f32 audio.
    #[tracing::instrument(skip(self, audio), fields(samples = audio.len()))]
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let recognizer = self
            .recognizer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let stream = recognizer.create_stream();
        stream.accept_waveform(SAMPLE_RATE as i32, audio);
        recognizer.decode(&stream);
        Ok(stream.get_result().map(|r| r.text).unwrap_or_default())
    }
}

/// Shared Parakeet instance for `lang`, loaded once per language per app run.
/// `None` means it isn't usable (models missing or load failed — already logged)
/// and the caller should fall back to the default engine chain. Each language has
/// its own `OnceLock`, so switching language lazily loads (then caches) that
/// language's model without disturbing the other.
pub fn parakeet(lang: Language) -> Option<&'static ParakeetStt> {
    static EN: OnceLock<Option<ParakeetStt>> = OnceLock::new();
    static ES: OnceLock<Option<ParakeetStt>> = OnceLock::new();
    let cell = match lang {
        Language::English => &EN,
        Language::Spanish => &ES,
    };
    cell.get_or_init(|| match ParakeetStt::new(lang) {
        Ok(p) => Some(p),
        Err(e) => {
            tracing::warn!(error = %e, lang = lang.tag(), "Parakeet unavailable");
            None
        }
    })
    .as_ref()
}

// ---------------------------------------------------------------------------
// Streaming STT (online Nemotron transducer, partial hypotheses)
// ---------------------------------------------------------------------------

/// Light online recognizer (streaming zipformer) for live partials. The
/// recognizer is `Send + Sync` with `&self` methods; each capture pipeline
/// creates its own [`sherpa_onnx::OnlineStream`] via [`Self::recognizer`],
/// feeds audio continuously, and reads partials.
pub struct StreamingStt {
    pub recognizer: OnlineRecognizer,
}

impl StreamingStt {
    pub fn new(lang: Language) -> Result<Self> {
        let sub = lang.sherpa_subdir("streaming");
        let root = find_model_root(&sub, "tokens.txt").ok_or_else(|| {
            anyhow!(
                "streaming ({}) model not found under {} — run scripts/fetch-models.ps1",
                lang.tag(),
                models_dir().join(&sub).display()
            )
        })?;
        let t0 = Instant::now();
        let mut config = OnlineRecognizerConfig::default();
        config.model_config.transducer.encoder = Some(find_onnx(&root, "encoder")?);
        config.model_config.transducer.decoder = Some(find_onnx(&root, "decoder")?);
        config.model_config.transducer.joiner = Some(find_onnx(&root, "joiner")?);
        config.model_config.tokens = Some(path_str(&root.join("tokens.txt")));
        config.model_config.num_threads = 2;
        config.decoding_method = Some("greedy_search".into());
        // Built-in endpointing replaces our VAD chunker on this path: rule2
        // finalizes after 1.2s of trailing silence post-speech (snappier than
        // the VAD's cut), rule3 caps run-on utterances. rule3 lowered from the
        // 20s default so the Parakeet second pass (~100ms/s of audio) always
        // fits inside the silence window the speculative decode exploits.
        config.enable_endpoint = true;
        // rule2 down from the 1.2s default: endpoints fire 300ms sooner after a
        // pause, and more utterances end on a natural pause (speculation-able)
        // instead of running into the rule3 cap mid-speech (where speculation
        // is impossible — there's no silence window before a forced cut).
        config.rule2_min_trailing_silence = 0.9;
        config.rule3_min_utterance_length = 10.0;
        let recognizer = OnlineRecognizer::create(&config)
            .ok_or_else(|| anyhow!("OnlineRecognizer::create failed (see sherpa-onnx logs)"))?;
        tracing::info!(
            load_ms = t0.elapsed().as_millis() as u64,
            model = %root.display(),
            "streaming model (sherpa-onnx) loaded"
        );
        Ok(Self { recognizer })
    }
}

/// Shared streaming recognizer for `lang`, loaded once per language per app run.
/// Same per-language `OnceLock` and `None` contract as [`parakeet`].
pub fn streaming(lang: Language) -> Option<&'static StreamingStt> {
    static EN: OnceLock<Option<StreamingStt>> = OnceLock::new();
    static ES: OnceLock<Option<StreamingStt>> = OnceLock::new();
    let cell = match lang {
        Language::English => &EN,
        Language::Spanish => &ES,
    };
    cell.get_or_init(|| match StreamingStt::new(lang) {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::warn!(error = %e, lang = lang.tag(), "streaming STT unavailable");
            None
        }
    })
    .as_ref()
}

// ---------------------------------------------------------------------------
// Kokoro TTS
// ---------------------------------------------------------------------------

/// Synthesize speech locally with Kokoro via sherpa-onnx.
///
/// Returns `Ok(None)` when the model isn't installed, so [`crate::tts`] falls
/// back to `OpenAI` cleanly instead of failing.
pub fn kokoro_tts(text: &str) -> Result<Option<Vec<u8>>> {
    static KOKORO: OnceLock<Option<OfflineTts>> = OnceLock::new();
    let Some(tts) = KOKORO.get_or_init(init_kokoro).as_ref() else {
        return Ok(None);
    };
    let t0 = Instant::now();
    let gen = GenerationConfig {
        sid: KOKORO_SID,
        speed: KOKORO_SPEED,
        ..Default::default()
    };
    let audio = tts
        .generate_with_config(text, &gen, None::<fn(&[f32], f32) -> bool>)
        .ok_or_else(|| anyhow!("kokoro synthesis failed (see sherpa-onnx logs)"))?;
    tracing::debug!(
        gen_ms = t0.elapsed().as_millis() as u64,
        sample_rate = audio.sample_rate(),
        samples = audio.samples().len(),
        "kokoro synthesized"
    );
    let wav =
        crate::cloud_stt::encode_wav_mono(audio.samples(), audio.sample_rate().cast_unsigned())?;
    Ok(Some(wav))
}

fn init_kokoro() -> Option<OfflineTts> {
    // Kokoro stays English-only: sherpa-onnx ships no Spanish voices, so Spanish
    // TTS goes through the OpenAI fallback in `crate::tts`.
    let Some(root) = find_model_root(Path::new("kokoro"), "model.onnx") else {
        tracing::info!(
            dir = %models_dir().join("kokoro").display(),
            "Kokoro model not present — TTS will use OpenAI (run scripts/fetch-models.ps1 for local)"
        );
        return None;
    };
    let t0 = Instant::now();
    // en-v0_19 needs no lexicon/dict (those are for the multi-lang variant).
    let config = OfflineTtsConfig {
        model: sherpa_onnx::OfflineTtsModelConfig {
            kokoro: OfflineTtsKokoroModelConfig {
                model: Some(path_str(&root.join("model.onnx"))),
                voices: Some(path_str(&root.join("voices.bin"))),
                tokens: Some(path_str(&root.join("tokens.txt"))),
                data_dir: Some(path_str(&root.join("espeak-ng-data"))),
                length_scale: 1.0,
                ..Default::default()
            },
            num_threads: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let Some(tts) = OfflineTts::create(&config) else {
        tracing::warn!(model = %root.display(), "Kokoro failed to load — TTS will use OpenAI");
        return None;
    };
    tracing::info!(
        load_ms = t0.elapsed().as_millis() as u64,
        model = %root.display(),
        "Kokoro TTS (sherpa-onnx) loaded"
    );
    Some(tts)
}
