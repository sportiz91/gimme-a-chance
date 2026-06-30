//! The transcription + answer language selected in the UI.
//!
//! Defaults to English; the user switches to Spanish from the overlay's language
//! dropdown. The choice takes effect on the next "Listen" (the STT engine is
//! rebuilt per session) and on the next answer (prompts are rebuilt per turn), so
//! no restart is needed. On-device Spanish models are loaded lazily on first use
//! and cached for the rest of the run (see `crate::stt`), so switching back and
//! forth never reloads.

use std::path::PathBuf;

/// A language the copilot can transcribe speech in and answer in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Language {
    #[default]
    English,
    Spanish,
}

impl Language {
    /// Parse the UI/string form (`"english"`/`"en"` | `"spanish"`/`"es"`).
    /// Unknown input yields `None` so the command can reject it.
    #[must_use]
    pub fn from_tag(s: &str) -> Option<Self> {
        match s {
            "english" | "en" => Some(Self::English),
            "spanish" | "es" => Some(Self::Spanish),
            _ => None,
        }
    }

    /// The lowercase tag the UI persists in localStorage and the commands exchange.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::English => "english",
            Self::Spanish => "spanish",
        }
    }

    /// ISO-639-1 code passed to Whisper (Groq cloud + local whisper-rs).
    #[must_use]
    pub fn whisper_code(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::Spanish => "es",
        }
    }

    /// Subdirectory under `models/sherpa/` holding this language's copy of an
    /// on-device model. English keeps the original flat layout so models already
    /// fetched keep working without migration; Spanish nests one level under `es/`.
    #[must_use]
    pub fn sherpa_subdir(self, model: &str) -> PathBuf {
        match self {
            Self::English => PathBuf::from(model),
            Self::Spanish => PathBuf::from("es").join(model),
        }
    }
}
