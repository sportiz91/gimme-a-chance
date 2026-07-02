use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("Audio error: {0}")]
    Audio(String),

    #[allow(dead_code)]
    #[error("Transcription error: {0}")]
    Transcription(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("Vision error: {0}")]
    Vision(String),

    #[error("Clipboard error: {0}")]
    Clipboard(String),

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_str(self.to_string().as_ref())
    }
}
