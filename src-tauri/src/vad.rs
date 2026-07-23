//! Voice Activity Detection chunker.
//!
//! Feeds 16kHz mono f32 audio through WebRTC's VAD in fixed-size frames and
//! emits variable-length chunks cut on detected silence (not on wall time).
//! This replaces the previous approach of cutting every 5s regardless of
//! content, which would slice words at chunk boundaries.
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use webrtc_vad::{SampleRate, Vad, VadMode};

pub const FRAME_MS: usize = 30;
pub const SAMPLE_RATE_HZ: usize = 16_000;
pub const FRAME_SAMPLES: usize = SAMPLE_RATE_HZ * FRAME_MS / 1000; // 480

/// How much audio preceding the first detected voice frame to prepend to the
/// emitted chunk. Gives whisper a tiny bit of context so the first word isn't
/// transcribed cold.
pub const PRE_SPEECH_PAD_MS: usize = 300;
/// How much consecutive silence after voice before we flush the chunk.
/// Too short → cuts mid-sentence. Too long → feels laggy.
pub const POST_SPEECH_PAD_MS: usize = 500;
/// Minimum voiced duration to bother transcribing (filters out blips/clicks).
/// Raised from 300ms: real interview utterances run longer, while most VAD
/// false-positives (clicks, coughs, "um") are shorter — so this drops a lot of
/// the `[BLANK_AUDIO]` garbage before it ever reaches whisper.
pub const MIN_CHUNK_MS: usize = 700;
/// Hard cap on chunk duration so one long sentence without pauses still flushes.
/// Lowered 15s→8s: with fast cloud STT (Groq ~0.4s) shorter caps cut the worst-case
/// lag during continuous speech without saturating the transcription step.
pub const MAX_CHUNK_MS: usize = 8_000;

const PRE_SPEECH_PAD_SAMPLES: usize = SAMPLE_RATE_HZ * PRE_SPEECH_PAD_MS / 1000;
const POST_SPEECH_PAD_FRAMES: usize = POST_SPEECH_PAD_MS / FRAME_MS;
const MIN_CHUNK_SAMPLES: usize = SAMPLE_RATE_HZ * MIN_CHUNK_MS / 1000;
const MAX_CHUNK_SAMPLES: usize = SAMPLE_RATE_HZ * MAX_CHUNK_MS / 1000;

#[derive(Debug, Clone, Copy)]
enum State {
    Idle,
    InVoice,
    TrailingSilence(usize), // count of consecutive silence frames
}

pub struct VadChunker {
    vad: Vad,
    state: State,
    // Rolling pre-roll buffer kept while Idle so we can prepend a small lead-in
    // when voice starts mid-frame.
    pre_roll: Vec<f32>,
    // Accumulator for the current voiced chunk.
    chunk: Vec<f32>,
    // Scratch for i16 conversion (VAD expects i16 PCM).
    i16_scratch: Vec<i16>,
}

pub enum ChunkAction {
    /// Keep accumulating, nothing to send yet.
    Continue,
    /// Chunk boundary reached; caller should send this buffer to whisper.
    Emit(Vec<f32>),
}

impl VadChunker {
    #[must_use]
    pub fn new() -> Self {
        let mut vad = Vad::new();
        vad.set_sample_rate(SampleRate::Rate16kHz);
        // Aggressive mode: fewer false positives on keyboard clicks, breathing,
        // and ambient noise. Quality mode was triggering on every background sound.
        vad.set_mode(VadMode::Aggressive);
        Self {
            vad,
            state: State::Idle,
            pre_roll: Vec::with_capacity(PRE_SPEECH_PAD_SAMPLES + FRAME_SAMPLES),
            chunk: Vec::with_capacity(MAX_CHUNK_SAMPLES + FRAME_SAMPLES),
            i16_scratch: vec![0i16; FRAME_SAMPLES],
        }
    }

    /// Is the chunker currently inside an utterance — voice heard, chunk not
    /// yet flushed? The trailing-silence countdown still counts: those words
    /// sit in the un-emitted chunk, so their transcription is still in flight.
    #[must_use]
    pub fn in_speech(&self) -> bool {
        !matches!(self.state, State::Idle)
    }

    /// Feed exactly one frame of 16kHz mono f32 samples (`FRAME_SAMPLES` long).
    /// Returns an action: either keep accumulating or emit the finished chunk.
    pub fn push_frame(&mut self, frame_f32: &[f32]) -> ChunkAction {
        debug_assert_eq!(frame_f32.len(), FRAME_SAMPLES);

        // Convert to i16 into reusable scratch (no alloc).
        for (dst, src) in self.i16_scratch.iter_mut().zip(frame_f32.iter()) {
            let scaled =
                (src * f32::from(i16::MAX)).clamp(f32::from(i16::MIN), f32::from(i16::MAX));
            *dst = scaled as i16;
        }

        let is_voice = self
            .vad
            .is_voice_segment(&self.i16_scratch)
            .unwrap_or(false);

        match self.state {
            State::Idle => {
                self.pre_roll.extend_from_slice(frame_f32);
                // Keep the pre-roll bounded — discard oldest samples beyond PRE_SPEECH_PAD.
                if self.pre_roll.len() > PRE_SPEECH_PAD_SAMPLES {
                    let excess = self.pre_roll.len() - PRE_SPEECH_PAD_SAMPLES;
                    self.pre_roll.drain(..excess);
                }
                if is_voice {
                    // Seed the chunk with the pre-roll so the first word has context.
                    self.chunk.clear();
                    self.chunk.extend_from_slice(&self.pre_roll);
                    self.pre_roll.clear();
                    self.state = State::InVoice;
                }
                ChunkAction::Continue
            }
            State::InVoice => {
                self.chunk.extend_from_slice(frame_f32);
                if self.chunk.len() >= MAX_CHUNK_SAMPLES {
                    // Hard cap hit — flush even mid-speech so whisper can catch up.
                    self.state = State::Idle;
                    return ChunkAction::Emit(std::mem::take(&mut self.chunk));
                }
                if !is_voice {
                    self.state = State::TrailingSilence(1);
                }
                ChunkAction::Continue
            }
            State::TrailingSilence(n) => {
                self.chunk.extend_from_slice(frame_f32);
                if is_voice {
                    // Voice resumed — back to InVoice.
                    self.state = State::InVoice;
                    return ChunkAction::Continue;
                }
                if n >= POST_SPEECH_PAD_FRAMES {
                    // Enough silence — flush if long enough, discard otherwise.
                    self.state = State::Idle;
                    if self.chunk.len() >= MIN_CHUNK_SAMPLES {
                        return ChunkAction::Emit(std::mem::take(&mut self.chunk));
                    }
                    self.chunk.clear();
                    return ChunkAction::Continue;
                }
                self.state = State::TrailingSilence(n + 1);
                ChunkAction::Continue
            }
        }
    }
}
