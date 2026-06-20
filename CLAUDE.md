# gimme-a-chance — Claude Code Instructions

## Project

Real-time interview copilot. Tauri v2 (Rust backend) + vanilla HTML/JS frontend.

## Repo Layout

- `src-tauri/` — Rust backend (Cargo project)
- `dist/` — Frontend (single index.html, no build step)
- `scripts/` — Setup scripts for Ubuntu 24.04

## Build (Windows)

```powershell
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:PATH = "C:\Program Files\CMake\bin;$env:PATH"
cd src-tauri
cargo build
```

## Before Every Commit

1. `cargo fmt --all` — format code
2. `cargo clippy --all-targets -- -D warnings` — zero warnings policy
3. Or run: `.\scripts\lint.ps1` (see `scripts/README.md` for all helpers)

## Code Style

- Clippy pedantic is enabled in `Cargo.toml [lints]`
- `unsafe_code` is forbidden
- Use `f64::from(x)` instead of `x as f64` for lossless casts
- Use inline format args: `format!("{name}")` not `format!("{}", name)`
- Numeric cast warnings are allowed in `audio.rs` (audio processing needs them)

## Key Dependencies

- `whisper-rs 0.16` — STT (whisper.cpp bindings)
- `sherpa-rs 0.6` (optional, feature `sherpa`) — on-device Parakeet STT + Kokoro TTS
- `cpal 0.15` — audio capture
- `tauri 2` — desktop framework
- `tokio` — async runtime
- `anyhow` — error handling

## Architecture

```
main.rs → lib.rs → audio.rs      (mic capture + transcription loop)
                  → claude.rs     (Claude CLI integration)
                  → transcriber.rs (Whisper model loading + inference)
                  → cloud_stt.rs  (Groq Whisper cloud STT)
                  → stt.rs        (sherpa-onnx: Parakeet STT + Kokoro TTS, feature `sherpa`)
                  → tts.rs        (simulate-interviewer TTS: Kokoro → OpenAI fallback)
```

Frontend communicates with Rust via Tauri IPC (invoke/listen).

## The `sherpa` Feature (on-device STT/TTS)

- Dev loop: `.\scripts\dev-sherpa.ps1` (defaults to the hybrid streaming engine;
  `-Engine sherpa` for chunked Parakeet).
- STT engine selection (`commands.rs`): `GIMME_STT_ENGINE` = `streaming` (hybrid:
  FastConformer live partials + Parakeet finals — the production engine) |
  `sherpa` (Parakeet per VAD chunk) | `whisper` (force local whisper) | unset
  (Groq cloud). Local whisper is always the grace fallback.
- Hybrid design (`audio.rs::streaming_loop`): a light online model only powers
  ephemeral `transcription-partial` events; on endpoint the buffered utterance is
  re-decoded with offline Parakeet for the final (~100ms per second of audio).
  Don't swap in heavy (0.6b) online models for partials — Nemotron saturated the
  CPU with dual capture and partials lagged behind real time. Finals and partials
  are RMS-gated: without the gate, faint speaker-bleed into the mic produces
  duplicated `[You]` lines.
- Models live in `%APPDATA%\gimme-a-chance\models\sherpa\{kokoro,parakeet,streaming,silero}`,
  fetched by `scripts/fetch-models.ps1` (Windows tar lacks bzip2 — the script falls
  back to WSL tar).
- Uses the OFFICIAL `sherpa-onnx` crate (k2-fsa) with `shared` libs. Do NOT switch
  to the `static` feature: the static prebuilts are /MT and multiply-define CRT
  symbols against whisper.cpp (LNK1169). History: the deprecated `sherpa-rs` crate
  force-linked msvcrtd into debug builds → "Debug Assertion Failed:
  _osfile(fh) & FOPEN" aborts; that's why it was replaced.
- If a rebuild fails with "Acceso denegado (os error 5)" on the exe, a previous
  instance is still running: `Stop-Process -Name gimme-a-chance -Force`.

## Acoustic Echo Cancellation (`aec.rs`, `dtln.rs`)

In "both" capture the interviewer's audio leaks from the headset earpiece into
the mic, so the mic pipeline transcribes the interviewer's words as ghost
`[You]` lines. The mic pipeline cancels this echo using the loopback as the
reference signal (routed over a channel from the interviewer pipeline). Engine
chosen by `GIMME_AEC_ENGINE`:

- **`aec3`** (default) — pure-Rust WebRTC AEC3. Lightweight, doesn't touch the
  interviewer pipeline's latency, kills the bleed cleanly in clean-turn speech.
- **`dtln`** — DTLN-aec (deep learning, Microsoft AEC Challenge) on the pure-Rust
  `tract` ONNX runtime. **Experimental, NOT recommended.** Measured: per-hop p50
  ~4ms but spikes to 29ms → backlog grows to ~1.7s under load (drops the user's
  audio), and the cleaned audio has artifacts that degrade the final ("hello" →
  "canoe"). Its CPU spikes also risk slowing the interviewer STT. Models aren't
  fetched by any script — they were hand-converted from breizhn/DTLN-aec TFLite
  to ONNX with tf2onnx and live in `%APPDATA%\gimme-a-chance\models\dtln`.

**Double-talk (both speaking at once) is intentionally left imperfect.** Real
interviews are turn-based: interviewer talks (you silent → bleed cancelled
cleanly) then you answer (interviewer silent → your mic is clean). Simultaneous
double-talk is rare and brief, and the info lost there is your own — not worth
the cost (DTLN proved that). AEC3 + the text-level bleed dedup cover the real
cases; don't reopen double-talk optimization without a concrete need.
