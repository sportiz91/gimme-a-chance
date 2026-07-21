# gimme-a-chance — Claude Code Instructions

## Project

Real-time interview copilot. Tauri v2 (Rust backend) + vanilla HTML/JS frontend.

## Repo Layout

- `src-tauri/` — Rust backend (Cargo project)
- `dist/` — Frontend (plain HTML/JS, no build step; index + answer pop-out + manager)
- `scripts/` — Windows PowerShell helpers: dev/build/run/lint/test + model fetcher (see `scripts/README.md`)

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

- `whisper-rs 0.16` — local whisper.cpp STT (last-resort fallback)
- `sherpa-onnx 1.13` (optional, feature `sherpa`, official k2-fsa bindings) — on-device STT (Parakeet/Canary finals, streaming partials) + Kokoro TTS; prebuilt `shared` DLLs
- `cpal 0.15` — audio capture (WASAPI mic + loopback)
- `tauri 2` — desktop framework
- `tokio` — async runtime
- `anyhow` — error handling

## Architecture

```
main.rs → lib.rs → audio.rs        (capture pipelines: VAD chunking / streaming loop, AEC plumbing)
                  → stt.rs          (sherpa-onnx: Parakeet/Canary finals, streaming partials, Kokoro TTS — feature `sherpa`)
                  → cloud_stt.rs    (Groq Whisper cloud STT)
                  → transcriber.rs  (local whisper.cpp fallback)
                  → aec.rs/dtln.rs  (echo cancellation: AEC3 default, DTLN experimental)
                  → backend.rs      (OpenAI/Groq chat: answers, vision, agent, state refresh)
                  → agent.rs        (rolling interview transcript + Interview State doc)
                  → commands.rs     (Tauri commands: listen, ask, describe, manager, STT config)
                  → storage.rs      (SQLite session persistence via writer thread)
                  → capture.rs/clipboard.rs (screen capture + clipboard ingestion)
                  → telemetry.rs/metrics.rs/latency.rs/crashlog.rs (JSONL logs, gauges, panic dumps)
                  → context_meter.rs (o200k token estimate anchored by API usage)
                  → tts.rs          (simulate-interviewer TTS: Kokoro → OpenAI fallback)
```

Frontend communicates with Rust via Tauri IPC (invoke/listen).

## The `sherpa` Feature (on-device STT/TTS)

- Dev loop: `.\scripts\dev.ps1` builds with `sherpa` by DEFAULT (`-CloudOnly`
  skips it for a faster pure-cloud compile).
- STT engine selection (`commands.rs::start_listening`) is driven from the UI
  (segmented ☁️ Nube / 💻 Local control + ⚡ parciales checkbox, persisted in
  localStorage, pushed via the `set_stt_config` command; read once per Listen).
  Flipping either control — or the language selector — mid-session
  auto-restarts the live Listen from JS (stop → 600ms drain → start; the old
  pipelines share `is_listening` with the new ones, hence the pause) so it
  applies immediately. No env var involved:
  - Local OFF (default) → Groq cloud (`whisper-large-v3-turbo`).
  - Local ON + partials ON → hybrid streaming (FastConformer/Kroko live
    partials + Parakeet/Canary finals — the production interview engine).
  - Local ON + partials OFF → Parakeet/Canary per VAD chunk (no zipformer).
  - Cloud with no `GROQ_API_KEY` falls back to Parakeet/Canary (if built +
    fetched), then local whisper. Mid-session request failures still fall back
    to local whisper (always loaded — never a model load on the hot path).
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
