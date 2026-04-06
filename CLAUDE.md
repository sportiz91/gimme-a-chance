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
3. Or run: `powershell.exe -ExecutionPolicy Bypass -File "C:\Users\SANTI\Desktop\webdev\scripts\lint-gimme.ps1"`

## Code Style

- Clippy pedantic is enabled in `Cargo.toml [lints]`
- `unsafe_code` is forbidden
- Use `f64::from(x)` instead of `x as f64` for lossless casts
- Use inline format args: `format!("{name}")` not `format!("{}", name)`
- Numeric cast warnings are allowed in `audio.rs` (audio processing needs them)

## Key Dependencies

- `whisper-rs 0.16` — STT (whisper.cpp bindings)
- `cpal 0.15` — audio capture
- `tauri 2` — desktop framework
- `tokio` — async runtime
- `anyhow` — error handling

## Architecture

```
main.rs → lib.rs → audio.rs      (mic capture + transcription loop)
                  → claude.rs     (Claude CLI integration)
                  → transcriber.rs (Whisper model loading + inference)
```

Frontend communicates with Rust via Tauri IPC (invoke/listen).
