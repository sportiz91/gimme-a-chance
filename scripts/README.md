# scripts/

Utility scripts for building, running, linting, and testing **gimme-a-chance**.

## Prerequisites (Windows)

- **Rust** (via rustup) — cargo must be on `PATH` or at `%USERPROFILE%\.cargo\bin`
- **LLVM** — for `bindgen` used by `whisper-rs`. Default lookup: `C:\Program Files\LLVM\bin`. Override with `$env:LIBCLANG_PATH`.
- **CMake** — to build `whisper.cpp` from source. Default lookup: `C:\Program Files\CMake\bin`.
- **Tauri CLI** (for `dev.ps1`) — `cargo install tauri-cli --version "^2.0"`

## Windows — PowerShell

| Script | What it does |
|---|---|
| `dev.ps1` | `cargo tauri dev` — live dev mode |
| `build.ps1` | `cargo build --release` — optimized binary |
| `run.ps1` | Run the release binary (requires `build.ps1` first) |
| `lint.ps1` | `cargo fmt` + `cargo clippy -D warnings` — matches CI |
| `test.ps1` | `cargo test` |

Run from the repo root:

```powershell
.\scripts\dev.ps1
```

Or from anywhere:

```powershell
powershell.exe -ExecutionPolicy Bypass -File C:\path\to\gimme-a-chance\scripts\dev.ps1
```

## Linux / Ubuntu 24.04

Install the system libraries Tauri v2 + audio capture need:

```bash
sudo apt-get install -y \
    libwebkit2gtk-4.1-dev \
    libappindicator3-dev \
    librsvg2-dev \
    patchelf \
    libssl-dev \
    libgtk-3-dev \
    libsoup-3.0-dev \
    libjavascriptcoregtk-4.1-dev \
    libasound2-dev \
    build-essential \
    pkg-config \
    cmake \
    libglib2.0-dev \
    clang
```

Then install Rust via [rustup](https://rustup.rs) and the Tauri CLI:

```bash
cargo install tauri-cli --version "^2.0"
```

Finally, download the Whisper model:

```bash
mkdir -p ~/.local/share/gimme-a-chance/models
curl -L -o ~/.local/share/gimme-a-chance/models/ggml-base.en.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

Build & run:

```bash
cd src-tauri && cargo tauri dev
```
