# gimme-a-chance

Real-time interview copilot for Windows. It listens to both sides of a call
(your microphone + system audio), transcribes live, keeps a rolling interview
context, and answers on demand through an AI agent — inside an always-on-top
overlay that is **invisible to screen sharing** (Meet/Zoom/Teams see nothing).

Built with Tauri v2 (Rust backend, plain HTML/JS frontend). Windows-only —
macOS/Linux have never been tested and are out of scope.

## Features

- **Dual capture** — microphone (you) + WASAPI loopback (interviewer), labeled
  per speaker, with AEC3 echo cancellation so headset bleed doesn't ghost into
  your channel.
- **Three STT modes**, switchable live from the toolbar (a flip mid-Listen
  auto-restarts the session):
  - ☁️ **Nube** (default): Groq `whisper-large-v3-turbo`.
  - 💻 **Local + ⚡ parciales**: on-device hybrid — live gray partial
    hypotheses while the speaker talks + high-quality finals (Parakeet for
    English, Canary for Spanish). Finals land at ~0ms after the endpoint
    thanks to speculative decoding. Audio never leaves the machine.
  - 💻 **Local** without partials: finals only, per VAD chunk.
- **Agent answers** (`Ctrl+Shift+Space`): an agent reads the whole interview so
  far (transcript + screen describes + clipboard) and helps with what's needed
  right now. Ask box for explicit questions. Normal or 🪨 caveman-terse
  response style.
- **Vision** (`Ctrl+Shift+Enter` to queue screenshots, `Ctrl+Shift+1` to
  describe): transcribes exercise statements / code from the screen into
  context.
- **Clipboard ingestion**: every copy becomes context (toggleable);
  `Ctrl+Shift+V` grabs the current selection via UI Automation without
  touching the target app.
- **Session recording**: every transcript line, describe, clip and Q&A is
  persisted to a local SQLite file; the 🗂 manager browses, annotates, exports
  to Markdown, and can inject a past session's context into a live interview.
- **Screen-share invisibility**: content-protected overlays, parked off-screen
  instead of hidden (`Ctrl+Shift+H`), no OS resize cursors — the share sees a
  normal desktop.
- English and Spanish, switchable live.

## Requirements

| Tool | Why | Install |
|---|---|---|
| Windows 10/11 x64 | WASAPI capture, WebView2 (ships with Win11) | — |
| Rust (via rustup) | toolchain is pinned by `rust-toolchain.toml`, rustup picks it up automatically | <https://rustup.rs> |
| LLVM | `bindgen` for whisper.cpp | default lookup `C:\Program Files\LLVM\bin`, override with `$env:LIBCLANG_PATH` |
| CMake | builds whisper.cpp | default lookup `C:\Program Files\CMake\bin` |
| Tauri CLI | only for the dev loop (`dev.ps1`) | `cargo install tauri-cli --version "^2"` |

## API keys

Two keys, both read from the environment on first run and then **seeded into
the Windows Credential Manager automatically** — after one successful run you
never need the env vars again:

```powershell
$env:OPENAI_API_KEY = "sk-..."   # required: answers, vision, agent
$env:GROQ_API_KEY   = "gsk_..."  # optional: cloud STT (without it, STT falls
                                 # back to on-device models, then local whisper)
.\scripts\dev.ps1                # or build.ps1 + run.ps1
```

## On-device models (optional but recommended)

The 💻 Local mode needs models fetched once per language (~1.8 GB total for
both, into `%APPDATA%\gimme-a-chance\models\`):

```powershell
.\scripts\fetch-models.ps1            # English (Parakeet + streaming + Kokoro TTS + VAD)
.\scripts\fetch-models.ps1 -Lang es   # Spanish (Canary + Kroko streaming)
```

Without them the Local switch simply degrades to the cloud chain with a warn
in the logs. The local whisper fallback model downloads itself on first run.

## Build & run

```powershell
# Dev loop (hot reload; full build with on-device STT by default)
.\scripts\dev.ps1
.\scripts\dev.ps1 -CloudOnly          # faster compile, cloud STT only

# Release ("production": the binary you double-click before a real interview)
.\scripts\build.ps1
.\scripts\run.ps1                     # warns if the exe is older than the last commit
```

The release exe lives at `src-tauri\target\release\gimme-a-chance.exe` — make
a shortcut to it. The sherpa/onnxruntime DLLs next to it **must travel with
it**; an exe copied alone silently degrades to cloud STT.

```powershell
# Before committing
.\scripts\lint.ps1                    # cargo fmt + clippy -D warnings (plain + sherpa)
.\scripts\test.ps1
```

## Using it

1. Pick capture source (🎙️+🔊 *Both* for a real interview), language, and STT
   mode, then press **Listen**.
2. Read the live transcript; press `Ctrl+Shift+Space` (or 🤖 Agent) whenever
   you want help — the agent decides what's needed from the full context.
3. Queue screenshots of exercise statements with `Ctrl+Shift+Enter`, describe
   them with `Ctrl+Shift+1`; copy anything to ingest it as context.
4. `Ctrl+Shift+H` hides/shows all overlays; the answer pop-out keeps working
   while the main window is hidden.
5. The **?** toolbar button lists every shortcut. The 🗂 button opens the
   session manager.

## Where your data lives

| What | Where |
|---|---|
| Session DB (transcripts, Q&A) | `%APPDATA%\gimme-a-chance\sessions.sqlite` — one file shared by debug and release builds |
| Models | `%APPDATA%\gimme-a-chance\models\` |
| Logs (JSONL, weekly rotation) | debug: `<repo>\logs\` · release: `%LOCALAPPDATA%\gimme-a-chance\logs\` |
| API keys | Windows Credential Manager (`gimme-a-chance` service) |

## Troubleshooting

- **Build fails: `` failed to remove file `gimme-a-chance.exe` `` (os error 5)**
  — a previous instance still runs:
  `Stop-Process -Name gimme-a-chance -Force`
- **`libclang` / CMake not found** — install LLVM + CMake or point
  `$env:LIBCLANG_PATH` / `$env:PATH` at them (the scripts do this for the
  default install paths).
- **Local mode transcribes nothing / falls back to cloud** — models not
  fetched, or the exe was separated from its DLLs. Check the JSONL log: it
  says exactly which engine each capture pipeline started with.
- **Overlay shows as a black box in a screen share** — never happens with
  the shipped binary (hide/show is implemented as off-screen parking
  precisely to avoid it); if you fork the window handling, read the comments
  in `lib.rs` first.

## License

Personal project, all rights reserved. Built for personal use; if you clone
it, use it responsibly and in accordance with your local rules and the terms
of whatever meeting you're in.
