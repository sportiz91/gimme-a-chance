# fetch-models.ps1 — download sherpa-onnx models for the `sherpa` feature
# (local Kokoro TTS + Parakeet STT + streaming STT) plus the local whisper fallback.
# Run once per language before `cargo build --features sherpa`.
#
#   .\scripts\fetch-models.ps1            # English (default)
#   .\scripts\fetch-models.ps1 -Lang es   # Spanish
#
# Target layout (read by src-tauri/src/stt.rs::models_dir() + transcriber.rs):
#   English:
#     %APPDATA%\gimme-a-chance\models\sherpa\kokoro\   (model.onnx, voices.bin, tokens.txt, espeak-ng-data)
#     %APPDATA%\gimme-a-chance\models\sherpa\parakeet\ (encoder/decoder/joiner .onnx, tokens.txt)
#     %APPDATA%\gimme-a-chance\models\sherpa\streaming\(encoder/decoder/joiner .onnx, tokens.txt)
#     %APPDATA%\gimme-a-chance\models\ggml-base.en.bin (whisper fallback — fetched manually, see note)
#   Spanish (nested under es\ so English stays put — no migration):
#     %APPDATA%\gimme-a-chance\models\sherpa\es\parakeet\  (Canary-180m-flash, pinned to es — offline finals)
#     %APPDATA%\gimme-a-chance\models\sherpa\es\streaming\ (Kroko streaming zipformer, Spanish)
#     %APPDATA%\gimme-a-chance\models\ggml-base.bin        (multilingual whisper fallback)
#   Shared:
#     %APPDATA%\gimme-a-chance\models\sherpa\silero\   (silero_vad.onnx — language-agnostic)
#
# NOTE: Kokoro (sherpa-onnx) ships NO Spanish voices, so Spanish TTS uses the
# OpenAI fallback in src-tauri/src/tts.rs — there is no Spanish Kokoro to fetch.
#
# Confirm the latest filenames at:
#   ASR : https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
#   TTS : https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
# Update the $*_URL variables below if they've changed, then re-run.

param(
    [ValidateSet("en", "es")]
    [string]$Lang = "en"
)

$ErrorActionPreference = "Stop"
$SherpaRoot = Join-Path $env:APPDATA "gimme-a-chance\models\sherpa"
$ModelsRoot = Join-Path $env:APPDATA "gimme-a-chance\models"   # whisper.cpp lives here

$ASR = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

# --- English (default) ---
$KOKORO_URL       = "$ASR/kokoro-en-v0_19.tar.bz2"
$PARAKEET_EN_URL  = "$ASR/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
# Partials-only model (finals are re-decoded with Parakeet). FastConformer 480ms
# int8: NeMo-family accuracy at ~115M params — 0.6b models lag on dual capture.
$STREAMING_EN_URL = "$ASR/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8.tar.bz2"

# --- Spanish ---
# Offline finals: NVIDIA Canary-180m-flash (en/es/de/fr), PINNED to Spanish in code
# via src_lang/tgt_lang="es" (see src-tauri/src/stt.rs). We do NOT use Parakeet-TDT
# v3: it auto-detects language per utterance and flips ~1/3 of Spanish utterances to
# English, and sherpa-onnx's offline transducer config has no language pin (verified
# in c-api.h). A monolingual fast-conformer-es was tried but its accuracy on
# rioplatense audio was poor; Canary keeps ~3.2% WER AND stays in Spanish.
$FINALS_ES_URL    = "$ASR/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2"
# Live partials: Kroko streaming zipformer (Spanish, ~124 MB). Light + low-latency
# — respects the "small+fast for ephemeral partials" rule (a 0.6b online model
# saturates the CPU under dual capture). License: CC-BY-SA-4.0 (fine for personal
# use; revisit if this is ever shipped as a closed-source product).
$STREAMING_ES_URL = "$ASR/sherpa-onnx-streaming-zipformer-es-kroko-2025-08-06.tar.bz2"
# Multilingual whisper for the Spanish offline fallback (the English `base.en`
# can't decode Spanish). ~142 MB.
$WHISPER_ES_URL   = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"

# Shared, language-agnostic VAD.
$SILERO_URL       = "$ASR/silero_vad.onnx"

# Download $url into $SherpaRoot\$destSubdir, extracting .tar.bz2 archives.
function Get-Archive($url, $destSubdir) {
    $dest = Join-Path $SherpaRoot $destSubdir
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $file = Join-Path $dest (Split-Path $url -Leaf)
    Write-Host "Downloading $url" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $url -OutFile $file
    if ($file -like "*.tar.bz2") {
        Write-Host "Extracting $file" -ForegroundColor DarkGray
        tar -xjf $file -C $dest    # Windows 10+ ships bsdtar
        if ($LASTEXITCODE -ne 0) {
            # bsdtar shells out to bzip2.exe for .bz2, which vanilla Windows
            # lacks — retry through WSL's GNU tar.
            Write-Host "tar.exe failed; retrying via WSL tar" -ForegroundColor Yellow
            $wfile = wsl.exe wslpath -u $file.Replace('\', '/')
            $wdest = wsl.exe wslpath -u $dest.Replace('\', '/')
            wsl.exe -e tar -xjf $wfile -C $wdest
            if ($LASTEXITCODE -ne 0) { throw "extraction failed for $file" }
        }
        Remove-Item $file
    }
}

# Download $url (no extraction) into an arbitrary directory — for the whisper .bin
# that lives in models\ rather than models\sherpa\.
function Get-File($url, $destDir) {
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    $file = Join-Path $destDir (Split-Path $url -Leaf)
    Write-Host "Downloading $url" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $url -OutFile $file
}

if ($Lang -eq "en") {
    Write-Host "Fetching ENGLISH sherpa-onnx models into $SherpaRoot" -ForegroundColor Green
    Get-Archive $KOKORO_URL       "kokoro"
    Get-Archive $PARAKEET_EN_URL  "parakeet"
    Get-Archive $STREAMING_EN_URL "streaming"
    Get-Archive $SILERO_URL       "silero"
    Write-Host "Done (English). The whisper fallback (ggml-base.en.bin) is fetched on first run if missing." -ForegroundColor Green
} else {
    Write-Host "Fetching SPANISH models into $SherpaRoot\es and $ModelsRoot" -ForegroundColor Green
    Get-Archive $FINALS_ES_URL    "es\parakeet"
    Get-Archive $STREAMING_ES_URL "es\streaming"
    Get-Archive $SILERO_URL       "silero"        # shared VAD (idempotent)
    Get-File    $WHISPER_ES_URL   $ModelsRoot     # multilingual whisper fallback
    Write-Host "Done (Spanish). Note: Spanish TTS uses OpenAI (no Spanish Kokoro)." -ForegroundColor Green
}

Write-Host "Now build with: cargo build --features sherpa" -ForegroundColor Green
