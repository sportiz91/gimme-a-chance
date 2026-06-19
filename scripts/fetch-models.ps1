# fetch-models.ps1 — download sherpa-onnx models for the `sherpa` feature
# (local Kokoro TTS + Parakeet STT + Nemotron streaming STT). Run once before
# `cargo build --features sherpa`.
#
# Target layout (read by src-tauri/src/stt.rs::models_dir()):
#   %APPDATA%\gimme-a-chance\models\sherpa\kokoro\      (model.onnx, voices.bin, tokens.txt, espeak-ng-data)
#   %APPDATA%\gimme-a-chance\models\sherpa\parakeet\    (encoder/decoder/joiner .onnx, tokens.txt)
#   %APPDATA%\gimme-a-chance\models\sherpa\silero\      (silero_vad.onnx)
#
# NOTE: sherpa-onnx publishes models as release archives. Confirm the latest
# filenames at:
#   TTS  : https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
#   ASR  : https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
#   VAD  : https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models  (silero_vad.onnx)
# Update the $*_URL variables below if they’ve changed, then re-run.

$ErrorActionPreference = "Stop"
$Root = Join-Path $env:APPDATA "gimme-a-chance\models\sherpa"

# Defaults known at time of writing — VERIFY against the release pages above.
$KOKORO_URL  = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2"
$PARAKEET_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
# Partials-only model (finals are re-decoded with Parakeet). FastConformer
# 480ms int8: NeMo-family accuracy at ~115M params — 0.6b models (Nemotron)
# lag on dual capture, icefall zipformer-2023 is fast but inaccurate.
$STREAMING_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8.tar.bz2"
$SILERO_URL  = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

function Get-Archive($url, $destSubdir) {
    $dest = Join-Path $Root $destSubdir
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

Write-Host "Fetching sherpa-onnx models into $Root" -ForegroundColor Green
Get-Archive $KOKORO_URL  "kokoro"
Get-Archive $PARAKEET_URL "parakeet"
Get-Archive $STREAMING_URL "streaming"
Get-Archive $SILERO_URL  "silero"
Write-Host "Done. Now build with: cargo build --features sherpa" -ForegroundColor Green
