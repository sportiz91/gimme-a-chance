# dev.ps1 — Run gimme-a-chance in live dev mode (cargo tauri dev)
#
# Builds with the `sherpa` feature by DEFAULT so the UI's 💻 local / ⚡ partials
# switches (on-device STT) and AEC3 actually work. -CloudOnly skips it for a
# faster pure-cloud build (the local switch then degrades to Groq with a warn).
#
# Usage:
#   .\scripts\dev.ps1                           (full build: sherpa on-device STT/TTS)
#   .\scripts\dev.ps1 -CloudOnly                (skip sherpa — faster compile, cloud STT only)
#   .\scripts\dev.ps1 -Features tracy           (extra features appended: tracy, flame)
#   .\scripts\dev.ps1 -Features tracy,flame     (both at once)
#   powershell.exe -ExecutionPolicy Bypass -File scripts\dev.ps1 -Features tracy

[CmdletBinding()]
param(
    [string]$Features,
    [switch]$CloudOnly
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot

# LLVM (bindgen / whisper-rs). Override via $env:LIBCLANG_PATH.
if (-not $env:LIBCLANG_PATH) {
    $env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
}

# CMake (whisper.cpp build). Prepend if present and not already in PATH.
$cmakeBin = "C:\Program Files\CMake\bin"
if ((Test-Path $cmakeBin) -and ($env:PATH -notlike "*$cmakeBin*")) {
    $env:PATH = "$cmakeBin;$env:PATH"
}

# Cargo bin. Prepend if present and not already in PATH.
$cargoBin = "$env:USERPROFILE\.cargo\bin"
if ((Test-Path $cargoBin) -and ($env:PATH -notlike "*$cargoBin*")) {
    $env:PATH = "$cargoBin;$env:PATH"
}

Set-Location $RepoRoot

# API-mode secrets: pull from the canonical WSL ~/.claude/.env (single source of
# truth — never hardcoded here, never committed). The app reads these from its
# environment and, on first run, seeds them into the Windows Credential Manager
# so later runs work even without this script. Skips any key already set.
foreach ($k in @('GROQ_API_KEY', 'OPENAI_API_KEY')) {
    if (-not (Get-Item "env:$k" -ErrorAction SilentlyContinue)) {
        $val = (wsl.exe bash -lc "grep -E '^$k=' ~/.claude/.env 2>/dev/null | head -1 | cut -d= -f2-").Trim().Trim('"').Trim("'")
        if ($val) {
            Set-Item -Path "env:$k" -Value $val
            Write-Host "Loaded $k from WSL ~/.claude/.env" -ForegroundColor DarkGray
        } else {
            Write-Host "WARN: $k not found in WSL ~/.claude/.env (API mode may fall back)" -ForegroundColor Yellow
        }
    }
}

# Default log level. Override by setting $env:RUST_LOG in your shell first.
if (-not $env:RUST_LOG) {
    $env:RUST_LOG = "info,gimme_a_chance_lib=debug"
}

$featureList = @()
if (-not $CloudOnly) { $featureList += 'sherpa' }
if ($Features) { $featureList += ($Features -split ',') }

if ($featureList.Count -gt 0) {
    $feat = $featureList -join ','
    Write-Host "Starting cargo tauri dev (RUST_LOG=$env:RUST_LOG, features=$feat)..." -ForegroundColor Cyan
    cargo tauri dev --features $feat
} else {
    Write-Host "Starting cargo tauri dev (RUST_LOG=$env:RUST_LOG)..." -ForegroundColor Cyan
    cargo tauri dev
}
