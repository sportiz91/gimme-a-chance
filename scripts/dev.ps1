# dev.ps1 — Run gimme-a-chance in live dev mode (cargo tauri dev)
#
# Usage:
#   .\scripts\dev.ps1
#   powershell.exe -ExecutionPolicy Bypass -File scripts\dev.ps1

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

# Default log level. Override by setting $env:RUST_LOG in your shell first.
if (-not $env:RUST_LOG) {
    $env:RUST_LOG = "info,gimme_a_chance_lib=debug"
}

Write-Host "Starting cargo tauri dev (RUST_LOG=$env:RUST_LOG)..." -ForegroundColor Cyan
cargo tauri dev
