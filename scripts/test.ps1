# test.ps1 — Run cargo test
#
# Usage:
#   .\scripts\test.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Manifest = Join-Path $RepoRoot "src-tauri\Cargo.toml"

if (-not $env:LIBCLANG_PATH) {
    $env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
}

$cmakeBin = "C:\Program Files\CMake\bin"
if ((Test-Path $cmakeBin) -and ($env:PATH -notlike "*$cmakeBin*")) {
    $env:PATH = "$cmakeBin;$env:PATH"
}

$cargoBin = "$env:USERPROFILE\.cargo\bin"
if ((Test-Path $cargoBin) -and ($env:PATH -notlike "*$cargoBin*")) {
    $env:PATH = "$cargoBin;$env:PATH"
}

Write-Host "`n=== cargo test ===" -ForegroundColor Cyan
cargo test --manifest-path $Manifest
exit $LASTEXITCODE
