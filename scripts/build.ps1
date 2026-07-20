# build.ps1 — Build gimme-a-chance in release mode
#
# Builds with the `sherpa` feature by DEFAULT — the release exe is the
# interview binary and must carry on-device STT (💻 Local / ⚡ parciales) + AEC3.
# Run after every feature lands on main so the Desktop shortcut is never stale.
#
# Usage:
#   .\scripts\build.ps1               (full build: sherpa on-device STT/TTS)
#   .\scripts\build.ps1 -CloudOnly    (skip sherpa — NOT for interviews)
#   powershell.exe -ExecutionPolicy Bypass -File scripts\build.ps1

[CmdletBinding()]
param(
    [switch]$CloudOnly
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$CargoDir = Join-Path $RepoRoot "src-tauri"

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

Set-Location $CargoDir

if ($CloudOnly) {
    Write-Host "`n=== Building release (cloud-only) ===" -ForegroundColor Cyan
    cargo build --release
} else {
    Write-Host "`n=== Building release (--features sherpa) ===" -ForegroundColor Cyan
    cargo build --release --features sherpa
}

if ($LASTEXITCODE -eq 0) {
    $exe = Join-Path $CargoDir "target\release\gimme-a-chance.exe"
    if (Test-Path $exe) {
        $size = [math]::Round((Get-Item $exe).Length / 1MB, 1)
        Write-Host "`nBuild OK: $exe ($size MB)" -ForegroundColor Green
    }
} else {
    Write-Host "`nBuild FAILED" -ForegroundColor Red
    exit $LASTEXITCODE
}
