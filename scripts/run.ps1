# run.ps1 — Run an already-built release binary
#
# Usage:
#   .\scripts\run.ps1
#
# Build first with scripts\build.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $RepoRoot "src-tauri\target\release\gimme-a-chance.exe"

if (-not (Test-Path $exe)) {
    Write-Host "No release build found at $exe" -ForegroundColor Red
    Write-Host "Run scripts\build.ps1 first." -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting gimme-a-chance..." -ForegroundColor Cyan
& $exe
