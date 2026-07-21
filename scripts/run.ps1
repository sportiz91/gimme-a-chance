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

# Freshness guard: an exe older than the last commit means the interview binary
# is missing features merged since it was built (the silent failure mode that
# burned a real interview on 2026-07-17).
Set-Location $RepoRoot
$exeTime = (Get-Item $exe).LastWriteTimeUtc
# Only commits that touch the binary's inputs count — docs/scripts commits
# must not cry wolf, or the warning trains you to ignore it.
$headUnix = git log -1 --format=%ct -- src-tauri dist 2>$null
if ($headUnix) {
    $headTime = [DateTimeOffset]::FromUnixTimeSeconds([int64]$headUnix).UtcDateTime
    if ($exeTime -lt $headTime) {
        Write-Host "WARN: exe ($($exeTime.ToString('yyyy-MM-dd HH:mm'))Z) is OLDER than the last commit ($($headTime.ToString('yyyy-MM-dd HH:mm'))Z)." -ForegroundColor Red
        Write-Host "      Run scripts\build.ps1 before a real interview." -ForegroundColor Yellow
    }
}

# The sherpa `shared` DLLs must sit next to the exe — without them the local
# STT engine silently degrades to cloud.
if (-not (Test-Path (Join-Path (Split-Path $exe) "sherpa-onnx-c-api.dll"))) {
    Write-Host "WARN: sherpa DLLs missing next to the exe (cloud-only build?) — no on-device STT." -ForegroundColor Yellow
}

Write-Host "Starting gimme-a-chance..." -ForegroundColor Cyan
& $exe
