# lint.ps1 — Run fmt + clippy. Use before committing.
#
# Usage:
#   .\scripts\lint.ps1
#   powershell.exe -ExecutionPolicy Bypass -File scripts\lint.ps1
#
# Matches the CI check (`.github/workflows/ci.yml`).

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

Write-Host "`n=== cargo fmt ===" -ForegroundColor Cyan
cargo fmt --all --manifest-path $Manifest
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green

Write-Host "`n=== cargo clippy ===" -ForegroundColor Cyan
cargo clippy --all-targets --manifest-path $Manifest -- -D warnings
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nClippy found warnings. Fix them before committing." -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green

Write-Host "`n=== All checks passed ===" -ForegroundColor Green
