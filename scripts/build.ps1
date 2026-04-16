# build.ps1 — Build gimme-a-chance in release mode
#
# Usage:
#   .\scripts\build.ps1
#   powershell.exe -ExecutionPolicy Bypass -File scripts\build.ps1

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

Write-Host "`n=== Building release ===" -ForegroundColor Cyan
cargo build --release

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
