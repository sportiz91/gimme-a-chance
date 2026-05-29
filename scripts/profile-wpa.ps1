# profile-wpa.ps1 - Record a Windows Performance trace of gimme-a-chance and
# open the resulting .etl in Windows Performance Analyzer.
#
# What it captures (via wpr.exe built-in recording profiles):
#   - GeneralProfile  broad coverage: CPU, images, interrupts, handles, processes
#   - CPU             CPU sampling with call stacks
#   - DiskIO          disk read/write events
#   - FileIO          file read/write events (useful for whisper model loads)
#   - Registry        registry access
# These are guaranteed to exist on any Windows 10/11. More advanced profiles like
# Audio.Verbose, ThreadScheduling, or ReferenceSet require installing the Windows ADK.
#
# Usage (must run as Administrator):
#   .\scripts\profile-wpa.ps1                     records for 60s, launches the app itself
#   .\scripts\profile-wpa.ps1 -Seconds 120        record 120s
#   .\scripts\profile-wpa.ps1 -NoLaunch           attach to an already running app (dev mode)
#   .\scripts\profile-wpa.ps1 -Open <path>        only open an existing .etl in WPA
#
# How it works:
#   By default, the script spawns `cargo tauri dev` in a side window so the webview
#   has its dev server to connect to. It waits for gimme-a-chance.exe to appear,
#   then starts ETW recording. When recording ends, it stops WPR, cleans up the
#   dev process tree, and opens WPA on the resulting .etl.
#
# Requirements:
#   - wpr.exe (ships with Windows)
#   - wpa.exe (Windows Performance Analyzer - install from Microsoft Store)
#   - Administrator privileges (ETW providers require elevation)

[CmdletBinding()]
param(
    [int]$Seconds = 60,
    [switch]$NoLaunch,
    [string]$Open,
    [int]$CompileTimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$Traces    = Join-Path $RepoRoot "logs\traces"
$DevScript = Join-Path $PSScriptRoot "dev.ps1"

function Ensure-Admin {
    $id = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object System.Security.Principal.WindowsPrincipal($id)
    if (-not $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Error "This script needs to run as Administrator (ETW requires it). Right-click PowerShell -> Run as Administrator."
        exit 1
    }
}

function Resolve-Wpa {
    $candidates = @(
        "$env:ProgramFiles\Windows Kits\10\Windows Performance Toolkit\wpa.exe",
        "${env:ProgramFiles(x86)}\Windows Kits\10\Windows Performance Toolkit\wpa.exe",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\wpa.exe"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    $wpa = Get-Command wpa.exe -ErrorAction SilentlyContinue
    if ($wpa) { return $wpa.Source }
    return $null
}

function Wait-ForApp {
    param([int]$TimeoutSeconds)
    $elapsed = 0
    while ($elapsed -lt $TimeoutSeconds) {
        $app = Get-Process -Name gimme-a-chance -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($app) { return $app }
        Start-Sleep -Seconds 2
        $elapsed += 2
        if (($elapsed % 10) -eq 0) {
            Write-Host "  still compiling/starting... ($elapsed s elapsed)" -ForegroundColor DarkGray
        }
    }
    return $null
}

function Stop-ProcessTree {
    param([int]$ProcessId)
    # taskkill /T kills the process and all its descendants.
    cmd.exe /c "taskkill /F /T /PID $ProcessId" 2>&1 | Out-Null
}

# --- Fast path: just open an existing trace ---
if ($Open) {
    $wpa = Resolve-Wpa
    if (-not $wpa) { Write-Error "wpa.exe not found. Install Windows Performance Analyzer from the Store."; exit 1 }
    & $wpa $Open
    exit 0
}

Ensure-Admin
New-Item -ItemType Directory -Force -Path $Traces | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$EtlPath   = Join-Path $Traces "gimme-a-chance-$timestamp.etl"

$devProcess = $null

try {
    # --- 1. Launch the app (unless the caller says it's already running) ---
    if (-not $NoLaunch) {
        Write-Host "Launching cargo tauri dev in a side window..." -ForegroundColor Cyan
        $devProcess = Start-Process -FilePath "powershell.exe" `
            -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $DevScript `
            -WorkingDirectory $RepoRoot `
            -PassThru

        Write-Host "Waiting for gimme-a-chance.exe to come up (up to $CompileTimeoutSeconds s)..." -ForegroundColor Cyan
        $app = Wait-ForApp -TimeoutSeconds $CompileTimeoutSeconds
        if (-not $app) {
            throw "gimme-a-chance.exe never appeared within $CompileTimeoutSeconds s. Compilation may have failed - check the dev window."
        }
        Write-Host "App is running (PID $($app.Id)). Letting it settle for 3s..." -ForegroundColor Green
        Start-Sleep -Seconds 3
    } else {
        $app = Get-Process -Name gimme-a-chance -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $app) {
            throw "-NoLaunch was specified but gimme-a-chance.exe is not running. Start it first, then rerun."
        }
        Write-Host "Attaching to existing app (PID $($app.Id))." -ForegroundColor Green
    }

    # --- 2. Start WPR recording ---
    Write-Host "Starting WPR recording..." -ForegroundColor Cyan
    wpr.exe -start GeneralProfile -start CPU -start DiskIO -start FileIO -start Registry -filemode
    if ($LASTEXITCODE -ne 0) {
        throw "wpr.exe failed to start. Check that you are running as Administrator and that no other tracing session is active (stop it with: wpr -cancel)."
    }

    Write-Host "Recording for $Seconds seconds. EXERCISE THE APP NOW (click Listen, ask a question)." -ForegroundColor Yellow
    Start-Sleep -Seconds $Seconds

    # --- 3. Stop WPR and save ---
    Write-Host "Stopping WPR and writing $EtlPath" -ForegroundColor Cyan
    wpr.exe -stop $EtlPath

    if (-not (Test-Path $EtlPath)) {
        throw "wpr.exe did not produce $EtlPath - check above output for errors."
    }

    $size = (Get-Item $EtlPath).Length / 1MB
    Write-Host ("Trace saved: {0} ({1:N1} MB)" -f $EtlPath, $size) -ForegroundColor Green

    # --- 4. Open WPA ---
    $wpa = Resolve-Wpa
    if ($wpa) {
        Write-Host "Opening in WPA (may take 30-60s to index)..." -ForegroundColor Cyan
        & $wpa $EtlPath
    } else {
        Write-Warning "wpa.exe not found. Install Windows Performance Analyzer from the Store, then open manually: $EtlPath"
    }
}
finally {
    # --- 5. Clean up: kill the dev window we spawned (if any) ---
    if ($devProcess -and -not $devProcess.HasExited) {
        Write-Host "Cleaning up dev process tree (PID $($devProcess.Id))..." -ForegroundColor DarkGray
        Stop-ProcessTree -ProcessId $devProcess.Id
    }
    # Also kill any lingering gimme-a-chance.exe that we might have started.
    if (-not $NoLaunch) {
        Get-Process -Name gimme-a-chance -ErrorAction SilentlyContinue | ForEach-Object {
            Stop-ProcessTree -ProcessId $_.Id
        }
    }
}
