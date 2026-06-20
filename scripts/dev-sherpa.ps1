# dev-sherpa.ps1 — dev loop with the on-device sherpa engines active.
# Usage:
#   .\scripts\dev-sherpa.ps1                    (streaming zipformer, live partials)
#   .\scripts\dev-sherpa.ps1 -Engine sherpa     (Parakeet offline, per VAD chunk)
#   .\scripts\dev-sherpa.ps1 -Aec dtln          (DTLN-aec echo cancellation; default aec3)
param([string]$Engine = "streaming", [string]$Aec = "")
$env:GIMME_STT_ENGINE = $Engine
if ($Aec) { $env:GIMME_AEC_ENGINE = $Aec }
& "$PSScriptRoot\dev.ps1" -Features sherpa
