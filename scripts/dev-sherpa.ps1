# dev-sherpa.ps1 — dev loop with the on-device sherpa engines active.
# Usage:
#   .\scripts\dev-sherpa.ps1                    (streaming zipformer, live partials)
#   .\scripts\dev-sherpa.ps1 -Engine sherpa     (Parakeet offline, per VAD chunk)
param([string]$Engine = "streaming")
$env:GIMME_STT_ENGINE = $Engine
& "$PSScriptRoot\dev.ps1" -Features sherpa
