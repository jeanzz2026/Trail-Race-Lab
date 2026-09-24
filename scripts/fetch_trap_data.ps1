$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VendorRoot = Join-Path $ProjectRoot "vendor"
$Target = Join-Path $VendorRoot "TRAP-data"

if (Test-Path -LiteralPath $Target) {
    Write-Host "TRAP-data already exists at $Target"
    exit 0
}

New-Item -ItemType Directory -Path $VendorRoot -Force | Out-Null
git clone --depth 1 https://github.com/ricfog/TRAP-data.git $Target
Write-Host "Downloaded upstream TRAP data to $Target"
