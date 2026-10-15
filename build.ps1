# Windows build script: compile DDTN research program PDF with Tectonic.
param(
    [string]$Tectonic = "$env:LOCALAPPDATA\tectonic\tectonic.exe",
    [switch]$RunMVM
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path $Tectonic)) {
    $found = (Get-Command tectonic -ErrorAction SilentlyContinue).Source
    if ($found) { $Tectonic = $found }
    else { throw "Tectonic not found. Install from https://tectonic-typesetting.github.io/" }
}

Write-Host "Building PDF with Tectonic..." -ForegroundColor Cyan
& $Tectonic "ddtn_research_program.tex"
if ($LASTEXITCODE -ne 0) { throw "PDF build failed." }
Write-Host "PDF built: $(Resolve-Path 'ddtn_research_program.pdf')" -ForegroundColor Green

if ($RunMVM) {
    Write-Host "Running MVM simulations..." -ForegroundColor Cyan
    Push-Location code
    python run_mvm.py
    Pop-Location
}
