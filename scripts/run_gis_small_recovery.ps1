param(
    [string]$Python = "C:\ProgramData\anaconda3\envs\mcda-xai\python.exe",
    [string]$Device = "cuda",
    [int]$Epochs = 25
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LogDir = "logs\gis_small_recovery"
$ResultsDir = "results\gis_small_recovery"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$step = "gis_small_recovery_s42_e$Epochs" + "_$stamp"
$outLog = Join-Path $LogDir "$step.out.log"
$errLog = Join-Path $LogDir "$step.err.log"

$args = @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\gis_roads_small_recovery.yaml",
    "--variant-overrides", "configs\gis_roads_small_recovery_overrides.yaml",
    "--variants", "baseline,az_cat,az_thesis",
    "--seeds", "42",
    "--epochs", "$Epochs",
    "--results-dir", $ResultsDir,
    "--run-name", "gis_roads_small_recovery_s42_e$Epochs",
    "--device", $Device
)

Write-Host "[GIS-SMALL] START $step"
Write-Host "[GIS-SMALL] LOG OUT: $outLog"
Write-Host "[GIS-SMALL] LOG ERR: $errLog"
Write-Host "[GIS-SMALL] CMD: $Python $($args -join ' ')"

$proc = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $ProjectRoot -RedirectStandardOutput $outLog -RedirectStandardError $errLog -WindowStyle Hidden -PassThru
Write-Host "[GIS-SMALL] PID=$($proc.Id)"
