$ErrorActionPreference = "Stop"

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ResultsDir = "results\global_roads_curated_subset_probe"
$LogDir = "logs\global_roads_curated_subset_probe"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$OutLog = Join-Path $LogDir "probe_$Timestamp.out.log"
$ErrLog = Join-Path $LogDir "probe_$Timestamp.err.log"

$Args = @(
    "scripts/run_drive_multiseed.py",
    "--config", "configs/global_roads_curated_subset_benchmark.yaml",
    "--variants", "baseline,az_thesis",
    "--seeds", "42",
    "--epochs", "20",
    "--results-dir", $ResultsDir,
    "--run-name", "global_roads_curated_subset_s42_e20",
    "--device", "cpu"
)

Write-Host "Starting curated subset probe..."
Write-Host "OUT: $OutLog"
Write-Host "ERR: $ErrLog"

& ".\.venv\Scripts\python.exe" @Args 1>> $OutLog 2>> $ErrLog

Write-Host "Done."
