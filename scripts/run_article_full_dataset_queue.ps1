$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = "C:\ProgramData\anaconda3\envs\mcda-xai\python.exe"
$ResultsDir = "results\article_full_dataset"
$LogDir = "logs\article_full_dataset"
$Overrides = "configs\article_az_thesis_name_overrides.yaml"

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Run-Step {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string[]]$Args
    )

    $Out = Join-Path $LogDir "$Name.out.log"
    $Err = Join-Path $LogDir "$Name.err.log"
    "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] START $Name" | Tee-Object -FilePath $Out -Append
    $proc = Start-Process -FilePath $Python -ArgumentList $Args -WorkingDirectory $Root -RedirectStandardOutput $Out -RedirectStandardError $Err -WindowStyle Hidden -PassThru -Wait
    if ($proc.ExitCode -ne 0) {
        "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] FAIL $Name exit=$($proc.ExitCode)" | Tee-Object -FilePath $Out -Append
        throw "$Name failed with exit code $($proc.ExitCode). See $Err"
    }
    "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] DONE $Name" | Tee-Object -FilePath $Out -Append
}

# DRIVE is already complete in article_latest_only. It is intentionally not
# rerun here because the local DRIVE test split has no manual labels.

Run-Step "01_chase_db1_full_azthesis_s42_e80" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\chase_db1_benchmark.yaml",
    "--variant-overrides", $Overrides,
    "--variants", "az_thesis",
    "--seeds", "42",
    "--results-dir", $ResultsDir,
    "--run-name", "chase_db1_full_azthesis_s42_e80",
    "--device", "cuda"
)

Run-Step "02_fives_full_azthesis_s42_e20" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\fives_benchmark.yaml",
    "--variant-overrides", $Overrides,
    "--variants", "az_thesis",
    "--seeds", "42",
    "--epochs", "20",
    "--results-dir", $ResultsDir,
    "--run-name", "fives_full_azthesis_s42_e20",
    "--device", "cuda"
)

Run-Step "03_arcade_syntax_full_azthesis_s42_e40" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\arcade_syntax_benchmark.yaml",
    "--variant-overrides", $Overrides,
    "--variants", "az_cat",
    "--seeds", "42",
    "--results-dir", $ResultsDir,
    "--run-name", "arcade_syntax_full_azthesis_s42_e40",
    "--device", "cuda"
)

Run-Step "04_arcade_stenosis_full_azthesis_s42_e40" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\arcade_stenosis_benchmark.yaml",
    "--variant-overrides", $Overrides,
    "--variants", "az_cat",
    "--seeds", "42",
    "--results-dir", $ResultsDir,
    "--run-name", "arcade_stenosis_full_azthesis_s42_e40",
    "--device", "cuda"
)

Run-Step "05_global_roads_full256_azthesis_s42_e30" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\global_roads_full_256_benchmark.yaml",
    "--variant-overrides", $Overrides,
    "--variants", "az_thesis",
    "--seeds", "42",
    "--results-dir", $ResultsDir,
    "--run-name", "global_roads_full256_azthesis_s42_e30",
    "--device", "cuda"
)

"[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] FULL_DATASET_QUEUE_DONE" | Tee-Object -FilePath (Join-Path $LogDir "full_dataset_queue.done.log") -Append
