$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = "C:\ProgramData\anaconda3\envs\mcda-xai\python.exe"
$ResultsDir = "results\article_final_latest"
$LogDir = "logs\article_final_latest"

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Run-Step {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string[]]$Args
    )

    $Out = Join-Path $LogDir "$Name.out.log"
    $Err = Join-Path $LogDir "$Name.err.log"
    $Stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$Stamp] START $Name" | Tee-Object -FilePath $Out -Append
    $proc = Start-Process -FilePath $Python -ArgumentList $Args -WorkingDirectory $Root -RedirectStandardOutput $Out -RedirectStandardError $Err -WindowStyle Hidden -PassThru -Wait
    $Stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    if ($proc.ExitCode -ne 0) {
        "[$Stamp] FAIL $Name exit=$($proc.ExitCode)" | Tee-Object -FilePath $Out -Append
        throw "$Name failed with exit code $($proc.ExitCode). See $Err"
    }
    "[$Stamp] DONE $Name" | Tee-Object -FilePath $Out -Append
}

Run-Step "01_drive_policyfix_ms" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\drive_benchmark.yaml",
    "--variant-overrides", "configs\drive_az_thesis_implfix_probe.yaml",
    "--variants", "baseline,az_thesis",
    "--seeds", "41,42,43",
    "--epochs", "20",
    "--results-dir", $ResultsDir,
    "--run-name", "drive_policyfix_ms_414243_e20_latest",
    "--device", "cuda"
)

Run-Step "02_chase_single_seed" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\chase_db1_benchmark.yaml",
    "--variants", "baseline,az_thesis",
    "--seeds", "42",
    "--epochs", "20",
    "--results-dir", $ResultsDir,
    "--run-name", "chase_db1_s42_e20_latest",
    "--device", "cuda"
)

Run-Step "03_fives_single_seed" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\fives_benchmark.yaml",
    "--variants", "baseline,az_thesis",
    "--seeds", "42",
    "--epochs", "10",
    "--results-dir", $ResultsDir,
    "--run-name", "fives_s42_e10_latest",
    "--device", "cuda"
)

Run-Step "04_arcade_syntax_single_seed" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\arcade_syntax_benchmark.yaml",
    "--variants", "baseline,az_thesis",
    "--seeds", "42",
    "--epochs", "20",
    "--results-dir", $ResultsDir,
    "--run-name", "arcade_syntax_s42_e20_latest",
    "--device", "cuda"
)

Run-Step "05_arcade_stenosis_single_seed" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\arcade_stenosis_benchmark.yaml",
    "--variants", "baseline,az_thesis",
    "--seeds", "42",
    "--epochs", "20",
    "--results-dir", $ResultsDir,
    "--run-name", "arcade_stenosis_s42_e20_latest",
    "--device", "cuda"
)

"[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] ALL DONE" | Tee-Object -FilePath (Join-Path $LogDir "queue.done.log") -Append
