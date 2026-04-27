$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = "C:\ProgramData\anaconda3\envs\mcda-xai\python.exe"
$ResultsDir = "results\article_latest_only"
$LogDir = "logs\article_latest_only"
$Overrides = "configs\article_az_only_stable_overrides.yaml"

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

Run-Step "11_drive_az_only_stable_ms" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\drive_benchmark.yaml",
    "--variant-overrides", $Overrides,
    "--variants", "az_thesis",
    "--seeds", "41,42,43",
    "--epochs", "20",
    "--results-dir", $ResultsDir,
    "--run-name", "drive_az_only_stable_ms_414243_e20",
    "--device", "cuda"
)

Run-Step "12_chase_az_only_stable_s42" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\chase_db1_benchmark.yaml",
    "--variant-overrides", $Overrides,
    "--variants", "az_thesis",
    "--seeds", "42",
    "--epochs", "20",
    "--results-dir", $ResultsDir,
    "--run-name", "chase_db1_az_only_stable_s42_e20",
    "--device", "cuda"
)

"[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] AZ ONLY DONE" | Tee-Object -FilePath (Join-Path $LogDir "az_only.done.log") -Append
