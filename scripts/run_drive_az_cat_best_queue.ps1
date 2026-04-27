$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = "C:\ProgramData\anaconda3\envs\mcda-xai\python.exe"
$LogDir = "logs\article_latest_only"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Run-Step {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string]$Config
    )

    $Out = Join-Path $LogDir "$Name.out.log"
    $Err = Join-Path $LogDir "$Name.err.log"
    "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] START $Name config=$Config" | Tee-Object -FilePath $Out -Append
    $proc = Start-Process -FilePath $Python -ArgumentList @("train.py", "--config", $Config) -WorkingDirectory $Root -RedirectStandardOutput $Out -RedirectStandardError $Err -WindowStyle Hidden -PassThru -Wait
    if ($proc.ExitCode -ne 0) {
        "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] FAIL $Name exit=$($proc.ExitCode)" | Tee-Object -FilePath $Out -Append
        throw "$Name failed with exit code $($proc.ExitCode). See $Err"
    }
    "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] DONE $Name" | Tee-Object -FilePath $Out -Append
}

Run-Step "21_drive_az_cat_best_repro_s42_e120" "configs\drive_az_cat_best_current.yaml"
Run-Step "22_drive_az_cat_recall_coremean_s42_e120" "configs\drive_az_cat_recall_current.yaml"

"[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] AZ_CAT_QUEUE_DONE" | Tee-Object -FilePath (Join-Path $LogDir "az_cat_best_queue.done.log") -Append
