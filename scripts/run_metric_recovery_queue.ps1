param(
    [string]$Python = "C:\ProgramData\anaconda3\envs\mcda-xai\python.exe",
    [string]$Device = "cuda",
    [int]$PollSeconds = 60,
    [int]$MaxGpuMemoryMiB = 2500
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LogDir = "logs\article_metric_recovery"
$ResultsDir = "results\article_metric_recovery"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsDir | Out-Null
$LockFile = Join-Path $LogDir "metric_recovery_queue.lock"

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$stamp] $Message"
    Write-Host $line
    Add-Content -Path (Join-Path $LogDir "metric_recovery_queue.out.log") -Value $line -Encoding UTF8
}

function Get-GpuMemoryUsedMiB {
    try {
        $raw = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $raw) {
            return 999999
        }
        return [int]($raw | Select-Object -First 1).Trim()
    } catch {
        return 999999
    }
}

function Test-OtherTrainingProcess {
    try {
        $currentPid = $PID
        $procs = Get-CimInstance Win32_Process | Where-Object {
            $_.ProcessId -ne $currentPid -and
            $_.CommandLine -and
            ($_.CommandLine -match "scripts\\run_drive_multiseed.py|scripts/run_drive_multiseed.py|train.py") -and
            ($_.CommandLine -notmatch "article_metric_recovery")
        }
        return [bool]$procs
    } catch {
        # Fallback for restricted PowerShell environments.
        return $false
    }
}

function Wait-ForSafeGpu {
    param([string]$StepName)
    while ($true) {
        $used = Get-GpuMemoryUsedMiB
        $busy = Test-OtherTrainingProcess
        if (-not $busy -and $used -le $MaxGpuMemoryMiB) {
            Write-Log "${StepName}: GPU is free enough, memory.used=${used}MiB"
            return
        }
        Write-Log "${StepName}: waiting, other_training=$busy, memory.used=${used}MiB"
        Start-Sleep -Seconds $PollSeconds
    }
}

function Run-Step {
    param(
        [string]$Name,
        [string[]]$Args
    )
    Wait-ForSafeGpu $Name
    $outLog = Join-Path $LogDir "$Name.out.log"
    $errLog = Join-Path $LogDir "$Name.err.log"
    Write-Log "START $Name"
    Write-Log "$Python $($Args -join ' ')"
    $proc = Start-Process -FilePath $Python -ArgumentList $Args -WorkingDirectory $ProjectRoot -RedirectStandardOutput $outLog -RedirectStandardError $errLog -WindowStyle Hidden -PassThru -Wait
    if ($proc.ExitCode -ne 0) {
        Write-Log "FAILED $Name exit=$($proc.ExitCode); see $errLog"
        throw "$Name failed"
    }
    Write-Log "DONE $Name"
}

function Run-StepIfMissing {
    param(
        [string]$Name,
        [string]$DoneFile,
        [string[]]$Args
    )
    if (Test-Path $DoneFile) {
        Write-Log "SKIP $Name (already exists: $DoneFile)"
        return
    }
    Run-Step $Name $Args
}

$Checkpoint = "results\article_full_dataset\fives_full_azthesis_s42_e20\az_thesis_seed42\checkpoint_best.pt"
if (-not (Test-Path $Checkpoint)) {
    throw "Required checkpoint not found: $Checkpoint"
}

Write-Log "Metric recovery queue started. Device=$Device, MaxGpuMemoryMiB=$MaxGpuMemoryMiB"
if (Test-Path $LockFile) {
    Write-Log "LOCK_EXISTS: another metric recovery queue is already running. Exit."
    exit 0
}

Set-Content -Path $LockFile -Value "$PID" -Encoding UTF8

try {
    Run-StepIfMissing "01_chase_transfer_from_fivesfull_s42_e20" (Join-Path $ResultsDir "chase_transfer_from_fivesfull_s42_e20\all_metrics.json") @(
        "scripts\run_drive_multiseed.py",
        "--config", "configs\chase_db1_transfer_from_fives_full20.yaml",
        "--variant-overrides", "configs\article_az_thesis_name_overrides.yaml",
        "--variants", "az_thesis",
        "--seeds", "42",
        "--results-dir", $ResultsDir,
        "--run-name", "chase_transfer_from_fivesfull_s42_e20",
        "--device", $Device
    )

    Write-Log "Metric recovery queue finished."
}
finally {
    Remove-Item -Path $LockFile -ErrorAction SilentlyContinue
}
