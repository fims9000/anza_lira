param(
    [string]$Python = "C:\ProgramData\anaconda3\envs\mcda-xai\python.exe",
    [string]$Device = "cuda",
    [int]$PollSeconds = 60,
    [int]$MaxGpuMemoryMiB = 2500
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LogDir = "logs\reviewer_baseline_pack"
$ResultsDir = "results\reviewer_baseline_pack"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsDir | Out-Null
$LockFile = Join-Path $LogDir "reviewer_baseline_pack_queue.lock"

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$stamp] $Message"
    Write-Host $line
    Add-Content -Path (Join-Path $LogDir "reviewer_baseline_pack_queue.out.log") -Value $line -Encoding UTF8
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
            ($_.CommandLine -notmatch "reviewer_baseline_pack")
        }
        return [bool]$procs
    } catch {
        # Some locked-down PowerShell policies block Win32_Process queries.
        # In this case we rely on GPU memory threshold only.
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

Write-Log "Reviewer baseline pack queue started. Device=$Device, MaxGpuMemoryMiB=$MaxGpuMemoryMiB"
if (Test-Path $LockFile) {
    Write-Log "LOCK_EXISTS: another reviewer baseline queue is already running. Exit."
    exit 0
}

Set-Content -Path $LockFile -Value "$PID" -Encoding UTF8

try {
    # Step 1: fast single-seed probe for all key variants.
    Run-StepIfMissing "01_drive_probe_s42_e40_all_variants" (Join-Path $ResultsDir "drive_probe_s42_e40_all_variants\all_metrics.json") @(
        "scripts\run_drive_multiseed.py",
        "--config", "configs\drive_az_cat_best_current.yaml",
        "--variant-overrides", "configs\reviewer_drive_lossmatched_overrides.yaml",
        "--variants", "baseline,attention_unet,az_no_fuzzy,az_no_aniso,az_cat",
        "--seeds", "42",
        "--epochs", "40",
        "--results-dir", $ResultsDir,
        "--run-name", "drive_probe_s42_e40_all_variants",
        "--device", $Device
    )

    # Step 2: final multi-seed head-to-head for article table.
    Run-StepIfMissing "02_drive_final_ms_414243_e120_headline" (Join-Path $ResultsDir "drive_final_ms_414243_e120_headline\all_metrics.json") @(
        "scripts\run_drive_multiseed.py",
        "--config", "configs\drive_az_cat_best_current.yaml",
        "--variant-overrides", "configs\reviewer_drive_lossmatched_overrides.yaml",
        "--variants", "baseline,attention_unet,az_cat",
        "--seeds", "41,42,43",
        "--epochs", "120",
        "--results-dir", $ResultsDir,
        "--run-name", "drive_final_ms_414243_e120_headline",
        "--device", $Device
    )

    Write-Log "Reviewer baseline pack queue finished."
}
finally {
    Remove-Item -Path $LockFile -ErrorAction SilentlyContinue
}
