param(
    [string]$Python = "C:\ProgramData\anaconda3\envs\mcda-xai\python.exe",
    [string]$Device = "cuda",
    [int]$PollSeconds = 45,
    [int]$MaxGpuMemoryMiB = 9000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LogDir = "logs\quick_small_queue"
$ResultsDir = "results\quick_small_queue"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsDir | Out-Null
$LockFile = Join-Path $LogDir "quick_small_queue.lock"

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$stamp] $Message"
    Write-Host $line
    Add-Content -Path (Join-Path $LogDir "quick_small_queue.out.log") -Value $line -Encoding UTF8
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

function Wait-ForGpuMemory {
    param([string]$StepName)
    while ($true) {
        $used = Get-GpuMemoryUsedMiB
        if ($used -le $MaxGpuMemoryMiB) {
            Write-Log "${StepName}: memory.used=${used}MiB, proceed."
            return
        }
        Write-Log "${StepName}: waiting, memory.used=${used}MiB (limit=$MaxGpuMemoryMiB)"
        Start-Sleep -Seconds $PollSeconds
    }
}

function Run-Step {
    param(
        [string]$Name,
        [string[]]$Args
    )
    Wait-ForGpuMemory $Name
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
    Run-Step -Name $Name -Args $Args
}

if (Test-Path $LockFile) {
    Write-Log "LOCK_EXISTS: another quick-small queue is already running. Exit."
    exit 0
}

Set-Content -Path $LockFile -Value "$PID" -Encoding UTF8
Write-Log "Quick-small queue started. Device=$Device, MaxGpuMemoryMiB=$MaxGpuMemoryMiB"

try {
    Run-StepIfMissing "01_drive_quick_s42_e20_az_vs_baseline" (Join-Path $ResultsDir "drive_quick_s42_e20_az_vs_baseline\all_metrics.json") @(
        "scripts\run_drive_multiseed.py",
        "--config", "configs\drive_benchmark.yaml",
        "--variant-overrides", "configs\quick_small_probe_overrides.yaml",
        "--variants", "baseline,az_thesis",
        "--seeds", "42",
        "--epochs", "20",
        "--results-dir", $ResultsDir,
        "--run-name", "drive_quick_s42_e20_az_vs_baseline",
        "--device", $Device
    )

    Run-StepIfMissing "02_chase_quick_s42_e20_az_vs_baseline" (Join-Path $ResultsDir "chase_quick_s42_e20_az_vs_baseline\all_metrics.json") @(
        "scripts\run_drive_multiseed.py",
        "--config", "configs\chase_db1_benchmark.yaml",
        "--variant-overrides", "configs\quick_small_probe_overrides.yaml",
        "--variants", "baseline,az_thesis",
        "--seeds", "42",
        "--epochs", "20",
        "--results-dir", $ResultsDir,
        "--run-name", "chase_quick_s42_e20_az_vs_baseline",
        "--device", $Device
    )

    Run-StepIfMissing "03_hrf_quick_s42_e20_az_vs_baseline" (Join-Path $ResultsDir "hrf_quick_s42_e20_az_vs_baseline\all_metrics.json") @(
        "scripts\run_drive_multiseed.py",
        "--config", "configs\hrf_segplus_benchmark.yaml",
        "--variant-overrides", "configs\quick_small_probe_overrides.yaml",
        "--variants", "baseline,az_thesis",
        "--seeds", "42",
        "--epochs", "20",
        "--results-dir", $ResultsDir,
        "--run-name", "hrf_quick_s42_e20_az_vs_baseline",
        "--device", $Device
    )

    Write-Log "Quick-small queue finished."
}
finally {
    Remove-Item -Path $LockFile -ErrorAction SilentlyContinue
}
