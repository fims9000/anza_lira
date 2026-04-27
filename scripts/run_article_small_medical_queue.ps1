param(
    [string]$Python = "C:\ProgramData\anaconda3\envs\mcda-xai\python.exe",
    [string]$Device = "cuda",
    [int]$PollSeconds = 60,
    [int]$MaxGpuMemoryMiB = 2500
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LogDir = "logs\article_small_medical"
$ResultsDir = "results\article_small_medical"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsDir | Out-Null

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$stamp] $Message"
    Write-Host $line
    Add-Content -Path (Join-Path $LogDir "small_medical_launcher.out.log") -Value $line -Encoding UTF8
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
    $currentPid = $PID
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $currentPid -and
        $_.CommandLine -and
        ($_.CommandLine -match "scripts\\run_drive_multiseed.py|scripts/run_drive_multiseed.py|train.py") -and
        ($_.CommandLine -notmatch "article_small_medical")
    }
    return [bool]$procs
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

Write-Log "Small medical AZ-Thesis queue started. Device=$Device, MaxGpuMemoryMiB=$MaxGpuMemoryMiB"

Run-Step "01_hrf_segplus_azthesis_s42_e60" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\hrf_segplus_benchmark.yaml",
    "--variant-overrides", "configs\article_az_thesis_name_overrides.yaml",
    "--variants", "az_thesis",
    "--seeds", "42",
    "--epochs", "60",
    "--results-dir", $ResultsDir,
    "--run-name", "hrf_segplus_azthesis_s42_e60",
    "--device", $Device
)

Run-Step "02_chase_db1_azthesis_s42_e80" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\chase_db1_benchmark.yaml",
    "--variant-overrides", "configs\article_az_thesis_name_overrides.yaml",
    "--variants", "az_thesis",
    "--seeds", "42",
    "--epochs", "80",
    "--results-dir", $ResultsDir,
    "--run-name", "chase_db1_azthesis_s42_e80",
    "--device", $Device
)

Run-Step "03_drive_azthesis_s42_e120" @(
    "scripts\run_drive_multiseed.py",
    "--config", "configs\drive_az_thesis_article_current.yaml",
    "--variant-overrides", "configs\article_az_thesis_name_overrides.yaml",
    "--variants", "az_thesis",
    "--seeds", "42",
    "--epochs", "120",
    "--results-dir", $ResultsDir,
    "--run-name", "drive_azthesis_s42_e120",
    "--device", $Device
)

Write-Log "Small medical AZ-Thesis queue finished."
