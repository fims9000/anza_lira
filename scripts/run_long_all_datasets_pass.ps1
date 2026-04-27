param(
    [string]$Python = "C:\ProgramData\anaconda3\envs\mcda-xai\python.exe",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LogDir = "logs\long_all_datasets"
$ResultsDir = "results\long_all_datasets"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsDir | Out-Null

$MasterLog = Join-Path $LogDir "long_all_datasets.out.log"
$MasterErr = Join-Path $LogDir "long_all_datasets.err.log"

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$stamp] $Message"
    Write-Host $line
    Add-Content -Path $MasterLog -Value $line -Encoding UTF8
}

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    Write-Log "START $Name"
    try {
        & $Action
        Write-Log "DONE $Name"
    } catch {
        $msg = $_.Exception.Message
        Write-Log "FAIL $Name :: $msg"
        Add-Content -Path $MasterErr -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Name :: $msg" -Encoding UTF8
    }
}

function Run-PowerShellScript {
    param(
        [string]$ScriptPath
    )
    $proc = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $ScriptPath) `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -PassThru `
        -Wait
    if ($proc.ExitCode -ne 0) {
        throw "Script failed: $ScriptPath (exit $($proc.ExitCode))"
    }
}

function Run-PythonStep {
    param(
        [string]$Name,
        [string[]]$Args
    )
    $outLog = Join-Path $LogDir "$Name.out.log"
    $errLog = Join-Path $LogDir "$Name.err.log"
    $proc = Start-Process -FilePath $Python `
        -ArgumentList $Args `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $outLog `
        -RedirectStandardError $errLog `
        -WindowStyle Hidden `
        -PassThru `
        -Wait
    if ($proc.ExitCode -ne 0) {
        throw "$Name failed (exit $($proc.ExitCode)); see $errLog"
    }
}

Write-Log "LONG PASS STARTED (device=$Device)"
Write-Log "Master log: $MasterLog"

Run-Step "article_latest_only_queue" {
    Run-PowerShellScript "scripts\run_article_latest_only_queue.ps1"
}

Run-Step "article_full_dataset_queue" {
    Run-PowerShellScript "scripts\run_article_full_dataset_queue.ps1"
}

Run-Step "article_small_medical_queue" {
    Run-PowerShellScript "scripts\run_article_small_medical_queue.ps1"
}

Run-Step "spacenet3_paris_large_azthesis_s42_e30" {
    Run-PythonStep "spacenet3_paris_large_azthesis_s42_e30" @(
        "scripts\run_drive_multiseed.py",
        "--config", "configs\spacenet3_paris_azthesis_large.yaml",
        "--variants", "az_thesis",
        "--seeds", "42",
        "--results-dir", "results",
        "--run-name", "spacenet3_paris_large_azthesis_s42_e30",
        "--device", $Device
    )
}

Write-Log "LONG PASS FINISHED"
