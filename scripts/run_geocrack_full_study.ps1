$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WindowsPython = Join-Path (Split-Path -Parent $ProjectRoot) "venv\Scripts\python.exe"
$LinuxPython = Join-Path (Split-Path -Parent $ProjectRoot) "venv/bin/python"
$Python = if (Test-Path $WindowsPython) { $WindowsPython } elseif (Test-Path $LinuxPython) { $LinuxPython } else { throw "Existing Code/venv Python was not found." }

Set-Location $ProjectRoot
& $Python scripts/geocrack_study.py full @args
exit $LASTEXITCODE
