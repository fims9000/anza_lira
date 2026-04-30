param(
    [string]$Config = "configs/drive_benchmark.yaml",
    [string]$Device = "cuda",
    [string]$Seed = "42",
    [int]$Epochs = 40
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LogDir = "logs/article3_regime_sweep"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $LogDir "regime_sweep_$Stamp.out.log"
$ErrLog = Join-Path $LogDir "regime_sweep_$Stamp.err.log"

$Rules = @(1, 2, 4, 8, 16)
$SeedInt = [int]$Seed

"Started regime-count sweep at $(Get-Date)" | Out-File -FilePath $OutLog -Encoding utf8
"Config=$Config Device=$Device Seed=$Seed Epochs=$Epochs" | Out-File -FilePath $OutLog -Encoding utf8 -Append

foreach ($R in $Rules) {
    $RunName = "article3_regime_R${R}_s${Seed}_e${Epochs}_$Stamp"
    "===== R=$R / $RunName =====" | Out-File -FilePath $OutLog -Encoding utf8 -Append
    python train.py `
        --config $Config `
        --variant az_thesis `
        --num-rules $R `
        --epochs $Epochs `
        --seed $SeedInt `
        --device $Device `
        --run-name $RunName `
        1>> $OutLog 2>> $ErrLog
}

"Finished regime-count sweep at $(Get-Date)" | Out-File -FilePath $OutLog -Encoding utf8 -Append
Write-Host "Logs:"
Write-Host "  $OutLog"
Write-Host "  $ErrLog"
