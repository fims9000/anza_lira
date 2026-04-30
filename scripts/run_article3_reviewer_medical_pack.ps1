param(
    [string]$Device = "cuda",
    [string]$Seeds = "41,42,43",
    [int]$Epochs = 60
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LogDir = "logs/article3_reviewer_medical_pack"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $LogDir "reviewer_medical_pack_$Stamp.out.log"
$ErrLog = Join-Path $LogDir "reviewer_medical_pack_$Stamp.err.log"

$Variants = "baseline,attention_unet,unet_plus_plus,az_no_fuzzy,az_no_aniso,az_thesis"
$Overrides = "configs/reviewer_drive_lossmatched_overrides.yaml"

$Commands = @(
    @("configs/drive_benchmark.yaml", "article3_reviewer_drive_ms_$Stamp"),
    @("configs/chase_db1_benchmark.yaml", "article3_reviewer_chase_ms_$Stamp"),
    @("configs/hrf_segplus_benchmark.yaml", "article3_reviewer_hrf_ms_$Stamp")
)

"Started reviewer medical pack at $(Get-Date)" | Out-File -FilePath $OutLog -Encoding utf8
"Device=$Device Seeds=$Seeds Epochs=$Epochs Variants=$Variants" | Out-File -FilePath $OutLog -Encoding utf8 -Append

foreach ($Item in $Commands) {
    $Config = $Item[0]
    $RunName = $Item[1]
    "===== $RunName / $Config =====" | Out-File -FilePath $OutLog -Encoding utf8 -Append
    python scripts/run_drive_multiseed.py `
        --config $Config `
        --variants $Variants `
        --seeds $Seeds `
        --epochs $Epochs `
        --device $Device `
        --variant-overrides $Overrides `
        --run-name $RunName `
        1>> $OutLog 2>> $ErrLog
}

"Finished reviewer medical pack at $(Get-Date)" | Out-File -FilePath $OutLog -Encoding utf8 -Append
Write-Host "Logs:"
Write-Host "  $OutLog"
Write-Host "  $ErrLog"
