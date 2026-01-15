param(
    [string]$Dataset = "nfcorpus",
    [string[]]$Sizes = @("1"),
    [int]$Repeats = 1,
    [int]$TopK = 10
)

# Activate venv if present
$venvActivate = Join-Path $PSScriptRoot '..\.venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
    Write-Host "Activating venv: $venvActivate"
    & $venvActivate
}
else {
    Write-Host "No venv activation found at $venvActivate; ensure your environment is configured" -ForegroundColor Yellow
}

# Run script from repository root
Set-Location (Join-Path $PSScriptRoot '..')
$sizeArg = $Sizes -join ' '
$cmd = "python tools\generate_latency_vs_size.py --dataset $Dataset --sizes $sizeArg --repeats $Repeats --top-k $TopK"
Write-Host "Running: $cmd"
Invoke-Expression $cmd
