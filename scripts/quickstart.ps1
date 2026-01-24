param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$RepoUrl,

    [Parameter(Mandatory=$false, Position=1)]
    [string]$Dir
)

function Fail([string]$msg, [int]$code=1) {
    Write-Error $msg
    exit $code
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git not found. Please install git from https://git-scm.com/downloads"
}

if (-not $Dir) {
    $Dir = ($RepoUrl -split '/')[-1] -replace '\.git$',''
}

if (Test-Path $Dir) {
    Write-Output "Directory '$Dir' exists. Pulling latest changes..."
    Push-Location $Dir
    git pull --rebase || Write-Warning "git pull failed; continuing"
    Pop-Location
} else {
    Write-Output "Cloning $RepoUrl into $Dir..."
    git clone $RepoUrl $Dir | Out-Null
}

Set-Location $Dir
if (-not (Test-Path './scripts/run_local.ps1') -and -not (Test-Path './scripts/run_local.sh')) {
    Fail "scripts/run_local.ps1 or scripts/run_local.sh not found in repo. This quickstart expects the project to include one of these scripts."
}

# Perform preflight checks in the target repo before launching
Write-Output "Running preflight checks..."

# Python version check
if (-not (Get-Command python -ErrorAction SilentlyContinue) -and -not (Get-Command python3 -ErrorAction SilentlyContinue)) {
    Fail "Python not found. Please install Python 3.11+"
}
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pyCmd = 'python'
} else {
    $pyCmd = 'python3'
}

try {
    & $pyCmd -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>$null
} catch {
    Fail "Python 3.11 or higher is required."
}

# Node version check
if (Get-Command node -ErrorAction SilentlyContinue) {
    $v = (node -v).TrimStart('v')
    $major = [int]($v.Split('.')[0])
    if ($major -lt 18) { Fail "Node.js v18 or higher is required. Found: v$v" }
} else {
    Fail "Node.js not found. Please install Node.js v18+."
}

# Disk space check - require at least 5 GB free
$cwd = Get-Location
$driveLetter = $cwd.Path.Split(':')[0]
$drive = Get-PSDrive -Name $driveLetter -ErrorAction SilentlyContinue
if (-not $drive) { $drive = Get-PSDrive -PSProvider FileSystem | Select-Object -First 1 }
if ($drive.Free -lt 5GB) { Fail "Not enough disk space: at least 5GB free is required. Available: $([math]::Round($drive.Free/1GB,2)) GB" }

if (Test-Path './scripts/run_local.ps1') {
    Write-Output "Launching local dev environment with ./scripts/run_local.ps1"
    powershell -ExecutionPolicy Bypass -File .\scripts\run_local.ps1
} else {
    Write-Output "Launching local dev environment with ./scripts/run_local.sh"
    bash ./scripts/run_local.sh
}
