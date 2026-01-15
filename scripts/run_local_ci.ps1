<#
.SYNOPSIS
    Runs the CI/CD pipeline locally on Windows.
    Replicates steps from .github/workflows/reusable-quality.yml and ci-fast.yml

.DESCRIPTION
    This script executes the following checks:
    1. Dependency Installation (CI, Lint, Editable install)
    2. Linting (Ruff, Black, Isort)
    3. Security (Bandit)
    4. Tests (Unit, Fast Integration, Smoke)
    
    It stops immediately if any step fails.
#>

$ErrorActionPreference = "Stop"

function Write-Header {
    param([string]$Title)
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host " $Title" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
}

function Run-Step {
    param([string]$Command, [string]$Description)
    Write-Host "-> $Description..." -ForegroundColor Yellow
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Step failed: $Description"
    }
    Write-Host "   OK" -ForegroundColor Green
}

# --- 1. Dependencies ---
Write-Header "1. Installing Dependencies"
Run-Step "pip install -r requirements/requirements-ci.txt" "Installing CI requirements"
Run-Step "pip install -r requirements/requirements-lint.txt" "Installing Lint requirements"
Run-Step "pip install -e ." "Installing project in editable mode"

# --- 2. Linting ---
Write-Header "2. Code Quality (Linting)"
Run-Step "ruff check cubo/ --select=E9,F63,F7,F82" "Running Ruff (Basic Checks)"
Run-Step "black --check cubo/" "Running Black (Format Check)"
Run-Step "isort --check-only cubo/" "Running Isort (Import Order Check)"

# --- 3. Security ---
Write-Header "3. Security Scan"
Run-Step "bandit -r cubo/ -ll -q" "Running Bandit Security Scan"

# --- 4. Tests ---
Write-Header "4. Running Tests"
Run-Step "pytest tests/unit -v --tb=short" "Running Unit Tests"
# "not slow" assumes pytest.ini or conftest.py has markers setup, replicating ci-fast.yml
Run-Step "pytest tests/integration -v --tb=short -m 'not slow'" "Running Fast Integration Tests"
Run-Step "pytest tests/smoke -v --tb=short" "Running Smoke Tests"

Write-Header "SUCCESS! All CI checks passed locally."
