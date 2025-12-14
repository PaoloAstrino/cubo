@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo CUBO AI - "Low-Spec" / Native Launcher
echo ============================================================
echo.

REM 1. Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VER=%%I
echo [OK] Found Python %PYTHON_VER%

REM 2. Check for Node.js (Optional but recommended for frontend)
node --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Node.js not found. Frontend might not work.
    echo Install Node.js from https://nodejs.org/ if you want the UI.
    set NODE_AVAILABLE=0
) else (
    echo [OK] Found Node.js
    set NODE_AVAILABLE=1
)

REM 3. Setup Virtual Environment
if not exist ".venv" (
    echo.
    echo [SETUP] Creating virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM 4. Activate Virtual Environment
echo [SETUP] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

REM 5. Install Dependencies
echo [SETUP] Checking/Installing dependencies...
pip install -e .
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

REM 6. Install Frontend Dependencies (if Node is available)
if "%NODE_AVAILABLE%"=="1" (
    if exist "frontend\package.json" (
        echo [SETUP] Installing frontend dependencies...
        cd frontend
        REM Check for pnpm, else use npm
        call pnpm --version >nul 2>&1
        if !errorlevel! equ 0 (
            call pnpm install
        ) else (
            echo [INFO] pnpm not found, using npm...
            call npm install
        )
        cd ..
    )
)

REM 7. Launch Full Stack
echo.
echo [START] Launching CUBO...
echo Press Ctrl+C to stop.
echo.

python scripts\start_fullstack.py

pause
