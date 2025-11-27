@echo off
REM Quick start script for CUBO full stack

echo ============================================================
echo CUBO Full Stack Startup
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.11+
    exit /b 1
)

REM Check if pnpm is installed
pnpm --version >nul 2>&1
if errorlevel 1 (
    echo Error: pnpm not found. Installing globally...
    npm install -g pnpm
)

REM Install backend dependencies
echo Installing backend dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install backend dependencies
    exit /b 1
)

REM Install frontend dependencies
echo Installing frontend dependencies...
cd frontend
pnpm install
if errorlevel 1 (
    echo Error: Failed to install frontend dependencies
    exit /b 1
)
cd ..

REM Start full stack
echo.
echo Starting CUBO full stack...
python scripts\start_fullstack.py %*
