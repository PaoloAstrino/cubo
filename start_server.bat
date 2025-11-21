@echo off
REM Start API server in a new window
cd /d "%~dp0"
start "CUBO API Server" cmd /k ".venv\Scripts\activate.bat && python start_api_server.py"
echo.
echo ========================================
echo CUBO API Server starting in new window
echo ========================================
echo.
echo The server will run in a separate window.
echo You can now run: python test_api_live.py
echo.
pause
