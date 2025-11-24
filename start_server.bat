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
echo You can now run live API smoke tests using the env var:
echo set CUBO_RUN_LIVE_TESTS=1 && pytest tests/api/test_api_live.py -q
echo.
pause
