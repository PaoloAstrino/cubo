@echo off
REM Clear Windows icon cache to force taskbar icon refresh

echo Clearing Windows icon cache...

REM Kill Windows Explorer
taskkill /f /im explorer.exe

REM Delete icon cache files
del /f /s /q /a "%LocalAppData%\IconCache.db" 2>nul
del /f /s /q /a "%LocalAppData%\Microsoft\Windows\Explorer\iconcache*" 2>nul

echo Icon cache cleared.
echo.
echo Press any key to restart Windows Explorer...
pause >nul

REM Restart Windows Explorer
start explorer.exe

echo.
echo Done! Now run the CUBO application again.
pause
