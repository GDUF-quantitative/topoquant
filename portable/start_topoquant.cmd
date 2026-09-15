@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

TopoQuant.exe
set "TOPOQUANT_EXIT=%ERRORLEVEL%"

echo.
if not "%TOPOQUANT_EXIT%"=="0" (
  echo TopoQuant exited with code %TOPOQUANT_EXIT%.
  echo Please keep this window open and take a screenshot of the error.
)
pause
exit /b %TOPOQUANT_EXIT%
