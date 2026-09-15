@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

TopoQuant.exe --portable-pipeline-self-test
set "TOPOQUANT_EXIT=%ERRORLEVEL%"
echo.
if "%TOPOQUANT_EXIT%"=="0" (
  echo Portable runtime is ready.
) else (
  echo Portable runtime check failed with code %TOPOQUANT_EXIT%.
)
pause
exit /b %TOPOQUANT_EXIT%
