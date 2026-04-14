@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0project\frontend\StartBrainfast.bat" (
  echo [ERROR] Missing launcher: project\frontend\StartBrainfast.bat
  pause
  exit /b 1
)

call "%~dp0project\frontend\StartBrainfast.bat"
if errorlevel 1 (
  echo.
  echo [ERROR] Brainfast failed to start.
  pause
  exit /b 1
)

exit /b 0
