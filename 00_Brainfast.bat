@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0project\frontend\StartBrainfast.bat" (
  echo [ERROR] Could not find project\frontend\StartBrainfast.bat
  pause
  exit /b 1
)

echo Starting Brainfast from D:\Brainfast ...
call "%~dp0project\frontend\StartBrainfast.bat"
