@echo off
setlocal

for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8787" ^| findstr "LISTENING"') do (
  echo Stopping Brainfast process %%P ...
  taskkill /PID %%P /F >nul 2>nul
)

echo Done.
exit /b 0
