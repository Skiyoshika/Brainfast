@echo off
setlocal
cd /d %~dp0

echo [1/3] Checking Python...
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found in PATH.
  echo Install Python 3.11+ and try again.
  pause
  exit /b 1
)

echo [2/3] Checking runtime...
python ..\scripts\check_env.py --config ..\configs\run_config.template.json
if errorlevel 1 (
  echo [ERROR] Runtime check failed.
  pause
  exit /b 1
)

echo [3/3] Starting Brainfast desktop launcher...
start "" pythonw desktop_app.py
if errorlevel 1 (
  echo [WARN] pythonw launch failed, falling back to python.
  start "" python desktop_app.py
)

echo Brainfast is starting. If the browser does not open automatically,
echo visit http://127.0.0.1:8787
exit /b 0
