@echo off
setlocal
cd /d %~dp0

python -m pip install pyinstaller
if errorlevel 1 (
  echo [ERROR] Failed to install pyinstaller.
  exit /b 1
)

python -m PyInstaller --noconfirm --onefile --windowed --name IdleBrainUI ^
  --hidden-import=flask ^
  --hidden-import=werkzeug ^
  --add-data "index.html;." ^
  --add-data "styles.css;." ^
  --add-data "app.js;." ^
  desktop_app.py

if errorlevel 1 (
  echo [ERROR] Build failed.
  exit /b 1
)

echo.
echo Build done. EXE at: dist\IdleBrainUI.exe

