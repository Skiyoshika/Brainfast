@echo off
setlocal
cd /d %~dp0

echo [1/4] Installing desktop build dependencies...
python -m pip install -r requirements-desktop-build.txt -q
if errorlevel 1 (
  echo [ERROR] pip install failed.
  pause & exit /b 1
)

echo [2/4] Building desktop bundle...
set "BRAINFAST_NO_PAUSE=1"
call build_desktop.bat
if errorlevel 1 (
  echo [ERROR] Desktop bundle build failed.
  pause & exit /b 1
)
set "BRAINFAST_NO_PAUSE="

echo [3/4] Checking NSIS...
where makensis >nul 2>nul
if errorlevel 1 (
  echo [ERROR] makensis was not found in PATH.
  echo Install NSIS and try again.
  pause & exit /b 1
)

set "VERSION_JSON=%~dp0..\version.json"
for /f "usebackq delims=" %%v in (`python -c "import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))['version'])" "%VERSION_JSON%"`) do set APP_VERSION=%%v

echo [4/6] Optional code signing for desktop EXE...
if defined BRAINFAST_CODESIGN_PFX if defined BRAINFAST_CODESIGN_PASSWORD (
  python ..\scripts\sign_release_assets.py --file "%CD%\dist\BrainfastUI\BrainfastUI.exe"
  if errorlevel 1 (
    echo [ERROR] Code signing failed.
    pause & exit /b 1
  )
) else (
  echo [INFO] Signing credentials not set. Skipping desktop EXE signing.
)

echo [5/6] Building installer...
makensis /DAPP_VERSION=%APP_VERSION% /DDIST_DIR="%CD%\dist\BrainfastUI" /DOUTPUT_FILE="%CD%\dist\Brainfast-%APP_VERSION%-setup.exe" "..\..\nsis\brainfast_setup.nsi"
if errorlevel 1 (
  echo [ERROR] NSIS build failed.
  pause & exit /b 1
)

echo [6/6] Optional code signing for installer...
if defined BRAINFAST_CODESIGN_PFX if defined BRAINFAST_CODESIGN_PASSWORD (
  python ..\scripts\sign_release_assets.py --file "%CD%\dist\Brainfast-%APP_VERSION%-setup.exe"
  if errorlevel 1 (
    echo [ERROR] Installer signing failed.
    pause & exit /b 1
  )
) else (
  echo [INFO] Signing credentials not set. Skipping installer signing.
)

echo Installer created at:
echo   dist\Brainfast-%APP_VERSION%-setup.exe
pause
