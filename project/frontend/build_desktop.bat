@echo off
setlocal
cd /d %~dp0

echo [1/4] Installing / updating dependencies...
python -m pip install -r requirements-desktop-build.txt -q
if errorlevel 1 (
  echo [ERROR] pip install failed.
  pause & exit /b 1
)

echo [2/4] Cleaning previous build...
if exist build\BrainfastUI rmdir /s /q build\BrainfastUI
if exist dist\BrainfastUI  rmdir /s /q dist\BrainfastUI

echo [3/4] Building with PyInstaller...
python -m PyInstaller --noconfirm BrainfastUI.spec
if errorlevel 1 (
  echo [ERROR] Build failed. See above for details.
  pause & exit /b 1
)

echo [4/4] Done!
echo.
echo  EXE location: dist\BrainfastUI\BrainfastUI.exe
echo  To distribute: zip the entire dist\BrainfastUI\ folder.
echo.
echo  First-run tip: the atlas file annotation_25.nii.gz is bundled.
echo  If you move the folder, keep all files together.
echo.
if /I "%BRAINFAST_NO_PAUSE%"=="1" exit /b 0
pause
