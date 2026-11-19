@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Please install the project first.
  pause
  exit /b 1
)
if not exist "%USERPROFILE%\.gazeguard_v1\trained_gaze.json" (
  echo No validated trained_gaze.json present in the app-data directory.
  echo Read TRAINING_GUIDE.md before enabling a model.
  pause
  exit /b 1
)
set "GAZEGUARD_ENABLE_TRAINED_GAZE=1"
".venv\Scripts\python.exe" run.py
pause
