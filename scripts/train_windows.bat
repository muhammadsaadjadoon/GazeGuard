@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Install GazeGuard first with start_windows.bat
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m training.train_landmarks --csv datasets/private/landmarks.csv --output models/trained_gaze.json
if errorlevel 1 echo Training failed: read TRAINING_GUIDE.md and collect additional labelled sessions.
pause
