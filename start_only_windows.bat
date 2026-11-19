@echo off
setlocal
cd /d "%~dp0"
if not exist "pyproject.toml" (
  echo ERROR: project root not found. Extract the entire ZIP.
  pause
  exit /b 2
)
if not exist ".venv\Scripts\python.exe" (
  echo Run start_windows.bat first to install the app.
  pause
  exit /b 3
)
".venv\Scripts\python.exe" run.py
if errorlevel 1 pause
