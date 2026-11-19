@echo off
setlocal
cd /d "%~dp0"
if not exist "pyproject.toml" (
  echo ERROR: pyproject.toml is missing from this folder.
  echo Re-extract the full archive into one folder, then use this script from that folder.
  pause
  exit /b 2
)
if not exist ".venv\Scripts\python.exe" (
  py -3.14 -m venv ".venv"
  if errorlevel 1 (
    echo ERROR: Python 3.14 x64 was not found. Install it and try again.
    pause
    exit /b 3
  )
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :install_error
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto :install_error
".venv\Scripts\python.exe" run.py
if errorlevel 1 (
  echo Application exited with an error.
  pause
  exit /b 5
)
exit /b 0
:install_error
echo Installation failed. Check your internet connection and Python version.
pause
exit /b 4
