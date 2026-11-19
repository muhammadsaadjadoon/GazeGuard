@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Install GazeGuard first with start_windows.bat
  pause
  exit /b 1
)
echo CLOSE THE DASHBOARD FIRST to release your webcam.
set /p PID=Enter anonymous participant ID (e.g. p01):
if "%PID%"=="" exit /b 1
echo Only continue if the participant agreed to numerical landmark capture.
set /p OK=Enter YES to confirm informed consent: 
if /i not "%OK%"=="YES" exit /b 1
".venv\Scripts\python.exe" -m training.collect_webcam --participant "%PID%" --consent
pause
