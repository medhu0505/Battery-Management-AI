@echo off
rem BMS Copilot launcher. Usage: run.bat [port]   (default port 8000)
setlocal
cd /d "%~dp0"
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  where python >nul 2>nul || (echo Python 3.12+ is required: https://www.python.org/downloads/ & goto :error)
  echo Setting up (first run only^)...
  python -m venv .venv || goto :error
  "%PY%" -m pip install --quiet --upgrade pip
  "%PY%" -m pip install -r requirements.txt || goto :error
)

if not exist "artifacts\model_report.json" (
  echo Training models (first run only, about 3 minutes^)...
  "%PY%" -m bms_copilot.train || goto :error
)

echo Starting BMS Copilot at http://127.0.0.1:%PORT%  (Ctrl+C to stop^)
if not defined BMS_NO_BROWSER start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep 6; Start-Process 'http://127.0.0.1:%PORT%'"
"%PY%" -m bms_copilot.api --port %PORT%
exit /b %errorlevel%

:error
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
