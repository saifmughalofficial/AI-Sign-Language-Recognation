@echo off
REM One-time setup for Windows: creates a virtual environment and installs everything (free).
cd /d "%~dp0"
echo === Sign Language Assistant setup ===
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during installation.
  pause
  exit /b 1
)
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-desktop.txt
if errorlevel 1 (
  echo.
  echo Some packages failed to install. If it was vosk or sounddevice, remove those lines
  echo from requirements-desktop.txt and run setup.bat again - the app still works without them.
)
python setup_dataset.py --generate-images --check
echo.
echo Setup finished. Start the app with run_app.bat
pause
