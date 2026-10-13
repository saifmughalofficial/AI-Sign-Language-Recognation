@echo off
REM Runs the whole training pipeline on the dataset you placed in dataset\raw
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate
echo [1/3] Preparing dataset...
python training\prepare_dataset.py
if errorlevel 1 goto fail
echo [2/3] Extracting hand landmarks...
python training\extract_features.py
if errorlevel 1 goto fail
echo [3/3] Training models...
python training\train_model.py
if errorlevel 1 goto fail
echo.
echo All done. Models are in the "models" folder. Start the app with run_app.bat
pause
exit /b 0
:fail
echo.
echo Something went wrong - read the message above. See README.md ^> Dataset setup.
pause
exit /b 1
