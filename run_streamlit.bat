@echo off
REM Sign Language Assistant - web edition (Streamlit). Opens http://localhost:8899
cd /d "%~dp0"
start "" http://localhost:8899
python -m streamlit run streamlit_app.py
pause
