@echo off
REM Sign Language Assistant - web edition (Streamlit). Opens http://localhost:8765
cd /d "%~dp0"
start "" http://localhost:8765
python -m streamlit run streamlit_app.py --server.port 8765
pause
