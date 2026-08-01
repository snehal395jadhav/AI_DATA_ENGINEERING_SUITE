@echo off
REM ╔══════════════════════════════════════════════════════════╗
REM ║  AI Data Engineering Suite — one-click launcher (Windows) ║
REM ╚══════════════════════════════════════════════════════════╝
cd /d "%~dp0"
echo Installing dependencies (first run only)...
python -m pip install -r requirements.txt
echo Starting AI Data Engineering Suite...
python -m streamlit run app.py
pause
