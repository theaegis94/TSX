@echo off
title TSX Signal Dashboard
cd /d "%~dp0"

echo.
echo  ================================================
echo    TSX Signal Dashboard
echo  ================================================
echo.

REM Verify Python is on PATH
where python >nul 2>nul
if errorlevel 1 (
    echo  ERROR: Python not found on PATH.
    echo  Install Python 3.10+ from https://www.python.org
    echo.
    pause
    exit /b 1
)

REM Verify streamlit is installed; install on first run
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo  First-time setup: installing streamlit + dependencies...
    python -m pip install --quiet streamlit yfinance pandas matplotlib requests
    if errorlevel 1 (
        echo.
        echo  ERROR: dependency install failed.
        pause
        exit /b 1
    )
)

echo  Dashboard will open at: http://localhost:8501
echo.
echo  To stop the server: close this window or press Ctrl+C
echo.
python -m streamlit run app.py
pause
