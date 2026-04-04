@echo off
REM Locetius v1.0 - Windows GUI Launcher
REM This file automatically detects the Python environment and launches the GUI

echo.
echo ==========================================
echo   Locetius v1.0 - QUBO Solver
echo   Launching GUI...
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Check if required packages are installed
python -c "import PyQt6" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Launch the GUI
echo Starting application...
python locetius_gui_v2.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Application failed to launch
    echo Check requirements.txt and dependencies
    pause
    exit /b 1
)

pause
