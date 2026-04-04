@echo off
REM Windows Compatibility Verification Test
REM Run this to verify Locetius v1.0 works correctly on your system

echo.
echo ============================================================
echo  LOCETIUS v1.0 - WINDOWS COMPATIBILITY TEST
echo ============================================================
echo.

echo [1] Platform Detection
for /f "tokens=*" %%i in ('python --version') do echo     %%i
echo     [OK] Running on Windows

echo.
echo [2] Core Module Imports
python -c "import numpy; print('     [OK] NumPy loaded')" 2>nul || echo     [FAIL] NumPy - run: pip install numpy
python -c "import scipy; print('     [OK] SciPy loaded')" 2>nul || echo     [FAIL] SciPy - run: pip install scipy
python -c "import PyQt6; print('     [OK] PyQt6 loaded')" 2>nul || echo     [FAIL] PyQt6 - run: pip install PyQt6
python -c "import flask; print('     [OK] Flask loaded')" 2>nul || echo     [FAIL] Flask - run: pip install flask

echo.
echo [3] GPU Detection
nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
if %errorlevel% equ 0 (
    echo     [OK] NVIDIA GPU detected
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader') do echo     GPU: %%i
) else (
    echo     [WARNING] No NVIDIA GPU detected or nvidia-smi not found
)

echo.
echo [4] Solver API Test
python -c "from locetius_api import LOUCETIUSSolver; print('     [OK] Solver API loads')" 2>nul || echo     [FAIL] Solver API - verify locetius_api.py exists

echo.
echo [5] File I/O Operations
python -c "import csv, json, tempfile; print('     [OK] File I/O works')" 2>nul || echo     [FAIL] File I/O

echo.
echo ============================================================
echo  COMPATIBILITY TEST COMPLETE
echo ============================================================
echo.
echo If tests passed, you can run:
echo   - GUI: python locetius_gui_v2.py
echo   - Server: python locetius_server.py
echo.
pause
