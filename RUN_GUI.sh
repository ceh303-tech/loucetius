#!/bin/bash
# Locetius v1.0 - Linux GUI Launcher
# This file automatically detects the Python environment and launches the GUI

echo ""
echo "=========================================="
echo "  Locetius v1.0 - QUBO Solver"
echo "  Launching GUI..."
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found in PATH"
    echo "Please install Python 3.10+ using:"
    echo "  Ubuntu/Debian: sudo apt install python3-dev python3-pip"
    echo "  Fedora: sudo dnf install python3-devel python3-pip"
    echo "  macOS: brew install python@3.10"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

# Check if PyQt6 is installed
if ! python3 -c "import PyQt6" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
fi

# Set LD_LIBRARY_PATH for CUDA if needed
if [ -z "$LD_LIBRARY_PATH" ]; then
    if [ -d "/usr/local/cuda/lib64" ]; then
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        echo "CUDA library path set"
    fi
fi

# Launch the GUI
echo "Starting application..."
python3 locetius_gui_v2.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Application failed to launch"
    echo "Check requirements.txt and dependencies"
    exit 1
fi
