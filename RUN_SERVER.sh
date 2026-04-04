#!/bin/bash
# Locetius v1.0 - REST API Server Launcher
# Start the server on port 8765

echo ""
echo "=========================================="
echo "  Locetius v1.0 - REST API Server"
echo "  Starting on port 8765..."
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    exit 1
fi

# Install dependencies if needed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Set LD_LIBRARY_PATH for CUDA
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

# Launch the server
echo "Starting REST API server..."
python3 locetius_server.py

if [ $? -ne 0 ]; then
    echo "ERROR: Server failed to start"
    exit 1
fi
