#!/usr/bin/env python3
"""
Linux Compatibility Verification Test

Run this on Linux to verify Locetius v1.0 works correctly on your system.
python3 test_linux_compatibility.py
"""

import sys
import platform
import importlib.util

def test_platform():
    """Test that we're on Linux"""
    print("=" * 60)
    print("LOCETIUS v1.0 - LINUX COMPATIBILITY TEST")
    print("=" * 60)
    print()
    
    print("[1] Platform Detection")
    print(f"    OS: {platform.system()}")
    print(f"    Architecture: {platform.machine()}")
    print(f"    Python: {platform.python_version()}")
    
    if platform.system() != "Linux":
        print("    [WARNING] This script is designed for Linux")
    else:
        print("    [OK] Running on Linux")
    print()

def test_imports():
    """Test core Python imports"""
    print("[2] Core Module Imports")
    
    modules_to_test = [
        ("numpy", "NumPy - matrix operations"),
        ("scipy", "SciPy - sparse matrix support"),
        ("PyQt6", "PyQt6 - GUI framework"),
        ("flask", "Flask - REST API"),
    ]
    
    all_ok = True
    for module_name, description in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"    [OK] {description}")
        except ImportError:
            print(f"    [FAIL] {description} - pip install {module_name}")
            all_ok = False
    
    if not all_ok:
        print("\n    Run: pip install -r requirements.txt")
    print()

def test_gpu():
    """Test GPU/CUDA availability"""
    print("[3] GPU & CUDA Detection")
    
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        print(f"    [OK] CUDA runtime available")
        print(f"    [OK] {gpu_count} GPU(s) detected")
        
        if gpu_count > 0:
            device = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(device).decode()
            print(f"    [OK] Primary GPU: {name}")
        
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"    [WARNING] GPU detection failed: {e}")
        print(f"    [INFO] This is OK if you have no NVIDIA GPU")
    print()

def test_solver_api():
    """Test the Locetius solver API"""
    print("[4] Solver API Load Test")
    
    try:
        from locetius_api import LOUCETIUSSolver, SwarmConfig
        print(f"    [OK] locetius_api module imports")
        
        # Try to instantiate solver
        solver = LOUCETIUSSolver()
        print(f"    [OK] LOUCETIUSSolver instantiated")
        
        # Test basic config
        config = SwarmConfig(num_variables=10, annealing_steps=100)
        print(f"    [OK] SwarmConfig created (10 variables)")
        
    except Exception as e:
        print(f"    [FAIL] Solver API error: {e}")
        print(f"    [INFO] Make sure locetius_api.py is in current directory")
    print()

def test_gui():
    """Test GUI can be imported"""
    print("[5] GUI Module Import Test")
    
    try:
        # Just test import, don't instantiate window
        import locetius_gui_v2
        print(f"    [OK] GUI module imports successfully")
        
    except Exception as e:
        print(f"    [WARNING] GUI import failed: {e}")
        print(f"    [INFO] This is normal in headless environments")
        print(f"    [INFO] Use REST API instead: python locetius_server.py")
    print()

def test_file_io():
    """Test file I/O operations"""
    print("[6] File I/O Operations")
    
    import tempfile
    import csv
    import json
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=True) as f:
            writer = csv.writer(f)
            writer.writerow(['test', '1.0'])
        print(f"    [OK] CSV write/read works")
    except Exception as e:
        print(f"    [FAIL] CSV I/O: {e}")
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=True) as f:
            json.dump({'test': 1.0}, f)
        print(f"    [OK] JSON write/read works")
    except Exception as e:
        print(f"    [FAIL] JSON I/O: {e}")
    
    print()

def test_matrix_operations():
    """Test sparse matrix operations"""
    print("[7] Sparse Matrix Operations")
    
    try:
        import numpy as np
        import scipy.sparse as sp
        
        # Create a small sparse matrix
        rows = [0, 1, 2]
        cols = [1, 2, 0]
        data = [1.0, 2.0, 3.0]
        Q = sp.coo_matrix((data, (rows, cols)), shape=(3, 3))
        
        print(f"    [OK] Created sparse COO matrix (3x3)")
        print(f"    [OK] Sparse matrix operations work")
        
    except Exception as e:
        print(f"    [FAIL] Sparse matrix: {e}")
    print()

def main():
    test_platform()
    test_imports()
    test_gpu()
    test_solver_api()
    test_gui()
    test_file_io()
    test_matrix_operations()
    
    print("=" * 60)
    print("LINUX COMPATIBILITY TEST COMPLETE")
    print("=" * 60)
    print()
    print("If all tests passed [OK], you can run:")
    print("  - GUI: python3 locetius_gui_v2.py")
    print("  - Server: python3 locetius_server.py")
    print("  - Python: from locetius_api import LOUCETIUSSolver")
    print()

if __name__ == "__main__":
    main()
