#!/usr/bin/env python3
"""
LOUCETIUS Quick Smoke Test
==========================
Verifies that the solver loads, runs, and produces sensible output.
Run this after cloning to check everything works on your machine.

Usage:
    python test_quick.py
"""

import sys
import time
import numpy as np

def main():
    print("=" * 60)
    print("  LOUCETIUS Quick Smoke Test")
    print("=" * 60)
    passed = 0
    failed = 0

    # -- Test 1: Import ------------------------------------------
    print("\n[1/5] Importing locetius_api ...", end=" ")
    try:
        from locetius_api import LOUCETIUSSolver, SwarmConfig, SolverMode
        print("OK")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        failed += 1
        print("\nCannot continue without the API. Check that locetius_api.py")
        print("and locetius_core.dll/.so are in this directory.")
        sys.exit(1)

    # -- Test 2: Solver loads (finds DLL/SO) ---------------------
    print("[2/5] Loading solver (DLL/SO) ...", end=" ")
    try:
        solver = LOUCETIUSSolver()
        print("OK")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        failed += 1
        print("\nThe DLL/SO could not be loaded. Check:")
        print("  - locetius_core.dll (Windows) or liblocetius_core.so (Linux)")
        print("  - CUDA 12.1+ drivers installed")
        print("  - NVIDIA GPU present")
        sys.exit(1)

    # -- Test 3: Tiny problem (3 variables) ----------------------
    print("[3/5] Solving 3-variable problem ...", end=" ")
    try:
        Q = np.array([
            [-1.0,  0.5,  0.3],
            [ 0.5, -2.0,  0.1],
            [ 0.3,  0.1, -1.5]
        ])
        config = SwarmConfig(
            num_variables=3,
            num_swarms=64,
            solver_mode=SolverMode.CONTINUOUS,
            annealing_steps=1000,
            engine='AUTO'
        )
        t0 = time.time()
        result = solver.solve(Q, config)
        dt = time.time() - t0

        assert result.best_solution is not None, "No solution returned"
        assert len(result.best_solution) == 3, f"Wrong solution length: {len(result.best_solution)}"
        assert result.best_energy is not None, "No energy returned"

        print(f"OK  (E={result.best_energy:.4f}, {dt:.2f}s)")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        failed += 1

    # -- Test 4: Medium problem (100 variables, random) ----------
    print("[4/5] Solving 100-variable random QUBO ...", end=" ")
    try:
        rng = np.random.default_rng(42)
        N = 100
        Q = rng.standard_normal((N, N))
        Q = (Q + Q.T) / 2  # symmetrise

        config = SwarmConfig(
            num_variables=N,
            num_swarms=64,
            solver_mode=SolverMode.COMBINATORIAL,
            annealing_steps=2000,
            engine='AUTO'
        )
        t0 = time.time()
        result = solver.solve(Q, config)
        dt = time.time() - t0

        assert result.best_solution is not None, "No solution returned"
        assert len(result.best_solution) == N, f"Wrong solution length"
        assert result.best_energy < 0, f"Energy should be negative for this problem, got {result.best_energy}"

        print(f"OK  (E={result.best_energy:.2f}, {dt:.2f}s)")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        failed += 1

    # -- Test 5: Sparse problem (scipy.sparse, 500 vars) --------
    print("[5/5] Solving 500-variable sparse QUBO ...", end=" ")
    try:
        import scipy.sparse
        rng = np.random.default_rng(123)
        N = 500
        density = 0.02  # ~2% non-zero
        Q_sparse = scipy.sparse.random(N, N, density=density, random_state=rng, format='coo')
        Q_sparse = (Q_sparse + Q_sparse.T) / 2  # symmetrise

        config = SwarmConfig(
            num_variables=N,
            num_swarms=64,
            solver_mode=SolverMode.SPATIAL,
            annealing_steps=2000,
            engine='AUTO'
        )
        t0 = time.time()
        result = solver.solve(Q_sparse, config)
        dt = time.time() - t0

        assert result.best_solution is not None, "No solution returned"
        assert len(result.best_solution) == N, f"Wrong solution length"

        print(f"OK  (E={result.best_energy:.2f}, {dt:.2f}s)")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        failed += 1

    # -- Summary -------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of 5")
    print("=" * 60)

    if failed == 0:
        print("\n  All tests passed. LOUCETIUS is ready to use.\n")
    else:
        print(f"\n  {failed} test(s) failed. Check the errors above.\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
