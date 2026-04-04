"""
Locetius v1.0 - Integration Examples
=====================================

Complete workflows showing how to use the Locetius API in Python
for various application domains.

This file can be run as a script to test the API with synthetic problems.

Author: Christian Hayes
Created: March 30, 2026
"""

import numpy as np
import scipy.sparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict

# Import Locetius API
from locetius_api import (
    LOUCETIUSSolver, SwarmConfig, SolverMode, PrecisionMode,
    solve_qubo, get_version, print_diagnostics
)


# ============================================================================
# EXAMPLE 1: Basic QUBO Solving with Convenience Function
# ============================================================================

def example_1_basic_solve():
    """
    Example 1: Simplest possible usage - just load and solve.
    
    Perfect for quick prototyping and one-off problems.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic QUBO Solving")
    print("=" * 70)
    
    # Create a small random QUBO problem
    N = 100
    Q = scipy.sparse.random(N, N, density=0.1, format='coo')
    
    # Solve with defaults (CONTINUOUS mode, Float32)
    result = solve_qubo(Q)
    
    print(result)
    return result


# ============================================================================
# EXAMPLE 2: Max-Cut Problem with Configuration
# ============================================================================

def example_2_max_cut():
    """
    Example 2: Solve a Max-Cut problem with explicit configuration.
    
    Max-Cut is a classic NP-hard problem:
    maximize: sum_{i<j} w_ij * (1 - 2*x_i*x_j)
    
    Which converts to QUBO:
    minimize: -sum_{i<j} w_ij * (1 - x_i - x_j + 2*x_i*x_j)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Max-Cut Problem (COMBINATORIAL Mode)")
    print("=" * 70)
    
    # Construct a simple Max-Cut problem
    # Graph: 4 nodes with weights
    edges = [
        (0, 1, 1.0),
        (1, 2, 2.0),
        (2, 3, 1.5),
        (3, 0, 1.0),
        (0, 2, 0.5),
    ]
    
    N = 4
    Q = scipy.sparse.lil_matrix((N, N), dtype=np.float64)
    
    for i, j, w in edges:
        Q[i, i] -= w
        Q[j, j] -= w
        Q[i, j] += 2 * w
    
    Q = Q.tocoo()
    
    # Create configuration for Max-Cut (COMBINATORIAL mode)
    config = SwarmConfig(
        num_variables=N,
        num_swarms=64,
        solver_mode=SolverMode.COMBINATORIAL,
        annealing_steps=5000,
        high_precision=False,  # Fast
    )
    
    solver = LOUCETIUSSolver()
    result = solver.solve(Q, config)
    
    print(f"Partition: {result.best_solution}")
    print(f"Max-Cut Energy: {result.best_energy:.4f}")
    print(f"Consensus: {result.consensus_percentage:.1f}%")
    
    # Interpret result
    cut_size = np.sum(result.best_solution)
    print(f"Solution: {cut_size} nodes in partition 0, {N - cut_size} in partition 1")
    
    return result


# ============================================================================
# EXAMPLE 3: Portfolio Optimization with High Precision
# ============================================================================

def example_3_portfolio_optimization():
    """
    Example 3: Portfolio optimization with strict precision requirements.
    
    Portfolio optimization often requires tight numerical precision to avoid
    floating-point errors in financial calculations.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Portfolio Optimization (HIGH PRECISION)")
    print("=" * 70)
    
    # Create a mock portfolio optimization QUBO
    # Variables: select which assets to include
    # Objective: maximize expected return, minimize variance
    
    N = 50  # 50 assets
    
    # Return vector (negated for minimization)
    mu = np.random.randn(N) * 0.01
    
    # Covariance matrix (risk)
    Sigma = np.random.randn(N, N)
    Sigma = (Sigma + Sigma.T) / 2
    
    # QUBO: minimize variance - return
    # Q = Sigma - diag(2*mu)
    Q = scipy.sparse.coo_matrix(Sigma)
    Q_diag = scipy.sparse.diags(-2 * mu)
    Q = (Q + Q_diag).tocoo()
    
    config_f32 = SwarmConfig(
        num_variables=N,
        solver_mode=SolverMode.CONTINUOUS,
        annealing_steps=10000,
        high_precision=False,  # Fast but less precise
    )
    
    config_f64 = SwarmConfig(
        num_variables=N,
        solver_mode=SolverMode.CONTINUOUS,
        annealing_steps=10000,
        high_precision=True,  # More precise
    )
    
    solver = LOUCETIUSSolver()
    
    print("Float32 (Fast):")
    start = time.time()
    result_f32 = solver.solve(Q, config_f32)
    print(f"  Energy:  {result_f32.best_energy:.10f}")
    print(f"  Time:    {time.time() - start:.2f}s")
    
    print("\nFloat64 (Precise):")
    start = time.time()
    result_f64 = solver.solve(Q, config_f64)
    print(f"  Energy:  {result_f64.best_energy:.10f}")
    print(f"  Time:    {time.time() - start:.2f}s")
    
    print(f"\nPrecision difference: {abs(result_f32.best_energy - result_f64.best_energy):.2e}")
    
    return result_f64  # Return high-precision result


# ============================================================================
# EXAMPLE 4: Large-Scale Problem with Sparse Matrix
# ============================================================================

def example_4_large_scale():
    """
    Example 4: Handling a large QUBO problem efficiently.
    
    Demonstrates sparse matrix use (99%+ compression vs dense).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Large-Scale QUBO (Sparse Matrix)")
    print("=" * 70)
    
    N = 5000  # 5K variables
    density = 0.001  # 0.1% sparse (typical for real problems)
    
    # Create sparse matrix
    Q = scipy.sparse.random(N, N, density=density, format='coo', dtype=np.float64)
    
    print(f"Problem: {N}x{N} matrix")
    print(f"Non-zeros: {Q.nnz:,}")
    
    # Memory comparison
    dense_bytes = N * N * 8  # float64
    sparse_bytes = Q.nnz * 3 * 4 + N * 4  # Rough estimate
    
    print(f"Dense memory: {dense_bytes / 1e9:.2f} GB")
    print(f"Sparse memory: {sparse_bytes / 1e6:.2f} MB")
    print(f"Compression: {100 - (sparse_bytes/dense_bytes)*100:.2f}%")
    
    # Solve
    config = SwarmConfig(
        num_variables=N,
        solver_mode=SolverMode.CONTINUOUS,
        annealing_steps=max(7500, int(N * 2.5)),  # Scale with problem size
    )
    
    solver = LOUCETIUSSolver()
    
    print(f"\nSolving with {config.annealing_steps} steps...")
    start = time.time()
    result = solver.solve(Q, config)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Energy: {result.best_energy:.6f}")
    print(f"Consensus: {result.consensus_percentage:.1f}%")
    print(f"Speed: {result.best_energy / elapsed:.0f} energy/second")
    
    return result


# ============================================================================
# EXAMPLE 5: Solver Mode Comparison
# ============================================================================

def example_5_mode_comparison():
    """
    Example 5: Compare all three solver modes on same problem.
    
    Shows trade-offs between modes.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Solver Mode Comparison")
    print("=" * 70)
    
    # Create test problem
    N = 200
    Q = scipy.sparse.random(N, N, density=0.05, format='coo')
    
    modes = [
        (SolverMode.SPATIAL, "SPATIAL (Topology)"),
        (SolverMode.COMBINATORIAL, "COMBINATORIAL (Max-Cut)"),
        (SolverMode.CONTINUOUS, "CONTINUOUS (Soft-Spin v2.5)"),
    ]
    
    results = {}
    solver = LOUCETIUSSolver()
    
    for mode_id, mode_name in modes:
        print(f"\n{mode_name}...")
        
        config = SwarmConfig(
            num_variables=N,
            solver_mode=mode_id,
            annealing_steps=5000,
        )
        
        start = time.time()
        result = solver.solve(Q, config)
        elapsed = time.time() - start
        
        results[mode_name] = result
        
        print(f"  Energy:     {result.best_energy:.6f}")
        print(f"  Time:       {elapsed:.4f}s")
        print(f"  Consensus:  {result.consensus_percentage:.1f}%")
    
    # Comparison table
    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)
    print(f"{'Mode':<30} {'Energy':>12} {'Consensus':>12}")
    print("-" * 70)
    
    for mode_name, result in results.items():
        print(f"{mode_name:<30} {result.best_energy:>12.6f} {result.consensus_percentage:>11.1f}%")
    
    return results


# ============================================================================
# EXAMPLE 6: Batch Processing Multiple Problems
# ============================================================================

def example_6_batch_processing():
    """
    Example 6: Efficiently solve a batch of related problems.
    
    Useful for parameter sweeps, sensitivity analysis, ensemble methods.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Batch Processing")
    print("=" * 70)
    
    # Create several problem instances
    problems = [
        ("Problem A", scipy.sparse.random(100, 100, density=0.1, format='coo')),
        ("Problem B", scipy.sparse.random(150, 150, density=0.08, format='coo')),
        ("Problem C", scipy.sparse.random(200, 200, density=0.05, format='coo')),
    ]
    
    solver = LOUCETIUSSolver()
    results_dict = {}
    
    print(f"Solving {len(problems)} problems...")
    start_total = time.time()
    
    for name, Q in problems:
        config = SwarmConfig(
            num_variables=Q.shape[0],
            annealing_steps=5000,
        )
        
        result = solver.solve(Q, config)
        results_dict[name] = result
        
        print(f"{name:12} N={Q.shape[0]:3d}  energy={result.best_energy:10.6f}  "
              f"time={result.wall_time:6.3f}s")
    
    total_time = time.time() - start_total
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per problem: {total_time / len(problems):.2f}s")
    
    return results_dict


# ============================================================================
# EXAMPLE 7: Loading QUBO from Files
# ============================================================================

def example_7_file_loading():
    """
    Example 7: Load QUBO matrices from various file formats.
    
    Supports: CSV, NPZ, JSON
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: File I/O")
    print("=" * 70)
    
    # Create sample data
    Q = scipy.sparse.random(50, 50, density=0.1, format='coo')
    work_dir = Path("./qubo_samples")
    work_dir.mkdir(exist_ok=True)
    
    # Example 1: Save and load NPZ (most efficient for sparse)
    print("Saving to NPZ...")
    npz_file = work_dir / "sample.npz"
    np.savez_compressed(npz_file, Q_row=Q.row, Q_col=Q.col, Q_data=Q.data, 
                        shape=Q.shape)
    print(f"  Saved: {npz_file}")
    
    # Load from NPZ
    loaded = np.load(npz_file)
    Q_loaded = scipy.sparse.coo_matrix(
        (loaded['Q_data'], (loaded['Q_row'], loaded['Q_col'])), 
        shape=loaded['shape']
    )
    print(f"  Loaded: {Q_loaded.shape}, nnz={Q_loaded.nnz}")
    
    # Example 2: Save and load COO JSON
    print("\nSaving to JSON (COO format)...")
    json_file = work_dir / "sample.json"
    json_data = {
        'row': Q.row.tolist(),
        'col': Q.col.tolist(),
        'data': Q.data.tolist(),
        'shape': Q.shape,
    }
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_file}")
    
    # Example 3: Save dense to CSV
    print("\nSaving dense to CSV...")
    csv_file = work_dir / "sample.csv"
    np.savetxt(csv_file, Q.toarray(), delimiter=',')
    print(f"  Saved: {csv_file}")
    
    print(f"\nFiles created in: {work_dir}")
    
    return work_dir


# ============================================================================
# EXAMPLE 8: Custom Problem Construction
# ============================================================================

def example_8_custom_problem():
    """
    Example 8: Build a QUBO from scratch for a specific application.
    
    Example: Ising spin glass with external field.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Custom Problem Construction")
    print("=" * 70)
    
    # Ising spin glass: H = -sum_{ij} J_ij * sigma_i * sigma_j - h_i * sigma_i
    # Convert to QUBO with sigma_i in {-1,+1} -> x_i in {0,1} via sigma_i = 2*x_i - 1
    
    N = 10
    
    # Random coupling constants (Gaussian)
    np.random.seed(42)
    couplings = {}
    for i in range(N):
        for j in range(i+1, N):
            if np.random.rand() < 0.3:  # 30% connected
                couplings[(i, j)] = np.random.randn() * 0.5
    
    # External field
    h = np.random.randn(N) * 0.3
    
    print(f"Ising model: {N} spins, {len(couplings)} couplings")
    
    # Build QUBO matrix
    # Ising: H = sum_{i<j} J_ij * sigma_i * sigma_j + sum_i h_i * sigma_i
    # Let sigma_i = 2*x_i - 1, then:
    # H = sum_{i<j} J_ij*(2*x_i-1)*(2*x_j-1) + sum_i h_i*(2*x_i-1)
    #   = sum_{i<j} 4*J_ij*x_i*x_j - 2*J_ij*(x_i+x_j) + J_ij
    #     + 2*h_i*x_i - h_i
    # QUBO: Q_{ii} = -2*sum_j J_ij - 2*h_i, Q_{ij} = 4*J_ij
    
    Q = scipy.sparse.lil_matrix((N, N), dtype=np.float64)
    
    # Diagonal terms
    for i in range(N):
        Q[i, i] = -2 * h[i]
        for j in range(N):
            if (i, j) in couplings:
                Q[i, i] -= 2 * couplings[(i, j)]
            elif (j, i) in couplings:
                Q[i, i] -= 2 * couplings[(j, i)]
    
    # Off-diagonal terms
    for (i, j), J in couplings.items():
        Q[i, j] = 4 * J
        Q[j, i] = 4 * J
    
    Q = Q.tocoo()
    
    # Solve
    config = SwarmConfig(
        num_variables=N,
        solver_mode=SolverMode.CONTINUOUS,
        annealing_steps=3000,
    )
    
    solver = LOUCETIUSSolver()
    result = solver.solve(Q, config)
    
    # Convert back to spins
    spins = 2 * result.best_solution - 1
    
    print(f"\nSolution:")
    print(f"  Spins: {spins}")
    print(f"  Energy: {result.best_energy:.6f}")
    print(f"  Consensus: {result.consensus_percentage:.1f}%")
    
    return result


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Run all examples."""
    
    print("\n")
    print("+" + "=" * 68 + "+")
    print("|" + "Locetius v1.0 - Integration Examples".center(68) + "|")
    print("|" + "Production Python API Demonstrations".center(68) + "|")
    print("+" + "=" * 68 + "+")
    
    # System info
    print_diagnostics()
    
    try:
        # Run examples
        example_1_basic_solve()
        example_2_max_cut()
        example_3_portfolio_optimization()
        example_4_large_scale()
        example_5_mode_comparison()
        example_6_batch_processing()
        example_7_file_loading()
        example_8_custom_problem()
        
        print("\n" + "=" * 70)
        print("[OK] ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70 + "\n")
        
    except OSError as e:
        print(f"\n[!] Note: C++ library not available ({e})")
        print("  This is expected if LOUCETIUS_core.dll/so not yet compiled.")
        print("  The API structure is ready for integration once the library is built.")
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
