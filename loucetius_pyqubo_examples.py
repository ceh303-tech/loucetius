"""
LOUCETIUS PyQUBO Integration Examples
======================================

Demonstrates how to use LOUCETIUS as a drop-in replacement for D-Wave
and other quantum solvers via PyQUBO.

These examples show that users don't need to rewrite their PyQUBO code -  - 
they just change one line!

Author: Christian Hayes
Date: April 2, 2026
"""

import numpy as np


# =============================================================================
# EXAMPLE 1: Simple Binary Optimization (Drop-in Replacement)
# =============================================================================

def example_1_drop_in_replacement():
    """
    Example 1: D-Wave code vs LOUCETIUS code (one line difference!)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Drop-in Replacement (D-Wave -> LOUCETIUS)")
    print("=" * 70)
    
    # First, define problem with PyQUBO (works for both!)
    try:
        from PyQUBO import Binary
    except ImportError:
        print("PyQUBO not installed. Install via: pip install pyqubo")
        return
    
    x, y, z = Binary('x'), Binary('y'), Binary('z')
    
    # Define objective: maximize x + y - 2yz (converted to minimization)
    model = -x - y + 2*y*z
    compiled_model = model.compile()
    
    print("\n[D-Wave Code]")
    print("from dwave.system import LeapHybridSampler")
    print("sampler = LeapHybridSampler()")
    print("result = sampler.sample(compiled_model)")
    
    print("\n[LOUCETIUS Code] <- SAME PROBLEM, ONE LINE DIFFERENT")
    print("from LOUCETIUS_pyqubo_api import sample")
    print("result = sample(compiled_model)")
    
    # Actually solve with LOUCETIUS
    from LOUCETIUS_pyqubo_api import sample
    
    result = sample(compiled_model, num_reads=64, annealing_steps=7500, engine='AUTO')
    
    print("\n[ok] Solved!")
    print(f"  Best solution: {result.lowest(1)[0]['solution']}")
    print(f"  Energy: {result.lowest(1)[0]['energy']:.6f}")


# =============================================================================
# EXAMPLE 2: Portfolio Optimization (Real-world Enterprise Use Case)
# =============================================================================

def example_2_portfolio_optimization():
    """
    Example 2: Portfolio rebalancing with constraints
    This is a typical aerospace/finance industry problem.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Portfolio Optimization with Constraints")
    print("=" * 70)
    
    try:
        from PyQUBO import Binary, Constraint, Penalty
    except ImportError:
        print("PyQUBO not installed. Install via: pip install pyqubo")
        return
    
    # Decision variables: which assets to include (0 or 1)
    n_assets = 20
    assets = [Binary(f'asset_{i}') for i in range(n_assets)]
    
    # Expected returns (example coefficients)
    returns = np.random.uniform(0.01, 0.05, n_assets)
    correlations = np.random.uniform(-0.3, 0.8, (n_assets, n_assets))
    
    print(f"\nPortfolio Optimization Problem:")
    print(f"  - Assets: {n_assets}")
    print(f"  - Return range: {returns.min():.4f} to {returns.max():.4f}")
    
    # Objective: maximize return, minimize correlation risk
    objective = sum(returns[i] * assets[i] for i in range(n_assets))
    risk = sum(
        correlations[i, j] * assets[i] * assets[j]
        for i in range(n_assets) for j in range(i+1, n_assets)
    )
    model = -objective + 0.5 * risk  # Minimize negative return + risk
    
    # Constraint: must select at least 5 and at most 10 assets
    min_assets_constraint = Constraint(
        sum(assets) >= 5,
        label="min_assets"
    )
    max_assets_constraint = Constraint(
        sum(assets) <= 10,
        label="max_assets"
    )
    
    # Compile with constraints
    compiled_model = model.compile()
    
    print(f"\nSolving with LOUCETIUS...")
    from LOUCETIUS_pyqubo_api import sample
    
    result = sample(
        compiled_model,
        num_reads=32,
        annealing_steps=10000,
        engine='HYBRID'  # Hybrid engine best for constrained problems
    )
    
    best = result.lowest(1)[0]
    print(f"\n[ok] Optimal Portfolio Found!")
    print(f"  Energy (objective): {best['energy']:.6f}")
    print(f"  Selected assets: {sum(best['solution'].values())}")
    print(f"  Asset composition: {dict(list(best['solution'].items())[:5])}... ({len(best['solution'])} total)")


# =============================================================================
# EXAMPLE 3: Traveling Salesman Problem (TSP)
# =============================================================================

def example_3_traveling_salesman():
    """
    Example 3: TSP encoded as QUBO via PyQUBO
    Standard optimization benchmark.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Traveling Salesman Problem (TSP)")
    print("=" * 70)
    
    try:
        from PyQUBO import Binary, Constraint, Penalty
    except ImportError:
        print("PyQUBO not installed. Install via: pip install pyqubo")
        return
    
    # TSP with 10 cities
    n_cities = 10
    
    # Random distance matrix
    np.random.seed(42)
    distances = np.random.uniform(1, 100, (n_cities, n_cities))
    # Make symmetric
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0)
    
    print(f"\nTraveling Salesman Problem:")
    print(f"  - Cities: {n_cities}")
    print(f"  - Binary variables needed: {n_cities * n_cities}")
    
    # Binary variables: x[i,j] = 1 if city i visited at step j
    x = {}
    for i in range(n_cities):
        for j in range(n_cities):
            x[i, j] = Binary(f'x_{i}_{j}')
    
    # Objective: minimize total distance
    objective = sum(
        distances[i, k] * x[i, j] * x[k, (j+1) % n_cities]
        for i in range(n_cities)
        for j in range(n_cities)
        for k in range(n_cities)
        if i != k
    )
    
    # Constraints: exactly one city per step, exactly one step per city
    constraints = 0
    for j in range(n_cities):
        constraints += Constraint(
            sum(x[i, j] for i in range(n_cities)) == 1,
            label=f"step_{j}"
        )
    
    for i in range(n_cities):
        constraints += Constraint(
            sum(x[i, j] for j in range(n_cities)) == 1,
            label=f"city_{i}"
        )
    
    model = objective + 100 * constraints
    compiled_model = model.compile()
    
    print(f"\nSolving TSP with LOUCETIUS...")
    from LOUCETIUS_pyqubo_api import sample
    
    result = sample(
        compiled_model,
        num_reads=16,
        annealing_steps=15000,
        engine='KERR'  # Kerr good for highly constrained problems
    )
    
    best = result.lowest(1)[0]
    print(f"\n[ok] TSP Solution Found!")
    print(f"  Energy (total distance): {best['energy']:.2f}")
    print(f"  Solution variables used: {sum(best['solution'].values())}")


# =============================================================================
# EXAMPLE 4: Batch Processing Multiple Problems
# =============================================================================

def example_4_batch_solving():
    """
    Example 4: Solve multiple problems efficiently (parameter sweeps)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Batch Solving Multiple Models")
    print("=" * 70)
    
    try:
        from PyQUBO import Binary
        from LOUCETIUS_pyqubo_api import sample_batch
    except ImportError:
        print("PyQUBO or LOUCETIUS not installed")
        return
    
    # Create 5 different models with varying problem sizes
    models = []
    for problem_id in range(5):
        n = 10 + problem_id * 5  # Increase size: 10, 15, 20, 25, 30
        vars_list = [Binary(f'x_{problem_id}_{i}') for i in range(n)]
        
        # Random objective
        np.random.seed(problem_id)
        weights = np.random.uniform(-1, 1, n)
        model = sum(weights[i] * vars_list[i] for i in range(n))
        models.append(model.compile())
    
    print(f"\nBatch processing:")
    print(f"  Problems: 5")
    print(f"  Sizes: 10, 15, 20, 25, 30 variables")
    
    # Solve all at once
    results = sample_batch(
        models,
        num_reads=32,
        annealing_steps=5000,
        engine='AUTO',
    )
    
    print(f"\n[ok] Batch complete!")
    for i, result in enumerate(results):
        best = result.lowest(1)[0]
        print(f"  Problem {i}: energy={best['energy']:.4f}")


# =============================================================================
# EXAMPLE 5: Real-Time Results Inspection
# =============================================================================

def example_5_results_inspection():
    """
    Example 5: Deep inspection of LOUCETIUS results
    Shows full diagnostics and solution analysis.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Results Inspection & Diagnostics")
    print("=" * 70)
    
    try:
        from PyQUBO import Binary
        from LOUCETIUS_pyqubo_api import sample
    except ImportError:
        print("PyQUBO or LOUCETIUS not installed")
        return
    
    # Simple 5-variable problem
    vars_list = [Binary(f'x_{i}') for i in range(5)]
    model = -sum(vars_list) + 2*sum(
        vars_list[i] * vars_list[j]
        for i in range(5) for j in range(i+1, 5)
    )
    compiled = model.compile()
    
    print(f"\nSolving 5-variable optimization...")
    result = sample(compiled, num_reads=10, annealing_steps=5000)
    
    print(f"\n[ok] Results Summary:")
    print(f"  - Number of reads: {result.num_reads}")
    print(f"  - Variable labels: {result.variable_labels}")
    
    print(f"\n  Top 3 Solutions:")
    for i, sol in enumerate(result.lowest(3)):
        print(f"    {i+1}. Energy={sol['energy']:.4f}, Solution={sol['solution']}")
    
    print(f"\n  All Energies: {[f'{e:.4f}' for e in result.energies]}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 70)
    print("LOUCETIUS x PYQUBO INTEGRATION EXAMPLES")
    print("Drop-in Replacement for D-Wave, Toshiba, and Others")
    print("=" * 70)
    
    # Run examples
    examples = [
        ("1", "Drop-in Replacement", example_1_drop_in_replacement),
        ("2", "Portfolio Optimization", example_2_portfolio_optimization),
        ("3", "Traveling Salesman Problem", example_3_traveling_salesman),
        ("4", "Batch Solving", example_4_batch_solving),
        ("5", "Results Inspection", example_5_results_inspection),
    ]
    
    if len(sys.argv) > 1:
        # Run specific example
        example_id = sys.argv[1]
        for ex_id, ex_name, ex_func in examples:
            if ex_id == example_id:
                try:
                    ex_func()
                except Exception as e:
                    print(f"\n[x] Error running example: {e}")
                    import traceback
                    traceback.print_exc()
                break
    else:
        # Run all
        for ex_id, ex_name, ex_func in examples:
            try:
                ex_func()
            except Exception as e:
                print(f"\n[x] Error in {ex_name}: {e}")
                continue
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
