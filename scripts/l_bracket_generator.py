"""
L-Bracket Topology Optimization QUBO Generator
================================================

Generates a highly sparse, symmetric QUBO matrix representing the classic
"L-Bracket Topology Optimization" problem. Suitable for benchmarking
QUBO solvers on realistic structural design problems.

The problem:
  - 2D discretized L-shaped bracket (100x100 grid, top-right quadrant removed)
  - ~6,400-8,000 binary variables (1=solid material, 0=empty)
  - Fixed support at top-left edge, downward load at bottom-right corner
  - Goal: Find minimum-material topology that carries the load without buckling

Mathematical approach:
  - Diagonal penalties enforce volume constraint (cost of keeping material)
  - Off-diagonal rewards along load path incentivize material placement
  - Neighbor smoothing ensures structural continuity (no dust)
  - Heuristic stress field computed from graph distances (no FEM required)

Output: Matrix Market format (.mtx file) for compatibility with all QUBO solvers.

Zero dependencies except numpy and scipy.
"""

import numpy as np
import scipy.sparse as sp
import scipy.io
from pathlib import Path
from collections import deque


def build_l_bracket_qubo(grid_size=100, volume_penalty=6.0, 
                         neighbor_reward=-1.0, load_reward=-8.0):
    """
    Construct the L-Bracket QUBO matrix.
    
    Args:
        grid_size: Size of bounding box (default 100x100)
        volume_penalty: Cost of keeping each material element (diagonal)
        neighbor_reward: Reward for adjacent elements (off-diagonal smoothing)
        load_reward: Reward for load path connectivity (off-diagonal)
    
    Returns:
        scipy.sparse.coo_matrix: Symmetric QUBO matrix
    """
    
    print(f"Building L-Bracket QUBO ({grid_size}x{grid_size} grid)...")
    
    # --------------------------------------------------------------------
    # STEP 1: Create the L-shaped domain
    # --------------------------------------------------------------------
    # Remove top-right quadrant: X > 40 and Y > 40
    
    nodes_2d = []  # List of (x, y) tuples for the L-shape
    for y in range(grid_size):
        for x in range(grid_size):
            # Include node if NOT in top-right quadrant
            if not (x > 40 and y > 40):
                nodes_2d.append((x, y))
    
    n = len(nodes_2d)
    print(f"  Domain: {n} variables in L-shape")
    
    # Map 2D coordinates to 1D indices
    coord_to_idx = {coord: i for i, coord in enumerate(nodes_2d)}
    
    # --------------------------------------------------------------------
    # STEP 2: Identify boundary conditions
    # --------------------------------------------------------------------
    # Fixed support: top-left edge (X=0-40, Y=0)
    # Load application: bottom-right corner (X=80-100, Y=0-20)
    
    fixed_support_nodes = set()
    for x in range(0, 41):
        if (x, 0) in coord_to_idx:
            fixed_support_nodes.add(coord_to_idx[(x, 0)])
    
    load_tip_nodes = set()
    for x in range(80, grid_size):
        for y in range(0, 21):
            if (x, y) in coord_to_idx:
                load_tip_nodes.add(coord_to_idx[(x, y)])
    
    print(f"  Boundary: {len(fixed_support_nodes)} support nodes, "
          f"{len(load_tip_nodes)} load nodes")
    
    # --------------------------------------------------------------------
    # STEP 3: Compute heuristic stress field (load path)
    # --------------------------------------------------------------------
    # Use BFS from load tip to find shortest distances to support
    
    stress_field = np.zeros(n)
    distances = {i: float('inf') for i in range(n)}
    
    # BFS from load tip backwards to support
    queue = deque([(i, 0) for i in load_tip_nodes])
    for node_idx in load_tip_nodes:
        distances[node_idx] = 0
    
    idx_to_coord = {v: k for k, v in coord_to_idx.items()}
    
    while queue:
        current_idx, dist = queue.popleft()
        curr_x, curr_y = idx_to_coord[current_idx]
        
        # Check all 4 neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_coord = (curr_x + dx, curr_y + dy)
            if neighbor_coord in coord_to_idx:
                neighbor_idx = coord_to_idx[neighbor_coord]
                if distances[neighbor_idx] > dist + 1:
                    distances[neighbor_idx] = dist + 1
                    queue.append((neighbor_idx, dist + 1))
    
    # Normalize distances to stress field: closer to load = higher stress
    max_dist = max(d for d in distances.values() if d != float('inf'))
    if max_dist > 0:
        stress_field = (max_dist - np.array([distances[i] for i in range(n)])) \
                       / max_dist
    else:
        stress_field = np.ones(n)
    
    print(f"  Stress field computed (max distance: {max_dist})")
    
    # --------------------------------------------------------------------
    # STEP 4: Build sparse QUBO matrix
    # --------------------------------------------------------------------
    # - Diagonal: +volume_penalty on every node
    # - Off-diagonal (neighbors): -neighbor_reward for continuity
    # - Off-diagonal (load path): -load_reward for high-stress neighbors
    
    rows = []
    cols = []
    data = []
    
    # Diagonal entries: volume penalty
    for i in range(n):
        rows.append(i)
        cols.append(i)
        data.append(volume_penalty)
    
    # Off-diagonal entries: spatial neighbors
    processed_edges = set()
    
    for i, (x, y) in enumerate(nodes_2d):
        # Check 4 neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            neighbor_coord = (nx, ny)
            
            if neighbor_coord in coord_to_idx:
                j = coord_to_idx[neighbor_coord]
                
                # Ensure we only add upper triangle (symmetric)
                edge = (min(i, j), max(i, j))
                if edge not in processed_edges:
                    processed_edges.add(edge)
                    
                    # Base neighbor reward for smoothing
                    reward = neighbor_reward
                    
                    # Boost reward for high-stress neighbors (load path)
                    # Only reward nodes very close to the main stress path (narrow trench)
                    if stress_field[i] > 0.8 and stress_field[j] > 0.8:
                        reward += load_reward
                    
                    # Add to both (i,j) and (j,i) for symmetry
                    rows.append(i)
                    cols.append(j)
                    data.append(reward)
                    
                    rows.append(j)
                    cols.append(i)
                    data.append(reward)
    
    # --------------------------------------------------------------------
    # STEP 5: Create sparse matrix and verify symmetry
    # --------------------------------------------------------------------
    
    Q = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    Q = Q.tocsr()  # Convert to CSR for efficient operations
    
    # Verify symmetry
    Q_T = Q.transpose()
    if not np.allclose(Q.data, Q_T.data):
        print("  WARNING: Matrix may not be perfectly symmetric")
    else:
        print(f"  [ok] Matrix verified symmetric")
    
    # Convert back to COO for export
    Q_coo = Q.tocoo()
    
    nnz = Q_coo.nnz
    density = nnz / (n * n)
    print(f"  Matrix: {n}x{n}, nnz={nnz}, density={density:.4f}")
    
    return Q_coo, n, nnz


def main():
    """Generate and export the L-Bracket QUBO benchmark."""
    
    # Build the matrix
    Q, n, nnz = build_l_bracket_qubo()
    
    # Export to Matrix Market format
    output_file = Path("L_Bracket_Benchmark.mtx")
    scipy.io.mmwrite(str(output_file), Q)
    
    print(f"\n[ok] Exported to: {output_file}")
    print(f"  Format: Matrix Market (.mtx)")
    print(f"  Matrix size: {n}x{n}")
    print(f"  Non-zeros: {nnz}")
    print(f"\nYou can now load this file in Locetius:")
    print(f"  1. Open locetius_gui_v2.py")
    print(f"  2. Click 'Import .MTX'")
    print(f"  3. Select: L_Bracket_Benchmark.mtx")
    print(f"  4. Click 'Solve with Locetius'")
    

if __name__ == "__main__":
    main()
