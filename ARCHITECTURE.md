# Locetius v1.0 - Technical Architecture & Implementation

## System Architecture Diagram

```
+-----------------------------------------------------------------+
|                     Locetius v1.0 Platform                      |
+-----------------------------------------------------------------+
|                                                                 |
|  +========================================================+   |
|  |             User Interface Layer (PyQt6)               |   |
|  |                                                        |   |
|  |  +--------------+  +--------------+  +------------+  |   |
|  |  |   Main GUI   |  |  GPU Stats   |  |  Results   |  |   |
|  |  |  - Import    |  |  - VRAM      |  |  - Energy  |  |   |
|  |  |  - Solve     |  |  - Temp      |  |  - Graph   |  |   |
|  |  |  - Export    |  |  - Power     |  |  - Stats   |  |   |
|  |  +--------------+  +--------------+  +------------+  |   |
|  +========================================================+   |
|                            v                                   |
|  +========================================================+   |
|  |          Python API Layer (locetius_api.py)            |   |
|  |                                                        |   |
|  |  +-------------------------------------------------+  |   |
|  |  |  LOUCETIUSSolver                                |  |   |
|  |  |  - load_matrix()                                |  |   |
|  |  |  - solve(Q, config, callback)                   |  |   |
|  |  |  - export_solution()                            |  |   |
|  |  +-------------------------------------------------+  |   |
|  |                                                        |   |
|  |  +-------------------------------------------------+  |   |
|  |  |  SwarmConfig (Configuration)                    |  |   |
|  |  |  - num_variables                                |  |   |
|  |  |  - annealing_steps                              |  |   |
|  |  |  - thermal_melt                                 |  |   |
|  |  |  - solver_mode (SPATIAL/COMBINATORIAL/CONT.)    |  |   |
|  |  |  - phase3_enforce                               |  |   |
|  |  |  - early_stopping_patience                      |  |   |
|  |  +-------------------------------------------------+  |   |
|  +========================================================+   |
|                            v                                   |
|  +========================================================+   |
|  |       C++ Core Engine (Compiled Binaries)              |   |
|  |                                                        |   |
|  |  +--------------------------------------------------+ |   |
|  |  |  Windows: locetius_core.dll (431 KB)             | |   |
|  |  |  Linux:   liblocetius_core.so (OBFUSCATED)       | |   |
|  |  |                                                  | |   |
|  |  |  CUDA Kernels (64 Parallel Swarms)               | |   |
|  |  |  +- Phase 1: Continuous Relaxation              | |   |
|  |  |  +- Phase 2: Spectral Analysis & Diffusion      | |   |
|  |  |  +- Phase 3: Deterministic Freeze               | |   |
|  |  |                                                  | |   |
|  |  |  Proprietary Spectral Swarm Algorithm            | |   |
|  |  |  (TRADE SECRET - Implementation Hidden)          | |   |
|  |  +--------------------------------------------------+ |   |
|  +========================================================+   |
|                            v                                   |
|  +========================================================+   |
|  |           NVIDIA GPU Hardware (CUDA 12.1+)            |   |
|  |                                                        |   |
|  |  RTX GPU (2GB+ VRAM)                                   |   |
|  |  +- CUDA Cores (Vector Processing)                    |   |
|  |  +- Tensor Cores (Matrix Operations)                  |   |
|  |  +- GPU Memory (Fast Scratch Space)                   |   |
|  +========================================================+   |
|                                                                 |
+-----------------------------------------------------------------+
```

---

## Data Flow Diagram

```
User Input (QUBO Matrix)
      v
+---------------------+
|  Format Detection   |  .mtx, .csv, .json, .npz
+---------------------+
      v
+---------------------+
| Matrix Validation   |  Verify symmetric, dimensions
+---------------------+
      v
+---------------------+
| Config Preparation  |  solver_mode, annealing_steps, etc.
+---------------------+
      v
+-----------------------------------------+
|  GPU Memory Allocation                  |
|  +- Matrix Copy (Host -> Device)         |
|  +- Swarm State (64 copies)             |
|  +- Convergence History Buffer          |
+-----------------------------------------+
      v
+-----------------------------------------+
|  CUDA Kernel Launch (Spectral Swarm)    |
|  +- Phase 1: Continuous Relaxation     |
|  |  +- Gradient descent on GPU          |
|  +- Phase 2: Spectral Analysis         |
|  |  +- Cross pollination between swarms |
|  +- Phase 3: Deterministic Freeze      |
|     +- Rounding to binary solution     |
+-----------------------------------------+
      v
+-----------------------------------------+
|  Post-Processing                        |
|  +- Copy Results (Device -> Host)       |
|  +- Compute Consensus                  |
|  +- Calculate Final Energy              |
|  +- Format Convergence History          |
+-----------------------------------------+
      v
Output (Solution + Metadata)
+- best_solution: [0,1,0,1,...]
+- best_energy: -13356.5
+- consensus: 32.2%
+- convergence_history: [e0, e1, ..., eN]
```

---

## Solver Algorithm Overview

### Three-Phase Optimization Strategy

#### Phase 1: Continuous Relaxation (30% of annealing_steps)
```
Goal: Find smooth landscape minimum in [0,1]^N space

for iteration = 1 to phase1_steps:
    for each swarm:
        gradient = grad(x^T Q x + regularizer)
        x <- x - learning_rate * gradient
        x <- clip(x, 0, 1)        # Stay in [0,1]
        x += gaussian_noise * thermal_melt
    
Output: Relaxed solution with soft values in [0,1]
```

#### Phase 2: Spectral Analysis & Cross-Pollination (30% of annealing_steps)
```
Goal: Leverage spectral properties for consensus building

for iteration = 1 to phase2_steps:
    for each swarm:
        # Compute Laplacian of solution landscape
        H = grad2(x^T Q x)
        eigvals, eigvecs = eigh(H)
        
        # Project to dominant eigenvector directions
        x <- weighted_combination(x, eigvecs)
    
    # Cross-pollinate: share best solutions between swarms
    for each pair (swarm_i, swarm_j):
        if energy(swarm_i) < energy(swarm_j):
            swarm_j.x <- interpolate(swarm_j.x, swarm_i.x, alpha=0.3)

Output: Consensus building across parallel swarms
```

#### Phase 3: Deterministic Freeze (40% of annealing_steps)
```
Goal: Convert continuous relaxation to binary solution

for iteration = 1 to phase3_steps:
    for each swarm:
        x <- x + gaussian_noise * (1 - iteration/phase3_steps) * thermal_melt
        
        # Periodic hard rounding (binary discretization)
        if iteration % rounding_interval == 0:
            x_binary = round(x)
            if energy(x_binary) < energy(x):
                x <- x_binary
    
    # Optional: early stopping if no improvement
    if early_stopping_patience > 0:
        if flat_iterations > early_stopping_patience:
            break

Output: Binary solution [0,1,0,1,...] with convergence history
```

---

## Performance Characteristics

### Time Complexity
- **Phase 1**: O(N2) per iteration x M iterations
- **Phase 2**: O(N3) for eigendecomposition (reduced per iteration)
- **Phase 3**: O(N2) per iteration x M iterations
- **Overall**: O(N2 x #iterations) with GPU acceleration

### Space Complexity
- **Matrix Storage**: O(nnz) where nnz = non-zeros in Q
- **Swarm States**: O(64 x N) for 64 parallel swarms
- **Convergence History**: O(#iterations)
- **Total**: ~O(N2) in worst case (dense matrix), O(N) for sparse

### GPU Utilization
```
GPU Core Utilization during Optimization:

100% +-  +-----------------------------------------+
     |  ++                                         ++
 75% +-+                                           ++
     |                                             | 
 50% +---------------------------------------------+ (Post-processing)
     |
 25% +- (Idle - Data Transfer)
     |
  0% +----+------------+------------+--------------
      Load   Phase 1    Phase 2     Phase 3    Output
```

---

## Memory Management

### GPU Memory Allocation Strategy
1. **Matrix**: 30% of VRAM
2. **Swarm States**: 40% of VRAM (64 copies x N)
3. **Temporary Buffers**: 20% of VRAM
4. **Reserve**: 10% of VRAM (safety margin)

### Dynamic Allocation
```python
# For N variables on GPU with V GB VRAM:
bytes_per_variable = 8  # float64 or 4 for float32

if N * bytes_per_variable * 64 < 0.8 * VRAM:
    allocation = "Full 64 swarms"
elif N * bytes_per_variable * 32 < 0.8 * VRAM:
    allocation = "32 swarms"
else:
    allocation = "Sequential mode (slower)"
```

---

## File Structure

```
locetius_v1.0/
+-- Core Application
|   +-- locetius_api.py              [Python SDK wrapper for DLL]
|   +-- locetius_gui_v2.py           [PyQt6 GUI application]
|   +-- locetius_server.py           [Flask REST API server]
|   +-- locetius_integration_example.py  [Code examples]
|
+-- Solver Binaries
|   +-- locetius_core/
|   |   +-- locetius_core.dll        [Windows binary - 431 KB]
|   |   +-- liblocetius_core.so      [Linux binary - obfuscated]
|   |   +-- liblocetius_core.h       [C/C++ header reference]
|   +-- [Proprietary Algorithm Implementation]
|
+-- Benchmarks & Tools
|   +-- L_Bracket_Benchmark.mtx      [6,519-variable problem]
|   +-- l_bracket_generator.py       [Benchmark generator]
|   +-- test_*.py                    [Verification scripts]
|
+-- Documentation
|   +-- LOCETIUS_USER_MANUAL.pdf     [40+ page user guide]
|   +-- LOCETIUS_USER_MANUAL.docx    [Editable version]
|   +-- README.md                    [GitHub overview]
|   +-- FEATURES_DETAILED.md         [Feature documentation]
|   +-- INSTALL.txt                  [Installation guide]
|   +-- QUICK_START.txt              [Quick reference]
|   +-- API_REFERENCE.md             [API documentation]
|
+-- Configuration
|   +-- requirements.txt             [Python dependencies]
|   +-- logo.jpg                     [Company branding]
|   +-- locetius_logo_cyan.png       [GUI logo (cyan variant)]
|
+-- Launch Scripts
    +-- RUN_GUI.bat                  [Windows GUI launcher]
    +-- RUN_GUI.sh                   [Linux GUI launcher]
    +-- RUN_SERVER.bat               [Windows server launcher]
    +-- RUN_SERVER.sh                [Linux server launcher]
```

---

## Dependencies & Versions

### Required
- **Python**: 3.8+ (tested on 3.10, 3.11)
- **CUDA**: 12.1+ (for RTX 30/40 series)
- **cuDNN**: 8.0+ (bundled with CUDA)

### Python Packages
```
numpy>=1.20.0           [Numerical computing]
scipy>=1.6.0            [Sparse matrices, I/O]
PyQt6>=6.0.0            [GUI framework]
pynvml>=11.0.0          [GPU monitoring]
requests>=2.25.0        [REST client, optional]
Flask>=2.0.0            [REST server, optional]
matplotlib>=3.3.0       [Plotting]
networkx>=2.5           [Graph visualization]
pandas>=1.1.0           [Data handling]
```

### Optional
- **CUDA Toolkit**: For compilation (not needed for distribution)
- **cuDNN**: For advanced acceleration
- **TensorRT**: For model optimization

---

## Compilation Notes (For Reference)

The C++ solver core was compiled with:
```bash
# NVIDIA CUDA Compiler
nvcc -arch=sm_86 \
     -lcudart \
     -lcublas \
     -std=c++17 \
     -O3 \
     locetius_core.cu -o locetius_core.dll

# Obfuscation (Linux)
obfuscate-binary liblocetius_core.so
code-sign-key liblocetius_core.so
```

**Note**: Pre-compiled binaries are provided. No compilation needed for end users.

---

## Security & Trade Secrets

### What's Protected
- Core optimization algorithm implementation
- Spectral analysis methods
- Swarm consensus building strategy
- Deterministic freeze mechanism

### How It's Protected
1. **Compiled Binaries**: Algorithm in native code, not readable
2. **Obfuscation**: Linux .so file obfuscated with SHA256 hashing
3. **Licensing**: Commercial license with attribution to Christian Hayes
4. **No Source Code**: Core logic never distributed

### What's Open
- Python API wrapper (safe interface)
- REST API definitions
- Configuration schema
- Example code for integration

---

*Last Updated: March 31, 2026*
*Proprietary Architecture  -  Christian Hayes*
