# LOUCETIUS - GPU-Accelerated QUBO Solver

A free, open-source, quantum-inspired optimisation engine for solving Quadratic
Unconstrained Binary Optimisation (QUBO) problems on NVIDIA GPUs.

Built with C++/LibTorch (compiled DLL/SO) and a Python API. Runs on any
CUDA-capable GPU.

## What It Does

LOUCETIUS solves QUBO problems - finding binary variable assignments that
minimise an energy function defined by a Q matrix. It ships four solver engines that
suit different problem structures:

| Engine | Method | Best For |
|--------|--------|----------|
| **LOCETIUS** | Spectral swarm annealing (64 parallel worlds) | Small-medium structured problems |
| **KERR** | Kerr nonlinear wave oscillator network | Constrained problems, fast convergence |
| **HYBRID** | Blended LOCETIUS + KERR with tunable weight | General-purpose, mixed problem types |
| **SPARSE** | Native cuSPARSE SpMM | Large sparse problems (N > 10,000) |

An automatic routing mode (`engine='AUTO'`) profiles the Q matrix and picks the best engine.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ceh303-tech/loucetius.git
cd loucetius

# 2. Install dependencies (PyTorch with CUDA first)
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. Verify everything works
python test_quick.py
```

### Minimal Example

```python
import numpy as np
from locetius_api import LOUCETIUSSolver, SwarmConfig, SolverMode

Q = np.array([
    [-1.0, 0.5, 0.3],
    [ 0.5,-2.0, 0.1],
    [ 0.3, 0.1,-1.5]
])

solver = LOUCETIUSSolver()
config = SwarmConfig(
    num_variables=3,
    num_swarms=64,
    solver_mode=SolverMode.CONTINUOUS,
    annealing_steps=7500,
    engine='AUTO'
)

result = solver.solve(Q, config)
print(f"Energy: {result.best_energy}")
print(f"Solution: {result.best_solution}")
print(f"Consensus: {result.consensus_percentage:.1f}%")
```

### With Hard Constraints

```python
constraints = {0: 1, 2: 0}  # Pin variable 0 to 1, variable 2 to 0
result = solver.solve(Q, config, hard_constraints=constraints)
```

## Architecture

LOUCETIUS uses a multi-engine pipeline rather than a single algorithm. A
topographical profiler scans the Q matrix structure (density, sparsity, barrier
heights) and routes the problem to the most suitable engine.

The main engines:

- **LOCETIUS (Thermal)** - Simulates 64 parallel annealing worlds with controlled
  thermal noise. Good at escaping shallow local minima on rugged landscapes.
- **KERR (Wave)** - Maps variables to continuous-valued oscillators with nonlinear
  coupling, borrowed from quantum optics. The waves naturally phase-lock into
  constraint-satisfying states. Sub-millisecond on medium problems.
- **HYBRID** - Injects wave-based momentum into the thermal simulation. A weight
  parameter (`locetius_weight`) controls the mix; w=0.75 works well in practice.
- **SPARSE** - Uses cuSPARSE for problems where the Q matrix is mostly zeros.
  Tested up to 15,000 variables on a single RTX GPU.

The core solver is a compiled C++/LibTorch DLL (Windows) and SO (Linux), called
via ctypes from Python.

## PyQUBO Compatibility

If you already have PyQUBO code written for D-Wave or other solvers, LOUCETIUS
works as a drop-in backend:

```python
# Before (D-Wave):
from dwave.system import LeapHybridSampler
sampler = LeapHybridSampler()
answer = sampler.sample(model)

# After (LOUCETIUS):
from loucetius_pyqubo_api import sample
answer = sample(model)
```

The `loucetius_pyqubo_api.py` module provides `sample()`, `sample_batch()`, and a D-Wave-compatible
`SampleSet` result format.

```bash
python loucetius_pyqubo_examples.py      # Run all examples
python loucetius_pyqubo_examples.py 1    # Drop-in replacement demo
python loucetius_pyqubo_examples.py 2    # Portfolio optimisation
python loucetius_pyqubo_examples.py 3    # TSP with constraints
```

## API Reference

### `LOUCETIUSSolver`

```python
solver = LOUCETIUSSolver(lib_path=None)  # Auto-finds DLL/SO
result = solver.solve(Q, config, linear_terms=None, hard_constraints=None)
```

### `SwarmConfig`

```python
config = SwarmConfig(
    num_variables=1000,
    num_swarms=64,              # 64 or 128
    solver_mode=SolverMode.CONTINUOUS,  # SPATIAL=0, COMBINATORIAL=1, CONTINUOUS=2
    annealing_steps=7500,
    high_precision=False,       # True for Float64
    temperature_init=1.0,
    temperature_final=0.01,
    penalty_init=0.5,
    penalty_final=12.0,
    engine='AUTO',              # AUTO, LOCETIUS, KERR, HYBRID, SPARSE
    locetius_weight=0.5,        # Hybrid blend: 0.0=pure Kerr, 1.0=pure Locetius
    pump_amplitude=0.7,         # Kerr pump strength
    kerr_nonlinearity=0.5       # Kerr coupling
)
```

### `SolveResult`

| Attribute | Description |
|-----------|-------------|
| `best_solution` | Binary vector (optimal assignment) |
| `best_energy` | Minimum energy found |
| `wall_time` | Computation time (seconds) |
| `consensus_percentage` | Agreement across parallel worlds (0-100%) |
| `solver_mode` | Which engine was used |

## GUI & REST Server

```bash
python locetius_gui_v2.py     # Interactive GUI with matrix editor
python locetius_server.py     # REST API on http://localhost:5000
```

REST example:
```bash
curl -X POST http://localhost:5000/solve \
  -H "Content-Type: application/json" \
  -d '{"Q": [[-1.0, 0.5], [0.5, -1.0]], "num_swarms": 64, "engine": "AUTO"}'
```

## Requirements

- **GPU:** NVIDIA with CUDA compute capability sm_86+ (RTX 20-series or newer)
- **CUDA:** 12.1+ (driver >= 527.41)
- **VRAM:** 4 GB minimum (24 GB for 128-swarm mode)
- **Python:** 3.8+
- **OS:** Windows 10/11 or Linux (Ubuntu 20.04+)

See [requirements.txt](requirements.txt) for Python dependencies.

## Test Results (RTX hardware)

From the benchmark suite using QPLIB and structured lattice instances:

| Test | Variables | Engine | Energy | Time | Consensus |
|------|-----------|--------|--------|------|-----------|
| 2D Grid (20x20) | 400 | LOCETIUS | -0.37 | 11.4s | 70.1% |
| 3D Dimer (3x8x8) | 384 | LOCETIUS | -1.44 | 14.2s | 46.7% |
| QPLIB_3506 | 496 | KERR | 1000.0 | 0.88s | - |
| QPLIB_3850 | 1,225 | KERR | 2868.0 | 0.11s | - |
| Quantum-hard (s28-qac) | 1,322 | KERR | 38232.0 | 5.0s | - |
| Synthetic sparse | 15,000 | SPARSE | - | ok | - |
| HYBRID sweep (w=0.75) | 400 | HYBRID | -1280.0 | <1s | - |

These are raw results without comparison to known optima. If you run LOUCETIUS on a
problem with a known optimal, please open an issue with your results.

## Known Limitations

- The LOCETIUS DLL engine uses a fixed 64-swarm topology. Requesting S=128 falls
  back to the KERR engine automatically.
- Not benchmarked head-to-head against commercial solvers (Gurobi, D-Wave, Toshiba
  CIM). Community benchmarking welcome.
- Dense problems above ~5,000 variables may exceed VRAM on consumer GPUs.

## Background

This solver grew out of a practical need - optimising aerofoil geometry for a
hand-launched glider project. Rather than paying for commercial optimisation
software, I built a GPU-accelerated solver from scratch. The approach was designed
visually (thinking about energy landscapes as physical terrain) rather than
derived from textbook equations.

If you're a researcher, engineer, or student who needs a free QUBO solver with GPU
acceleration, give it a try and let me know how it goes.

## Contributing

- Report bugs: open a GitHub issue
- Share benchmarks: run it on your problem and tell us how it did
- Suggest improvements: PRs welcome

## License

Dual-licensed:

1. **AGPLv3** - free for academic, research, and open-source use
2. **Commercial licence** - required for proprietary/closed-source use, contact ceh303@gmail.com

See [LICENSE](LICENSE) for full terms.

## Citation

If you use LOUCETIUS in published research, please cite:

```bibtex
@software{LOUCETIUS2026,
  author  = {Hayes, Christian},
  title   = {LOUCETIUS: GPU-Accelerated Quantum-Inspired QUBO Solver},
  year    = {2026},
  url     = {https://github.com/ceh303-tech/loucetius}
}
```

## Contact

Bug reports and questions: GitHub Issues  
Commercial licensing: ceh303@gmail.com

*Copyright 2026 Christian Hayes.*
