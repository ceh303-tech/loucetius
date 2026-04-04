# Locetius v1.0 - Complete Feature Documentation

## Table of Contents
1. [Solver Capabilities](#solver-capabilities)
2. [GUI Features](#gui-features)
3. [API Interface](#api-interface)
4. [Performance Metrics](#performance-metrics)
5. [Advanced Configuration](#advanced-configuration)

---

## Solver Capabilities

### Three Solver Modes

#### 1. **SPATIAL Mode**
Optimized for problems with spatial structure (graphs, grids, lattices)
- Respects adjacent node relationships
- Ideal for: Topology optimization, routing, circuit design
- Speed: Fast convergence on structured problems
- Memory: Efficient sparse matrix operations

#### 2. **COMBINATORIAL Mode**
General-purpose QUBO solver
- No assumptions about problem structure
- Ideal for: Portfolio optimization, feature selection, general optimization
- Speed: Medium; handles moderate-size problems
- Memory: Standard matrix storage

#### 3. **CONTINUOUS Mode**
Relaxation-based approach for continuous variable problems
- Solves continuous relaxation first, then binarizes
- Ideal for: Problems requiring floating-point intermediates
- Speed: Slower but gives relaxation bounds
- Memory: Higher (stores continuous solution state)

### Solver Configuration Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `num_variables` | 1-14,000 | N/A | Problem size |
| `annealing_steps` | 100-50,000 | 1,000 | Iterations per phase |
| `thermal_melt` (noise) | 0.0-1.0 | 0.2 | Stochastic perturbation strength |
| `phase3_enforce` | true/false | true | Enforce integer solution |
| `early_stopping_patience` | 50-5,000 | 0 (disabled) | Stop if energy flat for N steps |
| `solver_mode` | SPATIAL, COMBINATORIAL, CONTINUOUS | SPATIAL | Algorithm selection |
| `precision` | float32, float64 | float32 | Compute precision |

---

## GUI Features

### Input Section
- **Import MTX**: Load Matrix Market format files (.mtx)
- **Import CSV**: Load edge lists with [Node_A, Node_B, Weight] format
- **Random Test**: Generate NxN random test matrices with configurable density
- **File Preview**: Shows matrix dimensions, non-zero count, sparsity

### Hyperparameters Panel
```
[Logo] Locetius v1.0
--------------------------------
Active Engine Swarms: 64 (Optimal)  [<- Telemetry Badge]
Solver Mode: [SPATIAL v]
Precision: [float32 vs float64]
--------------------------------
Thermal Melt (Noise): |||||      [0.20]  <- Bright Cyan Display
Max Iterations: |||||||          [5000]  <- Bright Cyan Display
--------------------------------
 Enforce Phase 3 (Deterministic Freeze)  [<- Toggle]
 Early Stopping (Patience steps): [500]  [<- Config]
--------------------------------
[ SOLVE WITH LOCETIUS] <- Large Primary Button
```

### Results Section
- **Energy Convergence Chart**: Real-time line plot of energy vs iteration
- **Consensus Histogram**: Distribution of swarm binary agreement
- **Solution Summary**: Best energy, consensus %, execution time
- **[View Solution Graph]**: Opens network visualization (nodes colored by value)

### GPU Stats Tab
```
Status: * GPU Ready: NVIDIA RTX 2050
---------------------------------
Device: RTX 2050
Total VRAM: 4,096 MB
Compute Capability: 8.6
---------------------------------
GPU Utilization: ||||||       45%
Memory: |||          1,024 / 4,096 MB
Temperature: ||||         52degC (Green)
Power Draw: |||          8.4 / 80.0 W
Fan Speed: |||||         35%
GPU Clock: 2,010 MHz | Memory Clock: 5,000 MHz
```

### Export Options
1. **CSV Solution**: Variables and their final state (0 or 1)
2. **JSON Convergence**: Energy history, consensus, execution time
3. **STL Topology**: 3D mesh representation (for CAD/visualization)

### Server Configuration Tab
- **Connection Status**: Live indicator (* green/o grey)
- **Server URL**: Editable field (default: http://localhost:8765)
- **Test Connection**: One-shot health check
- **Auto-start**: Option to launch REST server on GUI startup

---

## API Interface

### Python SDK (`locetius_api.py`)

#### Core Classes

**`LOUCETIUSSolver`**
```python
solver = LOUCETIUSSolver()
result = solver.solve(Q, config, convergence_callback)
```

**`SwarmConfig` (Dataclass)**
```python
config = SwarmConfig(
    num_variables=100,
    annealing_steps=1000,
    thermal_melt=0.2,
    solver_mode="SPATIAL",
    precision="float32",
    phase3_enforce=True,
    early_stopping_patience=500
)
```

**`SolveResult` (Dataclass)**
```python
result.best_solution        # numpy array of 0/1 values
result.best_energy          # float, objective value
result.consensus            # float, 0-100 percent
result.convergence_history  # list of energy values
result.execution_time       # float, seconds
```

#### Example: Batch Solving
```python
import numpy as np
from locetius_api import LOUCETIUSSolver, SwarmConfig

solver = LOUCETIUSSolver()

# Solve multiple problems
for problem_size in [10, 50, 100, 500]:
    Q = np.random.randn(problem_size, problem_size)
    Q = (Q + Q.T) / 2
    
    config = SwarmConfig(num_variables=problem_size, annealing_steps=1000)
    result = solver.solve(Q, config)
    
    print(f"{problem_size:4d} vars: E={result.best_energy:8.2f}, "
          f"consensus={result.consensus:.1f}%, time={result.execution_time:.2f}s")
```

### REST API Server (`locetius_server.py`)

**Endpoints:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Server & GPU status |
| POST | `/solve` | Solve QUBO problem |

**Request Format:**
```json
{
  "Q": [[1.0, -0.5], [-0.5, 2.0]],
  "num_variables": 2,
  "annealing_steps": 500,
  "thermal_melt": 0.2,
  "solver_mode": "SPATIAL",
  "phase3_enforce": true,
  "early_stopping_patience": 0,
  "precision": "float32"
}
```

**Response Format:**
```json
{
  "best_solution": [1, 0],
  "best_energy": -1.5,
  "consensus": 85.3,
  "convergence_history": [-1.2, -1.35, -1.45, -1.5],
  "execution_time": 8.234
}
```

---

## Performance Metrics

### Benchmarks

Standard test: L-Bracket topology optimization problem (6,519 variables)

```
+============+==========+=========+===========+
|   Mode     |   Time   | Energy  | Consensus |
+============+==========+=========+===========+
| SPATIAL    |  ~45 sec | -13200  | 28%       |
| COMBINAT.  |  ~60 sec | -13356  | 32%       |
| CONTINUOUS |  ~90 sec | -13400* | 35%*      |
+============+==========+=========+===========+
*Relaxation bounds before binarization
```

### Scaling Analysis

```
Problem Size vs Execution Time

10,000 ms +-
          |
 5,000 ms +-      ++
          |    ++ | 
 2,500 ms +-  ++  |  
          | ++    |   
 1,000 ms ++      |  
          |       +-----------------
   500 ms +-------------------------
          |
   100 ms +----+----+----+----+----
               10   50   100  500 1000
            Problem Size (Variables)
```

### Memory Usage

| Problem Size | GPU Memory | Solver Mode |
|---|---|---|
| 10 variables | ~50 MB | Any |
| 100 variables | ~75 MB | SPATIAL |
| 1,000 variables | ~250 MB | COMBINATORIAL |
| 6,519 variables | ~1,200 MB | CONTINUOUS |

**Hardware**: NVIDIA RTX 2050 (4GB VRAM total)

---

## Advanced Configuration

### Environment Variables
```bash
# Force CPU mode (if GPU fails)
export LOCETIUS_DEVICE=cpu

# Enable verbose logging
export LOCETIUS_DEBUG=1

# Custom CUDA device selection
export CUDA_VISIBLE_DEVICES=0
```

### Configuration File (Optional)
Create `locetius_config.json`:
```json
{
  "default_precision": "float32",
  "default_solver_mode": "SPATIAL",
  "rest_server_port": 8765,
  "rest_server_host": "0.0.0.0",
  "gpu_device": 0,
  "max_variables": 14000,
  "verbosity": "INFO"
}
```

### GPU Optimization Tips

1. **Batch Processing**: Process multiple small problems together
2. **Precision Selection**:
   - `float32`: ~2x faster, slightly less accurate
   - `float64`: Higher precision, uses more memory
3. **Solver Mode Choice**:
   - SPATIAL: For graph/grid problems
   - COMBINATORIAL: For general problems
   - CONTINUOUS: When bounds needed

### Performance Tuning

| Setting | Impact | Recommendation |
|---------|---------|------------------|
| `thermal_melt` | High | Start at 0.2, increase for random problems |
| `annealing_steps` | High | 1000-5000 for most problems |
| `precision` | Medium | Use float32 unless accuracy critical |
| `early_stopping_patience` | Low | Disable for exploration, enable for production |

---

## Troubleshooting

### Common Issues

**Issue**: GPU not detected
```bash
# Check CUDA installation
nvidia-smi

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Issue**: Out of memory error
- Reduce `num_variables`
- Use `float32` instead of `float64`
- Enable early stopping

**Issue**: Slow convergence
- Increase `thermal_melt` (0.3-0.5 for harder problems)
- Try CONTINUOUS mode
- Increase `annealing_steps`

---

*Last Updated: March 31, 2026*
*For support, see LOCETIUS_USER_MANUAL.pdf*
