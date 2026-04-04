"""
Locetius v1.0 - REST API Server
====================================
Lightweight FastAPI server exposing the QUBO solver over HTTP.

Endpoints:
  POST /solve           -  Solve a QUBO problem (COO sparse format)
  POST /solve/dense     -  Solve a QUBO from a dense matrix
  GET  /health          -  Health check + GPU info
  GET  /version         -  Library version
  GET  /docs            -  Auto-generated OpenAPI docs (FastAPI)

Usage:
  pip install fastapi uvicorn numpy scipy
  python LOUCETIUS_server.py              # runs on http://localhost:8765

Quick test:
  curl http://localhost:8765/health
  curl -X POST http://localhost:8765/solve \
       -H "Content-Type: application/json" \
       -d '{"N":5,"rows":[0,1,2,3,4],"cols":[0,1,2,3,4],"values":[-1,-2,-1.5,-3,-0.5]}'

Author: Christian Hayes
Date: March 2026
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

# -- FastAPI ------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
except ImportError:
    print("ERROR: FastAPI/uvicorn not installed.")
    print("Run: pip install fastapi uvicorn")
    sys.exit(1)

# -- Locetius API -------------------------------------------------------------
# Add the server's own directory to sys.path so locetius_api.py is importable
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    import scipy.sparse
    from locetius_api import (
        LOUCETIUSSolver, SwarmConfig, SolverMode, PrecisionMode,
        get_version, LOUCETIUSCore,
    )
    _SOLVER_AVAILABLE = True
except ImportError as _e:
    _SOLVER_AVAILABLE = False
    _SOLVER_IMPORT_ERROR = str(_e)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("locetius.server")

# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Locetius v1.0 QUBO Solver API",
    description=(
        "GPU-accelerated QUBO/Ising solver via Spectral Swarm Engine.\n\n"
        "- Solver modes: SPATIAL (0), COMBINATORIAL (1), CONTINUOUS (2)\n"
        "- Input: COO sparse format or dense row-major float64\n"
        "- Output: binary solution, energy, consensus, wall time"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for local use (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------

class SolveRequest(BaseModel):
    """QUBO solve request in COO (Coordinate) sparse format."""
    N: int = Field(..., gt=0, le=100000, description="Number of variables")
    rows: List[int] = Field(..., description="COO row indices (0-based)")
    cols: List[int] = Field(..., description="COO column indices (0-based)")
    values: List[float] = Field(..., description="COO matrix values")
    # solver config (all optional)
    num_swarms: int = Field(64, ge=1, le=256, description="Parallel swarms")
    solver_mode: int = Field(2, ge=0, le=2, description="0=SPATIAL 1=COMBINATORIAL 2=CONTINUOUS")
    annealing_steps: int = Field(7500, ge=1, le=1000000, description="Optimization steps")
    high_precision: bool = Field(False, description="Float64 precision (slower)")

    @validator("rows", "cols", "values")
    def non_empty(cls, v):
        if len(v) == 0:
            raise ValueError("must not be empty")
        return v

    @validator("cols")
    def lengths_match_rows(cls, v, values):
        if "rows" in values and len(v) != len(values["rows"]):
            raise ValueError("rows and cols must have same length")
        return v

    @validator("values")
    def lengths_match_indices(cls, v, values):
        if "rows" in values and len(v) != len(values["rows"]):
            raise ValueError("values must have same length as rows/cols")
        return v


class SolveDenseRequest(BaseModel):
    """QUBO solve request from a dense row-major matrix."""
    matrix: List[List[float]] = Field(..., description="NxN dense QUBO matrix (row-major)")
    num_swarms: int = Field(64, ge=1, le=256)
    solver_mode: int = Field(2, ge=0, le=2)
    annealing_steps: int = Field(7500, ge=1, le=1000000)
    high_precision: bool = Field(False)


class SolveResponse(BaseModel):
    solution: List[int]
    energy: float
    consensus: List[float]
    consensus_pct: float
    wall_time_s: float
    solver_mode: str
    gpu_used: bool
    n_variables: int


class HealthResponse(BaseModel):
    status: str
    solver_available: bool
    version: Optional[str]
    gpu: Optional[str]
    vram_mb: Optional[int]
    compute_cap: Optional[str]
    uptime_s: float


# -----------------------------------------------------------------------------
# Startup  -  preload solver
# -----------------------------------------------------------------------------

_start_time = time.time()
_core: Optional[Any] = None
_gpu_info: Dict[str, Any] = {}

@app.on_event("startup")
async def startup():
    global _core, _gpu_info
    if not _SOLVER_AVAILABLE:
        logger.error(f"Locetius API not available: {_SOLVER_IMPORT_ERROR}")
        return
    try:
        _core = LOUCETIUSCore()
        # Query GPU info
        import ctypes
        dev_name = ctypes.create_string_buffer(256)
        vram = ctypes.c_int32(0)
        cc   = ctypes.c_int32(0)
        rc = _core.get_cuda_info(dev_name, ctypes.byref(vram), ctypes.byref(cc))
        if rc == 0:
            _gpu_info = {
                "device": dev_name.value.decode(),
                "vram_mb": int(vram.value),
                "compute_cap": f"{cc.value // 10}.{cc.value % 10}",
            }
            logger.info(f"GPU: {_gpu_info['device']}  VRAM={_gpu_info['vram_mb']}MB  cc={_gpu_info['compute_cap']}")
        else:
            _gpu_info = {}
            logger.warning("CUDA not available  -  will use CPU")
    except Exception as e:
        logger.error(f"Startup error: {e}")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_MODE_NAMES = {0: "SPATIAL", 1: "COMBINATORIAL", 2: "CONTINUOUS"}

def _run_solve(N, rows, cols, values, num_swarms, solver_mode, annealing_steps, high_precision):
    if not _SOLVER_AVAILABLE:
        raise HTTPException(503, detail=f"Solver not available: {_SOLVER_IMPORT_ERROR}")

    Q = scipy.sparse.coo_matrix(
        (np.array(values, dtype=np.float64),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(N, N)
    )
    config = SwarmConfig(
        num_variables=N,
        num_swarms=num_swarms,
        solver_mode=solver_mode,
        annealing_steps=annealing_steps,
        high_precision=high_precision,
    )
    solver = LOUCETIUSSolver()
    result = solver.solve(Q, config)

    return SolveResponse(
        solution=result.best_solution.tolist(),
        energy=float(result.best_energy),
        consensus=result.consensus.tolist(),
        consensus_pct=float(result.consensus_percentage),
        wall_time_s=float(result.wall_time),
        solver_mode=_MODE_NAMES.get(solver_mode, "CONTINUOUS"),
        gpu_used=bool(_gpu_info),
        n_variables=N,
    )


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check  -  returns GPU info and library status."""
    ver = None
    if _SOLVER_AVAILABLE and _core:
        try:
            ver = _core.get_version().decode()
        except Exception:
            pass

    return HealthResponse(
        status="ok" if _SOLVER_AVAILABLE else "degraded",
        solver_available=_SOLVER_AVAILABLE,
        version=ver,
        gpu=_gpu_info.get("device"),
        vram_mb=_gpu_info.get("vram_mb"),
        compute_cap=_gpu_info.get("compute_cap"),
        uptime_s=round(time.time() - _start_time, 1),
    )


@app.get("/version", tags=["System"])
async def version():
    """Return library version string."""
    if not _SOLVER_AVAILABLE:
        raise HTTPException(503, detail="Solver not available")
    return {"version": get_version()}


@app.post("/solve", response_model=SolveResponse, tags=["Solver"])
async def solve_coo(req: SolveRequest):
    """
    Solve a QUBO problem.

    Input matrix is provided in **COO (coordinate) sparse format**  -  identical
    to scipy.sparse.coo_matrix.  Only upper or lower triangle needed for
    symmetric matrices; the solver handles the rest.

    Example (5-variable diagonal QUBO, optimal solution is all-ones):
    ```json
    {
      "N": 5,
      "rows":   [0, 1, 2, 3, 4],
      "cols":   [0, 1, 2, 3, 4],
      "values": [-1, -2, -1.5, -3, -0.5]
    }
    ```
    """
    try:
        return _run_solve(
            req.N, req.rows, req.cols, req.values,
            req.num_swarms, req.solver_mode, req.annealing_steps, req.high_precision
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        logger.exception("Solve failed")
        raise HTTPException(500, detail=str(e))


@app.post("/solve/dense", response_model=SolveResponse, tags=["Solver"])
async def solve_dense(req: SolveDenseRequest):
    """
    Solve a QUBO from a **dense NxN matrix**.

    The matrix is converted to COO internally.  Prefer `/solve` (COO) for
    large problems  -  dense matrices eat memory fast.
    """
    try:
        arr = np.array(req.matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Expected square 2D matrix, got shape {arr.shape}")
        N = arr.shape[0]
        coo = scipy.sparse.coo_matrix(arr)
        return _run_solve(
            N, coo.row.tolist(), coo.col.tolist(), coo.data.tolist(),
            req.num_swarms, req.solver_mode, req.annealing_steps, req.high_precision
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        logger.exception("Dense solve failed")
        raise HTTPException(500, detail=str(e))


@app.get("/modes", tags=["Solver"])
async def solver_modes():
    """List available solver modes with descriptions."""
    return {
        "modes": [
            {"id": 0, "name": "SPATIAL",       "description": "Volumetric swarm topology  -  good for spatial problems"},
            {"id": 1, "name": "COMBINATORIAL", "description": "Max-cut / graph partitioning  -  binary combinatorics"},
            {"id": 2, "name": "CONTINUOUS",    "description": "Soft-spin v2.5  -  general QUBO / Ising (default)"},
        ]
    }


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Locetius v1.0 REST API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev)")
    args = parser.parse_args()

    print(f"""
+--------------------------------------------------+
|   LOUCETIUS v2.5  REST API Server                |
|   http://{args.host}:{args.port}                       |
|   Docs: http://{args.host}:{args.port}/docs             |
+--------------------------------------------------+
""")
    uvicorn.run(
        "locetius_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
