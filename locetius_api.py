"""
Locetius v1.0 - Production Python API Wrapper
==============================================

A high-performance Python interface to the Locetius C++ core library.
Wraps the compiled locetius_core.dll/so shared library using ctypes.

Supports:
  - Sparse and dense QUBO matrix input
  - Three solver modes (SPATIAL, COMBINATORIAL, CONTINUOUS)
  - Float32 (fast) and Float64 (precision) computation
  - Hard constraint enforcement
  - Consensus-based uncertainty quantification

Author: Christian Hayes
Date: March 2026
License: Proprietary  -  All Rights Reserved
"""

import ctypes
import os
import numpy as np
import scipy.sparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union, Tuple, Dict, Optional, List
import time
import platform
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- DLL search path: add parent directory so CUDA/LibTorch companion DLLs are
#    found when this package runs from a subdirectory (e.g., /final).
if os.name == 'nt':
    _dll_dirs_to_add = [
        Path(__file__).parent,           # this directory
        Path(__file__).parent.parent,    # one level up (source dev dir)
    ]
    _extra_paths = [str(d) for d in _dll_dirs_to_add if d.is_dir()]
    if _extra_paths:
        os.environ['PATH'] = os.pathsep.join(_extra_paths) + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        for _d in _dll_dirs_to_add:
            if _d.is_dir():
                try:
                    os.add_dll_directory(str(_d))
                except Exception:
                    pass


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class SolverMode:
    """Solver mode identifiers matching C++ enum."""
    SPATIAL = 0
    COMBINATORIAL = 1
    CONTINUOUS = 2

    @staticmethod
    def is_valid(mode: int) -> bool:
        """Validate solver mode."""
        return mode in [0, 1, 2]

    @staticmethod
    def name(mode: int) -> str:
        """Get human-readable name for mode."""
        names = {0: "SPATIAL", 1: "COMBINATORIAL", 2: "CONTINUOUS"}
        return names.get(mode, "UNKNOWN")


class PrecisionMode:
    """Precision mode identifiers."""
    FLOAT32 = 0
    FLOAT64 = 1

    @staticmethod
    def is_valid(mode: int) -> bool:
        return mode in [0, 1]

    @staticmethod
    def numpy_dtype(mode: int) -> np.dtype:
        """Get numpy dtype for precision mode."""
        return np.float64 if mode == 1 else np.float32

    @staticmethod
    def name(mode: int) -> str:
        return "FLOAT64" if mode == 1 else "FLOAT32"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SwarmConfig:
    """
    Configuration container for the Locetius solver.
    
    Maps directly to C++ struct for zero-copy parameter passing.
    """
    num_variables: int
    num_swarms: int = 64
    solver_mode: int = SolverMode.CONTINUOUS
    annealing_steps: int = 7500
    high_precision: bool = False
    temperature_init: float = 1.0
    temperature_final: float = 0.01
    penalty_init: float = 0.5
    penalty_final: float = 12.0
    phase3_enforce: bool = True
    early_stopping_patience: int = 0

    # -- Engine routing (Python-only, not sent to C struct) ------------------
    # "AUTO"     : traffic controller decides (default)
    # "LOCETIUS" : force C++ DLL (Spectral Swarm Engine)
    # "KERR"     : Kerr wave oscillator network
    # "HYBRID"   : blended Locetius + Kerr (locetius_weight controls blend)
    # "SPARSE"   : cuSPARSE SpMM for N>10K sparse problems
    engine: str = "AUTO"

    # Hybrid blend: 0.0 = pure Kerr, 1.0 = pure Locetius
    locetius_weight: float = 0.5

    # Kerr wave physics parameters
    pump_amplitude: float = 0.7
    kerr_nonlinearity: float = 0.5

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_variables <= 0:
            raise ValueError(f"num_variables must be > 0, got {self.num_variables}")
        # Support 64 (standard) or 128 (high-quality) parallel worlds.
        if self.num_swarms not in (64, 128):
            logger.warning(
                f"num_swarms must be 64 or 128; clamping {self.num_swarms} to 64"
            )
            self.num_swarms = 64
        if self.num_swarms <= 0 or self.num_swarms > 256:
            raise ValueError(f"num_swarms must be in [1, 256], got {self.num_swarms}")
        if not SolverMode.is_valid(self.solver_mode):
            raise ValueError(f"Invalid solver_mode: {self.solver_mode}")
        if self.annealing_steps <= 0:
            raise ValueError(f"annealing_steps must be > 0, got {self.annealing_steps}")
        if not (0.0 < self.temperature_final < self.temperature_init):
            raise ValueError(f"Temperature bounds invalid: T_init={self.temperature_init}, T_final={self.temperature_final}")
    
    def as_ctypes_struct(self) -> ctypes.Structure:
        """Convert to ctypes-compatible structure."""
        # Must include _reserved[32] so layout matches C struct exactly
        class CSwarmConfig(ctypes.Structure):
            _fields_ = [
                ("num_variables",   ctypes.c_int32),
                ("num_swarms",      ctypes.c_int32),
                ("solver_mode",     ctypes.c_int32),
                ("annealing_steps", ctypes.c_uint64),
                ("high_precision",  ctypes.c_bool),
                ("temperature_init",  ctypes.c_double),
                ("temperature_final", ctypes.c_double),
                ("penalty_init",    ctypes.c_double),
                ("penalty_final",   ctypes.c_double),
                ("_reserved",       ctypes.c_uint8 * 32),
            ]

        return CSwarmConfig(
            num_variables=self.num_variables,
            num_swarms=self.num_swarms,
            solver_mode=self.solver_mode,
            annealing_steps=self.annealing_steps,
            high_precision=self.high_precision,
            temperature_init=self.temperature_init,
            temperature_final=self.temperature_final,
            penalty_init=self.penalty_init,
            penalty_final=self.penalty_final,
        )


@dataclass
class SolveResult:
    """Container for solver results."""
    best_solution: np.ndarray  # [N] binary solution
    best_energy: float         # Best energy found
    wall_time: float           # Computation time (seconds)
    consensus: np.ndarray      # [N] consensus voting [0,1]
    consensus_percentage: float # % variables with >90% agreement
    num_steps: int
    solver_mode: str
    precision_mode: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'best_solution': self.best_solution.tolist(),
            'best_energy': float(self.best_energy),
            'wall_time': float(self.wall_time),
            'consensus': self.consensus.tolist(),
            'consensus_percentage': float(self.consensus_percentage),
            'num_steps': int(self.num_steps),
            'solver_mode': self.solver_mode,
            'precision_mode': self.precision_mode,
        }
    
    def __str__(self) -> str:
        """Pretty-print results."""
        lines = [
            "=" * 70,
            "LOCETIUS v1.0 SOLVER RESULTS",
            "=" * 70,
            f"Best Energy:           {self.best_energy:.6f}",
            f"Wall Time:             {self.wall_time:.4f} seconds",
            f"Consensus Agreement:   {self.consensus_percentage:.1f}%",
            f"Solver Mode:           {self.solver_mode}",
            f"Precision:             {self.precision_mode}",
            f"Optimization Steps:    {self.num_steps}",
            f"Solution Length:       {len(self.best_solution)}",
            "=" * 70,
        ]
        return "\n".join(lines)


# ============================================================================
# CORE API
# ============================================================================

class LOUCETIUSCore:
    """
    Low-level ctypes interface to locetius_core.dll/so.
    
    Handles:
      - Library loading (platform-aware)
      - C ABI marshaling
      - Memory safety
      - Error propagation
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        """
        Load the Locetius C++ core library.
        
        Args:
            lib_path: Explicit path to locetius_core.dll/so
                     If None, searches standard locations.
        
        Raises:
            OSError: If library cannot be found or loaded.
            RuntimeError: If required symbols are missing.
        """
        self._lib = None
        self._lib_path = lib_path
        self._load_library()
        self._bind_symbols()
    
    def _load_library(self):
        """Load platform-appropriate shared library."""
        search_paths = []
        
        if self._lib_path:
            search_paths.append(Path(self._lib_path))
        
        # Platform-specific default locations
        system = platform.system()
        if system == "Windows":
            search_paths.extend([
                Path.cwd() / "locetius_core.dll",
                Path.cwd() / "core files" / "locetius_core.dll",
                Path(__file__).parent / "locetius_core.dll",
            ])
            lib_name = "locetius_core.dll"
        else:  # Linux/macOS
            search_paths.extend([
                Path.cwd() / "liblocetius_core.so",
                Path.cwd() / "core files" / "liblocetius_core.so",
                Path(__file__).parent / "liblocetius_core.so",
                Path("/usr/local/lib/liblocetius_core.so"),
            ])
            lib_name = "liblocetius_core.so"
        
        # On Windows, add torch lib dir so DLL dependencies resolve
        if platform.system() == "Windows":
            try:
                import importlib.util
                torch_spec = importlib.util.find_spec("torch")
                if torch_spec:
                    torch_lib = str(Path(torch_spec.origin).parent / "lib")
                    os.add_dll_directory(torch_lib)
                    logger.debug(f"Added DLL directory: {torch_lib}")
                # Also add the DLL's own directory for co-located torch DLLs
                for p in search_paths:
                    if p.parent.is_dir():
                        try:
                            os.add_dll_directory(str(p.parent))
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"Could not set DLL search path: {e}")

        last_error = None
        for path in search_paths:
            if path.exists():
                try:
                    self._lib = ctypes.CDLL(str(path))
                    logger.info(f"Loaded Locetius core from: {path}")
                    return
                except OSError as e:
                    last_error = e
                    logger.debug(f"Failed to load from {path}: {e}")
        
        # If we get here, library not found
        raise OSError(
            f"Could not locate {lib_name}. "
            f"Searched: {[str(p) for p in search_paths[:3]]}. "
            f"Last error: {last_error}"
        )
    
    def _bind_symbols(self):
        """Bind C++ function signatures."""
        if not self._lib:
            raise RuntimeError("Library not loaded")
        
        try:
            # solve_qubo_coo(
            #   int N,
            #   int nnz,
            #   int* row_indices,    [nnz]
            #   int* col_indices,    [nnz]
            #   double* values,      [nnz]
            #   SwarmConfig* config,
            #   int* solution_out,   [N]  (output, must be pre-allocated)
            #   double* energy_out,  (output)
            #   double* consensus_out, [N] (output)
            #   double* timing_out   (output)
            # ) -> int (return code)
            
            self.solve_qubo_coo = self._lib.solve_qubo_coo
            self.solve_qubo_coo.argtypes = [
                ctypes.c_int,               # N
                ctypes.c_int,               # nnz
                ctypes.POINTER(ctypes.c_int),      # row_indices
                ctypes.POINTER(ctypes.c_int),      # col_indices
                ctypes.POINTER(ctypes.c_double),   # values
                ctypes.POINTER(ctypes.c_char),     # config (opaque)
                ctypes.POINTER(ctypes.c_int),      # solution_out
                ctypes.POINTER(ctypes.c_double),   # energy_out
                ctypes.POINTER(ctypes.c_double),   # consensus_out
                ctypes.POINTER(ctypes.c_double),   # timing_out
            ]
            self.solve_qubo_coo.restype = ctypes.c_int
            
            # Lifecycle
            self.initialize = self._lib.LOUCETIUS_initialize
            self.initialize.argtypes = []
            self.initialize.restype = ctypes.c_int
            self.initialize()

            # Version query
            self.get_version = self._lib.LOUCETIUS_get_version
            self.get_version.argtypes = []
            self.get_version.restype = ctypes.c_char_p

            # CUDA info
            self.get_cuda_info = self._lib.LOUCETIUS_get_cuda_info
            self.get_cuda_info.argtypes = [
                ctypes.POINTER(ctypes.c_char),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            self.get_cuda_info.restype = ctypes.c_int

            # Last error
            self.get_last_error = self._lib.LOUCETIUS_get_last_error
            self.get_last_error.argtypes = []
            self.get_last_error.restype = ctypes.c_char_p

            logger.info("Bound all C++ symbols successfully")
        
        except AttributeError as e:
            raise RuntimeError(f"Missing required symbol in C++ library: {e}")


# ============================================================================
# ENGINE MODULE LOADER
# ============================================================================

def _load_engine_module(module_name: str):
    """
    Import an engine module. Tries normal import first, then searches for a
    compiled .pyc in __pycache__ (for obfuscated distributions) or the parent
    working directory.
    """
    import importlib.util as _ilu
    import sys as _sys

    # 1. Already imported?
    if module_name in _sys.modules:
        return _sys.modules[module_name]

    # 2. Normal import (works when .py source is present)
    try:
        import importlib as _il
        return _il.import_module(module_name)
    except ImportError:
        pass

    # 3. Search for .pyc in __pycache__ beside this file, or in parent dir
    _here = Path(__file__).parent
    candidates = (
        list(_here.glob(f"__pycache__/{module_name}.cpython-*.pyc"))
        + list(_here.parent.glob(f"__pycache__/{module_name}.cpython-*.pyc"))
        + list(_here.parent.glob(f"{module_name}.py"))
    )
    if candidates:
        spec = _ilu.spec_from_file_location(module_name, candidates[0])
        mod = _ilu.module_from_spec(spec)
        _sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    raise ModuleNotFoundError(
        f"Engine module '{module_name}' not found. "
        f"Searched: {_here}/__pycache__/ and {_here.parent}/"
    )


# ============================================================================
# PUBLIC API
# ============================================================================

class LOUCETIUSSolver:
    """
    High-level Python API for LOUCETIUS v2.5.
    
    Usage:
        >>> config = SwarmConfig(num_variables=1000, solver_mode=SolverMode.CONTINUOUS)
        >>> solver = LOUCETIUSSolver()
        >>> Q = scipy.sparse.coo_matrix(...)  # QUBO matrix
        >>> result = solver.solve(Q, config)
        >>> print(result.best_energy)
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize solver.  The C++ core library is loaded lazily  -  only when
        the problem actually routes to the DLL (dense, N <= 10 000).  This avoids
        a LibTorch / PyTorch symbol conflict when large-sparse problems route to
        the pure-PyTorch SparseMergedSolver without ever touching the DLL.

        Args:
            lib_path: Optional explicit path to shared library.
        """
        self._lib_path = lib_path
        self.core: Optional[LOUCETIUSCore] = None  # loaded on first DLL-bound call
        logger.info("LOUCETIUS v2.5 API ready (DLL loads lazily on first dense solve)")

    def _ensure_core(self):
        """Load C++ library on first use."""
        if self.core is None:
            self.core = LOUCETIUSCore(self._lib_path)

    def solve(
        self,
        Q: Union[scipy.sparse.coo_matrix, np.ndarray],
        config: SwarmConfig,
        linear_terms: Optional[np.ndarray] = None,
        hard_constraints: Optional[Dict[int, int]] = None,
    ) -> SolveResult:
        """
        Solve a QUBO problem using LOUCETIUS v2.5.
        
        Args:
            Q: Sparse COO matrix (nnz x 3) or dense (N x N) QUBO matrix
            config: SwarmConfig with solver parameters
            linear_terms: Optional linear terms h (absorbed into Q diagonal)
            hard_constraints: Dict {variable_idx: {0,1}} for pinned variables
        
        Returns:
            SolveResult with best_solution, energy, time, consensus
        
        Raises:
            ValueError: If input validation fails
            RuntimeError: If C++ solver returns error code
            MemoryError: If allocation fails
        """
        start_time = time.time()
        
        # Convert input to COO format
        Q_coo = self._validate_and_convert_matrix(Q, config.num_variables)
        
        # Merge linear terms into Q diagonal if provided
        if linear_terms is not None:
            Q_coo = self._merge_linear_terms(Q_coo, linear_terms)
        
        # Apply hard constraints
        if hard_constraints:
            Q_coo = self._apply_hard_constraints(Q_coo, hard_constraints)

        N   = config.num_variables
        nnz = Q_coo.nnz
        max_entries = N * N
        sparsity = 1.0 - (nnz / max_entries) if max_entries > 0 else 0.0

        # -- Engine routing ------------------------------------------------
        engine = (config.engine or "AUTO").upper()

        if engine == "KERR":
            return self._solve_kerr_hybrid(Q_coo, config, hard_constraints, N, locetius_w=0.0)

        if engine == "HYBRID":
            return self._solve_kerr_hybrid(Q_coo, config, hard_constraints, N,
                                            locetius_w=config.locetius_weight)

        if engine == "SPARSE" or (engine == "AUTO" and N > 10_000 and sparsity > 0.95):
            return self._solve_sparse(Q_coo, config, hard_constraints, N, nnz)
        # -----------------------------------------------------------------
        # Default / "LOCETIUS" / "AUTO" (small or dense) -> C++ DLL

        # Prepare dense arrays for C++ interface
        nnz = Q_coo.nnz
        
        # Convert to int64 for safety (C++ may use 32-bit)
        rows = np.asarray(Q_coo.row, dtype=np.int32)
        cols = np.asarray(Q_coo.col, dtype=np.int32)
        data = np.asarray(Q_coo.data, dtype=np.float64)
        
        logger.info(f"Solving QUBO: N={N}, nnz={nnz}, mode={SolverMode.name(config.solver_mode)}")
        
        # Pre-allocate output buffers
        solution = np.zeros(N, dtype=np.int32)
        energy = np.array([0.0], dtype=np.float64)
        consensus = np.zeros(N, dtype=np.float64)
        timing = np.array([0.0], dtype=np.float64)
        
        # Convert config to C struct and pass by reference
        c_config = config.as_ctypes_struct()

        # Load DLL now (lazy  -  avoids LibTorch/PyTorch symbol conflict on sparse path)
        self._ensure_core()

        # Call C++ solver
        try:
            rc = self.core.solve_qubo_coo(
                N,
                nnz,
                rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.cast(ctypes.byref(c_config), ctypes.POINTER(ctypes.c_char)),
                solution.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                energy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                consensus.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                timing.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        except Exception as e:
            raise RuntimeError(f"C++ solver call failed: {e}")
        
        if rc != 0:
            # DLL returned error  -  fall back to KERR engine automatically
            err_msg = ""
            try:
                err_msg = self.core.get_last_error()
                if isinstance(err_msg, bytes):
                    err_msg = err_msg[:120].decode(errors="replace")
            except Exception:
                pass
            logger.warning(
                f"Locetius DLL returned code {rc}  -  falling back to KERR engine. "
                f"DLL error: {err_msg[:80] if err_msg else 'n/a'}"
            )
            return self._solve_kerr_hybrid(Q_coo, config, hard_constraints, N, locetius_w=0.0)

        wall_time = time.time() - start_time

        # Re-pin hard-constrained bits (solver may override them; penalty is strong
        # but not infinite, so post-process to guarantee the caller's contract).
        if hard_constraints:
            for v, c in hard_constraints.items():
                solution[v] = c

        # Consensus percentage: average swarm agreement across all variables.
        # Formula: mean(|frac_voting_1 - 0.5|) * 200
        #   0%   = every variable is perfectly 50/50 split across swarms (maximally uncertain)
        #   100% = every variable has unanimous swarm agreement (maximally decided)
        # This varies naturally with problem difficulty, mode, and annealing depth.
        consensus_pct = float(np.mean(np.abs(consensus - 0.5)) * 200.0)
        
        logger.info(f"Solved in {wall_time:.4f}s, energy={energy[0]:.6f}, consensus={consensus_pct:.1f}%")
        
        return SolveResult(
            best_solution=solution.astype(np.int32),
            best_energy=float(energy[0]),
            wall_time=float(timing[0]),
            consensus=consensus,
            consensus_percentage=float(consensus_pct),
            num_steps=config.annealing_steps,
            solver_mode=SolverMode.name(config.solver_mode),
            precision_mode=PrecisionMode.name(int(config.high_precision)),
        )

    def _solve_kerr_hybrid(
        self,
        Q_coo: scipy.sparse.coo_matrix,
        config: 'SwarmConfig',
        hard_constraints: Optional[Dict[int, int]],
        N: int,
        locetius_w: float = 0.0,
    ) -> 'SolveResult':
        """
        Route to the Python Kerr / Hybrid solver (merged_solver_cupy.py).

        locetius_w = 0.0 -> pure Kerr wave
        locetius_w = 1.0 -> pure Locetius thermal
        0 < locetius_w < 1 -> Hybrid blend
        """
        try:
            _msmod = _load_engine_module('merged_solver_cupy')
            MergedQuantumSolver = _msmod.MergedQuantumSolver
        except (ModuleNotFoundError, AttributeError) as exc:
            raise RuntimeError(
                "Merged (Kerr/Hybrid) solver not found. "
                "merged_solver_cupy is required for KERR and HYBRID engines."
            ) from exc

        mode_str = "KERR" if locetius_w == 0.0 else f"HYBRID(w={locetius_w:.2f})"
        logger.info(f"Routing to {mode_str} engine: N={N}, swarms={config.num_swarms}")

        import time as _time
        t0 = _time.time()

        solver = MergedQuantumSolver(
            Q_coo.shape[0],
            num_swarms=config.num_swarms,
            annealing_steps=config.annealing_steps,
            locetius_weight=locetius_w,
            pump_amplitude=config.pump_amplitude,
            kerr_nonlinearity=config.kerr_nonlinearity,
            dtype='float64' if config.high_precision else 'float32',
            seed=42,
        )
        result_dict = solver.solve(Q_coo)
        wall = _time.time() - t0

        sol_bin    = result_dict['solution']
        energy_val = result_dict['energy']

        solution = np.asarray(sol_bin, dtype=np.int32)
        if hard_constraints:
            for v, c in hard_constraints.items():
                solution[v] = c

        # Consensus from cross-swarm agreement (0-100 scale)
        raw_cons = float(result_dict.get('consensus', 0.5))
        consensus_pct = (raw_cons - 0.5) * 200.0 if raw_cons >= 0.5 else 0.0
        consensus_arr = np.full(N, raw_cons, dtype=np.float64)

        return SolveResult(
            best_solution=solution,
            best_energy=float(energy_val),
            wall_time=float(wall),
            consensus=consensus_arr,
            consensus_percentage=float(consensus_pct),
            num_steps=config.annealing_steps,
            solver_mode=mode_str,
            precision_mode=PrecisionMode.name(int(config.high_precision)),
        )

    def _solve_sparse(
        self,
        Q_coo: scipy.sparse.coo_matrix,
        config: 'SwarmConfig',
        hard_constraints: Optional[Dict[int, int]],
        N: int,
        nnz: int,
    ) -> 'SolveResult':
        """Route large sparse QUBOs to SparseMergedSolver (PyTorch + cuSPARSE)."""
        try:
            _ssmod = _load_engine_module('sparse_qubo_solver')
            SparseMergedSolver = _ssmod.SparseMergedSolver
        except (ModuleNotFoundError, AttributeError) as exc:
            raise RuntimeError(
                "SparseMergedSolver not found (sparse_qubo_solver.py missing). "
                "Cannot solve N>10K sparse QUBO without it."
            ) from exc

        logger.info(
            f"Routing to SparseMergedSolver: N={N}, nnz={nnz}, "
            f"sparsity={1 - nnz / (N * N):.4%}"
        )

        # Scale steps: more steps for smaller N within the sparse regime
        if N < 50_000:
            steps = min(config.annealing_steps, 2000)
        elif N < 200_000:
            steps = min(config.annealing_steps, 1000)
        else:
            steps = min(config.annealing_steps, 600)

        solver_sp = SparseMergedSolver(
            Q_coo,
            num_swarms=config.num_swarms,
            annealing_steps=steps,
            seed=42,
        )
        import time as _time
        t0 = _time.time()
        result_dict = solver_sp.solve()
        wall = _time.time() - t0

        sol_binary = result_dict['solution']
        energy_val = result_dict['energy']

        solution = sol_binary.astype(np.int32)
        if hard_constraints:
            for v, c in hard_constraints.items():
                solution[v] = c

        # Uniform 50 % consensus placeholder (sparse solver runs 1 best solution)
        consensus = np.full(N, 0.5, dtype=np.float64)
        consensus_pct = 0.0

        return SolveResult(
            best_solution=solution,
            best_energy=float(energy_val),
            wall_time=float(wall),
            consensus=consensus,
            consensus_percentage=consensus_pct,
            num_steps=steps,
            solver_mode="SPARSE_CUDA",
            precision_mode=PrecisionMode.name(int(config.high_precision)),
        )

    @staticmethod
    def _validate_and_convert_matrix(
        Q: Union[scipy.sparse.coo_matrix, np.ndarray],
        expected_size: int
    ) -> scipy.sparse.coo_matrix:
        """Convert input to COO format with validation."""
        if isinstance(Q, np.ndarray):
            if Q.ndim != 2:
                raise ValueError(f"Expected 2D matrix, got {Q.ndim}D")
            if Q.shape[0] != Q.shape[1]:
                raise ValueError(f"Expected square matrix, got {Q.shape}")
            if Q.shape[0] != expected_size:
                raise ValueError(f"Matrix size {Q.shape[0]} != config size {expected_size}")
            Q = scipy.sparse.coo_matrix(Q)
        
        elif isinstance(Q, scipy.sparse.coo_matrix):
            if Q.shape[0] != expected_size or Q.shape[1] != expected_size:
                raise ValueError(f"Matrix size {Q.shape} != config size {expected_size}")
        
        elif isinstance(Q, scipy.sparse.spmatrix):
            Q = Q.tocoo()
            if Q.shape[0] != expected_size or Q.shape[1] != expected_size:
                raise ValueError(f"Matrix size {Q.shape} != config size {expected_size}")
        else:
            raise TypeError(f"Expected numpy or scipy.sparse matrix, got {type(Q)}")
        
        return Q.tocoo()  # Ensure COO format
    
    @staticmethod
    def _merge_linear_terms(Q: scipy.sparse.coo_matrix, h: np.ndarray) -> scipy.sparse.coo_matrix:
        """Merge linear terms h into Q diagonal."""
        if h.shape[0] != Q.shape[0]:
            raise ValueError(f"Linear terms size {h.shape[0]} != matrix size {Q.shape[0]}")
        
        # Create diagonal matrix from h
        h_diag = scipy.sparse.diags(h, format='coo')
        
        # Add to Q
        Q_merged = (Q + h_diag).tocoo()
        Q_merged.sum_duplicates()  # Combine duplicate entries
        return Q_merged
    
    @staticmethod
    def _apply_hard_constraints(
        Q: scipy.sparse.coo_matrix,
        constraints: Dict[int, int]
    ) -> scipy.sparse.coo_matrix:
        """
        Enforce hard constraints via the penalty / variable-elimination method.

        For each pinned variable  v = c  (c in {0,1}):
          1. Remove all off-diagonal entries in row v and col v from Q.
          2. If c == 1  absorb the column coupling into the diagonal of the
             remaining free variables: Q'[j,j] += Q[v,j] + Q[j,v]  for j!=v.
          3. Set Q[v,v] to a large negative penalty so the DLL reliably picks
             x[v] = c  (for c=1: -P forces x[v]=1; for c=0: large +P diagonal
             makes x[v]=0 worse than any other choice -> set to +P instead).
          4. The final post-processing step re-pins the bit after extraction.
        """
        N = Q.shape[0]
        # Determine penalty magnitude from Q's own scale
        data_abs = np.abs(Q.data)
        P = float(data_abs.max()) * 1000.0 if data_abs.size > 0 else 1000.0

        # Work in LIL for efficient row/column surgery
        Q_lil = Q.tolil()

        for v, c in constraints.items():
            if not (0 <= v < N):
                raise ValueError(f"Constraint variable {v} out of range [0, {N})")
            if c not in (0, 1):
                raise ValueError(f"Constraint value for variable {v} must be 0 or 1, got {c}")

            if c == 1:
                # Absorb column v's coupling into remaining diagonals
                for j in range(N):
                    if j == v:
                        continue
                    contrib = Q_lil[v, j] + Q_lil[j, v]
                    if contrib != 0.0:
                        Q_lil[j, j] = (Q_lil[j, j] or 0.0) + contrib

            # Zero out full row v and column v
            Q_lil.rows[v] = []
            Q_lil.data[v] = []
            for j in range(N):
                if v in Q_lil.rows[j]:
                    idx = Q_lil.rows[j].index(v)
                    Q_lil.rows[j].pop(idx)
                    Q_lil.data[j].pop(idx)

            # Strong diagonal push: -P for pinned-to-1, +P for pinned-to-0
            Q_lil[v, v] = -P if c == 1 else P

        result = Q_lil.tocoo()
        result.sum_duplicates()
        return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def solve_qubo(
    Q: Union[scipy.sparse.coo_matrix, np.ndarray],
    solver_mode: int = SolverMode.CONTINUOUS,
    num_swarms: int = 64,
    annealing_steps: int = 7500,
    high_precision: bool = False,
    lib_path: Optional[str] = None,
) -> SolveResult:
    """
    Convenience function for solving a QUBO problem.
    
    Args:
        Q: QUBO matrix (sparse or dense)
        solver_mode: 0=SPATIAL, 1=COMBINATORIAL, 2=CONTINUOUS (default)
        num_swarms: Number of parallel swarms (1-256)
        annealing_steps: Number of optimization steps
        high_precision: Use Float64 instead of Float32
        lib_path: Optional path to LOUCETIUS_core library
    
    Returns:
        SolveResult with solution and metrics
    
    Example:
        >>> Q = scipy.sparse.random(100, 100, density=0.1, format='coo')
        >>> result = solve_qubo(Q)
        >>> print(result)
    """
    if isinstance(Q, np.ndarray):
        N = Q.shape[0]
    elif isinstance(Q, scipy.sparse.spmatrix):
        N = Q.shape[0]
    else:
        raise TypeError(f"Expected numpy/scipy matrix, got {type(Q)}")
    
    config = SwarmConfig(
        num_variables=N,
        num_swarms=num_swarms,
        solver_mode=solver_mode,
        annealing_steps=annealing_steps,
        high_precision=high_precision,
    )
    
    solver = LOUCETIUSSolver(lib_path)
    return solver.solve(Q, config)


# ============================================================================
# VERSION & DIAGNOSTICS
# ============================================================================

def get_version() -> str:
    """Get LOUCETIUS core library version."""
    try:
        core = LOUCETIUSCore()
        version_bytes = core.get_version()
        return version_bytes.decode('utf-8') if version_bytes else "Unknown"
    except Exception as e:
        return f"Error: {e}"


def print_diagnostics():
    """Print system diagnostics."""
    print("\n" + "=" * 70)
    print("LOUCETIUS v2.5 DIAGNOSTICS")
    print("=" * 70)
    print(f"Python Version:       {platform.python_version()}")
    print(f"Platform:             {platform.platform()}")
    print(f"NumPy Version:        {np.__version__}")
    print(f"SciPy Version:        {scipy.__version__}")
    
    try:
        version = get_version()
        print(f"LOUCETIUS Core:       {version}")
        print("Library Status:       OK LOADED")
    except Exception as e:
        print(f"Library Status:       FAILED ({e})")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Example usage
    print_diagnostics()
    
    # Create a small test problem
    N = 100
    Q = scipy.sparse.random(N, N, density=0.1, format='coo')
    
    try:
        result = solve_qubo(Q, solver_mode=SolverMode.CONTINUOUS)
        print(result)
    except OSError as e:
        print(f"Note: C++ library not available for testing. {e}")
        print("This is normal in development. The API is ready for integration.")
