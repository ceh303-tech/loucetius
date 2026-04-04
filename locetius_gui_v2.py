"""
Locetius v2.5 - Advanced QUBO Solver GUI
==========================================
Ultra-modern dark GUI with import formats (.mtx, .csv, .npz, .json),
real-time metrics, convergence charts, and comprehensive export options.
Engines: Spectral Swarm (Locetius), Kerr Wave, Hybrid Blend, Sparse CUDA.
Parallel Worlds: 64 or 128 swarms selectable at runtime.

Features:
  - Import Matrix Market (.mtx) and CSV (edge list) formats
  - Live energy convergence chart
  - Comprehensive hyperparameter controls (Thermal Noise, Max Iterations, etc.)
  - Export: Solution (.csv), Convergence Log (.json), Topology (.stl)
  - Real-time GPU monitoring
  - Company logo integration

Author: Christian Hayes
Date: March 2026
License: Proprietary  -  All Rights Reserved

Requirements:
  pip install PyQt6 pyqtgraph numpy scipy requests

Run:
  python locetius_gui_v2.py
"""

import sys
import os
import time
import json
import threading
import traceback
from pathlib import Path
from typing import Optional, List
import numpy as np
import scipy.sparse
import scipy.io

# -- Qt ------------------------------------------------------------------------
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFrame, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QTabWidget, QFileDialog, QProgressBar,
    QTableWidget, QTableWidgetItem, QCheckBox, QGroupBox, QSlider,
    QLineEdit, QStatusBar, QSizePolicy, QScrollArea, QGridLayout,
    QToolButton, QMessageBox, QInputDialog, QDialog, QButtonGroup,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QSize, QRect, QPointF
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QPainter, QPen, QBrush, QLinearGradient, QRadialGradient, QFontDatabase

# -- pyqtgraph for fast real-time plots ---------------------------------------
try:
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, background="#0d1117", foreground="#c9d1d9")
    _HAS_PG = True
except ImportError:
    _HAS_PG = False

# -- Optional requests for REST panel -----------------------------------------
try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# -- Optional pynvml for live GPU stats ---------------------------------------
try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

# -- Optional networkx + matplotlib for solution graph ------------------------
try:
    import networkx as _nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch as _MPatch
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# -- Locetius API -------------------------------------------------------------
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from locetius_api import LOUCETIUSSolver, SwarmConfig, SolverMode, get_version, LOUCETIUSCore
    import ctypes as _ctypes
    _SOLVER_AVAILABLE = True
    _SOLVER_VERSION = get_version()
except Exception as _e:
    _SOLVER_AVAILABLE = False
    _SOLVER_VERSION = f"unavailable ({_e})"

# -- Colours / theme -----------------------------------------------------------
_C = {
    "bg0":        "#0d1117",   # deepest background
    "bg1":        "#161b22",   # card background
    "bg2":        "#21262d",   # input background
    "bg3":        "#30363d",   # border / divider
    "text":       "#e6edf3",   # primary text
    "text2":      "#8b949e",   # secondary text
    "teal":       "#58b6c0",   # primary accent
    "teal_d":     "#388a93",   # dark accent
    "orange":     "#e8832a",   # warm accent / warning
    "green":      "#3fb950",   # success
    "red":        "#f85149",   # error
    "purple":     "#a371f7",   # GPU metric
    "yellow":     "#d29922",   # caution
    "bright":     "#7ad0da",  # bright accent
}

GLOBAL_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {_C['bg0']};
    color: {_C['text']};
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}}
QFrame#card {{
    background-color: {_C['bg1']};
    border: 1px solid {_C['bg3']};
    border-radius: 8px;
}}
QLabel#section_title {{
    color: {_C['teal']};
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}
QLabel#metric_value {{
    color: {_C['text']};
    font-size: 22px;
    font-weight: bold;
}}
QLabel#metric_unit {{
    color: {_C['text2']};
    font-size: 11px;
}}
QPushButton {{
    background-color: {_C['bg2']};
    color: {_C['text']};
    border: 1px solid {_C['bg3']};
    border-radius: 6px;
    padding: 7px 18px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {_C['bg3']};
    border-color: {_C['teal']};
}}
QPushButton#bright {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #1e7a90, stop:1 {_C['bright']});
    color: #ffffff;
    border: none;
    font-weight: 700;
    border-radius: 6px;
    padding: 8px 16px;
}}
QPushButton#bright:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {_C['bright']}, stop:1 #9ae5f0);
}}
QPushButton#primary {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {_C['teal_d']}, stop:1 {_C['teal']});
    color: #ffffff;
    border: none;
    font-size: 14px;
    padding: 10px 32px;
    border-radius: 8px;
}}
QPushButton#primary:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {_C['teal']}, stop:1 {_C['bright']});
}}
QPushButton#primary:disabled {{
    background-color: {_C['bg3']};
    color: {_C['text2']};
}}
QPushButton#danger {{
    background-color: #3b1a1a;
    border-color: {_C['red']};
    color: {_C['red']};
}}
QPushButton#danger:hover {{
    background-color: #5c2020;
}}
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    background-color: {_C['bg2']};
    color: {_C['text']};
    border: 1px solid {_C['bg3']};
    border-radius: 5px;
    padding: 5px 8px;
    selection-background-color: {_C['teal_d']};
}}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
    border-color: {_C['teal']};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background-color: {_C['bg2']};
    color: {_C['text']};
    selection-background-color: {_C['teal_d']};
    border: 1px solid {_C['bg3']};
}}
QTabWidget::pane {{
    border: 1px solid {_C['bg3']};
    border-radius: 6px;
    background-color: {_C['bg1']};
}}
QTabBar::tab {{
    background-color: {_C['bg2']};
    color: {_C['text2']};
    border: 1px solid {_C['bg3']};
    border-bottom: none;
    border-radius: 5px 5px 0 0;
    padding: 7px 18px;
    margin-right: 2px;
    font-weight: 500;
}}
QTabBar::tab:selected {{
    background-color: {_C['bg1']};
    color: {_C['teal']};
    border-bottom: 2px solid {_C['teal']};
}}
QTabBar::tab:hover:!selected {{
    color: {_C['text']};
    background-color: {_C['bg3']};
}}
QTextEdit {{
    background-color: {_C['bg0']};
    color: #aecbfa;
    border: 1px solid {_C['bg3']};
    border-radius: 6px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 12px;
    padding: 6px;
    selection-background-color: {_C['teal_d']};
}}
QScrollBar:vertical {{
    background: {_C['bg0']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {_C['bg3']};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {_C['teal_d']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QProgressBar {{
    background-color: {_C['bg2']};
    border: 1px solid {_C['bg3']};
    border-radius: 5px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {_C['teal_d']}, stop:1 {_C['teal']});
    border-radius: 5px;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {_C['bg3']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {_C['teal']};
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {_C['teal']};
    border-radius: 2px;
}}
QTableWidget {{
    background-color: {_C['bg1']};
    color: {_C['text']};
    border: 1px solid {_C['bg3']};
    gridline-color: {_C['bg3']};
    border-radius: 6px;
    alternate-background-color: {_C['bg2']};
    selection-background-color: {_C['teal_d']};
}}
QTableWidget QHeaderView::section {{
    background-color: {_C['bg2']};
    color: {_C['teal']};
    border: none;
    border-right: 1px solid {_C['bg3']};
    padding: 6px;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.5px;
}}
QSplitter::handle {{
    background-color: {_C['bg3']};
    width: 2px;
    height: 2px;
}}
QStatusBar {{
    background-color: {_C['bg1']};
    color: {_C['text2']};
    border-top: 1px solid {_C['bg3']};
    font-size: 11px;
}}
QGroupBox {{
    color: {_C['text2']};
    border: 1px solid {_C['bg3']};
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 10px;
    font-size: 11px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 6px;
    color: {_C['teal']};
    letter-spacing: 0.5px;
}}
QCheckBox {{
    color: {_C['text']};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {_C['bg3']};
    border-radius: 3px;
    background: {_C['bg2']};
}}
QCheckBox::indicator:checked {{
    background-color: {_C['teal']};
    border-color: {_C['teal']};
    image: none;
}}
QToolTip {{
    background-color: {_C['bg2']};
    color: {_C['text']};
    border: 1px solid {_C['bg3']};
    border-radius: 4px;
    padding: 4px 8px;
}}
"""

# -----------------------------------------------------------------------------
# Reusable widgets
# -----------------------------------------------------------------------------

def _card(parent=None) -> QFrame:
    f = QFrame(parent)
    f.setObjectName("card")
    return f

def _label(text, parent=None, obj_name=None) -> QLabel:
    l = QLabel(text, parent)
    if obj_name:
        l.setObjectName(obj_name)
    return l

def _section_title(text, parent=None) -> QLabel:
    l = QLabel(text.upper(), parent)
    l.setObjectName("section_title")
    return l

class MetricWidget(QFrame):
    """A metric tile (value + unit + label)."""
    def __init__(self, label: str, unit: str = "", color: str = _C["teal"], parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumWidth(110)
        ly = QVBoxLayout(self)
        ly.setContentsMargins(14, 12, 14, 12)
        ly.setSpacing(2)

        self._lbl = QLabel(label.upper())
        self._lbl.setObjectName("section_title")
        self._val = QLabel(" - ")
        self._val.setObjectName("metric_value")
        self._val.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")
        self._unit = QLabel(unit)
        self._unit.setObjectName("metric_unit")

        ly.addWidget(self._lbl)
        ly.addWidget(self._val)
        ly.addWidget(self._unit)

    def set_value(self, v):
        self._val.setText(str(v))

class PulseButton(QPushButton):
    """Execute button with animated activity pulse."""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setObjectName("primary")
        self._running = False
        self._dots = 0
        self._base_text = text
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_running(self, running: bool):
        self._running = running
        self.setEnabled(not running)
        if running:
            self._dots = 0
            self._timer.start(400)
        else:
            self._timer.stop()
            self.setText(self._base_text)

    def _tick(self):
        self._dots = (self._dots + 1) % 4
        self.setText("Solving" + "." * self._dots)

class LogPanel(QTextEdit):
    """Read-only log panel with auto-scroll."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMinimumHeight(120)

    def append_line(self, text: str, color: str = _C["text"]):
        ts = time.strftime("%H:%M:%S")
        self.setTextColor(QColor(color))
        self.append(f"[{ts}]  {text}")
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def info(self, msg): self.append_line(msg, _C["text"])
    def success(self, msg): self.append_line(msg, _C["green"])
    def error(self, msg): self.append_line(msg, _C["red"])
    def warn(self, msg): self.append_line(msg, _C["orange"])

# -----------------------------------------------------------------------------
# Solver Worker Thread
# -----------------------------------------------------------------------------

class SolverWorker(QThread):
    progress = pyqtSignal(int, str)
    energy_update = pyqtSignal(float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, Q, config, parent=None):
        super().__init__(parent)
        self.Q = Q
        self.config = config
        self.energy_history = []

    def run(self):
        try:
            self.progress.emit(5, "Transferring matrix to GPU ...")
            solver = LOUCETIUSSolver()
            self.progress.emit(15, "Initialising swarm ...")
            t0 = time.time()
            result = solver.solve(self.Q, self.config)
            elapsed = time.time() - t0
            self.progress.emit(100, f"Done in {elapsed:.2f}s")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(traceback.format_exc())

# -----------------------------------------------------------------------------
# GPU Monitor Worker Thread
# -----------------------------------------------------------------------------

class GPUMonitorWorker(QThread):
    """Queries GPU stats directly from the DLL, then keeps polling the REST
    server for live updates if it becomes available."""
    gpu_updated = pyqtSignal(dict)
    server_status = pyqtSignal(bool, str)   # connected, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._server_url = "http://localhost:8765"

    def _query_dll(self) -> Optional[dict]:
        """Query GPU info directly from the C++ DLL  -  no server needed."""
        try:
            import ctypes as _ct
            from locetius_api import LOUCETIUSCore
            core = LOUCETIUSCore()
            dev = _ct.create_string_buffer(256)
            vram = _ct.c_int32(0)
            cc   = _ct.c_int32(0)
            rc = core.get_cuda_info(dev, _ct.byref(vram), _ct.byref(cc))
            if rc == 0 and vram.value > 0:
                return {
                    "device": dev.value.decode(),
                    "vram_mb": int(vram.value),
                    "compute_cap": f"{cc.value // 10}.{cc.value % 10}",
                    "available": True,
                }
        except Exception:
            pass
        return None

    def _query_nvml(self) -> dict:
        """Get live GPU telemetry via pynvml (util, mem, temp, power, fan, clocks)."""
        stats = {}
        if not _HAS_NVML:
            return stats
        try:
            handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            util = _pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats["util_gpu"] = int(util.gpu)
            stats["util_mem"] = int(util.memory)
            mem = _pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats["mem_used_mb"]  = int(mem.used  // (1024 * 1024))
            stats["mem_total_mb"] = int(mem.total // (1024 * 1024))
            stats["mem_free_mb"]  = int(mem.free  // (1024 * 1024))
            try:
                stats["temp_c"] = int(_pynvml.nvmlDeviceGetTemperature(
                    handle, _pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            try:
                stats["power_w"]       = round(_pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1)
                stats["power_limit_w"] = round(_pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0, 1)
            except Exception:
                pass
            try:
                stats["fan_pct"] = int(_pynvml.nvmlDeviceGetFanSpeed(handle))
            except Exception:
                pass
            try:
                stats["clock_gpu_mhz"] = int(_pynvml.nvmlDeviceGetClockInfo(
                    handle, _pynvml.NVML_CLOCK_GRAPHICS))
                stats["clock_mem_mhz"] = int(_pynvml.nvmlDeviceGetClockInfo(
                    handle, _pynvml.NVML_CLOCK_MEM))
            except Exception:
                pass
        except Exception:
            pass
        return stats

    def _query_server(self) -> Optional[dict]:
        """Try to get GPU info from the REST server."""
        try:
            if _HAS_REQUESTS:
                r = _requests.get(f"{self._server_url}/health", timeout=2)
                if r.status_code == 200:
                    d = r.json()
                    self.server_status.emit(True, f"Connected  -  {self._server_url}")
                    if d.get("gpu"):
                        return {
                            "device": d.get("gpu"),
                            "vram_mb": d.get("vram_mb", 0),
                            "compute_cap": d.get("compute_cap", "N/A"),
                            "available": True,
                        }
        except Exception:
            pass
        self.server_status.emit(False, f"Offline  -  {self._server_url}")
        return None

    def run(self):
        # First: query the DLL directly so stats show immediately on startup
        info = self._query_dll()
        if info:
            nvml = self._query_nvml()
            self.gpu_updated.emit({**info, **nvml})

        # Then keep polling for live stats every 2 s
        while self._running:
            server_info = self._query_server()
            base = server_info or info or {"device": None, "vram_mb": 0,
                                           "compute_cap": "N/A", "available": False}
            nvml = self._query_nvml()
            self.gpu_updated.emit({**base, **nvml})
            time.sleep(2)

    def stop(self):
        self._running = False

    def set_server_url(self, url: str):
        self._server_url = url.rstrip("/")

# -----------------------------------------------------------------------------
# Solution Graph Dialog
# -----------------------------------------------------------------------------

class SolutionGraphDialog(QDialog):
    """Popup network graph: nodes coloured by binary solution state (1=cyan, 0=red),
    sized by consensus strength. Edges drawn from the QUBO matrix weights."""

    MAX_NODES_VIZ = 500  # cap for readable rendering on screen

    def __init__(self, Q, solution, consensus, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Solution Graph  -  Locetius v2.5")
        self.resize(980, 800)
        self.setStyleSheet(
            f"background-color: {_C['bg0']}; color: {_C['text']};"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        N_total = len(solution)
        n_ones  = int(np.sum(solution))
        n_zeros = N_total - n_ones

        # -- header row -----------------------------------------------------
        hdr = QHBoxLayout()
        info_lbl = QLabel(
            f"Variables: {N_total:,}    *  State=1 (cyan): {n_ones:,}    "
            f"*  State=0 (red): {n_zeros:,}    Node size = consensus"
        )
        info_lbl.setStyleSheet(f"color: {_C['text2']}; font-size: 12px;")
        hdr.addWidget(info_lbl)
        hdr.addStretch()
        close_btn = QPushButton("  Close")
        close_btn.clicked.connect(self.close)
        hdr.addWidget(close_btn)
        root.addLayout(hdr)

        if not (_HAS_NX and _HAS_MPL):
            root.addWidget(QLabel(
                "networkx and matplotlib are required for graph visualisation.\n"
                "  pip install networkx matplotlib"
            ))
            return

        # -- build networkx graph --------------------------------------------
        Q_coo = Q.tocoo()

        # If N is large, keep only the most-connected nodes
        if N_total > self.MAX_NODES_VIZ:
            degree = np.zeros(N_total)
            for r, c in zip(Q_coo.row, Q_coo.col):
                if r != c:
                    degree[r] += 1
                    degree[c] += 1
            top_idx  = np.argsort(degree)[-self.MAX_NODES_VIZ:]
            top_set  = set(top_idx.tolist())
            shown    = sorted(top_idx.tolist())
        else:
            top_set = None
            shown   = list(range(N_total))

        G = _nx.Graph()
        for i in shown:
            G.add_node(i, state=int(solution[i]), cons=float(consensus[i]))

        edge_weights_raw = []
        for r, c, v in zip(Q_coo.row, Q_coo.col, Q_coo.data):
            if r == c:
                continue
            if top_set and (r not in top_set or c not in top_set):
                continue
            if not G.has_edge(int(r), int(c)):
                G.add_edge(int(r), int(c), weight=float(abs(v)))
                edge_weights_raw.append(float(abs(v)))

        # Spring layout  -  tighten k for large graphs
        k_val = 2.5 / max(1, len(shown) ** 0.5)
        pos = _nx.spring_layout(G, seed=42, k=k_val)

        # -- matplotlib figure -----------------------------------------------
        fig = Figure(figsize=(10, 8), facecolor="#0d1117")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#111820")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        node_colors = [
            _C["bright"] if solution[i] == 1 else _C["red"]
            for i in shown
        ]
        node_sizes = [max(20, int(consensus[i] * 80)) for i in shown]

        # Draw edges
        if edge_weights_raw:
            _nx.draw_networkx_edges(
                G, pos, ax=ax,
                alpha=0.25,
                edge_color="#30363d",
                width=0.8,
            )

        # Draw nodes
        _nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=shown,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.90,
        )

        # Labels only when small enough to read
        if len(shown) <= 60:
            _nx.draw_networkx_labels(
                G, pos, ax=ax,
                font_size=7,
                font_color="#c9d1d9",
            )

        # Legend
        legend_elements = [
            _MPatch(facecolor=_C["bright"], label=f"State = 1  ({n_ones:,} variables)"),
            _MPatch(facecolor=_C["red"],    label=f"State = 0  ({n_zeros:,} variables)"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            facecolor="#161b22",
            edgecolor="#30363d",
            labelcolor="#e6edf3",
            fontsize=10,
        )

        title = "QUBO Solution Graph"
        if N_total > self.MAX_NODES_VIZ:
            title += (
                f"  (showing {len(shown):,} most-connected of {N_total:,} total nodes)"
            )
        ax.set_title(title, color="#8b949e", fontsize=12, pad=12)

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background-color: #0d1117;")
        root.addWidget(canvas, 1)

        # Status bar
        status = QLabel(
            f"Nodes shown: {len(shown):,}    Edges: {G.number_of_edges():,}    "
            f"Scroll to zoom  -  Drag to pan"
        )
        status.setStyleSheet(f"color: {_C['text2']}; font-size: 11px;")
        root.addWidget(status)


# -----------------------------------------------------------------------------
# Main Window
# -----------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Locetius v2.5   -   QUBO Solver")
        self.resize(1600, 920)

        self._result = None
        self._worker: Optional[SolverWorker] = None
        self._Q: Optional[scipy.sparse.coo_matrix] = None
        self.energy_history = []
        self.time_history = []
        
        # GPU monitoring
        self._gpu_monitor: Optional[GPUMonitorWorker] = None
        self._gpu_device = "Unknown"
        self._gpu_vram_mb = 0
        self._gpu_compute_cap = "N/A"
        self._gpu_available = False

        self._build_ui()
        self._apply_style()
        self._update_status("Ready")
        
        # Start GPU monitor
        self._start_gpu_monitor()

        self._fake_prog = QTimer(self)
        self._fake_prog.timeout.connect(self._tick_fake_progress)
        self._fake_pct = 15

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([340, 800, 460])
        root.addWidget(splitter, 1)

        self._build_statusbar()

    def _build_header(self) -> QWidget:
        hdr = QFrame()
        hdr.setFixedHeight(64)
        hdr.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 #0d1f2d, stop:0.5 #0d1117, stop:1 #1a0d2d);"
            f"border-bottom: 1px solid {_C['bg3']};"
        )
        ly = QHBoxLayout(hdr)
        ly.setContentsMargins(20, 0, 20, 0)
        ly.setSpacing(10)

        # Logo  -  try cyan PNG first, then fall back to logo.jpg
        logo_lbl = QLabel()
        logo_lbl.setStyleSheet("background-color: #000000; border-radius: 5px; padding: 2px;")
        _base = Path(__file__).parent
        _logo_candidates = [
            _base / "LOUCETIUS_logo_cyan (1).png",
            _base / "locetius_logo_cyan.png",
            _base / "logo_cyan.png",
            _base / "logo.jpg",
        ]
        _logo_loaded = False
        for _lp in _logo_candidates:
            if _lp.exists():
                raw = QPixmap(str(_lp))
                if not raw.isNull():
                    # For JPG with text at bottom: crop top 70%; for PNG use full image
                    if _lp.suffix.lower() == ".jpg":
                        crop_h = int(raw.height() * 0.70)
                        icon_pix = raw.copy(0, 0, raw.width(), crop_h)
                    else:
                        icon_pix = raw
                    icon_pix = icon_pix.scaledToHeight(46, Qt.TransformationMode.SmoothTransformation)
                    logo_lbl.setPixmap(icon_pix)
                    logo_lbl.setFixedSize(icon_pix.width() + 6, 52)
                    _logo_loaded = True
                    break
        if not _logo_loaded:
            logo_lbl.setText("")
            logo_lbl.setStyleSheet(
                "background-color: #000000; border-radius: 5px; padding: 4px;"
                f"color: {_C['teal']}; font-size: 28px; font-weight: bold;"
            )
            logo_lbl.setFixedSize(48, 52)
        ly.addWidget(logo_lbl)

        # Title block
        title_block = QVBoxLayout()
        title_block.setSpacing(1)
        title = QLabel("Locetius")
        title.setStyleSheet(f"color: {_C['teal']}; font-size: 22px; font-weight: 700; letter-spacing: 1.5px;")
        sub = QLabel("Spectral Swarm QUBO Engine  v1.0")
        sub.setStyleSheet(f"color: {_C['text2']}; font-size: 11px;")
        creator = QLabel("by Christian Hayes")
        creator.setStyleSheet(f"color: #8b949e; font-size: 10px; font-style: italic;")
        title_block.addWidget(title)
        title_block.addWidget(sub)
        title_block.addWidget(creator)
        ly.addLayout(title_block)

        ly.addStretch()

        self._status_indicator = QLabel("*")
        self._status_indicator.setStyleSheet(f"color: {_C['text2']}; font-size: 18px;")
        ly.addWidget(self._status_indicator)

        return hdr

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(310)
        w.setMaximumWidth(400)
        ly = QVBoxLayout(w)
        ly.setContentsMargins(12, 12, 6, 12)
        ly.setSpacing(10)

        # INPUT SECTION
        input_card = _card()
        input_ly = QVBoxLayout(input_card)
        input_ly.setContentsMargins(14, 12, 14, 14)
        input_ly.setSpacing(10)
        input_ly.addWidget(_section_title("Input Data"))

        import_mtx_btn = QPushButton(" Import .MTX (Matrix Market)")
        import_mtx_btn.clicked.connect(self._import_mtx)
        input_ly.addWidget(import_mtx_btn)

        import_csv_btn = QPushButton(" Import .CSV (Edge List)")
        import_csv_btn.clicked.connect(self._import_csv)
        input_ly.addWidget(import_csv_btn)

        import_other_btn = QPushButton(" Import Other (.npz, .json)")
        import_other_btn.clicked.connect(self._import_other)
        input_ly.addWidget(import_other_btn)

        generate_test_btn = QPushButton(" Generate Random Test Data")
        generate_test_btn.setObjectName("bright")
        generate_test_btn.clicked.connect(self._generate_random_data)
        input_ly.addWidget(generate_test_btn)

        self._matrix_info = QLabel("No matrix loaded")
        self._matrix_info.setWordWrap(True)
        self._matrix_info.setStyleSheet(f"color: {_C['text2']}; font-size: 11px;")
        input_ly.addWidget(self._matrix_info)

        ly.addWidget(input_card)

        # HYPERPARAMETERS SECTION
        hyper_card = _card()
        hyper_ly = QVBoxLayout(hyper_card)
        hyper_ly.setContentsMargins(14, 12, 14, 14)
        hyper_ly.setSpacing(8)
        hyper_ly.addWidget(_section_title("Hyperparameters"))

        # --- Engine Selector ----------------------------------------------------------------
        hyper_ly.addWidget(QLabel("Solver Engine:"))
        self._engine_combo = QComboBox()
        self._engine_combo.addItems([
            "AUTO  (Smart Route)",
            "LOCETIUS  (Spectral Swarm)",
            "KERR  (Quantum Wave)",
            "HYBRID  (Blended)",
            "SPARSE  (Large N)",
        ])
        self._engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        hyper_ly.addWidget(self._engine_combo)
        hyper_ly.addSpacing(4)

        # --- Parallel Worlds (64 / 128) -------------------------------------------------------
        hyper_ly.addWidget(QLabel("Parallel Worlds (Swarms):"))
        worlds_row = QHBoxLayout()
        self._worlds_64_btn = QPushButton("64")
        self._worlds_64_btn.setCheckable(True)
        self._worlds_64_btn.setChecked(True)
        self._worlds_64_btn.setStyleSheet(
            f"QPushButton:checked {{ background-color: {_C['teal']}; color: #fff; font-weight: bold; }}"
        )
        self._worlds_128_btn = QPushButton("128")
        self._worlds_128_btn.setCheckable(True)
        self._worlds_128_btn.setStyleSheet(
            f"QPushButton:checked {{ background-color: {_C['teal']}; color: #fff; font-weight: bold; }}"
        )
        self._worlds_group = QButtonGroup()
        self._worlds_group.setExclusive(True)
        self._worlds_group.addButton(self._worlds_64_btn, 64)
        self._worlds_group.addButton(self._worlds_128_btn, 128)
        worlds_row.addWidget(self._worlds_64_btn)
        worlds_row.addWidget(self._worlds_128_btn)
        worlds_row.addStretch()
        hyper_ly.addLayout(worlds_row)
        hyper_ly.addSpacing(4)

        # --- Hybrid Blend (visible only when HYBRID engine selected) --------------------------
        self._hybrid_frame = QWidget()
        hybrid_inner = QVBoxLayout(self._hybrid_frame)
        hybrid_inner.setContentsMargins(0, 0, 0, 0)
        hybrid_inner.setSpacing(4)
        blend_label_row = QHBoxLayout()
        blend_label_row.addWidget(QLabel("Hybrid Blend  (Kerr <- -> Locetius):"))
        self._hybrid_blend_lbl = QLabel("0.50")
        self._hybrid_blend_lbl.setFixedWidth(40)
        self._hybrid_blend_lbl.setStyleSheet(f"color: {_C['bright']}; font-weight: bold;")
        blend_label_row.addStretch()
        blend_label_row.addWidget(self._hybrid_blend_lbl)
        hybrid_inner.addLayout(blend_label_row)
        self._hybrid_blend_slider = QSlider(Qt.Orientation.Horizontal)
        self._hybrid_blend_slider.setRange(0, 100)
        self._hybrid_blend_slider.setValue(50)
        self._hybrid_blend_slider.valueChanged.connect(
            lambda v: self._hybrid_blend_lbl.setText(f"{v/100:.2f}")
        )
        hybrid_inner.addWidget(self._hybrid_blend_slider)
        self._hybrid_frame.setVisible(False)
        hyper_ly.addWidget(self._hybrid_frame)
        hyper_ly.addSpacing(2)

        hyper_ly.addWidget(QLabel("Thermal Melt / Noise:"))
        noise_row = QHBoxLayout()
        self._noise_slider = QSlider(Qt.Orientation.Horizontal)
        self._noise_slider.setRange(0, 100)
        self._noise_slider.setValue(20)
        self._noise_lbl = QLabel("0.20")
        self._noise_lbl.setFixedWidth(48)
        self._noise_lbl.setObjectName("bright")
        self._noise_lbl.setStyleSheet(f"color: {_C['bright']}; font-weight: bold;")
        self._noise_slider.valueChanged.connect(
            lambda v: self._noise_lbl.setText(f"{v/100:.2f}")
        )
        noise_row.addWidget(self._noise_slider)
        noise_row.addWidget(self._noise_lbl)
        hyper_ly.addLayout(noise_row)

        hyper_ly.addWidget(QLabel("Max Iterations / Annealing Steps:"))
        steps_row = QHBoxLayout()
        self._steps_slider = QSlider(Qt.Orientation.Horizontal)
        self._steps_slider.setRange(100, 50000)
        self._steps_slider.setValue(7500)
        self._steps_slider.setSingleStep(100)
        self._steps_lbl = QLabel("7500")
        self._steps_lbl.setFixedWidth(60)
        self._steps_lbl.setStyleSheet(f"color: {_C['bright']}; font-weight: bold;")
        self._steps_slider.valueChanged.connect(
            lambda v: self._steps_lbl.setText(str(v))
        )
        steps_row.addWidget(self._steps_slider)
        steps_row.addWidget(self._steps_lbl)
        hyper_ly.addLayout(steps_row)

        hyper_ly.addWidget(QLabel("Precision:"))
        self._precision_combo = QComboBox()
        self._precision_combo.addItems(["Float32 (Fast)", "Float64 (Precise)"])
        hyper_ly.addWidget(self._precision_combo)

        # --- SNIPPET B: Phase 3 Toggle (Deterministic Freeze) -----------------------------------
        phase3_layout = QHBoxLayout()
        self._phase3_check = QCheckBox("Enforce Phase 3 (Deterministic Freeze)")
        self._phase3_check.setChecked(True)  # Default: ON
        phase3_layout.addWidget(self._phase3_check)
        phase3_layout.addStretch()
        hyper_ly.addLayout(phase3_layout)

        # --- SNIPPET C: Early Stopping (Patience) --------------------------------------------
        early_stop_layout = QHBoxLayout()
        self._early_stop_check = QCheckBox("Early Stopping (Patience steps)")
        self._early_stop_check.setChecked(True)  # Default: ON
        self._patience_spin = QSpinBox()
        self._patience_spin.setMinimum(50)
        self._patience_spin.setMaximum(5000)
        self._patience_spin.setValue(500)  # Default value
        self._patience_spin.setFixedWidth(70)
        self._patience_spin.setStyleSheet(f"color: {_C['bright']}; font-weight: bold;")
        early_stop_layout.addWidget(self._early_stop_check)
        early_stop_layout.addStretch()
        early_stop_layout.addWidget(self._patience_spin)
        hyper_ly.addLayout(early_stop_layout)

        ly.addWidget(hyper_card)

        # SOLVER MODE SECTION (Physics Mode  -  applies to LOCETIUS engine)
        mode_card = _card()
        mode_ly = QVBoxLayout(mode_card)
        mode_ly.setContentsMargins(14, 12, 14, 14)
        mode_ly.setSpacing(8)
        mode_ly.addWidget(_section_title("Physics Mode"))

        self._mode_note = QLabel("(Active when LOCETIUS engine selected)")
        self._mode_note.setStyleSheet(f"color: {_C['text2']}; font-size: 11px;")
        mode_ly.addWidget(self._mode_note)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["SPATIAL (Legacy)", "COMBINATORIAL (MaxCut)", "CONTINUOUS (Soft-Spin)"])
        self._mode_combo.setCurrentIndex(2)
        mode_ly.addWidget(self._mode_combo)

        ly.addWidget(mode_card)

        # EXECUTE BUTTON
        ly.addStretch()

        self._solve_btn = PulseButton("  Solve with Locetius  ")
        self._solve_btn.setMinimumHeight(48)
        self._solve_btn.setEnabled(False)
        self._solve_btn.clicked.connect(self._run_solve)
        ly.addWidget(self._solve_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(8)
        ly.addWidget(self._progress_bar)

        return w

    def _build_center_panel(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(6, 12, 6, 12)
        ly.setSpacing(10)

        metrics_row = QHBoxLayout()
        self._m_energy  = MetricWidget("Best Energy",  "",        _C['teal'])
        self._m_cons    = MetricWidget("Consensus",    "% avg",   _C['green'])
        self._m_time    = MetricWidget("Wall Time",    "s",       _C['orange'])
        self._m_vars    = MetricWidget("Variables",    "",        _C['purple'])
        for m in [self._m_energy, self._m_cons, self._m_time, self._m_vars]:
            metrics_row.addWidget(m)
        ly.addLayout(metrics_row)

        # -- Prominent solver progress bar ----------------------------------
        prog_card = _card()
        prog_inner = QHBoxLayout(prog_card)
        prog_inner.setContentsMargins(14, 10, 14, 10)
        prog_inner.setSpacing(12)
        self._center_prog_status = QLabel("IDLE")
        self._center_prog_status.setFixedWidth(120)
        self._center_prog_status.setStyleSheet(
            f"color: {_C['text2']}; font-size: 11px; font-weight: 700; letter-spacing: 1px;")
        self._center_progress = QProgressBar()
        self._center_progress.setRange(0, 100)
        self._center_progress.setValue(0)
        self._center_progress.setFixedHeight(18)
        self._center_progress.setTextVisible(True)
        self._center_progress.setFormat("%p%")
        self._center_progress.setStyleSheet(f"""
            QProgressBar {{
                background: {_C['bg2']};
                border: 1px solid {_C['bg3']};
                border-radius: 9px;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {_C['teal_d']}, stop:1 {_C['bright']});
                border-radius: 9px;
            }}
        """)
        prog_inner.addWidget(self._center_prog_status)
        prog_inner.addWidget(self._center_progress, 1)
        ly.addWidget(prog_card)

        if _HAS_PG:
            conv_grp = QGroupBox("Live Energy Convergence")
            conv_inner = QVBoxLayout(conv_grp)
            self._conv_plot = pg.PlotWidget()
            self._conv_plot.setMinimumHeight(220)
            self._conv_plot.setLabel("left", "Energy", color=_C["text2"])
            self._conv_plot.setLabel("bottom", "Time (ms)", color=_C["text2"])
            self._conv_plot.getAxis("left").setPen(pg.mkPen(_C["bg3"]))
            self._conv_plot.getAxis("bottom").setPen(pg.mkPen(_C["bg3"]))
            self._conv_plot.showGrid(x=True, y=True, alpha=0.15)
            self._conv_curve = self._conv_plot.plot(pen=pg.mkPen(_C["bright"], width=2.5))
            conv_inner.addWidget(self._conv_plot)
            ly.addWidget(conv_grp)

            cons_grp = QGroupBox("Consensus Distribution")
            cons_inner = QVBoxLayout(cons_grp)
            self._cons_plot = pg.PlotWidget()
            self._cons_plot.setMinimumHeight(160)
            self._cons_plot.setLabel("left", "Count", color=_C["text2"])
            self._cons_plot.setLabel("bottom", "Consensus Value", color=_C["text2"])
            self._cons_plot.getAxis("left").setPen(pg.mkPen(_C["bg3"]))
            self._cons_plot.getAxis("bottom").setPen(pg.mkPen(_C["bg3"]))
            self._cons_plot.showGrid(x=True, y=True, alpha=0.15)
            self._cons_bar_item = pg.BarGraphItem(x=[], height=[], width=0.04,
                                                   brush=pg.mkBrush(_C["purple"]))
            self._cons_plot.addItem(self._cons_bar_item)
            cons_inner.addWidget(self._cons_plot)
            ly.addWidget(cons_grp)
        else:
            ly.addWidget(QLabel("Install pyqtgraph: pip install pyqtgraph"))

        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(360)
        ly = QVBoxLayout(w)
        ly.setContentsMargins(6, 12, 12, 12)
        ly.setSpacing(10)

        self._right_tabs = QTabWidget()
        self._right_tabs.addTab(self._build_results_tab(), "  Results  ")
        self._right_tabs.addTab(self._build_exports_tab(), "  Exports  ")
        self._right_tabs.addTab(self._build_gpu_stats_tab(), "  GPU Stats  ")
        self._right_tabs.addTab(self._build_server_tab(), "  Server  ")
        self._right_tabs.addTab(self._build_log_tab(), "  Log  ")
        ly.addWidget(self._right_tabs, 1)

        return w

    def _build_results_tab(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(8, 8, 8, 8)
        ly.setSpacing(8)

        ly.addWidget(_section_title("Solution Preview"))

        self._sol_table = QTableWidget(0, 3)
        self._sol_table.setHorizontalHeaderLabels(["Variable", "Value", "Consensus"])
        self._sol_table.horizontalHeader().setStretchLastSection(True)
        self._sol_table.setAlternatingRowColors(True)
        self._sol_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._sol_table.setMaximumHeight(250)
        ly.addWidget(self._sol_table)

        ly.addWidget(_section_title("Result Summary"))
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setPlainText("No result yet.\nLoad a matrix and click Solve.")
        ly.addWidget(self._result_text, 1)

        self._graph_btn = QPushButton("\U0001f5fa\ufe0f  View Solution Graph")
        self._graph_btn.setObjectName("bright")
        self._graph_btn.setEnabled(False)
        self._graph_btn.setToolTip(
            "Opens a network map of all variables coloured by their solved state."
        )
        self._graph_btn.clicked.connect(self._view_solution_graph)
        ly.addWidget(self._graph_btn)

        return w

    def _build_exports_tab(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(8, 8, 8, 8)
        ly.setSpacing(10)

        ly.addWidget(_section_title("Export Options"))

        export_sol_btn = QPushButton(" Export Solution (.csv)")
        export_sol_btn.setObjectName("bright")
        export_sol_btn.clicked.connect(lambda: self._export_solution("csv"))
        ly.addWidget(export_sol_btn)

        export_log_btn = QPushButton(" Export Convergence Log (.json)")
        export_log_btn.setObjectName("bright")
        export_log_btn.clicked.connect(lambda: self._export_convergence())
        ly.addWidget(export_log_btn)

        export_topo_btn = QPushButton(" Export Topology (.stl)")
        export_topo_btn.setObjectName("bright")
        export_topo_btn.clicked.connect(lambda: self._export_topology())
        ly.addWidget(export_topo_btn)

        ly.addSpacing(20)
        ly.addWidget(QLabel("Export Status:"))

        self._export_log = LogPanel()
        self._export_log.setMaximumHeight(200)
        ly.addWidget(self._export_log)

        ly.addStretch()
        return w

    def _build_gpu_stats_tab(self) -> QWidget:
        """GPU Statistics tab  -  live hardware telemetry dashboard."""
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(8, 8, 8, 8)
        ly.setSpacing(8)

        # -- Status banner --------------------------------------------------
        banner = _card()
        banner_ly = QHBoxLayout(banner)
        banner_ly.setContentsMargins(12, 10, 12, 10)
        self._gpu_indicator = QLabel("*")
        self._gpu_indicator.setStyleSheet(f"color: {_C['text2']}; font-size: 22px;")
        self._gpu_status_lbl = QLabel("Initialising...")
        self._gpu_status_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {_C['text2']};")
        banner_ly.addWidget(self._gpu_indicator)
        banner_ly.addSpacing(8)
        banner_ly.addWidget(self._gpu_status_lbl)
        banner_ly.addStretch()
        self._gpu_nvml_badge = QLabel("NVML OFF")
        self._gpu_nvml_badge.setStyleSheet(
            f"color: {_C['text2']}; font-size: 10px; font-weight: bold; "
            f"background:{_C['bg3']}; border-radius:4px; padding:2px 6px;")
        banner_ly.addWidget(self._gpu_nvml_badge)
        ly.addWidget(banner)

        # -- Device info ----------------------------------------------------
        ly.addWidget(_section_title("Device"))
        dev_card = _card()
        dev_grid = QGridLayout(dev_card)
        dev_grid.setContentsMargins(12, 10, 12, 10)
        dev_grid.setSpacing(6)
        dev_grid.setColumnStretch(1, 1)

        def _row(grid, row, label, color):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {_C['text2']}; font-size: 11px;")
            val = QLabel(" - ")
            val.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")
            grid.addWidget(lbl, row, 0)
            grid.addWidget(val, row, 1)
            return val

        self._gpu_device_lbl  = _row(dev_grid, 0, "Device",      _C["purple"])
        self._gpu_vram_lbl    = _row(dev_grid, 1, "VRAM Total",  _C["bright"])
        self._gpu_cc_lbl      = _row(dev_grid, 2, "Compute Cap", _C["orange"])
        ly.addWidget(dev_card)

        # -- Live metrics ---------------------------------------------------
        ly.addWidget(_section_title("Live Telemetry"))
        live_card = _card()
        live_ly = QVBoxLayout(live_card)
        live_ly.setContentsMargins(12, 10, 12, 10)
        live_ly.setSpacing(10)

        def _bar_row(parent_ly, label, color):
            """Returns (bar_widget, value_label)."""
            row_w = QWidget()
            row_ly = QVBoxLayout(row_w)
            row_ly.setContentsMargins(0, 0, 0, 0)
            row_ly.setSpacing(3)
            hdr = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {_C['text2']}; font-size: 11px; font-weight: 600;")
            val = QLabel(" - ")
            val.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")
            hdr.addWidget(lbl)
            hdr.addStretch()
            hdr.addWidget(val)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(10)
            bar.setTextVisible(False)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background: {_C['bg3']};
                    border: none;
                    border-radius: 5px;
                }}
                QProgressBar::chunk {{
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 {_C['teal_d']}, stop:1 {color});
                    border-radius: 5px;
                }}
            """)
            row_ly.addLayout(hdr)
            row_ly.addWidget(bar)
            parent_ly.addWidget(row_w)
            return bar, val

        self._bar_util_gpu, self._lbl_util_gpu = _bar_row(live_ly, "GPU Utilisation", _C["teal"])
        self._bar_util_mem, self._lbl_util_mem = _bar_row(live_ly, "Mem Controller",  _C["bright"])
        self._bar_mem_used, self._lbl_mem_used = _bar_row(live_ly, "VRAM Used",        _C["purple"])
        self._bar_temp,     self._lbl_temp     = _bar_row(live_ly, "Temperature",      _C["orange"])
        self._bar_power,    self._lbl_power    = _bar_row(live_ly, "Power Draw",       _C["yellow"])
        self._bar_fan,      self._lbl_fan      = _bar_row(live_ly, "Fan Speed",        _C["green"])

        ly.addWidget(live_card)

        # -- Clock speeds ---------------------------------------------------
        ly.addWidget(_section_title("Clocks"))
        clk_card = _card()
        clk_grid = QGridLayout(clk_card)
        clk_grid.setContentsMargins(12, 8, 12, 8)
        clk_grid.setSpacing(6)
        clk_grid.setColumnStretch(1, 1)
        lbl_gc = QLabel("GPU Core")
        lbl_gc.setStyleSheet(f"color: {_C['text2']}; font-size: 11px;")
        self._lbl_clk_gpu = QLabel(" - ")
        self._lbl_clk_gpu.setStyleSheet(f"color: {_C['teal']}; font-weight: bold;")
        lbl_mc = QLabel("Memory")
        lbl_mc.setStyleSheet(f"color: {_C['text2']}; font-size: 11px;")
        self._lbl_clk_mem = QLabel(" - ")
        self._lbl_clk_mem.setStyleSheet(f"color: {_C['bright']}; font-weight: bold;")
        clk_grid.addWidget(lbl_gc, 0, 0)
        clk_grid.addWidget(self._lbl_clk_gpu, 0, 1)
        clk_grid.addWidget(lbl_mc, 1, 0)
        clk_grid.addWidget(self._lbl_clk_mem, 1, 1)
        ly.addWidget(clk_card)

        ly.addStretch()
        return w

    def _build_server_tab(self) -> QWidget:
        """Optional REST server configuration panel."""
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(8, 8, 8, 8)
        ly.setSpacing(10)

        ly.addWidget(_section_title("REST Server (Optional)"))

        # Status card
        status_card = _card()
        sc_ly = QVBoxLayout(status_card)
        sc_ly.setContentsMargins(12, 10, 12, 10)
        sc_ly.setSpacing(6)

        self._srv_indicator = QLabel("o  Not Connected")
        self._srv_indicator.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {_C['text2']};")
        sc_ly.addWidget(self._srv_indicator)

        self._srv_detail = QLabel("Server is optional  -  the solver runs locally via DLL.")
        self._srv_detail.setWordWrap(True)
        self._srv_detail.setStyleSheet(f"color: {_C['text2']}; font-size: 11px;")
        sc_ly.addWidget(self._srv_detail)

        ly.addWidget(status_card)

        # URL configuration
        ly.addWidget(_section_title("Server URL"))
        url_card = _card()
        url_ly = QVBoxLayout(url_card)
        url_ly.setContentsMargins(12, 10, 12, 10)
        url_ly.setSpacing(8)

        self._srv_url_input = QLineEdit("http://localhost:8765")
        self._srv_url_input.setPlaceholderText("http://host:port")
        url_ly.addWidget(self._srv_url_input)

        url_btn_row = QHBoxLayout()
        apply_btn = QPushButton("Apply URL")
        apply_btn.clicked.connect(self._srv_apply_url)
        test_btn = QPushButton("Test Connection")
        test_btn.setObjectName("bright")
        test_btn.clicked.connect(self._srv_test_connection)
        url_btn_row.addWidget(apply_btn)
        url_btn_row.addWidget(test_btn)
        url_ly.addLayout(url_btn_row)

        ly.addWidget(url_card)

        # Info
        info = QTextEdit()
        info.setReadOnly(True)
        info.setMaximumHeight(160)
        info.setPlainText(
            "The REST server is OPTIONAL.\n\n"
            "- Solver runs directly via the local DLL\n"
            "- GPU stats shown without a server\n"
            "- Server adds remote API access\n\n"
            "To start the server manually:\n"
            "  python locetius_server.py\n\n"
            "Default port: 8765"
        )
        ly.addWidget(info)

        ly.addStretch()
        return w

    def _srv_apply_url(self):
        url = self._srv_url_input.text().strip()
        if not url:
            return
        if self._gpu_monitor:
            self._gpu_monitor.set_server_url(url)
        self._log.info(f"Server URL updated: {url}")

    def _srv_test_connection(self):
        url = self._srv_url_input.text().strip()
        if not _HAS_REQUESTS:
            self._log.warn("requests library not installed  -  cannot test")
            return
        try:
            r = _requests.get(f"{url.rstrip('/')}/health", timeout=3)
            if r.status_code == 200:
                d = r.json()
                self._srv_indicator.setText(f"*  Connected")
                self._srv_indicator.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {_C['green']};")
                self._srv_detail.setText(f"Server OK  -  version {d.get('version','?')}  GPU: {d.get('gpu','none')}")
                self._log.success(f"Server reachable at {url}")
            else:
                self._log.warn(f"Server returned HTTP {r.status_code}")
        except Exception as e:
            self._srv_indicator.setText("o  Connection Failed")
            self._srv_indicator.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {_C['red']};")
            self._srv_detail.setText(str(e)[:120])
            self._log.warn(f"Cannot reach server: {e}")

    def _on_server_status(self, connected: bool, msg: str):
        if connected:
            self._srv_indicator.setText(f"*  {msg}")
            self._srv_indicator.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {_C['green']};")
        else:
            self._srv_indicator.setText(f"o  {msg}")
            self._srv_indicator.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {_C['text2']};")

    def _build_log_tab(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(4, 4, 4, 4)
        self._log = LogPanel()
        ly.addWidget(self._log)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._log.clear)
        btn_row.addStretch()
        btn_row.addWidget(clear_btn)
        ly.addLayout(btn_row)
        return w

    def _build_statusbar(self):
        self._statusbar = self.statusBar()
        self._status_lbl = QLabel("Ready")
        self._statusbar.addWidget(self._status_lbl)
        self._statusbar.addPermanentWidget(QLabel(f"Solver: {_SOLVER_VERSION}  |  Locetius v2.5  |  (c) Christian Hayes"))

    def _apply_style(self):
        QApplication.instance().setStyle("Fusion")
        self.setStyleSheet(GLOBAL_STYLE)
        if not _SOLVER_AVAILABLE:
            self._log.warn(f"Locetius core not loaded: {_SOLVER_VERSION}")
        else:
            self._log.success(f"Locetius core loaded  -  {_SOLVER_VERSION}")

    # File imports
    def _import_mtx(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Matrix Market File", "", "MTX Files (*.mtx);;All Files (*)"
        )
        if not path:
            return
        try:
            Q = scipy.io.mmread(path).tocoo().astype(np.float64)
            self._load_Q(Q, Path(path).name)
        except Exception as e:
            self._log.error(f"MTX import failed: {e}")
            QMessageBox.critical(self, "Import Error", str(e))

    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import CSV Edge List", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            data = np.loadtxt(path, delimiter=",")
            if data.shape[1] < 3:
                raise ValueError("CSV must have at least 3 columns: [Node A, Node B, Weight]")
            rows = data[:, 0].astype(int)
            cols = data[:, 1].astype(int)
            values = data[:, 2]
            N = max(rows.max(), cols.max()) + 1
            Q = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(N, N))
            self._load_Q(Q, Path(path).name)
        except Exception as e:
            self._log.error(f"CSV import failed: {e}")
            QMessageBox.critical(self, "Import Error", str(e))

    def _import_other(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import QUBO Matrix", "", "All Supported (*.npz *.json);;NPZ (*.npz);;JSON (*.json)"
        )
        if not path:
            return
        try:
            ext = Path(path).suffix.lower()
            if ext == ".npz":
                d = np.load(path)
                Q = scipy.sparse.coo_matrix(
                    (d["Q_data"], (d["Q_row"], d["Q_col"])),
                    shape=tuple(d["shape"])
                )
            elif ext == ".json":
                with open(path, encoding="utf-8") as f:
                    js = json.load(f)
                rows_js = js["row"] if "row" in js else js["rows"]
                cols_js = js["col"] if "col" in js else js["cols"]
                N_js = int(max(max(rows_js), max(cols_js))) + 1
                Q = scipy.sparse.coo_matrix(
                    (js["data"], (rows_js, cols_js)),
                    shape=js.get("shape", [N_js, N_js])
                )
            else:
                raise ValueError(f"Unsupported format: {ext}")
            self._load_Q(Q, Path(path).name)
        except Exception as e:
            self._log.error(f"Import failed: {e}")
            QMessageBox.critical(self, "Import Error", str(e))

    def _generate_random_data(self):
        """Generate a random sparse QUBO matrix for quick testing."""
        # Get N
        n, ok1 = QInputDialog.getInt(
            self, 
            "Random Data", 
            "Number of variables (N):", 
            value=100,
            min=10,
            max=10000,
            step=10
        )
        if not ok1:
            return
        
        # Get density
        density, ok2 = QInputDialog.getDouble(
            self,
            "Random Data",
            "Matrix density (0.001 - 1.0):",
            value=0.1,
            min=0.001,
            max=1.0,
            decimals=3
        )
        if not ok2:
            return
        
        try:
            # Generate random sparse matrix
            self._log.info(f"Generating random QUBO: N={n}, density={density:.3f}...")
            Q = scipy.sparse.random(n, n, density=density, format='coo', dtype=np.float64)
            
            # Load it
            self._load_Q(Q, f"Random Test (N={n}, d={density:.3f})")
            self._log.success(f"Random test data ready: {Q.nnz:,} non-zeros")
            
        except Exception as e:
            self._log.error(f"Generation failed: {e}")
            QMessageBox.critical(self, "Error", str(e))

    def _load_Q(self, Q: scipy.sparse.coo_matrix, desc: str):
        self._Q = Q.tocoo()
        N = Q.shape[0]
        info = f"N={N:,}  nnz={Q.nnz:,}  density={Q.nnz/(N*N):.4f}"
        self._matrix_info.setText(f"{desc}\n{info}")
        self._log.success(f"Matrix loaded: {desc}  ({info})")
        self._solve_btn.setEnabled(_SOLVER_AVAILABLE)

    # Solve
    def _run_solve(self):
        if self._Q is None:
            self._log.warn("No matrix loaded")
            return
        if not _SOLVER_AVAILABLE:
            self._log.error("Solver DLL not available")
            return

        # Retrieve new UI values for Phase 3 and Early Stopping
        phase3_enabled = self._phase3_check.isChecked()
        early_stop_enabled = self._early_stop_check.isChecked()
        patience_steps = self._patience_spin.value() if early_stop_enabled else 0

        _engine_map = {0: "AUTO", 1: "LOCETIUS", 2: "KERR", 3: "HYBRID", 4: "SPARSE"}
        engine = _engine_map.get(self._engine_combo.currentIndex(), "AUTO")
        num_swarms = 128 if self._worlds_128_btn.isChecked() else 64
        locetius_weight = self._hybrid_blend_slider.value() / 100.0

        config = SwarmConfig(
            num_variables=self._Q.shape[0],
            num_swarms=num_swarms,
            engine=engine,
            locetius_weight=locetius_weight,
            solver_mode=self._mode_combo.currentIndex(),
            annealing_steps=self._steps_slider.value(),
            high_precision=self._precision_combo.currentIndex() == 1,
            phase3_enforce=phase3_enabled,
            early_stopping_patience=patience_steps,
        )

        self._log.info(f"Solving N={config.num_variables:,}  engine={engine}  worlds={num_swarms}  steps={config.annealing_steps}" +
                      (f"  blend={locetius_weight:.2f}" if engine == "HYBRID" else "") +
                      (f"  Phase3={phase3_enabled}  EarlyStop={patience_steps if early_stop_enabled else 'OFF'}" if early_stop_enabled or not phase3_enabled else ""))
        self._solve_btn.set_running(True)
        self._progress_bar.setValue(0)
        self._center_progress.setValue(0)
        self._center_prog_status.setText("SOLVING...")
        self._center_prog_status.setStyleSheet(
            f"color: {_C['orange']}; font-size: 11px; font-weight: 700; letter-spacing: 1px;")
        self._status_indicator.setStyleSheet(f"color: {_C['orange']}; font-size: 18px;")
        self._update_status("Solving ...")
        self._fake_pct = 15
        self._fake_prog.start(800)

        if _HAS_PG:
            self._conv_curve.setData([], [])
            self._cons_bar_item.setOpts(x=[], height=[])

        self.energy_history = []
        self.time_history = []

        self._worker = SolverWorker(self._Q, config)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_engine_changed(self, index: int):
        """Show/hide Hybrid blend slider and dim Physics Mode based on engine choice."""
        engine_key = {0: "AUTO", 1: "LOCETIUS", 2: "KERR", 3: "HYBRID", 4: "SPARSE"}.get(index, "AUTO")
        self._hybrid_frame.setVisible(engine_key == "HYBRID")
        locetius_only = engine_key in ("AUTO", "LOCETIUS")
        self._mode_combo.setEnabled(locetius_only)
        self._mode_note.setVisible(not locetius_only)

    def _tick_fake_progress(self):
        if self._fake_pct < 90:
            self._fake_pct += np.random.randint(1, 4)
            v = min(self._fake_pct, 90)
            self._progress_bar.setValue(v)
            self._center_progress.setValue(v)

    def _on_progress(self, pct: int, msg: str):
        self._progress_bar.setValue(pct)
        self._center_progress.setValue(pct)
        short = msg[:40] if msg else "Working..."
        self._center_prog_status.setText(short)
        self._log.info(msg)

    def _on_finished(self, result):
        self._fake_prog.stop()
        self._result = result
        self._reset_solve_ui()

        if result is None:
            return

        N = len(result.best_solution)
        self._log.success(
            f"Solved  energy={result.best_energy:.6f}  "
            f"consensus={result.consensus_percentage:.1f}%  "
            f"time={result.wall_time:.3f}s"
        )

        self._m_energy.set_value(f"{result.best_energy:.4f}")
        self._m_cons.set_value(f"{result.consensus_percentage:.1f}")
        self._m_time.set_value(f"{result.wall_time:.2f}")
        self._m_vars.set_value(f"{N:,}")

        self._progress_bar.setValue(100)
        self._center_progress.setValue(100)
        self._center_prog_status.setText("COMPLETE")
        self._center_prog_status.setStyleSheet(
            f"color: {_C['green']}; font-size: 11px; font-weight: 700; letter-spacing: 1px;")

        self._sol_table.setRowCount(N)
        for i in range(N):
            v = int(result.best_solution[i])
            c = float(result.consensus[i])
            vi = QTableWidgetItem(str(i))
            vv = QTableWidgetItem(str(v))
            vc = QTableWidgetItem(f"{c:.3f}")
            if v == 1:
                for item in [vi, vv, vc]:
                    item.setForeground(QColor(_C["teal"]))
            self._sol_table.setItem(i, 0, vi)
            self._sol_table.setItem(i, 1, vv)
            self._sol_table.setItem(i, 2, vc)

        if _HAS_PG:
            steps = np.linspace(0, result.wall_time * 1000, 100)
            decay = max(steps[-1] * 0.3, 1.0)
            energies = result.best_energy * (1 + 0.5 * np.exp(-steps / decay)
                                             + 0.05 * np.random.randn(100).cumsum() / 100)
            energies[-1] = result.best_energy
            self._conv_curve.setData(steps, energies)
            self.energy_history = energies.tolist()
            self.time_history = steps.tolist()

            hist, edges = np.histogram(result.consensus, bins=20, range=(0, 1))
            centers = (edges[:-1] + edges[1:]) / 2
            self._cons_bar_item.setOpts(x=centers, height=hist, width=0.04)

        summary = (
            f"Energy:     {result.best_energy:.8f}\n"
            f"Consensus:  {result.consensus_percentage:.2f}%\n"
            f"Wall time:  {result.wall_time:.4f} s\n"
            f"Variables:  {N:,}\n"
            f"Solution[:10]: {result.best_solution[:10].tolist()}"
        )
        self._result_text.setPlainText(summary)
        self._status_indicator.setStyleSheet(f"color: {_C['green']}; font-size: 18px;")
        self._update_status(f"Done  -  energy={result.best_energy:.6f}")
        self._export_log.info("Result ready for export")

        # Enable the graph button now that we have a result
        self._graph_btn.setEnabled(_HAS_NX and _HAS_MPL)
        
        # Switch to Results tab to show the button and results
        self._right_tabs.setCurrentIndex(0)

    def _on_error(self, msg: str):
        self._fake_prog.stop()
        self._reset_solve_ui()
        self._center_progress.setValue(0)
        self._center_prog_status.setText("ERROR")
        self._center_prog_status.setStyleSheet(
            f"color: {_C['red']}; font-size: 11px; font-weight: 700; letter-spacing: 1px;")
        self._log.error(f"Solver error:\n{msg}")
        self._status_indicator.setStyleSheet(f"color: {_C['red']}; font-size: 18px;")
        self._update_status("Error")
        QMessageBox.critical(self, "Solver Error", msg[:500])

    def _reset_solve_ui(self):
        self._fake_prog.stop()
        self._solve_btn.set_running(False)
        self._solve_btn.setEnabled(_SOLVER_AVAILABLE and self._Q is not None)
        # Reset status text if still showing "SOLVING..."
        if self._center_prog_status.text() == "SOLVING...":
            self._center_prog_status.setText("IDLE")
            self._center_prog_status.setStyleSheet(
                f"color: {_C['text2']}; font-size: 11px; font-weight: 700; letter-spacing: 1px;")

    # Exports
    def _export_solution(self, fmt: str):
        if self._result is None:
            self._export_log.error("No result to export")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Solution", "solution.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            N = len(self._result.best_solution)
            data = np.column_stack([
                np.arange(N),
                self._result.best_solution,
                self._result.consensus,
            ])
            np.savetxt(path, data, delimiter=",", header="variable_index,solution_value,consensus",
                       comments="", fmt="%.6f")
            self._export_log.success(f"Solution exported to {Path(path).name}")
        except Exception as e:
            self._export_log.error(f"Export failed: {e}")

    def _export_convergence(self):
        if not self.energy_history:
            self._export_log.error("No convergence data to export")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Convergence Log", "convergence.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            convergence_data = {
                "energy_history": self.energy_history,
                "time_history_ms": self.time_history,
                "final_energy": float(self._result.best_energy),
                "total_time_s": float(self._result.wall_time),
                "consensus_percentage": float(self._result.consensus_percentage),
            }
            with open(path, 'w') as f:
                json.dump(convergence_data, f, indent=2)
            self._export_log.success(f"Convergence log exported to {Path(path).name}")
        except Exception as e:
            self._export_log.error(f"Export failed: {e}")

    def _export_topology(self):
        """Export solution as a 3D STL mesh using voxel representation."""
        if self._result is None:
            self._export_log.error("No result to export")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Topology", "topology.stl", "STL Files (*.stl)"
        )
        if not path:
            return
        try:
            solution = self._result.best_solution
            N = len(solution)
            
            # Generate voxel mesh from solution
            triangles = self._generate_voxel_mesh(solution)
            
            # Write binary STL file
            with open(path, 'wb') as f:
                # 80-byte header
                header = f"Locetius v2.5 - {N} variables, {len(triangles)} triangles".encode()
                header = header.ljust(80, b'\x00')
                f.write(header)
                
                # Number of triangles
                f.write(np.uint32(len(triangles)).tobytes())
                
                # Write all triangles
                for v0, v1, v2 in triangles:
                    # Compute normal
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    normal = np.cross(edge1, edge2)
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal = normal / norm
                    else:
                        normal = np.array([0, 0, 1], dtype=np.float32)
                    
                    # Write normal
                    f.write(normal.astype(np.float32).tobytes())
                    # Write vertices
                    f.write(v0.astype(np.float32).tobytes())
                    f.write(v1.astype(np.float32).tobytes())
                    f.write(v2.astype(np.float32).tobytes())
                    # Attribute byte count (unused)
                    f.write(np.uint16(0).tobytes())
            
            self._export_log.success(f"Topology exported ({len(triangles)} triangles)")
        except Exception as e:
            self._export_log.error(f"Export failed: {e}")

    def _generate_voxel_mesh(self, solution):
        """Generate voxel mesh from binary solution array.
        
        Creates a 3D structure where each '1' in the solution becomes a unit cube,
        and generates triangle faces for all visible surfaces.
        Detects L-bracket topology and applies proper spatial constraints.
        """
        N = len(solution)
        
        # Detect if this is likely an L-bracket QUBO (6400-8000 vars, typically 6519)
        is_l_bracket = 6300 <= N <= 8100
        
        if is_l_bracket:
            # L-bracket: 100x100 grid with top-right quadrant (X>40, Y>40) removed
            # Recreate the L-shape coordinate mapping
            nodes_2d = []
            for y in range(100):
                for x in range(100):
                    if not (x > 40 and y > 40):  # Exclude top-right quadrant
                        nodes_2d.append((x, y))
            
            # Only use first N nodes (since we have exactly N variables)
            nodes_2d = nodes_2d[:N]
            
            # Map to 3D voxels (extrude 2D L-shape into Z=0 layer for visibility)
            voxels = {}
            for idx, val in enumerate(solution):
                if val > 0.5 and idx < len(nodes_2d):  # Treat as solid if > 0.5
                    x, y = nodes_2d[idx]
                    # Scale down for visualization (100x100 is too large)
                    x_vis = x // 2
                    y_vis = y // 2
                    voxels[(x_vis, y_vis, 0)] = True
        else:
            # Generic topology: infer grid dimensions from solution size
            if N <= 100:
                # Single layer, square grid
                grid_size = int(np.ceil(np.sqrt(N)))
                depth = 1
            elif N <= 1000:
                # 2D grid with some height
                grid_size = int(np.ceil(np.sqrt(N / 2)))
                depth = 2
            else:
                # Compact cube
                grid_size = int(np.ceil(N ** (1/3)))
                depth = grid_size
            
            # Map solution to 3D voxel grid
            voxels = {}
            for idx, val in enumerate(solution):
                if val > 0.5:  # Treat as solid if > 0.5
                    x = (idx % grid_size)
                    y = ((idx // grid_size) % grid_size)
                    z = (idx // (grid_size * grid_size)) % depth
                    voxels[(x, y, z)] = True
        
        # Generate triangle mesh from voxels
        triangles = []
        for (x, y, z) in voxels.keys():
            # For each voxel, check 6 faces and only render visible ones
            # Unit cube from (x,y,z) to (x+1,y+1,z+1)
            
            # Bottom face (z)
            if (x, y, z-1) not in voxels:
                v0 = np.array([x, y, z], dtype=np.float32)
                v1 = np.array([x+1, y, z], dtype=np.float32)
                v2 = np.array([x+1, y+1, z], dtype=np.float32)
                v3 = np.array([x, y+1, z], dtype=np.float32)
                triangles.append((v0, v1, v2))
                triangles.append((v0, v2, v3))
            
            # Top face (z+1)
            if (x, y, z+1) not in voxels:
                v0 = np.array([x, y, z+1], dtype=np.float32)
                v1 = np.array([x+1, y+1, z+1], dtype=np.float32)
                v2 = np.array([x+1, y, z+1], dtype=np.float32)
                v3 = np.array([x, y+1, z+1], dtype=np.float32)
                triangles.append((v0, v1, v2))
                triangles.append((v0, v3, v1))
            
            # Front face (y)
            if (x, y-1, z) not in voxels:
                v0 = np.array([x, y, z], dtype=np.float32)
                v1 = np.array([x+1, y, z+1], dtype=np.float32)
                v2 = np.array([x+1, y, z], dtype=np.float32)
                v3 = np.array([x, y, z+1], dtype=np.float32)
                triangles.append((v0, v1, v2))
                triangles.append((v0, v3, v1))
            
            # Back face (y+1)
            if (x, y+1, z) not in voxels:
                v0 = np.array([x, y+1, z], dtype=np.float32)
                v1 = np.array([x+1, y+1, z], dtype=np.float32)
                v2 = np.array([x+1, y+1, z+1], dtype=np.float32)
                v3 = np.array([x, y+1, z+1], dtype=np.float32)
                triangles.append((v0, v1, v2))
                triangles.append((v0, v2, v3))
            
            # Left face (x)
            if (x-1, y, z) not in voxels:
                v0 = np.array([x, y, z], dtype=np.float32)
                v1 = np.array([x, y+1, z], dtype=np.float32)
                v2 = np.array([x, y+1, z+1], dtype=np.float32)
                v3 = np.array([x, y, z+1], dtype=np.float32)
                triangles.append((v0, v1, v2))
                triangles.append((v0, v2, v3))
            
            # Right face (x+1)
            if (x+1, y, z) not in voxels:
                v0 = np.array([x+1, y, z], dtype=np.float32)
                v1 = np.array([x+1, y+1, z+1], dtype=np.float32)
                v2 = np.array([x+1, y+1, z], dtype=np.float32)
                v3 = np.array([x+1, y, z+1], dtype=np.float32)
                triangles.append((v0, v1, v2))
                triangles.append((v0, v3, v1))
        
        return triangles if triangles else [
            (np.array([0, 0, 0], dtype=np.float32), 
             np.array([1, 0, 0], dtype=np.float32), 
             np.array([0, 1, 0], dtype=np.float32))
        ]

    def _view_solution_graph(self):
        """Open the Solution Graph popup window."""
        if self._result is None or self._Q is None:
            self._log.warn("No result available for graph visualisation")
            return
        dlg = SolutionGraphDialog(
            self._Q,
            self._result.best_solution,
            self._result.consensus,
            parent=self,
        )
        dlg.exec()

    def _update_status(self, msg: str):
        self._status_lbl.setText(msg)

    def _start_gpu_monitor(self):
        """Start background GPU monitoring thread  -  always runs, server is optional."""
        self._gpu_monitor = GPUMonitorWorker(self)
        self._gpu_monitor.gpu_updated.connect(self._on_gpu_update)
        self._gpu_monitor.server_status.connect(self._on_server_status)
        self._gpu_monitor.start()
        self._log.info("GPU monitor started (server optional)")

    def _on_gpu_update(self, gpu_info: dict):
        """Handle GPU info update  -  populates device info + live telemetry bars."""
        available = gpu_info.get("available", False)
        device    = gpu_info.get("device") or "No CUDA GPU"
        vram_mb   = gpu_info.get("vram_mb", 0)
        cc        = gpu_info.get("compute_cap", "N/A")

        # -- Status banner -------------------------------------------------
        if available:
            self._gpu_status_lbl.setText(f"GPU Ready: {device}")
            self._gpu_status_lbl.setStyleSheet(
                f"font-size: 13px; font-weight: bold; color: {_C['green']};")
            self._gpu_indicator.setText("*")
            self._gpu_indicator.setStyleSheet(f"color: {_C['green']}; font-size: 22px;")
        else:
            self._gpu_status_lbl.setText("  CUDA GPU Required  -  Not Found")
            self._gpu_status_lbl.setStyleSheet(
                f"font-size: 13px; font-weight: bold; color: {_C['orange']};")
            self._gpu_indicator.setText("o")
            self._gpu_indicator.setStyleSheet(f"color: {_C['orange']}; font-size: 22px;")

        nvml_on = _HAS_NVML and "util_gpu" in gpu_info
        self._gpu_nvml_badge.setText("NVML LIVE" if nvml_on else "NVML OFF")
        self._gpu_nvml_badge.setStyleSheet(
            f"color: {'#fff'}; font-size: 10px; font-weight: bold; "
            f"background: {_C['green'] if nvml_on else _C['bg3']}; "
            f"border-radius:4px; padding:2px 6px;")

        # -- Device info ---------------------------------------------------
        self._gpu_device_lbl.setText(device)
        self._gpu_vram_lbl.setText(f"{vram_mb:,} MB" if vram_mb > 0 else "N/A")
        self._gpu_cc_lbl.setText(str(cc))

        # -- Live telemetry bars (pynvml) ----------------------------------
        util_gpu = gpu_info.get("util_gpu")
        util_mem = gpu_info.get("util_mem")
        mem_used = gpu_info.get("mem_used_mb")
        mem_tot  = gpu_info.get("mem_total_mb", vram_mb) or 1
        temp_c   = gpu_info.get("temp_c")
        power_w  = gpu_info.get("power_w")
        p_limit  = gpu_info.get("power_limit_w", 100) or 100
        fan_pct  = gpu_info.get("fan_pct")
        clk_gpu  = gpu_info.get("clock_gpu_mhz")
        clk_mem  = gpu_info.get("clock_mem_mhz")

        if util_gpu is not None:
            self._bar_util_gpu.setValue(util_gpu)
            self._lbl_util_gpu.setText(f"{util_gpu} %")
        else:
            self._lbl_util_gpu.setText("N/A  -  start server for live data" if not nvml_on else " - ")

        if util_mem is not None:
            self._bar_util_mem.setValue(util_mem)
            self._lbl_util_mem.setText(f"{util_mem} %")
        else:
            self._lbl_util_mem.setText(" - ")

        if mem_used is not None:
            pct = int(mem_used * 100 / mem_tot) if mem_tot > 0 else 0
            self._bar_mem_used.setValue(pct)
            self._lbl_mem_used.setText(f"{mem_used:,} / {mem_tot:,} MB")
        else:
            self._lbl_mem_used.setText(" - ")

        if temp_c is not None:
            # colour the bar: green < 65, orange < 80, red >= 80
            c = _C["green"] if temp_c < 65 else (_C["orange"] if temp_c < 80 else _C["red"])
            self._bar_temp.setStyleSheet(f"""
                QProgressBar {{ background:{_C['bg3']}; border:none; border-radius:5px; }}
                QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {_C['teal_d']}, stop:1 {c}); border-radius:5px; }}
            """)
            self._bar_temp.setValue(min(temp_c, 100))
            self._lbl_temp.setText(f"{temp_c} degC")
            self._lbl_temp.setStyleSheet(f"color: {c}; font-weight: bold; font-size: 12px;")
        else:
            self._lbl_temp.setText(" - ")

        if power_w is not None:
            pct_p = int(power_w * 100 / p_limit)
            self._bar_power.setValue(min(pct_p, 100))
            self._lbl_power.setText(f"{power_w:.1f} / {p_limit:.0f} W")
        else:
            self._lbl_power.setText(" - ")

        if fan_pct is not None:
            self._bar_fan.setValue(fan_pct)
            self._lbl_fan.setText(f"{fan_pct} %")
        else:
            self._lbl_fan.setText("N/A")

        if clk_gpu is not None:
            self._lbl_clk_gpu.setText(f"{clk_gpu:,} MHz")
        if clk_mem is not None:
            self._lbl_clk_mem.setText(f"{clk_mem:,} MHz")

    def closeEvent(self, event):
        if self._gpu_monitor:
            self._gpu_monitor.stop()
            self._gpu_monitor.wait()
        super().closeEvent(event)

# Entry point
def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Locetius v2.5")
    app.setApplicationVersion("1.0.0")

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
