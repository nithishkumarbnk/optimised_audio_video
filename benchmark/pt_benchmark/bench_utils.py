"""
Shared utilities for the PyTorch .pt benchmark suite.

Provides:
- synthetic frame generation
- result serialisation helpers
- GPU memory sampling
- CPU / wall-clock timing context manager
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic frame factory
# ---------------------------------------------------------------------------

def make_frame(height: int = 480, width: int = 640, seed: Optional[int] = None) -> np.ndarray:
    """Return a synthetic BGR uint8 frame.

    Args:
        height: Frame height in pixels.
        width:  Frame width in pixels.
        seed:   Optional RNG seed for reproducibility.

    Returns:
        ``uint8`` ndarray of shape ``(height, width, 3)``.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def make_frames(n: int, height: int = 480, width: int = 640) -> List[np.ndarray]:
    """Return a list of *n* synthetic BGR frames."""
    return [make_frame(height, width, seed=i) for i in range(n)]


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    """Wall-clock timing result from :func:`timed`."""

    elapsed_ms: float
    label: str = ""


@contextmanager
def timed(label: str = "") -> Generator[TimingResult, None, None]:
    """Context manager that measures wall-clock elapsed time in milliseconds.

    Usage::

        with timed("YOLO inference") as t:
            detections = YOLOService.detect(frame)
        print(t.elapsed_ms)
    """
    result = TimingResult(elapsed_ms=0.0, label=label)
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------

def gpu_memory_mb() -> Optional[float]:
    """Return current GPU allocated memory in MB, or ``None`` if CUDA is unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return None


def gpu_reserved_mb() -> Optional[float]:
    """Return current GPU reserved memory in MB, or ``None`` if CUDA is unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / (1024 ** 2)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# CPU utilisation helper
# ---------------------------------------------------------------------------

def cpu_percent() -> Optional[float]:
    """Return current process CPU utilisation percentage, or ``None``."""
    try:
        import psutil
        return psutil.Process(os.getpid()).cpu_percent(interval=0.05)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def stats(values: List[float]) -> Dict[str, float]:
    """Compute descriptive statistics for a list of floats.

    Returns a dict with keys: min, max, mean, median, p95, p99, std.
    """
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
        "std":    float(arr.std()),
    }


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------

def save_json(data: Any, filename: str) -> Path:
    """Serialise *data* to JSON and write to the benchmark outputs directory.

    Args:
        data:     JSON-serialisable object.
        filename: Output filename (e.g. ``"pt_latency.json"``).

    Returns:
        Absolute path to the written file.
    """
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    print(f"[bench_utils] Saved → {path}")
    return path
