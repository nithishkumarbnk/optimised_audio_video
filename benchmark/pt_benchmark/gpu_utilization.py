"""
GPU / CPU utilisation monitor for the vision pipeline.

Samples GPU memory and CPU utilisation while running the pipeline in a
background thread, then reports peak and average values.

Usage::

    python -m benchmark.pt_benchmark.gpu_utilization [--frames 50] [--gpu]

Output: benchmark/outputs/pt_gpu_utilization.json

Note: GPU metrics require CUDA. CPU metrics require ``psutil`` (optional).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.pt_benchmark.bench_utils import (
    make_frames,
    save_json,
    stats,
    gpu_memory_mb,
    gpu_reserved_mb,
    cpu_percent,
)
from app.services.yolo_service import YOLOService
from app.services.headset_service import HeadsetService
from app.services.proctor_service import ProctorService

_MODELS     = _ROOT / "app" / "models"
_YOLO_PT    = str(_MODELS / "yolov8n.pt")
_HEADSET_PT = str(_MODELS / "headset_model.pt")
_PROCTOR_PT = str(_MODELS / "proctoring.pt")


class _Sampler(threading.Thread):
    """Background thread that samples GPU/CPU metrics at 10 Hz."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.gpu_alloc_samples: List[float] = []
        self.gpu_reserved_samples: List[float] = []
        self.cpu_samples: List[float] = []
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            alloc = gpu_memory_mb()
            if alloc is not None:
                self.gpu_alloc_samples.append(alloc)
            reserved = gpu_reserved_mb()
            if reserved is not None:
                self.gpu_reserved_samples.append(reserved)
            cpu = cpu_percent()
            if cpu is not None:
                self.cpu_samples.append(cpu)
            time.sleep(0.1)

    def stop(self) -> None:
        self._stop_event.set()


def run(n_frames: int = 50, enable_gpu: bool = False) -> dict:
    """Run the GPU/CPU utilisation benchmark.

    Args:
        n_frames:   Number of synthetic frames to process.
        enable_gpu: Whether to request GPU inference.

    Returns:
        Dict with GPU/CPU utilisation statistics.
    """
    print(f"\n{'='*60}")
    print(f"  PT GPU/CPU Utilisation  |  frames={n_frames}  gpu={enable_gpu}")
    print(f"{'='*60}\n")

    YOLOService.initialize(_YOLO_PT, enable_gpu=enable_gpu)
    HeadsetService.initialize(_HEADSET_PT, enable_gpu=enable_gpu)
    ProctorService.initialize(_PROCTOR_PT, enable_gpu=enable_gpu)

    frames = make_frames(n_frames, height=480, width=640)

    sampler = _Sampler()
    sampler.start()

    print(f"Running pipeline on {n_frames} frames while sampling metrics...")
    t0 = time.perf_counter()
    for frame in frames:
        YOLOService.detect(frame)
        HeadsetService.classify(frame)
        ProctorService.analyze(frame)
    elapsed = time.perf_counter() - t0

    sampler.stop()
    sampler.join(timeout=1.0)

    results: dict = {
        "config": {"n_frames": n_frames, "enable_gpu": enable_gpu},
        "elapsed_s": round(elapsed, 3),
        "fps": round(n_frames / elapsed, 2) if elapsed > 0 else 0,
    }

    if sampler.gpu_alloc_samples:
        results["gpu_allocated_mb"] = stats(sampler.gpu_alloc_samples)
        print(f"GPU allocated  : mean={results['gpu_allocated_mb']['mean']:.1f}MB  "
              f"peak={results['gpu_allocated_mb']['max']:.1f}MB")
    else:
        print("GPU metrics    : N/A (CUDA not available or psutil not installed)")

    if sampler.cpu_samples:
        results["cpu_percent"] = stats(sampler.cpu_samples)
        print(f"CPU utilisation: mean={results['cpu_percent']['mean']:.1f}%  "
              f"peak={results['cpu_percent']['max']:.1f}%")
    else:
        print("CPU metrics    : N/A (psutil not installed)")

    print(f"Elapsed        : {elapsed:.2f}s  ({results['fps']:.1f} fps)")

    save_json(results, "pt_gpu_utilization.json")
    print("\nDone.\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="PT GPU/CPU utilisation benchmark")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames")
    parser.add_argument("--gpu",    action="store_true",  help="Enable GPU inference")
    args = parser.parse_args()
    run(n_frames=args.frames, enable_gpu=args.gpu)


if __name__ == "__main__":
    main()
