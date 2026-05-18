"""
FPS (frames-per-second) benchmark for the combined vision pipeline.

Runs the full YOLO + Headset + Proctor pipeline as fast as possible for a
fixed duration and reports sustained FPS.

Usage::

    python -m benchmark.pt_benchmark.fps_benchmark [--duration 10] [--gpu]

Output: benchmark/outputs/pt_fps.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.pt_benchmark.bench_utils import make_frame, save_json, gpu_memory_mb
from app.services.yolo_service import YOLOService
from app.services.headset_service import HeadsetService
from app.services.proctor_service import ProctorService

_MODELS     = _ROOT / "app" / "models"
_YOLO_PT    = str(_MODELS / "yolov8n.pt")
_HEADSET_PT = str(_MODELS / "headset_model.pt")
_PROCTOR_PT = str(_MODELS / "proctoring.pt")


def run(duration_s: float = 10.0, enable_gpu: bool = False) -> dict:
    """Run the FPS benchmark.

    Args:
        duration_s: How many seconds to run the pipeline loop.
        enable_gpu: Whether to request GPU inference.

    Returns:
        Dict with FPS statistics.
    """
    print(f"\n{'='*60}")
    print(f"  PT FPS Benchmark  |  duration={duration_s}s  gpu={enable_gpu}")
    print(f"{'='*60}\n")

    YOLOService.initialize(_YOLO_PT, enable_gpu=enable_gpu)
    HeadsetService.initialize(_HEADSET_PT, enable_gpu=enable_gpu)
    ProctorService.initialize(_PROCTOR_PT, enable_gpu=enable_gpu)

    # Pre-generate a pool of 20 frames to avoid numpy allocation overhead
    # skewing the timing.
    pool = [make_frame(480, 640, seed=i) for i in range(20)]

    frame_count = 0
    frame_times: list[float] = []

    print(f"Running pipeline loop for {duration_s}s...")
    deadline = time.perf_counter() + duration_s
    idx = 0

    while time.perf_counter() < deadline:
        frame = pool[idx % len(pool)]
        idx += 1

        t0 = time.perf_counter()
        YOLOService.detect(frame)
        HeadsetService.classify(frame)
        ProctorService.analyze(frame)
        elapsed = (time.perf_counter() - t0) * 1000.0

        frame_times.append(elapsed)
        frame_count += 1

    actual_duration = sum(frame_times) / 1000.0
    fps = frame_count / actual_duration if actual_duration > 0 else 0.0

    import numpy as np
    arr = np.array(frame_times)
    results = {
        "config": {"duration_s": duration_s, "enable_gpu": enable_gpu},
        "frames_processed": frame_count,
        "actual_duration_s": round(actual_duration, 3),
        "fps": round(fps, 2),
        "latency_ms": {
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "mean":   float(arr.mean()),
            "p95":    float(np.percentile(arr, 95)),
            "p99":    float(np.percentile(arr, 99)),
        },
    }

    alloc = gpu_memory_mb()
    if alloc is not None:
        results["gpu_allocated_mb"] = alloc

    print(f"\nFrames processed : {frame_count}")
    print(f"Actual duration  : {actual_duration:.2f}s")
    print(f"Sustained FPS    : {fps:.2f}")
    print(f"Mean latency     : {arr.mean():.1f}ms")
    print(f"p95 latency      : {float(np.percentile(arr, 95)):.1f}ms")

    save_json(results, "pt_fps.json")
    print("\nDone.\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="PT FPS benchmark")
    parser.add_argument("--duration", type=float, default=10.0, help="Run duration in seconds")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU inference")
    args = parser.parse_args()
    run(duration_s=args.duration, enable_gpu=args.gpu)


if __name__ == "__main__":
    main()
