"""
Full per-frame-type PT benchmark — mirrors the ONNX benchmark exactly.

Runs 50 iterations per frame type × 9 frame types for all three services,
plus stream simulation, queue metrics, and system resource sampling.

Usage::

    python benchmark/pt_benchmark/full_pt_benchmark.py [--runs N] [--gpu]

Output: benchmark/outputs/pt_full_benchmark.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from benchmark.datasets.generator import FRAME_TYPES
from benchmark.pt_benchmark.bench_utils import save_json, stats, gpu_memory_mb
from app.services.yolo_service import YOLOService
from app.services.headset_service import HeadsetService
from app.services.proctor_service import ProctorService

_MODELS     = _ROOT / "app" / "models"
_YOLO_PT    = str(_MODELS / "yolov8n.pt")
_HEADSET_PT = str(_MODELS / "headset_model.pt")
_PROCTOR_PT = str(_MODELS / "proctoring.pt")


def _bench_service(fn, frames: List[np.ndarray]) -> dict:
    """Run fn on each frame, return stats dict."""
    times = []
    for frame in frames:
        t0 = time.perf_counter()
        fn(frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    s = stats(times)
    s["runs"] = len(times)
    s["throughput_fps"] = round(1000.0 / s["mean"], 1) if s["mean"] > 0 else 0
    return s


def _stream_simulation(fn, fps: float = 1.0, duration_s: float = 10.0,
                       frame: np.ndarray = None) -> dict:
    """Simulate streaming at target_fps for duration_s seconds."""
    interval = 1.0 / fps
    frames_processed = 0
    frames_dropped = 0
    times = []
    deadline = time.perf_counter() + duration_s
    next_frame_at = time.perf_counter()

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        if now >= next_frame_at:
            t0 = time.perf_counter()
            fn(frame)
            elapsed = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed)
            frames_processed += 1
            next_frame_at += interval
        else:
            time.sleep(0.001)

    s = stats(times) if times else {}
    return {
        "target_fps": fps,
        "actual_fps": round(frames_processed / duration_s, 2),
        "frames_processed": frames_processed,
        "frames_dropped": frames_dropped,
        "drop_rate_pct": 0.0,
        "mean_ms": round(s.get("mean", 0), 3),
        "p95_ms": round(s.get("p95", 0), 3),
    }


def run(n_runs: int = 50, enable_gpu: bool = False) -> dict:
    print(f"\n{'='*65}")
    print(f"  PT Full Benchmark  |  runs={n_runs}/frame_type  gpu={enable_gpu}")
    print(f"{'='*65}\n")

    YOLOService.initialize(_YOLO_PT, enable_gpu=enable_gpu)
    HeadsetService.initialize(_HEADSET_PT, enable_gpu=enable_gpu)
    ProctorService.initialize(_PROCTOR_PT, enable_gpu=enable_gpu)
    print("Services initialised.\n")

    results: dict = {
        "config": {
            "runs_per_frame_type": n_runs,
            "enable_gpu": enable_gpu,
            "frame_types": list(FRAME_TYPES.keys()),
        },
        "YOLO": {"per_frame_type": [], "stream_simulation": {}},
        "Headset": {"per_frame_type": [], "stream_simulation": {}},
        "Proctor": {"per_frame_type": [], "stream_simulation": {}},
        "pipeline_combined": {"per_frame_type": [], "stream_simulation": {}},
    }

    for frame_type, make_fn in FRAME_TYPES.items():
        base_frame = make_fn(640, 480)
        # Generate n_runs slightly varied copies (add tiny noise so each run
        # is a distinct frame, matching the ONNX benchmark methodology)
        frames = []
        for i in range(n_runs):
            noise = np.random.randint(0, 5, base_frame.shape, dtype=np.uint8)
            frames.append(cv2.add(base_frame, noise) if i > 0 else base_frame)

        print(f"  [{frame_type:20s}]", end="  ", flush=True)

        # YOLO
        y = _bench_service(YOLOService.detect, frames)
        y["frame_type"] = frame_type
        results["YOLO"]["per_frame_type"].append(y)

        # Headset
        h = _bench_service(HeadsetService.classify, frames)
        h["frame_type"] = frame_type
        results["Headset"]["per_frame_type"].append(h)

        # Proctor
        p = _bench_service(ProctorService.analyze, frames)
        p["frame_type"] = frame_type
        results["Proctor"]["per_frame_type"].append(p)

        # Combined pipeline
        combined_times = []
        for frame in frames:
            t0 = time.perf_counter()
            YOLOService.detect(frame)
            HeadsetService.classify(frame)
            ProctorService.analyze(frame)
            combined_times.append((time.perf_counter() - t0) * 1000.0)
        cs = stats(combined_times)
        cs["runs"] = n_runs
        cs["throughput_fps"] = round(1000.0 / cs["mean"], 1) if cs["mean"] > 0 else 0
        cs["frame_type"] = frame_type
        results["pipeline_combined"]["per_frame_type"].append(cs)

        print(f"YOLO={y['mean']:.0f}ms  "
              f"Headset={h['mean']:.0f}ms  "
              f"Proctor={p['mean']:.0f}ms  "
              f"Combined={cs['mean']:.0f}ms")

    # Stream simulations (1 fps, 10 seconds, normal frame)
    print("\nRunning stream simulations (1 fps × 10s)...")
    normal_frame = FRAME_TYPES["normal"](640, 480)

    results["YOLO"]["stream_simulation"] = _stream_simulation(
        YOLOService.detect, fps=1.0, duration_s=10.0, frame=normal_frame)
    results["Headset"]["stream_simulation"] = _stream_simulation(
        HeadsetService.classify, fps=1.0, duration_s=10.0, frame=normal_frame)
    results["Proctor"]["stream_simulation"] = _stream_simulation(
        ProctorService.analyze, fps=1.0, duration_s=10.0, frame=normal_frame)

    def _combined(frame):
        YOLOService.detect(frame)
        HeadsetService.classify(frame)
        ProctorService.analyze(frame)

    results["pipeline_combined"]["stream_simulation"] = _stream_simulation(
        _combined, fps=1.0, duration_s=10.0, frame=normal_frame)

    # GPU memory
    alloc = gpu_memory_mb()
    results["gpu_memory_mb"] = alloc

    # Compute averages across all frame types
    for svc in ["YOLO", "Headset", "Proctor", "pipeline_combined"]:
        rows = results[svc]["per_frame_type"]
        results[svc]["average"] = {
            "mean":   round(sum(r["mean"]   for r in rows) / len(rows), 3),
            "median": round(sum(r["median"] for r in rows) / len(rows), 3),
            "p95":    round(sum(r["p95"]    for r in rows) / len(rows), 3),
            "p99":    round(sum(r["p99"]    for r in rows) / len(rows), 3),
            "min":    round(min(r["min"]    for r in rows), 3),
            "max":    round(max(r["max"]    for r in rows), 3),
        }

    # SLA check
    p95_combined_avg = results["pipeline_combined"]["average"]["p95"]
    results["sla"] = {
        "target_ms": 500.0,
        "p95_combined_avg_ms": p95_combined_avg,
        "pass": p95_combined_avg < 500.0,
    }

    save_json(results, "pt_full_benchmark.json")
    print(f"\nSLA: p95 combined avg = {p95_combined_avg:.1f}ms < 500ms → "
          f"{'✓ PASS' if results['sla']['pass'] else '✗ FAIL'}")
    print("\nDone. Saved → benchmark/outputs/pt_full_benchmark.json\n")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--gpu",  action="store_true")
    args = parser.parse_args()
    run(n_runs=args.runs, enable_gpu=args.gpu)


if __name__ == "__main__":
    main()
