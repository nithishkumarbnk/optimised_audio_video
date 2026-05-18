"""
Latency benchmark for all three PyTorch .pt vision services.

Measures per-frame inference latency (ms) for:
  - YOLOService.detect
  - HeadsetService.classify
  - ProctorService.analyze

Usage::

    python -m benchmark.pt_benchmark.latency_benchmark [--frames N] [--gpu]

Output: benchmark/outputs/pt_latency.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly.
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.pt_benchmark.bench_utils import (
    make_frames,
    save_json,
    stats,
    timed,
    gpu_memory_mb,
    gpu_reserved_mb,
)
from app.services.yolo_service import YOLOService
from app.services.headset_service import HeadsetService
from app.services.proctor_service import ProctorService

# Model paths relative to project root
_MODELS = _ROOT / "app" / "models"
_YOLO_PT    = str(_MODELS / "yolov8n.pt")
_HEADSET_PT = str(_MODELS / "headset_model.pt")
_PROCTOR_PT = str(_MODELS / "proctoring.pt")


def run(n_frames: int = 50, enable_gpu: bool = False) -> dict:
    """Run the latency benchmark and return the results dict.

    Args:
        n_frames:   Number of synthetic frames to process per service.
        enable_gpu: Whether to request GPU inference.

    Returns:
        Dict with per-service latency statistics.
    """
    print(f"\n{'='*60}")
    print(f"  PT Latency Benchmark  |  frames={n_frames}  gpu={enable_gpu}")
    print(f"{'='*60}\n")

    # ── Initialise services ──────────────────────────────────────────
    print("[1/4] Initialising services...")
    YOLOService.initialize(_YOLO_PT, enable_gpu=enable_gpu)
    HeadsetService.initialize(_HEADSET_PT, enable_gpu=enable_gpu)
    ProctorService.initialize(_PROCTOR_PT, enable_gpu=enable_gpu)
    print("      Services ready.\n")

    frames = make_frames(n_frames, height=480, width=640)

    results: dict = {
        "config": {"n_frames": n_frames, "enable_gpu": enable_gpu},
        "services": {},
    }

    # ── YOLO ─────────────────────────────────────────────────────────
    print(f"[2/4] Benchmarking YOLOService ({n_frames} frames)...")
    yolo_times: list[float] = []
    for frame in frames:
        with timed() as t:
            YOLOService.detect(frame)
        yolo_times.append(t.elapsed_ms)
    yolo_stats = stats(yolo_times)
    results["services"]["yolo"] = yolo_stats
    print(f"      mean={yolo_stats['mean']:.1f}ms  p95={yolo_stats['p95']:.1f}ms  "
          f"p99={yolo_stats['p99']:.1f}ms  max={yolo_stats['max']:.1f}ms")

    # ── Headset ───────────────────────────────────────────────────────
    print(f"[3/4] Benchmarking HeadsetService ({n_frames} frames)...")
    headset_times: list[float] = []
    for frame in frames:
        with timed() as t:
            HeadsetService.classify(frame)
        headset_times.append(t.elapsed_ms)
    headset_stats = stats(headset_times)
    results["services"]["headset"] = headset_stats
    print(f"      mean={headset_stats['mean']:.1f}ms  p95={headset_stats['p95']:.1f}ms  "
          f"p99={headset_stats['p99']:.1f}ms  max={headset_stats['max']:.1f}ms")

    # ── Proctor ───────────────────────────────────────────────────────
    print(f"[4/4] Benchmarking ProctorService ({n_frames} frames)...")
    proctor_times: list[float] = []
    for frame in frames:
        with timed() as t:
            ProctorService.analyze(frame)
        proctor_times.append(t.elapsed_ms)
    proctor_stats = stats(proctor_times)
    results["services"]["proctor"] = proctor_stats
    print(f"      mean={proctor_stats['mean']:.1f}ms  p95={proctor_stats['p95']:.1f}ms  "
          f"p99={proctor_stats['p99']:.1f}ms  max={proctor_stats['max']:.1f}ms")

    # ── Combined pipeline ─────────────────────────────────────────────
    print(f"\n[+]  Combined pipeline (YOLO + Headset + Proctor)...")
    pipeline_times: list[float] = []
    for frame in frames:
        with timed() as t:
            YOLOService.detect(frame)
            HeadsetService.classify(frame)
            ProctorService.analyze(frame)
        pipeline_times.append(t.elapsed_ms)
    pipeline_stats = stats(pipeline_times)
    results["services"]["pipeline_combined"] = pipeline_stats
    print(f"      mean={pipeline_stats['mean']:.1f}ms  p95={pipeline_stats['p95']:.1f}ms  "
          f"p99={pipeline_stats['p99']:.1f}ms  max={pipeline_stats['max']:.1f}ms")

    # ── GPU memory ────────────────────────────────────────────────────
    alloc = gpu_memory_mb()
    reserved = gpu_reserved_mb()
    if alloc is not None:
        results["gpu_memory"] = {"allocated_mb": alloc, "reserved_mb": reserved}
        print(f"\n[GPU] allocated={alloc:.1f}MB  reserved={reserved:.1f}MB")

    # ── SLA check ─────────────────────────────────────────────────────
    sla_ms = 500.0
    p95_combined = pipeline_stats["p95"]
    sla_pass = p95_combined < sla_ms
    results["sla"] = {
        "target_ms": sla_ms,
        "p95_combined_ms": p95_combined,
        "pass": sla_pass,
    }
    status = "✓ PASS" if sla_pass else "✗ FAIL"
    print(f"\n[SLA] p95 combined {p95_combined:.1f}ms < {sla_ms}ms → {status}")

    save_json(results, "pt_latency.json")
    print("\nDone.\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="PT latency benchmark")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames per service")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU inference")
    args = parser.parse_args()
    run(n_frames=args.frames, enable_gpu=args.gpu)


if __name__ == "__main__":
    main()
