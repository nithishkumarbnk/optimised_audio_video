"""
Master runner — executes all PT benchmark scripts in sequence and writes a
consolidated summary report to ``benchmark/outputs/pt_benchmark_report.json``.

Usage::

    python -m benchmark.pt_benchmark.run_all [--gpu] [--frames N] [--students N]

Runs:
  1. output_validation  — correctness checks
  2. latency_benchmark  — per-service latency statistics
  3. fps_benchmark      — sustained FPS over 10 seconds
  4. concurrent_stress  — N concurrent students
  5. gpu_utilization    — GPU/CPU utilisation sampling
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.pt_benchmark.bench_utils import save_json
import benchmark.pt_benchmark.output_validation as _val
import benchmark.pt_benchmark.latency_benchmark  as _lat
import benchmark.pt_benchmark.fps_benchmark       as _fps
import benchmark.pt_benchmark.concurrent_stress   as _stress
import benchmark.pt_benchmark.gpu_utilization     as _gpu


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all PT benchmarks")
    parser.add_argument("--gpu",      action="store_true", help="Enable GPU inference")
    parser.add_argument("--frames",   type=int, default=30, help="Frames for latency/validation")
    parser.add_argument("--students", type=int, default=8,  help="Concurrent students for stress test")
    args = parser.parse_args()

    wall_start = time.perf_counter()
    report: dict = {"config": vars(args), "results": {}}

    print("\n" + "="*60)
    print("  PT BENCHMARK SUITE — FULL RUN")
    print("="*60)

    # 1. Output validation
    print("\n[1/5] Output Validation")
    report["results"]["validation"] = _val.run(
        n_frames=args.frames, enable_gpu=args.gpu
    )

    # 2. Latency
    print("\n[2/5] Latency Benchmark")
    report["results"]["latency"] = _lat.run(
        n_frames=args.frames, enable_gpu=args.gpu
    )

    # 3. FPS
    print("\n[3/5] FPS Benchmark")
    report["results"]["fps"] = _fps.run(
        duration_s=10.0, enable_gpu=args.gpu
    )

    # 4. Concurrent stress
    print("\n[4/5] Concurrent Stress Test")
    report["results"]["stress"] = _stress.run(
        n_students=args.students,
        frames_per_student=args.frames,
        enable_gpu=args.gpu,
    )

    # 5. GPU/CPU utilisation
    print("\n[5/5] GPU/CPU Utilisation")
    report["results"]["utilization"] = _gpu.run(
        n_frames=args.frames, enable_gpu=args.gpu
    )

    wall_elapsed = time.perf_counter() - wall_start
    report["total_elapsed_s"] = round(wall_elapsed, 2)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    val_pass = report["results"]["validation"].get("passed", False)
    sla_pass = report["results"]["latency"].get("sla", {}).get("pass", False)
    fps_val  = report["results"]["fps"].get("fps", 0)
    errors   = report["results"]["stress"].get("summary", {}).get("errors", -1)

    print(f"  Validation   : {'✓ PASS' if val_pass else '✗ FAIL'}")
    print(f"  Latency SLA  : {'✓ PASS' if sla_pass else '✗ FAIL'}")
    print(f"  Sustained FPS: {fps_val:.1f}")
    print(f"  Stress errors: {errors}")
    print(f"  Total time   : {wall_elapsed:.1f}s")
    print("="*60 + "\n")

    save_json(report, "pt_benchmark_report.json")
    print("Full report saved to benchmark/outputs/pt_benchmark_report.json\n")


if __name__ == "__main__":
    main()
