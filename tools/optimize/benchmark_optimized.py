"""
Optimization Benchmark
=======================
Compares inference latency and model size between:
  - Original ONNX / PT models
  - Optimized INT8 ONNX models

Produces a JSON report at benchmark/outputs/optimization/benchmark_report.json

Usage:
    python tools/optimize/benchmark_optimized.py [--runs 50] [--gpu]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    LOG_FORMAT,
    STT_ASSETS,
    STT_OPTIMIZED_DIR,
    YOLO_ONNX_DIR,
    YOLO_INT8_DIR,
    BENCHMARK_RESULTS,
    get_providers,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("benchmark_optimized")


def _size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return round(path.stat().st_size / (1024 ** 2), 2)


def _make_session(onnx_path: Path, providers: list):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    return ort.InferenceSession(str(onnx_path), opts, providers=providers)


def _bench_session(session, dummy_inputs: dict, n_runs: int) -> dict:
    """Run n_runs inferences and return latency statistics."""
    # Warm-up
    for _ in range(3):
        session.run(None, dummy_inputs)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, dummy_inputs)
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    return {
        "runs": n_runs,
        "mean_ms":   round(float(arr.mean()), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms":    round(float(np.percentile(arr, 95)), 2),
        "p99_ms":    round(float(np.percentile(arr, 99)), 2),
        "min_ms":    round(float(arr.min()), 2),
        "max_ms":    round(float(arr.max()), 2),
    }


def bench_pair(name: str, original: Path, optimized: Path,
               providers: list, dummy_inputs: dict, n_runs: int) -> dict:
    """Benchmark an original vs optimized model pair.

    Returns:
        Dict with size and latency comparison.
    """
    result = {
        "model": name,
        "original": {"path": str(original), "size_mb": _size_mb(original)},
        "optimized": {"path": str(optimized), "size_mb": _size_mb(optimized)},
    }

    # Size reduction
    if result["original"]["size_mb"] > 0 and result["optimized"]["size_mb"] > 0:
        reduction = round(
            (1 - result["optimized"]["size_mb"] / result["original"]["size_mb"]) * 100, 1
        )
        result["size_reduction_pct"] = reduction
    else:
        result["size_reduction_pct"] = None

    # Benchmark original
    if original.exists():
        try:
            sess = _make_session(original, providers)
            result["original"]["latency"] = _bench_session(sess, dummy_inputs, n_runs)
            logger.info(
                f"[{name}] Original  mean={result['original']['latency']['mean_ms']}ms  "
                f"p95={result['original']['latency']['p95_ms']}ms"
            )
        except Exception as exc:
            logger.error(f"[{name}] Original benchmark failed: {exc}")
            result["original"]["latency"] = None

    # Benchmark optimized
    if optimized.exists():
        try:
            sess = _make_session(optimized, providers)
            result["optimized"]["latency"] = _bench_session(sess, dummy_inputs, n_runs)
            logger.info(
                f"[{name}] Optimized mean={result['optimized']['latency']['mean_ms']}ms  "
                f"p95={result['optimized']['latency']['p95_ms']}ms"
            )
        except Exception as exc:
            logger.error(f"[{name}] Optimized benchmark failed: {exc}")
            result["optimized"]["latency"] = None

    # Speedup
    orig_lat  = result.get("original",  {}).get("latency") or {}
    opt_lat   = result.get("optimized", {}).get("latency") or {}
    if orig_lat.get("mean_ms") and opt_lat.get("mean_ms"):
        speedup = round(orig_lat["mean_ms"] / opt_lat["mean_ms"], 2)
        result["speedup_x"] = speedup
        logger.info(f"[{name}] Speedup: {speedup}x")

    return result


def run(n_runs: int = 50, enable_gpu: bool = False) -> dict:
    """Run the full optimization benchmark.

    Returns:
        Full benchmark report dict.
    """
    providers = get_providers(enable_gpu)
    report    = {
        "config": {"n_runs": n_runs, "enable_gpu": enable_gpu, "providers": str(providers)},
        "stt": [],
        "yolo": [],
    }

    logger.info("=" * 60)
    logger.info(f"Optimization Benchmark  |  runs={n_runs}  gpu={enable_gpu}")
    logger.info("=" * 60)

    # ── STT models ────────────────────────────────────────────────────
    logger.info("\n--- STT Models ---")
    stt_pairs = [
        ("encoder",      STT_ASSETS / "encoder.onnx",      STT_OPTIMIZED_DIR / "encoder_int8.onnx"),
        ("rnnt_decoder", STT_ASSETS / "rnnt_decoder.onnx",  STT_OPTIMIZED_DIR / "rnnt_decoder_int8.onnx"),
        ("joint_enc",    STT_ASSETS / "joint_enc.onnx",     STT_OPTIMIZED_DIR / "joint_enc_int8.onnx"),
        ("joint_pred",   STT_ASSETS / "joint_pred.onnx",    STT_OPTIMIZED_DIR / "joint_pred_int8.onnx"),
    ]
    for name, orig, opt in stt_pairs:
        if not orig.exists():
            logger.warning(f"[{name}] Original not found, skipping")
            continue
        # Generate dummy input from model metadata
        try:
            import onnxruntime as ort
            sess_tmp = ort.InferenceSession(str(orig), providers=["CPUExecutionProvider"])
            dummy = {}
            for inp in sess_tmp.get_inputs():
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                dummy[inp.name] = np.random.randn(*shape).astype(np.float32)
        except Exception:
            dummy = {}

        result = bench_pair(name, orig, opt, providers, dummy, n_runs)
        report["stt"].append(result)

    # ── YOLO models ───────────────────────────────────────────────────
    logger.info("\n--- YOLO Models ---")
    yolo_dummy = {"images": np.random.randn(1, 3, 640, 640).astype(np.float32)}
    yolo_pairs = [
        ("yolov8n",       YOLO_ONNX_DIR / "yolov8n.onnx",       YOLO_INT8_DIR / "yolov8n_int8.onnx"),
        ("proctoring",    YOLO_ONNX_DIR / "proctoring.onnx",     YOLO_INT8_DIR / "proctoring_int8.onnx"),
        ("headset_model", YOLO_ONNX_DIR / "headset_model.onnx",  YOLO_INT8_DIR / "headset_model_int8.onnx"),
    ]
    for name, orig, opt in yolo_pairs:
        if not orig.exists():
            logger.warning(f"[{name}] Original ONNX not found — run export_pt_to_onnx.py first")
            continue
        result = bench_pair(name, orig, opt, providers, yolo_dummy, n_runs)
        report["yolo"].append(result)

    # ── Print summary table ───────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Model':20s} {'Orig MB':>8} {'Opt MB':>8} {'Size↓':>7} {'Orig ms':>9} {'Opt ms':>9} {'Speedup':>8}")
    logger.info("-" * 70)
    for section in ["stt", "yolo"]:
        for r in report[section]:
            orig_mb  = r["original"]["size_mb"]
            opt_mb   = r["optimized"]["size_mb"]
            size_red = f"{r.get('size_reduction_pct', 'N/A')}%"
            orig_ms  = r.get("original",  {}).get("latency", {}) or {}
            opt_ms   = r.get("optimized", {}).get("latency", {}) or {}
            speedup  = r.get("speedup_x", "N/A")
            logger.info(
                f"  {r['model']:20s} {orig_mb:>8.1f} {opt_mb:>8.1f} {size_red:>7} "
                f"{orig_ms.get('mean_ms', 'N/A'):>9} {opt_ms.get('mean_ms', 'N/A'):>9} "
                f"{speedup:>8}"
            )
    logger.info("=" * 70)

    # ── Save report ───────────────────────────────────────────────────
    report_path = BENCHMARK_RESULTS / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nReport saved → {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark original vs optimized models")
    parser.add_argument("--runs", type=int, default=50, help="Inference runs per model")
    parser.add_argument("--gpu",  action="store_true",  help="Use CUDA if available")
    args = parser.parse_args()
    run(n_runs=args.runs, enable_gpu=args.gpu)


if __name__ == "__main__":
    main()
