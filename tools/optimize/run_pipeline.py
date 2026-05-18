"""
Master Optimization Pipeline
==============================
Runs all optimization steps in the correct order:

  Step 1: Export PT → ONNX  (export_pt_to_onnx.py)
  Step 2: Quantize STT ONNX → INT8  (quantize_stt.py)
  Step 3: Quantize YOLO ONNX → INT8  (quantize_yolo.py)
  Step 4: Validate all optimized models  (validate_onnx.py)
  Step 5: Benchmark original vs optimized  (benchmark_optimized.py)

Usage:
    # Full pipeline
    python tools/optimize/run_pipeline.py

    # Skip PT export (if already done)
    python tools/optimize/run_pipeline.py --skip-export

    # Skip benchmarking (faster)
    python tools/optimize/run_pipeline.py --skip-benchmark

    # With GPU
    python tools/optimize/run_pipeline.py --gpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add tools/optimize to path so relative imports work
sys.path.insert(0, str(Path(__file__).parent))

from config import LOG_FORMAT, BENCHMARK_RESULTS

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("run_pipeline")


def run(
    skip_export: bool    = False,
    skip_benchmark: bool = False,
    enable_gpu: bool     = False,
    bench_runs: int      = 50,
) -> dict:
    """Execute the full optimization pipeline.

    Returns:
        Dict with results from each step.
    """
    pipeline_start = time.perf_counter()
    report = {}

    logger.info("\n" + "=" * 65)
    logger.info("  MODEL OPTIMIZATION PIPELINE")
    logger.info("=" * 65)

    # ── Step 1: PT → ONNX export ──────────────────────────────────────
    if not skip_export:
        logger.info("\n[STEP 1/5] Exporting PT models to ONNX...")
        import export_pt_to_onnx
        report["export"] = export_pt_to_onnx.run()
    else:
        logger.info("\n[STEP 1/5] Skipping PT export (--skip-export)")

    # ── Step 2: STT quantization ──────────────────────────────────────
    logger.info("\n[STEP 2/5] Quantizing STT ONNX models to INT8...")
    import quantize_stt
    report["stt_quantization"] = quantize_stt.run()

    # ── Step 3: YOLO quantization ─────────────────────────────────────
    logger.info("\n[STEP 3/5] Quantizing YOLO ONNX models to INT8...")
    import quantize_yolo
    report["yolo_quantization"] = quantize_yolo.run()

    # ── Step 4: Validation ────────────────────────────────────────────
    logger.info("\n[STEP 4/5] Validating all optimized models...")
    import validate_onnx
    report["validation"] = validate_onnx.run(enable_gpu=enable_gpu)

    # ── Step 5: Benchmark ─────────────────────────────────────────────
    if not skip_benchmark:
        logger.info(f"\n[STEP 5/5] Benchmarking ({bench_runs} runs per model)...")
        import benchmark_optimized
        report["benchmark"] = benchmark_optimized.run(n_runs=bench_runs, enable_gpu=enable_gpu)
    else:
        logger.info("\n[STEP 5/5] Skipping benchmark (--skip-benchmark)")

    # ── Final summary ─────────────────────────────────────────────────
    elapsed = round(time.perf_counter() - pipeline_start, 1)
    report["total_elapsed_s"] = elapsed

    val_results = report.get("validation", [])
    passed = sum(1 for r in val_results if r.get("inference_ok"))
    total  = len(val_results)

    logger.info("\n" + "=" * 65)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  Total time      : {elapsed}s")
    logger.info(f"  Validation      : {passed}/{total} models passed")

    # Save full pipeline report
    report_path = BENCHMARK_RESULTS / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"  Full report     : {report_path}")
    logger.info("=" * 65 + "\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full model optimization pipeline")
    parser.add_argument("--skip-export",    action="store_true", help="Skip PT→ONNX export")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip latency benchmarking")
    parser.add_argument("--gpu",            action="store_true", help="Use CUDA if available")
    parser.add_argument("--bench-runs",     type=int, default=50, help="Benchmark runs per model")
    args = parser.parse_args()

    run(
        skip_export=args.skip_export,
        skip_benchmark=args.skip_benchmark,
        enable_gpu=args.gpu,
        bench_runs=args.bench_runs,
    )


if __name__ == "__main__":
    main()
