"""
benchmark/run_benchmark.py
Main benchmark orchestrator — runs all benchmarks and generates the final report.

Usage:
    # Full benchmark (requires server running on port 8000)
    python benchmark/run_benchmark.py

    # Quick benchmark (fewer iterations)
    python benchmark/run_benchmark.py --quick

    # Specific benchmarks only
    python benchmark/run_benchmark.py --onnx-only
    python benchmark/run_benchmark.py --queue-only
    python benchmark/run_benchmark.py --ws-only

    # Custom student counts for WebSocket test
    python benchmark/run_benchmark.py --students 1 5 10 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from benchmark.datasets.generator import generate_dataset, DATASETS_DIR
from benchmark.metrics.collector import BenchmarkMetrics, MetricsCollector
from benchmark.runners.onnx_benchmark import run_all_models_benchmark
from benchmark.runners.audio_benchmark import run_audio_benchmark
from benchmark.stress.queue_stress_test import run_all_queue_tests
from benchmark.stress.websocket_stream_simulator import run_scalability_test
from benchmark.visualizations.generate_graphs import generate_all_charts
from benchmark.reports.report_generator import generate_all_reports

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("benchmark.main")

OUTPUT_DIR = Path(__file__).parent / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description="AI Proctoring Backend Benchmark")
    p.add_argument("--quick", action="store_true", help="Quick mode (fewer iterations)")
    p.add_argument("--onnx-only", action="store_true")
    p.add_argument("--audio-only", action="store_true")
    p.add_argument("--queue-only", action="store_true")
    p.add_argument("--ws-only", action="store_true")
    p.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    p.add_argument("--students", nargs="+", type=int, default=[1, 5, 10],
                   help="Student counts for WebSocket scalability test")
    p.add_argument("--ws-duration", type=float, default=15.0,
                   help="Duration per WebSocket test (seconds)")
    p.add_argument("--backend-url", default="ws://localhost:8000",
                   help="Backend WebSocket URL")
    return p.parse_args()


def run_full_benchmark(args) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = {}
    t_total = time.time()

    # ── 0. Generate test dataset ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 0: Generating test dataset")
    logger.info("=" * 60)
    dataset = generate_dataset()
    logger.info(f"Dataset ready: {len(dataset['frames'])} frame types, {len(dataset['audios'])} audio types")

    run_all = not any([args.onnx_only, args.audio_only, args.queue_only, args.ws_only])

    # ── 1. ONNX Benchmark ─────────────────────────────────────────────
    if run_all or args.onnx_only:
        logger.info("=" * 60)
        logger.info("STEP 1: ONNX Inference Benchmark")
        logger.info("=" * 60)
        warmup = 10 if args.quick else 50
        runs = 50 if args.quick else 500
        stream_dur = 10.0 if args.quick else 30.0

        onnx_results = run_all_models_benchmark(
            warmup_runs=warmup,
            benchmark_runs=runs,
            stream_duration_s=stream_dur,
        )
        combined["onnx"] = onnx_results
        (OUTPUT_DIR / "onnx_benchmark.json").write_text(json.dumps(onnx_results, indent=2))
        logger.info("ONNX benchmark complete")

    # ── 2. Audio Pipeline Benchmark ───────────────────────────────────
    if run_all or args.audio_only:
        logger.info("=" * 60)
        logger.info("STEP 2: Audio Pipeline Benchmark")
        logger.info("=" * 60)
        audio_results = run_audio_benchmark(audio_dir=DATASETS_DIR / "audio")
        combined["audio"] = audio_results
        (OUTPUT_DIR / "audio_benchmark.json").write_text(json.dumps(audio_results, indent=2))
        logger.info("Audio benchmark complete")

    # ── 3. Queue Stress Test ──────────────────────────────────────────
    if run_all or args.queue_only:
        logger.info("=" * 60)
        logger.info("STEP 3: Queue Stress Test")
        logger.info("=" * 60)
        queue_results = run_all_queue_tests()
        combined["queue"] = queue_results
        (OUTPUT_DIR / "queue_stress.json").write_text(json.dumps(queue_results, indent=2))
        logger.info("Queue stress test complete")

    # ── 4. WebSocket Scalability Test ─────────────────────────────────
    if run_all or args.ws_only:
        logger.info("=" * 60)
        logger.info("STEP 4: WebSocket Scalability Test")
        logger.info("=" * 60)
        logger.info(f"Testing student counts: {args.students}")
        logger.info("NOTE: Requires backend running at " + args.backend_url)

        try:
            ws_results = run_scalability_test(
                base_url=args.backend_url,
                student_counts=args.students,
                duration_s=args.ws_duration,
            )
            combined["websocket"] = ws_results
            (OUTPUT_DIR / "websocket_simulation.json").write_text(json.dumps(ws_results, indent=2))
            logger.info("WebSocket scalability test complete")
        except Exception as e:
            logger.warning(f"WebSocket test failed (is the backend running?): {e}")
            combined["websocket"] = {"error": str(e)}

    # ── 5. Generate Charts ────────────────────────────────────────────
    chart_paths = []
    if not args.no_charts:
        logger.info("=" * 60)
        logger.info("STEP 5: Generating Charts")
        logger.info("=" * 60)
        try:
            chart_paths = generate_all_charts(combined)
            logger.info(f"Generated {len(chart_paths)} charts")
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")

    # ── 6. Generate Reports ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Generating Reports")
    logger.info("=" * 60)
    saved = generate_all_reports(combined, chart_paths)

    total_elapsed = round(time.time() - t_total, 1)

    logger.info("=" * 60)
    logger.info(f"BENCHMARK COMPLETE in {total_elapsed}s")
    logger.info("=" * 60)
    for fmt, path in saved.items():
        logger.info(f"  {fmt.upper()}: {path}")
    if chart_paths:
        logger.info(f"  CHARTS: {len(chart_paths)} files in benchmark/outputs/charts/")

    return combined


if __name__ == "__main__":
    args = parse_args()
    results = run_full_benchmark(args)

    # Print quick summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    onnx = results.get("onnx", {})
    for model, data in onnx.items():
        stream = data.get("stream_simulation", {})
        print(f"  {model}: {stream.get('mean_ms', 0):.0f}ms avg, "
              f"{stream.get('actual_fps', 0):.2f} FPS, "
              f"{stream.get('drop_rate_pct', 0):.1f}% drops")

    queue = results.get("queue", {})
    tput = queue.get("throughput", {})
    if tput:
        print(f"  Queue: {tput.get('throughput_per_sec', 0):.0f} tasks/s, "
              f"wait={tput.get('wait_time_ms', {}).get('mean', 0):.1f}ms")

    ws = results.get("websocket", {})
    for key, data in ws.items():
        if isinstance(data, dict) and "num_students" in data:
            n = data["num_students"]
            c = data.get("connected", 0)
            print(f"  WebSocket {n} students: {c}/{n} connected, "
                  f"{data.get('total_events_received', 0)} events")

    print("=" * 60)
    print(f"Reports saved to: {OUTPUT_DIR.parent / 'reports' / 'generated'}")
    print(f"Metrics saved to: {OUTPUT_DIR}")
