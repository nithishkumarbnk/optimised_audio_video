"""
benchmark/reports/report_generator.py
Generates the final benchmark report in Markdown, CSV, and JSON formats.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parents[1] / "outputs"
REPORTS_DIR = Path(__file__).parents[1] / "reports" / "generated"


def _system_info() -> Dict:
    cpu = platform.processor() or platform.machine()
    mem = psutil.virtual_memory()
    return {
        "os": platform.system(),
        "python": platform.python_version(),
        "cpu": cpu,
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(mem.total / 1024**3, 1),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def _grade(value: float, thresholds: Dict[str, float]) -> str:
    """Return a grade emoji based on thresholds."""
    if value <= thresholds.get("excellent", float("inf")):
        return "🟢 Excellent"
    if value <= thresholds.get("good", float("inf")):
        return "🟡 Good"
    if value <= thresholds.get("acceptable", float("inf")):
        return "🟠 Acceptable"
    return "🔴 Poor"


def generate_markdown_report(benchmark_data: Dict, chart_paths: List[str] = None) -> str:
    """Generate a comprehensive Markdown benchmark report."""
    sys_info = _system_info()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    chart_paths = chart_paths or []

    onnx = benchmark_data.get("onnx", {})
    audio = benchmark_data.get("audio", {})
    queue = benchmark_data.get("queue", {})
    ws = benchmark_data.get("websocket", {})

    lines = [
        "# AI Proctoring Backend — Benchmark Report",
        f"\n**Generated:** {now}",
        f"**System:** {sys_info['cpu']} | {sys_info['cpu_cores']} cores / {sys_info['cpu_threads']} threads | {sys_info['ram_total_gb']} GB RAM",
        f"**OS:** {sys_info['os']} | Python {sys_info['python']}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]

    # Determine production readiness
    issues = []
    strengths = []

    # Check ONNX performance
    for model_name, model_data in onnx.items():
        stream = model_data.get("stream_simulation", {})
        mean_ms = stream.get("mean_ms", 0)
        drop_rate = stream.get("drop_rate_pct", 0)
        if mean_ms > 500:
            issues.append(f"{model_name} inference too slow ({mean_ms:.0f}ms > 500ms target)")
        else:
            strengths.append(f"{model_name} inference within target ({mean_ms:.0f}ms)")
        if drop_rate > 5:
            issues.append(f"{model_name} high frame drop rate ({drop_rate:.1f}%)")

    # Check queue performance
    bp = queue.get("backpressure", {})
    if bp.get("worker_saturation"):
        issues.append("Worker saturation detected under load")
    else:
        strengths.append("Worker pool handles load without saturation")

    # Check WebSocket scalability
    for key, data in ws.items():
        n = data.get("num_students", 0)
        connected_pct = data.get("connected", 0) / max(n, 1) * 100
        if connected_pct < 90 and n >= 10:
            issues.append(f"WebSocket connection rate {connected_pct:.0f}% at {n} students")
        elif n >= 10:
            strengths.append(f"WebSocket handles {n} concurrent students ({connected_pct:.0f}% connected)")

    if not issues:
        lines.append("✅ **System is PRODUCTION READY** — all benchmarks within acceptable thresholds.")
    elif len(issues) <= 2:
        lines.append("⚠️ **System is CONDITIONALLY READY** — minor issues detected.")
    else:
        lines.append("❌ **System requires optimization** before production deployment.")

    lines.append("")
    if strengths:
        lines.append("**Strengths:**")
        for s in strengths:
            lines.append(f"- ✅ {s}")
        lines.append("")
    if issues:
        lines.append("**Issues:**")
        for i in issues:
            lines.append(f"- ⚠️ {i}")
        lines.append("")

    # ONNX Section
    lines += [
        "---",
        "",
        "## 1. ONNX Inference Benchmark",
        "",
        "| Model | Mean (ms) | P95 (ms) | P99 (ms) | FPS | Grade |",
        "|-------|-----------|----------|----------|-----|-------|",
    ]
    for model_name, model_data in onnx.items():
        frame_results = model_data.get("per_frame_type", [])
        if frame_results:
            means = [r.get("mean_ms", 0) for r in frame_results]
            p95s = [r.get("p95_ms", 0) for r in frame_results]
            p99s = [r.get("p99_ms", 0) for r in frame_results]
            fpss = [r.get("throughput_fps", 0) for r in frame_results]
            mean = sum(means) / len(means)
            p95 = sum(p95s) / len(p95s)
            p99 = sum(p99s) / len(p99s)
            fps = sum(fpss) / len(fpss)
            grade = _grade(mean, {"excellent": 100, "good": 300, "acceptable": 500})
            lines.append(f"| {model_name} | {mean:.1f} | {p95:.1f} | {p99:.1f} | {fps:.1f} | {grade} |")

    lines += ["", "### Stream Simulation (1 FPS, 30s)", ""]
    lines += [
        "| Model | Actual FPS | Frames | Dropped | Drop Rate |",
        "|-------|-----------|--------|---------|-----------|",
    ]
    for model_name, model_data in onnx.items():
        stream = model_data.get("stream_simulation", {})
        lines.append(
            f"| {model_name} | {stream.get('actual_fps', 0):.2f} | "
            f"{stream.get('frames_processed', 0)} | {stream.get('frames_dropped', 0)} | "
            f"{stream.get('drop_rate_pct', 0):.1f}% |"
        )

    # Audio Section
    lines += [
        "",
        "---",
        "",
        "## 2. Audio Pipeline Benchmark",
        "",
        "| Stage | Mean (ms) | P95 (ms) | Throughput/s | Grade |",
        "|-------|-----------|----------|--------------|-------|",
    ]
    for stage, data in audio.items():
        if isinstance(data, dict) and "mean_ms" in data:
            mean = data.get("mean_ms", 0)
            p95 = data.get("p95_ms", 0)
            tput = data.get("throughput_per_sec", data.get("throughput_fps", "N/A"))
            grade = _grade(mean, {"excellent": 10, "good": 100, "acceptable": 500})
            lines.append(f"| {stage} | {mean} | {p95} | {tput} | {grade} |")

    stt = audio.get("stt", {})
    if stt and "mean_ms" in stt:
        lines += [
            "",
            f"**STT Transcription:** Mean={stt['mean_ms']}ms | "
            f"Sample: *\"{stt.get('sample_transcript', 'N/A')}\"*",
        ]

    # Queue Section
    lines += [
        "",
        "---",
        "",
        "## 3. Queue Stress Test",
        "",
    ]
    tput_data = queue.get("throughput", {})
    if tput_data:
        lines += [
            f"- **Throughput:** {tput_data.get('throughput_per_sec', 0):.0f} tasks/sec",
            f"- **Avg Wait Time:** {tput_data.get('wait_time_ms', {}).get('mean', 0):.2f}ms",
            f"- **P95 Wait Time:** {tput_data.get('wait_time_ms', {}).get('p95', 0):.2f}ms",
            f"- **Max Queue Depth:** {tput_data.get('max_queue_depth', 0)}",
            f"- **Worker Utilization:** {tput_data.get('avg_worker_busy_pct', 0):.1f}%",
        ]

    bp_data = queue.get("backpressure", {})
    if bp_data:
        lines += [
            "",
            f"**Backpressure Test:** Producer={bp_data.get('producer_rate_per_sec', 0)}/s, "
            f"Workers={bp_data.get('num_workers', 0)}, "
            f"Saturation={'⚠️ YES' if bp_data.get('worker_saturation') else '✅ NO'}",
        ]

    fd_data = queue.get("frame_drop", {})
    if fd_data:
        lines += [
            f"**Frame Drop Test:** Burst={fd_data.get('burst_size', 0)}, "
            f"Drop Rate={fd_data.get('drop_rate_pct', 0):.1f}% (throttle-based)",
        ]

    # WebSocket Section
    lines += [
        "",
        "---",
        "",
        "## 4. WebSocket Scalability",
        "",
        "| Students | Connected | Frames Sent | Events Received | P95 Latency | Drop Rate |",
        "|----------|-----------|-------------|-----------------|-------------|-----------|",
    ]
    for key, data in ws.items():
        n = data.get("num_students", 0)
        connected = data.get("connected", 0)
        frames = data.get("total_frames_sent", 0)
        events = data.get("total_events_received", 0)
        lat = data.get("event_latency_ms", {})
        p95 = lat.get("p95", "N/A") if isinstance(lat, dict) else "N/A"
        drop = data.get("drop_rate_pct", 0)
        lines.append(f"| {n} | {connected}/{n} | {frames} | {events} | {p95}ms | {drop:.1f}% |")

    # Charts Section
    if chart_paths:
        lines += [
            "",
            "---",
            "",
            "## 5. Charts",
            "",
        ]
        for path in chart_paths:
            name = Path(path).stem.replace("_", " ").title()
            rel = Path(path).name
            lines.append(f"![{name}](charts/{rel})")
            lines.append("")

    # Recommendations
    lines += [
        "",
        "---",
        "",
        "## 6. Recommendations",
        "",
    ]

    recs = [
        ("GPU Acceleration", "Enable CUDA (ENABLE_GPU=true) to reduce ONNX inference from ~2s to ~50ms per frame."),
        ("Worker Pool Size", "Increase WORKER_POOL_SIZE to match CPU core count for better throughput."),
        ("Audio Chunking", "Send 3-4s audio chunks every 4s to balance STT latency vs responsiveness."),
        ("Frame Rate", "1 FPS is optimal for CPU inference. Do not increase without GPU."),
        ("Horizontal Scaling", "For 50+ concurrent users, deploy multiple containers behind a load balancer."),
        ("STT Language", "Limit to 1-2 languages (en + primary language) to reduce STT latency from 30s to 2-3s."),
    ]
    for title, rec in recs:
        lines.append(f"- **{title}:** {rec}")

    lines += [
        "",
        "---",
        "",
        f"*Report generated by AI Proctoring Benchmark Framework — {now}*",
    ]

    return "\n".join(lines)


def generate_csv_metrics(benchmark_data: Dict) -> str:
    """Generate CSV with all benchmark metrics."""
    rows = []

    # ONNX metrics
    for model_name, model_data in benchmark_data.get("onnx", {}).items():
        for r in model_data.get("per_frame_type", []):
            rows.append({
                "category": "onnx_inference",
                "model": model_name,
                "frame_type": r.get("frame_type", ""),
                "mean_ms": r.get("mean_ms", 0),
                "p95_ms": r.get("p95_ms", 0),
                "p99_ms": r.get("p99_ms", 0),
                "min_ms": r.get("min_ms", 0),
                "max_ms": r.get("max_ms", 0),
                "throughput_fps": r.get("throughput_fps", 0),
                "runs": r.get("runs", 0),
            })

    # Audio metrics
    for stage, data in benchmark_data.get("audio", {}).items():
        if isinstance(data, dict) and "mean_ms" in data:
            rows.append({
                "category": "audio_pipeline",
                "stage": stage,
                "mean_ms": data.get("mean_ms", 0),
                "p95_ms": data.get("p95_ms", 0),
                "throughput_per_sec": data.get("throughput_per_sec", 0),
                "runs": data.get("runs", 0),
            })

    # WebSocket metrics
    for key, data in benchmark_data.get("websocket", {}).items():
        rows.append({
            "category": "websocket_scalability",
            "num_students": data.get("num_students", 0),
            "connected": data.get("connected", 0),
            "frames_sent": data.get("total_frames_sent", 0),
            "events_received": data.get("total_events_received", 0),
            "drop_rate_pct": data.get("drop_rate_pct", 0),
        })

    if not rows:
        return "category,metric,value\n"

    # Get all keys
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)

    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    return buf.getvalue()


def generate_summary_json(benchmark_data: Dict) -> Dict:
    """Generate a concise benchmark summary JSON."""
    sys_info = _system_info()

    # Find best/worst ONNX model
    onnx_summary = {}
    for model_name, model_data in benchmark_data.get("onnx", {}).items():
        stream = model_data.get("stream_simulation", {})
        onnx_summary[model_name] = {
            "mean_inference_ms": stream.get("mean_ms", 0),
            "p95_inference_ms": stream.get("p95_ms", 0),
            "actual_fps": stream.get("actual_fps", 0),
            "drop_rate_pct": stream.get("drop_rate_pct", 0),
        }

    # WebSocket max stable users
    max_stable = 0
    for key, data in benchmark_data.get("websocket", {}).items():
        n = data.get("num_students", 0)
        connected_pct = data.get("connected", 0) / max(n, 1) * 100
        if connected_pct >= 90:
            max_stable = max(max_stable, n)

    return {
        "generated_at": sys_info["timestamp"],
        "system": sys_info,
        "onnx_models": onnx_summary,
        "audio_pipeline": {
            "preprocessing_mean_ms": benchmark_data.get("audio", {}).get("preprocessing", {}).get("mean_ms", 0),
            "rule_engine_throughput_per_sec": benchmark_data.get("audio", {}).get("rule_engine", {}).get("throughput_per_sec", 0),
            "risk_engine_throughput_per_sec": benchmark_data.get("audio", {}).get("risk_engine", {}).get("throughput_per_sec", 0),
        },
        "queue": {
            "throughput_per_sec": benchmark_data.get("queue", {}).get("throughput", {}).get("throughput_per_sec", 0),
            "avg_wait_ms": benchmark_data.get("queue", {}).get("throughput", {}).get("wait_time_ms", {}).get("mean", 0),
            "worker_saturation_detected": benchmark_data.get("queue", {}).get("backpressure", {}).get("worker_saturation", False),
        },
        "websocket": {
            "max_stable_concurrent_users": max_stable,
        },
        "production_readiness": "READY" if max_stable >= 10 else "NEEDS_OPTIMIZATION",
    }


def generate_all_reports(benchmark_data: Dict, chart_paths: List[str] = None) -> Dict[str, str]:
    """Generate all report formats and save to disk."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}

    # Markdown report
    md = generate_markdown_report(benchmark_data, chart_paths)
    md_path = REPORTS_DIR / "final_report.md"
    md_path.write_text(md, encoding="utf-8")
    saved["markdown"] = str(md_path)
    logger.info(f"Markdown report: {md_path}")

    # CSV metrics
    csv_data = generate_csv_metrics(benchmark_data)
    csv_path = OUTPUT_DIR / "benchmark_metrics.csv"
    csv_path.write_text(csv_data, encoding="utf-8")
    saved["csv"] = str(csv_path)
    logger.info(f"CSV metrics: {csv_path}")

    # Summary JSON
    summary = generate_summary_json(benchmark_data)
    json_path = OUTPUT_DIR / "benchmark_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    saved["json"] = str(json_path)
    logger.info(f"Summary JSON: {json_path}")

    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Load all benchmark outputs
    combined = {}
    for json_file in OUTPUT_DIR.glob("*.json"):
        if json_file.name == "benchmark_summary.json":
            continue
        try:
            data = json.loads(json_file.read_text())
            key = json_file.stem.replace("_benchmark", "").replace("_stress", "").replace("_simulation", "")
            combined[key] = data
        except Exception as e:
            logger.warning(f"Could not load {json_file}: {e}")

    saved = generate_all_reports(combined)
    for fmt, path in saved.items():
        print(f"{fmt}: {path}")
