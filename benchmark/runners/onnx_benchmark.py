"""
benchmark/runners/onnx_benchmark.py
ONNX model inference benchmark runner.

Benchmarks:
- Single inference latency (CPU/CUDA)
- Warmup runs
- Batch testing
- Continuous stream simulation
- Per-model timing breakdown
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parents[2]))
from benchmark.datasets.generator import generate_all_frame_types, FRAME_TYPES
from benchmark.metrics.collector import BenchmarkMetrics, MetricsCollector

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parents[2] / "app" / "models"
OUTPUT_DIR = Path(__file__).parents[1] / "outputs"


def preprocess_frame(frame: np.ndarray, size: int = 640) -> np.ndarray:
    resized = cv2.resize(frame, (size, size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


class ONNXModelBenchmark:
    """Benchmarks a single ONNX model."""

    def __init__(
        self,
        model_path: str,
        model_name: str,
        providers: List[str] = None,
        warmup_runs: int = 50,
        benchmark_runs: int = 500,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.providers = providers or ["CPUExecutionProvider"]
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.session: Optional[ort.InferenceSession] = None

    def load(self) -> None:
        logger.info(f"Loading {self.model_name} from {self.model_path}")
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        load_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"{self.model_name} loaded in {load_ms}ms")
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape

    def _make_input(self) -> np.ndarray:
        """Create a random input tensor matching the model's expected shape."""
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in self.input_shape]
        return np.random.rand(*shape).astype(np.float32)

    def warmup(self) -> None:
        logger.info(f"Warming up {self.model_name} ({self.warmup_runs} runs)...")
        inp = self._make_input()
        for _ in range(self.warmup_runs):
            self.session.run(None, {self.input_name: inp})
        logger.info(f"Warmup complete for {self.model_name}")

    def run_latency_benchmark(self, frame_type: str = "normal") -> Dict:
        """Measure inference latency over N runs."""
        from benchmark.datasets.generator import FRAME_TYPES
        fn = FRAME_TYPES.get(frame_type, list(FRAME_TYPES.values())[0])
        frame = fn()
        inp = preprocess_frame(frame)

        latencies = []
        for i in range(self.benchmark_runs):
            t0 = time.perf_counter()
            self.session.run(None, {self.input_name: inp})
            latencies.append((time.perf_counter() - t0) * 1000)

        s = sorted(latencies)
        n = len(s)
        return {
            "model": self.model_name,
            "frame_type": frame_type,
            "runs": n,
            "mean_ms": round(sum(s) / n, 3),
            "median_ms": round(s[n // 2], 3),
            "p95_ms": round(s[int(n * 0.95)], 3),
            "p99_ms": round(s[int(n * 0.99)], 3),
            "min_ms": round(s[0], 3),
            "max_ms": round(s[-1], 3),
            "throughput_fps": round(1000 / (sum(s) / n), 1),
        }

    def run_all_frame_types(self) -> List[Dict]:
        """Benchmark across all frame types."""
        results = []
        for frame_type in FRAME_TYPES:
            result = self.run_latency_benchmark(frame_type)
            results.append(result)
            logger.info(
                f"{self.model_name} [{frame_type}]: "
                f"mean={result['mean_ms']}ms p95={result['p95_ms']}ms "
                f"fps={result['throughput_fps']}"
            )
        return results

    def run_stream_simulation(self, duration_s: float = 30.0, fps: float = 1.0) -> Dict:
        """Simulate continuous 1fps stream for duration_s seconds."""
        frame = FRAME_TYPES["normal"]()
        inp = preprocess_frame(frame)
        interval = 1.0 / fps

        latencies = []
        frames_processed = 0
        frames_dropped = 0
        start = time.perf_counter()

        while time.perf_counter() - start < duration_s:
            t0 = time.perf_counter()
            self.session.run(None, {self.input_name: inp})
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            frames_processed += 1

            elapsed = time.perf_counter() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                frames_dropped += 1

        total = time.perf_counter() - start
        s = sorted(latencies)
        n = len(s)
        return {
            "model": self.model_name,
            "duration_s": round(total, 2),
            "target_fps": fps,
            "actual_fps": round(frames_processed / total, 2),
            "frames_processed": frames_processed,
            "frames_dropped": frames_dropped,
            "drop_rate_pct": round(frames_dropped / max(frames_processed + frames_dropped, 1) * 100, 2),
            "mean_ms": round(sum(s) / n, 3) if n else 0,
            "p95_ms": round(s[int(n * 0.95)], 3) if n else 0,
        }


def run_all_models_benchmark(
    warmup_runs: int = 50,
    benchmark_runs: int = 500,
    stream_duration_s: float = 30.0,
) -> Dict:
    """Run benchmark for all three ONNX models."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = [
        ("yolov8n.onnx", "YOLO"),
        ("headset_model.onnx", "Headset"),
        ("proctoring.onnx", "Proctor"),
    ]

    all_results = {}

    for filename, name in models:
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue

        bench = ONNXModelBenchmark(
            model_path=str(model_path),
            model_name=name,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )
        bench.load()
        bench.warmup()

        frame_results = bench.run_all_frame_types()
        stream_result = bench.run_stream_simulation(duration_s=stream_duration_s)

        all_results[name] = {
            "per_frame_type": frame_results,
            "stream_simulation": stream_result,
        }

        logger.info(f"=== {name} Summary ===")
        logger.info(f"  Stream: {stream_result['actual_fps']} fps, "
                    f"drop_rate={stream_result['drop_rate_pct']}%")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    results = run_all_models_benchmark(warmup_runs=10, benchmark_runs=100, stream_duration_s=10.0)
    import json
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "onnx_benchmark.json").write_text(json.dumps(results, indent=2))
    print("Results saved to benchmark/outputs/onnx_benchmark.json")
