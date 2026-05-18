"""
benchmark/metrics/collector.py
Real-time metrics collector for the AI Proctoring benchmark framework.

Collects CPU, RAM, GPU, VRAM, latency, FPS, and queue metrics.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)

# Optional GPU support
try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False


@dataclass
class LatencySample:
    timestamp: float
    value_ms: float
    label: str = ""


@dataclass
class MetricsSnapshot:
    timestamp: float
    cpu_percent: float
    ram_usage_mb: float
    ram_percent: float
    gpu_percent: float
    vram_usage_mb: float
    inference_latency_ms: float = 0.0
    queue_delay_ms: float = 0.0
    websocket_latency_ms: float = 0.0
    fps: float = 0.0
    dropped_frames: int = 0
    active_sessions: int = 0


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics."""
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    # Latency samples
    inference_latencies: List[float] = field(default_factory=list)
    end_to_end_latencies: List[float] = field(default_factory=list)
    websocket_latencies: List[float] = field(default_factory=list)
    queue_delays: List[float] = field(default_factory=list)
    audio_processing_times: List[float] = field(default_factory=list)
    risk_engine_times: List[float] = field(default_factory=list)
    session_update_times: List[float] = field(default_factory=list)

    # Throughput
    frames_processed: int = 0
    frames_dropped: int = 0
    messages_dropped: int = 0
    websocket_disconnects: int = 0
    events_generated: int = 0

    # Resource snapshots
    resource_snapshots: List[MetricsSnapshot] = field(default_factory=list)

    # Worker metrics
    worker_utilization: List[float] = field(default_factory=list)
    queue_depths: List[int] = field(default_factory=list)

    def duration_s(self) -> float:
        end = self.end_time if self.end_time > 0 else time.time()
        return end - self.start_time

    def fps(self) -> float:
        dur = self.duration_s()
        return self.frames_processed / dur if dur > 0 else 0.0

    def drop_rate(self) -> float:
        total = self.frames_processed + self.frames_dropped
        return self.frames_dropped / total if total > 0 else 0.0

    def _stats(self, values: List[float]) -> Dict:
        if not values:
            return {"count": 0, "mean": 0, "median": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "stdev": 0}
        s = sorted(values)
        n = len(s)
        return {
            "count": n,
            "mean": round(statistics.mean(s), 2),
            "median": round(statistics.median(s), 2),
            "p95": round(s[int(n * 0.95)], 2),
            "p99": round(s[int(n * 0.99)], 2),
            "min": round(s[0], 2),
            "max": round(s[-1], 2),
            "stdev": round(statistics.stdev(s) if n > 1 else 0, 2),
        }

    def summary(self) -> Dict:
        snaps = self.resource_snapshots
        cpu_vals = [s.cpu_percent for s in snaps]
        ram_vals = [s.ram_usage_mb for s in snaps]
        gpu_vals = [s.gpu_percent for s in snaps]
        vram_vals = [s.vram_usage_mb for s in snaps]

        return {
            "name": self.name,
            "duration_s": round(self.duration_s(), 2),
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "drop_rate_pct": round(self.drop_rate() * 100, 2),
            "fps": round(self.fps(), 2),
            "events_generated": self.events_generated,
            "websocket_disconnects": self.websocket_disconnects,
            "inference_latency_ms": self._stats(self.inference_latencies),
            "end_to_end_latency_ms": self._stats(self.end_to_end_latencies),
            "websocket_latency_ms": self._stats(self.websocket_latencies),
            "queue_delay_ms": self._stats(self.queue_delays),
            "audio_processing_ms": self._stats(self.audio_processing_times),
            "risk_engine_ms": self._stats(self.risk_engine_times),
            "cpu_percent": self._stats(cpu_vals),
            "ram_usage_mb": self._stats(ram_vals),
            "gpu_percent": self._stats(gpu_vals),
            "vram_usage_mb": self._stats(vram_vals),
            "worker_utilization": self._stats(self.worker_utilization),
        }


class MetricsCollector:
    """Background metrics collector — polls system resources at fixed intervals."""

    def __init__(self, interval_s: float = 0.5):
        self._interval = interval_s
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._metrics: Optional[BenchmarkMetrics] = None
        self._lock = threading.Lock()

    def start(self, metrics: BenchmarkMetrics) -> None:
        self._metrics = metrics
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        logger.info(f"MetricsCollector started (interval={self._interval}s)")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("MetricsCollector stopped")

    def _collect_loop(self) -> None:
        proc = psutil.Process()
        while self._running:
            try:
                snap = self._take_snapshot(proc)
                with self._lock:
                    if self._metrics:
                        self._metrics.resource_snapshots.append(snap)
            except Exception as e:
                logger.debug(f"Metrics collection error: {e}")
            time.sleep(self._interval)

    def _take_snapshot(self, proc: psutil.Process) -> MetricsSnapshot:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_mb = mem.used / 1024 / 1024

        gpu_pct = 0.0
        vram_mb = 0.0
        if _GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_pct = float(util.gpu)
                vram_mb = mem_info.used / 1024 / 1024
            except Exception:
                pass

        return MetricsSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu,
            ram_usage_mb=round(ram_mb, 1),
            ram_percent=mem.percent,
            gpu_percent=gpu_pct,
            vram_usage_mb=round(vram_mb, 1),
        )

    def record_inference(self, latency_ms: float) -> None:
        with self._lock:
            if self._metrics:
                self._metrics.inference_latencies.append(latency_ms)

    def record_e2e(self, latency_ms: float) -> None:
        with self._lock:
            if self._metrics:
                self._metrics.end_to_end_latencies.append(latency_ms)

    def record_websocket(self, latency_ms: float) -> None:
        with self._lock:
            if self._metrics:
                self._metrics.websocket_latencies.append(latency_ms)

    def record_queue_delay(self, delay_ms: float) -> None:
        with self._lock:
            if self._metrics:
                self._metrics.queue_delays.append(delay_ms)

    def record_audio(self, latency_ms: float) -> None:
        with self._lock:
            if self._metrics:
                self._metrics.audio_processing_times.append(latency_ms)

    def record_risk_engine(self, latency_ms: float) -> None:
        with self._lock:
            if self._metrics:
                self._metrics.risk_engine_times.append(latency_ms)
