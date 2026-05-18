"""
Prometheus Metrics for STT Pipeline.

Exports metrics for:
- STT latency (load, preprocess, inference, total)
- Inference errors (timeout, decode, CUDA, etc.)
- Queue depth and wait times
- GPU memory and utilization
- Throughput (transcriptions/sec)
"""

from dataclasses import dataclass, field
from typing import Dict, List

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Metrics Definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # STT Pipeline Latency (milliseconds)
    stt_load_time_ms = Histogram(
        "stt_load_time_ms",
        "Audio file load time (ms)",
        buckets=[10, 50, 100, 250, 500, 1000],
    )

    stt_preprocess_time_ms = Histogram(
        "stt_preprocess_time_ms",
        "Audio preprocessing time (ms)",
        buckets=[10, 50, 100, 250, 500, 1000],
    )

    stt_inference_time_ms = Histogram(
        "stt_inference_time_ms",
        "Model inference time (ms)",
        ["language"],
        buckets=[100, 250, 500, 1000, 2000, 5000],
    )

    stt_total_time_ms = Histogram(
        "stt_total_time_ms",
        "Total transcription time (ms)",
        buckets=[500, 1000, 2000, 5000, 10000],
    )

    # STT Errors
    stt_errors_total = Counter(
        "stt_errors_total",
        "Total STT errors",
        ["error_type"],
    )

    # VAD Metrics
    stt_vad_rejected_total = Counter(
        "stt_vad_rejected_total",
        "Total audio chunks rejected by VAD",
    )

    stt_vad_confidence = Histogram(
        "stt_vad_confidence",
        "VAD confidence score (0-1)",
        buckets=[0.1, 0.3, 0.5, 0.7, 0.9],
    )

    # Transcription Results
    stt_transcriptions_total = Counter(
        "stt_transcriptions_total",
        "Total transcriptions completed",
        ["status"],  # success, vad_rejected, error
    )

    stt_transcript_length = Histogram(
        "stt_transcript_length",
        "Transcript character length",
        ["language"],
        buckets=[10, 50, 100, 250, 500, 1000],
    )

    # GPU Metrics
    gpu_memory_allocated_mb = Gauge(
        "gpu_memory_allocated_mb",
        "GPU memory allocated (MB)",
    )

    gpu_memory_reserved_mb = Gauge(
        "gpu_memory_reserved_mb",
        "GPU memory reserved (MB)",
    )

    gpu_memory_free_mb = Gauge(
        "gpu_memory_free_mb",
        "GPU memory free (MB)",
    )

    # Queue Metrics
    queue_depth = Gauge(
        "queue_depth",
        "Number of items in queue",
        ["queue_type"],  # audio, video
    )

    queue_wait_time_ms = Histogram(
        "queue_wait_time_ms",
        "Time spent waiting in queue (ms)",
        ["queue_type"],
        buckets=[10, 50, 100, 500, 1000, 5000],
    )

    # Throughput
    transcriptions_per_second = Gauge(
        "transcriptions_per_second",
        "Throughput: transcriptions per second",
    )

    # Model Status
    model_loaded = Gauge(
        "stt_model_loaded",
        "STT model loaded (1=yes, 0=no)",
    )

    cuda_available = Gauge(
        "cuda_available",
        "CUDA available (1=yes, 0=no)",
    )

else:
    # Fallback: no-op metrics (doesn't require prometheus_client)
    class _NoOpMetric:
        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    stt_load_time_ms = _NoOpMetric()
    stt_preprocess_time_ms = _NoOpMetric()
    stt_inference_time_ms = _NoOpMetric()
    stt_total_time_ms = _NoOpMetric()
    stt_errors_total = _NoOpMetric()
    stt_vad_rejected_total = _NoOpMetric()
    stt_vad_confidence = _NoOpMetric()
    stt_transcriptions_total = _NoOpMetric()
    stt_transcript_length = _NoOpMetric()
    gpu_memory_allocated_mb = _NoOpMetric()
    gpu_memory_reserved_mb = _NoOpMetric()
    gpu_memory_free_mb = _NoOpMetric()
    queue_depth = _NoOpMetric()
    queue_wait_time_ms = _NoOpMetric()
    transcriptions_per_second = _NoOpMetric()
    model_loaded = _NoOpMetric()
    cuda_available = _NoOpMetric()


# ---------------------------------------------------------------------------
# Metrics Helper: STT Statistics
# ---------------------------------------------------------------------------

@dataclass
class STTMetrics:
    """Accumulator for STT pipeline metrics."""

    request_id: str
    session_id: str
    language: str = ""
    status: str = "pending"  # pending, success, vad_rejected, error
    error_type: str = ""  # timeout, cuda_error, etc.

    load_time_ms: float = 0.0
    preprocess_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0

    vad_detected: bool = False
    vad_confidence: float = 0.0

    transcript_length: int = 0
    transcript_languages: List[str] = field(default_factory=list)

    def record(self) -> None:
        """Record metrics to Prometheus."""
        if self.load_time_ms > 0:
            stt_load_time_ms.observe(self.load_time_ms)

        if self.preprocess_time_ms > 0:
            stt_preprocess_time_ms.observe(self.preprocess_time_ms)

        if self.inference_time_ms > 0:
            stt_inference_time_ms.labels(language=self.language).observe(self.inference_time_ms)

        if self.total_time_ms > 0:
            stt_total_time_ms.observe(self.total_time_ms)

        if self.vad_confidence > 0:
            stt_vad_confidence.observe(self.vad_confidence)

        if not self.vad_detected:
            stt_vad_rejected_total.inc()

        stt_transcriptions_total.labels(status=self.status).inc()

        if self.error_type:
            stt_errors_total.labels(error_type=self.error_type).inc()

        if self.transcript_length > 0:
            stt_transcript_length.labels(language=self.language).observe(self.transcript_length)


# ---------------------------------------------------------------------------
# GPU Metrics Updater
# ---------------------------------------------------------------------------

def update_gpu_metrics() -> None:
    """Update GPU memory and utilization metrics."""
    try:
        import torch

        if torch.cuda.is_available():
            cuda_available.set(1)
            allocated = torch.cuda.memory_allocated(0) / (1024**2)
            reserved = torch.cuda.memory_reserved(0) / (1024**2)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            free = total - allocated

            gpu_memory_allocated_mb.set(allocated)
            gpu_memory_reserved_mb.set(reserved)
            gpu_memory_free_mb.set(free)
        else:
            cuda_available.set(0)

    except Exception:
        pass


def update_model_status(model_loaded: bool, cuda_available: bool) -> None:
    """Update model status metrics."""
    model_loaded_gauge = model_loaded
    cuda_available_gauge = 1 if cuda_available else 0

    if PROMETHEUS_AVAILABLE:
        globals()["model_loaded"].set(model_loaded_gauge)
        globals()["cuda_available"].set(cuda_available_gauge)
