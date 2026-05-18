"""
benchmark/runners/audio_benchmark.py
Audio pipeline benchmark — measures STT latency, preprocessing time,
and audio queue performance.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parents[2]))

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parents[1] / "outputs"


def benchmark_audio_preprocessing(audio_bytes: bytes, runs: int = 100) -> Dict:
    """Benchmark audio preprocessing (load + resample + normalize)."""
    import io
    import numpy as np
    import soundfile as sf
    import librosa

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        buf = io.BytesIO(audio_bytes)
        waveform, sr = sf.read(buf)
        waveform = np.array(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        latencies.append((time.perf_counter() - t0) * 1000)

    s = sorted(latencies)
    n = len(s)
    return {
        "stage": "preprocessing",
        "runs": n,
        "mean_ms": round(sum(s) / n, 3),
        "p95_ms": round(s[int(n * 0.95)], 3),
        "min_ms": round(s[0], 3),
        "max_ms": round(s[-1], 3),
    }


def benchmark_stt_single(audio_path: str, runs: int = 3) -> Dict:
    """Benchmark STT transcription for a single audio file."""
    from app.services.stt_service import STTService

    if STTService._model is None:
        logger.info("Initializing STT model for benchmark...")
        STTService.initialize()

    latencies = []
    transcripts = []

    for i in range(runs):
        logger.info(f"STT benchmark run {i+1}/{runs}...")
        t0 = time.perf_counter()
        text = STTService.transcribe(audio_path)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        transcripts.append(text or "")
        logger.info(f"  Run {i+1}: {lat:.0f}ms → '{(text or '')[:40]}'")

    s = sorted(latencies)
    n = len(s)
    return {
        "stage": "stt_transcription",
        "audio_path": audio_path,
        "runs": n,
        "mean_ms": round(sum(s) / n, 1),
        "min_ms": round(s[0], 1),
        "max_ms": round(s[-1], 1),
        "sample_transcript": transcripts[0][:80] if transcripts else "",
    }


def benchmark_rule_engine(texts: List[str], runs: int = 1000) -> Dict:
    """Benchmark rule engine performance."""
    from app.services.rule_engine import check_rules

    latencies = []
    for _ in range(runs):
        text = texts[_ % len(texts)]
        t0 = time.perf_counter()
        check_rules(text)
        latencies.append((time.perf_counter() - t0) * 1000)

    s = sorted(latencies)
    n = len(s)
    return {
        "stage": "rule_engine",
        "runs": n,
        "mean_ms": round(sum(s) / n, 4),
        "p95_ms": round(s[int(n * 0.95)], 4),
        "throughput_per_sec": round(1000 / (sum(s) / n), 0),
    }


def benchmark_risk_engine(runs: int = 10000) -> Dict:
    """Benchmark risk engine scoring performance."""
    from app.services.risk_engine import compute_score, classify
    import random

    all_events = [
        "mobile_detected", "multiple_persons", "headset_detected",
        "suspicious_transcript", "looking_away", "no_face",
        "long_speech", "keyword_detected", "bad_posture", "suspicious_movement"
    ]

    latencies = []
    for _ in range(runs):
        events = random.sample(all_events, k=random.randint(0, 5))
        t0 = time.perf_counter()
        score = compute_score(events)
        classify(score)
        latencies.append((time.perf_counter() - t0) * 1000)

    s = sorted(latencies)
    n = len(s)
    return {
        "stage": "risk_engine",
        "runs": n,
        "mean_ms": round(sum(s) / n, 5),
        "p99_ms": round(s[int(n * 0.99)], 5),
        "throughput_per_sec": round(1000 / (sum(s) / n), 0),
    }


def run_audio_benchmark(audio_dir: Path = None) -> Dict:
    """Run the complete audio pipeline benchmark."""
    from benchmark.datasets.generator import AUDIO_TYPES, generate_audio_bytes

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. Preprocessing benchmark
    logger.info("=== Audio Preprocessing Benchmark ===")
    speech_bytes = generate_audio_bytes("speech_like", duration_s=3.0)
    results["preprocessing"] = benchmark_audio_preprocessing(speech_bytes, runs=200)
    logger.info(f"  Preprocessing: mean={results['preprocessing']['mean_ms']}ms")

    # 2. Rule engine benchmark
    logger.info("=== Rule Engine Benchmark ===")
    test_texts = [
        "what is the answer to question five",
        "can you tell me how to solve this",
        "I need help with this problem",
        "normal conversation about the exam",
        "google search for the answer",
        "",
        None,
    ]
    results["rule_engine"] = benchmark_rule_engine(
        [t for t in test_texts if t], runs=5000
    )
    logger.info(f"  Rule engine: mean={results['rule_engine']['mean_ms']}ms "
                f"throughput={results['rule_engine']['throughput_per_sec']}/s")

    # 3. Risk engine benchmark
    logger.info("=== Risk Engine Benchmark ===")
    results["risk_engine"] = benchmark_risk_engine(runs=10000)
    logger.info(f"  Risk engine: mean={results['risk_engine']['mean_ms']}ms "
                f"throughput={results['risk_engine']['throughput_per_sec']}/s")

    # 4. STT benchmark (optional — slow)
    if audio_dir and (audio_dir / "speech_like.wav").exists():
        logger.info("=== STT Benchmark (3 runs) ===")
        try:
            results["stt"] = benchmark_stt_single(
                str(audio_dir / "speech_like.wav"), runs=3
            )
        except Exception as e:
            logger.warning(f"STT benchmark skipped: {e}")
            results["stt"] = {"error": str(e)}

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    import json
    from benchmark.datasets.generator import generate_dataset, DATASETS_DIR

    dataset = generate_dataset()
    results = run_audio_benchmark(audio_dir=DATASETS_DIR / "audio")
    (OUTPUT_DIR / "audio_benchmark.json").write_text(json.dumps(results, indent=2))
    print("Results saved to benchmark/outputs/audio_benchmark.json")
