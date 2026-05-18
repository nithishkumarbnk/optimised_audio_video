"""
benchmark/stress/websocket_stream_simulator.py
Real-time WebSocket stream simulator for concurrent student sessions.

Simulates 1, 10, 50, 100 concurrent students sending webcam frames
and audio chunks over WebSocket connections.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys_path_added = False
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    sys_path_added = True
except Exception:
    pass

logger = logging.getLogger(__name__)

try:
    import websockets
except ImportError:
    raise ImportError("Install websockets: pip install websockets")


@dataclass
class StudentSessionResult:
    session_id: str
    connected: bool = False
    connect_time_ms: float = 0.0
    frames_sent: int = 0
    frames_dropped: int = 0
    audio_sent: int = 0
    events_received: int = 0
    errors: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    disconnected_early: bool = False
    duration_s: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.95)]


def _make_jpeg_b64(frame_type: str = "normal") -> str:
    from benchmark.datasets.generator import generate_frame_bytes
    return base64.b64encode(generate_frame_bytes(frame_type)).decode()


def _make_wav_b64(duration_s: float = 2.0) -> str:
    from benchmark.datasets.generator import generate_audio_bytes
    return base64.b64encode(generate_audio_bytes("speech_like", duration_s)).decode()


async def simulate_student(
    base_url: str,
    session_id: str,
    duration_s: float,
    frame_interval_s: float = 1.0,
    audio_interval_s: float = 4.0,
    frame_types: List[str] = None,
    network_jitter_ms: float = 0.0,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> StudentSessionResult:
    """Simulate a single student WebSocket session."""
    result = StudentSessionResult(session_id=session_id)
    ws_url = f"{base_url}/ws/proctor/{session_id}"
    frame_types = frame_types or ["normal", "looking_away", "mobile", "no_face"]

    if semaphore:
        await semaphore.acquire()

    t_start = time.monotonic()

    try:
        t_connect = time.monotonic()
        async with websockets.connect(
            ws_url,
            ping_interval=None,
            open_timeout=15,
            close_timeout=5,
        ) as ws:
            result.connected = True
            result.connect_time_ms = round((time.monotonic() - t_connect) * 1000, 1)

            last_frame = 0.0
            last_audio = 0.0
            frame_idx = 0

            async def recv_loop():
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") == "ping":
                            await ws.send(json.dumps({"type": "pong"}))
                            continue
                        result.events_received += 1
                        # Record latency if payload has timing info
                        sent_at = msg.get("_sent_at")
                        if sent_at:
                            lat = (time.monotonic() - sent_at) * 1000
                            result.latencies_ms.append(lat)
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    result.errors += 1
                    logger.debug(f"[{session_id}] recv error: {e}")

            recv_task = asyncio.create_task(recv_loop())

            while time.monotonic() - t_start < duration_s:
                now = time.monotonic()

                # Send video frame at target FPS
                if now - last_frame >= frame_interval_s:
                    try:
                        ftype = frame_types[frame_idx % len(frame_types)]
                        frame_idx += 1
                        b64 = _make_jpeg_b64(ftype)

                        # Simulate network jitter
                        if network_jitter_ms > 0:
                            await asyncio.sleep(random.uniform(0, network_jitter_ms / 1000))

                        await ws.send(json.dumps({"type": "video", "frame": b64}))
                        result.frames_sent += 1
                        last_frame = now
                    except Exception as e:
                        result.frames_dropped += 1
                        logger.debug(f"[{session_id}] frame send error: {e}")

                # Send audio chunk
                if now - last_audio >= audio_interval_s:
                    try:
                        b64 = _make_wav_b64(duration_s=2.0)
                        await ws.send(json.dumps({"type": "audio", "audio": b64}))
                        result.audio_sent += 1
                        last_audio = now
                    except Exception as e:
                        result.errors += 1

                await asyncio.sleep(0.05)

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

    except asyncio.TimeoutError:
        result.errors += 1
        result.disconnected_early = True
    except Exception as e:
        result.errors += 1
        result.disconnected_early = True
        logger.debug(f"[{session_id}] connection error: {e}")
    finally:
        if semaphore:
            semaphore.release()

    result.duration_s = round(time.monotonic() - t_start, 2)
    return result


async def run_concurrent_simulation(
    base_url: str,
    num_students: int,
    duration_s: float,
    max_concurrent: int = None,
    frame_interval_s: float = 1.0,
    audio_interval_s: float = 4.0,
    network_jitter_ms: float = 0.0,
) -> Dict:
    """Run concurrent student simulation and return aggregated results."""
    max_concurrent = max_concurrent or num_students
    semaphore = asyncio.Semaphore(max_concurrent)

    logger.info(f"Starting simulation: {num_students} students, {duration_s}s duration")

    tasks = [
        simulate_student(
            base_url=base_url,
            session_id=f"bench-student-{i:04d}",
            duration_s=duration_s,
            frame_interval_s=frame_interval_s,
            audio_interval_s=audio_interval_s,
            network_jitter_ms=network_jitter_ms,
            semaphore=semaphore,
        )
        for i in range(num_students)
    ]

    t_start = time.monotonic()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_elapsed = round(time.monotonic() - t_start, 2)

    student_results = [r for r in results if isinstance(r, StudentSessionResult)]
    errors = len([r for r in results if not isinstance(r, StudentSessionResult)])

    connected = [r for r in student_results if r.connected]
    all_latencies = [lat for r in student_results for lat in r.latencies_ms]
    connect_times = [r.connect_time_ms for r in connected]

    total_frames = sum(r.frames_sent for r in student_results)
    total_dropped = sum(r.frames_dropped for r in student_results)
    total_audio = sum(r.audio_sent for r in student_results)
    total_events = sum(r.events_received for r in student_results)
    total_errors = sum(r.errors for r in student_results) + errors
    early_disc = sum(1 for r in student_results if r.disconnected_early)

    def pct(lst, p):
        if not lst:
            return 0
        s = sorted(lst)
        return round(s[int(len(s) * p)], 2)

    summary = {
        "num_students": num_students,
        "duration_s": duration_s,
        "total_elapsed_s": total_elapsed,
        "connected": len(connected),
        "failed_to_connect": num_students - len(connected),
        "disconnected_early": early_disc,
        "total_frames_sent": total_frames,
        "total_frames_dropped": total_dropped,
        "drop_rate_pct": round(total_dropped / max(total_frames + total_dropped, 1) * 100, 2),
        "total_audio_sent": total_audio,
        "total_events_received": total_events,
        "total_errors": total_errors,
        "connect_time_ms": {
            "mean": round(statistics.mean(connect_times), 1) if connect_times else 0,
            "max": round(max(connect_times), 1) if connect_times else 0,
        },
        "event_latency_ms": {
            "mean": round(statistics.mean(all_latencies), 1) if all_latencies else 0,
            "p50": pct(all_latencies, 0.50),
            "p95": pct(all_latencies, 0.95),
            "p99": pct(all_latencies, 0.99),
            "max": round(max(all_latencies), 1) if all_latencies else 0,
        } if all_latencies else {"note": "no latency data"},
    }

    logger.info(f"Simulation complete: {len(connected)}/{num_students} connected, "
                f"{total_events} events, drop_rate={summary['drop_rate_pct']}%")
    return summary


def run_scalability_test(
    base_url: str = "ws://localhost:8000",
    student_counts: List[int] = None,
    duration_s: float = 20.0,
) -> Dict:
    """Run scalability test across multiple student counts."""
    student_counts = student_counts or [1, 5, 10, 20]
    results = {}

    for count in student_counts:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {count} concurrent students...")
        result = asyncio.run(run_concurrent_simulation(
            base_url=base_url,
            num_students=count,
            duration_s=duration_s,
            max_concurrent=min(count, 20),  # cap concurrency to avoid overwhelming
        ))
        results[f"{count}_students"] = result

    return results


if __name__ == "__main__":
    import json
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    output_dir = Path(__file__).parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_scalability_test(
        base_url="ws://localhost:8000",
        student_counts=[1, 5, 10],
        duration_s=15.0,
    )
    (output_dir / "websocket_simulation.json").write_text(json.dumps(results, indent=2))
    print("Results saved to benchmark/outputs/websocket_simulation.json")
