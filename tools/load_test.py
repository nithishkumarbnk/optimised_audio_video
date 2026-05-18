"""
load_test.py — Stress test the AI Proctoring Backend with concurrent WebSocket sessions.

Simulates multiple concurrent exam sessions, each sending video frames and audio chunks.
Measures latency, throughput, dropped frames, and WebSocket responsiveness.

Usage:
    python tools/load_test.py --sessions 10 --duration 30
    python tools/load_test.py --sessions 50 --duration 60 --url ws://localhost:8000
    python tools/load_test.py --sessions 5 --duration 20 --report
"""

import asyncio
import argparse
import base64
import json
import math
import struct
import time
import statistics
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    import sys; sys.exit(1)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── FAKE DATA GENERATORS ──────────────────────────────────────────────────

def make_fake_jpeg() -> str:
    """Minimal valid JPEG as base64."""
    jpeg = (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
        b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
        b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e'
        b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
        b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'
        b'\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
        b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00\x00\x00\xff\xd9'
    )
    return base64.b64encode(jpeg).decode()


def make_fake_wav(duration_s: float = 2.0, sample_rate: int = 16000) -> str:
    """Generate a WAV sine wave as base64."""
    num_samples = int(sample_rate * duration_s)
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(num_samples)]
    data_size = num_samples * 2
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size
    )
    wav = header + struct.pack(f'<{num_samples}h', *samples)
    return base64.b64encode(wav).decode()


# ── SESSION METRICS ───────────────────────────────────────────────────────

@dataclass
class SessionMetrics:
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
    total_duration_s: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]


# ── SINGLE SESSION WORKER ─────────────────────────────────────────────────

async def run_session(
    base_url: str,
    session_id: str,
    duration: int,
    frame_interval: float = 1.0,
    audio_interval: float = 4.0,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> SessionMetrics:
    metrics = SessionMetrics(session_id=session_id)
    ws_url = f"{base_url}/ws/proctor/{session_id}"

    if semaphore:
        await semaphore.acquire()

    t_start = time.monotonic()

    try:
        t_connect = time.monotonic()
        async with websockets.connect(
            ws_url,
            ping_interval=None,
            open_timeout=10,
            close_timeout=5,
        ) as ws:
            metrics.connected = True
            metrics.connect_time_ms = round((time.monotonic() - t_connect) * 1000, 1)

            last_frame = 0.0
            last_audio = 0.0
            pending: Dict[str, float] = {}  # msg_id -> send_time

            async def recv_loop():
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get('type') == 'ping':
                            await ws.send(json.dumps({'type': 'pong'}))
                            continue
                        metrics.events_received += 1
                        # Track latency if we embedded a send timestamp
                        sent_at = msg.get('_sent_at')
                        if sent_at:
                            lat = (time.monotonic() - sent_at) * 1000
                            metrics.latencies_ms.append(lat)
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    metrics.errors += 1

            recv_task = asyncio.create_task(recv_loop())

            while time.monotonic() - t_start < duration:
                now = time.monotonic()

                if now - last_frame >= frame_interval:
                    try:
                        frame_b64 = make_fake_jpeg()
                        await ws.send(json.dumps({'type': 'video', 'frame': frame_b64}))
                        metrics.frames_sent += 1
                        last_frame = now
                    except Exception:
                        metrics.frames_dropped += 1

                if now - last_audio >= audio_interval:
                    try:
                        audio_b64 = make_fake_wav(duration_s=2.0)
                        await ws.send(json.dumps({'type': 'audio', 'audio': audio_b64}))
                        metrics.audio_sent += 1
                        last_audio = now
                    except Exception:
                        metrics.errors += 1

                await asyncio.sleep(0.05)

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

    except asyncio.TimeoutError:
        metrics.errors += 1
        metrics.disconnected_early = True
    except Exception as e:
        metrics.errors += 1
        metrics.disconnected_early = True
    finally:
        if semaphore:
            semaphore.release()

    metrics.total_duration_s = round(time.monotonic() - t_start, 2)
    return metrics


# ── LOAD TEST RUNNER ──────────────────────────────────────────────────────

async def run_load_test(
    base_url: str,
    num_sessions: int,
    duration: int,
    concurrency: int,
    frame_interval: float,
    audio_interval: float,
) -> List[SessionMetrics]:
    print(f"\n{'='*60}")
    print(f"  AI Proctoring Backend — Load Test")
    print(f"{'='*60}")
    print(f"  Sessions:    {num_sessions}")
    print(f"  Duration:    {duration}s per session")
    print(f"  Concurrency: {concurrency}")
    print(f"  Frame rate:  1 frame / {frame_interval}s")
    print(f"  Audio rate:  1 chunk / {audio_interval}s")
    print(f"  Backend:     {base_url}")
    print(f"{'='*60}\n")

    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        run_session(
            base_url=base_url,
            session_id=f"load-test-{i:04d}",
            duration=duration,
            frame_interval=frame_interval,
            audio_interval=audio_interval,
            semaphore=semaphore,
        )
        for i in range(num_sessions)
    ]

    t_start = time.monotonic()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_elapsed = round(time.monotonic() - t_start, 1)

    metrics_list = [r for r in results if isinstance(r, SessionMetrics)]
    errors = [r for r in results if not isinstance(r, SessionMetrics)]

    return metrics_list, total_elapsed, len(errors)


def print_report(metrics_list: List[SessionMetrics], total_elapsed: float, gather_errors: int):
    connected = [m for m in metrics_list if m.connected]
    failed = [m for m in metrics_list if not m.connected]
    early_disc = [m for m in metrics_list if m.disconnected_early]

    all_latencies = [lat for m in metrics_list for lat in m.latencies_ms]
    total_frames = sum(m.frames_sent for m in metrics_list)
    total_dropped = sum(m.frames_dropped for m in metrics_list)
    total_audio = sum(m.audio_sent for m in metrics_list)
    total_events = sum(m.events_received for m in metrics_list)
    total_errors = sum(m.errors for m in metrics_list) + gather_errors

    connect_times = [m.connect_time_ms for m in connected]

    print(f"\n{'='*60}")
    print(f"  LOAD TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total elapsed:       {total_elapsed}s")
    print(f"  Sessions attempted:  {len(metrics_list)}")
    print(f"  Connected:           {len(connected)}")
    print(f"  Failed to connect:   {len(failed)}")
    print(f"  Disconnected early:  {len(early_disc)}")
    print(f"")
    print(f"  Frames sent:         {total_frames}")
    print(f"  Frames dropped:      {total_dropped}")
    drop_rate = round(total_dropped / max(total_frames + total_dropped, 1) * 100, 1)
    print(f"  Drop rate:           {drop_rate}%")
    print(f"")
    print(f"  Audio chunks sent:   {total_audio}")
    print(f"  Events received:     {total_events}")
    print(f"  Total errors:        {total_errors}")
    print(f"")

    if connect_times:
        print(f"  Connect time avg:    {round(statistics.mean(connect_times), 1)}ms")
        print(f"  Connect time max:    {round(max(connect_times), 1)}ms")

    if all_latencies:
        print(f"")
        print(f"  Response latency:")
        print(f"    avg:  {round(statistics.mean(all_latencies), 1)}ms")
        print(f"    p50:  {round(statistics.median(all_latencies), 1)}ms")
        print(f"    p95:  {round(sorted(all_latencies)[int(len(all_latencies)*0.95)], 1)}ms")
        print(f"    max:  {round(max(all_latencies), 1)}ms")
    else:
        print(f"  No latency data (no events received)")

    print(f"{'='*60}\n")

    # Per-session summary (top 10 by events)
    top = sorted(metrics_list, key=lambda m: m.events_received, reverse=True)[:10]
    if top:
        print(f"  Top sessions by events received:")
        print(f"  {'Session':<25} {'Frames':>7} {'Audio':>6} {'Events':>7} {'Errors':>7} {'AvgLat':>8}")
        print(f"  {'-'*65}")
        for m in top:
            print(
                f"  {m.session_id:<25} {m.frames_sent:>7} {m.audio_sent:>6} "
                f"{m.events_received:>7} {m.errors:>7} {round(m.avg_latency_ms, 1):>7}ms"
            )
        print()


def main():
    parser = argparse.ArgumentParser(description="Load test the AI Proctoring Backend")
    parser.add_argument("--url", default="ws://localhost:8000", help="Backend WebSocket base URL")
    parser.add_argument("--sessions", type=int, default=10, help="Number of concurrent sessions")
    parser.add_argument("--duration", type=int, default=30, help="Duration per session (seconds)")
    parser.add_argument("--concurrency", type=int, default=None, help="Max concurrent connections (default: all)")
    parser.add_argument("--frame-interval", type=float, default=1.0, help="Seconds between frames")
    parser.add_argument("--audio-interval", type=float, default=4.0, help="Seconds between audio chunks")
    parser.add_argument("--report", action="store_true", default=True, help="Print detailed report")
    args = parser.parse_args()

    concurrency = args.concurrency or args.sessions

    metrics_list, total_elapsed, gather_errors = asyncio.run(
        run_load_test(
            base_url=args.url,
            num_sessions=args.sessions,
            duration=args.duration,
            concurrency=concurrency,
            frame_interval=args.frame_interval,
            audio_interval=args.audio_interval,
        )
    )

    if args.report:
        print_report(metrics_list, total_elapsed, gather_errors)


if __name__ == "__main__":
    main()
