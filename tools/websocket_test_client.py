"""
websocket_test_client.py — Interactive WebSocket test client for the AI Proctoring Backend.

Usage:
    python tools/websocket_test_client.py --session test-001 --url ws://localhost:8000
    python tools/websocket_test_client.py --session test-001 --send-frames --send-audio
"""

import asyncio
import base64
import json
import sys
import time
import argparse
import logging
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)

try:
    import numpy as np
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def make_fake_frame(width: int = 320, height: int = 240) -> str:
    """Generate a fake JPEG frame as base64."""
    if HAS_CV2:
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        # Draw a fake face circle
        cv2.circle(frame, (width // 2, height // 2), 60, (200, 180, 160), -1)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode()
    else:
        # Minimal valid JPEG (1x1 white pixel)
        minimal_jpeg = (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
            b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
            b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e'
            b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
            b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
            b'\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04'
            b'\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa'
            b'\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br'
            b'\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJ'
            b'STUVWXYZ\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00\x00\x00\xff\xd9'
        )
        return base64.b64encode(minimal_jpeg).decode()


def make_fake_audio(duration_ms: int = 1000, sample_rate: int = 16000) -> str:
    """Generate a fake WAV audio chunk as base64."""
    import struct
    import math

    num_samples = int(sample_rate * duration_ms / 1000)
    # Generate a simple sine wave at 440 Hz
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(num_samples)]

    # WAV header
    data_size = num_samples * 2
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size
    )
    audio_data = struct.pack(f'<{num_samples}h', *samples)
    wav_bytes = header + audio_data
    return base64.b64encode(wav_bytes).decode()


async def run_client(
    url: str,
    session_id: str,
    send_frames: bool = True,
    send_audio: bool = True,
    duration: int = 30,
    frame_interval: float = 1.0,
    audio_interval: float = 4.0,
):
    """Run a single WebSocket test client."""
    ws_url = f"{url}/ws/proctor/{session_id}"
    logger.info(f"Connecting to {ws_url}")

    events_received = 0
    frames_sent = 0
    audio_sent = 0
    start_time = time.monotonic()

    try:
        async with websockets.connect(ws_url, ping_interval=None) as ws:
            logger.info(f"Connected! Session: {session_id}")

            async def receive_loop():
                nonlocal events_received
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get('type') == 'ping':
                            await ws.send(json.dumps({'type': 'pong'}))
                            continue
                        events_received += 1
                        risk = msg.get('risk_level', msg.get('risk_score', '?'))
                        events = msg.get('events', [])
                        native = msg.get('native_text', '')
                        logger.info(
                            f"[{session_id}] Event #{events_received} | "
                            f"risk={risk} | events={events} | text={str(native)[:40]}"
                        )
                except websockets.exceptions.ConnectionClosed:
                    pass

            async def send_loop():
                nonlocal frames_sent, audio_sent
                last_frame = 0.0
                last_audio = 0.0

                while time.monotonic() - start_time < duration:
                    now = time.monotonic()

                    if send_frames and now - last_frame >= frame_interval:
                        frame_b64 = make_fake_frame()
                        await ws.send(json.dumps({'type': 'video', 'frame': frame_b64}))
                        frames_sent += 1
                        last_frame = now
                        logger.debug(f"[{session_id}] Frame #{frames_sent} sent")

                    if send_audio and now - last_audio >= audio_interval:
                        audio_b64 = make_fake_audio(duration_ms=3000)
                        await ws.send(json.dumps({'type': 'audio', 'audio': audio_b64}))
                        audio_sent += 1
                        last_audio = now
                        logger.info(f"[{session_id}] Audio chunk #{audio_sent} sent")

                    await asyncio.sleep(0.1)

            await asyncio.gather(receive_loop(), send_loop())

    except Exception as e:
        logger.error(f"[{session_id}] Error: {e}")

    elapsed = round(time.monotonic() - start_time, 1)
    logger.info(
        f"[{session_id}] Done | elapsed={elapsed}s | "
        f"frames={frames_sent} | audio={audio_sent} | events_received={events_received}"
    )
    return {"session_id": session_id, "frames": frames_sent, "audio": audio_sent, "events": events_received}


def main():
    parser = argparse.ArgumentParser(description="WebSocket test client for AI Proctoring Backend")
    parser.add_argument("--url", default="ws://localhost:8000", help="Backend WebSocket base URL")
    parser.add_argument("--session", default="test-session-001", help="Session ID")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--send-frames", action="store_true", default=True, help="Send video frames")
    parser.add_argument("--send-audio", action="store_true", default=True, help="Send audio chunks")
    parser.add_argument("--no-frames", action="store_true", help="Disable frame sending")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio sending")
    args = parser.parse_args()

    asyncio.run(run_client(
        url=args.url,
        session_id=args.session,
        send_frames=not args.no_frames,
        send_audio=not args.no_audio,
        duration=args.duration,
    ))


if __name__ == "__main__":
    main()
