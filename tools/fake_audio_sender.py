"""
fake_audio_sender.py — Send fake or real audio files to the audio analysis REST endpoint.

Usage:
    # Send a generated sine wave
    python tools/fake_audio_sender.py --session test-001

    # Send a real WAV file
    python tools/fake_audio_sender.py --session test-001 --file path/to/audio.wav

    # Send repeatedly
    python tools/fake_audio_sender.py --session test-001 --repeat 5 --delay 2
"""

import argparse
import base64
import json
import math
import struct
import sys
import time
import logging
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_wav(duration_s: float = 3.0, sample_rate: int = 16000, freq: float = 440.0) -> bytes:
    """Generate a WAV file with a sine wave tone."""
    num_samples = int(sample_rate * duration_s)
    samples = [int(32767 * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(num_samples)]
    data_size = num_samples * 2
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size
    )
    return header + struct.pack(f'<{num_samples}h', *samples)


def send_audio(
    base_url: str,
    session_id: str,
    audio_bytes: bytes,
    filename: str = "test_audio.wav",
) -> dict:
    """POST audio bytes to /api/v1/audio/analyze."""
    url = f"{base_url}/api/v1/audio/analyze"
    files = {"file": (filename, audio_bytes, "audio/wav")}
    data = {"session_id": session_id}

    t0 = time.monotonic()
    try:
        resp = requests.post(url, files=files, data=data, timeout=60)
        elapsed = round((time.monotonic() - t0) * 1000, 1)
        logger.info(f"Response: HTTP {resp.status_code} in {elapsed}ms")

        if resp.status_code == 200:
            result = resp.json()
            logger.info(f"  native_text: {str(result.get('native_text', ''))[:60]}")
            logger.info(f"  risk_score:  {result.get('risk_score')}")
            logger.info(f"  risk_level:  {result.get('risk_level')}")
            logger.info(f"  rule_flags:  {result.get('rule_flags')}")
            return result
        else:
            logger.error(f"  Error: {resp.text[:200]}")
            return {"error": resp.text}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Send audio to AI Proctoring Backend")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--session", default="test-session-001", help="Session ID")
    parser.add_argument("--file", default=None, help="Path to WAV file (default: generate sine wave)")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration of generated audio (seconds)")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to send")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between sends (seconds)")
    args = parser.parse_args()

    if args.file:
        audio_path = Path(args.file)
        if not audio_path.exists():
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        audio_bytes = audio_path.read_bytes()
        filename = audio_path.name
        logger.info(f"Using file: {args.file} ({len(audio_bytes)} bytes)")
    else:
        audio_bytes = generate_wav(duration_s=args.duration)
        filename = "generated_sine.wav"
        logger.info(f"Generated {args.duration}s sine wave ({len(audio_bytes)} bytes)")

    for i in range(args.repeat):
        logger.info(f"--- Send {i+1}/{args.repeat} ---")
        send_audio(args.url, args.session, audio_bytes, filename)
        if i < args.repeat - 1:
            time.sleep(args.delay)

    logger.info("Done.")


if __name__ == "__main__":
    main()
