"""
fake_video_sender.py — Send fake or real image frames to the video analysis REST endpoint.

Usage:
    # Send a generated random frame
    python tools/fake_video_sender.py --session test-001

    # Send a real image file
    python tools/fake_video_sender.py --session test-001 --file path/to/frame.jpg

    # Send repeatedly at 1fps
    python tools/fake_video_sender.py --session test-001 --repeat 10 --delay 1
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

try:
    import numpy as np
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_frame(width: int = 320, height: int = 240) -> bytes:
    """Generate a fake JPEG frame."""
    if HAS_CV2:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background
        frame[:] = (50, 50, 80)
        # Fake face
        cx, cy = width // 2, height // 2
        cv2.ellipse(frame, (cx, cy), (70, 90), 0, 0, 360, (200, 180, 160), -1)
        cv2.circle(frame, (cx - 20, cy - 10), 8, (60, 60, 60), -1)
        cv2.circle(frame, (cx + 20, cy - 10), 8, (60, 60, 60), -1)
        cv2.ellipse(frame, (cx, cy + 20), (25, 12), 0, 0, 180, (150, 100, 100), 2)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()
    else:
        # Minimal valid JPEG
        return (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
            b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
            b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e'
            b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
            b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
            b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00\x00\x00\xff\xd9'
        )


def send_frame(
    base_url: str,
    session_id: str,
    frame_bytes: bytes,
    filename: str = "frame.jpg",
) -> dict:
    """POST a frame to /api/v1/video/analyze-frame."""
    url = f"{base_url}/api/v1/video/analyze-frame"
    files = {"frame": (filename, frame_bytes, "image/jpeg")}
    data = {"session_id": session_id}

    t0 = time.monotonic()
    try:
        resp = requests.post(url, files=files, data=data, timeout=30)
        elapsed = round((time.monotonic() - t0) * 1000, 1)
        logger.info(f"Response: HTTP {resp.status_code} in {elapsed}ms")

        if resp.status_code == 200:
            result = resp.json()
            logger.info(f"  events:       {result.get('events', [])}")
            logger.info(f"  person_count: {result.get('person_count')}")
            logger.info(f"  risk_score:   {result.get('risk_score')}")
            logger.info(f"  risk_level:   {result.get('risk_level')}")
            return result
        else:
            logger.error(f"  Error: {resp.text[:200]}")
            return {"error": resp.text}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Send video frames to AI Proctoring Backend")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--session", default="test-session-001", help="Session ID")
    parser.add_argument("--file", default=None, help="Path to JPEG/PNG image file")
    parser.add_argument("--repeat", type=int, default=1, help="Number of frames to send")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between frames (seconds)")
    parser.add_argument("--width", type=int, default=320, help="Generated frame width")
    parser.add_argument("--height", type=int, default=240, help="Generated frame height")
    args = parser.parse_args()

    if args.file:
        frame_path = Path(args.file)
        if not frame_path.exists():
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        frame_bytes = frame_path.read_bytes()
        filename = frame_path.name
        logger.info(f"Using file: {args.file} ({len(frame_bytes)} bytes)")
    else:
        frame_bytes = generate_frame(args.width, args.height)
        filename = "generated_frame.jpg"
        logger.info(f"Generated {args.width}x{args.height} frame ({len(frame_bytes)} bytes)")

    for i in range(args.repeat):
        logger.info(f"--- Frame {i+1}/{args.repeat} ---")
        # Regenerate each frame if using generated frames (adds variety)
        if not args.file:
            frame_bytes = generate_frame(args.width, args.height)
        send_frame(args.url, args.session, frame_bytes, filename)
        if i < args.repeat - 1:
            time.sleep(args.delay)

    logger.info("Done.")


if __name__ == "__main__":
    main()
