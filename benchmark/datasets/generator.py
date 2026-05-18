"""
benchmark/datasets/generator.py
Generates synthetic test datasets for benchmarking.

Creates:
- Video frames (normal, suspicious, multi-person, mobile, low-light, blurred)
- Audio WAV files (silence, speech-like, noisy)
"""

from __future__ import annotations

import math
import random
import struct
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

DATASETS_DIR = Path(__file__).parent / "generated"


def ensure_dir() -> Path:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return DATASETS_DIR


# ── VIDEO FRAME GENERATORS ────────────────────────────────────────────────

def make_normal_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Single person, centered, good lighting."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (80, 70, 60)
    cx, cy = width // 2, height // 2
    # Face
    cv2.ellipse(frame, (cx, cy - 60), (70, 85), 0, 0, 360, (190, 155, 120), -1)
    # Eyes
    cv2.circle(frame, (cx - 22, cy - 75), 10, (40, 30, 20), -1)
    cv2.circle(frame, (cx + 22, cy - 75), 10, (40, 30, 20), -1)
    # Body
    cv2.rectangle(frame, (cx - 80, cy + 25), (cx + 80, height), (60, 80, 120), -1)
    # Hair
    cv2.ellipse(frame, (cx, cy - 100), (75, 45), 0, 180, 360, (30, 20, 15), -1)
    return frame


def make_no_face_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Empty room — no person."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (100, 90, 80)
    # Draw a desk
    cv2.rectangle(frame, (50, height - 100), (width - 50, height - 60), (120, 100, 80), -1)
    return frame


def make_multiple_persons_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Two people in frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (80, 70, 60)
    for cx in [width // 3, 2 * width // 3]:
        cy = height // 2 - 40
        cv2.ellipse(frame, (cx, cy), (55, 70), 0, 0, 360, (190, 155, 120), -1)
        cv2.rectangle(frame, (cx - 60, cy + 70), (cx + 60, height), (60, 80, 120), -1)
    return frame


def make_mobile_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Person holding a phone."""
    frame = make_normal_frame(width, height)
    # Draw phone
    cx = width // 2 + 120
    cy = height // 2
    cv2.rectangle(frame, (cx - 15, cy - 30), (cx + 15, cy + 30), (20, 20, 20), -1)
    cv2.rectangle(frame, (cx - 12, cy - 25), (cx + 12, cy + 25), (60, 120, 180), -1)
    return frame


def make_looking_away_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Person looking to the side."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (80, 70, 60)
    cx, cy = width // 4, height // 2 - 40
    cv2.ellipse(frame, (cx, cy), (65, 80), 0, 0, 360, (190, 155, 120), -1)
    cv2.rectangle(frame, (cx - 70, cy + 80), (cx + 70, height), (60, 80, 120), -1)
    return frame


def make_low_light_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Dark frame — low light conditions."""
    frame = make_normal_frame(width, height)
    dark = np.zeros_like(frame, dtype=np.uint8)
    frame = cv2.addWeighted(frame, 0.25, dark, 0.75, 0)
    return frame


def make_blurred_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Motion-blurred frame."""
    frame = make_normal_frame(width, height)
    return cv2.GaussianBlur(frame, (21, 21), 0)


def make_noisy_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Frame with heavy noise."""
    frame = make_normal_frame(width, height)
    noise = np.random.randint(0, 80, frame.shape, dtype=np.uint8)
    return cv2.add(frame, noise)


def make_headset_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Person wearing headset."""
    frame = make_normal_frame(width, height)
    cx, cy = width // 2, height // 2 - 60
    # Headset band
    cv2.ellipse(frame, (cx, cy - 30), (85, 50), 0, 180, 360, (30, 30, 30), 4)
    # Ear cups
    cv2.circle(frame, (cx - 85, cy - 30), 18, (20, 20, 20), -1)
    cv2.circle(frame, (cx + 85, cy - 30), 18, (20, 20, 20), -1)
    return frame


FRAME_TYPES = {
    "normal": make_normal_frame,
    "no_face": make_no_face_frame,
    "multiple_persons": make_multiple_persons_frame,
    "mobile": make_mobile_frame,
    "looking_away": make_looking_away_frame,
    "low_light": make_low_light_frame,
    "blurred": make_blurred_frame,
    "noisy": make_noisy_frame,
    "headset": make_headset_frame,
}


def generate_frame_bytes(frame_type: str = "normal", quality: int = 80) -> bytes:
    """Generate a JPEG frame of the given type."""
    fn = FRAME_TYPES.get(frame_type, make_normal_frame)
    frame = fn()
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def generate_all_frame_types(output_dir: Path = None) -> Dict[str, bytes]:
    """Generate one frame of each type and return as dict."""
    result = {}
    for name, fn in FRAME_TYPES.items():
        frame = fn()
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        result[name] = buf.tobytes()
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / f"{name}.jpg"), frame)
    return result


# ── AUDIO GENERATORS ──────────────────────────────────────────────────────

def _make_wav_header(num_samples: int, sample_rate: int = 16000) -> bytes:
    data_size = num_samples * 2
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", data_size,
    )


def make_silence_wav(duration_s: float = 3.0, sample_rate: int = 16000) -> bytes:
    n = int(sample_rate * duration_s)
    samples = [0] * n
    return _make_wav_header(n) + struct.pack(f"<{n}h", *samples)


def make_speech_like_wav(duration_s: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Simulate speech-like audio with varying frequency."""
    n = int(sample_rate * duration_s)
    samples = []
    for i in range(n):
        t = i / sample_rate
        # Mix of frequencies to simulate speech formants
        freq = 200 + 100 * math.sin(2 * math.pi * 3 * t)
        val = int(8000 * math.sin(2 * math.pi * freq * t) *
                  (0.5 + 0.5 * math.sin(2 * math.pi * 4 * t)))
        samples.append(max(-32767, min(32767, val)))
    return _make_wav_header(n) + struct.pack(f"<{n}h", *samples)


def make_noisy_wav(duration_s: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Audio with background noise."""
    n = int(sample_rate * duration_s)
    samples = [random.randint(-3000, 3000) for _ in range(n)]
    return _make_wav_header(n) + struct.pack(f"<{n}h", *samples)


def make_keyword_wav(duration_s: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Simulate audio that might trigger keyword detection (tone pattern)."""
    n = int(sample_rate * duration_s)
    samples = []
    for i in range(n):
        t = i / sample_rate
        val = int(15000 * math.sin(2 * math.pi * 440 * t) *
                  math.exp(-0.5 * ((t - duration_s / 2) ** 2)))
        samples.append(max(-32767, min(32767, val)))
    return _make_wav_header(n) + struct.pack(f"<{n}h", *samples)


AUDIO_TYPES = {
    "silence": make_silence_wav,
    "speech_like": make_speech_like_wav,
    "noisy": make_noisy_wav,
    "keyword": make_keyword_wav,
}


def generate_audio_bytes(audio_type: str = "speech_like", duration_s: float = 3.0) -> bytes:
    fn = AUDIO_TYPES.get(audio_type, make_speech_like_wav)
    return fn(duration_s=duration_s)


def generate_dataset(output_dir: Path = None) -> dict:
    """Generate the full benchmark dataset."""
    if output_dir is None:
        output_dir = ensure_dir()

    frames = generate_all_frame_types(output_dir / "frames")
    audios = {}
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in AUDIO_TYPES.items():
        data = fn(duration_s=3.0)
        audios[name] = data
        (audio_dir / f"{name}.wav").write_bytes(data)

    return {"frames": frames, "audios": audios}
