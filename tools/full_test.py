"""
full_test.py — End-to-end validation of all AI Proctoring Backend endpoints.
Run: python tools/full_test.py
"""
import asyncio
import base64
import json
import math
import struct
import sys
import time

import cv2
import numpy as np
import requests
import websockets

BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append(condition)
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")


def make_jpeg(width=320, height=240):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (60, 60, 80)
    cv2.circle(frame, (width // 2, height // 2), 60, (200, 180, 160), -1)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


def make_wav(duration_s=2.0, sr=16000):
    n = int(sr * duration_s)
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n)]
    data_size = n * 2
    hdr = struct.pack("<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16,
        b"data", data_size)
    return hdr + struct.pack(f"<{n}h", *samples)


# ── TEST 1: Health ────────────────────────────────────────────────────────
print("\n=== 1. Health Endpoint ===")
r = requests.get(f"{BASE}/health", timeout=5)
check("HTTP 200", r.status_code == 200)
body = r.json()
check("status=ok", body.get("status") == "ok")
check("yolo loaded", body.get("models", {}).get("yolo") is True)
check("headset loaded", body.get("models", {}).get("headset") is True)
check("proctor loaded", body.get("models", {}).get("proctor") is True)
check("stt loaded", body.get("models", {}).get("stt") is True)
check("uptime_seconds present", "uptime_seconds" in body)

# ── TEST 2: Video Frame ───────────────────────────────────────────────────
print("\n=== 2. Video Frame Endpoint ===")
jpeg = make_jpeg()
r = requests.post(f"{BASE}/api/v1/video/analyze-frame",
    files={"frame": ("f.jpg", jpeg, "image/jpeg")},
    data={"session_id": "e2e-test"}, timeout=15)
check("HTTP 200", r.status_code == 200, f"got {r.status_code}")
body = r.json()
check("session_id present", "session_id" in body)
check("events is list", isinstance(body.get("events"), list))
check("person_count present", "person_count" in body)
check("risk_score present", "risk_score" in body)
check("risk_level valid", body.get("risk_level") in ("LOW", "MEDIUM", "HIGH"))
check("detections is list", isinstance(body.get("detections"), list))
print(f"         events={body.get('events')} risk={body.get('risk_level')} persons={body.get('person_count')}")

# ── TEST 3: Video MIME rejection ──────────────────────────────────────────
print("\n=== 3. Video MIME Validation ===")
r = requests.post(f"{BASE}/api/v1/video/analyze-frame",
    files={"frame": ("f.txt", b"notanimage", "text/plain")},
    data={"session_id": "e2e-test"}, timeout=5)
check("HTTP 422 on bad MIME", r.status_code == 422, f"got {r.status_code}")

# ── TEST 4: Audio Endpoint ────────────────────────────────────────────────
print("\n=== 4. Audio Endpoint ===")
wav = make_wav(duration_s=2.0)
r = requests.post(f"{BASE}/api/v1/audio/analyze",
    files={"file": ("test.wav", wav, "audio/wav")},
    data={"session_id": "e2e-test"}, timeout=120)
check("HTTP 200", r.status_code == 200, f"got {r.status_code}")
body = r.json()
check("session_id present", "session_id" in body)
check("rule_flags is list", isinstance(body.get("rule_flags"), list))
check("risk_score present", "risk_score" in body)
check("risk_level valid", body.get("risk_level") in ("LOW", "MEDIUM", "HIGH"))
print(f"         native_text='{str(body.get('native_text',''))[:40]}' risk={body.get('risk_level')}")

# ── TEST 5: Audio MIME rejection ──────────────────────────────────────────
print("\n=== 5. Audio MIME Validation ===")
r = requests.post(f"{BASE}/api/v1/audio/analyze",
    files={"file": ("f.mp4", b"notaudio", "video/mp4")},
    data={"session_id": "e2e-test"}, timeout=5)
check("HTTP 422 on bad MIME", r.status_code == 422, f"got {r.status_code}")

# ── TEST 6: Frontend Static ───────────────────────────────────────────────
print("\n=== 6. Frontend Static Files ===")
r = requests.get(f"{BASE}/ui/", timeout=5)
check("HTTP 200", r.status_code == 200)
check("Contains HTML", "<!DOCTYPE html>" in r.text or "<html" in r.text.lower())

# ── TEST 7: WebSocket ─────────────────────────────────────────────────────
print("\n=== 7. WebSocket Endpoint ===")

async def ws_test():
    url = f"{WS_BASE}/ws/proctor/e2e-ws-test"
    try:
        async with websockets.connect(url, ping_interval=None, open_timeout=10) as ws:
            # Send a proper JPEG frame
            jpeg = make_jpeg(320, 240)
            b64 = base64.b64encode(jpeg).decode()
            t0 = time.monotonic()
            await ws.send(json.dumps({"type": "video", "frame": b64}))

            # Wait up to 12s for result
            result = None
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2)
                    msg = json.loads(raw)
                    if msg.get("type") == "ping":
                        await ws.send(json.dumps({"type": "pong"}))
                        continue
                    result = msg
                    break
                except asyncio.TimeoutError:
                    continue

            latency = round((time.monotonic() - t0) * 1000, 1)
            return result, latency
    except Exception as e:
        return None, str(e)

result, latency = asyncio.run(ws_test())
check("WebSocket connected and received result", result is not None, f"latency={latency}ms")
if result:
    check("WS result has events", "events" in result, str(result)[:80])
    check("WS result has risk_level", "risk_level" in result)
    print(f"         events={result.get('events')} risk={result.get('risk_level')} latency={latency}ms")

# ── SUMMARY ───────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
passed = sum(results)
total = len(results)
print(f"  Results: {passed}/{total} checks passed")
if passed == total:
    print(f"  \033[92mAll tests PASSED ✓\033[0m")
else:
    print(f"  \033[91m{total-passed} tests FAILED\033[0m")
print(f"{'='*50}\n")
sys.exit(0 if passed == total else 1)
