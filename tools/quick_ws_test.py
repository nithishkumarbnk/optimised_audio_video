"""Quick WebSocket latency test — 3 frames, measures round-trip time."""
import asyncio, json, base64, time
import numpy as np
import cv2
import websockets


async def quick_test():
    url = "ws://localhost:8000/ws/proctor/load-test-001"
    results = []
    async with websockets.connect(url, ping_interval=None, open_timeout=10) as ws:
        for i in range(3):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:] = (i * 30, 60, 80)
            cv2.circle(frame, (160, 120), 60, (200, 180, 160), -1)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            b64 = base64.b64encode(buf.tobytes()).decode()
            t0 = time.monotonic()
            await ws.send(json.dumps({"type": "video", "frame": b64}))
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10)
                data = json.loads(msg)
                if data.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                    msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    data = json.loads(msg)
                lat = round((time.monotonic() - t0) * 1000, 1)
                results.append({
                    "frame": i + 1,
                    "latency_ms": lat,
                    "events": data.get("events", []),
                    "risk": data.get("risk_level"),
                    "person_count": data.get("person_count"),
                })
            except asyncio.TimeoutError:
                results.append({"frame": i + 1, "latency_ms": "timeout", "events": [], "risk": None})
            await asyncio.sleep(1.1)

    print("\n=== WebSocket Latency Test Results ===")
    for r in results:
        print(f"  Frame {r['frame']}: latency={r['latency_ms']}ms  events={r['events']}  risk={r['risk']}  persons={r.get('person_count')}")
    print("======================================\n")


if __name__ == "__main__":
    asyncio.run(quick_test())
