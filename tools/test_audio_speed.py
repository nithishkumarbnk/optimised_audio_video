"""Measure audio transcription speed."""
import requests, struct, math, time

sr = 16000
n = sr * 4  # 4 second audio
samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n)]
data_size = n * 2
hdr = struct.pack('<4sI4s4sIHHIIHH4sI',
    b'RIFF', 36+data_size, b'WAVE', b'fmt ', 16, 1, 1,
    sr, sr*2, 2, 16, b'data', data_size)
wav = hdr + struct.pack(f'<{n}h', *samples)

print("Sending 4s audio chunk...")
t0 = time.time()
r = requests.post('http://localhost:8000/api/v1/audio/analyze',
    files={'file': ('test.wav', wav, 'audio/wav')},
    data={'session_id': 'speed-test'}, timeout=120)
elapsed = round(time.time() - t0, 1)
body = r.json()

native = str(body.get('native_text', ''))[:50]
print(f"Total time:      {elapsed}s")
print(f"STT elapsed:     {body.get('stt_elapsed_ms')}ms")
print(f"native_text:     {native!r}")
print(f"risk_level:      {body.get('risk_level')}")
print(f"rule_flags:      {body.get('rule_flags')}")
