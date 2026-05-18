"""Test bounding box detection in video response."""
import requests
import numpy as np
import cv2
import json

# Create a frame with a clear person-like shape
frame = np.zeros((480, 640, 3), dtype=np.uint8)
frame[:] = (40, 40, 60)
cv2.ellipse(frame, (320, 180), (70, 90), 0, 0, 360, (200, 180, 160), -1)  # head
cv2.rectangle(frame, (260, 270), (380, 450), (100, 80, 60), -1)  # body
_, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

r = requests.post(
    'http://localhost:8000/api/v1/video/analyze-frame',
    files={'frame': ('f.jpg', buf.tobytes(), 'image/jpeg')},
    data={'session_id': 'bbox-test'},
    timeout=15
)
body = r.json()
print('Status:', r.status_code)
print('Events:', body.get('events'))
print('Persons:', body.get('person_count'))
print('Risk:', body.get('risk_level'))
print('Detections:')
for d in body.get('detections', []):
    bb = d.get('bounding_box', {})
    name = d.get('class_name', '?')
    conf = d.get('confidence', 0)
    x1 = bb.get('x1', 0)
    y1 = bb.get('y1', 0)
    x2 = bb.get('x2', 0)
    y2 = bb.get('y2', 0)
    print(f'  {name} conf={conf:.2f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]')
if not body.get('detections'):
    print('  (no detections above threshold - try lowering YOLO_CONFIDENCE in .env)')
