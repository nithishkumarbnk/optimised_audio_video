"""Test bounding boxes with a more realistic frame."""
import requests
import numpy as np
import cv2

# Create a more realistic-looking frame with skin tones and structure
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Background - room-like
frame[:] = (120, 100, 80)

# Person body area
cv2.rectangle(frame, (220, 200), (420, 480), (80, 70, 60), -1)

# Face - skin tone ellipse
cv2.ellipse(frame, (320, 150), (80, 100), 0, 0, 360, (180, 140, 110), -1)

# Eyes
cv2.circle(frame, (290, 130), 15, (40, 30, 20), -1)
cv2.circle(frame, (350, 130), 15, (40, 30, 20), -1)
cv2.circle(frame, (290, 130), 8, (200, 200, 220), -1)
cv2.circle(frame, (350, 130), 8, (200, 200, 220), -1)

# Nose
cv2.ellipse(frame, (320, 165), (12, 8), 0, 0, 360, (160, 120, 90), -1)

# Mouth
cv2.ellipse(frame, (320, 195), (25, 10), 0, 0, 180, (120, 80, 70), 2)

# Hair
cv2.ellipse(frame, (320, 100), (85, 60), 0, 180, 360, (40, 30, 20), -1)

# Shirt
cv2.rectangle(frame, (230, 250), (410, 480), (60, 80, 140), -1)

# Add some noise to make it look more realistic
noise = np.random.randint(0, 20, frame.shape, dtype=np.uint8)
frame = cv2.add(frame, noise)

_, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

r = requests.post(
    'http://localhost:8000/api/v1/video/analyze-frame',
    files={'frame': ('face.jpg', buf.tobytes(), 'image/jpeg')},
    data={'session_id': 'bbox-real-test'},
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
    x1, y1, x2, y2 = bb.get('x1',0), bb.get('y1',0), bb.get('x2',0), bb.get('y2',0)
    print(f'  {name} conf={conf:.3f} box=[{x1:.0f},{y1:.0f} -> {x2:.0f},{y2:.0f}]')
if not body.get('detections'):
    print('  (no detections - YOLO needs real webcam frames for best results)')
    print('  The bounding box overlay is ready in the frontend at http://localhost:8000/ui/')

# Save the test frame so user can see what was sent
cv2.imwrite('tools/test_frame.jpg', frame)
print('\nTest frame saved to tools/test_frame.jpg')
