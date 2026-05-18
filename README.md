# AI Proctoring Backend

A production-grade FastAPI service that monitors exam candidates in real time. It ingests webcam video frames and microphone audio via REST endpoints and a WebSocket streaming interface, runs ONNX-based computer vision models and a multilingual speech-to-text pipeline, and returns structured risk assessments to connected clients.

---

## Overview

The backend is designed to handle 100+ concurrent proctoring sessions. Key capabilities:

- **Video analysis** — YOLOv8 object detection, headset classification, and behavioral analysis via ONNX models
- **Audio analysis** — Multilingual speech-to-text (AI4Bharat IndicConformer) followed by keyword rule matching and LLM-based risk assessment via OpenRouter
- **WebSocket streaming** — Low-latency bidirectional channel for continuous frame/audio submission with server-push risk events
- **Risk scoring** — Weighted event scoring with session-level cumulative tracking
- **Async worker queues** — ML inference is decoupled from I/O to prevent head-of-line blocking

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FastAPI (API Server)                        │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐  │
│  │ Audio Router │  │ Video Router │  │  WS Router   │  │ Health │  │
│  │/api/v1/audio │  │/api/v1/video │  │  /ws/proctor │  │/health │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────────┘  │
│         │                 │                  │                       │
│         ▼                 ▼                  ▼                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Security Middleware                              │   │
│  │       (CORS · Body-Size Limit · Origin Validation)           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐   │
│  │   Worker Manager     │    │        WebSocket Manager         │   │
│  │  Audio Queue (async) │    │  Connection Registry             │   │
│  │  Video Queue (async) │    │  Result Queues per connection    │   │
│  │  N audio workers     │    │  Heartbeat ping every 30s        │   │
│  │  N video workers     │    └──────────────────────────────────┘   │
│  └──────────────────────┘                                            │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                        Services Layer                         │   │
│  │  STT Service · LLM Service · YOLO · Headset · Proctor        │   │
│  │  Rule Engine · Risk Engine · Session Manager                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
  OpenRouter API             ONNX model files
  (LLM inference)            (yolov8n.onnx,
                              headset_model.onnx,
                              proctoring.onnx)
```

---

## Prerequisites

- Python 3.11+
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/) (for containerised deployment)
- An [OpenRouter](https://openrouter.ai) API key
- ONNX model files placed in `app/models/`:
  - `yolov8n.onnx`
  - `headset_model.onnx`
  - `proctoring.onnx`

---

## Local Setup

```bash
# 1. Clone the repository and enter the project directory
cd project/

# 2. Create and activate a virtual environment
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY (and any other values you want to override)

# 5. Start the development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

---

## Running with Docker

```bash
# 1. Configure environment variables
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY

# 2. Build and start the container
docker compose up --build

# Run in the background
docker compose up --build -d

# View logs
docker compose logs -f

# Stop the container
docker compose down
```

The service starts on port `8000`. The health endpoint is polled automatically by Docker's healthcheck every 30 seconds.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | **Yes** | — | API key for the OpenRouter LLM gateway. The application will not start without this. |
| `LOG_LEVEL` | No | `INFO` | Root logger level. One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `MAX_AUDIO_SIZE_MB` | No | `20` | Maximum audio upload size in MB. Requests exceeding this are rejected with HTTP 413. |
| `YOLO_CONFIDENCE` | No | `0.4` | YOLO detection confidence threshold (0.0–1.0). Lower values increase recall but also false positives. |
| `ENABLE_GPU` | No | `false` | Use CUDA for ONNX Runtime inference. Requires `onnxruntime-gpu` and a CUDA-capable GPU. |
| `WORKER_POOL_SIZE` | No | `4` | Number of async worker coroutines for each of the audio and video queues. |
| `CORS_ORIGINS` | No | `["*"]` | JSON array of allowed CORS origins. Use `["*"]` for development only. |

---

## API Endpoints

### POST /api/v1/audio/analyze

Transcribes an audio file, applies keyword rule matching, and runs LLM-based risk assessment.

**Request** — multipart/form-data

| Field | Type | Description |
|---|---|---|
| `file` | file | Audio file (WAV, MP3, M4A, OGG, FLAC). Max size controlled by `MAX_AUDIO_SIZE_MB`. |
| `session_id` | string | Unique identifier for the proctoring session. |

```bash
curl -X POST http://localhost:8000/api/v1/audio/analyze \
  -F "file=@recording.wav" \
  -F "session_id=exam-session-001"
```

**Response** — `200 OK`

```json
{
  "session_id": "exam-session-001",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "native_text": "what is the answer to question five",
  "translated_text": "what is the answer to question five",
  "rule_flags": ["answer", "question"],
  "llm_result": {
    "translated_text": "what is the answer to question five",
    "risk": "high",
    "confidence": 0.92,
    "reason": "Candidate is asking for exam answers"
  },
  "risk_score": 75,
  "risk_level": "HIGH"
}
```

---

### POST /api/v1/video/analyze-frame

Runs YOLO object detection, headset classification, and behavioral analysis on a single video frame.

**Request** — multipart/form-data

| Field | Type | Description |
|---|---|---|
| `file` | file | Image file (JPEG, PNG). |
| `session_id` | string | Unique identifier for the proctoring session. |

```bash
curl -X POST http://localhost:8000/api/v1/video/analyze-frame \
  -F "file=@frame.jpg" \
  -F "session_id=exam-session-001"
```

**Response** — `200 OK`

```json
{
  "session_id": "exam-session-001",
  "timestamp": "2024-01-15T10:30:05.000Z",
  "events": ["multiple_persons", "looking_away"],
  "person_count": 2,
  "risk_score": 60,
  "risk_level": "HIGH",
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.87,
      "bounding_box": { "x1": 120.0, "y1": 80.0, "x2": 400.0, "y2": 600.0 }
    },
    {
      "class_name": "cell phone",
      "confidence": 0.73,
      "bounding_box": { "x1": 310.0, "y1": 200.0, "x2": 360.0, "y2": 290.0 }
    }
  ]
}
```

---

### GET /health

Returns the current health status of the service, including model load status and worker pool state.

```bash
curl http://localhost:8000/health
```

**Response** — `200 OK`

```json
{
  "status": "healthy",
  "uptime_seconds": 3600.5,
  "models": {
    "yolo": true,
    "headset": true,
    "proctor": true,
    "stt": true
  },
  "workers": {
    "audio_workers": 4,
    "video_workers": 4,
    "audio_queue_size": 0,
    "video_queue_size": 0
  }
}
```

---

## WebSocket Usage

Connect to `/ws/proctor/{session_id}` to stream video frames and audio continuously. The server pushes risk event messages back as they are processed.

```javascript
const sessionId = "exam-session-001";
const ws = new WebSocket(`ws://localhost:8000/ws/proctor/${sessionId}`);

ws.onopen = () => {
  console.log("Connected to proctoring session:", sessionId);
};

// Send a video frame (base64-encoded image)
function sendVideoFrame(base64ImageData) {
  ws.send(JSON.stringify({
    type: "video",
    frame: base64ImageData   // base64-encoded JPEG or PNG
  }));
}

// Send an audio chunk (base64-encoded audio bytes)
function sendAudioChunk(base64AudioData) {
  ws.send(JSON.stringify({
    type: "audio",
    audio: base64AudioData   // base64-encoded WAV bytes
  }));
}

// Receive risk event messages from the server
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  // message shape: { event, risk_level, timestamp, session_id, details }
  console.log("Risk event:", message.event, "Level:", message.risk_level);

  if (message.risk_level === "HIGH") {
    // Trigger proctor alert
  }
};

ws.onclose = () => {
  console.log("Disconnected from proctoring session");
};

ws.onerror = (error) => {
  console.error("WebSocket error:", error);
};

// Example: capture and send a frame every second
setInterval(() => {
  const canvas = document.getElementById("webcam-canvas");
  const base64 = canvas.toDataURL("image/jpeg").split(",")[1];
  sendVideoFrame(base64);
}, 1000);
```

The server sends a WebSocket ping every 30 seconds to keep the connection alive. Clients should respond with a pong (most browser WebSocket implementations do this automatically).

---

## Project Structure

```
project/
├── app/
│   ├── main.py                  # FastAPI app factory, startup/shutdown hooks
│   ├── core/
│   │   ├── config.py            # Pydantic Settings — all env vars
│   │   ├── logger.py            # Structured logging setup
│   │   └── security.py          # Filename sanitisation, origin validation
│   ├── routers/
│   │   ├── audio.py             # POST /api/v1/audio/analyze
│   │   ├── video.py             # POST /api/v1/video/analyze-frame
│   │   ├── websocket.py         # WS /ws/proctor/{session_id}
│   │   └── health.py            # GET /health
│   ├── services/
│   │   ├── stt_service.py       # IndicConformer speech-to-text singleton
│   │   ├── llm_service.py       # OpenRouter LLM client with retry
│   │   ├── yolo_service.py      # YOLOv8 ONNX inference singleton
│   │   ├── headset_service.py   # Headset classifier ONNX singleton
│   │   ├── proctor_service.py   # Behavioral model ONNX singleton
│   │   ├── risk_engine.py       # Weighted risk scoring
│   │   ├── rule_engine.py       # Keyword-based rule matching
│   │   ├── session_manager.py   # In-memory session store (asyncio-safe)
│   │   ├── worker_manager.py    # Async audio/video worker queues
│   │   └── websocket_manager.py # WebSocket connection registry + heartbeat
│   ├── schemas/
│   │   ├── audio_schema.py      # AudioAnalysisRequest/Response, LLMResult
│   │   ├── video_schema.py      # VideoAnalysisResponse, Detection, BoundingBox
│   │   └── websocket_schema.py  # WSIncomingMessage, WSEventMessage
│   ├── utils/
│   │   ├── audio_utils.py       # Audio I/O helpers
│   │   ├── image_utils.py       # OpenCV decode/preprocess helpers
│   │   └── response_utils.py    # Consistent response shape builders
│   └── models/
│       ├── yolov8n.onnx         # YOLOv8n object detection model
│       ├── headset_model.onnx   # Headset binary classifier
│       └── proctoring.onnx      # Behavioral analysis model
├── tests/
│   ├── test_llm_service.py
│   ├── test_rule_engine.py
│   └── test_yolo_service.py
├── logs/                        # Runtime log files (gitignored, created at startup)
├── uploads/                     # Temporary audio uploads (gitignored, created at startup)
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

---

## Deployment Notes

**Single-worker design** — The container runs one uvicorn worker (`--workers 1`). This is intentional: ONNX models are loaded as in-process singletons and would be duplicated in RAM for each additional worker process. Scale horizontally by running multiple containers behind a load balancer instead.

**Model files** — ONNX model files are mounted read-only from the host (`./app/models:/app/app/models:ro`). They are not baked into the image, which keeps the image size manageable and allows model updates without rebuilding.

**GPU inference** — Set `ENABLE_GPU=true` in `.env` and replace `onnxruntime` with `onnxruntime-gpu` in `requirements.txt`. The Docker host must have the NVIDIA Container Toolkit installed and the container must be granted GPU access via `deploy.resources.reservations.devices` in `docker-compose.yml`.

**Persistent storage** — `logs/` and `uploads/` are bind-mounted from the host. Ensure the host directories are writable by the container process (UID 0 by default for the slim image).

**CORS** — The default `CORS_ORIGINS=["*"]` is suitable for development. In production, set this to the exact origin(s) of your frontend application, e.g. `CORS_ORIGINS=["https://proctor.example.com"]`.

**Health checks** — The `/health` endpoint is used by Docker's built-in healthcheck. It can also be wired into a load balancer or orchestrator (Kubernetes liveness/readiness probes) to automatically restart unhealthy instances.

**Secrets** — Never commit `.env` to version control. Use a secrets manager (AWS Secrets Manager, HashiCorp Vault, Docker Secrets) to inject `OPENROUTER_API_KEY` in production environments.
