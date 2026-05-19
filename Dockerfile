# ============================================================
# AI Proctoring Backend — Production Docker Image
# ============================================================
# Python 3.12 matches the development environment exactly.
# Uses opencv-python-headless (no X11/display dependencies).
# Models are expected via volume mount at /app/app/models
# (see docker-compose.yml).
# ============================================================

FROM python:3.12-slim

# ── Python runtime flags ──────────────────────────────────
# PYTHONUNBUFFERED: flush stdout/stderr immediately (visible in docker logs)
# PYTHONDONTWRITEBYTECODE: skip .pyc files (saves disk in containers)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── HuggingFace cache isolation ───────────────────────────
# Keeps transformers module cache inside the container filesystem,
# not in the user home directory. Prevents stale cache issues.
ENV HF_MODULES_CACHE=/app/.hf_modules_cache

# ── Working directory ─────────────────────────────────────
WORKDIR /app

# ── System dependencies ───────────────────────────────────
# libglib2.0-0  : required by opencv-python-headless
# libgomp1      : required by ONNX Runtime (OpenMP threading)
# ffmpeg        : required by audio conversion pipeline (webm/mp4 → wav)
# curl          : required by Docker HEALTHCHECK
# build-essential, gcc, g++, python3-dev : required to compile
#                 native Python extensions (webrtcvad, numba, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────
# Copy requirements first so Docker layer cache is reused
# when only source code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────
COPY app/ ./app/
COPY frontend/ ./frontend/

# ── Runtime directories ───────────────────────────────────
# uploads/ : temporary audio/video files from WebSocket clients
# logs/    : structured JSON application logs
# .hf_modules_cache/ : transformers module cache (see HF_MODULES_CACHE above)
RUN mkdir -p uploads logs .hf_modules_cache

# ── Port ─────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────
# start_period: 120s gives the STT model (~15s) + YOLO models (~3s)
# enough time to load before health checks begin.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────
# workers=1 is required: all three vision models are singletons loaded
# once at startup. Multiple workers would each load their own copy,
# multiplying RAM usage (3+ GB per worker).
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
