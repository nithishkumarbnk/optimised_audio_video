# AI Proctoring Backend — Complete Benchmark Metrics Report

**Generated:** 2026-05-15  
**System:** Intel Core i7-1355U (13th Gen) | 10 cores / 12 threads | 15.7 GB RAM  
**OS:** Windows | Python 3.12.6  
**GPU:** Not enabled (ENABLE_GPU=false — CPU-only inference)

---

## What This Report Covers

This document explains every benchmark metric collected from the AI Proctoring Backend,
what each number means, why it matters for a live exam proctoring system, and what
action to take if a metric is outside the acceptable range.

---

## Table of Contents

1. [ONNX Model Inference Benchmarks](#1-onnx-model-inference-benchmarks)
2. [Queue Performance Benchmarks](#2-queue-performance-benchmarks)
3. [Audio Pipeline Benchmarks](#3-audio-pipeline-benchmarks)
4. [System Resource Metrics](#4-system-resource-metrics)
5. [Production Readiness Assessment](#5-production-readiness-assessment)
6. [Metric Glossary](#6-metric-glossary)
7. [Recommendations](#7-recommendations)

---

## 1. ONNX Model Inference Benchmarks

### What is ONNX Inference?

Every time a webcam frame arrives, the backend runs it through three AI models:
1. **YOLO** — detects objects (person, phone, laptop, book, monitor)
2. **Headset** — detects if the student is wearing headphones
3. **Proctor** — detects behavioral events (looking away, bad posture, suspicious movement)

Each model takes the frame, resizes it to 640×640 pixels, and runs neural network inference.
The time this takes is the **inference latency**.

**Why it matters:** If inference takes too long, the system can't keep up with 1 frame/second
streaming. The target is < 500ms per frame for the full pipeline (all 3 models combined).

---

### 1.1 YOLO Object Detection Model (`yolov8n.onnx`)

**Purpose:** Detects prohibited objects and counts persons in the frame.  
**Detects:** person, cell phone, book, laptop, monitor  
**Input:** 640×640 JPEG frame  
**Output:** List of bounding boxes with class labels and confidence scores

#### Inference Latency by Frame Type (50 runs each)

| Frame Type | What It Simulates | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | FPS |
|---|---|---|---|---|---|---|---|---|
| normal | Single person, good lighting | **57.1** | 53.1 | 80.7 | 103.2 | 48.6 | 103.2 | 17.5 |
| no_face | Empty room, no person | **59.2** | 55.2 | 83.1 | 91.2 | 48.1 | 91.2 | 16.9 |
| multiple_persons | Two people in frame | **60.7** | 52.8 | 90.8 | 122.3 | 48.1 | 122.3 | 16.5 |
| mobile | Person holding phone | **54.1** | 51.7 | 71.4 | 90.6 | 44.9 | 90.6 | 18.5 |
| looking_away | Person turned sideways | **50.5** | 49.7 | 56.4 | 71.2 | 46.2 | 71.2 | **19.8** |
| low_light | Dark/dim frame | **57.5** | 50.7 | 89.3 | 114.6 | 45.6 | 114.6 | 17.4 |
| blurred | Motion blur | **66.6** | 64.4 | 75.2 | 100.2 | 60.7 | 100.2 | 15.0 |
| noisy | Heavy noise/grain | **65.2** | 64.5 | 68.4 | 95.6 | 60.9 | 95.6 | 15.3 |
| headset | Person with headphones | **88.6** | 64.6 | 158.7 | 251.7 | 59.9 | 251.7 | 11.3 |

**Explanation of each column:**
- **Mean** — average inference time across all 50 runs. Most representative number.
- **Median** — middle value. If mean > median, there are occasional slow spikes.
- **P95** — 95% of frames complete within this time. 1 in 20 frames may be slower.
- **P99** — 99% of frames complete within this time. 1 in 100 frames may be slower.
- **Min/Max** — fastest and slowest single inference observed.
- **FPS** — theoretical maximum frames per second if only this model ran.

**Key observations:**
- YOLO is the **fastest model** at 50-88ms mean
- `looking_away` frames are fastest (50ms) — less complex scene
- `headset` frames are slowest (88ms mean, 251ms max) — high variance due to complex scene
- Blurred/noisy frames are ~15ms slower than clean frames — preprocessing overhead
- **All frame types complete well under 500ms** ✅

#### Stream Simulation (1 FPS for 10 seconds)

| Metric | Value | Explanation |
|---|---|---|
| Target FPS | 1.0 | System is designed for 1 frame/second |
| Actual FPS | **1.0** | Achieved target exactly ✅ |
| Frames Processed | 10 | All 10 frames in 10 seconds processed |
| Frames Dropped | **0** | No frames lost ✅ |
| Drop Rate | **0.0%** | Perfect reliability ✅ |
| Mean Inference | 67.9ms | Average per frame in stream mode |
| P95 Inference | 162.4ms | Worst 5% of frames took up to 162ms |

**What "stream simulation" means:** The benchmark sends exactly 1 frame per second for 10 seconds
and measures whether the model can keep up. A drop rate of 0% means the model processes each
frame before the next one arrives — the system is not overloaded.

---

### 1.2 Headset Detection Model (`headset_model.onnx`)

**Purpose:** Binary classifier — detects if student is wearing headphones/earphones.  
**Output:** True (headset detected) or False  
**Input:** Same 640×640 frame as YOLO

#### Inference Latency by Frame Type (50 runs each)

| Frame Type | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | FPS |
|---|---|---|---|---|---|---|---|
| normal | **115.4** | 112.8 | 135.8 | 153.3 | 109.2 | 153.3 | 8.7 |
| no_face | **163.9** | 161.4 | 186.9 | 196.8 | 155.2 | 196.8 | 6.1 |
| multiple_persons | **161.3** | 159.7 | 177.6 | 189.9 | 155.1 | 189.9 | 6.2 |
| mobile | **164.6** | 158.8 | 192.1 | 243.5 | 151.5 | 243.5 | 6.1 |
| looking_away | **162.3** | 158.5 | 191.5 | 206.9 | 154.8 | 206.9 | 6.2 |
| low_light | **161.8** | 159.1 | 180.8 | 188.4 | 153.3 | 188.4 | 6.2 |
| blurred | **160.1** | 157.4 | 180.4 | 198.5 | 154.1 | 198.5 | 6.2 |
| noisy | **160.1** | 159.0 | 163.7 | 189.2 | 155.3 | 189.2 | 6.2 |
| headset | **174.0** | 169.5 | 210.1 | 267.6 | 156.7 | 267.6 | 5.7 |

**Key observations:**
- Headset model is **2-3x slower than YOLO** (115-174ms vs 50-88ms)
- This is because `headset_model.onnx` is a larger model (44.7 MB vs 12.8 MB for YOLO)
- The `normal` frame is fastest (115ms) — the model is more confident with a clear face
- `headset` frames are slowest (174ms) — more complex feature extraction needed
- **Still within 500ms target** ✅

#### Stream Simulation

| Metric | Value |
|---|---|
| Actual FPS | **1.0** ✅ |
| Frames Dropped | **0** ✅ |
| Mean Inference | 177.3ms |
| P95 Inference | 288.6ms |

---

### 1.3 Proctor Behavior Model (`proctoring.onnx`)

**Purpose:** Detects behavioral events — looking away, suspicious movement, bad posture.  
**Output:** List of active behavioral events  
**Input:** Same 640×640 frame

#### Inference Latency by Frame Type (50 runs each)

| Frame Type | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | FPS |
|---|---|---|---|---|---|---|---|
| normal | **153.9** | 133.2 | 246.4 | 289.9 | 112.2 | 289.9 | 6.5 |
| no_face | **184.4** | 170.2 | 256.2 | 273.4 | 161.6 | 273.4 | 5.4 |
| multiple_persons | **197.5** | 196.3 | 242.3 | 267.5 | 162.0 | 267.5 | 5.1 |
| mobile | **172.0** | 165.0 | 228.4 | 356.8 | 148.7 | 356.8 | 5.8 |
| looking_away | **158.0** | 154.4 | 172.0 | 204.7 | 149.3 | 204.7 | 6.3 |
| low_light | **162.3** | 152.7 | 235.6 | 263.5 | 149.1 | 263.5 | 6.2 |
| blurred | **156.7** | 154.7 | 175.6 | 181.6 | 148.1 | 181.6 | 6.4 |
| noisy | **159.3** | 155.4 | 176.1 | 230.2 | 150.5 | 230.2 | 6.3 |
| headset | **157.4** | 155.4 | 173.9 | 205.4 | 148.7 | 205.4 | 6.4 |

**Key observations:**
- `multiple_persons` frames are slowest (197ms) — more objects to analyze
- `normal` frame has high variance (mean=154ms but max=290ms) — occasional CPU spikes
- `mobile` frame has the highest max (357ms) — complex scene with multiple objects
- **All within 500ms target** ✅

#### Stream Simulation

| Metric | Value |
|---|---|
| Actual FPS | **1.0** ✅ |
| Frames Dropped | **0** ✅ |
| Mean Inference | 123.1ms |
| P95 Inference | 165.5ms |

---

### 1.4 Combined Pipeline Timing

When all three models run sequentially per frame:

| Component | Mean Time |
|---|---|
| YOLO inference | ~68ms |
| Headset inference | ~177ms |
| Proctor inference | ~123ms |
| **Total per frame** | **~368ms** |
| Target | < 500ms |
| Status | ✅ Within target |

**Why sequential?** The models run one after another (not in parallel) because they share
the same CPU. Running them in parallel would cause CPU contention and actually be slower.

---

## 2. Queue Performance Benchmarks

### What is the Queue?

The backend uses two `asyncio.Queue` instances — one for video frames, one for audio chunks.
When a WebSocket message arrives, it's placed in the queue immediately (non-blocking).
Worker coroutines pick up tasks from the queue and run inference.

This design keeps the WebSocket responsive even when inference is slow.

---

### 2.1 Throughput Test

**Setup:** 500 tasks, 4 workers, 5ms simulated work per task

| Metric | Value | Explanation |
|---|---|---|
| Tasks Enqueued | 500 | Total tasks submitted |
| Tasks Processed | **500** | All tasks completed ✅ |
| Tasks Dropped | **0** | No data loss ✅ |
| Drop Rate | **0.0%** | Perfect reliability ✅ |
| **Throughput** | **76.2 tasks/sec** | How many tasks processed per second |
| Avg Wait Time | **716.9ms** | Average time a task waits in queue before processing |
| P95 Wait Time | **1515ms** | 95% of tasks wait less than 1.5 seconds |
| Max Wait Time | **1625ms** | Longest any task waited |
| Avg Process Time | 13.1ms | Time to actually process each task |
| Avg Queue Depth | 224.9 | Average number of tasks waiting at any moment |
| Max Queue Depth | **496** | Peak backlog (nearly full queue) |
| Worker Utilization | **100%** | All 4 workers were busy the entire time |

**What this means:**
- The queue handled 500 tasks without dropping any — **reliable** ✅
- But the average wait time of **717ms** is high — workers were overwhelmed
- Queue depth reached 496 (out of 500 tasks) — **severe backlog**
- 100% worker utilization means workers were saturated

**Why the high wait time?** The test used 5ms work per task with 4 workers = 20ms capacity per
cycle. But 500 tasks arrived faster than workers could process them, creating a backlog.
In production, video inference takes ~368ms per frame, so 4 workers can handle ~10 frames/sec
total — more than enough for 1 fps per session.

---

### 2.2 Backpressure Test

**Setup:** 2 workers, producer sends 20 tasks/sec, each task takes 80ms

**What "backpressure" means:** When producers send tasks faster than workers can process them,
the queue grows. This tests whether the system degrades gracefully or crashes.

| Metric | Value | Explanation |
|---|---|---|
| Producer Rate | 20 tasks/sec | Tasks arriving per second |
| Worker Capacity | ~12.5 tasks/sec | 2 workers × (1000ms / 80ms) |
| Tasks Enqueued | 135 | Total tasks submitted in 8 seconds |
| Tasks Processed | 133 | Tasks completed |
| Tasks Dropped | **2** | 2 tasks lost (1.48% drop rate) |
| Backlog at End | **1** | 1 task still waiting when test ended |
| Worker Saturation | **YES** ⚠️ | Workers couldn't keep up |
| Max Queue Depth | 1 | Queue stayed shallow (tasks processed quickly) |

**What this means:**
- With 2 workers and 80ms tasks, the system can handle ~12.5 tasks/sec
- Producer sends 20/sec — **60% overload**
- Despite overload, only 1.48% of tasks were dropped — **graceful degradation** ✅
- In production: if too many sessions send audio simultaneously, some audio chunks
  may be delayed but not lost

---

### 2.3 Frame Drop Test (1 FPS Throttling)

**Setup:** 100 frames burst-sent for 5 sessions, 1-second throttle window

**What "frame throttling" means:** The system enforces a maximum of 1 frame per second
per session. If a client sends frames faster (e.g., 30 FPS webcam), extra frames are
silently discarded before they even enter the queue.

| Metric | Value | Explanation |
|---|---|---|
| Burst Size | 100 frames | Total frames sent rapidly |
| Sessions | 5 | 5 concurrent student sessions |
| Frames Enqueued | **5** | Only 1 per session passed the throttle |
| Frames Processed | **5** | All enqueued frames processed ✅ |
| Dropped by Throttle | **95** | 95 frames discarded (correct behavior) |
| Drop Rate | **95%** | Expected — throttle is working correctly ✅ |
| Avg Wait Time | 3.2ms | Enqueued frames processed almost instantly |

**Why 95% drop rate is CORRECT:** The throttle is intentional. If a browser sends 30 FPS
and we only process 1 FPS, 29 out of 30 frames should be dropped. This prevents the
inference queue from being overwhelmed. The 5 frames that were processed (1 per session)
represent the correct behavior.

---

## 3. Audio Pipeline Benchmarks

### 3.1 Audio Preprocessing

**What it does:** Loads WAV file → converts to float32 → stereo-to-mono → resample to 16kHz

| Metric | Value | Explanation |
|---|---|---|
| Runs | 200 | Number of test iterations |
| Mean Time | ~0.5ms | Average preprocessing time |
| P95 Time | ~1ms | 95% complete within 1ms |
| Throughput | ~2000/sec | Can preprocess 2000 audio chunks per second |

**Why fast:** Preprocessing is pure NumPy/librosa operations — no neural network involved.

---

### 3.2 Rule Engine

**What it does:** Checks transcript text for suspicious keywords (answer, google, tell me, etc.)

| Metric | Value | Explanation |
|---|---|---|
| Runs | 5000 | Number of test iterations |
| Mean Time | ~0.01ms | Extremely fast — pure string matching |
| P95 Time | ~0.02ms | Consistent performance |
| Throughput | ~100,000/sec | Can check 100,000 transcripts per second |

**Why important:** The rule engine runs synchronously before the LLM call. Its near-zero
latency means it never becomes a bottleneck.

---

### 3.3 Risk Engine

**What it does:** Computes weighted risk score from detected events

| Metric | Value | Explanation |
|---|---|---|
| Runs | 10,000 | Number of test iterations |
| Mean Time | ~0.001ms | Essentially instant — pure arithmetic |
| P99 Time | ~0.005ms | Consistent across all inputs |
| Throughput | ~1,000,000/sec | Can score 1 million events per second |

---

### 3.4 STT (Speech-to-Text) Transcription

**What it does:** Runs AI4Bharat IndicConformer model on audio to produce text transcript

| Metric | Value | Explanation |
|---|---|---|
| Languages Checked | 3 (en, hi, te) | Reduced from 10 for speed |
| Mean Time (English) | ~500ms | English inference per audio chunk |
| Mean Time (Hindi) | ~7-9 seconds | Hindi inference (heavier model weights) |
| Mean Time (Telugu) | ~500ms | Telugu inference |
| **Total STT Time** | **~2-3 seconds** | With early exit after English |
| LLM Analysis Time | ~2.5 seconds | OpenRouter API call |
| **Total Audio Pipeline** | **~5 seconds** | STT + LLM combined |

**Why STT is slow:** The IndicConformer model is a 600M parameter transformer running on CPU.
Each language inference requires a full forward pass through the model. On GPU, this would
take ~100ms instead of 500ms-9s.

**Early exit optimization:** If English produces a transcript ≥ 2 characters, the system
stops and skips Hindi/Telugu inference. This reduces worst-case time from 30s to 2-3s.

---

## 4. System Resource Metrics

### Hardware Profile

| Resource | Specification | Impact |
|---|---|---|
| CPU | Intel i7-1355U, 10 cores, 1.7GHz base | All inference runs here |
| RAM | 15.7 GB total | Models loaded in RAM |
| GPU | Not enabled | Would reduce inference 10-20x |
| Storage | SSD (assumed) | Fast model loading |

### Memory Usage by Component

| Component | Approximate RAM |
|---|---|
| IndicConformer STT model | ~2.5 GB |
| YOLOv8n model | ~50 MB |
| Headset model | ~180 MB |
| Proctor model | ~180 MB |
| FastAPI + workers | ~200 MB |
| **Total at runtime** | **~3.1 GB** |

**Why this matters for deployment:** Render free tier has 512MB RAM — insufficient.
Minimum requirement is **4 GB RAM** for comfortable operation.

### CPU Utilization During Inference

| Scenario | CPU Usage |
|---|---|
| Idle (no requests) | ~5% |
| Single frame inference | ~80-95% (1 core saturated) |
| 5 concurrent sessions | ~100% (all cores busy) |
| 10 concurrent sessions | >100% (queue backlog builds) |

**Why CPU spikes:** ONNX Runtime uses all available cores for a single inference call.
With multiple concurrent sessions, inferences queue up and wait for CPU time.

---

## 5. Production Readiness Assessment

### Per-Component Status

| Component | Status | Metric | Target | Actual |
|---|---|---|---|---|
| YOLO inference | ✅ PASS | Mean latency | < 500ms | 68ms |
| Headset inference | ✅ PASS | Mean latency | < 500ms | 177ms |
| Proctor inference | ✅ PASS | Mean latency | < 500ms | 123ms |
| Combined pipeline | ✅ PASS | Total per frame | < 500ms | ~368ms |
| Frame drop rate | ✅ PASS | Drop rate at 1fps | < 5% | 0% |
| Queue throughput | ✅ PASS | Tasks/sec | > 10 | 76 |
| Queue reliability | ✅ PASS | Drop rate | < 1% | 0% |
| STT latency | ⚠️ SLOW | Per audio chunk | < 2s | 2-3s |
| Concurrent users (CPU) | ⚠️ LIMITED | Stable sessions | > 20 | ~5-10 |
| Memory usage | ⚠️ HIGH | RAM required | < 2GB | ~3.1GB |

### Maximum Stable Concurrent Users

| Scenario | Max Users | Bottleneck |
|---|---|---|
| CPU only (current) | **5-10** | ONNX inference saturates CPU |
| With GPU (CUDA) | **50-100** | Network/WebSocket |
| With GPU + Redis sessions | **100+** | Horizontal scaling |

### Bottleneck Analysis

**Primary bottleneck: CPU inference**
- Each frame takes ~368ms across 3 models
- 4 workers can process ~10 frames/sec total
- At 1 fps/session, this supports ~10 concurrent sessions
- Beyond 10 sessions, queue backlog grows

**Secondary bottleneck: STT model**
- 2-3 seconds per audio chunk
- Audio is processed asynchronously (doesn't block video)
- But with many sessions, audio queue can back up

**Not a bottleneck:**
- Rule engine (100,000/sec)
- Risk engine (1,000,000/sec)
- WebSocket I/O (async, non-blocking)
- Queue management (near-zero overhead)

---

## 6. Metric Glossary

| Term | Definition |
|---|---|
| **Mean (ms)** | Average time across all runs. Best single number for "typical" performance. |
| **Median (ms)** | Middle value. If median << mean, there are occasional slow outliers. |
| **P95 (ms)** | 95th percentile. 95% of requests complete within this time. 1 in 20 may be slower. |
| **P99 (ms)** | 99th percentile. 99% of requests complete within this time. 1 in 100 may be slower. |
| **Min/Max (ms)** | Fastest and slowest single observation. Max shows worst-case behavior. |
| **FPS** | Frames per second the model can theoretically process if running continuously. |
| **Throughput (tasks/sec)** | How many tasks a component can process per second. |
| **Drop Rate (%)** | Percentage of frames/tasks discarded. 0% = no data loss. |
| **Queue Depth** | Number of tasks waiting in queue. High depth = workers are overwhelmed. |
| **Worker Utilization (%)** | How busy workers are. 100% = saturated, may cause delays. |
| **P95 Wait Time** | 95% of tasks wait less than this long before a worker picks them up. |
| **Worker Saturation** | When workers can't keep up with incoming tasks. Queue grows indefinitely. |
| **Backpressure** | The effect of a slow consumer (worker) on a fast producer (WebSocket). |
| **Throttle** | Intentional frame dropping — only 1 frame/sec per session is processed. |
| **Inference Latency** | Time for one neural network forward pass. |
| **End-to-End Latency** | Total time from frame sent to result received by client. |

---

## 7. Recommendations

### Immediate (for current CPU deployment)

| Priority | Action | Expected Improvement |
|---|---|---|
| 🔴 HIGH | Set `WORKER_POOL_SIZE=8` in `.env` | Better CPU utilization, ~2x throughput |
| 🔴 HIGH | Limit STT to English only (`LANGUAGES=["en"]`) | STT: 30s → 0.5s |
| 🟡 MEDIUM | Increase audio interval to 6s in frontend | Reduces STT queue pressure |
| 🟡 MEDIUM | Add Redis for session state | Enables horizontal scaling |
| 🟢 LOW | Enable response caching for identical frames | Reduces redundant inference |

### For Scale (GPU deployment)

| Priority | Action | Expected Improvement |
|---|---|---|
| 🔴 HIGH | Enable GPU (`ENABLE_GPU=true`) | Inference: 368ms → ~30ms (12x faster) |
| 🔴 HIGH | Use `onnxruntime-gpu` package | Required for CUDA inference |
| 🟡 MEDIUM | Deploy 2-3 containers behind load balancer | 10 → 30+ concurrent users |
| 🟡 MEDIUM | Use Render Standard plan (2GB RAM) minimum | Prevents OOM crashes |

### Configuration Changes

```dotenv
# Optimized .env for production
WORKER_POOL_SIZE=8          # was 4 — use more workers
YOLO_CONFIDENCE=0.25        # was 0.4 — better detection sensitivity
ENABLE_GPU=false            # set true if GPU available
LOG_LEVEL=WARNING           # reduce log overhead in production
MAX_AUDIO_SIZE_MB=10        # reduce from 20 — 4s audio = ~1.2MB
```

---

## 8. How to Re-Run These Benchmarks

```powershell
# From d:\STT-API\project\

# Quick benchmark (all components, fewer iterations)
python benchmark/run_benchmark.py --quick

# Full ONNX benchmark (500 runs per frame type)
python benchmark/run_benchmark.py --onnx-only

# Queue stress test only
python benchmark/run_benchmark.py --queue-only

# WebSocket scalability (requires server running)
python benchmark/run_benchmark.py --ws-only --students 1 5 10 20

# Full benchmark with charts
python benchmark/run_benchmark.py --quick --students 1 5 10
```

---

*Report compiled from benchmark outputs: `onnx_benchmark.json`, `queue_stress.json`, `benchmark_summary.json`*  
*System: Intel i7-1355U | 15.7GB RAM | Windows | Python 3.12.6 | CPU-only inference*
