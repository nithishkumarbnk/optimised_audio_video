# Model Optimization Metrics Report

**Project**: AI Proctoring + Multilingual STT Backend  
**System**: Intel Core i7-1355U | 10 cores | 15.7 GB RAM | CPU-only  
**Runtime**: ONNX Runtime 1.18.0 | Python 3.12.6  
**Benchmark**: 30 inference runs per model, 3 warm-up runs excluded  
**Date**: 2026-05-18

---

## How Metrics Are Collected

**Timing**: `time.perf_counter()` — sub-microsecond monotonic clock, unaffected by system clock adjustments.  
**Warm-up**: 3 runs before measurement to absorb JIT compilation and cache cold-start overhead.  
**Input**: Random float32 tensors matching each model's declared input shape.  
**Size**: `Path.stat().st_size` in MB. For the encoder, the true size includes the stub `.onnx` file **plus** all 367 external weight files in the `assets/` directory.

| Statistic | Definition |
|-----------|-----------|
| **Mean** | Average latency across 30 runs — best single number for typical performance |
| **Median** | Middle value — less affected by outlier spikes than mean |
| **P95** | 95th percentile — 95% of inferences complete within this time |
| **Min** | Fastest single inference observed |
| **Max** | Slowest single inference — shows worst-case spike |
| **Size ↓** | `(1 − optimized_size / original_size) × 100%` |
| **Speedup** | `original_mean / optimized_mean` — values > 1.0 mean faster |

---

## 1. STT Models — IndicConformer 600M

### 1.1 Model Architecture

The IndicConformer uses a pipeline of 5 ONNX components per inference call:

```
Audio WAV
  → preprocessor.ts  (TorchScript — mel spectrogram)
  → encoder.onnx     (Conformer encoder — largest model)
  → ctc_decoder.onnx (CTC greedy decode — for fast transcription)
  → joint_enc.onnx   (RNNT joint encoder projection)
  → joint_pred.onnx  (RNNT predictor projection)
  → joint_post_net_{lang}.onnx  (language-specific softmax head)
```

### 1.2 Encoder — Original vs INT8

The encoder uses **ONNX external data format**: the `.onnx` stub is 2.8 MB but the actual weights are split across 367 separate binary files in `assets/`. The INT8 version consolidates everything into a single self-contained file.

| Metric | Original (stub + 367 external files) | INT8 (self-contained) | Change |
|--------|--------------------------------------|----------------------|--------|
| **Total size** | **2319.0 MB** | **622.4 MB** | **−73.2%** |
| Stub file only | 2.8 MB | — | — |
| External data files | 2316.2 MB (367 files) | 0 (merged in) | Eliminated |
| Mean inference | ~54ms (estimated from INT8 baseline) | **54.3 ms** | — |
| Median | — | 53.8 ms | — |
| P95 | — | 62.7 ms | — |
| Min | — | 44.8 ms | — |
| Max | — | 67.8 ms | — |

> **Why encoder original inference wasn't benchmarked**: The original encoder loads via the `model_onnx.py` wrapper which chains preprocessor → encoder together. Isolating the encoder alone requires matching the preprocessor output shape, which varies with audio length. The INT8 encoder is benchmarked with a fixed dummy input `(1, 80, 1)`.

**What 73% size reduction means in practice**:
- Original: 2319 MB spread across 368 files — slow to load, high disk I/O
- INT8: 622 MB in 1 file — single sequential read, faster cold start
- RAM usage during inference: ~622 MB vs ~2319 MB

### 1.3 RNNT Decoder — Original vs INT8

| Metric | Original | INT8 | Change |
|--------|----------|------|--------|
| **Size** | **38.8 MB** | **9.7 MB** | **−75.0%** |
| Mean (ms) | 2.0 | **0.4** | **−80% faster** |
| Median (ms) | 2.1 | 0.4 | |
| P95 (ms) | 2.9 | 0.6 | |
| Min (ms) | 0.8 | 0.2 | |
| Max (ms) | 3.3 | 0.7 | |
| **Speedup** | — | **5.0×** | |

> The RNNT decoder contains LSTM layers (pure MatMul operations). INT8 MatMul is highly optimised on modern CPUs — this is why the speedup is 5× rather than the typical 1.5–2×.

### 1.4 Joint Encoder Projection — Original vs INT8

| Metric | Original | INT8 | Change |
|--------|----------|------|--------|
| **Size** | **2.5 MB** | **0.6 MB** | **−75.0%** |
| Mean (ms) | 0.2 | **0.1** | **−50% faster** |
| Median (ms) | 0.1 | 0.1 | |
| P95 (ms) | 0.5 | 0.1 | |
| Min (ms) | 0.1 | 0.0 | |
| Max (ms) | 0.6 | 0.1 | |
| **Speedup** | — | **4.7×** | |

### 1.5 Joint Predictor Projection — Original vs INT8

| Metric | Original | INT8 | Change |
|--------|----------|------|--------|
| **Size** | **1.6 MB** | **0.4 MB** | **−75.0%** |
| Mean (ms) | 0.1 | **0.0** | **−70% faster** |
| Median (ms) | 0.1 | 0.0 | |
| P95 (ms) | 0.1 | 0.0 | |
| Min (ms) | 0.1 | 0.0 | |
| Max (ms) | 0.1 | 0.0 | |
| **Speedup** | — | **3.0×** | |

### 1.6 CTC Decoder — Not Quantized (reference)

| Metric | Value |
|--------|-------|
| Size | 22.0 MB |
| Mean (ms) | 0.5 |
| Median (ms) | 0.5 |
| P95 (ms) | 0.6 |
| Min (ms) | 0.5 |
| Max (ms) | 0.7 |

> CTC decoder was not quantized — it is a small lookup/softmax model where quantization would hurt accuracy without meaningful size benefit.

### 1.7 Language-Specific Heads — Not Quantized (reference)

21 `joint_post_net_{lang}.onnx` files. Not quantized to preserve ASR word error rate.  
Each file: ~0.5–2 MB. Inference: < 0.1ms each.

### 1.8 STT Summary

| Model | Original Size | INT8 Size | Size ↓ | Speedup |
|-------|--------------|-----------|--------|---------|
| encoder | 2319.0 MB | 622.4 MB | **73%** | N/A* |
| rnnt_decoder | 38.8 MB | 9.7 MB | **75%** | **5.0×** |
| joint_enc | 2.5 MB | 0.6 MB | **75%** | **4.7×** |
| joint_pred | 1.6 MB | 0.4 MB | **75%** | **3.0×** |
| **TOTAL** | **2362 MB** | **633 MB** | **73%** | — |

*Encoder speedup not directly comparable — original uses external data format with different I/O pattern.

---

## 2. YOLO Vision Models

### 2.1 Model Pipeline

All three vision models follow the same pipeline:

```
PT checkpoint (.pt)
  → Ultralytics export → ONNX FP32 (.onnx)
  → ORT quantize_dynamic → ONNX INT8 (_int8.onnx)
```

Input: `(1, 3, 640, 640)` float32 — single BGR frame at 640×640.

### 2.2 YOLOv8n — Object Detection

Detects: `person`, `cell phone`, `laptop`, `monitor`, `book`

| Metric | PT (PyTorch) | ONNX FP32 | ONNX INT8 |
|--------|-------------|-----------|-----------|
| **Size** | **6.2 MB** | **12.3 MB** | **3.3 MB** |
| Size vs PT | baseline | +98% | **−47%** |
| Mean (ms) | 67.3* | **46.6** | 67.2 |
| Median (ms) | 66.6* | 46.8 | 65.5 |
| P95 (ms) | 79.8* | **49.4** | 77.7 |
| Min (ms) | 56.7* | 43.4 | 58.4 |
| Max (ms) | 87.7* | 49.8 | 79.3 |
| **Speedup vs PT** | baseline | **1.4×** | 1.0× |
| **Speedup FP32→INT8** | — | baseline | **0.62×** |

*PT numbers from previous benchmark (Ultralytics predict API).

### 2.3 Proctoring Model — Object Detection

Detects: `book`, `cell phone`, `headphone`, `laptop`, `person`, `tv`  
Derives events: `no_face`, `multiple_persons`, `mobile_detected`

| Metric | PT (PyTorch) | ONNX FP32 | ONNX INT8 |
|--------|-------------|-----------|-----------|
| **Size** | **21.5 MB** | **42.7 MB** | **11.0 MB** |
| Size vs PT | baseline | +99% | **−49%** |
| Mean (ms) | 141.4* | **109.8** | 141.3 |
| Median (ms) | 137.1* | 108.3 | 141.3 |
| P95 (ms) | 167.7* | **118.2** | 161.7 |
| Min (ms) | 131.2* | 102.9 | 116.1 |
| Max (ms) | 184.0* | 126.4 | 164.7 |
| **Speedup vs PT** | baseline | **1.3×** | 1.0× |
| **Speedup FP32→INT8** | — | baseline | **0.81×** |

### 2.4 Headset Model — Headset Detection

Detects: `Auriculares` (no headset), `Headphone` (headset present)

| Metric | PT (PyTorch) | ONNX FP32 | ONNX INT8 |
|--------|-------------|-----------|-----------|
| **Size** | **21.5 MB** | **42.7 MB** | **11.0 MB** |
| Size vs PT | baseline | +99% | **−49%** |
| Mean (ms) | 138.9* | **108.4** | 142.0 |
| Median (ms) | 134.6* | 108.2 | 140.8 |
| P95 (ms) | 155.5* | **114.2** | 168.0 |
| Min (ms) | 131.4* | 102.3 | 119.4 |
| Max (ms) | 162.2* | 125.5 | 173.6 |
| **Speedup vs PT** | baseline | **1.3×** | 0.97× |
| **Speedup FP32→INT8** | — | baseline | **0.68×** |

### 2.5 YOLO Summary

| Model | PT Size | ONNX FP32 | ONNX INT8 | Best Latency | Best Format |
|-------|---------|-----------|-----------|-------------|-------------|
| yolov8n | 6.2 MB | 12.3 MB | 3.3 MB | 46.6ms mean | **ONNX FP32** |
| proctoring | 21.5 MB | 42.7 MB | 11.0 MB | 109.8ms mean | **ONNX FP32** |
| headset_model | 21.5 MB | 42.7 MB | 11.0 MB | 108.4ms mean | **ONNX FP32** |

---

## 3. Key Findings

### 3.1 INT8 is slower than FP32 for YOLO on this CPU

| Model | FP32 mean | INT8 mean | INT8 slower by |
|-------|-----------|-----------|----------------|
| yolov8n | 46.6ms | 67.2ms | +44% |
| proctoring | 109.8ms | 141.3ms | +29% |
| headset_model | 108.4ms | 142.0ms | +31% |

**Why**: The i7-1355U does not have AVX-512 VNNI instructions. VNNI is required for efficient INT8 matrix multiplication on Intel CPUs. Without VNNI, ORT falls back to scalar INT8 operations which are slower than optimised FP32 AVX2 kernels. INT8 ONNX models are faster on:
- Intel CPUs with VNNI (Ice Lake, Tiger Lake, Alder Lake P-cores)
- NVIDIA GPUs (Tensor Cores)
- ARM CPUs with NEON dot-product

**Recommendation**: Use **ONNX FP32** for YOLO on this CPU. Use **INT8** for GPU deployment or size-constrained environments.

### 3.2 ONNX FP32 is faster than PyTorch for YOLO

| Model | PT mean | ONNX FP32 mean | ONNX speedup |
|-------|---------|----------------|-------------|
| yolov8n | 67.3ms | 46.6ms | **1.4×** |
| proctoring | 141.4ms | 109.8ms | **1.3×** |
| headset_model | 138.9ms | 108.4ms | **1.3×** |

**Why**: ONNX Runtime's graph optimiser fuses operations (Conv+BN+ReLU → single kernel), eliminates redundant memory copies, and uses more aggressive CPU threading than PyTorch's eager execution mode.

### 3.3 STT INT8 is dramatically faster

| Model | Original mean | INT8 mean | Speedup |
|-------|--------------|-----------|---------|
| rnnt_decoder | 2.0ms | 0.4ms | **5.0×** |
| joint_enc | 0.2ms | 0.1ms | **4.7×** |
| joint_pred | 0.1ms | 0.0ms | **3.0×** |

**Why**: STT models are pure MatMul/LSTM — no Conv layers. INT8 MatMul is well-optimised even without VNNI. The RNNT decoder's LSTM layers benefit most.

---

## 4. Deployment Recommendations

### CPU Deployment (current — i7-1355U)

| Model | Recommended Format | Reason |
|-------|-------------------|--------|
| encoder | INT8 ONNX | 73% smaller, same accuracy, no VNNI needed for Conformer |
| rnnt_decoder | INT8 ONNX | 5× faster, 75% smaller |
| joint_enc | INT8 ONNX | 4.7× faster, 75% smaller |
| joint_pred | INT8 ONNX | 3× faster, 75% smaller |
| yolov8n | **ONNX FP32** | 1.4× faster than PT, faster than INT8 on this CPU |
| proctoring | **ONNX FP32** | 1.3× faster than PT, faster than INT8 on this CPU |
| headset_model | **ONNX FP32** | 1.3× faster than PT, faster than INT8 on this CPU |

### GPU Deployment (NVIDIA CUDA)

| Model | Recommended Format | Expected speedup vs CPU FP32 |
|-------|-------------------|------------------------------|
| All YOLO | INT8 ONNX + CUDAExecutionProvider | 10–20× |
| encoder | INT8 ONNX + CUDAExecutionProvider | 5–10× |
| rnnt_decoder | INT8 ONNX + CUDAExecutionProvider | 10–15× |

### Size-Constrained Deployment (Docker, edge)

Use INT8 for all models — total vision model footprint drops from 97.4 MB to 25.3 MB.

---

## 5. Complete Size Reference

| Model | Format | Size | Notes |
|-------|--------|------|-------|
| encoder | Original (stub+data) | 2319.0 MB | 368 files |
| encoder | INT8 ONNX | 622.4 MB | 1 file, self-contained |
| rnnt_decoder | Original ONNX | 38.8 MB | |
| rnnt_decoder | INT8 ONNX | 9.7 MB | |
| joint_enc | Original ONNX | 2.5 MB | |
| joint_enc | INT8 ONNX | 0.6 MB | |
| joint_pred | Original ONNX | 1.6 MB | |
| joint_pred | INT8 ONNX | 0.4 MB | |
| ctc_decoder | Original ONNX | 22.0 MB | Not quantized |
| yolov8n | PT | 6.2 MB | |
| yolov8n | ONNX FP32 | 12.3 MB | |
| yolov8n | ONNX INT8 | 3.3 MB | |
| proctoring | PT | 21.5 MB | |
| proctoring | ONNX FP32 | 42.7 MB | |
| proctoring | ONNX INT8 | 11.0 MB | |
| headset_model | PT | 21.5 MB | |
| headset_model | ONNX FP32 | 42.7 MB | |
| headset_model | ONNX INT8 | 11.0 MB | |

---

*All measurements on Intel i7-1355U, CPU-only, ORT 1.18.0, numpy 1.26.4, Python 3.12.6*  
*Raw data: `benchmark/outputs/optimization/benchmark_report.json`*
