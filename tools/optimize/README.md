# Model Optimization Pipeline

Compresses and optimizes all AI models for low-latency CPU inference.

## Folder Structure

```
tools/optimize/
├── config.py                  # All paths and settings — edit this first
├── export_pt_to_onnx.py       # Step 1: PT → ONNX export (Ultralytics)
├── quantize_stt.py            # Step 2: STT ONNX → INT8
├── quantize_yolo.py           # Step 3: YOLO ONNX → INT8
├── validate_onnx.py           # Step 4: Validate all optimized models
├── benchmark_optimized.py     # Step 5: Latency + size comparison
├── ort_inference.py           # Production ORT inference wrappers
├── run_pipeline.py            # Master runner (all steps)
└── README.md                  # This file

app/models/optimized/          # Output directory (auto-created)
├── stt/                       # Quantized STT models
│   ├── encoder_int8.onnx
│   ├── rnnt_decoder_int8.onnx
│   ├── joint_enc_int8.onnx
│   ├── joint_pred_int8.onnx
│   ├── ctc_decoder.onnx       # Copied as-is (small)
│   ├── joint_post_net_*.onnx  # Copied as-is (language heads)
│   └── vocab.json
├── yolo_onnx/                 # FP32 ONNX exports from PT
│   ├── yolov8n.onnx
│   ├── proctoring.onnx
│   └── headset_model.onnx
└── yolo_int8/                 # INT8 quantized YOLO models
    ├── yolov8n_int8.onnx
    ├── proctoring_int8.onnx
    └── headset_model_int8.onnx

benchmark/outputs/optimization/
├── benchmark_report.json      # Latency + size comparison
└── pipeline_report.json       # Full pipeline run report
```

## Prerequisites

```powershell
# From D:\STT-API\tv\
venv\Scripts\pip.exe install onnx onnxruntime>=1.19.0 onnxruntime-tools onnxsim
```

## Step-by-Step Execution

All commands run from `D:\STT-API\tv\` with the venv activated.

### Option A — Run everything at once

```powershell
venv\Scripts\python.exe tools\optimize\run_pipeline.py
```

With GPU support:
```powershell
venv\Scripts\python.exe tools\optimize\run_pipeline.py --gpu
```

Skip benchmarking (faster):
```powershell
venv\Scripts\python.exe tools\optimize\run_pipeline.py --skip-benchmark
```

---

### Option B — Run each step individually

**Step 1: Export PT models to ONNX**
```powershell
venv\Scripts\python.exe tools\optimize\export_pt_to_onnx.py --imgsz 640 --opset 17
```
Output: `app/models/optimized/yolo_onnx/`

**Step 2: Quantize STT ONNX models to INT8**
```powershell
venv\Scripts\python.exe tools\optimize\quantize_stt.py
```
Output: `app/models/optimized/stt/`

**Step 3: Quantize YOLO ONNX models to INT8**
```powershell
venv\Scripts\python.exe tools\optimize\quantize_yolo.py
```
Output: `app/models/optimized/yolo_int8/`

**Step 4: Validate all optimized models**
```powershell
venv\Scripts\python.exe tools\optimize\validate_onnx.py
```

**Step 5: Benchmark original vs optimized**
```powershell
venv\Scripts\python.exe tools\optimize\benchmark_optimized.py --runs 50
```
Output: `benchmark/outputs/optimization/benchmark_report.json`

---

## What Gets Quantized and Why

| Model | Action | Reason |
|-------|--------|--------|
| `encoder.onnx` | INT8 dynamic | Largest model, biggest size/speed gain |
| `rnnt_decoder.onnx` | INT8 dynamic | Heavy LSTM layers benefit from INT8 |
| `joint_enc.onnx` | INT8 dynamic | Linear projection layers — safe to quantize |
| `joint_pred.onnx` | INT8 dynamic | Linear projection layers — safe to quantize |
| `joint_post_net_*.onnx` | **NOT quantized** | Language-specific heads — quantization hurts ASR accuracy |
| `ctc_decoder.onnx` | **NOT quantized** | Small model, no benefit |
| `yolov8n.pt` | Export → INT8 | Detection accuracy preserved at INT8 |
| `proctoring.pt` | Export → INT8 | Object detection — INT8 safe |
| `headset_model.pt` | Export → INT8 | Binary classifier — INT8 safe |

## Expected Results (CPU, Intel i7)

| Model | Original | INT8 | Size ↓ | Speed ↑ |
|-------|----------|------|--------|---------|
| encoder | ~350 MB | ~90 MB | ~75% | ~2–3× |
| rnnt_decoder | ~50 MB | ~13 MB | ~75% | ~2× |
| joint_enc | ~10 MB | ~3 MB | ~70% | ~1.5× |
| joint_pred | ~10 MB | ~3 MB | ~70% | ~1.5× |
| yolov8n | ~6 MB | ~2 MB | ~65% | ~1.3× |
| proctoring | ~44 MB | ~12 MB | ~73% | ~1.5× |
| headset_model | ~44 MB | ~12 MB | ~73% | ~1.5× |

## Deployment Configuration

### CPU-only (recommended for Docker)

```python
providers = ["CPUExecutionProvider"]

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 0   # use all cores
session_options.log_severity_level = 3     # suppress verbose logs
```

### GPU (NVIDIA CUDA)

```python
providers = [
    ("CUDAExecutionProvider", {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": 2 * 1024 ** 3,
    }),
    "CPUExecutionProvider",
]
```

## Important Constraints

- `per_channel=False` is used for all quantization — keeps RNNT decoding numerically stable
- `reduce_range=False` — avoids accuracy loss on non-VNNI CPUs (most cloud VMs)
- `joint_post_net_*` files are never quantized — they are language-specific softmax heads
  where INT8 precision loss directly degrades word error rate
- YOLO models use `dynamic=False` (fixed batch=1) for consistent realtime latency
