"""
Centralized configuration for the model optimization pipeline.

All paths, targets, and optimization settings are defined here.
Import this module in every other optimization script.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent          # D:\STT-API\tv
MODELS_ROOT  = PROJECT_ROOT / "app" / "models"

# ---------------------------------------------------------------------------
# STT ONNX model paths (source)
# ---------------------------------------------------------------------------

STT_ASSETS = MODELS_ROOT / "indic-conformer-600m" / "assets"

STT_ONNX_SOURCES = {
    "encoder":      STT_ASSETS / "encoder.onnx",
    "rnnt_decoder": STT_ASSETS / "rnnt_decoder.onnx",
    "joint_enc":    STT_ASSETS / "joint_enc.onnx",
    "joint_pred":   STT_ASSETS / "joint_pred.onnx",
    # ctc_decoder: small, no quantization needed
    "ctc_decoder":  STT_ASSETS / "ctc_decoder.onnx",
}

# joint_post_net_* files — language-specific heads, NOT quantized
STT_JOINT_POST_NETS = sorted(STT_ASSETS.glob("joint_post_net_*.onnx"))

# Models to apply INT8 dynamic quantization to
STT_QUANTIZE_TARGETS = ["encoder", "rnnt_decoder", "joint_enc", "joint_pred"]

# ---------------------------------------------------------------------------
# PT model paths (source)
# ---------------------------------------------------------------------------

PT_SOURCES = {
    "yolov8n":       MODELS_ROOT / "yolov8n.pt",
    "proctoring":    MODELS_ROOT / "proctoring.pt",
    "headset_model": MODELS_ROOT / "headset_model.pt",
}

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

OPTIMIZED_ROOT      = MODELS_ROOT / "optimized"
STT_OPTIMIZED_DIR   = OPTIMIZED_ROOT / "stt"
YOLO_ONNX_DIR       = OPTIMIZED_ROOT / "yolo_onnx"
YOLO_INT8_DIR       = OPTIMIZED_ROOT / "yolo_int8"
BENCHMARK_RESULTS   = PROJECT_ROOT / "benchmark" / "outputs" / "optimization"

# Create all output directories
for _d in [STT_OPTIMIZED_DIR, YOLO_ONNX_DIR, YOLO_INT8_DIR, BENCHMARK_RESULTS]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ONNX Runtime provider configuration
# ---------------------------------------------------------------------------

# CPU-first, optional CUDA
def get_providers(enable_gpu: bool = False) -> list:
    """Return the ORT execution provider list for the given device preference."""
    if enable_gpu:
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                return [
                    ("CUDAExecutionProvider", {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 2 * 1024 ** 3,  # 2 GB
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                    }),
                    "CPUExecutionProvider",
                ]
        except Exception:
            pass
    return ["CPUExecutionProvider"]

# ---------------------------------------------------------------------------
# Quantization settings
# ---------------------------------------------------------------------------

# Dynamic INT8 quantization — safe for transformer/conformer models
QUANT_WEIGHT_TYPE = "QInt8"   # weight quantization type
QUANT_ACTIVATION  = "QInt8"   # activation quantization type

# ONNX opset for exported PT models
ONNX_OPSET = 17

# ---------------------------------------------------------------------------
# Logging format
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
