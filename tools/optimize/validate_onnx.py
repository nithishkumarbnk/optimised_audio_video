"""
ONNX Model Validation
======================
Validates all optimized ONNX models:
  1. ONNX graph check (onnx.checker.check_model)
  2. ORT session load test
  3. Single inference smoke test with random input
  4. Output shape sanity check

Usage:
    python tools/optimize/validate_onnx.py [--gpu]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    LOG_FORMAT,
    STT_OPTIMIZED_DIR,
    YOLO_ONNX_DIR,
    YOLO_INT8_DIR,
    get_providers,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("validate_onnx")


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 ** 2), 2)


def validate_model(onnx_path: Path, providers: list,
                   dummy_inputs: Optional[dict] = None) -> dict:
    """Validate a single ONNX model.

    Args:
        onnx_path:    Path to the .onnx file.
        providers:    ORT execution providers list.
        dummy_inputs: Optional dict of {input_name: np.ndarray}.
                      If None, random inputs are generated from model metadata.

    Returns:
        Dict with validation results.
    """
    import onnx
    import onnxruntime as ort

    name = onnx_path.stem
    result = {
        "model": name,
        "path": str(onnx_path),
        "size_mb": _size_mb(onnx_path),
        "graph_check": False,
        "session_load": False,
        "inference_ok": False,
        "output_shapes": [],
        "inference_ms": None,
        "error": None,
    }

    # 1. ONNX graph check
    try:
        model_proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_proto)
        result["graph_check"] = True
        logger.info(f"[{name}] ✓ ONNX graph check passed")
    except Exception as exc:
        result["error"] = f"graph_check: {exc}"
        logger.error(f"[{name}] ✗ ONNX graph check failed: {exc}")
        return result

    # 2. ORT session load
    try:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.log_severity_level = 3  # suppress verbose ORT logs
        session = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers)
        result["session_load"] = True
        logger.info(f"[{name}] ✓ ORT session loaded")
    except Exception as exc:
        result["error"] = f"session_load: {exc}"
        logger.error(f"[{name}] ✗ ORT session load failed: {exc}")
        return result

    # 3. Inference smoke test
    try:
        if dummy_inputs is None:
            dummy_inputs = {}
            for inp in session.get_inputs():
                shape = [
                    d if isinstance(d, int) and d > 0 else 1
                    for d in inp.shape
                ]
                dtype_map = {
                    "tensor(float)":  np.float32,
                    "tensor(float16)": np.float16,
                    "tensor(int64)":  np.int64,
                    "tensor(int32)":  np.int32,
                }
                dtype = dtype_map.get(inp.type, np.float32)
                dummy_inputs[inp.name] = np.random.randn(*shape).astype(dtype)

        t0 = time.perf_counter()
        outputs = session.run(None, dummy_inputs)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        result["inference_ok"]   = True
        result["inference_ms"]   = elapsed_ms
        result["output_shapes"]  = [list(o.shape) for o in outputs]
        logger.info(
            f"[{name}] ✓ Inference OK in {elapsed_ms}ms — "
            f"outputs: {result['output_shapes']}"
        )
    except Exception as exc:
        result["error"] = f"inference: {exc}"
        logger.error(f"[{name}] ✗ Inference failed: {exc}")

    return result


def run(enable_gpu: bool = False) -> list[dict]:
    """Validate all optimized ONNX models.

    Returns:
        List of validation result dicts.
    """
    providers = get_providers(enable_gpu)
    results   = []

    logger.info("=" * 60)
    logger.info("ONNX Model Validation")
    logger.info(f"Providers: {providers}")
    logger.info("=" * 60)

    # ── STT optimized models ──────────────────────────────────────────
    logger.info("\n--- STT Models ---")
    for onnx_file in sorted(STT_OPTIMIZED_DIR.glob("*.onnx")):
        result = validate_model(onnx_file, providers)
        results.append(result)

    # ── YOLO ONNX (FP32) ─────────────────────────────────────────────
    logger.info("\n--- YOLO ONNX (FP32) ---")
    for onnx_file in sorted(YOLO_ONNX_DIR.glob("*.onnx")):
        # YOLO models expect (1, 3, 640, 640) float32 input
        dummy = {"images": np.random.randn(1, 3, 640, 640).astype(np.float32)}
        result = validate_model(onnx_file, providers, dummy_inputs=dummy)
        results.append(result)

    # ── YOLO INT8 ─────────────────────────────────────────────────────
    logger.info("\n--- YOLO ONNX (INT8) ---")
    for onnx_file in sorted(YOLO_INT8_DIR.glob("*.onnx")):
        dummy = {"images": np.random.randn(1, 3, 640, 640).astype(np.float32)}
        result = validate_model(onnx_file, providers, dummy_inputs=dummy)
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["inference_ok"])
    failed = len(results) - passed

    logger.info("\n" + "=" * 60)
    logger.info(f"VALIDATION SUMMARY: {passed} passed, {failed} failed")
    logger.info("=" * 60)
    for r in results:
        status = "✓" if r["inference_ok"] else "✗"
        logger.info(
            f"  {status} {r['model']:30s} {r['size_mb']:7.1f} MB  "
            f"{r['inference_ms'] or 'N/A':>8} ms"
        )
    if failed > 0:
        logger.warning(f"\n{failed} model(s) failed validation — check errors above.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate optimized ONNX models")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()
    run(enable_gpu=args.gpu)


if __name__ == "__main__":
    main()
