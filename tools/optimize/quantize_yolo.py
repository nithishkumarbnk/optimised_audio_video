"""
YOLO ONNX INT8 Quantization
=============================
Applies dynamic INT8 quantization to the exported YOLO ONNX models:
  - yolov8n.onnx       → yolov8n_int8.onnx
  - proctoring.onnx    → proctoring_int8.onnx
  - headset_model.onnx → headset_model_int8.onnx

Run AFTER export_pt_to_onnx.py has completed.

Usage:
    python tools/optimize/quantize_yolo.py
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from config import LOG_FORMAT, YOLO_ONNX_DIR, YOLO_INT8_DIR

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("quantize_yolo")

# ONNX models to quantize (must exist in YOLO_ONNX_DIR after export)
YOLO_QUANTIZE_TARGETS = [
    "yolov8n.onnx",
    "proctoring.onnx",
    "headset_model.onnx",
]


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 ** 2), 2)


def quantize_model(src: Path, dst: Path) -> dict:
    """Apply dynamic INT8 quantization to a YOLO ONNX model.

    Args:
        src: Source .onnx file.
        dst: Destination _int8.onnx file.

    Returns:
        Dict with size and timing metadata.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    name = src.stem
    size_before = _size_mb(src)
    logger.info(f"[{name}] Quantizing {src.name} ({size_before} MB) → {dst.name}")

    t0 = time.perf_counter()
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
        optimize_model=True,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    size_after = _size_mb(dst)
    reduction  = round((1 - size_after / size_before) * 100, 1)

    logger.info(
        f"[{name}] Done: {size_before} MB → {size_after} MB "
        f"({reduction}% reduction) in {elapsed}s"
    )
    return {
        "model": name,
        "src": str(src),
        "dst": str(dst),
        "size_before_mb": size_before,
        "size_after_mb": size_after,
        "reduction_pct": reduction,
        "elapsed_s": elapsed,
    }


def run() -> list[dict]:
    """Quantize all YOLO ONNX models to INT8.

    Returns:
        List of result dicts.
    """
    results = []

    logger.info("=" * 60)
    logger.info("YOLO ONNX INT8 Quantization Pipeline")
    logger.info(f"Source dir : {YOLO_ONNX_DIR}")
    logger.info(f"Output dir : {YOLO_INT8_DIR}")
    logger.info("=" * 60)

    for onnx_name in YOLO_QUANTIZE_TARGETS:
        src = YOLO_ONNX_DIR / onnx_name
        stem = src.stem  # e.g. "yolov8n"
        dst  = YOLO_INT8_DIR / f"{stem}_int8.onnx"

        if not src.exists():
            logger.warning(
                f"[{stem}] Source ONNX not found: {src}. "
                "Run export_pt_to_onnx.py first."
            )
            continue

        try:
            result = quantize_model(src, dst)
            results.append(result)
        except Exception as exc:
            logger.exception(f"[{stem}] Quantization failed: {exc}")

    # ── Summary ───────────────────────────────────────────────────────
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("QUANTIZATION SUMMARY")
        logger.info("=" * 60)
        total_before = sum(r["size_before_mb"] for r in results)
        total_after  = sum(r["size_after_mb"]  for r in results)
        total_reduction = round((1 - total_after / total_before) * 100, 1)
        for r in results:
            logger.info(
                f"  {r['model']:20s} {r['size_before_mb']:7.1f} MB → "
                f"{r['size_after_mb']:7.1f} MB  ({r['reduction_pct']}% ↓)"
            )
        logger.info(
            f"  {'TOTAL':20s} {total_before:7.1f} MB → "
            f"{total_after:7.1f} MB  ({total_reduction}% ↓)"
        )
        logger.info("=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize YOLO ONNX models to INT8")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
