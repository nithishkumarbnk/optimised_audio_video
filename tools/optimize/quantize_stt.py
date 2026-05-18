"""
STT ONNX Model Quantization
============================
Applies dynamic INT8 quantization to the four heavy STT ONNX models:
  - encoder.onnx        → encoder_int8.onnx
  - rnnt_decoder.onnx   → rnnt_decoder_int8.onnx
  - joint_enc.onnx      → joint_enc_int8.onnx
  - joint_pred.onnx     → joint_pred_int8.onnx

joint_post_net_*.onnx (language-specific heads) are intentionally skipped
to preserve ASR quality — they are small and quantization would hurt accuracy.

Usage:
    python tools/optimize/quantize_stt.py [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

from config import (
    LOG_FORMAT,
    STT_ASSETS,
    STT_OPTIMIZED_DIR,
    STT_ONNX_SOURCES,
    STT_QUANTIZE_TARGETS,
    STT_JOINT_POST_NETS,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("quantize_stt")


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 ** 2), 2)


def _true_size_mb(onnx_path: Path) -> float:
    """Return the true model size in MB, including ONNX external data files.

    ONNX models with large weights use the external data format: the .onnx
    file is a small stub and the actual weights live in sibling files in the
    same directory. This function sums the stub + all external data files so
    the size comparison is accurate.

    For self-contained models (no external data) this is identical to
    _size_mb().
    """
    total = onnx_path.stat().st_size
    parent = onnx_path.parent

    # External data files have no extension and live next to the .onnx stub.
    # They are named after the tensor they contain (e.g. onnx__MatMul_8067,
    # layers.0.conv.pointwise_conv1.weight, pre_encode.conv.0.weight, etc.)
    for sibling in parent.iterdir():
        if sibling == onnx_path:
            continue
        # Skip other .onnx files, .json, .ts, .py, .gitattributes
        if sibling.suffix in (".onnx", ".json", ".ts", ".py", ".gitattributes", ".md"):
            continue
        # Only count files that look like external data (no extension or
        # names matching known weight file patterns)
        if sibling.suffix == "" or sibling.name.startswith(("onnx__", "layers.", "pre_encode")):
            total += sibling.stat().st_size

    return round(total / (1024 ** 2), 2)


def quantize_model(name: str, src: Path, dst: Path) -> dict:
    """Apply dynamic INT8 quantization to a single ONNX model.

    Uses QUInt8 for models containing Conv layers (encoder) to avoid the
    ConvInteger op which requires ORT >= 1.19. Uses QInt8 for pure MatMul
    models (rnnt_decoder, joint_enc, joint_pred) for maximum compression.

    Args:
        name: Human-readable model name for logging.
        src:  Source .onnx file path.
        dst:  Destination _int8.onnx file path.

    Returns:
        Dict with size_before_mb, size_after_mb, reduction_pct, elapsed_s.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx

    size_before = _true_size_mb(src)

    # Detect Conv nodes — use QUInt8 to avoid ConvInteger (needs ORT>=1.19).
    # Pure MatMul models use QInt8 for better compression.
    model_proto = onnx.load(str(src))
    has_conv = any(node.op_type == "Conv" for node in model_proto.graph.node)
    weight_type = QuantType.QUInt8 if has_conv else QuantType.QInt8
    logger.info(
        f"[{name}] Quantizing {src.name} ({size_before} MB total) → {dst.name} "
        f"[{'QUInt8 (has Conv)' if has_conv else 'QInt8'}]"
    )

    t0 = time.perf_counter()
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=weight_type,
        per_channel=False,
        reduce_range=False,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    size_after = _size_mb(dst)
    reduction = round((1 - size_after / size_before) * 100, 1) if size_before > 0 else 0

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


def copy_unquantized(name: str, src: Path, dst_dir: Path) -> None:
    """Copy a model that should NOT be quantized to the output directory."""
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    logger.info(f"[{name}] Copied (no quantization): {src.name} → {dst}")


def run(dry_run: bool = False) -> list[dict]:
    """Run the full STT quantization pipeline.

    Args:
        dry_run: If True, only log what would happen without writing files.

    Returns:
        List of result dicts for each quantized model.
    """
    results = []

    logger.info("=" * 60)
    logger.info("STT ONNX Quantization Pipeline")
    logger.info(f"Source dir : {STT_ASSETS}")
    logger.info(f"Output dir : {STT_OPTIMIZED_DIR}")
    logger.info(f"Targets    : {STT_QUANTIZE_TARGETS}")
    logger.info("=" * 60)

    # ── Quantize the four heavy models ──────────────────────────────
    for name in STT_QUANTIZE_TARGETS:
        src = STT_ONNX_SOURCES[name]
        dst = STT_OPTIMIZED_DIR / f"{name}_int8.onnx"

        if not src.exists():
            logger.error(f"[{name}] Source not found: {src}")
            continue

        if dry_run:
            logger.info(f"[DRY RUN] Would quantize {src} → {dst}")
            continue

        try:
            result = quantize_model(name, src, dst)
            results.append(result)
        except Exception as exc:
            logger.exception(f"[{name}] Quantization failed: {exc}")

    # ── Copy ctc_decoder as-is (small, no benefit from quantization) ─
    ctc_src = STT_ONNX_SOURCES["ctc_decoder"]
    if ctc_src.exists() and not dry_run:
        copy_unquantized("ctc_decoder", ctc_src, STT_OPTIMIZED_DIR)

    # ── Copy joint_post_net_* as-is (language-specific heads) ────────
    logger.info(f"Copying {len(STT_JOINT_POST_NETS)} joint_post_net_* files (no quantization)...")
    for jpn in STT_JOINT_POST_NETS:
        if not dry_run:
            copy_unquantized(jpn.stem, jpn, STT_OPTIMIZED_DIR)

    # ── Copy other supporting assets ──────────────────────────────────
    for asset_name in ["joint_pre_net.onnx", "vocab.json", "language_masks.json"]:
        asset_src = STT_ASSETS / asset_name
        if asset_src.exists() and not dry_run:
            copy_unquantized(asset_name, asset_src, STT_OPTIMIZED_DIR)

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
        logger.info(f"  {'TOTAL':20s} {total_before:7.1f} MB → {total_after:7.1f} MB  ({total_reduction}% ↓)")
        logger.info("=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize STT ONNX models to INT8")
    parser.add_argument("--dry-run", action="store_true", help="Log only, do not write files")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
