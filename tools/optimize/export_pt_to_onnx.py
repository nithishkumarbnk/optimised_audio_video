"""
PT → ONNX Export
=================
Exports all three PyTorch .pt vision models to ONNX format using the
Ultralytics export API:
  - yolov8n.pt       → yolov8n.onnx
  - proctoring.pt    → proctoring.onnx
  - headset_model.pt → headset_model.onnx

Usage:
    python tools/optimize/export_pt_to_onnx.py [--imgsz 640] [--opset 17]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

from config import LOG_FORMAT, PT_SOURCES, YOLO_ONNX_DIR, ONNX_OPSET

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("export_pt_to_onnx")


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 ** 2), 2)


def export_model(name: str, pt_path: Path, output_dir: Path,
                 imgsz: int = 640, opset: int = ONNX_OPSET) -> dict:
    """Export a single .pt model to ONNX.

    Ultralytics export() writes the .onnx file next to the .pt file,
    then we move it to output_dir.

    Args:
        name:       Human-readable model name.
        pt_path:    Path to the .pt model file.
        output_dir: Directory to write the .onnx file into.
        imgsz:      Input image size (square).
        opset:      ONNX opset version.

    Returns:
        Dict with export metadata.
    """
    from ultralytics import YOLO

    if not pt_path.exists():
        raise FileNotFoundError(f"PT model not found: {pt_path}")

    size_pt = _size_mb(pt_path)
    logger.info(f"[{name}] Exporting {pt_path.name} ({size_pt} MB) to ONNX opset={opset}")

    t0 = time.perf_counter()

    model = YOLO(pt_path)
    # Ultralytics export() returns the path to the exported file
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,      # run onnx-simplifier for cleaner graph
        dynamic=False,       # fixed batch=1 for realtime inference
        half=False,          # FP32 — INT8 quantization done separately
        verbose=False,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    # Ultralytics writes the .onnx next to the .pt file
    exported_path = Path(exported) if exported else pt_path.with_suffix(".onnx")
    if not exported_path.exists():
        # Fallback: look for .onnx next to the .pt
        exported_path = pt_path.with_suffix(".onnx")

    dst = output_dir / exported_path.name
    shutil.move(str(exported_path), str(dst))

    size_onnx = _size_mb(dst)
    logger.info(
        f"[{name}] Exported: {size_pt} MB (PT) → {size_onnx} MB (ONNX) in {elapsed}s → {dst}"
    )

    return {
        "model": name,
        "pt_path": str(pt_path),
        "onnx_path": str(dst),
        "size_pt_mb": size_pt,
        "size_onnx_mb": size_onnx,
        "elapsed_s": elapsed,
        "imgsz": imgsz,
        "opset": opset,
    }


def run(imgsz: int = 640, opset: int = ONNX_OPSET) -> list[dict]:
    """Export all PT models to ONNX.

    Returns:
        List of export result dicts.
    """
    results = []

    logger.info("=" * 60)
    logger.info("PT → ONNX Export Pipeline")
    logger.info(f"Output dir : {YOLO_ONNX_DIR}")
    logger.info(f"Image size : {imgsz}x{imgsz}")
    logger.info(f"ONNX opset : {opset}")
    logger.info("=" * 60)

    for name, pt_path in PT_SOURCES.items():
        try:
            result = export_model(name, pt_path, YOLO_ONNX_DIR, imgsz=imgsz, opset=opset)
            results.append(result)
        except Exception as exc:
            logger.exception(f"[{name}] Export failed: {exc}")

    # ── Summary ───────────────────────────────────────────────────────
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("EXPORT SUMMARY")
        logger.info("=" * 60)
        for r in results:
            logger.info(
                f"  {r['model']:20s} {r['size_pt_mb']:7.1f} MB (PT) → "
                f"{r['size_onnx_mb']:7.1f} MB (ONNX)"
            )
        logger.info("=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PT models to ONNX")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--opset", type=int, default=ONNX_OPSET, help="ONNX opset version")
    args = parser.parse_args()
    run(imgsz=args.imgsz, opset=args.opset)


if __name__ == "__main__":
    main()
