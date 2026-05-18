"""
YOLO object detection service for the AI Proctoring Backend.

Wraps a YOLOv8 PyTorch model as a singleton via the Ultralytics YOLO API.

Public API
----------
* :meth:`YOLOService.initialize` — load the .pt model once at startup.
* :meth:`YOLOService.detect`     — run inference on a single BGR frame.

Design decisions
----------------
* Ultralytics ``YOLO.predict()`` handles letterbox resize, BGR→RGB, normalise,
  NMS, and coordinate rescaling back to the original frame resolution
  internally — no manual preprocessing pipeline is needed.
* ``torch.no_grad()`` is enforced inside ``predict()`` by Ultralytics; we
  additionally wrap the call to be explicit and future-proof.
* fp16 is enabled automatically when CUDA is available and ``enable_gpu=True``.
* The singleton is stored as a class-level attribute so the model is loaded
  exactly once per process and reused for every inference call.
* All exceptions during inference are caught and logged; the method returns
  ``[]`` so the worker pipeline is never interrupted by a bad frame.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from ultralytics import YOLO

from app.core.config import get_config
from app.schemas.video_schema import BoundingBox, Detection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO class index → human-readable name mapping
# ---------------------------------------------------------------------------

#: Only the five proctoring-relevant COCO classes are retained.
RELEVANT_CLASSES: dict[int, str] = {
    0:  "person",
    62: "monitor",
    63: "laptop",
    67: "cell phone",
    73: "book",
}


# ---------------------------------------------------------------------------
# Shared device configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeviceConfig:
    """Resolved PyTorch device settings shared by all three vision services."""

    device: torch.device
    use_fp16: bool
    device_name: str
    cuda_available: bool


def _resolve_device(enable_gpu: bool) -> DeviceConfig:
    """Resolve ``torch.device`` and fp16 flag from the ``enable_gpu`` flag.

    Args:
        enable_gpu: Whether GPU inference is requested via config.

    Returns:
        :class:`DeviceConfig` with ``device=cuda:0`` when CUDA is available
        and *enable_gpu* is ``True``, otherwise ``device=cpu``.
    """
    cuda_available = torch.cuda.is_available()

    if enable_gpu and cuda_available:
        device = torch.device("cuda:0")
        use_fp16 = True
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        if enable_gpu and not cuda_available:
            logger.warning(
                "CUDA requested but not available; falling back to CPU",
                extra={"cuda_available": False},
            )
        device = torch.device("cpu")
        use_fp16 = False
        device_name = "CPU"

    logger.info(
        "Device resolved",
        extra={
            "device": str(device),
            "use_fp16": use_fp16,
            "device_name": device_name,
            "cuda_available": cuda_available,
        },
    )
    return DeviceConfig(
        device=device,
        use_fp16=use_fp16,
        device_name=device_name,
        cuda_available=cuda_available,
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class YOLOService:
    """Singleton wrapper around a YOLOv8n PyTorch inference model.

    Usage::

        # Once at startup:
        YOLOService.initialize("app/models/yolov8n.pt", enable_gpu=False)

        # Per frame (called from WorkerManager via run_in_executor):
        detections: List[Detection] = YOLOService.detect(bgr_frame)
    """

    _model: Optional[YOLO] = None
    _device_cfg: Optional[DeviceConfig] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def initialize(cls, model_path: str, enable_gpu: bool) -> None:
        """Load ``yolov8n.pt`` via the Ultralytics YOLO API.

        Idempotent — a second call is a no-op if the model is already loaded.

        Args:
            model_path: Filesystem path to ``yolov8n.pt``.
            enable_gpu: Use ``cuda:0`` when CUDA is available.

        Raises:
            FileNotFoundError: If *model_path* does not exist on disk.
        """
        if cls._model is not None:
            logger.debug("YOLOService already initialised; skipping reload.")
            return

        t0 = time.perf_counter()
        cls._device_cfg = _resolve_device(enable_gpu)

        logger.info(
            "Loading YOLO PyTorch model",
            extra={
                "model_path": model_path,
                "device": str(cls._device_cfg.device),
                "use_fp16": cls._device_cfg.use_fp16,
            },
        )

        cls._model = YOLO(model_path)
        cls._model.to(cls._device_cfg.device)

        # Warm-up pass: eliminates first-inference JIT overhead so that the
        # first real frame is not penalised.
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            cls._model.predict(source=dummy, verbose=False, conf=0.9)
            logger.debug("YOLOService warm-up pass complete")
        except Exception:
            logger.debug("YOLOService warm-up pass skipped (non-fatal)")

        load_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "YOLO PyTorch model loaded successfully",
            extra={
                "model_path": model_path,
                "device": str(cls._device_cfg.device),
                "load_time_ms": round(load_ms, 2),
            },
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls, frame: np.ndarray) -> List[Detection]:
        """Run YOLOv8 inference on a single BGR frame.

        Ultralytics handles letterbox resize, BGR→RGB, normalisation, NMS,
        and coordinate rescaling back to the original frame resolution.

        Args:
            frame: BGR ``uint8`` ndarray with shape ``(H, W, 3)``.

        Returns:
            List of :class:`~app.schemas.video_schema.Detection` objects
            filtered to :data:`RELEVANT_CLASSES`.  Returns ``[]`` when the
            model is not initialised, no detections pass the confidence
            threshold, or any exception occurs.
        """
        if cls._model is None:
            logger.error("YOLOService.detect called before initialize()")
            return []

        try:
            return cls._run_inference(frame)
        except Exception:
            logger.exception("YOLO inference failed; returning empty detections")
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _run_inference(cls, frame: np.ndarray) -> List[Detection]:
        """Execute the Ultralytics YOLO inference pipeline.

        Args:
            frame: BGR ``uint8`` ndarray ``(H, W, 3)``.

        Returns:
            Filtered list of :class:`Detection` objects.
        """
        config = get_config()
        t0 = time.perf_counter()

        with torch.no_grad():
            results = cls._model.predict(  # type: ignore[union-attr]
                source=frame,
                device=cls._device_cfg.device,  # type: ignore[union-attr]
                half=cls._device_cfg.use_fp16,   # type: ignore[union-attr]
                conf=config.YOLO_CONFIDENCE,
                verbose=False,
            )

        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_idx = int(box.cls[0])
                if cls_idx not in RELEVANT_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(
                    Detection(
                        class_name=RELEVANT_CLASSES[cls_idx],
                        confidence=conf,
                        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "YOLO inference complete",
            extra={"count": len(detections), "elapsed_ms": round(elapsed_ms, 2)},
        )
        return detections
