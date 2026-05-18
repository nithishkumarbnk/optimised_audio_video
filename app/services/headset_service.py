"""
Headset detection service for the AI Proctoring Backend.

Model
-----
``headset_model.pt`` is an Ultralytics YOLOv8 detection model with two classes:
  0 → Auriculares  (no headset / ear without headset)
  1 → Headphone    (headset present)

Returns ``True`` when any detection of class 1 (Headphone) exceeds the
configured ``YOLO_CONFIDENCE`` threshold.

Design decisions
----------------
* Uses the Ultralytics YOLO API (same as YOLOService) for consistency and
  to benefit from the optimised preprocessing + NMS pipeline.
* Singleton pattern: model is loaded once at startup and reused per request.
* Warm-up pass eliminates first-inference JIT overhead.
* All exceptions during inference are caught; the method returns ``False``
  so the worker pipeline is never interrupted by a bad frame.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from ultralytics import YOLO

from app.core.config import get_config
from app.services.yolo_service import DeviceConfig, _resolve_device

logger = logging.getLogger(__name__)

# Class index for headset presence in headset_model.pt
_HEADSET_CLASS_IDX: int = 1   # "Headphone"


class HeadsetService:
    """Singleton Ultralytics YOLO headset presence classifier.

    Usage::

        # Once at startup:
        HeadsetService.initialize("app/models/headset_model.pt", enable_gpu=False)

        # Per frame:
        has_headset: bool = HeadsetService.classify(bgr_frame)
    """

    _model: Optional[YOLO] = None
    _device_cfg: Optional[DeviceConfig] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def initialize(cls, model_path: str, enable_gpu: bool) -> None:
        """Load ``headset_model.pt`` via the Ultralytics YOLO API.

        Idempotent — a second call is a no-op if the model is already loaded.

        Args:
            model_path: Filesystem path to ``headset_model.pt``.
            enable_gpu: Use ``cuda:0`` when CUDA is available.

        Raises:
            FileNotFoundError: If *model_path* does not exist on disk.
        """
        if cls._model is not None:
            logger.debug("HeadsetService already initialised; skipping reload.")
            return

        t0 = time.perf_counter()
        cls._device_cfg = _resolve_device(enable_gpu)

        logger.info(
            "Initialising HeadsetService",
            extra={
                "model_path": model_path,
                "device": str(cls._device_cfg.device),
                "use_fp16": cls._device_cfg.use_fp16,
            },
        )

        cls._model = YOLO(model_path)
        cls._model.to(cls._device_cfg.device)

        # Warm-up pass to eliminate first-inference JIT overhead.
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            cls._model.predict(source=dummy, verbose=False, conf=0.9)
            logger.debug("HeadsetService warm-up pass complete")
        except Exception:
            logger.debug("HeadsetService warm-up pass skipped (non-fatal)")

        load_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "HeadsetService initialised successfully",
            extra={
                "model_path": model_path,
                "device": str(cls._device_cfg.device),
                "load_time_ms": round(load_ms, 2),
                "classes": cls._model.names,
            },
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @classmethod
    def classify(cls, frame: np.ndarray) -> bool:
        """Return ``True`` if a headset (Headphone) is detected in the frame.

        Args:
            frame: BGR ``uint8`` ndarray of shape ``(H, W, 3)``.

        Returns:
            ``True`` when any detection of class 1 (Headphone) has confidence
            ≥ ``YOLO_CONFIDENCE``.  Returns ``False`` when the model is not
            initialised or any exception occurs.  Never raises.
        """
        if cls._model is None:
            logger.warning(
                "HeadsetService.classify called before initialize(); returning False."
            )
            return False

        try:
            config    = get_config()
            threshold = config.YOLO_CONFIDENCE

            with __import__("torch").no_grad():
                results = cls._model.predict(
                    source=frame,
                    device=cls._device_cfg.device,   # type: ignore[union-attr]
                    half=cls._device_cfg.use_fp16,    # type: ignore[union-attr]
                    conf=threshold,
                    verbose=False,
                )

            headset_detected = False
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == _HEADSET_CLASS_IDX:
                        headset_detected = True
                        break
                if headset_detected:
                    break

            logger.debug(
                "HeadsetService classification complete",
                extra={"headset_detected": headset_detected},
            )
            return headset_detected

        except Exception:
            logger.exception("HeadsetService.classify raised an unexpected exception.")
            return False
