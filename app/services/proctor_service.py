"""
Proctor behaviour detection service for the AI Proctoring Backend.

Model
-----
``proctoring.pt`` is an Ultralytics YOLOv8 object detection model with six
classes:
  0 → book
  1 → cell phone
  2 → headphone
  3 → laptop
  4 → person
  5 → tv

Behavioural events are **derived** from the detected objects:

  no_face           — no ``person`` detected in the frame
  multiple_persons  — more than one unique ``person`` detected
  mobile_detected   — ``cell phone`` detected (overrides worker_manager logic
                      so the event is also available from this service)

Note: ``looking_away``, ``suspicious_movement``, and ``bad_posture`` require
a pose/landmark model and cannot be derived from this object detector.
Those events are intentionally omitted rather than fabricated.

Design decisions
----------------
* Uses the Ultralytics YOLO API for consistency with YOLOService.
* Person deduplication uses IoU-based NMS (same logic as WorkerManager) to
  avoid counting the same person twice when YOLO fires multiple overlapping
  boxes.
* Singleton pattern: model is loaded once at startup and reused per request.
* Warm-up pass eliminates first-inference JIT overhead.
* All exceptions during inference are caught; the method returns ``[]``
  so the worker pipeline is never interrupted by a bad frame.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from app.core.config import get_config
from app.services.yolo_service import DeviceConfig, _resolve_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class indices in proctoring.pt
# ---------------------------------------------------------------------------

_CLS_BOOK       = 0
_CLS_CELL_PHONE = 1
_CLS_HEADPHONE  = 2
_CLS_LAPTOP     = 3
_CLS_PERSON     = 4
_CLS_TV         = 5

# IoU threshold for person deduplication
_PERSON_IOU_THRESHOLD: float = 0.3


def _iou(box_a: list, box_b: list) -> float:
    """Compute Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    iw   = max(0.0, ix2 - ix1)
    ih   = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _count_unique_persons(person_boxes: list) -> int:
    """Count unique persons using IoU-based deduplication.

    YOLO sometimes fires multiple overlapping boxes for the same person
    (e.g. body + face).  Boxes with IoU > ``_PERSON_IOU_THRESHOLD`` are
    merged into a single person.

    Args:
        person_boxes: List of ``[x1, y1, x2, y2, conf]`` for each person box.

    Returns:
        Number of unique persons.
    """
    if not person_boxes:
        return 0
    # Sort by confidence descending; greedily keep non-overlapping boxes.
    sorted_boxes = sorted(person_boxes, key=lambda b: b[4], reverse=True)
    kept: list = []
    for box in sorted_boxes:
        duplicate = any(_iou(box[:4], k[:4]) > _PERSON_IOU_THRESHOLD for k in kept)
        if not duplicate:
            kept.append(box)
    return len(kept)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ProctorService:
    """Singleton Ultralytics YOLO behavioural event detector.

    Derives proctoring events from detected objects in the frame.

    Usage::

        # Once at startup:
        ProctorService.initialize("app/models/proctoring.pt", enable_gpu=False)

        # Per frame:
        events: List[str] = ProctorService.analyze(bgr_frame)
    """

    _model: Optional[YOLO] = None
    _device_cfg: Optional[DeviceConfig] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def initialize(cls, model_path: str, enable_gpu: bool) -> None:
        """Load ``proctoring.pt`` via the Ultralytics YOLO API.

        Idempotent — a second call is a no-op if the model is already loaded.

        Args:
            model_path: Filesystem path to ``proctoring.pt``.
            enable_gpu: Use ``cuda:0`` when CUDA is available.

        Raises:
            FileNotFoundError: If *model_path* does not exist on disk.
        """
        if cls._model is not None:
            logger.debug("ProctorService already initialised; skipping reload.")
            return

        t0 = time.perf_counter()
        cls._device_cfg = _resolve_device(enable_gpu)

        logger.info(
            "Initialising ProctorService",
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
            logger.debug("ProctorService warm-up pass complete")
        except Exception:
            logger.debug("ProctorService warm-up pass skipped (non-fatal)")

        load_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "ProctorService initialised successfully",
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
    def analyze(cls, frame: np.ndarray) -> List[str]:
        """Detect behavioural events in a video frame.

        Runs the proctoring object detector and derives events:
          - ``no_face``          when no person is detected
          - ``multiple_persons`` when more than one unique person is detected
          - ``mobile_detected``  when a cell phone is detected

        Args:
            frame: BGR ``uint8`` ndarray of shape ``(H, W, 3)``.

        Returns:
            List of active event name strings.  Returns ``[]`` when the model
            is not initialised or any exception occurs.  Never raises.
        """
        if cls._model is None:
            logger.warning(
                "ProctorService.analyze called before initialize(); returning []."
            )
            return []

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

            person_boxes: list = []
            mobile_detected = False

            for result in results:
                for box in result.boxes:
                    cls_idx = int(box.cls[0])
                    conf    = float(box.conf[0])
                    xyxy    = box.xyxy[0].tolist()

                    if cls_idx == _CLS_PERSON:
                        person_boxes.append([*xyxy, conf])
                    elif cls_idx == _CLS_CELL_PHONE:
                        mobile_detected = True

            person_count = _count_unique_persons(person_boxes)

            active_events: List[str] = []

            if person_count == 0:
                active_events.append("no_face")
            elif person_count > 1:
                active_events.append("multiple_persons")

            if mobile_detected:
                active_events.append("mobile_detected")

            logger.debug(
                "ProctorService.analyze complete",
                extra={
                    "active_events": active_events,
                    "person_count": person_count,
                    "mobile_detected": mobile_detected,
                },
            )
            return active_events

        except Exception:
            logger.exception("ProctorService.analyze raised an unexpected exception.")
            return []
