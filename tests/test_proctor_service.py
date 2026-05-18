"""
Unit and property-based tests for ProctorService.

The ProctorService now uses the Ultralytics YOLO API (same as YOLOService)
and derives behavioural events from detected objects:
  - no_face          → no person detected
  - multiple_persons → more than one unique person detected
  - mobile_detected  → cell phone detected

Tests covered
-------------
2.1  initialize() with enable_gpu=False sets _model and _device_cfg
2.2  initialize() is idempotent — second call is a no-op
2.3  analyze() returns [] when model is not initialized
2.4  analyze() returns 'no_face' when no person is detected
2.5  analyze() returns 'multiple_persons' when >1 unique person detected
2.6  analyze() returns 'mobile_detected' when cell phone detected
2.7  analyze() returns [] when a single person is detected (clean frame)
2.8  analyze() returns [] (never raises) when inference throws an exception
2.9  Property test: for any valid BGR frame, analyze() always returns List[str] and never raises
2.10 Property test: all returned event names are in the known event set
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.proctor_service import ProctorService
from app.services.yolo_service import DeviceConfig

# Known events that ProctorService can emit
_KNOWN_EVENTS = {"no_face", "multiple_persons", "mobile_detected"}

# Class indices in proctoring.pt
_CLS_CELL_PHONE = 1
_CLS_PERSON     = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpu_device_cfg() -> DeviceConfig:
    return DeviceConfig(
        device=torch.device("cpu"),
        use_fp16=False,
        device_name="CPU",
        cuda_available=False,
    )


def _make_mock_box(cls_idx: int, conf: float, xyxy: list) -> MagicMock:
    """Return a mock Ultralytics box object."""
    box = MagicMock()
    box.cls  = [cls_idx]
    box.conf = [conf]
    box.xyxy = [torch.tensor(xyxy, dtype=torch.float32)]
    return box


def _make_mock_yolo_model(boxes: list) -> MagicMock:
    """Return a mock Ultralytics YOLO model whose predict() returns *boxes*.

    Each item in *boxes* is a dict: {cls_idx, conf, xyxy}.
    """
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = [
        _make_mock_box(b["cls_idx"], b["conf"], b["xyxy"]) for b in boxes
    ]
    mock_model.predict.return_value = [mock_result]
    return mock_model


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def valid_bgr_frame_strategy():
    return st.builds(
        lambda h, w: np.random.randint(0, 256, (h, w, 3), dtype=np.uint8),
        h=st.integers(min_value=64, max_value=640),
        w=st.integers(min_value=64, max_value=640),
    )


# ---------------------------------------------------------------------------
# 2.1 – 2.2: ProctorService.initialize
# ---------------------------------------------------------------------------

class TestProctorServiceInitialize:

    def setup_method(self):
        ProctorService._model = None
        ProctorService._device_cfg = None

    def test_initialize_cpu_sets_model_and_device_cfg(self, tmp_path):
        """2.1 initialize() with enable_gpu=False sets _model and _device_cfg."""
        mock_model = MagicMock()
        with patch("app.services.proctor_service.YOLO", return_value=mock_model):
            ProctorService.initialize(str(tmp_path / "proctoring.pt"), enable_gpu=False)

        assert ProctorService._model is mock_model
        assert ProctorService._device_cfg is not None
        assert ProctorService._device_cfg.device.type == "cpu"
        assert ProctorService._device_cfg.use_fp16 is False

    def test_initialize_is_idempotent(self, tmp_path):
        """2.2 A second call to initialize() is a no-op — YOLO() is called only once."""
        mock_model = MagicMock()
        with patch("app.services.proctor_service.YOLO", return_value=mock_model) as mock_yolo:
            ProctorService.initialize(str(tmp_path / "proctoring.pt"), enable_gpu=False)
            ProctorService.initialize(str(tmp_path / "proctoring.pt"), enable_gpu=False)

        mock_yolo.assert_called_once()
        assert ProctorService._model is mock_model

    def test_initialize_cuda_unavailable_falls_back_to_cpu(self, tmp_path):
        """initialize() with enable_gpu=True falls back to CPU when CUDA unavailable."""
        mock_model = MagicMock()
        with patch("app.services.proctor_service.YOLO", return_value=mock_model), \
             patch("torch.cuda.is_available", return_value=False):
            ProctorService.initialize(str(tmp_path / "proctoring.pt"), enable_gpu=True)

        assert ProctorService._device_cfg.device.type == "cpu"
        assert ProctorService._device_cfg.use_fp16 is False


# ---------------------------------------------------------------------------
# 2.3 – 2.8: ProctorService.analyze
# ---------------------------------------------------------------------------

class TestProctorServiceAnalyze:

    def setup_method(self):
        ProctorService._model = None
        ProctorService._device_cfg = None

    def test_analyze_returns_empty_when_not_initialized(self):
        """2.3 analyze() returns [] when model is not initialized."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert ProctorService.analyze(frame) == []

    def test_analyze_returns_no_face_when_no_person(self):
        """2.4 analyze() returns ['no_face'] when no person is detected."""
        mock_model = _make_mock_yolo_model([])  # no detections
        ProctorService._model = mock_model
        ProctorService._device_cfg = _cpu_device_cfg()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("app.services.proctor_service.get_config") as mock_cfg:
            mock_cfg.return_value.YOLO_CONFIDENCE = 0.4
            result = ProctorService.analyze(frame)

        assert "no_face" in result

    def test_analyze_returns_multiple_persons_when_two_persons(self):
        """2.5 analyze() returns 'multiple_persons' when two non-overlapping persons detected."""
        boxes = [
            {"cls_idx": _CLS_PERSON, "conf": 0.9, "xyxy": [10.0, 10.0, 100.0, 200.0]},
            {"cls_idx": _CLS_PERSON, "conf": 0.85, "xyxy": [300.0, 10.0, 500.0, 200.0]},
        ]
        mock_model = _make_mock_yolo_model(boxes)
        ProctorService._model = mock_model
        ProctorService._device_cfg = _cpu_device_cfg()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("app.services.proctor_service.get_config") as mock_cfg:
            mock_cfg.return_value.YOLO_CONFIDENCE = 0.4
            result = ProctorService.analyze(frame)

        assert "multiple_persons" in result
        assert "no_face" not in result

    def test_analyze_returns_mobile_detected_when_cell_phone(self):
        """2.6 analyze() returns 'mobile_detected' when a cell phone is detected."""
        boxes = [
            {"cls_idx": _CLS_PERSON,     "conf": 0.9,  "xyxy": [10.0, 10.0, 100.0, 200.0]},
            {"cls_idx": _CLS_CELL_PHONE, "conf": 0.85, "xyxy": [200.0, 50.0, 280.0, 120.0]},
        ]
        mock_model = _make_mock_yolo_model(boxes)
        ProctorService._model = mock_model
        ProctorService._device_cfg = _cpu_device_cfg()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("app.services.proctor_service.get_config") as mock_cfg:
            mock_cfg.return_value.YOLO_CONFIDENCE = 0.4
            result = ProctorService.analyze(frame)

        assert "mobile_detected" in result
        assert "no_face" not in result

    def test_analyze_returns_empty_for_single_clean_person(self):
        """2.7 analyze() returns [] when exactly one person is detected and no phone."""
        boxes = [
            {"cls_idx": _CLS_PERSON, "conf": 0.9, "xyxy": [10.0, 10.0, 200.0, 400.0]},
        ]
        mock_model = _make_mock_yolo_model(boxes)
        ProctorService._model = mock_model
        ProctorService._device_cfg = _cpu_device_cfg()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("app.services.proctor_service.get_config") as mock_cfg:
            mock_cfg.return_value.YOLO_CONFIDENCE = 0.4
            result = ProctorService.analyze(frame)

        assert result == []

    def test_analyze_returns_empty_on_inference_exception(self):
        """2.8 analyze() returns [] (never raises) when inference throws an exception."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("inference exploded")
        ProctorService._model = mock_model
        ProctorService._device_cfg = _cpu_device_cfg()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("app.services.proctor_service.get_config") as mock_cfg:
            mock_cfg.return_value.YOLO_CONFIDENCE = 0.4
            result = ProctorService.analyze(frame)

        assert result == []

    def test_overlapping_person_boxes_deduplicated(self):
        """Two heavily overlapping person boxes count as one person (IoU dedup)."""
        boxes = [
            {"cls_idx": _CLS_PERSON, "conf": 0.9,  "xyxy": [10.0, 10.0, 200.0, 400.0]},
            {"cls_idx": _CLS_PERSON, "conf": 0.85, "xyxy": [15.0, 15.0, 205.0, 405.0]},
        ]
        mock_model = _make_mock_yolo_model(boxes)
        ProctorService._model = mock_model
        ProctorService._device_cfg = _cpu_device_cfg()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("app.services.proctor_service.get_config") as mock_cfg:
            mock_cfg.return_value.YOLO_CONFIDENCE = 0.4
            result = ProctorService.analyze(frame)

        # Should be treated as one person → no multiple_persons event
        assert "multiple_persons" not in result
        assert "no_face" not in result


# ---------------------------------------------------------------------------
# 2.9: Property test — analyze() always returns List[str] and never raises
# ---------------------------------------------------------------------------

@given(frame=valid_bgr_frame_strategy())
@settings(max_examples=50)
def test_property_2_9_analyze_always_returns_list_of_str(frame: np.ndarray) -> None:
    """Property 2.9: For any valid BGR frame, analyze() always returns List[str]."""
    mock_model = _make_mock_yolo_model([])  # no detections → no_face
    original_model = ProctorService._model
    original_cfg   = ProctorService._device_cfg
    try:
        ProctorService._model      = mock_model
        ProctorService._device_cfg = _cpu_device_cfg()

        with patch("app.services.proctor_service.get_config") as mock_cfg:
            mock_cfg.return_value.YOLO_CONFIDENCE = 0.4
            result = ProctorService.analyze(frame)

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)
    finally:
        ProctorService._model      = original_model
        ProctorService._device_cfg = original_cfg


# ---------------------------------------------------------------------------
# 2.10: Property test — all returned event names are in the known event set
# ---------------------------------------------------------------------------

@given(frame=valid_bgr_frame_strategy())
@settings(max_examples=50)
def test_property_2_10_analyze_returns_only_known_event_names(frame: np.ndarray) -> None:
    """Property 2.10: All returned event names are in the known event set."""
    # Simulate a frame with a person + phone to maximise events returned
    boxes = [
        {"cls_idx": _CLS_PERSON,     "conf": 0.9,  "xyxy": [10.0, 10.0, 100.0, 200.0]},
        {"cls_idx": _CLS_PERSON,     "conf": 0.85, "xyxy": [300.0, 10.0, 500.0, 200.0]},
        {"cls_idx": _CLS_CELL_PHONE, "conf": 0.8,  "xyxy": [200.0, 50.0, 280.0, 120.0]},
    ]
    mock_model = _make_mock_yolo_model(boxes)
    original_model = ProctorService._model
    original_cfg   = ProctorService._device_cfg
    try:
        ProctorService._model      = mock_model
        ProctorService._device_cfg = _cpu_device_cfg()

        with patch("app.services.proctor_service.get_config") as mock_cfg:
            mock_cfg.return_value.YOLO_CONFIDENCE = 0.4
            result = ProctorService.analyze(frame)

        for event_name in result:
            assert event_name in _KNOWN_EVENTS, (
                f"Unexpected event {event_name!r} not in {_KNOWN_EVENTS}"
            )
    finally:
        ProctorService._model      = original_model
        ProctorService._device_cfg = original_cfg
