"""
Unit and property-based tests for HeadsetService.

The HeadsetService now uses the Ultralytics YOLO API (same as YOLOService).
headset_model.pt classes:
  0 → Auriculares  (no headset)
  1 → Headphone    (headset present)

Tests covered
-------------
1.1  initialize() with enable_gpu=False sets _model and _device_cfg
1.2  initialize() is idempotent — second call is a no-op
1.3  initialize() with CUDA unavailable falls back to CPU
1.4  classify() returns False when model is not initialized
1.5  classify() returns True when a Headphone (class 1) detection is above threshold
1.6  classify() returns False when only Auriculares (class 0) detected
1.7  classify() returns False when no detections above threshold
1.8  classify() returns False (never raises) when inference throws an exception
1.9  Property test: for any valid BGR frame, classify() always returns bool and never raises
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.headset_service import HeadsetService
from app.services.yolo_service import DeviceConfig

_HEADSET_CLASS_IDX = 1   # Headphone
_NO_HEADSET_CLASS  = 0   # Auriculares


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


def _make_mock_box(cls_idx: int, conf: float) -> MagicMock:
    box = MagicMock()
    box.cls  = [cls_idx]
    box.conf = [conf]
    box.xyxy = [torch.tensor([10.0, 10.0, 100.0, 100.0])]
    return box


def _make_mock_yolo_model(boxes: list) -> MagicMock:
    """Return a mock Ultralytics YOLO model whose predict() returns *boxes*.

    Each item in *boxes* is a dict: {cls_idx, conf}.
    """
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = [_make_mock_box(b["cls_idx"], b["conf"]) for b in boxes]
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
# 1.1 – 1.3: HeadsetService.initialize
# ---------------------------------------------------------------------------

class TestHeadsetServiceInitialize:

    def setup_method(self):
        HeadsetService._model = None
        HeadsetService._device_cfg = None

    def test_initialize_cpu_sets_model_and_device_cfg(self, tmp_path):
        """1.1 initialize() with enable_gpu=False sets _model and _device_cfg."""
        mock_model = MagicMock()
        with patch("app.services.headset_service.YOLO", return_value=mock_model):
            HeadsetService.initialize(str(tmp_path / "headset_model.pt"), enable_gpu=False)

        assert HeadsetService._model is mock_model
        assert HeadsetService._device_cfg is not None
        assert HeadsetService._device_cfg.device.type == "cpu"
        assert HeadsetService._device_cfg.use_fp16 is False

    def test_initialize_is_idempotent(self, tmp_path):
        """1.2 A second call to initialize() is a no-op — YOLO() is called only once."""
        mock_model = MagicMock()
        with patch("app.services.headset_service.YOLO", return_value=mock_model) as mock_yolo:
            HeadsetService.initialize(str(tmp_path / "headset_model.pt"), enable_gpu=False)
            HeadsetService.initialize(str(tmp_path / "headset_model.pt"), enable_gpu=False)

        mock_yolo.assert_called_once()
        assert HeadsetService._model is mock_model

    def test_initialize_cuda_unavailable_falls_back_to_cpu(self, tmp_path):
        """1.3 When CUDA is unavailable, enable_gpu=True still uses CPU."""
        mock_model = MagicMock()
        with patch("app.services.headset_service.YOLO", return_value=mock_model), \
             patch("torch.cuda.is_available", return_value=False):
            HeadsetService.initialize(str(tmp_path / "headset_model.pt"), enable_gpu=True)

        assert HeadsetService._device_cfg.device.type == "cpu"
        assert HeadsetService._device_cfg.use_fp16 is False


# ---------------------------------------------------------------------------
# 1.4 – 1.8: HeadsetService.classify
# ---------------------------------------------------------------------------

class TestHeadsetServiceClassify:

    def setup_method(self):
        HeadsetService._model = None
        HeadsetService._device_cfg = None

    def test_classify_returns_false_when_not_initialized(self):
        """1.4 classify() returns False when model is not initialized."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert HeadsetService.classify(frame) is False

    def test_classify_returns_true_when_headphone_detected(self):
        """1.5 classify() returns True when class 1 (Headphone) detected above threshold."""
        mock_model = _make_mock_yolo_model([
            {"cls_idx": _HEADSET_CLASS_IDX, "conf": 0.9},
        ])
        HeadsetService._model = mock_model
        HeadsetService._device_cfg = _cpu_device_cfg()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("app.services.headset_service.get_config") as mock_cfg:
            mock_cfg.return_value.YOLO_CONFIDENCE = 0.4
            result = HeadsetService.classify(frame)

        assert result is True

    def test_classify_returns_false_when_only_auriculares_detected(self):
        """1.6 classify() returns False when only class 0 (Auriculares) detected."""
        mock_model = _make_mock_yolo_model([
            {"cls_idx": _NO_HEADSET_CLASS, "conf": 0.9},
        ])
        HeadsetService._model = mock_model
        HeadsetService._device_cfg = _cpu_device_cfg()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("app.services.headset_service.get_config") as mock_cfg:
            mock_cfg.return_value.YOLO_CONFIDENCE = 0.4
            result = HeadsetService.classify(frame)

        assert result is False

    def test_classify_returns_false_when_no_detections(self):
        """1.7 classify() returns False when no detections above threshold."""
        mock_model = _make_mock_yolo_model([])
        HeadsetService._model = mock_model
        HeadsetService._device_cfg = _cpu_device_cfg()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("app.services.headset_service.get_config") as mock_cfg:
            mock_cfg.return_value.YOLO_CONFIDENCE = 0.4
            result = HeadsetService.classify(frame)

        assert result is False

    def test_classify_returns_false_on_inference_exception(self):
        """1.8 classify() returns False (never raises) when inference throws an exception."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("inference exploded")
        HeadsetService._model = mock_model
        HeadsetService._device_cfg = _cpu_device_cfg()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("app.services.headset_service.get_config") as mock_cfg:
            mock_cfg.return_value.YOLO_CONFIDENCE = 0.4
            result = HeadsetService.classify(frame)

        assert result is False


# ---------------------------------------------------------------------------
# 1.9: Property test — classify() always returns bool and never raises
# ---------------------------------------------------------------------------

@given(frame=valid_bgr_frame_strategy())
@settings(max_examples=50)
def test_property_1_9_classify_always_returns_bool(frame: np.ndarray) -> None:
    """Property 1.9: For any valid BGR frame, classify() always returns bool."""
    mock_model = _make_mock_yolo_model([
        {"cls_idx": _HEADSET_CLASS_IDX, "conf": 0.9},
    ])
    original_model = HeadsetService._model
    original_cfg   = HeadsetService._device_cfg
    try:
        HeadsetService._model      = mock_model
        HeadsetService._device_cfg = _cpu_device_cfg()

        with patch("app.services.headset_service.get_config") as mock_cfg:
            mock_cfg.return_value.YOLO_CONFIDENCE = 0.4
            result = HeadsetService.classify(frame)

        assert isinstance(result, bool)
    finally:
        HeadsetService._model      = original_model
        HeadsetService._device_cfg = original_cfg
