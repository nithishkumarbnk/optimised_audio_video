"""
Property-based tests for the YOLO Service.

**Validates: Requirements 8.5**

Properties tested
-----------------
Property 11: YOLO Detection Fields Completeness
    For any valid BGR frame, every Detection returned by ``YOLOService.detect``
    must have a non-null ``class_name``, a ``confidence`` in ``[0, 1]``, and a
    ``bounding_box`` where ``x1 <= x2`` and ``y1 <= y2``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.yolo_service import (
    RELEVANT_CLASSES,
    DeviceConfig,
    YOLOService,
    _resolve_device,
)


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


def _make_mock_yolo_model(detections: list) -> MagicMock:
    """Return a mock Ultralytics YOLO model whose predict() returns *detections*.

    Each item in *detections* is a dict with keys:
        cls_idx (int), conf (float), xyxy (list[float] of length 4)
    """
    mock_model = MagicMock()

    result_boxes = []
    for d in detections:
        box = MagicMock()
        box.cls = [d["cls_idx"]]
        box.conf = [d["conf"]]
        # box.xyxy[0] must be a tensor so that .tolist() works in yolo_service._run_inference
        box.xyxy = [torch.tensor(d["xyxy"], dtype=torch.float32)]
        result_boxes.append(box)

    mock_result = MagicMock()
    mock_result.boxes = result_boxes
    mock_model.predict.return_value = [mock_result]
    return mock_model


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def valid_bgr_frame_strategy():
    """Generate valid BGR frames as numpy uint8 arrays of varying sizes."""
    return st.builds(
        lambda h, w: np.random.randint(0, 256, (h, w, 3), dtype=np.uint8),
        h=st.integers(min_value=64, max_value=640),
        w=st.integers(min_value=64, max_value=640),
    )


def synthetic_detections_strategy():
    """Generate a list of synthetic detection dicts for the mock YOLO model."""
    relevant_indices = list(RELEVANT_CLASSES.keys())

    def _make(n_detections: int, seed: int):
        rng = np.random.default_rng(seed)
        detections = []
        for i in range(n_detections):
            x1 = float(rng.uniform(0, 500))
            y1 = float(rng.uniform(0, 400))
            x2 = x1 + float(rng.uniform(10, 100))
            y2 = y1 + float(rng.uniform(10, 100))
            cls_idx = relevant_indices[i % len(relevant_indices)]
            conf = float(rng.uniform(0.5, 1.0))
            detections.append({"cls_idx": cls_idx, "conf": conf, "xyxy": [x1, y1, x2, y2]})
        return detections

    return st.builds(
        _make,
        n_detections=st.integers(min_value=0, max_value=20),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )


# ---------------------------------------------------------------------------
# Unit tests — YOLOService.initialize
# ---------------------------------------------------------------------------


class TestYOLOServiceInitialize:
    """Tests for YOLOService.initialize."""

    def setup_method(self):
        YOLOService._model = None
        YOLOService._device_cfg = None

    def test_initialize_cpu_sets_model(self, tmp_path):
        """initialize with enable_gpu=False sets _model and _device_cfg."""
        mock_model = MagicMock()
        with patch("app.services.yolo_service.YOLO", return_value=mock_model):
            YOLOService.initialize(str(tmp_path / "model.pt"), enable_gpu=False)

        assert YOLOService._model is mock_model
        assert YOLOService._device_cfg is not None
        assert YOLOService._device_cfg.device.type == "cpu"
        assert YOLOService._device_cfg.use_fp16 is False

    def test_initialize_gpu_unavailable_falls_back_to_cpu(self, tmp_path):
        """When CUDA is unavailable, enable_gpu=True still uses CPU."""
        mock_model = MagicMock()
        with patch("app.services.yolo_service.YOLO", return_value=mock_model), \
             patch("torch.cuda.is_available", return_value=False):
            YOLOService.initialize(str(tmp_path / "model.pt"), enable_gpu=True)

        assert YOLOService._device_cfg.device.type == "cpu"
        assert YOLOService._device_cfg.use_fp16 is False

    def test_model_is_set_after_initialize(self, tmp_path):
        """_model is not None after initialize."""
        mock_model = MagicMock()
        with patch("app.services.yolo_service.YOLO", return_value=mock_model):
            YOLOService.initialize(str(tmp_path / "model.pt"), enable_gpu=False)

        assert YOLOService._model is not None


# ---------------------------------------------------------------------------
# Unit tests — YOLOService.detect without model
# ---------------------------------------------------------------------------


class TestYOLOServiceDetectWithoutModel:
    """detect() returns [] when model is not initialized."""

    def test_detect_without_model_returns_empty(self):
        original_model = YOLOService._model
        try:
            YOLOService._model = None
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = YOLOService.detect(frame)
            assert result == []
        finally:
            YOLOService._model = original_model


# ---------------------------------------------------------------------------
# Unit tests — YOLOService.detect with mocked model
# ---------------------------------------------------------------------------


class TestYOLOServiceDetectWithMockedModel:
    """detect() with a mocked Ultralytics YOLO model."""

    def _set_model(self, mock_model):
        YOLOService._model = mock_model
        YOLOService._device_cfg = _cpu_device_cfg()

    def test_detect_returns_list(self):
        """detect always returns a list."""
        detections = [
            {"cls_idx": 0, "conf": 0.9, "xyxy": [10.0, 20.0, 110.0, 120.0]},
        ]
        mock_model = _make_mock_yolo_model(detections)
        original_model = YOLOService._model
        original_cfg = YOLOService._device_cfg
        try:
            self._set_model(mock_model)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = YOLOService.detect(frame)
            assert isinstance(result, list)
        finally:
            YOLOService._model = original_model
            YOLOService._device_cfg = original_cfg

    def test_detect_returns_empty_for_no_detections(self):
        """No detections from model returns []."""
        mock_model = _make_mock_yolo_model([])
        original_model = YOLOService._model
        original_cfg = YOLOService._device_cfg
        try:
            self._set_model(mock_model)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = YOLOService.detect(frame)
            assert result == []
        finally:
            YOLOService._model = original_model
            YOLOService._device_cfg = original_cfg

    def test_detect_filters_irrelevant_classes(self):
        """Detections for non-relevant COCO classes are filtered out."""
        # class index 1 = bicycle — not in RELEVANT_CLASSES
        detections = [
            {"cls_idx": 1, "conf": 0.99, "xyxy": [10.0, 20.0, 110.0, 120.0]},
        ]
        mock_model = _make_mock_yolo_model(detections)
        original_model = YOLOService._model
        original_cfg = YOLOService._device_cfg
        try:
            self._set_model(mock_model)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = YOLOService.detect(frame)
            assert result == []
        finally:
            YOLOService._model = original_model
            YOLOService._device_cfg = original_cfg

    def test_detect_returns_person_detection(self):
        """High-confidence person detection (class 0) is returned."""
        detections = [
            {"cls_idx": 0, "conf": 0.99, "xyxy": [10.0, 20.0, 110.0, 120.0]},
        ]
        mock_model = _make_mock_yolo_model(detections)
        original_model = YOLOService._model
        original_cfg = YOLOService._device_cfg
        try:
            self._set_model(mock_model)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = YOLOService.detect(frame)
            assert len(result) == 1
            assert result[0].class_name == "person"
            assert result[0].confidence == pytest.approx(0.99)
        finally:
            YOLOService._model = original_model
            YOLOService._device_cfg = original_cfg

    def test_detect_exception_returns_empty_list(self):
        """Any exception during inference returns [] without raising."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("inference error")
        original_model = YOLOService._model
        original_cfg = YOLOService._device_cfg
        try:
            self._set_model(mock_model)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = YOLOService.detect(frame)
            assert result == []
        finally:
            YOLOService._model = original_model
            YOLOService._device_cfg = original_cfg


# ---------------------------------------------------------------------------
# Property 11: YOLO Detection Fields Completeness
# **Validates: Requirements 8.5**
# ---------------------------------------------------------------------------


@given(
    frame=valid_bgr_frame_strategy(),
    detections=synthetic_detections_strategy(),
)
@settings(max_examples=100)
def test_property_11_detection_fields_completeness(
    frame: np.ndarray,
    detections: list,
) -> None:
    """Property 11: Every Detection has non-null class_name, confidence in [0,1],
    and bounding_box with x1 <= x2 and y1 <= y2.

    **Validates: Requirements 8.5**
    """
    mock_model = _make_mock_yolo_model(detections)
    original_model = YOLOService._model
    original_cfg = YOLOService._device_cfg
    try:
        YOLOService._model = mock_model
        YOLOService._device_cfg = _cpu_device_cfg()
        result = YOLOService.detect(frame)

        assert isinstance(result, list), "detect must return a list"

        for det in result:
            assert det.class_name is not None, "class_name must not be None"
            assert len(det.class_name) > 0, "class_name must not be empty"
            assert det.class_name in RELEVANT_CLASSES.values(), (
                f"class_name {det.class_name!r} is not a relevant COCO class"
            )
            assert 0.0 <= det.confidence <= 1.0, (
                f"confidence {det.confidence} is outside [0, 1]"
            )
            bb = det.bounding_box
            assert bb.x1 <= bb.x2, f"bounding_box x1={bb.x1} > x2={bb.x2}"
            assert bb.y1 <= bb.y2, f"bounding_box y1={bb.y1} > y2={bb.y2}"
    finally:
        YOLOService._model = original_model
        YOLOService._device_cfg = original_cfg
