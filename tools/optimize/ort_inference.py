"""
ONNX Runtime Inference Wrappers
=================================
Production-ready ORT inference classes for the optimized models.
These replace the PyTorch services when running the INT8 ONNX variants.

Classes:
  - ORTSession        : Base session wrapper with logging and error handling
  - YOLOOrtInference  : YOLO object detection via ORT
  - STTOrtInference   : STT encoder inference via ORT (for latency testing)

Usage:
    from tools.optimize.ort_inference import YOLOOrtInference
    yolo = YOLOOrtInference("app/models/optimized/yolo_int8/yolov8n_int8.onnx")
    output = yolo.run(frame_np)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base ORT session wrapper
# ---------------------------------------------------------------------------

class ORTSession:
    """Thread-safe ONNX Runtime session wrapper.

    Loads the model once and exposes a `run()` method.
    All exceptions during inference are caught and logged — the method
    returns None so the caller can handle gracefully.
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: Optional[List] = None,
        enable_gpu: bool = False,
    ) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        if providers is None:
            if enable_gpu:
                available = ort.get_available_providers()
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "CUDAExecutionProvider" in available
                    else ["CPUExecutionProvider"]
                )
            else:
                providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # suppress verbose ORT logs
        # Use all available CPU threads for intra-op parallelism
        opts.intra_op_num_threads = 0  # 0 = use all cores

        t0 = time.perf_counter()
        self._session = ort.InferenceSession(
            str(self.model_path), opts, providers=providers
        )
        load_ms = round((time.perf_counter() - t0) * 1000, 1)

        self._input_names  = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]

        logger.info(
            "ORTSession loaded",
            extra={
                "model": self.model_path.name,
                "providers": providers,
                "load_ms": load_ms,
                "inputs": self._input_names,
            },
        )

    def run(self, inputs: Dict[str, np.ndarray]) -> Optional[List[np.ndarray]]:
        """Run inference.

        Args:
            inputs: Dict mapping input name → numpy array.

        Returns:
            List of output numpy arrays, or None on error.
        """
        try:
            return self._session.run(self._output_names, inputs)
        except Exception:
            logger.exception(f"ORT inference failed for {self.model_path.name}")
            return None

    @property
    def input_names(self) -> List[str]:
        return self._input_names

    @property
    def output_names(self) -> List[str]:
        return self._output_names


# ---------------------------------------------------------------------------
# YOLO ORT inference
# ---------------------------------------------------------------------------

class YOLOOrtInference:
    """YOLO object detection using an optimized ONNX Runtime session.

    Handles preprocessing (letterbox resize, BGR→RGB, normalise, CHW, batch)
    and postprocessing (confidence filter, class filter).

    Usage::

        yolo = YOLOOrtInference("app/models/optimized/yolo_int8/yolov8n_int8.onnx")
        detections = yolo.detect(bgr_frame, conf_threshold=0.4)
    """

    # COCO class indices relevant to proctoring
    RELEVANT_CLASSES = {0: "person", 62: "monitor", 63: "laptop", 67: "cell phone", 73: "book"}

    def __init__(self, model_path: str | Path, enable_gpu: bool = False) -> None:
        self._session = ORTSession(model_path, enable_gpu=enable_gpu)
        self._input_size = 640

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Letterbox resize + normalise → (1, 3, 640, 640) float32."""
        import cv2
        h, w = frame.shape[:2]
        scale = self._input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # Pad to square
        canvas = np.full((self._input_size, self._input_size, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]  # (1,3,640,640)
        return tensor, scale, (h, w)

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.4) -> List[Dict]:
        """Run detection on a BGR frame.

        Args:
            frame:          BGR uint8 ndarray (H, W, 3).
            conf_threshold: Minimum confidence to keep a detection.

        Returns:
            List of dicts: {class_name, confidence, bbox: [x1,y1,x2,y2]}.
        """
        tensor, scale, orig_shape = self._preprocess(frame)
        outputs = self._session.run({"images": tensor})
        if outputs is None:
            return []

        # Output shape: (1, num_classes+4, 8400) — YOLOv8 format
        raw = outputs[0][0]  # (num_classes+4, 8400)
        preds = raw.T        # (8400, num_classes+4)

        boxes   = preds[:, :4]
        scores  = preds[:, 4:]
        cls_ids = scores.argmax(axis=1)
        confs   = scores.max(axis=1)

        detections = []
        orig_h, orig_w = orig_shape
        for i, (conf, cls_id) in enumerate(zip(confs, cls_ids)):
            if conf < conf_threshold:
                continue
            if int(cls_id) not in self.RELEVANT_CLASSES:
                continue
            cx, cy, bw, bh = boxes[i]
            # Convert from 640x640 letterbox space back to original frame space
            x1 = (cx - bw / 2) / scale
            y1 = (cy - bh / 2) / scale
            x2 = (cx + bw / 2) / scale
            y2 = (cy + bh / 2) / scale
            detections.append({
                "class_name": self.RELEVANT_CLASSES[int(cls_id)],
                "confidence": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        return detections


# ---------------------------------------------------------------------------
# STT Encoder ORT inference (for latency testing / future integration)
# ---------------------------------------------------------------------------

class STTEncoderOrtInference:
    """Wraps the INT8 encoder ONNX for latency testing.

    The full STT pipeline still uses the model_onnx.py wrapper from the
    indic-conformer-600m directory. This class is provided for benchmarking
    the encoder in isolation.
    """

    def __init__(self, model_path: str | Path, enable_gpu: bool = False) -> None:
        self._session = ORTSession(model_path, enable_gpu=enable_gpu)

    def encode(self, audio_features: np.ndarray) -> Optional[np.ndarray]:
        """Run encoder inference.

        Args:
            audio_features: Float32 array of shape (1, T, 80) — mel features.

        Returns:
            Encoder output array or None on error.
        """
        input_name = self._session.input_names[0]
        outputs = self._session.run({input_name: audio_features})
        return outputs[0] if outputs else None
