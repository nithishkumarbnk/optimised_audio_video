"""
Image utility helpers for the AI Proctoring Backend.

Provides functions for decoding raw image bytes, preprocessing frames for
YOLO inference, and converting base64-encoded images to numpy arrays.
"""

import base64
from typing import Union

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR numpy array.

    Uses ``cv2.imdecode`` to support JPEG, PNG, and other OpenCV-compatible
    formats.

    Args:
        data: Raw image bytes (e.g. the body of a multipart file upload).

    Returns:
        A BGR numpy array with shape ``(H, W, 3)`` and dtype ``uint8``.

    Raises:
        ValueError: If ``data`` is empty or cannot be decoded as an image.
    """
    if not data:
        raise ValueError("Image data is empty.")

    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError(
            "cv2.imdecode returned None — the bytes could not be decoded as an image."
        )

    return frame


def preprocess_for_yolo(frame: np.ndarray) -> np.ndarray:
    """Preprocess a BGR frame for YOLOv8 ONNX inference.

    Applies the following transformations in order:

    1. Resize to 640 × 640 pixels.
    2. Convert colour order from BGR to RGB.
    3. Normalise pixel values to the range ``[0, 1]`` by dividing by 255.
    4. Transpose from HWC layout to CHW layout.
    5. Add a batch dimension so the final shape is ``(1, 3, 640, 640)``.

    Args:
        frame: A BGR numpy array with shape ``(H, W, 3)`` and dtype ``uint8``
            as returned by :func:`decode_image` or ``cv2.imread``.

    Returns:
        A float32 numpy array with shape ``(1, 3, 640, 640)`` ready to be
        passed as the input tensor to an ONNX inference session.
    """
    # 1. Resize to 640×640
    resized: np.ndarray = cv2.resize(frame, (640, 640))

    # 2. BGR → RGB
    rgb: np.ndarray = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 3. Normalise to [0, 1]
    normalised: np.ndarray = rgb.astype(np.float32) / 255.0

    # 4. HWC → CHW  (H=640, W=640, C=3) → (C=3, H=640, W=640)
    chw: np.ndarray = np.transpose(normalised, (2, 0, 1))

    # 5. Add batch dimension → (1, 3, 640, 640)
    batched: np.ndarray = np.expand_dims(chw, axis=0)

    return batched


def base64_to_frame(b64: str) -> np.ndarray:
    """Decode a base64-encoded image string into a BGR numpy array.

    Strips optional ``data:image/...;base64,`` prefixes before decoding so
    that both plain base64 strings and data-URI strings are accepted.

    Args:
        b64: A base64-encoded image string, optionally prefixed with a
            data-URI scheme (e.g. ``"data:image/jpeg;base64,/9j/4AAQ..."``).

    Returns:
        A BGR numpy array with shape ``(H, W, 3)`` and dtype ``uint8``.

    Raises:
        ValueError: If the base64 string cannot be decoded or the resulting
            bytes cannot be interpreted as an image.
    """
    # Strip data-URI prefix if present (e.g. "data:image/jpeg;base64,")
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        raw_bytes: bytes = base64.b64decode(b64)
    except Exception as exc:
        raise ValueError(f"Failed to base64-decode the image string: {exc}") from exc

    return decode_image(raw_bytes)
