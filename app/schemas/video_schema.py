"""
Pydantic schemas for video analysis endpoints.

This module defines the request/response models used by the video analysis
REST endpoint and WebSocket pipeline for structured data validation and
serialization.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Represents the coordinates of a detected object's bounding box.

    All coordinates are expressed as absolute pixel values relative to the
    original (pre-resize) frame dimensions.

    Attributes:
        x1: Left edge of the bounding box (horizontal start).
        y1: Top edge of the bounding box (vertical start).
        x2: Right edge of the bounding box (horizontal end).
        y2: Bottom edge of the bounding box (vertical end).
    """

    x1: float = Field(..., description="Left edge of the bounding box (pixels).")
    y1: float = Field(..., description="Top edge of the bounding box (pixels).")
    x2: float = Field(..., description="Right edge of the bounding box (pixels).")
    y2: float = Field(..., description="Bottom edge of the bounding box (pixels).")


class Detection(BaseModel):
    """Represents a single object detection result from the YOLO inference pipeline.

    Attributes:
        class_name: Human-readable COCO class label (e.g. ``"person"``,
            ``"cell phone"``).
        confidence: Detection confidence score in the range ``[0.0, 1.0]``.
        bounding_box: Pixel coordinates of the detected object within the frame.
    """

    class_name: str = Field(
        ...,
        description="COCO class label of the detected object (e.g. 'person', 'cell phone').",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score in [0.0, 1.0].",
    )
    bounding_box: BoundingBox = Field(
        ...,
        description="Pixel-coordinate bounding box of the detected object.",
    )


class VideoAnalysisResponse(BaseModel):
    """Response model returned by ``POST /api/v1/video/analyze-frame``.

    Encapsulates the full result of running a single video frame through the
    YOLO, Headset, and Proctor inference pipelines together with the Risk
    Engine scoring.

    Attributes:
        session_id: Unique identifier for the proctoring session this frame
            belongs to.
        timestamp: UTC timestamp at which the analysis was completed.
        events: List of behavioural event names detected in this frame
            (e.g. ``"no_face"``, ``"multiple_persons"``, ``"looking_away"``).
        person_count: Number of ``"person"`` class detections found by YOLO.
        risk_score: Numeric risk score computed by the Risk Engine for this
            frame's events.
        risk_level: Categorical risk classification — one of ``"LOW"``,
            ``"MEDIUM"``, or ``"HIGH"``.
        detections: Full list of object detections returned by the YOLO
            inference pipeline.
    """

    session_id: str = Field(
        ...,
        description="Unique identifier for the proctoring session.",
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp at which the frame analysis was completed.",
    )
    events: List[str] = Field(
        default_factory=list,
        description=(
            "Behavioural event names detected in this frame "
            "(e.g. 'no_face', 'multiple_persons', 'looking_away')."
        ),
    )
    person_count: int = Field(
        ...,
        ge=0,
        description="Number of 'person' class detections found by YOLO.",
    )
    risk_score: int = Field(
        ...,
        ge=0,
        description="Numeric risk score computed by the Risk Engine for this frame.",
    )
    risk_level: str = Field(
        ...,
        description="Categorical risk level: 'LOW', 'MEDIUM', or 'HIGH'.",
    )
    detections: List[Detection] = Field(
        default_factory=list,
        description="Full list of YOLO object detections for this frame.",
    )
