"""
Output validation benchmark for the three PyTorch .pt vision services.

Validates:
  - Detection confidence values are in [0.0, 1.0]
  - Bounding box coordinates satisfy x1 ≤ x2 and y1 ≤ y2
  - All class_name values are in the known RELEVANT_CLASSES set
  - HeadsetService always returns bool
  - ProctorService always returns List[str] with known event names
  - No service raises an exception on any valid frame

Usage::

    python -m benchmark.pt_benchmark.output_validation [--frames 30] [--gpu]

Output: benchmark/outputs/pt_validation.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.pt_benchmark.bench_utils import make_frames, save_json
from app.services.yolo_service import YOLOService, RELEVANT_CLASSES
from app.services.headset_service import HeadsetService
from app.services.proctor_service import ProctorService

_MODELS     = _ROOT / "app" / "models"
_YOLO_PT    = str(_MODELS / "yolov8n.pt")
_HEADSET_PT = str(_MODELS / "headset_model.pt")
_PROCTOR_PT = str(_MODELS / "proctoring.pt")

_KNOWN_CLASSES = set(RELEVANT_CLASSES.values())
# Events that ProctorService can emit (derived from object detections)
_KNOWN_EVENTS  = {"no_face", "multiple_persons", "mobile_detected"}


def run(n_frames: int = 30, enable_gpu: bool = False) -> dict:
    """Run output validation checks.

    Args:
        n_frames:   Number of synthetic frames to validate.
        enable_gpu: Whether to request GPU inference.

    Returns:
        Dict with per-service pass/fail counts and any violation details.
    """
    print(f"\n{'='*60}")
    print(f"  PT Output Validation  |  frames={n_frames}  gpu={enable_gpu}")
    print(f"{'='*60}\n")

    YOLOService.initialize(_YOLO_PT, enable_gpu=enable_gpu)
    HeadsetService.initialize(_HEADSET_PT, enable_gpu=enable_gpu)
    ProctorService.initialize(_PROCTOR_PT, enable_gpu=enable_gpu)

    frames = make_frames(n_frames, height=480, width=640)

    violations: List[str] = []
    yolo_checks = headset_checks = proctor_checks = 0

    for i, frame in enumerate(frames):
        # ── YOLO ──────────────────────────────────────────────────────
        try:
            detections = YOLOService.detect(frame)
            assert isinstance(detections, list), "detect() must return list"
            for det in detections:
                yolo_checks += 1
                if not (0.0 <= det.confidence <= 1.0):
                    violations.append(
                        f"[YOLO frame {i}] confidence {det.confidence} out of [0,1]"
                    )
                if det.class_name not in _KNOWN_CLASSES:
                    violations.append(
                        f"[YOLO frame {i}] unknown class_name {det.class_name!r}"
                    )
                bb = det.bounding_box
                if bb.x1 > bb.x2:
                    violations.append(
                        f"[YOLO frame {i}] bbox x1={bb.x1} > x2={bb.x2}"
                    )
                if bb.y1 > bb.y2:
                    violations.append(
                        f"[YOLO frame {i}] bbox y1={bb.y1} > y2={bb.y2}"
                    )
        except Exception as exc:
            violations.append(f"[YOLO frame {i}] raised exception: {exc}")

        # ── Headset ───────────────────────────────────────────────────
        try:
            result = HeadsetService.classify(frame)
            headset_checks += 1
            if not isinstance(result, bool):
                violations.append(
                    f"[Headset frame {i}] classify() returned {type(result).__name__}, expected bool"
                )
        except Exception as exc:
            violations.append(f"[Headset frame {i}] raised exception: {exc}")

        # ── Proctor ───────────────────────────────────────────────────
        try:
            events = ProctorService.analyze(frame)
            proctor_checks += 1
            if not isinstance(events, list):
                violations.append(
                    f"[Proctor frame {i}] analyze() returned {type(events).__name__}, expected list"
                )
            for evt in events:
                if not isinstance(evt, str):
                    violations.append(
                        f"[Proctor frame {i}] event {evt!r} is not a str"
                    )
                elif evt not in _KNOWN_EVENTS:
                    violations.append(
                        f"[Proctor frame {i}] unknown event {evt!r}"
                    )
        except Exception as exc:
            violations.append(f"[Proctor frame {i}] raised exception: {exc}")

    passed = len(violations) == 0
    results = {
        "config": {"n_frames": n_frames, "enable_gpu": enable_gpu},
        "checks": {
            "yolo_detections_checked": yolo_checks,
            "headset_frames_checked":  headset_checks,
            "proctor_frames_checked":  proctor_checks,
        },
        "violations": violations,
        "passed": passed,
    }

    status = "✓ ALL CHECKS PASSED" if passed else f"✗ {len(violations)} VIOLATION(S)"
    print(f"YOLO detections checked : {yolo_checks}")
    print(f"Headset frames checked  : {headset_checks}")
    print(f"Proctor frames checked  : {proctor_checks}")
    print(f"\nResult: {status}")
    if violations:
        for v in violations[:20]:
            print(f"  • {v}")
        if len(violations) > 20:
            print(f"  ... and {len(violations) - 20} more")

    save_json(results, "pt_validation.json")
    print("\nDone.\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="PT output validation")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to validate")
    parser.add_argument("--gpu",    action="store_true",  help="Enable GPU inference")
    args = parser.parse_args()
    run(n_frames=args.frames, enable_gpu=args.gpu)


if __name__ == "__main__":
    main()
