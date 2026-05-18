"""
Concurrent stress test for the vision pipeline.

Simulates N concurrent "students" each submitting frames to the pipeline
via a thread pool (mirroring the asyncio run_in_executor pattern used in
WorkerManager).

Usage::

    python -m benchmark.pt_benchmark.concurrent_stress [--students 10] [--frames 20] [--gpu]

Output: benchmark/outputs/pt_concurrent_stress.json
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.pt_benchmark.bench_utils import make_frame, save_json, stats
from app.services.yolo_service import YOLOService
from app.services.headset_service import HeadsetService
from app.services.proctor_service import ProctorService

_MODELS     = _ROOT / "app" / "models"
_YOLO_PT    = str(_MODELS / "yolov8n.pt")
_HEADSET_PT = str(_MODELS / "headset_model.pt")
_PROCTOR_PT = str(_MODELS / "proctoring.pt")


def _process_frame(frame_idx: int, student_id: int) -> dict:
    """Process one frame through the full pipeline and return timing info."""
    import numpy as np
    frame = make_frame(480, 640, seed=frame_idx * 1000 + student_id)

    t0 = time.perf_counter()
    detections = YOLOService.detect(frame)
    t1 = time.perf_counter()
    headset    = HeadsetService.classify(frame)
    t2 = time.perf_counter()
    events     = ProctorService.analyze(frame)
    t3 = time.perf_counter()

    return {
        "student_id":    student_id,
        "frame_idx":     frame_idx,
        "yolo_ms":       (t1 - t0) * 1000,
        "headset_ms":    (t2 - t1) * 1000,
        "proctor_ms":    (t3 - t2) * 1000,
        "total_ms":      (t3 - t0) * 1000,
        "n_detections":  len(detections),
        "n_events":      len(events),
        "headset":       headset,
    }


def run(n_students: int = 10, frames_per_student: int = 20, enable_gpu: bool = False) -> dict:
    """Run the concurrent stress test.

    Args:
        n_students:         Number of simulated concurrent students.
        frames_per_student: Frames each student submits.
        enable_gpu:         Whether to request GPU inference.

    Returns:
        Dict with aggregate statistics.
    """
    print(f"\n{'='*60}")
    print(f"  PT Concurrent Stress  |  students={n_students}  "
          f"frames/student={frames_per_student}  gpu={enable_gpu}")
    print(f"{'='*60}\n")

    YOLOService.initialize(_YOLO_PT, enable_gpu=enable_gpu)
    HeadsetService.initialize(_HEADSET_PT, enable_gpu=enable_gpu)
    ProctorService.initialize(_PROCTOR_PT, enable_gpu=enable_gpu)

    total_tasks = n_students * frames_per_student
    print(f"Submitting {total_tasks} tasks across {n_students} threads...\n")

    all_results: List[dict] = []
    errors = 0

    wall_start = time.perf_counter()

    # Use the same thread-pool pattern as WorkerManager._process_video
    with ThreadPoolExecutor(max_workers=n_students) as pool:
        futures = {
            pool.submit(_process_frame, frame_idx, student_id): (student_id, frame_idx)
            for student_id in range(n_students)
            for frame_idx in range(frames_per_student)
        }
        for future in as_completed(futures):
            try:
                all_results.append(future.result())
            except Exception as exc:
                errors += 1
                print(f"  [ERROR] {exc}")

    wall_elapsed = time.perf_counter() - wall_start
    throughput   = total_tasks / wall_elapsed if wall_elapsed > 0 else 0.0

    total_times   = [r["total_ms"]   for r in all_results]
    yolo_times    = [r["yolo_ms"]    for r in all_results]
    headset_times = [r["headset_ms"] for r in all_results]
    proctor_times = [r["proctor_ms"] for r in all_results]

    results = {
        "config": {
            "n_students":         n_students,
            "frames_per_student": frames_per_student,
            "total_tasks":        total_tasks,
            "enable_gpu":         enable_gpu,
        },
        "summary": {
            "wall_elapsed_s":  round(wall_elapsed, 3),
            "throughput_fps":  round(throughput, 2),
            "errors":          errors,
            "tasks_completed": len(all_results),
        },
        "latency_ms": {
            "total":   stats(total_times),
            "yolo":    stats(yolo_times),
            "headset": stats(headset_times),
            "proctor": stats(proctor_times),
        },
    }

    print(f"Wall time        : {wall_elapsed:.2f}s")
    print(f"Throughput       : {throughput:.1f} frames/s")
    print(f"Tasks completed  : {len(all_results)} / {total_tasks}")
    print(f"Errors           : {errors}")
    if total_times:
        import numpy as np
        arr = np.array(total_times)
        print(f"Total latency    : mean={arr.mean():.1f}ms  "
              f"p95={float(np.percentile(arr, 95)):.1f}ms  "
              f"max={arr.max():.1f}ms")

    save_json(results, "pt_concurrent_stress.json")
    print("\nDone.\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="PT concurrent stress test")
    parser.add_argument("--students", type=int, default=10, help="Concurrent students")
    parser.add_argument("--frames",   type=int, default=20, help="Frames per student")
    parser.add_argument("--gpu",      action="store_true",  help="Enable GPU inference")
    args = parser.parse_args()
    run(n_students=args.students, frames_per_student=args.frames, enable_gpu=args.gpu)


if __name__ == "__main__":
    main()
