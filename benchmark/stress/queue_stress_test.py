"""
benchmark/stress/queue_stress_test.py
Asyncio queue stress tester — measures throughput, wait times,
worker saturation, backlog growth, and frame dropping behavior.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

logger = logging.getLogger(__name__)


@dataclass
class QueueMetrics:
    enqueued: int = 0
    processed: int = 0
    dropped: int = 0
    enqueue_times_ms: List[float] = field(default_factory=list)
    wait_times_ms: List[float] = field(default_factory=list)
    process_times_ms: List[float] = field(default_factory=list)
    queue_depth_samples: List[int] = field(default_factory=list)
    worker_busy_pct: List[float] = field(default_factory=list)

    def summary(self) -> Dict:
        def stats(lst):
            if not lst:
                return {"mean": 0, "p95": 0, "max": 0}
            s = sorted(lst)
            n = len(s)
            return {
                "mean": round(statistics.mean(s), 3),
                "p95": round(s[int(n * 0.95)], 3),
                "max": round(s[-1], 3),
            }

        total = self.enqueued
        return {
            "enqueued": self.enqueued,
            "processed": self.processed,
            "dropped": self.dropped,
            "drop_rate_pct": round(self.dropped / max(total, 1) * 100, 2),
            "throughput_per_sec": round(
                self.processed / max(sum(self.process_times_ms) / 1000, 0.001), 1
            ),
            "enqueue_time_ms": stats(self.enqueue_times_ms),
            "wait_time_ms": stats(self.wait_times_ms),
            "process_time_ms": stats(self.process_times_ms),
            "avg_queue_depth": round(statistics.mean(self.queue_depth_samples), 1) if self.queue_depth_samples else 0,
            "max_queue_depth": max(self.queue_depth_samples) if self.queue_depth_samples else 0,
            "avg_worker_busy_pct": round(statistics.mean(self.worker_busy_pct), 1) if self.worker_busy_pct else 0,
        }


async def run_queue_throughput_test(
    num_workers: int = 4,
    num_tasks: int = 500,
    task_duration_ms: float = 10.0,
    queue_maxsize: int = 0,
) -> QueueMetrics:
    """Test raw queue throughput with simulated work."""
    metrics = QueueMetrics()
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
    worker_busy = [False] * num_workers

    async def worker(worker_id: int):
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=2.0)
                enqueued_at, task_id = item
                wait_ms = (time.monotonic() - enqueued_at) * 1000
                metrics.wait_times_ms.append(wait_ms)

                worker_busy[worker_id] = True
                t0 = time.monotonic()
                await asyncio.sleep(task_duration_ms / 1000)
                proc_ms = (time.monotonic() - t0) * 1000
                metrics.process_times_ms.append(proc_ms)
                metrics.processed += 1
                worker_busy[worker_id] = False
                queue.task_done()
            except asyncio.TimeoutError:
                break
            except asyncio.CancelledError:
                break

    async def monitor():
        while True:
            depth = queue.qsize()
            metrics.queue_depth_samples.append(depth)
            busy_count = sum(1 for b in worker_busy if b)
            metrics.worker_busy_pct.append(busy_count / num_workers * 100)
            await asyncio.sleep(0.1)

    workers = [asyncio.create_task(worker(i)) for i in range(num_workers)]
    monitor_task = asyncio.create_task(monitor())

    # Enqueue tasks
    for i in range(num_tasks):
        t0 = time.monotonic()
        try:
            if queue_maxsize > 0 and queue.qsize() >= queue_maxsize:
                metrics.dropped += 1
            else:
                await queue.put((time.monotonic(), i))
                metrics.enqueued += 1
                metrics.enqueue_times_ms.append((time.monotonic() - t0) * 1000)
        except asyncio.QueueFull:
            metrics.dropped += 1

    # Wait for all tasks to complete
    await queue.join()

    monitor_task.cancel()
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, monitor_task, return_exceptions=True)

    return metrics


async def run_backpressure_test(
    num_workers: int = 2,
    producer_rate_per_sec: float = 50.0,
    task_duration_ms: float = 100.0,
    duration_s: float = 10.0,
) -> Dict:
    """Test backpressure behavior when producers outpace workers."""
    queue: asyncio.Queue = asyncio.Queue()
    metrics = QueueMetrics()
    stop_event = asyncio.Event()

    async def producer():
        interval = 1.0 / producer_rate_per_sec
        while not stop_event.is_set():
            await queue.put((time.monotonic(), metrics.enqueued))
            metrics.enqueued += 1
            await asyncio.sleep(interval)

    async def worker():
        while not stop_event.is_set():
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
                enqueued_at, _ = item
                wait_ms = (time.monotonic() - enqueued_at) * 1000
                metrics.wait_times_ms.append(wait_ms)
                await asyncio.sleep(task_duration_ms / 1000)
                metrics.processed += 1
                queue.task_done()
            except asyncio.TimeoutError:
                continue

    async def depth_monitor():
        while not stop_event.is_set():
            metrics.queue_depth_samples.append(queue.qsize())
            await asyncio.sleep(0.2)

    prod_task = asyncio.create_task(producer())
    worker_tasks = [asyncio.create_task(worker()) for _ in range(num_workers)]
    monitor_task = asyncio.create_task(depth_monitor())

    await asyncio.sleep(duration_s)
    stop_event.set()

    prod_task.cancel()
    for w in worker_tasks:
        w.cancel()
    monitor_task.cancel()
    await asyncio.gather(prod_task, *worker_tasks, monitor_task, return_exceptions=True)

    metrics.dropped = metrics.enqueued - metrics.processed

    return {
        "test": "backpressure",
        "num_workers": num_workers,
        "producer_rate_per_sec": producer_rate_per_sec,
        "task_duration_ms": task_duration_ms,
        "duration_s": duration_s,
        "metrics": metrics.summary(),
        "backlog_at_end": queue.qsize(),
        "worker_saturation": metrics.processed < metrics.enqueued,
    }


async def run_frame_drop_test(
    num_workers: int = 4,
    burst_size: int = 200,
    task_duration_ms: float = 50.0,
    throttle_window_s: float = 1.0,
) -> Dict:
    """Test frame dropping behavior under burst load."""
    queue: asyncio.Queue = asyncio.Queue()
    metrics = QueueMetrics()
    last_processed: Dict[str, float] = {}

    async def throttled_enqueue(session_id: str, frame_id: int) -> bool:
        now = time.monotonic()
        last = last_processed.get(session_id, 0.0)
        if now - last < throttle_window_s:
            metrics.dropped += 1
            return False
        last_processed[session_id] = now
        await queue.put((time.monotonic(), session_id, frame_id))
        metrics.enqueued += 1
        return True

    async def worker():
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=2.0)
                enqueued_at = item[0]
                wait_ms = (time.monotonic() - enqueued_at) * 1000
                metrics.wait_times_ms.append(wait_ms)
                await asyncio.sleep(task_duration_ms / 1000)
                metrics.processed += 1
                queue.task_done()
            except asyncio.TimeoutError:
                break
            except asyncio.CancelledError:
                break

    workers = [asyncio.create_task(worker()) for _ in range(num_workers)]

    # Burst: send many frames rapidly for 5 sessions
    for i in range(burst_size):
        session_id = f"session-{i % 5}"
        await throttled_enqueue(session_id, i)

    await queue.join()
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    return {
        "test": "frame_drop",
        "burst_size": burst_size,
        "num_workers": num_workers,
        "enqueued": metrics.enqueued,
        "processed": metrics.processed,
        "dropped_by_throttle": metrics.dropped,
        "drop_rate_pct": round(metrics.dropped / burst_size * 100, 1),
        "avg_wait_ms": round(statistics.mean(metrics.wait_times_ms), 2) if metrics.wait_times_ms else 0,
    }


def run_all_queue_tests() -> Dict:
    """Run all queue stress tests."""
    results = {}

    logger.info("=== Queue Throughput Test ===")
    metrics = asyncio.run(run_queue_throughput_test(
        num_workers=4, num_tasks=500, task_duration_ms=5.0
    ))
    results["throughput"] = metrics.summary()
    logger.info(f"  Throughput: {results['throughput']['throughput_per_sec']}/s, "
                f"avg_wait={results['throughput']['wait_time_ms']['mean']}ms")

    logger.info("=== Backpressure Test ===")
    results["backpressure"] = asyncio.run(run_backpressure_test(
        num_workers=2, producer_rate_per_sec=20.0, task_duration_ms=80.0, duration_s=8.0
    ))
    logger.info(f"  Backlog at end: {results['backpressure']['backlog_at_end']}, "
                f"saturation={results['backpressure']['worker_saturation']}")

    logger.info("=== Frame Drop Test ===")
    results["frame_drop"] = asyncio.run(run_frame_drop_test(
        num_workers=4, burst_size=100, task_duration_ms=20.0
    ))
    logger.info(f"  Drop rate: {results['frame_drop']['drop_rate_pct']}%")

    return results


if __name__ == "__main__":
    import json
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    output_dir = Path(__file__).parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_all_queue_tests()
    (output_dir / "queue_stress.json").write_text(json.dumps(results, indent=2))
    print("Results saved to benchmark/outputs/queue_stress.json")
