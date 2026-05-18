"""
Worker Manager for the AI Proctoring Backend.

Manages two independent asyncio queues (audio and video) and two pools of
coroutine workers that consume from those queues.  This decouples WebSocket
I/O (fast) from ML inference (slow), preventing head-of-line blocking and
allowing the system to scale to 100+ concurrent sessions.

Architecture
------------
* ``audio_queue`` — receives :class:`AudioTask` items from WebSocket handlers.
* ``video_queue`` — receives :class:`VideoTask` items from WebSocket handlers.
* ``_audio_tasks`` — ``WORKER_POOL_SIZE`` asyncio Tasks, each running
  :meth:`WorkerManager._audio_worker`.
* ``_video_tasks`` — ``WORKER_POOL_SIZE`` asyncio Tasks, each running
  :meth:`WorkerManager._video_worker`.
* ``result_queues`` — per-connection :class:`asyncio.Queue` instances that
  workers write :class:`WorkerResult` objects into; the WebSocket handler
  reads from these to forward results to the client.
* ``last_frame_time`` — per-session monotonic timestamp used to throttle
  video frames to at most 1 fps.

Usage::

    manager = WorkerManager(llm_service=llm_svc)
    await manager.start(pool_size=4)

    # From a WebSocket handler:
    await manager.enqueue_audio(session_id, connection_id, audio_bytes)
    await manager.enqueue_video(session_id, connection_id, frame)

    result_q = manager.get_result_queue(connection_id)
    result: WorkerResult = await result_q.get()

    # On shutdown:
    await manager.stop()
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.schemas.audio_schema import LLMResult
from app.services.headset_service import HeadsetService
from app.services.proctor_service import ProctorService
from app.services.risk_engine import classify, compute_score
from app.services.rule_engine import check_rules
# from app.services.stt_service import STTService
# from app.services.stt_service_enhanced import STTService
from app.services.stt_service import STTService
from app.services.yolo_service import YOLOService
from app.utils.audio_utils import save_temp_audio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio format helpers
# ---------------------------------------------------------------------------

def _detect_audio_suffix(data: bytes) -> str:
    """Detect audio format from magic bytes and return file extension."""
    if data[:4] == b'RIFF':
        return ".wav"
    if data[:4] == b'fLaC':
        return ".flac"
    if data[:3] == b'ID3' or data[:2] == b'\xff\xfb':
        return ".mp3"
    if data[:4] == b'OggS':
        return ".ogg"
    # WebM/Matroska (browser MediaRecorder default)
    if data[:4] == b'\x1a\x45\xdf\xa3':
        return ".webm"
    # MP4/M4A
    if len(data) > 8 and data[4:8] in (b'ftyp', b'moov', b'mdat'):
        return ".mp4"
    # Default to webm (most browsers use this)
    return ".webm"


def _convert_to_wav(input_path: Path) -> Optional[Path]:
    """Convert audio file to WAV using ffmpeg or soundfile fallback.

    Returns the WAV path on success, None on failure.
    """
    import subprocess
    output_path = input_path.with_suffix(".converted.wav")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_path),
                "-ar", "16000", "-ac", "1", "-f", "wav",
                str(output_path)
            ],
            capture_output=True, timeout=30
        )
        if result.returncode == 0 and output_path.exists():
            return output_path
        logger.warning(
            "ffmpeg conversion failed",
            extra={"stderr": result.stderr.decode(errors="replace")[:200]},
        )
        return None
    except FileNotFoundError:
        # ffmpeg not installed — try librosa/soundfile directly
        try:
            import soundfile as sf
            import librosa
            import numpy as np_local
            data, sr = librosa.load(str(input_path), sr=16000, mono=True)
            sf.write(str(output_path), data, 16000, subtype="PCM_16")
            return output_path
        except Exception as e:
            logger.warning(f"Audio conversion fallback failed: {e}")
            return None
    except Exception as e:
        logger.warning(f"Audio conversion error: {e}")
        return None


def _count_unique_persons(person_detections: list) -> int:
    """Count unique persons using IoU-based deduplication.

    YOLO sometimes detects the same person multiple times (body + face).
    This merges overlapping boxes with IoU > 0.3 into a single person.
    """
    if not person_detections:
        return 0

    boxes = []
    for d in person_detections:
        bb = d.bounding_box
        boxes.append([bb.x1, bb.y1, bb.x2, bb.y2, d.confidence])

    boxes.sort(key=lambda b: b[4], reverse=True)
    kept = []
    for box in boxes:
        x1, y1, x2, y2, conf = box
        duplicate = False
        for k in kept:
            # Compute IoU
            ix1 = max(x1, k[0])
            iy1 = max(y1, k[1])
            ix2 = min(x2, k[2])
            iy2 = min(y2, k[3])
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (k[2] - k[0]) * (k[3] - k[1])
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0
            if iou > 0.3:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)

    return len(kept)


# ---------------------------------------------------------------------------
# Task and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AudioTask:
    """A unit of work representing a single audio chunk to be transcribed and analysed.

    Attributes
    ----------
    session_id:
        Unique identifier for the proctoring session this audio belongs to.
    connection_id:
        Unique identifier for the WebSocket connection that submitted the task.
        Used to route the :class:`WorkerResult` back to the correct client.
    audio_bytes:
        Raw audio bytes (WAV, MP3, etc.) to be processed by the STT pipeline.
    enqueued_at:
        Monotonic timestamp (from :func:`time.monotonic`) recorded when the
        task was placed on the queue.  Useful for latency tracking.
    """

    session_id: str
    connection_id: str
    audio_bytes: bytes
    enqueued_at: float  # time.monotonic()


@dataclass
class VideoTask:
    """A unit of work representing a single video frame to be analysed.

    Attributes
    ----------
    session_id:
        Unique identifier for the proctoring session this frame belongs to.
    connection_id:
        Unique identifier for the WebSocket connection that submitted the task.
    frame:
        BGR numpy array (H × W × 3, uint8) decoded from the client's base64
        payload.
    enqueued_at:
        Monotonic timestamp recorded when the task was placed on the queue.
    """

    session_id: str
    connection_id: str
    frame: np.ndarray
    enqueued_at: float  # time.monotonic()


@dataclass
class WorkerResult:
    """The result produced by a worker after processing an :class:`AudioTask`
    or :class:`VideoTask`.

    Attributes
    ----------
    session_id:
        Unique identifier for the proctoring session.
    connection_id:
        Unique identifier for the WebSocket connection; used to route the
        result back to the correct client.
    payload:
        A JSON-serialisable dictionary containing the full analysis result
        (e.g. transcript, risk score, detected events).
    error:
        A human-readable error message if processing failed, or ``None``
        when the task completed successfully.
    """

    session_id: str
    connection_id: str
    payload: dict  # serializable result dict
    error: Optional[str]


# ---------------------------------------------------------------------------
# WorkerManager
# ---------------------------------------------------------------------------


class WorkerManager:
    """Manages asyncio worker pools for audio and video inference tasks.

    The manager owns two :class:`asyncio.Queue` instances and two lists of
    :class:`asyncio.Task` objects.  Workers are spawned by :meth:`start` and
    cancelled by :meth:`stop`.

    Parameters
    ----------
    llm_service:
        An initialised :class:`~app.services.llm_service.LLMService` instance
        used by the audio processing pipeline.  Pass ``None`` to disable LLM
        analysis (the ``llm_result`` field in the payload will be ``None``).
    """

    def __init__(self, llm_service: Optional[object] = None) -> None:
        """Initialise queues, result stores, and worker task lists.

        Parameters
        ----------
        llm_service:
            Optional :class:`~app.services.llm_service.LLMService` instance.
            When provided, the audio pipeline calls ``llm_service.analyze``
            after rule-engine checks.
        """
        self.audio_queue: asyncio.Queue[AudioTask] = asyncio.Queue()
        self.video_queue: asyncio.Queue[VideoTask] = asyncio.Queue()

        # Per-connection result queues keyed by connection_id.
        self.result_queues: Dict[str, asyncio.Queue[WorkerResult]] = {}

        # Per-session last-processed-frame monotonic timestamp (for throttling).
        self.last_frame_time: Dict[str, float] = {}

        # Lists of running asyncio.Task objects for each worker pool.
        self._audio_tasks: List[asyncio.Task] = []
        self._video_tasks: List[asyncio.Task] = []

        # Optional LLM service injected at construction time.
        self._llm_service = llm_service

        logger.info("WorkerManager initialised")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, pool_size: int) -> None:
        """Spawn audio and video worker coroutines as asyncio Tasks.

        Creates ``pool_size`` audio workers and ``pool_size`` video workers.
        Each worker runs an infinite loop consuming from its respective queue
        until cancelled.

        Parameters
        ----------
        pool_size:
            Number of concurrent worker coroutines to spawn for each queue.
            Typically set to ``Config.WORKER_POOL_SIZE``.
        """
        for i in range(pool_size):
            audio_task = asyncio.ensure_future(self._audio_worker())
            audio_task.set_name(f"audio-worker-{i}")
            self._audio_tasks.append(audio_task)

            video_task = asyncio.ensure_future(self._video_worker())
            video_task.set_name(f"video-worker-{i}")
            self._video_tasks.append(video_task)

        logger.info(
            "WorkerManager started",
            extra={
                "pool_size": pool_size,
                "audio_workers": len(self._audio_tasks),
                "video_workers": len(self._video_tasks),
            },
        )

    async def stop(self) -> None:
        """Cancel all worker tasks and await their completion.

        Sends a :class:`asyncio.CancelledError` to every worker coroutine and
        waits for them to finish.  Workers catch ``CancelledError`` and exit
        their loop cleanly.
        """
        all_tasks = self._audio_tasks + self._video_tasks

        for task in all_tasks:
            task.cancel()

        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        self._audio_tasks.clear()
        self._video_tasks.clear()

        logger.info("WorkerManager stopped; all worker tasks cancelled")

    # ------------------------------------------------------------------
    # Worker coroutines
    # ------------------------------------------------------------------

    async def _audio_worker(self) -> None:
        """Coroutine that continuously consumes :class:`AudioTask` items.

        Loop behaviour:

        1. ``await audio_queue.get()`` — blocks until a task is available.
        2. Calls :meth:`_process_audio` to run the full STT → Rule → LLM →
           Risk pipeline.
        3. Places the :class:`WorkerResult` in the per-connection result queue.
        4. On :class:`asyncio.CancelledError`: breaks the loop (clean shutdown).
        5. On any other exception: logs the error and continues (the worker
           does NOT exit on transient failures — Requirement 12.5).
        6. Always calls ``audio_queue.task_done()`` in the ``finally`` block.
        """
        logger.debug("Audio worker started")
        while True:
            task: Optional[AudioTask] = None
            try:
                task = await self.audio_queue.get()
                result = await self._process_audio(task)
                # Ensure the result queue exists before putting.
                if task.connection_id not in self.result_queues:
                    self.result_queues[task.connection_id] = asyncio.Queue()
                await self.result_queues[task.connection_id].put(result)

            except asyncio.CancelledError:
                logger.debug("Audio worker received CancelledError; exiting")
                break

            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Audio worker encountered an unhandled exception; continuing",
                    exc_info=exc,
                )

            finally:
                # Always mark the task as done to unblock queue.join() callers.
                try:
                    self.audio_queue.task_done()
                except ValueError:
                    # task_done() called more times than get() — should not happen.
                    pass

    async def _video_worker(self) -> None:
        """Coroutine that continuously consumes :class:`VideoTask` items.

        Mirrors :meth:`_audio_worker` but processes video frames through the
        YOLO → Headset → Proctor → Risk pipeline.
        """
        logger.debug("Video worker started")
        while True:
            task: Optional[VideoTask] = None
            try:
                task = await self.video_queue.get()
                result = await self._process_video(task)
                # Ensure the result queue exists before putting.
                if task.connection_id not in self.result_queues:
                    self.result_queues[task.connection_id] = asyncio.Queue()
                await self.result_queues[task.connection_id].put(result)

            except asyncio.CancelledError:
                logger.debug("Video worker received CancelledError; exiting")
                break

            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Video worker encountered an unhandled exception; continuing",
                    exc_info=exc,
                )

            finally:
                try:
                    self.video_queue.task_done()
                except ValueError:
                    pass

    # ------------------------------------------------------------------
    # Enqueue helpers
    # ------------------------------------------------------------------

    async def enqueue_audio(
        self,
        session_id: str,
        connection_id: str,
        audio_bytes: bytes,
    ) -> None:
        """Create an :class:`AudioTask` and place it on the audio queue.

        Also registers a result queue for *connection_id* if one does not
        already exist.

        Parameters
        ----------
        session_id:
            Unique identifier for the proctoring session.
        connection_id:
            Unique identifier for the WebSocket connection.
        audio_bytes:
            Raw audio bytes to be transcribed and analysed.
        """
        if connection_id not in self.result_queues:
            self.result_queues[connection_id] = asyncio.Queue()

        task = AudioTask(
            session_id=session_id,
            connection_id=connection_id,
            audio_bytes=audio_bytes,
            enqueued_at=time.monotonic(),
        )
        await self.audio_queue.put(task)

        logger.debug(
            "Audio task enqueued",
            extra={
                "session_id": session_id,
                "connection_id": connection_id,
                "audio_bytes_len": len(audio_bytes),
            },
        )

    async def enqueue_video(
        self,
        session_id: str,
        connection_id: str,
        frame: np.ndarray,
    ) -> None:
        """Throttle-check and enqueue a :class:`VideoTask`.

        Implements 1-fps-per-session throttling (Requirement 13):

        * If the elapsed time since the last processed frame for *session_id*
          is less than 1.0 second, the frame is silently discarded.
        * Otherwise the monotonic timestamp is updated and the frame is
          enqueued.

        Also registers a result queue for *connection_id* if one does not
        already exist.

        Parameters
        ----------
        session_id:
            Unique identifier for the proctoring session.
        connection_id:
            Unique identifier for the WebSocket connection.
        frame:
            BGR numpy array (H × W × 3, uint8) to be analysed.
        """
        now = time.monotonic()
        last = self.last_frame_time.get(session_id, 0.0)

        if now - last < 1.0:
            # Silently discard — do NOT send an error to the client.
            logger.debug(
                "Video frame throttled (< 1s since last frame)",
                extra={
                    "session_id": session_id,
                    "elapsed_since_last": round(now - last, 3),
                },
            )
            return

        # Update the last-processed timestamp before enqueuing.
        self.last_frame_time[session_id] = now

        if connection_id not in self.result_queues:
            self.result_queues[connection_id] = asyncio.Queue()

        task = VideoTask(
            session_id=session_id,
            connection_id=connection_id,
            frame=frame,
            enqueued_at=now,
        )
        await self.video_queue.put(task)

        logger.debug(
            "Video task enqueued",
            extra={
                "session_id": session_id,
                "connection_id": connection_id,
                "frame_shape": frame.shape,
            },
        )

    # ------------------------------------------------------------------
    # Result queue accessor
    # ------------------------------------------------------------------

    def get_result_queue(self, connection_id: str) -> asyncio.Queue:
        """Return the result queue for a given connection.

        If no queue exists yet for *connection_id*, one is created and
        registered before being returned.

        Parameters
        ----------
        connection_id:
            Unique identifier for the WebSocket connection.

        Returns
        -------
        asyncio.Queue
            The :class:`asyncio.Queue` that workers write
            :class:`WorkerResult` objects into for this connection.
        """
        if connection_id not in self.result_queues:
            self.result_queues[connection_id] = asyncio.Queue()
        return self.result_queues[connection_id]

    # ------------------------------------------------------------------
    # Processing pipelines
    # ------------------------------------------------------------------

    async def _process_audio(self, task: AudioTask) -> WorkerResult:
        """Run the full audio analysis pipeline for a single :class:`AudioTask`.

        Pipeline steps
        --------------
        1. Save ``task.audio_bytes`` to a temporary WAV file.
        2. Call :meth:`STTService.transcribe` to obtain the native transcript.
        3. Call :func:`~app.services.rule_engine.check_rules` on the transcript.
        4. Call ``llm_service.analyze`` (if available) to get the LLM result.
        5. Build an events list from rule flags and LLM risk level.
        6. Call :func:`~app.services.risk_engine.compute_score` and
           :func:`~app.services.risk_engine.classify` to get the risk score
           and level.
        7. Return a :class:`WorkerResult` with the full payload dict.

        On any exception the method returns a :class:`WorkerResult` with
        ``error`` set to the exception message and an empty payload.

        Parameters
        ----------
        task:
            The :class:`AudioTask` to process.

        Returns
        -------
        WorkerResult
            Contains the full analysis payload or an error description.
        """
        temp_path: Optional[Path] = None
        converted_path: Optional[Path] = None
        try:
            t_start = time.monotonic()

            # 1. Detect audio format from magic bytes and save with correct extension.
            # Browser MediaRecorder sends webm/opus; soundfile needs WAV/FLAC/OGG.
            audio_bytes = task.audio_bytes
            suffix = _detect_audio_suffix(audio_bytes)
            temp_path = save_temp_audio(audio_bytes, suffix)

            # 2. Convert to WAV if not already WAV (soundfile can't read webm).
            stt_input_path = temp_path
            if suffix != ".wav":
                converted_path = _convert_to_wav(temp_path)
                if converted_path is not None:
                    stt_input_path = converted_path
                else:
                    # Conversion failed — skip STT
                    logger.warning(
                        "Audio conversion failed; skipping STT",
                        extra={"session_id": task.session_id, "suffix": suffix},
                    )
                    return WorkerResult(
                        session_id=task.session_id,
                        connection_id=task.connection_id,
                        payload={
                            "session_id": task.session_id,
                            "native_text": None,
                            "translated_text": None,
                            "rule_flags": [],
                            "llm_result": None,
                            "risk_score": 0,
                            "risk_level": "LOW",
                        },
                        error=None,
                    )

            # 3. Transcribe — run in thread executor to avoid blocking event loop.
            loop = asyncio.get_event_loop()
            native_text: Optional[str] = await loop.run_in_executor(
                None, STTService.transcribe, str(stt_input_path)
            )
            stt_elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)

            # 4. Rule engine.
            rule_flags: List[str] = check_rules(native_text)

            # 5. LLM analysis.
            llm_result: Optional[LLMResult] = None
            llm_elapsed_ms = 0.0
            if native_text and self._llm_service is not None:
                t_llm = time.monotonic()
                llm_result = await self._llm_service.analyze(native_text)
                llm_elapsed_ms = round((time.monotonic() - t_llm) * 1000, 1)

            # 6. Build events list for risk scoring.
            events: List[str] = list(rule_flags)
            if llm_result is not None and llm_result.risk in ("medium", "high"):
                events.append("suspicious_transcript")

            # 7. Risk scoring.
            risk_score: int = compute_score(events)
            risk_level: str = classify(risk_score)

            # 7. Assemble payload.
            translated_text: Optional[str] = (
                llm_result.translated_text if llm_result else None
            )
            payload = {
                "session_id": task.session_id,
                "connection_id": task.connection_id,
                "native_text": native_text,
                "translated_text": translated_text,
                "rule_flags": rule_flags,
                "llm_result": llm_result.model_dump() if llm_result else None,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "stt_elapsed_ms": stt_elapsed_ms,
                "llm_elapsed_ms": llm_elapsed_ms,
            }

            logger.info(
                "Audio task processed",
                extra={
                    "session_id": task.session_id,
                    "connection_id": task.connection_id,
                    "stt_elapsed_ms": stt_elapsed_ms,
                    "llm_elapsed_ms": llm_elapsed_ms,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                },
            )

            return WorkerResult(
                session_id=task.session_id,
                connection_id=task.connection_id,
                payload=payload,
                error=None,
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Audio processing pipeline failed",
                extra={
                    "session_id": task.session_id,
                    "connection_id": task.connection_id,
                    "error": str(exc),
                },
            )
            return WorkerResult(
                session_id=task.session_id,
                connection_id=task.connection_id,
                payload={},
                error=str(exc),
            )

        finally:
            # Always clean up the temporary audio file.
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to delete temp audio", extra={"path": str(temp_path)})
            if converted_path is not None:
                try:
                    converted_path.unlink(missing_ok=True)
                except Exception:
                    pass

    async def _process_video(self, task: VideoTask) -> WorkerResult:
        """Run the full video analysis pipeline for a single :class:`VideoTask`.

        Pipeline steps
        --------------
        1. Call :meth:`YOLOService.detect` to get object detections.
        2. Call :meth:`HeadsetService.classify` to check for headset presence.
        3. Call :meth:`ProctorService.analyze` to get behavioural events.
        4. Aggregate events:

           * Count ``"person"`` detections → ``person_count``.
           * Append ``"no_face"`` if ``person_count == 0``.
           * Append ``"multiple_persons"`` if ``person_count > 1``.
           * Append ``"headset_detected"`` if the headset classifier returns
             ``True``.
           * Extend with proctor behavioural events.

        5. Call :func:`~app.services.risk_engine.compute_score` and
           :func:`~app.services.risk_engine.classify`.
        6. Return a :class:`WorkerResult` with the full payload dict.

        On any exception the method returns a :class:`WorkerResult` with
        ``error`` set to the exception message and an empty payload.

        Parameters
        ----------
        task:
            The :class:`VideoTask` to process.

        Returns
        -------
        WorkerResult
            Contains the full analysis payload or an error description.
        """
        try:
            t_start = time.monotonic()
            loop = asyncio.get_event_loop()

            # 1. YOLO object detection — run in executor (CPU-bound).
            detections = await loop.run_in_executor(None, YOLOService.detect, task.frame)
            yolo_elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)

            # 2. Headset classification — run in executor.
            headset_detected: bool = await loop.run_in_executor(None, HeadsetService.classify, task.frame)

            # 3. Proctor behavioural analysis — run in executor.
            t_proctor = time.monotonic()
            proctor_events: List[str] = await loop.run_in_executor(None, ProctorService.analyze, task.frame)
            proctor_elapsed_ms = round((time.monotonic() - t_proctor) * 1000, 1)

            # 4. Aggregate events.
            # Count unique person detections — use IoU deduplication to avoid
            # counting the same person multiple times (body + face overlap).
            person_detections = [d for d in detections if d.class_name == "person"]
            person_count: int = _count_unique_persons(person_detections)

            # Also check for mobile phone detection
            mobile_detected = any(d.class_name == "cell phone" for d in detections)

            events: List[str] = []

            if person_count == 0:
                events.append("no_face")
            elif person_count > 1:
                events.append("multiple_persons")

            if mobile_detected:
                events.append("mobile_detected")

            if headset_detected:
                events.append("headset_detected")

            # Filter proctor events — don't duplicate no_face/multiple_persons
            filtered_proctor = [
                e for e in proctor_events
                if e not in ("no_face", "multiple_persons")
            ]
            events.extend(filtered_proctor)

            # 5. Risk scoring.
            risk_score: int = compute_score(events)
            risk_level: str = classify(risk_score)

            # 6. Assemble payload.
            # Serialise Detection objects to plain dicts.
            detections_payload = [
                {
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bounding_box": {
                        "x1": d.bounding_box.x1,
                        "y1": d.bounding_box.y1,
                        "x2": d.bounding_box.x2,
                        "y2": d.bounding_box.y2,
                    },
                }
                for d in detections
            ]

            payload = {
                "session_id": task.session_id,
                "connection_id": task.connection_id,
                "events": events,
                "person_count": person_count,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "detections": detections_payload,
                "yolo_elapsed_ms": yolo_elapsed_ms,
                "proctor_elapsed_ms": proctor_elapsed_ms,
            }

            logger.info(
                "Video task processed",
                extra={
                    "session_id": task.session_id,
                    "connection_id": task.connection_id,
                    "yolo_elapsed_ms": yolo_elapsed_ms,
                    "proctor_elapsed_ms": proctor_elapsed_ms,
                    "event_count": len(events),
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                },
            )

            return WorkerResult(
                session_id=task.session_id,
                connection_id=task.connection_id,
                payload=payload,
                error=None,
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Video processing pipeline failed",
                extra={
                    "session_id": task.session_id,
                    "connection_id": task.connection_id,
                    "error": str(exc),
                },
            )
            return WorkerResult(
                session_id=task.session_id,
                connection_id=task.connection_id,
                payload={},
                error=str(exc),
            )
