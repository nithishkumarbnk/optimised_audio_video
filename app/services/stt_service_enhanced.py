"""
Production-Grade Speech-to-Text Service for AI Proctoring Backend.

Wraps the AI4Bharat IndicConformer multilingual ASR model with:
- Async inference (non-blocking event loop)
- GPU support with CUDA memory management
- Per-language timeout (3 seconds)
- Graceful memory cleanup
- Structured logging + metrics

Usage::

    # At application startup:
    await STTService.initialize()

    # Per request (async):
    transcript = await STTService.transcribe_async("/path/to/audio.wav")
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import torch
from transformers import AutoModel

from app.utils.audio_utils import load_and_preprocess, detect_speech

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: HuggingFace model identifier for the IndicConformer ASR model.
_MODEL_ID: str = "ai4bharat/indic-conformer-600m-multilingual"

#: Language codes supported by the IndicConformer model.
LANGUAGES: List[str] = ["te", "hi", "ta", "kn", "ml", "bn", "mr", "gu", "pa", "en"]

#: Target sample rate required by the model (16 kHz).
_TARGET_SR: int = 16_000

#: Timeout per language inference (seconds)
_INFERENCE_TIMEOUT: float = 3.0

#: GPU memory fraction to allocate (80% of available)
_GPU_MEMORY_FRACTION: float = 0.8


# ---------------------------------------------------------------------------
# Service Class
# ---------------------------------------------------------------------------


class STTService:
    """Production-grade ASR service with GPU support, async, and timeouts.

    All public methods are classmethods — use STTService.method() directly.

    Features:
    - Lazy loading: model loaded once on first use
    - GPU acceleration: auto-detect CUDA, move model to GPU
    - Async inference: runs in thread pool to avoid event loop blocking
    - Per-language timeout: 3s max per language
    - Memory cleanup: explicit CUDA cache clearing
    - Graceful degradation: return partial results if language fails
    """

    # Class attributes (shared across all instances)
    _model: Optional[object] = None
    _device: Optional[str] = None
    _executor: Optional[ThreadPoolExecutor] = None
    _inference_semaphore: Optional[asyncio.Semaphore] = None

    # ---------------------------------------------------------------------------
    # Lifecycle: Initialize / Shutdown
    # ---------------------------------------------------------------------------

    @classmethod
    async def initialize(cls, gpu_memory_fraction: float = _GPU_MEMORY_FRACTION) -> None:
        """Load the IndicConformer model and initialize infrastructure.

        This method is idempotent: calling multiple times is safe (no-op after first call).

        Args:
            gpu_memory_fraction: Fraction of GPU memory to allocate (0.0-1.0).
                                Prevents OOM by limiting allocation. Default: 0.8 (80%).

        Raises:
            RuntimeError: If model loading fails (network, disk, CUDA errors).
        """
        if cls._model is not None:
            logger.info("STTService: model already loaded, skipping initialization")
            return

        logger.info(
            "STTService: initializing",
            extra={"model_id": _MODEL_ID, "gpu_fraction": gpu_memory_fraction},
        )

        # Detect and configure GPU
        if torch.cuda.is_available():
            cls._device = "cuda"
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            logger.info(
                "STTService: GPU detected, configuring CUDA",
                extra={
                    "cuda_device": torch.cuda.get_device_name(0),
                    "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    "memory_fraction": gpu_memory_fraction,
                },
            )
        else:
            cls._device = "cpu"
            logger.warning("STTService: CUDA not available, using CPU (will be slower)")

        # Load model
        load_start = time.time()
        try:
            model = AutoModel.from_pretrained(_MODEL_ID, trust_remote_code=True)
            cls._model = model.to(cls._device)
        except Exception as e:
            logger.error(f"STTService: model loading failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load IndicConformer model: {str(e)}") from e

        load_elapsed = time.time() - load_start

        # Create thread pool executor for sync→async wrapping
        cls._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt_inference")

        # Create semaphore: limit to 1 concurrent inference (GPU serialization)
        cls._inference_semaphore = asyncio.Semaphore(1)

        logger.info(
            "STTService: model loaded successfully",
            extra={
                "device": cls._device,
                "load_time_seconds": round(load_elapsed, 2),
                "model_param_count": sum(p.numel() for p in cls._model.parameters()),
            },
        )

    @classmethod
    async def shutdown(cls) -> None:
        """Gracefully shutdown the STT service."""
        logger.info("STTService: shutting down")

        if cls._executor:
            cls._executor.shutdown(wait=True)
            cls._executor = None

        # Clear GPU memory
        if cls._device == "cuda":
            torch.cuda.empty_cache()

        cls._model = None
        cls._inference_semaphore = None
        logger.info("STTService: shutdown complete")

    # ---------------------------------------------------------------------------
    # Public API: Transcription (Async)
    # ---------------------------------------------------------------------------

    @classmethod
    async def transcribe_async(cls, file_path: str) -> Optional[str]:
        """Transcribe an audio file asynchronously.

        Processing pipeline:
        1. Load and preprocess audio (resample, normalize, VAD)
        2. Validate audio (NaN, Inf, amplitude)
        3. Run CTC inference for each supported language
        4. Select best transcript (longest non-empty result)
        5. Cleanup: delete temp file and clear GPU memory

        Args:
            file_path: Path to audio file (.wav, .mp3, .flac, etc.)

        Returns:
            Optional[str]: Transcript string, or None if:
            - Model not initialized
            - Audio fails validation/VAD
            - All language inferences produce empty results

        Raises:
            RuntimeError: If initialize() was not called first.
        """
        if cls._model is None:
            raise RuntimeError("STTService.initialize() must be called before transcribe_async()")

        logger.info(
            "STTService: starting transcription",
            extra={"file_path": file_path},
        )
        request_start = time.time()

        try:
            # Load and preprocess audio
            logger.info("STTService: loading and preprocessing audio")
            load_start = time.time()

            waveform, sample_rate = load_and_preprocess(file_path)
            load_elapsed = time.time() - load_start

            logger.info(
                "STTService: audio preprocessing complete",
                extra={
                    "sample_rate": sample_rate,
                    "waveform_shape": waveform.shape,
                    "load_time_ms": round(load_elapsed * 1000, 1),
                },
            )

            # VAD check
            has_speech, vad_confidence = detect_speech(waveform, sr=sample_rate)
            if not has_speech:
                logger.warning(
                    "STTService: VAD rejected audio (no speech detected)",
                    extra={"vad_confidence": vad_confidence},
                )
                return None

            # Convert to tensor
            waveform_tensor = torch.tensor(waveform, device=cls._device).float().unsqueeze(0)
            logger.debug("STTService: converted waveform to tensor", extra={"device": cls._device})

            # Run multilingual inference
            logger.info(
                "STTService: starting multilingual inference",
                extra={"language_count": len(LANGUAGES)},
            )

            all_results: List[str] = []

            for index, lang in enumerate(LANGUAGES, start=1):
                try:
                    logger.info(
                        "STTService: starting inference",
                        extra={
                            "language": lang,
                            "progress": f"{index}/{len(LANGUAGES)}",
                        },
                    )

                    lang_start = time.time()

                    # Acquire semaphore: max 1 concurrent inference
                    async with cls._inference_semaphore:
                        # Run sync inference in thread pool with timeout
                        loop = asyncio.get_event_loop()
                        text = await asyncio.wait_for(
                            loop.run_in_executor(
                                cls._executor,
                                cls._infer_language_sync,
                                waveform_tensor,
                                lang,
                            ),
                            timeout=_INFERENCE_TIMEOUT,
                        )

                    lang_elapsed = time.time() - lang_start

                    logger.info(
                        "STTService: inference complete",
                        extra={
                            "language": lang,
                            "elapsed_ms": round(lang_elapsed * 1000, 1),
                            "result_length": len(text) if text else 0,
                        },
                    )

                    if text and text.strip():
                        all_results.append(text)
                    else:
                        logger.warning(
                            "STTService: empty transcript",
                            extra={"language": lang},
                        )

                except asyncio.TimeoutError:
                    logger.error(
                        "STTService: inference timeout",
                        extra={
                            "language": lang,
                            "timeout_seconds": _INFERENCE_TIMEOUT,
                        },
                    )
                    # Continue with next language
                    continue

                except Exception as lang_error:
                    logger.error(
                        "STTService: inference failed",
                        extra={
                            "language": lang,
                            "error": str(lang_error),
                        },
                        exc_info=True,
                    )
                    # Continue with next language (graceful degradation)
                    continue

                finally:
                    # Cleanup GPU memory after each inference
                    if cls._device == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

            # Select best transcript
            valid_results = [r.strip() for r in all_results if r and r.strip()]

            if not valid_results:
                logger.warning("STTService: no valid transcripts produced from any language")
                return None

            best_transcript = max(valid_results, key=len)

            total_elapsed = time.time() - request_start

            logger.info(
                "STTService: transcription complete",
                extra={
                    "total_time_ms": round(total_elapsed * 1000, 1),
                    "valid_results": len(valid_results),
                    "best_length": len(best_transcript),
                    "transcript_preview": best_transcript[:100],
                },
            )

            return best_transcript

        except Exception as e:
            total_elapsed = time.time() - request_start
            logger.error(
                "STTService: transcription pipeline failed",
                extra={
                    "error": str(e),
                    "total_time_ms": round(total_elapsed * 1000, 1),
                },
                exc_info=True,
            )
            return None

    # ---------------------------------------------------------------------------
    # Private: Sync Inference Worker (runs in thread pool)
    # ---------------------------------------------------------------------------

    @classmethod
    def _infer_language_sync(
        cls, waveform_tensor: torch.Tensor, language: str
    ) -> str:
        """Synchronous inference worker (runs in thread pool executor).

        This method must be synchronous because torch model inference is blocking.

        Args:
            waveform_tensor: Preprocessed waveform (batch_size=1, already on device)
            language: Language code (e.g., "te", "hi")

        Returns:
            str: Transcript string, or empty string if inference fails.
        """
        try:
            # Ensure tensor is on correct device
            if cls._device == "cuda":
                waveform_tensor = waveform_tensor.cuda()
            else:
                waveform_tensor = waveform_tensor.cpu()

            # Run CTC inference
            with torch.no_grad():
                text = cls._model(waveform_tensor, language, "ctc")

            return text if text else ""

        except Exception as e:
            logger.error(
                "STTService: sync inference error",
                extra={"language": language, "error": str(e)},
            )
            return ""

    # ---------------------------------------------------------------------------
    # Health & Diagnostics
    # ---------------------------------------------------------------------------

    @classmethod
    def is_ready(cls) -> bool:
        """Check if service is ready (model loaded, device available)."""
        if cls._model is None:
            return False

        if cls._device == "cuda":
            return torch.cuda.is_available()

        return True

    @classmethod
    def get_status(cls) -> dict:
        """Get service status and diagnostics."""
        status = {
            "model_loaded": cls._model is not None,
            "device": cls._device,
            "languages_supported": len(LANGUAGES),
            "inference_timeout_seconds": _INFERENCE_TIMEOUT,
        }

        if cls._device == "cuda" and torch.cuda.is_available():
            status.update({
                "cuda_available": True,
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_memory_reserved_mb": torch.cuda.memory_reserved(0) / (1024**2),
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated(0) / (1024**2),
            })
        else:
            status["cuda_available"] = False

        if cls._model:
            status["model_parameters"] = sum(p.numel() for p in cls._model.parameters())

        return status
