"""
Speech-to-Text service for the AI Proctoring Backend.

Wraps the AI4Bharat IndicConformer multilingual ASR model loaded from a
**local directory** (``app/models/indic-conformer-600m``) as a lazy-loaded
singleton.  No HuggingFace token or network access is required — the model
files are read directly from disk.

The model is initialised exactly once via :meth:`STTService.initialize`;
all subsequent calls to :meth:`STTService.transcribe` reuse the already-loaded
instance.

Supported languages
-------------------
Telugu (te), Hindi (hi), Tamil (ta), Kannada (kn), Malayalam (ml),
Bengali (bn), Marathi (mr), Gujarati (gu), Punjabi (pa), English (en).

Usage::

    # At application startup (once):
    STTService.initialize()

    # Per request:
    transcript = STTService.transcribe("/path/to/audio.wav")
    if transcript:
        print(transcript)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Absolute path to the locally downloaded IndicConformer model directory.
#: AutoModel.from_pretrained() accepts a local path just like a HuggingFace
#: model ID — no token or network access needed.
_MODEL_PATH: str = str(
    Path(__file__).parent.parent / "models" / "indic-conformer-600m"
)

#: Language codes to attempt during transcription.
#: Reduced to the two most common languages for speed.
#: Each language takes ~7–10s on CPU, so fewer = faster transcription.
LANGUAGES: List[str] = ["hi", "te"]

#: Minimum transcript length to accept as a valid result and stop early.
_MIN_TRANSCRIPT_LEN: int = 3

#: Target sample rate required by the model (16 kHz).
_TARGET_SR: int = 16_000


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------


class STTService:
    """Singleton wrapper around the AI4Bharat IndicConformer ASR model.

    The model is loaded lazily via :meth:`initialize` and stored as a
    class-level attribute so that it is shared across all call sites
    without re-loading.

    All public methods are *classmethods* — there is no need to
    instantiate this class.
    """

    #: Loaded model instance; ``None`` until :meth:`initialize` is called.
    _model: Optional[object] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def initialize(cls) -> None:
        """Load the IndicConformer model from the local model directory.

        This method is idempotent: calling it multiple times is safe and
        will not reload the model after the first successful call.

        The model is loaded with ``trust_remote_code=True`` as required
        by the AI4Bharat custom modelling code (``model_onnx.py`` inside
        the local model directory).  No HuggingFace token or network
        access is required.

        Logs the elapsed wall-clock time for the model load operation.

        Raises
        ------
        Exception
            Any exception raised by :func:`transformers.AutoModel.from_pretrained`
            is propagated to the caller (e.g. missing files, disk errors).
        """
        if cls._model is not None:
            logger.info("STTService: model already loaded, skipping initialization")
            return

        logger.info(
            "STTService: loading IndicConformer model",
            extra={"model_path": _MODEL_PATH},
        )
        load_start = time.time()

        cls._model = AutoModel.from_pretrained(
            _MODEL_PATH,
            trust_remote_code=True,
        )

        elapsed = round(time.time() - load_start, 2)
        logger.info(
            "STTService: model loaded successfully",
            extra={"model_path": _MODEL_PATH, "elapsed_seconds": elapsed},
        )

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    @classmethod
    def transcribe(cls, file_path: str) -> Optional[str]:
        """Transcribe an audio file using multilingual CTC inference.

        Processing pipeline
        -------------------
        1. Load the audio file with :func:`soundfile.read`.
        2. Cast the waveform to ``float32``.
        3. Convert stereo (or multi-channel) audio to mono by averaging
           all channels.
        4. Resample to 16 000 Hz with :func:`librosa.resample` if the
           original sample rate differs.
        5. Convert the waveform to a :class:`torch.Tensor` and add a
           batch dimension.
        6. Run CTC inference for each language in :data:`LANGUAGES`.
           Per-language exceptions are caught, logged, and skipped so
           that a single failing language does not abort the pipeline.
        7. Filter out empty/whitespace-only results.
        8. Return the transcript with the greatest character length, or
           ``None`` if all inferences produced empty results.

        Parameters
        ----------
        file_path:
            Absolute or relative path to the audio file.  Any format
            supported by *libsndfile* (WAV, FLAC, OGG, …) is accepted.

        Returns
        -------
        Optional[str]
            The best transcript string (longest non-empty result across
            all languages), or ``None`` if no transcript could be
            produced.

        Raises
        ------
        RuntimeError
            If :meth:`initialize` has not been called before
            :meth:`transcribe`.
        """
        if cls._model is None:
            raise RuntimeError(
                "STTService.initialize() must be called before transcribe()."
            )

        logger.info(
            "STTService: starting transcription",
            extra={"file_path": file_path},
        )
        overall_start = time.time()

        try:
            # ----------------------------------------------------------
            # 1. Load audio
            # ----------------------------------------------------------
            logger.info("STTService: loading audio file")
            waveform, sample_rate = sf.read(file_path)
            logger.info(
                "STTService: audio loaded",
                extra={"original_sample_rate": sample_rate},
            )

            # ----------------------------------------------------------
            # 2. Cast to float32
            # ----------------------------------------------------------
            waveform = np.array(waveform).astype(np.float32)
            logger.info("STTService: waveform cast to float32")

            # ----------------------------------------------------------
            # 3. Stereo → mono
            # ----------------------------------------------------------
            if waveform.ndim == 2:
                logger.info("STTService: converting stereo to mono")
                waveform = waveform.mean(axis=1)
                logger.info("STTService: stereo-to-mono conversion complete")

            # ----------------------------------------------------------
            # 4. Resample to 16 kHz
            # ----------------------------------------------------------
            if sample_rate != _TARGET_SR:
                logger.info(
                    "STTService: resampling audio",
                    extra={
                        "from_hz": sample_rate,
                        "to_hz": _TARGET_SR,
                    },
                )
                waveform = librosa.resample(
                    waveform,
                    orig_sr=sample_rate,
                    target_sr=_TARGET_SR,
                )
                sample_rate = _TARGET_SR
                logger.info("STTService: resampling complete")

            # ----------------------------------------------------------
            # 5. Convert to torch tensor and add batch dimension
            # ----------------------------------------------------------
            wav = torch.tensor(waveform).float().unsqueeze(0)
            logger.info("STTService: waveform converted to torch tensor with batch dim")

            # ----------------------------------------------------------
            # 6. Multilingual CTC inference
            # ----------------------------------------------------------
            logger.info(
                "STTService: running multilingual inference",
                extra={"language_count": len(LANGUAGES)},
            )
            all_results: List[str] = []

            for index, lang in enumerate(LANGUAGES, start=1):
                logger.info(
                    "STTService: starting inference",
                    extra={
                        "language": lang,
                        "index": index,
                        "total": len(LANGUAGES),
                    },
                )
                lang_start = time.time()

                try:
                    text = cls._model(wav, lang, "ctc")
                    lang_elapsed = round(time.time() - lang_start, 2)

                    logger.info(
                        "STTService: inference complete",
                        extra={
                            "language": lang,
                            "elapsed_seconds": lang_elapsed,
                        },
                    )

                    if text:
                        logger.info(
                            "STTService: transcript produced",
                            extra={"language": lang, "transcript": text},
                        )
                        all_results.append(text)
                        # Early exit: if English gives a good result, stop immediately
                        if lang == "en" and len(text.strip()) >= _MIN_TRANSCRIPT_LEN:
                            logger.info(
                                "STTService: early exit after English transcript"
                            )
                            break
                    else:
                        logger.warning(
                            "STTService: no transcript for language",
                            extra={"language": lang},
                        )

                except Exception as lang_error:  # noqa: BLE001
                    lang_elapsed = round(time.time() - lang_start, 2)
                    logger.error(
                        "STTService: inference failed for language",
                        extra={
                            "language": lang,
                            "elapsed_seconds": lang_elapsed,
                            "error": str(lang_error),
                        },
                    )
                    continue

            # ----------------------------------------------------------
            # 7. Filter empty results and select best transcript
            # ----------------------------------------------------------
            valid_results = [r.strip() for r in all_results if r and r.strip()]

            total_elapsed = round(time.time() - overall_start, 2)

            if not valid_results:
                logger.warning(
                    "STTService: no valid transcripts produced",
                    extra={"total_elapsed_seconds": total_elapsed},
                )
                return None

            # 8. Return the longest transcript
            best = max(valid_results, key=len)

            logger.info(
                "STTService: transcription complete",
                extra={
                    "total_elapsed_seconds": total_elapsed,
                    "valid_transcript_count": len(valid_results),
                    "best_transcript": best,
                },
            )
            return best

        except Exception:
            total_elapsed = round(time.time() - overall_start, 2)
            logger.exception(
                "STTService: transcription pipeline failed",
                extra={"total_elapsed_seconds": total_elapsed},
            )
            return None
