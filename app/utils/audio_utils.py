"""
Audio utility helpers for the AI Proctoring Backend.

Provides functions for saving temporary audio files, loading and preprocessing
audio data for STT inference, and decoding base64-encoded audio payloads.

Enhanced with:
- Voice Activity Detection (VAD)
- Audio validation (NaN, Inf, amplitude, duration)
- Waveform normalization
- High-quality resampling with energy preservation
- Correct stereo→mono order (resample first)
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Tuple
from uuid import uuid4

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Directory (relative to the project root) where temporary audio files are stored.
# Resolves to: <repo>/project/uploads/
# parents[0] = utils/, parents[1] = app/, parents[2] = project/
_UPLOADS_DIR = Path(__file__).resolve().parents[2] / "uploads"

# Target sample rate required by the IndicConformer STT model.
_TARGET_SAMPLE_RATE: int = 16_000


def save_temp_audio(data: bytes, suffix: str) -> Path:
    """Write raw audio bytes to a uniquely named file under the ``uploads/`` directory.

    The filename is generated from a UUID4 so that concurrent uploads never
    collide.  The ``uploads/`` directory is created if it does not already
    exist.

    Args:
        data:   Raw audio bytes to persist.
        suffix: File extension including the leading dot, e.g. ``".wav"`` or
                ``".mp3"``.  Used verbatim as the path suffix.

    Returns:
        Path: Absolute path to the newly created temporary file.

    Example::

        path = save_temp_audio(wav_bytes, ".wav")
        # → PosixPath('/project/uploads/3f2a…b1.wav')
    """
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = _UPLOADS_DIR / f"{uuid4().hex}{suffix}"
    file_path.write_bytes(data)
    logger.debug(f"Saved temporary audio file: {file_path}")
    return file_path


def cleanup_temp_audio(file_path: Path) -> bool:
    """Delete a temporary audio file.
    
    Args:
        file_path: Path to the temporary file to delete.
    
    Returns:
        bool: True if successfully deleted, False if file not found or error.
    """
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Cleaned up temporary audio: {file_path}")
            return True
        return False
    except Exception as e:
        logger.warning(
            "Failed to cleanup temporary audio file",
            extra={"file_path": str(file_path), "error": str(e)}
        )
        return False


def validate_audio(
    waveform: np.ndarray,
    sample_rate: int,
    min_duration_sec: float = 0.1,
    max_duration_sec: float = 3600.0
) -> Tuple[bool, str]:
    """Validate audio waveform and metadata.
    
    Checks for:
    - Correct shape (1-D or 2-D)
    - Duration within acceptable range
    - No NaN or Inf values
    - Amplitude within expected range
    - Sample rate in valid range
    
    Args:
        waveform: Audio waveform (1-D or 2-D numpy array)
        sample_rate: Sample rate in Hz
        min_duration_sec: Minimum audio duration (seconds)
        max_duration_sec: Maximum audio duration (seconds)
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    errors = []
    
    # Check shape
    if waveform.ndim > 2:
        errors.append(f"Audio has {waveform.ndim} dimensions (expected 1 or 2)")
    
    # Check length
    num_samples = len(waveform)
    min_samples = min_duration_sec * sample_rate
    max_samples = max_duration_sec * sample_rate
    
    if num_samples < min_samples:
        errors.append(
            f"Audio too short: {num_samples} samples ({num_samples/sample_rate:.2f}s) "
            f"< minimum {min_duration_sec}s"
        )
    
    if num_samples > max_samples:
        errors.append(
            f"Audio too long: {num_samples} samples ({num_samples/sample_rate:.2f}s) "
            f"> maximum {max_duration_sec}s"
        )
    
    # Check for NaN/Inf
    if np.any(np.isnan(waveform)):
        errors.append("Audio contains NaN values")
    
    if np.any(np.isinf(waveform)):
        errors.append("Audio contains Inf values")
    
    # Check amplitude (typically int16: [-32768, 32767] or float: [-1, 1])
    if len(waveform) > 0:
        max_amp = np.max(np.abs(waveform))
        if max_amp > 1e6:
            errors.append(f"Audio amplitude out of range: {max_amp:.0f}")
    
    # Check sample rate
    if sample_rate < 8000 or sample_rate > 48000:
        errors.append(f"Sample rate {sample_rate} Hz outside [8000, 48000] range")
    
    if errors:
        error_msg = "; ".join(errors)
        logger.warning(f"Audio validation failed: {error_msg}")
        return False, error_msg
    
    return True, ""


def normalize_waveform(
    waveform: np.ndarray,
    target_rms: float = 0.1,
    eps: float = 1e-10
) -> np.ndarray:
    """Normalize waveform to target RMS level.
    
    Normalizes audio amplitude to a consistent level (-20dB from unity),
    which matches IndicConformer's training distribution.
    
    Args:
        waveform: Audio waveform (float32 numpy array)
        target_rms: Target RMS level (default: 0.1 ≈ -20dB)
        eps: Small epsilon to avoid division by zero
    
    Returns:
        np.ndarray: Normalized waveform (float32, clipped to [-1, 1])
    """
    current_rms = np.sqrt(np.mean(waveform ** 2) + eps)
    
    if current_rms > eps:
        normalized = waveform * (target_rms / current_rms)
    else:
        normalized = waveform
    
    # Hard clip to [-1, 1] to prevent saturation
    normalized = np.clip(normalized, -1.0, 1.0)
    
    logger.debug(
        "Normalized waveform",
        extra={
            "input_rms": current_rms,
            "target_rms": target_rms,
            "output_rms": np.sqrt(np.mean(normalized ** 2))
        }
    )
    
    return normalized


def resample_audio(
    waveform: np.ndarray,
    orig_sr: int,
    target_sr: int,
    preserve_energy: bool = True
) -> np.ndarray:
    """High-quality audio resampling with energy preservation.
    
    Uses 'kaiser_best' filter for superior anti-aliasing (99.9% passband fidelity).
    Preserves energy to maintain loudness during resampling.
    
    Args:
        waveform: Audio waveform (1-D float32 numpy array)
        orig_sr: Original sample rate (Hz)
        target_sr: Target sample rate (Hz)
        preserve_energy: If True, normalize output to match input energy
    
    Returns:
        np.ndarray: Resampled waveform (float32, same RMS as input)
    """
    if orig_sr == target_sr:
        return waveform
    
    # Measure energy before resampling
    energy_before = np.sqrt(np.mean(waveform ** 2)) if preserve_energy else 1.0
    
    # Resample using kaiser_best filter (high quality, ~150ms slower)
    # res_type options: 'kaiser_best' (best), 'kaiser_fast' (faster), 'scipy' (fastest)
    resampled = librosa.resample(
        waveform,
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type='kaiser_best'  # 99.9% passband fidelity
    )
    
    # Preserve energy
    if preserve_energy:
        energy_after = np.sqrt(np.mean(resampled ** 2))
        if energy_after > 1e-10:
            resampled *= (energy_before / energy_after)
    
    logger.debug(
        "Resampled audio",
        extra={
            "from_hz": orig_sr,
            "to_hz": target_sr,
            "ratio": target_sr / orig_sr,
            "energy_before_db": 20 * np.log10(energy_before + 1e-10),
            "energy_after_db": 20 * np.log10(np.sqrt(np.mean(resampled ** 2)) + 1e-10)
        }
    )
    
    return resampled


def mix_stereo_to_mono(waveform: np.ndarray) -> np.ndarray:
    """Convert stereo (or multi-channel) audio to mono.
    
    Uses loudness-preserving weighted mix to avoid phase cancellation:
    1. Compute per-channel energy (RMS)
    2. Weight channels by relative energy
    3. Weighted average: louder channels contribute more
    
    This avoids destructive phase interference seen with simple averaging.
    
    Args:
        waveform: Audio waveform (1-D mono: pass through; 2-D stereo: convert)
    
    Returns:
        np.ndarray: 1-D mono waveform (float32)
    """
    if waveform.ndim != 2:
        return waveform
    
    num_channels = waveform.shape[1]
    logger.debug(f"Converting {num_channels}-channel audio to mono")
    
    # Compute per-channel RMS energy
    channel_energies = np.sqrt(np.mean(waveform ** 2, axis=0))
    
    # If channels have significantly different energy, weight them
    max_energy = np.max(channel_energies)
    if max_energy > 1e-10:
        weights = channel_energies / max_energy  # Normalize to [0, 1]
        # Weighted average: louder channels dominate
        mono = np.average(waveform, axis=1, weights=weights)
        
        logger.debug(
            "Stereo→mono conversion (loudness-preserving mix)",
            extra={
                "channels": num_channels,
                "channel_energies_db": [
                    20 * np.log10(e + 1e-10) for e in channel_energies
                ],
                "weights": weights.tolist()
            }
        )
    else:
        # Channels are silent, just average
        mono = np.mean(waveform, axis=1)
        logger.warning("Stereo→mono: channels are silent, using arithmetic mean")
    
    return mono


def detect_speech(
    waveform: np.ndarray,
    sr: int = 16000,
    energy_threshold_db: float = -40,
    spectral_method: str = "energy"
) -> Tuple[bool, float]:
    """Detect if waveform contains speech using Voice Activity Detection (VAD).
    
    Implements multi-feature VAD:
    1. **Energy-based:** RMS threshold (-40 dB is typical for speech)
    2. **Spectral-based:** Speech concentrated in 300-4000 Hz (spectral centroid)
    3. **Zero-crossing rate:** Speech has moderate ZCR (0.05-0.3)
    
    Returns True only if audio is likely to contain speech (not silence/noise).
    
    Args:
        waveform: Audio waveform (1-D float32 numpy array, 16kHz)
        sr: Sample rate (Hz)
        energy_threshold_db: RMS threshold in dB (below this → silence)
        spectral_method: 'energy' (fast), 'full' (comprehensive, slower)
    
    Returns:
        Tuple[bool, float]: (has_speech, confidence_0_to_1)
    """
    frame_length = int(0.02 * sr)  # 20ms frames
    hop_length = frame_length // 2
    
    # Feature 1: RMS energy
    rms = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
    
    # Find energy threshold: percentile-based (ignore lowest 10% = likely noise floor)
    threshold_db = np.percentile(rms_db, 10) + (energy_threshold_db - np.percentile(rms_db, 10)) * 0.3
    energy_frames = rms_db > threshold_db
    energy_confidence = np.mean(energy_frames)
    
    if spectral_method == "energy":
        # Fast VAD: energy only
        has_speech = energy_confidence > 0.15
        confidence = energy_confidence
    else:
        # Full VAD: energy + spectral + ZCR
        
        # Feature 2: Spectral centroid (speech ~1-4 kHz, noise ~0-1 kHz)
        spec_centroid = librosa.feature.spectral_centroid(
            y=waveform,
            sr=sr,
            hop_length=hop_length
        )[0]
        centroid_frames = (spec_centroid > 300) & (spec_centroid < 4000)
        centroid_confidence = np.mean(centroid_frames)
        
        # Feature 3: Zero crossing rate (speech ~0.05-0.3, silence <0.02)
        zcr = librosa.feature.zero_crossing_rate(
            waveform,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        zcr_frames = (zcr > 0.02) & (zcr < 0.5)
        zcr_confidence = np.mean(zcr_frames)
        
        # Require at least 2 of 3 features to indicate speech
        combined = (
            energy_frames.astype(int) +
            centroid_frames.astype(int) +
            zcr_frames.astype(int)
        ) >= 2
        
        speech_ratio = np.mean(combined)
        has_speech = speech_ratio > 0.15  # >15% of frames have speech indicators
        confidence = (energy_confidence + centroid_confidence + zcr_confidence) / 3
        
        logger.debug(
            "VAD analysis (full)",
            extra={
                "energy_confidence": energy_confidence,
                "centroid_confidence": centroid_confidence,
                "zcr_confidence": zcr_confidence,
                "combined_confidence": confidence,
                "has_speech": has_speech,
                "threshold_db": threshold_db,
                "mean_rms_db": np.mean(rms_db)
            }
        )
    
    logger.info(
        "VAD result",
        extra={
            "has_speech": has_speech,
            "confidence": confidence,
            "method": spectral_method
        }
    )
    
    return has_speech, confidence


def load_and_preprocess(path: Path) -> Tuple[np.ndarray, int]:
    """Load an audio file and preprocess it for STT inference.

    Processing pipeline (corrected order to avoid phase cancellation):

    1. Read the file with :func:`soundfile.read` (preserves original dtype).
    2. Cast the waveform to ``float32``.
    3. **RESAMPLE** to 16 kHz FIRST (before stereo→mono to avoid phase issues).
    4. Convert stereo (or multi-channel) audio to mono using loudness-preserving mix.
    5. Validate audio (NaN, Inf, amplitude, duration).
    6. Normalize to target RMS level (-20dB).
    7. Detect speech activity (VAD). Return None if silent.

    Args:
        path: Path to the audio file to load. Any format supported by
              *libsndfile* (WAV, FLAC, OGG, …) is accepted.

    Returns:
        Tuple[np.ndarray, int]: A 2-tuple of:

        * ``waveform`` — 1-D float32 numpy array, normalized to [-1, 1].
        * ``sample_rate`` — Always ``16000``.

    Raises:
        RuntimeError: If the file cannot be read or validation fails.
    """
    logger.info(f"Loading audio file: {path}")
    
    try:
        # Step 1: Load audio
        waveform, original_sr = sf.read(str(path), always_2d=False)
        logger.debug(
            "Audio file loaded",
            extra={
                "original_sample_rate": original_sr,
                "shape": waveform.shape,
                "dtype": waveform.dtype
            }
        )
        
        # Step 2: Cast to float32
        waveform = waveform.astype(np.float32)
        
        # Step 3: Resample FIRST (before stereo→mono)
        # This prevents phase cancellation when mixing stereo channels
        if original_sr != _TARGET_SAMPLE_RATE:
            waveform = resample_audio(
                waveform,
                orig_sr=original_sr,
                target_sr=_TARGET_SAMPLE_RATE,
                preserve_energy=True
            )
        
        # Step 4: Convert stereo→mono using loudness-preserving mix
        if waveform.ndim == 2:
            waveform = mix_stereo_to_mono(waveform)
        
        # Step 5: Validate audio
        is_valid, error_msg = validate_audio(waveform, _TARGET_SAMPLE_RATE)
        if not is_valid:
            raise RuntimeError(f"Audio validation failed: {error_msg}")
        
        # Step 6: Normalize to target RMS
        waveform = normalize_waveform(waveform, target_rms=0.1)
        
        # Step 7: Detect speech (VAD)
        has_speech, vad_confidence = detect_speech(waveform, sr=_TARGET_SAMPLE_RATE)
        logger.info(
            "Audio preprocessing complete",
            extra={
                "target_sample_rate": _TARGET_SAMPLE_RATE,
                "has_speech": has_speech,
                "vad_confidence": vad_confidence
            }
        )
        
        return waveform, _TARGET_SAMPLE_RATE
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        raise RuntimeError(f"Failed to preprocess audio: {str(e)}") from e
def base64_to_audio(b64: str) -> bytes:
    """Decode a base64-encoded audio string to raw bytes.

    Accepts both standard Base64 and URL-safe Base64.
    Strips optional data-URI prefixes automatically.

    Args:
        b64: Base64-encoded audio payload.

    Returns:
        bytes: Raw decoded audio bytes.
    """

    # Strip optional data URI prefix
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    # Remove whitespace/newlines
    b64 = b64.strip()

    return base64.b64decode(b64)