"""
Unit tests for audio utilities.

Tests VAD, validation, resampling, normalization, and stereo→mono conversion.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.utils.audio_utils import (
    validate_audio,
    normalize_waveform,
    resample_audio,
    mix_stereo_to_mono,
    detect_speech,
)


# ---------------------------------------------------------------------------
# Test Audio Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_wave_16khz():
    """Generate 1 second of 440 Hz sine wave at 16 kHz."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return waveform, sr


@pytest.fixture
def silence_16khz():
    """Generate 1 second of silence at 16 kHz."""
    sr = 16000
    duration = 1.0
    waveform = np.zeros(int(sr * duration), dtype=np.float32)
    return waveform, sr


@pytest.fixture
def stereo_audio():
    """Generate 1 second of stereo audio."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    left = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = 0.3 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    return stereo, sr


@pytest.fixture
def wav_file_16khz(sine_wave_16khz):
    """Create a temporary WAV file."""
    waveform, sr = sine_wave_16khz
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, waveform, sr)
        return Path(f.name)


# ---------------------------------------------------------------------------
# Tests: Validation
# ---------------------------------------------------------------------------

class TestValidateAudio:
    def test_valid_audio(self, sine_wave_16khz):
        """Valid audio should pass validation."""
        waveform, sr = sine_wave_16khz
        is_valid, error = validate_audio(waveform, sr)
        assert is_valid
        assert error == ""

    def test_audio_with_nan(self):
        """Audio with NaN should fail validation."""
        waveform = np.array([1.0, np.nan, 1.0], dtype=np.float32)
        is_valid, error = validate_audio(waveform, 16000)
        assert not is_valid
        assert "NaN" in error

    def test_audio_with_inf(self):
        """Audio with Inf should fail validation."""
        waveform = np.array([1.0, np.inf, 1.0], dtype=np.float32)
        is_valid, error = validate_audio(waveform, 16000)
        assert not is_valid
        assert "Inf" in error

    def test_audio_too_short(self):
        """Audio shorter than minimum duration should fail."""
        waveform = np.zeros(100, dtype=np.float32)  # ~6ms at 16kHz
        is_valid, error = validate_audio(waveform, 16000, min_duration_sec=0.1)
        assert not is_valid
        assert "too short" in error.lower()

    def test_audio_too_loud(self):
        """Audio with extreme amplitude should fail."""
        waveform = np.array([1e7], dtype=np.float32)  # Extremely loud
        is_valid, error = validate_audio(waveform, 16000)
        assert not is_valid
        assert "amplitude" in error.lower()

    def test_invalid_sample_rate(self):
        """Invalid sample rate should fail."""
        waveform = np.zeros(16000, dtype=np.float32)
        is_valid, error = validate_audio(waveform, 8)  # 8 Hz (invalid)
        assert not is_valid
        assert "sample rate" in error.lower()


# ---------------------------------------------------------------------------
# Tests: Normalization
# ---------------------------------------------------------------------------

class TestNormalizeWaveform:
    def test_normalize_preserves_dtype(self, sine_wave_16khz):
        """Normalization should preserve float32 dtype."""
        waveform, _ = sine_wave_16khz
        normalized = normalize_waveform(waveform)
        assert normalized.dtype == np.float32

    def test_normalize_clips_to_range(self):
        """Normalization should clip output to [-1, 1]."""
        waveform = np.array([5.0, -5.0, 10.0], dtype=np.float32)
        normalized = normalize_waveform(waveform, target_rms=0.1)
        assert np.all(normalized >= -1.0)
        assert np.all(normalized <= 1.0)

    def test_normalize_reduces_rms(self, sine_wave_16khz):
        """Normalization should reduce RMS for loud audio."""
        waveform, _ = sine_wave_16khz
        waveform_loud = waveform * 10  # 10x amplification
        normalized = normalize_waveform(waveform_loud, target_rms=0.1)
        
        rms_before = np.sqrt(np.mean(waveform_loud**2))
        rms_after = np.sqrt(np.mean(normalized**2))
        
        assert rms_after < rms_before


# ---------------------------------------------------------------------------
# Tests: Resampling
# ---------------------------------------------------------------------------

class TestResampleAudio:
    def test_resample_idempotent(self, sine_wave_16khz):
        """Resampling to same SR should return unchanged audio."""
        waveform, sr = sine_wave_16khz
        resampled = resample_audio(waveform, sr, sr)
        np.testing.assert_array_almost_equal(waveform, resampled)

    def test_resample_downsample_reduces_length(self, sine_wave_16khz):
        """Downsampling should reduce length."""
        waveform, sr = sine_wave_16khz
        target_sr = sr // 2
        resampled = resample_audio(waveform, sr, target_sr)
        assert len(resampled) < len(waveform)

    def test_resample_energy_preservation(self, sine_wave_16khz):
        """Energy should be preserved after resampling."""
        waveform, sr = sine_wave_16khz
        energy_before = np.sqrt(np.mean(waveform**2))
        
        resampled = resample_audio(waveform, sr, 8000, preserve_energy=True)
        energy_after = np.sqrt(np.mean(resampled**2))
        
        # Energy should be similar (allow small variance from resampling)
        assert abs(energy_before - energy_after) < 0.05


# ---------------------------------------------------------------------------
# Tests: Stereo to Mono Conversion
# ---------------------------------------------------------------------------

class TestMixStereoToMono:
    def test_mono_passthrough(self, sine_wave_16khz):
        """Mono audio should pass through unchanged."""
        waveform, _ = sine_wave_16khz
        mono = mix_stereo_to_mono(waveform)
        np.testing.assert_array_equal(waveform, mono)

    def test_stereo_conversion(self, stereo_audio):
        """Stereo audio should be converted to mono."""
        stereo, _ = stereo_audio
        mono = mix_stereo_to_mono(stereo)
        assert mono.ndim == 1
        assert len(mono) == stereo.shape[0]

    def test_stereo_loudness_preserved(self, stereo_audio):
        """Stereo→mono should preserve loudness."""
        stereo, _ = stereo_audio
        mono = mix_stereo_to_mono(stereo)
        
        # Energy should be reasonable (not completely canceled out)
        energy = np.sqrt(np.mean(mono**2))
        assert energy > 0.1  # Should have energy, not canceled


# ---------------------------------------------------------------------------
# Tests: Voice Activity Detection (VAD)
# ---------------------------------------------------------------------------

class TestDetectSpeech:
    def test_speech_detected_in_sine_wave(self, sine_wave_16khz):
        """Sine wave should be detected as speech-like."""
        waveform, sr = sine_wave_16khz
        has_speech, confidence = detect_speech(waveform, sr)
        assert has_speech or confidence > 0.1  # At least some confidence

    def test_silence_rejected(self, silence_16khz):
        """Silence should be rejected."""
        waveform, sr = silence_16khz
        has_speech, confidence = detect_speech(waveform, sr)
        # Silence should have low or zero confidence
        assert confidence < 0.5 or not has_speech

    def test_confidence_in_range(self, sine_wave_16khz):
        """Confidence should be in [0, 1]."""
        waveform, sr = sine_wave_16khz
        _, confidence = detect_speech(waveform, sr)
        assert 0.0 <= confidence <= 1.0

    def test_vad_returns_tuple(self, sine_wave_16khz):
        """VAD should return (bool, float) tuple."""
        waveform, sr = sine_wave_16khz
        result = detect_speech(waveform, sr)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], (bool, np.bool_))
        assert isinstance(result[1], (float, np.floating))


# ---------------------------------------------------------------------------
# Tests: Integration
# ---------------------------------------------------------------------------

class TestAudioPipelineIntegration:
    def test_full_pipeline_sine_wave(self, sine_wave_16khz):
        """Full pipeline should process sine wave successfully."""
        waveform, sr = sine_wave_16khz

        # Validate
        is_valid, _ = validate_audio(waveform, sr)
        assert is_valid

        # Normalize
        normalized = normalize_waveform(waveform)
        is_valid, _ = validate_audio(normalized, sr)
        assert is_valid

        # Detect speech
        has_speech, confidence = detect_speech(normalized, sr)
        assert confidence > 0

    def test_full_pipeline_stereo(self, stereo_audio):
        """Full pipeline should handle stereo input."""
        stereo, sr = stereo_audio

        # Convert to mono (should be done before resampling)
        mono = mix_stereo_to_mono(stereo)

        # Resample
        resampled = resample_audio(mono, sr, sr)

        # Validate
        is_valid, _ = validate_audio(resampled, sr)
        assert is_valid

    def test_pipeline_with_downsampling(self):
        """Pipeline should handle downsampling."""
        # Create 48 kHz audio
        sr_orig = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr_orig * duration))
        waveform_48k = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        # Resample to 16 kHz
        waveform_16k = resample_audio(waveform_48k, sr_orig, 16000)

        # Validate
        is_valid, _ = validate_audio(waveform_16k, 16000)
        assert is_valid


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_quiet_audio(self):
        """Very quiet audio should still be processable."""
        waveform = np.array([0.00001, 0.00001, 0.00001], dtype=np.float32)
        # Should not crash
        normalized = normalize_waveform(waveform)
        assert normalized.dtype == np.float32

    def test_very_loud_audio(self):
        """Very loud audio should be clipped safely."""
        waveform = np.array([0.9, 0.95, 1.0], dtype=np.float32)
        normalized = normalize_waveform(waveform)
        # Should be clipped to [-1, 1]
        assert np.all(normalized >= -1.0)
        assert np.all(normalized <= 1.0)

    def test_empty_waveform(self):
        """Empty waveform should be rejected."""
        waveform = np.array([], dtype=np.float32)
        is_valid, error = validate_audio(waveform, 16000)
        assert not is_valid

    def test_single_sample(self):
        """Single sample should be rejected (too short)."""
        waveform = np.array([0.5], dtype=np.float32)
        is_valid, error = validate_audio(waveform, 16000, min_duration_sec=0.1)
        assert not is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
