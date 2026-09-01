"""Stateful G.711 mu-law and sample-rate conversion for Twilio Media Streams.

Twilio carries 8 kHz G.711 mu-law. Dialt receives and emits 16 kHz PCM16.
Each direction owns a streaming SoXR resampler so filtering remains continuous across
WebSocket messages.
"""

from __future__ import annotations

import numpy as np
import soxr


TWILIO_SAMPLE_RATE = 8_000
DIALT_SAMPLE_RATE = 16_000
_MULAW_BIAS = 0x84
_MULAW_CLIP = 32_635


def decode_mulaw(data: bytes) -> np.ndarray:
    """Decode G.711 mu-law bytes to signed 16-bit PCM samples."""
    if not data:
        return np.empty(0, dtype="<i2")
    encoded = np.frombuffer(data, dtype=np.uint8)
    value = np.bitwise_not(encoded).astype(np.int32)
    sign = value & 0x80
    exponent = (value >> 4) & 0x07
    mantissa = value & 0x0F
    magnitude = ((mantissa << 3) + _MULAW_BIAS) << exponent
    decoded = magnitude - _MULAW_BIAS
    return np.where(sign != 0, -decoded, decoded).astype("<i2")


def encode_mulaw(samples: np.ndarray) -> bytes:
    """Encode signed PCM samples as G.711 mu-law bytes."""
    if samples.size == 0:
        return b""
    pcm = np.asarray(samples, dtype=np.int32)
    sign = np.where(pcm < 0, 0x80, 0).astype(np.int32)
    magnitude = np.minimum(np.abs(pcm), _MULAW_CLIP) + _MULAW_BIAS
    exponent = np.floor(np.log2(magnitude)).astype(np.int32) - 7
    exponent = np.clip(exponent, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    encoded = np.bitwise_not(sign | (exponent << 4) | mantissa) & 0xFF
    return encoded.astype(np.uint8).tobytes()


class TelephonyAudioBridge:
    """Convert streaming audio between Twilio mu-law 8 kHz and PCM16 16 kHz."""

    def __init__(self) -> None:
        self._to_dialt = soxr.ResampleStream(
            TWILIO_SAMPLE_RATE, DIALT_SAMPLE_RATE, 1, dtype="int16", quality="HQ"
        )
        self._to_twilio = soxr.ResampleStream(
            DIALT_SAMPLE_RATE, TWILIO_SAMPLE_RATE, 1, dtype="int16", quality="HQ"
        )
        self._pcm_byte_remainder = b""
        self._outbound_finished = False

    def twilio_to_dialt(self, data: bytes) -> bytes:
        """Decode and resample 8 kHz mu-law to 16 kHz PCM16."""
        samples = decode_mulaw(data)
        if samples.size == 0:
            return b""
        resampled = self._to_dialt.resample_chunk(samples)
        return resampled.astype("<i2", copy=False).tobytes()

    def dialt_to_twilio(self, data: bytes, *, final: bool = False) -> bytes:
        """Resample 16 kHz PCM16 and encode it as 8 kHz G.711 mu-law."""
        if self._outbound_finished:
            raise RuntimeError("outbound resampler has already been finalized")

        raw = self._pcm_byte_remainder + data
        usable_bytes = len(raw) - (len(raw) % 2)
        self._pcm_byte_remainder = raw[usable_bytes:]
        if final and self._pcm_byte_remainder:
            raise ValueError("final PCM16 chunk ends with a partial sample")
        samples = np.frombuffer(raw[:usable_bytes], dtype="<i2")
        resampled = self._to_twilio.resample_chunk(samples, last=final)
        self._outbound_finished = final
        return encode_mulaw(resampled)


def mulaw_8k_to_pcm16_16k(data: bytes) -> bytes:
    """One-shot compatibility wrapper."""
    bridge = TelephonyAudioBridge()
    samples = decode_mulaw(data)
    resampled = bridge._to_dialt.resample_chunk(samples, last=True)
    return resampled.astype("<i2", copy=False).tobytes()


def pcm16_16k_to_mulaw_8k(data: bytes) -> bytes:
    """One-shot compatibility wrapper."""
    return TelephonyAudioBridge().dialt_to_twilio(data, final=True)
