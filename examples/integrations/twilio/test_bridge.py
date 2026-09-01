import asyncio
from pathlib import Path

import bridge
import numpy as np
import pytest
from bridge import (
    PlaybackLedger,
    _signature_is_valid,
    execute_tool,
    get_settings,
    tool_manifest,
)
from telephony_audio import (
    TelephonyAudioBridge,
    decode_mulaw,
    mulaw_8k_to_pcm16_16k,
    pcm16_16k_to_mulaw_8k,
)
from twilio.request_validator import RequestValidator


def test_mulaw_silence_round_trip() -> None:
    pcm = mulaw_8k_to_pcm16_16k(bytes([0xFF]) * 160)
    assert len(pcm) == 640
    assert np.max(np.abs(np.frombuffer(pcm, dtype="<i2"))) <= 1
    round_trip = decode_mulaw(pcm16_16k_to_mulaw_8k(pcm))
    assert np.max(np.abs(round_trip)) == 0


def test_mulaw_tone_round_trip_preserves_shape() -> None:
    t = np.arange(320, dtype=np.float64) / 16_000
    source = (np.sin(2 * np.pi * 440 * t) * 12_000).astype("<i2")
    encoded = pcm16_16k_to_mulaw_8k(source.tobytes())
    decoded = np.frombuffer(mulaw_8k_to_pcm16_16k(encoded), dtype="<i2")
    assert len(encoded) == 160
    assert len(decoded) == len(source)
    assert np.corrcoef(source, decoded)[0, 1] > 0.98

def test_streamed_outbound_matches_one_shot_across_odd_byte_chunks() -> None:
    t = np.arange(16_000, dtype=np.float64) / 16_000
    source = (np.sin(2 * np.pi * 731 * t) * 12_000).astype("<i2").tobytes()
    expected = pcm16_16k_to_mulaw_8k(source)

    audio = TelephonyAudioBridge()
    chunks: list[bytes] = []
    offset = 0
    for size in (137, 503, 79, 1_019):
        while offset < len(source):
            chunk = source[offset : offset + size]
            offset += len(chunk)
            chunks.append(audio.dialt_to_twilio(chunk))
            if offset >= len(source):
                break
    chunks.append(audio.dialt_to_twilio(b"", final=True))
    actual_pcm = decode_mulaw(b"".join(chunks)).astype(np.int32)
    expected_pcm = decode_mulaw(expected).astype(np.int32)
    assert len(actual_pcm) == len(expected_pcm)
    difference = np.abs(actual_pcm - expected_pcm)
    assert np.mean(difference) < 1
    assert np.corrcoef(actual_pcm, expected_pcm)[0, 1] > 0.99999


def test_downsampling_rejects_out_of_band_tone() -> None:
    def output_rms(frequency: int) -> float:
        t = np.arange(16_000, dtype=np.float64) / 16_000
        source = (np.sin(2 * np.pi * frequency * t) * 12_000).astype("<i2")
        encoded = pcm16_16k_to_mulaw_8k(source.tobytes())
        decoded = decode_mulaw(encoded).astype(np.float64)
        return float(np.sqrt(np.mean(decoded * decoded)))

    passband_rms = output_rms(1_000)
    stopband_rms = output_rms(6_000)
    assert passband_rms > 7_000
    assert stopband_rms < passband_rms * 0.02


def test_final_partial_pcm_sample_is_rejected() -> None:
    with pytest.raises(ValueError, match="partial sample"):

        TelephonyAudioBridge().dialt_to_twilio(b"\x00", final=True)

def test_playback_ledger_tracks_unacknowledged_audio() -> None:
    ledger = PlaybackLedger()
    first = ledger.add(20)
    ledger.add(40)
    ledger.acknowledge(first)
    assert ledger.clear() == 40
    assert ledger.clear() == 0


def test_interruption_preserves_graceful_drain_and_hard_clear() -> None:
    ledger = PlaybackLedger()
    ledger.add(20)
    ledger.add(20)
    assert ledger.interruption(hard_clear=False) == (40, 0)
    assert ledger.pending_ms() == 40
    assert ledger.interruption(hard_clear=True) == (0, 40)
    assert ledger.pending_ms() == 0


def test_twilio_signatures_use_exact_http_and_websocket_urls(monkeypatch) -> None:
    monkeypatch.setenv("CONVERSE_API_KEY", "ck_test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "twilio-test-token")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://voice.example.com")
    get_settings.cache_clear()
    monkeypatch.delenv("CONVERSE_VOICE", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.voice is None
    validator = RequestValidator(settings.twilio_auth_token)

    form = {"CallSid": "CA123", "From": "+35310000000"}
    http_url = settings.http_url("/voice")
    http_signature = validator.compute_signature(http_url, form)
    assert _signature_is_valid(http_url, form, http_signature)

    websocket_url = settings.websocket_url("/media")
    websocket_signature = validator.compute_signature(websocket_url, {})
    assert _signature_is_valid(websocket_url, {}, websocket_signature)
    assert not _signature_is_valid(settings.http_url("/media"), {}, websocket_signature)
    get_settings.cache_clear()


def _configure_handoff(monkeypatch) -> None:
    monkeypatch.setenv("CONVERSE_API_KEY", "ck_test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "twilio-test-token")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC" + "1" * 32)
    monkeypatch.setenv("TWILIO_HUMAN_HANDOFF_URL", "https://customer.example/handoff")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://voice.example.com")
    get_settings.cache_clear()


def test_handoff_tool_is_optional_permissioned_and_has_no_destination(monkeypatch) -> None:
    _configure_handoff(monkeypatch)
    manifest = tool_manifest()
    assert len(manifest) == 1
    tool = manifest[0]
    assert tool["name"] == "request_human_handoff"
    assert tool["requires_permission"] is True
    assert tool["expected_duration"] == "seconds"
    assert set(tool["parameters"]["properties"]) == {"reason", "summary"}
    assert tool["parameters"]["additionalProperties"] is False

    monkeypatch.delenv("TWILIO_HUMAN_HANDOFF_URL")
    get_settings.cache_clear()
    assert tool_manifest() == []


def test_handoff_configuration_requires_account_sid_and_https(monkeypatch) -> None:
    _configure_handoff(monkeypatch)
    monkeypatch.delenv("TWILIO_ACCOUNT_SID")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="TWILIO_ACCOUNT_SID"):
        get_settings()

    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC" + "1" * 32)
    monkeypatch.setenv("TWILIO_HUMAN_HANDOFF_URL", "http://customer.example/handoff")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="must be an absolute https URL"):
        get_settings()


def test_handoff_redirects_to_customer_configuration_not_model_arguments(monkeypatch) -> None:
    _configure_handoff(monkeypatch)
    updates: list[dict] = []

    class FakeCalls:
        def __call__(self, call_sid: str):
            assert call_sid == "CA123"
            return self

        def update(self, **kwargs):
            updates.append(kwargs)

    class FakeClient:
        def __init__(self, account_sid: str, auth_token: str):
            assert account_sid == "AC" + "1" * 32
            assert auth_token == "twilio-test-token"
            self.calls = FakeCalls()

    monkeypatch.setattr(bridge, "Client", FakeClient)
    result = asyncio.run(execute_tool(
        "CA123",
        "request_human_handoff",
        {
            "reason": "The caller requested a representative.",
            "summary": "The caller needs help with an account question.",
            "destination": "https://attacker.example/redirect",
        },
    ))

    assert result == {"handoff_requested": True}
    assert updates == [{"url": "https://customer.example/handoff", "method": "POST"}]


def test_unknown_tool_still_fails_closed(monkeypatch) -> None:
    _configure_handoff(monkeypatch)
    with pytest.raises(RuntimeError, match="No handler configured"):
        asyncio.run(execute_tool("CA123", "unknown", {}))


def test_public_env_template_excludes_internal_smoke_credentials() -> None:
    env_template = Path(__file__).with_name("env.example").read_text()
    assert "TWILIO_BROWSER_" not in env_template
    assert "CONVERSE_SITE_ORIGIN" not in env_template
    assert env_template.count("TWILIO_ACCOUNT_SID") == 1
