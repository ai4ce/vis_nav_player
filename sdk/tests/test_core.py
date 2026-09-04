from __future__ import annotations

import pytest
from vis_nav_sdk import Action, ConfigError
from vis_nav_sdk import protocol as P
from vis_nav_sdk.config import (
    DEFAULT_SERVER,
    resolve_api_key,
    resolve_server,
    resolve_session_token,
    session_id_of,
)
from vis_nav_sdk.errors import http_error
from vis_nav_sdk.session import decode_image, websocket_url
from vis_nav_sdk.telemetry import StepTiming, Telemetry


def test_settings_resolve_in_order(monkeypatch):
    monkeypatch.delenv("VIS_NAV_SERVER", raising=False)
    monkeypatch.delenv("VIS_NAV_API_KEY", raising=False)
    assert resolve_server(None) == DEFAULT_SERVER
    monkeypatch.setenv("VIS_NAV_SERVER", "http://env:8000/")
    assert resolve_server(None) == "http://env:8000"
    assert resolve_server("http://arg ") == "http://arg"
    with pytest.raises(ConfigError, match="VIS_NAV_API_KEY"):
        resolve_api_key(None)
    monkeypatch.setenv("VIS_NAV_API_KEY", " env-key ")
    assert resolve_api_key(None) == "env-key"
    assert resolve_api_key("arg-key") == "arg-key"


@pytest.mark.parametrize(
    ("server", "expected"),
    [
        ("https://api.example.edu", "wss://api.example.edu/v1/sim/session/abc"),
        ("http://localhost:8000", "ws://localhost:8000/v1/sim/session/abc"),
        ("wss://api.example.edu/prefix/", "wss://api.example.edu/prefix/v1/sim/session/abc"),
        ("api.example.edu", "wss://api.example.edu/v1/sim/session/abc"),
    ],
)
def test_websocket_url(server, expected):
    assert websocket_url(server, "abc") == expected


def test_session_ids_are_quoted():
    assert websocket_url("https://h", "a/b c") == "wss://h/v1/sim/session/a%2Fb%20c"


def test_session_tokens_resolve_and_parse(monkeypatch):
    monkeypatch.delenv("VIS_NAV_SESSION", raising=False)
    with pytest.raises(ConfigError, match="Start"):
        resolve_session_token(None)
    with pytest.raises(ConfigError, match="not a session token"):
        resolve_session_token("dev-key-alice")
    token = "vns_abc123def456_s3cr3t-part_with_underscores"
    monkeypatch.setenv("VIS_NAV_SESSION", f" {token} ")
    assert resolve_session_token(None) == token
    assert session_id_of(token) == "abc123def456"


def test_binary_frames_round_trip():
    header = {"type": "obs", "seq": 3, "step": 12}
    payload = b"\xff\xd8jpeg"
    decoded, body = P.decode_binary_frame(P.encode_binary_frame(header, payload))
    assert decoded == header and body == payload
    with pytest.raises(ValueError):
        P.decode_binary_frame(b"\x01")
    with pytest.raises(ValueError):
        P.decode_binary_frame(b"\xff\xff\xff\x7f" + b"x" * 8)


def test_raw_frames_decode_without_a_codec():
    image = decode_image({"encoding": "raw", "shape": [2, 3, 3]}, bytes(range(18)))
    assert image.shape == (2, 3, 3) and image[1, 2, 2] == 17
    with pytest.raises(Exception, match="18 bytes"):
        decode_image({"encoding": "raw", "shape": [2, 3, 3]}, b"short")


def test_actions_are_the_original_bits():
    names = ["IDLE", "FORWARD", "BACKWARD", "LEFT", "RIGHT", "CHECKIN", "QUIT"]
    assert [int(Action[n]) for n in names] == [0, 1, 2, 4, 8, 16, 32]
    assert P.is_valid_step_action(int(Action.FORWARD | Action.LEFT))
    assert not P.is_valid_step_action(int(Action.CHECKIN))


def test_http_errors_explain_themselves():
    error = http_error(429)
    assert error.status == 429 and "attempts" in str(error)
    assert "(nope)" in str(http_error(418, "nope"))


def _timing(seq: int, rtt: float, server: float | None = 2.0, repeat: int = 4) -> StepTiming:
    return StepTiming(
        seq=seq,
        repeat=repeat,
        frames=1,
        payload_bytes=1000,
        rtt_ms=rtt,
        server_ms=server,
        decode_ms=0.5,
        think_ms=3.0,
        at=seq * 0.1,
    )


def test_telemetry_windows_and_summarises():
    telemetry = Telemetry(window=3)
    assert telemetry.last is None and telemetry.steps_per_second() is None
    for seq, rtt in enumerate([10.0, 20.0, 30.0, 40.0]):
        telemetry.record(_timing(seq, rtt))
    assert telemetry.count == 4 and telemetry.ticks == 16
    assert telemetry.last.rtt_ms == 40.0
    assert telemetry.median("rtt_ms") == 30.0  # window of 3: 20, 30, 40
    assert telemetry.last.network_ms == 38.0
    assert telemetry.steps_per_second() == pytest.approx(10.0)
    assert telemetry.ticks_per_second() == pytest.approx(40.0)
    summary = telemetry.summary()
    assert summary["steps"] == 4 and summary["payload_bytes"] == 4000


def test_network_share_is_unknown_without_the_server_figure():
    assert _timing(0, 10.0, server=None).network_ms is None
    assert _timing(0, 1.0, server=5.0).network_ms == 0.0
