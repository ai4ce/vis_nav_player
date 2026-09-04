"""
The wire protocol between the SDK and the gateway.

This file exists twice, byte for byte: ``sdk/vis_nav_sdk/protocol.py`` in the
``vis_nav_player`` repository and ``source/sim/protocol.py`` in ``vis_nav_server``. The
server does not depend on the SDK package -- it needs thirty constants and three small
functions, not a client -- and the SDK must not depend on the server. A test in the server
repository compares the two copies whenever both checkouts are side by side, so a change to
one that forgets the other fails there rather than as a student whose actions mean something
slightly different from what the server thinks they mean.

It has **no dependencies at all**, not even NumPy. Message *validation* is not here: the
gateway validates untrusted client input with Pydantic in ``source/sim/schemas.py``. What
lives here is the set of invariants both sides have to agree on: the version, the action
bits, the limits, the frame layout, and the error codes.

The full specification is ``docs/SIM_PROTOCOL.md`` in the server repository.
"""

from __future__ import annotations

import json
import struct
from enum import IntFlag
from typing import Any, Final

# --- versioning ---------------------------------------------------------------

PROTOCOL_VERSION: Final = 1
"""Bumped by changed semantics, not by additive fields.

The SDK refuses a version it does not implement rather than guessing, because guessing
produces a student who is silently playing a different game.
"""


# --- actions ------------------------------------------------------------------


class Action(IntFlag):
    """Unchanged from ``vis_nav_game.Action``.

    The values are pinned to the original ``enum.Flag`` in ``vis_nav_game/interface.py``
    so that trajectories recorded by the server stay directly comparable with the 794
    submissions in the archive, which store these same integers.

    Combinations are legal and meaningful: ``FORWARD | LEFT`` is an arc, not an error.
    ``BACKWARD`` inverts steering, so ``BACKWARD | LEFT`` turns clockwise the way reversing
    a car does. Opposing bits cancel.
    """

    IDLE = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 4
    RIGHT = 8
    CHECKIN = 16
    QUIT = 32


MOVEMENT_ACTIONS: Final = Action.FORWARD | Action.BACKWARD | Action.LEFT | Action.RIGHT
"""The bits a ``step`` message may carry.

``CHECKIN`` and ``QUIT`` are excluded on purpose. In the old in-process game they were
actions because there was nowhere else to put them; over a wire protocol they are
messages, and accepting them here too would give two ways to end a session that would
drift apart. A student who ORs them into a step gets ``bad_action`` rather than a
surprise.
"""


def is_valid_step_action(action: int) -> bool:
    """``IDLE`` (0) is valid: it burns a tick without moving, as it always did."""
    return action == 0 or (action & ~int(MOVEMENT_ACTIONS)) == 0


# --- limits -------------------------------------------------------------------

MAX_REPEAT: Final = 256
"""Upper bound on the ticks one ``step`` message may request.

``repeat`` is not an optimisation. ``nav_steps`` is the primary ranking key after goal
tier, and one round trip per step would make a 99th-percentile run (52,735 steps) take 17
minutes at 20 ms RTT and 88 minutes at 100 ms -- ranking students by step count while
gating them on their wifi. Real play was always long presses: the reference explorer holds
forward for 4-8 ticks and a turn for 6-10.

The cap exists so a single message cannot ask for unbounded work.
"""

DEFAULT_MAX_STEPS: Final = 60_000
"""Replaces the old 60-minute wall clock, which cannot mean anything when the simulation
is remote -- it would measure network and server load.

Chosen from the archive's 751 scored submissions: p50 1,490, p95 13,878, p99 52,735, max
142,803, and the best ``perfect`` run took 421. 60,000 clears p99 with headroom and
truncates a tail that was mostly failures (``failed`` averages 7,128 steps against 3,362
for ``perfect``). Overridable per challenge.
"""

DEFAULT_IDLE_TIMEOUT_S: Final = 120
DEFAULT_SESSION_TIMEOUT_S: Final = 3600


# --- camera -------------------------------------------------------------------

CAMERA_WIDTH: Final = 320
CAMERA_HEIGHT: Final = 240
COLOR_ORDER: Final = "bgr"
"""OpenCV's order, preserved from ``vis_nav_core.py:199`` (``img[:, :, 2::-1]``).

Stated on the wire rather than assumed because a channel swap is correct half the time by
accident and costs a week to find.
"""

TARGET_COUNT: Final = 4
TARGET_YAWS_DEG: Final = (0, 90, 180, 270)
"""Front, left, back, right. Positive yaw is counter-clockwise, so ``+y`` renders on the
image left.

``vis_nav_player``'s ``baseline.py:414`` labels these Front/Right/Back/Left, which is
wrong; the yaw is carried on the wire so nobody has to rely on a label again.
"""


# --- message types ------------------------------------------------------------

# server -> client
MSG_SESSION_READY: Final = "session.ready"
MSG_TARGET: Final = "target"
MSG_OBS: Final = "obs"
MSG_RESULT: Final = "result"
MSG_ERROR: Final = "error"

# client -> server
MSG_STEP: Final = "step"
MSG_ACK: Final = "ack"
MSG_CHECKIN: Final = "checkin"
MSG_ABORT: Final = "abort"


class ErrorCode:
    BAD_SEQ: Final = "bad_seq"
    BAD_ACTION: Final = "bad_action"
    BAD_REPEAT: Final = "bad_repeat"
    STEP_BUDGET_EXHAUSTED: Final = "step_budget_exhausted"
    NOT_READY: Final = "not_ready"
    PROTOCOL: Final = "protocol"
    WORKER_LOST: Final = "worker_lost"
    IDLE_TIMEOUT: Final = "idle_timeout"
    SESSION_TIMEOUT: Final = "session_timeout"
    INTERNAL: Final = "internal"


# --- frame framing ------------------------------------------------------------

_HEADER_LENGTH = struct.Struct("<I")
MAX_HEADER_BYTES: Final = 64 * 1024
"""A bound so a malformed length prefix cannot make the peer allocate arbitrarily.

Only the server sends binary frames today, but the SDK decodes them from a host the
student does not necessarily trust either (a lab machine, a proxy), and a decoder that
trusts a length field is a decoder with a denial-of-service in it.
"""


def encode_binary_frame(header: dict[str, Any], payload: bytes) -> bytes:
    """``uint32 LE header length | UTF-8 JSON header | payload``.

    Images travel as bytes beside a JSON header rather than base64 inside one: base64
    costs 33% more bandwidth plus an encode and a decode, on the hottest path in the
    protocol. A 52,735-step run is 790 MB of imagery even before that overhead.
    """
    encoded = json.dumps(header, separators=(",", ":")).encode()
    if len(encoded) > MAX_HEADER_BYTES:
        raise ValueError(f"header is {len(encoded)} bytes, over the {MAX_HEADER_BYTES} limit")
    return _HEADER_LENGTH.pack(len(encoded)) + encoded + payload


def decode_binary_frame(frame: bytes) -> tuple[dict[str, Any], bytes]:
    """Inverse of :func:`encode_binary_frame`.

    Raises ``ValueError`` on anything malformed. Every branch here is a frame a hostile or
    broken peer could send, so none of them may raise ``struct.error`` or
    ``JSONDecodeError`` at the caller.
    """
    if len(frame) < _HEADER_LENGTH.size:
        raise ValueError(
            f"frame is {len(frame)} bytes, too short for a {_HEADER_LENGTH.size}-byte length prefix"
        )
    (header_length,) = _HEADER_LENGTH.unpack_from(frame)
    if header_length > MAX_HEADER_BYTES:
        raise ValueError(f"header claims {header_length} bytes, over the {MAX_HEADER_BYTES} limit")
    end = _HEADER_LENGTH.size + header_length
    if len(frame) < end:
        raise ValueError(
            f"header claims {header_length} bytes but only "
            f"{len(frame) - _HEADER_LENGTH.size} follow the prefix"
        )
    try:
        header = json.loads(frame[_HEADER_LENGTH.size : end])
    except json.JSONDecodeError as exc:
        raise ValueError(f"header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        # ValueError, not TypeError: every raise in this function reports one malformed
        # frame, and a caller decoding untrusted bytes should need to catch exactly one
        # exception type to handle "this frame is bad".
        raise ValueError(  # noqa: TRY004
            f"header is {type(header).__name__}, expected an object"
        )
    if "type" not in header:
        raise ValueError("header has no 'type'")
    return header, frame[end:]
