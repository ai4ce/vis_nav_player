"""
One navigation session over the gateway's WebSocket.

Two things about the design are not free choices.

**The socket runs on its own thread.** The gateway measures a student's compute time as the
difference between two arrival times: an ``ack`` this SDK sends the instant an observation
is decoded, and the ``step`` that follows once the student's code has produced an action.
Because both travel the same path in the same direction, one-way latency and image
transfer cancel in the difference. That only holds if the ack is never queued behind the
student's own code -- so the socket, the ack, and the WebSocket pong are all serviced by a
dedicated IO thread, and a student who blocks for 400 ms in ``act()`` blocks nothing else.

**Decoding happens before the ack, on the IO thread.** JPEG decode is transport overhead;
charging it to the student would penalise nobody's algorithm and everybody's link.

See ``docs/SIM_PROTOCOL.md`` in the server repository.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np

from . import protocol as P
from .config import resolve_server, resolve_session_token, session_id_of
from .errors import SessionClosed, SimError, http_error
from .telemetry import StepTiming, Telemetry

Action = P.Action


# --- value types --------------------------------------------------------------


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    color_order: str
    encoding: str
    intrinsic_matrix: np.ndarray
    """3x3. The default worker renders ``fx = fy = 92, cx = 160, cy = 120`` at 320x240,
    but a challenge may use another camera; always read it from here."""

    @property
    def fx(self) -> float:
        return float(self.intrinsic_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.intrinsic_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.intrinsic_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.intrinsic_matrix[1, 2])


@dataclass(frozen=True)
class Limits:
    max_steps: int
    max_repeat: int
    attempts_used: int
    """Including this session."""
    attempts_allowed: int | None
    idle_timeout_s: float
    session_timeout_s: float
    motion_noise: bool = False
    """Whether each tick's speeds are perturbed. The seed is the server's; you never see it."""
    motion_noise_sigma: dict[str, float] | None = None
    """``{"linear_sigma": m/s, "angular_sigma": rad/s}`` when ``motion_noise`` is on."""


@dataclass(frozen=True)
class SessionInfo:
    """Everything known about a session before the first step."""

    session_id: str
    challenge: dict[str, Any]
    camera: Camera
    limits: Limits
    targets: list[np.ndarray]
    """Four views from the goal pose at yaw 0, 90, 180, 270 degrees: front, left, back,
    right. Same shape and colour order as observations."""


@dataclass(frozen=True)
class Observation:
    """One frame.

    Deliberately carries no pose, velocity, or collision flag. Inferring them is the task.
    """

    image: np.ndarray
    """``(height, width, 3)`` ``uint8``, **BGR** -- OpenCV's order."""
    step: int
    """Cumulative ticks used by this session. This is what becomes ``nav_steps``."""
    steps_left: int
    tick: int = 0
    """Index within a ``repeat`` run, ``0..repeat-1``."""
    final: bool = True


@dataclass(frozen=True)
class Result:
    goal_tier: str
    trans_error: float
    nav_steps: int
    job_id: str
    session_id: str
    think: dict[str, Any] = field(default_factory=dict)
    attempts_used: int = 0
    attempts_allowed: int | None = None

    @property
    def reached_goal(self) -> bool:
        return self.goal_tier in ("perfect", "partial")


# --- image decoding -----------------------------------------------------------


def decode_image(header: dict[str, Any], payload: bytes) -> np.ndarray:
    """Decode one frame according to the encoding *the frame itself* declares.

    Always returns BGR.
    """
    encoding = header.get("encoding", "jpeg")

    if encoding == "raw":
        shape = header.get("shape")
        if not (isinstance(shape, list) and len(shape) == 3):
            raise SimError(f"a raw frame needs a 3-element shape, got {shape!r}")
        height, width, channels = (int(n) for n in shape)
        expected = height * width * channels
        if len(payload) != expected:
            raise SimError(
                f"a raw {height}x{width}x{channels} frame needs {expected} bytes, "
                f"got {len(payload)}"
            )
        return np.frombuffer(payload, np.uint8).reshape(height, width, channels)

    if encoding != "jpeg":
        raise SimError(
            f"unknown image encoding {encoding!r}. Upgrade the SDK: git pull in "
            "vis_nav_player, then uv sync"
        )
    return _decode_jpeg(payload)


def _jpeg_decoder():
    """Prefer OpenCV: it decodes straight to BGR, which is the order on the wire. Pillow
    returns RGB and needs a channel reversal, so it is a fallback."""
    try:
        import cv2

        def decode_cv2(payload: bytes) -> np.ndarray:
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise SimError("could not decode the observation as a JPEG")
            return image

        return decode_cv2
    except ImportError:
        pass

    try:
        import io

        from PIL import Image

        def decode_pil(payload: bytes) -> np.ndarray:
            rgb = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
            return np.ascontiguousarray(rgb[:, :, ::-1])

        return decode_pil
    except ImportError:
        pass

    raise SimError(
        "no JPEG decoder available. Install one:\n"
        "    uv add opencv-python-headless   (recommended -- decodes directly to BGR)\n"
        "    uv add pillow                   (fallback)"
    )


_JPEG_DECODER: Any = None


def _decode_jpeg(payload: bytes) -> np.ndarray:
    global _JPEG_DECODER
    if _JPEG_DECODER is None:
        _JPEG_DECODER = _jpeg_decoder()
    return _JPEG_DECODER(payload)


# --- the IO thread ------------------------------------------------------------


class _Inbound:
    """What the IO thread hands to the calling thread."""

    __slots__ = ("kind", "payload", "received_at", "decode_ns", "payload_bytes")

    def __init__(
        self,
        kind: str,
        payload: Any,
        received_at: int = 0,
        decode_ns: int = 0,
        payload_bytes: int = 0,
    ) -> None:
        self.kind = kind
        self.payload = payload
        self.received_at = received_at
        self.decode_ns = decode_ns
        self.payload_bytes = payload_bytes


class _Pump:
    """Owns the event loop, the socket, and the ack.

    Nothing here ever calls into student code, which is the property that makes the
    gateway's timing measurement mean anything.
    """

    def __init__(self, url: str, headers: dict[str, str], open_timeout: float) -> None:
        self._url = url
        self._headers = headers
        self._open_timeout = open_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._outbound: asyncio.Queue[str | None] | None = None
        self._inbound: queue.Queue[_Inbound] = queue.Queue()
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vis-nav-sdk-io", daemon=True)

    # -- calling-thread API

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(self._open_timeout + 5.0):
            raise SimError(f"the SDK IO thread did not start within {self._open_timeout}s")
        if self._start_error is not None:
            raise self._start_error

    def send(self, message: dict[str, Any]) -> None:
        loop, outbound = self._loop, self._outbound
        if loop is None or outbound is None or self._closed.is_set():
            raise SessionClosed("the session is closed")
        loop.call_soon_threadsafe(outbound.put_nowait, json.dumps(message))

    def take(self, timeout: float) -> _Inbound:
        try:
            return self._inbound.get(timeout=timeout)
        except queue.Empty as exc:
            raise SimError(
                f"the gateway sent nothing for {timeout}s. If your method needs longer "
                "per step than that, raise recv_timeout on connect()."
            ) from exc

    def close(self) -> None:
        loop, outbound = self._loop, self._outbound
        if loop is not None and outbound is not None and not self._closed.is_set():
            loop.call_soon_threadsafe(outbound.put_nowait, None)
        self._thread.join(timeout=5.0)

    # -- IO thread

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:  # noqa: BLE001 - surfaced on the calling thread
            if not self._ready.is_set():
                self._start_error = exc
                self._ready.set()
            else:
                self._inbound.put(_Inbound("fatal", exc))
        finally:
            self._closed.set()
            self._inbound.put(_Inbound("closed", None))

    async def _main(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise SimError("the websockets package is required: uv add websockets") from exc

        try:
            await self._dial(websockets)
        except Exception as exc:
            raise _translate_handshake(exc) from exc

    async def _dial(self, websockets: Any) -> None:
        self._loop = asyncio.get_running_loop()
        self._outbound = asyncio.Queue()

        # `additional_headers` since websockets 14; `extra_headers` before it.
        try:
            async with websockets.connect(
                self._url,
                additional_headers=self._headers,
                open_timeout=self._open_timeout,
                max_size=16 * 1024 * 1024,
            ) as websocket:
                await self._serve(websocket)
                return
        except TypeError:
            pass

        async with websockets.connect(  # type: ignore[call-arg]
            self._url,
            extra_headers=self._headers,
            open_timeout=self._open_timeout,
            max_size=16 * 1024 * 1024,
        ) as websocket:
            await self._serve(websocket)

    async def _serve(self, websocket: Any) -> None:
        self._ready.set()
        writer = asyncio.create_task(self._write(websocket))
        try:
            await self._read(websocket)
        finally:
            writer.cancel()

    async def _write(self, websocket: Any) -> None:
        assert self._outbound is not None
        while True:
            message = await self._outbound.get()
            if message is None:
                await websocket.close()
                return
            await websocket.send(message)

    async def _read(self, websocket: Any) -> None:
        assert self._outbound is not None
        async for frame in websocket:
            received_at = time.perf_counter_ns()
            if isinstance(frame, bytes):
                header, payload = P.decode_binary_frame(frame)
                kind = header.get("type")
                if kind == P.MSG_OBS:
                    # Decode, *then* ack, then hand over. Decode is transport overhead;
                    # the ack marks the boundary after which everything is the student's.
                    image = decode_image(header, payload)
                    decoded_at = time.perf_counter_ns()
                    self._outbound.put_nowait(
                        json.dumps({"type": P.MSG_ACK, "seq": header.get("seq")})
                    )
                    self._inbound.put(
                        _Inbound(
                            P.MSG_OBS,
                            (header, image),
                            received_at=received_at,
                            decode_ns=decoded_at - received_at,
                            payload_bytes=len(payload),
                        )
                    )
                elif kind == P.MSG_TARGET:
                    self._inbound.put(
                        _Inbound(P.MSG_TARGET, (header, decode_image(header, payload)))
                    )
                else:
                    raise SimError(f"unexpected binary frame of type {kind!r}")
                continue

            message = json.loads(frame)
            if message.get("type") == "ping":
                # Answered here, never handed to the caller: the gateway uses the round
                # trip as a network baseline, so a pong that waited behind the student's
                # model would measure the student, not the network.
                self._outbound.put_nowait(json.dumps({"type": "pong", "id": message.get("id")}))
                continue
            self._inbound.put(_Inbound(str(message.get("type")), message))


# --- the session --------------------------------------------------------------


class Session:
    """One navigation attempt. Built by :func:`connect`."""

    def __init__(self, pump: _Pump, recv_timeout: float) -> None:
        self._pump = pump
        self._recv_timeout = recv_timeout
        self._seq = 0
        self._closed = False
        self._result: Result | None = None
        self._info: SessionInfo | None = None
        # When the last observation was handed to the caller. The next call's think time is
        # measured from here, so it covers everything the student did between two steps.
        self._returned_at: int | None = None

        self.initial_observation: Observation | None = None
        self.telemetry = Telemetry()

    # -- what the handshake told us

    @property
    def info(self) -> SessionInfo:
        assert self._info is not None
        return self._info

    @property
    def session_id(self) -> str:
        return self.info.session_id

    @property
    def challenge(self) -> dict[str, Any]:
        return self.info.challenge

    @property
    def camera(self) -> Camera:
        return self.info.camera

    @property
    def limits(self) -> Limits:
        return self.info.limits

    @property
    def targets(self) -> list[np.ndarray]:
        return self.info.targets

    @property
    def result(self) -> Result | None:
        return self._result

    @property
    def closed(self) -> bool:
        return self._closed

    # -- handshake

    def _handshake(self) -> None:
        ready = self._expect(P.MSG_SESSION_READY).payload
        version = ready.get("protocol")
        if version != P.PROTOCOL_VERSION:
            raise SimError(
                f"the gateway speaks protocol v{version}, this SDK speaks "
                f"v{P.PROTOCOL_VERSION}. Upgrade the SDK: git pull in vis_nav_player, "
                "then uv sync"
            )

        camera = ready["camera"]
        limits = ready["limits"]
        expected = ready.get("target_count", P.TARGET_COUNT)
        targets: list[tuple[int, np.ndarray]] = []
        while len(targets) < expected:
            header, image = self._expect(P.MSG_TARGET).payload
            targets.append((header["index"], image))

        self._info = SessionInfo(
            session_id=ready["session_id"],
            challenge=ready["challenge"],
            camera=Camera(
                width=camera["width"],
                height=camera["height"],
                color_order=camera["color_order"],
                encoding=camera.get("encoding", "jpeg"),
                intrinsic_matrix=np.asarray(camera["intrinsic_matrix"], dtype=np.float64),
            ),
            limits=Limits(
                max_steps=limits["max_steps"],
                max_repeat=limits["max_repeat"],
                attempts_used=limits["attempts_used"],
                attempts_allowed=limits.get("attempts_allowed"),
                idle_timeout_s=limits["idle_timeout_s"],
                session_timeout_s=limits["session_timeout_s"],
                motion_noise=limits.get("motion_noise", False),
                motion_noise_sigma=limits.get("motion_noise_sigma"),
            ),
            targets=[image for _, image in sorted(targets, key=lambda t: t[0])],
        )

        header, image = self._expect(P.MSG_OBS).payload
        self.initial_observation = _observation(header, image)
        self._returned_at = time.perf_counter_ns()

    # -- stepping

    def step(self, action: Action | int, repeat: int = 1, *, frames: str = "last") -> Observation:
        """Apply ``action`` for ``repeat`` ticks and return the resulting observation.

        Every tick is charged against ``nav_steps``; ``repeat`` buys wall clock, never
        score. A run of 1,500 steps at one round trip per tick is minutes of pure latency;
        drive in bursts of 4-8 ticks, which is also how the maze was meant to be driven.
        """
        return self.step_all(action, repeat, frames=frames)[-1]

    def step_all(
        self, action: Action | int, repeat: int = 1, *, frames: str = "last"
    ) -> list[Observation]:
        """As :meth:`step`, returning every observation the gateway sent (``frames="all"``
        or ``"every:N"`` for intermediate ticks)."""
        self._require_open()
        value = int(action)
        if not P.is_valid_step_action(value):
            raise SimError(
                f"action {value} is not a movement action. CHECKIN and QUIT are not step "
                "actions over the wire -- call checkin() or abort() instead."
            )
        limit = self.limits.max_repeat if self._info else P.MAX_REPEAT
        if not 1 <= repeat <= limit:
            raise SimError(f"repeat must be between 1 and {limit}, got {repeat}")

        seq = self._next_seq()
        message: dict[str, Any] = {
            "type": P.MSG_STEP,
            "seq": seq,
            "action": value,
            "repeat": repeat,
            "frames": frames,
        }
        think = self._think_ns()
        if think is not None:
            message["think_ns"] = think
        sent_at = time.perf_counter_ns()
        self._pump.send(message)

        collected: list[Observation] = []
        decode_ns = 0
        payload_bytes = 0
        while True:
            item = self._expect(P.MSG_OBS)
            header, image = item.payload
            decode_ns += item.decode_ns
            payload_bytes += item.payload_bytes
            collected.append(_observation(header, image))
            if header.get("final", True):
                break
        server_ns = header.get("server_ns")
        self.telemetry.record(
            StepTiming(
                seq=seq,
                repeat=repeat,
                frames=len(collected),
                payload_bytes=payload_bytes,
                rtt_ms=(item.received_at - sent_at) / 1e6,
                server_ms=None if server_ns is None else server_ns / 1e6,
                decode_ms=decode_ns / 1e6,
                think_ms=None if think is None else think / 1e6,
                at=time.perf_counter(),
            )
        )
        self._returned_at = time.perf_counter_ns()
        return collected

    def checkin(self) -> Result:
        """ "I am at the goal." Scores the session at the current pose and ends it."""
        self._require_open()
        message: dict[str, Any] = {"type": P.MSG_CHECKIN, "seq": self._next_seq()}
        think = self._think_ns()
        if think is not None:
            message["think_ns"] = think
        self._pump.send(message)

        payload = self._expect(P.MSG_RESULT).payload
        self._result = Result(
            goal_tier=payload["goal_tier"],
            trans_error=payload["trans_error"],
            nav_steps=payload["nav_steps"],
            job_id=payload["job_id"],
            session_id=payload["session_id"],
            think=payload.get("think", {}),
            attempts_used=payload.get("attempts_used", 0),
            attempts_allowed=payload.get("attempts_allowed"),
        )
        self._closed = True
        return self._result

    def abort(self, reason: str = "") -> None:
        """End without a score. **Still consumes an attempt** -- a free abort would let a
        student probe and discard until a run looked promising."""
        if self._closed:
            return
        with contextlib.suppress(SessionClosed):
            self._pump.send({"type": P.MSG_ABORT, "seq": self._next_seq(), "reason": reason[:200]})
        self._closed = True

    # -- plumbing

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def _think_ns(self) -> int | None:
        if self._returned_at is None:
            return None
        return time.perf_counter_ns() - self._returned_at

    def _require_open(self) -> None:
        if self._closed:
            raise SessionClosed(
                "this session is over. Sessions are single-use -- the trajectory is one "
                "continuous server-owned run, so there is no resume. Call connect() again "
                "for another attempt."
            )

    def _expect(self, kind: str) -> _Inbound:
        while True:
            item = self._pump.take(self._recv_timeout)
            if item.kind == kind:
                return item
            if item.kind == P.MSG_ERROR:
                message = item.payload
                error = SimError(
                    message.get("message", "the gateway reported an error"),
                    code=message.get("code"),
                )
                if message.get("fatal", True):
                    self._closed = True
                raise error
            if item.kind == "fatal":
                self._closed = True
                raise SimError(f"the connection failed: {item.payload}")
            if item.kind == "closed":
                self._closed = True
                raise SessionClosed("the gateway closed the connection")
            # Anything else is a message type we do not model yet; additive types must
            # not break an older SDK.

    def close(self) -> None:
        self.abort("client closed the session")
        self._pump.close()

    def __enter__(self) -> Session:  # noqa: PYI034 - typing.Self is 3.11+
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _observation(header: dict[str, Any], image: np.ndarray) -> Observation:
    return Observation(
        image=image,
        step=header["step"],
        steps_left=header["steps_left"],
        tick=header.get("tick", 0),
        final=header.get("final", True),
    )


def connect(
    token: str | None = None,
    *,
    server: str | None = None,
    recv_timeout: float = 60.0,
    open_timeout: float = 20.0,
) -> Session:
    """Redeem a session token from the challenge page and open the run it reserved.

    ``token`` falls back to ``$VIS_NAV_SESSION``; ``server`` to ``$VIS_NAV_SERVER`` and then
    the course server. A token opens exactly one session; the page hands out a new one each
    time you press Start.

        with connect("vns_...") as session:
            obs = session.initial_observation
            while not done:
                obs = session.step(Action.FORWARD, repeat=4)
            print(session.checkin())
    """
    from . import __version__

    token = resolve_session_token(token)
    url = websocket_url(resolve_server(server), session_id_of(token))
    headers = {"x-session-token": token, "x-sdk-version": __version__}
    pump = _Pump(url, headers, open_timeout)
    pump.start()
    session = Session(pump, recv_timeout)
    try:
        session._handshake()
    except BaseException:
        pump.close()
        raise
    return session


def websocket_url(server: str, session_id: str) -> str:
    parts = urlsplit(server if "//" in server else f"//{server}")
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(parts.scheme, "wss")
    path = parts.path.rstrip("/") + f"/v1/sim/session/{quote(session_id, safe='')}"
    return urlunsplit((scheme, parts.netloc, path, "", ""))


def _translate_handshake(exc: BaseException) -> BaseException:
    """Turn a websockets handshake rejection into something worth reading; anything
    unrecognised passes through unchanged rather than wrapped in a guess."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        return exc
    detail = None
    try:
        body = json.loads(getattr(response, "body", b"") or b"")
        if isinstance(body, dict) and isinstance(body.get("detail"), str):
            detail = body["detail"]
    except ValueError:
        pass
    return http_error(status, detail)
