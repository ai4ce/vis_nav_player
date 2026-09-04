"""
A pygame window: the camera view, the four target views, and what each step is costing --
round trip, the server's share, the network's share, decode, and the agent's own time.

pygame is an optional dependency (the ``viewer`` extra). Nothing else in the SDK imports
this module at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import SimError
from .session import Observation, SessionInfo
from .telemetry import Telemetry

if TYPE_CHECKING:
    from .agent import Agent

TARGET_LABELS = ("front", "left", "back", "right")

# -- palette
BG = (17, 19, 24)
CARD = (26, 29, 36)
EDGE = (42, 46, 56)
TEXT = (230, 230, 235)
DIM = (138, 144, 160)
FAINT = (70, 76, 90)
AMBER = (245, 177, 76)
GREEN = (76, 209, 132)
RED = (255, 107, 107)
BLUE = (108, 180, 255)
VIOLET = (178, 140, 255)

PAD = 12
GAP = 10
COLUMN_W = 360
RADIUS = 8

SANS = "sfprotext,helveticaneue,segoeui,inter,dejavusans,arial"
MONO = "sfmono,menlo,consolas,jetbrainsmono,dejavusansmono,monospace"


def open_viewer(viewer: Viewer | bool | None) -> Viewer | None:
    """Resolve the ``viewer`` argument of :func:`vis_nav_sdk.run`."""
    if viewer is False:
        return None
    if isinstance(viewer, Viewer):
        viewer.open()
        return viewer
    try:
        window = Viewer()
        window.open()
    except SimError:
        if viewer is True:
            raise
        return None
    return window


@dataclass(frozen=True)
class Keys:
    """The keyboard as seen by an agent, by pygame key name (``"up"``, ``"space"``, ...).

    ``held`` is what is down right now; ``tapped`` is everything pressed since the previous
    :meth:`Viewer.keys` call, so a tap shorter than one poll is still seen once.
    """

    held: frozenset[str]
    tapped: frozenset[str]

    def __contains__(self, name: str) -> bool:
        return name in self.held or name in self.tapped

    @property
    def any(self) -> bool:
        return bool(self.held or self.tapped)


class Viewer:
    """Owns the pygame display and the event queue. Agents read the keyboard through
    :meth:`keys` (see :attr:`Agent.viewer`)."""

    def __init__(self, scale: int = 2, title: str = "vis-nav", hold_s: float = 5.0) -> None:
        self.scale = scale
        self.title = title
        self.hold_s = hold_s
        self.closed = False
        self._pg: Any = None
        self._screen: Any = None
        self._fonts: dict[str, Any] = {}
        self._size: tuple[int, int] | None = None
        self._targets: list[Any] = []
        self._thumb: tuple[int, int] = (160, 120)
        self._info: SessionInfo | None = None
        self._held: set[str] = set()
        self._tapped: set[str] = set()

    # -- lifecycle

    def open(self) -> None:
        if self._pg is not None:
            return
        try:
            import pygame
        except ImportError as exc:
            raise SimError(
                "the viewer needs pygame: uv add pygame  (or run(..., viewer=False))"
            ) from exc
        pygame.init()
        pygame.display.set_caption(self.title)
        self._pg = pygame
        self._fonts = {
            "ui": pygame.font.SysFont(SANS, 13),
            "small": pygame.font.SysFont(SANS, 11),
            "title": pygame.font.SysFont(SANS, 14, bold=True),
            "big": pygame.font.SysFont(SANS, 22, bold=True),
            "mono": pygame.font.SysFont(MONO, 13),
            "mono_small": pygame.font.SysFont(MONO, 11),
        }
        self._resize((320 * self.scale + COLUMN_W + 3 * PAD, 240 * self.scale + 2 * PAD + 44))
        self._screen.fill(BG)
        self._text("connecting…", (PAD, PAD), DIM)
        pygame.display.flip()

    def attach(self, info: SessionInfo) -> None:
        self._info = info
        w = (COLUMN_W - 2 * PAD - GAP) // 2
        h = round(w * info.camera.height / info.camera.width)
        self._thumb = (w, h)
        self._targets = [
            self._pg.transform.smoothscale(self._surface(img), (w, h)) for img in info.targets
        ]

    def close(self) -> None:
        if self._pg is not None:
            self._pg.quit()
            self._pg = None
            self._screen = None

    # -- events

    def pump(self) -> None:
        """Drain the event queue. Sets :attr:`closed` when the window is closed."""
        if self._pg is None:
            return
        pg = self._pg
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.closed = True
            elif event.type == pg.KEYDOWN:
                name = pg.key.name(event.key)
                self._held.add(name)
                self._tapped.add(name)
            elif event.type == pg.KEYUP:
                self._held.discard(pg.key.name(event.key))
            elif event.type == pg.WINDOWFOCUSLOST:
                # The release will go to whichever window has focus now, not to us.
                self._held.clear()

    def keys(self) -> Keys:
        """Current keyboard state; clears the tap record.

        Tracked from key events rather than ``pygame.key.get_pressed()``: that array is
        indexed by keycode but only 512 long, and the arrow keys' SDL2 keycodes are above
        a billion, so they can never be found in it.
        """
        if self._pg is None:
            raise SimError("the viewer window is not open")
        tapped = frozenset(self._tapped)
        self._tapped.clear()
        return Keys(held=frozenset(self._held), tapped=tapped)

    # -- per step

    def update(self, obs: Observation, telemetry: Telemetry, agent: Agent) -> None:
        pg = self._pg
        if pg is None:
            return
        self.pump()

        h, w = obs.image.shape[:2]
        view = pg.transform.scale(self._surface(obs.image), (w * self.scale, h * self.scale))
        view_w, view_h = view.get_size()
        top_h = 44
        column_x = PAD + view_w + PAD

        agent_lines = list(agent.hud())
        cards = [
            ("targets", 2 * self._thumb[1] + 2 * 16 + 30),
            ("latency", 30 + 6 * 22 + 56),
        ]
        if agent_lines:
            cards.append(("agent", 30 + 20 * len(agent_lines) + 8))
        column_h = sum(height + GAP for _, height in cards) - GAP

        panel_w = view_w + PAD + COLUMN_W
        panel = self._panel(agent.panel(), panel_w)

        body_h = max(view_h + 24, column_h)
        total_w = PAD + view_w + PAD + COLUMN_W + PAD
        total_h = top_h + body_h + PAD + (panel.get_height() + 2 * PAD if panel else 0)
        self._resize((total_w, total_h))

        screen = self._screen
        screen.fill(BG)
        self._draw_topbar(obs, telemetry, total_w, top_h)

        # camera
        y0 = top_h
        self._card((PAD, y0, view_w, view_h + 24))
        screen.blit(view, (PAD, y0))
        pg.draw.rect(screen, EDGE, (PAD, y0, view_w, view_h), 1)
        self._text(
            f"camera {w}×{h}  ·  step {obs.step}  ·  {obs.steps_left} left",
            (PAD + 8, y0 + view_h + 5),
            DIM,
            "small",
        )

        # column
        y = y0
        for name, height in cards:
            rect = (column_x, y, COLUMN_W, height)
            self._card(rect)
            if name == "targets":
                self._draw_targets(rect)
            elif name == "latency":
                self._draw_latency(rect, telemetry)
            else:
                self._draw_agent(rect, agent_lines)
            y += height + GAP

        if panel is not None:
            py = top_h + body_h + PAD
            self._card((PAD, py, panel_w, panel.get_height() + PAD))
            screen.blit(panel, (PAD, py + PAD // 2))
        pg.display.flip()

    def hold(self, message: str, timeout: float | None = None) -> None:
        """Show ``message`` in a banner until a key is pressed, the window is closed, or
        ``timeout`` (default :attr:`hold_s`) passes."""
        pg = self._pg
        if pg is None or self.closed:
            return
        if timeout is None:
            timeout = self.hold_s
        w, h = self._size or (640, 480)
        colour = GREEN if message.startswith(("PERFECT", "PARTIAL")) else AMBER
        if message.startswith("FAILED"):
            colour = RED
        label = self._fonts["big"].render(message, True, TEXT)
        hint = self._fonts["small"].render("press any key to close", True, DIM)
        bw = max(label.get_width(), hint.get_width()) + 48
        bh = label.get_height() + hint.get_height() + 36
        x, y = (w - bw) // 2, (h - bh) // 2
        shade = pg.Surface((w, h), pg.SRCALPHA)
        shade.fill((0, 0, 0, 140))
        self._screen.blit(shade, (0, 0))
        pg.draw.rect(self._screen, CARD, (x, y, bw, bh), border_radius=RADIUS)
        pg.draw.rect(self._screen, colour, (x, y, bw, bh), 2, border_radius=RADIUS)
        self._screen.blit(label, (x + (bw - label.get_width()) // 2, y + 14))
        self._screen.blit(hint, (x + (bw - hint.get_width()) // 2, y + bh - hint.get_height() - 12))
        pg.display.flip()
        deadline = pg.time.get_ticks() + int(timeout * 1000)
        while pg.time.get_ticks() < deadline:
            for event in pg.event.get():
                if event.type in (pg.QUIT, pg.KEYDOWN):
                    return
            pg.time.wait(20)

    # -- sections

    def _draw_topbar(self, obs: Observation, telemetry: Telemetry, width: int, height: int) -> None:
        info = self._info
        name = info.challenge.get("name", "") if info else ""
        self._text(name or self.title, (PAD, 10), TEXT, "title")
        x = PAD + self._fonts["title"].size(name or self.title)[0] + 14
        if info is not None:
            x = self._badge(f"session {info.session_id}", (x, 9), DIM)
            allowed = info.limits.attempts_allowed
            used = info.limits.attempts_used
            attempts = f"attempt {used}/{allowed}" if allowed is not None else f"attempt {used}"
            x = self._badge(attempts, (x, 9), AMBER if allowed and used >= allowed else BLUE)
        rate = telemetry.steps_per_second()
        if rate is not None:
            self._badge(f"{rate:.0f} steps/s", (x, 9), DIM)

        if info is not None:
            bar_w = 220
            bx = width - PAD - bar_w
            frac = obs.step / max(1, info.limits.max_steps)
            self._text(
                f"{obs.step:,} / {info.limits.max_steps:,} ticks",
                (bx, 6),
                DIM,
                "small",
            )
            self._bar((bx, 26, bar_w, 6), frac, BLUE if frac < 0.8 else AMBER)

    def _draw_targets(self, rect: tuple[int, int, int, int]) -> None:
        x0, y0, _, _ = rect
        self._text("TARGETS", (x0 + PAD, y0 + 9), DIM, "small")
        self._text("views from the goal", (x0 + PAD + 58, y0 + 9), FAINT, "small")
        tw, th = self._thumb
        for i, surface in enumerate(self._targets[:4]):
            x = x0 + PAD + (i % 2) * (tw + GAP)
            y = y0 + 30 + (i // 2) * (th + 16)
            self._screen.blit(surface, (x, y))
            self._pg.draw.rect(self._screen, AMBER if i == 0 else EDGE, (x, y, tw, th), 1)
            self._text(TARGET_LABELS[i], (x, y + th + 2), AMBER if i == 0 else DIM, "small")

    def _draw_latency(self, rect: tuple[int, int, int, int], telemetry: Telemetry) -> None:
        x0, y0, w, _ = rect
        self._text("LATENCY", (x0 + PAD, y0 + 9), DIM, "small")
        last = telemetry.last
        if last is None:
            self._text("waiting for the first step…", (x0 + PAD, y0 + 34), FAINT)
            return
        self._text("last", (x0 + w - PAD - 122, y0 + 9), FAINT, "small")
        self._text("p50", (x0 + w - PAD - 46, y0 + 9), FAINT, "small")

        scale = max(20.0, 2 * (telemetry.median("rtt_ms") or 0.0))
        rows = [
            ("round trip", last.rtt_ms, telemetry.median("rtt_ms"), BLUE),
            ("  server", last.server_ms, telemetry.median("server_ms"), VIOLET),
            ("  network", last.network_ms, telemetry.median("network_ms"), BLUE),
            ("  decode", last.decode_ms, telemetry.median("decode_ms"), DIM),
            ("your code", last.think_ms, telemetry.median("think_ms"), AMBER),
        ]
        y = y0 + 30
        bar_x = x0 + PAD + 78
        bar_w = w - 2 * PAD - 78 - 130
        for label, value, median, colour in rows:
            self._text(label, (x0 + PAD, y + 3), TEXT if not label.startswith(" ") else DIM)
            if value is not None:
                self._bar((bar_x, y + 8, bar_w, 6), min(1.0, value / scale), colour)
            self._text(_ms(value), (x0 + w - PAD - 122, y + 3), TEXT, "mono")
            self._text(_ms(median), (x0 + w - PAD - 46, y + 3), DIM, "mono_small")
            y += 22

        rtts = telemetry.values("rtt_ms")
        self._text(
            f"repeat {last.repeat}  ·  {last.payload_bytes / 1024:.1f} KB/step  ·  "
            f"round trip, last {len(rtts)} steps, max {max(rtts):.0f} ms",
            (x0 + PAD, y + 4),
            FAINT,
            "small",
        )
        self._sparkline((x0 + PAD, y + 22, w - 2 * PAD, 28), rtts, BLUE)

    def _draw_agent(self, rect: tuple[int, int, int, int], lines: list[str]) -> None:
        x0, y0, w, h = rect
        self._text("AGENT", (x0 + PAD, y0 + 9), DIM, "small")
        self._screen.set_clip((x0 + PAD, y0, w - 2 * PAD, h))
        for i, line in enumerate(lines):
            self._text(line, (x0 + PAD, y0 + 30 + 20 * i), TEXT)
        self._screen.set_clip(None)

    def _panel(self, content: Any, width: int) -> Any:
        """Typeset what ``agent.panel()`` returned onto one surface ``width`` wide."""
        if content is None:
            return None
        pg = self._pg
        if isinstance(content, np.ndarray):
            ph, pw = content.shape[:2]
            return pg.transform.smoothscale(
                self._surface(content), (width, max(1, round(ph * width / pw)))
            )
        tiles = list(content)
        if not tiles:
            return None
        inner = width - 2 * PAD
        tw = min(200, (inner - GAP * (len(tiles) - 1)) // len(tiles))
        th = round(tw * 0.75)
        surface = pg.Surface((width, th + 22), pg.SRCALPHA)
        x = PAD
        for tile in tiles:
            image, label = tile[0], tile[1]
            colour = tile[2] if len(tile) > 2 else DIM
            surface.blit(pg.transform.smoothscale(self._surface(image), (tw, th)), (x, 0))
            pg.draw.rect(surface, colour, (x, 0, tw, th), 1)
            text = self._fonts["small"].render(label, True, colour)
            surface.blit(text, (x, th + 4), (0, 0, tw, text.get_height()))
            x += tw + GAP
        return surface

    # -- primitives

    def _card(self, rect: tuple[int, int, int, int]) -> None:
        self._pg.draw.rect(self._screen, CARD, rect, border_radius=RADIUS)
        self._pg.draw.rect(self._screen, EDGE, rect, 1, border_radius=RADIUS)

    def _badge(self, text: str, at: tuple[int, int], colour: tuple[int, int, int]) -> int:
        label = self._fonts["small"].render(text, True, colour)
        w, h = label.get_width() + 14, label.get_height() + 6
        self._pg.draw.rect(self._screen, CARD, (*at, w, h), border_radius=h // 2)
        self._pg.draw.rect(self._screen, EDGE, (*at, w, h), 1, border_radius=h // 2)
        self._screen.blit(label, (at[0] + 7, at[1] + 3))
        return at[0] + w + 8

    def _bar(
        self, rect: tuple[int, int, int, int], frac: float, colour: tuple[int, int, int]
    ) -> None:
        x, y, w, h = rect
        self._pg.draw.rect(self._screen, FAINT, rect, border_radius=h // 2)
        fill = max(h, round(w * max(0.0, min(1.0, frac))))
        self._pg.draw.rect(self._screen, colour, (x, y, fill, h), border_radius=h // 2)

    def _sparkline(
        self, rect: tuple[int, int, int, int], values: list[float], colour: tuple[int, int, int]
    ) -> None:
        x, y, w, h = rect
        self._pg.draw.rect(self._screen, BG, rect, border_radius=4)
        if len(values) < 2:
            return
        top = max(values) or 1.0
        n = len(values)
        points = [
            (x + 2 + i * (w - 4) / (n - 1), y + h - 2 - (h - 4) * min(1.0, v / top))
            for i, v in enumerate(values)
        ]
        self._pg.draw.aalines(self._screen, colour, False, points)

    def _text(
        self,
        text: str,
        at: tuple[int, int],
        colour: tuple[int, int, int],
        font: str = "ui",
    ) -> None:
        self._screen.blit(self._fonts[font].render(text, True, colour), at)

    def _surface(self, bgr: np.ndarray) -> Any:
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        return self._pg.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")

    def _resize(self, size: tuple[int, int]) -> None:
        if size != self._size:
            self._size = size
            self._screen = self._pg.display.set_mode(size)


def _ms(value: float | None) -> str:
    if value is None:
        return "  –"
    return f"{value:5.1f} ms" if value < 100 else f"{value:5.0f} ms"
