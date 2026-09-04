"""
Everything the SDK says to the terminal, in one place, through rich.

Kept deliberately small: a few panels, a confirm, a result table. `quiet=True` on `run()`
silences all of it; the SDK never prints anywhere else.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from .session import Result
from .telemetry import Telemetry


class UI:
    def __init__(self, quiet: bool = False) -> None:
        self.console = Console(quiet=quiet, highlight=False)
        self.quiet = quiet

    @property
    def interactive(self) -> bool:
        return not self.quiet and sys.stdin.isatty() and self.console.is_terminal

    # -- before the run

    def note(self, message: str, *, style: str = "dim") -> None:
        self.console.print(Text(message, style=style))

    def warn(self, message: str) -> None:
        self.console.print(
            Panel(message, border_style="yellow", title="notice", title_align="left")
        )

    def confirm(self, question: str, *, default: bool = True) -> bool:
        if not self.interactive:
            return default
        return Confirm.ask(question, default=default, console=self.console)

    def session_started(
        self,
        *,
        challenge_name: str,
        session_id: str,
        attempts_used: int,
        attempts_allowed: int | None,
        expires_at_ms: int,
        page_url: str,
        report_link: str | None,
    ) -> None:
        attempt = f"attempt {attempts_used + 1}"
        if attempts_allowed is not None:
            attempt += f" of {attempts_allowed}"
        minutes = max(0, round((expires_at_ms / 1000 - time.time()) / 60))
        body = Text()
        body.append(f"{challenge_name}\n", style="bold")
        body.append(f"{attempt}  ·  session {session_id}  ·  connect within {minutes} min\n\n")
        body.append("follow it live:  ", style="dim")
        body.append(page_url, style="link " + page_url)
        if report_link:
            body.append("\n\nwhen you are done, submit your report:  ", style="dim")
            body.append(report_link, style="bold link " + report_link)
        self.console.print(Panel(body, border_style="cyan", title="session", title_align="left"))

    # -- after the run

    def result(self, result: Result | None, reason: str | None, telemetry: Telemetry) -> None:
        if result is None:
            self.console.print(Panel(f"ended unscored: {reason}", border_style="yellow"))
        else:
            colour = {"perfect": "green", "partial": "green", "failed": "red"}.get(
                result.goal_tier, "white"
            )
            head = Text()
            head.append(result.goal_tier.upper(), style=f"bold {colour}")
            head.append(f"   {result.trans_error:.2f} m from the goal   {result.nav_steps:,} steps")
            if result.attempts_allowed is not None:
                head.append(
                    f"   ·   attempt {result.attempts_used} of {result.attempts_allowed}",
                    style="dim",
                )
            self.console.print(Panel(head, border_style=colour, title="result", title_align="left"))

        summary = telemetry.summary()
        if summary["steps"]:
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="dim")
            table.add_column(justify="right")
            table.add_row("round trips", f"{summary['steps']:,}")
            table.add_row("median round trip", _ms(summary["rtt_ms_p50"]))
            table.add_row("  server", _ms(summary["server_ms_p50"]))
            table.add_row("  network", _ms(summary["network_ms_p50"]))
            table.add_row("your code, per step", _ms(summary["think_ms_p50"]))
            if result is not None and result.think.get("mean_ms") is not None:
                table.add_row("  as the server measured it", _ms(result.think["mean_ms"]))
            self.console.print(table)

    def report_reminder(self, link: str) -> None:
        body = Text("Your run is recorded. Now submit your report:\n\n", style="bold")
        body.append(link, style="bold link " + link)
        self.console.print(
            Panel(body, border_style="magenta", title="one more thing", title_align="left")
        )

    def error(self, message: str) -> None:
        self.console.print(Panel(message, border_style="red", title="error", title_align="left"))


def _ms(value: Any) -> str:
    return "–" if value is None else f"{value:.1f} ms"
