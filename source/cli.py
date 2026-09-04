"""Command-line arguments shared by every agent script."""

from __future__ import annotations

import argparse
from pathlib import Path

from vis_nav_sdk import Client
from vis_nav_sdk.config import resolve_server, resolve_session_token, session_id_of


def parser(description: str, *, needs_api_key: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "session",
        nargs="?",
        default=None,
        help="session token from Start on the challenge page (or $VIS_NAV_SESSION)",
    )
    p.add_argument(
        "--server", default=None, help="API base URL (or $VIS_NAV_SERVER; default: course server)"
    )
    if needs_api_key:
        p.add_argument(
            "--api-key",
            default=None,
            help="your API key (or $VIS_NAV_API_KEY), for downloading the exploration data",
        )
    p.add_argument("--no-check", action="store_true", help="skip the pre-flight check")
    return p


def token(args: argparse.Namespace) -> str:
    return resolve_session_token(args.session)


def exploration_data(args: argparse.Namespace, data_dir: str | None) -> tuple[str, Path]:
    """``(challenge_id, dataset directory)`` for the session's challenge, downloading the
    dataset on first use. Needs the API key; the token alone does not identify you."""
    client = Client(args.api_key, server=resolve_server(args.server))
    challenge = client.session(session_id_of(token(args)))["challenge_id"]
    if data_dir:
        return challenge, Path(data_dir)
    print(f"fetching exploration data for {challenge}...")
    path = client.download_exploration_data(challenge, "data")
    print(f"  {path}")
    return challenge, path
