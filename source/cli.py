"""Command-line arguments shared by every agent script, and the ``.env`` they fall back on."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vis_nav_sdk import Client, SimError

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read ``KEY=value`` lines from ``.env`` into the environment, without overriding what is
    already set. Blank lines and ``#`` comments are skipped; values may be quoted. This is how
    the key and the challenge id are kept out of shell configuration and shell history: two
    lines in a file the repository ignores."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def parser(description: str) -> argparse.ArgumentParser:
    load_dotenv()
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--challenge",
        default=os.environ.get("VIS_NAV_CHALLENGE"),
        help="challenge id from the course site (or VIS_NAV_CHALLENGE in .env)",
    )
    p.add_argument("--api-key", default=None, help="your API key (or VIS_NAV_API_KEY in .env)")
    p.add_argument(
        "--server", default=None, help="API base URL (or $VIS_NAV_SERVER; default: course server)"
    )
    p.add_argument("--yes", action="store_true", help="start without asking")
    p.add_argument(
        "--no-browser", action="store_true", help="do not open the run's page on the site"
    )
    p.add_argument("--no-check", action="store_true", help="skip the pre-flight check")
    return p


def challenge(args: argparse.Namespace) -> str:
    if not args.challenge:
        raise SystemExit("--challenge (or VIS_NAV_CHALLENGE in .env) is required")
    return args.challenge


def run_options(args: argparse.Namespace) -> dict:
    return {
        "api_key": args.api_key,
        "server": args.server,
        "viewer": True,
        "check": not args.no_check,
        "confirm": False if args.yes else None,
        "browser": False if args.no_browser else None,
    }


def exploration_data(args: argparse.Namespace, data_dir: str | None) -> Path:
    """The dataset directory for the challenge, downloading it on first use."""
    if data_dir:
        return Path(data_dir)
    challenge_id = challenge(args)
    try:
        client = Client(args.api_key, server=args.server)
        path = client.download_exploration_data(challenge_id, "data")
    except SimError as exc:
        raise SystemExit(f"could not fetch the exploration data: {exc}") from None
    print(f"exploration data: {path}")
    return path
