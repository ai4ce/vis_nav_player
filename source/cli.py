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
    p.add_argument(
        "--local",
        nargs="?",
        const=os.environ.get("VIS_NAV_SEED", "7"),
        metavar="SEED",
        help=(
            "run on the local simulator instead of the server, in the maze this seed "
            "generates (default 7; the same seed is the same maze on every machine). "
            "No attempt is spent. Needs `uv sync --extra local` and the texture pack."
        ),
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
    """What to run on: ``local:<seed>`` with ``--local``, else the challenge id."""
    if args.local is not None:
        return f"local:{int(args.local)}"
    if not args.challenge:
        raise SystemExit("--challenge (or VIS_NAV_CHALLENGE in .env) is required, or --local")
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
    """The dataset directory for the challenge: downloaded on first use, or, with
    ``--local``, recorded on first use by the local simulator in the same format."""
    if data_dir:
        return Path(data_dir)
    challenge_id = challenge(args)
    if args.local is not None:
        return record_exploration_data(int(args.local))
    try:
        client = Client(args.api_key, server=args.server)
        path = client.download_exploration_data(challenge_id, "data")
    except SimError as exc:
        raise SystemExit(f"could not fetch the exploration data: {exc}") from None
    print(f"exploration data: {path}")
    return path


def record_exploration_data(seed: int, dest: str | Path = "data") -> Path:
    """``data/local-<seed>/``: what a challenge on this maze would hand out -- three drives
    through it, a frame every five ticks, ``target.jpg`` -- recorded here the first time."""
    path = Path(dest) / f"local-{seed}"
    if (path / "target.jpg").exists():
        return path
    try:
        import vis_nav_sim as sim
    except ImportError:
        raise SystemExit(
            "--local needs the vis-nav-sim package: run `uv sync --extra local`"
        ) from None
    try:
        textures = sim.Textures.find()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    print(f"recording exploration data for local maze {seed} into {path} ...", flush=True)
    world = sim.Simulator(textures, seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed)
    summary = world.record(path, routes=3, seed=seed, capture_every=5)
    print(f"exploration data: {path} ({summary['frames']} frames on {summary['routes']} routes)")
    return path
