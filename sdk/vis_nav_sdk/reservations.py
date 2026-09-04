"""
Tokens this machine has reserved but not yet redeemed.

The server keeps only a hash of a token, so a reservation made here can be resumed only
if the token is remembered here. Stored under the user's cache directory, readable by
the user alone; entries disappear when redeemed or expired.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "vis-nav" / "reservations.json"


def _load() -> dict[str, Any]:
    try:
        data = json.loads(_path().read_text())
    except (OSError, ValueError):
        return {}
    now_ms = time.time() * 1000
    return {
        k: v for k, v in data.items() if isinstance(v, dict) and v.get("expires_at", 0) > now_ms
    }


def _save(data: dict[str, Any]) -> None:
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1))
        os.chmod(path, 0o600)
    except OSError:
        pass  # a cache; losing it costs a prompt, not a run


def remember(server: str, created: dict[str, Any]) -> None:
    data = _load()
    data[created["session_id"]] = {
        "server": server,
        "token": created["token"],
        "expires_at": created["expires_at"],
    }
    _save(data)


def recall(server: str, session_id: str) -> str | None:
    entry = _load().get(session_id)
    if entry is None or entry.get("server") != server:
        return None
    return entry.get("token")


def forget(session_id: str) -> None:
    data = _load()
    if data.pop(session_id, None) is not None:
        _save(data)
