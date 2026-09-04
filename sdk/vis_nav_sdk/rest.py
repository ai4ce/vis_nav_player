"""
The HTTP side of the course API: who am I, what is this challenge, how many attempts do I
have left, give me the exploration data. Standard library only.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from .config import resolve_api_key, resolve_server
from .errors import SimError, http_error


class Client:
    """A key and a server, and the calls that need them.

    client = Client(api_key=KEY)
    print(client.me()["email"])
    data = client.download_exploration_data(CHALLENGE)
    print(client.quota(CHALLENGE))
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        server: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.server = resolve_server(server)
        self.api_key = resolve_api_key(api_key)
        self.timeout = timeout

    # -- account and challenge

    def me(self) -> dict[str, Any]:
        """Who this key belongs to: ``email``, ``name``, ``course``, ``group``."""
        return self._get("/me")

    def challenge(self, challenge_id: str) -> dict[str, Any]:
        return self._get(f"/challenges/{quote(challenge_id, safe='')}")

    def challenges(self, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        """Public challenges plus those of the caller's own course."""
        return self._get("/challenges", include_inactive=include_inactive)

    def quota(self, challenge_id: str) -> dict[str, Any]:
        """``attempts_used``, ``attempts_allowed`` (``None`` = unlimited), ``max_steps``,
        ``running`` (the team's live sessions), ``reservation`` (a session you started but
        have not connected to), ``final_submission_link``."""
        return self._get(f"/sim/{quote(challenge_id, safe='')}/quota")

    def start_session(self, challenge_id: str) -> dict[str, Any]:
        """Reserve a session: ``session_id``, a one-time ``token`` to redeem within
        ``expires_at``, ``page_url`` to follow it on the site, ``final_submission_link``.
        Nothing is spent until the token is redeemed. Replaces any reservation you already
        hold on this challenge."""
        return self._post(f"/sim/{quote(challenge_id, safe='')}/sessions")

    def sessions(self, challenge_id: str) -> list[dict[str, Any]]:
        """This student's past sessions on a challenge, newest first."""
        return self._get(f"/sim/{quote(challenge_id, safe='')}/sessions")

    def leaderboard(self, challenge_id: str) -> list[dict[str, Any]]:
        return self._get(f"/leaderboard/{quote(challenge_id, safe='')}")

    # -- exploration data

    def download_exploration_data(
        self, challenge_id: str, dest: str | Path = "data", *, force: bool = False
    ) -> Path:
        """Fetch and unpack the exploration dataset into ``dest/<challenge_id>/``.

        Returns that directory. Skipped if it already holds a dataset unless ``force``.
        Layout: ``target.jpg`` and ``traj_<i>/{<k>.jpg, data_info.json}``.
        """
        target = Path(dest) / challenge_id
        if not force and (target / "target.jpg").exists():
            return target

        url = f"{self.server}/challenges/{quote(challenge_id, safe='')}/exploration-data"
        request = urllib.request.Request(url, headers=self._headers())
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as handle:
            temp = Path(handle.name)
        try:
            try:
                with (
                    urllib.request.urlopen(request, timeout=self.timeout) as response,
                    temp.open("wb") as out,
                ):
                    shutil.copyfileobj(response, out, length=1 << 20)
            except urllib.error.HTTPError as exc:
                raise http_error(exc.code, _detail(exc)) from exc
            except urllib.error.URLError as exc:
                raise SimError(f"could not reach {self.server}: {exc.reason}") from exc

            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            with zipfile.ZipFile(temp) as archive:
                for member in archive.infolist():
                    if Path(member.filename).is_absolute() or ".." in member.filename:
                        raise SimError(f"refusing to unpack {member.filename!r}")
                archive.extractall(target)
        finally:
            temp.unlink(missing_ok=True)
        return target

    def session(self, session_id: str) -> dict[str, Any]:
        """One of your sessions, live: ``status`` (``pending`` / ``running`` / ``scored`` /
        ...), ``steps_used``, ``result`` once scored, and a ``live`` block while running."""
        return self._get(f"/sim/sessions/{quote(session_id, safe='')}")

    # -- plumbing

    def _headers(self) -> dict[str, str]:
        from . import __version__

        return {
            "x-api-key": self.api_key,
            "x-sdk-version": __version__,
            "accept": "application/json",
        }

    def _get(self, path: str, **params: Any) -> Any:
        url = self.server + path
        query = {k: v for k, v in params.items() if v is not None and v is not False}
        if query:
            url += "?" + urlencode(query)
        return self._send(urllib.request.Request(url, headers=self._headers()))

    def _post(self, path: str) -> Any:
        request = urllib.request.Request(self.server + path, method="POST", headers=self._headers())
        return self._send(request)

    def _send(self, request: urllib.request.Request) -> Any:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise http_error(exc.code, _detail(exc)) from exc
        except urllib.error.URLError as exc:
            raise SimError(f"could not reach {self.server}: {exc.reason}") from exc


def _detail(exc: urllib.error.HTTPError) -> str | None:
    try:
        body = json.loads(exc.read())
    except Exception:  # noqa: BLE001 - any non-JSON body is simply not a detail
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail if isinstance(detail, str) else None
