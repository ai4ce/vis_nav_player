from __future__ import annotations


class SimError(RuntimeError):
    """Anything the server refused, plus local protocol violations.

    ``code`` is one of ``protocol.ErrorCode`` when it came from the gateway, ``None`` when
    it was raised locally. ``status`` is the HTTP status when an HTTP request or the
    WebSocket upgrade was refused.
    """

    def __init__(self, message: str, *, code: str | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class SessionClosed(SimError):
    """The session is over: scored, aborted, timed out, or the connection dropped."""


class ConfigError(SimError):
    """A required setting (API key, server) was not given anywhere we look for it."""


#: What each HTTP rejection means, in terms a student can act on. Shared by the REST
#: client and the WebSocket upgrade, which are refused by the same checks.
HTTP_HELP: dict[int, str] = {
    401: (
        "the server did not recognise your credentials. A session token is single-use: press "
        "Start on the challenge page for a new one. An API key is the whole string shown on "
        "the course site, and it is not your email."
    ),
    403: (
        "your key is not allowed here. Either the challenge belongs to another course, "
        "its deadline has passed, it is no longer accepting runs, or the term has ended."
    ),
    404: "not found. Check the challenge id on the challenge page.",
    409: (
        "your team already has a session running -- finish or abort it first, since only "
        "one runs at a time. This token stays valid until it expires."
    ),
    429: (
        "you have used all your attempts for this challenge. Your best attempt is the one "
        "that counts, so nothing is lost. Ask a TA if the cap should be raised."
    ),
    503: (
        "no simulation worker is available right now. This is our problem, not yours -- "
        "try again shortly, and tell a TA if it persists."
    ),
}


def http_error(status: int, detail: str | None = None) -> SimError:
    text = HTTP_HELP.get(status, "the server refused the request.")
    if detail:
        text = f"{text} ({detail})"
    return SimError(f"HTTP {status}: {text}", status=status)
