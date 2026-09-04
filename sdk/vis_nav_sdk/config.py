"""
Where the SDK looks for its two settings, in order: explicit argument, environment
variable, built-in default. Only the server has a default.
"""

from __future__ import annotations

import os

from .errors import ConfigError

DEFAULT_SERVER = "https://visual-navigation-challenge-api.ai4ce.dev"
ENV_SERVER = "VIS_NAV_SERVER"
ENV_API_KEY = "VIS_NAV_API_KEY"

TOKEN_PREFIX = "vns_"


def resolve_server(server: str | None = None) -> str:
    value = server or os.environ.get(ENV_SERVER) or DEFAULT_SERVER
    return value.strip().rstrip("/")


def check_session_token(token: str) -> str:
    value = token.strip()
    if not value.startswith(TOKEN_PREFIX) or value.count("_") < 2:
        raise ConfigError(
            f"{value[:12]!r}... is not a session token. Tokens look like vns_<session>_<secret> "
            "and come from starting a session -- this is not your API key."
        )
    return value


def session_id_of(token: str) -> str:
    return token[len(TOKEN_PREFIX) :].split("_", 1)[0]


def resolve_api_key(api_key: str | None = None) -> str:
    value = (api_key or os.environ.get(ENV_API_KEY) or "").strip()
    if not value:
        raise ConfigError(
            "no API key. Pass api_key=... or set the environment variable "
            f"{ENV_API_KEY}. Your key is on the course site; treat it like a password "
            "and keep it out of code you commit."
        )
    return value
