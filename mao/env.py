"""Minimal .env loader (stdlib only). Loads KEY=VALUE lines from the repo-root
.env into os.environ without overriding variables already set in the shell."""

from __future__ import annotations

import os

_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def load_env(path: str = _ENV_PATH) -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


load_env()
