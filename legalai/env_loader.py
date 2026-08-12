"""Minimal .env loader (no third-party dependency).

The project's `.env` was previously only read by docker-compose, so running
`python analyze_results.py` directly would not see anything defined there --
including an API key. Importing this module fixes that.

Existing environment variables always win, so `$env:FOO = "x"` in PowerShell
still overrides the file.
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"

_loaded = False


def load_env(path: Path = ENV_FILE, override: bool = False) -> dict:
    """Read KEY=VALUE lines from a .env file into os.environ.

    Args:
        path: file to read; missing file is not an error.
        override: replace variables already present in the environment.

    Returns:
        The keys that were applied.
    """
    global _loaded
    applied = {}

    if not path.exists():
        _loaded = True
        return applied

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        # Strip matching quotes, and inline comments on unquoted values.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()

        if not key:
            continue
        if key in os.environ and not override:
            continue

        os.environ[key] = value
        applied[key] = value

    _loaded = True
    return applied


# Load on import so `import env_loader` is enough.
load_env()
