"""Shared helper: make sure Ollama is installed, running, and has the
embedding model - so setup.py/test.py/run.py never fail partway through
because Ollama wasn't ready. Embeddings always run on local Ollama regardless
of GENERATION_PROVIDER (see config.py), so this is needed even though
generation itself is local_peft, not Ollama.

Auto-install only targets Linux (a Vast.ai rental). On Windows/Mac this just
checks and tells you what to do - install from https://ollama.com/download.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request

OLLAMA_API = "http://localhost:11434"


def _reachable() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_API}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def ensure_ollama_ready(embedding_model: str = "nomic-embed-text") -> None:
    """Install (if needed), start (if needed), and pull the embedding model."""
    if shutil.which("ollama") is None:
        if platform.system() != "Linux":
            raise SystemExit(
                "Ollama is not installed, and auto-install here only targets Linux "
                "(a Vast.ai rental). Install it yourself: https://ollama.com/download"
            )
        print("[ollama] not found - installing (official Linux install script)...")
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
        print("[ollama] installed.")

    if _reachable():
        print("[ollama] already running.")
    else:
        print("[ollama] starting server in the background...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(30):
            if _reachable():
                break
            time.sleep(1)
        else:
            raise SystemExit(
                "[ollama] did not come up within 30s of starting. "
                "Check it manually with: ollama serve"
            )
        print("[ollama] running.")

    listing = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    if embedding_model in listing.stdout:
        print(f"[ollama] {embedding_model} already pulled.")
    else:
        print(f"[ollama] pulling {embedding_model}...")
        subprocess.run(["ollama", "pull", embedding_model], check=True)


if __name__ == "__main__":
    ensure_ollama_ready()
