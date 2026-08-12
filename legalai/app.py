"""One-command launcher for the Legal AI backend and frontend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"


def resolve_npm_command() -> str:
    """Resolve npm executable path in a Windows-safe way."""
    candidates = ["npm"]
    if os.name == "nt":
        # On Windows, npm is typically a cmd shim and may fail if invoked as bare "npm".
        candidates = ["npm.cmd", "npm.exe", "npm"]

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise RuntimeError("Required command not found: npm")


def run_blocking(command: list[str], cwd: Path):
    """Run a command and fail fast if it exits with an error."""
    result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}"
        )


def ensure_frontend_dependencies(skip_install: bool, npm_command: str):
    """Install frontend dependencies on first run if node_modules is missing."""
    if skip_install:
        return

    node_modules = FRONTEND_DIR / "node_modules"
    if node_modules.exists():
        return

    print("[launcher] Installing frontend dependencies...")
    run_blocking([npm_command, "install"], cwd=FRONTEND_DIR)


def build_frontend(npm_command: str):
    """Build frontend assets for production/static serving."""
    print("[launcher] Building frontend assets...")
    run_blocking([npm_command, "run", "build"], cwd=FRONTEND_DIR)


def start_backend(
    host: str,
    port: int,
    reload_enabled: bool,
    serve_static: bool = False,
    workers: int = 1,
) -> subprocess.Popen:
    """Start FastAPI backend through uvicorn in a child process."""
    env = os.environ.copy()
    if serve_static:
        env["LEGALAI_SERVE_STATIC"] = "1"

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if reload_enabled:
        command.append("--reload")
    elif workers > 1:
        command.extend(["--workers", str(workers)])

    print(f"[launcher] Starting backend on http://{host}:{port}")
    return subprocess.Popen(command, cwd=str(ROOT_DIR), env=env)


def start_frontend(
    npm_command: str,
    backend_host: str,
    backend_port: int,
    frontend_host: str,
    frontend_port: int,
) -> subprocess.Popen:
    """Start Vite frontend in a child process."""
    env = os.environ.copy()
    env.setdefault("VITE_API_BASE_URL", f"http://{backend_host}:{backend_port}")

    command = [
        npm_command,
        "run",
        "dev",
        "--",
        "--host",
        frontend_host,
        "--port",
        str(frontend_port),
    ]

    print(f"[launcher] Starting frontend on http://{frontend_host}:{frontend_port}")
    return subprocess.Popen(command, cwd=str(FRONTEND_DIR), env=env)


def terminate_process(process: subprocess.Popen, name: str):
    """Terminate a process gracefully, then force kill if needed."""
    if process.poll() is not None:
        return

    print(f"[launcher] Stopping {name}...")
    process.terminate()

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def monitor_processes(
    backend_process: subprocess.Popen | None,
    frontend_process: subprocess.Popen | None,
) -> int:
    """Keep launcher alive and stop sibling process if one exits unexpectedly."""
    while True:
        if backend_process and backend_process.poll() is not None:
            code = backend_process.returncode or 0
            if frontend_process:
                terminate_process(frontend_process, "frontend")
            return code

        if frontend_process and frontend_process.poll() is not None:
            code = frontend_process.returncode or 0
            if backend_process:
                terminate_process(backend_process, "backend")
            return code

        time.sleep(0.5)


def parse_args() -> argparse.Namespace:
    """Parse launcher command line options."""
    parser = argparse.ArgumentParser(
        description="Run Legal AI backend and frontend with one command.",
    )

    parser.add_argument("--backend-only", action="store_true", help="Run only backend")
    parser.add_argument("--frontend-only", action="store_true", help="Run only frontend")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Run production mode (backend serves built frontend assets)",
    )
    parser.add_argument(
        "--build-frontend",
        action="store_true",
        help="Build frontend before launching (recommended for production mode)",
    )
    parser.add_argument(
        "--skip-npm-install",
        action="store_true",
        help="Skip automatic npm install when node_modules is missing",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable backend auto-reload",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override chat model (sets OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override embedding model (sets OLLAMA_EMBEDDING_MODEL)",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Override Ollama base URL (sets OLLAMA_BASE_URL)",
    )
    parser.add_argument("--backend-host", default="127.0.0.1", help="Backend host")
    parser.add_argument("--backend-port", type=int, default=8000, help="Backend port")
    parser.add_argument("--frontend-host", default="127.0.0.1", help="Frontend host")
    parser.add_argument("--frontend-port", type=int, default=5173, help="Frontend port")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of backend workers in non-reload mode",
    )

    args = parser.parse_args()

    if args.backend_only and args.frontend_only:
        parser.error("Choose only one of --backend-only or --frontend-only")

    if args.production and args.frontend_only:
        parser.error("--production cannot be combined with --frontend-only")

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    return args


def main() -> int:
    """Entry point for one-command startup."""
    args = parse_args()
    npm_command = ""

    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model

    if args.embedding_model:
        os.environ["OLLAMA_EMBEDDING_MODEL"] = args.embedding_model

    if args.ollama_base_url:
        os.environ["OLLAMA_BASE_URL"] = args.ollama_base_url

    should_run_frontend_process = (not args.backend_only) and (not args.production)

    if should_run_frontend_process or args.build_frontend or args.production:
        npm_command = resolve_npm_command()

    if not FRONTEND_DIR.exists():
        raise RuntimeError(f"Frontend directory not found: {FRONTEND_DIR}")

    if not args.frontend_only:
        # Verify Python dependencies include uvicorn before launching.
        try:
            __import__("uvicorn")
        except ImportError as exc:
            raise RuntimeError(
                "uvicorn is not installed. Run: pip install -r requirements.txt"
            ) from exc

    if should_run_frontend_process or args.build_frontend or args.production:
        ensure_frontend_dependencies(skip_install=args.skip_npm_install, npm_command=npm_command)

    if args.build_frontend or args.production:
        build_frontend(npm_command=npm_command)

    backend_process = None
    frontend_process = None

    try:
        if args.model:
            print(f"[launcher] Chat model: {args.model}")
        if args.embedding_model:
            print(f"[launcher] Embedding model: {args.embedding_model}")
        if args.ollama_base_url:
            print(f"[launcher] Ollama URL: {args.ollama_base_url}")

        if not args.frontend_only:
            backend_process = start_backend(
                host=args.backend_host,
                port=args.backend_port,
                reload_enabled=not args.no_reload,
                serve_static=args.production,
                workers=args.workers,
            )

        if should_run_frontend_process:
            frontend_process = start_frontend(
                npm_command=npm_command,
                backend_host=args.backend_host,
                backend_port=args.backend_port,
                frontend_host=args.frontend_host,
                frontend_port=args.frontend_port,
            )

        if args.production:
            print(f"[launcher] Production mode running at http://{args.backend_host}:{args.backend_port}")
        elif args.backend_only:
            print(f"[launcher] Backend running at http://{args.backend_host}:{args.backend_port}")
        elif args.frontend_only:
            print(f"[launcher] Frontend running at http://{args.frontend_host}:{args.frontend_port}")
        else:
            print("[launcher] Legal AI is running:")
            print(f"  - Frontend: http://{args.frontend_host}:{args.frontend_port}")
            print(f"  - Backend:  http://{args.backend_host}:{args.backend_port}")

        print("[launcher] Press Ctrl+C to stop.")
        return monitor_processes(backend_process, frontend_process)

    except KeyboardInterrupt:
        print("\n[launcher] Ctrl+C received, shutting down...")
        return 0

    finally:
        if frontend_process:
            terminate_process(frontend_process, "frontend")
        if backend_process:
            terminate_process(backend_process, "backend")


if __name__ == "__main__":
    raise SystemExit(main())
