"""One-time setup for running the benchmark on a fresh Vast.ai box.

    python setup.py

Installs the Python dependencies (including the CUDA build of torch),
installs/starts Ollama and pulls the embedding model, creates .env from
.env.example if you don't have one yet, and runs the test suite as a final
sanity check. Safe to re-run - every step either no-ops or reinstalls cleanly.

After this: python test.py, then python run.py. See VASTAI_DEPLOY.md for the
full walkthrough and what each setting means.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"[setup] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def install_python_deps() -> None:
    # torch FIRST from PyTorch's own CUDA index, or pip silently grabs a
    # CPU-only wheel from PyPI and every model runs on the CPU.
    run([sys.executable, "-m", "pip", "install", "torch",
         "--index-url", "https://download.pytorch.org/whl/cu126"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements-finetune.txt"])


def ensure_env_file() -> None:
    env_file = ROOT / ".env"
    example = ROOT / ".env.example"
    if env_file.exists():
        print("[setup] .env already exists, leaving it alone.")
        return
    if not example.exists():
        print("[setup] WARNING: no .env.example to copy from; create .env yourself.")
        return
    shutil.copy(example, env_file)

    # Paired with run.py's/test.py's default of --concurrency 4: this is the
    # Vast.ai-tailored setup script, so it writes the settings that pairing
    # assumes rather than leaving them commented out as they are in
    # .env.example (which targets every environment, including an 8GB laptop
    # where this would be wrong). run.py's check_vram_fits() verifies this
    # actually fits before the full benchmark starts - see VASTAI_DEPLOY.md
    # for the VRAM math, which is genuinely tight on a 24GB card.
    with env_file.open("a", encoding="utf-8") as f:
        f.write(
            "\n# --- Set by setup.py: pairs with run.py/test.py's default "
            "--concurrency 4 ---\n"
            "LEGALAI_LOAD_IN_4BIT=0\n"
            "LEGALAI_GENERAL_POOL_SIZE=2\n"
        )

    print("[setup] Created .env from .env.example (+ LEGALAI_LOAD_IN_4BIT=0, "
          "LEGALAI_GENERAL_POOL_SIZE=2 for run.py's default concurrency).")
    print("[setup] >>> Edit .env and set DEEPSEEK_API_KEY before running the full benchmark. <<<")


def main() -> None:
    print("=== Legal AI: one-time Vast.ai setup ===")
    install_python_deps()
    ensure_env_file()

    from ollama_setup import ensure_ollama_ready

    ensure_ollama_ready()

    print("\n[setup] Checking LoRA adapters are present...")
    missing = [
        role for role in ("legal", "news", "general_qa")
        if not (ROOT / "adapters" / role / "adapter_config.json").exists()
    ]
    if missing:
        print(f"[setup] WARNING: no adapter for {missing}. The 'peft' arm needs these; "
              "they are normally already committed to the repo.")
    else:
        print("[setup] Adapters present for legal, news, general_qa.")

    print("\n[setup] Running the test suite...")
    run([sys.executable, "-m", "pytest", "tests", "-q"])

    print("\n=== Setup done ===")
    print("Next:")
    print("  python test.py     (a few minutes - sanity check before the full run)")
    print("  python run.py      (the full benchmark - see VASTAI_DEPLOY.md for a time estimate)")


if __name__ == "__main__":
    main()
