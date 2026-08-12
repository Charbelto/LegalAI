"""Backfill eval_history / learning_signal into adapters trained before that field existed.

    python finetune/backfill_eval_history.py
    python finetune/backfill_eval_history.py --log finetune/train_run.log

Why this exists
---------------
train_qlora.py records a per-epoch eval curve and a `learning_signal` verdict in
each adapter's training_meta.json, so validate_adapters.py can check whether
held-out loss actually improved. That check was added while a training run was
already in flight; Python had the previous version of the module loaded, so the
adapters produced by that run carry final eval metrics but no curve.

Rather than retrain (hours, for a metadata field), this reconstructs the curve
from the training log, which contains every eval line and the role banners that
delimit them. It writes only the two missing keys and never overwrites values
that already exist, so running it against adapters trained by the current code is
a no-op.

The reconstruction is marked `"source": "backfilled_from_log"` in the metadata,
because a value recovered from a log is not the same provenance as one recorded by
the trainer, and anything reading it should be able to tell.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FINETUNE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FINETUNE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: E402

# Role banner, or an eval line. Scanned in order so eval lines attach to whichever
# role was most recently announced.
_SCAN = re.compile(
    r"\[train\] role=(?P<role>\w+)\s+base="
    r"|'eval_loss': '(?P<loss>[0-9.]+)'"
    r"(?:.*?'eval_mean_token_accuracy': '(?P<acc>[0-9.]+)')?"
    r".*?'epoch': '(?P<epoch>[0-9.]+)'"
)


def parse_log(log_path: Path) -> dict:
    """Return {role: [{epoch, eval_loss, eval_mean_token_accuracy}, ...]}."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    curves: dict = {}
    role = None
    for match in _SCAN.finditer(text):
        if match.group("role"):
            role = match.group("role")
            curves.setdefault(role, [])
            continue
        if role is None:
            continue
        curves[role].append(
            {
                "epoch": round(float(match.group("epoch")), 3),
                "eval_loss": float(match.group("loss")),
                "eval_mean_token_accuracy": (
                    float(match.group("acc")) if match.group("acc") else None
                ),
            }
        )
    return curves


def learning_signal(history: list) -> dict | None:
    """Same computation train_qlora.py performs, so the two agree."""
    if len(history) < 2:
        return None
    first = history[0]["eval_loss"]
    best = min(point["eval_loss"] for point in history)
    return {
        "first_eval_loss": round(first, 4),
        "best_eval_loss": round(best, 4),
        "final_eval_loss": round(history[-1]["eval_loss"], 4),
        "improvement": round(first - best, 4),
        "improved": bool(first - best > 0.01),
        "best_epoch": min(history, key=lambda p: p["eval_loss"])["epoch"],
        "source": "backfilled_from_log",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log", default=str(FINETUNE_DIR / "train_run.log"))
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"No training log at {log_path}")

    curves = parse_log(log_path)
    if not curves:
        raise SystemExit(f"No eval lines found in {log_path}")

    updated, skipped, missing = [], [], []
    for role in config.LOCAL_PEFT_ROLES:
        meta_path = ROOT_DIR / config.LOCAL_ADAPTER_DIR / role / "training_meta.json"
        if not meta_path.exists():
            missing.append(role)
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("eval_history"):
            skipped.append(role)
            continue
        history = curves.get(role) or []
        if not history:
            print(f"[backfill] {role}: no eval lines in the log for this role")
            continue

        meta["eval_history"] = history
        meta["learning_signal"] = learning_signal(history)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        signal = meta["learning_signal"] or {}
        verdict = "improved" if signal.get("improved") else "DID NOT IMPROVE"
        print(
            f"[backfill] {role}: {len(history)} eval points, "
            f"{signal.get('first_eval_loss')} -> best {signal.get('best_eval_loss')} "
            f"(epoch {signal.get('best_epoch')}) -> {verdict}"
        )
        if signal and not signal.get("improved"):
            print(
                f"[backfill] WARNING {role} shows no learning. Check "
                f"finetune/data/manifest.json token_lengths - examples over the "
                f"training sequence limit are right-truncated, removing the target."
            )
        updated.append(role)

    if skipped:
        print(f"[backfill] already had a curve, left untouched: {skipped}")
    if missing:
        print(f"[backfill] not trained yet: {missing}")
    print(f"[backfill] updated {len(updated)} adapter(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
