"""Measure each BASE model's held-out loss, with no adapter attached.

    python finetune/base_eval_loss.py

Why this is needed
------------------
train_qlora.py evaluates after each epoch, so the earliest eval point already
reflects a full epoch of training. The `learning_signal` verdict derived from
that curve therefore answers "did it keep improving after epoch 1?" and cannot
answer "did it learn anything at all?" - the base -> epoch-1 gain is invisible to
it.

That distinction decides how to read a flat curve. If the base model scores far
worse than epoch 1, the adapter learned substantially and then stopped; if the
base scores about the same, the adapter never learned. Those call for opposite
decisions, so the number is worth measuring rather than assuming.

Loads one model at a time and frees it before the next, so this needs the VRAM of
a single expert.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FINETUNE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FINETUNE_DIR.parent
DATA_DIR = FINETUNE_DIR / "data"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: E402

REPORT_PATH = FINETUNE_DIR / "base_eval_loss.json"
ROLE_TO_DOMAIN = {"legal": "legal", "news": "news", "general_qa": "general"}


def main():
    import torch
    import transformers

    # Reuse the trainer's own rendering so the comparison is apples to apples:
    # a different prompt format would change the loss for reasons unrelated to
    # the weights.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_tq", FINETUNE_DIR / "train_qlora.py")
    trainer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer_module)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.LOCAL_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.LOCAL_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    results = {}
    for role, domain in ROLE_TO_DOMAIN.items():
        base_model_id = config.LOCAL_PEFT_ROLES[role]["base_model"]
        val_path = DATA_DIR / f"{domain}_val.jsonl"
        if not val_path.exists():
            print(f"[base-eval] no val set for {domain}; skipping")
            continue
        records = [json.loads(line) for line in open(val_path, encoding="utf-8") if line.strip()]

        print(f"[base-eval] {role}: loading {base_model_id} (no adapter)...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_id, trust_remote_code=config.LOCAL_TRUST_REMOTE_CODE
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quant_config,
            dtype=compute_dtype,
            device_map={"": 0},
            trust_remote_code=config.LOCAL_TRUST_REMOTE_CODE,
            attn_implementation=config.LOCAL_ATTN_IMPLEMENTATION,
        )
        model.eval()

        total_loss, total_tokens, correct, counted = 0.0, 0, 0, 0
        with torch.inference_mode():
            for record in records:
                text = trainer_module.render_example(record, tokenizer)
                ids = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                )["input_ids"].to(model.device)
                if ids.shape[-1] < 2:
                    continue
                out = model(input_ids=ids, labels=ids)
                # Token count weighting, so long and short examples contribute
                # proportionally - the same convention HF's Trainer uses.
                n_tokens = ids.shape[-1] - 1
                total_loss += float(out.loss) * n_tokens
                total_tokens += n_tokens
                predictions = out.logits[:, :-1].argmax(-1)
                targets = ids[:, 1:]
                correct += int((predictions == targets).sum())
                counted += int(targets.numel())

        results[role] = {
            "base_model": base_model_id,
            "val_examples": len(records),
            "base_eval_loss": round(total_loss / total_tokens, 4) if total_tokens else None,
            "base_mean_token_accuracy": round(correct / counted, 4) if counted else None,
        }
        print(
            f"[base-eval] {role}: base_eval_loss="
            f"{results[role]['base_eval_loss']} "
            f"acc={results[role]['base_mean_token_accuracy']}"
        )

        del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    # Compare against each adapter's epoch-1 and best values.
    print(f"\n{'role':11s} {'base':>8} {'ep1':>8} {'best':>8} | {'base->ep1':>10} {'ep1->best':>10}")
    for role, info in results.items():
        meta_path = ROOT_DIR / config.LOCAL_ADAPTER_DIR / role / "training_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        history = meta.get("eval_history") or []
        if not history:
            continue
        base = info["base_eval_loss"]
        ep1 = history[0]["eval_loss"]
        best = min(point["eval_loss"] for point in history)
        info["epoch1_eval_loss"] = ep1
        info["best_eval_loss"] = best
        info["base_to_epoch1_gain"] = round(base - ep1, 4)
        info["epoch1_to_best_gain"] = round(ep1 - best, 4)
        info["learned_from_base"] = bool(base - ep1 > 0.01)
        print(
            f"{role:11s} {base:8.3f} {ep1:8.3f} {best:8.3f} | "
            f"{base - ep1:+10.3f} {ep1 - best:+10.3f}"
        )

    REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[base-eval] written to {REPORT_PATH.name}")
    print(
        "[base-eval] A large positive base->ep1 gain means the adapter DID learn, and a "
        "flat ep1->best curve only means it stopped improving. A near-zero base->ep1 gain "
        "means it never learned at all."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
