"""QLoRA fine-tuning: one adapter per domain expert.

    python finetune/train_qlora.py --role legal
    python finetune/train_qlora.py --role all
    python finetune/train_qlora.py --role news --epochs 2 --rank 32

Reads finetune/data/<domain>_train.jsonl (built by prepare_datasets.py), trains a
LoRA adapter on the role's 4-bit-quantised base model, and writes it to
adapters/<role>/ where local_models.py looks for it.

Hyperparameters follow the pivot plan's Section 5 Step 2 defaults: r=16,
alpha=32, dropout=0.05, 4-bit NF4, lr 2e-4, 3 epochs, effective batch 8. They
are starting points, not tuned values - every one is a CLI flag, and whatever
was actually used is written into adapters/<role>/training_meta.json so the
paper's fine-tuning-protocol table is transcribed from the run rather than from
this docstring.

VRAM note: training loads ONE model at a time, so it has the whole card to
itself. That is a different constraint from inference, where all three models
must co-reside - see finetune/check_vram.py for that check.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

FINETUNE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FINETUNE_DIR.parent
DATA_DIR = FINETUNE_DIR / "data"

# Import the project's config so model ids and adapter paths have exactly one
# source of truth shared with the serving path.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import config  # noqa: E402

# role -> dataset file stem produced by prepare_datasets.py
ROLE_TO_DOMAIN = {"legal": "legal", "news": "news", "general_qa": "general"}

# Hyperparameters, and why they are these values rather than the plan's.
#
# The plan proposed r=16/alpha=32, lr 2e-4, 3 epochs, 1024 tokens. A first pass at
# 3 epochs / 512 tokens / 2e-4 produced adapters that reduced held-out loss
# substantially (46-72%) and yet generated DEGENERATE TEXT in the deployed graph:
# against a real 1400-token prompt the legal expert returned ungrammatical output
# with invented terms, where the same base model answered correctly and cited
# Annex III. Two things were wrong at once and both are corrected here:
#
#   * Over-strong adaptation. Held-out loss bottomed at epoch 1-2 and rose after,
#     mean answer length collapsed (161 -> 68 words on validation probes), and
#     fluency went with it - the signature of a small dataset overwriting general
#     capability. Hence 1 epoch at lr 5e-5 (4x lower) instead of 3 at 2e-4.
#   * Train/serve context mismatch. Training at 512 tokens while serving prompts
#     run 1373-2141 (experts) and ~4500 (aggregator) means the adapter was only
#     ever fitted on sequences 3-9x shorter than it is used at. 1024 covers the
#     bulk of real expert prompts and is already known to fit this GPU.
#
# Cost: ~178 optimiser steps per adapter, roughly 50 minutes each.
DEFAULTS = {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "learning_rate": 5e-5,
    "epochs": 1,
    "batch_size": 1,
    "grad_accum": 8,          # effective batch 8, sized to fit beside a 4-bit 3B
    "max_seq_length": 1024,
    "warmup_ratio": 0.03,
    "seed": 20260803,
}

# NOTE ON CONSISTENCY, which matters more than any individual value here:
# every adapter must be trained with the SAME budget. If one expert gets more
# epochs, longer sequences or more examples than another, then a topology
# difference is partly "which graph position happens to hold the best-trained
# expert" - a confound stacked on top of the model-position one the paper already
# discloses. Changing a default therefore means retraining ALL THREE, not just
# the ones not yet built. Hence --role all is the default.

# Attention projections plus the MLP: the plan specifies attention projections
# as the standard choice, and the MLP projections are included because they are
# where most LoRA-on-decoder recipes get their remaining headroom for very
# little extra memory. Names are the union across the three architectures
# (Llama/Qwen use q,k,v,o + gate,up,down; Phi-3 fuses them into qkv_proj and
# gate_up_proj), and peft silently ignores names a given model lacks.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "qkv_proj", "gate_up_proj",
]


def _require(module_names: List[str]):
    missing = []
    for name in module_names:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(
            "Missing training dependencies: " + ", ".join(missing) + "\n"
            "    pip install -r requirements-finetune.txt"
        )


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(
            f"No training data at {path}.\n"
            "Build it first:  python finetune/prepare_datasets.py"
        )
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"{path} is empty.")
    return records


def render_example(record: Dict[str, str], tokenizer) -> str:
    """Render one record through the target model's own chat template.

    Training in the same format the model is served in matters: an adapter
    trained on "### Instruction:" plaintext but served through the model's chat
    template is being asked to generalise across a format shift, which is a
    common and entirely avoidable reason a fine-tune appears not to have worked.
    """
    user_content = record.get("instruction", "").strip()
    extra = record.get("input", "").strip()
    if extra:
        user_content = f"{user_content}\n\n{extra}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": record.get("output", "").strip()},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"User: {user_content}\n\nAssistant: {record['output'].strip()}"


def _ensure_distinct_pad_token(tokenizer, base_model_id: str):
    """Guarantee pad_token_id != eos_token_id before SFT.

    Defensive hygiene, NOT a fix for the general_qa degeneration - that hypothesis
    was tested and refuted, and this docstring previously claimed otherwise.

    The reasoning was: SFT collators mask padding out of the labels by comparing
    against `pad_token_id`, so if pad and eos are the same token every genuine
    end-of-turn token is masked too and the model never learns to stop. Granite
    3.1 is exactly that case - `<|end_of_text|>` (id 0) is its eos, bos AND unk,
    and it has no other special token - while Llama 3.2 and Qwen2.5 ship distinct
    pad tokens. It fit the symptom precisely: only the Granite adapter ran to the
    full token cap and emitted invented non-words.

    It was wrong. Retraining with a distinct pad token produced an eval curve
    essentially identical to the colliding run ([3.457, 2.044, 1.739, 1.722] vs
    [3.48, 2.03, 1.737, 1.721]) and the adapter still degenerated. TRL evidently
    derives its label mask from the chat template rather than from pad-token
    identity, so the collision was never affecting the objective.

    Kept because a distinct pad token is correct regardless - some collators and
    some trainer configurations DO mask by token id, and relying on the two
    coinciding is fragile. An in-vocabulary token is adopted rather than a new
    one, since adding a token would require resizing the embedding matrix of a
    frozen 4-bit base, which LoRA does not cover and which would make the adapter
    non-portable. Pad identity is otherwise irrelevant: those positions are
    excluded by the attention mask.
    """
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        return  # already distinct - Llama and Qwen take this path

    vocab = tokenizer.get_vocab()
    # Prefer a real pad token if the model has one; otherwise any in-range token
    # that cannot appear in chat-formatted training text.
    for candidate in ("<|pad|>", "<pad>", "<fim_pad>", "<fim_prefix>", "<unk>"):
        token_id = vocab.get(candidate)
        if token_id is not None and token_id != tokenizer.eos_token_id:
            tokenizer.pad_token = candidate
            print(
                f"[train] {base_model_id}: pad_token collided with eos "
                f"({tokenizer.eos_token!r}); using {candidate!r} (id {token_id}) as pad "
                f"so end-of-turn tokens are not masked out of the labels."
            )
            return

    raise SystemExit(
        f"[train] {base_model_id}: pad_token equals eos_token "
        f"({tokenizer.eos_token!r}) and no distinct in-vocabulary candidate was "
        f"found. Training would mask every stop token out of the labels and "
        f"produce an adapter that never terminates. Add a pad token and resize "
        f"embeddings, or pick a different base model."
    )


def _total_steps(n_examples: int, args: argparse.Namespace) -> int:
    """Optimiser steps for the whole run, for sizing the eval interval."""
    per_epoch = max(1, n_examples // max(1, args.batch_size * args.grad_accum))
    return max(1, int(per_epoch * args.epochs))


def _existing_adapter_meta(role: str):
    """Return a finished adapter's training_meta.json, or None.

    Requires BOTH the adapter weights and the metadata: a directory holding only
    `checkpoints/` is a trainer that started and did not finish, and must not be
    mistaken for a completed adapter.
    """
    out_dir = ROOT_DIR / config.LOCAL_ADAPTER_DIR / role
    if not (out_dir / "adapter_config.json").exists():
        return None
    meta_path = out_dir / "training_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


# Parameters that must match across all three adapters for the topology
# comparison to be clean. Anything affecting how much or how hard an expert was
# trained belongs here; cosmetic fields do not.
_BUDGET_KEYS = (
    ("epochs", "epochs"),
    ("max_seq_length", "max_seq_length"),
    ("lora_rank", "rank"),
    ("lora_alpha", "alpha"),
    ("lora_dropout", "dropout"),
    ("learning_rate", "learning_rate"),
    ("effective_batch_size", None),   # derived; compared separately below
)


def _hyperparameter_mismatch(existing: Dict[str, Any], args: argparse.Namespace):
    """List (name, existing, requested) for any budget parameter that differs."""
    recorded = existing.get("hyperparameters", {}) or {}
    differences = []
    for meta_key, arg_name in _BUDGET_KEYS:
        if arg_name is None:
            continue
        old = recorded.get(meta_key)
        new = getattr(args, arg_name, None)
        if old is None or new is None:
            continue
        # Numeric compare with tolerance: 3 and 3.0 are the same budget.
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            if abs(float(old) - float(new)) > 1e-9:
                differences.append((meta_key, old, new))
        elif old != new:
            differences.append((meta_key, old, new))

    old_batch = recorded.get("effective_batch_size")
    new_batch = args.batch_size * args.grad_accum
    if old_batch is not None and old_batch != new_batch:
        differences.append(("effective_batch_size", old_batch, new_batch))

    # Training-set size is part of the budget too, and lives outside
    # hyperparameters because prepare_datasets.py controls it.
    old_examples = existing.get("train_examples")
    domain = ROLE_TO_DOMAIN[existing.get("role", "")] if existing.get("role") in ROLE_TO_DOMAIN else None
    if old_examples is not None and domain:
        current_path = DATA_DIR / f"{domain}_train.jsonl"
        if current_path.exists():
            current_examples = sum(1 for line in open(current_path, encoding="utf-8") if line.strip())
            if current_examples != old_examples:
                differences.append(("train_examples", old_examples, current_examples))

    return differences


def train_role(role: str, args: argparse.Namespace) -> Dict[str, Any]:
    _require(["torch", "transformers", "peft", "trl", "bitsandbytes", "datasets"])
    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    spec = config.LOCAL_PEFT_ROLES[role]
    base_model_id = spec["base_model"]
    domain = ROLE_TO_DOMAIN[role]
    out_dir = ROOT_DIR / config.LOCAL_ADAPTER_DIR / role
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"[train] role={role}  base={base_model_id}  domain={domain}")
    print("=" * 78)

    train_records = load_jsonl(DATA_DIR / f"{domain}_train.jsonl")
    val_path = DATA_DIR / f"{domain}_val.jsonl"
    val_records = load_jsonl(val_path) if val_path.exists() else []

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    _ensure_distinct_pad_token(tokenizer, base_model_id)
    tokenizer.padding_side = "right"

    train_ds = Dataset.from_dict(
        {"text": [render_example(r, tokenizer) for r in train_records]}
    )
    eval_ds = (
        Dataset.from_dict({"text": [render_example(r, tokenizer) for r in val_records]})
        if val_records
        else None
    )
    print(f"[train] {len(train_ds)} train / {len(eval_ds) if eval_ds else 0} eval examples")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.LOCAL_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.LOCAL_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        dtype=compute_dtype,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False       # incompatible with gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[train] trainable {trainable:,} / {total:,} params ({100 * trainable / total:.3f}%)")

    sft_config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",             # the adapter is saved explicitly below
        # Evaluate on a step interval rather than per epoch. With a single epoch,
        # per-epoch evaluation yields one data point, which is not a curve and
        # cannot show whether the run saturated or started degrading - the very
        # diagnostic that mattered most on the previous attempt. Four points per
        # run keeps that visible at any epoch count.
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=max(20, _total_steps(len(train_ds), args) // 4) if eval_ds else None,
        bf16=compute_dtype is torch.bfloat16,
        fp16=compute_dtype is torch.float16,
        optim="paged_adamw_8bit",       # the QLoRA paper's optimiser
        max_length=args.max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        packing=False,
        seed=args.seed,
        report_to=[],
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    duration_s = round(time.perf_counter() - started, 1)

    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    eval_metrics = trainer.evaluate() if eval_ds else {}

    # Per-epoch eval history, not just the final value.
    #
    # This is the diagnostic that caught a silently broken adapter: with 83% of
    # NewsQA examples truncated past the sequence limit, held-out loss sat at
    # 1.920 -> 1.923 -> 1.956 with token accuracy static at 0.568. The FINAL
    # numbers alone look like a plausible converged run; only the flat curve
    # reveals that nothing was learnt. Recording the trajectory means
    # validate_adapters.py can check it, and the paper can report it, instead of
    # it depending on someone reading the training log at the right moment.
    eval_history = [
        {
            "epoch": round(float(entry.get("epoch", 0)), 3),
            "eval_loss": float(entry["eval_loss"]),
            "eval_mean_token_accuracy": (
                float(entry["eval_mean_token_accuracy"])
                if "eval_mean_token_accuracy" in entry
                else None
            ),
        }
        for entry in getattr(trainer.state, "log_history", [])
        if "eval_loss" in entry
    ]

    # Did held-out loss actually improve over training? A non-improving curve
    # means the adapter is inert regardless of how the final loss reads.
    learning_signal = None
    if len(eval_history) >= 2:
        first_loss = eval_history[0]["eval_loss"]
        best_loss = min(point["eval_loss"] for point in eval_history)
        learning_signal = {
            "first_eval_loss": round(first_loss, 4),
            "best_eval_loss": round(best_loss, 4),
            "final_eval_loss": round(eval_history[-1]["eval_loss"], 4),
            "improvement": round(first_loss - best_loss, 4),
            "improved": bool(first_loss - best_loss > 0.01),
            "best_epoch": min(eval_history, key=lambda p: p["eval_loss"])["epoch"],
        }
        if not learning_signal["improved"]:
            print(
                f"[train] WARNING role={role}: held-out loss never improved "
                f"({first_loss:.4f} -> best {best_loss:.4f}). This adapter has probably "
                f"learnt nothing. The usual cause is examples exceeding "
                f"--max-seq-length, which right-truncates the TARGET away - check "
                f"finetune/data/manifest.json token_lengths before benchmarking it."
            )

    meta = {
        "role": role,
        "domain": domain,
        "base_model": base_model_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": duration_s,
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds) if eval_ds else 0,
        "hyperparameters": {
            "lora_rank": args.rank,
            "lora_alpha": args.alpha,
            "lora_dropout": args.dropout,
            "target_modules": TARGET_MODULES,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "per_device_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "max_seq_length": args.max_seq_length,
            "warmup_ratio": args.warmup_ratio,
            "optimizer": "paged_adamw_8bit",
            "lr_scheduler": "cosine",
            "seed": args.seed,
        },
        "quantization": {
            "load_in_4bit": True,
            "quant_type": config.LOCAL_QUANT_TYPE,
            "double_quant": config.LOCAL_DOUBLE_QUANT,
            "compute_dtype": str(compute_dtype).replace("torch.", ""),
        },
        "trainable_params": int(trainable),
        "total_params": int(total),
        "final_train_loss": float(result.training_loss) if result.training_loss is not None else None,
        "eval_metrics": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        "eval_history": eval_history,
        "learning_signal": learning_signal,
        "peak_vram_mib": (
            round(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None
        ),
        "platform": {
            "python": sys.version.split()[0],
            "torch": __import__("torch").__version__,
            "transformers": transformers.__version__,
            "system": platform.system(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"[train] role={role} done in {duration_s}s  loss={meta['final_train_loss']}  "
        f"peak_vram={meta['peak_vram_mib']} MiB  -> {out_dir}"
    )

    # Free the card before the next role loads; without this, three sequential
    # trainings in one process accumulate and OOM on the second or third.
    del trainer, model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--role",
        choices=list(ROLE_TO_DOMAIN.keys()) + ["all"],
        default="all",
        help="Which expert to train (default: all three, sequentially).",
    )
    parser.add_argument("--rank", type=int, default=DEFAULTS["rank"])
    parser.add_argument("--alpha", type=int, default=DEFAULTS["alpha"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--epochs", type=float, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    parser.add_argument("--max-seq-length", type=int, default=DEFAULTS["max_seq_length"])
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULTS["warmup_ratio"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if an adapter already exists, overwriting it. Without "
        "this, completed adapters are skipped so an interrupted --role all run "
        "resumes instead of redoing ~50 minutes of work per expert.",
    )
    args = parser.parse_args()

    # Training touches one model at a time, so serving-time adapter loading must
    # not interfere; nothing here reads LEGALAI_USE_ADAPTERS.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    roles = list(ROLE_TO_DOMAIN.keys()) if args.role == "all" else [args.role]

    # Resume by default. Each adapter takes ~50 minutes, so re-invoking after an
    # interruption must not silently retrain and overwrite the ones that already
    # finished. --force retrains regardless.
    summaries = []
    for role in roles:
        existing = _existing_adapter_meta(role)
        if existing is not None and not args.force:
            mismatch = _hyperparameter_mismatch(existing, args)
            if mismatch:
                # Refusing rather than skipping: an adapter trained on a
                # different budget than its peers is the confound this whole
                # protocol exists to avoid (see the consistency note above). The
                # only correct resolutions are to retrain everything on one
                # budget, or to match the existing one - never to proceed with a
                # mixed set.
                raise SystemExit(
                    f"\n[train] ABORT: adapters/{role} already exists but was trained with "
                    f"DIFFERENT settings than this invocation:\n"
                    + "\n".join(f"    {name}: existing={old!r} requested={new!r}"
                               for name, old, new in mismatch)
                    + "\n\nMixing training budgets across experts makes any topology "
                      "difference partly 'which position holds the best-trained expert'.\n"
                    "Either match the existing settings, or retrain ALL THREE on the new "
                    "ones:\n"
                    "    Remove-Item -Recurse -Force adapters\n"
                    "    python finetune\\train_qlora.py --role all\n"
                    "(or pass --force to overwrite just this one, accepting the confound)."
                )
            print(
                f"[train] role={role} already trained "
                f"({existing.get('train_examples')} examples, "
                f"{existing.get('duration_s')}s, loss={existing.get('final_train_loss')}) "
                f"- skipping. Pass --force to retrain."
            )
            summaries.append(existing)
            continue
        summaries.append(train_role(role, args))

    print("\n[train] summary")
    for meta in summaries:
        print(
            f"  {meta['role']:11s} loss={meta['final_train_loss']!s:8s} "
            f"{meta['duration_s']}s  peak={meta['peak_vram_mib']} MiB"
        )
    print(
        "\n[train] Next: validate the adapters actually changed behaviour before "
        "benchmarking:\n    python finetune/validate_adapters.py"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
