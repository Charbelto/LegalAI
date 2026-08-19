"""Step 0: can all three expert models actually co-reside in 8GB?

    python finetune/check_vram.py                 # load all three, then generate
    python finetune/check_vram.py --no-generate   # load only
    python finetune/check_vram.py --concurrent    # also test true parallel inference

The entire pivot rests on an assumption nobody had tested: that three 3-4B
models at 4-bit quantisation fit together on an 8GB laptop GPU and can serve the
PARALLEL topology concurrently. The plan calls this Step 0 for a reason -
everything downstream depends on it. This script answers it with measurements
rather than arithmetic, and writes them to finetune/vram_report.json so the
paper's hardware claims are transcribed from a real run.

What it reports
---------------
* Free VRAM before and after each model loads (torch.cuda.mem_get_info, i.e.
  the driver's view, which includes the desktop compositor's ~1.5GB - not just
  this process's allocations, because that is what actually determines whether
  the third model fits).
* One short generation per model, to prove the weights are usable and not merely
  resident.
* With --concurrent, three simultaneous generations from three threads, which is
  what the PARALLEL and Graph Engineering topologies do. Reports wall-clock against the
  sequential sum so the speedup (or lack of it) is visible.

If it fails, the fallbacks in plan order are: shorter context
(LEGALAI_LOCAL_MAX_INPUT_TOKENS), smaller variants, and only as a last resort
sequential loading - which would have to be disclosed in Threats to Validity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

FINETUNE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FINETUNE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: E402
import local_models  # noqa: E402

REPORT_PATH = FINETUNE_DIR / "vram_report.json"

PROBES = {
    "legal": "In one sentence, what obligation does the EU AI Act place on providers of high-risk AI systems?",
    "news": "In one sentence, what is a regulatory sandbox?",
    "general_qa": "In one sentence, what is the difference between a law and a regulation?",
}


def _gpu_snapshot():
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "free_mib": round(free / 1024**2),
        "total_mib": round(total / 1024**2),
        "used_mib": round((total - free) / 1024**2),
        "torch_allocated_mib": round(torch.cuda.memory_allocated() / 1024**2),
        "torch_reserved_mib": round(torch.cuda.memory_reserved() / 1024**2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-generate", action="store_true", help="Load only, do not generate.")
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Also run three simultaneous generations (what PARALLEL/Graph Engineering do).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--kv-context",
        type=int,
        default=2176 + 1024,
        help="Context length (input + output tokens) to budget the KV cache at for "
        "all three models simultaneously. Default 3200 = the measured worst-case "
        "expert prompt (2141 tokens on the Granite tokenizer) plus the 1024-token "
        "generation budget. This is the concurrent phase; the aggregator's longer "
        "prompt runs alone and so does not multiply.",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit(
            "torch is not installed. pip install -r requirements-finetune.txt"
        )

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "load_in_4bit": config.LOCAL_LOAD_IN_4BIT,
        "quant_type": config.LOCAL_QUANT_TYPE,
        "double_quant": config.LOCAL_DOUBLE_QUANT,
        "compute_dtype": config.LOCAL_COMPUTE_DTYPE,
        "use_adapters": config.LOCAL_PEFT_USE_ADAPTERS,
        "arm": local_models.arm_name(),
        "max_input_tokens": config.LOCAL_MAX_INPUT_TOKENS,
        "roles": [],
        "baseline": _gpu_snapshot(),
    }

    print("=" * 78)
    print("Step 0: concurrent-residency check for three local expert models")
    print("=" * 78)
    if report["baseline"]:
        print(
            f"GPU: {report['gpu']}  total {report['baseline']['total_mib']} MiB, "
            f"{report['baseline']['free_mib']} MiB free before loading "
            f"({report['baseline']['used_mib']} MiB already in use by other processes)"
        )
    print(f"Adapters: {'ON (peft arm)' if config.LOCAL_PEFT_USE_ADAPTERS else 'OFF (base control arm)'}")
    for row in local_models.describe_roles():
        state = "present" if row["adapter_present"] else "MISSING"
        print(f"  {row['role']:11s} {row['base_model']:42s} adapter={state}")
    print()

    models = {}
    ok = True
    for role in config.LOCAL_PEFT_ROLES:
        before = _gpu_snapshot()
        started = time.perf_counter()
        try:
            chat = local_models.LocalChatModel(role=role, temperature=0.0, max_new_tokens=args.max_new_tokens)
        except Exception as exc:
            print(f"[step0] FAILED to load {role}: {exc}")
            report["roles"].append({"role": role, "loaded": False, "error": str(exc)})
            ok = False
            break
        load_s = round(time.perf_counter() - started, 2)
        after = _gpu_snapshot()
        models[role] = chat

        entry = {
            "role": role,
            "loaded": True,
            "base_model": chat._loaded.base_model_id,
            "adapter_dir": chat._loaded.adapter_dir,
            "load_s": load_s,
            "free_before_mib": before["free_mib"] if before else None,
            "free_after_mib": after["free_mib"] if after else None,
            "model_cost_mib": (before["free_mib"] - after["free_mib"]) if before and after else None,
        }
        print(
            f"[step0] loaded {role:11s} in {load_s:6.1f}s  "
            f"cost {entry['model_cost_mib']} MiB  free now {entry['free_after_mib']} MiB"
        )
        report["roles"].append(entry)

    report["after_all_loads"] = _gpu_snapshot()
    if report["baseline"] and report["after_all_loads"]:
        report["all_three_models_mib"] = (
            report["baseline"]["free_mib"] - report["after_all_loads"]["free_mib"]
        )
        print(
            f"\n[step0] all three resident: {report['all_three_models_mib']} MiB of weights, "
            f"{report['after_all_loads']['free_mib']} MiB free"
        )

    # Generation proves the weights are usable, not just resident: a model can
    # load and then OOM on the first forward pass once activations allocate.
    if ok and not args.no_generate:
        print("\n[step0] sequential generation probe")
        sequential_total = 0.0
        for role, chat in models.items():
            started = time.perf_counter()
            try:
                message = chat.invoke(PROBES[role])
                elapsed = time.perf_counter() - started
                sequential_total += elapsed
                text = message.content.replace("\n", " ")[:110]
                print(
                    f"  {role:11s} {elapsed:6.2f}s  "
                    f"{message.response_metadata['completion_tokens']:3d} tok  {text}"
                )
                for entry in report["roles"]:
                    if entry["role"] == role:
                        entry["generate_s"] = round(elapsed, 2)
                        entry["completion_tokens"] = message.response_metadata["completion_tokens"]
                        entry["sample"] = message.content[:400]
            except Exception as exc:
                print(f"  {role:11s} GENERATION FAILED: {exc}")
                ok = False
                for entry in report["roles"]:
                    if entry["role"] == role:
                        entry["generate_error"] = str(exc)
        report["sequential_generation_s"] = round(sequential_total, 2)
        report["peak_vram_after_generation"] = _gpu_snapshot()

    # The real question for PARALLEL/Graph Engineering: do three experts overlap or queue?
    if ok and args.concurrent and not args.no_generate:
        print("\n[step0] concurrent generation probe (3 threads, what PARALLEL does)")
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                role: pool.submit(models[role].invoke, PROBES[role]) for role in models
            }
            errors = {}
            for role, future in futures.items():
                try:
                    future.result()
                except Exception as exc:
                    errors[role] = str(exc)
        wall = round(time.perf_counter() - started, 2)
        report["concurrent_generation_s"] = wall
        report["concurrent_errors"] = errors
        if report.get("sequential_generation_s"):
            report["concurrency_speedup"] = round(
                report["sequential_generation_s"] / wall, 2
            )
            print(
                f"  wall {wall}s vs sequential {report['sequential_generation_s']}s "
                f"-> speedup {report['concurrency_speedup']}x"
            )
        if errors:
            print(f"  concurrent errors: {errors}")
            ok = False
        report["peak_vram_after_concurrent"] = _gpu_snapshot()

    # KV-cache budget. Residency alone is not the test: the three models must
    # also have room for their key/value caches at the context length the
    # benchmark actually uses. This is where the first attempt failed - all three
    # models fitted, leaving 252 MiB, but their combined KV cache needs far more
    # than that, because Phi-3.5-mini has no grouped-query attention (32 KV
    # heads) and costs 384 KiB per token against Llama 3.2 3B's 112 and
    # Qwen2.5 3B's 36. Computed from each model's own config, not estimated.
    if ok and report["baseline"]:
        # Budget the CONCURRENT phase, which is what actually stresses the card:
        # the three experts running at once in PARALLEL/Graph Engineering. Their measured
        # prompts are ~2100 tokens, so --kv-context defaults to that plus the
        # generation budget. config.LOCAL_MAX_INPUT_TOKENS is deliberately much
        # higher (it has to clear the aggregator's ~4500-token prompt) but the
        # aggregator runs alone, so budgeting three models at that ceiling would
        # reject a configuration that works fine.
        budget = {
            "context_tokens": args.kv_context,
            "ceiling_max_input_tokens": config.LOCAL_MAX_INPUT_TOKENS,
            "note": "concurrent expert phase; the aggregator's longer prompt runs alone",
        }
        per_model = []
        total_kv_mib = 0.0
        for role, chat in models.items():
            model_config = chat._loaded.model.config
            layers = getattr(model_config, "num_hidden_layers", 0)
            heads = getattr(model_config, "num_attention_heads", 0)
            kv_heads = getattr(model_config, "num_key_value_heads", heads) or heads
            head_dim = getattr(model_config, "head_dim", None) or (
                getattr(model_config, "hidden_size", 0) // heads if heads else 0
            )
            # 2 tensors (K and V) x 2 bytes (bf16/fp16) per head-dim element.
            kib_per_token = layers * kv_heads * head_dim * 2 * 2 / 1024
            kv_mib = kib_per_token * budget["context_tokens"] / 1024
            total_kv_mib += kv_mib
            per_model.append(
                {
                    "role": role,
                    "layers": layers,
                    "kv_heads": kv_heads,
                    "head_dim": head_dim,
                    "grouped_query_attention": bool(kv_heads and heads and kv_heads < heads),
                    "kib_per_token": round(kib_per_token, 1),
                    "kv_cache_mib_at_context": round(kv_mib),
                }
            )
        budget["per_model"] = per_model
        budget["total_kv_cache_mib"] = round(total_kv_mib)
        budget["free_after_weights_mib"] = report["after_all_loads"]["free_mib"]
        budget["headroom_mib"] = round(report["after_all_loads"]["free_mib"] - total_kv_mib)
        budget["fits"] = budget["headroom_mib"] > 0
        report["kv_budget"] = budget

        print(
            f"\n[step0] KV-cache budget for the concurrent expert phase at "
            f"{budget['context_tokens']} tokens/model "
            f"(ceiling LEGALAI_LOCAL_MAX_INPUT_TOKENS={config.LOCAL_MAX_INPUT_TOKENS} "
            f"applies to the aggregator, which runs alone)"
        )
        for entry in per_model:
            print(
                f"  {entry['role']:11s} {entry['layers']:3d}L x {entry['kv_heads']:2d}kv x "
                f"{entry['head_dim']:3d}d = {entry['kib_per_token']:6.0f} KiB/tok -> "
                f"{entry['kv_cache_mib_at_context']:5d} MiB"
                f"{'' if entry['grouped_query_attention'] else '   (no GQA)'}"
            )
        print(
            f"  total KV {budget['total_kv_cache_mib']} MiB vs "
            f"{budget['free_after_weights_mib']} MiB free -> headroom "
            f"{budget['headroom_mib']} MiB"
        )
        if not budget["fits"]:
            ok = False
            print(
                "[step0] FAIL the three models are resident but cannot hold their KV "
                "caches concurrently at this context length."
            )

    report["verdict"] = "PASS" if ok else "FAIL"
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"Step 0 verdict: {report['verdict']}   (report: {REPORT_PATH.name})")
    if not ok:
        print(
            "Fallbacks, in the plan's order of preference:\n"
            "  1. Lower LEGALAI_LOCAL_MAX_INPUT_TOKENS (shorter context, same models)\n"
            "  2. Swap a base model for a smaller variant\n"
            "  3. Last resort: sequential loading, which must be disclosed in\n"
            "     Threats to Validity - the advisor asked for true concurrency."
        )
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
