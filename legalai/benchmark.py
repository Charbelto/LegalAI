"""Benchmark harness: run Legal AI across topologies, queries, and repeats.

Design notes (these matter for the validity of the numbers this produces):

* Canned COMPL-AI demo answers are forced OFF. They bypass the multi-agent
  workflow entirely and report synthetic 1.0 ms telemetry, which silently
  flattened between-topology differences in earlier runs.
* Repeats only carry information if the model samples. LEGALAI_DETERMINISTIC
  defaults to "0" here and each repeat gets its own seed, so text-quality
  variance (and therefore confidence intervals) is real. Set
  LEGALAI_DETERMINISTIC=1 explicitly if you want greedy decoding, in which case
  repeats measure latency variance only.
* One warm-up request per mode is issued and discarded so that first-call model
  loading does not land in the measured data.
* Smoke runs write to a separate file so they can never be mistaken for, or
  mixed into, a full run.
* Since the PEFT pivot each run also belongs to an ARM: "peft" (each expert
  carries its LoRA adapter) or "base" (identical base weights, no adapter). The
  arm is recorded on every row and cross-checked against what the server
  actually reports, because the two arms are indistinguishable from the outside
  and mislabelling one would invalidate RQ2 silently.
"""

import argparse
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parent
URL_BASE = os.getenv("URL_BASE", "http://127.0.0.1:8000").rstrip("/")
URL = f"{URL_BASE}/chat"
HEALTH_URL = f"{URL_BASE}/health"
# Was "/config", a route that doesn't exist on this backend (backend/main.py
# only defines "/runtime") - every preflight() check below silently no-opped
# on a 404 body ({"detail": "Not Found"}) instead of ever seeing real config,
# including the abort-if-COMPL-AI-enabled safety check.
CONFIG_URL = f"{URL_BASE}/runtime"

# Three structurally distinct coordination patterns, and only these three.
#
#   all                strict sequential chain  (legal -> news -> general_qa)
#   parallel           full concurrency         (all three at once)
#   graph_engineering  converging dependency    (legal, news -> general_qa + loop engineering verification)
#
# The other topologies - single, legal_first, planner_based, verify_only,
# legal_news_parallel - remain fully implemented and selectable in
# graph/workflow.py and through the API/UI. They are excluded from the
# benchmarked set and from the paper for scope, not because they are broken.
# SINGLE in particular is no longer a baseline: once each expert is a separately
# fine-tuned model, "one agent vs many" is a different question from "which
# structure combines specialised agents best", which is what this experiment now
# measures.
MODES = [
    "all",
    "parallel",
    "graph_engineering",
]

# The two experimental arms. Both are benchmarked, so the paper can report
# whether PEFT specialisation itself improved anything (RQ2) rather than only
# ranking topologies among already-specialised agents.
ARMS = ["peft", "base"]

# Seed for (repeat r) is SEED_BASE + r, recorded on every row.
SEED_BASE = int(os.getenv("BENCH_SEED_BASE", "1000"))


def _force_experiment_env(arm: str = None):
    """Pin the environment the server subprocess inherits."""
    os.environ["LEGALAI_ENABLE_COMPL_AI"] = "0"      # never canned answers in an experiment
    os.environ.setdefault("LEGALAI_DETERMINISTIC", "0")  # sample, so repeats vary
    os.environ.setdefault("LEGALAI_NUM_CTX", "8192")     # avoid silent truncation
    # A live web/EUR-Lex lookup would make the legal expert's context depend on
    # what the internet returned that minute, so two runs of the same query
    # would not be comparable. Same reasoning as fetch_news=False below.
    os.environ["LEGALAI_ENABLE_EURLEX_LIVE"] = "0"
    if arm is not None:
        # This is what actually selects the arm: local_models reads it when it
        # resolves each role's model. Set before the server subprocess starts.
        os.environ["LEGALAI_USE_ADAPTERS"] = "1" if arm == "peft" else "0"


# How long to wait for the server to become healthy.
#
# Was 25 attempts (~25s), which is not enough for this stack: importing
# backend.main pulls in chromadb, langchain, langgraph and pymupdf, and on a cold
# filesystem cache that alone can exceed a minute on Windows. The failure looked
# like a hang rather than a timeout - the log filled with identical health-check
# errors while the server was in fact still importing, and it became healthy
# shortly after benchmark.py had already given up.
SERVER_START_TIMEOUT_S = int(os.getenv("BENCH_SERVER_START_TIMEOUT_S", "300"))

# Per-request timeout. Was 600s, which is fine for a hosted API but not for local
# 2-3B generation: one pass through the graph makes ~8 model calls and takes 5-8
# minutes, and for graph_engineering the terminal validator (Loop Engineering) can
# send the whole graph round again (MAX_ITERATIONS=2). That mode's retries are
# normal rather than exceptional - in the pre-pivot run, when the validator still
# ran unconditionally for every mode, 508 of 630 rows showed 20+ graph steps
# against 14-17 for a single pass - so a 10-minute ceiling would convert the
# system's ordinary behaviour into recorded failures. ALL and PARALLEL no longer
# reach the validator at all (see graph/workflow.py route_after_aggregator) and so
# never retry, but the timeout still has to clear graph_engineering's worst case.
REQUEST_TIMEOUT_S = int(os.getenv("BENCH_REQUEST_TIMEOUT_S", "3600"))


def start_server():
    """Start the uvicorn FastAPI server as a subprocess and wait until healthy."""
    print("[benchmark] Starting FastAPI server...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT_DIR),
        stdout=None,
        stderr=None,
    )

    deadline = time.perf_counter() + SERVER_START_TIMEOUT_S
    attempt = 0
    last_error = None
    while time.perf_counter() < deadline:
        attempt += 1

        # If the subprocess has already exited there is nothing to wait for, and
        # retrying for another five minutes only hides its exit code.
        exit_code = proc.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"The server subprocess exited with code {exit_code} before becoming "
                f"healthy. Its traceback was printed above (stdout/stderr are "
                f"inherited). Common causes: a missing adapter for the requested arm, "
                f"port 8000 already in use, or an import error."
            )

        try:
            resp = requests.get(HEALTH_URL, timeout=5)
            if resp.status_code == 200:
                waited = SERVER_START_TIMEOUT_S - (deadline - time.perf_counter())
                print(f"[benchmark] Server is ready after {waited:.0f}s ({attempt} checks)")
                return proc
            last_error = f"status {resp.status_code}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}"

        # Report progress sparsely: one line per attempt turned the log into
        # hundreds of identical tracebacks that obscured the real failure.
        if attempt == 1 or attempt % 10 == 0:
            remaining = max(0, deadline - time.perf_counter())
            print(
                f"[benchmark]   still waiting for the server "
                f"({attempt} checks, {remaining:.0f}s left; last: {last_error})"
            )
        time.sleep(1)

    proc.terminate()
    proc.wait()
    raise RuntimeError(
        f"Server did not become healthy within {SERVER_START_TIMEOUT_S}s "
        f"(last error: {last_error}). Raise BENCH_SERVER_START_TIMEOUT_S if this "
        f"machine is simply slow to import the stack."
    )


def preflight(repeats: int, arm: str = None) -> dict:
    """Verify the server we are about to measure is configured for an experiment."""
    try:
        server_config = requests.get(CONFIG_URL, timeout=5).json()
    except Exception as exc:
        raise RuntimeError(f"Could not read {CONFIG_URL}: {exc}") from exc

    if server_config.get("compl_ai_enabled"):
        raise RuntimeError(
            "ABORT: the server has COMPL-AI canned answers enabled. Those bypass the "
            "workflow and fake telemetry. Restart it with LEGALAI_ENABLE_COMPL_AI=0."
        )

    # Arm cross-check. benchmark.py reuses an already-running server when it
    # finds one, and a server left over from the *other* arm looks identical
    # from the outside - so a two-arm run could silently label 270 base-model
    # runs as "peft". The server reports which arm it actually loaded; refuse to
    # measure it if that disagrees with what was asked for.
    if arm is not None:
        server_arm = server_config.get("generation_arm")
        if server_arm is None:
            raise RuntimeError(
                "ABORT: --arm was requested but the server does not report "
                "generation_arm. It is probably running an older build, or "
                "GENERATION_PROVIDER is not local_peft. Restart it."
            )
        if server_arm != arm:
            raise RuntimeError(
                f"ABORT: --arm {arm} was requested but the running server reports "
                f"arm '{server_arm}'. A stale server from the other arm would "
                f"mislabel every row. Stop it and let benchmark.py start its own, "
                f"or restart it with LEGALAI_USE_ADAPTERS="
                f"{'1' if arm == 'peft' else '0'}."
            )
        missing = [
            row["role"]
            for row in server_config.get("local_models", [])
            if row.get("adapter_requested") and not row.get("adapter_present")
        ]
        if missing:
            raise RuntimeError(
                f"ABORT: arm 'peft' needs LoRA adapters but these roles have none: "
                f"{missing}. Train them first (python finetune/train_qlora.py) "
                f"or run --arm base."
            )
        print(f"[benchmark] Arm check OK: server reports arm='{server_arm}'")
        for row in server_config.get("local_models", []):
            print(
                f"[benchmark]   {row['role']:11s} {row['base_model']:42s} "
                f"adapter={'on' if row.get('adapter_requested') else 'off'}"
            )

    if server_config.get("deterministic") and repeats > 1:
        print(
            f"[benchmark] WARNING server reports deterministic=True with REPEATS={repeats}. "
            "Decoding is greedy, so every repeat will return identical text and all "
            "text-quality CIs will be +/-0.00. Restart the server with "
            "LEGALAI_DETERMINISTIC=0 for variance-bearing repeats."
        )

    print(
        f"[benchmark] Server config OK: provider={server_config.get('generation_provider', 'ollama')} "
        f"model={server_config.get('chat_model')} "
        f"num_ctx={server_config.get('num_ctx')} deterministic={server_config.get('deterministic')} "
        f"compl_ai_enabled={server_config.get('compl_ai_enabled')}"
    )
    return server_config


def warm_up(dataset):
    """Issue one throwaway request per mode so model-load time is not measured."""
    if not dataset:
        return
    probe = dataset[0]["query"]
    print(f"[benchmark] Warm-up: 1 discarded request per mode ({len(MODES)} total)...")
    for mode in MODES:
        try:
            requests.post(
                URL,
                json={
                    "message": probe,
                    "session_id": f"warmup_{mode}",
                    "fetch_news": False,
                    "expert_execution_mode": mode,
                    "seed": SEED_BASE - 1,
                },
                timeout=REQUEST_TIMEOUT_S,
            )
            print(f"[benchmark]   warmed {mode}")
        except Exception as exc:
            print(f"[benchmark]   warm-up failed for {mode}: {exc}")


def _execute_one(item: dict, mode: str, rep: int, repeats: int, arm: str) -> dict:
    """Run one (query, mode, repeat) combination and return its result row.

    Safe to call from multiple threads at once: each call only touches its own
    local variables and does its own HTTP round-trip, so concurrent calls
    don't share any mutable state on the client side.
    """
    q_id = item["id"]
    seed = SEED_BASE + rep
    payload = {
        "message": item["query"],
        # Arm in the session id too: the same (mode, query, repeat) is run once
        # per arm, and a shared session id would let arm 1's history leak into
        # arm 2's memory node.
        "session_id": f"bench_{arm}_{mode}_{q_id}_{rep}",
        "fetch_news": False,
        "expert_execution_mode": mode,
        "seed": seed,
    }

    print(
        f"[benchmark] -> {q_id} | {arm.upper()} | {mode.upper()} | "
        f"repeat {rep + 1}/{repeats} | seed {seed}..."
    )
    # Absolute start time, recorded per row so time-correlated drift is
    # detectable afterwards instead of invisible. This matters because the two
    # arms run as separate sequential passes rather than interleaved: over a
    # multi-day run a thermally throttling GPU could make the second arm
    # systematically slower for reasons unrelated to the adapters, and elapsed_s
    # is one of the metrics in the arm comparison. Topology comparisons are not
    # exposed to this, since modes are interleaved across queries within an arm.
    started_iso = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    try:
        resp = requests.post(URL, json=payload, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        elapsed_s = round(time.perf_counter() - start, 3)

        row = {
            "query_id": q_id,
            "query_type": item["type"],
            "mode": mode,
            "arm": arm,
            "repeat": rep,
            "seed": seed,
            "gold": item["gold"],
            "gold_doc_ids": item.get("gold_doc_ids", []),
            "gold_status": item.get("gold_status"),
            "gold_needs_review": item.get("needs_review"),
            "response": data.get("response", ""),
            "started_at_utc": started_iso,
            "elapsed_s": elapsed_s,
            "backend_ms": data.get("workflow_elapsed_ms"),
            "route": data.get("route"),
            "timings": data.get("agent_timings_ms", {}),
            "steps": len(data.get("thinking_log", [])),
            "retrieved_ids": data.get("retrieved_ids", []),
            "prompt_tokens": data.get("prompt_tokens"),
            "completion_tokens": data.get("completion_tokens"),
            # abstention as a measured outcome, not a silent rewrite
            "abstained": data.get("abstained", False),
            "abstained_experts": data.get("abstained_experts", []),
            "experts_run": data.get("experts_run", 0),
            "expert_abstention_rate": data.get("expert_abstention_rate", 0.0),
            "truncation_warnings": data.get("truncation_warnings", []),
            "success": True,
        }
        flag = " ABSTAINED" if row["abstained"] else ""
        print(
            f"[benchmark]    ok {q_id} | {arm} | {mode} | repeat {rep} | {elapsed_s}s | tokens "
            f"{row['prompt_tokens']}/{row['completion_tokens']}{flag}"
        )
        return row
    except Exception as exc:
        # The server's actual detail (the real exception, not just requests'
        # generic "500 Server Error: ...") lives in the response body -
        # without this, every server-side failure looked identical here and
        # in benchmark_runs.jsonl, regardless of what actually broke.
        detail = str(exc)
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                detail = response.json().get("detail", detail)
            except Exception:
                detail = response.text or detail
        print(f"[benchmark]    ERROR {q_id} | {arm} | {mode} | repeat {rep}: {detail}")
        return {
            "query_id": q_id,
            "query_type": item["type"],
            "mode": mode,
            "arm": arm,
            "repeat": rep,
            "seed": seed,
            "success": False,
            "error": detail,
        }


def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="1 query x all modes x 1 repeat, separate output file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing runs file")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the discarded warm-up pass")
    parser.add_argument(
        "--arm",
        choices=ARMS,
        default=os.getenv("BENCH_ARM", "peft"),
        help="Experimental arm. 'peft' = each expert carries its LoRA adapter; "
        "'base' = the identical base models with no adapter (the control that "
        "makes RQ2 answerable). Sets LEGALAI_USE_ADAPTERS for the server "
        "subprocess and is cross-checked against what the server reports. The "
        "two arms cannot share one process (six models will not fit in 8GB), so "
        "run them as two passes - run_experiment.ps1 does this for you.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the runs file instead of truncating it. Used for the "
        "second arm of a two-arm run, so both arms land in one file with an "
        "'arm' column rather than in two files analysis would have to stitch.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("BENCH_CONCURRENCY", "1")),
        help="Max in-flight requests. 1 (default) preserves the original sequential "
        "behaviour - safe everywhere, and the only setting that measures each "
        "request's latency in true isolation. Raise this against "
        "GENERATION_PROVIDER=deepseek (or another hosted API) freely - the server "
        "no longer serializes requests there. For local_peft on a VRAM-rich GPU, "
        "raising it is now also supported: see config.LOCAL_MODEL_POOL_SIZE and "
        "local_models.py's per-replica CUDA streams, which let concurrent requests "
        "genuinely overlap instead of only queueing on one shared model lock. Doing "
        "so trades per-request latency purity for wall-clock throughput - recorded "
        "elapsed_s at concurrency > 1 includes queueing/contention delay, not just "
        "the topology's own work, so latency comparisons remain valid ACROSS "
        "topologies (the same concurrency level is applied uniformly to all three) "
        "but are no longer isolated-request latency numbers; disclose the "
        "concurrency level used alongside any latency figure. A shared local "
        "Ollama backend (embeddings) can still choke on true concurrency without "
        "OLLAMA_NUM_PARALLEL raised, and per-request seeds are race-free regardless "
        "of provider now that local generation seeds via a per-call torch.Generator "
        "rather than the global RNG (see local_models.py LocalChatModel.invoke).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of (query, mode, repeat) combinations attempted, for a "
        "quick sample/timing check. Writes to benchmark_runs_sample.jsonl, never to "
        "benchmark_runs.jsonl, so a sample run can never be mistaken for, or "
        "overwrite, real experiment data.",
    )
    args = parser.parse_args()

    _force_experiment_env(arm=args.arm)

    repeats = 1 if args.smoke else int(os.getenv("BENCH_REPEATS", "3"))
    concurrency = max(1, args.concurrency)

    dataset_path = ROOT_DIR / "eval_dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Smoke and sample (--limit) output are kept strictly separate from full-run
    # output, so neither can be mistaken for, or overwrite, the real thing.
    if args.smoke:
        runs_file = ROOT_DIR / "benchmark_runs_smoke.jsonl"
    elif args.limit is not None:
        runs_file = ROOT_DIR / "benchmark_runs_sample.jsonl"
    else:
        runs_file = ROOT_DIR / "benchmark_runs.jsonl"

    print(f"[benchmark] ARM: {args.arm} (LEGALAI_USE_ADAPTERS={os.environ.get('LEGALAI_USE_ADAPTERS')})")
    if args.smoke:
        dataset = dataset[:1]
        print(f"[benchmark] SMOKE mode: 1 query x {len(MODES)} modes x 1 repeat -> {runs_file.name}")
    else:
        print(
            f"[benchmark] FULL mode: {len(dataset)} queries x {len(MODES)} modes x {repeats} repeats "
            f"= {len(dataset) * len(MODES) * repeats} runs for arm '{args.arm}' -> {runs_file.name}"
        )
        if args.limit is not None:
            print(f"[benchmark] --limit {args.limit}: sample run, capped at {args.limit} runs")
    if concurrency > 1:
        print(f"[benchmark] concurrency={concurrency} (max in-flight requests)")

    existing_runs = set()
    if args.resume and runs_file.exists():
        print(f"[benchmark] Resume mode: reading existing runs from {runs_file}")
        with open(runs_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("success"):
                        # Keyed on arm too. Without it, resuming a two-arm run
                        # would treat arm 2's cells as already done because arm 1
                        # covered the same (query, mode, repeat) triple, and 270
                        # runs would silently vanish. Rows written before the
                        # pivot have no arm; treat them as the peft arm's
                        # predecessors rather than matching every arm.
                        existing_runs.add(
                            (
                                data["query_id"],
                                data["mode"],
                                data["repeat"],
                                data.get("arm", "peft"),
                            )
                        )
                except Exception as exc:
                    print(f"[benchmark] Warning parsing resume line: {exc}")

    server_process = None
    try:
        resp = requests.get(HEALTH_URL, timeout=2)
        if resp.status_code != 200:
            print(f"[benchmark] Health check returned {resp.status_code}. Starting a new server...")
            server_process = start_server()
        else:
            print("[benchmark] Server is already running. Reusing it.")
    except Exception:
        server_process = start_server()

    # Preflight sits inside its own try/finally: it is the step most likely to
    # abort (arm mismatch, missing adapters, COMPL-AI enabled), and it used to run
    # outside any cleanup, so a failed check left the server subprocess alive. The
    # next arm's invocation would then find a server already listening, reuse it,
    # and be measuring the PREVIOUS arm's loaded models. The arm cross-check would
    # catch that too, but leaking the process turns one clear failure into two.
    try:
        server_config = preflight(repeats, arm=args.arm)
    except Exception:
        if server_process:
            print("[benchmark] Preflight failed; stopping the server we started.")
            server_process.terminate()
            server_process.wait()
        raise

    started_at = time.perf_counter()
    started_iso = datetime.now(timezone.utc).isoformat()

    if not args.no_warmup:
        warm_up(dataset)

    work_items = []
    for item, mode, rep in itertools.product(dataset, MODES, range(repeats)):
        q_id = item["id"]
        if (q_id, mode, rep, args.arm) in existing_runs:
            print(f"[benchmark] Skipping completed run: {q_id} | {args.arm} | {mode} | repeat {rep}")
            continue
        work_items.append((item, mode, rep))

    if args.limit is not None and len(work_items) > args.limit:
        print(f"[benchmark] Truncating {len(work_items)} planned runs down to --limit {args.limit}")
        work_items = work_items[: args.limit]

    completed = 0
    failed = 0
    try:
        mode_str = "a" if (args.resume or args.append) else "w"
        with open(runs_file, mode_str, encoding="utf-8") as out:
            if concurrency <= 1:
                for item, mode, rep in work_items:
                    row = _execute_one(item, mode, rep, repeats, args.arm)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    completed += row.get("success", False)
                    failed += not row.get("success", False)
            else:
                print(f"[benchmark] Dispatching {len(work_items)} runs, concurrency={concurrency}...")
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [
                        pool.submit(_execute_one, item, mode, rep, repeats, args.arm)
                        for item, mode, rep in work_items
                    ]
                    for future in as_completed(futures):
                        row = future.result()
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()
                        completed += row.get("success", False)
                        failed += not row.get("success", False)

        duration_s = round(time.perf_counter() - started_at, 2)
        meta = {
            "started_at_utc": started_iso,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": duration_s,  # measured, not hardcoded
            "runs_file": runs_file.name,
            "smoke": bool(args.smoke),
            "modes": MODES,
            "arm": args.arm,
            "appended": bool(args.append),
            "repeats": repeats,
            "queries": len(dataset),
            "query_types": sorted({item["type"] for item in dataset}),
            "planned_runs": len(dataset) * len(MODES) * repeats,
            "attempted_runs": len(work_items),
            "limit": args.limit,
            "concurrency": concurrency,
            "completed_runs": completed,
            "failed_runs": failed,
            "warmup_excluded": not args.no_warmup,
            "seed_base": SEED_BASE,
            "seeds": [SEED_BASE + r for r in range(repeats)],
            "server_config": server_config,
            "env": {
                key: os.environ.get(key)
                for key in (
                    "LEGALAI_DETERMINISTIC",
                    "LEGALAI_NUM_CTX",
                    "LEGALAI_NUM_PREDICT",
                    "LEGALAI_ENABLE_COMPL_AI",
                    "GENERATION_PROVIDER",
                    "OLLAMA_MODEL",
                    "OLLAMA_NUM_PARALLEL",
                    "OLLAMA_BASE_URL",
                    "DEEPSEEK_MODEL",
                    # local_peft provenance: which arm, which base models, and
                    # that live legal search stayed off.
                    "LEGALAI_USE_ADAPTERS",
                    "LEGALAI_ADAPTER_DIR",
                    "LEGALAI_LEGAL_BASE_MODEL",
                    "LEGALAI_NEWS_BASE_MODEL",
                    "LEGALAI_GENERAL_BASE_MODEL",
                    "LEGALAI_LOCAL_MAX_INPUT_TOKENS",
                    "LEGALAI_ENABLE_EURLEX_LIVE",
                )
            },
            "platform": {
                "python": sys.version.split()[0],
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
        }
        if args.smoke:
            meta_path = ROOT_DIR / "run_meta_smoke.json"
        elif args.limit is not None:
            meta_path = ROOT_DIR / "run_meta_sample.json"
        else:
            meta_path = ROOT_DIR / "run_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Per-arm copy as well. run_meta.json is what analysis reads, but arm 2
        # of a two-arm run overwrites it, and losing arm 1's timings/seed record
        # would make that half of the experiment unprovenanced.
        if not args.smoke and args.limit is None:
            arm_meta_path = ROOT_DIR / f"run_meta_{args.arm}.json"
            arm_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[benchmark] Per-arm metadata saved to {arm_meta_path.name}")

        print(f"[benchmark] Runs saved to {runs_file}")
        print(f"[benchmark] Metadata saved to {meta_path}")
        print(f"[benchmark] {completed} succeeded, {failed} failed, {duration_s}s total")
        if not args.smoke and args.limit is None:
            print("[benchmark] Next: python analyze_results.py")

    finally:
        if server_process:
            print("[benchmark] Stopping FastAPI server...")
            server_process.terminate()
            server_process.wait()
            print("[benchmark] Server stopped.")


if __name__ == "__main__":
    run_benchmark()
