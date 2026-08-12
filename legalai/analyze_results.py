"""Analysis and statistics script for Legal AI benchmark runs."""

import argparse
import itertools
import os
import re
import json
import math
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import evaluate_workflows as eval_funcs
import llm_judge
import config

ROOT_DIR = Path(__file__).resolve().parent
RUNS_FILE = ROOT_DIR / "benchmark_runs.jsonl"
DATASET_FILE = ROOT_DIR / "eval_dataset.json"

# Token cost config (configurable $/1k rates)
INPUT_COST_PER_1K = 0.00015   # $0.15 per million
OUTPUT_COST_PER_1K = 0.0006   # $0.60 per million

# Metrics carried into the paired significance tests.
TEST_METRICS = [
    "judge_average",
    "rouge_l",
    "elapsed_s",
    "cost",
    "abstained_flag",
    "judge_per_1k_tokens",  # token-budget-normalised quality
]

# Minimum number of paired queries for a two-sided Wilcoxon signed-rank test to be
# able to reach p < 0.05 at all. Below this the comparison is reported as
# not-tested (NaN) rather than as a null result.
MIN_PAIRS = 6


def validate_dataset_schema(dataset):
    """Validate eval_dataset.json schema."""
    for i, row in enumerate(dataset):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset entry {i} is not a dictionary.")
        for field in ["id", "type", "query", "gold"]:
            if field not in row:
                raise ValueError(f"Dataset entry {i} is missing required field '{field}'")
        if row["type"] not in ["simple", "decomposable", "routing"]:
            raise ValueError(f"Dataset entry {i} has invalid type '{row['type']}', must be simple/decomposable/routing")
        if not isinstance(row["gold_doc_ids"], list):
            raise ValueError(f"Dataset entry {i} gold_doc_ids must be a list")
    print(f"[analysis] Dataset schema validation passed: {len(dataset)} rows.")


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size between two groups x and y."""
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    greater = 0
    less = 0
    for val_x in x:
        for val_y in y:
            if val_x > val_y:
                greater += 1
            elif val_x < val_y:
                less += 1
    return (greater - less) / (n_x * n_y)


def compute_retrieval_metrics(retrieved_ids, gold_doc_ids, relevance_annotated=False):
    """Compute precision@5, recall@5 and reciprocal rank (RR).

    Returns (None, None, None) unless a human has annotated which chunks are
    actually relevant. `gold_doc_ids` used to be a copy of whatever the retriever
    returned while the gold answer was drafted, which made precision@5 equal to
    1.0 by construction and the metric meaningless.
    """
    if not relevance_annotated or not gold_doc_ids:
        return None, None, None

    gold_set = set(gold_doc_ids)
    retrieved_top5 = retrieved_ids[:5]
    retrieved_set = set(retrieved_top5)

    # Precision@5
    p_5 = len(retrieved_set.intersection(gold_set)) / 5.0

    # Recall@5
    r_5 = len(retrieved_set.intersection(gold_set)) / len(gold_set)

    # Reciprocal Rank (RR)
    rr = 0.0
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in gold_set:
            rr = 1.0 / rank
            break

    return p_5, r_5, rr


def _compare_paired(left_pq, right_pq, left_name, right_name, test_metrics, min_pairs):
    """One paired comparison between two per-query frames.

    Both frames are indexed by query_id with one row per query (repeats already
    averaged by the caller). Returns a flat dict of columns.

    Sign convention, stated once and used everywhere: a positive
    `<metric>_median_diff` and a positive `<metric>_cliffs_delta` both mean the
    LEFT side scored higher. `comparison` is always "<left>_vs_<right>", so the
    name tells you which direction positive points.
    """
    common_idx = left_pq.index.intersection(right_pq.index)
    row = {
        "comparison": f"{left_name}_vs_{right_name}",
        "left": left_name,
        "right": right_name,
        "n_queries_paired": len(common_idx),
    }

    for metric in test_metrics:
        if metric not in left_pq.columns or metric not in right_pq.columns:
            # A metric absent from this subset (e.g. no judge scores at all)
            # is recorded as not-tested rather than silently skipped.
            row[f"{metric}_n"] = 0
            row[f"{metric}_left_mean"] = np.nan
            row[f"{metric}_right_mean"] = np.nan
            row[f"{metric}_median_diff"] = np.nan
            row[f"{metric}_underpowered"] = True
            row[f"{metric}_p"] = np.nan
            row[f"{metric}_cliffs_delta"] = np.nan
            continue

        l_series = pd.to_numeric(left_pq.loc[common_idx, metric], errors="coerce")
        r_series = pd.to_numeric(right_pq.loc[common_idx, metric], errors="coerce")
        usable = l_series.notna() & r_series.notna()
        l_vals = l_series[usable].to_numpy(dtype=float)
        r_vals = r_series[usable].to_numpy(dtype=float)
        n_pairs = len(l_vals)

        row[f"{metric}_n"] = n_pairs
        row[f"{metric}_left_mean"] = float(np.mean(l_vals)) if n_pairs else np.nan
        row[f"{metric}_right_mean"] = float(np.mean(r_vals)) if n_pairs else np.nan
        row[f"{metric}_median_diff"] = (
            float(np.median(l_vals - r_vals)) if n_pairs else np.nan
        )
        row[f"{metric}_underpowered"] = bool(n_pairs < min_pairs)

        if n_pairs < min_pairs or np.all(l_vals == r_vals):
            # Report as not-tested rather than as p = 1.0, which reads as
            # evidence of no difference when it is really absence of data.
            p_val = np.nan
        else:
            try:
                p_val = float(wilcoxon(l_vals, r_vals).pvalue)
            except Exception as exc:
                print(f"      [analysis] Wilcoxon warning for {row['comparison']} {metric}: {exc}")
                p_val = np.nan

        row[f"{metric}_p"] = p_val
        row[f"{metric}_cliffs_delta"] = cliffs_delta(l_vals, r_vals) if n_pairs else np.nan

    return row


def _holm_correct(sig, test_metrics):
    """Add Holm-corrected p-values across the rows of one comparison family."""
    if sig.empty:
        return sig
    for metric in test_metrics:
        column = f"{metric}_p"
        if column not in sig.columns:
            continue
        p_vals = pd.to_numeric(sig[column], errors="coerce").to_numpy(dtype=float)
        corrected = np.full(p_vals.shape, np.nan, dtype=float)
        testable = ~np.isnan(p_vals)
        if testable.sum() > 0:
            _, p_corr, _, _ = multipletests(p_vals[testable], alpha=0.05, method="holm")
            corrected[testable] = p_corr
        sig[f"{metric}_p_holm"] = corrected
    return sig


def paired_tests(frame, test_metrics, min_pairs=MIN_PAIRS):
    """Wilcoxon + Cliff's delta for every PAIR of topologies, per query.

    Changed by the PEFT pivot. Previously every multi-agent topology was tested
    against the single-agent baseline, because the question was "does multi-agent
    beat single-agent". SINGLE is no longer in the compared set - once each
    expert is a separately fine-tuned model, that question is not the one this
    experiment answers - so the comparison is now all-pairs among the topologies
    present: ALL vs PARALLEL, ALL vs DAG, PARALLEL vs DAG. With three topologies
    that is three tests, and Holm corrects across exactly that family of three,
    per metric.

    The experimental unit is still the QUERY, not the repeat. Repeats of the same
    query are correlated samples of one item, so pairing on (query_id, repeat)
    inflates n and breaks the independence assumption of the Wilcoxon
    signed-rank test. Repeats are averaged within each (query_id, mode) cell
    first, and modes are paired on query_id.
    """
    available = [m for m in test_metrics if m in frame.columns]
    per_query = frame.groupby(["query_id", "mode"])[available].mean().reset_index()

    modes = sorted(per_query["mode"].unique())
    if len(modes) < 2:
        print(f"[analysis]   fewer than 2 topologies in this subset ({modes}); skipping")
        return pd.DataFrame()

    by_mode = {mode: per_query[per_query["mode"] == mode].set_index("query_id") for mode in modes}

    rows = [
        _compare_paired(by_mode[left], by_mode[right], left, right, test_metrics, min_pairs)
        for left, right in itertools.combinations(modes, 2)
    ]
    return _holm_correct(pd.DataFrame(rows), test_metrics)


def arm_tests(frame, test_metrics, min_pairs=MIN_PAIRS):
    """RQ2: within each topology, does the PEFT arm beat the untuned base arm?

    This is the ablation the pivot plan flags as the gap worth closing. Without
    it the paper can only say "these fine-tuned agents behave this way under
    these topologies"; with it, it can say whether the fine-tuning did anything
    at all. Pairing is on query_id within a topology, so each query contributes
    one peft value and one base value - the same paired design as the topology
    tests, just varying the arm instead of the structure.

    Returns an empty frame when the run only covered one arm, rather than
    inventing a comparison.
    """
    if "arm" not in frame.columns:
        return pd.DataFrame()
    arms = sorted(a for a in frame["arm"].dropna().unique())
    if len(arms) < 2:
        print(f"[analysis]   only one arm present ({arms}); no ablation to test")
        return pd.DataFrame()

    available = [m for m in test_metrics if m in frame.columns]
    per_query = frame.groupby(["query_id", "mode", "arm"])[available].mean().reset_index()

    rows = []
    for mode in sorted(per_query["mode"].unique()):
        subset = per_query[per_query["mode"] == mode]
        peft = subset[subset["arm"] == "peft"].set_index("query_id")
        base = subset[subset["arm"] == "base"].set_index("query_id")
        if peft.empty or base.empty:
            print(f"[analysis]   mode={mode}: missing one arm; skipping ablation test")
            continue
        row = _compare_paired(peft, base, "peft", "base", test_metrics, min_pairs)
        row["mode"] = mode
        row["comparison"] = f"{mode}:peft_vs_base"
        rows.append(row)

    # Holm across the family of topologies (3 tests), per metric.
    return _holm_correct(pd.DataFrame(rows), test_metrics)


def analyze():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=int(os.getenv("JUDGE_WORKERS", "8")),
        help="Max concurrent LLM judge calls (default 8). The judge talks directly "
        "to the provider's API (see llm_judge.py call_provider), not through our "
        "local FastAPI server, so it isn't bound by the thread-pool limit that "
        "caps benchmark.py's concurrency.",
    )
    args = parser.parse_args()
    judge_workers = max(1, args.judge_workers)

    # 1. Load dataset & validate
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_FILE}")
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    validate_dataset_schema(dataset)

    # 2. Load runs
    if not RUNS_FILE.exists():
        print(f"[analysis] Error: Runs file {RUNS_FILE} does not exist. Run benchmark first.")
        return

    runs = []
    failed_count = 0
    with open(RUNS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if not data.get("success", False):
                failed_count += 1
                continue
            runs.append(data)

    print(f"[analysis] Loaded {len(runs)} successful runs (dropped {failed_count} failed runs).")
    if not runs:
        print("[analysis] No successful runs to analyze.")
        return

    df = pd.DataFrame(runs)

    # 3. Compute quality metrics per run
    print("[analysis] Calculating lexical similarity metrics vs gold answers...")
    bleu_1, bleu_4 = [], []
    rouge_1_p, rouge_1_r, rouge_1_f = [], [], []
    rouge_2_p, rouge_2_r, rouge_2_f = [], [], []
    rouge_l_p, rouge_l_r, rouge_l_f = [], [], []
    jaccards, char_jaccards, cosines, levenshteins = [], [], [], []

    # Retrieval metrics
    p_5s, r_5s, rrs = [], [], []

    # Real Cost
    costs = []

    # Inputs for the LLM judge, built in df row order and scored in a separate
    # concurrent pass below - kept as a parallel list (not dispatched inline)
    # specifically so judge_accs[i] etc. stay aligned with df.iloc[i] regardless
    # of which judge call happens to finish first.
    judge_inputs = []

    for idx, row in df.iterrows():
        response = row.get("response", "")
        gold = row.get("gold", "")
        query_id = row.get("query_id")

        # Find query text from dataset
        q_item = next((item for item in dataset if item["id"] == query_id), None)
        query_text = q_item["query"] if q_item else ""

        # Lexical quality
        bleus = eval_funcs.calculate_bleu(gold, response)
        bleu_1.append(bleus[0])
        bleu_4.append(bleus[3])

        r1_p, r1_r, r1_f = eval_funcs.calculate_rouge_n(gold, response, 1)
        r2_p, r2_r, r2_f = eval_funcs.calculate_rouge_n(gold, response, 2)
        rl_p, rl_r, rl_f = eval_funcs.calculate_rouge_l(gold, response)

        rouge_1_p.append(r1_p)
        rouge_1_r.append(r1_r)
        rouge_1_f.append(r1_f)

        rouge_2_p.append(r2_p)
        rouge_2_r.append(r2_r)
        rouge_2_f.append(r2_f)

        rouge_l_p.append(rl_p)
        rouge_l_r.append(rl_r)
        rouge_l_f.append(rl_f)

        jaccards.append(eval_funcs.word_jaccard(gold, response))
        char_jaccards.append(eval_funcs.char_jaccard(gold, response))
        cosines.append(eval_funcs.cosine_similarity_tf(gold, response))
        levenshteins.append(eval_funcs.levenshtein_similarity(gold, response))

        judge_inputs.append((query_text, gold, response))

        # Retrieval
        annotated = bool(q_item.get("relevance_annotated", False)) if q_item else False
        p_5, r_5, rr = compute_retrieval_metrics(
            row.get("retrieved_ids", []),
            (q_item or {}).get("gold_doc_ids", []),
            relevance_annotated=annotated,
        )
        p_5s.append(p_5)
        r_5s.append(r_5)
        rrs.append(rr)

        # Real Cost
        pt = row.get("prompt_tokens") or 0
        ct = row.get("completion_tokens") or 0
        cost = (pt * INPUT_COST_PER_1K + ct * OUTPUT_COST_PER_1K) / 1000.0
        costs.append(cost)

    # LLM Judge, dispatched concurrently. ThreadPoolExecutor.map() returns
    # results in the same order as its inputs regardless of completion order
    # (unlike as_completed(), which would return in completion order and
    # silently misalign judge scores against the wrong rows) - that guarantee
    # is what keeps judge_accs[i] etc. lined up with df.iloc[i] below.
    print(f"[analysis] Scoring {len(judge_inputs)} runs with the LLM judge (judge-workers={judge_workers})...")
    judge_accs, judge_comps, judge_grounds, judge_avgs = [], [], [], []
    judge_ok = []

    def _score_one(indexed_args):
        i, (query_text, gold, response) = indexed_args
        print(f"  [LLM Judge] Scoring run {i+1}/{len(judge_inputs)}...")
        return llm_judge.judge(query_text, gold, response)

    with ThreadPoolExecutor(max_workers=judge_workers) as pool:
        judge_results = list(pool.map(_score_one, enumerate(judge_inputs)))

    # LLM Judge. A failed judge call returns ok=False; recording it as 1/1/1
    # would fabricate a real (very bad) score, so those rows are excluded from
    # judge statistics while still contributing latency/cost data.
    for res in judge_results:
        if res.get("ok", True) and res.get("accuracy") is not None:
            judge_accs.append(res["accuracy"])
            judge_comps.append(res["completeness"])
            judge_grounds.append(res["groundedness"])
            judge_avgs.append((res["accuracy"] + res["completeness"] + res["groundedness"]) / 3.0)
            judge_ok.append(True)
        else:
            print(f"    [LLM Judge] EXCLUDED (judge failure): {res.get('rationale')}")
            judge_accs.append(np.nan)
            judge_comps.append(np.nan)
            judge_grounds.append(np.nan)
            judge_avgs.append(np.nan)
            judge_ok.append(False)

    df["bleu_1"] = bleu_1
    df["bleu_4"] = bleu_4
    
    df["rouge_1_p"] = rouge_1_p
    df["rouge_1_r"] = rouge_1_r
    df["rouge_1_f"] = rouge_1_f
    
    df["rouge_2_p"] = rouge_2_p
    df["rouge_2_r"] = rouge_2_r
    df["rouge_2_f"] = rouge_2_f
    
    df["rouge_l_p"] = rouge_l_p
    df["rouge_l_r"] = rouge_l_r
    df["rouge_l_f"] = rouge_l_f
    
    # Keep aliases for compat
    df["rouge_1"] = rouge_1_f
    df["rouge_2"] = rouge_2_f
    df["rouge_l"] = rouge_l_f

    df["jaccard"] = jaccards
    df["char_jaccard"] = char_jaccards
    df["cosine"] = cosines
    df["levenshtein"] = levenshteins

    df["judge_accuracy"] = judge_accs
    df["judge_completeness"] = judge_comps
    df["judge_groundedness"] = judge_grounds
    df["judge_average"] = judge_avgs
    df["judge_ok"] = judge_ok
    judge_failures = int(len(df) - sum(judge_ok))
    if judge_failures:
        print(
            f"[analysis] WARNING {judge_failures}/{len(df)} judge calls failed and were "
            "excluded from judge metrics (they remain in latency/cost metrics)."
        )

    annotated_count = sum(1 for item in dataset if item.get("relevance_annotated"))
    if annotated_count == 0:
        print(
            "[analysis] Retrieval metrics SKIPPED: no query has annotated relevant "
            "chunks. Run scripts/annotate_relevance.py, or delete every retrieval "
            "claim from the paper - do not report self-referential values."
        )
    else:
        print(f"[analysis] Retrieval metrics computed for {annotated_count}/{len(dataset)} annotated queries.")

    df["precision_at_5"] = p_5s
    df["recall_at_5"] = r_5s
    df["mrr"] = rrs
    df["cost"] = costs

    # Text structural metrics
    word_counts, char_counts, sentence_counts, paragraph_counts = [], [], [], []
    avg_word_lens, avg_sent_lens, ttrs = [], [], []
    citations_list, temporal_refs_list, bullet_points_list = [], [], []

    for idx, row in df.iterrows():
        resp = row.get("response", "")
        tokens = eval_funcs.custom_tokenize(resp)
        wc = len(resp.split())
        cc = len(resp)
        
        sc = len(re.split(r"[.!?]+", resp)) - 1
        if sc <= 0:
            sc = 1
            
        pc = len([p for p in resp.split("\n\n") if p.strip()])
        if pc <= 0:
            pc = 1
            
        awl = sum(len(w) for w in tokens) / len(tokens) if tokens else 0.0
        asl = wc / sc
        ttr_val = len(set(tokens)) / len(tokens) if tokens else 0.0
        
        cits = len(re.findall(r"(?i)article\s+\d+", resp))
        temps = len(re.findall(r"\b(202\d|may|june|july|january|february|march|april)\b", resp.lower()))
        bullets = len(re.findall(r"^\s*[-*+•\d+\.]", resp, re.MULTILINE))
        
        word_counts.append(wc)
        char_counts.append(cc)
        sentence_counts.append(sc)
        paragraph_counts.append(pc)
        avg_word_lens.append(awl)
        avg_sent_lens.append(asl)
        ttrs.append(ttr_val)
        citations_list.append(cits)
        temporal_refs_list.append(temps)
        bullet_points_list.append(bullets)

    df["word_count"] = word_counts
    df["char_count"] = char_counts
    df["sentence_count"] = sentence_counts
    df["paragraph_count"] = paragraph_counts
    df["avg_word_len"] = avg_word_lens
    df["avg_sent_len"] = avg_sent_lens
    df["ttr"] = ttrs
    df["citations"] = citations_list
    df["temporal_refs"] = temporal_refs_list
    df["bullet_points"] = bullet_points_list

    # Compute operational/overhead metrics
    df["net_overhead"] = df["elapsed_s"] - (df["backend_ms"] / 1000.0)
    df["net_overhead"] = df["net_overhead"].apply(lambda x: max(0.0, x))
    df["lat_per_step"] = df.apply(lambda row: row["backend_ms"] / row["steps"] if row["steps"] > 0 else 0.0, axis=1)
    df["words_per_sec"] = df.apply(lambda row: row["word_count"] / row["elapsed_s"] if row["elapsed_s"] > 0 else 0.0, axis=1)

    # Token-budget fairness. The 2026 single-vs-multi papers control for the
    # reasoning-token budget; without a normalised view, "cheaper and better" can
    # be mistaken for a like-for-like win. These express quality per unit of spend.
    total_tokens = (
        pd.to_numeric(df["prompt_tokens"], errors="coerce").fillna(0)
        + pd.to_numeric(df["completion_tokens"], errors="coerce").fillna(0)
    )
    df["total_tokens"] = total_tokens
    df["judge_per_1k_tokens"] = np.where(
        total_tokens > 0, df["judge_average"] / (total_tokens / 1000.0), np.nan
    )
    df["rouge_l_per_1k_tokens"] = np.where(
        total_tokens > 0, df["rouge_l"] / (total_tokens / 1000.0), np.nan
    )
    df["judge_per_second"] = np.where(
        df["elapsed_s"] > 0, df["judge_average"] / df["elapsed_s"], np.nan
    )

    # Extract individual node timings
    nodes_order = ["planner", "router", "memory", "retrieval", "legal", "news", "general_qa", "aggregator", "validator", "response"]
    for node in nodes_order:
        df[f"timing_{node}"] = df["timings"].apply(lambda t: t.get(node, 0.0) if isinstance(t, dict) else 0.0)

    # Abstention as a measured outcome (older runs predate these fields)
    if "abstained" not in df.columns:
        df["abstained"] = False
    if "expert_abstention_rate" not in df.columns:
        df["expert_abstention_rate"] = 0.0
    df["abstained_flag"] = df["abstained"].fillna(False).astype(int)
    df["expert_abstention_rate"] = pd.to_numeric(df["expert_abstention_rate"], errors="coerce").fillna(0.0)

    # Arm (peft / base). Rows written before the PEFT pivot carry no arm; label
    # them explicitly rather than letting them join whichever arm sorts first.
    if "arm" not in df.columns:
        df["arm"] = "unknown"
    df["arm"] = df["arm"].fillna("unknown").astype(str)
    arms_present = sorted(df["arm"].unique())
    print(f"[analysis] Arms present: {arms_present}")
    if "unknown" in arms_present:
        print(
            "[analysis] WARNING some rows have no 'arm' field. They predate the PEFT "
            "pivot and are NOT interchangeable with either arm; re-run the benchmark "
            "rather than pooling them."
        )

    abstain_by_mode = df.groupby(["arm", "mode"])["abstained_flag"].mean()
    print("[analysis] Abstention rate by arm and mode:")
    for (arm_name, mode_name), rate in abstain_by_mode.items():
        print(f"    {arm_name:6s} {mode_name:22s} {rate:.1%}")

    # Time-correlated latency drift check.
    #
    # The arms run as separate sequential passes, not interleaved, so anything
    # that changes slowly over a multi-day run - GPU thermal throttling above
    # all - lands disproportionately on whichever arm ran second. That would
    # show up as a latency difference between arms with no causal relationship
    # to the adapters, and elapsed_s is one of the tested metrics. Topology
    # comparisons are unaffected, since modes are interleaved within an arm.
    #
    # Reported rather than corrected: silently regressing latency on wall-clock
    # time would hide a real measurement problem behind an adjustment.
    drift_report = {}
    if "started_at_utc" in df.columns and df["started_at_utc"].notna().any():
        times = pd.to_datetime(df["started_at_utc"], errors="coerce", utc=True)
        df["_run_started"] = times
        for arm in arms_present:
            arm_rows = df[(df["arm"] == arm) & times.notna()]
            if len(arm_rows) < 10:
                continue
            first, last = arm_rows["_run_started"].min(), arm_rows["_run_started"].max()
            # Latency of the first vs last decile of each arm, in run order.
            ordered = arm_rows.sort_values("_run_started")
            decile = max(1, len(ordered) // 10)
            early = pd.to_numeric(ordered["elapsed_s"].head(decile), errors="coerce").mean()
            late = pd.to_numeric(ordered["elapsed_s"].tail(decile), errors="coerce").mean()
            drift_pct = ((late - early) / early * 100.0) if early else float("nan")
            drift_report[arm] = {
                "started_utc": str(first),
                "finished_utc": str(last),
                "duration_hours": round((last - first).total_seconds() / 3600.0, 2),
                "mean_elapsed_s_first_decile": round(float(early), 2),
                "mean_elapsed_s_last_decile": round(float(late), 2),
                "drift_pct": round(float(drift_pct), 1),
            }
        df.drop(columns=["_run_started"], inplace=True, errors="ignore")

        if drift_report:
            print("[analysis] Within-arm latency drift (first vs last decile, run order):")
            for arm, info in drift_report.items():
                print(
                    f"    {arm:6s} {info['mean_elapsed_s_first_decile']:7.2f}s -> "
                    f"{info['mean_elapsed_s_last_decile']:7.2f}s "
                    f"({info['drift_pct']:+.1f}%) over {info['duration_hours']}h"
                )
            worst = max(abs(i["drift_pct"]) for i in drift_report.values())
            if worst > 15.0:
                print(
                    f"[analysis] WARNING latency drifted {worst:.0f}% within an arm. The arms "
                    "ran as separate passes, so drift of this size confounds any "
                    "peft-vs-base LATENCY comparison (quality metrics are unaffected). "
                    "Report the arm latency difference as uninterpretable, or re-run with "
                    "the arms interleaved."
                )
    else:
        print(
            "[analysis] NOTE rows carry no started_at_utc, so time-correlated latency "
            "drift cannot be checked. Pre-pivot runs predate the field."
        )

    # 4. Aggregate metrics per mode
    print("[analysis] Aggregating results per mode...")
    metrics_to_agg = [
        "bleu_1", "bleu_4", 
        "rouge_1_p", "rouge_1_r", "rouge_1_f",
        "rouge_2_p", "rouge_2_r", "rouge_2_f",
        "rouge_l_p", "rouge_l_r", "rouge_l_f",
        "bleu_1", "bleu_4", "rouge_1", "rouge_2", "rouge_l", "jaccard", "char_jaccard", "cosine", "levenshtein",
        "judge_accuracy", "judge_completeness", "judge_groundedness", "judge_average",
        "precision_at_5", "recall_at_5", "mrr",
        "elapsed_s", "backend_ms", "net_overhead", "lat_per_step", "words_per_sec", "steps", "prompt_tokens", "completion_tokens", "cost",
        "word_count", "char_count", "sentence_count", "paragraph_count", "avg_word_len", "avg_sent_len", "ttr",
        "citations", "temporal_refs", "bullet_points",
        "abstained_flag", "expert_abstention_rate",
        "total_tokens", "judge_per_1k_tokens", "rouge_l_per_1k_tokens", "judge_per_second",
    ] + [f"timing_{node}" for node in nodes_order]

    # De-duplicate while preserving order (bleu_1/bleu_4 were listed twice, which
    # produced duplicate columns in by_query_type.csv).
    metrics_to_agg = list(dict.fromkeys(metrics_to_agg))

    # One row per (mode, arm): the arm is an experimental factor, so collapsing
    # over it would average a fine-tuned system together with its own untuned
    # control and report the mean as if it were one system.
    agg_rows = []
    cells = sorted(set(zip(df["mode"], df["arm"])))

    for mode, arm in cells:
        mode_df = df[(df["mode"] == mode) & (df["arm"] == arm)]
        row_summary = {"mode": mode, "arm": arm}
        for metric in metrics_to_agg:
            # Filter non-null values (e.g. retrieval metrics might be null for some queries)
            series = mode_df[metric].dropna()
            n = len(series)
            if n == 0:
                row_summary[f"{metric}_mean"] = np.nan
                row_summary[f"{metric}_std"] = np.nan
                row_summary[f"{metric}_ci"] = np.nan
                row_summary[f"{metric}_n"] = 0
                continue

            mean = series.mean()
            std = series.std(ddof=1) if n > 1 else 0.0
            sem = std / math.sqrt(n)
            # 95% Confidence Interval margin
            ci_margin = 1.96 * sem

            row_summary[f"{metric}_mean"] = round(mean, 5)
            row_summary[f"{metric}_std"] = round(std, 5)
            row_summary[f"{metric}_ci"] = round(ci_margin, 5)
            row_summary[f"{metric}_n"] = n

        agg_rows.append(row_summary)

    summary_df = pd.DataFrame(agg_rows)
    # Output analysis_summary.csv
    summary_df.to_csv(ROOT_DIR / "analysis_summary.csv", index=False)
    print(f"[analysis] Saved analysis summary to {ROOT_DIR}/analysis_summary.csv")

    # 5. Grouped by query_type (Interaction H2)
    print("[analysis] Analyzing interaction (H2) by query type...")
    type_mode_agg = (
        df.groupby(["query_type", "mode", "arm"])[metrics_to_agg].mean().reset_index()
    )
    type_mode_agg.to_csv(ROOT_DIR / "by_query_type.csv", index=False)
    print(f"[analysis] Saved query type interaction to {ROOT_DIR}/by_query_type.csv")

    # 6. Paired significance tests.
    #
    # Two families of comparison since the PEFT pivot:
    #   RQ1/RQ3 - all pairs of topologies (ALL / PARALLEL / DAG), run separately
    #             within each arm, because comparing a tuned topology against an
    #             untuned one would confound structure with specialisation.
    #   RQ2     - peft vs base within each topology (the ablation).
    #
    # In both, the experimental unit is the QUERY, not the individual repeat, and
    # Holm corrects within each family - see paired_tests/arm_tests.
    test_metrics = list(TEST_METRICS)

    sig_frames = []
    for arm in arms_present:
        arm_df = df[df["arm"] == arm]
        print(f"[analysis] Pairwise topology tests within arm '{arm}' ({len(arm_df)} runs)...")
        overall_sig = paired_tests(arm_df, test_metrics)
        if not overall_sig.empty:
            overall_sig.insert(0, "scope", "overall")
            overall_sig.insert(1, "arm", arm)
            sig_frames.append(overall_sig)

        # H2 interaction: the same paired tests within each query type. If
        # topology matters at all it should matter most on decomposable and
        # routing queries, so a single pooled test would hide it.
        for qtype, subset in arm_df.groupby("query_type"):
            print(f"[analysis]   arm={arm} query_type={qtype} ({len(subset)} runs)")
            type_sig = paired_tests(subset, test_metrics)
            if not type_sig.empty:
                type_sig.insert(0, "scope", f"query_type={qtype}")
                type_sig.insert(1, "arm", arm)
                sig_frames.append(type_sig)

    # RQ2 ablation: does specialisation itself help, within each topology?
    print("[analysis] Ablation tests (peft vs base) per topology...")
    ablation_sig = arm_tests(df, test_metrics)
    if not ablation_sig.empty:
        ablation_sig.insert(0, "scope", "arm_ablation")
        ablation_sig.insert(1, "arm", "peft_vs_base")
        sig_frames.append(ablation_sig)
    else:
        print(
            "[analysis] NOTE no ablation results. RQ2 is unanswerable from this run - "
            "benchmark both arms (run_experiment.ps1 does both by default) or drop RQ2 "
            "from the paper rather than implying the control existed."
        )

    sig_df = pd.concat(sig_frames, ignore_index=True) if sig_frames else pd.DataFrame()

    if not sig_df.empty:
        underpowered_cols = [c for c in sig_df.columns if c.endswith("_underpowered")]
        if underpowered_cols and sig_df[underpowered_cols].any().any():
            print(
                f"[analysis] WARNING some comparisons have fewer than {MIN_PAIRS} paired "
                "queries and are reported as not-tested (p = NaN), not as null results."
            )

    sig_df.to_csv(ROOT_DIR / "significance.csv", index=False)
    print(f"[analysis] Saved paired significance to {ROOT_DIR}/significance.csv")

    # 7. Consistency variance
    print("[analysis] Analyzing consistency variance across repeats...")
    consistency = (
        df.groupby(["query_id", "mode", "arm"])["judge_average"].var().reset_index()
    )
    consistency.rename(columns={"judge_average": "variance"}, inplace=True)
    
    # 8. Create results.json bundle
    run_meta_path = ROOT_DIR / "run_meta.json"
    run_meta = {}
    if run_meta_path.exists():
        try:
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[analysis] Warning: could not read run_meta.json: {exc}")

    # Per-arm run metadata, so a two-arm run keeps both halves' provenance even
    # though run_meta.json only holds whichever arm finished last.
    arm_meta = {}
    for arm in arms_present:
        arm_meta_path = ROOT_DIR / f"run_meta_{arm}.json"
        if arm_meta_path.exists():
            try:
                arm_meta[arm] = json.loads(arm_meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[analysis] Warning: could not read {arm_meta_path.name}: {exc}")

    provenance = {
        "runs_file": RUNS_FILE.name,
        "runs_analyzed": int(len(df)),
        "queries": int(df["query_id"].nunique()),
        "query_types": sorted(df["query_type"].dropna().unique().tolist()),
        "modes": sorted(df["mode"].dropna().unique().tolist()),
        "arms": arms_present,
        "runs_per_arm": {
            arm: int((df["arm"] == arm).sum()) for arm in arms_present
        },
        "repeats_per_cell": int(df.groupby(["query_id", "mode", "arm"]).size().max()),
        "judge_model": getattr(llm_judge, "JUDGE_MODEL", None),
        "judge_provider": getattr(llm_judge, "JUDGE_PROVIDER", None),
        # Read from the dataset rather than llm_judge.GOLD_MODEL. That constant
        # names whichever model first drafted the reference answers and goes
        # stale the moment one is revised, which would credit the paper's ground
        # truth to a model that did not write it.
        "gold_model": "; ".join(
            sorted({str(i["gold_model"]) for i in dataset if i.get("gold_model")})
        )
        or getattr(llm_judge, "GOLD_MODEL", None),
        # How those answers were produced, so a reader can tell a raw draft from
        # a reviewed one without opening the dataset.
        "gold_status": {
            status: sum(1 for i in dataset if i.get("gold_status") == status)
            for status in sorted(
                {str(i["gold_status"]) for i in dataset if i.get("gold_status")}
            )
        },
        "system_model": getattr(llm_judge, "SYSTEM_MODEL", None),
        # Which model actually generated each expert's answers, and whether its
        # adapter was applied. Read from config (the single source of truth the
        # serving path also uses) rather than retyped into the paper by hand.
        "expert_models": [
            {
                "role": role,
                "base_model": spec["base_model"],
                "finetune_dataset": spec["dataset"],
                "adapter_dir": spec["adapter"],
            }
            for role, spec in getattr(config, "LOCAL_PEFT_ROLES", {}).items()
        ],
        "coordinator_role": getattr(config, "LOCAL_COORDINATOR_ROLE", None),
        "coordinator_uses_adapter": getattr(config, "LOCAL_COORDINATOR_USE_ADAPTER", None),
        "local_load_in_4bit": getattr(config, "LOCAL_LOAD_IN_4BIT", None),
        "local_quant_type": getattr(config, "LOCAL_QUANT_TYPE", None),
        "ablation_tested": not ablation_sig.empty,
        # Recorded explicitly (not inferred from run_meta.json's client-side env
        # block, which only reflects benchmark.py's own process and can miss
        # variables that only ever lived in .env inside the server subprocess) -
        # snapshot_run.py relies on this being accurate.
        "generation_provider": getattr(config, "GENERATION_PROVIDER", None),
        "judge_failures_excluded": judge_failures,
        "queries_with_relevance_annotation": annotated_count,
        "retrieval_metrics_reportable": bool(annotated_count),
        "gold_needs_review": bool(df.get("gold_needs_review", pd.Series(dtype=bool)).fillna(False).any()),
        "min_pairs_for_test": MIN_PAIRS,
        "test_metrics": list(TEST_METRICS),
        "latency_drift_by_arm": drift_report,
        "run_meta": run_meta,
        "run_meta_by_arm": arm_meta,
    }
    print(
        f"[analysis] Provenance: {provenance['queries']} queries x "
        f"{len(provenance['modes'])} topologies x {len(arms_present)} arm(s), "
        f"judge={provenance['judge_model']}, gold={provenance['gold_model']}"
    )
    if provenance["gold_needs_review"]:
        print("[analysis] WARNING gold answers are still flagged needs_review; do not "
              "describe them as expert-curated.")

    results_bundle = {
        "provenance": provenance,
        "summary": summary_df.to_dict(orient="records"),
        "significance": sig_df.to_dict(orient="records"),
        # Split out as well as included above, so the paper's RQ2 subsection can
        # be written from one small block instead of filtering the whole frame.
        "ablation": ablation_sig.to_dict(orient="records") if not ablation_sig.empty else [],
        "by_query_type": type_mode_agg.to_dict(orient="records"),
        "consistency": consistency.to_dict(orient="records")
    }

    with open(ROOT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_bundle, f, indent=2, ensure_ascii=False)
    print(f"[analysis] Saved complete results.json bundle")


if __name__ == "__main__":
    analyze()
