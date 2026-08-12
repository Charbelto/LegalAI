"""LLM-as-a-judge correctness scorer for Legal AI responses.

Supports three providers:

* `deepseek` (current default) - hosted DeepSeek V4 Flash. Cheap and strong
  enough to judge reliably; this is the "focus" provider for this experiment.
* `openai`  - any hosted OpenAI-compatible endpoint (OpenAI itself, or another
  provider that speaks the same API), kept as an alternative hosted option.
* `ollama`  - a local model, free but weaker. Not removed - still fully
  supported, just no longer the default. Switch back any time with
  JUDGE_PROVIDER=ollama.

Generation stays local (Ollama) in all cases; only judging goes over the
network for a hosted provider, which is the cheap part of the pipeline.

Cost safety, because the budget here is a few dollars:

* Every scored answer is cached by content hash, so re-running the analysis never
  pays twice for the same (query, gold, answer) triple.
* Real token usage is read from the API response, not estimated.
* Spend accumulates in `judge_spend.json` across runs and is checked against
  `JUDGE_BUDGET_USD` before every call. Exceeding it raises rather than
  continuing to spend.
* `python llm_judge.py --check` makes exactly one call and reports model,
  latency, tokens and cost so the setup can be validated for a fraction of a cent.

Configuration (put these in legalai/.env):

    # Hosted, DeepSeek (current default):
    JUDGE_PROVIDER=deepseek
    JUDGE_MODEL=deepseek-v4-flash
    DEEPSEEK_API_KEY=sk-...
    JUDGE_BUDGET_USD=3.00
    JUDGE_PRICE_IN_PER_M=0.14      # verify against current pricing
    JUDGE_PRICE_OUT_PER_M=0.28

    # Hosted, OpenAI (alternative):
    #   JUDGE_PROVIDER=openai
    #   JUDGE_MODEL=<model id>
    #   OPENAI_API_KEY=sk-...

    # Local, free (alternative, still fully supported):
    #   JUDGE_PROVIDER=ollama
    #   JUDGE_MODEL=gemma4:12b

Usage:
    python llm_judge.py --check
    python llm_judge.py --spend
    python llm_judge.py --validate judge_validation.csv
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import env_loader  # noqa: F401  (imported for its side effect: loads .env)

ROOT_DIR = Path(__file__).resolve().parent
CACHE_FILE = ROOT_DIR / "judge_cache.json"
SPEND_FILE = ROOT_DIR / "judge_spend.json"

JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "deepseek").strip().lower()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "deepseek-v4-flash")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Hosted providers: both speak the OpenAI chat-completions schema, so they share
# the same HTTP call code (_call_hosted) and differ only in key/base_url/name.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")

# Metadata only - no captured values here. api_key/base_url are looked up from
# the live module globals at call time (via _provider_config below) so that
# tests (and any runtime reconfiguration) that patch OPENAI_API_KEY /
# DEEPSEEK_API_KEY etc. directly are honoured, instead of a stale value frozen
# at import time.
HOSTED_PROVIDERS = {
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url_var": "OPENAI_BASE_URL",
        "models_url": "https://api.openai.com/v1/models",
    },
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY",
        "base_url_var": "DEEPSEEK_BASE_URL",
        "models_url": "https://api.deepseek.com/v1/models",
    },
}


def _provider_config(provider: str) -> dict:
    """Live api_key/base_url for a hosted provider, read from current globals."""
    meta = HOSTED_PROVIDERS[provider]
    return {
        "api_key": globals()[meta["key_env"]],
        "base_url": globals()[meta["base_url_var"]],
        "key_env": meta["key_env"],
        "models_url": meta["models_url"],
    }

# Prices change; these are only used to track spend against the budget. Set them
# from the current pricing page. Defaults below are DeepSeek V4 Flash's
# cache-miss rates - update if you switch JUDGE_PROVIDER to openai.
PRICE_IN_PER_M = float(os.getenv("JUDGE_PRICE_IN_PER_M", "0.14"))
PRICE_OUT_PER_M = float(os.getenv("JUDGE_PRICE_OUT_PER_M", "0.28"))
BUDGET_USD = float(os.getenv("JUDGE_BUDGET_USD", "3.00"))

# Model that drafted the reference answers, and the model being graded. A judge
# sharing a family with either shows self-preference bias.
GOLD_MODEL = os.getenv("GOLD_MODEL", "llama3.1:8b")
try:
    import config as _config

    # The system under test is whatever GENERATION_PROVIDER actually resolves
    # to - not always OLLAMA_MODEL. Getting this wrong doesn't just mislabel
    # provenance in results.json; it also silences the self-preference-bias
    # check below (deepseek judging deepseek's own answers would otherwise
    # look like "deepseek judging qwen2.5" and pass silently).
    #
    # local_peft needs its own branch: it serves three separately fine-tuned
    # experts, so there is no single model under test. Without this the else
    # arm reported OLLAMA_MODEL - a model that never ran - and the bias check
    # compared the judge against a family absent from the experiment.
    if _config.GENERATION_PROVIDER == "deepseek":
        SYSTEM_MODELS = [_config.DEEPSEEK_MODEL]
    elif _config.GENERATION_PROVIDER == "local_peft":
        SYSTEM_MODELS = [
            spec["base_model"] for spec in _config.LOCAL_PEFT_ROLES.values()
        ] or [_config.OLLAMA_MODEL]
    else:
        SYSTEM_MODELS = [_config.OLLAMA_MODEL]
except Exception:  # pragma: no cover
    SYSTEM_MODELS = [os.getenv("OLLAMA_MODEL", "qwen2.5")]

# Kept as a string for provenance and display; the bias check below iterates
# SYSTEM_MODELS so a multi-expert run is checked model by model.
SYSTEM_MODEL = "; ".join(dict.fromkeys(SYSTEM_MODELS))


def _family(model_name: str) -> str:
    return re.split(r"[:/\-]", str(model_name).strip().lower())[0]


# Self-preference-bias check: applies regardless of which provider the judge
# runs on. It used to be gated to JUDGE_PROVIDER == "ollama" on the assumption
# that a hosted judge could never share a family with local generation - that
# assumption broke the moment generation could also be hosted (e.g.
# GENERATION_PROVIDER=deepseek + JUDGE_PROVIDER=deepseek is the same model
# judging its own answers).
_bias_clashes = [m for m in SYSTEM_MODELS if _family(JUDGE_MODEL) == _family(m)]
if _bias_clashes:
    print(
        f"[judge] WARNING judge '{JUDGE_MODEL}' shares a family with "
        f"{len(_bias_clashes)} model(s) under test: {', '.join(_bias_clashes)}. "
        "It will favour their outputs; pick a different judge."
    )
if JUDGE_MODEL == GOLD_MODEL:
    print(
        f"[judge] WARNING judge and gold-answer model are both '{JUDGE_MODEL}'. "
        "Self-preference bias is well documented; set JUDGE_MODEL to a different model."
    )

# --------------------------------------------------------------------------- #
# Cache and spend tracking
# --------------------------------------------------------------------------- #

_cache = {}
if CACHE_FILE.exists():
    try:
        _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[judge] Warning: failed to load cache: {exc}")

# Guards _cache and the two on-disk files below (judge_cache.json, spend log) -
# analyze_results.py now scores rows concurrently, and concurrent
# read-modify-write of a shared dict + "serialize the whole thing to disk"
# call is a lost-update / torn-write hazard otherwise. Only ever held around
# the fast in-memory + small-file-write parts, never around the network call
# itself - holding it there would serialize every judge call again.
_state_lock = threading.Lock()


def save_cache():
    try:
        CACHE_FILE.write_text(json.dumps(_cache, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"[judge] Warning: failed to save cache: {exc}")


def load_spend() -> dict:
    if SPEND_FILE.exists():
        try:
            return json.loads(SPEND_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"total_usd": 0.0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def record_spend(prompt_tokens: int, completion_tokens: int) -> dict:
    """Add one call's real usage to the cumulative spend log."""
    with _state_lock:
        spend = load_spend()
        cost = (prompt_tokens / 1e6) * PRICE_IN_PER_M + (completion_tokens / 1e6) * PRICE_OUT_PER_M
        spend["total_usd"] = round(spend.get("total_usd", 0.0) + cost, 6)
        spend["calls"] = spend.get("calls", 0) + 1
        spend["prompt_tokens"] = spend.get("prompt_tokens", 0) + int(prompt_tokens)
        spend["completion_tokens"] = spend.get("completion_tokens", 0) + int(completion_tokens)
        spend["last_call_usd"] = round(cost, 6)
        spend["provider"] = JUDGE_PROVIDER
        spend["model"] = JUDGE_MODEL
        spend["prices_per_m"] = {"in": PRICE_IN_PER_M, "out": PRICE_OUT_PER_M}
        try:
            SPEND_FILE.write_text(json.dumps(spend, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[judge] Warning: failed to write spend log: {exc}")
        return spend


def check_budget():
    """Raise before making a paid call if the cumulative budget is exhausted."""
    if JUDGE_PROVIDER not in HOSTED_PROVIDERS:
        return
    spend = load_spend()
    if spend.get("total_usd", 0.0) >= BUDGET_USD:
        raise RuntimeError(
            f"Judge budget exhausted: ${spend['total_usd']:.4f} spent of ${BUDGET_USD:.2f} "
            f"over {spend.get('calls', 0)} calls. Raise JUDGE_BUDGET_USD to continue, or "
            "switch JUDGE_PROVIDER=ollama. Scores already cached are kept."
        )


def get_cache_key(query: str, gold: str, answer: str) -> str:
    combined = f"{JUDGE_PROVIDER}|{JUDGE_MODEL}|{query.strip()}|||{gold.strip()}|||{answer.strip()}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

JUDGE_PROMPT_TEMPLATE = """You are an expert LLM-as-a-judge evaluating compliance answers based on the official EU AI Act.
You will be given a User Query, a Gold Standard reference answer, and an Assistant Answer.
Your task is to rate the Assistant Answer on three metrics:
1. "accuracy" (1-5): How factually accurate is the Assistant Answer compared to the Gold Standard reference answer?
2. "completeness" (1-5): Does the Assistant Answer cover all the core points and requirements mentioned in the Gold Standard reference answer?
3. "groundedness" (1-5): Is the Assistant Answer grounded in the EU AI Act (does it cite specific Articles, and does not hallucinate or make unsupported claims)?

Note: refusing to answer for lack of authoritative support is a legitimate response in this
domain. Score such an answer on its own terms rather than treating the refusal as a factual error.

Your response must be a strict JSON object with exactly the following keys:
{{
  "accuracy": <integer 1-5>,
  "completeness": <integer 1-5>,
  "groundedness": <integer 1-5>,
  "rationale": "<a single-line rationale summarizing the evaluation>"
}}

Do not include any other text, notes, markdown formatting, or explanations. Respond with ONLY the JSON object.

User Query:
{query}

Gold Standard reference answer:
{gold}

Assistant Answer:
{answer}

JSON Response:"""


def build_prompt(query: str, gold: str, answer: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(query=query, gold=gold, answer=answer)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


def _call_ollama(prompt: str) -> tuple:
    """Return (content, prompt_tokens, completion_tokens)."""
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)
    response = llm.invoke(prompt)
    meta = getattr(response, "response_metadata", {}) or {}
    return (
        response.content.strip(),
        int(meta.get("prompt_eval_count", 0) or 0),
        int(meta.get("eval_count", 0) or 0),
    )


def _call_hosted(prompt: str, provider: str, timeout: int = 90) -> tuple:
    """Call a hosted OpenAI-compatible chat completions API (openai or deepseek).

    Returns (content, prompt_tok, completion_tok).
    """
    import requests

    info = _provider_config(provider)
    api_key = info["api_key"]
    base_url = info["base_url"]
    key_env = info["key_env"]
    models_url = info["models_url"]

    if not api_key:
        raise RuntimeError(
            f"{key_env} is not set. Add it to legalai/.env as {key_env}=sk-..."
        )
    if not os.getenv("JUDGE_MODEL"):
        raise RuntimeError(
            f"JUDGE_MODEL must be set explicitly for the {provider} provider (model ids change). "
            "List what your key can use with:\n"
            f"  curl {models_url} -H \"Authorization: Bearer $env:{key_env}\""
        )

    check_budget()

    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(4):
        response = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {}) or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            record_spend(prompt_tokens, completion_tokens)
            return content, prompt_tokens, completion_tokens

        body = response.text[:400]

        # Some newer models reject a non-default temperature; drop it and retry once.
        if response.status_code == 400 and "temperature" in body.lower() and "temperature" in payload:
            print("[judge] Model rejected 'temperature'; retrying without it.")
            payload.pop("temperature")
            continue

        if response.status_code in (401, 403):
            raise RuntimeError(f"{provider} auth failed ({response.status_code}): {body}")

        if response.status_code == 404:
            raise RuntimeError(
                f"Model '{JUDGE_MODEL}' not found for this key ({body}). List available models:\n"
                f"  curl {models_url} -H \"Authorization: Bearer $env:{key_env}\""
            )

        if response.status_code == 429 or response.status_code >= 500:
            wait = 2 ** attempt * 2
            print(f"[judge] HTTP {response.status_code}; retrying in {wait}s")
            last_error = f"HTTP {response.status_code}: {body}"
            time.sleep(wait)
            continue

        raise RuntimeError(f"{provider} request failed ({response.status_code}): {body}")

    raise RuntimeError(f"{provider} request failed after retries: {last_error}")


def call_provider(prompt: str) -> tuple:
    if JUDGE_PROVIDER in HOSTED_PROVIDERS:
        return _call_hosted(prompt, JUDGE_PROVIDER)
    if JUDGE_PROVIDER == "ollama":
        return _call_ollama(prompt)
    raise RuntimeError(
        f"Unknown JUDGE_PROVIDER '{JUDGE_PROVIDER}' (expected ollama, openai, or deepseek)"
    )


# --------------------------------------------------------------------------- #
# Judging
# --------------------------------------------------------------------------- #


def _parse_scores(content: str) -> dict:
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    # Tolerate a model that wraps the object in prose.
    if not content.lstrip().startswith("{"):
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)
    return json.loads(content)


def judge(query: str, gold: str, answer: str, retries: int = 3) -> dict:
    """Score one answer against the reference.

    Returns a dict with accuracy/completeness/groundedness (1-5 or None),
    rationale, judge_model and an `ok` flag. `ok=False` means the judge call
    failed and the row must be excluded from judge statistics, never scored as 1.
    """
    if not gold or not answer:
        return {
            "accuracy": 1,
            "completeness": 1,
            "groundedness": 1,
            "rationale": "Empty gold reference or empty response.",
            "judge_model": JUDGE_MODEL,
            "ok": True,
        }

    key = get_cache_key(query, gold, answer)
    with _state_lock:
        if key in _cache:
            return _cache[key]

    prompt = build_prompt(query, gold, answer)

    for attempt in range(retries):
        try:
            content, _prompt_tokens, _completion_tokens = call_provider(prompt)
            data = _parse_scores(content)

            result = {
                "accuracy": max(1, min(5, int(data.get("accuracy", 1)))),
                "completeness": max(1, min(5, int(data.get("completeness", 1)))),
                "groundedness": max(1, min(5, int(data.get("groundedness", 1)))),
                "rationale": str(data.get("rationale", "No rationale supplied.")).strip(),
                "judge_model": JUDGE_MODEL,
                "judge_provider": JUDGE_PROVIDER,
                "ok": True,
            }

            with _state_lock:
                _cache[key] = result
                save_cache()
            return result

        except RuntimeError as exc:
            # Configuration, auth or budget problems will not fix themselves.
            print(f"[judge] Fatal: {exc}")
            raise

        except Exception as exc:
            print(f"[judge] Retry {attempt + 1}/{retries} failed to score response: {exc}")
            if attempt == retries - 1:
                # Not cached, and flagged so the analysis excludes it.
                return {
                    "accuracy": None,
                    "completeness": None,
                    "groundedness": None,
                    "rationale": f"Failed to get valid judge output: {exc}",
                    "judge_model": JUDGE_MODEL,
                    "judge_provider": JUDGE_PROVIDER,
                    "ok": False,
                }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def print_spend():
    spend = load_spend()
    print(f"Provider        : {JUDGE_PROVIDER}")
    print(f"Model           : {JUDGE_MODEL}")
    print(f"Calls billed    : {spend.get('calls', 0)}")
    print(f"Prompt tokens   : {spend.get('prompt_tokens', 0):,}")
    print(f"Completion tok. : {spend.get('completion_tokens', 0):,}")
    print(f"Spent           : ${spend.get('total_usd', 0.0):.4f} of ${BUDGET_USD:.2f} budget")
    print(f"Cached scores   : {len(_cache)}")
    print(f"Prices used     : ${PRICE_IN_PER_M}/M in, ${PRICE_OUT_PER_M}/M out")
    print("(Prices are for budget tracking only - verify them against current pricing.)")


def preflight_check():
    """One live call, to validate configuration before spending on 1200."""
    print(f"[judge] Provider={JUDGE_PROVIDER} model={JUDGE_MODEL}")
    if JUDGE_PROVIDER in HOSTED_PROVIDERS:
        info = _provider_config(JUDGE_PROVIDER)
        print(f"[judge] {info['key_env']} {'present' if info['api_key'] else 'MISSING'}")

    query = "What is a high-risk AI system under the EU AI Act?"
    gold = (
        "High-risk AI systems are those listed in Annex III or used as safety components "
        "of products covered by Union harmonisation legislation, per Article 6."
    )
    answer = (
        "Under Article 6, an AI system is high-risk if it is a safety component of a "
        "regulated product or falls within the Annex III use cases, such as biometric "
        "identification or employment decisions."
    )

    started = time.perf_counter()
    content, prompt_tokens, completion_tokens = call_provider(build_prompt(query, gold, answer))
    elapsed = time.perf_counter() - started

    print(
        f"[judge] Responded in {elapsed:.1f}s, {prompt_tokens} prompt / "
        f"{completion_tokens} completion tokens"
    )
    try:
        scores = _parse_scores(content)
        print(f"[judge] Parsed OK: {scores}")
    except Exception as exc:
        print(f"[judge] PARSE FAILED: {exc}\nRaw output:\n{content[:500]}")
        print("[judge] The model is not returning strict JSON. Try another model.")
        return

    per_call = (prompt_tokens / 1e6) * PRICE_IN_PER_M + (completion_tokens / 1e6) * PRICE_OUT_PER_M
    print(f"\n[judge] Cost this call : ${per_call:.6f}")
    for n in (240, 720, 1200):
        print(f"[judge] Projected {n:>4} calls: ${per_call * n:.2f}")
    print()
    print_spend()


def validate_agreement(csv_path: str):
    """Print agreement metrics between human and judge scores."""
    import csv

    path = Path(csv_path)
    if not path.exists():
        print(f"Error: validation file not found at {path}")
        sys.exit(1)

    print(f"[judge] Validating against human scores in {path}...")

    with open(path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        print("Error: empty CSV file")
        sys.exit(1)

    judge_scores = {"accuracy": [], "completeness": [], "groundedness": []}
    human_scores = {"accuracy": [], "completeness": [], "groundedness": []}
    skipped = 0

    for index, row in enumerate(rows):
        try:
            humans = {
                "accuracy": float(row["human_accuracy"]),
                "completeness": float(row["human_completeness"]),
                "groundedness": float(row["human_groundedness"]),
            }
        except KeyError as exc:
            print(f"Error: CSV row {index} missing column: {exc}")
            print("Ensure human_accuracy, human_completeness, human_groundedness exist.")
            sys.exit(1)
        except (TypeError, ValueError):
            skipped += 1
            continue

        print(f"Scoring row {index + 1}/{len(rows)}...")
        result = judge(row.get("query", ""), row.get("gold", ""), row.get("answer", ""))
        if not result.get("ok"):
            skipped += 1
            continue

        for dimension in judge_scores:
            judge_scores[dimension].append(result[dimension])
            human_scores[dimension].append(humans[dimension])

    n = len(judge_scores["accuracy"])
    if n == 0:
        print("No comparable rows (did you fill in the human_* columns?)")
        sys.exit(1)

    def mae(a, b):
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    def pearson(a, b):
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
        var_a = sum((x - mean_a) ** 2 for x in a) ** 0.5
        var_b = sum((y - mean_b) ** 2 for y in b) ** 0.5
        return cov / (var_a * var_b) if var_a and var_b else float("nan")

    print("\n=== Judge validation ===")
    print(f"Judge      : {JUDGE_PROVIDER}/{JUDGE_MODEL}")
    print(f"Sample size: {n} ({skipped} rows skipped)")
    total_mae = 0.0
    for dimension in ("accuracy", "completeness", "groundedness"):
        dim_mae = mae(judge_scores[dimension], human_scores[dimension])
        corr = pearson(judge_scores[dimension], human_scores[dimension])
        total_mae += dim_mae
        print(f"{dimension:14s} MAE={dim_mae:.3f}  r={corr:+.3f}")
    print(f"{'overall':14s} MAE={total_mae / 3:.3f}")
    print(
        "\nReport this in the paper. MAE below ~0.5 on a 1-5 scale is good agreement; "
        "above ~1.0 means the judge is not measuring what you are."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", type=str, help="CSV of human-scored answers")
    parser.add_argument("--check", action="store_true", help="one live call to validate setup")
    parser.add_argument("--spend", action="store_true", help="show cumulative judge spend")
    args = parser.parse_args()

    if args.check:
        preflight_check()
    elif args.spend:
        print_spend()
    elif args.validate:
        validate_agreement(args.validate)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
