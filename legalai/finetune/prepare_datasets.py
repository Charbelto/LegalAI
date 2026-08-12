"""Build the three per-domain supervised fine-tuning sets.

    python finetune/prepare_datasets.py                  # all three
    python finetune/prepare_datasets.py --domain legal   # just one
    python finetune/prepare_datasets.py --max-examples 3000

Each domain becomes a JSONL file of {"instruction", "input", "output"} records
under finetune/data/, which train_qlora.py renders through the target model's
own chat template.

Domains, and why these datasets
-------------------------------
legal   LegalBench (nguha/legalbench). Already cited in the paper's related
        work, so no new citation to verify. 162 tasks; a stratified sample
        across tasks is taken rather than all of one task, so the legal expert
        learns legal-reasoning register broadly instead of memorising one
        task's answer format.

        IMPORTANT, measured rather than assumed: LegalBench is a short-answer
        *evaluation* benchmark. 105 of the 112 loadable tasks have mean answers
        of 10 characters or fewer; the median answer is 3 characters ("Yes",
        "No", "UCC") and only 42 of 2375 sampled rows exceed 40 characters.
        Training the legal expert directly on those targets teaches it to emit
        bare labels, which fights the IRAC-structured, article-citing response
        the deployed legal agent is prompted for (config.LEGAL_PROMPT) and is
        scored against by the LLM judge. The PEFT arm would then score WORSE
        than its own base control for a reason that has nothing to do with
        specialisation, and the paper would be reporting a format artefact as a
        finding.

        --legal-format irac (default) therefore renders each answer into the
        system's own response structure. Every field is filled from real dataset
        content: the task's question, the facts it supplies, and its
        ground-truth determination. Nothing is invented and no model generates
        any part of a training target. --legal-format raw reproduces the bare
        label for comparison. The manifest records which was used, and the
        paper's fine-tuning protocol has to state it.

news    NewsQA (Trischler et al. 2017). The canonical `Maluuba/newsqa` repo is
        a loader script that requires manually downloading Microsoft's
        newsqa-data-v1.csv plus the CNN stories, so the default here is
        `lucadiliello/newsqa`, a directly-loadable preprocessed copy of the
        same data. --news-dataset switches back to the official path if you
        have the manual files. The paper cites Trischler et al. either way;
        which mirror was used is recorded in the manifest.

general Dolly-15k (databricks/databricks-dolly-15k). Human-written, CC BY-SA
        3.0 - chosen over Alpaca specifically to avoid training on another
        company's model outputs and the licensing footnote that implies.

Leakage control
---------------
The pivot plan asks explicitly that no query or source text be shared between
the legal and news sets, since both touch AI-regulation-adjacent current
events. Enforced two ways, both reported in the manifest:
  * exact dedupe within each domain, by normalised instruction+input hash;
  * cross-domain overlap removal - any news record whose normalised text
    collides with a legal record (exact hash, or >=0.6 token-Jaccard against a
    legal record sharing a rare token) is dropped from news, not from legal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"

# Per domain. The plan proposed 2000-3000; 1500 is inside the spirit of that
# range and was chosen after measuring training cost on the target hardware
# (2500 examples at 1024 tokens ran ~4 h per adapter, ~11 h for three). Applied
# identically to all three domains - an uneven per-domain count would mean one
# expert was trained harder than another, which confounds the topology
# comparison. See the consistency note in train_qlora.py.
DEFAULT_MAX_EXAMPLES = 1500
DEFAULT_SEED = 20260803
JACCARD_THRESHOLD = 0.60
MIN_OUTPUT_CHARS = 2

# Character budget for the `input` field of a training example.
#
# This is sized against the TRAINING sequence limit (train_qlora.py's
# --max-seq-length, 512 tokens), not the serving context. Getting that backwards
# caused a real failure: at 6000 characters, 83% of NewsQA examples rendered past
# 512 tokens, and because the article precedes the question and answer, the
# tokenizer's right-truncation removed the target from most of the training set.
# The news adapter consequently trained on article prefixes with no answer and
# its held-out loss never moved (1.920 -> 1.956 across three epochs, token
# accuracy static at 0.568). It looked like a converged run, not a broken one.
#
# At roughly 3.5 characters per token, 2800 characters of input plus a ~450
# character target sits inside a 1024-token training window. Applied to all three
# domains, because an uneven truncation rate across experts (once 83% / 22% / 9%)
# is itself an inconsistency in effective training budget.
#
# Raised from 1400 alongside the move to 1024-token training. The two must move
# together: the cap exists to keep examples inside the trainer's sequence limit,
# and deliberately filling more of that window is the point here rather than a
# side effect. Adapters fitted only on short sequences generated degenerate text
# when served 1400-4500-token prompts, so training examples that reach further
# toward real prompt lengths are part of the fix.
MAX_INPUT_CHARS = 2800

# Verified after generation rather than assumed - see _report_token_lengths().
# Keep in step with train_qlora.py's --max-seq-length.
DEFAULT_TRAIN_SEQ_TOKENS = 1024

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


# --------------------------------------------------------------------------- #
# Text normalisation / dedupe helpers
# --------------------------------------------------------------------------- #


def _norm(text: str) -> str:
    lowered = _NON_ALNUM.sub(" ", str(text or "").lower())
    return _WS.sub(" ", lowered).strip()


def _hash(record: Dict[str, str]) -> str:
    key = _norm(record.get("instruction", "")) + "||" + _norm(record.get("input", ""))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _tokens(record: Dict[str, str]) -> set:
    return set(_norm(record.get("instruction", "") + " " + record.get("input", "")).split())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _valid(record: Dict[str, str]) -> bool:
    if not str(record.get("instruction", "")).strip():
        return False
    if len(str(record.get("output", "")).strip()) < MIN_OUTPUT_CHARS:
        return False
    if len(str(record.get("input", ""))) > MAX_INPUT_CHARS:
        return False
    return True


def _dedupe(records: Iterable[Dict[str, str]]) -> tuple[List[Dict[str, str]], int]:
    seen = set()
    kept = []
    dropped = 0
    for record in records:
        digest = _hash(record)
        if digest in seen:
            dropped += 1
            continue
        seen.add(digest)
        kept.append(record)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _require_datasets():
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "This script needs the HF datasets library:\n"
            "    pip install -r requirements-finetune.txt\n"
            f"(import failed: {exc})"
        ) from exc
    return datasets


def _format_legal_output(task: str, question: str, facts: str, answer: str, style: str) -> str:
    """Render a LegalBench answer in the response format the agent is deployed in.

    Only real dataset content is used - the task name, the question text, the
    facts the task supplies, and its ground-truth answer. The structure mirrors
    the Issue/Rule/Application/Conclusion skeleton of config.LEGAL_PROMPT so the
    adapter reinforces that format instead of overriding it with a bare label.

    Fields the dataset genuinely does not provide (a statutory citation, a
    confidence justification) are deliberately OMITTED rather than filled with
    plausible-looking text. Inventing them would be training the model to
    hallucinate citations, which is the single worst failure mode for a legal
    assistant.
    """
    if style == "raw":
        return answer

    lines = [f"**Issue**: {question.strip()}" if question.strip() else f"**Issue**: {task.replace('_', ' ')}"]
    if facts.strip():
        # Truncated: the facts are already in the prompt, this section exists to
        # teach the model to restate what it relied on, not to memorise inputs.
        condensed = _WS.sub(" ", facts.strip())[:400]
        lines.append(f"**Application**: {condensed}")
    lines.append(f"**Conclusion**: {answer.strip()}")
    return "\n".join(lines)


def load_legalbench(
    max_examples: int, rng: random.Random, legal_format: str = "irac"
) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Stratified sample across LegalBench tasks.

    LegalBench is 162 separate configs. Loading one gives a model that learns
    one narrow answer format; the point of a "legal expert" is breadth of legal
    register, so examples are drawn round-robin across as many tasks as load
    successfully. Each task supplies its own natural-language instruction in
    `base_prompt`, which is exactly the supervision signal wanted here.
    """
    datasets = _require_datasets()
    from datasets import get_dataset_config_names

    print("[prepare] enumerating LegalBench tasks...")
    configs = list(get_dataset_config_names("nguha/legalbench"))
    rng.shuffle(configs)
    print(f"[prepare]   {len(configs)} tasks available")

    per_task: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    tasks_loaded, tasks_failed = [], []
    # Round-robin needs a pool from many tasks; cap per-task pull so one huge
    # task cannot dominate, and stop enumerating once there is plenty to sample.
    per_task_cap = max(8, (max_examples // 40) + 8)

    for name in configs:
        if sum(len(v) for v in per_task.values()) >= max_examples * 3:
            break
        try:
            split = datasets.load_dataset("nguha/legalbench", name, split="test")
        except Exception as exc:
            tasks_failed.append({"task": name, "error": str(exc)[:200]})
            continue

        columns = set(split.column_names)
        if "answer" not in columns:
            tasks_failed.append({"task": name, "error": "no 'answer' column"})
            continue
        # Everything except the answer and bookkeeping columns is the question.
        text_columns = [c for c in split.column_names if c not in {"answer", "index"}]

        rows = []
        for row in split.select(range(min(len(split), per_task_cap))):
            instruction = (
                f"You are a legal expert. Complete the following legal reasoning task "
                f"({name.replace('_', ' ')}). Structure your answer with the issue, "
                f"how the rule applies to the facts, and your conclusion."
            )
            body_parts = []
            for column in text_columns:
                value = str(row.get(column) or "").strip()
                if value:
                    body_parts.append(f"{column.replace('_', ' ').title()}: {value}")
            answer = str(row.get("answer") or "").strip()
            body = "\n".join(body_parts)
            # LegalBench tasks put the question in a 'question' column when they
            # have one; otherwise the task's own base_prompt is the question, and
            # the task name is the best available stand-in.
            question = str(row.get("question") or "").strip()
            record = {
                "instruction": instruction,
                "input": body,
                "output": _format_legal_output(name, question, body, answer, legal_format),
                "source": f"legalbench/{name}",
                "raw_answer": answer,
            }
            if _valid(record):
                rows.append(record)
        if rows:
            per_task[name] = rows
            tasks_loaded.append(name)

    # Round-robin across tasks so the sample is spread, not front-loaded.
    pools = {k: list(v) for k, v in per_task.items()}
    for pool in pools.values():
        rng.shuffle(pool)
    selected: List[Dict[str, str]] = []
    while len(selected) < max_examples and any(pools.values()):
        for name in list(pools.keys()):
            if not pools[name]:
                del pools[name]
                continue
            selected.append(pools[name].pop())
            if len(selected) >= max_examples:
                break

    meta = {
        "hf_id": "nguha/legalbench",
        "tasks_available": len(configs),
        "tasks_used": len(tasks_loaded),
        "tasks_failed": len(tasks_failed),
        "tasks_failed_detail": tasks_failed[:10],
        "sampling": "round-robin across tasks, test split",
        "answer_format": legal_format,
        "mean_raw_answer_chars": (
            round(sum(len(r.get("raw_answer", "")) for r in selected) / len(selected), 1)
            if selected
            else 0
        ),
    }
    return selected, meta


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _window_around_answer(context: str, answer: str, budget: int) -> str:
    """Return at most `budget` characters of `context` containing `answer`.

    Keeps the answer roughly centred so the model sees the evidence on both
    sides. Falls back to the opening of the article when the answer cannot be
    located, which is the best available option but is why
    _format_news_output() re-checks locatability against the windowed text
    rather than the full article.
    """
    context = context.strip()
    if len(context) <= budget:
        return context

    position = context.find(answer)
    if position < 0:
        position = context.lower().find(answer.lower())
    if position < 0:
        return context[:budget]

    half = max(0, (budget - len(answer)) // 2)
    start = max(0, position - half)
    end = min(len(context), start + budget)
    start = max(0, end - budget)

    snippet = context[start:end]
    # Trim partial words at the edges so the model is not shown fragments.
    if start > 0:
        snippet = snippet.partition(" ")[2]
        snippet = "... " + snippet
    if end < len(context):
        snippet = snippet.rpartition(" ")[0] + " ..."
    return snippet.strip()


def _format_news_output(answer: str, context: str, style: str) -> str:
    """Answer span, plus the article sentence that contains it.

    NewsQA answers are extractive spans and average 25 characters, which is the
    same format-mismatch problem LegalBench has in a milder form: an adapter
    trained on bare spans answers in bare spans, and the LLM judge scores that as
    incomplete on a task whose prompt asks for dates, sources and key facts.

    The fix uses only text that is already in the dataset - the answer span and
    the sentence of the source article containing it. It gives the news expert
    the habit of quoting its supporting evidence, which is precisely what
    config.NEWS_PROMPT asks for, and it raises target length without inventing a
    single word. When the span cannot be located in the article, the bare answer
    is kept rather than attaching an unrelated sentence.
    """
    answer = answer.strip()
    if style == "raw" or not answer:
        return answer

    position = context.find(answer)
    if position < 0:
        # Case-insensitive retry before giving up.
        lowered = context.lower().find(answer.lower())
        position = lowered
    if position < 0:
        return answer

    # Widen to the enclosing sentence.
    start = max(context.rfind(".", 0, position), context.rfind("\n", 0, position)) + 1
    end_candidates = [context.find(mark, position + len(answer)) for mark in (".", "!", "?", "\n")]
    end_candidates = [e for e in end_candidates if e > 0]
    end = min(end_candidates) + 1 if end_candidates else min(len(context), position + len(answer) + 200)
    sentence = _WS.sub(" ", context[start:end]).strip()

    if not sentence or sentence == answer:
        return answer
    return f"{answer}\n\nSupporting detail from the article: \"{sentence}\""


def load_newsqa(
    max_examples: int, rng: random.Random, hf_id: str, news_format: str = "grounded"
) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
    """NewsQA: answer a question about a news article, grounded in that article."""
    datasets = _require_datasets()
    print(f"[prepare] loading NewsQA from {hf_id}...")

    last_error = None
    dataset = None
    for split in ("train", "validation", "test"):
        try:
            dataset = datasets.load_dataset(hf_id, split=split)
            print(f"[prepare]   using split={split} ({len(dataset)} rows)")
            break
        except Exception as exc:
            last_error = exc
    if dataset is None:
        raise SystemExit(
            f"Could not load {hf_id}: {last_error}\n"
            "The official Maluuba/newsqa needs a manual download of "
            "newsqa-data-v1.csv plus the CNN stories. Pass "
            "--news-dataset lucadiliello/newsqa for the preprocessed mirror."
        )

    columns = set(dataset.column_names)
    context_key = next((k for k in ("context", "story_text", "text", "passage") if k in columns), None)
    question_key = next((k for k in ("question", "questions") if k in columns), None)
    answer_key = next((k for k in ("answers", "answer", "labels") if k in columns), None)
    if not (context_key and question_key and answer_key):
        raise SystemExit(
            f"{hf_id} has unexpected columns {sorted(columns)}; "
            "expected a context, a question and an answer field."
        )

    def _first_answer(value: Any) -> str:
        """NewsQA answers arrive as SQuAD-style dicts, lists, or bare strings."""
        if isinstance(value, dict):
            for key in ("text", "answer_start", "value"):
                inner = value.get(key)
                if isinstance(inner, list) and inner:
                    return str(inner[0])
                if isinstance(inner, str) and inner.strip():
                    return inner
            return ""
        if isinstance(value, (list, tuple)):
            return _first_answer(value[0]) if value else ""
        return str(value or "")

    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    # Oversample: many NewsQA rows are unanswerable or have empty spans.
    indices = indices[: max_examples * 4]

    records = []
    for index in indices:
        row = dataset[index]
        context = str(row.get(context_key) or "").strip()
        question = _first_answer(row.get(question_key)) if not isinstance(row.get(question_key), str) else str(row.get(question_key))
        answer = _first_answer(row.get(answer_key)).strip()
        if not (context and question and answer):
            continue
        # Window the article around the answer rather than taking its opening
        # characters. A blind head-truncation frequently cuts away the very
        # sentence the answer comes from, which asks the model to produce a span
        # that is no longer in its context - training it to guess. The window
        # keeps the answer and its surrounding sentences.
        windowed = _window_around_answer(context, answer, MAX_INPUT_CHARS - 200)
        record = {
            "instruction": (
                "You are a news analyst. Answer the question using only the news "
                "article provided. Be specific about dates, organisations and facts."
            ),
            "input": f"Article:\n{windowed}\n\nQuestion: {question.strip()}",
            "output": _format_news_output(answer, windowed, news_format),
            "source": hf_id,
            "raw_answer": answer,
        }
        if _valid(record):
            records.append(record)
        if len(records) >= max_examples:
            break

    meta = {
        "hf_id": hf_id,
        "rows_available": len(dataset),
        "columns": sorted(columns),
        "answer_format": news_format,
        "mean_raw_answer_chars": (
            round(sum(len(r.get("raw_answer", "")) for r in records) / len(records), 1)
            if records
            else 0
        ),
        "note": (
            "preprocessed mirror of Trischler et al. (2017) NewsQA"
            if hf_id != "Maluuba/newsqa"
            else "official Maluuba loader (requires manual source files)"
        ),
    }
    return records, meta


def load_dolly(
    max_examples: int, rng: random.Random, context_fraction: float = 0.7
) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Dolly-15k: general instruction following, human-written.

    `context_fraction` biases sampling toward examples that HAVE a context field
    (Dolly's closed_qa, information_extraction and summarization categories),
    rather than taking a uniform sample.

    This is a fix for a measured failure, not a preference. Uniform sampling gave
    only 27% context-bearing examples and a median rendered length of 175 tokens,
    against 298 for legal and 642 for news. The general expert is served
    1368-token prompts in the graph - instruction plus retrieved context - so its
    adapter was fitted almost entirely on a shape it never meets in production.
    The result was not mild: at 1368 tokens the adapter emitted invented
    non-words and ran to the full 1024-token cap on every call (213s), while its
    own base model answered correctly in 19s. The other two adapters, whose
    training examples are 2-4x longer, are unaffected.

    Biasing toward context-bearing examples raises the median without leaving
    Dolly, so the dataset and its citation are unchanged and the training budget
    (epochs, learning rate, sequence limit, example count) stays identical across
    all three experts. It equalises the general expert's example shape with the
    other two rather than giving it any advantage.
    """
    datasets = _require_datasets()
    print(f"[prepare] loading Dolly-15k (context_fraction={context_fraction})...")
    dataset = datasets.load_dataset("databricks/databricks-dolly-15k", split="train")

    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    # Partition first, then fill the two quotas, so the ratio is controlled rather
    # than left to whatever the shuffle happened to produce.
    with_context, without_context = [], []
    for index in indices:
        row = dataset[index]
        record = {
            "instruction": str(row.get("instruction") or "").strip(),
            "input": str(row.get("context") or "").strip()[:MAX_INPUT_CHARS],
            "output": str(row.get("response") or "").strip(),
            "source": f"dolly15k/{row.get('category', 'unknown')}",
        }
        if not _valid(record):
            continue
        (with_context if record["input"] else without_context).append(record)

    want_context = int(max_examples * context_fraction)
    chosen = with_context[:want_context]
    chosen += without_context[: max_examples - len(chosen)]
    # If Dolly runs short of context-bearing rows, top up from the rest rather
    # than returning fewer examples than the other domains get.
    if len(chosen) < max_examples:
        chosen += with_context[len(chosen) : max_examples]
    rng.shuffle(chosen)

    categories: Dict[str, int] = defaultdict(int)
    for record in chosen:
        categories[record["source"].split("/")[-1]] += 1

    meta = {
        "hf_id": "databricks/databricks-dolly-15k",
        "rows_available": len(dataset),
        "license": "CC BY-SA 3.0",
        "context_fraction_requested": context_fraction,
        "context_bearing_available": len(with_context),
        "context_bearing_selected": sum(1 for r in chosen if r["input"]),
        "categories": dict(sorted(categories.items(), key=lambda kv: -kv[1])),
    }
    return chosen, meta


# --------------------------------------------------------------------------- #
# Cross-domain leakage control
# --------------------------------------------------------------------------- #


def remove_cross_domain_overlap(
    legal: List[Dict[str, str]], news: List[Dict[str, str]]
) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Drop news records that overlap the legal set.

    Direction matters: the legal expert's dataset is the one the paper's central
    claim rests on, so overlap is removed from news. Two passes - exact
    normalised hash, then token-Jaccard against legal records that share a rare
    token (an inverted index on the rarest term keeps this from being a 2500x2500
    all-pairs comparison).
    """
    legal_hashes = {_hash(record) for record in legal}
    legal_tokens = [_tokens(record) for record in legal]

    # Inverted index: token -> legal record indices, restricted to tokens that
    # are not ubiquitous (a token in most records tells us nothing).
    postings: Dict[str, List[int]] = defaultdict(list)
    for index, tokens in enumerate(legal_tokens):
        for token in tokens:
            postings[token].append(index)
    common_cutoff = max(2, len(legal_tokens) // 10)
    rare_postings = {t: idxs for t, idxs in postings.items() if len(idxs) <= common_cutoff}

    kept, dropped_exact, dropped_fuzzy = [], 0, 0
    for record in news:
        if _hash(record) in legal_hashes:
            dropped_exact += 1
            continue
        tokens = _tokens(record)
        candidates = set()
        for token in tokens:
            candidates.update(rare_postings.get(token, ()))
        if any(_jaccard(tokens, legal_tokens[i]) >= JACCARD_THRESHOLD for i in candidates):
            dropped_fuzzy += 1
            continue
        kept.append(record)

    report = {
        "news_dropped_exact_match": dropped_exact,
        "news_dropped_near_duplicate": dropped_fuzzy,
        "jaccard_threshold": JACCARD_THRESHOLD,
        "news_kept": len(kept),
    }
    return kept, report


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _report_token_lengths(domain: str, records: List[Dict[str, str]], seq_tokens: int):
    """Tokenise the rendered examples and report how many exceed the training limit.

    This exists because character-level caps are only a proxy. The failure this
    guards against is silent and expensive: examples longer than the trainer's
    sequence limit are right-truncated, so for any dataset whose target sits at
    the END of the example (all three of ours) the model is trained on an input
    with no answer attached. Loss then plateaus at a plausible-looking value and
    the run reports success. Measuring here, with each domain's own tokenizer and
    the trainer's own rendering function, is the only way to know.

    Returns None when the tokenizer cannot be loaded (offline, say) rather than
    blocking dataset preparation.
    """
    role_for_domain = {"legal": "legal", "news": "news", "general": "general_qa"}
    role = role_for_domain.get(domain)
    if role is None:
        return None
    try:
        from transformers import AutoTokenizer

        sys.path.insert(0, str(ROOT_DIR))
        import config as project_config
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_train_qlora", Path(__file__).resolve().parent / "train_qlora.py"
        )
        trainer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer_module)

        model_id = project_config.LOCAL_PEFT_ROLES[role]["base_model"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        print(f"[prepare] {domain}: could not measure token lengths ({exc})")
        return None

    lengths = sorted(
        len(tokenizer(trainer_module.render_example(record, tokenizer))["input_ids"])
        for record in records
    )
    if not lengths:
        return None
    over = sum(1 for value in lengths if value > seq_tokens)
    stats = {
        "tokenizer": model_id,
        "train_seq_tokens": seq_tokens,
        "median_tokens": lengths[len(lengths) // 2],
        "p95_tokens": lengths[int(0.95 * len(lengths))],
        "max_tokens": lengths[-1],
        "examples_over_limit": over,
        "pct_over_limit": round(100.0 * over / len(lengths), 1),
    }
    print(
        f"[prepare] {domain:8s} rendered tokens: median={stats['median_tokens']} "
        f"p95={stats['p95_tokens']} max={stats['max_tokens']} | "
        f"over {seq_tokens}: {over}/{len(lengths)} ({stats['pct_over_limit']}%)"
    )
    return stats


def _write_jsonl(path: Path, records: List[Dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--domain",
        choices=["legal", "news", "general", "all"],
        default="all",
        help="Which domain to build (default: all three).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help=f"Examples per domain (default {DEFAULT_MAX_EXAMPLES}).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--news-dataset",
        default="lucadiliello/newsqa",
        help="NewsQA source. Use Maluuba/newsqa only if you have the manual files.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Held-out fraction per domain, for the training loss curve.",
    )
    parser.add_argument(
        "--legal-format",
        choices=["irac", "raw"],
        default="irac",
        help="How LegalBench answers are rendered as training targets. 'irac' "
        "(default) wraps the real question/facts/answer in the system's own "
        "response structure; 'raw' emits the bare label, which measurably teaches "
        "the legal expert to answer in 3 characters. See the module docstring.",
    )
    parser.add_argument(
        "--general-context-fraction",
        type=float,
        default=0.7,
        help="Fraction of the general (Dolly) set that must carry a context field. "
        "Uniform sampling gave 27%% and a 175-token median, far shorter than the "
        "1368-token prompts the general expert is served, and its adapter degenerated "
        "at that length. See load_dolly().",
    )
    parser.add_argument(
        "--news-format",
        choices=["grounded", "raw"],
        default="grounded",
        help="How NewsQA answers are rendered. 'grounded' (default) appends the "
        "article sentence containing the answer span - real dataset text, and what "
        "config.NEWS_PROMPT asks the agent to do. 'raw' emits the bare 25-character "
        "span.",
    )
    parser.add_argument(
        "--train-seq-tokens",
        type=int,
        default=DEFAULT_TRAIN_SEQ_TOKENS,
        help="The sequence limit training will run at (train_qlora.py "
        "--max-seq-length). Used only to CHECK the generated examples fit; keep it "
        "in step with the trainer or the check measures the wrong thing.",
    )
    parser.add_argument(
        "--max-pct-over-seq-limit",
        type=float,
        default=10.0,
        help="Warn when more than this percentage of a domain's examples exceed "
        "the training sequence limit. Over-long examples are right-truncated, "
        "which removes the target and trains the model on an input with no answer.",
    )
    parser.add_argument(
        "--min-mean-output-chars",
        type=int,
        default=40,
        help="Warn when a domain's mean training target is shorter than this. The "
        "deployed agents are prompted for multi-sentence answers and judged "
        "against long gold references, so a very short mean target is a "
        "format mismatch that would surface as a fake quality finding.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    wanted = {"legal", "news", "general"} if args.domain == "all" else {args.domain}

    manifest: Dict[str, Any] = {
        "seed": args.seed,
        "max_examples_per_domain": args.max_examples,
        "val_fraction": args.val_fraction,
        "domains": {},
    }

    raw: Dict[str, List[Dict[str, str]]] = {}
    source_meta: Dict[str, Dict[str, Any]] = {}

    if "legal" in wanted:
        records, meta = load_legalbench(args.max_examples, rng, legal_format=args.legal_format)
        raw["legal"], source_meta["legal"] = records, meta
    if "news" in wanted:
        records, meta = load_newsqa(
            args.max_examples, rng, args.news_dataset, news_format=args.news_format
        )
        raw["news"], source_meta["news"] = records, meta
    if "general" in wanted:
        records, meta = load_dolly(
            args.max_examples, rng, context_fraction=args.general_context_fraction
        )
        raw["general"], source_meta["general"] = records, meta

    # Within-domain dedupe first, so the cross-domain pass works on clean sets.
    dedupe_report = {}
    for domain, records in raw.items():
        raw[domain], dropped = _dedupe(records)
        dedupe_report[domain] = {"duplicates_removed": dropped}

    # Cross-domain leakage: only meaningful when both sets are being built. When
    # only one is, say so in the manifest rather than implying a check ran.
    if "legal" in raw and "news" in raw:
        raw["news"], overlap_report = remove_cross_domain_overlap(raw["legal"], raw["news"])
        manifest["cross_domain_overlap"] = overlap_report
    else:
        manifest["cross_domain_overlap"] = {
            "status": "not run",
            "reason": "needs both legal and news in the same invocation; "
            "rebuild with --domain all to enforce it",
        }

    for domain, records in raw.items():
        rng.shuffle(records)
        n_val = max(1, int(len(records) * args.val_fraction)) if records else 0
        val, train = records[:n_val], records[n_val:]

        train_path = DATA_DIR / f"{domain}_train.jsonl"
        val_path = DATA_DIR / f"{domain}_val.jsonl"
        _write_jsonl(train_path, train)
        _write_jsonl(val_path, val)

        manifest["domains"][domain] = {
            "source": source_meta[domain],
            "dedupe": dedupe_report[domain],
            "train_examples": len(train),
            "val_examples": len(val),
            "train_file": train_path.name,
            "val_file": val_path.name,
            "mean_output_chars": (
                round(sum(len(r["output"]) for r in train) / len(train), 1) if train else 0
            ),
            "token_lengths": _report_token_lengths(domain, train, args.train_seq_tokens),
        }
        print(
            f"[prepare] {domain:8s} train={len(train):5d} val={len(val):4d} "
            f"-> {train_path.relative_to(ROOT_DIR)}"
        )

    # Format-mismatch guard. This is the check that would have caught the raw
    # LegalBench problem before three hours of GPU time: a domain whose mean
    # training target is a few characters long will produce an adapter that
    # answers in a few characters, no matter how good the training loss looks.
    manifest["target_length_check"] = {
        "min_mean_output_chars": args.min_mean_output_chars,
        "domains": {},
    }
    warned = []
    for domain, info in manifest["domains"].items():
        mean_chars = info["mean_output_chars"]
        ok = mean_chars >= args.min_mean_output_chars
        manifest["target_length_check"]["domains"][domain] = {
            "mean_output_chars": mean_chars,
            "ok": ok,
        }
        if not ok:
            warned.append((domain, mean_chars))

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[prepare] manifest -> {manifest_path.relative_to(ROOT_DIR)}")

    if manifest["cross_domain_overlap"].get("status") == "not run":
        print(
            "[prepare] NOTE cross-domain leakage check did not run (single domain). "
            "Rebuild with --domain all before training for the paper."
        )

    # Truncation guard. An uneven rate across domains is itself a problem: it
    # means the experts were effectively trained on different amounts of signal,
    # which confounds any later comparison between graph positions.
    truncation_warned = []
    for domain, info in manifest["domains"].items():
        stats = info.get("token_lengths")
        if not stats:
            continue
        if stats["pct_over_limit"] > args.max_pct_over_seq_limit:
            truncation_warned.append((domain, stats))

    for domain, stats in truncation_warned:
        print(
            f"[prepare] WARNING '{domain}': {stats['pct_over_limit']}% of examples exceed the "
            f"{stats['train_seq_tokens']}-token training limit (median "
            f"{stats['median_tokens']}, max {stats['max_tokens']}). Those are "
            f"right-truncated, which removes the TARGET and trains the model on an input "
            f"with no answer - held-out loss then plateaus and the run looks converged "
            f"when the adapter has learnt nothing. Shorten the inputs "
            f"(MAX_INPUT_CHARS) or raise --max-seq-length, then retrain ALL adapters."
        )
    if not truncation_warned and any(
        info.get("token_lengths") for info in manifest["domains"].values()
    ):
        rates = {
            d: i["token_lengths"]["pct_over_limit"]
            for d, i in manifest["domains"].items()
            if i.get("token_lengths")
        }
        print(f"[prepare] token-length check passed for all domains (over-limit rates: {rates})")

    for domain, mean_chars in warned:
        print(
            f"[prepare] WARNING '{domain}' targets average only {mean_chars} characters, "
            f"below the {args.min_mean_output_chars}-character floor. An adapter trained "
            f"on these will answer just as briefly, which the LLM judge will score as "
            f"incomplete - a format artefact that looks exactly like 'specialisation "
            f"made quality worse'. Check the manifest before training."
        )
    if warned:
        print(
            "[prepare] For the legal domain specifically, --legal-format irac is the "
            "intended fix; --legal-format raw reproduces the bare LegalBench labels."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
