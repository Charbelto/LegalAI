"""Check that each LoRA adapter actually changed its model's behaviour.

    python finetune/validate_adapters.py
    python finetune/validate_adapters.py --role legal --max-new-tokens 300

The pivot plan is explicit that each fine-tune "should be validated by generating
a handful of test answers before moving on ... rather than assuming the fine-tune
worked". This script does that: for each role it generates the same domain probes
twice - once with the adapter loaded, once from the identical base weights - and
prints both side by side, plus cheap quantitative signals of whether anything
moved.

It deliberately does NOT decide for you whether the fine-tune is good. It reports
divergence and domain-marker counts; reading the pairs is the actual check. A
diff rate near zero means the adapter is doing nothing and the "peft" arm would
be measuring the base models under a different label - which would silently
invalidate RQ2.

One model at a time: the adapter and base variants of the same architecture are
loaded and freed in sequence, so this needs the same VRAM as a single expert, not
all three.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

FINETUNE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FINETUNE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: E402
import local_models  # noqa: E402

REPORT_PATH = FINETUNE_DIR / "adapter_validation.json"

# Domain-appropriate probes. The legal ones deliberately ask for the kind of
# article-citing, structured answer the LegalBench adapter is supposed to
# encourage, since "does it cite articles more" is the concrete claim.
PROBES: Dict[str, List[str]] = {
    "legal": [
        "Under the EU AI Act, what conformity-assessment obligations apply to a provider "
        "of a high-risk AI system before placing it on the market?",
        "Is an AI system used to evaluate creditworthiness of natural persons high-risk "
        "under the EU AI Act? Explain your reasoning.",
        "What penalties may a member state impose for placing a prohibited AI practice on the market?",
    ],
    "news": [
        "Article:\nThe European Commission announced on Tuesday that it would delay parts of "
        "its digital rulebook by twelve months, citing industry readiness concerns. The "
        "delay affects transparency obligations originally due in August.\n\n"
        "Question: Why was the delay announced, and what does it affect?",
        "Article:\nRegulators in three countries opened a joint inquiry into a facial "
        "recognition vendor after reports it scraped images without consent. The vendor "
        "denies wrongdoing and says it will cooperate.\n\n"
        "Question: What triggered the inquiry?",
    ],
    "general_qa": [
        "What is the difference between a regulation and a directive in EU law?",
        "Explain what a vector database does, in two or three sentences.",
        "Write a short, polite reply declining a meeting invitation.",
    ],
}

BASE_LOSS_PATH = FINETUNE_DIR / "base_eval_loss.json"


def _load_base_losses() -> Dict[str, Dict]:
    """Base-model held-out losses from finetune/base_eval_loss.py, or {}."""
    if not BASE_LOSS_PATH.exists():
        return {}
    try:
        return json.loads(BASE_LOSS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


_ARTICLE_RE = re.compile(r"(?i)\barticle\s+\d+")
_STRUCTURE_RE = re.compile(r"(?i)\b(issue|rule|application|conclusion|sources|confidence)\s*[:*]")


def _signals(text: str) -> Dict[str, float]:
    words = text.split()
    return {
        "chars": len(text),
        "words": len(words),
        "article_citations": len(_ARTICLE_RE.findall(text)),
        "irac_markers": len(_STRUCTURE_RE.findall(text)),
        "type_token_ratio": round(len(set(w.lower() for w in words)) / len(words), 3) if words else 0.0,
    }


def _retrieval_context(query: str, n_docs: int = 5) -> str:
    """Real retrieved statutory context for a probe, or "" if unavailable.

    Probing with bare one-line questions is what let a broken set of adapters
    pass validation. Those probes are ~50 tokens; deployed expert prompts are
    1373-2141 and the aggregator's ~4500. The previous adapters looked merely
    terse at 50 tokens and produced ungrammatical, confabulated text at 1400 -
    a failure mode that is invisible unless the probe is realistically long.
    """
    try:
        sys.path.insert(0, str(ROOT_DIR))
        from agents.retrieval import RetrievalAgent

        state = RetrievalAgent().invoke(
            {"query": query, "route": "legal", "session_id": "validate",
             "retrieved_docs": [], "chat_history": []}
        )
        docs = state.get("retrieved_docs", [])[:n_docs]
        parts = []
        for index, doc in enumerate(docs, 1):
            content = getattr(doc, "page_content", None)
            if content is None and isinstance(doc, dict):
                content = doc.get("page_content") or doc.get("content") or ""
            parts.append(f"[Document {index}]:\n{content}\n")
        return "\n".join(parts)
    except Exception as exc:
        print(f"[validate] could not retrieve context ({exc}); probing without it")
        return ""


def _deployed_prompt(role: str, query: str, context: str) -> str:
    """Render a probe through the role's REAL agent prompt template.

    Not an approximation of a deployed prompt - the actual one. This distinction
    produced a wrong verdict: an earlier version wrapped probes in an improvised
    "You are a legal expert ... citing specific Articles" scaffold, and the legal
    adapter answered in 36 words where its base model used 213, which read as a
    regression. Through config.LEGAL_PROMPT, the same adapter produced 687 words
    of correctly-structured IRAC citing the right article, while the base model
    fabricated a citation.

    The adapters are trained on targets shaped like these templates, so they are
    format-sensitive by construction. Validating them under any other scaffold
    measures a condition they will never meet in production.
    """
    import utils as project_utils

    templates = {
        "legal": config.LEGAL_PROMPT,
        "news": config.NEWS_PROMPT,
        "general_qa": config.GENERAL_QA_PROMPT,
    }
    template = templates.get(role)
    if template is None:
        return query
    return template.format(
        context=context or "No relevant documents found.",
        query=query,
        chat_history="No previous conversation.",
        current_date=project_utils.get_current_date(),
    )


def _generate(role: str, use_adapters: bool, probes: List[str], max_new_tokens: int) -> List[str]:
    """Generate all probes for one role under one arm, then free the model."""
    # LOCAL_PEFT_USE_ADAPTERS is read by local_models at model-resolution time,
    # so flipping it here is what selects which variant gets loaded. Reset by the
    # caller; local_models caches on (base model, arm) so the two variants are
    # separate entries and cannot bleed into each other.
    original = config.LOCAL_PEFT_USE_ADAPTERS
    config.LOCAL_PEFT_USE_ADAPTERS = use_adapters
    chat = None
    try:
        chat = local_models.LocalChatModel(
            role=role, temperature=0.0, max_new_tokens=max_new_tokens
        )
        return [chat.invoke(probe).content for probe in probes]
    finally:
        config.LOCAL_PEFT_USE_ADAPTERS = original
        # Drop this frame's reference to the model BEFORE unloading. Order
        # matters: unload_all() clears the registry and calls
        # torch.cuda.empty_cache(), but `chat` (via chat._loaded.model) is still a
        # live local here, so the allocation survives and empty_cache() frees
        # nothing. This function is called six times - two arms x three roles -
        # so the leak accumulates ~12GB of 4-bit weights on an 8GB card and OOMs
        # partway through validation.
        del chat
        local_models.unload_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--role",
        choices=list(config.LOCAL_PEFT_ROLES.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--realistic-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap probes in real retrieved context so they approximate deployed "
        "prompt lengths (default). Short probes cannot reveal the long-context "
        "degeneration that a previous adapter set exhibited only at ~1400 tokens. "
        "Use --no-realistic-length for the old bare-question behaviour.",
    )
    parser.add_argument(
        "--min-diff-ratio",
        type=float,
        default=0.05,
        help="Below this mean text divergence, the adapter is flagged as inert.",
    )
    args = parser.parse_args()

    roles = list(config.LOCAL_PEFT_ROLES.keys()) if args.role == "all" else [args.role]
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_new_tokens": args.max_new_tokens,
        "min_diff_ratio": args.min_diff_ratio,
        "roles": [],
    }
    all_ok = True

    for role in roles:
        adapter_dir = local_models.adapter_path(role)
        if not (adapter_dir / "adapter_config.json").exists():
            print(f"[validate] {role}: NO ADAPTER at {adapter_dir} - train it first. Skipping.")
            report["roles"].append(
                {"role": role, "status": "missing_adapter", "adapter_dir": str(adapter_dir)}
            )
            all_ok = False
            continue

        probes = PROBES[role]
        if args.realistic_length:
            # Every role is rendered through its OWN agent template, so each
            # adapter is judged in the format it is actually served in. Retrieved
            # context is fetched per probe for the legal role (its corpus is the
            # Act); the news probes carry their own article text and the general
            # role is context-free by design, matching how those agents run.
            probes = [
                _deployed_prompt(
                    role, probe, _retrieval_context(probe) if role == "legal" else ""
                )
                for probe in probes
            ]

        print("=" * 78)
        print(f"[validate] role={role}  base={config.LOCAL_PEFT_ROLES[role]['base_model']}")
        approx_tokens = max(len(p) for p in probes) // 4
        scaffold = "real agent prompt" if args.realistic_length else "bare question"
        warning = "" if approx_tokens > 300 else "  SHORT - may hide long-context failure"
        print(f"[validate] longest probe ~{approx_tokens} tokens ({scaffold}){warning}")
        print("=" * 78)

        print("[validate]   generating with adapter...")
        tuned = _generate(role, True, probes, args.max_new_tokens)
        print("[validate]   generating from base weights...")
        base = _generate(role, False, probes, args.max_new_tokens)

        comparisons = []
        for probe, tuned_text, base_text in zip(probes, tuned, base):
            diff_ratio = 1.0 - difflib.SequenceMatcher(None, base_text, tuned_text).ratio()
            comparison = {
                "probe": probe[:160],
                "diff_ratio": round(diff_ratio, 4),
                "tuned": tuned_text,
                "base": base_text,
                "tuned_signals": _signals(tuned_text),
                "base_signals": _signals(base_text),
            }
            comparisons.append(comparison)

            print(f"\n--- probe: {probe[:100]}")
            print(f"    divergence from base: {diff_ratio:.1%}")
            print(f"    [TUNED] {tuned_text[:420]}")
            print(f"    [BASE ] {base_text[:420]}")

        mean_diff = sum(c["diff_ratio"] for c in comparisons) / len(comparisons)
        inert = mean_diff < args.min_diff_ratio
        role_meta = {}
        meta_path = adapter_dir / "training_meta.json"
        if meta_path.exists():
            try:
                role_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                role_meta = {}

        # Second, independent check: did held-out loss actually improve during
        # training? Output divergence and the loss curve fail in different ways,
        # so both are reported. An adapter can diverge from base while having
        # learnt nothing useful (it was trained on truncated targets), and it can
        # in principle learn while changing output only subtly. Requiring both to
        # look healthy is what makes "the adapters work" a checked claim.
        # Named learning_signal, not `signal`: the marker-aggregation loop below
        # already binds `signal`, and reusing the name silently rebound this to a
        # string ("words") by the time it was read again.
        learning_signal = role_meta.get("learning_signal") or {}

        # The verdict must be measured against the BASE model, not against epoch 1.
        #
        # The first evaluation happens after a full epoch, so an epoch-1-to-epoch-3
        # comparison answers "did it keep improving?" and is blind to everything
        # learnt in epoch 1. Judged on that alone the news adapter looked inert -
        # yet its base loss is 2.994 against 1.630 after one epoch, a 46%
        # reduction. Reporting that as "learnt nothing" would have been flatly
        # wrong, and would have argued for discarding a working adapter.
        #
        # base_eval_loss.json is therefore authoritative when present; the flat
        # curve is demoted to a note about when training saturated.
        base_losses = _load_base_losses()
        base_entry = base_losses.get(role) or {}
        base_loss = base_entry.get("base_eval_loss")
        history = role_meta.get("eval_history") or []
        best_loss = min((p["eval_loss"] for p in history), default=None)

        gain_from_base = None
        if base_loss is not None and best_loss is not None:
            gain_from_base = round(base_loss - best_loss, 4)

        if gain_from_base is not None:
            never_learned = gain_from_base <= 0.05
            if not never_learned and learning_signal.get("improved") is False:
                print(
                    f"[validate] NOTE {role}: training saturated after epoch 1 "
                    f"(held-out loss {learning_signal.get('first_eval_loss')} -> "
                    f"{learning_signal.get('final_eval_loss')}), but the adapter did learn: "
                    f"base {base_loss} -> best {best_loss} (gain {gain_from_base}). "
                    f"This is a note about training length, not an inert adapter."
                )
        else:
            # No base measurement available. Fall back to the curve, and say so -
            # this verdict is weaker than it looks.
            never_learned = learning_signal.get("improved") is False
            if never_learned:
                print(
                    f"[validate] WARNING {role}: held-out loss did not improve after epoch 1 "
                    f"and no base-model measurement exists to check whether it learnt at all. "
                    f"Run: python finetune/base_eval_loss.py"
                )

        if never_learned and gain_from_base is not None:
            print(
                f"[validate] WARNING {role}: adapter is no better than its base model "
                f"(base {base_loss} vs best {best_loss}). It should not be benchmarked as a "
                f"specialised expert."
            )

        entry = {
            "role": role,
            "status": "inert" if (inert or never_learned) else "ok",
            "adapter_dir": str(adapter_dir),
            "mean_diff_ratio": round(mean_diff, 4),
            "train_examples": role_meta.get("train_examples"),
            "final_train_loss": role_meta.get("final_train_loss"),
            "eval_history": role_meta.get("eval_history"),
            "learning_signal": learning_signal or None,
            "base_eval_loss": base_loss,
            "best_eval_loss": best_loss,
            "gain_from_base": gain_from_base,
            "saturated_after_epoch1": learning_signal.get("improved") is False,
            "failed_divergence_check": bool(inert),
            "failed_loss_curve_check": bool(never_learned),
            "comparisons": comparisons,
        }
        # Aggregate the domain-marker signals so "does the legal adapter cite
        # more articles" is answerable from the report, not just by eye.
        for marker in ("article_citations", "irac_markers", "words"):
            entry[f"tuned_mean_{marker}"] = round(
                sum(c["tuned_signals"][marker] for c in comparisons) / len(comparisons), 2
            )
            entry[f"base_mean_{marker}"] = round(
                sum(c["base_signals"][marker] for c in comparisons) / len(comparisons), 2
            )
        report["roles"].append(entry)

        # Answer-shape summary alongside the verdict.
        #
        # The two pass/fail criteria - divergence from base, and held-out loss
        # gain - are both about whether the adapter CHANGED the model, not whether
        # it improved it. An adapter can satisfy both while answering far worse:
        # that is exactly what happened when the legal adapter scored 92.6%
        # divergence and a 1.80 loss gain while producing a third of its base
        # model's words. These figures are printed next to the verdict so a PASS
        # cannot be read as "this adapter is good" without seeing them.
        word_ratio = (
            entry["tuned_mean_words"] / entry["base_mean_words"]
            if entry["base_mean_words"]
            else float("nan")
        )
        entry["word_ratio_vs_base"] = round(word_ratio, 2)
        print(
            f"\n[validate] {role}: divergence {mean_diff:.1%}, "
            f"loss improved={learning_signal.get('improved', 'unknown')}, "
            f"words {entry['tuned_mean_words']:.0f} vs base {entry['base_mean_words']:.0f} "
            f"({word_ratio:.2f}x), citations {entry['tuned_mean_article_citations']:.1f} vs "
            f"{entry['base_mean_article_citations']:.1f} -> {entry['status'].upper()}"
        )
        if word_ratio < 0.5 or word_ratio > 4.0:
            print(
                f"[validate] NOTE {role} answer length differs from base by "
                f"{word_ratio:.2f}x. Neither pass criterion looks at this, so read it "
                f"before treating the verdict as an endorsement: judged completeness "
                f"and lexical overlap are both measured against long references."
            )
        if never_learned:
            all_ok = False
        if inert:
            print(
                f"[validate] WARNING adapter for '{role}' barely changes output. Either the "
                "training did not take (check final_train_loss and the LoRA target modules "
                "actually matched this architecture) or the adapter is not being applied. "
                "Do NOT benchmark the peft arm until this is resolved - it would report the "
                "base models under a 'peft' label."
            )
            all_ok = False

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[validate] full transcripts -> {REPORT_PATH.name}")
    print(f"[validate] verdict: {'PASS' if all_ok else 'ATTENTION NEEDED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
