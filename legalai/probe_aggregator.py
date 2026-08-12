"""Isolate the aggregator at increasing prompt sizes.

    python probe_aggregator.py

The experts are confirmed good at production prompt length, so the degenerate
answers coming out of /chat must originate downstream. The aggregator is the
prime suspect on two counts: it consumed 25 of one failing run's 46 minutes, and
it is the only node that sees retrieved context AND all three expert answers at
once - roughly 4500 tokens, against the 1373-2141 an expert sees.

It runs on the coordination model (the general expert's base weights with the
adapter disabled), so this also tests whether an unadapted 2B model can do the
merging job at that length.

Method: hold the model fixed and vary only how many expert answers are fed in,
using the REAL expert outputs recorded in finetune/adapter_validation.json rather
than regenerating them. If output degrades as the prompt grows, the failure is
length-driven and capping the aggregator's input is the fix. If it degrades even
with one expert, the node needs a different model.
"""

import json
import sys
from pathlib import Path

import config
import local_models
import utils as Utils
from agents.retrieval import RetrievalAgent

QUERY = "What is a high-risk AI system under the EU AI Act?"
VALIDATION = Path("finetune/adapter_validation.json")


def real_expert_outputs():
    """The actual tuned expert answers recorded during validation."""
    if not VALIDATION.exists():
        raise SystemExit("Run finetune/validate_adapters.py first - this reuses its outputs.")
    report = json.loads(VALIDATION.read_text(encoding="utf-8"))
    outputs = {}
    for entry in report["roles"]:
        comparisons = entry.get("comparisons") or []
        if comparisons:
            outputs[entry["role"]] = comparisons[0]["tuned"]
    return outputs


def main():
    experts = real_expert_outputs()
    print("real expert outputs available:", {k: len(v.split()) for k, v in experts.items()}, "words")

    state = RetrievalAgent().invoke(
        {"query": QUERY, "route": "legal", "session_id": "aggprobe",
         "retrieved_docs": [], "chat_history": []}
    )
    docs = state.get("retrieved_docs", [])
    context_parts = []
    for index, doc in enumerate(docs[:5], 1):
        content = getattr(doc, "page_content", None)
        if content is None and isinstance(doc, dict):
            content = doc.get("page_content") or doc.get("content") or ""
        context_parts.append(f"[Document {index}]:\n{content}\n")
    context = "\n".join(context_parts)

    # The aggregator runs on the coordination model: the general expert's base
    # weights with no adapter. Ask for it the same way the agent does.
    chat = local_models.LocalChatModel(role="aggregator", temperature=0.3)
    tokenizer = chat._loaded.tokenizer
    print(f"aggregator model: {chat._loaded.base_model_id}  adapter={chat._loaded.adapter_dir}")

    # Increasing input sizes, one expert at a time.
    ladders = [
        ("1 expert (legal)", ["legal"]),
        ("2 experts (legal+news)", ["legal", "news"]),
        ("3 experts (all)", ["legal", "news", "general_qa"]),
    ]

    for label, roles in ladders:
        blocks = [f"[{r}]:\n{experts[r]}" for r in roles if r in experts]
        expert_output = "\n\n".join(blocks)
        prompt = config.AGGREGATOR_PROMPT.format(
            current_date=Utils.get_current_date(),
            chat_history="No previous conversation.",
            context=context,
            agent_type=", ".join(roles),
            expert_output=expert_output,
            query=QUERY,
        )
        tokens = len(tokenizer(prompt)["input_ids"])
        message = chat.invoke(prompt)
        answer = message.content
        meta = message.response_metadata
        print(f"\n=== {label} | prompt {tokens} tokens ===")
        print(f"    out: {len(answer.split())} words, {meta['completion_tokens']} tokens, "
              f"hit_cap={meta.get('hit_token_cap')}, {meta['generation_ms'] / 1000:.0f}s")
        print("    " + answer[:500].replace("\n", "\n    "))

    local_models.unload_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
