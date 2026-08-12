"""Call the legal expert directly, at production prompt length, tuned vs base.

    python probe_legal_expert.py

Reads the LegalAgent's RAW output. Going through /chat cannot answer "is the legal
adapter usable", because the answer you see there has passed through the
aggregator, validator and response nodes - any of which could be the thing
degrading it. This bypasses all of them.

Uses the real retrieval stack and the real config.LEGAL_PROMPT, so the prompt is
the 1373-2141 tokens the expert actually sees in production rather than a short
standalone probe. That length is what broke the first adapter set while short
probes still looked acceptable.
"""

import sys

import config
import local_models
import utils as Utils
from agents.legal import LegalAgent
from agents.retrieval import RetrievalAgent

QUERIES = [
    "What is a high-risk AI system under the EU AI Act?",
    "What conformity-assessment obligations apply to a provider of a high-risk AI system?",
]


def run(use_adapters: bool):
    config.LOCAL_PEFT_USE_ADAPTERS = use_adapters
    local_models.unload_all()
    label = "TUNED" if use_adapters else "BASE "
    outputs = []
    agent = LegalAgent()
    retrieval = RetrievalAgent()
    for query in QUERIES:
        state = retrieval.invoke(
            {"query": query, "route": "legal", "session_id": "probe",
             "retrieved_docs": [], "chat_history": []}
        )
        docs = state.get("retrieved_docs", [])
        # Exactly what the agent will render, so the token count below is the real
        # production prompt length rather than an approximation.
        rendered = config.LEGAL_PROMPT.format(
            context=agent._format_context(docs),
            query=query,
            chat_history="No previous conversation.",
            current_date=Utils.get_current_date(),
        )
        tokens = len(local_models.get_loaded_model("legal").tokenizer(rendered)["input_ids"])

        result = agent.invoke(dict(state))
        answer = result["agent_outputs"]["legal"]
        outputs.append((query, tokens, answer))
        print(f"\n[{label}] prompt={tokens} tokens | answer={len(answer.split())} words, {len(answer)} chars")
        print(f"    Q: {query[:80]}")
        print("    " + answer[:600].replace("\n", "\n    "))
    local_models.unload_all()
    return outputs


print("=" * 78)
print("LEGAL EXPERT, RAW OUTPUT, PRODUCTION PROMPT LENGTH")
print("=" * 78)
tuned = run(True)
base = run(False)

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
for (q, t_tok, t_ans), (_, b_tok, b_ans) in zip(tuned, base):
    tw, bw = len(t_ans.split()), len(b_ans.split())
    abstain = config.ABSTENTION_SENTENCE[:40]
    print(f"  prompt {t_tok} tokens")
    print(f"    tuned: {tw:4d} words  abstained={t_ans.strip().startswith(abstain)}")
    print(f"    base : {bw:4d} words  abstained={b_ans.strip().startswith(abstain)}")
    print(f"    ratio: {tw / bw:.2f}x" if bw else "    ratio: n/a")
sys.exit(0)
