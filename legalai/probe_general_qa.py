"""Diagnose why the general_qa node dominates run time.

    python probe_general_qa.py

In a full run general_qa took 572s - 56% of the wall clock, and 12x the legal
expert on the same hardware, despite Granite 2B being the SMALLEST of the three
models. That asymmetry says bug rather than model size.

Leading hypothesis: it runs to the 1024-token cap without emitting a stop token,
the same signature as the aggregator defect but in a node whose adapter IS
correctly applied. `hit_token_cap` in the generation metadata answers it directly.

Compares tuned vs base at the prompt length the node actually sees in the graph
(retrieved context included), because validation only exercised it at ~262 tokens.
"""

import sys
import time

import config
import local_models
import utils as Utils
from agents.general_qa import GeneralQAAgent
from agents.retrieval import RetrievalAgent

QUERY = "What is a high-risk AI system under the EU AI Act?"


def run(use_adapters: bool):
    config.LOCAL_PEFT_USE_ADAPTERS = use_adapters
    local_models.unload_all()
    label = "TUNED" if use_adapters else "BASE "

    agent = GeneralQAAgent()
    state = RetrievalAgent().invoke(
        {"query": QUERY, "route": "legal", "session_id": "gqprobe",
         "retrieved_docs": [], "chat_history": []}
    )

    # Exactly the prompt the agent will render, to report the real token count.
    rendered = config.GENERAL_QA_PROMPT.format(
        context=agent._format_context(state.get("retrieved_docs", [])),
        query=QUERY,
        chat_history="No previous conversation.",
        current_date=Utils.get_current_date(),
    )
    tokenizer = local_models.get_loaded_model("general_qa").tokenizer
    prompt_tokens = len(tokenizer(rendered)["input_ids"])

    started = time.perf_counter()
    result = agent.invoke(dict(state))
    elapsed = time.perf_counter() - started
    answer = result["agent_outputs"]["general_qa"]

    tokens = result.get("agent_tokens", {}).get("general_qa", {})
    print(f"\n[{label}] prompt={prompt_tokens} tok | {elapsed:.1f}s | "
          f"completion={tokens.get('completion')} tok | {len(answer.split())} words")
    print(f"    tok/s = {tokens.get('completion', 0) / elapsed:.1f}")
    print("    " + answer[:400].replace("\n", "\n    "))
    local_models.unload_all()
    return elapsed, tokens.get("completion", 0), len(answer.split())


print("=" * 78)
print("general_qa node: is it running to the token cap?")
print(f"LLM_NUM_PREDICT (cap) = {config.LLM_NUM_PREDICT}")
print("=" * 78)

t_time, t_tok, t_words = run(True)
b_time, b_tok, b_words = run(False)

print("\n" + "=" * 78)
print(f"  TUNED: {t_time:6.1f}s  {t_tok:5d} completion tokens  {t_words:4d} words"
      f"{'   <-- HIT CAP' if t_tok >= config.LLM_NUM_PREDICT else ''}")
print(f"  BASE : {b_time:6.1f}s  {b_tok:5d} completion tokens  {b_words:4d} words"
      f"{'   <-- HIT CAP' if b_tok >= config.LLM_NUM_PREDICT else ''}")
print("=" * 78)
if t_tok >= config.LLM_NUM_PREDICT:
    print("The tuned node never emits a stop token: it generates the full budget every")
    print("call. That is the whole cost, and it is a generation defect, not model size.")
sys.exit(0)
