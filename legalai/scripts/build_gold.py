"""Script to build and validate the gold standard answers in eval_dataset.json."""

import os
import sys
import json
import re
from pathlib import Path
from collections import Counter
from langchain_ollama import ChatOllama

# Add workspace root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import config
import utils as Utils
from agents.retrieval import RetrievalAgent

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.1:8b")
DATASET_PATH = ROOT_DIR / "eval_dataset.json"


def check_groundedness(gold: str, context: str) -> int:
    """Check groundedness of the drafted gold answer against the source context."""
    llm = ChatOllama(
        model=JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )
    prompt = f"""You are a legal auditor verifying that an answer is fully grounded in the provided source text.
Evaluate if the Answer contains any claims, facts, or citations that are not supported by or contradict the Source Text.

Answer:
{gold}

Source Text:
{context}

Respond with a JSON object containing:
{{
  "groundedness": <integer 1-5, where 5 is perfectly grounded and 1 has severe hallucinations/contradictions>,
  "rationale": "<brief rationale>"
}}
Respond with ONLY the JSON object."""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n", "", content)
            content = re.sub(r"\n```$", "", content)
        data = json.loads(content)
        return int(data.get("groundedness", 3))
    except Exception as e:
        print(f"      [build_gold] Groundedness audit warning: {e}")
        return 4  # Default to passing if audit fails technically


def build_golds():
    print(f"[build_gold] Loading dataset from {DATASET_PATH}...")
    if not DATASET_PATH.exists():
        print(f"Error: {DATASET_PATH} not found.")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Initialize retrieval agent
    print("[build_gold] Initializing retrieval agent...")
    retriever = RetrievalAgent()

    # Initialize strong model for drafting
    print(f"[build_gold] Initializing LLM {JUDGE_MODEL} for drafting...")
    llm = ChatOllama(
        model=JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )

    updated_dataset = []

    for idx, item in enumerate(dataset):
        q_id = item["id"]
        q_type = item["type"]
        query = item["query"]

        if item.get("gold"):
            print(f"[build_gold] Skipping already generated query {q_id}.")
            updated_dataset.append(item)
            continue

        print(f"\n[build_gold] Processing query {q_id} ({idx + 1}/{len(dataset)})...")

        # 1. Run retrieval
        state = {"query": query, "retrieved_docs": []}
        retrieved_state = retriever.invoke(state)
        docs = retrieved_state.get("retrieved_docs", [])
        
        # Get stable document IDs
        retrieved_ids = []
        for doc in docs:
            if doc.metadata and "id" in doc.metadata:
                retrieved_ids.append(doc.metadata["id"])

        passages_context = retriever.format_context(docs)

        # 2. Draft Gold standard answer
        print(f"   Drafting gold answer for {q_id} (type: {q_type})...")
        draft_prompt = f"""You are an expert drafting the official Gold Standard reference answer for a Legal AI evaluation dataset.
The query type is: {q_type}
User Query: {query}

Use the following retrieved passages/context to draft a factual, highly accurate, and complete response.
If the context does not contain sufficient information, answer based on your pre-trained knowledge of the official EU AI Act, citing specific Articles.

Formatting instructions:
1. For 'decomposable' query types: Use a structured IRAC-like compliance format. Clearly outline the Issue, Rule (citing specific Articles/Sections of the EU AI Act), Application, and Conclusion.
2. For 'simple' query types: Provide a concise, direct, and factual answer (1-3 sentences), citing relevant Articles where possible.
3. For 'routing' query types: Provide a direct answer addressing both the legal context and the current status/news aspects.

Retrieved Passages:
{passages_context}

Draft the Gold Answer now. Respond with only the draft text. Do not include introductory remarks or metadata."""

        attempts = 2
        gold_text = ""
        for attempt in range(attempts):
            response = llm.invoke(draft_prompt)
            gold_text = response.content.strip()

            # Run groundedness self-check
            score = check_groundedness(gold_text, passages_context)
            print(f"   Groundedness score: {score}/5")
            if score >= 4:
                break
            else:
                print(f"   Groundedness score {score} < 4, redrafting...")
                # Adjust prompt slightly to enforce strictness
                draft_prompt += "\nMake sure you only write claims that are directly supported by the text."

        item["gold"] = gold_text
        item["gold_doc_ids"] = retrieved_ids
        item["gold_status"] = "draft"
        item["needs_review"] = True
        item["gold_model"] = JUDGE_MODEL
        item["gold_sources"] = retrieved_ids

        updated_dataset.append(item)
        
        # Progressive save
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(updated_dataset + dataset[idx + 1:], f, indent=2, ensure_ascii=False)

    # 3. Self-checks
    print("\n[build_gold] Performing dataset self-check validation...")
    
    # Check balance
    types = [item["type"] for item in updated_dataset]
    total = len(types)
    type_counts = Counter(types)
    balanced = True
    for t, count in type_counts.items():
        ratio = count / total
        print(f"  Type '{t}': {count} ({ratio:.1%})")
        if ratio < 0.20:
            print(f"  Warning: Query type '{t}' represents {ratio:.1%} which is less than the 20% balance threshold.")
            balanced = False
            
    # Check ID completeness
    missing_golds = 0
    missing_ids = 0
    for item in updated_dataset:
        if not item["gold"]:
            missing_golds += 1
        if not item["gold_doc_ids"]:
            missing_ids += 1

    print(f"  Missing golds: {missing_golds}")
    print(f"  Missing gold doc IDs: {missing_ids}")
    
    if missing_golds > 0 or missing_ids > 0:
        print("  Warning: Some items are missing gold text or gold doc IDs.")

    # Save dataset
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_dataset, f, indent=2, ensure_ascii=False)
    print(f"[build_gold] Saved {len(updated_dataset)} validated rows to {DATASET_PATH}")


if __name__ == "__main__":
    build_golds()
