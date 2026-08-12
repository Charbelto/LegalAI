"""Annotate which retrieved chunks are actually relevant to each query.

Why this exists: `gold_doc_ids` used to be a copy of whatever the retriever
returned while the gold answer was being drafted, so precision@5 measured against
it was 1.0 by construction. Retrieval metrics are now suppressed until a human has
marked the genuinely relevant chunks here.

Usage:
    python scripts/annotate_relevance.py                 # annotate un-annotated queries
    python scripts/annotate_relevance.py --redo q05      # re-annotate one query
    python scripts/annotate_relevance.py --k 10          # show more candidates
    python scripts/annotate_relevance.py --status        # how far through you are

At each query: enter the numbers of the relevant chunks (e.g. "1 3 4"),
"n" if none are relevant, "s" to skip, or "q" to save and quit.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

DATASET_FILE = ROOT_DIR / "eval_dataset.json"


def load_dataset():
    if not DATASET_FILE.exists():
        raise SystemExit(f"Dataset not found at {DATASET_FILE}")
    return json.loads(DATASET_FILE.read_text(encoding="utf-8"))


def save_dataset(dataset):
    DATASET_FILE.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def show_status(dataset):
    annotated = [item for item in dataset if item.get("relevance_annotated")]
    print(f"Annotated: {len(annotated)}/{len(dataset)} queries")
    by_type = {}
    for item in dataset:
        key = item.get("type", "?")
        done, total = by_type.get(key, (0, 0))
        by_type[key] = (done + bool(item.get("relevance_annotated")), total + 1)
    for key, (done, total) in sorted(by_type.items()):
        print(f"  {key:14s} {done}/{total}")
    if len(annotated) == 0:
        print("\nRetrieval metrics are suppressed until at least one query is annotated.")


def get_retriever():
    """Import the project's retrieval stack lazily (needs Ollama + Chroma)."""
    import chromadb
    from chromadb.utils import embedding_functions

    import config

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        model_name=config.OLLAMA_EMBEDDING_MODEL,
        url=f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/embeddings",
    )
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIRECTORY)
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME, embedding_function=ollama_ef
    )
    return collection


def candidates_for(collection, query, k):
    result = collection.query(query_texts=[query], n_results=k)
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    return list(zip(ids, documents, metadatas))


def annotate(dataset, collection, k, only_id=None):
    changed = 0
    for item in dataset:
        if only_id and item["id"] != only_id:
            continue
        if not only_id and item.get("relevance_annotated"):
            continue

        print("\n" + "=" * 78)
        print(f"{item['id']}  [{item.get('type')}]  {item['query']}")
        print("=" * 78)

        try:
            candidates = candidates_for(collection, item["query"], k)
        except Exception as exc:
            print(f"Retrieval failed: {exc}")
            break

        if not candidates:
            print("No candidates returned - is the corpus ingested?")
            continue

        for index, (chunk_id, document, metadata) in enumerate(candidates, 1):
            name = (metadata or {}).get("name") or (metadata or {}).get("source") or "?"
            snippet = " ".join((document or "").split())[:260]
            print(f"\n [{index}] id={chunk_id}  source={name}")
            print(f"     {snippet}...")

        print("\nReference answer (for context):")
        print("   " + " ".join(item.get("gold", "").split())[:400] + "...")

        answer = input("\nRelevant chunk numbers (space separated) / n=none / s=skip / q=quit: ").strip().lower()

        if answer == "q":
            break
        if answer == "s":
            continue
        if answer == "n":
            item["gold_doc_ids"] = []
            item["relevance_annotated"] = True
            item["relevance_annotated_by"] = "human"
            changed += 1
            print("  -> recorded: no relevant chunks retrieved")
            continue

        try:
            picked = [int(token) for token in answer.split()]
        except ValueError:
            print("  -> not understood, skipping")
            continue

        selected = [
            candidates[number - 1][0]
            for number in picked
            if 1 <= number <= len(candidates)
        ]
        item["gold_doc_ids"] = selected
        item["relevance_annotated"] = True
        item["relevance_annotated_by"] = "human"
        changed += 1
        print(f"  -> recorded {len(selected)} relevant chunk(s): {selected}")

    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8, help="candidates to show per query")
    parser.add_argument("--redo", type=str, default=None, help="re-annotate a single query id")
    parser.add_argument("--status", action="store_true", help="show progress and exit")
    args = parser.parse_args()

    dataset = load_dataset()

    if args.status:
        show_status(dataset)
        return

    show_status(dataset)
    collection = get_retriever()
    changed = annotate(dataset, collection, args.k, only_id=args.redo)

    if changed:
        save_dataset(dataset)
        print(f"\nSaved {changed} annotation(s) to {DATASET_FILE.name}")
    else:
        print("\nNothing changed.")
    show_status(dataset)


if __name__ == "__main__":
    main()
