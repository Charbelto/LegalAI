"""Shrink the news snapshot so it serves routing queries without burying the Act.

Embedding 39 full articles produced 612 news chunks against 644 Act chunks, and
the real hybrid retriever then returned news for 12 of the 20 statute queries.
Fixing the 10 routing queries by breaking 12 statute ones is a net loss.

Volume is the lever, not presence. On a recency query the reranker gives news
+0.15 and a 2026 effective_from a further +0.2, against legislation's +0.2
authority boost -- so news wins those queries on scoring alone and only needs to
exist, not dominate. Truncating each article to its lede keeps all 39 sources
represented at roughly a tenth of the chunk count.

embed_articles_from_files(clear_existing=True) deletes only source_type="news"
and leaves the statutory corpus alone, so this swaps the news set in place.
"""

import json
import os
import shutil
import sys

sys.path.insert(0, r"C:\Users\Charbel\Desktop\Legal AI\legalai")
os.chdir(r"C:\Users\Charbel\Desktop\Legal AI\legalai")

import chromadb

import embed
import utils as Utils

TRUNC_CHARS = 2000
TRUNC_DIR = "articles_lede"


def truncate(text: str, limit: int) -> str:
    """Cut to `limit` chars, backing up to the last sentence end so the final
    chunk is not a half sentence the embedder has to make sense of."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", ".\n", "! ", "? "):
        idx = cut.rfind(sep)
        if idx > limit * 0.5:
            return cut[: idx + 1]
    idx = cut.rfind(" ")
    return cut[:idx] if idx > 0 else cut


def main():
    client = chromadb.PersistentClient(path="chroma_storage")
    col = client.get_collection("collection_1")
    before = col.count()
    leg_before = len(col.get(where={"source_type": "legislation"}, include=[])["ids"])
    print(f"BEFORE: {before} chunks ({leg_before} legislation)")

    meta = Utils.load_articles(Utils.ARTICLES_FILE)
    print(f"articles on disk: {len(meta)}")

    if os.path.isdir(TRUNC_DIR):
        shutil.rmtree(TRUNC_DIR)
    os.makedirs(TRUNC_DIR, exist_ok=True)

    trimmed = []
    for art in meta:
        src = art.get("file", "")
        if not os.path.exists(src):
            continue
        content = Utils.load_article_content(src) or ""
        if len(content.strip()) < 300:
            continue
        lede = truncate(content, TRUNC_CHARS)
        dest = os.path.join(TRUNC_DIR, os.path.basename(src))
        with open(dest, "w", encoding="utf-8") as f:
            f.write(lede)
        entry = dict(art)
        entry["file"] = dest
        trimmed.append(entry)

    total_chars = sum(os.path.getsize(a["file"]) for a in trimmed)
    print(f"trimmed {len(trimmed)} articles -> ~{total_chars//1000}k chars "
          f"(expect ~{total_chars//1000} chunks at 1000 chars each)")

    embed.embed_articles_from_files(trimmed, clear_existing=True)

    col = client.get_collection("collection_1")
    leg_after = len(col.get(where={"source_type": "legislation"}, include=[])["ids"])
    news_after = len(col.get(where={"source_type": "news"}, include=[])["ids"])
    print(f"\nAFTER: {col.count()} chunks = {leg_after} legislation + {news_after} news")
    print(f"news share: {100*news_after/col.count():.1f}%")
    print(f"Act intact: {'YES' if leg_after == leg_before else 'NO - ' + str(leg_before-leg_after) + ' LOST'}")

    snap_path = "news_snapshot_20260806.json"
    if os.path.exists(snap_path):
        snap = json.load(open(snap_path, encoding="utf-8"))
        snap["trimmed_to_lede_chars"] = TRUNC_CHARS
        snap["news_chunks_final"] = news_after
        snap["legislation_chunks"] = leg_after
        json.dump(snap, open(snap_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"updated {snap_path}")


if __name__ == "__main__":
    main()
