"""Build a FROZEN news snapshot so the 10 routing queries are benchmarkable.

The routing queries ask for current developments. The corpus is statute-only, so
they were unanswerable. This adds real news alongside the Act -- it does not
replace it.

Two things make this safe, and both matter:

1. clear_existing=False. The service path (backend/service.py:485) fetches with
   clear_existing=True, which deletes every chunk and leaves 44 news chunks where
   the Act used to be. That is bug 8. embed.py assigns ids from
   collection.count(), so appending yields ids 644+ and cannot collide with the
   Act's 0-643.

2. Fetched ONCE, here, out of band. The benchmark keeps fetch_news=False, so all
   540 runs read an identical corpus. Live fetching during the run would make
   context depend on what the web returned that minute and the arms would not be
   comparable.

The pre-existing articles.json holds 20 junk rows from the accidental fetch
("Legal AI software vendors", "2026 in the United Kingdom - Wikipedia"). They are
moved aside rather than embedded.
"""

import json
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, r"C:\Users\Charbel\Desktop\Legal AI\legalai")
os.chdir(r"C:\Users\Charbel\Desktop\Legal AI\legalai")

import chromadb

import embed
import utils as Utils

STAMP = "20260806"

# One targeted search per routing query, derived from the query itself.
TOPICS = {
    "q21": "EU AI Office establishment operations news 2026",
    "q22": "EU AI Act general-purpose AI GPAI models regulation update 2026",
    "q23": "EU US AI safety cooperation agreement announcement",
    "q24": "EU AI Act member states national market surveillance authorities designation 2026",
    "q25": "EU AI Office guidelines general-purpose AI model classification 2026",
    "q26": "EU AI Act impact on US technology companies 2026",
    "q27": "EU AI Act notified bodies conformity assessment high-risk systems 2026",
    "q28": "EU AI Act Article 5 prohibitions enforcement 2026",
    "q29": "EU AI Act penalties fines imposed companies 2026",
    "q30": "EU AI Act harmonised standards standardisation request CEN CENELEC 2026",
}
PER_TOPIC = 4


def baseline_check():
    client = chromadb.PersistentClient(path="chroma_storage")
    col = client.get_collection("collection_1")
    ids = col.get(include=[])["ids"]
    nums = sorted(int(i) for i in ids if i.isdigit())
    print(f"BEFORE: {col.count()} chunks, id range {nums[0]}-{nums[-1]}")
    return set(ids)


def stash_stale():
    if os.path.exists("articles.json"):
        shutil.move("articles.json", f"articles.stale_backup_{STAMP}.json")
        print(f"moved stale articles.json -> articles.stale_backup_{STAMP}.json")
    if os.path.isdir("articles"):
        dest = f"articles_stale_backup_{STAMP}"
        if not os.path.exists(dest):
            shutil.move("articles", dest)
            print(f"moved stale articles/ -> {dest}/")
    os.makedirs("articles", exist_ok=True)


def main():
    before_ids = baseline_check()
    stash_stale()

    seen_urls = set()
    manifest = []
    for qid, query in TOPICS.items():
        print(f"\n--- {qid}: {query}")
        try:
            articles = Utils.fetch_news_articles(query, PER_TOPIC)
        except Exception as exc:  # a dead source must not abort the snapshot
            print(f"  fetch failed: {exc}")
            continue

        fresh = []
        for a in articles:
            url = a.get("url", "")
            body = (a.get("content") or "").strip()
            if not url or url in seen_urls or len(body) < 400:
                continue
            seen_urls.add(url)
            fresh.append(a)
            manifest.append(
                {"qid": qid, "search": query, "title": a.get("title"),
                 "url": url, "chars": len(body)}
            )
            print(f"  + {len(body):6} ch  {str(a.get('title'))[:70]}")

        if fresh:
            Utils.save_online_articles(query, fresh)

    if not manifest:
        print("\nNO ARTICLES FETCHED - corpus untouched.")
        return

    print(f"\nembedding {len(manifest)} articles (clear_existing=False)...")
    meta = Utils.load_articles(Utils.ARTICLES_FILE)
    embed.embed_articles_from_files(meta, clear_existing=False)

    # The whole point of the exercise: the Act must still be there.
    client = chromadb.PersistentClient(path="chroma_storage")
    col = client.get_collection("collection_1")
    after_ids = set(col.get(include=[])["ids"])
    lost = before_ids - after_ids
    print(f"\nAFTER: {col.count()} chunks")
    print(f"Act chunks lost: {len(lost)}  {'OK' if not lost else 'CORPUS DAMAGED'}")

    snap = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "act_chunks_before": len(before_ids),
        "total_chunks_after": col.count(),
        "news_articles": len(manifest),
        "articles": manifest,
    }
    with open(f"news_snapshot_{STAMP}.json", "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    print(f"manifest -> news_snapshot_{STAMP}.json")


if __name__ == "__main__":
    main()
