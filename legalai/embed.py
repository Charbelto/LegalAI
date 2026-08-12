import utils as Utils
from tqdm import tqdm
import requests
import fitz  # PyMuPDF
from chromadb.utils import embedding_functions
import chromadb
import os as OS
from typing import Callable, Optional
import config

def pdf_to_text(url, headers=None):
    """Download one PDF and extract its text. Returns "" on any failure.

    A silent "" here is how the corpus ended up with a single empty document:
    the caller embedded the empty string without checking. fetch_eu_ai_act_text()
    below is the safe entry point - it verifies the payload really is a PDF and
    raises rather than returning nothing.
    """
    try:
        response = requests.get(url, timeout=60, headers=headers or {})
        response.raise_for_status()
        pdf_data = response.content
        if not pdf_data.startswith(b"%PDF-"):
            print(
                f"[embed] {url} returned {len(pdf_data)} bytes that are not a PDF "
                f"(status {response.status_code}, "
                f"content-type {response.headers.get('content-type', '?')})"
            )
            return ""
        document = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


def pdf_to_text_from_file(path):
    """Extract text from a PDF already on disk (the manual-download fallback)."""
    document = fitz.open(path)
    return "".join(document.load_page(i).get_text() for i in range(len(document)))


def fetch_eu_ai_act_text(min_chars=200_000):
    """Fetch the EU AI Act text, trying each configured source in turn.

    Raises RuntimeError if no source yields a plausible full text, rather than
    returning "" - embedding an empty string produced a one-document collection
    that looked superficially fine (`get_db_document_count()` returned 1) while
    making every legal query abstain. The length floor catches a source that
    returns a cover page or an error document formatted as a valid PDF.
    """
    attempts = []
    for source in Utils.EUROPEAN_ACT_SOURCES:
        label = source.get("label", source["url"])
        print(f"[embed] trying {label}")
        text = pdf_to_text(source["url"], headers=source.get("headers"))
        if len(text) >= min_chars:
            print(f"[embed] got {len(text):,} characters from {label}")
            return text, label
        attempts.append(f"{label}: {len(text)} chars")
        print(f"[embed]   too short ({len(text)} chars), trying next source")

    raise RuntimeError(
        "Could not fetch the EU AI Act from any configured source.\n  "
        + "\n  ".join(attempts)
        + "\n\nEvery legal query would abstain against an empty corpus, so this is "
        "fatal rather than a warning. Fix by either adding a working URL to "
        "utils.EUROPEAN_ACT_SOURCES, or downloading the PDF manually and running:\n"
        "    python embed.py --pdf path\\to\\ai_act.pdf"
    )

def split_text_into_sections(text, min_chars_per_section):
    paragraphs = text.split('\n')
    sections = []
    current_section = ""
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_length + paragraph_length + 2 <= min_chars_per_section:  # +2 for the double newline
            current_section += paragraph + '\n\n'
            current_length += paragraph_length + 2  # +2 for the double newline
        else:
            if current_section:
                sections.append(current_section.strip())
            current_section = paragraph + '\n\n'
            current_length = paragraph_length + 2  # +2 for the double newline

    if current_section:  # Add the last section
        sections.append(current_section.strip())

    return sections

def embed_text_in_chromadb(
    text,
    document_name,
    document_description,
    persist_directory=config.CHROMA_PERSIST_DIRECTORY,
    clear_existing=False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    metadata_template: Optional[dict] = None
):
    """Embed text into ChromaDB with optional progress callback.

    Args:
        text: Text content to embed.
        document_name: Name of the document.
        document_description: Description of the document.
        persist_directory: Directory for ChromaDB persistence.
        clear_existing: If True, clear existing documents before adding.
        progress_callback: Optional callback(current, total, status_message).
        metadata_template: Optional template dictionary to merge into chunk metadatas.
    """
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        model_name=config.OLLAMA_EMBEDDING_MODEL,
        url=f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/embeddings",
    )
    documents = split_text_into_sections(text, 1000)
    total_docs = len(documents)

    if progress_callback:
        progress_callback(0, total_docs, f"Preparing to embed {total_docs} document chunks...")

    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = config.CHROMA_COLLECTION_NAME
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ollama_ef,
    )

    # Optionally clear existing documents
    if clear_existing:
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)
            print(f"Cleared {len(existing_ids)} existing documents")
            if progress_callback:
                progress_callback(0, total_docs, f"Cleared {len(existing_ids)} existing documents")

    # Generate metadata for each chunk
    metadatas = []
    for chunk in documents:
        meta = {
            "name": document_name,
            "description": document_description,
            "authority_rank": 3,  # Default to news/medium
            "source_type": "news",
            "effective_from": Utils.get_current_date()
        }
        if metadata_template:
            meta.update(metadata_template)

        # If it's the official AI Act, apply temporal logic based on article rules
        if document_name == "Artificial Intelligence Act":
            meta["authority_rank"] = 1
            meta["source_type"] = "legislation"
            
            chunk_lower = chunk.lower()
            if "article 5" in chunk_lower or "prohibited" in chunk_lower:
                meta["effective_from"] = "2025-02-02"
            elif "chapter v" in chunk_lower or "gpai" in chunk_lower or "general purpose" in chunk_lower:
                meta["effective_from"] = "2025-08-02"
            elif "article 6" in chunk_lower or "high-risk" in chunk_lower or "annex iii" in chunk_lower:
                meta["effective_from"] = "2026-08-02"
            else:
                meta["effective_from"] = "2024-08-02"

        metadatas.append(meta)

    # create ids from the current count
    count = collection.count()
    print(f"Collection currently contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]

    # load the documents in batches of 100
    batch_size = 100
    for i in tqdm(
        range(0, len(documents), batch_size), desc="Adding documents", unit_scale=batch_size
    ):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],  # type: ignore
        )
        if progress_callback:
            current = min(i + batch_size, total_docs)
            progress_callback(current, total_docs, f"Embedded {current}/{total_docs} chunks...")

    new_count = collection.count()
    print(f"Added {new_count - count} documents")

    if progress_callback:
        progress_callback(total_docs, total_docs, f"Complete! Added {new_count - count} documents")


def embed_articles_from_files(
    articles_metadata,
    clear_existing=False,
    persist_directory=config.CHROMA_PERSIST_DIRECTORY,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
):
    """Embed scraped articles from local files into ChromaDB individually with metadata."""
    total_articles = len(articles_metadata)

    if progress_callback:
        progress_callback(0, total_articles, f"Loading {total_articles} articles...")

    # Clear previous NEWS chunks only.
    #
    # This used to delete every id in the collection, so the first news fetch wiped
    # the EU AI Act corpus and every later legal query abstained for lack of
    # authoritative context. Statutory documents (source_type != "news") are now
    # preserved and their presence is verified afterwards.
    if clear_existing:
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            model_name=config.OLLAMA_EMBEDDING_MODEL,
            url=f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/embeddings",
        )
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=ollama_ef,
        )

        total_before = collection.count()
        news_ids = []
        try:
            news_ids = collection.get(where={"source_type": "news"}).get("ids", []) or []
        except Exception as exc:
            print(f"[embed] Warning: could not query news chunks ({exc}); nothing cleared")

        if news_ids:
            collection.delete(ids=news_ids)
            print(f"[embed] Cleared {len(news_ids)} news chunk(s); kept {total_before - len(news_ids)} non-news chunk(s)")
        else:
            print("[embed] No previous news chunks to clear")

        remaining = collection.count()
        if remaining == 0:
            print(
                "[embed] WARNING the collection is now empty. The statutory corpus is "
                "missing and legal queries will abstain until it is re-ingested."
            )

    for idx, article in enumerate(articles_metadata):
        file_path = article.get('file', '')
        if OS.path.exists(file_path):
            content = Utils.load_article_content(file_path)
            if content:
                # Add article metadata header
                header = f"\n\n=== Article: {article.get('title', 'Untitled')} ===\n"
                header += f"Source: {article.get('source', 'Unknown')}\n"
                header += f"URL: {article.get('url', 'N/A')}\n\n"
                full_text = header + content

                # Extract date or use current
                fetched_at = article.get('fetched_at', '')
                effective_date = fetched_at[:10] if fetched_at else Utils.get_current_date()

                meta_template = {
                    "authority_rank": 3,
                    "source_type": "news",
                    "effective_from": effective_date,
                    "url": article.get('url', 'N/A')
                }

                embed_text_in_chromadb(
                    full_text,
                    document_name=article.get('title', 'News Article'),
                    document_description=f"Source: {article.get('source', 'Unknown')}",
                    persist_directory=persist_directory,
                    clear_existing=False,
                    metadata_template=meta_template
                )

        if progress_callback:
            progress_callback(idx + 1, total_articles, f"Loaded & Embedded {idx + 1}/{total_articles} articles...")
   
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest the EU AI Act into ChromaDB (the legal expert's corpus)."
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Read the Act from a local PDF instead of downloading it. Use this when "
        "every remote source is unreachable (EUR-Lex serves non-browser clients an "
        "AWS WAF challenge).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Clear the collection first. Use this when the store holds stale content - "
        "a live news fetch with clear_existing=True replaces the Act with news chunks, "
        "which is how the corpus silently shrank to 44 news chunks before the pivot.",
    )
    args = parser.parse_args()

    document_name = "Artificial Intelligence Act"
    document_description = "Artificial Intelligence Act"

    if args.pdf:
        text = pdf_to_text_from_file(args.pdf)
        source_label = f"local file {args.pdf}"
        if len(text) < 200_000:
            raise SystemExit(
                f"{args.pdf} yielded only {len(text)} characters; that is not the full Act."
            )
        print(f"[embed] got {len(text):,} characters from {source_label}")
    else:
        text, source_label = fetch_eu_ai_act_text()

    embed_text_in_chromadb(
        text,
        document_name,
        document_description,
        clear_existing=args.replace,
    )
    print(f"[embed] source: {source_label}")
    print(f"[embed] collection now holds {Utils.get_db_document_count()} chunks")

