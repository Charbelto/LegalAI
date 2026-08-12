"""Retrieval Agent - Performs vector search on ChromaDB."""

from typing import Any, Dict, List
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from agents.base import BaseAgent
import config
import utils as Utils


import math
import re
from typing import Any, Dict, List
from collections import Counter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from agents.base import BaseAgent
import config
import utils as Utils


class BM25Retriever:
    """Pure-python implementation of BM25 for sparse retrieval."""

    def __init__(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus  # List of {"content": str, "doc": Document}
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.df = Counter()
        self.doc_tfs = []
        self.N = len(corpus)
        self.k1 = 1.5
        self.b = 0.75
        self._build()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _build(self):
        total_len = 0
        for doc in self.corpus:
            tokens = self._tokenize(doc["content"])
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            
            tf = Counter(tokens)
            self.doc_tfs.append(tf)
            
            for word in tf.keys():
                self.df[word] += 1
                
        self.avg_doc_len = total_len / self.N if self.N > 0 else 1.0

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        score = 0.0
        tf = self.doc_tfs[doc_idx]
        d_len = self.doc_len[doc_idx]
        
        for token in query_tokens:
            if token not in tf:
                continue
            
            df = self.df.get(token, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            
            term_tf = tf[token]
            num = term_tf * (self.k1 + 1)
            denom = term_tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len))
            score += idf * (num / denom)
            
        return score

    def search(self, query: str, top_k: int = 10) -> List[tuple[float, Document]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or self.N == 0:
            return []
            
        scores = []
        for i in range(self.N):
            s = self.score(query_tokens, i)
            if s > 0:
                scores.append((s, self.corpus[i]["doc"]))
                
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]


class RetrievalAgent(BaseAgent):
    """Retrieval Agent searches the vector store for relevant documents using hybrid search."""

    def __init__(self):
        """Initialize the Retrieval Agent."""
        super().__init__(temperature=0.0)
        self.embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
        self.db = None
        self._init_db()

    def _init_db(self):
        """Initialize the ChromaDB connection."""
        try:
            self.db = Chroma(
                persist_directory=config.CHROMA_PERSIST_DIRECTORY,
                embedding_function=self.embeddings,
                collection_name=config.CHROMA_COLLECTION_NAME,
            )
            self.log(f"Connected to ChromaDB at {config.CHROMA_PERSIST_DIRECTORY}")
        except Exception as e:
            self.log(f"Error connecting to ChromaDB: {e}")
            self.db = None

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hybrid BM25 + dense search and metadata-aware reranking.

        Args:
            state: The current state with 'query'.

        Returns:
            Updated state with 'retrieved_docs' populated.
        """
        query = state.get("query", "")
        self.log(f"Retrieving documents for: {query[:50]}...")

        if self.db is None:
            self.log("Database not initialized, attempting to reconnect")
            self._init_db()

        if self.db is None:
            self.log("Failed to connect to database, returning empty results")
            state["retrieved_docs"] = []
            return state

        try:
            # 1. Dense retrieval (Top 10)
            dense_docs = []
            try:
                dense_results = self.db.similarity_search_with_relevance_scores(query, k=10)
                dense_docs = [doc for doc, score in dense_results]
            except Exception as e:
                self.log(f"Dense vector search failed: {e}")
                # Fallback to standard similarity search
                dense_docs = self.db.similarity_search(query, k=10)

            # 2. Sparse BM25 retrieval (Top 10)
            bm25_docs = []
            all_chunks = self.db._collection.get(include=["documents", "metadatas"])
            content_to_id = {}
            if all_chunks and all_chunks.get("documents"):
                chunk_ids = all_chunks.get("ids", [])
                corpus = []
                for idx, (text, meta) in enumerate(zip(all_chunks["documents"], all_chunks.get("metadatas", []))):
                    db_id = chunk_ids[idx] if idx < len(chunk_ids) else f"chunk_{idx}"
                    meta_dict = dict(meta) if meta else {}
                    meta_dict["id"] = db_id
                    content_to_id[text] = db_id
                    corpus.append({
                        "content": text,
                        "doc": Document(page_content=text, metadata=meta_dict)
                    })
                bm25_retriever = BM25Retriever(corpus)
                bm25_results = bm25_retriever.search(query, top_k=10)
                bm25_docs = [doc for score, doc in bm25_results]

            # 2.5 Attach IDs to dense docs as well
            for doc in dense_docs:
                if doc.page_content in content_to_id:
                    doc.metadata = doc.metadata or {}
                    doc.metadata["id"] = content_to_id[doc.page_content]
                else:
                    import hashlib
                    h = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:8]
                    doc_name = doc.metadata.get("name", "doc") if doc.metadata else "doc"
                    doc.metadata = doc.metadata or {}
                    doc.metadata["id"] = f"{doc_name}_{h}"

            # 3. Merge & Deduplicate
            merged_docs = {}
            for rank, doc in enumerate(dense_docs):
                merged_docs[doc.page_content] = {
                    "doc": doc,
                    "dense_rank": rank,
                    "bm25_rank": 999
                }
            for rank, doc in enumerate(bm25_docs):
                if doc.page_content in merged_docs:
                    merged_docs[doc.page_content]["bm25_rank"] = rank
                else:
                    merged_docs[doc.page_content] = {
                        "doc": doc,
                        "dense_rank": 999,
                        "bm25_rank": rank
                    }

            # 4. Reranking algorithm
            # Incorporates relevance score proxy, authority_rank, and temporal metadata
            is_recency_query = any(word in query.lower() for word in ["recent", "news", "update", "this week", "latest", "2025", "2026"])
            scored_docs = []

            for content, info in merged_docs.items():
                doc = info["doc"]
                meta = doc.metadata or {}
                
                # Normalize ranks
                dense_score = (10 - info["dense_rank"]) / 10 if info["dense_rank"] != 999 else 0.0
                bm25_score = (10 - info["bm25_rank"]) / 10 if info["bm25_rank"] != 999 else 0.0
                text_score = 0.5 * dense_score + 0.5 * bm25_score

                # Authority boost: rank 1 (legislation) -> +0.2, rank 2 (official guidelines/Q&A) -> +0.1, rank 3 -> 0
                authority_rank = int(meta.get("authority_rank", 3))
                authority_boost = 0.1 * (3 - authority_rank)

                # Temporal/Recency boost
                recency_boost = 0.0
                if is_recency_query:
                    # News or newer effective dates get boosted
                    effective_from = str(meta.get("effective_from", ""))
                    if "2025" in effective_from:
                        recency_boost += 0.1
                    elif "2026" in effective_from:
                        recency_boost += 0.2
                    if meta.get("source_type") == "news":
                        recency_boost += 0.15

                total_score = text_score + authority_boost + recency_boost
                scored_docs.append((total_score, doc))

            # Sort by total score and keep top 5
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_docs = [doc for score, doc in scored_docs[:5]]

            state["retrieved_docs"] = top_docs
            self.log(f"Retrieved and reranked {len(top_docs)} documents using hybrid search.")

        except Exception as e:
            self.log(f"Error during hybrid retrieval: {e}")
            state["retrieved_docs"] = []

        return state

    def format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into context string.

        Args:
            docs: List of Document objects.

        Returns:
            Formatted context string with citation-grade metadata details.
        """
        if not docs:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            meta = doc.metadata or {}
            source = meta.get("name", "Unknown Source")
            effective = meta.get("effective_from", "N/A")
            authority = meta.get("authority_rank", 3)
            auth_str = "Legislation" if authority == 1 else "Official Guidelines" if authority == 2 else "News/Articles"

            context_parts.append(
                f"[Document {i}]: {source}\n"
                f"Source Type: {auth_str} | Effective Date: {effective}\n"
                f"Content:\n{content}\n"
            )

        return "\n".join(context_parts)
