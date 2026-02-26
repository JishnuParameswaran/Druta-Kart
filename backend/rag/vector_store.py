"""
Druta Kart - RAG vector store.

Primary path: ChromaDB with sentence-transformer embeddings.
Fallback path: keyword search over raw knowledge-base text files (no ML needed).

The fallback ensures complaint_agent can always retrieve policy context even
when ChromaDB is not initialised (e.g. during unit tests or cold starts).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_KB_DIR = Path(__file__).parent / "knowledge_base"
_CHROMA_DIR = Path(__file__).parent / "chroma_db"
_COLLECTION_NAME = "druta_kart_kb"

# Module-level cache
_chroma_collection = None


# ---------------------------------------------------------------------------
# ChromaDB (primary path)
# ---------------------------------------------------------------------------

def _get_collection():
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    try:
        import chromadb  # type: ignore
        from rag.embeddings import get_embedding_function  # type: ignore

        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        _chroma_collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )
        logger.info("ChromaDB collection '%s' loaded.", _COLLECTION_NAME)
        return _chroma_collection
    except Exception as exc:
        logger.warning("ChromaDB unavailable (%s); will use text fallback.", exc)
        return None


# ---------------------------------------------------------------------------
# Text fallback (keyword relevance scoring)
# ---------------------------------------------------------------------------

def _load_knowledge_base() -> List[dict]:
    """Load all .txt files from knowledge_base/ as plain text chunks."""
    chunks = []
    if not _KB_DIR.exists():
        return chunks
    for txt_file in sorted(_KB_DIR.glob("*.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8")
            # Split on double-newline to get rough paragraphs
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) > 30:
                    chunks.append({"source": txt_file.stem, "text": para})
        except Exception as exc:
            logger.warning("Failed to read %s: %s", txt_file, exc)
    return chunks


def _keyword_search(query: str, chunks: List[dict], n: int = 3) -> List[str]:
    """Return top-n chunks by keyword overlap with the query."""
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk["text"].lower().split())
        score = len(query_words & chunk_words)
        if score > 0:
            scored.append((score, chunk["text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:n]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_knowledge_base(query: str, n_results: int = 3) -> str:
    """Query the knowledge base and return relevant context as a single string.

    Tries ChromaDB first; falls back to keyword search over raw text files.

    Args:
        query:     Natural language query (e.g. "refund policy for damaged items").
        n_results: Number of top results to return.

    Returns:
        Newline-separated relevant passages, or empty string if nothing found.
    """
    if not query or not query.strip():
        return ""

    # -- Try ChromaDB --
    collection = _get_collection()
    if collection is not None:
        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count() or 1),
            )
            docs = results.get("documents", [[]])[0]
            if docs:
                return "\n\n".join(docs)
        except Exception as exc:
            logger.warning("ChromaDB query failed: %s; falling back to text search.", exc)

    # -- Text fallback --
    chunks = _load_knowledge_base()
    if not chunks:
        logger.warning("Knowledge base is empty; no RAG context available.")
        return ""

    top = _keyword_search(query, chunks, n=n_results)
    return "\n\n".join(top)


def reset_collection() -> None:
    """Reset the cached ChromaDB collection (useful in tests)."""
    global _chroma_collection
    _chroma_collection = None
