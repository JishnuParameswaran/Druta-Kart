"""
Druta Kart - RAG vector store.

Primary path: Supabase pgvector (match_documents RPC).
Fallback path: keyword search over raw knowledge-base text files.

The fallback ensures complaint_agent can always retrieve policy context even
when Supabase is unreachable (e.g. during unit tests or cold starts).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_KB_DIR = Path(__file__).parent / "knowledge_base"


# ---------------------------------------------------------------------------
# Supabase pgvector (primary path)
# ---------------------------------------------------------------------------

def _pgvector_search(query: str, n_results: int = 3) -> List[str]:
    """Query Supabase pgvector via the match_documents stored function."""
    try:
        from rag.embeddings import embed_texts
        from db.supabase_client import get_client

        embeddings = embed_texts([query])
        if not embeddings:
            return []

        client = get_client()
        result = client.rpc(
            "match_documents",
            {"query_embedding": embeddings[0], "match_count": n_results},
        ).execute()

        if result.data:
            return [row["content"] for row in result.data]
        return []
    except Exception as exc:
        logger.warning("pgvector search failed (%s); will use text fallback.", exc)
        return []


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

    Tries Supabase pgvector first; falls back to keyword search over raw text files.

    Args:
        query:     Natural language query (e.g. "refund policy for damaged items").
        n_results: Number of top results to return.

    Returns:
        Newline-separated relevant passages, or empty string if nothing found.
    """
    if not query or not query.strip():
        return ""

    # -- Try pgvector --
    docs = _pgvector_search(query, n_results)
    if docs:
        return "\n\n".join(docs)

    # -- Text fallback --
    chunks = _load_knowledge_base()
    if not chunks:
        logger.warning("Knowledge base is empty; no RAG context available.")
        return ""

    top = _keyword_search(query, chunks, n=n_results)
    return "\n\n".join(top)


def reset_collection() -> None:
    """No-op kept for test compatibility (was ChromaDB-specific)."""
    pass
