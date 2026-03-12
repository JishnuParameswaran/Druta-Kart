"""
Druta Kart - Knowledge Base Ingestion Script.

Reads .txt files from rag/knowledge_base/, chunks them into meaningful
passages, and upserts them into Supabase pgvector (documents table).

Usage:
    cd backend
    python -m rag.ingest          # module mode
    python rag/ingest.py          # direct invocation

Idempotent: re-running deletes all existing rows and re-inserts fresh embeddings,
so it is safe to run after editing knowledge-base files.

Prerequisites (run once in Supabase SQL editor):
    See supabase/migrations/001_pgvector.sql
"""
from __future__ import annotations

import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Iterator

# Allow running directly: python rag/ingest.py
_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

logger = logging.getLogger(__name__)

_KB_DIR = _HERE / "knowledge_base"

# Minimum characters for a chunk to be worth embedding
_MIN_CHUNK_LEN = 30

# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def _chunk_faq(text: str) -> Iterator[str]:
    """Yield individual Q&A pairs from an FAQ document.

    Splits on lines beginning with "Q:" so each question+answer is one chunk,
    giving the retriever the best chance of matching a user query to the
    correct answer without irrelevant context mixed in.
    """
    blocks = re.split(r"(?=^Q:)", text, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if block and len(block) >= _MIN_CHUNK_LEN:
            yield block


def _chunk_sections(text: str) -> Iterator[str]:
    """Yield passages from a policy document.

    Splits on Markdown ## headings first (keeping the heading with its body),
    then further splits each section on blank lines to get paragraph-sized
    chunks. This preserves the section heading as context for each paragraph.
    """
    # Split on ## headings, keeping the delimiter with the following content
    sections = re.split(r"(?=^##\s)", text, flags=re.MULTILINE)
    for section in sections:
        # Strip the overall document title (lines before any ## heading)
        # then break each section into paragraphs
        for para in section.split("\n\n"):
            para = para.strip()
            if len(para) >= _MIN_CHUNK_LEN:
                yield para


def _chunk_file(path: Path) -> list[dict]:
    """Chunk a single knowledge-base file into a list of chunk dicts.

    Returns:
        List of dicts with keys: id, text, metadata.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Cannot read %s: %s", path, exc)
        return []

    stem = path.stem  # e.g. "faq", "refund_policy"
    if "faq" in stem.lower():
        passages = list(_chunk_faq(text))
    else:
        passages = list(_chunk_sections(text))

    chunks = []
    for idx, passage in enumerate(passages):
        # Stable, content-based ID so re-ingesting the same text is idempotent
        chunk_id = hashlib.sha1(
            f"{stem}:{idx}:{passage}".encode()
        ).hexdigest()[:16]
        chunks.append({
            "id": chunk_id,
            "text": passage,
            "metadata": {"source": stem, "chunk_index": idx},
        })

    return chunks


def load_chunks() -> list[dict]:
    """Load and chunk all knowledge-base .txt files.

    Returns:
        Flat list of chunk dicts across all files.
    """
    if not _KB_DIR.exists():
        logger.error("Knowledge base directory not found: %s", _KB_DIR)
        return []

    all_chunks: list[dict] = []
    for txt_file in sorted(_KB_DIR.glob("*.txt")):
        file_chunks = _chunk_file(txt_file)
        logger.info("  %s → %d chunks", txt_file.name, len(file_chunks))
        all_chunks.extend(file_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest(reset: bool = True) -> int:
    """Ingest the knowledge base into Supabase pgvector.

    Args:
        reset: If True (default), delete all existing rows before ingesting
               so stale entries from removed documents are purged.
               Set to False for additive / incremental upsert.

    Returns:
        Number of chunks successfully upserted.
    """
    from rag.embeddings import embed_texts
    from db.supabase_client import get_client

    client = get_client()
    chunks = load_chunks()
    if not chunks:
        logger.warning("No chunks to ingest.")
        return 0

    if reset:
        client.table("documents").delete().neq("id", "").execute()
        logger.info("Cleared existing documents from Supabase.")

    # Embed and upsert in batches
    batch_size = 50
    ingested = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)

        rows = [
            {
                "id": c["id"],
                "content": c["text"],
                "metadata": c["metadata"],
                "embedding": emb,
            }
            for c, emb in zip(batch, embeddings)
        ]
        client.table("documents").upsert(rows).execute()
        ingested += len(rows)
        logger.info("  Upserted %d / %d chunks.", ingested, len(chunks))

    logger.info("Ingestion complete: %d chunks in Supabase pgvector.", ingested)
    return ingested


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting knowledge-base ingestion into Supabase pgvector...")
    logger.info("Knowledge base : %s", _KB_DIR)

    chunks = load_chunks()
    logger.info("Total chunks to ingest: %d", len(chunks))

    n = ingest(reset=True)
    logger.info("Done. %d chunks ingested.", n)
    sys.exit(0 if n > 0 else 1)
