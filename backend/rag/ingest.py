"""
Druta Kart - Knowledge Base Ingestion Script.

Reads .txt files from rag/knowledge_base/, chunks them into meaningful
passages, and upserts them into ChromaDB.

Usage:
    cd backend
    python -m rag.ingest          # module mode
    python rag/ingest.py          # direct invocation

Idempotent: re-running performs a full collection reset followed by a fresh
upsert, so it is safe to run after editing knowledge-base files.
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
_CHROMA_DIR = _HERE / "chroma_db"
_COLLECTION_NAME = "druta_kart_kb"

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
    """Ingest the knowledge base into ChromaDB.

    Args:
        reset: If True (default), delete and recreate the collection before
               ingesting so stale entries from removed documents are purged.
               Set to False for additive / incremental upsert.

    Returns:
        Number of chunks successfully upserted.
    """
    try:
        import chromadb  # type: ignore
    except ImportError:
        logger.error("chromadb not installed. Run: pip install chromadb")
        return 0

    from rag.embeddings import get_embedding_function

    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(_COLLECTION_NAME)
            logger.info("Deleted existing collection '%s'.", _COLLECTION_NAME)
        except Exception:
            pass  # Collection may not exist yet on first run

    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )

    chunks = load_chunks()
    if not chunks:
        logger.warning("No chunks to ingest.")
        return 0

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Upsert in batches to keep memory usage bounded
    batch_size = 100
    ingested = 0
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        ingested += len(ids[start:end])
        logger.info("  Upserted %d / %d chunks.", ingested, len(chunks))

    logger.info(
        "Ingestion complete: %d chunks in collection '%s'.",
        ingested,
        _COLLECTION_NAME,
    )
    return ingested


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting knowledge-base ingestion...")
    logger.info("Knowledge base : %s", _KB_DIR)
    logger.info("ChromaDB path  : %s", _CHROMA_DIR)

    chunks = load_chunks()
    logger.info("Total chunks to ingest: %d", len(chunks))

    n = ingest(reset=True)
    logger.info("Done. %d chunks ingested.", n)
    sys.exit(0 if n > 0 else 1)
