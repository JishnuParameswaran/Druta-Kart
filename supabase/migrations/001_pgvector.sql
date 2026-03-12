-- Druta Kart: pgvector setup for RAG knowledge base
-- Run this once in Supabase SQL Editor (Dashboard → SQL Editor → New Query)

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Documents table (replaces ChromaDB)
CREATE TABLE IF NOT EXISTS documents (
    id       TEXT PRIMARY KEY,
    content  TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(384)          -- all-MiniLM-L6-v2 produces 384-dim vectors
);

-- 3. IVFFlat index for fast cosine similarity search
--    (lists = 10 works well for small KB; raise to 100 if > 10k rows)
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);

-- 4. Similarity search function called by vector_store.py
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(384),
    match_count     int DEFAULT 3
)
RETURNS TABLE(id text, content text, metadata jsonb, similarity float)
LANGUAGE SQL STABLE AS $$
    SELECT
        id,
        content,
        metadata,
        1 - (embedding <=> query_embedding) AS similarity
    FROM documents
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;

-- 5. RLS: allow service_role full access (backend uses service_role_key)
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_role_full_access" ON documents
    USING (true)
    WITH CHECK (true);
