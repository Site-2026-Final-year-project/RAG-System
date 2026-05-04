-- Speed up pgvector similarity search after large manual syncs (350+ PDFs → many chunks).
-- Without an index on `embedding`, Postgres scans the whole table every chat turn.
--
-- Requires pgvector with HNSW support (common on pgvector ≥ 0.5). On older installs use
-- `002_rag_kb_chunks_embedding_ivfflat.sql` instead.
--
-- Run during low traffic; creation locks writes and uses RAM/CPU.

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_rag_kb_chunks_embedding_hnsw
  ON rag_kb_chunks
  USING hnsw (embedding vector_l2_ops);
