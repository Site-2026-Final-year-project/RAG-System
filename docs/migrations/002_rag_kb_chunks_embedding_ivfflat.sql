-- Fallback if HNSW is unavailable: approximate IVF index (tune `lists` to ~ rows/1000 for large tables).

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_rag_kb_chunks_embedding_ivfflat
  ON rag_kb_chunks
  USING ivfflat (embedding vector_l2_ops)
  WITH (lists = 200);

-- After building, refresh planner stats:
-- ANALYZE rag_kb_chunks;
