-- RAG chunk store + vector index (runs once on empty Docker volume).
-- Matches docs/schema_rag_pgvector.sql so retrieval stays fast after large manual syncs.

CREATE TABLE IF NOT EXISTS rag_kb_chunks (
  id VARCHAR(36) PRIMARY KEY,
  scope VARCHAR(16) NOT NULL,
  owner_user_id VARCHAR(128),
  manual_id VARCHAR(36),
  chunk_index INT NOT NULL DEFAULT 0,
  content TEXT NOT NULL,
  embedding vector(384) NOT NULL,
  embedding_model VARCHAR(128) NOT NULL DEFAULT 'all-MiniLM-L6-v2',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_rag_kb_chunks_scope_owner
  ON rag_kb_chunks (scope, owner_user_id);

CREATE INDEX IF NOT EXISTS ix_rag_kb_chunks_manual
  ON rag_kb_chunks (manual_id);

CREATE INDEX IF NOT EXISTS ix_rag_kb_chunks_embedding_hnsw
  ON rag_kb_chunks
  USING hnsw (embedding vector_l2_ops);

CREATE TABLE IF NOT EXISTS rag_user_profile (
  user_id VARCHAR(128) PRIMARY KEY,
  vehicle_meta TEXT NOT NULL DEFAULT '',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
