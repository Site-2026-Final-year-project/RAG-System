-- Run on the same PostgreSQL database used by your Express admin app.
-- Requires: CREATE privilege and ability to create extensions (or DBA runs CREATE EXTENSION once).

CREATE EXTENSION IF NOT EXISTS vector;

-- Chunk store for RAG (global + per-user manuals). Express can INSERT rows and call the Python
-- embed/sync job, or use scripts/build_index.py and scripts/upload_manual.py to fill embeddings.

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

-- Optional vehicle hint per app user (replaces local models/<user>/meta.txt when using Postgres).

CREATE TABLE IF NOT EXISTS rag_user_profile (
  user_id VARCHAR(128) PRIMARY KEY,
  vehicle_meta TEXT NOT NULL DEFAULT '',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
