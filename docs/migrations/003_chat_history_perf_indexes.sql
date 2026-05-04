-- Improves /sessions and /sessions/{id}/messages performance on deployed Postgres.
-- Run on the DB used by the RAG chat API.

-- For listing sessions by user ordered by most recent.
CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_updated
  ON chat_sessions (user_id, updated_at DESC);

-- For cursor pagination in chat history:
-- WHERE session_id = ? AND (created_at,id) < (?,?) ORDER BY created_at DESC, id DESC
CREATE INDEX IF NOT EXISTS ix_chat_messages_session_created_id
  ON chat_messages (session_id, created_at DESC, id DESC);

