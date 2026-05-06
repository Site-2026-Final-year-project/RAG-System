-- Adds rolling chat summary to sessions (for long-conversation coherence).
-- Run on the DB used by the RAG chat API.

ALTER TABLE chat_sessions
  ADD COLUMN IF NOT EXISTS chat_summary TEXT DEFAULT '';

