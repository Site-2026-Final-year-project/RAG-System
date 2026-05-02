-- Run once on the DB that stores `chat_sessions` (same as RAG Chat API DATABASE_URL).
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS vehicle_id VARCHAR(36);
CREATE INDEX IF NOT EXISTS ix_chat_sessions_vehicle_id ON chat_sessions (vehicle_id);
