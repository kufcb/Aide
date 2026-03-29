-- Agent memory schema for pgvector
-- 注意：vector 维度默认 768（nomic-embed-text 常见维度），
-- 如果你实际 embedding 维度不同，请在执行前修改下方 VECTOR(768)。

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS agent_memories (
    id BIGSERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT,
    memory_type TEXT NOT NULL CHECK (memory_type IN ('profile', 'episodic', 'semantic', 'procedural')),
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    access_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_accessed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_agent_memories_hnsw
    ON agent_memories USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_agent_memories_scope_time
    ON agent_memories (agent_id, user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_memories_type
    ON agent_memories (memory_type);

CREATE INDEX IF NOT EXISTS idx_agent_memories_expiry
    ON agent_memories (expires_at);
