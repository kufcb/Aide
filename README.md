# Aide

## Agent 记忆系统（nomic-embed-text + pgvector）

### 1. 执行数据库初始化 SQL

请手动执行：

- `sql/memory/001_init_agent_memory.sql`

### 2. 配置环境变量

在 `.env` 增加：

```env
MEMORY_ENABLED=true
MEMORY_PG_DSN=postgresql://user:password@127.0.0.1:5432/aide
MEMORY_AGENT_ID=aide-react-agent
MEMORY_EMBEDDING_MODEL=nomic-embed-text
MEMORY_OLLAMA_BASE_URL=http://127.0.0.1:11434

MEMORY_RECALL_TOP_K=40
MEMORY_RETRIEVE_TOP_K=8
MEMORY_MIN_SIMILARITY=0.35
MEMORY_DEDUP_THRESHOLD=0.92

MEMORY_TTL_PROFILE_DAYS=0
MEMORY_TTL_PROCEDURAL_DAYS=180
MEMORY_TTL_SEMANTIC_DAYS=90
MEMORY_TTL_EPISODIC_DAYS=30
```

### 3. 使用方式

`/chat/stream` 请求体现在支持：

- `msg`: 用户消息
- `user_id`: 用户 ID（用于隔离记忆）
- `session_id`: 会话 ID（用于同会话轻微加权）
- `agent_id`: Agent ID

示例：

```json
{
  "msg": "我喜欢简洁回答，以后都用要点式",
  "user_id": "u_1001",
  "session_id": "s_20260329_01",
  "agent_id": "aide-react-agent"
}
```
