"""
项目配置文件
存放 API Key 等敏感信息
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _get_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# 智谱 AI API Key
# 优先从环境变量读取，如果没有则使用默认值
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
# GLM-4.5-Air  GLM-4.7-Flash
# 模型配置
MODEL_NAME = os.getenv("MODEL_NAME", "GLM-4.7-Flash")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.5"))

# 记忆系统配置
MEMORY_ENABLED = _get_bool("MEMORY_ENABLED", True)
MEMORY_AGENT_ID = os.getenv("MEMORY_AGENT_ID", "aide-react-agent")
MEMORY_PG_DSN = os.getenv("MEMORY_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:15432/aide")
MEMORY_EMBEDDING_MODEL = os.getenv("MEMORY_EMBEDDING_MODEL", "nomic-embed-text")
MEMORY_OLLAMA_BASE_URL = os.getenv("MEMORY_OLLAMA_BASE_URL", "http://127.0.0.1:11434")

MEMORY_RECALL_TOP_K = int(os.getenv("MEMORY_RECALL_TOP_K", "40"))
MEMORY_RETRIEVE_TOP_K = int(os.getenv("MEMORY_RETRIEVE_TOP_K", "8"))
MEMORY_MIN_SIMILARITY = float(os.getenv("MEMORY_MIN_SIMILARITY", "0.35"))
MEMORY_DEDUP_THRESHOLD = float(os.getenv("MEMORY_DEDUP_THRESHOLD", "0.92"))

MEMORY_TTL_PROFILE_DAYS = int(os.getenv("MEMORY_TTL_PROFILE_DAYS", "0"))  # 0 代表不过期
MEMORY_TTL_PROCEDURAL_DAYS = int(os.getenv("MEMORY_TTL_PROCEDURAL_DAYS", "180"))
MEMORY_TTL_SEMANTIC_DAYS = int(os.getenv("MEMORY_TTL_SEMANTIC_DAYS", "90"))
MEMORY_TTL_EPISODIC_DAYS = int(os.getenv("MEMORY_TTL_EPISODIC_DAYS", "30"))
