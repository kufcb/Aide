import re
from typing import List, Optional

from config import (
    MEMORY_AGENT_ID,
    MEMORY_DEDUP_THRESHOLD,
    MEMORY_EMBEDDING_MODEL,
    MEMORY_ENABLED,
    MEMORY_MIN_SIMILARITY,
    MEMORY_OLLAMA_BASE_URL,
    MEMORY_PG_DSN,
    MEMORY_RECALL_TOP_K,
    MEMORY_RETRIEVE_TOP_K,
    MEMORY_TTL_EPISODIC_DAYS,
    MEMORY_TTL_PROCEDURAL_DAYS,
    MEMORY_TTL_PROFILE_DAYS,
    MEMORY_TTL_SEMANTIC_DAYS,
)
from logs.logging_server import logger
from memory.models import MemoryCandidate, MemoryRecord

try:
    from memory.embedder import NomicEmbedder
    from memory.store import PgVectorMemoryStore
except Exception:  # pragma: no cover - 兜底，避免依赖未安装时服务崩溃
    NomicEmbedder = None
    PgVectorMemoryStore = None

_PROFILE_HINTS = (
    "我叫",
    "我的名字",
    "叫我",
    "我是",
    "我来自",
    "我在",
)

_PREFERENCE_HINTS = (
    "我喜欢",
    "我不喜欢",
    "我希望",
    "我想要",
    "偏好",
    "不要",
    "别用",
    "以后请",
)

_PROCEDURAL_HINTS = (
    "请用",
    "回复时",
    "格式",
    "步骤",
    "先",
    "再",
    "最后",
)


class MemoryService:
    def __init__(self):
        self.enabled = False
        self._embedder = None
        self._store = None

        if not MEMORY_ENABLED:
            logger.info("Memory disabled by MEMORY_ENABLED")
            return

        if not MEMORY_PG_DSN:
            logger.info("Memory disabled because MEMORY_PG_DSN is empty")
            return

        if NomicEmbedder is None or PgVectorMemoryStore is None:
            logger.info("Memory disabled because dependencies are not available")
            return

        try:
            self._embedder = NomicEmbedder(
                model=MEMORY_EMBEDDING_MODEL,
                base_url=MEMORY_OLLAMA_BASE_URL,
            )
            if not self._embedder.health_check():
                logger.warning(
                    "Memory embedding service is unreachable at startup; "
                    "running in degraded mode with runtime retries "
                    "(base_url=%s, model=%s)",
                    MEMORY_OLLAMA_BASE_URL,
                    MEMORY_EMBEDDING_MODEL,
                )
            self._store = PgVectorMemoryStore(
                dsn=MEMORY_PG_DSN,
                agent_id=MEMORY_AGENT_ID,
                recall_top_k=MEMORY_RECALL_TOP_K,
                retrieve_top_k=MEMORY_RETRIEVE_TOP_K,
                min_similarity=MEMORY_MIN_SIMILARITY,
                dedup_threshold=MEMORY_DEDUP_THRESHOLD,
            )
            self.enabled = True
            logger.info("Memory service enabled")
        except Exception:
            logger.exception("Failed to initialize memory service")

    def build_prompt_context(self, user_input: str, user_id: str, session_id: str) -> str:
        if not self.enabled or not user_input.strip():
            return ""

        try:
            embedding = self._embed_with_retry(user_input)
            memories = self._store.retrieve(
                embedding=embedding,
                user_id=user_id,
                session_id=session_id,
            )
            return self._format_memories(memories)
        except Exception:
            logger.exception("Memory retrieval failed, fallback to recent memories")
            try:
                memories = self._store.retrieve_recent(
                    user_id=user_id,
                    session_id=session_id,
                )
                return self._format_memories(memories)
            except Exception:
                logger.exception("Fallback memory retrieval failed")
                return ""

    def write_user_input(
        self,
        user_input: str,
        user_id: str,
        session_id: str,
    ) -> None:
        if not self.enabled:
            return

        user_text = self._sanitize(user_input)
        if not user_text:
            return

        candidates = self._extract_user_candidates(user_text)
        self._write_candidates(
            candidates=candidates,
            user_id=user_id,
            session_id=session_id,
            shared_embedding_text=user_text,
        )

    def write_turn(
        self,
        user_input: str,
        assistant_output: str,
        user_id: str,
        session_id: str,
    ) -> None:
        if not self.enabled:
            return

        user_text = self._sanitize(user_input)
        assistant_text = self._sanitize(assistant_output)
        if not user_text:
            return

        episodic_candidate = self._build_episodic_candidate(
            user_text=user_text,
            assistant_text=assistant_text,
        )
        if episodic_candidate is None:
            return

        self._write_candidates(
            candidates=[episodic_candidate],
            user_id=user_id,
            session_id=session_id,
        )

    def _extract_user_candidates(self, user_text: str) -> List[MemoryCandidate]:
        candidates: List[MemoryCandidate] = []
        dedup_guard = set()

        for sentence in self._split_sentences(user_text):
            clean_sentence = sentence.strip()
            if len(clean_sentence) < 6:
                continue
            if clean_sentence in dedup_guard:
                continue

            if any(hint in clean_sentence for hint in _PROFILE_HINTS):
                dedup_guard.add(clean_sentence)
                candidates.append(
                    MemoryCandidate(
                        content=clean_sentence,
                        memory_type="profile",
                        importance=0.90,
                        confidence=0.88,
                        ttl_days=self._ttl_or_none(MEMORY_TTL_PROFILE_DAYS),
                        metadata={"source": "user_profile"},
                    )
                )
                continue

            if any(hint in clean_sentence for hint in _PREFERENCE_HINTS):
                dedup_guard.add(clean_sentence)
                candidates.append(
                    MemoryCandidate(
                        content=clean_sentence,
                        memory_type="semantic",
                        importance=0.80,
                        confidence=0.82,
                        ttl_days=self._ttl_or_none(MEMORY_TTL_SEMANTIC_DAYS),
                        metadata={"source": "user_preference"},
                    )
                )
                continue

            if any(hint in clean_sentence for hint in _PROCEDURAL_HINTS):
                dedup_guard.add(clean_sentence)
                candidates.append(
                    MemoryCandidate(
                        content=clean_sentence,
                        memory_type="procedural",
                        importance=0.75,
                        confidence=0.80,
                        ttl_days=self._ttl_or_none(MEMORY_TTL_PROCEDURAL_DAYS),
                        metadata={"source": "interaction_rule"},
                    )
                )

        return candidates[:6]

    def _build_episodic_candidate(
        self,
        user_text: str,
        assistant_text: str,
    ) -> Optional[MemoryCandidate]:
        if not user_text:
            return None

        # 通用回合记忆：帮助模型回想近期事件。
        episodic_text = (
            f"用户说：{user_text}\n"
            f"助手回复：{assistant_text[:240]}"
        ).strip()
        if len(episodic_text) < 8:
            return None

        return MemoryCandidate(
            content=episodic_text,
            memory_type="episodic",
            importance=0.45,
            confidence=0.70,
            ttl_days=self._ttl_or_none(MEMORY_TTL_EPISODIC_DAYS),
            metadata={"source": "turn_summary"},
        )

    def _write_candidates(
        self,
        candidates: List[MemoryCandidate],
        user_id: str,
        session_id: str,
        shared_embedding_text: Optional[str] = None,
    ) -> None:
        if not candidates:
            return

        shared_embedding = None
        if shared_embedding_text:
            try:
                shared_embedding = self._embed_with_retry(shared_embedding_text)
            except Exception:
                logger.exception("Memory shared embedding failed")

        for candidate in candidates:
            try:
                embedding = shared_embedding
                if embedding is None:
                    embedding = self._embed_with_retry(candidate.content)

                self._store.upsert_candidate(
                    embedding=embedding,
                    user_id=user_id,
                    session_id=session_id,
                    candidate=candidate,
                )
            except Exception:
                logger.exception(
                    "Memory candidate write failed (type=%s, source=%s)",
                    candidate.memory_type,
                    (candidate.metadata or {}).get("source", "unknown"),
                )

    def _embed_with_retry(self, text: str, retries: int = 1):
        error = None
        for attempt in range(retries + 1):
            try:
                return self._embedder.embed(text)
            except Exception as exc:  # pragma: no cover - 依赖外部服务
                error = exc
                if attempt < retries:
                    logger.warning(
                        "Memory embedding failed, retrying (%s/%s)",
                        attempt + 1,
                        retries + 1,
                    )
        raise error

    @staticmethod
    def _format_memories(memories: List[MemoryRecord]) -> str:
        if not memories:
            return ""

        lines = [
            "以下是与当前用户相关的历史记忆，请仅在相关时使用；如果不确定，请忽略这些记忆："
        ]
        for idx, memory in enumerate(memories, start=1):
            text = memory.content.replace("\n", " ").strip()
            if len(text) > 160:
                text = text[:157] + "..."
            lines.append(f"{idx}. [{memory.memory_type}] {text}")

        return "\n".join(lines)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"[。！？!?；;\n]+", text)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _sanitize(text: str) -> str:
        text = text.replace("\x00", "").strip()
        if len(text) > 1500:
            return text[:1500]
        return text

    @staticmethod
    def _ttl_or_none(days: int):
        if days <= 0:
            return None
        return days


_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
