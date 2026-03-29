import math
from datetime import datetime
from typing import Iterable, List

import psycopg

from memory.models import MemoryCandidate, MemoryRecord


class PgVectorMemoryStore:
    def __init__(
        self,
        dsn: str,
        agent_id: str,
        recall_top_k: int = 40,
        retrieve_top_k: int = 8,
        min_similarity: float = 0.35,
        dedup_threshold: float = 0.92,
    ):
        self.dsn = dsn
        self.agent_id = agent_id
        self.recall_top_k = recall_top_k
        self.retrieve_top_k = retrieve_top_k
        self.min_similarity = min_similarity
        self.dedup_threshold = dedup_threshold

    @staticmethod
    def _vector_literal(vector: List[float]) -> str:
        return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"

    @staticmethod
    def _freshness(created_at: datetime) -> float:
        now = datetime.now(created_at.tzinfo) if created_at.tzinfo else datetime.now()
        age_days = max((now - created_at).total_seconds() / 86400, 0)
        return math.exp(-age_days / 30)

    def retrieve(
        self,
        embedding: List[float],
        user_id: str,
        session_id: str,
    ) -> List[MemoryRecord]:
        vector = self._vector_literal(embedding)

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        content,
                        memory_type,
                        importance,
                        confidence,
                        created_at,
                        metadata,
                        1 - (embedding <=> %s::vector) AS similarity,
                        CASE WHEN session_id = %s THEN 1 ELSE 0 END AS same_session
                    FROM agent_memories
                    WHERE agent_id = %s
                      AND user_id = %s
                      AND (expires_at IS NULL OR expires_at > now())
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector, session_id, self.agent_id, user_id, vector, self.recall_top_k),
                )
                rows = cur.fetchall()

        scored = []
        for row in rows:
            similarity = float(row[7])
            if similarity < self.min_similarity:
                continue

            freshness = self._freshness(row[5])
            same_session = float(row[8])
            score = 0.70 * similarity + 0.20 * float(row[3]) + 0.10 * freshness + 0.05 * same_session
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[: self.retrieve_top_k]

        records = [
            MemoryRecord(
                id=row[0],
                content=row[1],
                memory_type=row[2],
                importance=float(row[3]),
                confidence=float(row[4]),
                created_at=row[5],
                metadata=row[6] or {},
                similarity=float(row[7]),
            )
            for _, row in selected
        ]

        self._mark_accessed(record.id for record in records)
        return records

    def retrieve_recent(
        self,
        user_id: str,
        session_id: str,
        limit: int = None,
    ) -> List[MemoryRecord]:
        row_limit = limit or self.retrieve_top_k

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        content,
                        memory_type,
                        importance,
                        confidence,
                        created_at,
                        metadata
                    FROM agent_memories
                    WHERE agent_id = %s
                      AND user_id = %s
                      AND (expires_at IS NULL OR expires_at > now())
                    ORDER BY
                        CASE WHEN session_id = %s THEN 0 ELSE 1 END,
                        CASE memory_type
                            WHEN 'profile' THEN 0
                            WHEN 'semantic' THEN 1
                            WHEN 'procedural' THEN 2
                            ELSE 3
                        END,
                        importance DESC,
                        created_at DESC
                    LIMIT %s
                    """,
                    (self.agent_id, user_id, session_id, row_limit),
                )
                rows = cur.fetchall()

        records = [
            MemoryRecord(
                id=row[0],
                content=row[1],
                memory_type=row[2],
                importance=float(row[3]),
                confidence=float(row[4]),
                created_at=row[5],
                metadata=row[6] or {},
                similarity=0.0,
            )
            for row in rows
        ]

        self._mark_accessed(record.id for record in records)
        return records

    def _mark_accessed(self, memory_ids: Iterable[int]) -> None:
        ids = list(memory_ids)
        if not ids:
            return

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE agent_memories
                    SET access_count = access_count + 1,
                        last_accessed_at = now()
                    WHERE id = ANY(%s)
                    """,
                    (ids,),
                )

    def upsert_candidate(
        self,
        embedding: List[float],
        user_id: str,
        session_id: str,
        candidate: MemoryCandidate,
    ) -> None:
        vector = self._vector_literal(embedding)
        metadata = candidate.metadata or {}

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, 1 - (embedding <=> %s::vector) AS similarity
                    FROM agent_memories
                    WHERE agent_id = %s
                      AND user_id = %s
                      AND memory_type = %s
                      AND (expires_at IS NULL OR expires_at > now())
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                    """,
                    (vector, self.agent_id, user_id, candidate.memory_type, vector),
                )
                similar = cur.fetchone()

                if similar and float(similar[1]) >= self.dedup_threshold:
                    cur.execute(
                        """
                        UPDATE agent_memories
                        SET content = CASE
                                WHEN char_length(%s) > char_length(content) THEN %s
                                ELSE content
                            END,
                            importance = GREATEST(importance, %s),
                            confidence = GREATEST(confidence, %s),
                            metadata = metadata || %s::jsonb,
                            last_accessed_at = now(),
                            access_count = access_count + 1,
                            session_id = %s
                        WHERE id = %s
                        """,
                        (
                            candidate.content,
                            candidate.content,
                            candidate.importance,
                            candidate.confidence,
                            psycopg.types.json.Jsonb(metadata),
                            session_id,
                            similar[0],
                        ),
                    )
                    return

                if candidate.ttl_days is None:
                    expires_at = None
                else:
                    expires_at = f"{candidate.ttl_days} days"

                if expires_at is None:
                    cur.execute(
                        """
                        INSERT INTO agent_memories (
                            agent_id,
                            user_id,
                            session_id,
                            memory_type,
                            content,
                            embedding,
                            importance,
                            confidence,
                            metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s, %s::jsonb)
                        """,
                        (
                            self.agent_id,
                            user_id,
                            session_id,
                            candidate.memory_type,
                            candidate.content,
                            vector,
                            candidate.importance,
                            candidate.confidence,
                            psycopg.types.json.Jsonb(metadata),
                        ),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO agent_memories (
                            agent_id,
                            user_id,
                            session_id,
                            memory_type,
                            content,
                            embedding,
                            importance,
                            confidence,
                            expires_at,
                            metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s::vector, %s, %s,
                            now() + (%s)::interval,
                            %s::jsonb
                        )
                        """,
                        (
                            self.agent_id,
                            user_id,
                            session_id,
                            candidate.memory_type,
                            candidate.content,
                            vector,
                            candidate.importance,
                            candidate.confidence,
                            expires_at,
                            psycopg.types.json.Jsonb(metadata),
                        ),
                    )
