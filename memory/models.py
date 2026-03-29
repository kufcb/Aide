from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class MemoryRecord:
    id: int
    content: str
    memory_type: str
    importance: float
    confidence: float
    similarity: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryCandidate:
    content: str
    memory_type: str
    importance: float = 0.5
    confidence: float = 0.5
    ttl_days: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
