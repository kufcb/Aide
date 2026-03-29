import time
from typing import List

from langchain_ollama import OllamaEmbeddings


class NomicEmbedder:
    def __init__(self, model: str, base_url: str = ""):
        kwargs = {"model": model}
        if base_url:
            kwargs["base_url"] = base_url
        self._embeddings = OllamaEmbeddings(**kwargs)

    def embed(self, text: str, retries: int = 2) -> List[float]:
        last_error = None
        for attempt in range(retries + 1):
            try:
                return self._embeddings.embed_query(text)
            except Exception as e:
                last_error = e
                # Retry on transient errors: 502, 503, 504, connection errors
                error_text = str(e).lower()
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                is_transient = (
                    status_code in (502, 503, 504)
                    or "502" in error_text
                    or "503" in error_text
                    or "504" in error_text
                    or "connection" in error_text
                    or "timeout" in error_text
                )
                if not is_transient or attempt == retries:
                    raise
                time.sleep(0.5 * (attempt + 1))
        raise last_error

    def health_check(self) -> bool:
        try:
            self._embeddings.embed_query("health check")
            return True
        except Exception:
            return False
