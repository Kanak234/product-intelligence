"""
Configuration for the platform services.

Everything is environment-driven and everything has a default that means
"switched off". Importing this module must never fail and never connect to
anything — the Streamlit deployment imports the application with none of these
services present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # Persistence. Empty string means "no database"; the app then runs stateless.
    database_url: str = os.getenv("DATABASE_URL", "")

    # Cache and Celery broker. Empty means "no cache"; every request recomputes.
    redis_url: str = os.getenv("REDIS_URL", "")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    # Vector store. ":memory:" runs qdrant embedded, no server required.
    qdrant_url: str = os.getenv("QDRANT_URL", ":memory:")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "products")
    # Cosine score above which a stored product becomes a duplicate *candidate*.
    # This is a recall knob, not a precision one: candidates are confirmed
    # afterwards by comparing specifications, so a lower value here costs a
    # little search work and never produces a false duplicate.
    duplicate_threshold: float = float(os.getenv("DUPLICATE_THRESHOLD", "0.80"))

    # Local model.
    enable_llm: bool = _flag("ENABLE_LLM", False)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

    # Pipeline behaviour.
    batch_concurrency: int = int(os.getenv("BATCH_CONCURRENCY", "4"))
    stream_chunk_size: int = int(os.getenv("STREAM_CHUNK_SIZE", "500"))

    @property
    def persistence_enabled(self) -> bool:
        return bool(self.database_url)

    @property
    def cache_enabled(self) -> bool:
        return bool(self.redis_url)

    def describe(self) -> dict:
        """Safe summary for the /health endpoint. Never leaks credentials."""
        return {
            "persistence": "on" if self.persistence_enabled else "off",
            "cache": "on" if self.cache_enabled else "off",
            "vector_store": "embedded" if self.qdrant_url == ":memory:" else "server",
            "llm": "on" if self.enable_llm else "off",
            "llm_model": self.ollama_model if self.enable_llm else None,
        }


def get_settings() -> Settings:
    """Read settings fresh. Not cached, so tests can vary the environment."""
    return Settings(
        database_url=os.getenv("DATABASE_URL", ""),
        redis_url=os.getenv("REDIS_URL", ""),
        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
        qdrant_url=os.getenv("QDRANT_URL", ":memory:"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "products"),
        duplicate_threshold=float(os.getenv("DUPLICATE_THRESHOLD", "0.80")),
        enable_llm=_flag("ENABLE_LLM", False),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        batch_concurrency=int(os.getenv("BATCH_CONCURRENCY", "4")),
        stream_chunk_size=int(os.getenv("STREAM_CHUNK_SIZE", "500")),
    )
