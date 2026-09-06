"""
Cache — stage 3 of DOCKER.md.

Enrichment is deterministic when the model is off, so the same input always
produces the same output and is safe to cache indefinitely. With the model on
it is close enough to deterministic (temperature 0) that a bounded TTL is fine.

As everywhere else in this layer, a missing or unreachable Redis is not an
error. `Cache.available` goes False, every call becomes a no-op, and the
pipeline simply recomputes.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .config import Settings, get_settings

KEY_PREFIX = "pi:v1"


def cache_key(fingerprint: str, use_llm: bool) -> str:
    """
    Namespaced key.

    `use_llm` is part of the key on purpose: a deterministic result and a
    model-enriched result for the same product are different answers and must
    never be served for one another.
    """
    mode = "llm" if use_llm else "det"
    return f"{KEY_PREFIX}:{mode}:{fingerprint}"


class Cache:
    """Redis-backed result cache that degrades to a no-op."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client = None
        self.hits = 0
        self.misses = 0

        if self.settings.cache_enabled:
            try:
                import redis  # local import: optional dependency

                client = redis.Redis.from_url(
                    self.settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                client.ping()
                self._client = client
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def ping(self) -> bool:
        if not self.available:
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    # -- operations --------------------------------------------------------

    def get(self, fingerprint: str, use_llm: bool = False) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        try:
            blob = self._client.get(cache_key(fingerprint, use_llm))
        except Exception:
            return None

        if blob is None:
            self.misses += 1
            return None
        try:
            value = json.loads(blob)
        except json.JSONDecodeError:
            # Corrupt entry is a miss, not a crash. Drop it and move on.
            self.misses += 1
            self.delete(fingerprint, use_llm)
            return None

        self.hits += 1
        return value

    def set(
        self, fingerprint: str, result: Dict[str, Any],
        use_llm: bool = False, ttl: Optional[int] = None,
    ) -> bool:
        if not self.available:
            return False
        try:
            self._client.set(
                cache_key(fingerprint, use_llm),
                json.dumps(result, default=str),
                ex=ttl if ttl is not None else self.settings.cache_ttl_seconds,
            )
            return True
        except Exception:
            return False

    def delete(self, fingerprint: str, use_llm: bool = False) -> bool:
        if not self.available:
            return False
        try:
            return bool(self._client.delete(cache_key(fingerprint, use_llm)))
        except Exception:
            return False

    def clear(self) -> int:
        """Remove every key this application owns. Leaves other keys alone."""
        if not self.available:
            return 0
        try:
            removed = 0
            for key in self._client.scan_iter(match=f"{KEY_PREFIX}:*", count=500):
                removed += self._client.delete(key)
            return removed
        except Exception:
            return 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "available": self.available,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
