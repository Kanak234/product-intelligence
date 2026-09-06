"""
Vector search — stage 4 of DOCKER.md.

Purpose: near-duplicate detection across a catalogue. The same pump listed
twice under two supplier part numbers is the problem this solves, and it is one
that no per-record rule can see.

Two embedding paths:

1. The local model's embedding endpoint, when Ollama is reachable.
2. A deterministic lexical fallback — hashed character trigrams — when it is
   not.

The fallback is not as good as a real embedding model at paraphrase, but it is
reproducible, needs no model server, and is genuinely effective at catching the
near-identical strings that dominate real catalogue duplication. It also means
this layer is testable without a GPU, which matters more than the last few
points of recall.

`QDRANT_URL=":memory:"` runs Qdrant embedded in-process, so no server is
required for development or tests.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Settings, get_settings

VECTOR_SIZE = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# -- deterministic fallback embedding ---------------------------------------

def _trigrams(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    joined = " ".join(tokens)
    if len(joined) < 3:
        return tokens or [joined]
    return [joined[i : i + 3] for i in range(len(joined) - 2)]


def lexical_embedding(text: str, size: int = VECTOR_SIZE) -> List[float]:
    """
    Hashed character-trigram vector, L2-normalised.

    Deterministic across processes and machines: the hash is sha1 rather than
    Python's salted `hash()`, so a vector written today matches one computed
    tomorrow in a different process.
    """
    vector = [0.0] * size
    for gram in _trigrams(text):
        digest = hashlib.sha1(gram.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % size
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def product_text(result: Dict[str, Any]) -> str:
    """Text used for similarity: name, description and specification values."""
    parts: List[str] = [str(result.get("name") or "")]

    description = result.get("enriched_description")
    if isinstance(description, dict):
        description = description.get("value")
    if isinstance(description, str):
        parts.append(description)

    for key, field in (result.get("specifications") or {}).items():
        value = field.get("value") if isinstance(field, dict) else field
        unit = field.get("unit") if isinstance(field, dict) else None
        parts.append(f"{key} {value}{f' {unit}' if unit else ''}")

    return " ".join(p for p in parts if p)


# -- store -------------------------------------------------------------------

class VectorStore:
    """
    Qdrant-backed similarity index.

    Like the other services, unavailability is not fatal: `available` goes
    False and `upsert`/`search` become no-ops.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedder: Optional[Any] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection = self.settings.qdrant_collection
        self.embed = embedder or lexical_embedding
        self._client = None

        try:
            from qdrant_client import QdrantClient  # local import: optional
            from qdrant_client.models import Distance, VectorParams

            url = self.settings.qdrant_url
            self._client = (
                QdrantClient(location=":memory:") if url == ":memory:"
                # check_compatibility is off because it makes a version call on
                # construction; an unreachable server should surface as
                # `available == False`, not as a warning on import.
                else QdrantClient(url=url, check_compatibility=False)
            )
            existing = {c.name for c in self._client.get_collections().collections}
            if self.collection not in existing:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                )
        except Exception:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def upsert(self, product_id: str, result: Dict[str, Any]) -> bool:
        if not self.available:
            return False
        try:
            from qdrant_client.models import PointStruct

            self._client.upsert(
                collection_name=self.collection,
                points=[PointStruct(
                    id=_point_id(product_id),
                    vector=self.embed(product_text(result)),
                    payload={
                        "product_id": product_id,
                        "name": result.get("name"),
                        "category": _category_value(result),
                        # Carried so duplicate confirmation needs no database
                        # round trip. Cosine alone cannot tell a duplicate from
                        # a different size in the same family; the specs can.
                        "specs": comparable_specs(result),
                    },
                )],
            )
            return True
        except Exception:
            return False

    def upsert_many(self, items: Sequence[Tuple[str, Dict[str, Any]]]) -> int:
        return sum(1 for pid, res in items if self.upsert(pid, res))

    def search(self, result: Dict[str, Any], limit: int = 5,
               min_score: float = 0.0) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        try:
            hits = self._client.query_points(
                collection_name=self.collection,
                query=self.embed(product_text(result)),
                limit=limit,
                score_threshold=min_score or None,
            ).points
        except Exception:
            return []

        return [{
            "product_id": (h.payload or {}).get("product_id"),
            "name": (h.payload or {}).get("name"),
            "category": (h.payload or {}).get("category"),
            "specs": (h.payload or {}).get("specs") or {},
            "score": round(float(h.score), 4),
        } for h in hits]

    def find_duplicates(self, result: Dict[str, Any], threshold: Optional[float] = None,
                        exclude_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Near-duplicates of `result`, confirmed against their specifications.

        Similarity alone is not sufficient and measurement says so plainly. With
        the lexical fallback embedding, an 11 kW motor and an 18.5 kW motor in
        the same product family score 0.87, while a genuine duplicate written
        with different spacing scores 0.85. Any single cosine threshold
        therefore either misses real duplicates or flags different products.

        So the vector search is used only to generate candidates, and each
        candidate is confirmed by comparing specifications: two records that
        disagree on a measured value are different products, whatever their
        text similarity. That check is unit-aware and reuses the engine's own
        comparison, so 11 kW and 11000 W agree while 11 kW and 18.5 kW do not.
        """
        cutoff = self.settings.duplicate_threshold if threshold is None else threshold
        candidates = self.search(result, limit=10, min_score=cutoff)

        mine = comparable_specs(result)
        confirmed = []
        for candidate in candidates:
            if candidate.get("product_id") == exclude_id:
                continue
            conflict = spec_conflict(mine, candidate.get("specs") or {})
            if conflict is None:
                confirmed.append(candidate)
            else:
                candidate["rejected_because"] = conflict
        return confirmed

    def find_candidates(self, result: Dict[str, Any], threshold: Optional[float] = None,
                        exclude_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Similarity hits without spec confirmation. Useful for inspection."""
        cutoff = self.settings.duplicate_threshold if threshold is None else threshold
        return [
            c for c in self.search(result, limit=10, min_score=cutoff)
            if c.get("product_id") != exclude_id
        ]

    def count(self) -> int:
        if not self.available:
            return 0
        try:
            return int(self._client.count(collection_name=self.collection).count)
        except Exception:
            return 0

    def reset(self) -> bool:
        if not self.available:
            return False
        try:
            from qdrant_client.models import Distance, VectorParams

            self._client.delete_collection(self.collection)
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            return True
        except Exception:
            return False


def _point_id(product_id: str) -> int:
    """Qdrant wants an int or a UUID; map any string id to a stable int."""
    return int.from_bytes(hashlib.sha1(product_id.encode("utf-8")).digest()[:8], "big") >> 1


def _category_value(result: Dict[str, Any]) -> Optional[str]:
    category = result.get("category")
    if isinstance(category, dict):
        return category.get("value")
    return category


# -- specification comparison ------------------------------------------------

def comparable_specs(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    The numeric specifications worth comparing, as {key: [value, unit]}.

    Only values that are actually measurements: comparing free-text fields
    would reject duplicates over harmless wording differences.
    """
    out: Dict[str, Any] = {}
    for key, field in (result.get("specifications") or {}).items():
        if not isinstance(field, dict):
            continue
        value = field.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[key] = [float(value), field.get("unit") or ""]
    return out


def spec_conflict(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[str]:
    """
    First disagreement between two spec maps, or None if they are compatible.

    Keys present in only one side are not a conflict: incomplete data is normal
    and says nothing about whether two records describe the same product.
    """
    from product_intelligence import units

    for key in sorted(set(a) & set(b)):
        value_a, unit_a = a[key]
        value_b, unit_b = b[key]
        try:
            agrees = units.values_agree(value_a, unit_a, value_b, unit_b)
        except Exception:
            agrees = abs(value_a - value_b) <= 1e-9
        if not agrees:
            return f"{key}: {value_a}{unit_a} vs {value_b}{unit_b}"
    return None
