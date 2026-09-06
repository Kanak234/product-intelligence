"""
Service layer — the single place where the engine and the infrastructure meet.

The engine knows nothing about databases, caches or model servers, and that is
deliberate: it is why the Streamlit deployment could drop all of them without
touching a line of `product_intelligence/`. This module is the seam.

Construct `ProductService()` with nothing configured and it behaves exactly
like the current Streamlit prototype: deterministic, stateless, in-process.
Configure services through the environment and each one activates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from product_intelligence import ProductIntelligencePipeline, RawProduct

from .cache import Cache
from .config import Settings, get_settings
from .db import Repository
from .vector import VectorStore


def build_llm_router(settings: Settings):
    """Router if the model is switched on and importable, else None."""
    if not settings.enable_llm:
        return None
    try:
        from app.services.llm.ollama_provider import OllamaProvider
        from app.services.llm.router import LLMRouter

        return LLMRouter({
            "ollama": OllamaProvider(
                base_url=settings.ollama_base_url,
                default_model=settings.ollama_model,
            )
        })
    except Exception:
        return None


class ProductService:
    """Enrichment with caching, persistence and duplicate detection layered on."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        repository: Optional[Repository] = None,
        cache: Optional[Cache] = None,
        vectors: Optional[VectorStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository if repository is not None else Repository(self.settings)
        self.cache = cache if cache is not None else Cache(self.settings)
        self.vectors = vectors if vectors is not None else VectorStore(self.settings)

        # A configured database is useless without tables, and nothing else in
        # the request path would create them.
        if self.repository.available:
            self.repository.ensure_schema()

        self.llm_router = build_llm_router(self.settings)
        self.pipeline = ProductIntelligencePipeline(
            llm_router=self.llm_router,
            model=self.settings.ollama_model if self.llm_router else None,
        )

    @property
    def llm_enabled(self) -> bool:
        return self.llm_router is not None

    # -- single product ----------------------------------------------------

    async def enrich(
        self,
        name: str,
        description: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        use_llm: Optional[bool] = None,
        persist: bool = True,
        find_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        Enrich one product, using the cache when it can and populating it when
        it cannot.

        The returned dict is the engine's own output plus a `platform` block
        describing what the infrastructure did. Nothing inside the engine's
        output is modified — a caller that ignores `platform` gets exactly the
        result the standalone prototype would produce.
        """
        want_llm = self.llm_enabled if use_llm is None else (use_llm and self.llm_enabled)
        raw = RawProduct(
            name=name.strip(),
            description=(description or "").strip(),
            attributes=dict(attributes or {}),
        )

        fingerprint = _fingerprint_of(raw)
        cached = self.cache.get(fingerprint, use_llm=want_llm) if fingerprint else None
        if cached is not None:
            cached = dict(cached)
            # A cache hit must not mean the record silently never reaches the
            # database. The cache can easily outlive it — a wiped database, a
            # restored backup, a Redis that was simply never cleared — and
            # without this check those products would be absent from storage
            # forever while every request kept succeeding.
            persisted_id = None
            if persist and self.repository.available:
                digest = cached.get("fingerprint")
                if digest and self.repository.find_by_hash(digest) is None:
                    persisted_id = self.repository.save_product(cached)
                    if persisted_id and self.vectors.available:
                        self.vectors.upsert(persisted_id, cached)

            cached["platform"] = {
                "cache": "hit",
                "persisted_id": persisted_id,
                "duplicates": [],
                "llm_used": want_llm,
            }
            return cached

        result = await self.pipeline.process(raw, use_llm=want_llm)

        # Store under the same key the lookup used. Using the engine's own
        # post-processing fingerprint here instead would mean every lookup
        # missed, because the lookup happens before the engine has run.
        self.cache.set(fingerprint, result, use_llm=want_llm)

        duplicates: List[Dict[str, Any]] = []
        if find_duplicates and self.vectors.available:
            duplicates = self.vectors.find_duplicates(result)

        product_id = self.repository.save_product(result) if persist else None
        if product_id and self.vectors.available:
            self.vectors.upsert(product_id, result)

        enriched = dict(result)
        enriched["platform"] = {
            "cache": "miss",
            "persisted_id": product_id,
            "duplicates": duplicates,
            "llm_used": want_llm,
        }
        return enriched

    # -- catalogue ---------------------------------------------------------

    async def enrich_catalogue(
        self,
        records: List[Dict[str, Any]],
        use_llm: Optional[bool] = None,
        persist: bool = True,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Batch enrichment plus the cross-record consistency report."""
        want_llm = self.llm_enabled if use_llm is None else (use_llm and self.llm_enabled)
        raws = [
            RawProduct(
                name=str(r.get("name", "")).strip(),
                description=str(r.get("description", "") or "").strip(),
                attributes=dict(r.get("attributes") or {}),
            )
            for r in records
            if str(r.get("name", "")).strip()
        ]

        batch = await self.pipeline.process_batch(
            raws, use_llm=want_llm, concurrency=self.settings.batch_concurrency
        )

        if persist and self.repository.available:
            ids = self.repository.save_products(batch.get("products", []), job_id=job_id)
            if self.vectors.available:
                self.vectors.upsert_many(list(zip(ids, batch.get("products", []))))

        batch["platform"] = {
            "persisted": persist and self.repository.available,
            "job_id": job_id,
            "llm_used": want_llm,
        }
        return batch

    async def enrich_catalogue_streaming(
        self,
        records: List[Dict[str, Any]],
        use_llm: Optional[bool] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Bounded-memory catalogue run.

        `process_batch` holds every record in memory at once, which is wrong for
        a real catalogue. This streams chunks and persists as it goes, so peak
        memory stays flat regardless of size. Cross-record consistency is not
        produced here — it needs the whole set by definition.
        """
        want_llm = self.llm_enabled if use_llm is None else (use_llm and self.llm_enabled)
        raws = [
            RawProduct(
                name=str(r.get("name", "")).strip(),
                description=str(r.get("description", "") or "").strip(),
                attributes=dict(r.get("attributes") or {}),
            )
            for r in records
            if str(r.get("name", "")).strip()
        ]

        processed = 0
        async for chunk in self.pipeline.process_stream(
            raws,
            use_llm=want_llm,
            concurrency=self.settings.batch_concurrency,
            chunk_size=self.settings.stream_chunk_size,
        ):
            products = chunk.get("products", [])
            if self.repository.available:
                ids = self.repository.save_products(products, job_id=job_id)
                if self.vectors.available:
                    self.vectors.upsert_many(list(zip(ids, products)))
            processed += len(products)
            if job_id:
                self.repository.update_job(job_id, status="running", processed=processed)

        if job_id:
            self.repository.update_job(
                job_id, status="done", processed=processed,
                summary={"total": len(raws), "processed": processed},
            )
        return {"job_id": job_id, "total": len(raws), "processed": processed}

    # -- diagnostics -------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        return {
            "engine": "ok",
            "database": self.repository.ping(),
            "cache": self.cache.ping(),
            "vector_store": self.vectors.available,
            "llm": self.llm_enabled,
            "config": self.settings.describe(),
        }


def _fingerprint_of(raw: RawProduct) -> str:
    """Fingerprint before processing, for the cache lookup."""
    from .db import content_hash

    return content_hash(raw.name, raw.description or "", raw.attributes)
