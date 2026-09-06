"""
Product Intelligence engine.

A deterministic-first pipeline that turns limited product information into
structured, validated, explainable catalogue data.

    from app.services.product_intelligence import ProductIntelligencePipeline, RawProduct

    pipeline = ProductIntelligencePipeline(llm_router=router)
    result = await pipeline.process(RawProduct(name="ABB 2.2kW 415V 3-phase motor"))
    print(result["validation"]["score"], result["explanation"]["summary"])

The deterministic layers (extract, classify, derive, validate, explain) have no
third-party dependencies and run without a model server, a database, or a
network. The language model is an optional gap-filler, never the source of
truth.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from .schema import (
    Evidence,
    EnrichedProduct,
    FieldValue,
    Origin,
    ORIGIN_TRUST,
    RawProduct,
)
from .units import UnitError, convert, to_canonical, values_agree
from .extractor import SpecExtractor, SPEC_PATTERNS
from .taxonomy import TAXONOMY, Category, TaxonomyClassifier, get_category, all_leaf_names
from .enricher import ProductEnricher
from .validator import (
    CatalogConsistencyChecker,
    Finding,
    ProductValidator,
    Severity,
)
from .explainer import ProductExplainer
from .ingestion import (
    IngestionError,
    IngestionResult,
    SUPPORTED_FORMATS,
    ingest,
)
from .evaluation import Evaluator, GroundTruthCase, load_dataset

__all__ = [
    "ProductIntelligencePipeline",
    "DEFAULT_CHUNK_SIZE",
    "RawProduct",
    "EnrichedProduct",
    "FieldValue",
    "Evidence",
    "Origin",
    "ORIGIN_TRUST",
    "ProductEnricher",
    "ProductValidator",
    "ProductExplainer",
    "CatalogConsistencyChecker",
    "SpecExtractor",
    "TaxonomyClassifier",
    "Category",
    "TAXONOMY",
    "SPEC_PATTERNS",
    "Finding",
    "Severity",
    "get_category",
    "all_leaf_names",
    "ingest",
    "IngestionResult",
    "IngestionError",
    "SUPPORTED_FORMATS",
    "Evaluator",
    "GroundTruthCase",
    "load_dataset",
    "convert",
    "to_canonical",
    "values_agree",
    "UnitError",
]

__version__ = "2.1.0"

#: Records held in memory at once by `process_stream`. Chosen so a chunk's
#: peak stays comfortably inside a small container while still amortising the
#: per-chunk overhead.
DEFAULT_CHUNK_SIZE = 500


class ProductIntelligencePipeline:
    """
    One entry point for the whole flow: enrich -> validate -> explain.

    This is what the API and the Celery worker both call, so a product
    processed synchronously and one processed in a batch go through exactly
    the same code and produce exactly the same shape of result.
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        model: Optional[str] = None,
        strict_validation: bool = False,
    ) -> None:
        self.enricher = ProductEnricher(llm_router=llm_router, model=model)
        self.validator = ProductValidator(strict=strict_validation)
        self.explainer = ProductExplainer()
        self.catalog_checker = CatalogConsistencyChecker()

    async def process(
        self,
        raw: RawProduct,
        use_llm: bool = True,
        include_trace: bool = False,
    ) -> Dict[str, Any]:
        product = await self.process_to_object(raw, use_llm=use_llm)
        return product.to_dict(include_trace=include_trace)

    async def process_to_object(
        self, raw: RawProduct, use_llm: bool = True
    ) -> EnrichedProduct:
        product = await self.enricher.enrich(raw, use_llm=use_llm)
        self.validator.validate(product)
        self.explainer.explain(product)
        return product

    async def process_stream(
        self,
        raws: List[RawProduct],
        use_llm: bool = True,
        concurrency: int = 4,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a catalogue in bounded chunks, yielding each chunk's results.

        `process_batch` holds every enriched record in memory at once. That is
        fine for a page of products and wrong for a catalogue: benchmarked at
        50,000 records it peaked near 950 MB, which would exhaust a modest
        container well before the CPU became the limit.

        This keeps only `chunk_size` records live, so peak memory is flat
        regardless of catalogue size. The caller streams results out - to a
        database, a CSV response, a queue - instead of collecting them.

        Cross-record consistency is deliberately *not* run here: it compares
        records against each other and therefore needs the whole set. Call
        `catalog_checker.check()` separately if you need it, accepting the
        memory that implies, or run it per chunk for a local approximation.
        """
        for offset in range(0, len(raws), chunk_size):
            chunk = raws[offset : offset + chunk_size]
            products = await self.enricher.enrich_many(
                chunk, use_llm=use_llm, concurrency=concurrency
            )
            for product in products:
                self.validator.validate(product)
                self.explainer.explain(product)

            yield {
                "offset": offset,
                "count": len(products),
                "products": [p.to_dict() for p in products],
            }

    async def process_batch(
        self,
        raws: List[RawProduct],
        use_llm: bool = True,
        concurrency: int = 4,
        include_trace: bool = False,
    ) -> Dict[str, Any]:
        """Enrich a whole catalogue and add the cross-record consistency report."""
        products = await self.enricher.enrich_many(
            raws, use_llm=use_llm, concurrency=concurrency
        )
        for product in products:
            self.validator.validate(product)
            self.explainer.explain(product)

        catalog_report = self.catalog_checker.check(products)
        valid = sum(1 for p in products if p.validation.get("is_valid"))

        return {
            "products": [p.to_dict(include_trace=include_trace) for p in products],
            "catalog_consistency": catalog_report,
            "summary": {
                "total": len(products),
                "valid": valid,
                "invalid": len(products) - valid,
                "mean_validation_score": round(
                    sum(p.validation.get("score", 0.0) for p in products) / len(products), 4
                ) if products else 0.0,
                "mean_completeness": round(
                    sum(p.completeness() for p in products) / len(products), 4
                ) if products else 0.0,
            },
        }
