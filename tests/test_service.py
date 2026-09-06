"""
Service layer tests.

The single most important property here: with nothing configured, the service
must return exactly what the standalone Streamlit prototype returns. Every
infrastructure feature is an addition on top of an unchanged engine result, and
if that stops being true the cloud deployment and the Compose stack have
silently become different products.
"""

from __future__ import annotations

import pytest

# The platform layer is optional. On a deployment that installed only
# requirements.txt these modules are absent, and this file must skip
# rather than fail collection.
pytest.importorskip("sqlalchemy", reason="service layer requires SQLAlchemy (requirements-platform.txt)")

from pi_platform.cache import Cache
from pi_platform.config import Settings
from pi_platform.db import Repository
from pi_platform.service import ProductService, build_llm_router
from pi_platform.vector import VectorStore
from product_intelligence import ProductIntelligencePipeline, RawProduct

MOTOR = "ABB M3BP 160MLA 11kW 400V 50Hz IE3 three-phase motor"
MOTOR_VARIANT = "ABB M3BP160MLA 11 kW 400 V IE3 3-phase motor"
BIGGER_MOTOR = "ABB M3BP 180MLA 18.5kW 400V 50Hz IE3 three-phase motor"


@pytest.fixture
def stateless(stateless_settings) -> ProductService:
    return ProductService(stateless_settings)


@pytest.fixture
def persistent(sqlite_settings) -> ProductService:
    service = ProductService(sqlite_settings)
    service.repository.create_schema()
    service.vectors.reset()
    yield service
    service.repository.drop_schema()


class TestStatelessParity:
    """Nothing configured — the cloud deployment's exact configuration."""

    def test_no_services_are_active(self, stateless):
        assert stateless.repository.available is False
        assert stateless.cache.available is False
        assert stateless.llm_enabled is False

    async def test_engine_output_is_untouched(self, stateless):
        direct = await ProductIntelligencePipeline().process(
            RawProduct(name=MOTOR), use_llm=False
        )
        through_service = await stateless.enrich(MOTOR)
        through_service.pop("platform")
        assert through_service == direct

    async def test_platform_block_reports_nothing_happened(self, stateless):
        result = await stateless.enrich(MOTOR)
        assert result["platform"]["persisted_id"] is None
        assert result["platform"]["cache"] == "miss"

    async def test_repeated_calls_are_stable(self, stateless):
        first = await stateless.enrich(MOTOR)
        second = await stateless.enrich(MOTOR)
        assert first["specifications"] == second["specifications"]

    async def test_catalogue_works_without_persistence(self, stateless, sample_records):
        batch = await stateless.enrich_catalogue(sample_records)
        assert batch["summary"]["total"] == 3
        assert batch["platform"]["persisted"] is False

    async def test_health_reports_everything_off(self, stateless):
        health = stateless.health()
        assert health["engine"] == "ok"
        assert health["database"] is False and health["cache"] is False


class TestPersistence:
    async def test_enrichment_is_stored(self, persistent):
        result = await persistent.enrich(MOTOR)
        product_id = result["platform"]["persisted_id"]
        assert product_id
        assert persistent.repository.get_product(product_id)["name"] == MOTOR

    async def test_persist_false_stores_nothing(self, persistent):
        result = await persistent.enrich(MOTOR, persist=False)
        assert result["platform"]["persisted_id"] is None
        assert persistent.repository.count_products() == 0

    async def test_blank_names_are_dropped_from_a_catalogue(self, persistent):
        batch = await persistent.enrich_catalogue(
            [{"name": MOTOR}, {"name": "   "}, {"name": ""}]
        )
        assert batch["summary"]["total"] == 1

    async def test_catalogue_rows_are_linked_to_the_job(self, persistent, sample_records):
        job_id = persistent.repository.create_job(total=len(sample_records))
        await persistent.enrich_catalogue(sample_records, job_id=job_id)
        assert len(persistent.repository.products_for_job(job_id)) == 3

    async def test_streaming_persists_every_record(self, persistent, sample_records):
        job_id = persistent.repository.create_job(total=len(sample_records))
        outcome = await persistent.enrich_catalogue_streaming(sample_records, job_id=job_id)
        assert outcome["processed"] == 3
        assert persistent.repository.count_products() == 3

    async def test_streaming_marks_the_job_done(self, persistent, sample_records):
        job_id = persistent.repository.create_job(total=len(sample_records))
        await persistent.enrich_catalogue_streaming(sample_records, job_id=job_id)
        job = persistent.repository.get_job(job_id)
        assert job["status"] == "done" and job["processed"] == 3


class TestDuplicateDetection:
    async def test_a_reworded_duplicate_is_flagged(self, persistent):
        await persistent.enrich(MOTOR)
        result = await persistent.enrich(MOTOR_VARIANT)
        assert [d["name"] for d in result["platform"]["duplicates"]] == [MOTOR]

    async def test_a_different_size_is_not_flagged(self, persistent):
        await persistent.enrich(MOTOR)
        result = await persistent.enrich(BIGGER_MOTOR)
        assert result["platform"]["duplicates"] == []

    async def test_the_first_product_has_no_duplicates(self, persistent):
        result = await persistent.enrich(MOTOR)
        assert result["platform"]["duplicates"] == []


class TestCaching:
    @pytest.fixture
    def cached(self, redis_settings, tmp_path):
        settings = Settings(
            database_url=f"sqlite:///{tmp_path/'c.db'}",
            redis_url=redis_settings.redis_url,
            qdrant_url=":memory:",
        )
        service = ProductService(settings)
        service.repository.create_schema()
        service.cache.clear()
        yield service
        service.cache.clear()
        service.repository.drop_schema()

    async def test_second_identical_request_hits_the_cache(self, cached):
        assert (await cached.enrich(MOTOR))["platform"]["cache"] == "miss"
        assert (await cached.enrich(MOTOR))["platform"]["cache"] == "hit"

    async def test_a_cache_hit_does_not_persist_again(self, cached):
        await cached.enrich(MOTOR)
        await cached.enrich(MOTOR)
        assert cached.repository.count_products() == 1

    async def test_a_cache_hit_still_stores_a_record_the_database_lacks(self, cached):
        """
        The cache can outlive the database — a wipe, a restore, a Redis that
        was never cleared. A hit must not mean the product is silently absent
        from storage forever.
        """
        await cached.enrich(MOTOR)
        cached.repository.drop_schema()
        cached.repository.create_schema()
        assert cached.repository.count_products() == 0

        result = await cached.enrich(MOTOR)
        assert result["platform"]["cache"] == "hit"
        assert result["platform"]["persisted_id"] is not None
        assert cached.repository.count_products() == 1

    async def test_a_cache_hit_honours_persist_false(self, cached):
        await cached.enrich(MOTOR, persist=False)
        result = await cached.enrich(MOTOR, persist=False)
        assert result["platform"]["cache"] == "hit"
        assert cached.repository.count_products() == 0

    async def test_a_cached_result_matches_the_computed_one(self, cached):
        computed = await cached.enrich(MOTOR)
        served = await cached.enrich(MOTOR)
        assert served["specifications"] == computed["specifications"]

    async def test_different_products_do_not_share_an_entry(self, cached):
        await cached.enrich(MOTOR)
        assert (await cached.enrich(BIGGER_MOTOR))["platform"]["cache"] == "miss"


class TestLLMWiring:
    def test_router_is_none_when_disabled(self, stateless_settings):
        assert build_llm_router(stateless_settings) is None

    def test_router_is_built_when_enabled(self):
        settings = Settings(enable_llm=True, ollama_base_url="http://127.0.0.1:1")
        assert build_llm_router(settings) is not None

    def test_service_reports_llm_off_by_default(self, stateless):
        assert stateless.llm_enabled is False

    async def test_use_llm_true_is_ignored_when_no_model_is_configured(self, stateless):
        result = await stateless.enrich(MOTOR, use_llm=True)
        assert all(
            field["origin"] != "llm" for field in result["specifications"].values()
        )


class TestInjection:
    """Components are injectable, which is what makes the layer testable."""

    async def test_accepts_prebuilt_components(self, stateless_settings):
        service = ProductService(
            stateless_settings,
            repository=Repository(stateless_settings),
            cache=Cache(stateless_settings),
            vectors=VectorStore(stateless_settings),
        )
        assert (await service.enrich(MOTOR))["name"] == MOTOR
