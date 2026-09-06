"""
Persistence tests.

The behaviour that matters most is not "rows can be written" — it is that the
application works identically when there is no database at all. Half of these
tests are about the stateless path.
"""

from __future__ import annotations

import pytest

# The platform layer is optional. On a deployment that installed only
# requirements.txt these modules are absent, and this file must skip
# rather than fail collection.
pytest.importorskip("sqlalchemy", reason="persistence requires SQLAlchemy (requirements-platform.txt)")

from pi_platform.db import Repository, content_hash
from product_intelligence import ProductIntelligencePipeline, RawProduct

MOTOR = "ABB M3BP 160MLA 11kW 400V 50Hz IE3 three-phase motor"


@pytest.fixture(scope="module")
async def enriched():
    return await ProductIntelligencePipeline().process(
        RawProduct(name=MOTOR), use_llm=False
    )


class TestStatelessMode:
    """No database configured. Every call must be a harmless no-op."""

    def test_reports_unavailable(self, stateless_settings):
        assert Repository(stateless_settings).available is False

    def test_ping_is_false_not_an_exception(self, stateless_settings):
        assert Repository(stateless_settings).ping() is False

    async def test_save_returns_none(self, stateless_settings, enriched):
        assert Repository(stateless_settings).save_product(enriched) is None

    def test_reads_return_empty(self, stateless_settings):
        repo = Repository(stateless_settings)
        assert repo.recent_products() == []
        assert repo.get_product("anything") is None
        assert repo.get_job("anything") is None
        assert repo.count_products() == 0

    def test_schema_calls_are_safe(self, stateless_settings):
        assert Repository(stateless_settings).create_schema() is False

    def test_unusable_url_degrades_instead_of_raising(self):
        from pi_platform.config import Settings

        repo = Repository(Settings(database_url="not-a-real-url://x"))
        assert repo.available is False

    def test_unreachable_server_degrades_instead_of_raising(self):
        """
        Stage 1 of the Compose stack: DATABASE_URL points at a `db` service
        that has not been started. The app must come up stateless, not crash
        on the first write.
        """
        from pi_platform.config import Settings

        repo = Repository(Settings(
            database_url="postgresql+psycopg2://postgres:postgres@127.0.0.1:1/nothing"
        ))
        assert repo.available is False
        assert repo.save_product({"name": "x"}) is None


class TestContentHash:
    def test_is_stable_across_calls(self):
        assert content_hash("Pump A", "desc") == content_hash("Pump A", "desc")

    def test_ignores_surrounding_whitespace(self):
        assert content_hash("  Pump A  ", "desc") == content_hash("Pump A", "desc")

    def test_attribute_order_does_not_matter(self):
        assert content_hash("P", "", {"a": 1, "b": 2}) == content_hash("P", "", {"b": 2, "a": 1})

    def test_different_products_differ(self):
        assert content_hash("Pump A") != content_hash("Pump B")


class TestProductStorage:
    async def test_roundtrip_is_lossless(self, repository, enriched):
        product_id = repository.save_product(enriched)
        assert repository.get_product(product_id) == enriched

    async def test_metadata_columns_are_populated(self, repository, enriched):
        repository.save_product(enriched)
        row = repository.recent_products(1)[0]
        assert row["name"] == MOTOR
        assert row["category"]
        assert row["spec_count"] > 0
        assert 0.0 <= row["validation_score"] <= 1.0

    async def test_llm_flag_false_for_deterministic_result(self, repository, enriched):
        repository.save_product(enriched)
        assert repository.recent_products(1)[0]["llm_used"] is False

    async def test_llm_flag_true_when_a_field_came_from_the_model(self, repository, enriched):
        doctored = dict(enriched)
        doctored["specifications"] = dict(enriched["specifications"])
        doctored["specifications"]["ip_rating"] = {"value": "IP55", "origin": "llm", "confidence": 0.6}
        repository.save_product(doctored)
        assert repository.recent_products(1)[0]["llm_used"] is True

    async def test_lookup_by_fingerprint(self, repository, enriched):
        repository.save_product(enriched)
        found = repository.find_by_hash(enriched["fingerprint"])
        assert found == enriched

    async def test_missing_id_returns_none(self, repository):
        assert repository.get_product("00000000-0000-0000-0000-000000000000") is None

    async def test_recent_is_newest_first(self, repository, enriched):
        for name in ("First product", "Second product", "Third product"):
            result = dict(enriched)
            result["name"] = name
            repository.save_product(result)
        assert repository.recent_products(3)[0]["name"] == "Third product"

    async def test_batch_save_returns_all_ids(self, repository, enriched):
        ids = repository.save_products([enriched, enriched, enriched])
        assert len(ids) == 3
        assert repository.count_products() == 3


class TestJobs:
    def test_job_starts_queued(self, repository):
        job_id = repository.create_job(total=10)
        assert repository.get_job(job_id)["status"] == "queued"

    def test_progress_updates(self, repository):
        job_id = repository.create_job(total=10)
        repository.update_job(job_id, status="running", processed=4)
        job = repository.get_job(job_id)
        assert (job["status"], job["processed"]) == ("running", 4)

    def test_completion_sets_finished_at(self, repository):
        job_id = repository.create_job(total=2)
        assert repository.get_job(job_id)["finished_at"] is None
        repository.update_job(job_id, status="done", processed=2)
        assert repository.get_job(job_id)["finished_at"] is not None

    def test_failure_records_the_error(self, repository):
        job_id = repository.create_job(total=2)
        repository.update_job(job_id, status="failed", error="broker unreachable")
        job = repository.get_job(job_id)
        assert job["status"] == "failed" and "broker" in job["error"]

    def test_updating_a_missing_job_is_false(self, repository):
        assert repository.update_job("nope", status="done") is False

    async def test_products_are_linked_to_their_job(self, repository, enriched):
        job_id = repository.create_job(total=2)
        repository.save_products([enriched, enriched], job_id=job_id)
        assert len(repository.products_for_job(job_id)) == 2


class TestPostgres:
    """
    The same behaviour against the real engine used in production.

    SQLite is forgiving in ways Postgres is not — JSON handling and type
    coercion especially — so passing on SQLite alone would not be evidence that
    the Compose stack works.
    """

    def test_connects(self, postgres_settings):
        assert Repository(postgres_settings).ping() is True

    async def test_json_roundtrip_survives_postgres(self, postgres_settings, enriched):
        repo = Repository(postgres_settings)
        repo.create_schema()
        product_id = repo.save_product(enriched)
        try:
            assert repo.get_product(product_id) == enriched
        finally:
            repo.drop_schema()
