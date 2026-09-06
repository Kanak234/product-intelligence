"""
Worker tests.

`process_catalogue` and `process_single` are plain functions with the Celery
task decorators wrapped around them, so these tests exercise exactly the code a
worker runs without needing a broker or a second process. The broker round trip
itself is Celery's responsibility, not this application's.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# The platform layer is optional. On a deployment that installed only
# requirements.txt these modules are absent, and this file must skip
# rather than fail collection.
pytest.importorskip("celery", reason="worker requires Celery (requirements-platform.txt)")

from pi_platform.service import ProductService
from pi_platform.worker import celery_app, process_catalogue, process_single

MOTOR = "ABB M3BP 160MLA 11kW 400V 50Hz IE3 three-phase motor"


@pytest.fixture
def env_service(sqlite_settings, monkeypatch):
    """
    Configure through the environment, the way a real worker is configured.

    This is the more faithful fixture: the worker builds its own service from
    settings, so patching the environment tests the wiring as well as the work.
    """
    monkeypatch.setenv("DATABASE_URL", sqlite_settings.database_url)
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.delenv("REDIS_URL", raising=False)

    service = ProductService(sqlite_settings)
    service.repository.create_schema()
    yield service
    service.repository.drop_schema()


class TestCeleryConfiguration:
    def test_app_is_configured(self):
        assert celery_app is not None

    def test_tasks_are_registered(self):
        assert "product_intel.process_catalogue" in celery_app.tasks
        assert "product_intel.process_single" in celery_app.tasks

    def test_uses_json_serialisation(self):
        assert celery_app.conf.task_serializer == "json"

    def test_time_limits_allow_a_long_catalogue(self):
        assert celery_app.conf.task_time_limit >= 3600

    def test_soft_limit_is_below_the_hard_limit(self):
        assert celery_app.conf.task_soft_time_limit < celery_app.conf.task_time_limit

    def test_broker_is_configured_from_the_environment(self):
        assert celery_app.conf.broker_url


class TestCatalogueProcessing:
    def test_processes_every_record(self, env_service, sample_records):
        outcome = process_catalogue(sample_records)
        assert outcome["processed"] == 3

    def test_creates_a_job_when_none_is_given(self, env_service, sample_records):
        outcome = process_catalogue(sample_records)
        assert env_service.repository.get_job(outcome["job_id"]) is not None

    def test_marks_the_job_done(self, env_service, sample_records):
        outcome = process_catalogue(sample_records)
        assert env_service.repository.get_job(outcome["job_id"])["status"] == "done"

    def test_persists_the_results(self, env_service, sample_records):
        process_catalogue(sample_records)
        assert env_service.repository.count_products() == 3

    def test_reuses_an_existing_job_id(self, env_service, sample_records):
        job_id = env_service.repository.create_job(total=3)
        outcome = process_catalogue(sample_records, job_id=job_id)
        assert outcome["job_id"] == job_id

    def test_records_progress_on_the_job(self, env_service, sample_records):
        outcome = process_catalogue(sample_records)
        assert env_service.repository.get_job(outcome["job_id"])["processed"] == 3

    def test_empty_catalogue_is_not_an_error(self, env_service):
        assert process_catalogue([])["processed"] == 0

    def test_records_without_names_are_skipped(self, env_service):
        outcome = process_catalogue([{"name": MOTOR}, {"name": "  "}])
        assert outcome["processed"] == 1


class TestSingleProcessing:
    def test_returns_an_enriched_product(self, env_service):
        result = process_single(MOTOR)
        assert result["specifications"]["power"]["value"] == 11000

    def test_persists_the_result(self, env_service):
        process_single(MOTOR)
        assert env_service.repository.count_products() == 1


class TestFailureHandling:
    def test_a_failing_job_is_marked_failed(self, env_service, sample_records):
        job_id = env_service.repository.create_job(total=3)

        with patch(
            "pi_platform.service.ProductService.enrich_catalogue_streaming",
            side_effect=RuntimeError("model server exploded"),
        ):
            with pytest.raises(RuntimeError):
                process_catalogue(sample_records, job_id=job_id)

        job = env_service.repository.get_job(job_id)
        assert job["status"] == "failed"
        assert "exploded" in job["error"]
