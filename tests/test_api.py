"""
API tests.

These run against the real FastAPI application with a real service behind it —
no route mocking. The service is injected so that persistence points at a
throwaway SQLite file rather than requiring Postgres, but every request goes
through the same code the Compose stack runs.
"""

from __future__ import annotations

import pytest

# The platform layer is optional. On a deployment that installed only
# requirements.txt these modules are absent, and this file must skip
# rather than fail collection.
pytest.importorskip("fastapi", reason="API requires FastAPI (requirements-platform.txt)")

from pi_platform.api import app, set_service
from pi_platform.service import ProductService

MOTOR = "ABB M3BP 160MLA 11kW 400V 50Hz IE3 three-phase motor"
MOTOR_VARIANT = "ABB M3BP160MLA 11 kW 400 V IE3 3-phase motor"


@pytest.fixture
def client(sqlite_settings):
    from fastapi.testclient import TestClient

    service = ProductService(sqlite_settings)
    service.repository.create_schema()
    service.vectors.reset()
    set_service(service)

    with TestClient(app) as test_client:
        yield test_client

    service.repository.drop_schema()
    set_service(None)


@pytest.fixture
def stateless_client(stateless_settings):
    """No database — the configuration the cloud deployment would use."""
    from fastapi.testclient import TestClient

    set_service(ProductService(stateless_settings))
    with TestClient(app) as test_client:
        yield test_client
    set_service(None)


class TestHealth:
    def test_health_is_reachable(self, client):
        assert client.get("/health").status_code == 200

    def test_health_reports_each_service(self, client):
        body = client.get("/health").json()
        assert body["engine"] == "ok"
        assert body["database"] is True
        assert set(body["config"]) >= {"persistence", "cache", "llm"}

    def test_health_leaks_no_credentials(self, client):
        assert "password" not in client.get("/health").text.lower()

    def test_config_endpoint(self, client):
        assert client.get("/config").json()["llm"] == "off"

    def test_health_works_with_nothing_configured(self, stateless_client):
        body = stateless_client.get("/health").json()
        assert body["engine"] == "ok" and body["database"] is False


class TestEnrich:
    def test_returns_structured_specifications(self, client):
        body = client.post("/enrich", json={"name": MOTOR}).json()
        assert body["specifications"]["power"]["value"] == 11000

    def test_every_field_carries_provenance(self, client):
        body = client.post("/enrich", json={"name": MOTOR}).json()
        assert all(
            "origin" in field and "confidence" in field
            for field in body["specifications"].values()
        )

    def test_result_is_persisted_by_default(self, client):
        body = client.post("/enrich", json={"name": MOTOR}).json()
        product_id = body["platform"]["persisted_id"]
        assert client.get(f"/products/{product_id}").status_code == 200

    def test_persist_false_is_honoured(self, client):
        body = client.post("/enrich", json={"name": MOTOR, "persist": False}).json()
        assert body["platform"]["persisted_id"] is None

    def test_description_and_attributes_are_accepted(self, client):
        response = client.post("/enrich", json={
            "name": "Centrifugal pump",
            "description": "Stainless steel pump rated to 3 bar",
            "attributes": {"supplier": "Grundfos"},
        })
        assert response.status_code == 200

    def test_blank_name_is_rejected(self, client):
        assert client.post("/enrich", json={"name": "   "}).status_code == 422

    def test_missing_name_is_rejected(self, client):
        assert client.post("/enrich", json={}).status_code == 422

    def test_empty_name_is_rejected(self, client):
        assert client.post("/enrich", json={"name": ""}).status_code == 422

    def test_works_without_a_database(self, stateless_client):
        body = stateless_client.post("/enrich", json={"name": MOTOR}).json()
        assert body["specifications"]["power"]["value"] == 11000
        assert body["platform"]["persisted_id"] is None


class TestCatalogue:
    def test_processes_every_record(self, client, sample_records):
        body = client.post("/catalogue", json={"products": sample_records}).json()
        assert body["summary"]["total"] == 3

    def test_includes_cross_record_consistency(self, client, sample_records):
        body = client.post("/catalogue", json={"products": sample_records}).json()
        assert "catalog_consistency" in body

    def test_rejects_an_empty_list(self, client):
        assert client.post("/catalogue", json={"products": []}).status_code == 422

    def test_rejects_records_with_no_usable_name(self, client):
        response = client.post("/catalogue", json={"products": [{"name": "  "}]})
        assert response.status_code == 422

    def test_streaming_mode_returns_counts(self, client, sample_records):
        body = client.post(
            "/catalogue", json={"products": sample_records, "stream": True}
        ).json()
        assert body["processed"] == 3 and body["job_id"]

    def test_streaming_job_can_be_polled(self, client, sample_records):
        job_id = client.post(
            "/catalogue", json={"products": sample_records, "stream": True}
        ).json()["job_id"]
        assert client.get(f"/jobs/{job_id}").json()["status"] == "done"


class TestLookups:
    def test_unknown_product_is_404(self, client):
        assert client.get("/products/00000000-0000-0000-0000-000000000000").status_code == 404

    def test_unknown_job_is_404(self, client):
        assert client.get("/jobs/not-a-job").status_code == 404

    def test_recent_products_lists_what_was_stored(self, client):
        client.post("/enrich", json={"name": MOTOR})
        assert len(client.get("/products").json()["products"]) == 1

    def test_limit_is_validated(self, client):
        assert client.get("/products?limit=0").status_code == 422
        assert client.get("/products?limit=9999").status_code == 422


class TestSimilarity:
    def test_finds_a_stored_near_match(self, client):
        client.post("/enrich", json={"name": MOTOR})
        matches = client.post("/similar", json={"name": MOTOR_VARIANT}).json()["matches"]
        assert matches and matches[0]["name"] == MOTOR

    def test_does_not_persist_the_query_product(self, client):
        client.post("/similar", json={"name": MOTOR_VARIANT})
        assert client.get("/products").json()["products"] == []

    def test_empty_index_returns_no_matches(self, client):
        assert client.post("/similar", json={"name": MOTOR}).json()["matches"] == []


class TestFrontend:
    """
    The browser frontend is served by the API itself, same origin, so that no
    CORS configuration is needed for the page to call its own backend.
    """

    def test_root_serves_the_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.text.lstrip().startswith("<!DOCTYPE html>")

    def test_page_is_html(self, client):
        assert "text/html" in client.get("/").headers["content-type"]

    def test_root_is_not_in_the_api_schema(self, client):
        assert "/" not in client.get("/openapi.json").json()["paths"]

    def test_page_fetches_no_third_party_assets(self, client):
        """
        This platform is offline-first. A page that pulls fonts or scripts from
        a CDN contradicts that and breaks in an air-gapped deployment, which is
        exactly where it is meant to run.
        """
        page = client.get("/").text
        assert "//fonts.googleapis.com" not in page
        assert "//cdn." not in page
        assert "http://" not in page.split("<script")[0].replace("http://localhost", "")

    def test_page_declares_the_same_trust_ceilings_as_the_engine(self, client):
        """
        The gauge draws each origin's ceiling client-side, so the numbers are
        duplicated in the page. If the engine's ceilings change and the page is
        not updated, the gauge quietly lies about the rule it exists to show.
        """
        import re

        from product_intelligence.schema import ORIGIN_TRUST

        page = client.get("/").text
        for origin, ceiling in ORIGIN_TRUST.items():
            key = getattr(origin, "value", str(origin))
            # The table in the page is column-aligned, so allow any run of
            # whitespace between the origin name and its ceiling.
            pattern = rf'"{re.escape(key)}",\s+{ceiling:.2f}'
            assert re.search(pattern, page), f"page is missing {key} at {ceiling}"


class TestDocumentation:
    def test_openapi_schema_is_generated(self, client):
        schema = client.get("/openapi.json").json()
        assert "/enrich" in schema["paths"]
        assert "/catalogue" in schema["paths"]
