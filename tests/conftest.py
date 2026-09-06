"""
Shared fixtures for the platform tests.

Design rule: as much as possible must run with no external service. The
persistence tests therefore run against SQLite by default and additionally
against Postgres when `DATABASE_URL` points at one, and the vector tests use
Qdrant's embedded in-process mode. Only the Redis tests genuinely require a
server, and they skip cleanly when there isn't one.

This keeps `pytest` green in the same environment that runs on Streamlit Cloud,
while still exercising the real code paths under Docker Compose.
"""

from __future__ import annotations

import os

import pytest

from pi_platform.config import Settings


def _postgres_url() -> str | None:
    url = os.getenv("DATABASE_URL", "")
    return url if url.startswith("postgresql") else None


def _redis_reachable(url: str) -> bool:
    try:
        import redis

        client = redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
        return bool(client.ping())
    except Exception:
        return False


@pytest.fixture
def sqlite_settings(tmp_path) -> Settings:
    """Persistence against a throwaway SQLite file. Always available."""
    return Settings(
        database_url=f"sqlite:///{tmp_path/'test.db'}",
        redis_url="",
        qdrant_url=":memory:",
        enable_llm=False,
    )


@pytest.fixture
def stateless_settings() -> Settings:
    """Nothing configured — exactly how the Streamlit deployment runs."""
    return Settings(database_url="", redis_url="", qdrant_url=":memory:", enable_llm=False)


@pytest.fixture
def postgres_settings() -> Settings:
    """Real Postgres, or skip. Exercised under Docker Compose and in CI."""
    url = _postgres_url()
    if not url:
        pytest.skip("no PostgreSQL DATABASE_URL configured")
    return Settings(database_url=url, redis_url="", qdrant_url=":memory:", enable_llm=False)


@pytest.fixture
def redis_settings() -> Settings:
    """Real Redis, or skip."""
    url = os.getenv("REDIS_URL", "")
    if not url or not _redis_reachable(url):
        pytest.skip("no reachable Redis at REDIS_URL")
    return Settings(database_url="", redis_url=url, qdrant_url=":memory:", enable_llm=False)


@pytest.fixture
def repository(sqlite_settings):
    from pi_platform.db import Repository

    repo = Repository(sqlite_settings)
    repo.create_schema()
    yield repo
    repo.drop_schema()


@pytest.fixture
def sample_records():
    return [
        {"name": "ABB M3BP 160MLA 11kW 400V 50Hz IE3 three-phase motor"},
        {"name": "Grundfos CR 5-10 centrifugal pump 3 bar stainless steel"},
        {"name": "Siemens 3RV2011 motor protection circuit breaker 400V 6.3A"},
    ]
