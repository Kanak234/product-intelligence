"""
Cache tests.

The no-op path is tested unconditionally because it is the path the cloud
deployment takes. The Redis path is tested whenever a server is reachable.
"""

from __future__ import annotations

import pytest

# The platform layer is optional. On a deployment that installed only
# requirements.txt these modules are absent, and this file must skip
# rather than fail collection.
pytest.importorskip("redis", reason="cache requires redis-py (requirements-platform.txt)")

from pi_platform.cache import Cache, cache_key


class TestCacheKey:
    def test_deterministic_mode_and_llm_mode_use_different_keys(self):
        assert cache_key("abc", use_llm=False) != cache_key("abc", use_llm=True)

    def test_keys_are_namespaced(self):
        assert cache_key("abc", False).startswith("pi:v1")

    def test_different_products_get_different_keys(self):
        assert cache_key("abc", False) != cache_key("def", False)


class TestDisabledCache:
    """No Redis configured: every operation is a safe no-op."""

    def test_reports_unavailable(self, stateless_settings):
        assert Cache(stateless_settings).available is False

    def test_get_returns_none(self, stateless_settings):
        assert Cache(stateless_settings).get("abc") is None

    def test_set_returns_false(self, stateless_settings):
        assert Cache(stateless_settings).set("abc", {"a": 1}) is False

    def test_ping_is_false(self, stateless_settings):
        assert Cache(stateless_settings).ping() is False

    def test_clear_is_zero(self, stateless_settings):
        assert Cache(stateless_settings).clear() == 0

    def test_unreachable_server_degrades_instead_of_raising(self):
        from pi_platform.config import Settings

        cache = Cache(Settings(redis_url="redis://127.0.0.1:1/0"))
        assert cache.available is False
        assert cache.get("abc") is None


class TestRedisCache:
    @pytest.fixture
    def cache(self, redis_settings):
        cache = Cache(redis_settings)
        cache.clear()
        yield cache
        cache.clear()

    def test_connects(self, cache):
        assert cache.available is True and cache.ping() is True

    def test_miss_then_hit(self, cache):
        assert cache.get("fp1") is None
        cache.set("fp1", {"name": "Pump", "specifications": {}})
        assert cache.get("fp1")["name"] == "Pump"

    def test_roundtrip_preserves_nested_structure(self, cache):
        payload = {"specifications": {"power": {"value": 11000, "origin": "extracted"}}}
        cache.set("fp2", payload)
        assert cache.get("fp2") == payload

    def test_llm_and_deterministic_results_do_not_collide(self, cache):
        cache.set("fp3", {"mode": "deterministic"}, use_llm=False)
        cache.set("fp3", {"mode": "model"}, use_llm=True)
        assert cache.get("fp3", use_llm=False)["mode"] == "deterministic"
        assert cache.get("fp3", use_llm=True)["mode"] == "model"

    def test_delete_removes_the_entry(self, cache):
        cache.set("fp4", {"a": 1})
        cache.delete("fp4")
        assert cache.get("fp4") is None

    def test_expiry_is_applied(self, cache):
        cache.set("fp5", {"a": 1}, ttl=60)
        ttl = cache._client.ttl(cache_key("fp5", False))
        assert 0 < ttl <= 60

    def test_corrupt_entry_is_treated_as_a_miss(self, cache):
        cache._client.set(cache_key("fp6", False), "{not json")
        assert cache.get("fp6") is None

    def test_clear_only_removes_our_keys(self, cache):
        cache._client.set("unrelated:key", "keep me")
        cache.set("fp7", {"a": 1})
        cache.clear()
        assert cache.get("fp7") is None
        assert cache._client.get("unrelated:key") == "keep me"
        cache._client.delete("unrelated:key")

    def test_stats_track_hits_and_misses(self, cache):
        cache.get("missing-one")
        cache.set("fp8", {"a": 1})
        cache.get("fp8")
        stats = cache.stats()
        assert stats["hits"] == 1 and stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
