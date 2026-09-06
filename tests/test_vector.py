"""
Vector search tests.

Qdrant runs embedded in-process, so these need no server and no GPU. The
embedding under test is the deterministic lexical fallback; a real embedding
model would score higher on paraphrase, but every property asserted here —
determinism, normalisation, ordering, thresholds — must hold for either.
"""

from __future__ import annotations

import pytest

# The platform layer is optional. On a deployment that installed only
# requirements.txt these modules are absent, and this file must skip
# rather than fail collection.
pytest.importorskip("qdrant_client", reason="vector search requires qdrant-client (requirements-platform.txt)")

from pi_platform.vector import (
    VectorStore, cosine, lexical_embedding, product_text, VECTOR_SIZE,
)
from product_intelligence import ProductIntelligencePipeline, RawProduct

MOTOR = "ABB M3BP 160MLA 11kW 400V 50Hz IE3 three-phase motor"
MOTOR_VARIANT = "ABB M3BP160MLA 11 kW 400 V IE3 3-phase motor"
PUMP = "Grundfos CR 5-10 centrifugal pump 3 bar stainless steel"


async def _enrich(name: str):
    return await ProductIntelligencePipeline().process(RawProduct(name=name), use_llm=False)


@pytest.fixture(scope="module")
async def products():
    return {
        "motor": await _enrich(MOTOR),
        "variant": await _enrich(MOTOR_VARIANT),
        "pump": await _enrich(PUMP),
    }


class TestEmbedding:
    def test_has_the_declared_dimension(self):
        assert len(lexical_embedding("a motor")) == VECTOR_SIZE

    def test_is_deterministic(self):
        assert lexical_embedding("a motor") == lexical_embedding("a motor")

    def test_is_normalised(self):
        vector = lexical_embedding("11kW three phase induction motor")
        assert abs(sum(v * v for v in vector) ** 0.5 - 1.0) < 1e-9

    def test_empty_text_does_not_crash(self):
        assert len(lexical_embedding("")) == VECTOR_SIZE

    def test_identical_text_scores_one(self):
        vector = lexical_embedding(MOTOR)
        assert cosine(vector, vector) == pytest.approx(1.0)

    def test_near_identical_text_scores_higher_than_unrelated(self):
        base = lexical_embedding(MOTOR)
        near = cosine(base, lexical_embedding(MOTOR_VARIANT))
        far = cosine(base, lexical_embedding(PUMP))
        assert near > far

    def test_case_and_punctuation_are_ignored(self):
        assert lexical_embedding("11kW Motor!") == lexical_embedding("11kw motor")


class TestProductText:
    async def test_includes_the_name(self, products):
        assert "M3BP" in product_text(products["motor"])

    async def test_includes_specification_values(self, products):
        text = product_text(products["motor"])
        assert "power" in text and "11000" in text

    def test_handles_a_bare_result(self):
        assert product_text({"name": "Widget"}) == "Widget"


class TestVectorStore:
    @pytest.fixture
    def store(self, stateless_settings):
        store = VectorStore(stateless_settings)
        store.reset()
        return store

    def test_embedded_mode_is_available_without_a_server(self, store):
        assert store.available is True

    async def test_upsert_then_count(self, store, products):
        store.upsert("p-motor", products["motor"])
        assert store.count() == 1

    async def test_upsert_is_idempotent_for_the_same_id(self, store, products):
        store.upsert("p-motor", products["motor"])
        store.upsert("p-motor", products["motor"])
        assert store.count() == 1

    async def test_search_ranks_the_closest_first(self, store, products):
        store.upsert("p-motor", products["motor"])
        store.upsert("p-pump", products["pump"])
        hits = store.search(products["variant"], limit=2)
        assert hits[0]["product_id"] == "p-motor"

    async def test_search_returns_payload_fields(self, store, products):
        store.upsert("p-motor", products["motor"])
        hit = store.search(products["motor"], limit=1)[0]
        assert hit["name"] == MOTOR and hit["category"]

    async def test_finds_a_near_duplicate(self, store, products):
        store.upsert("p-motor", products["motor"])
        matches = store.find_duplicates(products["variant"])
        assert [m["product_id"] for m in matches] == ["p-motor"]

    async def test_does_not_flag_an_unrelated_product(self, store, products):
        store.upsert("p-motor", products["motor"])
        assert store.find_duplicates(products["pump"]) == []

    async def test_excludes_the_product_itself(self, store, products):
        store.upsert("p-motor", products["motor"])
        assert store.find_duplicates(products["motor"], exclude_id="p-motor") == []

    async def test_threshold_can_be_overridden(self, store, products):
        store.upsert("p-motor", products["motor"])
        assert store.find_duplicates(products["variant"], threshold=0.99) == []

    async def test_search_payload_carries_comparable_specs(self, store, products):
        store.upsert("p-motor", products["motor"])
        hit = store.search(products["motor"], limit=1)[0]
        assert "power" in hit["specs"]

    async def test_reset_empties_the_collection(self, store, products):
        store.upsert("p-motor", products["motor"])
        store.reset()
        assert store.count() == 0

    async def test_batch_upsert_counts_successes(self, store, products):
        written = store.upsert_many([("a", products["motor"]), ("b", products["pump"])])
        assert written == 2

    async def test_search_on_an_empty_collection_is_empty(self, store, products):
        assert store.search(products["motor"]) == []


class TestUnavailableStore:
    def test_bad_url_degrades_to_unavailable(self):
        from pi_platform.config import Settings

        store = VectorStore(Settings(qdrant_url="http://127.0.0.1:1"))
        assert store.available is False
        assert store.search({"name": "x"}) == []
        assert store.upsert("id", {"name": "x"}) is False
        assert store.count() == 0
