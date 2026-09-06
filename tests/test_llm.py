"""
Tests for the platform's own local language model layer.

These run against a stub Ollama server rather than a real model. That is a
deliberate boundary: what needs testing here is the contract — request shape,
response parsing, failure handling, and above all that model output cannot
overwrite a value found in the source text. None of that is a property of the
model, and testing it against a 7B model would make the suite slow,
non-deterministic and dependent on a GPU.

What these tests do NOT prove is the quality of a real model's answers. That is
measured separately by the accuracy thresholds in test_product_intelligence.py,
run with the model enabled.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

# The platform layer is optional. On a deployment that installed only
# requirements.txt these modules are absent, and this file must skip
# rather than fail collection.
pytest.importorskip("httpx", reason="local model client requires httpx (requirements-platform.txt)")

from app.services.llm.base import ChatMessage, LLMUnavailable
from app.services.llm.ollama_provider import (
    OllamaProvider, parse_json_object, strip_json_fences,
)
from app.services.llm.router import LLMRouter
from product_intelligence import ProductIntelligencePipeline, RawProduct

MOTOR = "ABB M3BP 160MLA 11kW 400V 50Hz IE3 three-phase motor"

# A deliberately adversarial model response: two legitimate gap fills, one
# attempt to overwrite a value read from the source, one key outside the
# allowed schema, and one unparseable unit.
ADVERSARIAL = {
    "specifications": {
        "ip_rating": {"value": "IP55", "unit": None, "basis": "standard for this frame"},
        "insulation_class": {"value": "F", "unit": None, "basis": "nameplate"},
        "power": {"value": 999, "unit": "kW", "basis": "contradicts the source"},
        "colour": {"value": "blue", "unit": None, "basis": "not in the schema"},
        "weight": {"value": 95, "unit": "furlongs", "basis": "nonsense unit"},
    },
    "attributes": ["Premium efficiency IE3"],
    "description": "A three-phase induction motor rated at 11 kW for continuous industrial duty.",
}


class _StubHandler(BaseHTTPRequestHandler):
    payload: dict = ADVERSARIAL
    fenced: bool = True
    status: int = 200
    requests: list = []

    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"models": [{"name": "qwen2.5:7b-instruct"}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)

        if type(self).status != 200:
            self.send_response(type(self).status)
            self.end_headers()
            self.wfile.write(b"{}")
            return

        content = json.dumps(type(self).payload)
        if type(self).fenced:
            content = f"```json\n{content}\n```"

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"message": {"content": content}}).encode())


@pytest.fixture(scope="module")
def stub_server():
    handler = type("Handler", (_StubHandler,), {"requests": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}", handler
    server.shutdown()


@pytest.fixture
def provider(stub_server):
    url, handler = stub_server
    handler.payload = ADVERSARIAL
    handler.fenced = True
    handler.status = 200
    handler.requests.clear()
    return OllamaProvider(base_url=url, default_model="qwen2.5:7b-instruct")


# -- response parsing --------------------------------------------------------

class TestResponseParsing:
    @pytest.mark.parametrize("text,expected", [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('Sure, here you go: {"a": 1} — hope that helps', {"a": 1}),
        ("no json at all", {}),
        ("", {}),
        ("[1, 2, 3]", {}),
    ])
    def test_parses_realistic_model_output(self, text, expected):
        assert parse_json_object(text) == expected

    def test_strip_fences_leaves_plain_json_alone(self):
        assert strip_json_fences('{"a": 1}') == '{"a": 1}'


# -- provider ----------------------------------------------------------------

class TestOllamaProvider:
    async def test_health_true_when_server_reachable(self, provider):
        assert await provider.health() is True

    async def test_health_false_when_server_absent(self):
        dead = OllamaProvider(base_url="http://127.0.0.1:1")
        assert await dead.health() is False

    async def test_complete_returns_content(self, provider):
        out = await provider.complete([ChatMessage(role="user", content="hi")])
        assert "ip_rating" in out

    async def test_sends_temperature_zero(self, provider, stub_server):
        _, handler = stub_server
        await provider.complete([ChatMessage(role="user", content="hi")])
        assert handler.requests[-1]["options"]["temperature"] == 0.0

    async def test_sends_messages_in_order(self, provider, stub_server):
        _, handler = stub_server
        await provider.complete([
            ChatMessage(role="system", content="be terse"),
            ChatMessage(role="user", content="hello"),
        ])
        sent = handler.requests[-1]["messages"]
        assert [m["role"] for m in sent] == ["system", "user"]

    async def test_does_not_stream(self, provider, stub_server):
        _, handler = stub_server
        await provider.complete([ChatMessage(role="user", content="hi")])
        assert handler.requests[-1]["stream"] is False

    async def test_raises_on_server_error(self, provider, stub_server):
        _, handler = stub_server
        handler.status = 500
        with pytest.raises(Exception):
            await provider.complete([ChatMessage(role="user", content="hi")])
        handler.status = 200

    async def test_lists_available_models(self, provider):
        assert "qwen2.5:7b-instruct" in await provider.available_models()

    async def test_available_models_empty_when_down(self):
        assert await OllamaProvider(base_url="http://127.0.0.1:1").available_models() == []


# -- router ------------------------------------------------------------------

class TestRouter:
    def test_default_router_has_ollama(self):
        assert "ollama" in LLMRouter().provider_names()

    def test_get_provider_unknown_name_raises(self):
        with pytest.raises(LLMUnavailable):
            LLMRouter().get_provider("does-not-exist")

    def test_rejects_empty_provider_map(self):
        with pytest.raises(ValueError):
            LLMRouter({})

    async def test_auto_select_picks_healthy_provider(self, provider):
        router = LLMRouter({"ollama": provider})
        assert await router.auto_select_provider() == "ollama"

    async def test_auto_select_skips_dead_provider(self, provider):
        router = LLMRouter({
            "dead": OllamaProvider(base_url="http://127.0.0.1:1"),
            "live": provider,
        })
        assert await router.auto_select_provider() == "live"

    async def test_raises_when_nothing_healthy(self):
        router = LLMRouter({"dead": OllamaProvider(base_url="http://127.0.0.1:1")})
        with pytest.raises(LLMUnavailable):
            await router.auto_select_provider()
        assert await router.any_healthy() is False


# -- the part that actually matters ------------------------------------------

class TestTrustBoundary:
    """
    The engine must treat model output as a proposal, never as fact.

    These are the tests that make it safe to point a language model at the
    pipeline at all. If any of them fail, the model can corrupt data that was
    read directly from the source, and the whole provenance model is a lie.
    """

    @pytest.fixture
    async def result(self, provider):
        pipeline = ProductIntelligencePipeline(
            llm_router=LLMRouter({"ollama": provider}), model="qwen2.5:7b-instruct"
        )
        return await pipeline.process(RawProduct(name=MOTOR), use_llm=True)

    async def test_model_fills_genuine_gaps(self, result):
        assert result["specifications"]["ip_rating"]["value"] == "IP55"
        assert result["specifications"]["insulation_class"]["value"] == "F"

    async def test_filled_gaps_are_marked_as_model_output(self, result):
        assert result["specifications"]["ip_rating"]["origin"] == "llm"

    async def test_model_confidence_stays_under_its_ceiling(self, result):
        assert result["specifications"]["ip_rating"]["confidence"] <= 0.65

    async def test_model_cannot_overwrite_an_extracted_value(self, result):
        power = result["specifications"]["power"]
        assert power["origin"] == "extracted"
        assert power["value"] == 11000  # not the 999 kW the model proposed

    async def test_keys_outside_the_schema_are_rejected(self, result):
        assert "colour" not in result["specifications"]

    async def test_values_with_unusable_units_are_rejected(self, result):
        assert "weight" not in result["specifications"]

    async def test_extracted_values_outrank_model_values(self, result):
        specs = result["specifications"]
        assert specs["power"]["confidence"] > specs["ip_rating"]["confidence"]


class TestDegradation:
    """A missing or broken model must degrade the output, never break it."""

    async def test_pipeline_runs_with_no_model_server(self):
        router = LLMRouter({"ollama": OllamaProvider(base_url="http://127.0.0.1:1")})
        pipeline = ProductIntelligencePipeline(llm_router=router)
        result = await pipeline.process(RawProduct(name=MOTOR), use_llm=True)
        assert result["specifications"]["power"]["value"] == 11000
        assert all(
            field["origin"] != "llm" for field in result["specifications"].values()
        )

    async def test_unparseable_response_is_survivable(self, provider, stub_server):
        _, handler = stub_server
        handler.payload = {"not": "the expected shape"}
        pipeline = ProductIntelligencePipeline(llm_router=LLMRouter({"ollama": provider}))
        result = await pipeline.process(RawProduct(name=MOTOR), use_llm=True)
        assert result["specifications"]["power"]["value"] == 11000

    async def test_use_llm_false_never_calls_the_model(self, provider, stub_server):
        _, handler = stub_server
        handler.requests.clear()
        pipeline = ProductIntelligencePipeline(llm_router=LLMRouter({"ollama": provider}))
        await pipeline.process(RawProduct(name=MOTOR), use_llm=False)
        assert handler.requests == []

    async def test_deterministic_result_is_unchanged_by_model_availability(self, provider):
        plain = await ProductIntelligencePipeline().process(
            RawProduct(name=MOTOR), use_llm=False
        )
        with_router = await ProductIntelligencePipeline(
            llm_router=LLMRouter({"ollama": provider})
        ).process(RawProduct(name=MOTOR), use_llm=False)
        assert plain["specifications"].keys() == with_router["specifications"].keys()
