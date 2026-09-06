# Local LLM — the platform's own model

**Requirement:** the platform should carry its own language model inside it, and
run its analysis on that model — not on any external API. No OpenAI key, no
Anthropic key, no data leaving the machine.

The engine was built anticipating this, and the layer is now built. This
document explains how it works and what remains unproven.

**Status: implemented.** The layer ships in `app/services/llm/` and is covered
by 34 tests. It is *off by default* (`ENABLE_LLM=0`), because the cloud
deployment has no model server and must stay deterministic.

What is verified and what is not:

- **Verified** — the request shape, response parsing, health checking, provider
  failover, and above all the trust boundary: the tests drive the real pipeline
  against a stub model server that deliberately lies, and confirm the engine
  rejects every lie.
- **Not verified** — the answer quality of a real 7B model. No model was pulled
  in the environment where this was written. Run the accuracy suite with
  `ENABLE_LLM=1` against a real Ollama container before trusting the numbers.

---

## 1. Why the engine is already ready for this

`ProductEnricher.__init__` takes an optional `llm_router`:

```python
def __init__(
    self,
    llm_router: Optional[Any] = None,
    model: Optional[str] = None,
    provider_name: Optional[str] = None,
    llm_timeout: float = 45.0,
) -> None:
```

With `llm_router=None`, the LLM stage is skipped and everything else runs
normally. Supply a router and stage 4 of `enrich()` activates. No other code
changes.

Better still, the engine already treats model output as untrusted:

- `Origin.LLM` has a trust ceiling of **0.65**, below `EXTRACTED` (0.95)
- `_merge_llm_specs` rejects any key not in `LLM_ALLOWED_SPECS`
- It rejects any key already known from `INPUT` or `EXTRACTED`
- It rejects values whose units cannot be canonicalised
- Every rejection is recorded with a reason
- `_call_llm` returns `None` on timeout or any exception, and enrichment
  continues

So a hallucinating model degrades the output; it cannot corrupt it. The
hallucination-rate threshold of ≤ 0.05 in the test suite is what holds this
honest, and it must keep passing once the model is wired in.

---

## 2. Recommended stack

**Ollama** as the model server. It is a single container, exposes a simple
HTTP API, manages model weights itself, and runs on CPU when there is no GPU.

**Model:** start with `qwen2.5:7b-instruct` — strong at structured extraction
and reliable at emitting JSON. On a constrained machine, `qwen2.5:3b-instruct`
or `phi3:mini` work; expect more rejected proposals, which the merge logic will
absorb.

```yaml
# add to docker-compose.yml — see DOCKER.md, stage 5
  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes: [ollama_models:/root/.ollama]
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 10s
      timeout: 5s
      retries: 20        # first model pull is slow; be generous

volumes:
  ollama_models:
```

Keep the weights in a named volume. Re-pulling several gigabytes on every
`docker compose down` is a fast route back to giving up on Docker.

---

## 3. What was implemented

The three modules below exist in `app/services/llm/`. This section documents
what they do and why; it is no longer a to-do list.

`_call_llm` in the engine does this:

```python
try:
    from app.services.llm.base import ChatMessage
except Exception:
    return None
```

Before this layer existed that import failed, so the stage no-opped. Providing
the module — plus a router and a provider — is what switches it on. **No change
to the engine was required**, which was the point of building it this way.

### 3.1 `app/services/llm/base.py`

```python
# app/services/llm/base.py
from dataclasses import dataclass

@dataclass
class ChatMessage:
    role: str      # "system" | "user" | "assistant"
    content: str
```

### 3.2 `app/services/llm/ollama_provider.py`

One required method: `async complete(messages, model) -> str`.

```python
# app/services/llm/ollama_provider.py
import httpx
from typing import List, Optional
from .base import ChatMessage


class OllamaProvider:
    def __init__(self, base_url: str = "http://ollama:11434",
                 default_model: str = "qwen2.5:7b-instruct") -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    async def complete(self, messages: List[ChatMessage],
                       model: Optional[str] = None) -> str:
        payload = {
            "model": model or self.default_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": 0.0,   # extraction, not creativity
                "num_predict": 1024,
            },
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False
```

`temperature=0.0` matters. This is a structured-extraction task; sampling
variance produces schema violations, which produce rejections, which lower
recall for no benefit.

### 3.3 `app/services/llm/router.py`

`_call_llm` calls `get_provider(name)` and optionally
`await auto_select_provider()`.

```python
# app/services/llm/router.py
class LLMRouter:
    def __init__(self, providers: dict) -> None:
        self.providers = providers

    def get_provider(self, name=None):
        if name is None:
            name = next(iter(self.providers))
        return self.providers[name]

    async def auto_select_provider(self):
        for name, provider in self.providers.items():
            if await provider.health():
                return name
        raise RuntimeError("no healthy LLM provider")
```

`auto_select_provider` is what lets the platform run happily with the model
server down — the failure is caught in `_call_llm` and enrichment continues
deterministically.

### 3.4 The response schema the engine expects

This is easy to get wrong and produces a silent failure when you do: the model
returns valid JSON, the engine parses it, and nothing merges. The engine asks
for a **nested** object, not a flat map of specifications:

```json
{
  "specifications": {
    "ip_rating": {"value": "IP55", "unit": null, "basis": "one short justification"}
  },
  "attributes": ["short selling point"],
  "description": "2-3 sentence commerce-ready description",
  "category_suggestion": "only if you disagree with the assigned category"
}
```

A flat `{"ip_rating": "IP55"}` parses cleanly and is then discarded, because
`_stage_llm` reads `payload.get("specifications", {})`. If model output is
never appearing in results, check this first.

Markdown fences are tolerated — both the engine's parser and this layer's
`parse_json_object` strip them — as is a sentence wrapped around the object.

### 3.5 Wiring it up

```python
router = LLMRouter({"ollama": OllamaProvider()})
pipeline = ProductIntelligencePipeline(llm_router=router,
                                       model="qwen2.5:7b-instruct")
result = await pipeline.process(raw, use_llm=True)
```

That is the entire integration. In practice you would not construct it by
hand: `pi_platform.service.ProductService` does this from environment settings,
so `ENABLE_LLM=1` is the only switch needed.

---

## 4. Behaviour that must be preserved

**The model fills gaps only.** It must never override an extracted or input
value. `_merge_llm_specs` already enforces this; do not relax it.

**Graceful degradation is mandatory.** With the model server unreachable, the
platform must still work, still pass its tests, and still produce output —
merely with fewer filled fields. `_call_llm` returning `None` on failure is the
mechanism. Never let a model failure raise into the pipeline.

**Provenance stays visible.** Every model-supplied field renders as
`Origin.LLM` in the UI, at its lower trust level. The user must always be able
to see which values the model produced versus which came from the source text.

**The accuracy gates keep applying.** Category accuracy ≥ 0.90, spec F1 ≥ 0.90,
hallucination rate ≤ 0.05. Run the suite with the model enabled and again with
it disabled. Both must pass. If enabling the model pushes hallucination above
the threshold, the constraint is the model's — tighten the prompt, tighten
`LLM_ALLOWED_SPECS`, or use a better model. Do not raise the threshold.

**Determinism is a feature of Mode A.** The Streamlit Cloud deployment stays
model-free and therefore reproducible. Do not make the public demo depend on a
model server that will not exist there.

---

## 5. Honest constraints

Streamlit Community Cloud gives roughly 1 GB of RAM and no GPU. **A 7B model
cannot run there.** Even a 3B model is not realistic.

So the local LLM belongs to Mode C — the Docker Compose simulation on your own
machine — and not to the public demo. That is not a limitation of the design;
it is what "local model" means. The public link demonstrates the deterministic
engine; the local stack demonstrates the full platform.

If you eventually want a hosted deployment *with* the model, that needs a real
VM with adequate RAM — at which point the Compose file from `DOCKER.md` is
already the deployment artefact, which is a good part of why it is worth
writing properly.

Expect CPU-only inference at roughly 5–20 tokens/second for a 7B model.
Per-product enrichment will take seconds, not milliseconds. This is the correct
argument for the Celery worker in stage 3: batch enrichment belongs in a
background job, not in a request.
