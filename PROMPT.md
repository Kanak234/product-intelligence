# PROMPT.md

A briefing you can paste at the start of a future session — with an AI
assistant, a collaborator, or yourself in six months — so that work resumes
without re-deriving context.

`ARCHITECTURE.md`, `DOCKER.md` and `LOCAL_LLM.md` are the detail. This is the
summary plus the standing instructions.

---

## Paste from here

I am working on a **product intelligence platform**. It takes messy product
records — a name, free text, loose attributes — and returns structured,
validated, fully explained products: normalised specifications, a category, a
confidence and an origin for every field, graded validation findings, and a
readable explanation of how each conclusion was reached.

### Current state

Two working configurations from one codebase.

```
streamlit_app.py          UI and styling only
app_core.py               all logic; imports no Streamlit, therefore testable
product_intelligence/     the engine, ~3,850 lines, pure Python, in-process
pi_platform/              platform services: db, cache, vector, api, worker
web/index.html            browser frontend, served by the API at /
app/services/llm/         the platform's own local model (Ollama)
Dockerfile                two targets: base (app) and test
docker-compose.yml        full stack, staged by profile
tests/                    358 tests with all services; 179 with none
data/eval/                ground_truth.jsonl and hard_cases.jsonl
```

**Mode A** — Streamlit Community Cloud. One process, `requirements.txt` only,
deterministic, no database and no model. Deployed and working.

**Mode C** — Docker Compose locally. Postgres, Redis, Celery, Qdrant, FastAPI
and a local Ollama model around the same unchanged engine.

The engine's entry point:

```python
pipeline = ProductIntelligencePipeline()   # llm_router=None → deterministic
result = await pipeline.process(RawProduct(name=...), use_llm=False)
```

The engine was never modified to support any of this. `pi_platform/service.py`
is the only seam where the engine meets infrastructure, and every service
degrades to a no-op when unconfigured — which is exactly why the same code runs
as a single stateless process on the cloud.

### The invariant that matters most

Every field carries an `Origin`, and origin caps confidence:

```
INPUT 1.00 > EXTRACTED 0.95 > DERIVED 0.85 > TAXONOMY 0.80 > LLM 0.65 > DEFAULT 0.30
```

A model-generated value can never outrank a value that was present in the
source. **Do not weaken this.** It is what makes adding a language model a
contained change rather than a dangerous one.

### Three deployment modes — always say which you mean

- **Mode A — Cloud.** Streamlit Community Cloud. Deterministic, no model, no
  database. Working today. This is the public demo.
- **Mode B — Local.** `streamlit run streamlit_app.py`. Same as A. Working today.
- **Mode C — Full stack.** Docker Compose on my own machine, running the whole
  simulated production environment: FastAPI, Postgres, Redis, Celery, Qdrant,
  and the local LLM. **Planned, not built.** This is where development will
  happen once it exists.

Mode C is a local simulation of production. It is the same engine with real
infrastructure around it. **Modes A and C must never diverge in engine
behaviour** — the test suite is the contract, and it runs in both.

### Where I am going

**1. Verify the Docker images actually build.** All five compose stages are
written and the application code behind them is tested against real PostgreSQL,
real Redis, embedded Qdrant and a real Celery worker — but **no Docker image
was ever built**, because no daemon was available. Run
`docker compose up --build`, then `docker compose --profile test run --rm test`
and check for 358 passed. This is the first thing to do.

**2. Prove the local model works with real weights.** The model layer is
implemented and tested against a stub server that deliberately lies; the
engine correctly rejects every lie. What is untested is a real model's answer
quality. Pull `qwen2.5:7b-instruct`, set `ENABLE_LLM=1`, and run the accuracy
suite. If hallucination rate exceeds 0.05, fix the prompt or the model — do not
raise the threshold.

**3. Port the frame-size fix back to the main backend** (see below).

### Standing rules

- **Mode A must never break.** Streamlit Cloud must keep working with a plain
  `pip install -r requirements.txt`. If a change makes the app require Docker
  to start, the change is wrong. This is what guarantees a working fallback at
  all times.
- **There are two frontends and they are not interchangeable.**
  `streamlit_app.py` calls the engine in-process and works everywhere;
  `web/index.html` calls the API over HTTP and therefore only works in Mode C.
  The page has no build step and must not load third-party assets — this
  platform is offline-first.
- **Keep `app_core.py` free of Streamlit imports.** Logic there, presentation in
  `streamlit_app.py`. If an `if` decides *what the answer is* rather than *how
  it looks*, it belongs in `app_core.py`.
- **The accuracy gates are not negotiable:** category accuracy ≥ 0.90, spec
  F1 ≥ 0.90, hallucination rate ≤ 0.05. If enabling the model breaches the
  hallucination gate, fix the prompt or the model — do not raise the threshold.
- **The LLM fills gaps only.** It must never overwrite an `INPUT` or
  `EXTRACTED` value. `_merge_llm_specs` enforces this; leave it enforced.
- **Degrade, never crash.** Model server unreachable, `pypdf` missing, database
  down — the app must still start and still produce output. This property is
  already present throughout; preserve it.
- **The suite passes before and after any change.** 358 with all services up,
  179 in a bare environment where the platform tests skip. A skip is fine; a
  failure is not.
- **Never tune a threshold to make a test pass.** When stage 4 flagged
  duplicates on similarity alone, a different product scored *higher* than a
  real duplicate — no threshold could work. The fix was to change the design
  (confirm candidates against specifications), not the number. If a constant
  needs tuning to pass, the design is probably wrong.
- **One thing at a time, with a verifiable exit criterion.** This applies to
  Docker stages and to everything else.

### Known outstanding item

An engine bug was found and fixed **in this prototype only**: IEC motor frame
designations ending in `L` (`100L`, `160L`, `180L`, `200L`, `225L`) were being
read as volumes in litres by the `EXT.VOLUME.BARE` rule in `extractor.py`.
Fixed with negative lookbehinds for `frame` / `frame size`, covered by eight
regression tests.

**The same bug is still present in the main backend archive.** It is a one-line
change. Port it back.

Two more traps worth knowing:

- Do not name a package `platform` — it shadows the standard library module.
  The services package is `pi_platform` for this reason.
- Beware fixtures that do work the real startup path does not. Two bugs got
  through that way: nothing created the database tables, and a cache hit meant
  a record was never stored. Both were found by running the API for real
  against an empty database, not by the tests.
- The engine expects a **nested** LLM response, `{"specifications": {...}}`. A
  flat map parses cleanly and is then silently discarded.

## Paste to here
