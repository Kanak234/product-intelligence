# Architecture — as it stands today

This document describes exactly what is in this folder right now, why it is
shaped this way, and which parts are permanent versus which were simplified to
get a deployment out of the door.

Read this first. `DOCKER.md` and `LOCAL_LLM.md` describe what comes next and
both assume you have read this one.

---

## 1. What this application is

A product intelligence engine. You give it a messy product record — a name, a
free-text description, a handful of loose attributes — and it returns a
structured, validated, fully explained product:

- **Specifications** extracted and normalised to canonical units
- **A category**, assigned by a rule-based taxonomy classifier
- **A confidence and an origin for every single field**
- **Validation findings**, graded by severity
- **A human-readable explanation** of how each conclusion was reached

The design principle that matters most: **nothing in the output is
unattributed**. Every value carries an `Origin` recording where it came from,
and origin caps confidence. This is what makes the engine safe to point at a
language model later — a model can only ever fill gaps, never overwrite a fact
that was present in the source.

---

## 2. Current deployment shape

```
Browser
   │
   ▼
Streamlit process  ← single Python process, no network calls out
   │
   ├── streamlit_app.py     presentation only
   ├── app_core.py          pure logic, no Streamlit imports
   └── product_intelligence/  the engine, in-process
```

There is no database, no queue, no vector store, no API server, no container.
The engine is pure Python and runs in the same process as the UI.

The platform services (`pi_platform/`, `app/services/llm/`) exist in the same
repository but are **not imported by the Streamlit app at all** and are not in
`requirements.txt`. Mode A remains a single process with nothing to configure.

This is a deliberate reduction, not the intended end state. The engine itself
is unchanged from the full backend — the same `ProductIntelligencePipeline`
that a FastAPI route or a Celery worker would call is called directly here.
That is precisely why the reduction was safe: the engine never depended on the
infrastructure around it.

### Deployment modes — the important clarification

There are three distinct ways this project can run, and they are often
confused with each other. Be explicit about which one you mean.

| Mode | What runs | Where | Status |
|---|---|---|---|
| **A. Cloud (current)** | Streamlit process only | Streamlit Community Cloud | **Working today** |
| **B. Local** | Streamlit process only | Your machine, `streamlit run` | **Working today** |
| **C. Full stack** | Streamlit + FastAPI + Postgres + Redis + Celery + Qdrant + local LLM | Docker Compose, your machine | **Built — see `DOCKER.md`** |

Mode C is a *local simulation* of a production deployment. It is not a
different product; it is the same engine with real infrastructure around it so
that persistence, background jobs, semantic search and the language model can
be exercised properly. Mode C is what you develop against. Mode A is what you
show people.

**Mode A and Mode C must never diverge in engine behaviour.** The test suite is
the contract that enforces this: the same tests run in both.

### Two frontends, on purpose

| Frontend | Runs in | Talks to | Available in |
|---|---|---|---|
| `streamlit_app.py` | A Python process | The engine, in-process | Modes A, B, C |
| `web/index.html` | The browser | The API over HTTP | Mode C only |

They are not duplicates competing for the same job. Streamlit can call the
engine directly because it *is* Python; a browser cannot, so the HTML frontend
needs the API behind it. That is why the HTML page does not and cannot replace
Streamlit on the cloud deployment — there is no API there to call.

The page is served by FastAPI itself at `/`, same origin as the API, so no CORS
configuration is needed. It uses no build step, no framework and no third-party
assets: an offline-first platform that fetched fonts from a CDN on every page
load would contradict its own premise and break in an air-gapped deployment.

One duplication is deliberate and guarded: the page draws each origin's trust
ceiling client-side, so those six numbers exist in both `schema.py` and
`index.html`. A test asserts they match, and fails if either drifts.

### Deploying Mode A

Push this folder to GitHub, then at `share.streamlit.io` create a new app
pointing at `streamlit_app.py` on the `main` branch. `requirements.txt` and
`.streamlit/config.toml` are already in place; nothing else needs configuring.
There are no secrets, no environment variables and no external services, which
is the entire reason this deploys cleanly.

---

## 3. File-by-file

```
.
├── streamlit_app.py            500 lines  UI, layout, CSS. No business logic.
├── app_core.py                 285 lines  All logic the UI needs. No Streamlit import.
├── product_intelligence/                  The engine (~3,850 lines)
│   ├── __init__.py             201  ProductIntelligencePipeline — the single entry point
│   ├── schema.py               231  Origin, ORIGIN_TRUST, FieldValue, Evidence, RawProduct, EnrichedProduct
│   ├── units.py                252  Unit registry, canonicalisation, tolerant comparison
│   ├── extractor.py            484  Regex spec extraction (SPEC_PATTERNS)
│   ├── taxonomy.py             421  Rule-based category tree and classifier
│   ├── enricher.py             618  Orchestrates extract → classify → derive → LLM → describe
│   ├── validator.py            556  Per-product rules + CatalogConsistencyChecker
│   ├── explainer.py            323  Turns evidence chains into prose
│   ├── ingestion.py            565  CSV, TSV, JSON, JSONL, HTML, TXT, MD, PDF, XLSX readers
│   ├── evaluation.py           497  Scoring against ground truth
│   └── benchmark.py            301  Throughput and memory measurement
├── tests/
│   ├── test_product_intelligence.py  765  Engine tests, incl. enforced accuracy thresholds
│   ├── test_app_core.py              325  Logic layer, incl. frame-size regression
│   └── test_ui.py                    112  Headless Streamlit AppTest driving the real app
├── data/eval/
│   ├── ground_truth.jsonl                 Accuracy dataset
│   └── hard_cases.jsonl                   Adversarial dataset
├── pi_platform/                           Platform services (Mode C only)
│   ├── config.py                          Environment-driven settings
│   ├── db.py                              Postgres persistence (SQLAlchemy)
│   ├── cache.py                           Redis result cache
│   ├── vector.py                          Qdrant near-duplicate detection
│   ├── service.py                         The seam: engine + infrastructure
│   ├── api.py                             FastAPI routes + serves web/
│   └── worker.py                          Celery background jobs
├── app/services/llm/                      The platform's own local model
│   ├── base.py                            ChatMessage — the engine imports this
│   ├── ollama_provider.py                 Local Ollama client
│   └── router.py                          Health-checked provider selection
├── web/index.html                         Browser frontend, served by the API
├── Dockerfile                             Two targets: base (app) and test
├── docker-compose.yml                     Full stack, staged by profile
├── requirements.txt                       streamlit, pypdf, openpyxl
├── requirements-dev.txt                   + pytest, pytest-asyncio
├── requirements-platform.txt              Mode C only: fastapi, sqlalchemy, ...
├── pytest.ini                             testpaths, pythonpath, asyncio_mode
├── .streamlit/config.toml                 Theme and headless server settings
├── ARCHITECTURE.md                        This file
├── DOCKER.md                              Roadmap back to the full stack
├── LOCAL_LLM.md                           Spec for the built-in language model
└── PROMPT.md                              Paste-able brief for a future session
```

### The `app_core.py` / `streamlit_app.py` split

This split is load-bearing and should be preserved.

`app_core.py` imports no Streamlit. Every function in it is a plain function
over plain data, which means the whole logic layer is testable without a
browser, a server or a session. `streamlit_app.py` imports `app_core` and does
nothing but arrange the results on screen.

If you find yourself writing an `if` statement inside `streamlit_app.py` that
decides *what the answer is* rather than *how it looks*, it belongs in
`app_core.py`.

Key functions in `app_core.py`:

| Function | Purpose |
|---|---|
| `parse_attributes(text)` | Parses `key: value` / `key = value` lines. Skips blanks, `#` comments and separator-less lines. Last value wins. |
| `build_raw(name, description, attrs)` | Constructs a `RawProduct`, trimming whitespace. |
| `analyse_product(raw)` | Runs the pipeline for one product. |
| `analyse_catalogue(raws)` | Runs the batch pipeline plus cross-record consistency. |
| `provenance_breakdown(result)` | Counts fields by origin. |
| `traceability(result)` | Fraction of fields originating from input/extracted/derived. Empty → `0.0`. |
| `spec_rows(result)` | Specification rows sorted by trust, descending. |
| `flat_rows_to_csv(rows)` | CSV export using the union of all columns. |
| `results_to_json(results)` | JSON export. |

`analyse_product` and `analyse_catalogue` wrap the engine's async API in a
`_run` helper. This exists because Streamlit executes scripts on a worker
thread with no running event loop; calling `asyncio.run` naively there fails
intermittently. Do not remove the wrapper.

---

## 4. How the engine works

### The pipeline

```python
from product_intelligence import ProductIntelligencePipeline, RawProduct

pipeline = ProductIntelligencePipeline()          # llm_router=None → deterministic
result = await pipeline.process(
    RawProduct(name="ABB M3BP 160MLA 11kW 400V IE3"),
    use_llm=False,
)
```

Other entry points on the same object:

- `process_to_object(raw, use_llm)` — returns the `EnrichedProduct` rather than a dict
- `process_batch(raws, use_llm, concurrency)` — whole catalogue plus consistency report
- `process_stream(raws, chunk_size=...)` — bounded-memory streaming for large catalogues

`process_batch` holds every record in memory; at 50,000 records that peaks near
950 MB. `process_stream` keeps only `chunk_size` records live and yields chunks
as they complete, so peak memory stays flat. Use `process_stream` for anything
resembling a real catalogue. Note that cross-record consistency is deliberately
*not* run during streaming, because it needs the entire set to compare against.

### Stages inside `enricher.enrich()`

1. **Extract** — `SpecExtractor` runs `SPEC_PATTERNS` over the concatenated
   text. Hits become `Origin.EXTRACTED`.
2. **Classify** — `TaxonomyClassifier` assigns a leaf category.
   `Origin.TAXONOMY`.
3. **Derive** — computed fields (e.g. full-load current from kW and V).
   `Origin.DERIVED`.
4. **LLM** — *only if* `use_llm=True` **and** an `llm_router` was supplied.
   Currently neither is true in this deployment, so this stage is skipped
   entirely. `Origin.LLM`.
5. **Describe** — `explainer` renders the evidence chain into prose.

Then `validator.validate()` and `explainer.explain()` run, in that order.

### Origin and trust — the core invariant

```python
class Origin(str, Enum):
    INPUT     = "input"       # supplied verbatim by the user or source system
    EXTRACTED = "extracted"   # deterministic regex/parser hit on source text
    DERIVED   = "derived"     # computed from other fields
    TAXONOMY  = "taxonomy"    # rule-based classifier
    LLM       = "llm"         # generated by the language model
    DEFAULT   = "default"     # category default, weakest evidence

ORIGIN_TRUST = {
    Origin.INPUT:     1.00,
    Origin.EXTRACTED: 0.95,
    Origin.DERIVED:   0.85,
    Origin.TAXONOMY:  0.80,
    Origin.LLM:       0.65,
    Origin.DEFAULT:   0.30,
}
```

A value's confidence can never exceed its origin's ceiling. An LLM guess at
0.65 cannot outrank an extracted value at 0.95. This is the mechanism that
stops enrichment from silently corrupting ground truth, and it is why adding a
language model later is a contained change rather than a risky one.

`_merge_llm_specs` enforces this at merge time: proposed keys outside
`LLM_ALLOWED_SPECS` are rejected, keys already known from `INPUT` or
`EXTRACTED` are rejected, values with unrecognised units are rejected. Every
rejection is recorded with a reason.

### Validation

`ProductValidator` produces `Finding` objects with severities `CRITICAL`,
`MAJOR`, `MINOR`, `INFO`, each carrying a stable `rule_id` such as
`VAL.CONF.LLM_INFERRED`. Rule IDs are part of the public contract — treat
renaming one as a breaking change.

`CatalogConsistencyChecker` runs across a whole batch, catching contradictions
between records that no single-record check could see.

### Ingestion

`ingest(filename, content, content_type)` dispatches on extension and MIME
type. Supported: `.csv .tsv .json .jsonl .ndjson .html .htm .txt .md .pdf
.xlsx .xlsm`.

`pypdf` and `openpyxl` are imported lazily. If they are missing the app still
starts and PDF/Excel uploads return a clear message rather than crashing. Keep
this property.

---

## 5. Tests

```bash
pip install -r requirements-dev.txt
pytest
```

**358 tests, all passing** with every service running. In a clean environment
with only `requirements.txt` installed — the cloud configuration — 179 pass and
the platform tests skip cleanly, which was verified in a fresh virtualenv
rather than assumed.

The suite is not decorative. `test_product_intelligence.py` enforces accuracy
thresholds against `data/eval/`, and the run fails if quality regresses:

- Category accuracy **≥ 0.90**
- Specification F1 **≥ 0.90**
- Hallucination rate **≤ 0.05**

`test_ui.py` uses Streamlit's headless `AppTest` to drive the actual app —
filling the form, pressing the button, reading the rendered output — so a UI
break fails the suite rather than being discovered in production.

Verification performed on this package: the archive was extracted to a clean
directory and the suite re-run there, giving 179 passes on a fresh extract. The
app was booted and returned HTTP 200, and screenshots were captured from a real
browser.

### One known divergence from the main backend

While reviewing rendered output, a genuine engine bug surfaced: IEC motor frame
designations ending in `L` — `100L`, `160L`, `180L`, `200L`, `225L` — were
being read as volumes in litres by the `EXT.VOLUME.BARE` rule in
`extractor.py`.

It is fixed **here** with negative lookbehinds for `frame` and `frame size`
(with and without a colon), and covered by eight parametrised regression tests.
Real volumes still extract correctly.

**This fix exists only in this prototype copy. The same bug is still present in
the main backend archive.** It is a one-line change to `EXT.VOLUME.BARE`.
Port it back.

---

## 6. What was removed, and what that cost

> This table describes **Mode A**. Everything marked "removed" now exists again
> in Mode C; it is removed from the cloud deployment on purpose, not missing.

| Removed from Mode A | Was responsible for | Consequence in Mode A |
|---|---|---|
| Docker / Compose | Reproducible multi-service environment | Runs as one process; setup is `pip install` |
| PostgreSQL | Persistence, job records, audit history | Nothing is saved between page loads |
| Redis | Caching, Celery broker | Every request recomputes from scratch |
| Qdrant | Vector search, semantic dedupe | No similarity search |
| Neo4j | Relationship graph | No cross-product graph queries |
| FastAPI | HTTP API, auth, rate limiting | No programmatic access; UI only |
| Next.js | Production frontend | Streamlit is the only interface |
| Local LLM | Gap-filling enrichment | Deterministic extraction only |
| Celery | Background jobs | Everything runs in the request |

Nothing in the engine was removed. Everything above was *around* the engine.

The honest summary: what remains is complete, tested and correct, but it is a
single-user, stateless, deterministic slice. Restoring the rest is additive
work, not a rewrite — which is the whole point of having kept the engine free
of infrastructure dependencies.

Note that `enricher.py` and `evaluation.py` each contain one `app.*` import.
Both are lazy and guarded by `try/except`, which is why the engine runs
standalone. `_call_llm` imports `app.services.llm.base.ChatMessage` inside the
function and returns `None` if the import fails. That module now exists, so the
LLM stage activates whenever a router is supplied — with no change to the
engine. See `LOCAL_LLM.md`.
