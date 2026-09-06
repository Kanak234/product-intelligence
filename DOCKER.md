# Docker — roadmap back to the full stack

Docker was removed from this project deliberately, to get a working, tested,
publicly deployable application shipped. That was the right call and it is not
a decision to be embarrassed about: the deployment works, the tests pass, and
the engine was proven to be genuinely independent of its infrastructure.

This document is the plan for bringing the full stack back **when there is a
reason to**, without repeating the debugging that made it painful the first
time.

**All five stages are now implemented.** `Dockerfile`, `.dockerignore` and
`docker-compose.yml` ship in this folder, and the application code each stage
needs — persistence, cache, API, worker, vector search, local model — lives in
`pi_platform/` and `app/services/llm/`, covered by the test suite.

The compose file uses **profiles** so the stages can still be brought up one at
a time, which is the part that matters. Nothing here is dead configuration:
every service has code behind it that actually uses it.

**What has been verified, and what has not.** The application code was tested
against real PostgreSQL 16, real Redis, embedded Qdrant, a real Celery worker
consuming from Redis, and a stub model server — 358 tests. The **Docker images
themselves were never built**, because no Docker daemon was available in the
environment where this was written. Every claim in this document about
application behaviour is tested; every claim about container behaviour is
reasoned. Check the exit criteria yourself.

Streamlit Community Cloud does not read the `Dockerfile`. It installs
`requirements.txt` and runs `streamlit_app.py`, exactly as before. This was
checked rather than assumed: the suite was run in a clean virtualenv containing
only `requirements.txt` and `requirements-dev.txt`, where the platform tests
skip cleanly and the app still boots and serves HTTP 200.

---

## 1. Decide whether you actually need it

Be honest about the trigger. Add the stack back when you need something the
current shape genuinely cannot do:

- **Persistence** — results must survive a page reload, or you need audit history
- **A real API** — something other than a browser needs to call this
- **The local LLM** — a model server is a separate process by nature (see `LOCAL_LLM.md`)
- **Scale** — catalogues large enough to need background jobs and a queue
- **Semantic search** — deduplication or similarity, requiring a vector store

If none of these apply, the current deployment is not a compromise. It is the
correct size for the problem.

---

## 2. Target architecture

```
                          ┌──────────────────┐
                          │  Streamlit / UI  │
                          └────────┬─────────┘
                                   │ HTTP
                          ┌────────▼─────────┐
                          │     FastAPI      │
                          │  (the engine)    │
                          └────────┬─────────┘
        ┌──────────────┬───────────┼───────────┬──────────────┐
        ▼              ▼           ▼           ▼              ▼
   ┌─────────┐   ┌─────────┐  ┌────────┐  ┌────────┐   ┌───────────┐
   │Postgres │   │  Redis  │  │ Qdrant │  │ Celery │   │  Ollama   │
   │persist  │   │ cache + │  │ vector │  │ worker │   │ local LLM │
   │  audit  │   │ broker  │  │ search │  │  jobs  │   │           │
   └─────────┘   └─────────┘  └────────┘  └────────┘   └───────────┘
```

The engine — `product_intelligence/` — sits inside the FastAPI service and is
**not modified**. It is the same package that runs in-process today. The
services around it supply persistence, concurrency and the model.

---

## 3. Build it in stages

The single biggest cause of the original difficulty was standing everything up
at once, so that any failure was ambiguous. Do not do that again. Each stage
below must be independently verifiable before you start the next one.

### Stage 1 — Containerise only what already works — **DONE**

One service. No database, no model, no queue. Prove that the app you have runs
inside a container identically to how it runs on your machine.

Shipped in this folder: `Dockerfile`, `.dockerignore`, `docker-compose.yml`.

The Dockerfile has two targets. `base` is the application image; `test` is the
same image plus pytest, so the suite can be run inside the container rather
than only outside it. Once tests only run outside Docker the two environments
begin to drift, and you lose the ability to tell which one is wrong.

```bash
docker compose up --build                       # http://localhost:8501
docker compose --profile test run --rm test     # expect: 179 passed
docker compose down
```

Or without compose:

```bash
docker build -t product-intel .
docker run --rm -p 8501:8501 product-intel
```

**Exit criteria:** the app loads at `localhost:8501`, and the test service
reports 179 passed inside the container. Do not proceed until both are true.

Three details in the shipped files that are doing real work:

- `--server.address=0.0.0.0` is mandatory. Streamlit binds to `127.0.0.1` by
  default, which inside a container means unreachable from the host. This one
  line accounts for a great many "the container starts but I can't open it"
  hours.
- `requirements.txt` is copied and installed **before** the source. Editing a
  `.py` file then leaves the pip layer cached, so rebuilds take seconds.
- The image runs as `appuser`, not root, with a real `$HOME` — Streamlit writes
  configuration into the home directory and fails oddly without one.

**What was verified, and what was not.** The exact `CMD` and the healthcheck
were run and confirmed to return HTTP 200 on `/_stcore/health`, the compose
file parses, and the 179 tests pass on a clean extract of this archive. The
image build itself was not executed, because no Docker daemon was available in
the environment where these files were written. Run `docker compose up --build`
and check it against the exit criteria above before trusting it.

### Stage 2 — Persistence — **DONE**

```bash
docker compose --profile data up --build
```

Brings up Postgres and Redis alongside the app. `pi_platform/db.py` provides
`Repository`, storing enriched products and batch jobs.

`depends_on` with `condition: service_healthy` is the important part of the
compose file. Plain `depends_on` waits for the container to *start*, not for
Postgres to accept connections — so the app races ahead, fails to connect, and
exits. The healthcheck is what makes startup deterministic.

The application does not require any of this. `Repository` connects once at
construction and, if nothing answers, reports `available == False` and every
call becomes a no-op. That is why the stage 1 `app` service works while
`DATABASE_URL` still points at a `db` service that has not been started.

**Exit criteria:** `/health` reports `"database": true`, and a product enriched
through the API can be fetched back by id.

### Stage 3 — Cache and background jobs — **DONE**

`pi_platform/cache.py` caches results in Redis; `pi_platform/worker.py` defines
the Celery tasks. Catalogue enrichment belongs in a worker, not a request —
with the model enabled a single product takes seconds on CPU, so a thousand
records is minutes of work and an HTTP request would time out long before it
finished.

The cache key includes whether the model was used, so a deterministic result
and a model-enriched result for the same product can never be served for one
another.

```bash
docker compose --profile full up --build      # api + worker + qdrant
```

**Exit criteria:** submitting a catalogue with `"stream": true` returns a job id
immediately, `/jobs/{id}` reaches `done`, and enriching the same product twice
reports `"cache": "hit"` the second time.

### Stage 4 — Vector search — **DONE**

`pi_platform/vector.py`. Near-duplicate detection across a catalogue: the same
pump listed twice under two supplier part numbers is the problem this solves,
and no per-record rule can see it.

Read section 5 below before tuning anything here — the naive version of this
feature does not work, and the reason is instructive.

`QDRANT_URL=":memory:"` runs Qdrant embedded in-process, which is why the tests
need no server.

**Exit criteria:** two spellings of the same product are reported as duplicates
of one another, and two genuinely different products are not.

### Stage 5 — The local model — **DONE, but unproven**

```bash
docker compose --profile llm up --build
docker compose exec ollama ollama pull qwen2.5:7b-instruct   # several GB, slow
docker compose exec api sh -c 'ENABLE_LLM=1 python -c "..."'  # or set it in compose
```

See `LOCAL_LLM.md`. `ENABLE_LLM` defaults to `0` even when the container is up,
because a model that is still being pulled would otherwise make every
enrichment wait for a timeout.

This stage is last for a reason: it is the heaviest service, the slowest to
pull, and the only one whose output is non-deterministic — the worst possible
thing to be debugging simultaneously with a networking problem.

**Exit criteria:** with `ENABLE_LLM=1`, some fields come back with
`"origin": "llm"`, and the accuracy suite still passes — category accuracy
≥ 0.90, spec F1 ≥ 0.90, hallucination rate ≤ 0.05.

## 4. Rules that prevent the original pain

**One service at a time.** Every stage above has an exit criterion. Meet it
before moving on. A failure then has exactly one plausible cause.

**Read the logs before changing anything.** `docker compose logs -f <service>`.
The overwhelming majority of compose failures announce themselves clearly in
the logs and are then fixed by guesswork instead of by reading.

**Service names are hostnames.** From inside the app container, Postgres is at
`db:5432`, not `localhost:5432`. `localhost` inside a container means that
container. This is the second most common source of lost hours.

**Healthchecks, not `sleep`.** Startup order is a real problem and
`condition: service_healthy` is the real solution.

**Never bake secrets into the image.** Environment variables and a
`.env` file that is gitignored.

**Keep the tests runnable in the container.** `docker compose run --rm app
pytest` should always work. The moment tests only run outside Docker, the two
environments start drifting and you lose the ability to tell which one is
wrong.

**Never let Docker become a prerequisite for Mode A.** The Streamlit Cloud
deployment must keep working with a plain `pip install -r requirements.txt`
throughout. If a change makes the app require a container to start, that change
is wrong. This constraint is what guarantees you always have something working
to fall back to.

---

## 5. Common failures, with causes

| Symptom | Cause | Fix |
|---|---|---|
| Container runs, browser shows nothing | Streamlit bound to `127.0.0.1` | `--server.address=0.0.0.0` |
| `connection refused` to Postgres | App started before DB was ready | Healthcheck + `condition: service_healthy` |
| `could not translate host name "db"` | Using `localhost` instead of the service name | Use the compose service name |
| Rebuild is slow every time | `COPY . .` placed before `pip install` | Copy `requirements.txt` first |
| Data lost on `docker compose down` | No named volume | Add `volumes: [pgdata:/var/lib/postgresql/data]` |
| Port already in use | Local Postgres/Redis already running | Change the host-side port, e.g. `5433:5432` |
| Works locally, fails on Streamlit Cloud | A Docker-only dependency crept into the app | Keep Mode A dependency-clean |

---

## 6. Things this build actually turned up

Recorded because each one cost real time and would otherwise cost it again.

**A package named `platform` shadows the standard library.** The services layer
was briefly called `platform/`, which silently breaks any dependency that does
`import platform`. It is now `pi_platform/`. Do not rename it back.

**`pytest.ini` set `asyncio_mode = auto` while `pytest-asyncio` was not in
`requirements-dev.txt`.** On this machine the plugin happened to be installed
globally, so the suite looked green. In a clean virtualenv pytest warns
"Unknown config option: asyncio_mode" and every `async def` test silently does
not run. A suite that tests nothing still reports success — which is worse than
a failing one. Fixed; the dependency is now declared.

**Similarity alone cannot identify duplicates.** The first version of stage 4
flagged duplicates on a cosine threshold. Measured on real enriched products:

| Pair | Score | Actually? |
|---|---|---|
| Same motor, different spacing | 0.85 | duplicate |
| Same motor, typo | 0.98 | duplicate |
| **160MLA 11 kW vs 180MLA 18.5 kW** | **0.87** | **different product** |
| Different brand, different rating | 0.80 | different product |
| Motor vs pump | 0.18 | different product |

A genuinely different motor scored *higher* than a genuine duplicate. No single
threshold works, and tuning one until the tests pass would have hidden that.
The fix was to change the design, not the number: vector search now generates
*candidates*, and each candidate is confirmed by comparing specifications using
the engine's own unit-aware comparison. 11 kW and 11000 W agree; 11 kW and
18.5 kW do not. The threshold became a recall knob, where being slightly wrong
is harmless.

**Nothing created the database tables.** `/health` reported the database as
connected, and the first write then failed with `relation "products" does not
exist`. The tests had been calling `create_schema()` themselves, so they never
caught it — a fixture was doing work the real startup path did not. Fixed:
`ProductService` calls `ensure_schema()` on construction. Note this is table
creation, not migration; introduce Alembic once the schema starts changing.

**A cache hit meant a record was never stored.** The cache outlives the
database easily — a wipe, a restored backup, a Redis nobody cleared — and the
original code returned early on a hit without ever writing the row. Every
request kept succeeding while the product stayed permanently absent from
storage. Fixed: on a hit, the row is written if the database does not already
have it. Both failure modes now have regression tests.

**The engine wants a nested LLM response.** `{"specifications": {...}}`, not a
flat map. A flat response parses fine and is then silently discarded. See
`LOCAL_LLM.md` section 3.4.

## 7. Relationship to Streamlit Cloud

These two deployments do not compete. Streamlit Community Cloud does not run
your Dockerfile — it installs `requirements.txt` and runs
`streamlit_app.py`. That is unaffected by anything in this document, provided
you follow the last rule in section 4.

The intended end state:

- **Streamlit Cloud** — the public, deterministic demo. Anyone can open a link
  and see the engine work. No model, no database, no cost.
- **Docker Compose, locally** — the full simulation, with persistence,
  background jobs and the local LLM. Where development happens.

Same engine. Same tests. Two surfaces.
