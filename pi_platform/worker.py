"""
Background jobs — stage 3 of DOCKER.md.

Catalogue enrichment belongs in a worker, not in a request. With the local
model enabled a single product takes seconds on CPU, so a thousand-record
catalogue is minutes of work: an HTTP request would time out long before it
finished.

Importing this module does not require Redis. Celery is configured lazily and
the tasks are plain functions underneath, so they can be called and tested
directly without a broker.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

BROKER_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    from celery import Celery

    celery_app = Celery("product_intel", broker=BROKER_URL, backend=BROKER_URL)
    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        # A catalogue job is long by design; do not let the default limits kill it.
        task_time_limit=60 * 60,
        task_soft_time_limit=55 * 60,
        worker_max_tasks_per_child=50,
    )
except Exception:  # pragma: no cover - celery is an optional dependency
    celery_app = None


def _run(coro):
    """Run an async coroutine from Celery's synchronous worker context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


# -- the actual work, independent of Celery ---------------------------------

def process_catalogue(
    records: List[Dict[str, Any]],
    job_id: Optional[str] = None,
    use_llm: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Enrich a catalogue in bounded memory, persisting as it goes.

    Plain function on purpose: the tests call this directly, with no broker and
    no worker process, and exercise exactly the code that runs in production.
    """
    from .service import ProductService

    service = ProductService()

    if job_id is None:
        job_id = service.repository.create_job(total=len(records))

    try:
        return _run(
            service.enrich_catalogue_streaming(records, use_llm=use_llm, job_id=job_id)
        )
    except Exception as exc:
        if job_id:
            service.repository.update_job(job_id, status="failed", error=str(exc))
        raise


def process_single(
    name: str, description: str = "", use_llm: Optional[bool] = None
) -> Dict[str, Any]:
    from .service import ProductService

    return _run(
        ProductService().enrich(name=name, description=description, use_llm=use_llm)
    )


# -- Celery task registration ------------------------------------------------

if celery_app is not None:  # pragma: no branch

    @celery_app.task(name="product_intel.process_catalogue", bind=True)
    def process_catalogue_task(self, records, job_id=None, use_llm=None):
        return process_catalogue(records, job_id=job_id, use_llm=use_llm)

    @celery_app.task(name="product_intel.process_single")
    def process_single_task(name, description="", use_llm=None):
        return process_single(name, description=description, use_llm=use_llm)
