"""
HTTP API — the programmatic surface the Streamlit prototype does not have.

Routes are thin. All behaviour lives in `ProductService`, so the API and the
Celery worker and the Streamlit app all take the same path through the engine
and cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .service import ProductService

app = FastAPI(
    title="Product Intelligence API",
    version="2.1",
    description="Structured, validated, fully attributed product data.",
)

_service: Optional[ProductService] = None


def get_service() -> ProductService:
    """Lazily built so importing this module never touches a network service."""
    global _service
    if _service is None:
        _service = ProductService()
    return _service


def set_service(service: Optional[ProductService]) -> None:
    """Inject a service. Used by the tests to supply configured instances."""
    global _service
    _service = service


# -- the browser frontend ----------------------------------------------------

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """
    Serve the HTML frontend from the API itself.

    Same origin on purpose: opening the file directly from disk would make
    every request cross-origin and require CORS to be opened up, which is a
    worse default than simply serving the page from the process that answers
    its calls.
    """
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="frontend not installed")
    return FileResponse(index)


# -- request models ----------------------------------------------------------

class ProductIn(BaseModel):
    name: str = Field(..., min_length=1, description="Product name or title line")
    description: str = ""
    attributes: Dict[str, Any] = Field(default_factory=dict)


class EnrichRequest(ProductIn):
    use_llm: Optional[bool] = None
    persist: bool = True


class CatalogueRequest(BaseModel):
    products: List[ProductIn] = Field(..., min_length=1)
    use_llm: Optional[bool] = None
    persist: bool = True
    stream: bool = Field(
        False,
        description=(
            "Bounded-memory mode. Persists as it goes and returns counts rather "
            "than the full result set. Use for large catalogues."
        ),
    )


# -- routes ------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    """Which services are actually reachable. Safe to expose; no credentials."""
    return get_service().health()


@app.get("/config")
def config() -> Dict[str, Any]:
    return get_settings().describe()


@app.post("/enrich")
async def enrich(request: EnrichRequest) -> Dict[str, Any]:
    if not request.name.strip():
        raise HTTPException(status_code=422, detail="name must not be blank")

    return await get_service().enrich(
        name=request.name,
        description=request.description,
        attributes=request.attributes,
        use_llm=request.use_llm,
        persist=request.persist,
    )


@app.post("/catalogue")
async def catalogue(request: CatalogueRequest) -> Dict[str, Any]:
    service = get_service()
    records = [p.model_dump() for p in request.products]

    if not any(r["name"].strip() for r in records):
        raise HTTPException(status_code=422, detail="no product had a name")

    if request.stream:
        job_id = service.repository.create_job(total=len(records))
        return await service.enrich_catalogue_streaming(
            records, use_llm=request.use_llm, job_id=job_id
        )

    return await service.enrich_catalogue(
        records, use_llm=request.use_llm, persist=request.persist
    )


@app.get("/products/{product_id}")
def get_product(product_id: str) -> Dict[str, Any]:
    result = get_service().repository.get_product(product_id)
    if result is None:
        raise HTTPException(status_code=404, detail="product not found")
    return result


@app.get("/products")
def recent_products(limit: int = Query(20, ge=1, le=200)) -> Dict[str, Any]:
    return {"products": get_service().repository.recent_products(limit)}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = get_service().repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.post("/similar")
async def similar(request: ProductIn, limit: int = Query(5, ge=1, le=50)) -> Dict[str, Any]:
    """Nearest catalogue entries to a candidate product, without persisting it."""
    service = get_service()
    result = await service.enrich(
        name=request.name,
        description=request.description,
        attributes=request.attributes,
        persist=False,
        find_duplicates=False,
    )
    return {"matches": service.vectors.search(result, limit=limit)}
