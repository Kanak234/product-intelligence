"""
Core logic for the Product Intelligence prototype.

Deliberately free of Streamlit imports so every function here is testable in a
bare environment. The UI layer (streamlit_app.py) does presentation only.

The pipeline runs deterministically: no model server, no database, no network.
An LLM is optional in the engine and simply absent here, which is why the
prototype behaves identically on a laptop and on a hosted runner.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
from typing import Any, Dict, List, Optional, Tuple

from product_intelligence import (
    IngestionError,
    IngestionResult,
    ORIGIN_TRUST,
    Origin,
    ProductIntelligencePipeline,
    RawProduct,
    SUPPORTED_FORMATS,
    ingest,
)

__all__ = [
    "ORIGIN_STYLE",
    "SEVERITY_STYLE",
    "SUPPORTED_FORMATS",
    "analyse_catalogue",
    "analyse_product",
    "build_raw",
    "flat_rows_to_csv",
    "ingest_upload",
    "origin_style",
    "parse_attributes",
    "provenance_breakdown",
    "results_to_json",
    "severity_style",
    "traceability",
]


# =========================================================================
# presentation vocabulary
#
# Provenance is the product's whole argument, so it gets a first-class colour
# system rather than incidental styling. Order matches ORIGIN_TRUST.
# =========================================================================

ORIGIN_STYLE: Dict[str, Tuple[str, str]] = {
    Origin.INPUT.value: ("From source", "#0B6E4F"),
    Origin.EXTRACTED.value: ("Read from text", "#0E7C7B"),
    Origin.DERIVED.value: ("Computed", "#2F5DD1"),
    Origin.TAXONOMY.value: ("Rule matched", "#5B4BC4"),
    Origin.LLM.value: ("Model inferred", "#8A5CF6"),
    Origin.DEFAULT.value: ("Category default", "#8A7A66"),
}

SEVERITY_STYLE: Dict[str, Tuple[str, str]] = {
    "critical": ("Critical", "#B3261E"),
    "major": ("Major", "#C24A0F"),
    "minor": ("Minor", "#B08000"),
    "info": ("Info", "#4A6572"),
}


def origin_style(origin: Optional[str]) -> Tuple[str, str]:
    """Label and colour for a provenance tag. Unknown tags stay neutral."""
    if not origin:
        return ("Unknown", "#5A6B75")
    return ORIGIN_STYLE.get(str(origin).lower(), (str(origin).title(), "#5A6B75"))


def severity_style(severity: Optional[str]) -> Tuple[str, str]:
    if not severity:
        return ("Info", "#4A6572")
    return SEVERITY_STYLE.get(str(severity).lower(), (str(severity).title(), "#4A6572"))


# =========================================================================
# input handling
# =========================================================================


def parse_attributes(text: str) -> Dict[str, str]:
    """
    Turn free-form ``key: value`` lines into an attribute map.

    Accepts ``:`` or ``=`` as the separator, ignores blank lines, and keeps the
    last value when a key repeats. A line with no separator is skipped rather
    than guessed at, so a typo never becomes a silent fake specification.
    """
    attributes: Dict[str, str] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        separator = None
        for candidate in (":", "="):
            if candidate in line:
                separator = candidate
                break
        if separator is None:
            continue
        key, _, value = line.partition(separator)
        key = key.strip()
        value = value.strip()
        if key and value:
            attributes[key] = value
    return attributes


def build_raw(
    name: str,
    description: str = "",
    category: str = "",
    attributes: Optional[Dict[str, str]] = None,
    source: str = "manual",
    source_ref: str = "",
) -> RawProduct:
    """Assemble a RawProduct, trimming the whitespace a form inevitably adds."""
    return RawProduct(
        name=(name or "").strip(),
        description=(description or "").strip(),
        category=(category or "").strip(),
        source=source,
        source_ref=source_ref,
        attributes=dict(attributes or {}),
    )


def ingest_upload(content: Any, filename: str) -> IngestionResult:
    """
    Parse an uploaded catalogue file into RawProducts.

    Raises IngestionError with a readable message; the caller shows it as-is
    rather than a stack trace.
    """
    return ingest(content, filename=filename)


# =========================================================================
# running the pipeline
# =========================================================================


def _run(coro: Any) -> Any:
    """
    Run a coroutine from synchronous code.

    Streamlit executes scripts on a worker thread with no event loop, so
    asyncio.run is correct there. The fallback covers a thread that already
    owns a loop, which would otherwise raise instead of simply working.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}

    def _worker() -> None:
        result["value"] = asyncio.run(coro)

    import threading

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    return result["value"]


def analyse_product(raw: RawProduct, include_trace: bool = True) -> Dict[str, Any]:
    """Enrich, validate and explain one product. Deterministic, no network."""
    pipeline = ProductIntelligencePipeline()
    return _run(pipeline.process(raw, use_llm=False, include_trace=include_trace))


def analyse_catalogue(raws: List[RawProduct]) -> Dict[str, Any]:
    """Process a whole catalogue and add the cross-record consistency report."""
    pipeline = ProductIntelligencePipeline()
    return _run(pipeline.process_batch(raws, use_llm=False))


# =========================================================================
# reading the results
# =========================================================================


def provenance_breakdown(result: Dict[str, Any]) -> Dict[str, int]:
    """
    Count specifications by where their value came from.

    This is the number that answers the question a buyer actually asks: how
    much of this record is the supplier's own data, and how much did we make up?
    """
    counts: Dict[str, int] = {}
    for field in (result.get("specifications") or {}).values():
        origin = str(field.get("origin", "unknown"))
        counts[origin] = counts.get(origin, 0) + 1
    return counts


def traceability(result: Dict[str, Any]) -> float:
    """
    Fraction of specifications traceable to the source text or a computation
    over it, as opposed to model inference or a category default.

    Returns 0.0 for a record with no specifications, which reads correctly as
    "nothing here is traceable" rather than a misleading perfect score.
    """
    counts = provenance_breakdown(result)
    total = sum(counts.values())
    if not total:
        return 0.0
    trusted = sum(
        count
        for origin, count in counts.items()
        if origin in {Origin.INPUT.value, Origin.EXTRACTED.value, Origin.DERIVED.value}
    )
    return round(trusted / total, 4)


def spec_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Specifications flattened for display, strongest provenance first."""
    rows: List[Dict[str, Any]] = []
    for key, field in (result.get("specifications") or {}).items():
        origin = str(field.get("origin", ""))
        value = field.get("value")
        unit = field.get("unit") or ""
        rows.append(
            {
                "field": key,
                "value": f"{value} {unit}".strip() if unit else value,
                "origin": origin,
                "confidence": float(field.get("confidence") or 0.0),
                "raw": field.get("raw") or "",
                "evidence": field.get("evidence") or [],
                "_trust": ORIGIN_TRUST.get(Origin(origin), 0.0) if origin in Origin._value2member_map_ else 0.0,
            }
        )
    rows.sort(key=lambda row: (-row["_trust"], row["field"]))
    return rows


# =========================================================================
# export
# =========================================================================


def flat_rows_to_csv(rows: List[Dict[str, Any]]) -> bytes:
    """
    Serialise flat product rows to CSV bytes.

    Columns are the union of every row's keys, so a catalogue whose records
    carry different specifications still exports as one rectangular file
    instead of losing the columns the first row happened not to have.
    """
    if not rows:
        return b""
    columns: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue().encode("utf-8")


def results_to_json(payload: Any) -> bytes:
    """Pretty JSON bytes for download. Falls back to str for odd values."""
    return json.dumps(payload, indent=2, default=str).encode("utf-8")
