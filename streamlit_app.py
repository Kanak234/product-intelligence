"""
Product Intelligence — prototype UI.

Runs the deterministic enrichment pipeline directly in the Streamlit process:
no database, no container, no model server. Everything shown on screen is
produced by the same engine the full platform's API calls.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List

import streamlit as st

from app_core import (
    IngestionError,
    SUPPORTED_FORMATS,
    analyse_catalogue,
    analyse_product,
    build_raw,
    flat_rows_to_csv,
    ingest_upload,
    origin_style,
    parse_attributes,
    provenance_breakdown,
    results_to_json,
    severity_style,
    spec_rows,
    traceability,
)

st.set_page_config(
    page_title="Product Intelligence",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================================
# styling
#
# The subject is industrial measurement, so the page borrows from instrument
# panels and datasheets: condensed labels, monospaced values, hairline rules,
# and one colour system that carries meaning rather than decoration.
# =========================================================================

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #10202B;
  --ink-soft: #48606D;
  --rule: #D3DBDF;
  --panel: #FFFFFF;
  --wash: #EDF1F2;
}

html, body, [class*="css"], .stApp { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
.stApp { background: var(--wash); }
.block-container { padding-top: 2.2rem; max-width: 1280px; }

h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: .01em; color: var(--ink); }

.masthead { border-bottom: 2px solid var(--ink); padding-bottom: .7rem; margin-bottom: 1.4rem; }
.masthead .title {
  font-family: 'Barlow Condensed', sans-serif; font-size: 2.5rem; font-weight: 600;
  line-height: 1; color: var(--ink); text-transform: uppercase; letter-spacing: .02em;
}
.masthead .sub {
  font-family: 'IBM Plex Mono', monospace; font-size: .72rem; color: var(--ink-soft);
  text-transform: uppercase; letter-spacing: .13em; margin-top: .45rem;
}

.eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: .68rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .15em; color: var(--ink-soft);
  border-bottom: 1px solid var(--rule); padding-bottom: .35rem; margin: 1.5rem 0 .8rem;
}

.panel {
  background: var(--panel); border: 1px solid var(--rule);
  border-left: 3px solid var(--ink); padding: 1rem 1.1rem; margin-bottom: .8rem;
}

.headline {
  font-family: 'Barlow Condensed', sans-serif; font-size: 1.55rem; font-weight: 600;
  color: var(--ink); line-height: 1.15;
}
.headline .code {
  font-family: 'IBM Plex Mono', monospace; font-size: .72rem; color: var(--ink-soft);
  letter-spacing: .08em; display: block; margin-top: .3rem; font-weight: 500;
}

/* the provenance meter: the page's signature element */
.spec { border-bottom: 1px solid var(--rule); padding: .6rem 0; }
.spec:last-child { border-bottom: none; }
.spec-head { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }
.spec-name {
  font-family: 'IBM Plex Mono', monospace; font-size: .74rem; color: var(--ink-soft);
  text-transform: uppercase; letter-spacing: .08em;
}
.spec-value {
  font-family: 'IBM Plex Mono', monospace; font-size: 1.02rem; font-weight: 600;
  color: var(--ink); text-align: right;
}
.spec-meta { display: flex; align-items: center; gap: .55rem; margin-top: .4rem; }
.chip {
  font-family: 'IBM Plex Mono', monospace; font-size: .62rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .09em; color: #FFF;
  padding: .13rem .45rem; white-space: nowrap;
}
.meter { flex: 1; height: 3px; background: var(--rule); }
.meter > span { display: block; height: 100%; }
.conf {
  font-family: 'IBM Plex Mono', monospace; font-size: .68rem; color: var(--ink-soft);
  min-width: 2.6rem; text-align: right;
}
.quote {
  font-family: 'IBM Plex Mono', monospace; font-size: .7rem; color: var(--ink-soft);
  margin-top: .35rem; padding-left: .6rem; border-left: 2px solid var(--rule);
}

.stat { background: var(--panel); border: 1px solid var(--rule); padding: .8rem .9rem; height: 100%; }
.stat .k {
  font-family: 'IBM Plex Mono', monospace; font-size: .62rem; color: var(--ink-soft);
  text-transform: uppercase; letter-spacing: .1em;
}
.stat .v {
  font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 600;
  color: var(--ink); line-height: 1.1; margin-top: .2rem;
}
.stat .n { font-family: 'IBM Plex Mono', monospace; font-size: .64rem; color: var(--ink-soft); }

.finding { background: var(--panel); border: 1px solid var(--rule); border-left: 3px solid; padding: .7rem .9rem; margin-bottom: .5rem; }
.finding .top { display: flex; align-items: center; gap: .5rem; margin-bottom: .3rem; }
.finding .rule { font-family: 'IBM Plex Mono', monospace; font-size: .64rem; color: var(--ink-soft); letter-spacing: .06em; }
.finding .msg { font-size: .88rem; color: var(--ink); }
.finding .fix { font-size: .8rem; color: var(--ink-soft); margin-top: .25rem; }

.why { font-size: .87rem; color: var(--ink); line-height: 1.55; margin-bottom: .4rem; padding-left: .9rem; position: relative; }
.why:before { content: "—"; position: absolute; left: 0; color: var(--ink-soft); }

.legend { display: flex; flex-wrap: wrap; gap: .9rem; margin-top: .3rem; }
.legend .item { display: flex; align-items: center; gap: .35rem; font-family: 'IBM Plex Mono', monospace; font-size: .64rem; color: var(--ink-soft); }
.legend .dot { width: 9px; height: 9px; }

.stButton > button {
  font-family: 'Barlow Condensed', sans-serif; text-transform: uppercase;
  letter-spacing: .06em; font-weight: 600; font-size: 1rem;
  border-radius: 0; border: 1px solid var(--ink); background: var(--ink); color: #FFF;
}
.stButton > button:hover { background: #1E3746; border-color: #1E3746; color: #FFF; }
.stTabs [data-baseweb="tab"] { font-family: 'Barlow Condensed', sans-serif; text-transform: uppercase; letter-spacing: .05em; font-size: 1rem; }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--rule); }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


# =========================================================================
# rendering
# =========================================================================


def render_stat(label: str, value: str, note: str = "") -> str:
    return (
        f'<div class="stat"><div class="k">{esc(label)}</div>'
        f'<div class="v">{esc(value)}</div>'
        f'<div class="n">{esc(note)}</div></div>'
    )


def render_specs(result: Dict[str, Any]) -> None:
    rows = spec_rows(result)
    if not rows:
        st.info("No specifications were found. Add a description or a datasheet line with figures in it.")
        return

    blocks: List[str] = []
    for row in rows:
        label, colour = origin_style(row["origin"])
        pct = int(round(row["confidence"] * 100))
        evidence = row["evidence"][0] if row["evidence"] else {}
        quote = evidence.get("source_text") or ""
        quote_html = f'<div class="quote">{esc(quote)}</div>' if quote else ""
        blocks.append(
            '<div class="spec">'
            f'<div class="spec-head"><span class="spec-name">{esc(row["field"].replace("_", " "))}</span>'
            f'<span class="spec-value">{esc(row["value"])}</span></div>'
            '<div class="spec-meta">'
            f'<span class="chip" style="background:{colour}">{esc(label)}</span>'
            f'<span class="meter"><span style="width:{pct}%;background:{colour}"></span></span>'
            f'<span class="conf">{pct}%</span></div>'
            f"{quote_html}</div>"
        )
    st.markdown(f'<div class="panel">{"".join(blocks)}</div>', unsafe_allow_html=True)


def render_legend(result: Dict[str, Any]) -> None:
    counts = provenance_breakdown(result)
    if not counts:
        return
    items = []
    for origin, count in sorted(counts.items(), key=lambda pair: -pair[1]):
        label, colour = origin_style(origin)
        items.append(
            f'<div class="item"><span class="dot" style="background:{colour}"></span>'
            f"{esc(label)} · {count}</div>"
        )
    st.markdown(f'<div class="legend">{"".join(items)}</div>', unsafe_allow_html=True)


def render_findings(result: Dict[str, Any]) -> None:
    findings = (result.get("validation") or {}).get("findings") or []
    if not findings:
        st.markdown(
            '<div class="panel">Every rule passed. Nothing needs a human here.</div>',
            unsafe_allow_html=True,
        )
        return

    for finding in findings:
        label, colour = severity_style(finding.get("severity"))
        suggestion = finding.get("suggestion") or ""
        fix = f'<div class="fix">{esc(suggestion)}</div>' if suggestion else ""
        st.markdown(
            f'<div class="finding" style="border-left-color:{colour}">'
            f'<div class="top"><span class="chip" style="background:{colour}">{esc(label)}</span>'
            f'<span class="rule">{esc(finding.get("rule_id", ""))} · {esc(finding.get("field", ""))}</span></div>'
            f'<div class="msg">{esc(finding.get("message", ""))}</div>{fix}</div>',
            unsafe_allow_html=True,
        )


def render_reasoning(result: Dict[str, Any]) -> None:
    explanation = result.get("explanation") or {}
    summary = explanation.get("summary")
    if summary:
        st.markdown(f'<div class="panel">{esc(summary)}</div>', unsafe_allow_html=True)

    reasoning = explanation.get("category_reasoning") or {}
    reasons = reasoning.get("why") or []
    if reasons:
        st.markdown('<div class="eyebrow">Why this category</div>', unsafe_allow_html=True)
        st.markdown(
            "".join(f'<div class="why">{esc(reason)}</div>' for reason in reasons),
            unsafe_allow_html=True,
        )

    alternatives = reasoning.get("alternatives") or result.get("category_alternatives") or []
    if alternatives:
        st.markdown('<div class="eyebrow">Categories considered and rejected</div>', unsafe_allow_html=True)
        st.dataframe(
            [
                {
                    "Category": alt.get("category", ""),
                    "Code": alt.get("code", ""),
                    "Score": alt.get("score", 0),
                    "Confidence": alt.get("confidence", 0),
                }
                for alt in alternatives
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_single(result: Dict[str, Any]) -> None:
    category = result.get("category") or {}
    path = " › ".join(result.get("category_path") or []) or "Unclassified"
    metrics = result.get("metrics") or {}
    validation = result.get("validation") or {}

    st.markdown(
        f'<div class="panel"><div class="headline">{esc(result.get("name", ""))}'
        f'<span class="code">{esc(path)} · fingerprint {esc(result.get("fingerprint", ""))}</span></div></div>',
        unsafe_allow_html=True,
    )

    trace_pct = f"{int(round(traceability(result) * 100))}%"
    columns = st.columns(4)
    stats = [
        ("Traceable to source", trace_pct, "not model-invented"),
        ("Validation", f"{validation.get('score', 0):.2f}", str(validation.get("status", ""))),
        ("Completeness", f"{metrics.get('completeness', 0):.0%}", "fields populated"),
        ("Specifications", str(metrics.get("spec_count", 0)), f"category conf {category.get('confidence', 0):.2f}"),
    ]
    for column, (label, value, note) in zip(columns, stats):
        column.markdown(render_stat(label, value, note), unsafe_allow_html=True)

    description = (result.get("enriched_description") or {}).get("value")
    if description:
        st.markdown('<div class="eyebrow">Generated description</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="panel">{esc(description)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Specifications and where each came from</div>', unsafe_allow_html=True)
    render_legend(result)
    render_specs(result)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="eyebrow">Validation findings</div>', unsafe_allow_html=True)
        render_findings(result)
    with right:
        st.markdown('<div class="eyebrow">Reasoning</div>', unsafe_allow_html=True)
        render_reasoning(result)

    keywords = result.get("keywords") or []
    if keywords:
        st.markdown('<div class="eyebrow">Search keywords</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="panel">{esc(", ".join(keywords))}</div>', unsafe_allow_html=True)

    st.download_button(
        "Download this record as JSON",
        data=results_to_json(result),
        file_name=f"product-{result.get('fingerprint', 'record')}.json",
        mime="application/json",
    )


# =========================================================================
# page
# =========================================================================

st.markdown(
    '<div class="masthead"><div class="title">Product Intelligence</div>'
    '<div class="sub">Deterministic enrichment · validation · explainable provenance</div></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="eyebrow">How this runs</div>', unsafe_allow_html=True)
    st.markdown(
        "Every value on screen comes from rule-based extraction, unit conversion and "
        "physics-derived computation. No model server is contacted, so results are "
        "identical on every run and every machine."
    )
    st.markdown('<div class="eyebrow">Provenance scale</div>', unsafe_allow_html=True)
    legend = []
    for origin in ("input", "extracted", "derived", "taxonomy", "llm", "default"):
        label, colour = origin_style(origin)
        legend.append(
            f'<div class="item"><span class="dot" style="background:{colour}"></span>{esc(label)}</div>'
        )
    st.markdown(f'<div class="legend" style="flex-direction:column;gap:.4rem">{"".join(legend)}</div>',
                unsafe_allow_html=True)

tab_single, tab_batch = st.tabs(["Single product", "Catalogue file"])

with tab_single:
    st.markdown('<div class="eyebrow">Describe the product</div>', unsafe_allow_html=True)
    examples = {
        "Induction motor": (
            "ABB M2BAX 2.2kW 415V 3-phase 1440rpm IE3 induction motor",
            "Foot mounted TEFC squirrel cage motor, IP55, frame 100L, 50Hz",
        ),
        "Circuit breaker": (
            "Siemens 3RV2011-1JA10 motor protection circuit breaker 690V 10A",
            "Thermal magnetic release, IP20, screw terminals, 3 pole",
        ),
        "Centrifugal pump": (
            "Grundfos CR 5-10 vertical multistage centrifugal pump 2.2kW",
            "Stainless steel 304, 5 m3/h at 60 m head, 415V 50Hz, mechanical seal",
        ),
        "Blank": ("", ""),
    }
    choice = st.radio("Start from", list(examples), horizontal=True, label_visibility="collapsed")
    default_name, default_description = examples[choice]

    name = st.text_input("Product name or title", value=default_name,
                         placeholder="Manufacturer, model, ratings — whatever the source system holds")
    description = st.text_area("Description", value=default_description, height=90,
                               placeholder="Datasheet text, catalogue blurb, supplier notes")

    left, right = st.columns(2)
    category_hint = left.text_input("Category hint", placeholder="Optional — leave blank to let the classifier decide")
    attribute_text = right.text_area("Known attributes", height=80,
                                     placeholder="One per line\nBrand: ABB\nframe = 100L")

    if st.button("Analyse product", type="primary"):
        if not name.strip():
            st.error("Enter a product name. Everything else is optional.")
        else:
            raw = build_raw(name, description, category_hint, parse_attributes(attribute_text))
            with st.spinner("Extracting, classifying, validating…"):
                st.session_state["single"] = analyse_product(raw)

    if st.session_state.get("single"):
        render_single(st.session_state["single"])

with tab_batch:
    st.markdown('<div class="eyebrow">Upload a catalogue</div>', unsafe_allow_html=True)
    st.markdown(
        f"Accepted: {', '.join(SUPPORTED_FORMATS)}. A CSV needs a name column; description "
        "and category are used when present."
    )
    upload = st.file_uploader("Catalogue file", type=[fmt.lstrip(".") for fmt in SUPPORTED_FORMATS],
                              label_visibility="collapsed")

    if upload is not None and st.button("Process catalogue", type="primary"):
        try:
            ingested = ingest_upload(upload.getvalue(), upload.name)
        except IngestionError as error:
            st.error(f"Could not read that file: {error}")
        else:
            if not ingested.products:
                st.warning("The file parsed, but no products were found in it.")
            else:
                with st.spinner(f"Processing {len(ingested.products)} products…"):
                    st.session_state["batch"] = analyse_catalogue(ingested.products)
                    st.session_state["ingest_report"] = ingested.to_dict()

    report = st.session_state.get("ingest_report")
    batch = st.session_state.get("batch")

    if report:
        skipped = len(report.get("skipped") or [])
        st.markdown(
            f'<div class="panel">Read <strong>{report.get("ingested", 0)}</strong> of '
            f'{report.get("row_count", 0)} rows from {esc(report.get("source_ref", ""))} '
            f"({esc(report.get('source_type', ''))}). Skipped {skipped}.</div>",
            unsafe_allow_html=True,
        )
        for warning in report.get("warnings") or []:
            st.warning(warning)

    if batch:
        summary = batch.get("summary") or {}
        columns = st.columns(4)
        stats = [
            ("Products", str(summary.get("total", 0)), "processed"),
            ("Valid", str(summary.get("valid", 0)), f"{summary.get('invalid', 0)} need review"),
            ("Mean validation", f"{summary.get('mean_validation_score', 0):.2f}", "0–1 scale"),
            ("Mean completeness", f"{summary.get('mean_completeness', 0):.0%}", "fields populated"),
        ]
        for column, (label, value, note) in zip(columns, stats):
            column.markdown(render_stat(label, value, note), unsafe_allow_html=True)

        products = batch.get("products") or []
        st.markdown('<div class="eyebrow">Catalogue</div>', unsafe_allow_html=True)
        st.dataframe(
            [
                {
                    "Name": product.get("name", ""),
                    "Category": (product.get("category") or {}).get("value", ""),
                    "Specs": (product.get("metrics") or {}).get("spec_count", 0),
                    "Traceable": traceability(product),
                    "Validation": (product.get("validation") or {}).get("score", 0),
                    "Status": (product.get("validation") or {}).get("status", ""),
                }
                for product in products
            ],
            use_container_width=True,
            hide_index=True,
        )

        consistency = batch.get("catalog_consistency") or {}
        issues = consistency.get("findings") or consistency.get("issues") or []
        if issues:
            st.markdown('<div class="eyebrow">Cross-record consistency</div>', unsafe_allow_html=True)
            st.dataframe(issues, use_container_width=True, hide_index=True)

        flat = []
        for product in products:
            row = {
                "name": product.get("name", ""),
                "category": (product.get("category") or {}).get("value", ""),
                "category_path": " > ".join(product.get("category_path") or []),
                "description": (product.get("enriched_description") or {}).get("value", ""),
                "keywords": ", ".join(product.get("keywords") or []),
                "validation_status": (product.get("validation") or {}).get("status", ""),
                "validation_score": (product.get("validation") or {}).get("score", 0),
                "completeness": (product.get("metrics") or {}).get("completeness", 0),
            }
            for key, field in sorted((product.get("specifications") or {}).items()):
                unit = field.get("unit") or ""
                row[f"spec.{key}"] = f"{field.get('value')} {unit}".strip() if unit else field.get("value")
            flat.append(row)

        left, right = st.columns(2)
        left.download_button("Download enriched catalogue (CSV)", data=flat_rows_to_csv(flat),
                             file_name="enriched-catalogue.csv", mime="text/csv")
        right.download_button("Download full result (JSON)", data=results_to_json(batch),
                              file_name="enriched-catalogue.json", mime="application/json")

        st.markdown('<div class="eyebrow">Inspect one record</div>', unsafe_allow_html=True)
        names = [product.get("name", f"Record {index}") for index, product in enumerate(products)]
        selected = st.selectbox("Record", names, label_visibility="collapsed")
        if selected in names:
            render_single(products[names.index(selected)])
