"""
Explainability.

The explanation is *read off* the pipeline, not generated after the fact. Every
claim it makes traces to a `FieldValue.origin`, an `Evidence` record, a
classifier score, or a validator finding that already exists on the object.
Nothing here calls a language model and nothing here is hardcoded — if the
pipeline did not record it, the explainer does not assert it.

That distinction matters for the "explainable outputs" requirement: an
explanation a model writes about its own output is a plausible story; an
explanation derived from provenance is an audit trail.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .schema import EnrichedProduct, FieldValue, Origin
from .taxonomy import get_category
from . import units


#: Plain-language gloss for each origin, shown in the UI legend.
ORIGIN_LABELS: Dict[str, Dict[str, str]] = {
    Origin.INPUT.value: {
        "label": "From source",
        "meaning": "Supplied verbatim by the source system or user. Treated as truth.",
        "trust": "highest",
    },
    Origin.EXTRACTED.value: {
        "label": "Extracted",
        "meaning": "Matched by a deterministic rule against the source text. Reproducible.",
        "trust": "high",
    },
    Origin.DERIVED.value: {
        "label": "Computed",
        "meaning": "Calculated from other known fields using a physical relationship.",
        "trust": "medium-high",
    },
    Origin.TAXONOMY.value: {
        "label": "Classified",
        "meaning": "Assigned by the scored taxonomy classifier from matched terms.",
        "trust": "medium-high",
    },
    Origin.LLM.value: {
        "label": "Model-inferred",
        "meaning": "Proposed by the local language model to fill a gap. Verify before publishing.",
        "trust": "medium",
    },
    Origin.DEFAULT.value: {
        "label": "Fallback",
        "meaning": "No signal found; a category default was applied.",
        "trust": "low",
    },
}


class ProductExplainer:
    """Builds a structured, per-field explanation from pipeline provenance."""

    def explain(self, product: EnrichedProduct) -> Dict[str, Any]:
        explanation = {
            "summary": self._summary(product),
            "category_reasoning": self._category_reasoning(product),
            "field_provenance": self._field_provenance(product),
            "reasoning_chain": self._reasoning_chain(product),
            "validation_reasoning": self._validation_reasoning(product),
            "trust_breakdown": self._trust_breakdown(product),
            "review_queue": self._review_queue(product),
            "legend": ORIGIN_LABELS,
        }
        product.explanation = explanation
        return explanation

    # -- sections ---------------------------------------------------------

    def _summary(self, product: EnrichedProduct) -> str:
        counts = self._origin_counts(product)
        total = sum(counts.values())
        category = str(product.category.value) if product.category else "no category"
        if total == 0:
            return (
                f"No specifications could be recovered for '{product.raw.name}'. "
                f"It was classified as {category} on name and description alone."
            )

        pieces: List[str] = []
        for origin in (Origin.INPUT, Origin.EXTRACTED, Origin.DERIVED, Origin.LLM):
            n = counts.get(origin.value, 0)
            if n:
                pieces.append(f"{n} {ORIGIN_LABELS[origin.value]['label'].lower()}")

        verified = counts.get(Origin.INPUT.value, 0) + counts.get(Origin.EXTRACTED.value, 0)
        share = round(100 * verified / total) if total else 0

        return (
            f"'{product.raw.name}' was classified as {category} and enriched to "
            f"{total} specification(s): {', '.join(pieces)}. "
            f"{share}% of fields trace directly to the source text rather than model inference."
        )

    def _category_reasoning(self, product: EnrichedProduct) -> Dict[str, Any]:
        if not product.category:
            return {"assigned": None, "why": "No category was assigned."}

        category = get_category(str(product.category.value))
        evidence = [e.to_dict() for e in product.category.evidence]
        matched = next(
            (e.get("source_text") for e in evidence if e.get("rule_id", "").startswith("TAX.")),
            None,
        )

        why: List[str] = []
        if product.category.origin is Origin.INPUT:
            why.append("The source system supplied this category and it matched the taxonomy exactly.")
        else:
            why.append(
                f"The classifier scored every leaf in the taxonomy; "
                f"{product.category.value} won with confidence {product.category.confidence:.2f}."
            )
            if matched:
                why.append(f"Deciding signals: {matched}.")

        supporting_specs = []
        if category:
            supporting_specs = sorted(set(category.expected_specs) & set(product.specifications))
            if supporting_specs:
                why.append(
                    f"{len(supporting_specs)} extracted specification(s) are characteristic of "
                    f"this category: {', '.join(supporting_specs)}."
                )

        if product.category.confidence < 0.45:
            why.append(
                "Confidence is low — the runner-up categories below are close enough that a "
                "human should confirm."
            )

        return {
            "assigned": product.category.value,
            "code": category.code if category else None,
            "path": product.category_path,
            "confidence": product.category.confidence,
            "origin": product.category.origin.value,
            "why": why,
            "supporting_specifications": supporting_specs,
            "alternatives": product.category_alternatives,
            "evidence": evidence,
        }

    def _field_provenance(self, product: EnrichedProduct) -> List[Dict[str, Any]]:
        """One row per specification: value, where it came from, and proof."""
        rows: List[Dict[str, Any]] = []
        for key, fv in sorted(product.specifications.items()):
            rows.append({
                "field": key,
                "display": self._display(fv),
                "value": fv.value,
                "unit": fv.unit,
                "origin": fv.origin.value,
                "origin_label": ORIGIN_LABELS[fv.origin.value]["label"],
                "confidence": fv.confidence,
                "verifiable": fv.origin in (Origin.INPUT, Origin.EXTRACTED),
                "source_text": fv.raw,
                "evidence": [e.to_dict() for e in fv.evidence],
                "why": self._why_field(key, fv),
            })
        return rows

    def _reasoning_chain(self, product: EnrichedProduct) -> List[Dict[str, Any]]:
        """The actual stages the pipeline ran, in order, with what each produced."""
        chain: List[Dict[str, Any]] = []
        for step in product.pipeline_trace:
            stage = step.get("stage")
            detail = {k: v for k, v in step.items() if k != "stage"}
            chain.append({
                "stage": stage,
                "description": _STAGE_DESCRIPTIONS.get(stage, stage),
                "result": detail,
            })
        return chain

    def _validation_reasoning(self, product: EnrichedProduct) -> Dict[str, Any]:
        report = product.validation or {}
        findings = report.get("findings", [])
        if not report:
            return {"status": "not_validated", "explanation": "Validation has not been run."}

        blocking = [f for f in findings if f.get("severity") in ("critical", "major")]
        advisory = [f for f in findings if f.get("severity") in ("minor", "info")]

        if report.get("is_valid"):
            explanation = (
                f"Passed {report.get('checked_rules', 0)} checks with a score of "
                f"{report.get('score')}. "
                + (f"{len(advisory)} advisory note(s) remain." if advisory else "No issues found.")
            )
        else:
            reasons = "; ".join(f.get("message", "") for f in blocking[:3])
            explanation = (
                f"Failed validation on {len(blocking)} blocking issue(s): {reasons}"
            )

        return {
            "status": report.get("status"),
            "score": report.get("score"),
            "consistency_score": report.get("consistency_score"),
            "explanation": explanation,
            "blocking_issues": blocking,
            "advisory_issues": advisory,
        }

    def _trust_breakdown(self, product: EnrichedProduct) -> Dict[str, Any]:
        counts = self._origin_counts(product)
        total = sum(counts.values()) or 1
        verified = counts.get(Origin.INPUT.value, 0) + counts.get(Origin.EXTRACTED.value, 0)
        computed = counts.get(Origin.DERIVED.value, 0)
        inferred = counts.get(Origin.LLM.value, 0)

        return {
            "by_origin": counts,
            "total_fields": sum(counts.values()),
            "verifiable_ratio": round(verified / total, 4),
            "computed_ratio": round(computed / total, 4),
            "model_inferred_ratio": round(inferred / total, 4),
            "mean_confidence": product.mean_confidence(),
            "completeness": product.completeness(),
            "interpretation": (
                "Every field marked 'From source' or 'Extracted' can be traced to an exact "
                "span of the input text. Model-inferred fields are the only ones that could "
                "be wrong without the source being wrong."
            ),
        }

    def _review_queue(self, product: EnrichedProduct) -> List[Dict[str, Any]]:
        """What a human should actually look at, ranked. Drives the UI's action list."""
        queue: List[Dict[str, Any]] = []

        for key, fv in product.specifications.items():
            if fv.origin is Origin.LLM:
                queue.append({
                    "priority": 2 if fv.confidence < 0.6 else 3,
                    "field": key,
                    "reason": "Inferred by the model with no supporting text in the source.",
                    "action": f"Confirm {key.replace('_', ' ')} against the datasheet.",
                })

        for finding in (product.validation or {}).get("findings", []):
            severity = finding.get("severity")
            if severity in ("critical", "major"):
                queue.append({
                    "priority": 1,
                    "field": finding.get("field"),
                    "reason": finding.get("message"),
                    "action": finding.get("suggestion") or "Correct the source data and re-run.",
                })

        if product.category and product.category.confidence < 0.45:
            alternatives = ", ".join(
                a["category"] for a in product.category_alternatives[:2]
            ) or "none"
            queue.append({
                "priority": 1,
                "field": "category",
                "reason": (
                    f"Category confidence is {product.category.confidence:.2f}; the classifier "
                    "did not find a clear winner."
                ),
                "action": f"Confirm the category. Close alternatives: {alternatives}.",
            })

        queue.sort(key=lambda item: (item["priority"], str(item.get("field"))))
        return queue

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _origin_counts(product: EnrichedProduct) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for fv in product.specifications.values():
            counts[fv.origin.value] = counts.get(fv.origin.value, 0) + 1
        return counts

    @staticmethod
    def _display(fv: FieldValue) -> str:
        if isinstance(fv.value, list):
            return ", ".join(str(v) for v in fv.value)
        if fv.unit and isinstance(fv.value, (int, float)) and not isinstance(fv.value, bool):
            return units.format_value(float(fv.value), fv.unit)
        return str(fv.value)

    @staticmethod
    def _why_field(key: str, fv: FieldValue) -> str:
        label = key.replace("_", " ")
        if fv.origin is Origin.INPUT:
            return f"The source record already contained {label}; it was normalised, not changed."
        if fv.origin is Origin.EXTRACTED:
            span = next((e for e in fv.evidence if e.kind == "text_span"), None)
            if span and span.source_text:
                return (
                    f"Found in the source text near \"{span.source_text.strip()}\" "
                    f"by rule {span.rule_id}."
                )
            return f"Matched against a controlled vocabulary by rule "  \
                   f"{fv.evidence[0].rule_id if fv.evidence else 'unknown'}."
        if fv.origin is Origin.DERIVED:
            detail = fv.evidence[0].detail if fv.evidence else ""
            return f"Not stated in the source; {detail[0].lower() + detail[1:] if detail else 'computed from other fields.'}"
        if fv.origin is Origin.LLM:
            detail = fv.evidence[0].detail if fv.evidence else ""
            return f"Not present in the source text. Model's stated basis: {detail}"
        return "Applied as a category default because no signal was found."


_STAGE_DESCRIPTIONS: Dict[str, str] = {
    "extract": "Ran deterministic pattern and vocabulary rules over the source text.",
    "classify": "Scored the product against every leaf in the taxonomy.",
    "derive": "Computed additional fields from physical relationships between known values.",
    "llm": "Asked the local model to fill only the fields the rules could not.",
    "complete": "Pipeline finished.",
    "error": "The pipeline failed for this record.",
}
