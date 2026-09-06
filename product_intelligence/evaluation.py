"""
Evaluation harness.

"Improve accuracy and consistency of product data" is only a claim until it is
measured, so this module scores the pipeline against a labelled ground-truth
set and reports numbers that can be quoted, regressed against, and defended.

Metrics
-------
* **Category accuracy** and **top-3 accuracy** — did the classifier pick the
  right leaf, and was the right answer at least in the shortlist?
* **Specification precision / recall / F1** — per field, then micro-averaged.
  A value counts as correct only if it matches the expected value *after* unit
  normalisation, within tolerance. Getting 2.2 kW from "3 HP" is a hit; getting
  it from nowhere is a hallucination and is counted as a false positive.
* **Hallucination rate** — the share of emitted specs that are not in the
  ground truth at all. This is the number that keeps the LLM honest.
* **Completeness lift** — fields per record before vs after enrichment, which
  is the headline "limited input → rich output" claim from the brief.

Run it from the CLI:

    python -m app.services.product_intelligence.evaluation \\
        --dataset data/eval/ground_truth.jsonl --no-llm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import EnrichedProduct, Origin, RawProduct
from .enricher import ProductEnricher
from .validator import ProductValidator, CatalogConsistencyChecker
from . import units


#: Relative tolerance when comparing numeric specs after normalisation.
NUMERIC_TOLERANCE = 0.05


@dataclass
class GroundTruthCase:
    """One labelled example."""

    name: str
    description: str = ""
    category: str = ""
    expected_category: str = ""
    expected_specs: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GroundTruthCase":
        return cls(
            name=payload["name"],
            description=payload.get("description", ""),
            category=payload.get("input_category", ""),
            expected_category=payload.get("expected_category", ""),
            expected_specs=payload.get("expected_specs", {}) or {},
            attributes=payload.get("attributes", {}) or {},
            notes=payload.get("notes", ""),
        )

    def to_raw(self) -> RawProduct:
        return RawProduct(
            name=self.name,
            description=self.description,
            category=self.category,
            source="eval",
            source_ref=f"eval:{self.name}",
            attributes=self.attributes,
        )


@dataclass
class CaseResult:
    name: str
    expected_category: str
    predicted_category: str
    category_correct: bool
    category_in_top3: bool
    category_confidence: float
    true_positives: List[str]
    false_positives: List[str]
    false_negatives: List[str]
    value_mismatches: List[Dict[str, Any]]
    input_field_count: int
    output_field_count: int
    validation_score: float
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expected_category": self.expected_category,
            "predicted_category": self.predicted_category,
            "category_correct": self.category_correct,
            "category_in_top3": self.category_in_top3,
            "category_confidence": self.category_confidence,
            "spec_hits": self.true_positives,
            "spec_hallucinated": self.false_positives,
            "spec_missed": self.false_negatives,
            "value_mismatches": self.value_mismatches,
            "fields_in": self.input_field_count,
            "fields_out": self.output_field_count,
            "validation_score": self.validation_score,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class Evaluator:
    """Scores the enrichment pipeline against a labelled dataset."""

    def __init__(
        self,
        enricher: Optional[ProductEnricher] = None,
        tolerance: float = NUMERIC_TOLERANCE,
    ) -> None:
        self.enricher = enricher or ProductEnricher()
        self.validator = ProductValidator()
        self.tolerance = tolerance

    # -- public API -------------------------------------------------------

    async def run(
        self, cases: List[GroundTruthCase], use_llm: bool = False, concurrency: int = 4
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        raws = [case.to_raw() for case in cases]
        enriched = await self.enricher.enrich_many(raws, use_llm=use_llm, concurrency=concurrency)

        results: List[CaseResult] = []
        for case, product in zip(cases, enriched):
            case_started = time.perf_counter()
            self.validator.validate(product)
            results.append(
                self._score_case(case, product, (time.perf_counter() - case_started) * 1000)
            )

        catalog = CatalogConsistencyChecker().check(enriched)
        total_seconds = time.perf_counter() - started

        return {
            "summary": self._aggregate(results, catalog, total_seconds, use_llm),
            "per_case": [r.to_dict() for r in results],
            "catalog_consistency": {
                "score": catalog["catalog_consistency_score"],
                "duplicate_groups": len(catalog["duplicate_groups"]),
                "outliers": len(catalog["outliers"]),
                "unit_conflicts": len(catalog["unit_conflicts"]),
            },
        }

    # -- scoring ----------------------------------------------------------

    def _score_case(
        self, case: GroundTruthCase, product: EnrichedProduct, elapsed_ms: float
    ) -> CaseResult:
        predicted = str(product.category.value) if product.category else ""
        alternatives = [alt["category"] for alt in product.category_alternatives[:2]]
        correct = _same_category(predicted, case.expected_category)
        in_top3 = correct or any(
            _same_category(alt, case.expected_category) for alt in alternatives
        )

        true_positives: List[str] = []
        false_negatives: List[str] = []
        value_mismatches: List[Dict[str, Any]] = []

        for key, expected in case.expected_specs.items():
            actual = product.specifications.get(key)
            if actual is None:
                false_negatives.append(key)
                continue
            matched, detail = self._values_match(expected, actual.value, actual.unit)
            if matched:
                true_positives.append(key)
            else:
                false_negatives.append(key)
                value_mismatches.append({
                    "field": key,
                    "expected": expected,
                    "got": detail,
                    "origin": actual.origin.value,
                })

        # A spec we emitted that the ground truth does not mention at all.
        # Derived fields are excluded: they are computed, not claimed as read
        # from the source, and the dataset does not label them.
        false_positives = [
            key for key, fv in product.specifications.items()
            if key not in case.expected_specs and fv.origin in (Origin.LLM,)
        ]

        return CaseResult(
            name=case.name,
            expected_category=case.expected_category,
            predicted_category=predicted,
            category_correct=correct,
            category_in_top3=in_top3,
            category_confidence=product.category.confidence if product.category else 0.0,
            true_positives=sorted(true_positives),
            false_positives=sorted(false_positives),
            false_negatives=sorted(false_negatives),
            value_mismatches=value_mismatches,
            input_field_count=_input_field_count(case),
            output_field_count=len(product.specifications),
            validation_score=product.validation.get("score", 0.0),
            elapsed_ms=elapsed_ms,
        )

    def _values_match(
        self, expected: Any, actual: Any, actual_unit: Optional[str]
    ) -> Tuple[bool, Any]:
        """
        Compare an expected value against what the pipeline produced.

        Expected values may be written as "2.2 kW" or as a bare number in the
        canonical unit; both are accepted, because the point of the check is
        the physical quantity, not the notation.
        """
        display = (
            units.format_value(float(actual), actual_unit)
            if actual_unit and isinstance(actual, (int, float))
            else actual
        )

        if isinstance(expected, str):
            parsed = _parse_measurement(expected)
            if parsed is not None:
                number, unit = parsed
                try:
                    canonical, _, _ = units.to_canonical(number, unit)
                except units.UnitError:
                    canonical = number
                if isinstance(actual, (int, float)):
                    return _close(canonical, float(actual), self.tolerance), display
                return False, display
            # Non-numeric string: compare case- and space-insensitively.
            if isinstance(actual, list):
                return (
                    expected.strip().lower() in [str(a).strip().lower() for a in actual],
                    display,
                )
            return str(actual).strip().lower() == expected.strip().lower(), display

        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return _close(float(expected), float(actual), self.tolerance), display

        if isinstance(expected, list):
            actual_list = actual if isinstance(actual, list) else [actual]
            expected_set = {str(e).strip().lower() for e in expected}
            actual_set = {str(a).strip().lower() for a in actual_list}
            return expected_set.issubset(actual_set), display

        return str(expected).strip().lower() == str(actual).strip().lower(), display

    def _aggregate(
        self,
        results: List[CaseResult],
        catalog: Dict[str, Any],
        total_seconds: float,
        use_llm: bool,
    ) -> Dict[str, Any]:
        n = len(results) or 1

        tp = sum(len(r.true_positives) for r in results)
        fp = sum(len(r.false_positives) for r in results)
        fn = sum(len(r.false_negatives) for r in results)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        emitted = sum(r.output_field_count for r in results)
        fields_in = sum(r.input_field_count for r in results)

        return {
            "cases": len(results),
            "mode": "llm_assisted" if use_llm else "deterministic_only",
            "category_accuracy": round(sum(r.category_correct for r in results) / n, 4),
            "category_top3_accuracy": round(sum(r.category_in_top3 for r in results) / n, 4),
            "mean_category_confidence": round(
                sum(r.category_confidence for r in results) / n, 4
            ),
            "spec_precision": round(precision, 4),
            "spec_recall": round(recall, 4),
            "spec_f1": round(f1, 4),
            "spec_true_positives": tp,
            "spec_false_positives": fp,
            "spec_false_negatives": fn,
            "hallucination_rate": round(fp / emitted, 4) if emitted else 0.0,
            "completeness": {
                "mean_fields_in": round(fields_in / n, 2),
                "mean_fields_out": round(emitted / n, 2),
                "lift_multiple": round(emitted / fields_in, 2) if fields_in else None,
            },
            "mean_validation_score": round(
                sum(r.validation_score for r in results) / n, 4
            ),
            "catalog_consistency_score": catalog["catalog_consistency_score"],
            "throughput": {
                "total_seconds": round(total_seconds, 3),
                "records_per_second": round(len(results) / total_seconds, 2)
                if total_seconds > 0 else None,
                "mean_ms_per_record": round(1000 * total_seconds / n, 1),
            },
        }


# =========================================================================
# dataset loading + CLI
# =========================================================================


def load_dataset(path: str | Path) -> List[GroundTruthCase]:
    """Load a .jsonl (one case per line) or .json (array) ground-truth file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Ground-truth dataset not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    cases: List[GroundTruthCase] = []

    if file_path.suffix == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                cases.append(GroundTruthCase.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{file_path}:{line_no} is malformed: {exc}") from exc
    else:
        payload = json.loads(text)
        records = payload if isinstance(payload, list) else payload.get("cases", [])
        cases = [GroundTruthCase.from_dict(record) for record in records]

    if not cases:
        raise ValueError(f"No cases found in {file_path}")
    return cases


def format_report(report: Dict[str, Any]) -> str:
    """Human-readable summary for the terminal and the demo."""
    s = report["summary"]
    lines = [
        "",
        "=" * 66,
        "  PRODUCT INTELLIGENCE — EVALUATION REPORT",
        "=" * 66,
        f"  Cases evaluated       : {s['cases']}",
        f"  Mode                  : {s['mode']}",
        "",
        "  CLASSIFICATION",
        f"    Category accuracy   : {s['category_accuracy']:.1%}",
        f"    Top-3 accuracy      : {s['category_top3_accuracy']:.1%}",
        f"    Mean confidence     : {s['mean_category_confidence']:.3f}",
        "",
        "  SPECIFICATION EXTRACTION",
        f"    Precision           : {s['spec_precision']:.1%}",
        f"    Recall              : {s['spec_recall']:.1%}",
        f"    F1                  : {s['spec_f1']:.1%}",
        f"    Hallucination rate  : {s['hallucination_rate']:.1%}",
        f"    TP / FP / FN        : {s['spec_true_positives']} / "
        f"{s['spec_false_positives']} / {s['spec_false_negatives']}",
        "",
        "  ENRICHMENT LIFT",
        f"    Fields in  (mean)   : {s['completeness']['mean_fields_in']}",
        f"    Fields out (mean)   : {s['completeness']['mean_fields_out']}",
        f"    Lift                : {s['completeness']['lift_multiple']}x",
        "",
        "  QUALITY",
        f"    Mean validation     : {s['mean_validation_score']:.3f}",
        f"    Catalog consistency : {s['catalog_consistency_score']:.3f}",
        "",
        "  THROUGHPUT",
        f"    Records/second      : {s['throughput']['records_per_second']}",
        f"    Mean ms/record      : {s['throughput']['mean_ms_per_record']}",
        "=" * 66,
    ]

    failures = [c for c in report["per_case"] if not c["category_correct"]]
    if failures:
        lines.append(f"  MISCLASSIFIED ({len(failures)}):")
        for case in failures[:10]:
            lines.append(
                f"    - {case['name'][:44]:<44} expected {case['expected_category']}, "
                f"got {case['predicted_category']}"
            )
        lines.append("=" * 66)

    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> int:
    cases = load_dataset(args.dataset)

    llm_router = None
    if not args.no_llm:
        try:
            from app.services.llm.router import LLMRouter
            llm_router = LLMRouter()
        except Exception as exc:
            print(f"[warn] LLM router unavailable ({exc}); running deterministic-only.")

    evaluator = Evaluator(enricher=ProductEnricher(llm_router=llm_router, model=args.model))
    report = await evaluator.run(cases, use_llm=not args.no_llm, concurrency=args.concurrency)

    print(format_report(report))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.output}")

    threshold_failures = []
    if report["summary"]["category_accuracy"] < args.min_accuracy:
        threshold_failures.append(
            f"category accuracy {report['summary']['category_accuracy']:.3f} "
            f"< {args.min_accuracy}"
        )
    if report["summary"]["spec_f1"] < args.min_f1:
        threshold_failures.append(
            f"spec F1 {report['summary']['spec_f1']:.3f} < {args.min_f1}"
        )

    if threshold_failures:
        print("\nFAILED thresholds: " + "; ".join(threshold_failures))
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the product intelligence pipeline against ground truth."
    )
    parser.add_argument("--dataset", default="data/eval/ground_truth.jsonl")
    parser.add_argument("--no-llm", action="store_true",
                        help="Deterministic layer only (fast, no model server needed).")
    parser.add_argument("--model", default=None, help="Model name to use when the LLM is enabled.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", default=None, help="Write the full JSON report here.")
    parser.add_argument("--min-accuracy", type=float, default=0.0,
                        help="Exit non-zero below this category accuracy (for CI).")
    parser.add_argument("--min-f1", type=float, default=0.0,
                        help="Exit non-zero below this spec F1 (for CI).")
    return asyncio.run(_main_async(parser.parse_args()))


# -- helpers ---------------------------------------------------------------


def _close(expected: float, actual: float, tolerance: float) -> bool:
    if expected == actual:
        return True
    scale = max(abs(expected), abs(actual))
    return scale > 0 and abs(expected - actual) / scale <= tolerance


def _parse_measurement(text: str) -> Optional[Tuple[float, str]]:
    """Split '2.2 kW' into (2.2, 'kW'). Returns None if it is not a measurement."""
    import re

    match = re.match(r"^\s*(-?\d[\d,]*(?:\.\d+)?)\s*([A-Za-z°/³\"%]+.*?)\s*$", text)
    if not match:
        return None
    try:
        number = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = match.group(2).strip()
    return (number, unit) if units.resolve(unit) else None


def _same_category(a: str, b: str) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()


def _input_field_count(case: GroundTruthCase) -> int:
    """How many fields the source actually gave us, before enrichment."""
    count = len(case.attributes)
    if case.description:
        count += 1
    if case.category:
        count += 1
    return max(count, 1)


if __name__ == "__main__":
    raise SystemExit(main())
