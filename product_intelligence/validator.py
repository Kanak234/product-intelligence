"""
Validation and consistency engine.

Three classes of check, all deterministic:

* **Completeness** — does the record carry the specs its category requires?
* **Plausibility** — is each value inside the physically sensible range for
  that dimension and category? (A 900 kW desk fan is a data-entry error.)
* **Coherence** — do the fields agree *with each other*? Electrical power,
  voltage, current and power factor are not independent; if a record claims
  2.2 kW at 415 V three-phase but 40 A, one of those numbers is wrong. This is
  the check that catches errors no single-field validator can see.

Every finding carries a severity, the rule that produced it, and the observed
vs expected values, so the UI can show *why* something failed rather than a
bare score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from .schema import EnrichedProduct, FieldValue
from .taxonomy import Category, get_category
from . import units


class Severity:
    CRITICAL = "critical"   # record is unusable as-is
    MAJOR = "major"         # likely wrong, needs review
    MINOR = "minor"         # suspicious or incomplete
    INFO = "info"           # advisory only


#: Penalty applied to the 0-1 validation score per finding.
SEVERITY_WEIGHT: Dict[str, float] = {
    Severity.CRITICAL: 0.40,
    Severity.MAJOR: 0.18,
    Severity.MINOR: 0.06,
    Severity.INFO: 0.0,
}


@dataclass
class Finding:
    rule_id: str
    severity: str
    field: str
    message: str
    observed: Any = None
    expected: Any = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


#: Physically sensible ranges in canonical units, applied to every product.
#: Deliberately wide — this catches unit-entry blunders and typos, not
#: unusual-but-real products.
GLOBAL_RANGES: Dict[str, Tuple[float, float]] = {
    "power": (0.1, 5_000_000.0),           # W
    "voltage": (1.0, 400_000.0),           # V
    "current": (0.001, 10_000.0),          # A
    "frequency": (1.0, 100_000.0),         # Hz
    "rotational_speed": (1.0, 200_000.0),  # rpm
    "torque": (0.001, 500_000.0),          # Nm
    "weight": (0.001, 500_000.0),          # kg
    "pressure": (0.001, 10_000.0),         # bar
    "flow_rate": (0.0001, 100_000.0),      # m3/h
    "max_temperature": (-273.15, 3_000.0), # C
    "length": (0.1, 100_000.0),            # mm
    "diameter": (0.1, 50_000.0),           # mm
    "width": (0.1, 50_000.0),              # mm
    "height": (0.1, 100_000.0),            # mm
    "volume": (0.001, 1_000_000.0),        # L
    "noise_level": (0.0, 200.0),           # dB
    "efficiency": (0.0, 100.0),            # %
}

#: Tighter, category-specific ranges layered on top of the global ones.
CATEGORY_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "MOT.IND": {
        "power": (30.0, 15_000_000.0),
        "voltage": (12.0, 15_000.0),
        "rotational_speed": (100.0, 30_000.0),
    },
    "MOT.SRV": {
        "power": (5.0, 100_000.0),
        "torque": (0.01, 5_000.0),
        "rotational_speed": (10.0, 60_000.0),
    },
    "PMP.CENT": {
        "flow_rate": (0.01, 50_000.0),
        "pressure": (0.1, 500.0),
        "rotational_speed": (100.0, 10_000.0),
    },
    "PMP.SUB": {"flow_rate": (0.01, 20_000.0), "pressure": (0.1, 400.0)},
    "AIR.CMP": {"pressure": (0.5, 500.0), "power": (100.0, 2_000_000.0)},
    "VLV.BALL": {"pressure": (0.1, 700.0), "max_temperature": (-200.0, 800.0)},
    "VLV.GATE": {"pressure": (0.1, 700.0), "max_temperature": (-200.0, 800.0)},
    "ELC.SWG": {"voltage": (12.0, 40_000.0), "current": (0.1, 6_300.0)},
    "INS.PRS": {"pressure": (0.001, 5_000.0)},
    "BRG.ROLL": {"diameter": (1.0, 3_000.0)},
}

#: Standard mains voltages. A voltage far from any of these on a mains-powered
#: product is usually a typo (e.g. 41.5 V instead of 415 V).
STANDARD_VOLTAGES = (
    12, 24, 36, 48, 110, 120, 208, 220, 230, 240, 380, 400, 415, 440, 480,
    525, 600, 660, 690, 1000, 3300, 6600, 11000, 33000,
)

#: Typical synchronous speeds at 50/60 Hz for 2/4/6/8/10-pole machines.
SYNCHRONOUS_SPEEDS = (3600, 3000, 1800, 1500, 1200, 1000, 900, 750, 720, 600)


class ProductValidator:
    """Validates an `EnrichedProduct` and produces a scored, explained report."""

    def __init__(self, strict: bool = False) -> None:
        #: In strict mode, MINOR findings also block the "valid" status.
        self.strict = strict

    # -- public API -------------------------------------------------------

    def validate(self, product: EnrichedProduct) -> Dict[str, Any]:
        category = self._resolve_category(product)
        findings: List[Finding] = []

        findings += self._check_identity(product)
        findings += self._check_required_specs(product, category)
        findings += self._check_ranges(product, category)
        findings += self._check_electrical_coherence(product)
        findings += self._check_mechanical_coherence(product)
        findings += self._check_dimensional_coherence(product)
        findings += self._check_confidence(product)

        score = self._score(findings)
        blocking = {Severity.CRITICAL, Severity.MAJOR}
        if self.strict:
            blocking = blocking | {Severity.MINOR}
        is_valid = not any(f.severity in blocking for f in findings)

        report = {
            "is_valid": is_valid,
            "status": "valid" if is_valid else "invalid",
            "score": score,
            "consistency_score": self._consistency_score(findings),
            "completeness": product.completeness(),
            "findings": [f.to_dict() for f in findings],
            "counts": {
                sev: sum(1 for f in findings if f.severity == sev)
                for sev in (Severity.CRITICAL, Severity.MAJOR, Severity.MINOR, Severity.INFO)
            },
            "checked_rules": self._rules_run(category),
        }
        product.validation = report
        return report

    # -- individual check groups -----------------------------------------

    def _check_identity(self, product: EnrichedProduct) -> List[Finding]:
        out: List[Finding] = []
        name = (product.raw.name or "").strip()
        if not name:
            out.append(Finding("VAL.ID.NAME_MISSING", Severity.CRITICAL, "name",
                               "Product name is empty; the record cannot be catalogued."))
        elif len(name) < 3:
            out.append(Finding(
                "VAL.ID.NAME_SHORT", Severity.MAJOR, "name",
                "Product name is too short to identify the item.",
                observed=name,
                suggestion="Include the product type and model, e.g. 'Centrifugal Pump CP-200'.",
            ))
        if not product.category or not product.category.value:
            out.append(Finding("VAL.ID.NO_CATEGORY", Severity.MAJOR, "category",
                               "No category could be assigned."))
        elif product.category.confidence < 0.35:
            out.append(Finding(
                "VAL.ID.WEAK_CATEGORY", Severity.MINOR, "category",
                "Category assignment has low confidence.",
                observed=round(product.category.confidence, 3),
                expected=">= 0.35",
                suggestion="Add product type keywords to the name or description.",
            ))
        return out

    def _check_required_specs(
        self, product: EnrichedProduct, category: Optional[Category]
    ) -> List[Finding]:
        if not category:
            return []
        out: List[Finding] = []
        for spec in category.required_specs:
            if spec not in product.specifications:
                out.append(Finding(
                    f"VAL.REQ.{spec.upper()}", Severity.MAJOR, spec,
                    f"{category.name} records require '{spec}' but it is missing.",
                    expected=f"'{spec}' present",
                    suggestion=f"Add {spec.replace('_', ' ')} to the source data or datasheet.",
                ))
        missing_expected = [
            s for s in category.expected_specs
            if s not in category.required_specs and s not in product.specifications
        ]
        if missing_expected:
            out.append(Finding(
                "VAL.REQ.EXPECTED_GAPS", Severity.MINOR, "specifications",
                f"{len(missing_expected)} specification(s) typical for this category are absent.",
                observed=missing_expected,
                suggestion="Enrichment can infer these, but a datasheet gives higher confidence.",
            ))
        return out

    def _check_ranges(
        self, product: EnrichedProduct, category: Optional[Category]
    ) -> List[Finding]:
        out: List[Finding] = []
        cat_ranges = CATEGORY_RANGES.get(category.code, {}) if category else {}

        for key, fv in product.specifications.items():
            if not isinstance(fv.value, (int, float)) or isinstance(fv.value, bool):
                continue
            value = float(fv.value)

            if value < 0 and key != "max_temperature":
                out.append(Finding(
                    f"VAL.RANGE.NEGATIVE.{key.upper()}", Severity.CRITICAL, key,
                    f"Negative {key.replace('_', ' ')} is physically impossible.",
                    observed=units.format_value(value, fv.unit or ""),
                    expected="> 0",
                ))
                continue

            lo, hi = cat_ranges.get(key, GLOBAL_RANGES.get(key, (None, None)))
            if lo is None:
                continue

            if value < lo or value > hi:
                severity = Severity.MAJOR if key in cat_ranges else Severity.MINOR
                out.append(Finding(
                    f"VAL.RANGE.{key.upper()}", severity, key,
                    f"{key.replace('_', ' ').title()} is outside the plausible range"
                    + (f" for {category.name}." if key in cat_ranges and category else "."),
                    observed=units.format_value(value, fv.unit or ""),
                    expected=f"{lo:g} - {hi:g} {fv.unit or ''}".strip(),
                    suggestion=self._unit_slip_hint(value, lo, hi, fv),
                ))

        volt = product.specifications.get("voltage")
        if volt and isinstance(volt.value, (int, float)):
            v = float(volt.value)
            if v > 60 and not any(abs(v - std) / std <= 0.12 for std in STANDARD_VOLTAGES):
                nearest = min(STANDARD_VOLTAGES, key=lambda s: abs(s - v))
                out.append(Finding(
                    "VAL.RANGE.NONSTANDARD_VOLTAGE", Severity.MINOR, "voltage",
                    "Voltage does not match any standard supply voltage.",
                    observed=f"{v:g} V", expected=f"nearest standard is {nearest} V",
                    suggestion=f"Verify against the nameplate; {nearest} V is the closest standard.",
                ))
        return out

    def _check_electrical_coherence(self, product: EnrichedProduct) -> List[Finding]:
        """P = V*I*sqrt(3)*pf for three-phase, P = V*I*pf for single-phase."""
        out: List[Finding] = []
        specs = product.specifications
        power, voltage, current = specs.get("power"), specs.get("voltage"), specs.get("current")

        if not (power and voltage and current):
            return out
        if not all(isinstance(f.value, (int, float)) for f in (power, voltage, current)):
            return out

        p, v, i = float(power.value), float(voltage.value), float(current.value)
        if v <= 0 or i <= 0:
            return out

        phase = specs.get("phase")
        is_three_phase = bool(phase and str(phase.value).startswith("3"))
        root3 = math.sqrt(3) if is_three_phase else 1.0
        phase_label = "three-phase" if is_three_phase else "single-phase"

        implied_pf = p / (v * i * root3)

        if implied_pf > 1.05:
            out.append(Finding(
                "VAL.COH.POWER_TOO_HIGH", Severity.MAJOR, "power",
                f"Rated power exceeds what {phase_label} {v:g} V x {i:g} A can deliver "
                f"(implied power factor {implied_pf:.2f} > 1).",
                observed=f"{p:g} W at {v:g} V, {i:g} A",
                expected=f"<= {v * i * root3:.0f} W",
                suggestion="Check whether the power figure is input power or the current is per-phase.",
            ))
        elif implied_pf < 0.35:
            out.append(Finding(
                "VAL.COH.POWER_TOO_LOW", Severity.MINOR, "current",
                f"Implied power factor of {implied_pf:.2f} is unusually low for {phase_label} equipment.",
                observed=f"{p:g} W at {v:g} V, {i:g} A",
                expected="power factor 0.5 - 1.0",
                suggestion="Current may include inrush or be quoted for a different voltage.",
            ))

        freq, speed = specs.get("frequency"), specs.get("rotational_speed")
        if (
            freq and speed
            and isinstance(freq.value, (int, float))
            and isinstance(speed.value, (int, float))
        ):
            f_hz, n_rpm = float(freq.value), float(speed.value)
            if f_hz in (50.0, 60.0) and n_rpm > 0:
                poles = (120.0 * f_hz) / n_rpm
                nearest_sync = min(SYNCHRONOUS_SPEEDS, key=lambda s: abs(s - n_rpm))
                # Induction machines run just below synchronous speed (slip),
                # so allow ~10% below but very little above.
                if not (0.90 * nearest_sync <= n_rpm <= 1.02 * nearest_sync):
                    out.append(Finding(
                        "VAL.COH.SPEED_VS_FREQ", Severity.MINOR, "rotational_speed",
                        f"Speed is not consistent with any standard pole count at {f_hz:g} Hz "
                        f"(implies {poles:.1f} poles).",
                        observed=f"{n_rpm:g} rpm at {f_hz:g} Hz",
                        expected=f"near {nearest_sync} rpm",
                        suggestion="Verify the speed, or state that the drive is inverter-fed.",
                    ))
        return out

    def _check_mechanical_coherence(self, product: EnrichedProduct) -> List[Finding]:
        """P = T*omega. Power, torque and speed cannot be chosen independently."""
        out: List[Finding] = []
        specs = product.specifications
        power, torque, speed = specs.get("power"), specs.get("torque"), specs.get("rotational_speed")
        if not (power and torque and speed):
            return out
        if not all(isinstance(f.value, (int, float)) for f in (power, torque, speed)):
            return out

        p, t, n = float(power.value), float(torque.value), float(speed.value)
        if n <= 0 or t <= 0:
            return out

        implied_power = t * (2 * math.pi * n / 60.0)   # W
        if implied_power <= 0:
            return out
        ratio = p / implied_power

        if ratio > 1.35 or ratio < 0.65:
            out.append(Finding(
                "VAL.COH.POWER_TORQUE_SPEED", Severity.MAJOR, "torque",
                "Power, torque and speed are mutually inconsistent (P = T x 2*pi*n/60).",
                observed=f"{p:g} W stated vs {implied_power:.0f} W implied by {t:g} Nm at {n:g} rpm",
                expected=f"within +/-35% of {implied_power:.0f} W",
                suggestion="Torque may be a starting/peak figure rather than the rated value.",
            ))
        return out

    def _check_dimensional_coherence(self, product: EnrichedProduct) -> List[Finding]:
        out: List[Finding] = []
        specs = product.specifications

        dia, width = specs.get("diameter"), specs.get("width")
        if (
            dia and width
            and isinstance(dia.value, (int, float))
            and isinstance(width.value, (int, float))
            and float(width.value) > float(dia.value) * 6
        ):
            out.append(Finding(
                "VAL.COH.ASPECT_RATIO", Severity.MINOR, "width",
                "Width is disproportionately large compared with diameter.",
                observed=f"width {width.value:g} mm vs diameter {dia.value:g} mm",
                suggestion="Check whether the two values use the same unit.",
            ))

        # Weight vs bounding volume: implied density outside 0.02-25 g/cm3
        # means a unit error somewhere.
        weight = specs.get("weight")
        dims = [specs.get(k) for k in ("length", "width", "height")]
        if weight and all(d is not None for d in dims):
            try:
                volume_mm3 = 1.0
                for d in dims:
                    volume_mm3 *= float(d.value)
                if volume_mm3 > 0:
                    density = float(weight.value) * 1e6 / volume_mm3   # g/cm3
                    if density > 25 or density < 0.02:
                        out.append(Finding(
                            "VAL.COH.DENSITY", Severity.MINOR, "weight",
                            f"Implied density of {density:.2f} g/cm3 is outside the range of "
                            "any common industrial material.",
                            observed=f"{weight.value:g} kg over the stated envelope",
                            expected="0.02 - 25 g/cm3",
                            suggestion="One of weight or the dimensions is likely in the wrong unit.",
                        ))
            except (TypeError, ValueError):
                pass
        return out

    def _check_confidence(self, product: EnrichedProduct) -> List[Finding]:
        out: List[Finding] = []
        llm_only = [k for k, fv in product.specifications.items() if fv.origin.value == "llm"]
        if llm_only:
            out.append(Finding(
                "VAL.CONF.LLM_INFERRED", Severity.INFO, "specifications",
                f"{len(llm_only)} specification(s) were inferred by the model rather than "
                "read from the source text.",
                observed=sorted(llm_only),
                suggestion="Confirm against a datasheet before publishing to the catalogue.",
            ))
        weak = [k for k, fv in product.specifications.items() if fv.confidence < 0.5]
        if weak:
            out.append(Finding(
                "VAL.CONF.LOW", Severity.MINOR, "specifications",
                f"{len(weak)} specification(s) carry confidence below 0.5.",
                observed=sorted(weak),
            ))
        return out

    # -- scoring -----------------------------------------------------------

    @staticmethod
    def _score(findings: List[Finding]) -> float:
        score = 1.0
        for finding in findings:
            score -= SEVERITY_WEIGHT.get(finding.severity, 0.0)
        return round(max(0.0, min(1.0, score)), 4)

    @staticmethod
    def _consistency_score(findings: List[Finding]) -> float:
        """Score restricted to cross-field coherence rules only."""
        score = 1.0
        for finding in findings:
            if finding.rule_id.startswith("VAL.COH."):
                score -= SEVERITY_WEIGHT.get(finding.severity, 0.0)
        return round(max(0.0, min(1.0, score)), 4)

    @staticmethod
    def _unit_slip_hint(value: float, lo: float, hi: float, fv: FieldValue) -> Optional[str]:
        """If x1000 or /1000 lands the value in range, say so explicitly."""
        for factor, label in ((1000.0, "x1000"), (0.001, "/1000"), (0.01, "/100"), (100.0, "x100")):
            if lo <= value * factor <= hi:
                return (
                    f"Value would be plausible if scaled {label} - check the source unit "
                    f"(currently {fv.unit or 'unitless'})."
                )
        return None

    @staticmethod
    def _rules_run(category: Optional[Category]) -> int:
        base = 14
        if category:
            base += len(category.required_specs) + len(CATEGORY_RANGES.get(category.code, {}))
        return base

    @staticmethod
    def _resolve_category(product: EnrichedProduct) -> Optional[Category]:
        if not product.category or not product.category.value:
            return None
        return get_category(str(product.category.value))


class CatalogConsistencyChecker:
    """
    Cross-record checks over a whole catalogue.

    Single-record validation cannot see duplicates, or a spec that is an
    outlier only relative to its peers. This runs across the batch and is what
    makes the engine useful at catalogue scale rather than one product at a
    time.
    """

    def __init__(self, z_threshold: float = 3.0) -> None:
        self.z_threshold = z_threshold

    def check(self, products: List[EnrichedProduct]) -> Dict[str, Any]:
        duplicates = self._find_duplicates(products)
        outliers = self._find_outliers(products)
        unit_conflicts = self._find_unit_conflicts(products)

        total_issues = len(duplicates) + len(outliers) + len(unit_conflicts)
        denominator = max(len(products), 1)

        return {
            "record_count": len(products),
            "duplicate_groups": duplicates,
            "outliers": outliers,
            "unit_conflicts": unit_conflicts,
            "catalog_consistency_score": round(max(0.0, 1.0 - total_issues / denominator), 4),
        }

    @staticmethod
    def _find_duplicates(products: List[EnrichedProduct]) -> List[Dict[str, Any]]:
        buckets: Dict[str, List[str]] = {}
        for product in products:
            buckets.setdefault(product.raw.fingerprint(), []).append(product.raw.name)
        return [
            {"fingerprint": fp, "names": names, "count": len(names)}
            for fp, names in buckets.items()
            if len(names) > 1
        ]

    def _find_outliers(self, products: List[EnrichedProduct]) -> List[Dict[str, Any]]:
        """Per-category z-score outliers on numeric specs."""
        grouped: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        for product in products:
            cat = str(product.category.value) if product.category else "Uncategorised"
            for key, fv in product.specifications.items():
                if isinstance(fv.value, (int, float)) and not isinstance(fv.value, bool):
                    grouped.setdefault((cat, key), []).append((product.raw.name, float(fv.value)))

        out: List[Dict[str, Any]] = []
        for (cat, key), samples in grouped.items():
            if len(samples) < 4:
                continue
            values = [v for _, v in samples]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            stdev = math.sqrt(variance)
            if stdev == 0:
                continue
            for name, value in samples:
                z = abs(value - mean) / stdev
                if z > self.z_threshold:
                    out.append({
                        "product": name,
                        "category": cat,
                        "field": key,
                        "value": value,
                        "z_score": round(z, 2),
                        "category_mean": round(mean, 4),
                        "message": (
                            f"{key} is {z:.1f} standard deviations from the {cat} average "
                            f"of {mean:.4g}."
                        ),
                    })
        return out

    @staticmethod
    def _find_unit_conflicts(products: List[EnrichedProduct]) -> List[Dict[str, Any]]:
        """The same field carrying different canonical units across records."""
        seen: Dict[str, Dict[str, List[str]]] = {}
        for product in products:
            for key, fv in product.specifications.items():
                if fv.unit:
                    seen.setdefault(key, {}).setdefault(fv.unit, []).append(product.raw.name)
        return [
            {
                "field": key,
                "units": sorted(by_unit),
                "message": f"Field '{key}' appears with {len(by_unit)} different canonical units.",
                "examples": {u: names[:3] for u, names in by_unit.items()},
            }
            for key, by_unit in seen.items()
            if len(by_unit) > 1
        ]
