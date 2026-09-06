"""
The enrichment pipeline.

Order matters, and it is the opposite of the obvious one. The deterministic
layer runs *first* and the language model runs *last*, constrained to the gaps
the rules could not fill:

    1. extract   - regex + controlled vocabulary over the raw text
    2. classify  - scored taxonomy match
    3. derive    - compute what physics lets us compute for free
    4. llm       - fill remaining gaps only, never overwrite (see ORIGIN_TRUST)
    5. describe  - commerce-ready description

Two consequences worth stating out loud in a demo:

* The pipeline degrades gracefully. With no model server reachable, steps 1-3
  and 5 still produce a usable, scored, explainable record. `llm_used` in the
  trace says which mode ran.
* The model can never silently contradict the source data, because
  `EnrichedProduct.set_spec` refuses a lower-trust value. Hallucination is
  contained to fields where there was nothing to contradict.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from .schema import Evidence, EnrichedProduct, FieldValue, Origin, RawProduct
from .extractor import SpecExtractor
from .taxonomy import AMBIGUOUS_THRESHOLD, TaxonomyClassifier, get_category
from . import units

try:  # structlog is present in the API/worker, absent in bare unit tests
    import structlog
    logger = structlog.get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


#: Fields the LLM is allowed to propose. Anything outside this list is dropped,
#: which stops the model inventing bespoke keys that break the catalogue schema.
LLM_ALLOWED_SPECS = {
    "power", "voltage", "current", "frequency", "phase", "rotational_speed",
    "torque", "weight", "pressure", "flow_rate", "max_temperature", "length",
    "width", "height", "diameter", "volume", "material", "mounting",
    "ip_rating", "insulation_class", "efficiency_class", "efficiency",
    "duty_cycle", "noise_level", "nominal_bore", "thread_size",
    "protection_class", "certifications",
}

ENRICH_SYSTEM_PROMPT = (
    "You are an industrial product data specialist. You output JSON only - no "
    "prose, no markdown fences. If you do not know a value, omit the key "
    "entirely rather than guessing. Never contradict a value marked as already "
    "known."
)


def build_gap_prompt(
    raw: RawProduct,
    known_specs: Dict[str, Any],
    category_name: str,
    wanted: List[str],
) -> str:
    """Prompt the model with what we already know and ask only for the gaps."""
    known_block = json.dumps(known_specs, indent=2) if known_specs else "{}"
    wanted_block = ", ".join(wanted) if wanted else "(none)"
    return f"""Product name: {raw.name}
Source description: {raw.description or "(none provided)"}
Assigned category: {category_name}

Already extracted from the source text (authoritative - do not change these):
{known_block}

Fill ONLY these missing fields, if and only if they are stated in or reliably
implied by the source text or are standard for this exact product type:
{wanted_block}

Return JSON in exactly this shape:
{{
  "specifications": {{"field_name": {{"value": <number or string>, "unit": "<unit or null>", "basis": "<one short sentence of justification>"}}}},
  "attributes": ["short selling point", "..."],
  "description": "2-3 sentence commerce-ready product description",
  "category_suggestion": "<category name, only if you disagree with the assigned one>"
}}"""


class ProductEnricher:
    """
    Runs the full pipeline for one product.

    `llm_router` is optional. Pass None (or leave the model servers down) and
    the enricher runs in deterministic-only mode.
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        model: Optional[str] = None,
        provider_name: Optional[str] = None,
        llm_timeout: float = 45.0,
    ) -> None:
        self.llm_router = llm_router
        self.model = model
        self.provider_name = provider_name
        self.llm_timeout = llm_timeout
        self.extractor = SpecExtractor()
        self.classifier = TaxonomyClassifier()

    # -- public API -------------------------------------------------------

    async def enrich(self, raw: RawProduct, use_llm: bool = True) -> EnrichedProduct:
        product = EnrichedProduct(raw=raw)
        text = raw.text_blob()

        self._stage_extract(product, text)
        self._stage_classify(product, text)
        self._stage_derive(product)

        llm_used = False
        if use_llm and self.llm_router is not None:
            llm_used = await self._stage_llm(product, text)

        self._stage_describe(product, text, llm_used)

        product.trace("complete", llm_used=llm_used, spec_count=len(product.specifications))
        return product

    async def enrich_many(
        self, raws: List[RawProduct], use_llm: bool = True, concurrency: int = 4
    ) -> List[EnrichedProduct]:
        """Bounded-concurrency batch enrichment, order preserved."""
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def run(raw: RawProduct) -> EnrichedProduct:
            async with semaphore:
                try:
                    return await self.enrich(raw, use_llm=use_llm)
                except Exception as exc:  # one bad record must not kill the batch
                    logger.warning("enrichment failed", product=raw.name, error=str(exc))
                    failed = EnrichedProduct(raw=raw)
                    failed.trace("error", message=str(exc))
                    return failed

        return list(await asyncio.gather(*(run(r) for r in raws)))

    # -- stages -----------------------------------------------------------

    def _stage_extract(self, product: EnrichedProduct, text: str) -> None:
        extracted = self.extractor.extract(text)
        for key, fv in extracted.items():
            product.set_spec(key, fv)

        # Explicit key/value attributes from the source system outrank text
        # extraction, since somebody already structured them.
        for key, value in (product.raw.attributes or {}).items():
            norm_key = _normalise_key(key)
            fv = _field_from_attribute(norm_key, str(value))
            if fv:
                product.set_spec(norm_key, fv)

        product.keywords = self.extractor.extract_keywords(text)
        product.trace("extract", found=sorted(extracted), keyword_count=len(product.keywords))

    def _stage_classify(self, product: EnrichedProduct, text: str) -> None:
        # A category supplied by the source system is treated as input truth.
        if product.raw.category and get_category(product.raw.category):
            category = get_category(product.raw.category)
            product.category = FieldValue(
                value=category.name,
                origin=Origin.INPUT,
                confidence=1.0,
                evidence=[Evidence(
                    kind="rule",
                    detail="Category supplied by the source system and matched to the taxonomy",
                    source_text=product.raw.category,
                    rule_id="TAX.SOURCE_PROVIDED",
                )],
            )
            product.category_path = list(category.path)
            product.category_alternatives = []
            product.trace("classify", source="input", category=category.name)
            return

        fv, category, ranked = self.classifier.classify(
            product.raw.name, product.raw.description, product.specifications
        )
        product.category = fv
        product.category_path = list(category.path)
        product.category_alternatives = [
            {
                "category": c.category.name,
                "code": c.category.code,
                "score": c.score,
                "confidence": c.normalised,
                "why": c.matched_terms[:4],
            }
            for c in ranked[1:4]
        ]
        product.trace(
            "classify",
            source="taxonomy",
            category=category.name,
            confidence=fv.confidence,
            ambiguous=fv.confidence < AMBIGUOUS_THRESHOLD,
        )

    def _stage_derive(self, product: EnrichedProduct) -> None:
        """Compute what physics gives us for free from what we already have."""
        specs = product.specifications
        derived: List[str] = []

        # Current from power, voltage and phase (assumed pf 0.85, eff 0.9).
        if "current" not in specs and {"power", "voltage"} <= set(specs):
            power, voltage = specs["power"], specs["voltage"]
            if _numeric(power) and _numeric(voltage) and float(voltage.value) > 0:
                phase = specs.get("phase")
                root3 = math.sqrt(3) if phase and str(phase.value).startswith("3") else 1.0
                pf, eff = 0.85, 0.90
                amps = float(power.value) / (float(voltage.value) * root3 * pf * eff)
                product.set_spec("current", FieldValue(
                    value=round(amps, 2), origin=Origin.DERIVED, confidence=0.7, unit="A",
                    evidence=[Evidence(
                        kind="computation",
                        detail=(
                            f"Estimated from P/(V x {'sqrt(3) x ' if root3 > 1 else ''}"
                            f"pf {pf} x eff {eff}) using {power.value:g} W at {voltage.value:g} V"
                        ),
                        rule_id="DER.CURRENT.FROM_POWER",
                    )],
                ))
                derived.append("current")

        # Torque from power and speed.
        if "torque" not in specs and {"power", "rotational_speed"} <= set(specs):
            power, speed = specs["power"], specs["rotational_speed"]
            if _numeric(power) and _numeric(speed) and float(speed.value) > 0:
                nm = float(power.value) / (2 * math.pi * float(speed.value) / 60.0)
                product.set_spec("torque", FieldValue(
                    value=round(nm, 2), origin=Origin.DERIVED, confidence=0.72, unit="Nm",
                    evidence=[Evidence(
                        kind="computation",
                        detail=(
                            f"Computed as P/(2*pi*n/60) from {power.value:g} W "
                            f"at {speed.value:g} rpm"
                        ),
                        rule_id="DER.TORQUE.FROM_POWER_SPEED",
                    )],
                ))
                derived.append("torque")

        # Frequency implied by a standard synchronous speed.
        if "frequency" not in specs and "rotational_speed" in specs:
            speed = specs["rotational_speed"]
            if _numeric(speed):
                n = float(speed.value)
                for hz, syncs in ((50, (3000, 1500, 1000, 750, 600)), (60, (3600, 1800, 1200, 900, 720))):
                    if any(0.90 * s <= n <= 1.02 * s for s in syncs):
                        product.set_spec("frequency", FieldValue(
                            value=hz, origin=Origin.DERIVED, confidence=0.6, unit="Hz",
                            evidence=[Evidence(
                                kind="computation",
                                detail=f"{n:g} rpm matches a standard {hz} Hz synchronous speed",
                                rule_id="DER.FREQ.FROM_SPEED",
                            )],
                        ))
                        derived.append("frequency")
                        break

        # IP rating implies an environment attribute worth surfacing.
        ip = specs.get("ip_rating")
        if ip and isinstance(ip.value, str) and len(ip.value) == 4:
            try:
                water = int(ip.value[3])
                if water >= 5:
                    product.attributes.append("Suitable for washdown environments")
            except ValueError:
                pass

        product.trace("derive", derived=derived)

    async def _stage_llm(self, product: EnrichedProduct, text: str) -> bool:
        """Ask the model for the gaps only. Returns True if it contributed."""
        category = get_category(str(product.category.value)) if product.category else None
        wanted = [
            spec for spec in (category.expected_specs if category else ())
            if spec not in product.specifications
        ]
        # Even with no category gaps, a generated description is still useful.
        if not wanted and product.enriched_description:
            return False

        known = {
            key: (
                units.format_value(fv.value, fv.unit) if fv.unit and _numeric(fv)
                else fv.value
            )
            for key, fv in product.specifications.items()
        }
        prompt = build_gap_prompt(
            product.raw, known, str(product.category.value) if product.category else "Unknown", wanted
        )

        raw_response = await self._call_llm(prompt)
        if raw_response is None:
            product.trace("llm", status="unavailable", fallback="deterministic_only")
            return False

        payload = _parse_json_response(raw_response)
        if payload is None:
            product.trace("llm", status="unparseable", chars=len(raw_response))
            logger.warning("LLM returned unparseable JSON", product=product.raw.name)
            return False

        accepted, rejected = self._merge_llm_specs(product, payload.get("specifications", {}))

        for attr in payload.get("attributes", []) or []:
            if isinstance(attr, str) and 3 < len(attr) < 120:
                product.attributes.append(attr.strip())

        description = payload.get("description")
        if isinstance(description, str) and len(description.strip()) > 30:
            product.enriched_description = FieldValue(
                value=description.strip(),
                origin=Origin.LLM,
                confidence=0.65,
                evidence=[Evidence(
                    kind="llm",
                    detail=f"Generated by {self.model or 'local model'} from the source text and extracted specs",
                    rule_id="LLM.DESCRIPTION",
                )],
            )

        suggestion = payload.get("category_suggestion")
        if (
            isinstance(suggestion, str)
            and product.category
            and product.category.confidence < AMBIGUOUS_THRESHOLD
            and get_category(suggestion)
        ):
            product.category_alternatives.insert(0, {
                "category": get_category(suggestion).name,
                "code": get_category(suggestion).code,
                "score": None,
                "confidence": 0.6,
                "why": ["model disagreed with the rule classifier on a low-confidence match"],
            })

        product.trace("llm", status="ok", accepted=accepted, rejected=rejected)
        return bool(accepted) or product.enriched_description is not None

    def _stage_describe(self, product: EnrichedProduct, text: str, llm_used: bool) -> None:
        """Guarantee a description exists even with no model available."""
        if product.enriched_description and product.enriched_description.value:
            return

        category_name = str(product.category.value) if product.category else "industrial product"
        bits: List[str] = []
        for key in ("power", "voltage", "flow_rate", "pressure", "rotational_speed",
                    "nominal_bore", "torque", "material", "ip_rating"):
            fv = product.specifications.get(key)
            if not fv:
                continue
            label = key.replace("_", " ")
            if fv.unit and _numeric(fv):
                bits.append(f"{label} of {units.format_value(fv.value, fv.unit)}")
            else:
                bits.append(f"{label} {fv.value}")
            if len(bits) >= 4:
                break

        sentence = f"{product.raw.name} is a {category_name.lower()}"
        if bits:
            sentence += " rated at " + ", ".join(bits[:-1])
            sentence += (" and " if len(bits) > 1 else " ") + bits[-1]
        sentence += "."

        if product.raw.description:
            sentence += " " + product.raw.description.strip().rstrip(".") + "."

        product.enriched_description = FieldValue(
            value=sentence,
            origin=Origin.DERIVED,
            confidence=0.75,
            evidence=[Evidence(
                kind="computation",
                detail=(
                    "Composed deterministically from the assigned category and the "
                    f"{len(bits)} highest-signal extracted specification(s)"
                    + ("" if llm_used else "; no model server was reachable")
                ),
                rule_id="DER.DESCRIPTION.TEMPLATE",
            )],
        )

    # -- LLM plumbing -----------------------------------------------------

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Single model call with timeout and failover. None on any failure."""
        if self.llm_router is None:
            return None

        try:
            from app.services.llm.base import ChatMessage  # local import: optional dep
        except Exception:  # pragma: no cover
            return None

        messages = [
            ChatMessage(role="system", content=ENRICH_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]

        provider_name = self.provider_name
        try:
            if provider_name is None and hasattr(self.llm_router, "auto_select_provider"):
                provider_name = await asyncio.wait_for(
                    self.llm_router.auto_select_provider(), timeout=10.0
                )
            provider = self.llm_router.get_provider(provider_name)
            return await asyncio.wait_for(
                provider.complete(messages, self.model), timeout=self.llm_timeout
            )
        except asyncio.TimeoutError:
            logger.warning("LLM call timed out", timeout=self.llm_timeout)
            return None
        except Exception as exc:
            logger.warning("LLM call failed", error=str(exc))
            return None

    def _merge_llm_specs(
        self, product: EnrichedProduct, proposed: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Validate, normalise and merge model-proposed specs. Gaps only."""
        accepted: List[str] = []
        rejected: List[str] = []

        if not isinstance(proposed, dict):
            return accepted, rejected

        for raw_key, payload in proposed.items():
            key = _normalise_key(raw_key)
            if key not in LLM_ALLOWED_SPECS:
                rejected.append(f"{raw_key}: not in allowed schema")
                continue
            if key in product.specifications:
                existing = product.specifications[key]
                if existing.origin in (Origin.INPUT, Origin.EXTRACTED):
                    rejected.append(f"{key}: already known from source ({existing.origin.value})")
                    continue

            value, unit, basis = _unpack_llm_payload(payload)
            if value is None:
                rejected.append(f"{key}: no usable value")
                continue

            if unit:
                try:
                    value, unit, _ = units.to_canonical(float(value), unit)
                except (units.UnitError, TypeError, ValueError):
                    rejected.append(f"{key}: unrecognised unit {unit!r}")
                    continue

            fv = FieldValue(
                value=value,
                origin=Origin.LLM,
                confidence=0.6,
                unit=unit,
                evidence=[Evidence(
                    kind="llm",
                    detail=basis or "Proposed by the local model to fill a gap in the source data",
                    rule_id="LLM.SPEC_GAPFILL",
                )],
            )
            if product.set_spec(key, fv):
                accepted.append(key)
            else:
                rejected.append(f"{key}: lower trust than existing value")

        return accepted, rejected


# -- module helpers -------------------------------------------------------


def _numeric(fv: FieldValue) -> bool:
    return isinstance(fv.value, (int, float)) and not isinstance(fv.value, bool)


def _normalise_key(key: str) -> str:
    """Map the many names a source uses for one field onto our schema."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
    return _KEY_ALIASES.get(cleaned, cleaned)


_KEY_ALIASES: Dict[str, str] = {
    "kw": "power", "power_rating": "power", "rated_power": "power", "motor_power": "power",
    "wattage": "power", "hp": "power", "output_power": "power",
    "volts": "voltage", "supply_voltage": "voltage", "rated_voltage": "voltage",
    "operating_voltage": "voltage", "input_voltage": "voltage",
    "amps": "current", "amperage": "current", "rated_current": "current",
    "full_load_current": "current", "fla": "current",
    "rpm": "rotational_speed", "speed": "rotational_speed", "shaft_speed": "rotational_speed",
    "mass": "weight", "net_weight": "weight", "gross_weight": "weight",
    "max_pressure": "pressure", "working_pressure": "pressure", "head": "pressure",
    "discharge_pressure": "pressure",
    "flow": "flow_rate", "capacity": "flow_rate", "discharge": "flow_rate",
    "temperature": "max_temperature", "operating_temperature": "max_temperature",
    "max_temp": "max_temperature", "temp": "max_temperature",
    "body_material": "material", "construction": "material", "material_of_construction": "material",
    "moc": "material",
    "ingress_protection": "ip_rating", "ip": "ip_rating", "protection_rating": "ip_rating",
    "dn": "nominal_bore", "nominal_diameter": "nominal_bore", "size": "nominal_bore",
    "bore": "nominal_bore", "port_size": "nominal_bore",
    "dia": "diameter", "outer_diameter": "diameter", "od": "diameter",
    "freq": "frequency", "hz": "frequency",
    "phases": "phase", "no_of_phases": "phase",
    "cert": "certifications", "certification": "certifications", "approvals": "certifications",
    "standards": "certifications",
    "mount": "mounting", "mounting_type": "mounting",
    "noise": "noise_level", "sound_level": "noise_level",
    "eff": "efficiency", "efficiency_pct": "efficiency",
    "ie_class": "efficiency_class",
}


def _field_from_attribute(key: str, value: str) -> Optional[FieldValue]:
    """Parse a source-system key/value pair, normalising units where present."""
    text = value.strip()
    if not text:
        return None

    match = re.match(r"^(-?\d[\d,]*(?:\.\d+)?)\s*([A-Za-z°/³\"%]+.*)?$", text)
    if match:
        try:
            number = float(match.group(1).replace(",", ""))
        except ValueError:
            number = None
        unit_text = (match.group(2) or "").strip()
        if number is not None:
            if unit_text:
                try:
                    canonical_value, canonical_unit, _ = units.to_canonical(number, unit_text)
                    return FieldValue(
                        value=canonical_value, origin=Origin.INPUT, confidence=1.0,
                        unit=canonical_unit, raw=text,
                        evidence=[Evidence(
                            kind="rule",
                            detail=f"Supplied by the source system as '{key}: {text}'",
                            source_text=text, rule_id="EXT.ATTR.STRUCTURED",
                        )],
                    )
                except units.UnitError:
                    pass
            else:
                return FieldValue(
                    value=number, origin=Origin.INPUT, confidence=1.0, raw=text,
                    evidence=[Evidence(
                        kind="rule",
                        detail=f"Supplied by the source system as '{key}: {text}'",
                        source_text=text, rule_id="EXT.ATTR.STRUCTURED",
                    )],
                )

    return FieldValue(
        value=text, origin=Origin.INPUT, confidence=1.0, raw=text,
        evidence=[Evidence(
            kind="rule",
            detail=f"Supplied by the source system as '{key}: {text}'",
            source_text=text, rule_id="EXT.ATTR.STRUCTURED",
        )],
    )


def _unpack_llm_payload(payload: Any) -> Tuple[Any, Optional[str], Optional[str]]:
    """Accept both {"value": x, "unit": y} and a bare scalar."""
    if isinstance(payload, dict):
        return payload.get("value"), payload.get("unit"), payload.get("basis")
    if isinstance(payload, (int, float, str, list)):
        return payload, None, None
    return None, None, None


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from a model response, fences and preamble included."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost balanced {...} block.
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(cleaned)):
        if cleaned[index] == "{":
            depth += 1
        elif cleaned[index] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start : index + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
