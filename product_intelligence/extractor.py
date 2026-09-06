"""
Deterministic specification extraction.

Pulls structured, unit-normalised specifications out of the messy free text
that industrial catalogues actually contain. Every hit records the exact
character span that produced it, which is what makes the explainability layer
honest rather than decorative.

This runs *before* the LLM. Anything the regex layer can prove, the LLM is not
allowed to overwrite (see `FieldValue.beats` / `ORIGIN_TRUST`). The LLM only
fills gaps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .schema import Evidence, FieldValue, Origin
from . import units


@dataclass(frozen=True)
class SpecPattern:
    """One extraction rule."""

    rule_id: str
    field: str
    pattern: str
    dimension: Optional[str] = None     # expected physical dimension, if any
    unit_group: Optional[int] = 2       # regex group holding the unit
    value_group: int = 1
    confidence: float = 0.9
    flags: int = re.IGNORECASE

    def compiled(self) -> re.Pattern:
        return re.compile(self.pattern, self.flags)


#: Number fragment reused across patterns: 2, 2.2, 2,200, .5, -5.
#: Negatives are captured deliberately. A negative weight is a data error, and
#: the validator can only flag what the extractor surfaces — dropping it here
#: would let the bad record through looking merely incomplete.
_NUM = r"(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?|-?\.\d+)"

#: Optional qualifier that sits between a label and its number. Catalogues are
#: full of "range up to 25 bar" and "rated to 16 bar"; without this the label
#: and the value never end up adjacent and the pattern misses.
_QUAL = r"(?:\s*(?:range|rating|rated(?:\s+(?:to|for|at))?|up\s+to|max\.?|maximum|of|is|upto)\b)*\s*[:=]?\s*(?:0\s*-\s*)?"

#: Labelled patterns are the most reliable: "Power: 2.2 kW".
#: Trailing-label patterns catch the equally common "63 A rated current".
#: Bare patterns are the fallback for "2.2 kW motor", and carry lower
#: confidence precisely because they have no label to corroborate them.
SPEC_PATTERNS: List[SpecPattern] = [
    # -- electrical -------------------------------------------------------
    SpecPattern(
        "EXT.POWER.LABELLED", "power",
        rf"\b(?:rated\s+)?(?:power|output|motor\s+rating|rating|consumption){_QUAL}{_NUM}\s*(kW|W|HP|BHP|kVA|VA)\b",
        dimension="power", confidence=0.95,
    ),
    SpecPattern(
        "EXT.POWER.TRAILING", "power",
        rf"\b{_NUM}\s*(kW|W|HP|BHP|kVA)\s+(?:rated\s+)?(?:power|output|motor|rating)\b",
        dimension="power", confidence=0.93,
    ),
    SpecPattern(
        "EXT.POWER.BARE", "power",
        rf"\b{_NUM}\s*(kW|HP|BHP|kVA)\b",
        dimension="power", confidence=0.85,
    ),
    SpecPattern(
        # Bare watts are riskier than bare kW because "W" collides with model
        # codes, so require a following boundary that is not an alphanumeric.
        "EXT.POWER.BARE_W", "power",
        rf"\b{_NUM}\s*(W)\b(?![-\w])",
        dimension="power", confidence=0.78,
    ),
    SpecPattern(
        "EXT.VOLTAGE.LABELLED", "voltage",
        rf"\b(?:voltage|supply|input\s+voltage|operating\s+voltage|coil\s+voltage|grade){_QUAL}{_NUM}\s*(V|VAC|VDC|kV|volts?)\b",
        dimension="voltage", confidence=0.95,
    ),
    SpecPattern(
        "EXT.VOLTAGE.BARE", "voltage",
        rf"\b{_NUM}\s*(V|VAC|VDC|kV)\b(?!\w)",
        dimension="voltage", confidence=0.82,
    ),
    SpecPattern(
        "EXT.CURRENT.LABELLED", "current",
        rf"\b(?:current|full\s*load\s*current|fla|rated\s+current|current\s+rating|consumption){_QUAL}{_NUM}\s*(A|mA|amps?|amperes?)\b",
        dimension="current", confidence=0.95,
    ),
    SpecPattern(
        "EXT.CURRENT.TRAILING", "current",
        rf"\b{_NUM}\s*(A|mA|amps?)\s+(?:rated\s+current|current|output|consumption|load)\b",
        dimension="current", confidence=0.93,
    ),
    SpecPattern(
        # Bare amps: "24 VDC 20 A", "63 A, 3 pole". Lower confidence, and the
        # trailing boundary keeps it off part numbers like "A20".
        "EXT.CURRENT.BARE", "current",
        rf"\b{_NUM}\s*(A|mA)\b(?![-\w])",
        dimension="current", confidence=0.75,
    ),
    SpecPattern(
        "EXT.FREQUENCY", "frequency",
        rf"\b(?:frequency|freq)?\s*[:=]?\s*{_NUM}\s*(Hz|kHz)\b",
        dimension="frequency", confidence=0.85,
    ),
    SpecPattern(
        "EXT.PHASE", "phase",
        r"\b(single|three|3|1)[\s-]*phase\b",
        unit_group=None, confidence=0.92,
    ),
    # -- mechanical -------------------------------------------------------
    SpecPattern(
        "EXT.SPEED", "rotational_speed",
        rf"\b(?:speed|rpm|rotation(?:al)?\s+speed)?\s*[:=]?\s*{_NUM}\s*(rpm|r/min|RPM)\b",
        dimension="rotational_speed", confidence=0.92,
    ),
    SpecPattern(
        "EXT.TORQUE", "torque",
        rf"\b(?:torque|rated\s+torque|output\s+torque){_QUAL}{_NUM}\s*(Nm|N-m|kgm|lbft)\b",
        dimension="torque", confidence=0.94,
    ),
    SpecPattern(
        "EXT.TORQUE.BARE", "torque",
        rf"\b{_NUM}\s*(Nm|N-m)\b",
        dimension="torque", confidence=0.8,
    ),
    SpecPattern(
        "EXT.WEIGHT", "weight",
        rf"\b(?:weight|mass|net\s+weight|gross\s+weight|weighs){_QUAL}{_NUM}\s*(kg|kgs|g|grams?|t|tonnes?|lbs?|pounds?)\b",
        dimension="mass", confidence=0.94,
    ),
    SpecPattern(
        # Bare mass: catalogues routinely end a spec list with ", 62 kg."
        # Restricted to kg/g so it cannot collide with tonnes-as-"t" or amps.
        "EXT.WEIGHT.BARE", "weight",
        rf"\b{_NUM}\s*(kg|g)\b(?![-\w])",
        dimension="mass", confidence=0.72,
    ),
    SpecPattern(
        "EXT.PRESSURE", "pressure",
        rf"\b(?:pressure|max\.?\s*pressure|working\s+pressure|operating\s+pressure|discharge\s+pressure|head|PN\d*){_QUAL}{_NUM}\s*(bar|mbar|psi|kPa|MPa|Pa|kgf/cm2|atm)\b",
        dimension="pressure", confidence=0.94,
    ),
    SpecPattern(
        "EXT.PRESSURE.BARE", "pressure",
        rf"\b{_NUM}\s*(bar|psi)\b(?![-\w])",
        dimension="pressure", confidence=0.78,
    ),
    SpecPattern(
        "EXT.FLOW", "flow_rate",
        rf"\b(?:flow(?:\s*rate)?|air\s*flow|airflow|air\s+delivery|free\s+air\s+delivery|capacity|discharge|delivery){_QUAL}{_NUM}\s*(m3/h|m3/hr|LPM|l/min|l/h|LPS|l/s|GPM|CFM|cmh)\b",
        dimension="flow", confidence=0.94,
    ),
    SpecPattern(
        "EXT.FLOW.BARE", "flow_rate",
        rf"\b{_NUM}\s*(m3/h|m3/hr|LPM|l/min|l/h|CFM|GPM)\b",
        dimension="flow", confidence=0.8,
    ),
    SpecPattern(
        "EXT.TEMPERATURE", "max_temperature",
        rf"\b(?:max\.?\s*temp(?:erature)?|operating\s+temp(?:erature)?|temp(?:erature)?){_QUAL}(-?\d+(?:\.\d+)?)\s*(°?C|°?F|K|celsius)\b",
        dimension="temperature", confidence=0.9,
    ),
    SpecPattern(
        "EXT.TEMPERATURE.BARE", "max_temperature",
        r"\b(-?\d+(?:\.\d+)?)\s*(°C|°F)\b",
        dimension="temperature", confidence=0.8,
    ),
    SpecPattern(
        "EXT.NOISE", "noise_level",
        rf"\b(?:noise(?:\s*level)?|sound)\s*[:=]?\s*{_NUM}\s*(dBA?|dB)\b",
        dimension="noise", confidence=0.9,
    ),
    # -- dimensional ------------------------------------------------------
    SpecPattern(
        "EXT.BORE.DN", "nominal_bore",
        r"\bDN\s*(\d{1,4})\b",
        unit_group=None, confidence=0.95,
    ),
    SpecPattern(
        "EXT.LENGTH.LABELLED", "length",
        rf"\b(?:length|long)\s*[:=]?\s*{_NUM}\s*(mm|cm|m|in|inch|inches|ft)\b",
        dimension="length", confidence=0.92,
    ),
    SpecPattern(
        "EXT.DIAMETER", "diameter",
        rf"\b(?:diameter|dia\.?|bore|ø)\s*[:=]?\s*{_NUM}\s*(mm|cm|m|in|inch|inches)\b",
        dimension="length", confidence=0.93,
    ),
    SpecPattern(
        "EXT.HEIGHT", "height",
        rf"\b(?:height|tall)\s*[:=]?\s*{_NUM}\s*(mm|cm|m|in|inch|inches|ft)\b",
        dimension="length", confidence=0.92,
    ),
    SpecPattern(
        "EXT.WIDTH", "width",
        rf"\b(?:width|wide)\s*[:=]?\s*{_NUM}\s*(mm|cm|m|in|inch|inches|ft)\b",
        dimension="length", confidence=0.92,
    ),
    SpecPattern(
        "EXT.VOLUME", "volume",
        rf"\b(?:volume|tank|receiver|pack\s+size|pack|drum|container|cartridge){_QUAL}{_NUM}\s*(L|litres?|liters?|ml|m3|gal)\b",
        dimension="volume", confidence=0.88,
    ),
    SpecPattern(
        # The lookbehinds keep IEC motor frame designations out of the volume
        # field. "frame 100L" is a shaft-height code, not a hundred litres, and
        # frames ending in L (100L, 160L, 180L, 200L, 225L) are common enough
        # that without this guard most motor records gain a fictitious volume.
        "EXT.VOLUME.BARE", "volume",
        rf"(?<!frame )(?<!frame: )(?<!frame size )(?<!frame size: )"
        rf"\b{_NUM}\s*(L|litres?|liters?|ml)\b(?![-\w])",
        dimension="volume", confidence=0.75,
    ),
    # -- coded / categorical ---------------------------------------------
    SpecPattern(
        "EXT.IP_RATING", "ip_rating",
        r"\bIP\s*-?\s*(\d{2})\b", unit_group=None, confidence=0.97,
    ),
    SpecPattern(
        "EXT.INSULATION_CLASS", "insulation_class",
        r"\b(?:insulation\s+class|class)\s*[:=]?\s*([A-FH])\b",
        unit_group=None, confidence=0.9,
    ),
    SpecPattern(
        "EXT.EFFICIENCY_CLASS", "efficiency_class",
        r"\b(IE[1-5])\b", unit_group=None, confidence=0.96,
    ),
    SpecPattern(
        "EXT.MOUNTING", "mounting",
        r"\b(foot\s+mounted|flange\s+mounted|face\s+mounted|wall\s+mounted|din\s+rail|panel\s+mount(?:ed)?|skid\s+mounted)\b",
        unit_group=None, confidence=0.88,
    ),
    SpecPattern(
        "EXT.DUTY_CYCLE", "duty_cycle",
        r"\b(S[1-9])\s*(?:duty)?\b", unit_group=None, confidence=0.85,
    ),
    SpecPattern(
        "EXT.PROTECTION", "protection_class",
        r"\b(ATEX|Ex\s*d|Ex\s*e|explosion[\s-]?proof|flame[\s-]?proof)\b",
        unit_group=None, confidence=0.9,
    ),
    SpecPattern(
        # Case-sensitive on purpose, and blocked from matching a following "/".
        # Under IGNORECASE this pattern read the "m3" in "30 m3/h" as an M3
        # metric thread, silently inventing a spec from a flow-rate unit.
        "EXT.THREAD", "thread_size",
        r"\b(M\d{1,3}(?:\s*x\s*\d+(?:\.\d+)?)?|G\d{1,3}(?:/\d)?|BSP|NPT)\b(?!\d)",
        unit_group=None, confidence=0.8, flags=0,
    ),
    SpecPattern(
        "EXT.EFFICIENCY_PCT", "efficiency",
        rf"\b(?:efficiency)\s*[:=]?\s*{_NUM}\s*(%)",
        unit_group=None, confidence=0.92,
    ),
]

#: Materials are matched as a controlled vocabulary rather than free text so
#: that "SS 316", "stainless steel 316" and "AISI 316" collapse to one value.
MATERIAL_VOCAB: Dict[str, Tuple[str, ...]] = {
    "Stainless Steel 316": ("ss 316", "ss316", "aisi 316", "stainless steel 316", "316 stainless", "sus316"),
    "Stainless Steel 304": ("ss 304", "ss304", "aisi 304", "stainless steel 304", "304 stainless", "sus304"),
    "Stainless Steel": ("stainless steel", "stainless", "inox"),
    "Cast Iron": ("cast iron", "ci body", "grey iron", "gray iron"),
    "Ductile Iron": ("ductile iron", "sg iron", "nodular iron"),
    "Carbon Steel": ("carbon steel", "mild steel", "ms body", "a105"),
    "Aluminium": ("aluminium", "aluminum", "alu ", "al alloy"),
    "Brass": ("brass",),
    "Bronze": ("bronze", "gunmetal"),
    "Polypropylene": ("polypropylene", "pp body"),
    "PVC": ("pvc", "upvc", "cpvc"),
    "PTFE": ("ptfe", "teflon"),
    "Nylon": ("nylon", "polyamide", "pa66"),
    "Rubber": ("rubber", "epdm", "nbr", "viton"),
    "Ceramic": ("ceramic", "alumina"),
}

#: Certifications, matched case-insensitively as whole tokens.
CERTIFICATION_VOCAB: Tuple[str, ...] = (
    "CE", "UL", "CSA", "ATEX", "IECEx", "ISO 9001", "ISO 14001", "RoHS",
    "REACH", "BIS", "ISI", "API 610", "API 682", "ASME", "DIN", "JIS",
    "NEMA", "CRN", "EAC", "PED", "SIL 2", "SIL 3",
)


class SpecExtractor:
    """Runs every pattern over the source text and normalises the hits."""

    def __init__(self, patterns: Optional[List[SpecPattern]] = None) -> None:
        self.patterns = patterns or SPEC_PATTERNS
        self._compiled = [(p, p.compiled()) for p in self.patterns]

    # -- public API -------------------------------------------------------

    def extract(self, text: str) -> Dict[str, FieldValue]:
        """Extract every recognisable specification from `text`."""
        if not text:
            return {}

        found: Dict[str, FieldValue] = {}

        for pattern, regex in self._compiled:
            for match in regex.finditer(text):
                fv = self._build_field_value(pattern, match, text)
                if fv is None:
                    continue
                existing = found.get(pattern.field)
                if existing is None or fv.beats(existing):
                    found[pattern.field] = fv

        material = self._extract_material(text)
        if material:
            found["material"] = material

        certs = self._extract_certifications(text)
        if certs:
            found["certifications"] = certs

        return found

    def extract_keywords(self, text: str, limit: int = 12) -> List[str]:
        """Frequency-ranked content tokens, stopwords removed. Deterministic."""
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())
        counts: Dict[str, int] = {}
        for token in tokens:
            if token in _STOPWORDS or token.isdigit():
                continue
            counts[token] = counts.get(token, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [word for word, _ in ranked[:limit]]

    # -- internals --------------------------------------------------------

    def _build_field_value(
        self, pattern: SpecPattern, match: re.Match, text: str
    ) -> Optional[FieldValue]:
        raw_value = match.group(pattern.value_group)
        if raw_value is None:
            return None
        raw_value = raw_value.strip()

        span = [match.start(), match.end()]
        snippet = text[max(0, span[0] - 20) : min(len(text), span[1] + 20)].strip()

        # Categorical rules (no unit group): keep the matched token as-is.
        if pattern.unit_group is None:
            value = self._normalise_categorical(pattern.field, raw_value)
            return FieldValue(
                value=value,
                origin=Origin.EXTRACTED,
                confidence=pattern.confidence,
                raw=match.group(0).strip(),
                evidence=[
                    Evidence(
                        kind="text_span",
                        detail=f"Matched rule {pattern.rule_id} on source text",
                        source_text=snippet,
                        span=span,
                        rule_id=pattern.rule_id,
                    )
                ],
            )

        # Measured rules: parse the number, resolve and convert the unit.
        try:
            numeric = float(raw_value.replace(",", ""))
        except ValueError:
            return None

        unit_raw = match.group(pattern.unit_group)
        if not unit_raw:
            return None

        try:
            value, canon_unit, dim_name = units.to_canonical(numeric, unit_raw)
        except units.UnitError:
            return None

        # Reject a hit whose unit belongs to the wrong physical dimension.
        if pattern.dimension and dim_name != pattern.dimension:
            return None

        converted = unit_raw.strip().lower() != canon_unit.lower()
        detail = f"Matched rule {pattern.rule_id} on source text"
        if converted:
            detail += f"; converted {numeric:g} {unit_raw.strip()} -> {value:g} {canon_unit}"

        return FieldValue(
            value=value,
            origin=Origin.EXTRACTED,
            confidence=pattern.confidence,
            unit=canon_unit,
            raw=match.group(0).strip(),
            evidence=[
                Evidence(
                    kind="text_span",
                    detail=detail,
                    source_text=snippet,
                    span=span,
                    rule_id=pattern.rule_id,
                )
            ],
        )

    @staticmethod
    def _normalise_categorical(field_name: str, raw: str) -> str:
        token = raw.strip()
        if field_name == "phase":
            low = token.lower()
            return "3-phase" if low in {"three", "3"} else "1-phase"
        if field_name == "ip_rating":
            return f"IP{token}"
        if field_name == "nominal_bore":
            return f"DN{token}"
        if field_name == "mounting":
            return token.title()
        if field_name == "protection_class":
            return token.upper().replace("  ", " ")
        return token.upper() if len(token) <= 4 else token

    @staticmethod
    def _extract_material(text: str) -> Optional[FieldValue]:
        low = text.lower()
        # Longest alias first so "stainless steel 316" wins over "stainless steel".
        candidates: List[Tuple[int, str, str]] = []
        for canonical, aliases in MATERIAL_VOCAB.items():
            for alias in aliases:
                idx = low.find(alias)
                if idx >= 0:
                    candidates.append((len(alias), canonical, alias))
        if not candidates:
            return None
        _, canonical, alias = max(candidates, key=lambda c: c[0])
        return FieldValue(
            value=canonical,
            origin=Origin.EXTRACTED,
            confidence=0.93,
            raw=alias,
            evidence=[
                Evidence(
                    kind="rule",
                    detail=f"Matched controlled material vocabulary entry '{alias}'",
                    source_text=alias,
                    rule_id="EXT.MATERIAL.VOCAB",
                )
            ],
        )

    @staticmethod
    def _extract_certifications(text: str) -> Optional[FieldValue]:
        hits: List[str] = []
        for cert in CERTIFICATION_VOCAB:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(cert)}(?![A-Za-z0-9])", text, re.IGNORECASE):
                hits.append(cert)
        if not hits:
            return None
        return FieldValue(
            value=sorted(set(hits)),
            origin=Origin.EXTRACTED,
            confidence=0.9,
            evidence=[
                Evidence(
                    kind="rule",
                    detail=f"Found {len(hits)} certification token(s) in source text",
                    source_text=", ".join(hits),
                    rule_id="EXT.CERT.VOCAB",
                )
            ],
        )


_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was", "has",
    "have", "can", "will", "all", "any", "its", "our", "your", "which", "used",
    "use", "using", "made", "also", "such", "into", "than", "then", "they",
    "them", "not", "but", "one", "two", "per", "via", "new", "high", "low",
    "product", "products", "item", "items", "type", "types", "model",
}
