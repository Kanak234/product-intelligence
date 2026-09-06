"""
Industrial product taxonomy and a deterministic classifier.

Two jobs:

1. Hold a hierarchical category tree (UNSPSC-style, three levels deep) with
   the signals that identify each leaf.
2. Score a product against every leaf and return a ranked list. Because it is
   scored rather than first-match, the runner-up categories are real
   alternatives with real numbers behind them — which is exactly what the
   "explainable outputs" requirement needs.

The classifier is deterministic, so its accuracy can be measured against a
ground-truth set and reported as a number. The LLM only gets consulted when
the top score is below `AMBIGUOUS_THRESHOLD`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .schema import Evidence, FieldValue, Origin

#: Below this score the rule classifier admits it is unsure and the pipeline
#: asks the LLM for a second opinion.
AMBIGUOUS_THRESHOLD = 0.45

#: Scoring weights. Name hits count more than description hits because a
#: product's name is the highest-signal field in every catalogue we tested.
W_NAME_STRONG = 4.0
W_NAME_WEAK = 1.5
W_DESC_STRONG = 2.0
W_DESC_WEAK = 0.75
W_SPEC_MATCH = 1.0
W_NEGATIVE = -3.0


@dataclass(frozen=True)
class Category:
    """A leaf category and the evidence that identifies it."""

    code: str
    path: Tuple[str, ...]
    strong: Tuple[str, ...] = ()       # near-decisive terms
    weak: Tuple[str, ...] = ()         # supporting terms
    negative: Tuple[str, ...] = ()     # terms that rule this leaf out
    expected_specs: Tuple[str, ...] = ()   # specs typical of this category
    required_specs: Tuple[str, ...] = ()   # specs a valid record must carry

    @property
    def name(self) -> str:
        return self.path[-1]

    @property
    def parent_path(self) -> Tuple[str, ...]:
        return self.path[:-1]


TAXONOMY: Tuple[Category, ...] = (
    # -- Pumps ------------------------------------------------------------
    Category(
        "PMP.CENT", ("Industrial Equipment", "Pumps", "Centrifugal Pump"),
        strong=("centrifugal pump", "end suction pump", "monoblock pump", "volute pump"),
        weak=("impeller", "pump", "casing", "suction", "discharge", "head"),
        negative=("diaphragm", "peristaltic", "gear pump", "screw pump"),
        expected_specs=("flow_rate", "pressure", "power", "material", "rotational_speed"),
        required_specs=("flow_rate", "power"),
    ),
    Category(
        "PMP.SUB", ("Industrial Equipment", "Pumps", "Submersible Pump"),
        strong=("submersible pump", "borewell pump", "sump pump", "dewatering pump"),
        weak=("submersible", "borewell", "immersed", "well", "pump"),
        expected_specs=("flow_rate", "pressure", "power", "ip_rating"),
        required_specs=("flow_rate", "power"),
    ),
    Category(
        "PMP.POS", ("Industrial Equipment", "Pumps", "Positive Displacement Pump"),
        strong=("diaphragm pump", "gear pump", "screw pump", "peristaltic pump", "lobe pump",
                "positive displacement pump", "dosing pump", "metering pump"),
        weak=("dosing", "metering", "displacement", "pump"),
        expected_specs=("flow_rate", "pressure", "power", "material"),
        required_specs=("flow_rate",),
    ),
    # -- Motors & Drives --------------------------------------------------
    Category(
        "MOT.IND", ("Industrial Equipment", "Motors & Drives", "AC Induction Motor"),
        strong=("induction motor", "squirrel cage motor", "ac motor", "tefc motor",
                "three phase motor", "3 phase motor"),
        weak=("motor", "stator", "rotor", "winding", "torque"),
        negative=("servo", "stepper", "dc motor", "gear motor"),
        expected_specs=("power", "voltage", "rotational_speed", "phase", "efficiency_class",
                        "insulation_class", "ip_rating", "mounting"),
        required_specs=("power", "voltage", "rotational_speed"),
    ),
    Category(
        "MOT.SRV", ("Industrial Equipment", "Motors & Drives", "Servo & Stepper Motor"),
        strong=("servo motor", "stepper motor", "servomotor", "brushless dc motor", "bldc motor"),
        weak=("encoder", "servo", "stepper", "positioning", "closed loop"),
        expected_specs=("power", "voltage", "torque", "rotational_speed"),
        required_specs=("torque", "voltage"),
    ),
    Category(
        "MOT.VFD", ("Industrial Equipment", "Motors & Drives", "Variable Frequency Drive"),
        strong=("variable frequency drive", "vfd", "frequency inverter", "ac drive",
                "variable speed drive", "soft starter"),
        weak=("inverter", "drive", "speed control", "modbus"),
        expected_specs=("power", "voltage", "current", "frequency", "ip_rating"),
        required_specs=("power", "voltage"),
    ),
    Category(
        "MOT.GBX", ("Industrial Equipment", "Motors & Drives", "Gearbox & Speed Reducer"),
        strong=("gearbox", "speed reducer", "worm gear", "helical gearbox", "planetary gearbox",
                "gear reducer", "geared motor"),
        weak=("ratio", "gear", "reduction", "output shaft"),
        expected_specs=("torque", "rotational_speed", "power", "mounting", "weight"),
        required_specs=("torque",),
    ),
    # -- Valves & Fittings ------------------------------------------------
    Category(
        "VLV.BALL", ("Industrial Equipment", "Valves & Fittings", "Ball Valve"),
        strong=("ball valve",),
        weak=("valve", "seat", "quarter turn", "lever", "flanged"),
        expected_specs=("nominal_bore", "pressure", "material", "max_temperature", "thread_size"),
        required_specs=("nominal_bore", "pressure"),
    ),
    Category(
        "VLV.GATE", ("Industrial Equipment", "Valves & Fittings", "Gate & Globe Valve"),
        strong=("gate valve", "globe valve", "knife gate valve"),
        weak=("valve", "wedge", "rising stem", "flanged"),
        expected_specs=("nominal_bore", "pressure", "material", "max_temperature"),
        required_specs=("nominal_bore", "pressure"),
    ),
    Category(
        "VLV.CTRL", ("Industrial Equipment", "Valves & Fittings", "Control & Solenoid Valve"),
        strong=("solenoid valve", "control valve", "pneumatic valve", "actuated valve",
                "butterfly valve"),
        weak=("actuator", "coil", "pilot", "valve", "positioner"),
        expected_specs=("nominal_bore", "pressure", "voltage", "material", "ip_rating"),
        required_specs=("nominal_bore",),
    ),
    # -- Bearings & Power Transmission -------------------------------------
    Category(
        "BRG.ROLL", ("Industrial Equipment", "Power Transmission", "Bearing"),
        strong=("ball bearing", "roller bearing", "tapered bearing", "thrust bearing",
                "pillow block", "plummer block", "bearing unit"),
        weak=("bearing", "race", "cage", "seal", "grease", "bore"),
        expected_specs=("diameter", "width", "max_temperature", "rotational_speed", "material"),
        required_specs=("diameter",),
    ),
    Category(
        "PTX.BELT", ("Industrial Equipment", "Power Transmission", "Belt, Chain & Coupling"),
        strong=("v-belt", "timing belt", "roller chain", "drive coupling", "flexible coupling",
                "conveyor belt", "sprocket"),
        weak=("belt", "chain", "coupling", "pulley", "pitch", "tension"),
        expected_specs=("length", "width", "torque", "material"),
    ),
    # -- Instrumentation --------------------------------------------------
    Category(
        "INS.PRS", ("Industrial Equipment", "Instrumentation", "Pressure Instrument"),
        strong=("pressure transmitter", "pressure gauge", "pressure sensor", "pressure switch",
                "manometer"),
        weak=("pressure", "transmitter", "gauge", "4-20ma", "diaphragm", "sensor"),
        expected_specs=("pressure", "voltage", "current", "ip_rating", "material", "max_temperature"),
        required_specs=("pressure",),
    ),
    Category(
        "INS.TMP", ("Industrial Equipment", "Instrumentation", "Temperature Instrument"),
        strong=("temperature transmitter", "thermocouple", "rtd sensor", "pt100",
                "temperature sensor", "thermowell"),
        weak=("temperature", "probe", "thermal", "sensor", "4-20ma"),
        expected_specs=("max_temperature", "voltage", "ip_rating", "material", "length"),
        required_specs=("max_temperature",),
    ),
    Category(
        "INS.FLW", ("Industrial Equipment", "Instrumentation", "Flow Instrument"),
        strong=("flow meter", "flowmeter", "rotameter", "magnetic flow meter", "ultrasonic flow meter",
                "mass flow meter"),
        weak=("flow", "meter", "totaliser", "totalizer", "4-20ma"),
        expected_specs=("flow_rate", "nominal_bore", "pressure", "voltage", "ip_rating"),
        required_specs=("flow_rate",),
    ),
    Category(
        "INS.LVL", ("Industrial Equipment", "Instrumentation", "Level Instrument"),
        strong=("level transmitter", "level switch", "radar level", "ultrasonic level sensor",
                "float switch"),
        weak=("level", "tank", "float", "probe", "4-20ma"),
        expected_specs=("length", "voltage", "ip_rating", "max_temperature", "material"),
    ),
    # -- Electrical & Control ---------------------------------------------
    Category(
        "ELC.PLC", ("Electrical & Control", "Automation", "PLC & Controller"),
        strong=("programmable logic controller", "plc", "plc cpu", "hmi panel", "rtu controller"),
        weak=("controller", "ladder", "modbus", "profibus", "ethernet/ip", "i/o module"),
        expected_specs=("voltage", "current", "ip_rating", "mounting"),
        required_specs=("voltage",),
    ),
    Category(
        "ELC.SWG", ("Electrical & Control", "Power Distribution", "Switchgear & Protection"),
        strong=("circuit breaker", "mccb", "mcb", "contactor", "overload relay", "isolator switch",
                "distribution board"),
        weak=("breaker", "relay", "protection", "pole", "trip", "busbar"),
        expected_specs=("voltage", "current", "frequency", "ip_rating", "mounting"),
        required_specs=("voltage", "current"),
    ),
    Category(
        "ELC.CBL", ("Electrical & Control", "Power Distribution", "Cable & Wiring"),
        strong=("power cable", "control cable", "armoured cable", "flexible cable", "cable gland"),
        weak=("cable", "conductor", "insulation", "core", "sheath", "copper"),
        expected_specs=("voltage", "current", "length", "material"),
    ),
    Category(
        "ELC.TRF", ("Electrical & Control", "Power Distribution", "Transformer & Power Supply"),
        strong=("transformer", "smps", "power supply unit", "isolation transformer", "ups system"),
        weak=("supply", "primary", "secondary", "kva", "rectifier"),
        expected_specs=("power", "voltage", "current", "frequency", "ip_rating", "weight"),
        required_specs=("power", "voltage"),
    ),
    # -- Air & Fluid Handling ---------------------------------------------
    Category(
        "AIR.CMP", ("Industrial Equipment", "Air & Gas", "Air Compressor"),
        strong=("air compressor", "screw compressor", "reciprocating compressor", "scroll compressor"),
        weak=("compressor", "cfm", "receiver", "air end", "unloader"),
        expected_specs=("power", "pressure", "flow_rate", "voltage", "noise_level", "volume"),
        required_specs=("power", "pressure"),
    ),
    Category(
        "AIR.FAN", ("Industrial Equipment", "Air & Gas", "Fan & Blower"),
        strong=("centrifugal blower", "axial fan", "industrial fan", "exhaust fan", "roots blower"),
        weak=("fan", "blower", "airflow", "impeller", "cfm", "ventilation"),
        expected_specs=("flow_rate", "power", "voltage", "rotational_speed", "noise_level"),
        required_specs=("flow_rate", "power"),
    ),
    Category(
        "FLT.IND", ("Industrial Equipment", "Filtration", "Filter & Separator"),
        strong=("filter cartridge", "bag filter", "cyclone separator", "air filter", "oil filter",
                "strainer"),
        weak=("filter", "micron", "element", "separator", "housing"),
        expected_specs=("flow_rate", "pressure", "material", "max_temperature", "nominal_bore"),
    ),
    # -- Safety & Consumables ---------------------------------------------
    Category(
        "SAF.PPE", ("Safety & MRO", "Personal Protective Equipment", "PPE"),
        strong=("safety helmet", "safety gloves", "safety goggles", "ear muffs", "respirator",
                "safety harness", "safety shoes"),
        weak=("safety", "protective", "helmet", "gloves", "goggles", "hazard"),
        expected_specs=("material", "certifications", "weight"),
    ),
    Category(
        "MRO.TOOL", ("Safety & MRO", "Tools", "Hand & Power Tool"),
        strong=("torque wrench", "impact wrench", "angle grinder", "drill machine", "hand tool kit",
                "bench vice"),
        weak=("tool", "wrench", "spanner", "grinder", "drill", "chuck"),
        expected_specs=("power", "voltage", "torque", "rotational_speed", "weight"),
    ),
    Category(
        "MRO.LUB", ("Safety & MRO", "Consumables", "Lubricant & Chemical"),
        strong=("industrial lubricant", "hydraulic oil", "gear oil", "grease cartridge",
                "cutting fluid", "rust preventive"),
        weak=("lubricant", "oil", "grease", "viscosity", "iso vg", "additive"),
        expected_specs=("volume", "max_temperature", "material"),
    ),
)

#: Fallback when nothing scores above zero.
UNCLASSIFIED = Category(
    "GEN.UNK", ("Industrial Equipment", "Uncategorised", "General Industrial Equipment")
)

_BY_CODE: Dict[str, Category] = {c.code: c for c in TAXONOMY}
_BY_NAME: Dict[str, Category] = {c.name.lower(): c for c in TAXONOMY}


def get_category(code_or_name: str) -> Optional[Category]:
    key = (code_or_name or "").strip()
    return _BY_CODE.get(key.upper()) or _BY_NAME.get(key.lower())


def all_leaf_names() -> List[str]:
    return [c.name for c in TAXONOMY]


@dataclass
class Classification:
    """One scored candidate."""

    category: Category
    score: float
    normalised: float
    matched_terms: List[str] = field(default_factory=list)
    penalised_terms: List[str] = field(default_factory=list)


class TaxonomyClassifier:
    """Deterministic scoring classifier over `TAXONOMY`."""

    def __init__(self, taxonomy: Sequence[Category] = TAXONOMY) -> None:
        self.taxonomy = tuple(taxonomy)

    def rank(
        self,
        name: str,
        description: str = "",
        specs: Optional[Dict[str, object]] = None,
        top_n: int = 5,
    ) -> List[Classification]:
        """Score every leaf and return the best `top_n`, highest first."""
        name_l = f" {(name or '').lower()} "
        desc_l = f" {(description or '').lower()} "
        spec_keys = set(specs or {})

        results: List[Classification] = []
        for category in self.taxonomy:
            score = 0.0
            matched: List[str] = []
            penalised: List[str] = []

            for term in category.strong:
                if _contains(name_l, term):
                    score += W_NAME_STRONG
                    matched.append(f"name:'{term}'")
                elif _contains(desc_l, term):
                    score += W_DESC_STRONG
                    matched.append(f"description:'{term}'")

            for term in category.weak:
                if _contains(name_l, term):
                    score += W_NAME_WEAK
                    matched.append(f"name:'{term}'")
                elif _contains(desc_l, term):
                    score += W_DESC_WEAK
                    matched.append(f"description:'{term}'")

            for term in category.negative:
                if _contains(name_l, term) or _contains(desc_l, term):
                    score += W_NEGATIVE
                    penalised.append(term)

            overlap = spec_keys & set(category.expected_specs)
            if overlap:
                score += W_SPEC_MATCH * len(overlap)
                matched.append(f"specs:{sorted(overlap)}")

            if score > 0:
                results.append(Classification(category, round(score, 4), 0.0, matched, penalised))

        if not results:
            return [Classification(UNCLASSIFIED, 0.0, 0.0, [], [])]

        results.sort(key=lambda r: (-r.score, r.category.code))

        # Normalise into a 0-1 confidence using the margin over the runner-up.
        # A clear winner scores high; a near-tie scores low, which is exactly
        # when we want the LLM to weigh in.
        top = results[0].score
        runner_up = results[1].score if len(results) > 1 else 0.0
        ceiling = max(top, 1.0)
        for res in results:
            share = res.score / ceiling
            if res is results[0]:
                margin = (top - runner_up) / top if top else 0.0
                res.normalised = round(min(0.99, 0.55 * min(share, 1.0) + 0.45 * margin), 4)
            else:
                res.normalised = round(min(0.85, share * 0.6), 4)

        return results[:top_n]

    def classify(
        self,
        name: str,
        description: str = "",
        specs: Optional[Dict[str, object]] = None,
    ) -> Tuple[FieldValue, Category, List[Classification]]:
        """Return (category FieldValue, winning Category, all ranked candidates)."""
        ranked = self.rank(name, description, specs)
        best = ranked[0]

        evidence = [
            Evidence(
                kind="rule",
                detail=(
                    f"Taxonomy classifier scored {best.score:g} for {best.category.name} "
                    f"({best.category.code}); matched {len(best.matched_terms)} signal(s)"
                ),
                source_text="; ".join(best.matched_terms[:6]) or None,
                rule_id=f"TAX.{best.category.code}",
            )
        ]
        if best.penalised_terms:
            evidence.append(
                Evidence(
                    kind="rule",
                    detail=f"Negative signals reduced the score: {', '.join(best.penalised_terms)}",
                    rule_id="TAX.NEGATIVE",
                )
            )
        if len(ranked) > 1:
            evidence.append(
                Evidence(
                    kind="rule",
                    detail=(
                        f"Runner-up was {ranked[1].category.name} at score "
                        f"{ranked[1].score:g}; margin {best.score - ranked[1].score:g}"
                    ),
                    rule_id="TAX.MARGIN",
                )
            )

        fv = FieldValue(
            value=best.category.name,
            origin=Origin.TAXONOMY if best.category is not UNCLASSIFIED else Origin.DEFAULT,
            confidence=best.normalised if best.category is not UNCLASSIFIED else 0.2,
            evidence=evidence,
        )
        return fv, best.category, ranked


def _contains(haystack: str, term: str) -> bool:
    """Whole-token containment, so 'pump' does not match 'pumpkin'."""
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", haystack) is not None
