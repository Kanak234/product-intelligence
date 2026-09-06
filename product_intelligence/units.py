"""
Unit normalisation for industrial product specifications.

This is the layer that makes "Accuracy & Consistency" measurable. Industrial
catalogues express the same physical quantity a dozen different ways — 2.2 kW,
2200 W, 3 HP; 415V, 415 volts, 415 VAC; 50 mm, 5 cm, 1.97". Downstream
comparison, dedup, and search all break unless these collapse onto one
canonical representation.

Everything here is deterministic and dependency-free, so the same conversion
runs identically in the API, the Celery worker, and the evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Dimension:
    """A physical dimension and its canonical SI-ish unit."""

    name: str
    canonical: str


POWER = Dimension("power", "W")
VOLTAGE = Dimension("voltage", "V")
CURRENT = Dimension("current", "A")
FREQUENCY = Dimension("frequency", "Hz")
LENGTH = Dimension("length", "mm")
MASS = Dimension("mass", "kg")
PRESSURE = Dimension("pressure", "bar")
FLOW = Dimension("flow", "m3/h")
TEMPERATURE = Dimension("temperature", "C")
SPEED = Dimension("rotational_speed", "rpm")
TORQUE = Dimension("torque", "Nm")
VOLUME = Dimension("volume", "L")
ANGLE = Dimension("angle", "deg")
NOISE = Dimension("noise", "dB")

#: unit alias -> (dimension, multiplier, offset)
#: Value in canonical units = raw * multiplier + offset.
_UNIT_TABLE: Dict[str, Tuple[Dimension, float, float]] = {
    # power ---------------------------------------------------------------
    "w": (POWER, 1.0, 0.0),
    "watt": (POWER, 1.0, 0.0),
    "watts": (POWER, 1.0, 0.0),
    "kw": (POWER, 1_000.0, 0.0),
    "kilowatt": (POWER, 1_000.0, 0.0),
    "mw": (POWER, 1_000_000.0, 0.0),
    "hp": (POWER, 745.699872, 0.0),
    "bhp": (POWER, 745.699872, 0.0),
    "ps": (POWER, 735.49875, 0.0),
    "va": (POWER, 1.0, 0.0),
    "kva": (POWER, 1_000.0, 0.0),
    # voltage -------------------------------------------------------------
    "v": (VOLTAGE, 1.0, 0.0),
    "volt": (VOLTAGE, 1.0, 0.0),
    "volts": (VOLTAGE, 1.0, 0.0),
    "vac": (VOLTAGE, 1.0, 0.0),
    "vdc": (VOLTAGE, 1.0, 0.0),
    "kv": (VOLTAGE, 1_000.0, 0.0),
    "mv": (VOLTAGE, 0.001, 0.0),
    # current -------------------------------------------------------------
    "a": (CURRENT, 1.0, 0.0),
    "amp": (CURRENT, 1.0, 0.0),
    "amps": (CURRENT, 1.0, 0.0),
    "ampere": (CURRENT, 1.0, 0.0),
    "amperes": (CURRENT, 1.0, 0.0),
    "ma": (CURRENT, 0.001, 0.0),
    # frequency -----------------------------------------------------------
    "hz": (FREQUENCY, 1.0, 0.0),
    "hertz": (FREQUENCY, 1.0, 0.0),
    "khz": (FREQUENCY, 1_000.0, 0.0),
    # length --------------------------------------------------------------
    "mm": (LENGTH, 1.0, 0.0),
    "millimetre": (LENGTH, 1.0, 0.0),
    "millimeter": (LENGTH, 1.0, 0.0),
    "cm": (LENGTH, 10.0, 0.0),
    "m": (LENGTH, 1_000.0, 0.0),
    "metre": (LENGTH, 1_000.0, 0.0),
    "meter": (LENGTH, 1_000.0, 0.0),
    "in": (LENGTH, 25.4, 0.0),
    "inch": (LENGTH, 25.4, 0.0),
    "inches": (LENGTH, 25.4, 0.0),
    '"': (LENGTH, 25.4, 0.0),
    "ft": (LENGTH, 304.8, 0.0),
    "feet": (LENGTH, 304.8, 0.0),
    # mass ----------------------------------------------------------------
    "kg": (MASS, 1.0, 0.0),
    "kgs": (MASS, 1.0, 0.0),
    "kilogram": (MASS, 1.0, 0.0),
    "kilograms": (MASS, 1.0, 0.0),
    "g": (MASS, 0.001, 0.0),
    "gram": (MASS, 0.001, 0.0),
    "grams": (MASS, 0.001, 0.0),
    "t": (MASS, 1_000.0, 0.0),
    "ton": (MASS, 1_000.0, 0.0),
    "tonne": (MASS, 1_000.0, 0.0),
    "lb": (MASS, 0.45359237, 0.0),
    "lbs": (MASS, 0.45359237, 0.0),
    "pound": (MASS, 0.45359237, 0.0),
    # pressure ------------------------------------------------------------
    "bar": (PRESSURE, 1.0, 0.0),
    "bars": (PRESSURE, 1.0, 0.0),
    "mbar": (PRESSURE, 0.001, 0.0),
    "pa": (PRESSURE, 1e-5, 0.0),
    "kpa": (PRESSURE, 0.01, 0.0),
    "mpa": (PRESSURE, 10.0, 0.0),
    "psi": (PRESSURE, 0.0689476, 0.0),
    "atm": (PRESSURE, 1.01325, 0.0),
    "kgf/cm2": (PRESSURE, 0.980665, 0.0),
    # flow ----------------------------------------------------------------
    "m3/h": (FLOW, 1.0, 0.0),
    "m3/hr": (FLOW, 1.0, 0.0),
    "cmh": (FLOW, 1.0, 0.0),
    "lpm": (FLOW, 0.06, 0.0),
    "l/min": (FLOW, 0.06, 0.0),
    "lps": (FLOW, 3.6, 0.0),
    "l/s": (FLOW, 3.6, 0.0),
    "l/h": (FLOW, 0.001, 0.0),
    "gpm": (FLOW, 0.2271247, 0.0),
    "cfm": (FLOW, 1.699011, 0.0),
    # temperature ---------------------------------------------------------
    "c": (TEMPERATURE, 1.0, 0.0),
    "°c": (TEMPERATURE, 1.0, 0.0),
    "celsius": (TEMPERATURE, 1.0, 0.0),
    "k": (TEMPERATURE, 1.0, -273.15),
    "f": (TEMPERATURE, 5.0 / 9.0, -32.0 * 5.0 / 9.0),
    "°f": (TEMPERATURE, 5.0 / 9.0, -32.0 * 5.0 / 9.0),
    # rotational speed ----------------------------------------------------
    "rpm": (SPEED, 1.0, 0.0),
    "r/min": (SPEED, 1.0, 0.0),
    "rps": (SPEED, 60.0, 0.0),
    # torque --------------------------------------------------------------
    "nm": (TORQUE, 1.0, 0.0),
    "n-m": (TORQUE, 1.0, 0.0),
    "kgm": (TORQUE, 9.80665, 0.0),
    "lbft": (TORQUE, 1.355818, 0.0),
    # volume --------------------------------------------------------------
    "l": (VOLUME, 1.0, 0.0),
    "litre": (VOLUME, 1.0, 0.0),
    "liter": (VOLUME, 1.0, 0.0),
    "litres": (VOLUME, 1.0, 0.0),
    "ml": (VOLUME, 0.001, 0.0),
    "m3": (VOLUME, 1_000.0, 0.0),
    "gal": (VOLUME, 3.785412, 0.0),
    # misc ----------------------------------------------------------------
    "deg": (ANGLE, 1.0, 0.0),
    "db": (NOISE, 1.0, 0.0),
    "dba": (NOISE, 1.0, 0.0),
}

#: Units that are genuinely ambiguous without context. `m` is metres in a
#: dimension field but minutes in a duration field; `t` is tonnes but also
#: "turns". The extractor resolves these using the field name, never blindly.
AMBIGUOUS_UNITS = {"m", "t", "a", "c", "k", "f", "in"}


class UnitError(ValueError):
    """Raised when a unit string cannot be resolved."""


def canonical_unit(dimension: Dimension) -> str:
    return dimension.canonical


def resolve(unit: str) -> Optional[Tuple[Dimension, float, float]]:
    """Look up a unit alias. Case- and space-insensitive."""
    if not unit:
        return None
    key = unit.strip().lower().replace(" ", "").replace("³", "3").replace("^3", "3")
    key = key.rstrip(".")
    return _UNIT_TABLE.get(key)


def convert(value: float, from_unit: str, to_unit: Optional[str] = None) -> Tuple[float, str]:
    """
    Convert `value` from `from_unit` to `to_unit` (default: canonical unit
    of the source dimension). Returns (converted_value, unit_symbol).
    """
    src = resolve(from_unit)
    if src is None:
        raise UnitError(f"Unknown unit: {from_unit!r}")
    dim, mult, offset = src
    base = value * mult + offset

    if to_unit is None or resolve(to_unit) == src:
        return _round(base), dim.canonical

    dst = resolve(to_unit)
    if dst is None:
        raise UnitError(f"Unknown target unit: {to_unit!r}")
    if dst[0] is not dim:
        raise UnitError(
            f"Cannot convert {dim.name} ({from_unit}) to {dst[0].name} ({to_unit})"
        )
    _, dmult, doffset = dst
    return _round((base - doffset) / dmult), to_unit


def to_canonical(value: float, unit: str) -> Tuple[float, str, str]:
    """Convert to canonical units. Returns (value, unit, dimension_name)."""
    src = resolve(unit)
    if src is None:
        raise UnitError(f"Unknown unit: {unit!r}")
    dim, mult, offset = src
    return _round(value * mult + offset), dim.canonical, dim.name


def same_dimension(unit_a: str, unit_b: str) -> bool:
    a, b = resolve(unit_a), resolve(unit_b)
    return bool(a and b and a[0] is b[0])


def dimension_of(unit: str) -> Optional[str]:
    found = resolve(unit)
    return found[0].name if found else None


def values_agree(
    value_a: float, unit_a: str, value_b: float, unit_b: str, tolerance: float = 0.02
) -> bool:
    """
    Do two measurements describe the same quantity within `tolerance`
    (relative)? Used for cross-source consistency checks: a catalogue saying
    3 HP and a datasheet saying 2.2 kW are consistent, not contradictory.
    """
    if not same_dimension(unit_a, unit_b):
        return False
    a, _, _ = to_canonical(value_a, unit_a)
    b, _, _ = to_canonical(value_b, unit_b)
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    if scale == 0:
        return True
    return abs(a - b) / scale <= tolerance


def format_value(value: float, unit: str) -> str:
    """Render a measurement without trailing float noise."""
    if isinstance(value, float) and value.is_integer():
        return f"{int(value)} {unit}".strip()
    return f"{value:g} {unit}".strip()


def _round(value: float, places: int = 10) -> float:
    rounded = round(value, places)
    return int(rounded) if isinstance(rounded, float) and rounded.is_integer() else rounded
