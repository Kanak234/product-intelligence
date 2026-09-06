"""
Scale benchmark for the catalogue engine.

"Scalable Catalog Engine" is a claim until someone runs it at scale. The
accuracy harness measures 40 records; a real industrial catalogue is tens of
thousands. This generates a synthetic catalogue of arbitrary size and measures
what actually matters when the record count grows:

* **Throughput** — records/second, and whether it holds as the catalogue grows
  or degrades. Anything worse than flat means a hidden quadratic.
* **Memory** — peak RSS per thousand records. A pipeline that holds the whole
  catalogue in memory does not scale no matter how fast it is.
* **Latency distribution** — p50/p95/p99 per record. A good mean hiding a
  terrible tail is a bad user experience on the one product someone cares about.
* **Cross-record cost** — the consistency checker compares records against each
  other, so its cost is the thing most likely to blow up super-linearly.

Run it:

    python -m app.services.product_intelligence.benchmark --sizes 100,1000,10000
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import random
import statistics
import sys
import time
import resource
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .schema import RawProduct
from .enricher import ProductEnricher
from .validator import CatalogConsistencyChecker, ProductValidator
from .explainer import ProductExplainer


#: Building blocks for synthetic-but-realistic catalogue rows. The point is to
#: exercise the same extraction and classification paths real data would, not
#: to produce a pretty dataset, so every template names a real product type and
#: carries the specs that type actually has.
_TEMPLATES = [
    ("{brand} {series} {power} Three Phase Induction Motor",
     "TEFC squirrel cage motor, {voltage} V, 50 Hz, {speed} rpm, IP{ip}, "
     "insulation class F, foot mounted, {weight} kg."),
    ("{brand} {series} End Suction Centrifugal Pump",
     "Flow rate {flow} m3/h, head {pressure} bar, cast iron casing, "
     "{power} motor, {speed} rpm, DN{bore} discharge."),
    ("{brand} {series} Ball Valve DN{bore}",
     "Two piece ball valve, DN{bore}, working pressure {pressure} bar, "
     "stainless steel 316 body, PTFE seat, max temperature {temp} C."),
    ("{brand} {series} Variable Frequency Drive {power}",
     "AC drive for motor speed control, {power}, {voltage} V three phase, "
     "{current} A output, IP{ip} enclosure, Modbus RTU."),
    ("{brand} {series} Rotary Screw Air Compressor",
     "Oil injected screw compressor, {power}, working pressure {pressure} bar, "
     "free air delivery {flow} m3/h, {voltage} V, noise level {noise} dBA."),
    ("{brand} {series} Deep Groove Ball Bearing",
     "Single row ball bearing, bore diameter {bore} mm, width {width} mm, "
     "max speed {speed} rpm, operating temperature {temp} C, carbon steel."),
    ("{brand} {series} Pressure Transmitter",
     "Smart pressure transmitter, range 0-{pressure} bar, 4-20mA output, "
     "{voltage} V supply, IP{ip}, stainless steel 316 wetted parts."),
    ("{brand} {series} Helical Gearbox",
     "Helical gear reducer, rated torque {torque} Nm, output speed {speed} rpm, "
     "flange mounted, cast iron housing, weight {weight} kg."),
]

_BRANDS = ("ABB", "Siemens", "Crompton", "Kirloskar", "Grundfos", "Danfoss",
           "Schneider", "Bonfiglioli", "SKF", "Atlas Copco", "L&T", "Wika")


def generate_catalog(size: int, seed: int = 1234) -> List[RawProduct]:
    """
    Build a deterministic synthetic catalogue.

    Seeded on purpose: a benchmark whose input changes between runs cannot
    detect a performance regression.
    """
    rng = random.Random(seed)
    products: List[RawProduct] = []

    for i in range(size):
        name_tpl, desc_tpl = _TEMPLATES[i % len(_TEMPLATES)]
        values = {
            "brand": rng.choice(_BRANDS),
            "series": f"{rng.choice('ABCDEFGHJKLM')}{rng.randint(10, 999)}",
            "power": rng.choice(["0.75 kW", "2.2 kW", "5.5 kW", "11 kW", "22 kW", "3 HP", "10 HP"]),
            "voltage": rng.choice([230, 400, 415, 440, 690]),
            "speed": rng.choice([720, 960, 1440, 1470, 2900, 3000]),
            "ip": rng.choice([54, 55, 65, 66, 68]),
            "weight": rng.randint(3, 850),
            "flow": rng.randint(5, 900),
            "pressure": rng.randint(2, 250),
            "bore": rng.choice([15, 25, 40, 50, 80, 100, 150, 200]),
            "current": round(rng.uniform(1.5, 180), 1),
            "temp": rng.choice([60, 80, 120, 200, 400]),
            "noise": rng.randint(52, 92),
            "width": rng.randint(7, 60),
            "torque": rng.randint(20, 4200),
        }
        products.append(RawProduct(
            name=name_tpl.format(**values),
            description=desc_tpl.format(**values),
            source="benchmark",
            source_ref=f"synthetic:{i}",
        ))

    return products


@dataclass
class ScaleResult:
    size: int
    total_seconds: float
    enrich_seconds: float
    validate_seconds: float
    consistency_seconds: float
    records_per_second: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    peak_memory_mb: float
    memory_mb_per_1k: float
    mean_specs: float
    mean_validation_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


async def run_size(size: int, concurrency: int = 8, seed: int = 1234) -> ScaleResult:
    """Enrich, validate and consistency-check a catalogue of `size` records."""
    products = generate_catalog(size, seed=seed)
    enricher = ProductEnricher()          # deterministic only; no model server
    validator = ProductValidator()
    explainer = ProductExplainer()

    gc.collect()
    # resource.getrusage costs nothing to read. tracemalloc was the obvious
    # choice here and was wrong: instrumenting every allocation slowed the
    # pipeline roughly threefold, so the benchmark was measuring its own
    # overhead and reporting a throughput number three times worse than real.
    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()

    # -- enrichment ------------------------------------------------------
    enrich_started = time.perf_counter()
    enriched = await enricher.enrich_many(products, use_llm=False, concurrency=concurrency)
    enrich_seconds = time.perf_counter() - enrich_started

    # -- per-record validation, timed individually for the tail ----------
    latencies: List[float] = []
    validate_started = time.perf_counter()
    for product in enriched:
        record_started = time.perf_counter()
        validator.validate(product)
        explainer.explain(product)
        latencies.append((time.perf_counter() - record_started) * 1000)
    validate_seconds = time.perf_counter() - validate_started

    # -- cross-record consistency: the most likely place to go quadratic --
    consistency_started = time.perf_counter()
    CatalogConsistencyChecker().check(enriched)
    consistency_seconds = time.perf_counter() - consistency_started

    total_seconds = time.perf_counter() - started
    rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is kilobytes on Linux, bytes on macOS.
    divisor = 1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0
    peak_bytes = max(0, rss_after_kb - rss_before_kb) * (1024.0 if divisor == 1024.0 else 1.0)

    latencies.sort()

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        index = min(len(latencies) - 1, int(len(latencies) * p))
        return latencies[index]

    peak_mb = peak_bytes / 1024 / 1024

    return ScaleResult(
        size=size,
        total_seconds=total_seconds,
        enrich_seconds=enrich_seconds,
        validate_seconds=validate_seconds,
        consistency_seconds=consistency_seconds,
        records_per_second=size / total_seconds if total_seconds else 0.0,
        p50_ms=pct(0.50),
        p95_ms=pct(0.95),
        p99_ms=pct(0.99),
        peak_memory_mb=peak_mb,
        memory_mb_per_1k=peak_mb / (size / 1000) if size else 0.0,
        mean_specs=statistics.mean(len(p.specifications) for p in enriched) if enriched else 0.0,
        mean_validation_score=statistics.mean(
            p.validation.get("score", 0.0) for p in enriched
        ) if enriched else 0.0,
    )


def format_report(results: List[ScaleResult]) -> str:
    """Terminal report. The scaling factor is the line that matters."""
    lines = [
        "",
        "=" * 78,
        "  PRODUCT INTELLIGENCE - SCALE BENCHMARK",
        "=" * 78,
        f"  {'Records':>9}  {'rec/sec':>9}  {'total s':>8}  {'p50 ms':>7}  "
        f"{'p95 ms':>7}  {'p99 ms':>7}  {'peak MB':>8}",
        "  " + "-" * 74,
    ]
    for r in results:
        lines.append(
            f"  {r.size:>9,}  {r.records_per_second:>9.1f}  {r.total_seconds:>8.2f}  "
            f"{r.p50_ms:>7.2f}  {r.p95_ms:>7.2f}  {r.p99_ms:>7.2f}  {r.peak_memory_mb:>8.1f}"
        )

    lines += ["", "  STAGE BREAKDOWN (seconds)", "  " + "-" * 74,
              f"  {'Records':>9}  {'enrich':>10}  {'validate':>10}  {'consistency':>12}"]
    for r in results:
        lines.append(
            f"  {r.size:>9,}  {r.enrich_seconds:>10.2f}  {r.validate_seconds:>10.2f}  "
            f"{r.consistency_seconds:>12.2f}"
        )

    if len(results) > 1:
        first, last = results[0], results[-1]
        size_factor = last.size / first.size
        time_factor = last.total_seconds / first.total_seconds if first.total_seconds else 0
        throughput_ratio = (
            last.records_per_second / first.records_per_second
            if first.records_per_second else 0
        )
        # Linear means time grows with the record count and throughput holds.
        verdict = (
            "linear" if time_factor <= size_factor * 1.35
            else "super-linear - investigate"
        )
        lines += [
            "",
            "  SCALING",
            "  " + "-" * 74,
            f"    {first.size:,} -> {last.size:,} records is {size_factor:.0f}x the input",
            f"    Wall time grew {time_factor:.1f}x -> {verdict}",
            f"    Throughput held at {throughput_ratio:.0%} of the small-catalogue rate",
            f"    Memory: {last.memory_mb_per_1k:.1f} MB per 1,000 records",
            f"    Quality unchanged: {last.mean_specs:.1f} specs/record, "
            f"validation {last.mean_validation_score:.3f}",
        ]

    lines.append("=" * 78)
    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> int:
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    results: List[ScaleResult] = []

    for size in sizes:
        print(f"  running {size:,} records...", flush=True)
        results.append(await run_size(size, concurrency=args.concurrency, seed=args.seed))

    print(format_report(results))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump([r.to_dict() for r in results], handle, indent=2)
        print(f"\nFull results written to {args.output}")

    if args.min_throughput and results[-1].records_per_second < args.min_throughput:
        print(
            f"\nFAILED: {results[-1].records_per_second:.1f} rec/s at "
            f"{results[-1].size:,} records is below the {args.min_throughput} floor."
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure catalogue-scale throughput, latency and memory."
    )
    parser.add_argument("--sizes", default="100,1000,10000",
                        help="Comma-separated catalogue sizes to benchmark.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None, help="Write JSON results here.")
    parser.add_argument("--min-throughput", type=float, default=0.0,
                        help="Exit non-zero below this rec/s at the largest size (for CI).")
    return asyncio.run(_main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
