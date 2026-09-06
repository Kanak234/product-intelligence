"""
Tests for the product intelligence engine.

Deliberately dependency-light: the deterministic core imports nothing outside
the standard library, so these run in a bare environment with only pytest —
no database, no model server, no Docker.

    cd backend && pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from product_intelligence import (
    CatalogConsistencyChecker,
    ProductExplainer,
    ProductIntelligencePipeline,
    ProductValidator,
    RawProduct,
    SpecExtractor,
    TaxonomyClassifier,
    ingest,
)
from product_intelligence.enricher import (
    ProductEnricher,
    _normalise_key,
    _parse_json_response,
)
from product_intelligence.schema import EnrichedProduct, FieldValue, Origin
from product_intelligence import units


# =========================================================================
# units
# =========================================================================


class TestUnits:
    @pytest.mark.parametrize(
        "value,unit,expected,expected_unit",
        [
            (2.2, "kW", 2200, "W"),
            (3, "HP", 2237.099616, "W"),
            (1, "kV", 1000, "V"),
            (5, "cm", 50, "mm"),
            (1, "inch", 25.4, "mm"),
            (1, "t", 1000, "kg"),
            (1, "psi", 0.0689476, "bar"),
            (60, "LPM", 3.6, "m3/h"),
            (1, "rps", 60, "rpm"),
        ],
    )
    def test_converts_to_canonical(self, value, unit, expected, expected_unit):
        got, got_unit, _ = units.to_canonical(value, unit)
        assert got == pytest.approx(expected, rel=1e-6)
        assert got_unit == expected_unit

    def test_temperature_offsets_are_applied(self):
        assert units.to_canonical(273.15, "K")[0] == pytest.approx(0.0, abs=1e-9)
        assert units.to_canonical(212, "F")[0] == pytest.approx(100.0, abs=1e-6)

    def test_rejects_unknown_unit(self):
        with pytest.raises(units.UnitError):
            units.to_canonical(1, "furlongs")

    def test_rejects_cross_dimension_conversion(self):
        with pytest.raises(units.UnitError):
            units.convert(1, "kW", "bar")

    def test_hp_and_kw_are_recognised_as_the_same_quantity(self):
        # The whole point of normalisation: a catalogue saying 3 HP and a
        # datasheet saying 2.2 kW are not in conflict.
        assert units.values_agree(3, "HP", 2.2, "kW", tolerance=0.02)

    def test_genuinely_different_values_do_not_agree(self):
        assert not units.values_agree(3, "HP", 5.5, "kW")

    def test_unit_aliases_are_case_and_space_insensitive(self):
        assert units.to_canonical(1, "KW")[0] == 1000
        assert units.to_canonical(1, " kW ")[0] == 1000


# =========================================================================
# extraction
# =========================================================================


class TestExtractor:
    @pytest.fixture
    def extractor(self):
        return SpecExtractor()

    def test_extracts_and_normalises_power(self, extractor):
        specs = extractor.extract("Rated power 3 HP motor")
        assert specs["power"].value == pytest.approx(2237.1, rel=1e-4)
        assert specs["power"].unit == "W"

    def test_records_the_source_span(self, extractor):
        specs = extractor.extract("Motor rating: 2.2 kW at 415 V")
        evidence = specs["power"].evidence[0]
        assert evidence.kind == "text_span"
        assert evidence.rule_id.startswith("EXT.POWER")
        assert evidence.span is not None

    def test_extracted_values_are_marked_verifiable(self, extractor):
        specs = extractor.extract("415 V supply")
        assert specs["voltage"].origin is Origin.EXTRACTED

    @pytest.mark.parametrize(
        "text,field,expected",
        [
            ("IP66 rated enclosure", "ip_rating", "IP66"),
            ("IP 54 protection", "ip_rating", "IP54"),
            ("three phase supply", "phase", "3-phase"),
            ("single phase supply", "phase", "1-phase"),
            ("DN80 flanged", "nominal_bore", "DN80"),
            ("IE3 efficiency class", "efficiency_class", "IE3"),
            ("foot mounted design", "mounting", "Foot Mounted"),
        ],
    )
    def test_categorical_normalisation(self, extractor, text, field, expected):
        assert extractor.extract(text)[field].value == expected

    def test_material_vocabulary_collapses_synonyms(self, extractor):
        for text in ("SS 316 body", "AISI 316 wetted parts", "stainless steel 316"):
            assert extractor.extract(text)["material"].value == "Stainless Steel 316"

    def test_longest_material_alias_wins(self, extractor):
        # "stainless steel 316" must beat the shorter "stainless steel".
        assert extractor.extract("stainless steel 316 body")["material"].value == (
            "Stainless Steel 316"
        )

    def test_collects_certifications(self, extractor):
        certs = extractor.extract("CE marked, ISO 9001 certified, ATEX approved")["certifications"]
        assert set(certs.value) >= {"CE", "ISO 9001", "ATEX"}

    def test_negative_values_are_captured_not_dropped(self, extractor):
        # The validator can only flag impossible values the extractor surfaces.
        assert extractor.extract("weight -5 kg")["weight"].value == -5

    def test_wrong_dimension_unit_is_rejected(self, extractor):
        # "50 Hz" must never be read as a power figure.
        specs = extractor.extract("50 Hz supply")
        assert "power" not in specs
        assert specs["frequency"].value == 50

    def test_labelled_beats_bare_on_confidence(self, extractor):
        labelled = extractor.extract("Rated current: 12 A")["current"]
        bare = extractor.extract("24 VDC 12 A")["current"]
        assert labelled.confidence > bare.confidence

    def test_returns_nothing_for_empty_input(self, extractor):
        assert extractor.extract("") == {}

    def test_keywords_are_deterministic(self, extractor):
        text = "Industrial centrifugal pump with stainless steel impeller"
        assert extractor.extract_keywords(text) == extractor.extract_keywords(text)


# =========================================================================
# taxonomy
# =========================================================================


class TestTaxonomy:
    @pytest.fixture
    def classifier(self):
        return TaxonomyClassifier()

    @pytest.mark.parametrize(
        "name,description,expected",
        [
            ("ABB Induction Motor 5 HP", "three phase TEFC motor", "AC Induction Motor"),
            ("Kirloskar Centrifugal Pump", "end suction pump with impeller", "Centrifugal Pump"),
            ("Audco Ball Valve DN50", "two piece flanged ball valve", "Ball Valve"),
            ("SKF Deep Groove Ball Bearing", "single row bearing", "Bearing"),
            ("Danfoss VFD 15kW", "variable frequency drive", "Variable Frequency Drive"),
            ("Atlas Copco Screw Compressor", "oil injected air compressor", "Air Compressor"),
        ],
    )
    def test_classifies_common_products(self, classifier, name, description, expected):
        fv, category, _ = classifier.classify(name, description)
        assert category.name == expected
        assert fv.confidence > 0.3

    def test_negative_signals_prevent_a_wrong_match(self, classifier):
        # "gear pump" must not land in Centrifugal Pump despite "pump".
        _, category, _ = classifier.classify("Gear Pump GP-2", "positive displacement gear pump")
        assert category.name == "Positive Displacement Pump"

    def test_returns_ranked_alternatives(self, classifier):
        # A close-coupled pump-motor set genuinely matches several leaves, which
        # is exactly when the runner-ups need to be surfaced to the user.
        _, _, ranked = classifier.classify(
            "Pump Motor Assembly", "close coupled motor driven centrifugal pump unit"
        )
        assert len(ranked) > 1
        assert ranked[0].score >= ranked[1].score

    def test_unknown_product_falls_back_without_crashing(self, classifier):
        fv, category, _ = classifier.classify("Zorblax 3000", "an entirely unknown object")
        assert category.name == "General Industrial Equipment"
        assert fv.confidence < 0.35

    def test_whole_token_matching_only(self, classifier):
        # 'pumpkin' must not trigger the pump categories.
        _, category, _ = classifier.classify("Pumpkin Seed Oil", "food grade oil")
        assert "Pump" not in category.name

    def test_confidence_reflects_the_margin(self, classifier):
        clear, _, _ = classifier.classify(
            "Centrifugal Pump CP-200", "end suction centrifugal pump with volute casing"
        )
        vague, _, _ = classifier.classify("Industrial Unit", "equipment")
        assert clear.confidence > vague.confidence


# =========================================================================
# schema / trust ordering
# =========================================================================


class TestTrustOrdering:
    def test_extracted_beats_llm(self):
        extracted = FieldValue(2200, Origin.EXTRACTED, 0.8, "W")
        inferred = FieldValue(9999, Origin.LLM, 0.99, "W")
        assert extracted.beats(inferred)
        assert not inferred.beats(extracted)

    def test_input_beats_everything(self):
        supplied = FieldValue(415, Origin.INPUT, 0.5, "V")
        for origin in (Origin.EXTRACTED, Origin.DERIVED, Origin.TAXONOMY, Origin.LLM):
            assert supplied.beats(FieldValue(230, origin, 0.99, "V"))

    def test_confidence_is_capped_by_origin_trust(self):
        # An LLM value cannot claim certainty it has not earned.
        assert FieldValue(1, Origin.LLM, 0.99).confidence <= 0.65

    def test_set_spec_refuses_a_lower_trust_overwrite(self):
        product = EnrichedProduct(raw=RawProduct(name="test"))
        assert product.set_spec("power", FieldValue(2200, Origin.EXTRACTED, 0.9, "W"))
        assert not product.set_spec("power", FieldValue(5000, Origin.LLM, 0.6, "W"))
        assert product.specifications["power"].value == 2200


# =========================================================================
# validation
# =========================================================================


def _build(name: str, **specs) -> EnrichedProduct:
    """Assemble a product directly, bypassing extraction, for validator tests."""
    product = EnrichedProduct(raw=RawProduct(name=name))
    for key, (value, unit) in specs.items():
        product.set_spec(key, FieldValue(value, Origin.EXTRACTED, 0.9, unit))
    return product


class TestValidator:
    @pytest.fixture
    def validator(self):
        return ProductValidator()

    def test_flags_negative_measurements_as_critical(self, validator):
        product = _build("Test Motor", weight=(-5, "kg"))
        report = validator.validate(product)
        assert report["counts"]["critical"] >= 1
        assert not report["is_valid"]

    def test_catches_impossible_power_factor(self, validator):
        # 50 kW cannot be drawn from 230 V at 2 A.
        product = _build("Bad Record", power=(50000, "W"), voltage=(230, "V"), current=(2, "A"))
        report = validator.validate(product)
        rules = {f["rule_id"] for f in report["findings"]}
        assert "VAL.COH.POWER_TOO_HIGH" in rules

    def test_accepts_a_physically_sound_record(self, validator):
        # 2.2 kW at 415 V single-phase, 6 A -> pf ~0.88. Plausible.
        product = _build("Good Record", power=(2200, "W"), voltage=(415, "V"), current=(6, "A"))
        report = validator.validate(product)
        assert "VAL.COH.POWER_TOO_HIGH" not in {f["rule_id"] for f in report["findings"]}

    def test_catches_power_torque_speed_mismatch(self, validator):
        # 1000 Nm at 1500 rpm implies ~157 kW, not 1 kW.
        product = _build(
            "Impossible Gearbox",
            power=(1000, "W"), torque=(1000, "Nm"), rotational_speed=(1500, "rpm"),
        )
        report = validator.validate(product)
        assert "VAL.COH.POWER_TORQUE_SPEED" in {f["rule_id"] for f in report["findings"]}

    def test_consistent_power_torque_speed_passes(self, validator):
        # 14.6 Nm at 1440 rpm implies ~2.2 kW.
        product = _build(
            "Sound Motor",
            power=(2200, "W"), torque=(14.6, "Nm"), rotational_speed=(1440, "rpm"),
        )
        report = validator.validate(product)
        assert "VAL.COH.POWER_TORQUE_SPEED" not in {f["rule_id"] for f in report["findings"]}

    def test_flags_nonstandard_supply_voltage(self, validator):
        product = _build("Odd Motor", voltage=(317, "V"))
        report = validator.validate(product)
        assert "VAL.RANGE.NONSTANDARD_VOLTAGE" in {f["rule_id"] for f in report["findings"]}

    def test_standard_voltage_is_not_flagged(self, validator):
        product = _build("Normal Motor", voltage=(415, "V"))
        report = validator.validate(product)
        assert "VAL.RANGE.NONSTANDARD_VOLTAGE" not in {f["rule_id"] for f in report["findings"]}

    def test_empty_name_is_critical(self, validator):
        report = validator.validate(EnrichedProduct(raw=RawProduct(name="")))
        assert report["counts"]["critical"] >= 1

    def test_findings_explain_themselves(self, validator):
        product = _build("Bad", power=(50000, "W"), voltage=(230, "V"), current=(2, "A"))
        report = validator.validate(product)
        for finding in report["findings"]:
            assert finding["message"]
            assert finding["rule_id"]
            assert finding["severity"] in ("critical", "major", "minor", "info")

    def test_score_falls_as_findings_accumulate(self, validator):
        clean = validator.validate(_build("Clean", voltage=(415, "V")))
        broken = validator.validate(_build("Broken", weight=(-5, "kg"), voltage=(317, "V")))
        assert broken["score"] < clean["score"]


class TestCatalogConsistency:
    def test_detects_duplicates_across_records(self):
        products = [
            EnrichedProduct(raw=RawProduct(name="Ball Valve DN50")),
            EnrichedProduct(raw=RawProduct(name="ball valve dn50")),   # same after squashing
            EnrichedProduct(raw=RawProduct(name="Gate Valve DN80")),
        ]
        report = CatalogConsistencyChecker().check(products)
        assert len(report["duplicate_groups"]) == 1
        assert report["duplicate_groups"][0]["count"] == 2

    def test_detects_a_statistical_outlier(self):
        products = []
        for i, power in enumerate([2200, 2300, 2250, 2280, 2210, 900000]):
            product = _build(f"Motor {i}", power=(power, "W"))
            product.category = FieldValue("AC Induction Motor", Origin.TAXONOMY, 0.8)
            products.append(product)
        report = CatalogConsistencyChecker(z_threshold=1.9).check(products)
        assert any(o["value"] == 900000 for o in report["outliers"])

    def test_clean_catalog_scores_one(self):
        products = [_build(f"Valve {i}", pressure=(16, "bar")) for i in range(5)]
        assert CatalogConsistencyChecker().check(products)["catalog_consistency_score"] == 1.0


# =========================================================================
# pipeline
# =========================================================================


class TestPipeline:
    @pytest.fixture
    def pipeline(self):
        return ProductIntelligencePipeline()   # no LLM router

    def test_runs_without_a_model_server(self, pipeline):
        result = asyncio.run(pipeline.process(
            RawProduct(
                name="ABB 2.2kW Three Phase Induction Motor",
                description="415 V, 50 Hz, 1440 rpm, IP55, foot mounted",
            ),
            use_llm=False,
        ))
        assert result["category"]["value"] == "AC Induction Motor"
        assert result["specifications"]["power"]["value"] == 2200
        assert result["validation"]["status"] in ("valid", "invalid")
        assert result["explanation"]["summary"]

    def test_derives_torque_from_power_and_speed(self, pipeline):
        product = asyncio.run(pipeline.process_to_object(
            RawProduct(name="Motor", description="2.2 kW at 1440 rpm"), use_llm=False
        ))
        torque = product.specifications["torque"]
        assert torque.origin is Origin.DERIVED
        assert torque.value == pytest.approx(14.59, rel=0.02)

    def test_derived_fields_state_their_working(self, pipeline):
        product = asyncio.run(pipeline.process_to_object(
            RawProduct(name="Motor", description="2.2 kW at 1440 rpm"), use_llm=False
        ))
        assert "2*pi*n/60" in product.specifications["torque"].evidence[0].detail

    def test_source_attributes_outrank_text_extraction(self, pipeline):
        product = asyncio.run(pipeline.process_to_object(
            RawProduct(
                name="Motor",
                description="approximately 3 kW",
                attributes={"Power": "2.2 kW"},
            ),
            use_llm=False,
        ))
        assert product.specifications["power"].origin is Origin.INPUT
        assert product.specifications["power"].value == 2200

    def test_always_produces_a_description(self, pipeline):
        product = asyncio.run(pipeline.process_to_object(
            RawProduct(name="Ball Valve DN50"), use_llm=False
        ))
        assert product.enriched_description.value

    def test_batch_preserves_order_and_reports_consistency(self, pipeline):
        raws = [
            RawProduct(name="Motor A", description="2.2 kW 415 V"),
            RawProduct(name="Pump B", description="30 m3/h 4 kW"),
            RawProduct(name="Valve C", description="DN50 16 bar"),
        ]
        result = asyncio.run(pipeline.process_batch(raws, use_llm=False))
        assert [p["name"] for p in result["products"]] == ["Motor A", "Pump B", "Valve C"]
        assert result["summary"]["total"] == 3
        assert "catalog_consistency_score" in result["catalog_consistency"]

    def test_one_bad_record_does_not_kill_the_batch(self, pipeline):
        raws = [RawProduct(name="Good Motor 2.2kW"), RawProduct(name="")]
        result = asyncio.run(pipeline.process_batch(raws, use_llm=False))
        assert result["summary"]["total"] == 2

    def test_flat_export_shape(self, pipeline):
        product = asyncio.run(pipeline.process_to_object(
            RawProduct(name="Motor", description="2.2 kW 415 V"), use_llm=False
        ))
        row = product.to_flat_dict()
        assert row["name"] == "Motor"
        assert "spec.power" in row


# =========================================================================
# LLM merge behaviour (no server required)
# =========================================================================


class TestLLMMerge:
    def test_rejects_fields_outside_the_schema(self):
        enricher = ProductEnricher()
        product = EnrichedProduct(raw=RawProduct(name="Motor"))
        accepted, rejected = enricher._merge_llm_specs(
            product, {"vibe": {"value": "excellent"}, "voltage": {"value": 415, "unit": "V"}}
        )
        assert "voltage" in accepted
        assert any("vibe" in r for r in rejected)

    def test_cannot_overwrite_an_extracted_value(self):
        enricher = ProductEnricher()
        product = EnrichedProduct(raw=RawProduct(name="Motor"))
        product.set_spec("power", FieldValue(2200, Origin.EXTRACTED, 0.9, "W"))
        accepted, rejected = enricher._merge_llm_specs(
            product, {"power": {"value": 99, "unit": "kW"}}
        )
        assert "power" not in accepted
        assert product.specifications["power"].value == 2200

    def test_normalises_units_on_accepted_values(self):
        enricher = ProductEnricher()
        product = EnrichedProduct(raw=RawProduct(name="Motor"))
        enricher._merge_llm_specs(product, {"power": {"value": 5, "unit": "HP"}})
        assert product.specifications["power"].unit == "W"
        assert product.specifications["power"].value == pytest.approx(3728.5, rel=1e-3)

    def test_rejects_unparseable_units(self):
        enricher = ProductEnricher()
        product = EnrichedProduct(raw=RawProduct(name="Motor"))
        _, rejected = enricher._merge_llm_specs(product, {"power": {"value": 5, "unit": "zorks"}})
        assert rejected and "power" not in product.specifications

    @pytest.mark.parametrize(
        "raw,expected_key",
        [("Rated Power", "power"), ("RPM", "rotational_speed"), ("MOC", "material"),
         ("full load current", "current"), ("Body Material", "material")],
    )
    def test_key_aliases_map_onto_the_schema(self, raw, expected_key):
        assert _normalise_key(raw) == expected_key

    @pytest.mark.parametrize(
        "response",
        [
            '{"a": 1}',
            '```json\n{"a": 1}\n```',
            'Here is the JSON you asked for:\n{"a": 1}\nHope that helps!',
        ],
    )
    def test_parses_json_out_of_messy_model_output(self, response):
        assert _parse_json_response(response) == {"a": 1}

    def test_returns_none_when_there_is_no_json(self):
        assert _parse_json_response("I could not complete that request.") is None


# =========================================================================
# explainability
# =========================================================================


class TestExplainer:
    @pytest.fixture
    def explained(self):
        pipeline = ProductIntelligencePipeline()
        return asyncio.run(pipeline.process_to_object(
            RawProduct(
                name="ABB 2.2kW Motor",
                description="415 V, 50 Hz, 1440 rpm, IP55, three phase",
            ),
            use_llm=False,
        ))

    def test_every_field_carries_provenance(self, explained):
        rows = explained.explanation["field_provenance"]
        assert rows
        for row in rows:
            assert row["origin"]
            assert row["why"]
            assert row["evidence"]

    def test_extracted_fields_are_marked_verifiable(self, explained):
        rows = {r["field"]: r for r in explained.explanation["field_provenance"]}
        assert rows["voltage"]["verifiable"] is True

    def test_reasoning_chain_matches_the_stages_that_ran(self, explained):
        stages = [step["stage"] for step in explained.explanation["reasoning_chain"]]
        assert "extract" in stages and "classify" in stages and "derive" in stages

    def test_category_reasoning_names_the_deciding_signals(self, explained):
        reasoning = explained.explanation["category_reasoning"]
        assert reasoning["assigned"] == "AC Induction Motor"
        assert reasoning["why"]
        assert reasoning["evidence"]

    def test_trust_breakdown_sums_correctly(self, explained):
        breakdown = explained.explanation["trust_breakdown"]
        assert breakdown["total_fields"] == sum(breakdown["by_origin"].values())
        assert 0.0 <= breakdown["verifiable_ratio"] <= 1.0

    def test_review_queue_is_priority_sorted(self, explained):
        queue = explained.explanation["review_queue"]
        priorities = [item["priority"] for item in queue]
        assert priorities == sorted(priorities)

    def test_explanation_is_json_serialisable(self, explained):
        json.dumps(explained.explanation)   # must not raise


# =========================================================================
# ingestion
# =========================================================================


class TestIngestion:
    def test_reads_a_csv_catalogue(self):
        csv_text = (
            "Product Name,Description,Category,Voltage\n"
            "Motor A,2.2 kW three phase,AC Induction Motor,415 V\n"
            "Pump B,30 m3/h centrifugal,Centrifugal Pump,415 V\n"
        )
        result = ingest(csv_text, "catalog.csv")
        assert len(result.products) == 2
        assert result.products[0].name == "Motor A"
        assert result.products[0].attributes["Voltage"] == "415 V"

    def test_sniffs_a_semicolon_delimiter(self):
        result = ingest("name;description\nMotor;2.2 kW\n", "euro.csv")
        assert result.products[0].name == "Motor"

    def test_skips_rows_with_no_name_and_reports_them(self):
        result = ingest("name,description\n,orphan row\nMotor,2.2 kW\n", "c.csv")
        assert len(result.products) == 1
        assert len(result.skipped) == 1

    def test_rejects_a_file_with_no_name_column(self):
        from product_intelligence import IngestionError

        with pytest.raises(IngestionError):
            ingest("colour,size\nred,large\n", "bad.csv")

    def test_reads_a_json_array(self):
        payload = json.dumps([{"name": "Motor A", "description": "2.2 kW"}])
        assert ingest(payload, "products.json").products[0].name == "Motor A"

    def test_reads_jsonl(self):
        payload = '{"name": "A", "description": "x"}\n{"name": "B", "description": "y"}\n'
        assert len(ingest(payload, "products.jsonl").products) == 2

    def test_flattens_nested_json(self):
        payload = json.dumps([{"name": "Motor", "specs": {"voltage": "415 V"}}])
        product = ingest(payload, "p.json").products[0]
        assert "specs_voltage" in product.attributes

    def test_scrapes_a_product_page(self):
        html = """
        <html><head><title>Centrifugal Pump CP-200 | Acme</title></head>
        <body><h1>Centrifugal Pump CP-200</h1>
        <p>A robust end suction pump for industrial water transfer duties everywhere.</p>
        <table><tr><th>Flow Rate</th><td>30 m3/h</td></tr>
        <tr><th>Power</th><td>4 kW</td></tr></table>
        <script>var x = 1;</script></body></html>
        """
        product = ingest(html, "page.html").products[0]
        assert product.name == "Centrifugal Pump CP-200"
        assert product.attributes["Flow Rate"] == "30 m3/h"
        assert "var x" not in product.description

    def test_splits_a_document_on_headings(self):
        text = (
            "CENTRIFUGAL PUMP CP-200\n"
            "Flow: 30 m3/h\nPower: 4 kW\n\n"
            "BALL VALVE BV-50\n"
            "Size: DN50\nPressure: 16 bar\n"
        )
        result = ingest(text, "catalog.txt")
        assert len(result.products) == 2
        assert result.products[0].attributes["Flow"] == "30 m3/h"

    def test_dispatches_on_content_when_the_extension_is_unknown(self):
        assert ingest('[{"name": "Motor"}]', "mystery").products[0].name == "Motor"

    def test_ingested_records_flow_through_the_pipeline(self):
        result = ingest(
            "name,description\nABB Motor,2.2 kW 415 V 1440 rpm three phase\n", "c.csv"
        )
        pipeline = ProductIntelligencePipeline()
        enriched = asyncio.run(pipeline.process_batch(result.products, use_llm=False))
        assert enriched["products"][0]["category"]["value"] == "AC Induction Motor"


# =========================================================================
# evaluation harness + regression thresholds
# =========================================================================


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"


class TestEvaluation:
    @pytest.mark.skipif(
        not (DATA_DIR / "ground_truth.jsonl").exists(), reason="dataset not present"
    )
    def test_meets_the_baseline_thresholds(self):
        """
        Regression guard. These thresholds are the numbers we quote, so a
        change that degrades them should fail CI rather than ship quietly.
        """
        from product_intelligence.evaluation import Evaluator, load_dataset

        cases = load_dataset(DATA_DIR / "ground_truth.jsonl")
        report = asyncio.run(Evaluator().run(cases, use_llm=False))
        summary = report["summary"]

        assert summary["category_accuracy"] >= 0.90
        assert summary["spec_f1"] >= 0.90
        assert summary["hallucination_rate"] <= 0.05

    @pytest.mark.skipif(
        not (DATA_DIR / "hard_cases.jsonl").exists(), reason="dataset not present"
    )
    def test_holds_up_on_the_adversarial_set(self):
        from product_intelligence.evaluation import Evaluator, load_dataset

        cases = load_dataset(DATA_DIR / "hard_cases.jsonl")
        report = asyncio.run(Evaluator().run(cases, use_llm=False))

        # Lower bar than the tuned set, on purpose: this set exists to be hard.
        assert report["summary"]["category_accuracy"] >= 0.80
        assert report["summary"]["category_top3_accuracy"] >= 0.90
        assert report["summary"]["spec_f1"] >= 0.85


# =========================================================================
# streaming at catalogue scale
# =========================================================================


class TestStreaming:
    """
    `process_stream` exists because `process_batch` holds the whole catalogue
    in memory — benchmarked near 950 MB at 50,000 records. These guard the
    properties that make streaming worth having.
    """

    @pytest.fixture
    def pipeline(self):
        return ProductIntelligencePipeline()

    def _collect(self, pipeline, raws, **kwargs):
        async def run():
            chunks = []
            async for chunk in pipeline.process_stream(raws, use_llm=False, **kwargs):
                chunks.append(chunk)
            return chunks

        return asyncio.run(run())

    def test_yields_every_record_exactly_once(self, pipeline):
        from product_intelligence.benchmark import generate_catalog

        raws = generate_catalog(120)
        chunks = self._collect(pipeline, raws, chunk_size=25)

        names = [p["name"] for c in chunks for p in c["products"]]
        assert len(names) == 120
        assert names == [r.name for r in raws], "order must be preserved"

    def test_chunk_size_is_respected(self, pipeline):
        from product_intelligence.benchmark import generate_catalog

        chunks = self._collect(pipeline, generate_catalog(45), chunk_size=20)
        assert [c["count"] for c in chunks] == [20, 20, 5]
        assert [c["offset"] for c in chunks] == [0, 20, 40]

    def test_streamed_records_match_the_batch_path(self, pipeline):
        """Streaming must change memory behaviour, not results."""
        from product_intelligence.benchmark import generate_catalog

        raws = generate_catalog(30)
        streamed = [p for c in self._collect(pipeline, raws, chunk_size=7) for p in c["products"]]
        batched = asyncio.run(pipeline.process_batch(raws, use_llm=False))["products"]

        assert len(streamed) == len(batched)
        for a, b in zip(streamed, batched):
            assert a["category"]["value"] == b["category"]["value"]
            assert a["specifications"].keys() == b["specifications"].keys()
            assert a["validation"]["score"] == b["validation"]["score"]

    def test_empty_input_yields_nothing(self, pipeline):
        assert self._collect(pipeline, []) == []

    def test_records_are_validated_and_explained(self, pipeline):
        from product_intelligence.benchmark import generate_catalog

        chunks = self._collect(pipeline, generate_catalog(10), chunk_size=5)
        for product in (p for c in chunks for p in c["products"]):
            assert product["validation"]["status"] in ("valid", "invalid")
            assert product["explanation"]["summary"]


class TestSyntheticCatalog:
    def test_generation_is_deterministic(self):
        from product_intelligence.benchmark import generate_catalog

        assert [p.name for p in generate_catalog(50)] == [
            p.name for p in generate_catalog(50)
        ]

    def test_generated_records_are_classifiable(self):
        """A benchmark on unclassifiable junk would measure nothing useful."""
        from product_intelligence.benchmark import generate_catalog

        pipeline = ProductIntelligencePipeline()
        result = asyncio.run(pipeline.process_batch(generate_catalog(24), use_llm=False))
        uncategorised = [
            p for p in result["products"]
            if p["category"]["value"] == "General Industrial Equipment"
        ]
        assert not uncategorised, f"{len(uncategorised)} synthetic records failed to classify"
        assert result["summary"]["mean_completeness"] > 0.8
