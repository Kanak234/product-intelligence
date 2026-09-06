"""
Tests for the prototype's core layer.

These cover the code the UI depends on: input parsing, pipeline invocation
from synchronous context, provenance accounting, and export. Runs with only
pytest installed — no Streamlit, no database, no model server.

    pytest tests/ -v
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from app_core import (
    ORIGIN_STYLE,
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
from product_intelligence import Origin

MOTOR = "ABB M2BAX 2.2kW 415V 3-phase 1440rpm IE3 induction motor"
MOTOR_DESCRIPTION = "Foot mounted TEFC squirrel cage motor, IP55, frame 100L, 50Hz"


@pytest.fixture(scope="module")
def result():
    """One analysed motor, shared across the read-only assertions below."""
    return analyse_product(build_raw(MOTOR, MOTOR_DESCRIPTION))


@pytest.fixture(scope="module")
def batch():
    """A small mixed catalogue: motor, breaker, pump."""
    return analyse_catalogue(
        [
            build_raw(MOTOR, MOTOR_DESCRIPTION),
            build_raw("Siemens 3RV2011 circuit breaker 690V 10A", "IP20, 3 pole"),
            build_raw("Grundfos CR 5-10 pump 2.2kW", "Stainless steel, 415V 50Hz"),
        ]
    )



class TestParseAttributes:
    def test_colon_separated_pairs(self):
        assert parse_attributes("Brand: ABB\nSeries: M2BAX") == {"Brand": "ABB", "Series": "M2BAX"}

    def test_equals_is_also_accepted(self):
        assert parse_attributes("frame = 100L") == {"frame": "100L"}

    def test_lines_without_a_separator_are_skipped_not_guessed(self):
        """A typo must never become a silent fake specification."""
        assert parse_attributes("Brand: ABB\nthis line is prose") == {"Brand": "ABB"}

    def test_blank_and_comment_lines_ignored(self):
        assert parse_attributes("\n# a note\nBrand: ABB\n\n") == {"Brand": "ABB"}

    def test_later_value_wins_for_a_repeated_key(self):
        assert parse_attributes("Brand: ABB\nBrand: Siemens") == {"Brand": "Siemens"}

    def test_value_containing_a_colon_survives_intact(self):
        assert parse_attributes("Note: rated at 415V: 50Hz") == {"Note": "rated at 415V: 50Hz"}

    def test_empty_input_is_an_empty_map(self):
        assert parse_attributes("") == {}
        assert parse_attributes(None) == {}


class TestBuildRaw:
    def test_whitespace_is_trimmed(self):
        raw = build_raw("  Motor  ", "  desc  ", " cat ")
        assert raw.name == "Motor"
        assert raw.description == "desc"
        assert raw.category == "cat"

    def test_attributes_are_copied_not_aliased(self):
        source = {"Brand": "ABB"}
        raw = build_raw("Motor", attributes=source)
        source["Brand"] = "Siemens"
        assert raw.attributes["Brand"] == "ABB"

    def test_defaults_are_safe(self):
        raw = build_raw("Motor")
        assert raw.attributes == {}
        assert raw.source == "manual"


class TestAnalyseProduct:
    def test_runs_from_synchronous_code(self, result):
        """The UI calls this without an event loop; it must not raise."""
        assert isinstance(result, dict)

    def test_classifies_the_motor(self, result):
        assert result["category"]["value"] == "AC Induction Motor"
        assert result["category_path"][0] == "Industrial Equipment"

    def test_extracts_the_ratings_from_the_name(self, result):
        specs = result["specifications"]
        assert specs["power"]["value"] == 2200
        assert specs["power"]["unit"] == "W"
        assert specs["voltage"]["value"] == 415

    def test_derives_torque_that_was_never_stated(self, result):
        """Enrichment must add real value, not just reformat the input."""
        assert "torque" in result["specifications"]
        assert result["specifications"]["torque"]["origin"] == Origin.DERIVED.value

    def test_nothing_is_attributed_to_a_model(self, result):
        """No model server runs here, so no field may claim LLM provenance."""
        origins = {field["origin"] for field in result["specifications"].values()}
        assert Origin.LLM.value not in origins

    def test_every_value_carries_evidence(self, result):
        for name, field in result["specifications"].items():
            assert field["evidence"], f"{name} has no evidence"

    def test_validation_and_explanation_are_present(self, result):
        assert "score" in result["validation"]
        assert result["explanation"]["summary"]

    def test_trace_is_included_when_asked(self):
        traced = analyse_product(build_raw(MOTOR), include_trace=True)
        assert traced["pipeline_trace"]

    def test_is_deterministic_across_runs(self):
        first = analyse_product(build_raw(MOTOR, MOTOR_DESCRIPTION))
        second = analyse_product(build_raw(MOTOR, MOTOR_DESCRIPTION))
        assert first["specifications"] == second["specifications"]
        assert first["validation"]["score"] == second["validation"]["score"]

    def test_a_bare_name_still_produces_a_record(self):
        result = analyse_product(build_raw("Unlabelled part"))
        assert result["name"] == "Unlabelled part"
        assert "validation" in result


class TestProvenance:
    def test_breakdown_counts_every_specification(self, result):
        counts = provenance_breakdown(result)
        assert sum(counts.values()) == len(result["specifications"])

    def test_traceability_is_a_fraction(self, result):
        assert 0.0 <= traceability(result) <= 1.0

    def test_deterministic_run_is_fully_traceable(self, result):
        """Extracted and derived only, so nothing is unaccounted for."""
        assert traceability(result) == 1.0

    def test_empty_record_scores_zero_not_one(self):
        """A record with nothing in it must not read as perfectly traceable."""
        assert traceability({"specifications": {}}) == 0.0
        assert traceability({}) == 0.0

    def test_model_inferred_values_lower_the_score(self):
        synthetic = {
            "specifications": {
                "a": {"origin": Origin.EXTRACTED.value},
                "b": {"origin": Origin.LLM.value},
            }
        }
        assert traceability(synthetic) == 0.5


class TestSpecRows:
    def test_rows_are_ordered_by_provenance_strength(self):
        result = analyse_product(build_raw(MOTOR, MOTOR_DESCRIPTION))
        rows = spec_rows(result)
        trusts = [row["_trust"] for row in rows]
        assert trusts == sorted(trusts, reverse=True)

    def test_unit_is_joined_to_the_value_for_display(self):
        rows = spec_rows(analyse_product(build_raw(MOTOR)))
        power = next(row for row in rows if row["field"] == "power")
        assert power["value"] == "2200 W"

    def test_handles_a_record_with_no_specifications(self):
        assert spec_rows({"specifications": {}}) == []


class TestStyling:
    def test_every_origin_has_a_label_and_colour(self):
        for origin in Origin:
            label, colour = origin_style(origin.value)
            assert label and colour.startswith("#")

    def test_unknown_origin_falls_back_without_raising(self):
        label, colour = origin_style("something-new")
        assert colour.startswith("#")

    def test_missing_origin_is_handled(self):
        assert origin_style(None)[0] == "Unknown"

    def test_severities_have_styles(self):
        for severity in ("critical", "major", "minor", "info"):
            label, colour = severity_style(severity)
            assert label and colour.startswith("#")

    def test_style_map_covers_the_enum_exactly(self):
        assert set(ORIGIN_STYLE) == {origin.value for origin in Origin}


class TestCatalogue:
    def test_processes_every_record(self, batch):
        assert batch["summary"]["total"] == 3
        assert len(batch["products"]) == 3

    def test_includes_cross_record_consistency(self, batch):
        assert "catalog_consistency" in batch

    def test_summary_scores_are_in_range(self, batch):
        assert 0.0 <= batch["summary"]["mean_validation_score"] <= 1.0
        assert 0.0 <= batch["summary"]["mean_completeness"] <= 1.0

    def test_products_are_classified_differently(self, batch):
        categories = {(product["category"] or {}).get("value") for product in batch["products"]}
        assert len(categories) > 1


class TestIngestUpload:
    def test_reads_a_csv(self):
        content = b"name,description\nABB 2.2kW motor,TEFC IP55\nSiemens breaker,IP20\n"
        result = ingest_upload(content, "catalogue.csv")
        assert len(result.products) == 2
        assert result.products[0].name == "ABB 2.2kW motor"

    def test_reads_json(self):
        content = json.dumps([{"name": "ABB motor", "description": "2.2kW"}]).encode()
        result = ingest_upload(content, "catalogue.json")
        assert len(result.products) == 1

    def test_records_the_source_reference(self):
        result = ingest_upload(b"name\nWidget\n", "supplier-feed.csv")
        assert result.source_ref == "supplier-feed.csv"

    def test_ingested_products_flow_into_the_pipeline(self):
        result = ingest_upload(b"name\nABB M2BAX 2.2kW 415V motor\n", "c.csv")
        analysed = analyse_product(result.products[0])
        assert analysed["specifications"]["power"]["value"] == 2200


class TestExport:
    def test_csv_has_a_header_and_a_row_per_product(self):
        rows = [{"name": "A", "category": "Motor"}, {"name": "B", "category": "Pump"}]
        parsed = list(csv.DictReader(io.StringIO(flat_rows_to_csv(rows).decode())))
        assert len(parsed) == 2
        assert parsed[0]["name"] == "A"

    def test_columns_are_the_union_across_rows(self):
        """A later record's extra specification must not be silently dropped."""
        rows = [{"name": "A"}, {"name": "B", "spec.power": "2200 W"}]
        text = flat_rows_to_csv(rows).decode()
        assert "spec.power" in text.splitlines()[0]
        assert list(csv.DictReader(io.StringIO(text)))[1]["spec.power"] == "2200 W"

    def test_missing_values_become_empty_not_missing_columns(self):
        rows = [{"name": "A"}, {"name": "B", "spec.power": "2200 W"}]
        parsed = list(csv.DictReader(io.StringIO(flat_rows_to_csv(rows).decode())))
        assert parsed[0]["spec.power"] == ""

    def test_empty_export_is_empty_bytes(self):
        assert flat_rows_to_csv([]) == b""

    def test_json_export_round_trips(self):
        payload = {"name": "A", "score": 0.94}
        assert json.loads(results_to_json(payload).decode()) == payload

    def test_json_export_survives_unserialisable_values(self):
        assert b"Origin.LLM" in results_to_json({"origin": Origin.LLM}) or b"llm" in results_to_json(
            {"origin": Origin.LLM}
        )

    def test_full_result_exports_without_error(self):
        result = analyse_product(build_raw(MOTOR, MOTOR_DESCRIPTION))
        assert json.loads(results_to_json(result).decode())["name"] == MOTOR


class TestFrameSizeIsNotAVolume:
    """
    Regression guard for a false positive found while reviewing the UI.

    IEC motor frame designations ending in L (100L, 160L, 180L) were being read
    by the bare volume rule as a measurement in litres, so most motor records
    gained a specification that was never in the source.
    """

    @pytest.mark.parametrize(
        "description",
        [
            "Foot mounted TEFC motor, frame 100L",
            "frame size 180L",
            "Frame: 160L IP55",
            "frame size: 225L",
        ],
    )
    def test_frame_designations_do_not_become_a_volume(self, description):
        result = analyse_product(build_raw("ABB motor 2.2kW 415V", description))
        assert "volume" not in result["specifications"], description

    @pytest.mark.parametrize(
        "description,expected",
        [
            ("Tank capacity 100L", 100),
            ("20 L receiver", 20),
            ("5L drum", 5),
            ("Air receiver volume 500 litres", 500),
        ],
    )
    def test_real_volumes_are_still_read(self, description, expected):
        result = analyse_product(build_raw("Compressor 2.2kW", description))
        assert result["specifications"]["volume"]["value"] == expected
