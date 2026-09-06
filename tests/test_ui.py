"""
End-to-end tests for the Streamlit UI.

These drive the real app headlessly through Streamlit's AppTest harness: the
script runs, widgets are set, the button is pressed, and the rendered output is
inspected. A crash in rendering fails here rather than in front of an audience.

Skipped automatically if Streamlit is not installed, so the core suite still
runs in a bare environment.

    pytest tests/test_ui.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1", reason="streamlit not installed")
AppTest = streamlit_testing.AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
TIMEOUT = 120


@pytest.fixture(scope="module")
def app():
    """The app as first loaded, before any interaction."""
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


@pytest.fixture(scope="module")
def analysed():
    """The app after analysing the pre-filled motor example."""
    instance = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    instance.button[0].click().run()
    return instance


class TestInitialRender:
    def test_the_script_runs_without_raising(self, app):
        assert not app.exception

    def test_the_analyse_control_is_present(self, app):
        assert any("Analyse" in button.label for button in app.button)

    def test_the_form_fields_are_present(self, app):
        labels = [widget.label for widget in app.text_input]
        assert "Product name or title" in labels

    def test_an_example_is_prefilled_so_the_demo_starts_instantly(self, app):
        name = next(w for w in app.text_input if w.label == "Product name or title")
        assert name.value, "the first example should populate the name field"

    def test_no_results_are_shown_before_analysing(self, app):
        assert "single" not in app.session_state or not app.session_state["single"]


class TestAnalyseFlow:
    def test_pressing_analyse_does_not_crash(self, analysed):
        assert not analysed.exception

    def test_a_result_is_stored(self, analysed):
        assert analysed.session_state["single"]

    def test_the_motor_is_classified(self, analysed):
        assert analysed.session_state["single"]["category"]["value"] == "AC Induction Motor"

    def test_specifications_reach_the_page(self, analysed):
        result = analysed.session_state["single"]
        assert result["metrics"]["spec_count"] > 0

    def test_the_download_control_is_offered(self, analysed):
        assert analysed.download_button, "the record should be downloadable after analysis"

    def test_metrics_are_rendered_as_markdown(self, analysed):
        rendered = " ".join(block.value for block in analysed.markdown)
        assert "Traceable to source" in rendered
        assert "Validation" in rendered


class TestValidation:
    def test_an_empty_name_is_refused_with_a_readable_error(self):
        instance = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        name = next(w for w in instance.text_input if w.label == "Product name or title")
        name.set_value("").run()
        instance.button[0].click().run()

        assert not instance.exception
        assert instance.error, "an empty name should produce an error, not a blank page"
        assert "name" in instance.error[0].value.lower()

    def test_a_bare_name_with_no_description_still_works(self):
        instance = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        name = next(w for w in instance.text_input if w.label == "Product name or title")
        name.set_value("Unlabelled industrial part").run()
        description = next(w for w in instance.text_area if w.label == "Description")
        description.set_value("").run()
        instance.button[0].click().run()

        assert not instance.exception
        assert instance.session_state["single"]["name"] == "Unlabelled industrial part"

    def test_freeform_attributes_are_accepted(self):
        instance = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        attributes = next(w for w in instance.text_area if w.label == "Known attributes")
        attributes.set_value("Brand: ABB\nframe = 100L").run()
        instance.button[0].click().run()

        assert not instance.exception
        assert instance.session_state["single"]
