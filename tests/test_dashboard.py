from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "secom.data").is_file(), reason="needs cached SECOM data")
def test_dashboard_renders_demo_batch_and_reacts_to_controls():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=300).run()
    assert not app.exception
    assert app.title[0].value == "Fab inspection monitor"
    text = " ".join(m.value for m in app.markdown)
    assert "<b>62 of 314</b>" in text and "<b>5 of 17</b>" in text
    assert [t.label for t in app.tabs][0] == "Queue · 62"
    assert "Paul Lai" in text and "holai@ucsd.edu" in text and "UC San Diego" in text
    assert app.expander[0].label == "How to use this monitor"
    assert "sensor_000" in " ".join(m.value for m in app.expander[0].markdown)

    app.sidebar.slider[0].set_value(30).run()
    assert not app.exception
    assert "<b>94 of 314</b>" in " ".join(m.value for m in app.markdown)

    first = app.session_state["unit_pick"]
    app.button[1].click().run()  # Next
    assert not app.exception and app.session_state["unit_pick"] != first

    app.sidebar.radio[0].set_value("mspc").run()  # still at the 30% budget: MSPC finds 3
    assert not app.exception
    assert "<b>3 of 17</b>" in " ".join(m.value for m in app.markdown)
