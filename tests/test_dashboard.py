from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "secom.data").is_file(), reason="needs cached SECOM data")
def test_dashboard_renders_demo_batch():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=300).run()
    assert not app.exception
    assert app.title[0].value == "Fab inspection monitor"
    assert [m.label for m in app.metric][:2] == ["Units in batch", "Inspection queue"]
    app.sidebar.slider[0].set_value(30).run()
    assert not app.exception
    assert app.metric[1].value == str(int(0.30 * 314))
