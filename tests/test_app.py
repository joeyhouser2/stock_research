"""Smoke test for the Streamlit dashboard.

Streamlit only executes the script when a client connects, so duplicate-widget-id
bugs don't show up at server boot — they need a full render. AppTest does exactly
that headlessly, catching the StreamlitDuplicateElementId class of error. No
network or GPU: no Run button is clicked, so only widget construction is exercised.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_dashboard_renders_without_errors():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    # All six tools are present as tabs.
    assert len(at.button) >= 6
