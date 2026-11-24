"""End-to-end Streamlit check that a physical ladder lane is never measured."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).parent


def test_ladder_lane_keeps_its_center_but_never_gets_an_roi_or_measurement():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception

    sample = (ROOT / "sample_blot.tif").read_bytes()
    app.file_uploader[0].upload(
        "sample_blot.tif", sample, "image/tiff"
    ).run(timeout=60)
    assert not app.exception

    app.text_area[1].set_value("Ladder\nWT\nKO").run(timeout=60)
    assert not app.exception
    assert app.multiselect[0].value == ["Ladder"]

    layout = app.session_state["layouts"]["sample_blot.tif"]
    assert len(layout["lane_x"]) == 3
    assert {roi["lane"] for roi in layout["rois"]} == {"WT", "KO"}
    assert all(roi["lane"] != "Ladder" for roi in layout["rois"])

    quantify = next(
        button for button in app.button
        if button.label == "Run densitometry on all blots"
    )
    quantify.click().run(timeout=60)
    assert not app.exception

    measurements = app.session_state["measurements"]
    assert set(measurements["lane"]) == {"WT", "KO"}
    assert "Ladder" not in set(measurements["lane"])

