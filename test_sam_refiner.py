"""Pure tests for optional SAM ROI refinement; no SAM package or weights needed."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from wblot import sam_refiner


def test_sam_refiner_import_is_lazy():
    code = (
        "import sys; import wblot.sam_refiner; "
        "assert 'torch' not in sys.modules; "
        "assert 'sam2' not in sys.modules; print('ok')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "ok"


def test_prepare_roi_crop_is_rgb_local_and_clamped():
    image = np.arange(80 * 120, dtype=np.float32).reshape(80, 120)
    image[0, 0] = np.nan
    crop, prompt, bounds = sam_refiner.prepare_roi_crop(
        image,
        {"x": 0, "y": 1, "w": 20, "h": 8},
        context_scale=4.0,
    )
    assert crop.dtype == np.uint8
    assert crop.ndim == 3 and crop.shape[2] == 3
    assert bounds[0] == 0
    assert bounds[1] == 0
    assert prompt.tolist() == [0.0, 1.0, 20.0, 9.0]
    assert np.isfinite(crop).all()


def test_recenter_preserves_locked_size_and_invalidates_approval():
    result = sam_refiner.RefinementResult(
        backend=sam_refiner.BACKEND,
        checkpoint_sha256="a" * 64,
        device="cpu",
        crop_box_xyxy=(0, 0, 100, 80),
        prompt_box_xyxy=(20.0, 20.0, 40.0, 30.0),
        old_box_xywh=(20, 20, 20, 10),
        proposed_center_xy=(3.0, 4.0),
        mask_box_xyxy=(0, 1, 8, 7),
        mask_score=0.91,
        mask_area=30,
        displacement_pixels=4.0,
    )
    original = {
        "roi_id": "r1",
        "x": 20,
        "y": 20,
        "w": 20,
        "h": 10,
        "source": "gelgenie",
        "annotation_approved": True,
        "review_required": False,
    }
    updated = sam_refiner.recenter_fixed_roi(
        original,
        result,
        {"w": 24, "h": 12},
        (80, 100),
    )
    assert (updated["x"], updated["y"]) == (0, 0)
    assert (updated["w"], updated["h"]) == (24, 12)
    assert updated["annotation_approved"] is False
    assert updated["review_required"] is True
    assert updated["source"] == "sam2_refined"
    assert original["x"] == 20


def test_refine_box_uses_one_box_prompt_and_nearest_component(monkeypatch):
    class FakePredictor:
        def __init__(self):
            self.image = None
            self.box = None
            self.multimask = None

        def set_image(self, image):
            self.image = image

        def predict(self, *, box, multimask_output):
            self.box = np.asarray(box)
            self.multimask = multimask_output
            mask = np.zeros(self.image.shape[:2], dtype=bool)
            center_x = int(round((self.box[0] + self.box[2]) / 2))
            center_y = int(round((self.box[1] + self.box[3]) / 2))
            mask[center_y - 2:center_y + 3, center_x - 8:center_x + 9] = True
            mask[2:5, 2:5] = True
            return mask[None, :, :], np.asarray([0.88]), None

    predictor = FakePredictor()
    monkeypatch.setattr(
        sam_refiner,
        "_verified_checkpoint",
        lambda path: (
            Path("/tmp/fake.pt"),
            {"sha256": "b" * 64, "verified_official": True},
        ),
    )
    monkeypatch.setattr(sam_refiner, "_resolve_device", lambda device: "cpu")
    monkeypatch.setattr(
        sam_refiner,
        "_load_predictor",
        lambda path, digest, device: predictor,
    )
    roi = {"x": 40, "y": 30, "w": 30, "h": 10}
    result = sam_refiner.refine_box(
        np.arange(120 * 160, dtype=np.uint8).reshape(120, 160),
        roi,
        "ignored.pt",
    )
    assert predictor.image.ndim == 3
    assert predictor.box.shape == (4,)
    assert predictor.multimask is False
    assert result.mask_score == pytest.approx(0.88)
    assert result.mask_area == 85
    assert result.displacement_pixels < 1.0


def test_refine_box_rejects_implausibly_displaced_mask(monkeypatch):
    class FarPredictor:
        def set_image(self, image):
            self.shape = image.shape

        def predict(self, *, box, multimask_output):
            mask = np.zeros(self.shape[:2], dtype=bool)
            mask[-8:-2, -18:-2] = True
            return mask[None, :, :], np.asarray([0.99]), None

    monkeypatch.setattr(
        sam_refiner,
        "_verified_checkpoint",
        lambda path: (
            Path("/tmp/fake.pt"),
            {"sha256": "c" * 64, "verified_official": True},
        ),
    )
    monkeypatch.setattr(sam_refiner, "_resolve_device", lambda device: "cpu")
    monkeypatch.setattr(
        sam_refiner,
        "_load_predictor",
        lambda path, digest, device: FarPredictor(),
    )
    with pytest.raises(sam_refiner.SAMRefinerError, match="far"):
        sam_refiner.refine_box(
            np.zeros((300, 300), dtype=np.uint8),
            {"x": 140, "y": 140, "w": 12, "h": 6},
            "ignored.pt",
            context_scale=8.0,
        )


def test_refinement_history_records_no_quantitative_mask():
    result = sam_refiner.RefinementResult(
        backend=sam_refiner.BACKEND,
        checkpoint_sha256="d" * 64,
        device="cuda",
        crop_box_xyxy=(0, 0, 100, 80),
        prompt_box_xyxy=(20.0, 20.0, 40.0, 30.0),
        old_box_xywh=(20, 20, 20, 10),
        proposed_center_xy=(31.0, 26.0),
        mask_box_xyxy=(21, 22, 41, 30),
        mask_score=0.93,
        mask_area=110,
        displacement_pixels=1.4,
    )
    entry = sam_refiner.refinement_history_entry(
        "r1",
        {"source": "gelgenie", "status": "pretrained proposal"},
        {"x": 21, "y": 21, "w": 20, "h": 10},
        result,
    )
    assert entry["roi_id"] == "r1"
    assert entry["previous_source"] == "gelgenie"
    assert entry["new_box_xywh"] == [21, 21, 20, 10]
    assert entry["quantitative_mask_used"] is False
    assert entry["human_review_required"] is True
