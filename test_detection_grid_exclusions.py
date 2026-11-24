"""Focused regression tests for raw-grid recovery and non-sample lanes."""
from __future__ import annotations

import numpy as np

from wblot import detection


def test_raw_grid_recovers_one_broad_row_with_ten_lanes():
    height, width = 240, 760
    yy, xx = np.mgrid[:height, :width]
    expected_y = 120
    expected_x = np.asarray([75 + 68 * index for index in range(10)])
    gray = np.full((height, width), 240.0)
    for center_x in expected_x:
        gray -= 150.0 * np.exp(
            -(
                (xx - center_x) ** 2 / (2 * 16.0**2)
                + (yy - expected_y) ** 2 / (2 * 4.0**2)
            )
        )

    grid = detection.infer_raw_grid(gray)

    assert grid.target_centers == (expected_y,)
    assert len(grid.lane_centers) == 10
    assert np.max(np.abs(np.asarray(grid.lane_centers) - expected_x)) <= 2


def test_excluded_lane_keeps_geometry_but_never_gets_a_band_or_fallback():
    height, width = 130, 190
    yy, xx = np.mgrid[:height, :width]
    targets = ["A", "B"]
    lanes = ["Sample 1", "Ladder", "Sample 2"]
    target_centers = [35, 95]
    lane_centers = [30, 95, 160]
    gray = np.full((height, width), 235.0)
    for target_index, center_y in enumerate(target_centers):
        for lane_index, center_x in enumerate(lane_centers):
            if target_index == 1 and lane_index == 2:
                continue
            gray -= 130.0 * np.exp(
                -(
                    (xx - center_x) ** 2 / (2 * 11.0**2)
                    + (yy - center_y) ** 2 / (2 * 4.0**2)
                )
            )

    result = detection.detect_bands(
        gray,
        targets,
        lanes,
        lane_centers_hint=lane_centers,
        target_centers_hint=target_centers,
        excluded_lanes={"ladder"},
    )

    assert result.lane_centers == lane_centers
    assert all(band.lane != "Ladder" for band in result.bands)
    assert any("3 bands for 4 target/lane cells" in warning for warning in result.warnings)

    # Completion also enforces the exclusion for proposals supplied by another
    # backend, not only for bands created by detect_bands().
    result.bands.append(
        detection.DetectedBand(
            target="A",
            lane="Ladder",
            x=84,
            y=28,
            w=22,
            h=14,
            confidence=0.99,
        )
    )
    added = detection.ensure_complete_grid(
        result,
        targets,
        lanes,
        gray.shape,
        gray=gray,
        excluded_lanes="LADDER",
    )

    assert added == 1
    assert result.lane_centers == lane_centers
    assert all(band.lane != "Ladder" for band in result.bands)
    assert {(band.target, band.lane) for band in result.bands} == {
        (target, lane)
        for target in targets
        for lane in ("Sample 1", "Sample 2")
    }

