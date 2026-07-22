"""Generate a synthetic 3-lane x 3-row Western blot (TIFF) for demoing the app.

Rows (top->bottom): Insig, Prmt9, GAPDH (loading control).
Lanes (left->right): WT, KO, Rescue.
Bands are dark on a light background (film-style), so use "bands are dark" = ON.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

W, H = 360, 300
rng = np.random.default_rng(7)

# light background with mild noise
img = np.full((H, W), 235.0)
img += rng.normal(0, 3.0, size=(H, W))

# lane x-centres and row y-centres
lane_x = [80, 180, 280]
row_y = {"Insig": 60, "Prmt9": 150, "GAPDH": 240}

# relative "true" amounts (darkness). GAPDH ~ even loading.
amounts = {
    "Insig": [1.0, 0.4, 0.9],    # KO lane loses Insig
    "Prmt9": [0.3, 1.0, 0.6],    # KO lane gains Prmt9
    "GAPDH": [1.0, 0.95, 1.05],  # roughly even loading
}


def add_band(img, cx, cy, darkness, sx=26, sy=9):
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
    img -= darkness * 150.0 * g  # subtract -> darker band
    return img


for target, amt in amounts.items():
    for i, cx in enumerate(lane_x):
        img = add_band(img, cx, row_y[target], amt[i])

img = np.clip(img, 0, 255).astype(np.uint8)
Image.fromarray(img, mode="L").save("sample_blot.tif")
print("Wrote sample_blot.tif", img.shape)
