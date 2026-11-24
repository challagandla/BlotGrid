"""Render the BlotGrid app icon (Linear palette) to packaging/icon.png.

PyInstaller converts the PNG to .ico/.icns at build time (needs Pillow).
"""
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 1024
CANVAS, SURFACE, HAIRLINE, ACCENT, INK = "#010102", "#141516", "#34343a", "#5e6ad2", "#f7f8f8"


def render(size: int = SIZE) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = size // 16
    d.rounded_rectangle([pad, pad, size - pad, size - pad], radius=size // 5,
                        fill=CANVAS, outline=HAIRLINE, width=max(2, size // 128))
    inner = size // 4
    cell = (size - 2 * inner) // 3
    gap = cell // 6
    for row in range(3):
        for col in range(3):
            x0 = inner + col * cell + gap // 2
            y0 = inner + row * cell + gap // 2
            box = [x0, y0, x0 + cell - gap, y0 + cell - gap]
            # One row of "bands" in the accent colour; the rest are lanes.
            fill = ACCENT if row == 1 else (INK if (row, col) == (0, 0) else SURFACE)
            d.rounded_rectangle(box, radius=cell // 6, fill=fill)
    return img


if __name__ == "__main__":
    out = Path(__file__).with_name("icon.png")
    render().resize((512, 512), Image.LANCZOS).save(out, optimize=True)
    print(out)
