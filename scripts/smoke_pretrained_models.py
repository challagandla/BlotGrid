#!/usr/bin/env python3
"""Run local inference smoke checks for BlotGrid's verified pretrained models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from wblot import detection, gelgenie, image_io, sam_refiner  # noqa: E402


TARGETS = ["Insig", "Prmt9", "GAPDH"]
LANES = ["WT", "KO", "Rescue"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify GelGenie and optional SAM 2.1 inference locally."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=REPOSITORY_ROOT / "sample_blot.tif",
    )
    parser.add_argument(
        "--gelgenie-model",
        type=Path,
        default=REPOSITORY_ROOT / gelgenie.DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / sam_refiner.DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-sam", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loaded = image_io.load_image(args.image, name=args.image.name)
    gel_result = gelgenie.predict_bands(
        loaded.gray,
        TARGETS,
        LANES,
        args.gelgenie_model,
        max_value=loaded.max_value,
        probability_threshold=0.5,
        review_threshold=0.75,
        device=args.device,
        polarity="dark",
    )
    output = {
        "image": str(args.image.resolve()),
        "gelgenie": {
            "direct_proposals": len(gel_result.bands),
            "providers": gelgenie.available_providers(),
            "model": gelgenie.model_provenance(args.gelgenie_model),
        },
    }
    if not args.skip_sam:
        classical = detection.detect_bands(
            loaded.gray,
            TARGETS,
            LANES,
            polarity="dark",
        )
        if not classical.bands:
            raise SystemExit("error: classical detector found no ROI for SAM prompt")
        prompt_roi = classical.bands[0].as_row()
        refinement = sam_refiner.refine_box(
            image_io.to_display(loaded),
            prompt_roi,
            args.sam_checkpoint,
            device=args.device,
        )
        updated = sam_refiner.recenter_fixed_roi(
            prompt_roi,
            refinement,
            {"w": prompt_roi["w"], "h": prompt_roi["h"]},
            loaded.gray.shape,
        )
        output["sam2"] = {
            "device": refinement.device,
            "mask_score": refinement.mask_score,
            "old_box_xywh": refinement.old_box_xywh,
            "new_box_xywh": [
                updated["x"],
                updated["y"],
                updated["w"],
                updated["h"],
            ],
            "size_preserved": (
                updated["w"] == prompt_roi["w"]
                and updated["h"] == prompt_roi["h"]
            ),
            "checkpoint": sam_refiner.checkpoint_provenance(
                args.sam_checkpoint
            ),
        }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
