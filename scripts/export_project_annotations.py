#!/usr/bin/env python3
"""Export approved BlotGrid project ROIs as COCO band annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from wblot.ml_detector import MLDetectorError  # noqa: E402
from wblot.ml_training import export_project_to_coco  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert approved BlotGrid ROIs to category-1 COCO boxes. "
            "Saved image hashes and alignment are honored when present."
        )
    )
    parser.add_argument("project", type=Path, help="reviewed BlotGrid project JSON")
    parser.add_argument("images", type=Path, help="original upload image directory")
    parser.add_argument("output", type=Path, help="COCO JSON output path")
    parser.add_argument(
        "--export-images",
        type=Path,
        help=(
            "copy registered working images here; this happens automatically "
            "beside the COCO JSON when any saved alignment is nonzero"
        ),
    )
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help=(
            "include unapproved automatic ROIs as explicitly marked pseudo-labels"
        ),
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="include disabled ROIs (still requires approval unless overridden)",
    )
    parser.add_argument(
        "--include-fallback",
        action="store_true",
        help="include fallback ROIs (still requires approval unless overridden)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing COCO/working-image outputs",
    )
    parser.add_argument(
        "--allow-unverified-images",
        action="store_true",
        help=(
            "allow a trusted legacy project with no saved image SHA-256; "
            "checksum mismatches are never bypassed"
        ),
    )
    parser.add_argument(
        "--allow-incomplete-annotations",
        action="store_true",
        help=(
            "allow trusted provisional positive images without a completed "
            "whole-image annotation attestation"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        coco = export_project_to_coco(
            args.project,
            args.images,
            args.output,
            include_inactive=args.include_inactive,
            include_fallback=args.include_fallback,
            include_unreviewed=args.include_unreviewed,
            export_image_root=args.export_images,
            allow_unverified=args.allow_unverified_images,
            allow_incomplete_annotations=args.allow_incomplete_annotations,
            force=args.force,
        )
    except MLDetectorError as exc:
        parser.exit(2, f"error: {exc}\n")
    summary = dict(coco["blotgrid"]["export_summary"])
    summary["output"] = str(args.output.resolve())
    summary["annotation_policy"] = coco["blotgrid"]["annotation_policy"]
    summary["image_root"] = coco["blotgrid"]["image_root"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
