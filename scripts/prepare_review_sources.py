#!/usr/bin/env python3
"""Plan, fetch, and build the public Western-blot human-review corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from wblot import review_sources  # noqa: E402


def _display_summary(result: object, root: Path) -> object:
    if not isinstance(result, dict):
        return result
    if result.get("schema") == "blotgrid.public_western_fetch":
        return {
            key: value
            for key, value in result.items()
            if key != "receipts"
        } | {
            "receipt_manifest": str(root / "fetch-manifest.json"),
        }
    if "fetch" in result and "build" in result:
        return {
            "fetch": _display_summary(result["fetch"], root),
            "build": _display_summary(result["build"], root),
        }
    return result.get("counts", result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare HPA, Mendeley, BioImage Archive, and Zenodo Western-blot "
            "images for fail-closed manual review in BlotGrid."
        )
    )
    parser.add_argument(
        "command",
        choices=("plan", "fetch", "build", "all"),
        help=(
            "plan performs no network access; fetch acquires the complete "
            "pinned scope; build creates review projects from local payloads; "
            "all runs fetch then build"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/public_western"),
        help="Generated data root (default: data/public_western)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Bounded concurrent downloads for fetch/all (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Images per BlotGrid review project (default: 20)",
    )
    parser.add_argument(
        "--gelgenie-model",
        type=Path,
        help=(
            "Optional verified local GelGenie ONNX model. Proposals remain "
            "unapproved and excluded from ground truth."
        ),
    )
    parser.add_argument(
        "--include-all-hpa",
        action="store_true",
        help=(
            "Canonicalize every acquired HPA image during build. The review "
            "queue still uses the deterministic stratified 256-image subset."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace stale generated downloads/extractions. Source selection "
            "and integrity checks remain enforced."
        ),
    )
    parser.add_argument(
        "--skip-hpa-xml-revalidation",
        action="store_true",
        help=(
            "Reuse the pinned rich HPA index without independently reparsing "
            "the XML in fetch. Intended only after parser-validation.json exists."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.expanduser().resolve()
    try:
        if args.command == "plan":
            result = review_sources.plan_sources(root)
        elif args.command == "fetch":
            result = review_sources.fetch_sources(
                root,
                workers=args.workers,
                force=args.force,
                validate_hpa_xml=not args.skip_hpa_xml_revalidation,
            )
        elif args.command == "build":
            result = review_sources.build_review_corpus(
                root,
                batch_size=args.batch_size,
                gelgenie_model=args.gelgenie_model,
                include_all_hpa=args.include_all_hpa,
                force_extract=args.force,
            )
        else:
            if args.include_all_hpa:
                raise review_sources.ReviewSourceError(
                    "--include-all-hpa applies to build; all intentionally "
                    "builds the bounded first-review corpus after full fetch."
                )
            if args.skip_hpa_xml_revalidation:
                fetch = review_sources.fetch_sources(
                    root,
                    workers=args.workers,
                    force=args.force,
                    validate_hpa_xml=False,
                )
                build = review_sources.build_review_corpus(
                    root,
                    batch_size=args.batch_size,
                    gelgenie_model=args.gelgenie_model,
                    force_extract=args.force,
                )
                result = {"fetch": fetch, "build": build}
            else:
                result = review_sources.prepare_all(
                    root,
                    workers=args.workers,
                    batch_size=args.batch_size,
                    gelgenie_model=args.gelgenie_model,
                    force=args.force,
                )
    except review_sources.ReviewSourceError as exc:
        raise SystemExit(f"Public Western preparation failed: {exc}") from exc
    summary = _display_summary(result, root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
