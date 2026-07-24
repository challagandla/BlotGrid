#!/usr/bin/env python3
"""Fail-closed publication check for a BlotGrid detector checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from wblot.ml_detector import (  # noqa: E402
    MLDetectorError,
    sanitize_checkpoint_for_release,
    verified_checkpoint_provenance,
)
from wblot.ml_training import checkpoint_publication_assessment  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a checkpoint sidecar, licence lineage, public metadata "
            "allowlist, and initializer redistribution status."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--release-output",
        type=Path,
        help=(
            "required distinct output for the sanitized checkpoint that may be "
            "published; the local training archive is never certified directly"
        ),
    )
    parser.add_argument(
        "--attribution-output",
        type=Path,
        help="write the verified attribution bundle after a passing assessment",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.checkpoint.expanduser().resolve()
    source_sidecar = source.with_name(source.name + ".metadata.json")
    try:
        verified_checkpoint_provenance(source)
    except MLDetectorError as exc:
        print(json.dumps({"eligible_for_publication": False, "error": str(exc)}))
        return 2
    if args.release_output is None:
        print(
            json.dumps(
                {
                    "eligible_for_publication": False,
                    "error": (
                        "A distinct --release-output is required. The local "
                        "PyTorch training archive is never publication-safe."
                    ),
                }
            )
        )
        return 2

    destination = args.release_output.expanduser().resolve()
    destination_sidecar = destination.with_name(
        destination.name + ".metadata.json"
    )
    attribution_destination = (
        args.attribution_output.expanduser().resolve()
        if args.attribution_output
        else None
    )
    if attribution_destination in {
        destination,
        destination_sidecar,
        source,
        source_sidecar,
    }:
        print(
            json.dumps(
                {
                    "eligible_for_publication": False,
                    "error": (
                        "--attribution-output must be distinct from the input and "
                        "release checkpoints and their metadata sidecars."
                    ),
                }
            )
        )
        return 2
    if attribution_destination is not None and (
        attribution_destination.exists()
        or attribution_destination.is_symlink()
    ):
        print(
            json.dumps(
                {
                    "eligible_for_publication": False,
                    "error": "Attribution output must not already exist.",
                }
            )
        )
        return 2
    if (
        destination.exists()
        or destination.is_symlink()
        or destination_sidecar.exists()
        or destination_sidecar.is_symlink()
    ):
        print(
            json.dumps(
                {
                    "eligible_for_publication": False,
                    "error": "Release output and sidecar must not already exist.",
                }
            )
        )
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=destination.name + ".release-stage.",
        dir=destination.parent,
    ) as staging_directory:
        staged = Path(staging_directory) / destination.name
        try:
            provenance = sanitize_checkpoint_for_release(
                source,
                staged,
            )
        except MLDetectorError as exc:
            print(
                json.dumps(
                    {"eligible_for_publication": False, "error": str(exc)}
                )
            )
            return 2
        assessment = checkpoint_publication_assessment(provenance)
        if not assessment["eligible_for_publication"]:
            print(json.dumps(assessment, indent=2, ensure_ascii=False))
            return 1
        staged_sha256 = provenance["sha256"]
        try:
            os.replace(staged, destination)
            os.replace(
                staged.with_name(staged.name + ".metadata.json"),
                destination_sidecar,
            )
        except OSError as exc:
            destination.unlink(missing_ok=True)
            destination_sidecar.unlink(missing_ok=True)
            print(
                json.dumps(
                    {
                        "eligible_for_publication": False,
                        "error": f"Could not finalize release output: {exc}",
                    }
                )
            )
            return 2
    try:
        provenance = verified_checkpoint_provenance(destination)
    except MLDetectorError as exc:
        destination.unlink(missing_ok=True)
        destination_sidecar.unlink(missing_ok=True)
        print(json.dumps({"eligible_for_publication": False, "error": str(exc)}))
        return 2
    if provenance.get("sha256") != staged_sha256:
        destination.unlink(missing_ok=True)
        destination_sidecar.unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "eligible_for_publication": False,
                    "error": "Release checkpoint changed while it was finalized.",
                }
            )
        )
        return 2
    provenance["release_artifact_sanitized"] = True
    assessment = checkpoint_publication_assessment(provenance)
    if not assessment["eligible_for_publication"]:
        destination.unlink(missing_ok=True)
        destination_sidecar.unlink(missing_ok=True)
        print(json.dumps(assessment, indent=2, ensure_ascii=False))
        return 1
    assessment["release_checkpoint"] = str(destination)
    print(json.dumps(assessment, indent=2, ensure_ascii=False))
    if attribution_destination is not None:
        attribution_destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=attribution_destination.name + ".",
            suffix=".tmp",
            dir=attribution_destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    assessment["attribution_bundle"],
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                handle.write("\n")
            os.replace(temporary, attribution_destination)
        finally:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
