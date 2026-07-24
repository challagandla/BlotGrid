#!/usr/bin/env python3
"""Download and verify the pinned open-source models supported by BlotGrid."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from wblot import gelgenie, sam_refiner  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_pinned(
    destination: Path,
    *,
    url: str,
    expected_size: int,
    expected_sha256: str,
    label: str,
    force: bool,
) -> tuple[Path, str]:
    """Stream, bound, hash, and atomically install one allowlisted artifact."""
    destination = destination.expanduser()
    if destination.is_symlink():
        raise RuntimeError("Model destination symlinks are not accepted.")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        digest = _sha256(destination)
        if (
            digest == expected_sha256
            and destination.stat().st_size == expected_size
        ):
            return destination, "already_verified"
        raise RuntimeError(
            f"{destination} already exists but is not the pinned official model. "
            "Move it aside or pass --force to replace only this model file."
        )

    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BlotGrid-pretrained-model-installer/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open(
            "xb"
        ) as output:
            total = 0
            while True:
                block = response.read(1024**2)
                if not block:
                    break
                total += len(block)
                if total > expected_size:
                    raise RuntimeError(
                        f"{label} download exceeded the pinned model size."
                    )
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if partial.stat().st_size != expected_size:
            raise RuntimeError(
                f"{label} download size does not match the pinned model."
            )
        digest = _sha256(partial)
        if digest != expected_sha256:
            raise RuntimeError(
                f"{label} download SHA-256 does not match the pinned model."
            )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination, "downloaded"


def download_gelgenie(destination: Path, *, force: bool = False) -> dict:
    """Atomically install the immutable official GelGenie Universal ONNX file."""
    destination, status = _download_pinned(
        destination,
        url=gelgenie.MODEL_URL,
        expected_size=gelgenie.MODEL_SIZE_BYTES,
        expected_sha256=gelgenie.MODEL_SHA256,
        label="GelGenie",
        force=force,
    )
    result = {"status": status, **gelgenie.model_provenance(destination)}
    _write_metadata_sidecar(
        destination,
        result,
        source_url=gelgenie.MODEL_URL,
        upstream_repository="https://github.com/mattaq31/GelGenie",
        model_card=f"https://huggingface.co/{gelgenie.MODEL_ID}",
    )
    return result


def download_sam2(destination: Path, *, force: bool = False) -> dict:
    """Atomically install the immutable official SAM 2.1 Hiera Small weights."""
    destination, status = _download_pinned(
        destination,
        url=sam_refiner.CHECKPOINT_URL,
        expected_size=sam_refiner.CHECKPOINT_SIZE_BYTES,
        expected_sha256=sam_refiner.CHECKPOINT_SHA256,
        label="SAM 2.1 Hiera Small",
        force=force,
    )
    result = {
        "status": status,
        **sam_refiner.checkpoint_provenance(destination),
    }
    _write_metadata_sidecar(
        destination,
        result,
        source_url=sam_refiner.CHECKPOINT_URL,
        upstream_repository="https://github.com/facebookresearch/sam2",
        model_card="https://github.com/facebookresearch/sam2#model-description",
    )
    return result


def _write_metadata_sidecar(
    destination: Path,
    payload: dict,
    *,
    source_url: str,
    upstream_repository: str,
    model_card: str,
) -> None:
    """Write model attribution/provenance beside the ignored local weights."""
    sidecar = destination.with_name(destination.name + ".metadata.json")
    serializable = {
        **payload,
        "source_url": source_url,
        "upstream_repository": upstream_repository,
        "model_card": model_card,
    }
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=sidecar.name + ".",
        suffix=".tmp",
        dir=sidecar.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, sidecar)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and SHA-256 verify BlotGrid pretrained models."
    )
    parser.add_argument(
        "--model",
        choices=["gelgenie-universal", "sam2.1-hiera-small"],
        default="gelgenie-universal",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing invalid file at the exact destination path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.model == "gelgenie-universal":
            destination = (
                args.destination
                or REPOSITORY_ROOT / gelgenie.DEFAULT_MODEL_PATH
            )
            result = download_gelgenie(destination, force=args.force)
        else:
            destination = (
                args.destination
                or REPOSITORY_ROOT / sam_refiner.DEFAULT_CHECKPOINT_PATH
            )
            result = download_sam2(destination, force=args.force)
    except (
        OSError,
        RuntimeError,
        gelgenie.GelGenieError,
        sam_refiner.SAMRefinerError,
    ) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
