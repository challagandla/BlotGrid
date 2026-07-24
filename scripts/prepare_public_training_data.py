#!/usr/bin/env python3
"""Download and prepare the immutable GelGenie v3 public bootstrap corpus.

The generated detector data are intentionally described as out-of-domain:
GelGenie's masks annotate electrophoresis gels, not a Western-blot validation
cohort.  Downloads and generated files live below ``data/`` by default and are
therefore not redistributed by BlotGrid.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import sys
import tempfile
from typing import Any, Iterable
import urllib.request
import zipfile
import zlib

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION
import scipy
from scipy import ndimage


ZENODO_RECORD = "14641949"
ZENODO_DOI = "10.5281/zenodo.14641949"
ZENODO_BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"
RGP_URL = (
    "https://dbarchive.biosciencedbc.jp/data/rgp-caps/LATEST/"
    "rgp_caps_electrophoresis_image.zip"
)
RGP_LICENSE_URL = "https://dbarchive.biosciencedbc.jp/en/rgp-caps/lic.html"
RGP_ATTRIBUTION = (
    "RGP: 332 PCR-based genetic markers on rice chromosomes "
    "© National Institute of Agrobiological Sciences licensed under "
    "CC Attribution-Share Alike 2.1 Japan"
)
MAX_ARCHIVE_MEMBERS = 20_000
MAX_EXTRACTED_BYTES = 50 * 1024**3
MAX_MEMBER_BYTES = 4 * 1024**3
CHUNK_BYTES = 1024 * 1024
IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
}


@dataclass(frozen=True)
class SourceFile:
    name: str
    url: str
    size: int
    algorithm: str
    digest: str
    collection: str
    extract: bool = True


ZENODO_FILES = (
    ("external_gels.zip", 96_670_165, "29b6ff6ca68b6c7768758fc2ad87678a"),
    ("matthew_gels_2.zip", 51_907_776, "2a361b56f2a4ac8c1899dcaf9b7c38f7"),
    ("matthew_gels.zip", 756_367_174, "bfe792d8afd10bbeb2f6024f45e512ef"),
    ("nathan_gels.zip", 4_010_557, "0a2adc5dd17fea74f69f2b8ce3ffc626"),
    (
        "quantitation_ladder_gels.zip",
        23_314_210,
        "627f7340d5372511691088c4e0195a06",
    ),
    (
        "stella_gels_for_finetuning.zip",
        24_730_774,
        "ddab6a30d8e6b7ff337b9065280849a5",
    ),
    ("lsdb_gels.zip", 84_488, "e73aab6d1cd067fe840bdedcbdd8df1d"),
    ("readme.txt", 2_729, "ff286b8ac93a11c4e709e89950ebcc8e"),
)
EXPECTED_LABELLED_IMAGES = {
    "external_gels.zip": 25,
    "matthew_gels_2.zip": 85,
    "matthew_gels.zip": 301,
    "nathan_gels.zip": 37,
    "quantitation_ladder_gels.zip": 35,
    "stella_gels_for_finetuning.zip": 26,
    "lsdb_gels.zip": 66,
}
GELGENIE_CREATORS = [
    "Matthew Aquilina",
    "Nathan Wu",
    "Kiros Kwan",
    "Filip Bušić",
    "James Dodd",
    "Laura Nicolás-Sáenz",
    "Alan O'Callaghan",
    "Peter Bankhead",
    "Katherine Dunn",
]
GELGENIE_DATA_COLLECTORS = [
    "Siyuan Stella Wang",
    "Yichen Zhao",
    "Thomas Mayer",
    "Huangchen Cui",
    "Joana Reis",
    "Ricarda Törner",
]
GELGENIE_CITATION = (
    "Aquilina, Matthew; Wu, Nathan; Kwan, Kiros; Bušić, Filip; Dodd, James; "
    "Nicolás-Sáenz, Laura; O'Callaghan, Alan; Bankhead, Peter; Dunn, Katherine "
    "(2025). Segmented Gel Electrophoresis Images used for Training GelGenie "
    "Models (v3). Zenodo. https://doi.org/10.5281/zenodo.14641949"
)
SOURCES = tuple(
    SourceFile(
        name=name,
        url=f"{ZENODO_BASE}/{name}/content",
        size=size,
        algorithm="md5",
        digest=digest,
        collection="gelgenie_v3",
        extract=name.endswith(".zip"),
    )
    for name, size, digest in ZENODO_FILES
) + (
    SourceFile(
        name="rgp_caps_electrophoresis_image.zip",
        url=RGP_URL,
        size=30_105_023,
        algorithm="sha256",
        digest="b0a8ee39d8db7eec745ac5aaec970e78eef65b04a7a0fd964960eb639b511d46",
        collection="rgp_caps",
    ),
)


class PreparationError(RuntimeError):
    """Unsafe, incomplete, or internally inconsistent public data."""


def _digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verified(path: Path, source: SourceFile) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == source.size
        and _digest(path, source.algorithm).lower() == source.digest.lower()
    )


def download_source(
    source: SourceFile,
    destination: Path,
    *,
    force: bool = False,
    timeout: int = 120,
) -> str:
    """Atomically download one source, enforcing its exact size and digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _verified(destination, source):
        return "already_verified"
    if destination.exists() and not force:
        raise PreparationError(
            f"Existing download is not the pinned {source.name}: {destination}. "
            "Use --force only if it is safe to replace that generated file."
        )
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "BlotGrid-public-data/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) != source.size:
                raise PreparationError(
                    f"{source.name} server size changed: {declared} != {source.size}."
                )
            received = 0
            hasher = hashlib.new(source.algorithm)
            with partial.open("xb") as handle:
                while True:
                    chunk = response.read(min(CHUNK_BYTES, source.size + 1 - received))
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > source.size:
                        raise PreparationError(
                            f"{source.name} exceeded its pinned byte count."
                        )
                    handle.write(chunk)
                    hasher.update(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        if received != source.size:
            raise PreparationError(
                f"{source.name} is truncated: {received} != {source.size} bytes."
            )
        if hasher.hexdigest().lower() != source.digest.lower():
            raise PreparationError(f"{source.name} checksum does not match its pin.")
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return "downloaded"


def _safe_member_path(name: str) -> PurePosixPath:
    if "\x00" in name or "\\" in name:
        raise PreparationError(f"Unsafe ZIP member path: {name!r}.")
    member = PurePosixPath(name)
    if member.is_absolute() or not member.parts or ".." in member.parts:
        raise PreparationError(f"Unsafe ZIP member path: {name!r}.")
    return member


def safe_extract(
    archive: Path,
    destination: Path,
    *,
    archive_sha256: str,
    force: bool = False,
) -> str:
    """Extract a pinned ZIP atomically without links or path traversal."""
    marker_name = ".blotgrid-extraction.json"
    marker = destination / marker_name
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = {}
        if payload.get("archive_sha256") == archive_sha256:
            if _extraction_matches_archive(archive, destination, marker_name):
                return "already_verified"
            if not force:
                raise PreparationError(
                    f"Extracted files are missing or corrupt beneath {destination}. "
                    "Use --force to rebuild that generated directory."
                )
    if destination.exists() and not force:
        raise PreparationError(
            f"Extraction directory is incomplete or stale: {destination}. "
            "Use --force only if it is safe to replace that generated directory."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise PreparationError(
                    f"{archive.name} contains too many members: {len(members)}."
                )
            total = 0
            for info in members:
                _safe_member_path(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise PreparationError(
                        f"{archive.name} contains a symbolic link: {info.filename}."
                    )
                if info.flag_bits & 0x1:
                    raise PreparationError(
                        f"{archive.name} contains an encrypted member."
                    )
                if info.file_size > MAX_MEMBER_BYTES:
                    raise PreparationError(
                        f"{archive.name} member is too large: {info.filename}."
                    )
                total += info.file_size
                if total > MAX_EXTRACTED_BYTES:
                    raise PreparationError(
                        f"{archive.name} exceeds the extraction byte limit."
                    )
            for info in members:
                relative = _safe_member_path(info.filename)
                target = temporary.joinpath(*relative.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                copied = 0
                with bundle.open(info) as source_handle, target.open("xb") as output:
                    while True:
                        chunk = source_handle.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        copied += len(chunk)
                        if copied > info.file_size:
                            raise PreparationError(
                                f"ZIP member expanded past its declared size: "
                                f"{info.filename}."
                            )
                        output.write(chunk)
                if copied != info.file_size:
                    raise PreparationError(
                        f"ZIP member is truncated: {info.filename}."
                    )
        _atomic_json(
            temporary / marker_name,
            {
                "archive": archive.name,
                "archive_sha256": archive_sha256,
                "members": len(members),
                "uncompressed_bytes": total,
            },
        )
        backup = temporary.with_name(temporary.name + ".previous")
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except Exception:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "extracted"


def _extraction_matches_archive(
    archive: Path,
    destination: Path,
    marker_name: str,
) -> bool:
    """Verify every extracted file against its pinned ZIP size and CRC."""
    try:
        with zipfile.ZipFile(archive) as bundle:
            expected_files: set[str] = set()
            for info in bundle.infolist():
                relative = _safe_member_path(info.filename)
                if info.is_dir():
                    continue
                relative_name = relative.as_posix()
                expected_files.add(relative_name)
                path = destination.joinpath(*relative.parts)
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != info.file_size
                ):
                    return False
                checksum = 0
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
                        checksum = zlib.crc32(chunk, checksum)
                if checksum & 0xFFFFFFFF != info.CRC:
                    return False
        actual_files = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file() and path.name != marker_name
        }
        return actual_files == expected_files
    except (OSError, ValueError, zipfile.BadZipFile, PreparationError):
        return False


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _source_directory(raw_root: Path, source: SourceFile) -> Path:
    return raw_root / Path(source.name).stem


def download_all(
    output: Path,
    *,
    force: bool = False,
    sources: Iterable[SourceFile] = SOURCES,
) -> list[dict[str, Any]]:
    downloads = output / "downloads"
    records = []
    for source in sources:
        path = downloads / source.name
        status = download_source(source, path, force=force)
        print(f"{status}: {source.name}")
        records.append(
            {
                "name": source.name,
                "url": source.url,
                "size_bytes": source.size,
                "digest_algorithm": source.algorithm,
                "digest": source.digest,
                "sha256": _digest(path),
                "collection": source.collection,
                "status": status,
            }
        )
    return records


def extract_all(
    output: Path,
    *,
    force: bool = False,
    sources: Iterable[SourceFile] = SOURCES,
) -> list[dict[str, Any]]:
    downloads = output / "downloads"
    raw_root = output / "raw"
    records = []
    for source in sources:
        archive = downloads / source.name
        if not _verified(archive, source):
            raise PreparationError(
                f"Missing or invalid pinned download: {archive}. "
                "Run the full or download-only mode first."
            )
        if not source.extract:
            continue
        sha256 = _digest(archive)
        destination = _source_directory(raw_root, source)
        status = safe_extract(
            archive,
            destination,
            archive_sha256=sha256,
            force=force,
        )
        print(f"{status}: {source.name}")
        records.append(
            {
                "archive": source.name,
                "archive_sha256": sha256,
                "directory": str(destination.relative_to(output)),
                "verified": True,
            }
        )
    return records


def _is_image(path: Path) -> bool:
    return (
        path.is_file()
        and not path.name.startswith(".")
        and "__MACOSX" not in path.parts
        and path.suffix.casefold() in IMAGE_SUFFIXES
    )


def _index_images(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if _is_image(path):
                index[path.stem.casefold()].append(path)
    return index


def _component_boxes(mask_path: Path) -> tuple[list[list[float]], tuple[int, int]]:
    try:
        with Image.open(mask_path) as image:
            image.load()
            array = np.asarray(image)
            size = (int(image.width), int(image.height))
    except (OSError, ValueError) as exc:
        raise PreparationError(f"Could not decode mask: {mask_path}.") from exc
    if array.ndim == 3:
        channels = array[..., :3] if array.shape[-1] == 4 else array
        foreground = np.any(channels != 0, axis=-1)
    elif array.ndim == 2:
        foreground = array != 0
    else:
        raise PreparationError(f"Unsupported mask dimensions: {mask_path}.")
    labels, count = ndimage.label(
        foreground,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    boxes: list[list[float]] = []
    for label_index, slices in enumerate(
        ndimage.find_objects(labels, max_label=count), start=1
    ):
        if slices is None:
            continue
        y_slice, x_slice = slices
        width = int(x_slice.stop - x_slice.start)
        height = int(y_slice.stop - y_slice.start)
        if width < 1 or height < 1:
            continue
        pixel_count = int(np.count_nonzero(labels[slices] == label_index))
        if pixel_count < 1:
            continue
        boxes.append(
            [
                float(x_slice.start),
                float(y_slice.start),
                float(width),
                float(height),
            ]
        )
    return boxes, size


def _image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            return int(image.width), int(image.height)
    except (OSError, ValueError) as exc:
        raise PreparationError(f"Could not decode image: {path}.") from exc


def _relative(path: Path, output: Path) -> str:
    try:
        return str(path.resolve().relative_to(output.resolve()))
    except ValueError:
        return str(path.resolve())


def collect_pairs(
    output: Path,
    *,
    sources: Iterable[SourceFile] = SOURCES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Pair every upstream mask with its exact image, including LSDB/RGP."""
    raw_root = output / "raw"
    sources_by_name = {source.name: source for source in sources}
    rgp_source = sources_by_name.get("rgp_caps_electrophoresis_image.zip")
    rgp_index = (
        _index_images(_source_directory(raw_root, rgp_source))
        if rgp_source is not None
        else {}
    )
    pairs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    masks_by_archive: Counter[str] = Counter()
    pairs_by_archive: Counter[str] = Counter()
    pairs_by_partition: Counter[str] = Counter()
    rejections_by_reason: Counter[str] = Counter()
    mask_to_image = {
        "masks": ("images", "train"),
        "val_masks": ("val_images", "val"),
        "test_masks": ("test_images", "test"),
    }
    for source in sources:
        if not source.extract or source.collection == "rgp_caps":
            continue
        source_root = _source_directory(raw_root, source)
        if not source_root.is_dir():
            raise PreparationError(f"Missing extracted source: {source_root}.")
        external = source.name == "external_gels.zip"
        mask_directories = sorted(
            path
            for path in source_root.rglob("*")
            if path.is_dir() and path.name.casefold() in mask_to_image
        )
        for mask_directory in mask_directories:
            image_directory_name, upstream_partition = mask_to_image[
                mask_directory.name.casefold()
            ]
            partition = "external" if external else upstream_partition
            local_image_index = _index_images(
                mask_directory.parent / image_directory_name
            )
            for mask_path in sorted(mask_directory.iterdir()):
                if not _is_image(mask_path):
                    continue
                counts["masks_seen"] += 1
                masks_by_archive[source.name] += 1
                key = mask_path.stem.casefold()
                candidates = local_image_index.get(key, [])
                image_origin = source.name
                if not candidates and source.name == "lsdb_gels.zip":
                    candidates = rgp_index.get(key, [])
                    image_origin = "rgp_caps_electrophoresis_image.zip"
                if len(candidates) != 1:
                    rejected.append(
                        {
                            "reason": (
                                "missing_source_image"
                                if not candidates
                                else "ambiguous_source_image"
                            ),
                            "source_archive": source.name,
                            "partition": partition,
                            "mask": _relative(mask_path, output),
                            "candidate_count": len(candidates),
                        }
                    )
                    counts["unpaired_masks"] += 1
                    rejections_by_reason[rejected[-1]["reason"]] += 1
                    continue
                image_path = candidates[0]
                try:
                    boxes, mask_size = _component_boxes(mask_path)
                    image_size = _image_size(image_path)
                except PreparationError as exc:
                    rejected.append(
                        {
                            "reason": "decode_failure",
                            "source_archive": source.name,
                            "partition": partition,
                            "mask": _relative(mask_path, output),
                            "image": _relative(image_path, output),
                            "detail": str(exc),
                        }
                    )
                    counts["decode_failures"] += 1
                    rejections_by_reason["decode_failure"] += 1
                    continue
                if image_size != mask_size:
                    rejected.append(
                        {
                            "reason": "image_mask_dimension_mismatch",
                            "source_archive": source.name,
                            "partition": partition,
                            "mask": _relative(mask_path, output),
                            "image": _relative(image_path, output),
                            "image_size": list(image_size),
                            "mask_size": list(mask_size),
                        }
                    )
                    counts["dimension_mismatches"] += 1
                    rejections_by_reason["image_mask_dimension_mismatch"] += 1
                    continue
                if not boxes:
                    rejected.append(
                        {
                            "reason": "empty_mask_not_used_as_unattested_negative",
                            "source_archive": source.name,
                            "partition": partition,
                            "mask": _relative(mask_path, output),
                            "image": _relative(image_path, output),
                        }
                    )
                    counts["empty_masks"] += 1
                    rejections_by_reason[
                        "empty_mask_not_used_as_unattested_negative"
                    ] += 1
                    continue
                pairs.append(
                    {
                        "source_archive": source.name,
                        "image_source_archive": image_origin,
                        "partition": partition,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "image": _relative(image_path, output),
                        "mask": _relative(mask_path, output),
                        "image_sha256": _digest(image_path),
                        "mask_sha256": _digest(mask_path),
                        "width": image_size[0],
                        "height": image_size[1],
                        "boxes": boxes,
                    }
                )
                counts[f"paired_{partition}"] += 1
                pairs_by_archive[source.name] += 1
                pairs_by_partition[partition] += 1
    counts["pairs_total"] = len(pairs)
    collection = {
        **dict(sorted(counts.items())),
        "masks_by_archive": dict(sorted(masks_by_archive.items())),
        "pairs_by_archive": dict(sorted(pairs_by_archive.items())),
        "pairs_by_partition": dict(sorted(pairs_by_partition.items())),
        "rejections_by_reason": dict(sorted(rejections_by_reason.items())),
    }
    return pairs, rejected, collection


def _validate_official_completeness(
    sources: Iterable[SourceFile],
    rejected: list[dict[str, Any]],
    collection_counts: dict[str, Any],
) -> None:
    """Fail closed if the pinned v3 corpus is not represented completely."""
    names = {source.name for source in sources}
    official_names = {source.name for source in SOURCES}
    if names != official_names:
        return
    masks = collection_counts.get("masks_by_archive", {})
    pairs = collection_counts.get("pairs_by_archive", {})
    if masks != EXPECTED_LABELLED_IMAGES:
        raise PreparationError(
            "Pinned GelGenie v3 mask counts changed or were not fully discovered: "
            f"{masks!r} != {EXPECTED_LABELLED_IMAGES!r}."
        )
    if pairs != EXPECTED_LABELLED_IMAGES or rejected:
        reasons = Counter(str(item.get("reason", "unknown")) for item in rejected)
        raise PreparationError(
            "Pinned GelGenie v3 contains an unpaired, unreadable, empty, or "
            "dimension-mismatched labelled image. Refusing a partial corpus. "
            f"paired={pairs!r}; rejections={dict(sorted(reasons.items()))!r}."
        )
    if sum(pairs.values()) != 575:
        raise PreparationError(
            f"Pinned GelGenie v3 completeness count is {sum(pairs.values())}, "
            "not the published 575 labelled gels."
        )


def _copy_verified(
    source: Path,
    destination: Path,
    sha256: str,
    *,
    force: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _digest(destination) == sha256:
        return
    if destination.exists() and not force:
        raise PreparationError(
            f"Generated image path has unexpected content: {destination}."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, length=CHUNK_BYTES)
            output.flush()
            os.fsync(output.fileno())
        if _digest(temporary) != sha256:
            raise PreparationError(f"Copied image checksum changed: {source}.")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_outputs(
    output: Path,
    pairs: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    collection_counts: dict[str, int],
    source_records: list[dict[str, Any]],
    extraction_records: list[dict[str, Any]],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Quarantine leakage/conflicts and write deterministic COCO files."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pairs:
        groups[record["image_sha256"]].append(record)
    usable: list[dict[str, Any]] = []
    heldout = list(rejected)
    duplicate_count = 0
    quarantined_groups = 0
    quarantined_records = 0
    for image_sha256, group in sorted(groups.items()):
        partitions = {record["partition"] for record in group}
        mask_sha256s = {record["mask_sha256"] for record in group}
        if len(partitions) > 1 or len(mask_sha256s) > 1:
            reason = "cross_partition_duplicate_or_conflicting_labels"
            quarantined_groups += 1
            quarantined_records += len(group)
            for record in sorted(group, key=lambda item: (item["partition"], item["mask"])):
                heldout.append(
                    {
                        **_public_record(record),
                        "reason": reason,
                        "quarantine_image_sha256": image_sha256,
                        "quarantine_partitions": sorted(partitions),
                        "quarantine_mask_sha256s": sorted(mask_sha256s),
                        "cross_partition_duplicate": len(partitions) > 1,
                        "conflicting_labels": len(mask_sha256s) > 1,
                    }
                )
            continue
        ordered = sorted(
            group,
            key=lambda item: (
                item["partition"],
                item["source_archive"],
                item["image"],
                item["mask"],
            ),
        )
        canonical = ordered[0]
        for duplicate in ordered[1:]:
            duplicate_count += 1
            heldout.append(
                {
                    **_public_record(duplicate),
                    "reason": "deduplicated_identical_image_and_mask_within_partition",
                    "canonical_image": canonical["image"],
                    "canonical_mask": canonical["mask"],
                }
            )
        if canonical["partition"] in {"train", "val"}:
            usable.append(canonical)
        else:
            heldout.append(
                {
                    **_public_record(canonical),
                    "reason": f"upstream_{canonical['partition']}_held_out",
                }
            )

    usable.sort(
        key=lambda item: (
            0 if item["partition"] == "train" else 1,
            item["source_archive"],
            item["image"],
        )
    )
    if not any(record["partition"] == "train" for record in usable):
        raise PreparationError(
            "No clean training images remain after duplicate/conflict quarantine."
        )
    if not any(record["partition"] == "val" for record in usable):
        raise PreparationError(
            "No clean validation images remain after duplicate/conflict quarantine."
        )

    image_dir = output / "images"
    expected_image_names: set[str] = set()
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    split = {"train": [], "val": []}
    annotation_id = 1
    for image_id, record in enumerate(usable, start=1):
        safe_source = Path(record["source_archive"]).stem.replace(" ", "_")
        filename = (
            f"{safe_source}__{record['partition']}__"
            f"{record['image_sha256'][:12]}__{Path(record['image']).name}"
        )
        destination = image_dir / filename
        expected_image_names.add(filename)
        _copy_verified(
            Path(record["image_path"]),
            destination,
            record["image_sha256"],
            force=force,
        )
        split[record["partition"]].append(image_id)
        coco_images.append(
            {
                "id": image_id,
                "file_name": filename,
                "width": record["width"],
                "height": record["height"],
                "license": (
                    2
                    if record["image_source_archive"]
                    == "rgp_caps_electrophoresis_image.zip"
                    else 1
                ),
                "blotgrid": {
                    "annotation_complete": True,
                    "negative_image": False,
                    "original_sha256": record["image_sha256"],
                    "study_group": (
                        f"public:{Path(record['source_archive']).stem}:"
                        f"{record['image_sha256']}"
                    ),
                    "source_archive": record["source_archive"],
                    "image_source_archive": record["image_source_archive"],
                    "source_partition": record["partition"],
                    "source_image": record["image"],
                    "source_mask": record["mask"],
                    "source_mask_sha256": record["mask_sha256"],
                    "annotation_origin": "upstream_hand_labelled_mask_components",
                    "out_of_domain": True,
                    "domain": "gel_electrophoresis_not_western_blot_validation",
                },
            }
        )
        for bbox in record["boxes"]:
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                    "attributes": {
                        "annotation_origin": "upstream_hand_labelled_mask_component",
                        "source_mask_sha256": record["mask_sha256"],
                    },
                }
            )
            annotation_id += 1

    if image_dir.is_dir():
        unexpected = sorted(
            path
            for path in image_dir.iterdir()
            if path.name not in expected_image_names
        )
        if unexpected and not force:
            raise PreparationError(
                "Generated image directory contains stale/unexpected entries: "
                + ", ".join(path.name for path in unexpected[:10])
                + (" ..." if len(unexpected) > 10 else "")
                + ". Use --force to reconcile that generated directory."
            )
        for path in unexpected:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    annotations_payload = {
        "info": {
            "description": (
                "Out-of-domain GelGenie v3 public gel-electrophoresis bootstrap "
                "boxes; not a Western-blot validation cohort."
            ),
            "source_doi": ZENODO_DOI,
            "license": (
                "Mixed: GelGenie items CC BY 4.0; RGP-origin LSDB images "
                "CC BY-SA 2.1 Japan"
            ),
            "out_of_domain": True,
        },
        "licenses": [
            {
                "id": 1,
                "name": "Creative Commons Attribution 4.0 International",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            },
            {
                "id": 2,
                "name": "Creative Commons Attribution-ShareAlike 2.1 Japan",
                "url": RGP_LICENSE_URL,
            },
        ],
        "categories": [{"id": 1, "name": "band", "supercategory": "gel"}],
        "images": coco_images,
        "annotations": coco_annotations,
    }
    heldout_payload = {
        "description": (
            "Records excluded from fitting: upstream test/external data, "
            "duplicate/conflicting image groups, and invalid pairs."
        ),
        "out_of_domain": True,
        "records": sorted(
            heldout,
            key=lambda item: (
                str(item.get("reason", "")),
                str(item.get("source_archive", "")),
                str(item.get("image", "")),
                str(item.get("mask", "")),
            ),
        ),
    }
    annotations_path = output / "annotations.json"
    split_path = output / "split.json"
    heldout_path = output / "heldout.json"
    _atomic_json(annotations_path, annotations_payload)
    _atomic_json(split_path, split)
    _atomic_json(heldout_path, heldout_payload)

    manifest = {
        "schema_version": 1,
        "dataset_name": "BlotGrid public GelGenie v3 bootstrap",
        "out_of_domain": True,
        "domain_warning": (
            "Electrophoresis gels are not a Western-blot validation cohort. "
            "Do not use these metrics as evidence of Western-blot accuracy."
        ),
        "sources": {
            "gelgenie": {
                "record": ZENODO_RECORD,
                "doi": ZENODO_DOI,
                "license": "CC BY 4.0",
                "record_url": f"https://zenodo.org/records/{ZENODO_RECORD}",
                "creators": GELGENIE_CREATORS,
                "data_collectors": GELGENIE_DATA_COLLECTORS,
                "citation": GELGENIE_CITATION,
                "published_labelled_images": 575,
                "published_counts_by_archive": EXPECTED_LABELLED_IMAGES,
            },
            "rgp_caps": {
                "doi": "10.18908/lsdba.nbdc00318-05-000",
                "license": "CC BY-SA 2.1 Japan",
                "license_url": RGP_LICENSE_URL,
                "required_attribution": RGP_ATTRIBUTION,
            },
        },
        "downloads": source_records,
        "extractions": extraction_records,
        "pair_collection": collection_counts,
        "counts": {
            "usable_images": len(coco_images),
            "train_images": len(split["train"]),
            "validation_images": len(split["val"]),
            "band_boxes": len(coco_annotations),
            "heldout_records": len(heldout_payload["records"]),
            "deduplicated_records": duplicate_count,
            "quarantined_image_groups": quarantined_groups,
            "quarantined_records": quarantined_records,
        },
        "generated_files": {
            "annotations.json": {
                "sha256": _digest(annotations_path),
                "size_bytes": annotations_path.stat().st_size,
            },
            "split.json": {
                "sha256": _digest(split_path),
                "size_bytes": split_path.stat().st_size,
            },
            "heldout.json": {
                "sha256": _digest(heldout_path),
                "size_bytes": heldout_path.stat().st_size,
            },
        },
        "conversion": {
            "mask_foreground": "nonzero pixels",
            "connectivity": 8,
            "component_output": "one bounding box per connected component",
            "leakage_policy": (
                "Quarantine every image-SHA group spanning partitions or carrying "
                "conflicting mask SHA-256 values; deduplicate identical pairs "
                "within one partition."
            ),
            "code": {
                "path": "scripts/prepare_public_training_data.py",
                "sha256": _digest(Path(__file__).resolve()),
            },
            "environment": {
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "executable": sys.executable,
                "numpy_version": np.__version__,
                "pillow_version": PILLOW_VERSION,
                "scipy_version": scipy.__version__,
            },
        },
    }
    _atomic_json(output / "dataset_manifest.json", manifest)
    return manifest


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "source_archive",
            "image_source_archive",
            "partition",
            "image",
            "mask",
            "image_sha256",
            "mask_sha256",
            "width",
            "height",
            "boxes",
        )
    }


def prepare(
    output: Path,
    *,
    force: bool = False,
    sources: Iterable[SourceFile] = SOURCES,
) -> dict[str, Any]:
    source_list = tuple(sources)
    extraction_records = extract_all(output, force=force, sources=source_list)
    pairs, rejected, collection_counts = collect_pairs(
        output, sources=source_list
    )
    _validate_official_completeness(
        source_list,
        rejected,
        collection_counts,
    )
    source_records = []
    for source in source_list:
        path = output / "downloads" / source.name
        source_records.append(
            {
                "name": source.name,
                "url": source.url,
                "size_bytes": source.size,
                "digest_algorithm": source.algorithm,
                "digest": source.digest,
                "sha256": _digest(path),
                "collection": source.collection,
            }
        )
    return build_outputs(
        output,
        pairs,
        rejected,
        collection_counts,
        source_records,
        extraction_records,
        force=force,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare the immutable GelGenie v3 public bootstrap "
            "dataset plus the pinned RGP source archive."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/public_gelgenie"),
        help="Generated/download root (default: data/public_gelgenie).",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "download-only", "prepare-only"),
        default="full",
        help="Full pipeline, downloads only, or prepare verified downloads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace invalid generated downloads/extractions under --output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode in {"full", "download-only"}:
        download_all(output, force=args.force)
    if args.mode in {"full", "prepare-only"}:
        manifest = prepare(output, force=args.force)
        counts = manifest["counts"]
        print(
            "prepared: "
            f"{counts['train_images']} train, "
            f"{counts['validation_images']} validation, "
            f"{counts['band_boxes']} boxes, "
            f"{counts['heldout_records']} held-out records"
        )
        print(f"manifest: {output / 'dataset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
