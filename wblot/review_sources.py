"""Prepare provenance-preserving public Western-blot review projects.

This module deliberately separates three states:

* downloaded source payloads (immutable, verified where upstream supplied pins);
* canonical review images (validated single images, with conversions recorded);
* human-approved annotations (created later in BlotGrid).

Automatic GelGenie rectangles are proposals only.  This importer never marks a
rectangle approved, never marks an image complete, and never manufactures a
grid-filling rectangle.
"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import html
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote, unquote, urlsplit
import urllib.request
import xml.parsers.expat
import zipfile
import zlib

import numpy as np
from PIL import Image, UnidentifiedImageError


class ReviewSourceError(RuntimeError):
    """A source, archive, image, or generated review artifact failed closed."""


SCHEMA = "blotgrid.public_western_review"
SCHEMA_VERSION = 1
CHUNK_BYTES = 1024 * 1024
MAX_DOWNLOAD_BYTES = 4 * 1024**3
MAX_HPA_IMAGE_BYTES = 32 * 1024**2
# The release XML exceeds the 32-bit ISIZE field used by gzip. Keep a finite
# ceiling well above the pinned release while rejecting decompression bombs.
MAX_HPA_XML_UNCOMPRESSED_BYTES = 16 * 1024**3
MAX_IMAGE_PIXELS = 100_000_000
MAX_IMAGE_DIMENSION = 100_000
MAX_TIFF_ARRAY_BYTES = 512 * 1024**2
MAX_REVIEW_PROPOSAL_PIXELS = 1_500_000
MAX_ZIP_MEMBERS = 20_000
MAX_ZIP_MEMBER_BYTES = 4 * 1024**3
MAX_ZIP_EXTRACTED_BYTES = 50 * 1024**3
MAX_ZIP_COMPRESSION_RATIO = 250.0
IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
}

HPA_XML_SHA256 = (
    "a8d12feab9d1c9846787fdd3f5efbe843d4247a898227830ff870ac18287f8ca"
)
HPA_UNIQUE_TSV_SHA256 = (
    "b30d43925a9ca43baa39149d11040f5e5e01a0fe248bd201e37cf9e8451fc5d0"
)
HPA_REVIEW_TSV_SHA256 = (
    "78382664e207c2f41980f9053cafdd4134217f4767a360fe79928f44c634b9fa"
)
HPA_IMPORT_JSON_SHA256 = (
    "be80a441a7d1e08b10cb0d49f0dd7cc78fa06f7be049499a6223df80e37945b4"
)
MENDELEY_PAYLOAD_MANIFEST_SHA256 = (
    "b13e4dd402d9d65c3ce54e6d8367f6da30a0c3d2d50f8d0ee4d5c40f46d0d7e3"
)
MENDELEY_BULK_MANIFEST_SHA256 = (
    "eb9745771b373a3b6fac115161e495c39132d26142db1a9ccdf6e718c1056275"
)
BIA2849_PAYLOAD_MANIFEST_SHA256 = (
    "c9d776e2b5d2ac94e61aac4da41e94f3cfe6b8b2fd102824034d378721d7f7b1"
)
BIA2306_PAYLOAD_MANIFEST_SHA256 = (
    "f170edd86cdee1ae539ad67fa58039b98e7ff73048ab5b3a11855b7b46ac51ef"
)
ZENODO_PAYLOAD_MANIFEST_SHA256 = (
    "aeba4b1632e748a477121aefe23bb9067a6a6e04f6229d25be239f3480b066e2"
)
HPA_UNIQUE_COUNT = 11_968
HPA_REVIEW_COUNT = 256
HPA_REVIEW_STRATA = {"enhanced": 32, "supported": 128, "uncertain": 96}
HPA_PRIMARY_DOI = "10.1126/science.1260419"

CC_BY_4 = {
    "id": 1,
    "name": "Creative Commons Attribution 4.0 International",
    "url": "https://creativecommons.org/licenses/by/4.0/",
    "spdx_identifier": "CC-BY-4.0",
    "publication_assessment": {
        "status": "cleared",
        "redistribution_cleared": True,
        "commercial_use_cleared": True,
        "attribution_required": True,
        "share_alike_required": False,
        "obligations": ["attribution"],
    },
}
CC0_1 = {
    "id": 2,
    "name": "Creative Commons Zero v1.0 Universal",
    "url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "spdx_identifier": "CC0-1.0",
    "publication_assessment": {
        "status": "cleared",
        "redistribution_cleared": True,
        "commercial_use_cleared": True,
        "attribution_required": False,
        "share_alike_required": False,
        "obligations": [],
    },
}
LICENSES = [CC_BY_4, CC0_1]

HPA_CITATION = (
    "Uhlén M et al. Tissue-based map of the human proteome. Science. "
    "2015;347(6220):1260419. doi:10.1126/science.1260419"
)
MENDELEY_CITATION = (
    "Zhou H, Stamler JS (2023). An enzyme that selectively S-nitrosylates "
    "proteins to regulate insulin signaling. Mendeley Data, V2. "
    "doi:10.17632/zz3ktsyjds.2"
)
BIOSTUDIES_2849_CITATION = (
    "A strategy for targeted protein degradation in a cell-line with low "
    "homology-directed repair activity. BioImage Archive. "
    "doi:10.6019/S-BIAD2849"
)
BIOSTUDIES_2306_CITATION = (
    "Ubiquitin pathway blockade reveals endogenous ADP-ribosylation marking "
    "PARP7 and AHR for degradation. BioImage Archive. "
    "doi:10.6019/S-BIAD2306"
)
ZENODO_CITATION = (
    "Loft A (2021). Western blot images related to \"A macrophage-hepatocyte "
    "glucocorticoid receptor axis coordinates fasting ketogenesis\". Zenodo. "
    "doi:10.5281/zenodo.5810203"
)

SOURCES = [
    {
        "id": "hpa_v25_1",
        "name": "Human Protein Atlas Western blot assay panels, release 25.1",
        "version": "25.1",
        "doi": HPA_PRIMARY_DOI,
        "url": "https://www.proteinatlas.org/about/download",
        "license_id": 1,
        "required_attribution": HPA_CITATION,
        "publication_assessment": {
            "status": "cleared",
            "redistribution_cleared": True,
            "attribution_resolved": True,
            "license_id": 1,
        },
    },
    {
        "id": "mendeley_zz3ktsyjds_v2",
        "name": (
            "An enzyme that selectively S-nitrosylates proteins to regulate "
            "insulin signaling"
        ),
        "version": "2",
        "doi": "10.17632/zz3ktsyjds.2",
        "url": "https://data.mendeley.com/datasets/zz3ktsyjds/2",
        "license_id": 1,
        "required_attribution": MENDELEY_CITATION,
        "publication_assessment": {
            "status": "cleared",
            "redistribution_cleared": True,
            "attribution_resolved": True,
            "license_id": 1,
        },
    },
    {
        "id": "bia_s_biad2849",
        "name": (
            "A strategy for targeted protein degradation in a cell-line with "
            "low homology-directed repair activity"
        ),
        "doi": "10.6019/S-BIAD2849",
        "url": "https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2849",
        "license_id": 2,
        "required_attribution": BIOSTUDIES_2849_CITATION,
        "publication_assessment": {
            "status": "cleared",
            "redistribution_cleared": True,
            "attribution_resolved": True,
            "license_id": 2,
        },
    },
    {
        "id": "bia_s_biad2306",
        "name": (
            "Ubiquitin pathway blockade reveals endogenous ADP-ribosylation "
            "marking PARP7 and AHR for degradation"
        ),
        "doi": "10.6019/S-BIAD2306",
        "url": "https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2306",
        "license_id": 2,
        "required_attribution": BIOSTUDIES_2306_CITATION,
        "publication_assessment": {
            "status": "cleared",
            "redistribution_cleared": True,
            "attribution_resolved": True,
            "license_id": 2,
        },
    },
    {
        "id": "zenodo_5810203",
        "name": (
            "Western blot images related to \"A macrophage-hepatocyte "
            "glucocorticoid receptor axis coordinates fasting ketogenesis\""
        ),
        "doi": "10.5281/zenodo.5810203",
        "url": "https://zenodo.org/records/5810203",
        "license_id": 1,
        "required_attribution": ZENODO_CITATION,
        "publication_assessment": {
            "status": "cleared",
            "redistribution_cleared": True,
            "attribution_resolved": True,
            "license_id": 1,
        },
    },
]
SOURCES_BY_ID = {str(source["id"]): source for source in SOURCES}


@dataclass(frozen=True)
class DownloadSpec:
    """One exact upstream payload and its local destination."""

    key: str
    url: str
    destination: Path
    size: int | None = None
    algorithm: str | None = None
    digest: str | None = None
    max_bytes: int = MAX_DOWNLOAD_BYTES


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Return a streaming digest without loading a scientific payload in RAM."""
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewSourceError(f"Could not read valid JSON from {path}.") from exc


def _load_pinned_json(path: Path, expected_sha256: str) -> Any:
    if (
        path.is_symlink()
        or not path.is_file()
        or file_digest(path) != expected_sha256
    ):
        raise ReviewSourceError(
            f"Pinned metadata manifest changed or is missing: {path}."
        )
    return _load_json(path)


def _require_list_manifest(
    path: Path,
    expected_sha256: str,
) -> list[dict[str, Any]]:
    payload = _load_pinned_json(path, expected_sha256)
    if not isinstance(payload, list) or not all(
        isinstance(record, dict) for record in payload
    ):
        raise ReviewSourceError(f"{path} must be a JSON list of objects.")
    return payload


def _safe_manifest_path(value: Any, *, label: str) -> PurePosixPath:
    try:
        return _safe_member_path(str(value))
    except ReviewSourceError as exc:
        raise ReviewSourceError(f"Unsafe {label}: {value!r}.") from exc


def _safe_basename(value: Any, *, label: str) -> str:
    relative = _safe_manifest_path(value, label=label)
    if len(relative.parts) != 1:
        raise ReviewSourceError(f"{label} must be a basename: {value!r}.")
    return relative.name


def _contained_destination(base: Path, basename: str) -> Path:
    name = _safe_basename(basename, label="download filename")
    destination = base / name
    resolved_base = base.resolve()
    resolved_destination = destination.resolve()
    if (
        resolved_destination == resolved_base
        or resolved_base not in resolved_destination.parents
    ):
        raise ReviewSourceError(
            f"Download destination escapes its source directory: {destination}."
        )
    return destination


def _require_https_url(
    value: Any,
    *,
    host: str,
    path_prefix: str,
    exact_path: str | None = None,
    label: str,
) -> str:
    url = str(value)
    parsed = urlsplit(url)
    decoded_path = unquote(parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.hostname != host
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not decoded_path.startswith(path_prefix)
        or (exact_path is not None and decoded_path != exact_path)
    ):
        raise ReviewSourceError(f"{label} has an unexpected URL: {url!r}.")
    return url


def _valid_single_frame_jpeg(path: Path) -> bool:
    """Return whether *path* is a fully decodable, single-frame JPEG."""
    try:
        with Image.open(path) as image:
            if (
                image.format != "JPEG"
                or int(getattr(image, "n_frames", 1)) != 1
                or image.width < 1
                or image.height < 1
                or image.width > MAX_IMAGE_DIMENSION
                or image.height > MAX_IMAGE_DIMENSION
                or image.width * image.height > MAX_IMAGE_PIXELS
            ):
                return False
            image.load()
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError, UnidentifiedImageError):
        return False
    return True


def _existing_download_verified(spec: DownloadSpec) -> bool:
    path = spec.destination
    if path.is_symlink() or not path.is_file():
        return False
    size = path.stat().st_size
    if spec.size is not None and size != spec.size:
        return False
    if size > spec.max_bytes:
        return False
    if spec.key.startswith("hpa:") and (
        path.name != _basename_from_url(spec.url)
        or not _valid_single_frame_jpeg(path)
    ):
        return False
    if spec.algorithm and spec.digest:
        return file_digest(path, spec.algorithm).lower() == spec.digest.lower()
    receipt_path = path.with_name(path.name + ".download.json")
    if receipt_path.is_file():
        receipt = _load_json(receipt_path)
        return (
            receipt.get("url") == spec.url
            and receipt.get("bytes") == size
            and receipt.get("sha256") == file_digest(path)
        )
    # A byte count is not a content identity. Checksum-less BioStudies assets
    # must originate in this downloader over the pinned HTTPS URL before their
    # local SHA-256 receipt can be trusted on later runs.
    return False


def _write_download_receipt(spec: DownloadSpec) -> dict[str, Any]:
    path = spec.destination
    receipt = {
        "schema": "blotgrid.verified_download",
        "schema_version": 1,
        "key": spec.key,
        "url": spec.url,
        "bytes": path.stat().st_size,
        "sha256": file_digest(path),
        "upstream_digest": (
            f"{spec.algorithm}:{spec.digest}"
            if spec.algorithm and spec.digest
            else None
        ),
        "upstream_size": spec.size,
    }
    _atomic_json(path.with_name(path.name + ".download.json"), receipt)
    return receipt


def _validate_download_url(key: str, value: Any) -> str:
    """Enforce the immutable upstream namespace associated with a spec key."""
    if key.startswith("hpa:"):
        image_id = key.split(":", 1)[1]
        if not re.fullmatch(r"[0-9]+", image_id):
            raise ReviewSourceError(f"Invalid HPA image key: {key!r}.")
        url = _require_https_url(
            value,
            host="v25.proteinatlas.org",
            path_prefix="/images_",
            label=key,
        )
        path = unquote(urlsplit(url).path)
        expected_paths = {
            f"/images_western_blot/{image_id}_medium.jpg",
            f"/images_wb_sirna/{image_id}_medium.jpg",
        }
        if path not in expected_paths:
            raise ReviewSourceError(f"{key} has an unexpected HPA path.")
        return url
    if key.startswith(("S-BIAD2849:", "S-BIAD2306:")):
        accession, source_path = key.split(":", 1)
        url = str(value)
        parsed = urlsplit(url)
        decoded_path = unquote(parsed.path)
        primary_path = f"/biostudies/files/{accession}/{source_path}"
        fire_mirror_path = (
            f"/biostudies/fire/S-BIAD/{accession[-3:]}/{accession}/"
            f"Files/{source_path}"
        )
        public_mirror_path = (
            f"/pub/databases/biostudies/S-BIAD/{accession[-3:]}/"
            f"{accession}/Files/{source_path}"
        )
        valid_location = (
            parsed.hostname == "www.ebi.ac.uk"
            and decoded_path == primary_path
        ) or (
            parsed.hostname == "ftp.ebi.ac.uk"
            and decoded_path in {fire_mirror_path, public_mirror_path}
        )
        if (
            parsed.scheme != "https"
            or parsed.port is not None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not valid_location
        ):
            raise ReviewSourceError(
                f"{key} has an unexpected URL: {url!r}."
            )
        return url
    if key.startswith("zenodo:5810203:"):
        return _require_https_url(
            value,
            host="zenodo.org",
            path_prefix="/api/records/5810203/files/",
            label=key,
        )
    if key == "mendeley:zz3ktsyjds-v2.zip":
        return _require_https_url(
            value,
            host="prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com",
            path_prefix="/zz3ktsyjds-2.zip",
            exact_path="/zz3ktsyjds-2.zip",
            label=key,
        )
    raise ReviewSourceError(f"Unknown download source key: {key!r}.")


def download_verified(
    spec: DownloadSpec,
    *,
    force: bool = False,
    timeout: int = 120,
    attempts: int = 3,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Atomically download one bounded payload and write a SHA-256 receipt."""
    _validate_download_url(spec.key, spec.url)
    spec.destination.parent.mkdir(parents=True, exist_ok=True)
    if _existing_download_verified(spec):
        receipt = _write_download_receipt(spec)
        receipt["status"] = "already_verified"
        return receipt
    if spec.destination.exists() and not force:
        raise ReviewSourceError(
            f"Existing payload does not match its source pin: {spec.destination}. "
            "Use --force only for generated download payloads."
        )

    partial = spec.destination.with_name(spec.destination.name + ".part")
    partial.unlink(missing_ok=True)
    last_error: BaseException | None = None
    for attempt in range(max(1, attempts)):
        request = urllib.request.Request(
            spec.url,
            headers={"User-Agent": "BlotGrid-public-western/1.0"},
        )
        try:
            with opener(request, timeout=timeout) as response:
                if not (spec.algorithm and spec.digest):
                    final_url_getter = getattr(response, "geturl", None)
                    final_url = (
                        final_url_getter()
                        if callable(final_url_getter)
                        else spec.url
                    )
                    _validate_download_url(spec.key, final_url)
                if spec.key.startswith("hpa:"):
                    content_type = str(
                        response.headers.get("Content-Type", "")
                    ).split(";", 1)[0].strip().lower()
                    if content_type not in {"image/jpeg", "image/jpg"}:
                        raise ReviewSourceError(
                            f"{spec.key} returned unexpected content type "
                            f"{content_type or '<missing>'}."
                        )
                declared_text = response.headers.get("Content-Length")
                try:
                    declared = (
                        int(declared_text)
                        if declared_text is not None
                        else None
                    )
                except (TypeError, ValueError) as exc:
                    raise ReviewSourceError(
                        f"{spec.key} returned an invalid Content-Length."
                    ) from exc
                if declared is not None and declared < 0:
                    raise ReviewSourceError(
                        f"{spec.key} returned a negative Content-Length."
                    )
                if spec.size is not None and declared is not None:
                    if declared != spec.size:
                        raise ReviewSourceError(
                            f"{spec.key} server size changed: "
                            f"{declared} != {spec.size}."
                        )
                if declared is not None and declared > spec.max_bytes:
                    raise ReviewSourceError(
                        f"{spec.key} exceeds its download byte limit."
                    )
                received = 0
                upstream_hasher = (
                    hashlib.new(spec.algorithm) if spec.algorithm else None
                )
                with partial.open("xb") as output:
                    while True:
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > spec.max_bytes:
                            raise ReviewSourceError(
                                f"{spec.key} exceeded its download byte limit."
                            )
                        if spec.size is not None and received > spec.size:
                            raise ReviewSourceError(
                                f"{spec.key} exceeded its pinned byte count."
                            )
                        output.write(chunk)
                        if upstream_hasher is not None:
                            upstream_hasher.update(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            if spec.size is not None and received != spec.size:
                raise ReviewSourceError(
                    f"{spec.key} is truncated: {received} != {spec.size} bytes."
                )
            if declared is not None and received != declared:
                raise ReviewSourceError(
                    f"{spec.key} response length changed: "
                    f"{received} != {declared} bytes."
                )
            if spec.algorithm and spec.digest:
                assert upstream_hasher is not None
                if upstream_hasher.hexdigest().lower() != spec.digest.lower():
                    raise ReviewSourceError(
                        f"{spec.key} checksum does not match its upstream pin."
                    )
            if spec.key.startswith("hpa:") and not _valid_single_frame_jpeg(
                partial
            ):
                raise ReviewSourceError(
                    f"{spec.key} is not a valid single-frame JPEG."
                )
            os.replace(partial, spec.destination)
            receipt = _write_download_receipt(spec)
            receipt["status"] = "downloaded"
            return receipt
        except Exception as exc:
            partial.unlink(missing_ok=True)
            last_error = exc
            if attempt + 1 < max(1, attempts):
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def _safe_member_path(name: str) -> PurePosixPath:
    if "\x00" in name or "\\" in name:
        raise ReviewSourceError(f"Unsafe ZIP member path: {name!r}.")
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ReviewSourceError(f"Unsafe ZIP member path: {name!r}.")
    return relative


def _validated_zip_members(archive: Path) -> list[zipfile.ZipInfo]:
    try:
        bundle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReviewSourceError(f"{archive} is not a valid ZIP archive.") from exc
    with bundle:
        members = bundle.infolist()
        if len(members) > MAX_ZIP_MEMBERS:
            raise ReviewSourceError(
                f"{archive.name} contains too many members: {len(members)}."
            )
        seen: set[str] = set()
        total = 0
        for info in members:
            relative = _safe_member_path(info.orig_filename)
            normalized = relative.as_posix()
            if normalized in seen:
                raise ReviewSourceError(
                    f"{archive.name} has a duplicate member: {normalized}."
                )
            seen.add(normalized)
            if info.flag_bits & 0x1:
                raise ReviewSourceError(
                    f"{archive.name} contains an encrypted member: {normalized}."
                )
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ReviewSourceError(
                    f"{archive.name} contains a symbolic link: {normalized}."
                )
            if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ReviewSourceError(
                    f"{archive.name} contains a special file: {normalized}."
                )
            if info.file_size > MAX_ZIP_MEMBER_BYTES:
                raise ReviewSourceError(
                    f"{archive.name} member is too large: {normalized}."
                )
            total += info.file_size
            if total > MAX_ZIP_EXTRACTED_BYTES:
                raise ReviewSourceError(
                    f"{archive.name} exceeds the extraction byte limit."
                )
            if (
                info.file_size > CHUNK_BYTES
                and info.compress_size > 0
                and info.file_size / info.compress_size
                > MAX_ZIP_COMPRESSION_RATIO
            ):
                raise ReviewSourceError(
                    f"{archive.name} contains a compression-ratio bomb: "
                    f"{normalized}."
                )
        return members


def _extraction_matches(
    archive: Path,
    destination: Path,
    selected_names: set[str],
) -> bool:
    try:
        with zipfile.ZipFile(archive) as bundle:
            info_by_name = {
                _safe_member_path(info.orig_filename).as_posix(): info
                for info in bundle.infolist()
                if not info.is_dir()
            }
            for name in selected_names:
                info = info_by_name.get(name)
                path = destination.joinpath(*PurePosixPath(name).parts)
                if (
                    info is None
                    or path.is_symlink()
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
        actual = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file() and path.name != ".blotgrid-extraction.json"
        }
        return actual == selected_names
    except (
        OSError,
        ValueError,
        zipfile.BadZipFile,
        ReviewSourceError,
    ):
        return False


def safe_extract_zip(
    archive: Path,
    destination: Path,
    *,
    select: Callable[[str], bool] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Safely and atomically extract selected regular files from a ZIP."""
    members = _validated_zip_members(archive)
    selected = [
        info
        for info in members
        if not info.is_dir()
        and (select is None or select(_safe_member_path(info.orig_filename).as_posix()))
    ]
    selected_names = {
        _safe_member_path(info.orig_filename).as_posix() for info in selected
    }
    archive_sha256 = file_digest(archive)
    marker = destination / ".blotgrid-extraction.json"
    if marker.is_file():
        payload = _load_json(marker)
        if (
            payload.get("archive_sha256") == archive_sha256
            and payload.get("selected_members") == sorted(selected_names)
            and _extraction_matches(archive, destination, selected_names)
        ):
            return {**payload, "status": "already_verified"}
    if destination.exists() and not force:
        raise ReviewSourceError(
            f"Extraction directory is incomplete or stale: {destination}. "
            "Use --force only for generated extraction directories."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        with zipfile.ZipFile(archive) as bundle:
            for info in selected:
                relative = _safe_member_path(info.orig_filename)
                target = temporary.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                copied = 0
                with bundle.open(info) as source, target.open("xb") as output:
                    while True:
                        chunk = source.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        copied += len(chunk)
                        if copied > info.file_size:
                            raise ReviewSourceError(
                                "ZIP member expanded past its declared size: "
                                f"{info.orig_filename}."
                            )
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if copied != info.file_size:
                    raise ReviewSourceError(
                        f"ZIP member is truncated: {info.orig_filename}."
                    )
        receipt = {
            "schema": "blotgrid.safe_zip_extraction",
            "schema_version": 1,
            "archive": archive.name,
            "archive_sha256": archive_sha256,
            "selected_members": sorted(selected_names),
            "selected_bytes": sum(info.file_size for info in selected),
        }
        _atomic_json(temporary / marker.name, receipt)
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
        return {**receipt, "status": "extracted"}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def stream_hpa_western_metadata(xml_gz: Path) -> list[dict[str, Any]]:
    """Stream official HPA XML and deduplicate HPA-owned WB/WBsiRNA URLs.

    The returned records are intentionally compact.  The separately pinned rich
    import JSON is used for review metadata after its URL set is proven equal to
    this independent parse of the official bulk XML.
    """
    if not xml_gz.is_file() or xml_gz.is_symlink():
        raise ReviewSourceError(f"HPA XML is missing or unsafe: {xml_gz}.")

    by_url: dict[str, dict[str, Any]] = {}
    stack: list[str] = []
    entry: dict[str, str] | None = None
    antibody_id = ""
    blot: dict[str, Any] | None = None
    image_type = ""
    capture: str | None = None
    text_parts: list[str] = []

    def local(name: str) -> str:
        return name.rsplit("}", 1)[-1].rsplit(":", 1)[-1]

    def start(name: str, attrs: Mapping[str, str]) -> None:
        nonlocal entry, antibody_id, blot, image_type, capture, text_parts
        tag = local(name)
        parent = stack[-1] if stack else ""
        stack.append(tag)
        if tag == "entry":
            entry = {
                "url": str(attrs.get("url", "")).strip(),
                "version": str(attrs.get("version", "")).strip(),
                "gene_name": "",
                "ensembl_id": "",
            }
        elif tag == "name" and parent == "entry" and entry is not None:
            capture = "gene_name"
            text_parts = []
        elif (
            tag == "identifier"
            and parent == "entry"
            and entry is not None
            and attrs.get("db") == "Ensembl"
        ):
            entry["ensembl_id"] = str(attrs.get("id", "")).strip()
        elif tag == "antibody":
            antibody_id = str(attrs.get("id", "")).strip()
        elif tag == "westernBlot":
            source = str(attrs.get("source", "")).strip()
            technology = str(attrs.get("technology", "")).strip()
            blot = {
                "relevant": (
                    source == "HPA"
                    and technology in {"WB", "WBsiRNA"}
                ),
                "source": source,
                "technology": technology,
                "validation": "",
                "description": "",
                "lanes": [],
            }
        elif tag == "verification" and blot and blot["relevant"]:
            blot["description"] = str(
                attrs.get("description", "")
            ).strip()
            capture = "validation"
            text_parts = []
        elif tag == "lane" and blot and blot["relevant"]:
            blot["lanes"].append(
                {
                    "lane_id": attrs.get("laneId"),
                    "content": attrs.get("laneContent", ""),
                }
            )
        elif tag == "image" and blot and blot["relevant"]:
            image_type = str(attrs.get("imageType", "")).strip()
        elif tag == "imageUrl" and blot and blot["relevant"]:
            capture = "image_url"
            text_parts = []

    def characters(value: str) -> None:
        if capture is not None:
            text_parts.append(value)

    def end(name: str) -> None:
        nonlocal entry, antibody_id, blot, image_type, capture, text_parts
        tag = local(name)
        value = "".join(text_parts).strip() if capture is not None else ""
        if tag == "name" and capture == "gene_name" and entry is not None:
            entry["gene_name"] = value
            capture = None
            text_parts = []
        elif tag == "verification" and capture == "validation" and blot:
            blot["validation"] = value.lower()
            capture = None
            text_parts = []
        elif tag == "imageUrl" and capture == "image_url" and blot and entry:
            capture = None
            text_parts = []
            if value:
                mapped = {
                    "image_url": value,
                    "hpa_entry_version": entry["version"],
                    "entry_url": entry["url"],
                    "ensembl_id": entry["ensembl_id"],
                    "gene_name": entry["gene_name"],
                    "antibody_id": antibody_id,
                    "source": blot["source"],
                    "technology": blot["technology"],
                    "validation": blot["validation"],
                    "validation_description": blot["description"],
                    "image_type": image_type,
                    "lanes": list(blot["lanes"]),
                    "source_row_count": 1,
                }
                existing = by_url.get(value)
                if existing is None:
                    by_url[value] = mapped
                else:
                    existing["source_row_count"] = (
                        int(existing["source_row_count"]) + 1
                    )
                    for key in ("entry_url", "ensembl_id", "gene_name"):
                        aliases = existing.setdefault(
                            f"{key}_aliases",
                            [existing[key]],
                        )
                        if mapped[key] not in aliases:
                            aliases.append(mapped[key])
        if tag == "image":
            image_type = ""
        elif tag == "westernBlot":
            blot = None
        elif tag == "antibody":
            antibody_id = ""
        elif tag == "entry":
            entry = None
        if stack:
            stack.pop()

    try:
        parser = xml.parsers.expat.ParserCreate(namespace_separator="}")
        parser.StartElementHandler = start
        parser.CharacterDataHandler = characters
        parser.EndElementHandler = end
        hasher = hashlib.sha256()
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        decoded_bytes = 0
        with xml_gz.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
                hasher.update(chunk)
                decoded = decompressor.decompress(chunk)
                decoded_bytes += len(decoded)
                if decoded_bytes > MAX_HPA_XML_UNCOMPRESSED_BYTES:
                    raise ReviewSourceError(
                        "HPA XML exceeds its decompression byte limit."
                    )
                if decoded:
                    parser.Parse(decoded, False)
        final = decompressor.flush()
        decoded_bytes += len(final)
        if decoded_bytes > MAX_HPA_XML_UNCOMPRESSED_BYTES:
            raise ReviewSourceError(
                "HPA XML exceeds its decompression byte limit."
            )
        if final:
            parser.Parse(final, False)
        if (
            not decompressor.eof
            or decompressor.unused_data
            or hasher.hexdigest() != HPA_XML_SHA256
        ):
            raise ReviewSourceError(
                "HPA XML SHA-256 or gzip framing does not match release 25.1."
            )
        parser.Parse(b"", True)
    except (
        OSError,
        EOFError,
        zlib.error,
        xml.parsers.expat.ExpatError,
    ) as exc:
        raise ReviewSourceError("HPA release XML could not be parsed.") from exc
    return [by_url[url] for url in sorted(by_url)]


def stream_hpa_western_urls(xml_gz: Path) -> set[str]:
    """Quickly validate the exact HPA-owned WB URL population.

    The full release would require a very large number of XML callbacks. This
    scanner still
    decompresses and hashes every byte, but only retains complete
    ``westernBlot`` elements before checking their explicit source and
    technology attributes.
    """
    if not xml_gz.is_file() or xml_gz.is_symlink():
        raise ReviewSourceError(f"HPA XML is missing or unsafe: {xml_gz}.")
    opening = b"<westernBlot"
    closing = b"</westernBlot>"
    maximum_block_bytes = 16 * 1024**2
    buffer = b""
    urls: set[str] = set()
    attribute = re.compile(
        rb"\b(source|technology)\s*=\s*([\"'])(.*?)\2",
        re.DOTALL,
    )
    image_url = re.compile(
        rb"<imageUrl(?:\s[^>]*)?>(.*?)</imageUrl\s*>",
        re.DOTALL,
    )

    def consume(decoded: bytes) -> None:
        nonlocal buffer
        buffer += decoded
        while True:
            start = buffer.find(opening)
            if start < 0:
                buffer = buffer[-(len(opening) - 1) :]
                return
            end = buffer.find(closing, start + len(opening))
            if end < 0:
                buffer = buffer[start:]
                if len(buffer) > maximum_block_bytes:
                    raise ReviewSourceError(
                        "HPA westernBlot element exceeds its byte limit."
                    )
                return
            end += len(closing)
            block = buffer[start:end]
            buffer = buffer[end:]
            tag_end = block.find(b">")
            if tag_end < 0:
                raise ReviewSourceError("Malformed HPA westernBlot start tag.")
            attributes = {
                match.group(1).decode("ascii"): match.group(3).decode(
                    "utf-8"
                ).strip()
                for match in attribute.finditer(block[: tag_end + 1])
            }
            if (
                attributes.get("source") != "HPA"
                or attributes.get("technology") not in {"WB", "WBsiRNA"}
            ):
                continue
            for match in image_url.finditer(block):
                value = html.unescape(
                    match.group(1).decode("utf-8").strip()
                )
                if not re.fullmatch(
                    r"https://v25\.proteinatlas\.org/"
                    r"(?:images_western_blot|images_wb_sirna)/"
                    r"[0-9]+_medium\.jpg",
                    value,
                ):
                    raise ReviewSourceError(
                        f"Unexpected HPA Western image URL: {value!r}."
                    )
                urls.add(value)

    try:
        hasher = hashlib.sha256()
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        decoded_bytes = 0
        with xml_gz.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
                hasher.update(chunk)
                decoded = decompressor.decompress(chunk)
                decoded_bytes += len(decoded)
                if decoded_bytes > MAX_HPA_XML_UNCOMPRESSED_BYTES:
                    raise ReviewSourceError(
                        "HPA XML exceeds its decompression byte limit."
                    )
                if decoded:
                    consume(decoded)
        final = decompressor.flush()
        decoded_bytes += len(final)
        if decoded_bytes > MAX_HPA_XML_UNCOMPRESSED_BYTES:
            raise ReviewSourceError(
                "HPA XML exceeds its decompression byte limit."
            )
        if final:
            consume(final)
        if opening in buffer:
            raise ReviewSourceError("HPA XML ended inside a westernBlot element.")
        if (
            not decompressor.eof
            or decompressor.unused_data
            or hasher.hexdigest() != HPA_XML_SHA256
        ):
            raise ReviewSourceError(
                "HPA XML SHA-256 or gzip framing does not match release 25.1."
            )
    except (OSError, EOFError, UnicodeError, zlib.error) as exc:
        raise ReviewSourceError("HPA release XML could not be scanned.") from exc
    return urls


def _load_hpa_import_json(metadata_dir: Path) -> dict[str, Any]:
    path = metadata_dir / "hpa-wb-v25.1-import.json"
    if file_digest(path) != HPA_IMPORT_JSON_SHA256:
        raise ReviewSourceError("HPA rich import JSON SHA-256 changed.")
    payload = _load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema")
        != "blotgrid.hpa_western_blot_metadata"
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("records"), list)
    ):
        raise ReviewSourceError("HPA rich import JSON has an unsupported schema.")
    records = payload["records"]
    if len(records) != HPA_UNIQUE_COUNT:
        raise ReviewSourceError(
            f"HPA import JSON has {len(records)} records; "
            f"expected {HPA_UNIQUE_COUNT}."
        )
    urls: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ReviewSourceError("HPA import records must be objects.")
        image = record.get("image")
        western = record.get("western_blot")
        if not isinstance(image, dict) or not isinstance(western, dict):
            raise ReviewSourceError("HPA import record lacks image/WB metadata.")
        url = str(image.get("url", ""))
        if (
            western.get("source") != "HPA"
            or western.get("technology") not in {"WB", "WBsiRNA"}
            or not re.fullmatch(
                r"https://v25\.proteinatlas\.org/"
                r"(?:images_western_blot|images_wb_sirna)/"
                r"[^/]+_medium\.jpg",
                url,
            )
        ):
            raise ReviewSourceError(
                f"HPA import record is outside the pinned WB scope: {url!r}."
            )
        if url in urls:
            raise ReviewSourceError(f"Duplicate HPA image URL: {url}.")
        urls.add(url)
    dataset = payload.get("dataset")
    if (
        not isinstance(dataset, dict)
        or dataset.get("release") != "25.1"
        or dataset.get("records") != HPA_UNIQUE_COUNT
        or dataset.get("source_metadata_sha256") != HPA_XML_SHA256
    ):
        raise ReviewSourceError("HPA import dataset metadata is inconsistent.")
    return payload


def _load_tsv(path: Path, expected_sha256: str) -> list[dict[str, str]]:
    if file_digest(path) != expected_sha256:
        raise ReviewSourceError(f"Pinned TSV SHA-256 changed: {path}.")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewSourceError(f"Could not parse pinned TSV: {path}.") from exc


def load_hpa_metadata(
    root: Path,
    *,
    validate_xml: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load rich HPA metadata and its deterministic stratified review subset."""
    metadata_dir = root / "downloads" / "metadata"
    payload = _load_hpa_import_json(metadata_dir)
    records = payload["records"]
    by_url = {
        str(record["image"]["url"]): record
        for record in records
    }

    unique_rows = _load_tsv(
        metadata_dir / "hpa-wb-v25.1-unique.tsv",
        HPA_UNIQUE_TSV_SHA256,
    )
    if len(unique_rows) != HPA_UNIQUE_COUNT:
        raise ReviewSourceError("Pinned HPA unique TSV row count changed.")
    unique_urls = [row.get("image_url", "") for row in unique_rows]
    if len(set(unique_urls)) != HPA_UNIQUE_COUNT or set(unique_urls) != set(
        by_url
    ):
        raise ReviewSourceError(
            "HPA rich import JSON and unique TSV URL sets differ."
        )

    review_rows = _load_tsv(
        metadata_dir / "hpa-wb-v25.1-review-256.tsv",
        HPA_REVIEW_TSV_SHA256,
    )
    if len(review_rows) != HPA_REVIEW_COUNT:
        raise ReviewSourceError("Pinned HPA review TSV row count changed.")
    strata: dict[str, int] = {}
    review_urls: list[str] = []
    for row in review_rows:
        url = row.get("image_url", "")
        validation = row.get("wb_validation", "")
        if url not in by_url:
            raise ReviewSourceError(
                f"HPA review subset URL is absent from full index: {url}."
            )
        strata[validation] = strata.get(validation, 0) + 1
        review_urls.append(url)
    if strata != HPA_REVIEW_STRATA or len(set(review_urls)) != HPA_REVIEW_COUNT:
        raise ReviewSourceError(
            "HPA review subset is not the pinned 32/128/96 stratification."
        )

    parser_receipt = root / "downloads" / "hpa" / "parser-validation.json"
    xml_gz = root / "downloads" / "hpa" / "proteinatlas.xml.gz"
    receipt_valid = False
    if parser_receipt.is_file():
        receipt = _load_json(parser_receipt)
        receipt_valid = (
            receipt.get("xml_sha256") == HPA_XML_SHA256
            and receipt.get("import_json_sha256") == HPA_IMPORT_JSON_SHA256
            and receipt.get("unique_urls") == HPA_UNIQUE_COUNT
            and xml_gz.is_file()
            and xml_gz.stat().st_size == 729_899_935
            and file_digest(xml_gz) == HPA_XML_SHA256
        )
    if validate_xml and not receipt_valid:
        parsed_urls = stream_hpa_western_urls(xml_gz)
        if parsed_urls != set(by_url):
            missing = sorted(set(by_url) - parsed_urls)[:3]
            extra = sorted(parsed_urls - set(by_url))[:3]
            raise ReviewSourceError(
                "Independent HPA XML parse disagrees with pinned import JSON; "
                f"missing={missing}, extra={extra}."
            )
        _atomic_json(
            parser_receipt,
            {
                "schema": "blotgrid.hpa_xml_parser_validation",
                "schema_version": 1,
                "xml_sha256": HPA_XML_SHA256,
                "import_json_sha256": HPA_IMPORT_JSON_SHA256,
                "unique_urls": len(parsed_urls),
                "scope": {
                    "source": "HPA",
                    "technology": ["WB", "WBsiRNA"],
                    "deduplicated_by": "image_url",
                },
            },
        )
        receipt_valid = True
    if validate_xml and not receipt_valid:
        raise ReviewSourceError(
            "HPA XML scope has not passed independent pinned validation. "
            "Run the fetch command without --skip-hpa-xml-revalidation."
        )
    return records, [by_url[url] for url in review_urls]


def _manifest_paths(root: Path) -> dict[str, Path]:
    metadata = root / "downloads" / "metadata"
    return {
        "mendeley": metadata / "zz3ktsyjds-v2-payload-manifest.json",
        "mendeley_bulk": metadata / "zz3ktsyjds-v2-bulk-zip-metadata.json",
        "bia2849": metadata / "S-BIAD2849-wb-payload-manifest.json",
        "bia2306": metadata / "S-BIAD2306-wb-payload-manifest.json",
        "zenodo": metadata / "zenodo-5810203-payload-manifest.json",
    }


def _validate_biostudies_records(
    records: Sequence[Mapping[str, Any]],
    *,
    accession: str,
    role: str,
    suffixes: set[str],
) -> None:
    paths: set[str] = set()
    basenames: set[str] = set()
    for record in records:
        value = str(record.get("path", ""))
        relative = _safe_manifest_path(
            value,
            label=f"{accession} source path",
        )
        basename = relative.name
        if (
            not value.isascii()
            or ":" in value
            or value in paths
            or basename in basenames
            or record.get("type") != "file"
            or record.get("role") != role
            or Path(basename).suffix.lower() not in suffixes
            or int(record.get("size", -1)) < 1
        ):
            raise ReviewSourceError(
                f"{accession} source manifest entry is invalid."
            )
        paths.add(value)
        basenames.add(basename)
        expected_url = (
            f"https://www.ebi.ac.uk/biostudies/files/{accession}/"
            f"{quote(value, safe='/')}"
        )
        if record.get("url") != expected_url:
            raise ReviewSourceError(f"{accession} source URL changed.")


def validate_manifests(root: Path) -> dict[str, Any]:
    """Validate exact source selections before any network or extraction work."""
    paths = _manifest_paths(root)
    mendeley = _require_list_manifest(
        paths["mendeley"],
        MENDELEY_PAYLOAD_MANIFEST_SHA256,
    )
    bia2849_all = _require_list_manifest(
        paths["bia2849"],
        BIA2849_PAYLOAD_MANIFEST_SHA256,
    )
    bia2306 = _require_list_manifest(
        paths["bia2306"],
        BIA2306_PAYLOAD_MANIFEST_SHA256,
    )
    zenodo_all = _require_list_manifest(
        paths["zenodo"],
        ZENODO_PAYLOAD_MANIFEST_SHA256,
    )
    bulk = _load_pinned_json(
        paths["mendeley_bulk"],
        MENDELEY_BULK_MANIFEST_SHA256,
    )

    if (
        len(mendeley) != 186
        or any(
            _safe_basename(
                record.get("filename", ""),
                label="Mendeley filename",
            ).lower().endswith(".png")
            is False
            for record in mendeley
        )
        or len({record.get("filename") for record in mendeley}) != 186
        or len({record.get("id") for record in mendeley}) != 186
    ):
        raise ReviewSourceError(
            "Mendeley selection must be exactly 186 uniquely named PNG files."
        )
    for record in mendeley:
        file_id = str(record.get("id", ""))
        if not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}",
            file_id,
        ):
            raise ReviewSourceError("Mendeley file ID is not a canonical UUID.")
        expected_url = (
            "https://data.mendeley.com/public-files/datasets/zz3ktsyjds/"
            f"files/{file_id}/file_downloaded"
        )
        if record.get("download_url") != expected_url:
            raise ReviewSourceError("Mendeley per-file URL changed.")
        if (
            int(record.get("size", -1)) < 1
            or int(record.get("bytes", -1)) != int(record["size"])
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(record.get("sha256", "")),
            )
        ):
            raise ReviewSourceError("Mendeley per-file pin is invalid.")
    bia2849 = [
        record
        for record in bia2849_all
        if record.get("role") == "raw_western_blot_image"
        and Path(str(record.get("path", ""))).suffix.lower()
        in {".tif", ".tiff"}
    ]
    if len(bia2849) != 42:
        raise ReviewSourceError(
            "S-BIAD2849 selection must be exactly 42 raw Western TIFF files."
        )
    _validate_biostudies_records(
        bia2849,
        accession="S-BIAD2849",
        role="raw_western_blot_image",
        suffixes={".tif", ".tiff"},
    )
    if (
        len(bia2306) != 38
        or any(
            record.get("role") != "western_blot_source_archive"
            or Path(str(record.get("path", ""))).suffix.lower() != ".zip"
            for record in bia2306
        )
    ):
        raise ReviewSourceError(
            "S-BIAD2306 selection must be exactly 38 Western source ZIPs."
        )
    _validate_biostudies_records(
        bia2306,
        accession="S-BIAD2306",
        role="western_blot_source_archive",
        suffixes={".zip"},
    )
    zenodo = [
        record
        for record in zenodo_all
        if str(record.get("extension", "")).lower() in {"tif", "tiff"}
    ]
    if len(zenodo) != 20:
        raise ReviewSourceError(
            "Zenodo 5810203 selection must be exactly 20 original TIFF files."
        )
    zenodo_names: set[str] = set()
    for record in zenodo:
        filename = _safe_basename(
            record.get("filename", ""),
            label="Zenodo filename",
        )
        if (
            not filename.isascii()
            or ":" in filename
            or filename in zenodo_names
            or Path(filename).suffix.lower() not in {".tif", ".tiff"}
            or str(record.get("extension", "")).lower()
            != Path(filename).suffix.lower().lstrip(".")
            or int(record.get("bytes", -1)) < 1
            or not re.fullmatch(
                r"md5:[0-9a-f]{32}",
                str(record.get("checksum", "")),
            )
        ):
            raise ReviewSourceError("Zenodo TIFF manifest entry is invalid.")
        zenodo_names.add(filename)
        expected_url = (
            "https://zenodo.org/api/records/5810203/files/"
            f"{quote(filename, safe='')}/content"
        )
        if record.get("download_url") != expected_url:
            raise ReviewSourceError("Zenodo per-file URL changed.")
    if (
        not isinstance(bulk, dict)
        or int(bulk.get("size", -1)) != 579_758_452
        or bulk.get("sha256_hash")
        != "e967686aa374b4090366fd8a0117680f9ee52840d73abddb69780d09543ac78a"
        or bulk.get("url")
        != (
            "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1."
            "amazonaws.com/zz3ktsyjds-2.zip"
        )
    ):
        raise ReviewSourceError("Mendeley bulk ZIP pin changed.")
    return {
        "mendeley": mendeley,
        "mendeley_bulk": bulk,
        "bia2849": bia2849,
        "bia2306": bia2306,
        "zenodo": zenodo,
    }


def _basename_from_url(url: str) -> str:
    name = PurePosixPath(url.split("?", 1)[0]).name
    _safe_member_path(name)
    if "/" in name:
        raise ReviewSourceError(f"URL did not resolve to a basename: {url}.")
    return name


def hpa_download_path(root: Path, record: Mapping[str, Any]) -> Path:
    url = str(record["image"]["url"])
    return root / "downloads" / "hpa" / "images" / _basename_from_url(url)


def download_specs(
    root: Path,
    *,
    validate_hpa_xml: bool = True,
) -> tuple[list[DownloadSpec], dict[str, int]]:
    """Return exact download specs; creating the list performs no download."""
    manifests = validate_manifests(root)
    hpa_records, _review = load_hpa_metadata(
        root,
        validate_xml=validate_hpa_xml,
    )
    specs: list[DownloadSpec] = []

    bulk = manifests["mendeley_bulk"]
    specs.append(
        DownloadSpec(
            key="mendeley:zz3ktsyjds-v2.zip",
            url=str(bulk["url"]),
            destination=(
                root
                / "downloads"
                / "mendeley"
                / "zz3ktsyjds-v2.zip"
            ),
            size=int(bulk["size"]),
            algorithm="sha256",
            digest=str(bulk["sha256_hash"]),
        )
    )
    for record in manifests["bia2849"]:
        basename = _safe_manifest_path(
            record["path"],
            label="S-BIAD2849 source path",
        ).name
        base = (
            root
            / "downloads"
            / "biostudies"
            / "S-BIAD2849"
            / "files"
        )
        specs.append(
            DownloadSpec(
                key=f"S-BIAD2849:{record['path']}",
                url=str(record["url"]),
                destination=_contained_destination(base, basename),
                size=int(record["size"]),
            )
        )
    for record in manifests["bia2306"]:
        basename = _safe_manifest_path(
            record["path"],
            label="S-BIAD2306 source path",
        ).name
        base = (
            root
            / "downloads"
            / "biostudies"
            / "S-BIAD2306"
            / "archives"
        )
        specs.append(
            DownloadSpec(
                key=f"S-BIAD2306:{record['path']}",
                url=str(record["url"]),
                destination=_contained_destination(base, basename),
                size=int(record["size"]),
            )
        )
    for record in manifests["zenodo"]:
        filename = _safe_basename(
            record["filename"],
            label="Zenodo filename",
        )
        checksum = str(record["checksum"])
        algorithm, digest = checksum.split(":", 1)
        base = root / "downloads" / "zenodo" / "5810203" / "files"
        specs.append(
            DownloadSpec(
                key=f"zenodo:5810203:{filename}",
                url=str(record["download_url"]),
                destination=_contained_destination(base, filename),
                size=int(record["bytes"]),
                algorithm=algorithm,
                digest=digest,
            )
        )
    for record in hpa_records:
        url = str(record["image"]["url"])
        specs.append(
            DownloadSpec(
                key=f"hpa:{record['image']['id']}",
                url=url,
                destination=hpa_download_path(root, record),
                max_bytes=MAX_HPA_IMAGE_BYTES,
            )
        )
    return specs, {
        "hpa_images": len(hpa_records),
        "mendeley_png": len(manifests["mendeley"]),
        "S-BIAD2849_tiff": len(manifests["bia2849"]),
        "S-BIAD2306_zip": len(manifests["bia2306"]),
        "zenodo_tiff": len(manifests["zenodo"]),
    }


def fetch_sources(
    root: Path,
    *,
    workers: int = 4,
    force: bool = False,
    validate_hpa_xml: bool = True,
) -> dict[str, Any]:
    """Fetch every exact payload, including all 11,968 HPA panels."""
    specs, expected = download_specs(
        root,
        validate_hpa_xml=validate_hpa_xml,
    )
    receipts: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(download_verified, spec, force=force): spec
            for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                receipts.append(future.result())
            except Exception as exc:
                failures.append(f"{spec.key}: {exc}")
    if failures:
        raise ReviewSourceError(
            f"{len(failures)} download(s) failed; first failures: "
            + " | ".join(sorted(failures)[:5])
        )
    summary = {
        "schema": "blotgrid.public_western_fetch",
        "schema_version": 1,
        "expected": expected,
        "payloads": len(receipts),
        "downloaded": sum(
            receipt.get("status") == "downloaded" for receipt in receipts
        ),
        "already_verified": sum(
            receipt.get("status") == "already_verified"
            for receipt in receipts
        ),
        "receipts": sorted(receipts, key=lambda receipt: receipt["key"]),
    }
    _atomic_json(root / "fetch-manifest.json", summary)
    return summary


def plan_sources(root: Path) -> dict[str, Any]:
    """Write a no-download acquisition plan with exact expected selections."""
    manifests = validate_manifests(root)
    hpa_records, hpa_review = load_hpa_metadata(root, validate_xml=False)
    plan = {
        "schema": "blotgrid.public_western_plan",
        "schema_version": 1,
        "downloads": {
            "hpa_full_index_and_images": len(hpa_records),
            "hpa_first_review_subset": len(hpa_review),
            "mendeley_png": len(manifests["mendeley"]),
            "S-BIAD2849_raw_tiff": len(manifests["bia2849"]),
            "S-BIAD2306_wb_archives": len(manifests["bia2306"]),
            "zenodo_original_tiff": len(manifests["zenodo"]),
        },
        "source_dois": [source["doi"] for source in SOURCES],
        "licenses": LICENSES,
        "notes": [
            "fetch validates the HPA XML URL set and acquires all 11,968 panels",
            "build uses the stratified 256-panel HPA review subset by default",
            "no automatic proposal is approved training ground truth",
        ],
    }
    _atomic_json(root / "plan.json", plan)
    return plan


def _require_local_download(spec: DownloadSpec) -> dict[str, Any]:
    """Verify a build input without initiating network access."""
    if _existing_download_verified(spec):
        receipt = _write_download_receipt(spec)
        receipt["status"] = "already_verified"
        return receipt
    raise ReviewSourceError(
        f"Required payload is missing or unverified: {spec.destination}. "
        "Run the fetch command first."
    )


def _slug(value: str, *, maximum: int = 70) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    normalized = re.sub(r"_+", "_", normalized)
    return (normalized or "image")[:maximum]


def _study_group(source_id: str, source_image: str, hint: str = "") -> str:
    if source_id == "mendeley_zz3ktsyjds_v2":
        group = hint or PurePosixPath(source_image).parent.name
    elif source_id == "bia_s_biad2306":
        # Separate target/channel archives from this accession frequently
        # contain the same experiment. Without defensible donor-level metadata,
        # conservatively keep the whole DOI in one automatic-split component.
        group = "doi_10.6019_S-BIAD2306"
    elif source_id == "zenodo_5810203":
        match = re.match(r"(Figure_[A-Za-z0-9]+)", Path(source_image).stem)
        group = match.group(1) if match else Path(source_image).stem
    elif source_id == "hpa_v25_1":
        group = hint or Path(source_image).stem
    else:
        group = hint or PurePosixPath(source_image).parent.as_posix()
    return f"{source_id}:{_slug(group, maximum=120)}"


def _candidate(
    *,
    path: Path,
    source_id: str,
    upstream_url: str,
    upstream_path: str,
    source_image: str,
    study_group: str,
    source_archive: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = SOURCES_BY_ID[source_id]
    return {
        "path": path,
        "source_id": source_id,
        "license_id": int(source["license_id"]),
        "source_doi": str(source["doi"]),
        "source_url": str(source["url"]),
        "required_attribution": str(source["required_attribution"]),
        "source_archive": source_archive,
        "source_image": source_image,
        "upstream_url": upstream_url,
        "upstream_path": upstream_path,
        "study_group": study_group,
        **dict(extra or {}),
    }


def _materialize_mendeley(
    root: Path,
    manifests: Mapping[str, Any],
    *,
    force: bool,
) -> list[dict[str, Any]]:
    bulk = manifests["mendeley_bulk"]
    archive = root / "downloads" / "mendeley" / "zz3ktsyjds-v2.zip"
    _require_local_download(
        DownloadSpec(
            key="mendeley:zz3ktsyjds-v2.zip",
            url=str(bulk["url"]),
            destination=archive,
            size=int(bulk["size"]),
            algorithm="sha256",
            digest=str(bulk["sha256_hash"]),
        )
    )
    extracted = root / "raw" / "mendeley" / "zz3ktsyjds-v2"
    safe_extract_zip(
        archive,
        extracted,
        select=lambda name: Path(name).suffix.lower() == ".png",
        force=force,
    )
    extracted_files = [
        path for path in extracted.rglob("*") if path.is_file() and path.suffix.lower() == ".png"
    ]
    by_basename: dict[str, list[Path]] = {}
    for path in extracted_files:
        by_basename.setdefault(path.name, []).append(path)
    if len(extracted_files) != 186:
        raise ReviewSourceError(
            f"Mendeley archive yielded {len(extracted_files)} PNGs; expected 186."
        )

    candidates: list[dict[str, Any]] = []
    for record in manifests["mendeley"]:
        matches = by_basename.get(str(record["filename"]), [])
        if len(matches) != 1:
            raise ReviewSourceError(
                f"Mendeley manifest filename is not one-to-one in the archive: "
                f"{record['filename']!r}."
            )
        path = matches[0]
        expected_size = int(record.get("bytes", record["size"]))
        if (
            path.stat().st_size != expected_size
            or file_digest(path) != str(record["sha256"])
        ):
            raise ReviewSourceError(
                f"Mendeley extracted PNG failed its per-file pin: {path.name}."
            )
        member = path.relative_to(extracted).as_posix()
        candidates.append(
            _candidate(
                path=path,
                source_id="mendeley_zz3ktsyjds_v2",
                upstream_url=str(record["download_url"]),
                upstream_path=member,
                source_archive=archive.name,
                source_image=member,
                study_group=_study_group(
                    "mendeley_zz3ktsyjds_v2",
                    member,
                    str(record.get("folder_name", "")),
                ),
                extra={
                    "upstream_sha256": str(record["sha256"]),
                    "upstream_file_id": str(record["id"]),
                },
            )
        )
    return candidates


def _materialize_bia2849(
    root: Path,
    manifests: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in manifests["bia2849"]:
        basename = _safe_manifest_path(
            record["path"],
            label="S-BIAD2849 source path",
        ).name
        base = (
            root
            / "downloads"
            / "biostudies"
            / "S-BIAD2849"
            / "files"
        )
        path = _contained_destination(base, basename)
        _require_local_download(
            DownloadSpec(
                key=f"S-BIAD2849:{record['path']}",
                url=str(record["url"]),
                destination=path,
                size=int(record["size"]),
            )
        )
        upstream_path = str(record["path"])
        candidates.append(
            _candidate(
                path=path,
                source_id="bia_s_biad2849",
                upstream_url=str(record["url"]),
                upstream_path=upstream_path,
                source_image=upstream_path,
                study_group=_study_group(
                    "bia_s_biad2849",
                    upstream_path,
                    PurePosixPath(upstream_path).parent.as_posix(),
                ),
                extra={"upstream_checksum_status": "not_provided"},
            )
        )
    return candidates


def _materialize_bia2306(
    root: Path,
    manifests: Mapping[str, Any],
    *,
    force: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for record in manifests["bia2306"]:
        archive_name = _safe_manifest_path(
            record["path"],
            label="S-BIAD2306 source path",
        ).name
        base = (
            root
            / "downloads"
            / "biostudies"
            / "S-BIAD2306"
            / "archives"
        )
        archive = _contained_destination(base, archive_name)
        _require_local_download(
            DownloadSpec(
                key=f"S-BIAD2306:{record['path']}",
                url=str(record["url"]),
                destination=archive,
                size=int(record["size"]),
            )
        )
        archive_key = (
            _slug(Path(archive_name).stem, maximum=90)
            + "-"
            + hashlib.sha256(archive_name.encode("utf-8")).hexdigest()[:10]
        )
        extracted = (
            root / "raw" / "biostudies" / "S-BIAD2306" / archive_key
        )
        safe_extract_zip(
            archive,
            extracted,
            select=lambda name: Path(name).suffix.lower() in IMAGE_SUFFIXES,
            force=force,
        )
        image_paths = sorted(
            (
                path
                for path in extracted.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: path.relative_to(extracted).as_posix(),
        )
        full_tiffs = [
            path for path in image_paths if path.suffix.lower() in {".tif", ".tiff"}
        ]
        selected = full_tiffs or [
            path
            for path in image_paths
            if path.name.lower() not in {"th.jpg", "thumbnail.jpg"}
        ]
        if not selected:
            raise ReviewSourceError(
                f"S-BIAD2306 archive has no reviewable image member: {archive_name}."
            )
        for path in image_paths:
            if path in selected:
                continue
            exclusions.append(
                {
                    "source_id": "bia_s_biad2306",
                    "source_archive": archive_name,
                    "source_image": path.relative_to(extracted).as_posix(),
                    "reason": (
                        "thumbnail_superseded_by_full_resolution_tiff"
                        if full_tiffs
                        else "nonprimary_image_member"
                    ),
                }
            )
        for path in selected:
            member = path.relative_to(extracted).as_posix()
            upstream_path = f"{record['path']}#{member}"
            candidates.append(
                _candidate(
                    path=path,
                    source_id="bia_s_biad2306",
                    upstream_url=str(record["url"]),
                    upstream_path=upstream_path,
                    source_archive=archive_name,
                    source_image=member,
                    study_group=_study_group(
                        "bia_s_biad2306",
                        member,
                        archive_name,
                    ),
                    extra={"upstream_checksum_status": "not_provided"},
                )
            )
    return candidates, exclusions


def _materialize_zenodo(
    root: Path,
    manifests: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in manifests["zenodo"]:
        filename = _safe_basename(
            record["filename"],
            label="Zenodo filename",
        )
        base = root / "downloads" / "zenodo" / "5810203" / "files"
        path = _contained_destination(base, filename)
        algorithm, digest = str(record["checksum"]).split(":", 1)
        _require_local_download(
            DownloadSpec(
                key=f"zenodo:5810203:{filename}",
                url=str(record["download_url"]),
                destination=path,
                size=int(record["bytes"]),
                algorithm=algorithm,
                digest=digest,
            )
        )
        candidates.append(
            _candidate(
                path=path,
                source_id="zenodo_5810203",
                upstream_url=str(record["download_url"]),
                upstream_path=filename,
                source_image=filename,
                study_group=_study_group("zenodo_5810203", filename),
                extra={"upstream_checksum": str(record["checksum"])},
            )
        )
    return candidates


def _hpa_specific_attribution(record: Mapping[str, Any]) -> str:
    genes = ", ".join(str(value) for value in record.get("gene_names", []))
    antibody = str(record.get("antibody", {}).get("id", ""))
    entry_url = str(record.get("canonical_entry_url", ""))
    return (
        "Image credit: Human Protein Atlas. Western blot for "
        f"{genes or 'unspecified gene'}, antibody {antibody or 'unspecified'}, "
        f"HPA v25, {entry_url}; licensed under CC BY 4.0."
    )


def _materialize_hpa(
    root: Path,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        path = hpa_download_path(root, record)
        url = str(record["image"]["url"])
        _require_local_download(
            DownloadSpec(
                key=f"hpa:{record['image']['id']}",
                url=url,
                destination=path,
                max_bytes=MAX_HPA_IMAGE_BYTES,
            )
        )
        antibody_id = str(record.get("antibody", {}).get("id", ""))
        entry_url = str(record.get("canonical_entry_url", ""))
        validation = str(
            record.get("western_blot", {}).get("validation", "")
        )
        candidates.append(
            _candidate(
                path=path,
                source_id="hpa_v25_1",
                upstream_url=url,
                upstream_path=str(record["image"]["id"]),
                source_image=path.name,
                study_group=_study_group(
                    "hpa_v25_1",
                    path.name,
                    antibody_id or str(record["image"]["id"]),
                ),
                extra={
                    "canonical_entry_url": entry_url,
                    "image_credit": "Human Protein Atlas",
                    "specific_attribution": _hpa_specific_attribution(record),
                    "hpa_validation": validation,
                    "hpa_validation_description": str(
                        record.get("western_blot", {}).get(
                            "validation_description",
                            "",
                        )
                    ),
                    "hpa_technology": str(
                        record.get("western_blot", {}).get("technology", "")
                    ),
                    "hpa_lanes": record.get("western_blot", {}).get(
                        "lanes",
                        [],
                    ),
                    "hpa_gene_names": record.get("gene_names", []),
                    "hpa_ensembl_ids": record.get("ensembl_ids", []),
                    "hpa_antibody_id": antibody_id,
                    "processed_asset_notice": (
                        "HPA rendered medium assay panel; not a raw scan"
                    ),
                },
            )
        )
    return candidates


def _bit_depth_for_mode(mode: str) -> int:
    if mode in {"1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr"}:
        return 8
    if mode.startswith("I;16"):
        return 16
    return 32


def _pillow_metadata(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            frames = int(getattr(image, "n_frames", 1))
            metadata = {
                "width": int(image.width),
                "height": int(image.height),
                "frames": frames,
                "mode": str(image.mode),
                "format": str(image.format or ""),
                "bit_depth": _bit_depth_for_mode(str(image.mode)),
            }
            if frames != 1:
                raise ReviewSourceError(
                    f"unsupported_multi_frame_image:{frames}"
                )
            if image.width < 1 or image.height < 1:
                raise ReviewSourceError("invalid_image_dimensions")
            if (
                image.width > MAX_IMAGE_DIMENSION
                or image.height > MAX_IMAGE_DIMENSION
                or image.width * image.height > MAX_IMAGE_PIXELS
            ):
                raise ReviewSourceError(
                    "image_dimensions_exceed_resource_limit"
                )
            image.load()
        with Image.open(path) as image:
            image.verify()
        return metadata
    except ReviewSourceError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ReviewSourceError(f"unsupported_image:{exc}") from exc


def _validate_tiff_level(
    level: Any,
    *,
    label: str,
) -> tuple[int, int, int, np.dtype[Any]]:
    axes = str(level.axes)
    shape = tuple(int(value) for value in level.shape)
    if axes == "YX" and len(shape) == 2:
        height, width = shape
        samples = 1
    elif axes == "YXS" and len(shape) == 3 and shape[2] in {3, 4}:
        height, width, samples = shape
    else:
        raise ReviewSourceError(f"ambiguous_tiff_axes:{axes}:{shape}")
    try:
        dtype = np.dtype(level.dtype)
    except (TypeError, ValueError) as exc:
        raise ReviewSourceError(f"unsupported_tiff_dtype:{level.dtype}") from exc
    decoded_bytes = height * width * samples * int(dtype.itemsize)
    if (
        height < 1
        or width < 1
        or height > MAX_IMAGE_DIMENSION
        or width > MAX_IMAGE_DIMENSION
        or height * width > MAX_IMAGE_PIXELS
        or dtype.kind not in {"b", "u", "i", "f"}
        or dtype.itemsize < 1
        or dtype.itemsize > 8
        or decoded_bytes > MAX_TIFF_ARRAY_BYTES
    ):
        raise ReviewSourceError(
            f"tiff_resource_limit:{label}:{shape}:{dtype}"
        )
    return height, width, samples, dtype


def _collapse_grayscale_rgb(
    array: np.ndarray,
    *,
    photometric: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(array)
    if (
        values.ndim != 3
        or values.shape[2] not in {3, 4}
        or photometric != "RGB"
    ):
        raise ReviewSourceError(
            "ambiguous_tiff_color_encoding:"
            f"{values.shape}:{photometric or '<missing>'}"
        )
    if not (
        np.array_equal(values[..., 0], values[..., 1])
        and np.array_equal(values[..., 0], values[..., 2])
    ):
        raise ReviewSourceError("nonidentical_rgb_channels")
    if values.shape[2] == 4:
        dtype = np.dtype(values.dtype)
        opaque = (
            True
            if dtype.kind == "b"
            else np.all(values[..., 3] == np.iinfo(dtype).max)
            if dtype.kind in {"u", "i"}
            else np.all(values[..., 3] == 1.0)
        )
        if not bool(opaque):
            raise ReviewSourceError("nonopaque_tiff_alpha")
    return values[..., 0], {
        "method": (
            "identical_rgb_channels_to_grayscale_v1"
            if values.shape[2] == 3
            else "identical_rgb_opaque_alpha_to_grayscale_v1"
        ),
        "source_samples": int(values.shape[2]),
        "channels_bit_identical": True,
        "alpha_fully_opaque": values.shape[2] == 4,
    }


def _uint16_from_array(array: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(array)
    if values.ndim != 2:
        raise ReviewSourceError(
            f"ambiguous_tiff_axes: expected 2-D YX, got {values.shape}"
        )
    finite = np.isfinite(values)
    if not finite.any():
        raise ReviewSourceError("tiff_has_no_finite_pixels")
    finite_values = values[finite].astype(np.float64, copy=False)
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    working = np.where(finite, values, minimum).astype(np.float64, copy=False)
    if values.dtype == np.uint16:
        output = np.asarray(working, dtype=np.uint16)
        method = "identity_uint16_v1"
    elif values.dtype == np.uint8:
        output = np.asarray(working, dtype=np.uint16) * np.uint16(257)
        method = "uint8_full_range_to_uint16_v1"
    elif maximum > minimum:
        output = np.rint(
            np.clip((working - minimum) / (maximum - minimum), 0.0, 1.0)
            * 65535.0
        ).astype(np.uint16)
        method = "finite_minmax_linear_to_uint16_v1"
    else:
        output = np.zeros(values.shape, dtype=np.uint16)
        method = "constant_to_zero_uint16_v1"
    return output, {
        "method": method,
        "source_dtype": str(values.dtype),
        "source_shape": [int(value) for value in values.shape],
        "source_finite_min": minimum,
        "source_finite_max": maximum,
        "source_nonfinite_pixels": int(values.size - np.count_nonzero(finite)),
        "output_dtype": "uint16",
    }


def _convert_tiff_primary_series(
    source: Path,
    destination: Path,
) -> tuple[Path, dict[str, Any]]:
    """Convert one unambiguous highest-resolution TIFF series to 16-bit PNG."""
    try:
        import tifffile
    except ImportError as exc:
        raise ReviewSourceError(
            "tifffile is required to inspect scientific TIFF pyramids."
        ) from exc

    source_sha256 = file_digest(source)
    if source.stat().st_size > MAX_TIFF_ARRAY_BYTES:
        raise ReviewSourceError("tiff_file_exceeds_resource_limit")
    try:
        with tifffile.TiffFile(source) as bundle:
            if len(bundle.series) != 1:
                raise ReviewSourceError(
                    f"ambiguous_tiff_series:{len(bundle.series)}"
                )
            series = bundle.series[0]
            levels = list(getattr(series, "levels", ()) or (series,))
            if len(levels) > 16 or len(bundle.pages) > 256:
                raise ReviewSourceError("tiff_structure_exceeds_resource_limit")
            level_dimensions: list[tuple[int, int]] = []
            for level in levels:
                height, width, _samples, _dtype = _validate_tiff_level(
                    level,
                    label="pyramid_level",
                )
                level_dimensions.append((height, width))
            level_shapes = [
                [int(value) for value in level.shape] for level in levels
            ]
            if any(
                level_dimensions[index][0] > level_dimensions[index - 1][0]
                or level_dimensions[index][1]
                > level_dimensions[index - 1][1]
                for index in range(1, len(level_dimensions))
            ):
                raise ReviewSourceError("invalid_tiff_pyramid_order")
            array = levels[0].asarray()
            first_page = series.pages[0]
            photometric = str(
                getattr(getattr(first_page, "photometric", None), "name", "")
                or ""
            )
            color_collapse: dict[str, Any] | None = None
            if str(levels[0].axes) == "YXS":
                array, color_collapse = _collapse_grayscale_rgb(
                    array,
                    photometric=photometric,
                )
            software = str(
                getattr(bundle.pages[0], "software", "") or ""
            )
            page_count = len(bundle.pages)
    except ReviewSourceError:
        raise
    except Exception as exc:
        raise ReviewSourceError(f"unsupported_tiff:{exc}") from exc

    output, scale = _uint16_from_array(array)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".png",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(output).save(temporary, format="PNG")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    conversion = {
        **scale,
        "color_collapse": color_collapse,
        "source_sha256": source_sha256,
        "selected_series": 0,
        "selected_level": 0,
        "series_count": 1,
        "pyramid_levels": len(level_shapes),
        "level_shapes": level_shapes,
        "tiff_pages": page_count,
        "source_software": software or None,
        "conversion_software": {
            "name": "tifffile",
            "version": str(tifffile.__version__),
        },
        "derived_sha256": file_digest(destination),
        "derived_format": "PNG",
    }
    return destination, conversion


def _prepare_image(
    candidate: Mapping[str, Any],
    derived_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any] | None]:
    source = Path(candidate["path"])
    suffix = source.suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        return source, _pillow_metadata(source), None
    if source.stat().st_size > MAX_TIFF_ARRAY_BYTES:
        raise ReviewSourceError("tiff_file_exceeds_resource_limit")

    # Scientific TIFFs require structural inspection.  Single-series pyramids
    # are not independent frames: level 0 is the full-resolution raw channel,
    # and lower levels are FILETYPE.REDUCEDIMAGE display levels.
    try:
        import tifffile
    except ImportError as exc:
        raise ReviewSourceError(
            "tifffile is required to validate TIFF review sources."
        ) from exc
    try:
        with tifffile.TiffFile(source) as bundle:
            if len(bundle.series) != 1:
                raise ReviewSourceError(
                    f"ambiguous_tiff_series:{len(bundle.series)}"
                )
            series = bundle.series[0]
            levels = list(getattr(series, "levels", ()) or (series,))
            if len(levels) > 16 or len(bundle.pages) > 256:
                raise ReviewSourceError("tiff_structure_exceeds_resource_limit")
            for level in levels:
                _validate_tiff_level(level, label="pyramid_level")
            if len(levels) == 1 and str(series.axes) == "YXS":
                metadata = _pillow_metadata(source)
                first_page = series.pages[0]
                photometric = str(
                    getattr(
                        getattr(first_page, "photometric", None),
                        "name",
                        "",
                    )
                    or ""
                )
                try:
                    with Image.open(source) as pillow_image:
                        pillow_array = np.asarray(pillow_image)
                    _collapse_grayscale_rgb(
                        pillow_array,
                        photometric=photometric,
                    )
                except ReviewSourceError as exc:
                    if "nonidentical_rgb_channels" in str(exc):
                        # Preserve genuine color scans byte-for-byte. BlotGrid's
                        # existing image loader records RGB metadata and uses
                        # its fixed luminance conversion only for detection and
                        # measurement; the review copy remains the native TIFF.
                        return source, metadata, None
                    raise
            needs_conversion = (
                len(levels) > 1
                or str(series.axes) != "YX"
                or len(series.shape) != 2
            )
    except ReviewSourceError:
        raise
    except Exception as exc:
        # Pillow may still support simple legacy TIFFs that tifffile declines.
        try:
            return source, _pillow_metadata(source), None
        except ReviewSourceError:
            raise ReviewSourceError(f"unsupported_tiff:{exc}") from exc

    if not needs_conversion:
        try:
            return source, _pillow_metadata(source), None
        except ReviewSourceError as exc:
            if "unsupported_multi_frame_image" in str(exc):
                raise
            # A one-series YX scientific TIFF can be made explicitly loadable
            # without guessing a page or channel.
            needs_conversion = True
    if needs_conversion:
        source_sha = file_digest(source)
        name = (
            f"{_slug(source.stem, maximum=80)}"
            f"__{source_sha[:16]}__fullres.png"
        )
        destination = (
            derived_root
            / str(candidate["source_id"])
            / name
        )
        receipt_path = destination.with_name(destination.name + ".conversion.json")
        if destination.is_file() and receipt_path.is_file():
            receipt = _load_json(receipt_path)
            if (
                receipt.get("source_sha256") == source_sha
                and receipt.get("derived_sha256") == file_digest(destination)
            ):
                return destination, _pillow_metadata(destination), receipt
        destination, conversion = _convert_tiff_primary_series(
            source,
            destination,
        )
        _atomic_json(receipt_path, conversion)
        return destination, _pillow_metadata(destination), conversion
    raise AssertionError("unreachable TIFF preparation state")


def _link_or_copy_verified(source: Path, destination: Path, sha256: str) -> None:
    """Create an inode-independent verified copy.

    Review images are intentionally mutable working files. Hard links would
    let a reviewer overwrite canonical or downloaded source bytes in place.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if (
            not destination.is_symlink()
            and file_digest(destination) == sha256
            and source.stat().st_ino != destination.stat().st_ino
        ):
            return
        raise ReviewSourceError(
            f"Canonical filename collision or corruption: {destination}."
        )
    if destination.exists():
        raise ReviewSourceError(
            f"Canonical destination is not a regular file: {destination}."
        )
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with source.open("rb") as input_handle, temporary.open("xb") as output:
            shutil.copyfileobj(input_handle, output, CHUNK_BYTES)
            output.flush()
            os.fsync(output.fileno())
        if file_digest(temporary) != sha256:
            raise ReviewSourceError(
                f"Canonical copy digest changed for {source}."
            )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _assign_leakage_components(images: list[dict[str, Any]]) -> None:
    """Connect study groups that share exact rendered content.

    Training already unions ``study_group`` and ``original_sha256``. Encoding
    the same connection here keeps separately exported review batches safe too
    and makes exact-content aliases explicit in the corpus manifest.
    """
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

    groups_by_sha: dict[str, set[str]] = {}
    counts_by_sha: dict[str, int] = {}
    for image in images:
        group = str(image["study_group"])
        digest = str(image["sha256"])
        find(group)
        groups_by_sha.setdefault(digest, set()).add(group)
        counts_by_sha[digest] = counts_by_sha.get(digest, 0) + 1
    for groups in groups_by_sha.values():
        ordered = sorted(groups)
        for group in ordered[1:]:
            union(ordered[0], group)

    components: dict[str, set[str]] = {}
    for group in parent:
        components.setdefault(find(group), set()).add(group)
    safe_group_by_original: dict[str, str] = {}
    for groups in components.values():
        if len(groups) == 1:
            only = next(iter(groups))
            safe_group_by_original[only] = only
            continue
        digest = hashlib.sha256(
            "\0".join(sorted(groups)).encode("utf-8")
        ).hexdigest()[:24]
        linked = f"blotgrid_exact_content_component:{digest}"
        for group in groups:
            safe_group_by_original[group] = linked

    for image in images:
        original_group = str(image["study_group"])
        safe_group = safe_group_by_original[original_group]
        if safe_group != original_group:
            image["upstream_study_group"] = original_group
            image["study_group"] = safe_group
        digest = str(image["sha256"])
        image["exact_content_sha256"] = digest
        image["exact_content_count"] = counts_by_sha[digest]


def canonicalize_candidates(
    root: Path,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate candidates and create collision-free canonical review images."""
    canonical_root = root / "canonical" / "images"
    derived_root = root / "derived"
    images: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    seen_filenames: set[str] = set()
    for candidate in candidates:
        source_path = Path(candidate["path"])
        try:
            prepared, image_metadata, conversion = _prepare_image(
                candidate,
                derived_root,
            )
        except ReviewSourceError as exc:
            quarantined.append(
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "path"
                }
                | {
                    "source_path": str(source_path),
                    "source_sha256": (
                        file_digest(source_path)
                        if source_path.is_file() and not source_path.is_symlink()
                        else None
                    ),
                    "reason": str(exc),
                }
            )
            continue

        sha256 = file_digest(prepared)
        path_key = hashlib.sha256(
            (
                str(candidate["source_id"])
                + "\0"
                + str(candidate["upstream_path"])
            ).encode("utf-8")
        ).hexdigest()[:10]
        suffix = prepared.suffix.lower()
        filename = (
            f"{_slug(str(candidate['source_id']), maximum=35)}"
            f"__{_slug(Path(str(candidate['source_image'])).stem, maximum=70)}"
            f"__{path_key}__{sha256[:12]}{suffix}"
        )
        if filename in seen_filenames:
            raise ReviewSourceError(
                f"Canonical naming collision for {filename}."
            )
        seen_filenames.add(filename)
        destination = canonical_root / filename
        _link_or_copy_verified(prepared, destination, sha256)
        record = {
            key: value for key, value in candidate.items() if key != "path"
        }
        record.update(
            {
                "project_filename": filename,
                "canonical_path": destination.relative_to(root).as_posix(),
                "sha256": sha256,
                "bytes": destination.stat().st_size,
                **image_metadata,
                "conversion": conversion,
                "original_source_sha256": file_digest(source_path),
            }
        )
        images.append(record)
    _assign_leakage_components(images)
    return images, quarantined


def _empty_layout(image: Mapping[str, Any], warning: str) -> dict[str, Any]:
    width = int(image["width"])
    height = int(image["height"])
    return {
        "mode": "review_import",
        "annotation_complete": False,
        "negative_image": False,
        "detection_backend": "none",
        "detector_provenance": {},
        "rois": [],
        "roi_size": {
            "w": max(6, int(width * 0.12)),
            "h": max(4, int(height * 0.08)),
        },
        "lane_x": [],
        "band_width": max(12, int(width * 0.12)),
        "targets": {},
        "bands_dark": True,
        "polarity_confidence": 0.0,
        "detection_warnings": [warning],
        "detected_count": 0,
        "directly_detected_count": 0,
        "fallback_count": 0,
        "next_roi_id": 0,
    }


def _gelgenie_layout(
    root: Path,
    image: Mapping[str, Any],
    model_path: Path,
) -> dict[str, Any]:
    """Create unapproved GelGenie proposals without grid completion."""
    from wblot import gelgenie, image_io

    path = root / str(image["canonical_path"])
    loaded = image_io.load_image(path, name=str(image["project_filename"]))
    proposal_gray = loaded.gray
    proposal_width = loaded.width
    proposal_height = loaded.height
    proposal_resized = False
    if loaded.width * loaded.height > MAX_REVIEW_PROPOSAL_PIXELS:
        scale = math.sqrt(
            MAX_REVIEW_PROPOSAL_PIXELS / (loaded.width * loaded.height)
        )
        proposal_width = max(8, int(math.floor(loaded.width * scale)))
        proposal_height = max(8, int(math.floor(loaded.height * scale)))
        proposal_gray = np.asarray(
            Image.fromarray(
                np.asarray(loaded.gray, dtype=np.float32),
                mode="F",
            ).resize(
                (proposal_width, proposal_height),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.float64,
        )
        proposal_resized = True
    try:
        result = gelgenie.predict_bands(
            proposal_gray,
            ["Band"],
            ["Unassigned"],
            model_path,
            max_value=loaded.max_value,
            probability_threshold=0.50,
            review_threshold=0.75,
            device="cpu",
            normalization="auto",
            polarity="auto",
        )
    except Exception as exc:
        return _empty_layout(
            image,
            f"GelGenie proposal generation failed: {exc}",
        )
    if proposal_resized:
        scale_x = loaded.width / proposal_width
        scale_y = loaded.height / proposal_height
        for band in result.bands:
            x0 = int(round(band.x * scale_x))
            y0 = int(round(band.y * scale_y))
            x1 = int(round((band.x + band.w) * scale_x))
            y1 = int(round((band.y + band.h) * scale_y))
            band.x = min(max(0, x0), loaded.width - 1)
            band.y = min(max(0, y0), loaded.height - 1)
            band.w = max(1, min(loaded.width, x1) - band.x)
            band.h = max(1, min(loaded.height, y1) - band.y)
            band.profile_span = float(band.h)
        result.lane_centers = [
            min(loaded.width - 1, max(0, int(round(value * scale_x))))
            for value in result.lane_centers
        ]
        result.target_centers = [
            min(loaded.height - 1, max(0, int(round(value * scale_y))))
            for value in result.target_centers
        ]
        result.warnings.append(
            "Proposal inference used a bilinear downscaled working copy "
            f"({proposal_width}x{proposal_height} from "
            f"{loaded.width}x{loaded.height}); boxes were mapped back to "
            "the untouched native-resolution review image."
        )
    rows = [
        {
            **band.as_row(),
            "roi_id": f"roi_{index}",
            "annotation_approved": False,
            # Explicitly identify the initial machine suggestion.  The normal
            # exporter derives pseudo-label status from approval and excludes
            # every one of these by default.
            "pseudo_label": True,
            "review_required": True,
        }
        for index, band in enumerate(result.bands)
    ]
    widths = [int(row["w"]) for row in rows]
    heights = [int(row["h"]) for row in rows]
    roi_size = {
        "w": (
            int(math.ceil(float(np.quantile(widths, 0.90))))
            if widths
            else max(6, int(loaded.width * 0.12))
        ),
        "h": (
            int(math.ceil(float(np.quantile(heights, 0.90))))
            if heights
            else max(4, int(loaded.height * 0.08))
        ),
    }
    return {
        "mode": "automatic",
        "annotation_complete": False,
        "negative_image": False,
        "detection_backend": "gelgenie_hybrid",
        "detector_provenance": {
            **gelgenie.model_provenance(model_path),
            "backend": "gelgenie_hybrid",
            "probability_threshold": 0.50,
            "review_threshold": 0.75,
            "requested_device": "cpu",
            "execution_provider": gelgenie.runtime_provider("cpu"),
            "available_providers": gelgenie.available_providers(),
            "normalization": "auto",
            "input_inverted": bool(result.bands_are_dark),
            "proposal_input": {
                "native_width": loaded.width,
                "native_height": loaded.height,
                "model_width": proposal_width,
                "model_height": proposal_height,
                "resized": proposal_resized,
                "resampling": (
                    "bilinear"
                    if proposal_resized
                    else "none"
                ),
            },
        },
        "rois": rows,
        "roi_size": roi_size,
        "lane_x": result.lane_centers,
        "band_width": (
            int(round(float(np.median(widths))))
            if widths
            else max(12, int(loaded.width * 0.12))
        ),
        "targets": {
            "Band": {
                "y": (
                    int(result.target_centers[0])
                    if result.target_centers
                    else loaded.height // 2
                ),
                "h": roi_size["h"],
            }
        },
        "bands_dark": bool(result.bands_are_dark),
        "polarity_confidence": float(result.polarity_confidence),
        "detection_warnings": list(result.warnings)
        + [
            "Every imported proposal is unapproved and requires human review.",
            "No grid-completion fallback was run.",
        ],
        "detected_count": len(rows),
        "directly_detected_count": len(rows),
        "fallback_count": 0,
        "next_roi_id": len(rows),
    }


def _lineage_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact public-safe source record accepted by release gates."""
    return {
        "id": source["id"],
        "doi": source["doi"],
        "license_id": source["license_id"],
        "required_attribution": source["required_attribution"],
        "publication_assessment": source["publication_assessment"],
    }


def _lineage_image(image: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "source_id",
        "license_id",
        "source_doi",
        "source_url",
        "required_attribution",
        "source_archive",
        "source_image",
        "upstream_url",
        "upstream_path",
        "upstream_sha256",
        "upstream_checksum",
        "upstream_checksum_status",
        "upstream_file_id",
        "canonical_entry_url",
        "image_credit",
        "specific_attribution",
        "hpa_validation",
        "hpa_validation_description",
        "hpa_technology",
        "hpa_lanes",
        "hpa_gene_names",
        "hpa_ensembl_ids",
        "hpa_antibody_id",
        "processed_asset_notice",
        "conversion",
        "original_source_sha256",
    )
    return {field: image.get(field) for field in fields if image.get(field) is not None}


def build_project(
    root: Path,
    batch: Sequence[Mapping[str, Any]],
    *,
    batch_id: str,
    gelgenie_model: Path | None = None,
) -> dict[str, Any]:
    """Build one fail-closed BlotGrid review project."""
    layouts: dict[str, Any] = {}
    for image in batch:
        if gelgenie_model is None:
            layout = _empty_layout(
                image,
                "No automatic proposals requested; add every true band manually.",
            )
        else:
            layout = _gelgenie_layout(root, image, gelgenie_model)
        # Belt-and-suspenders: generated projects can never arrive approved.
        layout["annotation_complete"] = False
        layout["negative_image"] = False
        layout["fallback_count"] = 0
        for roi in layout.get("rois", []):
            roi["annotation_approved"] = False
            roi["review_required"] = True
            roi["pseudo_label"] = True
        layouts[str(image["project_filename"])] = layout

    source_ids = sorted({str(image["source_id"]) for image in batch})
    sources = [_lineage_source(SOURCES_BY_ID[source_id]) for source_id in source_ids]
    license_ids = {int(source["license_id"]) for source in sources}
    licenses = [
        dict(record) for record in LICENSES if int(record["id"]) in license_ids
    ]
    source_dois = [str(SOURCES_BY_ID[source_id]["doi"]) for source_id in source_ids]
    image_names = [str(image["project_filename"]) for image in batch]
    project = {
        "targets": ["Band"],
        "control": "",
        "lanes": ["Unassigned"],
        "bands_dark": True,
        "polarity_mode": "auto",
        "detection_sensitivity": 0.65,
        "detection_backend": (
            "gelgenie_hybrid" if gelgenie_model is not None else "classical"
        ),
        "ml_checkpoint": "models/blotgrid_detector.pt",
        "ml_device": "cpu",
        "ml_score_threshold": 0.50,
        "ml_review_threshold": 0.75,
        "gelgenie_model": (
            str(gelgenie_model)
            if gelgenie_model is not None
            else "models/gelgenie_universal.onnx"
        ),
        "gelgenie_probability_threshold": 0.50,
        "gelgenie_review_threshold": 0.75,
        "gelgenie_normalization": "auto",
        "bg_method": "local",
        "bg_margin": 6,
        "rb_radius": 50,
        "bg_fixed": 0.0,
        "multiband": "sum",
        "measurement_mode": "profile",
        "layouts": layouts,
        "replicate": {
            name: index for index, name in enumerate(image_names, start=1)
        },
        "study_group": {
            str(image["project_filename"]): str(image["study_group"])
            for image in batch
        },
        "annotation_complete": {name: False for name in image_names},
        "alignment_applied": {name: 0.0 for name in image_names},
        "image_provenance": {
            str(image["project_filename"]): {
                "sha256": str(image["sha256"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "bit_depth": int(image["bit_depth"]),
            }
            for image in batch
        },
        "source_lineage": {
            "schema_version": 1,
            "info": {
                "title": f"BlotGrid public Western review {batch_id}",
                "version": "1.0",
                "source_dois": source_dois,
            },
            "licenses": licenses,
            "sources": sources,
            "images": {
                str(image["project_filename"]): _lineage_image(image)
                for image in batch
            },
        },
        "evaluation_task": {},
    }
    return project


def _manual_labeling_text(
    *,
    review_images: int,
    batches: int,
    proposals: bool,
) -> str:
    proposal_sentence = (
        "GelGenie proposals are present, but every proposal is unapproved and "
        "requires review."
        if proposals
        else "No proposals were requested; add all true-band boxes manually."
    )
    return f"""# BlotGrid public Western manual review

This queue contains {review_images} images in {batches} batches. {proposal_sentence}
Nothing in these projects is training ground truth until you complete the steps below.

## Balanced pilot order

For the first cross-source pilot, review these complete batches before working
straight through the queue: `batch_0001` (S-BIAD2306), `batch_0005`
(S-BIAD2849), `batch_0007` (HPA), `batch_0020` (Mendeley), and `batch_0029`
(mostly Zenodo). This yields an inspectable multi-source tranche, but it is not
a release-grade validation set and does not waive review of the remaining
batches. Never split the S-BIAD2306 DOI group across train and validation.

## Work one batch at a time

1. In BlotGrid, load `review/batches/<batch>/project.initial.json` and click
   **Restore project**.
2. Upload every file from that batch's `images/` directory. Do not rename files:
   BlotGrid verifies the saved SHA-256 before applying any proposal.
3. Open the ROI tab and inspect the entire image at useful zoom. The neutral
   target/lane labels (`Band` / `Unassigned`) are intentional.
4. Move/resize proposals to tightly cover one real protein band. Add every
   missed true band. Disable every false proposal.
5. Exclude molecular-weight marker/ladder lanes, labels and other text,
   image borders, scan edges, dust, scratches, saturated blocks, lane-wide
   smear and background gradients. HPA panels commonly have a marker lane on
   the left; the saved HPA lane metadata is a warning, not an automatic label.
6. Put separate boxes around resolved doublets or triplets. Do not merge them
   into one broad box. Include faint bands when they are genuinely band-shaped;
   weak signal is not a reason to omit a true band.
7. Check that every active ROI is a true band. Enter/retain the pre-filled
   study group. Tick the whole-image attestation, then use **Approve all active
   AI labels**. For a genuinely band-free image, disable all boxes and attest
   the complete negative image.
8. Save the reviewed project and place it at
   `review/completed/<batch>.reviewed.json`. Do not overwrite
   `project.initial.json`.

## Release gate

- An ROI is usable only when `annotation_approved` is literally `true`.
- A positive or negative image is usable only when its whole-image
  `annotation_complete` attestation is literally `true`.
- Unapproved proposals are pseudo-labels. Never train or report validation
  metrics from them as human ground truth.
- Do not approve a proposal merely because the model drew it.
- HPA assets are rendered medium assay panels, not raw scans. Keep that stratum
  separate in interpretation and preserve the per-image HPA credit and URL.

## Sources and licences

| Source | DOI | Licence |
|---|---|---|
| Human Protein Atlas v25.1 | `{HPA_PRIMARY_DOI}` | CC BY 4.0 |
| Mendeley Data `zz3ktsyjds`, V2 | `10.17632/zz3ktsyjds.2` | CC BY 4.0 |
| BioImage Archive `S-BIAD2849` | `10.6019/S-BIAD2849` | CC0 1.0 |
| BioImage Archive `S-BIAD2306` | `10.6019/S-BIAD2306` | CC0 1.0 |
| Zenodo `5810203` | `10.5281/zenodo.5810203` | CC BY 4.0 |

The project JSON contains exact per-image lineage, upstream paths/URLs, hashes,
conversion records where needed, and required attribution. Preserve it when
saving reviewed projects.
"""


def _write_review_queue(
    root: Path,
    queue: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "batch",
        "project",
        "image",
        "source_id",
        "source_doi",
        "study_group",
        "hpa_validation",
        "proposal_count",
        "annotation_complete",
        "status",
        "review_note",
    ]
    lines: list[str] = []
    buffer = tempfile.SpooledTemporaryFile(mode="w+", newline="", encoding="utf-8")
    try:
        writer = csv.DictWriter(
            buffer,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(queue)
        buffer.seek(0)
        lines.append(buffer.read())
    finally:
        buffer.close()
    _atomic_text(root / "REVIEW_QUEUE.tsv", "".join(lines))


def _build_review_corpus_inplace(
    root: Path,
    *,
    batch_size: int = 20,
    gelgenie_model: Path | None = None,
    include_all_hpa: bool = False,
    force_extract: bool = False,
) -> dict[str, Any]:
    """Build canonical images, fail-closed projects, and a manual review queue."""
    root = root.expanduser().resolve()
    if batch_size < 1 or batch_size > 200:
        raise ReviewSourceError("batch_size must be between 1 and 200.")
    for generated_dir in (
        "raw",
        "derived",
        "canonical",
        "review",
        "review/completed",
    ):
        (root / generated_dir).mkdir(parents=True, exist_ok=True)
    manifests = validate_manifests(root)
    hpa_all, hpa_review = load_hpa_metadata(root, validate_xml=True)
    hpa_records = hpa_all if include_all_hpa else hpa_review

    candidates: list[dict[str, Any]] = []
    candidates.extend(
        _materialize_mendeley(root, manifests, force=force_extract)
    )
    candidates.extend(_materialize_bia2849(root, manifests))
    bia2306, exclusions = _materialize_bia2306(
        root,
        manifests,
        force=force_extract,
    )
    candidates.extend(bia2306)
    candidates.extend(_materialize_zenodo(root, manifests))
    candidates.extend(_materialize_hpa(root, hpa_records))
    images, quarantined = canonicalize_candidates(root, candidates)
    exact_content_sets: dict[str, list[str]] = {}
    for image in images:
        exact_content_sets.setdefault(str(image["sha256"]), []).append(
            str(image["project_filename"])
        )
    duplicate_content_sets = {
        digest: sorted(names)
        for digest, names in exact_content_sets.items()
        if len(names) > 1
    }

    review_hpa_urls = {
        str(record["image"]["url"]) for record in hpa_review
    }
    review_images = [
        image
        for image in images
        if image["source_id"] != "hpa_v25_1"
        or image["upstream_url"] in review_hpa_urls
    ]
    review_images.sort(
        key=lambda image: (
            str(image["source_id"]),
            str(image["project_filename"]),
        )
    )
    if gelgenie_model is not None:
        gelgenie_model = gelgenie_model.expanduser().resolve()
        if not gelgenie_model.is_file():
            raise ReviewSourceError(
                f"GelGenie model is missing: {gelgenie_model}."
            )

    batches_root = root / "review" / "batches"
    queue: list[dict[str, Any]] = []
    batch_count = math.ceil(len(review_images) / batch_size)
    for batch_index in range(batch_count):
        batch = review_images[
            batch_index * batch_size : (batch_index + 1) * batch_size
        ]
        batch_id = f"batch_{batch_index + 1:04d}"
        batch_dir = batches_root / batch_id
        image_dir = batch_dir / "images"
        project = build_project(
            root,
            batch,
            batch_id=batch_id,
            gelgenie_model=gelgenie_model,
        )
        for image in batch:
            source = root / str(image["canonical_path"])
            destination = image_dir / str(image["project_filename"])
            _link_or_copy_verified(
                source,
                destination,
                str(image["sha256"]),
            )
            layout = project["layouts"][str(image["project_filename"])]
            note = (
                "Exclude HPA marker lane and panel labels"
                if image["source_id"] == "hpa_v25_1"
                else ""
            )
            queue.append(
                {
                    "batch": batch_id,
                    "project": (
                        batch_dir / "project.initial.json"
                    ).relative_to(root).as_posix(),
                    "image": destination.relative_to(root).as_posix(),
                    "source_id": image["source_id"],
                    "source_doi": image["source_doi"],
                    "study_group": image["study_group"],
                    "hpa_validation": image.get("hpa_validation", ""),
                    "proposal_count": len(layout.get("rois", [])),
                    "annotation_complete": "false",
                    "status": "pending_human_review",
                    "review_note": note,
                }
            )
        _atomic_json(batch_dir / "project.initial.json", project)

    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "licenses": LICENSES,
        "sources": SOURCES,
        "hpa": {
            "full_index_records": len(hpa_all),
            "first_review_records": len(hpa_review),
            "canonicalized_records": len(hpa_records),
            "full_acquisition_command": "prepare_review_sources.py fetch",
            "asset_stratum": "rendered_medium_assay_panel_not_raw_scan",
        },
        "counts": {
            "input_candidates": len(candidates),
            "canonical_images": len(images),
            "review_images": len(review_images),
            "quarantined_images": len(quarantined),
            "excluded_nonprimary_members": len(exclusions),
            "review_batches": batch_count,
            "exact_duplicate_content_sets": len(duplicate_content_sets),
        },
        "images": images,
        "exact_duplicate_content_sets": duplicate_content_sets,
        "quarantined": quarantined,
        "excluded_nonprimary_members": exclusions,
        "annotation_policy": {
            "targets": ["Band"],
            "lanes": ["Unassigned"],
            "all_annotation_complete": False,
            "all_proposals_annotation_approved": False,
            "all_proposals_review_required": True,
            "grid_completion_used": False,
            "pseudo_labels_are_training_ground_truth": False,
            "automatic_split_uses_study_group_and_exact_content_sha256": True,
        },
    }
    _atomic_json(root / "dataset_manifest.json", manifest)
    _write_review_queue(root, queue)
    _atomic_text(
        root / "MANUAL_LABELING.md",
        _manual_labeling_text(
            review_images=len(review_images),
            batches=batch_count,
            proposals=gelgenie_model is not None,
        ),
    )
    return manifest


_GENERATED_CORPUS_PATHS = (
    "raw",
    "derived",
    "canonical",
    "review",
    "dataset_manifest.json",
    "REVIEW_QUEUE.tsv",
    "MANUAL_LABELING.md",
)


def _validate_staged_corpus(
    staging: Path,
    manifest: Mapping[str, Any],
) -> None:
    counts = manifest.get("counts", {})
    review_images = int(counts.get("review_images", -1))
    review_batches = int(counts.get("review_batches", -1))
    if review_images < 1 or review_batches < 1:
        raise ReviewSourceError("Staged review corpus is unexpectedly empty.")
    for relative in _GENERATED_CORPUS_PATHS:
        path = staging / relative
        if path.is_symlink() or not path.exists():
            raise ReviewSourceError(
                f"Staged corpus artifact is missing or unsafe: {relative}."
            )
    batch_root = staging / "review" / "batches"
    actual_batches = sorted(
        path.name
        for path in batch_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    )
    expected_batches = [
        f"batch_{index + 1:04d}" for index in range(review_batches)
    ]
    if actual_batches != expected_batches:
        raise ReviewSourceError("Staged review batch set is not exact.")
    project_count = len(list(batch_root.glob("*/project.initial.json")))
    image_count = len(list(batch_root.glob("*/images/*")))
    if project_count != review_batches or image_count != review_images:
        raise ReviewSourceError(
            "Staged review projects do not match declared corpus counts."
        )
    queue_rows = (
        staging / "REVIEW_QUEUE.tsv"
    ).read_text(encoding="utf-8").splitlines()
    if len(queue_rows) != review_images + 1:
        raise ReviewSourceError("Staged review queue row count changed.")


def _commit_staged_corpus(root: Path, staging: Path) -> None:
    """Replace all generated corpus artifacts with rollback on any failure."""
    backup = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.previous-", dir=root.parent)
    )
    backed_up: list[str] = []
    installed: list[str] = []
    committed = False
    try:
        for relative in _GENERATED_CORPUS_PATHS:
            source = staging / relative
            destination = root / relative
            previous = backup / relative
            if destination.is_symlink():
                raise ReviewSourceError(
                    f"Refusing to replace generated symlink: {destination}."
                )
            if destination.exists():
                os.replace(destination, previous)
                backed_up.append(relative)
            os.replace(source, destination)
            installed.append(relative)
        committed = True
    except Exception:
        for relative in reversed(installed):
            destination = root / relative
            failed = staging / relative
            if destination.exists() and not destination.is_symlink():
                os.replace(destination, failed)
        for relative in reversed(backed_up):
            previous = backup / relative
            destination = root / relative
            if previous.exists():
                os.replace(previous, destination)
        raise
    finally:
        if backup.exists():
            if committed:
                shutil.rmtree(backup)
            else:
                try:
                    backup.rmdir()
                except OSError:
                    # A non-empty backup only remains if rollback itself failed.
                    pass


def build_review_corpus(
    root: Path,
    *,
    batch_size: int = 20,
    gelgenie_model: Path | None = None,
    include_all_hpa: bool = False,
    force_extract: bool = False,
) -> dict[str, Any]:
    """Build and atomically install a complete, stale-free review corpus."""
    root = root.expanduser().resolve()
    if not (root / "downloads").is_dir():
        raise ReviewSourceError(
            f"Download root is missing: {root / 'downloads'}."
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.build-", dir=root.parent)
    )
    try:
        os.symlink(root / "downloads", staging / "downloads")
        manifest = _build_review_corpus_inplace(
            staging,
            batch_size=batch_size,
            gelgenie_model=gelgenie_model,
            include_all_hpa=include_all_hpa,
            force_extract=force_extract,
        )
        _validate_staged_corpus(staging, manifest)
        _commit_staged_corpus(root, staging)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def prepare_all(
    root: Path,
    *,
    workers: int = 4,
    batch_size: int = 20,
    gelgenie_model: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Fetch the complete source scope, then build the first review corpus."""
    fetch = fetch_sources(root, workers=workers, force=force)
    build = build_review_corpus(
        root,
        batch_size=batch_size,
        gelgenie_model=gelgenie_model,
        force_extract=force,
    )
    return {"fetch": fetch, "build": build}
