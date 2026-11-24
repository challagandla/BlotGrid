"""Self-contained optional training and annotation-export support.

The module contains no top-level PyTorch imports.  COCO export and metric
helpers therefore remain usable in the base BlotGrid environment.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

from .ml_detector import (
    CHECKPOINT_FORMAT_VERSION,
    DEFAULT_ARCHITECTURE,
    MAX_CHECKPOINT_METADATA_BYTES,
    MAX_TRAINABLE_BACKBONE_LAYERS,
    MLDetectorError,
    SUPPORTED_ARCHITECTURES,
    TRANSFER_TRAINABLE_BACKBONE_LAYERS,
    _load_checkpoint,
    _loads_unique_json,
    _metadata_sidecar,
    _safe_metadata,
    _sha256_file,
    build_model,
    build_model_with_provenance,
    normalize_grayscale,
    validate_initialization_provenance,
)


class MLTrainingError(MLDetectorError):
    """Invalid dataset, split, training option, or resume checkpoint."""


DATASET_LICENSE_LINEAGE_SCHEMA_VERSION = 1
MAX_DATASET_LICENSE_LINEAGE_BYTES = 512 * 1024
PUBLICATION_METADATA_SCHEMA_VERSION = 2
MAX_CHECKPOINT_HISTORY = 500

_PUBLICATION_LICENSE_POLICIES: dict[str, dict[str, Any]] = {
    "CC0-1.0": {
        "names": {
            "Creative Commons Zero v1.0 Universal",
        },
        "url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "attribution_required": False,
        "share_alike_required": False,
        "obligations": [],
    },
    "CC-BY-4.0": {
        "names": {
            "Creative Commons Attribution 4.0",
            "Creative Commons Attribution 4.0 International",
        },
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution_required": True,
        "share_alike_required": False,
        "obligations": ["attribution"],
    },
    "CC-BY-SA-2.1-JP": {
        "names": {
            "Creative Commons Attribution-ShareAlike 2.1 Japan",
        },
        "url": "https://creativecommons.org/licenses/by-sa/2.1/jp/",
        "attribution_required": True,
        "share_alike_required": True,
        "obligations": ["attribution", "share_alike"],
    },
}
_LICENSE_PUBLICATION_ASSESSMENT_FIELDS = {
    "status",
    "redistribution_cleared",
    "commercial_use_cleared",
    "attribution_required",
    "share_alike_required",
    "obligations",
}
_SOURCE_PUBLICATION_ASSESSMENT_FIELDS = {
    "status",
    "redistribution_cleared",
    "attribution_resolved",
    "license_id",
}


def _review_required_license_assessment() -> dict[str, Any]:
    return {
        "status": "review_required",
        "redistribution_cleared": False,
        "commercial_use_cleared": False,
        "attribution_required": False,
        "share_alike_required": False,
        "obligations": [],
    }


def _review_required_source_assessment(
    license_id: int | None,
) -> dict[str, Any]:
    return {
        "status": "review_required",
        "redistribution_cleared": False,
        "attribution_resolved": False,
        "license_id": license_id,
    }


def _read_json_object(source: str | os.PathLike[str] | Mapping[str, Any]) -> dict:
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source)
        try:
            payload = _loads_unique_json(path.read_text(encoding="utf-8"))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise MLTrainingError(
                f"Could not read JSON file {path}: {exc}"
            ) from exc
    if not isinstance(payload, dict):
        raise MLTrainingError("JSON input must contain an object at its top level.")
    return payload


def _validated_json_copy(value: Any, *, label: str, depth: int = 0) -> Any:
    """Copy JSON metadata exactly, failing instead of silently truncating it."""
    if depth > 8:
        raise MLTrainingError(f"{label} exceeds the maximum metadata depth.")
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MLTrainingError(f"{label} contains a non-finite number.")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 10000:
            raise MLTrainingError(f"{label} contains too many list items.")
        return [
            _validated_json_copy(
                item,
                label=f"{label}[{index}]",
                depth=depth + 1,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        if len(value) > 10000:
            raise MLTrainingError(f"{label} contains too many object fields.")
        copied: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, (str, int, float, bool)):
                raise MLTrainingError(f"{label} contains a non-JSON object key.")
            key = str(raw_key)
            if len(key) > 256:
                raise MLTrainingError(f"{label} contains an overlong object key.")
            copied[key] = _validated_json_copy(
                item,
                label=f"{label}.{key}",
                depth=depth + 1,
            )
        return copied
    raise MLTrainingError(
        f"{label} contains unsupported metadata type {type(value).__name__}."
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _dataset_lineage_digest(lineage: Mapping[str, Any]) -> str:
    payload = dict(lineage)
    payload.pop("sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _lineage_image_digest(
    image_id: int,
    *,
    original_sha256: str | None,
    source_id: str | None,
    license_id: int | None,
) -> str:
    """Return a stable opaque image reference without retaining local names."""
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "image_id": int(image_id),
                "original_sha256": original_sha256,
                "source_id": source_id,
                "license_id": license_id,
            }
        )
    ).hexdigest()


def _lineage_digest_for_image_record(
    image_id: int,
    image: Mapping[str, Any],
) -> str:
    metadata = image.get("blotgrid")
    if not isinstance(metadata, Mapping):
        metadata = {}
    raw_license_id = image.get("license")
    license_id = int(raw_license_id) if raw_license_id is not None else None
    source_id = str(metadata.get("source_id", "")).strip() or None
    original_sha256 = (
        str(metadata.get("original_sha256", "")).strip().lower() or None
    )
    return _lineage_image_digest(
        image_id,
        original_sha256=original_sha256,
        source_id=source_id,
        license_id=license_id,
    )


def _validate_dataset_license_lineage(
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    copied = _validated_json_copy(lineage, label="dataset licence lineage")
    if (
        type(copied.get("schema_version")) is not int
        or copied.get("schema_version")
        != DATASET_LICENSE_LINEAGE_SCHEMA_VERSION
    ):
        raise MLTrainingError(
            "Dataset licence lineage has an unsupported schema version."
        )
    declared = str(copied.get("sha256", "")).lower()
    if len(declared) != 64 or any(
        character not in "0123456789abcdef" for character in declared
    ):
        raise MLTrainingError("Dataset licence lineage has no valid SHA-256.")
    actual = _dataset_lineage_digest(copied)
    if declared != actual:
        raise MLTrainingError("Dataset licence lineage SHA-256 is inconsistent.")
    allowed_top_level = {
        "schema_version",
        "privacy_profile",
        "dataset_info",
        "licenses",
        "sources",
        "images",
        "summary",
        "sha256",
    }
    if set(copied) - allowed_top_level:
        raise MLTrainingError(
            "Dataset licence lineage contains non-public metadata fields."
        )
    if copied.get("privacy_profile") != "public_allowlist_v1":
        raise MLTrainingError(
            "Dataset licence lineage lacks the public metadata allowlist marker."
        )
    dataset_info = copied.get("dataset_info")
    if not isinstance(dataset_info, Mapping) or (
        set(dataset_info) - {"title", "version", "dois"}
    ):
        raise MLTrainingError("Dataset public identity metadata is invalid.")
    if not isinstance(dataset_info.get("dois", []), list):
        raise MLTrainingError("Dataset public DOI metadata is invalid.")
    for field in ("title", "version"):
        if field in dataset_info and (
            not isinstance(dataset_info[field], str)
            or not dataset_info[field].strip()
            or len(dataset_info[field]) > 256
            or any(ord(character) < 32 for character in dataset_info[field])
        ):
            raise MLTrainingError("Dataset public identity text is invalid.")
    dois = dataset_info.get("dois", [])
    if len(dois) > 100 or any(
        not isinstance(doi, str)
        or not doi.strip()
        or len(doi) > 256
        or any(ord(character) < 32 for character in doi)
        for doi in dois
    ):
        raise MLTrainingError("Dataset public DOI metadata is invalid.")
    if (
        dois != sorted(set(dois))
        or any(not _is_public_doi(doi) for doi in dois)
    ):
        raise MLTrainingError(
            "Dataset public DOI metadata is not a unique canonical DOI list."
        )

    raw_licenses = copied.get("licenses")
    raw_sources = copied.get("sources")
    raw_images = copied.get("images")
    summary = copied.get("summary")
    if not all(
        isinstance(value, list)
        for value in (raw_licenses, raw_sources, raw_images)
    ) or not isinstance(summary, Mapping):
        raise MLTrainingError("Dataset licence lineage collections are invalid.")
    licenses_by_id: dict[int, Mapping[str, Any]] = {}
    for license_record in raw_licenses:
        if not isinstance(license_record, Mapping) or (
            set(license_record)
            - {
                "id",
                "name",
                "url",
                "spdx_identifier",
                "publication_assessment",
            }
        ):
            raise MLTrainingError("Dataset licence record is not public-safe.")
        try:
            license_id = int(license_record["id"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise MLTrainingError("Dataset licence record has an invalid id.") from exc
        if license_id in licenses_by_id:
            raise MLTrainingError("Dataset licence ids must be unique.")
        for field in ("name", "url", "spdx_identifier"):
            if field in license_record and (
                not isinstance(license_record[field], str)
                or not license_record[field].strip()
                or len(license_record[field]) > 512
                or any(
                    ord(character) < 32
                    for character in license_record[field]
                )
            ):
                raise MLTrainingError("Dataset licence text is invalid.")
        assessment = license_record.get("publication_assessment")
        if not isinstance(assessment, Mapping) or (
            set(assessment) != _LICENSE_PUBLICATION_ASSESSMENT_FIELDS
        ):
            raise MLTrainingError(
                "Dataset licence publication assessment is incomplete."
            )
        if assessment.get("status") not in {
            "cleared",
            "review_required",
            "prohibited",
        }:
            raise MLTrainingError(
                "Dataset licence publication assessment status is invalid."
            )
        for field in (
            "redistribution_cleared",
            "commercial_use_cleared",
            "attribution_required",
            "share_alike_required",
        ):
            if not isinstance(assessment.get(field), bool):
                raise MLTrainingError(
                    "Dataset licence publication assessment flags must be booleans."
                )
        obligations = assessment.get("obligations")
        if (
            not isinstance(obligations, list)
            or len(obligations) != len(set(obligations))
            or any(
                obligation not in {"attribution", "share_alike"}
                for obligation in obligations
            )
        ):
            raise MLTrainingError(
                "Dataset licence publication obligations are invalid."
            )
        licenses_by_id[license_id] = license_record

    sources_by_id: dict[str, Mapping[str, Any]] = {}
    for source in raw_sources:
        if not isinstance(source, Mapping) or (
            set(source)
            - {
                "id",
                "doi",
                "license_id",
                "required_attribution",
                "publication_assessment",
            }
        ):
            raise MLTrainingError("Dataset source record is not public-safe.")
        source_id = str(source.get("id", "")).strip()
        if (
            not source_id
            or len(source_id) > 128
            or any(ord(character) < 32 for character in source_id)
            or source_id in sources_by_id
        ):
            raise MLTrainingError("Dataset source ids must be non-empty and unique.")
        source_license_id = source.get("license_id")
        if source_license_id is not None:
            try:
                source_license_id = int(source_license_id)
            except (TypeError, ValueError, OverflowError) as exc:
                raise MLTrainingError(
                    "Dataset source has an invalid licence id."
                ) from exc
            if source_license_id not in licenses_by_id:
                raise MLTrainingError("Dataset source refers to an unknown licence.")
        for field in ("doi", "required_attribution"):
            if field in source and (
                not isinstance(source[field], str)
                or not source[field].strip()
                or len(source[field]) > 2048
                or any(ord(character) < 32 for character in source[field])
            ):
                raise MLTrainingError("Dataset source public text is invalid.")
        assessment = source.get("publication_assessment")
        if not isinstance(assessment, Mapping) or (
            set(assessment) != _SOURCE_PUBLICATION_ASSESSMENT_FIELDS
        ):
            raise MLTrainingError(
                "Dataset source publication assessment is incomplete."
            )
        if assessment.get("status") not in {
            "cleared",
            "review_required",
            "prohibited",
        } or not isinstance(assessment.get("redistribution_cleared"), bool) or (
            not isinstance(assessment.get("attribution_resolved"), bool)
        ):
            raise MLTrainingError(
                "Dataset source publication assessment is invalid."
            )
        assessed_license_id = assessment.get("license_id")
        if assessed_license_id is not None:
            try:
                assessed_license_id = int(assessed_license_id)
            except (TypeError, ValueError, OverflowError) as exc:
                raise MLTrainingError(
                    "Dataset source assessment has an invalid licence id."
                ) from exc
        if assessed_license_id != source_license_id:
            raise MLTrainingError(
                "Dataset source assessment is not bound to its licence."
            )
        sources_by_id[source_id] = source

    seen_image_digests: set[str] = set()
    unresolved_licenses = 0
    unresolved_sources = 0
    for image in raw_images:
        if not isinstance(image, Mapping) or (
            set(image)
            - {"image_digest", "content_sha256", "license_id", "source_id"}
        ):
            raise MLTrainingError("Dataset image lineage is not public-safe.")
        image_digest = str(image.get("image_digest", "")).lower()
        if (
            len(image_digest) != 64
            or any(character not in "0123456789abcdef" for character in image_digest)
            or image_digest in seen_image_digests
        ):
            raise MLTrainingError(
                "Dataset image lineage digests must be unique SHA-256 values."
            )
        seen_image_digests.add(image_digest)
        content_sha256 = image.get("content_sha256")
        if content_sha256 is not None:
            content_sha256 = str(content_sha256).lower()
            if len(content_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in content_sha256
            ):
                raise MLTrainingError(
                    "Dataset image content digest is not a SHA-256 value."
                )
        license_id = image.get("license_id")
        source_id = image.get("source_id")
        if license_id is None:
            unresolved_licenses += 1
        else:
            try:
                license_id = int(license_id)
            except (TypeError, ValueError, OverflowError) as exc:
                raise MLTrainingError(
                    "Dataset image has an invalid licence id."
                ) from exc
            if license_id not in licenses_by_id:
                raise MLTrainingError("Dataset image refers to an unknown licence.")
        if source_id is None:
            unresolved_sources += 1
        elif str(source_id) not in sources_by_id:
            raise MLTrainingError("Dataset image refers to an unknown source.")
        if source_id is not None and license_id is not None:
            source_license = sources_by_id[str(source_id)].get("license_id")
            if source_license is not None:
                try:
                    source_license = int(source_license)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise MLTrainingError(
                        "Dataset source has an invalid licence id."
                    ) from exc
                if source_license != license_id:
                    raise MLTrainingError(
                        "Dataset image licence conflicts with its declared source."
                    )
    expected_summary = {
        "images": len(raw_images),
        "images_without_declared_license": unresolved_licenses,
        "images_without_declared_source": unresolved_sources,
    }
    if dict(summary) != expected_summary:
        raise MLTrainingError("Dataset licence lineage summary is inconsistent.")
    return copied


def _extract_dataset_license_lineage(
    coco: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize COCO licence/source metadata into a checkpoint-safe contract."""
    raw_info = coco.get("info", {})
    if raw_info is None:
        raw_info = {}
    if not isinstance(raw_info, Mapping):
        raise MLTrainingError("COCO 'info' must be an object when provided.")
    dataset_info: dict[str, Any] = {}
    title = raw_info.get("title")
    if title is not None:
        dataset_info["title"] = str(title).strip()
    version = raw_info.get("version")
    if version is not None:
        dataset_info["version"] = str(version).strip()
    raw_dois = raw_info.get("source_dois", raw_info.get("dois", []))
    if raw_dois is None:
        raw_dois = []
    if not isinstance(raw_dois, list):
        raise MLTrainingError("COCO dataset DOIs must be a list.")
    dataset_info["dois"] = sorted(
        {
            str(value).strip()
            for value in raw_dois
            if str(value).strip()
        }
    )

    raw_licenses = coco.get("licenses", [])
    if raw_licenses is None:
        raw_licenses = []
    if not isinstance(raw_licenses, list):
        raise MLTrainingError("COCO 'licenses' must be a list when provided.")
    licenses: list[dict[str, Any]] = []
    licenses_by_id: dict[int, dict[str, Any]] = {}
    for index, raw_license in enumerate(raw_licenses):
        if not isinstance(raw_license, Mapping) or "id" not in raw_license:
            raise MLTrainingError(f"COCO licence {index} requires an id.")
        try:
            license_id = int(raw_license["id"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise MLTrainingError("COCO licence ids must be integers.") from exc
        if license_id in licenses_by_id:
            raise MLTrainingError(f"Duplicate COCO licence id: {license_id}.")
        copied_license: dict[str, Any] = {"id": license_id}
        for field in ("name", "url", "spdx_identifier"):
            if raw_license.get(field) is not None:
                copied_license[field] = str(raw_license[field]).strip()
        raw_assessment = raw_license.get("publication_assessment")
        if raw_assessment is None:
            copied_license[
                "publication_assessment"
            ] = _review_required_license_assessment()
        elif isinstance(raw_assessment, Mapping):
            copied_license["publication_assessment"] = _validated_json_copy(
                raw_assessment,
                label=f"COCO licence {license_id} publication assessment",
            )
        else:
            raise MLTrainingError(
                f"COCO licence {license_id} publication assessment must be an object."
            )
        licenses_by_id[license_id] = copied_license
        licenses.append(copied_license)
    licenses.sort(key=lambda item: int(item["id"]))

    raw_blotgrid = coco.get("blotgrid", {})
    if raw_blotgrid is None:
        raw_blotgrid = {}
    if not isinstance(raw_blotgrid, Mapping):
        raise MLTrainingError("COCO 'blotgrid' metadata must be an object.")
    lineage_schema = raw_blotgrid.get("license_lineage_schema_version")
    if (
        lineage_schema is not None
        and (
            type(lineage_schema) is not int
            or lineage_schema != DATASET_LICENSE_LINEAGE_SCHEMA_VERSION
        )
    ):
        raise MLTrainingError(
            "COCO licence-lineage schema marker is not supported."
        )
    raw_sources = raw_blotgrid.get("sources", [])
    if raw_sources is None:
        raw_sources = []
    if not isinstance(raw_sources, list):
        raise MLTrainingError("COCO BlotGrid sources must be a list.")
    sources: list[dict[str, Any]] = []
    sources_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, Mapping):
            raise MLTrainingError(f"COCO source {index} must be an object.")
        source_id = str(raw_source.get("id", "")).strip()
        if not source_id:
            raise MLTrainingError(f"COCO source {index} requires a non-empty id.")
        if source_id in sources_by_id:
            raise MLTrainingError(f"Duplicate COCO source id: {source_id!r}.")
        copied_source: dict[str, Any] = {"id": source_id}
        for field in ("doi", "required_attribution"):
            if raw_source.get(field) is not None:
                copied_source[field] = str(raw_source[field]).strip()
        if raw_source.get("license_id") is not None:
            try:
                source_license_id = int(raw_source["license_id"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise MLTrainingError(
                    f"COCO source {source_id!r} has an invalid licence id."
                ) from exc
            if source_license_id not in licenses_by_id:
                raise MLTrainingError(
                    f"COCO source {source_id!r} refers to unknown licence "
                    f"{source_license_id}."
                )
            copied_source["license_id"] = source_license_id
        else:
            source_license_id = None
        raw_assessment = raw_source.get("publication_assessment")
        if raw_assessment is None:
            copied_source[
                "publication_assessment"
            ] = _review_required_source_assessment(source_license_id)
        elif isinstance(raw_assessment, Mapping):
            copied_source["publication_assessment"] = _validated_json_copy(
                raw_assessment,
                label=f"COCO source {source_id!r} publication assessment",
            )
        else:
            raise MLTrainingError(
                f"COCO source {source_id!r} publication assessment must be an object."
            )
        sources_by_id[source_id] = copied_source
        sources.append(copied_source)
    sources.sort(key=lambda item: str(item["id"]))

    raw_images = coco.get("images")
    if not isinstance(raw_images, list):
        raise MLTrainingError("COCO JSON must contain an 'images' list.")
    image_lineage: list[dict[str, Any]] = []
    seen_image_ids: set[int] = set()
    unresolved_licenses = 0
    unresolved_sources = 0
    for index, raw_image in enumerate(raw_images):
        if not isinstance(raw_image, Mapping) or "id" not in raw_image:
            raise MLTrainingError(f"COCO image {index} requires an id.")
        try:
            image_id = int(raw_image["id"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise MLTrainingError("COCO image ids must be integers.") from exc
        if image_id in seen_image_ids:
            raise MLTrainingError(f"Duplicate COCO image id: {image_id}.")
        seen_image_ids.add(image_id)

        raw_license_id = raw_image.get("license")
        license_id: int | None = None
        if raw_license_id is not None:
            try:
                license_id = int(raw_license_id)
            except (TypeError, ValueError, OverflowError) as exc:
                raise MLTrainingError(
                    f"COCO image {image_id} has an invalid licence id."
                ) from exc
            if license_id not in licenses_by_id:
                raise MLTrainingError(
                    f"COCO image {image_id} refers to unknown licence {license_id}."
                )
        else:
            unresolved_licenses += 1

        raw_image_metadata = raw_image.get("blotgrid", {})
        if raw_image_metadata is None:
            raw_image_metadata = {}
        if not isinstance(raw_image_metadata, Mapping):
            raise MLTrainingError(
                f"COCO image {image_id} BlotGrid metadata must be an object."
            )
        raw_source_id = raw_image_metadata.get("source_id")
        source_id = (
            str(raw_source_id).strip() if raw_source_id is not None else None
        )
        if source_id == "":
            source_id = None
        if source_id is not None and source_id not in sources_by_id:
            raise MLTrainingError(
                f"COCO image {image_id} refers to unknown source {source_id!r}."
            )
        if source_id is None:
            unresolved_sources += 1
        source = sources_by_id.get(source_id or "", {})
        source_license_id = source.get("license_id")
        if (
            license_id is not None
            and source_license_id is not None
            and license_id != int(source_license_id)
        ):
            raise MLTrainingError(
                f"COCO image {image_id} licence does not match source {source_id!r}."
            )

        for image_key, source_key in (
            ("source_doi", "doi"),
            ("required_attribution", "required_attribution"),
        ):
            direct = raw_image_metadata.get(image_key)
            inherited = source.get(source_key)
            if direct is not None and inherited is not None and direct != inherited:
                raise MLTrainingError(
                    f"COCO image {image_id} {image_key} conflicts with source "
                    f"{source_id!r}."
                )
        original_sha256 = str(
            raw_image_metadata.get("original_sha256", "")
        ).strip().lower()
        if original_sha256 and (
            len(original_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in original_sha256
            )
        ):
            raise MLTrainingError(
                f"COCO image {image_id} has an invalid original_sha256."
            )
        image_lineage.append(
            {
                "image_digest": _lineage_image_digest(
                    image_id,
                    original_sha256=original_sha256 or None,
                    source_id=source_id,
                    license_id=license_id,
                ),
                "content_sha256": original_sha256 or None,
                "license_id": license_id,
                "source_id": source_id,
            }
        )
    image_lineage.sort(key=lambda item: str(item["image_digest"]))

    lineage: dict[str, Any] = {
        "schema_version": DATASET_LICENSE_LINEAGE_SCHEMA_VERSION,
        "privacy_profile": "public_allowlist_v1",
        "dataset_info": dataset_info,
        "licenses": licenses,
        "sources": sources,
        "images": image_lineage,
        "summary": {
            "images": len(image_lineage),
            "images_without_declared_license": unresolved_licenses,
            "images_without_declared_source": unresolved_sources,
        },
    }
    encoded = _canonical_json_bytes(lineage)
    if len(encoded) > MAX_DATASET_LICENSE_LINEAGE_BYTES:
        raise MLTrainingError(
            "Dataset licence lineage exceeds the 512 KiB checkpoint metadata "
            "budget; use a smaller, explicitly catalogued dataset."
        )
    lineage["sha256"] = hashlib.sha256(encoded).hexdigest()
    return lineage


def _safe_image_path(root: Path, filename: str) -> Path:
    """Resolve a COCO/project filename beneath its declared image directory."""
    root = root.expanduser().resolve()
    normalized = str(filename).replace("\\", "/")
    candidates = [root / normalized, root / Path(normalized).name]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            return resolved
    raise MLTrainingError(
        f"Image {filename!r} was not found beneath {root}."
    )


def _load_image_array(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise MLTrainingError(f"Could not decode training image: {path}") from exc
    return array


def _xywh_to_xyxy(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, box_width, box_height = (float(value) for value in bbox)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        return None
    if box_width <= 0 or box_height <= 0:
        return None
    x0 = float(np.clip(x, 0.0, width))
    y0 = float(np.clip(y, 0.0, height))
    x1 = float(np.clip(x + box_width, 0.0, width))
    y1 = float(np.clip(y + box_height, 0.0, height))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return [x0, y0, x1, y1]


def _annotation_is_pseudo(annotation: Mapping[str, Any]) -> bool:
    attributes = annotation.get("attributes")
    if not isinstance(attributes, Mapping):
        return True
    for field in ("pseudo_label", "annotation_approved", "review_required"):
        if field in attributes and not isinstance(attributes[field], bool):
            raise MLTrainingError(
                f"COCO annotation attribute {field!r} must be a literal boolean."
            )
    pseudo_label = attributes.get("pseudo_label")
    approved = attributes.get("annotation_approved")
    review_required = attributes.get("review_required")
    if (
        pseudo_label is True and approved is True
    ) or (
        pseudo_label is False
        and (approved is False or review_required is True)
    ) or (
        approved is True and review_required is True
    ):
        raise MLTrainingError(
            "COCO annotation pseudo/approval attributes are contradictory."
        )
    return not (
        pseudo_label is False
        and approved is True
        and review_required is False
    )


def _is_attested_negative(image: Mapping[str, Any]) -> bool:
    metadata = image.get("blotgrid")
    return bool(
        isinstance(metadata, Mapping)
        and metadata.get("annotation_complete") is True
        and metadata.get("negative_image") is True
    )


class CocoBandDataset:
    """Minimal COCO bounding-box dataset for category ``1 = band``."""

    def __init__(
        self,
        annotations: str | os.PathLike[str] | Mapping[str, Any],
        image_root: str | os.PathLike[str],
        *,
        image_ids: Iterable[int] | None = None,
        training: bool = False,
        include_pseudo_labels: bool = False,
        require_checksums: bool = False,
        known_checksums: Mapping[int, str] | None = None,
    ) -> None:
        self.coco = _read_json_object(annotations)
        self.license_lineage = _extract_dataset_license_lineage(self.coco)
        self.image_root = Path(image_root).expanduser().resolve()
        self.training = bool(training)
        self.include_pseudo_labels = bool(include_pseudo_labels)
        images = self.coco.get("images")
        annotations_list = self.coco.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations_list, list):
            raise MLTrainingError(
                "COCO JSON must contain 'images' and 'annotations' lists."
            )

        self.images_by_id: dict[int, dict[str, Any]] = {}
        for raw in images:
            if not isinstance(raw, dict) or "id" not in raw or "file_name" not in raw:
                raise MLTrainingError(
                    "Every COCO image requires an id and file_name."
                )
            try:
                image_id = int(raw["id"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise MLTrainingError("COCO image ids must be integers.") from exc
            if image_id in self.images_by_id:
                raise MLTrainingError(f"Duplicate COCO image id: {image_id}.")
            filename = str(raw["file_name"])
            self.images_by_id[image_id] = dict(raw)

        self.annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        seen_annotation_ids: set[int] = set()
        for raw in annotations_list:
            if not isinstance(raw, dict):
                continue
            try:
                image_id = int(raw.get("image_id"))
                category_id = int(raw.get("category_id"))
            except (TypeError, ValueError, OverflowError):
                continue
            if image_id not in self.images_by_id or category_id != 1:
                continue
            if _annotation_is_pseudo(raw) and not self.include_pseudo_labels:
                continue
            if "id" in raw:
                try:
                    annotation_id = int(raw["id"])
                except (TypeError, ValueError, OverflowError):
                    annotation_id = -1
                if annotation_id in seen_annotation_ids:
                    raise MLTrainingError(
                        f"Duplicate COCO annotation id: {annotation_id}."
                    )
                seen_annotation_ids.add(annotation_id)
            self.annotations_by_image[image_id].append(dict(raw))

        selected = (
            sorted(self.images_by_id)
            if image_ids is None
            else [int(value) for value in image_ids]
        )
        if len(selected) != len(set(selected)):
            raise MLTrainingError("Dataset image ids must be unique.")
        unknown = sorted(set(selected) - set(self.images_by_id))
        if unknown:
            raise MLTrainingError(f"Unknown COCO image ids: {unknown}.")
        self.image_ids = selected
        self.image_paths: dict[int, Path] = {}
        self.image_checksums: dict[int, str] = {}
        self.valid_box_counts: dict[int, int] = {}
        self.attested_negative_ids: set[int] = set()
        known = {int(key): str(value) for key, value in (known_checksums or {}).items()}
        for image_id in self.image_ids:
            image_record = self.images_by_id[image_id]
            filename = str(image_record["file_name"])
            image_path = _safe_image_path(self.image_root, filename)
            self.image_paths[image_id] = image_path
            actual_width, actual_height = _image_dimensions(image_path)
            try:
                declared_width = int(image_record.get("width", actual_width))
                declared_height = int(image_record.get("height", actual_height))
            except (TypeError, ValueError, OverflowError) as exc:
                raise MLTrainingError(
                    f"Invalid COCO dimensions for image {filename!r}."
                ) from exc
            if (declared_width, declared_height) != (actual_width, actual_height):
                raise MLTrainingError(
                    f"COCO dimensions for {filename!r} do not match the image file."
                )

            metadata = image_record.get("blotgrid")
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("annotation_complete") is not True
            ):
                raise MLTrainingError(
                    f"COCO image {filename!r} lacks literal whole-image "
                    "annotation_complete=true attestation."
                )
            expected_sha256 = None
            if metadata.get("working_pixels_exported") is True:
                expected_sha256 = metadata.get("exported_working_sha256")
            if not expected_sha256:
                expected_sha256 = metadata.get("original_sha256")
            if expected_sha256 is None and require_checksums:
                raise MLTrainingError(
                    f"COCO image {filename!r} has no verifiable SHA-256 metadata. "
                    "Use an explicit allow-unverified override only for trusted data."
                )
            actual_sha256 = known.get(image_id) or _sha256_file(image_path)
            if expected_sha256 is not None:
                expected = str(expected_sha256).strip().lower()
                if (
                    len(expected) != 64
                    or any(character not in "0123456789abcdef" for character in expected)
                ):
                    raise MLTrainingError(
                        f"COCO image {filename!r} has an invalid expected SHA-256."
                    )
                if actual_sha256.lower() != expected:
                    raise MLTrainingError(
                        f"Image checksum mismatch for {filename!r}; training pixels "
                        "do not match the exported annotation coordinate system."
                    )
            self.image_checksums[image_id] = actual_sha256

            valid_boxes, _ = self._boxes_for_image(
                image_id, width=actual_width, height=actual_height
            )
            self.valid_box_counts[image_id] = len(valid_boxes)
            if _is_attested_negative(image_record):
                self.attested_negative_ids.add(image_id)
            elif not len(valid_boxes):
                raise MLTrainingError(
                    f"Image {filename!r} has no usable category-1 boxes after "
                    "clipping and is not an attested complete negative."
                )

    def __len__(self) -> int:
        return len(self.image_ids)

    def _boxes_for_image(
        self,
        image_id: int,
        *,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        boxes: list[list[float]] = []
        crowds: list[int] = []
        for annotation in self.annotations_by_image.get(image_id, []):
            converted = _xywh_to_xyxy(
                annotation.get("bbox"), width=width, height=height
            )
            if converted is None:
                continue
            boxes.append(converted)
            crowds.append(int(bool(annotation.get("iscrowd", 0))))
        return (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(crowds, dtype=np.int64),
        )

    @staticmethod
    def _augment(
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply box-safe polarity, contrast, and spatial augmentations."""
        image = np.asarray(image, dtype=np.float32)
        boxes = np.asarray(boxes, dtype=np.float32).copy()
        height, width = image.shape
        if random.random() < 0.5:
            image = 1.0 - image
        contrast = random.uniform(0.72, 1.30)
        midpoint = float(np.median(image))
        image = np.clip((image - midpoint) * contrast + midpoint, 0.0, 1.0)
        gamma = random.uniform(0.82, 1.18)
        image = np.power(image, gamma, dtype=np.float32)
        if random.random() < 0.5:
            image = image[:, ::-1].copy()
            if len(boxes):
                old_x0 = boxes[:, 0].copy()
                old_x1 = boxes[:, 2].copy()
                boxes[:, 0] = width - old_x1
                boxes[:, 2] = width - old_x0
        if random.random() < 0.25:
            image = image[::-1, :].copy()
            if len(boxes):
                old_y0 = boxes[:, 1].copy()
                old_y1 = boxes[:, 3].copy()
                boxes[:, 1] = height - old_y1
                boxes[:, 3] = height - old_y0
        return np.ascontiguousarray(image), boxes

    def __getitem__(self, index: int):
        # Importing through the shared runtime guard keeps this module optional.
        from .ml_detector import _require_ml_runtime

        torch, _ = _require_ml_runtime()
        image_id = self.image_ids[index]
        raw = _load_image_array(self.image_paths[image_id])
        image = normalize_grayscale(raw)
        height, width = image.shape
        boxes, crowds = self._boxes_for_image(
            image_id, width=width, height=height
        )
        if self.training:
            image, boxes = self._augment(image, boxes)
        rgb = np.ascontiguousarray(np.repeat(image[None, :, :], 3, axis=0))
        image_tensor = torch.from_numpy(rgb)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        if len(boxes):
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )
        else:
            area = torch.zeros((0,), dtype=torch.float32)
        target = {
            "boxes": boxes_tensor,
            "labels": labels,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.as_tensor(crowds, dtype=torch.int64),
        }
        return image_tensor, target


def _collate_detection_batch(batch):
    return tuple(zip(*batch))


def _resolve_split_value(
    value: Any,
    images_by_id: Mapping[int, Mapping[str, Any]],
) -> int:
    filenames = {
        str(image["file_name"]): image_id
        for image_id, image in images_by_id.items()
    }
    basename_matches: dict[str, list[int]] = defaultdict(list)
    for image_id, image in images_by_id.items():
        filename = str(image["file_name"]).replace("\\", "/")
        basename_matches[Path(filename).name].append(image_id)
    if isinstance(value, str):
        if value in filenames:
            return filenames[value]
        normalized = value.replace("\\", "/")
        if normalized in filenames:
            return filenames[normalized]
        matches = basename_matches.get(Path(normalized).name, [])
        if len(matches) == 1:
            return matches[0]
    try:
        image_id = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MLTrainingError(f"Unknown split item: {value!r}.") from exc
    if image_id not in images_by_id:
        raise MLTrainingError(f"Unknown split image id: {image_id}.")
    return image_id


def image_leakage_groups(
    images_by_id: Mapping[int, Mapping[str, Any]],
) -> list[list[int]]:
    """Group images connected by a study id or original-image SHA-256."""
    image_ids = sorted(int(value) for value in images_by_id)
    parent = {image_id: image_id for image_id in image_ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    token_owner: dict[tuple[str, str], int] = {}
    for image_id in image_ids:
        metadata = images_by_id[image_id].get("blotgrid")
        if not isinstance(metadata, Mapping):
            continue
        tokens: list[tuple[str, str]] = []
        study_group = str(metadata.get("study_group") or "").strip()
        original_sha256 = str(metadata.get("original_sha256") or "").strip().lower()
        if study_group:
            tokens.append(("study_group", study_group))
        if original_sha256:
            tokens.append(("original_sha256", original_sha256))
        for token in tokens:
            if token in token_owner:
                union(image_id, token_owner[token])
            else:
                token_owner[token] = image_id

    groups: dict[int, list[int]] = defaultdict(list)
    for image_id in image_ids:
        groups[find(image_id)].append(image_id)
    return sorted(
        (sorted(group) for group in groups.values()),
        key=lambda group: (group[0], len(group)),
    )


def make_train_val_split(
    images_by_id: Mapping[int, Mapping[str, Any]],
    *,
    validation_fraction: float = 0.2,
    seed: int = 1337,
    split: str | os.PathLike[str] | Mapping[str, Any] | None = None,
    train_only_ids: Iterable[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Create or validate a disjoint image-level train/validation split."""
    all_ids = sorted(int(value) for value in images_by_id)
    if len(all_ids) < 2:
        raise MLTrainingError(
            "Training requires at least two images for a held-out validation split."
        )
    required_train = {int(value) for value in (train_only_ids or [])}
    unknown_train_only = sorted(required_train - set(all_ids))
    if unknown_train_only:
        raise MLTrainingError(
            f"Unknown train-only image ids: {unknown_train_only}."
        )
    leakage_groups = image_leakage_groups(images_by_id)
    if split is not None:
        payload = _read_json_object(split)
        if not isinstance(payload.get("train"), list) or not isinstance(
            payload.get("val"), list
        ):
            raise MLTrainingError(
                "Split JSON must contain 'train' and 'val' lists."
            )
        train_ids = [
            _resolve_split_value(value, images_by_id)
            for value in payload["train"]
        ]
        val_ids = [
            _resolve_split_value(value, images_by_id)
            for value in payload["val"]
        ]
        if not train_ids or not val_ids:
            raise MLTrainingError("Train and validation splits must both be non-empty.")
        if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
            raise MLTrainingError("Split JSON contains duplicate images.")
        overlap = set(train_ids) & set(val_ids)
        if overlap:
            raise MLTrainingError(
                f"Train/validation leakage: image ids appear in both splits: "
                f"{sorted(overlap)}."
            )
        for group in leakage_groups:
            group_set = set(group)
            if group_set & set(train_ids) and group_set & set(val_ids):
                raise MLTrainingError(
                    "Train/validation leakage: images sharing a study_group or "
                    f"original_sha256 were divided across splits: {group}."
                )
        pseudo_in_validation = sorted(required_train & set(val_ids))
        if pseudo_in_validation:
            raise MLTrainingError(
                "Pseudo-label images are train-only and cannot appear in validation: "
                f"{pseudo_in_validation}."
            )
        covered = set(train_ids) | set(val_ids)
        if covered != set(all_ids):
            missing = sorted(set(all_ids) - covered)
            unknown = sorted(covered - set(all_ids))
            raise MLTrainingError(
                f"Split JSON must assign every dataset image exactly once "
                f"(missing={missing}, unknown={unknown})."
            )
        return train_ids, val_ids

    missing_study_groups: list[int] = []
    for image_id in all_ids:
        metadata = images_by_id[image_id].get("blotgrid")
        study_group = (
            str(metadata.get("study_group") or "").strip()
            if isinstance(metadata, Mapping)
            else ""
        )
        if not study_group:
            missing_study_groups.append(image_id)
    if missing_study_groups:
        raise MLTrainingError(
            "Automatic splitting requires a non-empty study_group for every "
            "image. Add the missing groups or provide a deliberate --split-json "
            f"(missing image ids: {missing_study_groups})."
        )

    try:
        fraction = float(validation_fraction)
    except (TypeError, ValueError) as exc:
        raise MLTrainingError("validation_fraction must be numeric.") from exc
    if not 0.0 < fraction < 1.0:
        raise MLTrainingError("validation_fraction must be between 0 and 1.")
    forced_group_indexes = {
        index
        for index, group in enumerate(leakage_groups)
        if set(group) & required_train
    }
    candidate_groups = [
        group
        for index, group in enumerate(leakage_groups)
        if index not in forced_group_indexes
    ]
    if not candidate_groups:
        raise MLTrainingError(
            "No clean leakage group remains for validation; pseudo-label groups "
            "are train-only."
        )
    random.Random(int(seed)).shuffle(candidate_groups)
    maximum_validation_groups = (
        len(candidate_groups)
        if forced_group_indexes
        else len(candidate_groups) - 1
    )
    if maximum_validation_groups < 1:
        raise MLTrainingError(
            "At least two independent study/checksum groups are required for "
            "held-out validation."
        )
    target_validation_images = max(1, int(round(len(all_ids) * fraction)))
    chosen_validation_groups: list[list[int]] = []
    validation_images = 0
    for group in candidate_groups:
        if len(chosen_validation_groups) >= maximum_validation_groups:
            break
        if chosen_validation_groups and validation_images >= target_validation_images:
            break
        chosen_validation_groups.append(group)
        validation_images += len(group)
    val_ids = sorted(
        image_id
        for group in chosen_validation_groups
        for image_id in group
    )
    train_ids = sorted(set(all_ids) - set(val_ids))
    if not train_ids or not val_ids:
        raise MLTrainingError(
            "Could not create non-empty leakage-safe train and validation splits."
        )
    return train_ids, val_ids


def split_manifest(
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    images_by_id: Mapping[int, Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "seed": int(seed),
        "train": [int(value) for value in train_ids],
        "val": [int(value) for value in val_ids],
        "train_files": [
            str(images_by_id[int(value)]["file_name"]) for value in train_ids
        ],
        "val_files": [
            str(images_by_id[int(value)]["file_name"]) for value in val_ids
        ],
        "leakage_groups": image_leakage_groups(images_by_id),
    }


def box_iou_matrix(
    first: Iterable[Iterable[float]],
    second: Iterable[Iterable[float]],
) -> np.ndarray:
    """Return an ``N x M`` IoU matrix for ``[x0, y0, x1, y1]`` boxes."""
    boxes_a = np.asarray(list(first), dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(list(second), dtype=np.float64).reshape(-1, 4)
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    size = np.clip(bottom_right - top_left, 0.0, None)
    intersection = size[..., 0] * size[..., 1]
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * np.clip(
        boxes_a[:, 3] - boxes_a[:, 1], 0.0, None
    )
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * np.clip(
        boxes_b[:, 3] - boxes_b[:, 1], 0.0, None
    )
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def detection_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute greedy one-to-one box precision, recall, F1, and matched IoU."""
    if not 0.0 <= float(score_threshold) <= 1.0:
        raise MLTrainingError("score_threshold must be between 0 and 1.")
    if not 0.0 < float(iou_threshold) <= 1.0:
        raise MLTrainingError("iou_threshold must be in (0, 1].")
    true_positive = false_positive = false_negative = 0
    matched_ious: list[float] = []
    image_count = 0
    for record in records:
        image_count += 1
        predictions = np.asarray(
            record.get("pred_boxes", []), dtype=np.float64
        ).reshape(-1, 4)
        scores = np.asarray(
            record.get("pred_scores", []), dtype=np.float64
        ).reshape(-1)
        ground_truth = np.asarray(
            record.get("gt_boxes", []), dtype=np.float64
        ).reshape(-1, 4)
        if len(scores) != len(predictions):
            raise MLTrainingError("Prediction boxes and scores have different lengths.")
        keep = np.isfinite(scores) & (scores >= float(score_threshold))
        predictions = predictions[keep]
        scores = scores[keep]
        order = np.argsort(-scores)
        predictions = predictions[order]
        ious = box_iou_matrix(predictions, ground_truth)
        used_ground_truth: set[int] = set()
        for prediction_index in range(len(predictions)):
            if not len(ground_truth):
                false_positive += 1
                continue
            available = [
                index
                for index in range(len(ground_truth))
                if index not in used_ground_truth
            ]
            if not available:
                false_positive += 1
                continue
            best = max(available, key=lambda index: ious[prediction_index, index])
            best_iou = float(ious[prediction_index, best])
            if best_iou >= float(iou_threshold):
                true_positive += 1
                used_ground_truth.add(best)
                matched_ious.append(best_iou)
            else:
                false_positive += 1
        false_negative += len(ground_truth) - len(used_ground_truth)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else (1.0 if false_negative == 0 else 0.0)
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "images": int(image_count),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": (
            float(np.mean(matched_ious)) if matched_ious else 0.0
        ),
        "score_threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
    }


def _seed_everything(seed: int) -> None:
    from .ml_detector import _require_ml_runtime

    torch, _ = _require_ml_runtime()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, TypeError):
        pass


def _epoch_seed(seed: int, epoch: int) -> int:
    """Derive a stable independent RNG seed for a zero-based epoch."""
    payload = f"blotgrid-training-epoch-v1:{int(seed)}:{int(epoch)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    from .ml_detector import _require_ml_runtime

    torch, _ = _require_ml_runtime()
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _resolve_training_device(requested: str):
    from .ml_detector import _resolve_device

    return _resolve_device(requested)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_sidecar_preflight(
    metadata: Mapping[str, Any],
    *,
    include_optimizer: bool,
) -> dict[str, Any]:
    checkpoint_metadata = dict(metadata)
    checkpoint_metadata["optimizer_state_included"] = bool(include_optimizer)
    sidecar = dict(_safe_metadata(checkpoint_metadata))
    if sidecar != checkpoint_metadata:
        raise MLTrainingError(
            "Checkpoint metadata contains unsupported or over-limit values; "
            "refusing to write a silently truncated provenance sidecar."
        )
    candidate = dict(sidecar)
    candidate["checkpoint_sha256"] = "0" * 64
    try:
        encoded = json.dumps(
            candidate,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise MLTrainingError(
            "Checkpoint metadata could not be serialized safely."
        ) from exc
    if len(encoded) > MAX_CHECKPOINT_METADATA_BYTES:
        raise MLTrainingError(
            "Checkpoint metadata exceeds the 1 MiB verified-sidecar limit."
        )
    return sidecar


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    metadata: Mapping[str, Any],
    include_optimizer: bool = True,
) -> None:
    from .ml_detector import _require_ml_runtime

    torch, _ = _require_ml_runtime()
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = dict(metadata)
    checkpoint_metadata["optimizer_state_included"] = bool(include_optimizer)
    sidecar = _checkpoint_sidecar_preflight(
        metadata,
        include_optimizer=include_optimizer,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    checkpoint_payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": model.state_dict(),
        "metadata": checkpoint_metadata,
    }
    if include_optimizer:
        checkpoint_payload["optimizer_state_dict"] = optimizer.state_dict()
    try:
        torch.save(checkpoint_payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar["checkpoint_sha256"] = _sha256_file(path)
    _atomic_json_write(_metadata_sidecar(path), sidecar)


def _evaluate_model(
    model: Any,
    loader: Any,
    device: Any,
    *,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    from .ml_detector import _require_ml_runtime

    torch, _ = _require_ml_runtime()
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets):
                labels = output["labels"].detach().cpu().numpy()
                band_mask = labels == 1
                records.append(
                    {
                        "pred_boxes": output["boxes"].detach().cpu().numpy()[band_mask],
                        "pred_scores": output["scores"].detach().cpu().numpy()[band_mask],
                        "gt_boxes": target["boxes"].detach().cpu().numpy(),
                    }
                )
    return detection_metrics(
        records,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )


def _move_optimizer_to_device(optimizer: Any, device: Any) -> None:
    from .ml_detector import _require_ml_runtime

    torch, _ = _require_ml_runtime()
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _validated_resume_optimizer_state(
    checkpoint_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return a structurally complete optimizer state or reject continuation."""
    optimizer_state = checkpoint_payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise MLTrainingError(
            "Resume checkpoint lacks optimizer state. Use it for inference, or "
            "start a new named training run instead of silently restarting SGD."
        )
    state = optimizer_state.get("state")
    parameter_groups = optimizer_state.get("param_groups")
    if not isinstance(state, Mapping) or not isinstance(parameter_groups, list):
        raise MLTrainingError(
            "Resume checkpoint optimizer state is incomplete or malformed."
        )
    if not parameter_groups or any(
        not isinstance(group, Mapping)
        or not isinstance(group.get("params"), list)
        for group in parameter_groups
    ):
        raise MLTrainingError(
            "Resume checkpoint optimizer parameter groups are incomplete."
        )
    return optimizer_state


def _checkpoint_paths(
    output: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """Return the stable best and last-epoch checkpoint paths for one run."""
    output_path = Path(output).expanduser().resolve()
    last_path = output_path.with_name(
        output_path.stem + ".last" + (output_path.suffix or ".pt")
    )
    return output_path, last_path


def _training_environment(
    torch: Any,
    torchvision: Any,
    resolved_device: Any,
) -> dict[str, Any]:
    """Capture the runtime and exact training-source hashes."""
    source_root = Path(__file__).resolve().parents[1]
    source_sha256: dict[str, str] = {}
    for relative in (
        "wblot/ml_training.py",
        "wblot/ml_detector.py",
        "scripts/train_detector.py",
        "requirements-ml.txt",
    ):
        path = source_root / relative
        if path.is_file():
            source_sha256[relative] = _sha256_file(path)
    cudnn_version = None
    if hasattr(torch.backends, "cudnn"):
        try:
            cudnn_version = torch.backends.cudnn.version()
        except (AttributeError, RuntimeError):
            cudnn_version = None
    device_name = "CPU"
    device_capability = None
    device_total_memory = None
    if getattr(resolved_device, "type", "") == "cuda":
        try:
            device_name = str(torch.cuda.get_device_name(resolved_device))
            device_capability = list(
                torch.cuda.get_device_capability(resolved_device)
            )
            device_total_memory = int(
                torch.cuda.get_device_properties(resolved_device).total_memory
            )
        except (AssertionError, RuntimeError, TypeError, ValueError):
            device_name = str(resolved_device)
    libc_name, libc_version = platform.libc_ver()
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "libc_name": libc_name,
            "libc_version": libc_version,
        },
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "torchvision_git_revision": str(
            getattr(getattr(torchvision, "version", None), "git_version", "")
            or ""
        ),
        "numpy_version": str(np.__version__),
        "pillow_version": str(PILLOW_VERSION),
        "cuda_build": str(torch.version.cuda) if torch.version.cuda else None,
        "cudnn_version": int(cudnn_version) if cudnn_version else None,
        "device": str(resolved_device),
        "device_name": device_name,
        "device_capability": device_capability,
        "device_total_memory": device_total_memory,
        "deterministic_algorithms": True,
        "epoch_seed_scheme": "sha256(blotgrid-training-epoch-v1:seed:epoch)",
        "source_sha256": source_sha256,
    }


def _validate_resume_metadata(
    metadata: Mapping[str, Any],
    training_config: Mapping[str, Any],
    training_environment: Mapping[str, Any],
    *,
    dataset_license_lineage: Mapping[str, Any],
) -> None:
    """Reject a continuation whose data-independent run contract drifted."""
    if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise MLTrainingError(
            "Resume checkpoint predates the current licence-lineage checkpoint "
            "format. Start a new named run; old checkpoints remain inference-only."
        )
    saved_config = metadata.get("training_config")
    saved_environment = metadata.get("training_environment")
    if not isinstance(saved_config, Mapping) or not isinstance(
        saved_environment, Mapping
    ):
        raise MLTrainingError(
            "Resume checkpoint lacks the complete training contract. Start a "
            "new named run rather than continuing an older-format checkpoint."
        )
    for key, current_value in training_config.items():
        if key == "epochs_requested":
            continue
        if saved_config.get(key) != current_value:
            raise MLTrainingError(
                f"Resume training option {key!r} changed "
                f"({saved_config.get(key)!r} != {current_value!r}). "
                "Only the requested epoch count may change."
            )
    for key in (
        "python_version",
        "python_implementation",
        "platform",
        "torch_version",
        "torchvision_version",
        "torchvision_git_revision",
        "numpy_version",
        "pillow_version",
        "cuda_build",
        "cudnn_version",
        "device",
        "device_name",
        "device_capability",
        "device_total_memory",
        "deterministic_algorithms",
        "epoch_seed_scheme",
        "source_sha256",
    ):
        if saved_environment.get(key) != training_environment.get(key):
            raise MLTrainingError(
                f"Resume environment field {key!r} changed. Start a new named "
                "run so the checkpoint remains reproducible."
            )

    saved_lineage = metadata.get("dataset_license_lineage")
    if not isinstance(saved_lineage, Mapping):
        raise MLTrainingError(
            "Resume checkpoint lacks dataset licence lineage. Start a new named run."
        )
    validated_saved_lineage = _validate_dataset_license_lineage(saved_lineage)
    validated_current_lineage = _validate_dataset_license_lineage(
        dataset_license_lineage
    )
    if validated_saved_lineage != validated_current_lineage:
        raise MLTrainingError(
            "Resume dataset licence lineage differs from the current COCO dataset."
        )
    expected_image_digests = {
        str(record["image_digest"])
        for record in validated_saved_lineage["images"]
    }
    saved_partitions: list[set[str]] = []
    for key in ("train_image_digests", "val_image_digests"):
        values = metadata.get(key)
        if not isinstance(values, list):
            raise MLTrainingError(
                "Resume checkpoint lacks its public-safe split lineage."
            )
        digests = {str(value) for value in values}
        if len(digests) != len(values) or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        ):
            raise MLTrainingError(
                "Resume checkpoint split lineage contains invalid digests."
            )
        saved_partitions.append(digests)
    if saved_partitions[0] & saved_partitions[1] or (
        saved_partitions[0] | saved_partitions[1]
    ) != expected_image_digests:
        raise MLTrainingError(
            "Resume checkpoint split lineage does not cover the dataset exactly."
        )

    initialization = metadata.get("initialization_provenance")
    if not isinstance(initialization, Mapping):
        raise MLTrainingError(
            "Resume checkpoint lacks exact initialization provenance."
        )
    architecture = str(training_config.get("architecture", ""))
    try:
        initialization = validate_initialization_provenance(
            dict(initialization),
            architecture,
            torchvision_version=str(
                training_environment.get("torchvision_version", "")
            ),
            torchvision_git_revision=str(
                training_environment.get("torchvision_git_revision", "")
            ),
            require_verified_artifact=True,
        )
    except MLDetectorError as exc:
        raise MLTrainingError(
            f"Resume checkpoint initialization provenance is invalid: {exc}"
        ) from exc
    mode = str(initialization["mode"])
    pretrained_requested = bool(
        saved_config.get("pretrained_transfer_learning_requested")
    )
    if pretrained_requested != (mode == "torchvision_pretrained"):
        raise MLTrainingError(
            "Resume checkpoint initialization provenance conflicts with its "
            "training configuration."
        )

    if metadata.get("optimizer_state_included") is not True:
        raise MLTrainingError(
            "Resume checkpoint was not saved with optimizer state. Use it for "
            "inference, or start a new named training run."
        )
    epoch = metadata.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise MLTrainingError(
            "Resume checkpoint has no valid completed epoch."
        )
    best_epoch = metadata.get("best_epoch")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch < 0
        or best_epoch > epoch
    ):
        raise MLTrainingError(
            "Resume checkpoint has no valid best validation epoch."
        )
    best_metrics = metadata.get("best_metrics")
    current_metrics = metadata.get("current_metrics")
    history = metadata.get("history")
    if not isinstance(best_metrics, Mapping) or (
        set(best_metrics) != _PUBLIC_METRIC_FIELDS
    ):
        raise MLTrainingError(
            "Resume checkpoint lacks complete best validation metrics."
        )
    if not isinstance(current_metrics, Mapping) or (
        set(current_metrics) != _PUBLIC_METRIC_FIELDS
    ):
        raise MLTrainingError(
            "Resume checkpoint lacks complete current validation metrics."
        )
    if (
        not isinstance(history, list)
        or not history
        or len(history) > MAX_CHECKPOINT_HISTORY
    ):
        raise MLTrainingError(
            "Resume checkpoint lacks a bounded, non-empty training history."
        )

    count_fields = {"images", "true_positive", "false_positive", "false_negative"}
    unit_interval_fields = {
        "precision",
        "recall",
        "f1",
        "mean_matched_iou",
        "score_threshold",
        "iou_threshold",
    }

    def validate_metrics(record: Mapping[str, Any], *, label: str) -> None:
        for field in count_fields:
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MLTrainingError(
                    f"Resume checkpoint {label} field {field!r} is invalid."
                )
        true_positive = int(record["true_positive"])
        false_positive = int(record["false_positive"])
        false_negative = int(record["false_negative"])
        expected_precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        expected_recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        expected_f1 = (
            2.0
            * expected_precision
            * expected_recall
            / (expected_precision + expected_recall)
            if expected_precision + expected_recall
            else 0.0
        )
        if any(
            not math.isclose(
                float(record[field]),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for field, expected in (
                ("precision", expected_precision),
                ("recall", expected_recall),
                ("f1", expected_f1),
            )
        ):
            raise MLTrainingError(
                f"Resume checkpoint {label} metric arithmetic is inconsistent."
            )
        mean_iou = float(record["mean_matched_iou"])
        iou_threshold = float(record["iou_threshold"])
        if (
            true_positive == 0
            and not math.isclose(
                mean_iou,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ) or (
            true_positive > 0
            and mean_iou + 1e-12 < iou_threshold
        ):
            raise MLTrainingError(
                f"Resume checkpoint {label} mean IoU is inconsistent."
            )
        for field in unit_interval_fields:
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MLTrainingError(
                    f"Resume checkpoint {label} field {field!r} is invalid."
                )
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise MLTrainingError(
                    f"Resume checkpoint {label} field {field!r} is invalid."
                )

    validate_metrics(best_metrics, label="best metrics")
    validate_metrics(current_metrics, label="current metrics")
    history_epochs: list[int] = []
    for index, record in enumerate(history):
        if not isinstance(record, Mapping) or (
            set(record) != _PUBLIC_HISTORY_FIELDS
        ):
            raise MLTrainingError(
                f"Resume checkpoint history record {index} is incomplete."
            )
        record_epoch = record.get("epoch")
        if (
            isinstance(record_epoch, bool)
            or not isinstance(record_epoch, int)
            or record_epoch < 0
        ):
            raise MLTrainingError(
                f"Resume checkpoint history record {index} has an invalid epoch."
            )
        train_loss = record.get("train_loss")
        if (
            isinstance(train_loss, bool)
            or not isinstance(train_loss, (int, float))
            or not math.isfinite(float(train_loss))
            or float(train_loss) < 0.0
        ):
            raise MLTrainingError(
                f"Resume checkpoint history record {index} has an invalid loss."
            )
        validate_metrics(record, label=f"history record {index}")
        history_epochs.append(record_epoch)
    if any(
        current != previous + 1
        for previous, current in zip(history_epochs, history_epochs[1:])
    ) or history_epochs[-1] != epoch or history_epochs[0] != max(
        0,
        epoch - MAX_CHECKPOINT_HISTORY + 1,
    ) or len(history_epochs) != min(epoch + 1, MAX_CHECKPOINT_HISTORY):
        raise MLTrainingError(
            "Resume checkpoint history is not a contiguous continuation ending "
            "at its declared epoch."
        )
    if any(
        current_metrics[field] != history[-1][field]
        for field in _PUBLIC_METRIC_FIELDS
    ):
        raise MLTrainingError(
            "Resume checkpoint current metrics do not match its final history record."
        )
    best_f1 = float(best_metrics["f1"])
    current_f1 = float(current_metrics["f1"])
    best_iou = float(best_metrics["mean_matched_iou"])
    current_iou = float(current_metrics["mean_matched_iou"])
    if best_f1 < current_f1 or (
        best_f1 == current_f1 and best_iou < current_iou
    ):
        raise MLTrainingError(
            "Resume checkpoint best metrics are inconsistent with current metrics."
        )
    history_by_epoch = {
        int(record["epoch"]): record for record in history
    }
    best_record = history_by_epoch.get(best_epoch)
    if best_record is not None:
        if any(
            best_metrics[field] != best_record[field]
            for field in _PUBLIC_METRIC_FIELDS
        ):
            raise MLTrainingError(
                "Resume checkpoint best metrics do not match best_epoch history."
            )
    elif best_epoch >= history_epochs[0]:
        raise MLTrainingError(
            "Resume checkpoint best_epoch is not bound to its retained history."
        )
    history_best_rank = max(
        (
            float(record["f1"]),
            float(record["mean_matched_iou"]),
        )
        for record in history
    )
    if (best_f1, best_iou) < history_best_rank:
        raise MLTrainingError(
            "Resume checkpoint best metrics are below retained history."
        )


_PUBLIC_CHECKPOINT_METADATA_FIELDS = {
    "format_version",
    "publication_metadata_schema_version",
    "architecture",
    "num_classes",
    "categories",
    "epoch",
    "epochs_requested",
    "best_epoch",
    "best_metrics",
    "current_metrics",
    "selection_metric",
    "history",
    "training_config",
    "training_environment",
    "initialization_provenance",
    "pretrained_transfer_learning",
    "trainable_backbone_layers",
    "trainable_parameter_count",
    "seed",
    "score_threshold",
    "iou_threshold",
    "train_image_digests",
    "val_image_digests",
    "annotation_sha256",
    "annotation_complete_images",
    "dataset_license_lineage",
    "image_manifest",
    "image_manifest_sha256",
    "allow_unverified_images",
    "pseudo_label_annotations",
    "allow_pseudo_labels",
    "created_utc",
    "torch_version",
    "torchvision_version",
    "torchvision_git_revision",
    "device",
    "optimizer_state_included",
    "checkpoint_sha256",
}
_PUBLIC_TRAINING_CONFIG_FIELDS = {
    "architecture",
    "optimizer",
    "epochs_requested",
    "batch_size",
    "learning_rate",
    "momentum",
    "weight_decay",
    "validation_fraction",
    "seed",
    "num_workers",
    "score_threshold",
    "iou_threshold",
    "pretrained_transfer_learning_requested",
    "trainable_backbone_layers",
    "backbone_normalization",
    "allow_pseudo_labels",
    "allow_unverified_images",
}
_PUBLIC_TRAINING_ENVIRONMENT_FIELDS = {
    "python_version",
    "python_implementation",
    "platform",
    "torch_version",
    "torchvision_version",
    "torchvision_git_revision",
    "numpy_version",
    "pillow_version",
    "cuda_build",
    "cudnn_version",
    "device",
    "device_name",
    "device_capability",
    "device_total_memory",
    "deterministic_algorithms",
    "epoch_seed_scheme",
    "source_sha256",
}
_PUBLIC_TRAINING_SOURCE_FILES = {
    "wblot/ml_training.py",
    "wblot/ml_detector.py",
    "scripts/train_detector.py",
    "requirements-ml.txt",
}
_PUBLIC_METRIC_FIELDS = {
    "images",
    "true_positive",
    "false_positive",
    "false_negative",
    "precision",
    "recall",
    "f1",
    "mean_matched_iou",
    "score_threshold",
    "iou_threshold",
}
_PUBLIC_HISTORY_FIELDS = _PUBLIC_METRIC_FIELDS | {"epoch", "train_loss"}
_DOI_PATTERN = re.compile(
    r"10\.[0-9]{4,9}/[-._;()/:A-Z0-9]+",
    re.IGNORECASE,
)


def _is_sha256_text(value: Any) -> bool:
    if not isinstance(value, str) or value != value.lower():
        return False
    digest = value
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _is_public_doi(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value == value.strip()
        and len(value) <= 256
        and _DOI_PATTERN.fullmatch(value) is not None
    )


def _attribution_mentions_exact_doi(attribution: Any, doi: str) -> bool:
    if not isinstance(attribution, str) or len(attribution) < 32:
        return False
    return doi.lower() in {
        match.group(0).lower() for match in _DOI_PATTERN.finditer(attribution)
    }


def _public_metadata_allowlist_violations(
    metadata: Mapping[str, Any],
) -> list[str]:
    """Return every field-level breach of the generated public schema."""
    def valid_int(value: Any, *, minimum: int = 0) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, int)
            and value >= minimum
        )

    def valid_number(
        value: Any,
        *,
        minimum: float = 0.0,
        maximum: float | None = None,
    ) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        numeric = float(value)
        return (
            math.isfinite(numeric)
            and numeric >= minimum
            and (maximum is None or numeric <= maximum)
        )

    def valid_version_token(value: Any, *, allow_empty: bool = False) -> bool:
        if not isinstance(value, str):
            return False
        if not value:
            return allow_empty
        return (
            len(value) <= 128
            and value[0].isdigit()
            and all(
                character.isascii()
                and (character.isalnum() or character in ".-_+")
                for character in value
            )
        )

    def valid_platform_record(value: Any) -> bool:
        if not isinstance(value, Mapping) or set(value) != {
            "system",
            "release",
            "machine",
            "libc_name",
            "libc_version",
        }:
            return False
        libc_name = value.get("libc_name")
        libc_version = value.get("libc_version")
        return (
            value.get("system") == "Linux"
            and value.get("machine")
            in {"x86_64", "amd64", "aarch64", "arm64", "ppc64le", "s390x"}
            and valid_version_token(value.get("release"))
            and libc_name in {"", "glibc", "musl", "libc"}
            and valid_version_token(libc_version, allow_empty=True)
        )

    def valid_metrics(record: Any) -> bool:
        if not isinstance(record, Mapping) or set(record) != _PUBLIC_METRIC_FIELDS:
            return False
        if not all(
            valid_int(record.get(field))
            for field in (
                "images",
                "true_positive",
                "false_positive",
                "false_negative",
            )
        ) or not all(
            valid_number(record.get(field), maximum=1.0)
            for field in (
                "precision",
                "recall",
                "f1",
                "mean_matched_iou",
                "score_threshold",
                "iou_threshold",
            )
        ):
            return False
        true_positive = record["true_positive"]
        false_positive = record["false_positive"]
        false_negative = record["false_negative"]
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        expected_precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )
        expected_recall = (
            true_positive / recall_denominator
            if recall_denominator
            else 0.0
        )
        expected_f1 = (
            2.0
            * expected_precision
            * expected_recall
            / (expected_precision + expected_recall)
            if expected_precision + expected_recall
            else 0.0
        )
        mean_matched_iou = float(record["mean_matched_iou"])
        iou_threshold = float(record["iou_threshold"])
        if (
            true_positive == 0
            and not math.isclose(
                mean_matched_iou,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ) or (
            true_positive > 0
            and mean_matched_iou + 1e-12 < iou_threshold
        ):
            return False
        return all(
            math.isclose(
                float(record[field]),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for field, expected in (
                ("precision", expected_precision),
                ("recall", expected_recall),
                ("f1", expected_f1),
            )
        )

    violations = [
        f"metadata.{field}"
        for field in sorted(set(metadata) - _PUBLIC_CHECKPOINT_METADATA_FIELDS)
    ]
    missing = _PUBLIC_CHECKPOINT_METADATA_FIELDS - set(metadata)
    violations.extend(
        f"metadata.{field} (missing)" for field in sorted(missing)
    )

    config = metadata.get("training_config")
    if not isinstance(config, Mapping):
        violations.append("metadata.training_config")
    else:
        violations.extend(
            f"metadata.training_config.{field}"
            for field in sorted(set(config) - _PUBLIC_TRAINING_CONFIG_FIELDS)
        )
        violations.extend(
            f"metadata.training_config.{field} (missing)"
            for field in sorted(_PUBLIC_TRAINING_CONFIG_FIELDS - set(config))
        )
        if (
            config.get("architecture") not in SUPPORTED_ARCHITECTURES
            or config.get("optimizer") != "SGD"
            or config.get("backbone_normalization")
            not in {"BatchNorm2d", "FrozenBatchNorm2d"}
            or not valid_int(config.get("epochs_requested"), minimum=1)
            or not valid_int(config.get("batch_size"), minimum=1)
            or not valid_int(config.get("seed"))
            or not valid_int(config.get("num_workers"))
            or not valid_int(config.get("trainable_backbone_layers"))
            or not valid_number(config.get("learning_rate"), minimum=1e-300)
            or not valid_number(config.get("momentum"), maximum=1.0)
            or not valid_number(config.get("weight_decay"))
            or not valid_number(
                config.get("validation_fraction"),
                minimum=1e-300,
                maximum=1.0,
            )
            or config.get("validation_fraction") >= 1.0
            or not valid_number(config.get("score_threshold"), maximum=1.0)
            or not valid_number(
                config.get("iou_threshold"),
                minimum=1e-300,
                maximum=1.0,
            )
            or not isinstance(
                config.get("pretrained_transfer_learning_requested"), bool
            )
            or not isinstance(config.get("allow_pseudo_labels"), bool)
            or not isinstance(config.get("allow_unverified_images"), bool)
            or config.get("backbone_normalization")
            != (
                "FrozenBatchNorm2d"
                if config.get("pretrained_transfer_learning_requested")
                else "BatchNorm2d"
            )
        ):
            violations.append("metadata.training_config values")

    environment = metadata.get("training_environment")
    if not isinstance(environment, Mapping):
        violations.append("metadata.training_environment")
    else:
        violations.extend(
            f"metadata.training_environment.{field}"
            for field in sorted(
                set(environment) - _PUBLIC_TRAINING_ENVIRONMENT_FIELDS
            )
        )
        violations.extend(
            f"metadata.training_environment.{field} (missing)"
            for field in sorted(
                _PUBLIC_TRAINING_ENVIRONMENT_FIELDS - set(environment)
            )
        )
        source_hashes = environment.get("source_sha256")
        if not isinstance(source_hashes, Mapping) or (
            set(source_hashes) != _PUBLIC_TRAINING_SOURCE_FILES
        ):
            violations.append("metadata.training_environment.source_sha256")
        elif any(
            not _is_sha256_text(value) for value in source_hashes.values()
        ):
            violations.append(
                "metadata.training_environment.source_sha256 values"
            )
        version_fields = (
            "python_version",
            "torch_version",
            "torchvision_version",
            "numpy_version",
            "pillow_version",
        )
        git_revision = environment.get("torchvision_git_revision")
        device = environment.get("device")
        device_name = environment.get("device_name")
        device_name_valid = (
            device_name == "CPU"
            if device == "cpu"
            else (
                isinstance(device_name, str)
                and len(device_name) <= 128
                and device_name.startswith(
                    (
                        "NVIDIA ",
                        "Tesla ",
                        "GeForce ",
                        "Quadro ",
                        "AMD ",
                        "Radeon ",
                        "Intel ",
                    )
                )
                and all(
                    character.isascii()
                    and (
                        character.isalnum()
                        or character in " .-_+()"
                    )
                    for character in device_name
                )
            )
        )
        if (
            any(
                not valid_version_token(environment.get(field))
                for field in version_fields
            )
            or environment.get("python_implementation") != "CPython"
            or not valid_platform_record(environment.get("platform"))
            or not (
                isinstance(git_revision, str)
                and len(git_revision) in {40, 64}
                and all(
                    character in "0123456789abcdef"
                    for character in git_revision.lower()
                )
            )
            or not (
                device == "cpu"
                or (
                    isinstance(device, str)
                    and (
                        device == "cuda"
                        or (
                            device.startswith("cuda:")
                            and device.removeprefix("cuda:").isdigit()
                        )
                    )
                )
            )
            or not device_name_valid
        ):
            violations.append("metadata.training_environment text values")
        if (
            environment.get("deterministic_algorithms") is not True
            or environment.get("epoch_seed_scheme")
            != "sha256(blotgrid-training-epoch-v1:seed:epoch)"
            or not (
                environment.get("cuda_build") is None
                or valid_version_token(environment.get("cuda_build"))
            )
            or not (
                environment.get("cudnn_version") is None
                or valid_int(environment.get("cudnn_version"), minimum=1)
            )
            or not (
                environment.get("device_total_memory") is None
                or valid_int(environment.get("device_total_memory"), minimum=1)
            )
        ):
            violations.append("metadata.training_environment runtime values")
        capability = environment.get("device_capability")
        if capability is not None and (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(not valid_int(value) for value in capability)
        ):
            violations.append("metadata.training_environment.device_capability")

    for key in ("best_metrics", "current_metrics"):
        metrics = metadata.get(key)
        if not valid_metrics(metrics):
            violations.append(f"metadata.{key}")
    history = metadata.get("history")
    if (
        not isinstance(history, list)
        or not history
        or len(history) > MAX_CHECKPOINT_HISTORY
    ):
        violations.append("metadata.history")
    else:
        for index, record in enumerate(history):
            if not isinstance(record, Mapping) or (
                set(record) != _PUBLIC_HISTORY_FIELDS
            ) or not valid_metrics(
                {
                    field: record.get(field)
                    for field in _PUBLIC_METRIC_FIELDS
                }
            ) or not valid_int(
                record.get("epoch")
            ) or not valid_number(record.get("train_loss")):
                violations.append(f"metadata.history[{index}]")
                break

    partitions: dict[str, list[str]] = {}
    for field in ("train_image_digests", "val_image_digests"):
        partition = metadata.get(field)
        if (
            not isinstance(partition, list)
            or not partition
            or any(not _is_sha256_text(value) for value in partition)
            or len(partition) != len(set(partition))
        ):
            violations.append(f"metadata.{field}")
        else:
            partitions[field] = partition
    if len(partitions) == 2 and set(
        partitions["train_image_digests"]
    ) & set(partitions["val_image_digests"]):
        violations.append("metadata split overlap")
    validation_images = len(partitions.get("val_image_digests", []))
    metric_records = [
        metadata.get("best_metrics"),
        metadata.get("current_metrics"),
    ]
    if isinstance(history, list):
        metric_records.extend(history)
    if validation_images and any(
        isinstance(record, Mapping)
        and record.get("images") != validation_images
        for record in metric_records
    ):
        violations.append("metadata validation metric cardinality")

    if (
        type(metadata.get("format_version")) is not int
        or metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION
        or type(metadata.get("publication_metadata_schema_version")) is not int
        or metadata.get("publication_metadata_schema_version")
        != PUBLICATION_METADATA_SCHEMA_VERSION
        or type(metadata.get("num_classes")) is not int
        or metadata.get("num_classes") != 2
        or metadata.get("categories") != {"0": "background", "1": "band"}
    ):
        violations.append("metadata generated schema values")
    if (
        metadata.get("architecture") not in SUPPORTED_ARCHITECTURES
        or not valid_int(metadata.get("epoch"))
        or not valid_int(metadata.get("epochs_requested"), minimum=1)
        or not valid_int(metadata.get("best_epoch"))
        or metadata.get("best_epoch") > metadata.get("epoch", -1)
        or metadata.get("selection_metric")
        != "validation_f1_then_mean_matched_iou"
        or not isinstance(metadata.get("pretrained_transfer_learning"), bool)
        or not valid_int(metadata.get("trainable_backbone_layers"))
        or not valid_int(metadata.get("trainable_parameter_count"), minimum=1)
        or not valid_int(metadata.get("seed"))
        or not valid_number(metadata.get("score_threshold"), maximum=1.0)
        or not valid_number(
            metadata.get("iou_threshold"),
            minimum=1e-300,
            maximum=1.0,
        )
        or not _is_sha256_text(metadata.get("annotation_sha256"))
        or not valid_int(
            metadata.get("annotation_complete_images"),
            minimum=1,
        )
        or not isinstance(metadata.get("allow_unverified_images"), bool)
        or not valid_int(metadata.get("pseudo_label_annotations"))
        or not isinstance(metadata.get("allow_pseudo_labels"), bool)
        or not isinstance(metadata.get("optimizer_state_included"), bool)
        or not _is_sha256_text(metadata.get("checkpoint_sha256"))
    ):
        violations.append("metadata generated scalar values")
    created_utc = metadata.get("created_utc")
    try:
        parsed_created = datetime.fromisoformat(str(created_utc).replace("Z", "+00:00"))
    except ValueError:
        parsed_created = None
    if parsed_created is None or parsed_created.tzinfo is None:
        violations.append("metadata.created_utc")
    if isinstance(config, Mapping) and (
        config.get("architecture") != metadata.get("architecture")
        or config.get("epochs_requested") != metadata.get("epochs_requested")
        or config.get("seed") != metadata.get("seed")
        or config.get("score_threshold") != metadata.get("score_threshold")
        or config.get("iou_threshold") != metadata.get("iou_threshold")
        or config.get("trainable_backbone_layers")
        != metadata.get("trainable_backbone_layers")
        or config.get("allow_pseudo_labels")
        != metadata.get("allow_pseudo_labels")
        or config.get("allow_unverified_images")
        != metadata.get("allow_unverified_images")
        or config.get("pretrained_transfer_learning_requested")
        != metadata.get("pretrained_transfer_learning")
    ):
        violations.append("metadata.training_config cross-field values")
    if isinstance(environment, Mapping) and (
        metadata.get("torch_version") != environment.get("torch_version")
        or metadata.get("torchvision_version")
        != environment.get("torchvision_version")
        or metadata.get("torchvision_git_revision")
        != environment.get("torchvision_git_revision")
        or metadata.get("device") != environment.get("device")
    ):
        violations.append("metadata.training_environment cross-field values")
    epoch = metadata.get("epoch")
    epochs_requested = metadata.get("epochs_requested")
    if (
        not valid_int(epoch)
        or not valid_int(epochs_requested, minimum=1)
        or epoch >= epochs_requested
    ):
        violations.append("metadata epoch continuity")
    if isinstance(history, list) and history and all(
        isinstance(record, Mapping) for record in history
    ):
        history_epochs = [record.get("epoch") for record in history]
        if (
            any(not valid_int(value) for value in history_epochs)
            or any(
                current != previous + 1
                for previous, current in zip(
                    history_epochs,
                    history_epochs[1:],
                )
            )
            or history_epochs[-1] != epoch
            or history_epochs[0]
            != max(0, epoch - MAX_CHECKPOINT_HISTORY + 1)
            or len(history_epochs) != min(epoch + 1, MAX_CHECKPOINT_HISTORY)
        ):
            violations.append("metadata.history epoch continuity")
        current_metrics = metadata.get("current_metrics")
        if isinstance(current_metrics, Mapping) and any(
            current_metrics.get(field) != history[-1].get(field)
            for field in _PUBLIC_METRIC_FIELDS
        ):
            violations.append("metadata.current_metrics history binding")
        best_metrics = metadata.get("best_metrics")
        best_epoch = metadata.get("best_epoch")
        if isinstance(best_metrics, Mapping) and isinstance(
            current_metrics, Mapping
        ):
            try:
                best_rank = (
                    float(best_metrics["f1"]),
                    float(best_metrics["mean_matched_iou"]),
                )
                current_rank = (
                    float(current_metrics["f1"]),
                    float(current_metrics["mean_matched_iou"]),
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                violations.append("metadata.best_metrics selection binding")
            else:
                if best_rank < current_rank:
                    violations.append(
                        "metadata.best_metrics selection binding"
                    )
                history_by_epoch = {
                    record.get("epoch"): record
                    for record in history
                    if isinstance(record, Mapping)
                    and valid_int(record.get("epoch"))
                }
                best_record = history_by_epoch.get(best_epoch)
                if best_record is not None:
                    if any(
                        best_metrics.get(field) != best_record.get(field)
                        for field in _PUBLIC_METRIC_FIELDS
                    ):
                        violations.append(
                            "metadata.best_metrics history binding"
                        )
                elif (
                    not valid_int(best_epoch)
                    or not history_epochs
                    or best_epoch >= history_epochs[0]
                ):
                    violations.append(
                        "metadata.best_metrics history binding"
                    )
                try:
                    history_best_rank = max(
                        (
                            float(record["f1"]),
                            float(record["mean_matched_iou"]),
                        )
                        for record in history
                    )
                except (KeyError, TypeError, ValueError, OverflowError):
                    violations.append(
                        "metadata.best_metrics history binding"
                    )
                else:
                    if best_rank < history_best_rank:
                        violations.append(
                            "metadata.best_metrics history binding"
                        )
        for index, record in enumerate(history):
            if any(
                record.get(field) != metadata.get(field)
                for field in ("score_threshold", "iou_threshold")
            ):
                violations.append(
                    f"metadata.history[{index}] threshold binding"
                )
                break
    for key in ("best_metrics", "current_metrics"):
        metrics = metadata.get(key)
        if isinstance(metrics, Mapping) and any(
            metrics.get(field) != metadata.get(field)
            for field in ("score_threshold", "iou_threshold")
        ):
            violations.append(f"metadata.{key} threshold binding")
    return violations


def checkpoint_publication_assessment(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure fail-closed assessment for publishing a trained checkpoint."""
    blockers: list[dict[str, str]] = []

    def block(code: str, message: str) -> None:
        blockers.append({"code": code, "message": message})

    if provenance.get("metadata_verified") is not True:
        block(
            "unverified_sidecar",
            "The checkpoint metadata sidecar is absent or does not match the model.",
        )
        metadata: Mapping[str, Any] = {}
    else:
        candidate = provenance.get("metadata")
        metadata = candidate if isinstance(candidate, Mapping) else {}
        if not metadata:
            block("missing_metadata", "Verified checkpoint metadata is empty.")
    if provenance.get("embedded_metadata_verified") is not True:
        block(
            "unverified_embedded_metadata",
            "Restricted checkpoint metadata does not exactly match its sidecar.",
        )
    if provenance.get("container_verified") is not True:
        block(
            "unverified_checkpoint_container",
            "Restricted checkpoint payload does not match the generated public "
            "model and optimizer container contract.",
        )
    if provenance.get("release_artifact_sanitized") is not True:
        block(
            "unsanitized_release_artifact",
            "Publication requires a distinct checkpoint rewritten from validated "
            "logical tensors; the local PyTorch archive is not releasable.",
        )
    verified_trainable_count = provenance.get(
        "verified_trainable_parameter_count"
    )
    if (
        type(verified_trainable_count) is not int
        or verified_trainable_count <= 0
        or metadata.get("trainable_parameter_count")
        != verified_trainable_count
    ):
        block(
            "unverified_checkpoint_container",
            "The declared trainable parameter count is not bound to the strict "
            "loaded model.",
        )
    if metadata.get("checkpoint_sha256") != provenance.get("sha256"):
        block(
            "checkpoint_digest_binding",
            "Checkpoint metadata SHA-256 does not match the assessed artifact.",
        )
    if (
        type(metadata.get("format_version")) is not int
        or metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION
    ):
        block(
            "checkpoint_schema",
            f"Publication requires checkpoint format {CHECKPOINT_FORMAT_VERSION}.",
        )
    if (
        type(metadata.get("publication_metadata_schema_version")) is not int
        or metadata.get("publication_metadata_schema_version")
        != PUBLICATION_METADATA_SCHEMA_VERSION
    ):
        block(
            "publication_schema",
            "Checkpoint metadata lacks the public allowlist schema marker.",
        )
    metadata_violations = _public_metadata_allowlist_violations(metadata)
    if metadata_violations:
        block(
            "private_metadata",
            "Checkpoint metadata differs from the public allowlist: "
            + ", ".join(metadata_violations),
        )

    lineage: dict[str, Any] | None = None
    raw_lineage = metadata.get("dataset_license_lineage")
    if isinstance(raw_lineage, Mapping):
        try:
            lineage = _validate_dataset_license_lineage(raw_lineage)
        except MLTrainingError as exc:
            block("dataset_lineage", str(exc))
    else:
        block("dataset_lineage", "Checkpoint has no dataset licence lineage.")

    if lineage is not None:
        summary = lineage["summary"]
        if summary["images_without_declared_license"]:
            block(
                "unresolved_image_licenses",
                "One or more training images have no declared licence.",
            )
        if summary["images_without_declared_source"]:
            block(
                "unresolved_image_sources",
                "One or more training images have no declared public source.",
            )
        licenses = {int(item["id"]): item for item in lineage["licenses"]}
        for license_id, license_record in licenses.items():
            if not str(license_record.get("name", "")).strip() or not (
                str(license_record.get("url", "")).strip()
                or str(license_record.get("spdx_identifier", "")).strip()
            ):
                block(
                    "incomplete_license",
                    f"Licence {license_id} lacks a public name and URL/SPDX id.",
                )
                continue
            identifier = str(
                license_record.get("spdx_identifier", "")
            ).strip()
            policy = _PUBLICATION_LICENSE_POLICIES.get(identifier)
            assessment = license_record.get("publication_assessment")
            if policy is None:
                block(
                    "license_not_cleared",
                    f"Licence {license_id} is not in the reviewed publication "
                    "policy allowlist.",
                )
                continue
            if (
                license_record.get("name") not in policy["names"]
                or license_record.get("url") != policy["url"]
            ):
                block(
                    "license_identity",
                    f"Licence {license_id} does not match its reviewed SPDX policy.",
                )
                continue
            expected_assessment = {
                "status": "cleared",
                "redistribution_cleared": True,
                "commercial_use_cleared": True,
                "attribution_required": policy["attribution_required"],
                "share_alike_required": policy["share_alike_required"],
                "obligations": policy["obligations"],
            }
            if assessment != expected_assessment:
                block(
                    "license_not_cleared",
                    f"Licence {license_id} lacks explicit redistribution and "
                    "commercial-use clearance with complete obligations.",
                )
                continue
            if policy["share_alike_required"]:
                block(
                    "share_alike_release_review",
                    f"Licence {license_id} requires a separately bound compatible "
                    "checkpoint release-licence decision.",
                )
        source_dois: set[str] = set()
        for source in lineage["sources"]:
            source_id = str(source["id"])
            license_id = source.get("license_id")
            if license_id is None or int(license_id) not in licenses:
                block(
                    "source_license",
                    f"Source {source_id!r} has no resolved licence.",
                )
            source_doi = str(source.get("doi", "")).strip()
            attribution = str(source.get("required_attribution", "")).strip()
            if not _is_public_doi(source_doi):
                block(
                    "source_identity",
                    f"Source {source_id!r} has no resolvable public DOI.",
                )
            else:
                source_dois.add(source_doi)
            if not _attribution_mentions_exact_doi(
                attribution,
                source_doi,
            ):
                block(
                    "source_attribution",
                    f"Source {source_id!r} lacks DOI-bound required attribution.",
                )
            expected_source_assessment = {
                "status": "cleared",
                "redistribution_cleared": True,
                "attribution_resolved": True,
                "license_id": license_id,
            }
            if source.get(
                "publication_assessment"
            ) != expected_source_assessment:
                block(
                    "source_not_cleared",
                    f"Source {source_id!r} lacks an explicit licence-bound "
                    "publication assessment.",
                )
        dataset_dois = set(lineage["dataset_info"].get("dois", []))
        if dataset_dois != source_dois:
            block(
                "dataset_source_identity",
                "Dataset DOI identity does not exactly match its declared sources.",
            )
        expected_digests = {
            str(record["image_digest"]) for record in lineage["images"]
        }
        partition_digests: list[set[str]] = []
        for key in ("train_image_digests", "val_image_digests"):
            raw_partition = metadata.get(key)
            if not isinstance(raw_partition, list):
                block(
                    "split_lineage",
                    "Checkpoint lacks its public-safe train/validation split lineage.",
                )
                partition_digests = []
                break
            values = {str(value) for value in raw_partition}
            if len(values) != len(raw_partition) or any(
                len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
                for value in values
            ):
                block(
                    "split_lineage",
                    "Checkpoint split lineage contains invalid image digests.",
                )
                partition_digests = []
                break
            partition_digests.append(values)
        if len(partition_digests) == 2 and (
            partition_digests[0] & partition_digests[1]
            or (partition_digests[0] | partition_digests[1])
            != expected_digests
        ):
            block(
                "split_lineage",
                "Checkpoint split lineage does not cover the dataset exactly.",
            )

    architecture = str(metadata.get("architecture", ""))
    environment = metadata.get("training_environment")
    initialization = metadata.get("initialization_provenance")
    if not isinstance(environment, Mapping) or not isinstance(
        initialization, Mapping
    ):
        block(
            "initializer_lineage",
            "Checkpoint lacks its complete runtime and initializer lineage.",
        )
    else:
        try:
            validated_initialization = validate_initialization_provenance(
                dict(initialization),
                architecture,
                torchvision_version=str(
                    environment.get("torchvision_version", "")
                ),
                torchvision_git_revision=str(
                    environment.get("torchvision_git_revision", "")
                ),
                require_verified_artifact=True,
            )
        except MLDetectorError as exc:
            block("initializer_lineage", str(exc))
        else:
            transfer_used = (
                validated_initialization["mode"] == "torchvision_pretrained"
            )
            if metadata.get("pretrained_transfer_learning") is not transfer_used:
                block(
                    "initializer_lineage",
                    "Checkpoint transfer-learning flags conflict with its "
                    "initializer provenance.",
                )
            if validated_initialization["mode"] == "torchvision_pretrained":
                block(
                    "pretrained_redistribution_review",
                    "TorchVision COCO initialization is review-required and is "
                    "not cleared here for derived-checkpoint redistribution.",
                )

    raw_manifest = metadata.get("image_manifest")
    manifest_fields = {
        "image_digest",
        "sha256",
        "study_group_sha256",
        "original_sha256",
        "license_id",
        "source_id",
    }
    if not isinstance(raw_manifest, list) or not raw_manifest:
        block(
            "image_manifest",
            "Checkpoint has no non-empty public-safe training image manifest.",
        )
    else:
        if metadata.get("annotation_complete_images") != len(raw_manifest):
            block(
                "whole_image_attestation",
                "Checkpoint whole-image annotation attestation does not cover "
                "its complete training manifest.",
            )
        manifest_by_digest: dict[str, Mapping[str, Any]] = {}
        for index, record in enumerate(raw_manifest):
            if not isinstance(record, Mapping) or set(record) != manifest_fields:
                block(
                    "private_metadata",
                    f"Image manifest record {index} differs from its public schema.",
                )
                continue
            image_digest = str(record.get("image_digest", "")).lower()
            if (
                not _is_sha256_text(image_digest)
                or image_digest in manifest_by_digest
            ):
                block(
                    "image_manifest",
                    f"Image manifest record {index} has a duplicate or invalid "
                    "image_digest.",
                )
                continue
            manifest_by_digest[image_digest] = record
            for field in (
                "sha256",
                "study_group_sha256",
                "original_sha256",
            ):
                value = record.get(field)
                if (
                    field in {"sha256", "study_group_sha256"}
                    and not _is_sha256_text(value)
                ) or (
                    field == "original_sha256"
                    and value is not None
                    and not _is_sha256_text(value)
                ):
                    block(
                        "image_manifest",
                        f"Image manifest record {index} has an invalid {field}.",
                    )
            if lineage is not None:
                lineage_records = {
                    str(item["image_digest"]): item
                    for item in lineage["images"]
                }
                expected = lineage_records.get(image_digest)
                if expected is None or any(
                    record.get(field) != expected.get(lineage_field)
                    for field, lineage_field in (
                        ("original_sha256", "content_sha256"),
                        ("license_id", "license_id"),
                        ("source_id", "source_id"),
                    )
                ):
                    block(
                        "image_manifest_lineage",
                        f"Image manifest record {index} conflicts with dataset lineage.",
                    )
        if lineage is not None:
            lineage_digests = {
                str(item["image_digest"]) for item in lineage["images"]
            }
            if set(manifest_by_digest) != lineage_digests:
                block(
                    "image_manifest_lineage",
                    "Image manifest does not cover dataset lineage exactly.",
                )
        declared_manifest_digest = metadata.get("image_manifest_sha256")
        actual_manifest_digest = hashlib.sha256(
            _canonical_json_bytes(raw_manifest)
        ).hexdigest()
        if declared_manifest_digest != actual_manifest_digest:
            block(
                "image_manifest_digest",
                "Image manifest SHA-256 is absent or inconsistent.",
            )
        raw_train = metadata.get("train_image_digests")
        raw_val = metadata.get("val_image_digests")
        if isinstance(raw_train, list) and isinstance(raw_val, list):
            train_digests = {str(value) for value in raw_train}
            val_digests = {str(value) for value in raw_val}
            for grouping_field in (
                "study_group_sha256",
                "original_sha256",
                "sha256",
            ):
                train_groups = {
                    manifest_by_digest[digest].get(grouping_field)
                    for digest in train_digests & set(manifest_by_digest)
                    if manifest_by_digest[digest].get(grouping_field) is not None
                }
                val_groups = {
                    manifest_by_digest[digest].get(grouping_field)
                    for digest in val_digests & set(manifest_by_digest)
                    if manifest_by_digest[digest].get(grouping_field) is not None
                }
                if train_groups & val_groups:
                    block(
                        "split_group_leakage",
                        "Train/validation partitions share a sealed study or "
                        "original-content group.",
                    )
                    break

    if metadata.get("allow_unverified_images") is not False:
        block(
            "unverified_training_images",
            "Publication requires verified checksums for every training image.",
        )
    if (
        metadata.get("allow_pseudo_labels") is not False
        or metadata.get("pseudo_label_annotations") != 0
    ):
        block(
            "pseudo_label_training",
            "Publication requires every training annotation to carry literal "
            "human approval with pseudo_label=false and review_required=false.",
        )

    attribution_bundle = None
    if not blockers and lineage is not None:
        attribution_bundle = {
            "schema_version": 1,
            "checkpoint_sha256": str(provenance.get("sha256", "")),
            "dataset": lineage["dataset_info"],
            "licenses": lineage["licenses"],
            "sources": lineage["sources"],
            "initializer": {
                "mode": initialization.get("mode"),
                "architecture": architecture,
            },
        }
    return {
        "schema_version": 1,
        "eligible_for_publication": not blockers,
        "blockers": blockers,
        "attribution_bundle": attribution_bundle,
    }


def train_detector(
    annotations: str | os.PathLike[str],
    image_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    architecture: str = DEFAULT_ARCHITECTURE,
    pretrained: bool = True,
    epochs: int = 20,
    batch_size: int = 2,
    learning_rate: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    validation_fraction: float = 0.2,
    seed: int = 1337,
    split_json: str | os.PathLike[str] | None = None,
    device: str = "auto",
    num_workers: int = 0,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    resume: str | os.PathLike[str] | None = None,
    allow_pseudo_labels: bool = False,
    allow_unverified_images: bool = False,
) -> dict[str, Any]:
    """Fine-tune a one-class TorchVision detector and save the best model."""
    from .ml_detector import _require_ml_runtime

    torch, torchvision = _require_ml_runtime()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise MLTrainingError(
            f"architecture must be one of: {', '.join(SUPPORTED_ARCHITECTURES)}"
        )
    output_path, last_path = _checkpoint_paths(output)
    if output_path.exists() and resume is None:
        raise MLTrainingError(
            f"Output checkpoint already exists: {output_path}. "
            "Choose a new path or use --resume."
        )
    if resume is not None:
        resume_path = Path(resume).expanduser().resolve()
        if resume_path not in {output_path, last_path}:
            raise MLTrainingError(
                "For safe checkpoint continuity, --resume must name either the "
                f"best checkpoint ({output_path.name}) or its last-epoch "
                f"checkpoint ({last_path.name})."
            )
    if int(epochs) < 1 or int(batch_size) < 1 or int(num_workers) < 0:
        raise MLTrainingError(
            "epochs and batch_size must be positive; num_workers cannot be negative."
        )
    if float(learning_rate) <= 0 or float(weight_decay) < 0:
        raise MLTrainingError(
            "learning_rate must be positive and weight_decay cannot be negative."
        )
    _seed_everything(int(seed))
    resolved_device = _resolve_training_device(device)
    trainable_backbone_layers = (
        TRANSFER_TRAINABLE_BACKBONE_LAYERS
        if pretrained
        else MAX_TRAINABLE_BACKBONE_LAYERS[architecture]
    )
    training_config = {
        "architecture": architecture,
        "optimizer": "SGD",
        "epochs_requested": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "momentum": float(momentum),
        "weight_decay": float(weight_decay),
        "validation_fraction": float(validation_fraction),
        "seed": int(seed),
        "num_workers": int(num_workers),
        "score_threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
        "pretrained_transfer_learning_requested": bool(pretrained),
        "trainable_backbone_layers": int(trainable_backbone_layers),
        "backbone_normalization": (
            "FrozenBatchNorm2d" if pretrained else "BatchNorm2d"
        ),
        "allow_pseudo_labels": bool(allow_pseudo_labels),
        "allow_unverified_images": bool(allow_unverified_images),
    }
    training_environment = _training_environment(
        torch, torchvision, resolved_device
    )

    coco_payload = _read_json_object(annotations)
    annotation_path = Path(annotations).expanduser().resolve()
    annotation_sha256 = _sha256_file(annotation_path)
    raw_annotations = coco_payload.get("annotations", [])
    pseudo_annotations = [
        annotation
        for annotation in raw_annotations
        if isinstance(annotation, Mapping) and _annotation_is_pseudo(annotation)
    ]
    if pseudo_annotations and not allow_pseudo_labels:
        raise MLTrainingError(
            f"COCO contains {len(pseudo_annotations)} pseudo-label annotation(s). "
            "Training refuses them by default; use the explicit pseudo-label "
            "override only for training, never validation."
        )
    pseudo_image_ids: set[int] = set()
    for annotation in pseudo_annotations:
        try:
            pseudo_image_ids.add(int(annotation["image_id"]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise MLTrainingError(
                "Pseudo-label annotation has an invalid image_id."
            ) from exc
    full_dataset = CocoBandDataset(
        coco_payload,
        image_root,
        training=False,
        include_pseudo_labels=bool(allow_pseudo_labels),
        require_checksums=not bool(allow_unverified_images),
    )
    if not full_dataset.image_ids:
        raise MLTrainingError("The COCO dataset contains no images.")
    annotation_count = sum(
        len(full_dataset.annotations_by_image.get(image_id, []))
        for image_id in full_dataset.image_ids
    )
    if annotation_count == 0:
        raise MLTrainingError("The COCO dataset contains no category-1 band boxes.")
    image_id_by_digest = {
        _lineage_digest_for_image_record(image_id, image): image_id
        for image_id, image in full_dataset.images_by_id.items()
    }
    if len(image_id_by_digest) != len(full_dataset.images_by_id):
        raise MLTrainingError("Dataset image lineage digests are not unique.")

    resume_state = resume_metadata = resume_payload = None
    resume_optimizer_state = None
    if resume is not None:
        resume_state, resume_metadata, resume_payload = _load_checkpoint(
            resume, device=torch.device("cpu")
        )
        if resume_payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise MLTrainingError(
                "Resume checkpoint predates the current licence-lineage "
                "checkpoint format. Old checkpoints remain inference-only."
            )
        checkpoint_architecture = str(
            resume_metadata.get("architecture", DEFAULT_ARCHITECTURE)
        )
        if checkpoint_architecture != architecture:
            raise MLTrainingError(
                f"Resume checkpoint uses {checkpoint_architecture}, not {architecture}."
            )
        previous_annotation_sha256 = str(
            resume_metadata.get("annotation_sha256", "")
        ).lower()
        if previous_annotation_sha256 != annotation_sha256.lower():
            raise MLTrainingError(
                "Resume checkpoint annotation SHA-256 differs from the current "
                "COCO dataset; start a new training run for changed annotations."
            )
        _validate_resume_metadata(
            resume_metadata,
            training_config,
            training_environment,
            dataset_license_lineage=full_dataset.license_lineage,
        )
        resume_optimizer_state = _validated_resume_optimizer_state(
            resume_payload
        )

    split_source: str | os.PathLike[str] | Mapping[str, Any] | None = split_json
    if split_source is None and resume_metadata:
        saved_train = resume_metadata.get("train_image_digests")
        saved_val = resume_metadata.get("val_image_digests")
        if not isinstance(saved_train, list) or not isinstance(saved_val, list):
            raise MLTrainingError(
                "Resume checkpoint lacks its public-safe split lineage."
            )
        try:
            split_source = {
                "train": [
                    image_id_by_digest[str(value)] for value in saved_train
                ],
                "val": [
                    image_id_by_digest[str(value)] for value in saved_val
                ],
            }
        except KeyError as exc:
            raise MLTrainingError(
                "Resume checkpoint split image lineage differs from the "
                "current dataset."
            ) from exc
    train_ids, val_ids = make_train_val_split(
        full_dataset.images_by_id,
        validation_fraction=validation_fraction,
        seed=seed,
        split=split_source,
        train_only_ids=pseudo_image_ids,
    )
    if resume_metadata:
        current_train_digests = {
            _lineage_digest_for_image_record(
                image_id, full_dataset.images_by_id[image_id]
            )
            for image_id in train_ids
        }
        current_val_digests = {
            _lineage_digest_for_image_record(
                image_id, full_dataset.images_by_id[image_id]
            )
            for image_id in val_ids
        }
        if current_train_digests != set(
            resume_metadata["train_image_digests"]
        ) or current_val_digests != set(
            resume_metadata["val_image_digests"]
        ):
            raise MLTrainingError(
                "Resume train/validation split differs from the checkpoint."
            )
    train_dataset = CocoBandDataset(
        full_dataset.coco,
        image_root,
        image_ids=train_ids,
        training=True,
        include_pseudo_labels=bool(allow_pseudo_labels),
        require_checksums=not bool(allow_unverified_images),
        known_checksums=full_dataset.image_checksums,
    )
    validation_dataset = CocoBandDataset(
        full_dataset.coco,
        image_root,
        image_ids=val_ids,
        training=False,
        include_pseudo_labels=False,
        require_checksums=not bool(allow_unverified_images),
        known_checksums=full_dataset.image_checksums,
    )
    if set(val_ids) & pseudo_image_ids:
        raise MLTrainingError("Pseudo-label images cannot be used for validation.")
    train_positive_boxes = sum(train_dataset.valid_box_counts.values())
    validation_positive_boxes = sum(validation_dataset.valid_box_counts.values())
    if train_positive_boxes < 1:
        raise MLTrainingError("The training split contains no usable positive boxes.")
    if validation_positive_boxes < 1:
        raise MLTrainingError(
            "The validation split contains no usable approved positive boxes."
        )

    loader_options = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "collate_fn": _collate_detection_batch,
        "worker_init_fn": _seed_worker if int(num_workers) else None,
        "pin_memory": resolved_device.type == "cuda",
    }
    validation_generator = torch.Generator()
    validation_generator.manual_seed(_epoch_seed(int(seed), -1))
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        generator=validation_generator,
        **loader_options,
    )

    if resume_state is None:
        model, initialization_provenance = build_model_with_provenance(
            architecture,
            pretrained=bool(pretrained),
            trainable_backbone_layers=trainable_backbone_layers,
            freeze_backbone_batch_norm=bool(pretrained),
        )
    else:
        model = build_model(
            architecture,
            pretrained=False,
            trainable_backbone_layers=trainable_backbone_layers,
            freeze_backbone_batch_norm=bool(pretrained),
        )
        initialization_provenance = _validated_json_copy(
            resume_metadata["initialization_provenance"],
            label="resume initialization provenance",
        )
    model.to(resolved_device)
    if resume_state is not None:
        try:
            model.load_state_dict(resume_state, strict=True)
        except Exception as exc:
            raise MLTrainingError(
                "Resume checkpoint weights do not match the selected architecture."
            ) from exc
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameter_count = sum(parameter.numel() for parameter in parameters)
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(learning_rate),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            _move_optimizer_to_device(optimizer, resolved_device)
        except (ValueError, RuntimeError, KeyError) as exc:
            raise MLTrainingError(
                "Resume optimizer state is incompatible with this training setup."
            ) from exc
    transfer_learning_used = (
        initialization_provenance.get("mode") == "torchvision_pretrained"
    )

    start_epoch = (
        int(resume_metadata.get("epoch", -1)) + 1 if resume_metadata else 0
    )
    previous_best = (
        resume_metadata.get("best_metrics", {}) if resume_metadata else {}
    )
    best_epoch = (
        int(resume_metadata.get("best_epoch", -1))
        if resume_metadata
        else -1
    )
    best_f1 = float(previous_best.get("f1", -1.0))
    best_iou = float(previous_best.get("mean_matched_iou", -1.0))
    history = list(resume_metadata.get("history", [])) if resume_metadata else []
    if start_epoch >= int(epochs):
        raise MLTrainingError(
            f"Resume checkpoint is already at epoch {start_epoch}; "
            f"--epochs must be greater than {start_epoch}."
        )

    lineage_by_digest = {
        str(record["image_digest"]): record
        for record in full_dataset.license_lineage["images"]
    }
    image_manifest: list[dict[str, Any]] = []
    for image_id in full_dataset.image_ids:
        image_record = full_dataset.images_by_id[image_id]
        blotgrid_metadata = image_record.get("blotgrid")
        if not isinstance(blotgrid_metadata, Mapping):
            blotgrid_metadata = {}
        original_sha256 = (
            str(blotgrid_metadata.get("original_sha256", "")).strip().lower()
            or None
        )
        image_digest = _lineage_digest_for_image_record(image_id, image_record)
        lineage_record = lineage_by_digest[image_digest]
        study_group = str(blotgrid_metadata.get("study_group", "")).strip()
        image_manifest.append(
            {
                "image_digest": image_digest,
                "sha256": full_dataset.image_checksums[image_id],
                "study_group_sha256": (
                    hashlib.sha256(
                        f"blotgrid-study-group-v1:{study_group}".encode("utf-8")
                    ).hexdigest()
                    if study_group
                    else None
                ),
                "original_sha256": original_sha256,
                "license_id": lineage_record["license_id"],
                "source_id": lineage_record["source_id"],
            }
        )
    manifest_bytes = json.dumps(
        image_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    image_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    split_path = output_path.with_name(output_path.name + ".split.json")
    manifest = split_manifest(
        train_ids, val_ids, full_dataset.images_by_id, seed=int(seed)
    )
    if split_json is not None:
        manifest["source_split_json"] = Path(split_json).name
    _atomic_json_write(split_path, manifest)

    best_metrics: dict[str, Any] = dict(previous_best)
    static_metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "publication_metadata_schema_version": PUBLICATION_METADATA_SCHEMA_VERSION,
        "architecture": architecture,
        "num_classes": 2,
        "categories": {"0": "background", "1": "band"},
        "epochs_requested": int(epochs),
        "selection_metric": "validation_f1_then_mean_matched_iou",
        "training_config": training_config,
        "training_environment": training_environment,
        "initialization_provenance": initialization_provenance,
        "pretrained_transfer_learning": transfer_learning_used,
        "trainable_backbone_layers": int(trainable_backbone_layers),
        "trainable_parameter_count": int(trainable_parameter_count),
        "seed": int(seed),
        "score_threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
        "train_image_digests": sorted(
            _lineage_digest_for_image_record(
                image_id, full_dataset.images_by_id[image_id]
            )
            for image_id in train_ids
        ),
        "val_image_digests": sorted(
            _lineage_digest_for_image_record(
                image_id, full_dataset.images_by_id[image_id]
            )
            for image_id in val_ids
        ),
        "annotation_sha256": annotation_sha256,
        "annotation_complete_images": len(full_dataset.image_ids),
        "dataset_license_lineage": full_dataset.license_lineage,
        "image_manifest": image_manifest,
        "image_manifest_sha256": image_manifest_sha256,
        "allow_unverified_images": bool(allow_unverified_images),
        "pseudo_label_annotations": len(pseudo_annotations),
        "allow_pseudo_labels": bool(allow_pseudo_labels),
        # Retain these top-level fields for compatibility with older
        # provenance readers; the complete record is above.
        "torch_version": training_environment["torch_version"],
        "torchvision_version": training_environment["torchvision_version"],
        "torchvision_git_revision": training_environment[
            "torchvision_git_revision"
        ],
        "device": training_environment["device"],
    }
    maximum_float = -1.7976931348623157e308
    reserved_history_record = {
        "epoch": 999999999,
        "train_loss": maximum_float,
        "images": 999999999,
        "true_positive": 999999999,
        "false_positive": 999999999,
        "false_negative": 999999999,
        "precision": maximum_float,
        "recall": maximum_float,
        "f1": maximum_float,
        "mean_matched_iou": maximum_float,
        "score_threshold": maximum_float,
        "iou_threshold": maximum_float,
    }
    _checkpoint_sidecar_preflight(
        {
            **static_metadata,
            "epoch": 999999999,
            "best_epoch": 999999999,
            "best_metrics": reserved_history_record,
            "current_metrics": reserved_history_record,
            "history": [reserved_history_record] * MAX_CHECKPOINT_HISTORY,
            "created_utc": "9999-12-31T23:59:59.999999+00:00",
        },
        include_optimizer=True,
    )
    for epoch in range(start_epoch, int(epochs)):
        epoch_seed = _epoch_seed(int(seed), int(epoch))
        _seed_everything(epoch_seed)
        train_generator = torch.Generator()
        train_generator.manual_seed(epoch_seed)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            generator=train_generator,
            **loader_options,
        )
        model.train()
        losses: list[float] = []
        for images, targets in train_loader:
            device_images = [image.to(resolved_device) for image in images]
            device_targets = [
                {
                    key: (
                        value.to(resolved_device)
                        if hasattr(value, "to")
                        else value
                    )
                    for key, value in target.items()
                }
                for target in targets
            ]
            loss_components = model(device_images, device_targets)
            loss = sum(loss_components.values())
            if not bool(torch.isfinite(loss)):
                raise MLTrainingError(
                    f"Non-finite training loss at epoch {epoch + 1}."
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = _evaluate_model(
            model,
            validation_loader,
            resolved_device,
            score_threshold=float(score_threshold),
            iou_threshold=float(iou_threshold),
        )
        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            **metrics,
        }
        history.append(epoch_record)
        # Keep sidecars compact for long runs without losing recent context.
        history = history[-MAX_CHECKPOINT_HISTORY:]
        is_best = (
            metrics["f1"] > best_f1 + 1e-12
            or (
                abs(metrics["f1"] - best_f1) <= 1e-12
                and metrics["mean_matched_iou"] > best_iou
            )
        )
        if is_best:
            best_f1 = float(metrics["f1"])
            best_iou = float(metrics["mean_matched_iou"])
            best_metrics = dict(metrics)
            best_epoch = int(epoch)

        metadata = {
            **static_metadata,
            "epoch": int(epoch),
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "current_metrics": metrics,
            "history": history,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        _save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            metadata=metadata,
        )
        if is_best:
            _save_checkpoint(
                output_path,
                model=model,
                optimizer=optimizer,
                metadata=metadata,
            )
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"loss={epoch_record['train_loss']:.4f} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} "
            f"iou={metrics['mean_matched_iou']:.3f}"
        )

    return {
        "best_checkpoint": str(output_path),
        "last_checkpoint": str(last_path),
        "split_json": str(split_path),
        "architecture": architecture,
        "device": str(resolved_device),
        "best_metrics": best_metrics,
        "epochs_completed": int(epochs - start_epoch),
        "train_images": len(train_ids),
        "validation_images": len(val_ids),
    }


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except (OSError, ValueError) as exc:
        raise MLTrainingError(f"Could not read image dimensions: {path}") from exc


def _project_image_value(container: Any, filename: str, default: Any) -> Any:
    if not isinstance(container, Mapping):
        return default
    if filename in container:
        return container[filename]
    normalized = str(filename).replace("\\", "/")
    basename = Path(normalized).name
    if normalized in container:
        return container[normalized]
    matches = [
        value
        for key, value in container.items()
        if Path(str(key).replace("\\", "/")).name == basename
    ]
    return matches[0] if len(matches) == 1 else default


def _verify_project_image_hash(
    image_path: Path,
    provenance: Any,
    *,
    filename: str,
    allow_unverified: bool,
) -> str:
    actual = _sha256_file(image_path)
    if not isinstance(provenance, Mapping) or not provenance.get("sha256"):
        if not allow_unverified:
            raise MLTrainingError(
                f"Project image {filename!r} has no saved SHA-256 provenance. "
                "Re-upload and save the project again, or use the explicit "
                "allow-unverified override for a trusted legacy project."
            )
        return actual
    expected = str(provenance["sha256"]).strip().lower()
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise MLTrainingError(
            f"Project image provenance for {filename!r} has an invalid SHA-256."
        )
    if actual.lower() != expected:
        raise MLTrainingError(
            f"Image provenance mismatch for {filename!r}: the selected source "
            "file is not the upload used to create these ROIs."
        )
    return actual


def _safe_relative_filename(filename: str) -> Path:
    normalized = str(filename).replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute() or ".." in candidate.parts:
        candidate = Path(candidate.name)
    if not candidate.name:
        raise MLTrainingError(f"Invalid project image filename: {filename!r}.")
    return candidate


def _write_working_image(
    source: Path,
    destination: Path,
    *,
    angle: float,
    force: bool,
) -> str:
    """Write the image whose coordinate system the reviewed ROIs use."""
    if destination.exists() and not force:
        raise MLTrainingError(
            f"Export image already exists: {destination}. Use --force to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if abs(angle) < 1e-9:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=destination.name + ".",
            suffix=destination.suffix + ".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return _sha256_file(destination)

    from . import alignment, image_io

    try:
        with Image.open(source) as original:
            image_format = original.format
        loaded = image_io.load_image(source, name=source.name)
        working = alignment.rotate_loaded(loaded, angle)
    except (OSError, ValueError) as exc:
        raise MLTrainingError(
            f"Could not recreate aligned working pixels for {source.name!r}."
        ) from exc
    if working.bit_depth <= 8:
        output_array = np.rint(
            np.clip(working.gray, 0.0, 255.0)
        ).astype(np.uint8)
    elif working.bit_depth <= 16:
        output_array = np.rint(
            np.clip(working.gray, 0.0, 65535.0)
        ).astype(np.uint16)
    elif image_format == "TIFF":
        output_array = np.asarray(working.gray, dtype=np.float32)
    else:
        # Pillow cannot write floating-point PNG/JPEG. Preserve the coordinate
        # system and monotonic contrast in a 16-bit representation.
        maximum = max(float(working.max_value), 1.0)
        output_array = np.rint(
            np.clip(working.gray / maximum, 0.0, 1.0) * 65535.0
        ).astype(np.uint16)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=destination.suffix + ".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(output_array).save(temporary, format=image_format)
        os.replace(temporary, destination)
    except (OSError, ValueError, KeyError) as exc:
        raise MLTrainingError(
            f"Could not write aligned training image: {destination}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(destination)


_PROJECT_LINEAGE_IMAGE_RESERVED_FIELDS = {
    "license",
    "mode",
    "bands_dark",
    "polarity_confidence",
    "detection_warnings",
    "replicate",
    "study_group",
    "original_sha256",
    "provenance_verified",
    "alignment_applied_degrees",
    "exported_working_sha256",
    "working_pixels_exported",
    "annotation_complete",
    "negative_image",
}


def _project_source_lineage(
    payload: Mapping[str, Any],
    layouts: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate optional project lineage without weakening legacy exports."""
    raw = payload.get("source_lineage")
    if raw is None or raw == {}:
        return None
    if not isinstance(raw, Mapping):
        raise MLTrainingError("Project source_lineage must be an object.")
    unexpected = set(raw) - {
        "schema_version",
        "info",
        "licenses",
        "sources",
        "images",
    }
    if unexpected:
        raise MLTrainingError(
            "Project source_lineage contains unsupported field(s): "
            + ", ".join(sorted(str(value) for value in unexpected))
            + "."
        )
    schema_version = raw.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != DATASET_LICENSE_LINEAGE_SCHEMA_VERSION
    ):
        raise MLTrainingError(
            "Project source_lineage schema_version is not supported."
        )

    raw_info = raw.get("info", {})
    if not isinstance(raw_info, Mapping):
        raise MLTrainingError("Project source_lineage info must be an object.")
    unsupported_info = set(raw_info) - {"title", "version", "source_dois"}
    if unsupported_info:
        raise MLTrainingError(
            "Project source_lineage info contains unsupported field(s): "
            + ", ".join(sorted(str(value) for value in unsupported_info))
            + "."
        )
    info = _validated_json_copy(raw_info, label="Project source_lineage info")
    source_dois = info.get("source_dois", [])
    if not isinstance(source_dois, list) or any(
        not isinstance(value, str) or not value.strip()
        for value in source_dois
    ):
        raise MLTrainingError(
            "Project source_lineage info.source_dois must be a list of "
            "non-empty strings."
        )
    if len(set(source_dois)) != len(source_dois):
        raise MLTrainingError(
            "Project source_lineage info.source_dois contains duplicates."
        )

    raw_licenses = raw.get("licenses", [])
    raw_sources = raw.get("sources", [])
    raw_images = raw.get("images")
    if not isinstance(raw_licenses, list):
        raise MLTrainingError("Project source_lineage licenses must be a list.")
    if not isinstance(raw_sources, list):
        raise MLTrainingError("Project source_lineage sources must be a list.")
    if not isinstance(raw_images, Mapping):
        raise MLTrainingError("Project source_lineage images must be an object.")
    licenses = _validated_json_copy(
        raw_licenses, label="Project source_lineage licenses"
    )
    sources = _validated_json_copy(
        raw_sources, label="Project source_lineage sources"
    )

    license_ids: set[int] = set()
    for index, record in enumerate(licenses):
        if not isinstance(record, dict) or type(record.get("id")) is not int:
            raise MLTrainingError(
                f"Project source_lineage licence {index} requires an integer id."
            )
        license_id = int(record["id"])
        if license_id in license_ids:
            raise MLTrainingError(
                f"Duplicate project source_lineage licence id: {license_id}."
            )
        license_ids.add(license_id)

    sources_by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(sources):
        if not isinstance(record, dict):
            raise MLTrainingError(
                f"Project source_lineage source {index} must be an object."
            )
        source_id = record.get("id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise MLTrainingError(
                f"Project source_lineage source {index} requires a non-empty id."
            )
        source_id = source_id.strip()
        if source_id in sources_by_id:
            raise MLTrainingError(
                f"Duplicate project source_lineage source id: {source_id!r}."
            )
        record["id"] = source_id
        license_id = record.get("license_id")
        if type(license_id) is not int or license_id not in license_ids:
            raise MLTrainingError(
                f"Project source_lineage source {source_id!r} must refer to a "
                "declared integer licence id."
            )
        sources_by_id[source_id] = record

    layout_names = {str(value) for value in layouts}
    if any(not isinstance(value, str) for value in raw_images):
        raise MLTrainingError(
            "Project source_lineage image keys must be project filenames."
        )
    lineage_names = set(raw_images)
    missing = sorted(layout_names - lineage_names)
    extra = sorted(lineage_names - layout_names)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(repr(value) for value in missing))
        if extra:
            details.append("unknown " + ", ".join(repr(value) for value in extra))
        raise MLTrainingError(
            "Project source_lineage image mapping does not exactly match "
            "project layouts (" + "; ".join(details) + ")."
        )

    images: dict[str, dict[str, Any]] = {}
    for filename in sorted(layout_names):
        raw_image = raw_images[filename]
        if not isinstance(raw_image, Mapping):
            raise MLTrainingError(
                f"Project source_lineage image {filename!r} must be an object."
            )
        image = _validated_json_copy(
            raw_image,
            label=f"Project source_lineage image {filename!r}",
        )
        collisions = set(image) & _PROJECT_LINEAGE_IMAGE_RESERVED_FIELDS
        if collisions:
            raise MLTrainingError(
                f"Project source_lineage image {filename!r} uses reserved "
                "field(s): "
                + ", ".join(sorted(collisions))
                + "."
            )
        source_id = image.get("source_id")
        if not isinstance(source_id, str) or source_id.strip() not in sources_by_id:
            raise MLTrainingError(
                f"Project source_lineage image {filename!r} must refer to a "
                "declared source_id."
            )
        source_id = source_id.strip()
        image["source_id"] = source_id
        source = sources_by_id[source_id]
        source_license_id = int(source["license_id"])
        image_license_id = image.get("license_id", source_license_id)
        if (
            type(image_license_id) is not int
            or image_license_id not in license_ids
            or image_license_id != source_license_id
        ):
            raise MLTrainingError(
                f"Project source_lineage image {filename!r} licence does not "
                f"match source {source_id!r}."
            )
        image["license_id"] = image_license_id
        for image_field, source_field in (
            ("source_doi", "doi"),
            ("source_url", "url"),
            ("required_attribution", "required_attribution"),
        ):
            inherited = source.get(source_field)
            direct = image.get(image_field)
            if direct is not None and inherited is not None and direct != inherited:
                raise MLTrainingError(
                    f"Project source_lineage image {filename!r} {image_field} "
                    f"conflicts with source {source_id!r}."
                )
            if direct is None and inherited is not None:
                image[image_field] = inherited
        images[filename] = image

    return {
        "schema_version": DATASET_LICENSE_LINEAGE_SCHEMA_VERSION,
        "info": info,
        "licenses": licenses,
        "sources": sources,
        "images": images,
    }


def export_project_to_coco(
    project: str | os.PathLike[str] | Mapping[str, Any],
    image_root: str | os.PathLike[str],
    output: str | os.PathLike[str] | None = None,
    *,
    include_inactive: bool = False,
    include_fallback: bool = False,
    include_unreviewed: bool = False,
    export_image_root: str | os.PathLike[str] | None = None,
    allow_unverified: bool = False,
    allow_incomplete_annotations: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Convert reviewed BlotGrid project ROIs to category-1 COCO boxes."""
    payload = _read_json_object(project)
    layouts = payload.get("layouts")
    if not isinstance(layouts, dict):
        raise MLTrainingError("BlotGrid project JSON has no 'layouts' object.")
    source_lineage = _project_source_lineage(payload, layouts)
    root = Path(image_root).expanduser().resolve()
    output_path = Path(output).expanduser().resolve() if output is not None else None
    if output_path is not None and output_path.exists() and not force:
        raise MLTrainingError(
            f"Output already exists: {output_path}. Use --force to replace it."
        )
    alignments = payload.get("alignment_applied", {})
    image_provenance = payload.get("image_provenance", {})
    study_groups = payload.get("study_group", {})
    alignment_angles: dict[str, float] = {}
    for filename in layouts:
        raw_angle = _project_image_value(alignments, str(filename), 0.0)
        try:
            angle = float(raw_angle)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MLTrainingError(
                f"Invalid saved alignment angle for {filename!r}."
            ) from exc
        if not math.isfinite(angle) or abs(angle) > 15.0:
            raise MLTrainingError(
                f"Saved alignment angle for {filename!r} is outside ±15 degrees."
            )
        alignment_angles[str(filename)] = angle
    has_alignment = any(abs(value) >= 1e-9 for value in alignment_angles.values())
    if export_image_root is not None:
        exported_root = Path(export_image_root).expanduser().resolve()
    elif has_alignment:
        if output_path is None:
            raise MLTrainingError(
                "Aligned project ROIs require output or export_image_root so the "
                "registered working images can be written."
            )
        exported_root = output_path.with_name(output_path.stem + "_images")
    else:
        exported_root = None
    if exported_root is not None:
        if exported_root == root:
            raise MLTrainingError(
                "The working-image export directory must differ from the original "
                "image directory; source uploads are never overwritten."
            )
        destinations: dict[Path, str] = {}
        for raw_filename in layouts:
            filename = str(raw_filename)
            destination = (exported_root / _safe_relative_filename(filename)).resolve()
            source = _safe_image_path(root, filename)
            if destination == source:
                raise MLTrainingError(
                    f"Working-image export would overwrite source image {filename!r}."
                )
            if destination in destinations:
                raise MLTrainingError(
                    f"Project filenames {destinations[destination]!r} and "
                    f"{filename!r} map to the same export path."
                )
            if destination.exists() and not force:
                raise MLTrainingError(
                    f"Export image already exists: {destination}. "
                    "Use --force to replace it."
                )
            destinations[destination] = filename
    source_name = (
        Path(project).name
        if isinstance(project, (str, os.PathLike))
        else "in-memory-project.json"
    )
    coco: dict[str, Any] = {
        "info": {
            "description": "Reviewed BlotGrid Western-blot band boxes",
            "version": "1.0",
            "date_created": datetime.now(timezone.utc).isoformat(),
            "source_project": source_name,
            "source_project_saved_at": payload.get("_saved_at"),
        },
        "licenses": (
            source_lineage["licenses"] if source_lineage is not None else []
        ),
        "categories": [
            {"id": 1, "name": "band", "supercategory": "western_blot"}
        ],
        "images": [],
        "annotations": [],
        "blotgrid": {
            "targets": _safe_metadata(payload.get("targets", [])),
            "lanes": _safe_metadata(payload.get("lanes", [])),
            "control": _safe_metadata(payload.get("control", "")),
            "filters": {
                "include_inactive": bool(include_inactive),
                "include_fallback": bool(include_fallback),
                "include_unreviewed": bool(include_unreviewed),
                "allow_unverified": bool(allow_unverified),
                "allow_incomplete_annotations": bool(
                    allow_incomplete_annotations
                ),
            },
            "annotation_policy": (
                "approved_and_pseudo_labels"
                if include_unreviewed
                else "human_approved_only"
            ),
            "pseudo_labels_included": bool(include_unreviewed),
            "image_root": (
                exported_root.name if exported_root is not None else root.name
            ),
            "source_metadata": {
                "polarity_mode": _safe_metadata(payload.get("polarity_mode")),
                "detection_sensitivity": _safe_metadata(
                    payload.get("detection_sensitivity")
                ),
                "measurement_mode": _safe_metadata(
                    payload.get("measurement_mode")
                ),
            },
            "warnings": [],
        },
    }
    if source_lineage is not None:
        for field in ("title", "version", "source_dois"):
            if field in source_lineage["info"]:
                coco["info"][field] = source_lineage["info"][field]
        coco["blotgrid"]["license_lineage_schema_version"] = (
            DATASET_LICENSE_LINEAGE_SCHEMA_VERSION
        )
        coco["blotgrid"]["sources"] = source_lineage["sources"]
    annotation_id = 1
    skipped_inactive = skipped_fallback = skipped_unreviewed = 0
    skipped_invalid = review_count = pseudo_label_count = 0
    dropped_unattested_empty = attested_negative_count = 0
    incomplete_positive_count = 0
    for image_id, filename in enumerate(sorted(layouts, key=str), start=1):
        layout = layouts[filename]
        if not isinstance(layout, dict):
            coco["blotgrid"]["warnings"].append(
                f"Layout for {filename!r} was not an object and was skipped."
            )
            continue
        image_path = _safe_image_path(root, str(filename))
        provenance = _project_image_value(
            image_provenance, str(filename), {}
        )
        original_sha256 = _verify_project_image_hash(
            image_path,
            provenance,
            filename=str(filename),
            allow_unverified=bool(allow_unverified),
        )
        width, height = _image_dimensions(image_path)
        preview_rois = layout.get("rois", [])
        preview_has_positive = False
        if isinstance(preview_rois, list):
            for preview_roi in preview_rois:
                if not isinstance(preview_roi, dict):
                    continue
                if not bool(preview_roi.get("use", True)) and not include_inactive:
                    continue
                if (
                    str(preview_roi.get("source", "unknown")) == "fallback"
                    and not include_fallback
                ):
                    continue
                if (
                    preview_roi.get("annotation_approved") is not True
                    and not include_unreviewed
                ):
                    continue
                preview_box = _xywh_to_xyxy(
                    [
                        preview_roi.get("x"),
                        preview_roi.get("y"),
                        preview_roi.get("w"),
                        preview_roi.get("h"),
                    ],
                    width=width,
                    height=height,
                )
                if preview_box is not None:
                    preview_has_positive = True
                    break
        if (
            preview_has_positive
            and layout.get("annotation_complete") is not True
            and not allow_incomplete_annotations
        ):
            raise MLTrainingError(
                f"Layout for {filename!r} has positive band annotations but "
                "whole-image annotation_complete is not true. Complete and "
                "attest the image review before export, or use the explicit "
                "allow-incomplete override for trusted provisional data."
            )
        if (
            preview_has_positive
            and layout.get("annotation_complete") is True
            and layout.get("negative_image") is True
        ):
            raise MLTrainingError(
                f"Layout for {filename!r} is marked as a complete negative but "
                "also exports band annotations."
            )
        angle = alignment_angles[str(filename)]
        exported_sha256 = original_sha256
        if exported_root is not None:
            relative_name = _safe_relative_filename(str(filename))
            exported_path = exported_root / relative_name
            exported_sha256 = _write_working_image(
                image_path,
                exported_path,
                angle=angle,
                force=force,
            )
        lineage_image = (
            dict(source_lineage["images"][str(filename)])
            if source_lineage is not None
            else {}
        )
        image_license_id = lineage_image.pop("license_id", None)
        image_metadata = {
            **lineage_image,
            "mode": _safe_metadata(layout.get("mode")),
            "bands_dark": _safe_metadata(layout.get("bands_dark")),
            "polarity_confidence": _safe_metadata(
                layout.get("polarity_confidence")
            ),
            "detection_warnings": _safe_metadata(
                layout.get("detection_warnings", [])
            ),
            "replicate": _safe_metadata(
                payload.get("replicate", {}).get(filename)
                if isinstance(payload.get("replicate"), dict)
                else None
            ),
            "study_group": _safe_metadata(
                _project_image_value(study_groups, str(filename), None)
            ),
            "original_sha256": original_sha256,
            "provenance_verified": bool(
                isinstance(provenance, Mapping)
                and provenance.get("sha256")
            ),
            "alignment_applied_degrees": angle,
            "exported_working_sha256": exported_sha256,
            "working_pixels_exported": exported_root is not None,
            "annotation_complete": layout.get("annotation_complete") is True,
            "negative_image": layout.get("negative_image") is True,
        }
        image_record = {
            "id": image_id,
            "file_name": str(filename),
            "width": width,
            "height": height,
            "blotgrid": image_metadata,
        }
        if image_license_id is not None:
            image_record["license"] = image_license_id
        coco["images"].append(image_record)
        annotation_start = len(coco["annotations"])
        rois = layout.get("rois", [])
        if not isinstance(rois, list):
            coco["blotgrid"]["warnings"].append(
                f"Layout for {filename!r} has no reviewed ROI list."
            )
            coco["images"].pop()
            dropped_unattested_empty += 1
            continue
        if not rois:
            coco["blotgrid"]["warnings"].append(
                f"Layout for {filename!r} contains no ROIs."
            )
        for roi in rois:
            if not isinstance(roi, dict):
                skipped_invalid += 1
                continue
            active = bool(roi.get("use", True))
            source = str(roi.get("source", "unknown"))
            annotation_approved = roi.get("annotation_approved") is True
            if not active and not include_inactive:
                skipped_inactive += 1
                continue
            if source == "fallback" and not include_fallback:
                skipped_fallback += 1
                continue
            if not annotation_approved and not include_unreviewed:
                skipped_unreviewed += 1
                continue
            converted = _xywh_to_xyxy(
                [
                    roi.get("x"),
                    roi.get("y"),
                    roi.get("w"),
                    roi.get("h"),
                ],
                width=width,
                height=height,
            )
            if converted is None:
                skipped_invalid += 1
                continue
            x0, y0, x1, y1 = converted
            box_width, box_height = x1 - x0, y1 - y0
            review_required = bool(roi.get("review_required", False))
            review_count += int(review_required)
            is_pseudo_label = not annotation_approved
            pseudo_label_count += int(is_pseudo_label)
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x0, y0, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "segmentation": [],
                    "attributes": {
                        "roi_id": _safe_metadata(roi.get("roi_id")),
                        "target": _safe_metadata(roi.get("target")),
                        "lane": _safe_metadata(roi.get("lane")),
                        "source": source,
                        "status": _safe_metadata(roi.get("status")),
                        "confidence": _safe_metadata(roi.get("confidence")),
                        "review_required": review_required,
                        "active": active,
                        "annotation_approved": annotation_approved,
                        "pseudo_label": is_pseudo_label,
                    },
                }
            )
            annotation_id += 1
        retained_count = len(coco["annotations"]) - annotation_start
        is_attested_negative = bool(
            layout.get("annotation_complete") is True
            and layout.get("negative_image") is True
        )
        if retained_count and is_attested_negative:
            raise MLTrainingError(
                f"Layout for {filename!r} is marked as a complete negative but "
                "also exports band annotations."
            )
        if (
            retained_count
            and layout.get("annotation_complete") is not True
        ):
            if not allow_incomplete_annotations:
                raise MLTrainingError(
                    f"Layout for {filename!r} has positive band annotations but "
                    "whole-image annotation_complete is not true. Complete and "
                    "attest the image review before export, or use the explicit "
                    "allow-incomplete override for trusted provisional data."
                )
            incomplete_positive_count += 1
            coco["blotgrid"]["warnings"].append(
                f"Image {filename!r} contains positive annotations without a "
                "complete whole-image review attestation."
            )
        if retained_count == 0:
            if is_attested_negative:
                attested_negative_count += 1
            else:
                coco["images"].pop()
                dropped_unattested_empty += 1
                coco["blotgrid"]["warnings"].append(
                    f"Image {filename!r} was dropped because no retained band "
                    "annotations remain and it lacks a complete-negative attestation."
                )

    coco["blotgrid"]["export_summary"] = {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "skipped_inactive": skipped_inactive,
        "skipped_fallback": skipped_fallback,
        "skipped_unreviewed": skipped_unreviewed,
        "skipped_invalid": skipped_invalid,
        "included_review_required": review_count,
        "included_pseudo_labels": pseudo_label_count,
        "attested_negative_images": attested_negative_count,
        "dropped_unattested_empty_images": dropped_unattested_empty,
        "incomplete_positive_images": incomplete_positive_count,
    }
    if review_count:
        coco["blotgrid"]["warnings"].append(
            f"{review_count} included ROI(s) still carry review_required=true; "
            "confirm these boxes before training."
        )

    if pseudo_label_count:
        coco["blotgrid"]["warnings"].append(
            f"{pseudo_label_count} unapproved ROI(s) were included as pseudo-labels. "
            "Do not treat pseudo-label validation metrics as independent evidence."
        )

    if source_lineage is not None:
        validated_lineage = _extract_dataset_license_lineage(coco)
        lineage_summary = validated_lineage["summary"]
        if (
            lineage_summary["images_without_declared_license"]
            or lineage_summary["images_without_declared_source"]
        ):
            raise MLTrainingError(
                "Project source_lineage did not resolve every exported image to "
                "a declared source and licence."
            )

    if output_path is not None:
        _atomic_json_write(output_path, coco)
    return coco
