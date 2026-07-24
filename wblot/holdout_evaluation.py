"""Fail-closed helpers for sealed Western-blot detector evaluation.

The module deliberately contains no optional ML imports. Detector inference is
orchestrated by ``scripts/evaluate_detector.py``; validation, matching and
native-pixel downstream comparisons remain usable in BlotGrid's base runtime.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from . import densitometry, image_io
from .ml_training import PUBLICATION_METADATA_SCHEMA_VERSION, box_iou_matrix


METHODS = ("classical", "gelgenie", "local")
PHASES = ("direct", "pre_review", "post_review")
STRATA_DIMENSIONS = (
    "scanner",
    "exposure",
    "target",
    "background",
    "acquisition_style",
)
ARTIFACT_CATEGORIES = {
    2: "border",
    3: "text",
    4: "smear",
    5: "blank_cell",
}


class EvaluationError(ValueError):
    """Unsafe or incomplete holdout input, review, or metric request."""


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not read JSON object: {source}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"JSON input must contain an object: {source}")
    return payload


def _json_bytes(payload: Any, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2 if pretty else None,
            sort_keys=not pretty,
            separators=None if pretty else (",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_json(path: str | os.PathLike[str], payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(payload, pretty=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(
    path: str | os.PathLike[str],
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in materialized:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(str(key))
        fieldnames = ordered or ["empty"]
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in materialized:
                serializable = {}
                for key in fieldnames:
                    value = row.get(key)
                    if isinstance(value, (dict, list, tuple)):
                        value = json.dumps(
                            value,
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                    if isinstance(value, float) and not math.isfinite(value):
                        value = ""
                    serializable[key] = value
                writer.writerow(serializable)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_image_path(root: Path, filename: str) -> Path:
    root = root.expanduser().resolve()
    normalized = str(filename).replace("\\", "/")
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise EvaluationError(f"Image escapes the declared root: {filename!r}") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise EvaluationError(f"Holdout image is missing or unsafe: {filename!r}")
    return candidate


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"Could not decode holdout image: {path.name}") from exc


def _xywh_to_xyxy(
    bbox: Any,
    *,
    width: int,
    height: int,
    label: str,
) -> list[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise EvaluationError(f"{label} requires a four-value COCO bbox.")
    try:
        x, y, w, h = [float(value) for value in bbox]
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError(f"{label} bbox is not numeric.") from exc
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        raise EvaluationError(f"{label} bbox contains a non-finite value.")
    if w <= 0 or h <= 0:
        raise EvaluationError(f"{label} bbox must have positive dimensions.")
    if w < 1.0 or h < 1.0:
        raise EvaluationError(f"{label} bbox dimensions must be at least one pixel.")
    if x < 0.0 or y < 0.0 or x + w > width or y + h > height:
        raise EvaluationError(f"{label} bbox is outside the image bounds.")
    return [x, y, x + w, y + h]


def _string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise EvaluationError(f"{label} must be a non-empty list.")
    result = [str(item).strip() for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise EvaluationError(f"{label} values must be non-empty and unique.")
    return result


def _is_sha256(value: Any) -> bool:
    digest = str(value).strip().lower()
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _canonical_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError("Binding payload is not canonical JSON.") from exc
    return hashlib.sha256(encoded).hexdigest()


def proposal_images_sha256(images: Mapping[str, Any]) -> str:
    """Return the canonical digest used to bind one method's proposals."""
    if not isinstance(images, Mapping):
        raise EvaluationError("Proposal images must be an object.")
    return _canonical_sha256(images)


def review_task_binding_sha256(
    *,
    run_nonce: str,
    run_binding_sha256: str,
    protocol_id: str,
    method: str,
    task_id: str,
    proposal_sha256: str,
) -> str:
    """Bind a blinded review task to one evaluator run and proposal payload."""
    return _canonical_sha256(
        {
            "schema_version": 1,
            "run_nonce": run_nonce,
            "run_binding_sha256": run_binding_sha256,
            "task_id": task_id,
            "protocol_id": protocol_id,
            "method": method,
            "proposal_sha256": proposal_sha256,
        }
    )


def _validated_digest_set(value: Any, *, label: str) -> set[str]:
    if isinstance(value, Mapping):
        raw_values = list(value.values())
    elif isinstance(value, list):
        raw_values = list(value)
    else:
        raise EvaluationError(f"{label} must be a non-empty list or object.")
    if not raw_values:
        raise EvaluationError(f"{label} must be non-empty.")
    digests = [str(item).strip().lower() for item in raw_values]
    if any(not _is_sha256(item) for item in digests):
        raise EvaluationError(f"{label} contains an invalid SHA-256 digest.")
    if len(digests) != len(set(digests)):
        raise EvaluationError(f"{label} contains duplicate SHA-256 digests.")
    return set(digests)


def _development_inventory(
    ledger: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    if ledger.get("schema_version") != 1:
        raise EvaluationError("Development ledger schema_version must be 1.")
    if ledger.get("complete") is not True:
        raise EvaluationError("Development ledger requires complete=true.")
    if ledger.get("completeness_attested") is not True:
        raise EvaluationError(
            "Development ledger requires completeness_attested=true."
        )
    declared_groups = set(
        _string_list(
            ledger.get("study_groups"),
            label="development ledger study_groups",
        )
    )
    declared_hashes = _validated_digest_set(
        ledger.get("image_sha256"),
        label="development ledger image_sha256",
    )
    model_selection = ledger.get("model_selection")
    inventory_groups: set[str] = set()
    inventory_hashes: set[str] = set()
    if isinstance(model_selection, Mapping):
        inventory_groups = set(
            _string_list(
                model_selection.get("study_groups"),
                label="model_selection study_groups",
            )
        )
        inventory_hashes = _validated_digest_set(
            model_selection.get("image_sha256"),
            label="model_selection image_sha256",
        )
    elif isinstance(model_selection, list) and model_selection:
        for index, record in enumerate(model_selection):
            if not isinstance(record, Mapping):
                raise EvaluationError(
                    f"model_selection record {index} must be an object."
                )
            group = str(record.get("study_group", "")).strip()
            digest = str(record.get("image_sha256", "")).strip().lower()
            if not group or not _is_sha256(digest):
                raise EvaluationError(
                    f"model_selection record {index} is incomplete or invalid."
                )
            if digest in inventory_hashes:
                raise EvaluationError(
                    "model_selection contains duplicate image SHA-256 digests."
                )
            inventory_groups.add(group)
            inventory_hashes.add(digest)
    else:
        raise EvaluationError(
            "Development ledger requires a non-empty model_selection inventory."
        )
    if (
        inventory_groups != declared_groups
        or inventory_hashes != declared_hashes
    ):
        raise EvaluationError(
            "Development ledger model_selection inventory must exactly match "
            "its declared study_groups and image_sha256 digests."
        )
    return declared_groups, declared_hashes


def _checkpoint_metadata(provenance: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(provenance, Mapping) or not provenance:
        raise EvaluationError("Local checkpoint provenance is required.")
    if provenance.get("metadata_verified") is not True:
        raise EvaluationError(
            "The local checkpoint needs a SHA-verified metadata sidecar."
        )
    if provenance.get("embedded_metadata_verified") is not True:
        raise EvaluationError(
            "The local checkpoint sidecar must exactly match restricted-load "
            "embedded metadata."
        )
    if provenance.get("container_verified") is not True:
        raise EvaluationError(
            "The local checkpoint payload must match the generated model and "
            "optimizer container contract."
        )
    metadata = provenance.get("metadata")
    if not isinstance(metadata, Mapping) or not metadata:
        raise EvaluationError(
            "The local checkpoint needs non-empty verified metadata."
        )
    if metadata.get("format_version") != 2:
        raise EvaluationError("Checkpoint format_version must be 2.")
    if (
        metadata.get("publication_metadata_schema_version")
        != PUBLICATION_METADATA_SCHEMA_VERSION
    ):
        raise EvaluationError(
            "Checkpoint metadata requires publication schema_version "
            f"{PUBLICATION_METADATA_SCHEMA_VERSION}."
        )
    return metadata


def validate_holdout_inputs(
    coco: Mapping[str, Any],
    annotations_path: str | os.PathLike[str],
    image_root: str | os.PathLike[str],
    seal: Mapping[str, Any],
    ledger: Mapping[str, Any],
    checkpoint_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize a sealed, fully annotated holdout cohort."""
    if seal.get("schema_version") != 1:
        raise EvaluationError("Holdout seal schema_version must be 1.")
    if not str(seal.get("sealed_utc", "")).strip():
        raise EvaluationError("Holdout seal requires a non-empty sealed_utc.")
    annotations_sha256 = sha256_file(annotations_path)
    if str(seal.get("annotations_sha256", "")).lower() != annotations_sha256:
        raise EvaluationError("Holdout annotation SHA-256 does not match its seal.")
    development_groups, development_hashes = _development_inventory(ledger)
    checkpoint_metadata = _checkpoint_metadata(checkpoint_provenance)

    raw_categories = coco.get("categories")
    if not isinstance(raw_categories, list):
        raise EvaluationError("COCO categories must be a list.")
    category_names: dict[int, str] = {}
    for raw in raw_categories:
        if not isinstance(raw, Mapping):
            continue
        try:
            category_names[int(raw["id"])] = str(raw["name"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    if category_names.get(1) != "band":
        raise EvaluationError("COCO category 1 must be named 'band'.")
    for category_id, name in ARTIFACT_CATEGORIES.items():
        if category_names.get(category_id) != name:
            raise EvaluationError(
                f"COCO category {category_id} must be named {name!r}."
            )

    raw_images = coco.get("images")
    raw_annotations = coco.get("annotations")
    if not isinstance(raw_images, list) or not isinstance(raw_annotations, list):
        raise EvaluationError("COCO images and annotations must be lists.")
    root = Path(image_root).expanduser().resolve()
    seal_hashes = seal.get("image_sha256")
    if not isinstance(seal_hashes, Mapping):
        raise EvaluationError("Holdout seal requires an image_sha256 object.")
    seal_groups = set(
        _string_list(seal.get("study_groups"), label="seal study_groups")
    )

    images: dict[int, dict[str, Any]] = {}
    filenames: set[str] = set()
    cohort_groups: set[str] = set()
    cohort_hashes: set[str] = set()
    for index, raw in enumerate(raw_images):
        if not isinstance(raw, Mapping):
            raise EvaluationError(f"COCO image {index} must be an object.")
        try:
            image_id = int(raw["id"])
            filename = str(raw["file_name"])
            declared_width = int(raw["width"])
            declared_height = int(raw["height"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise EvaluationError(
                f"COCO image {index} has invalid identity or dimensions."
            ) from exc
        if (
            not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or Path(filename).name != filename
        ):
            raise EvaluationError(
                "Holdout filenames must be flat basenames so blinded projects "
                "can be restored with exact Streamlit uploads."
            )
        if image_id in images or filename in filenames:
            raise EvaluationError("Holdout image ids and filenames must be unique.")
        path = _safe_image_path(root, filename)
        if _image_dimensions(path) != (declared_width, declared_height):
            raise EvaluationError(
                f"COCO dimensions do not match holdout image {filename!r}."
            )
        metadata = raw.get("blotgrid")
        if not isinstance(metadata, Mapping):
            raise EvaluationError(f"Image {filename!r} lacks BlotGrid metadata.")
        if metadata.get("annotation_complete") is not True:
            raise EvaluationError(
                f"Image {filename!r} lacks whole-image annotation attestation."
            )
        study_group = str(metadata.get("study_group", "")).strip()
        if not study_group:
            raise EvaluationError(f"Image {filename!r} has no study_group.")
        targets = _string_list(
            metadata.get("targets"), label=f"{filename} targets"
        )
        lanes = _string_list(metadata.get("lanes"), label=f"{filename} lanes")
        lane_hints = metadata.get("lane_centers_hint")
        target_hints = metadata.get("target_centers_hint")
        if not isinstance(lane_hints, list) or len(lane_hints) != len(lanes):
            raise EvaluationError(f"Image {filename!r} has invalid lane hints.")
        if not isinstance(target_hints, list) or len(target_hints) != len(targets):
            raise EvaluationError(f"Image {filename!r} has invalid target hints.")
        try:
            lane_hints = [int(value) for value in lane_hints]
            target_hints = [int(value) for value in target_hints]
        except (TypeError, ValueError, OverflowError) as exc:
            raise EvaluationError(
                f"Image {filename!r} hints must be integer coordinates."
            ) from exc
        if metadata.get("bands_are_dark") not in {True, False}:
            raise EvaluationError(
                f"Image {filename!r} requires a literal bands_are_dark boolean."
            )
        control_target = str(metadata.get("control_target", "")).strip()
        reference_lane = str(metadata.get("reference_lane", "")).strip()
        if control_target not in targets or reference_lane not in lanes:
            raise EvaluationError(
                f"Image {filename!r} has invalid control_target/reference_lane."
            )
        roi_size = metadata.get("review_roi_size")
        if not isinstance(roi_size, Mapping):
            raise EvaluationError(f"Image {filename!r} lacks review_roi_size.")
        try:
            review_w, review_h = int(roi_size["w"]), int(roi_size["h"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise EvaluationError(
                f"Image {filename!r} has invalid review_roi_size."
            ) from exc
        if review_w < 1 or review_h < 1:
            raise EvaluationError(
                f"Image {filename!r} review_roi_size must be positive."
            )
        strata = metadata.get("strata")
        if not isinstance(strata, Mapping):
            raise EvaluationError(f"Image {filename!r} lacks evaluation strata.")
        normalized_strata = {}
        for dimension in STRATA_DIMENSIONS:
            if dimension == "target":
                continue
            value = str(strata.get(dimension, "")).strip()
            if not value:
                raise EvaluationError(
                    f"Image {filename!r} lacks stratum {dimension!r}."
                )
            normalized_strata[dimension] = value
        actual_sha256 = sha256_file(path)
        expected = str(
            seal_hashes.get(str(image_id), seal_hashes.get(filename, ""))
        ).lower()
        declared = str(metadata.get("original_sha256", "")).lower()
        if expected != actual_sha256 or declared != actual_sha256:
            raise EvaluationError(
                f"Image SHA-256 does not match seal/metadata for {filename!r}."
            )
        if actual_sha256 in cohort_hashes:
            raise EvaluationError(
                "Holdout images must not contain duplicate image bytes."
            )
        cohort_groups.add(study_group)
        cohort_hashes.add(actual_sha256)
        filenames.add(filename)
        images[image_id] = {
            "id": image_id,
            "file_name": filename,
            "path": str(path),
            "sha256": actual_sha256,
            "width": declared_width,
            "height": declared_height,
            "study_group": study_group,
            "targets": targets,
            "lanes": lanes,
            "lane_centers_hint": lane_hints,
            "target_centers_hint": target_hints,
            "bands_are_dark": bool(metadata["bands_are_dark"]),
            "control_target": control_target,
            "reference_lane": reference_lane,
            "review_roi_size": {"w": review_w, "h": review_h},
            "strata": normalized_strata,
        }
    if cohort_groups != seal_groups:
        raise EvaluationError("Sealed study_groups do not exactly match the cohort.")
    if set(str(key) for key in seal_hashes) not in (
        {str(image_id) for image_id in images},
        filenames,
    ):
        raise EvaluationError("Seal image_sha256 keys do not exactly match images.")
    first_image = images[min(images)]
    for image in images.values():
        for field in ("targets", "lanes", "control_target", "reference_lane"):
            if image[field] != first_image[field]:
                raise EvaluationError(
                    "All images in one blinded review run must share targets, "
                    f"lanes, control target and reference lane ({field} differs)."
                )

    overlap_groups = cohort_groups & development_groups
    overlap_hashes = cohort_hashes & development_hashes
    if overlap_groups or overlap_hashes:
        raise EvaluationError(
            "Sealed holdout overlaps the declared development ledger."
        )
    checkpoint_manifest = checkpoint_metadata.get("image_manifest")
    if not isinstance(checkpoint_manifest, list) or not checkpoint_manifest:
        raise EvaluationError(
            "Checkpoint image_manifest must be a non-empty list."
        )
    allowed_manifest_fields = {
        "image_digest",
        "sha256",
        "study_group_sha256",
        "original_sha256",
        "license_id",
        "source_id",
    }
    checkpoint_groups: set[str] = set()
    checkpoint_hashes: set[str] = set()
    checkpoint_image_digests: set[str] = set()
    for index, record in enumerate(checkpoint_manifest):
        if not isinstance(record, Mapping):
            raise EvaluationError(
                f"Checkpoint image_manifest record {index} must be an object."
            )
        unexpected = set(record) - allowed_manifest_fields
        if unexpected:
            raise EvaluationError(
                "Checkpoint image_manifest contains non-public metadata fields."
            )
        image_digest = str(record.get("image_digest", "")).strip().lower()
        digest = str(record.get("sha256", "")).strip().lower()
        group_digest = (
            str(record.get("study_group_sha256", "")).strip().lower()
        )
        if not all(
            _is_sha256(value)
            for value in (image_digest, digest, group_digest)
        ):
            raise EvaluationError(
                f"Checkpoint image_manifest record {index} has invalid digests."
            )
        if image_digest in checkpoint_image_digests or digest in checkpoint_hashes:
            raise EvaluationError(
                "Checkpoint image_manifest contains duplicate image ancestry."
            )
        checkpoint_image_digests.add(image_digest)
        checkpoint_hashes.add(digest)
        checkpoint_groups.add(group_digest)
        original_value = record.get("original_sha256")
        if original_value is not None:
            original = str(original_value).strip().lower()
            if not _is_sha256(original):
                raise EvaluationError(
                    f"Checkpoint image_manifest record {index} has an invalid "
                    "original_sha256."
                )
            checkpoint_hashes.add(original)
        license_id = record.get("license_id")
        if license_id is not None and (
            isinstance(license_id, bool) or not isinstance(license_id, int)
        ):
            raise EvaluationError(
                f"Checkpoint image_manifest record {index} has invalid license_id."
            )
        source_id = record.get("source_id")
        if source_id is not None and (
            not isinstance(source_id, str) or not source_id.strip()
        ):
            raise EvaluationError(
                f"Checkpoint image_manifest record {index} has invalid source_id."
            )
    declared_manifest_sha256 = checkpoint_metadata.get("image_manifest_sha256")
    if declared_manifest_sha256 is not None and (
        not _is_sha256(declared_manifest_sha256)
        or str(declared_manifest_sha256).lower()
        != _canonical_sha256(checkpoint_manifest)
    ):
        raise EvaluationError("Checkpoint image_manifest SHA-256 is inconsistent.")
    cohort_group_digests = {
        hashlib.sha256(
            f"blotgrid-study-group-v1:{group}".encode("utf-8")
        ).hexdigest()
        for group in cohort_groups
    }
    if cohort_group_digests & checkpoint_groups or cohort_hashes & checkpoint_hashes:
        raise EvaluationError("Sealed holdout overlaps the local checkpoint.")

    ground_truth: dict[int, list[dict[str, Any]]] = {
        image_id: [] for image_id in images
    }
    artifacts: dict[int, list[dict[str, Any]]] = {
        image_id: [] for image_id in images
    }
    artifact_counts = {name: 0 for name in ARTIFACT_CATEGORIES.values()}
    annotation_ids: set[int] = set()
    for index, raw in enumerate(raw_annotations):
        if not isinstance(raw, Mapping):
            raise EvaluationError(f"COCO annotation {index} must be an object.")
        try:
            annotation_id = int(raw["id"])
            image_id = int(raw["image_id"])
            category_id = int(raw["category_id"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise EvaluationError(
                f"COCO annotation {index} has invalid identity."
            ) from exc
        if annotation_id in annotation_ids or image_id not in images:
            raise EvaluationError("COCO annotation ids/images are invalid.")
        annotation_ids.add(annotation_id)
        image = images[image_id]
        bbox = _xywh_to_xyxy(
            raw.get("bbox"),
            width=image["width"],
            height=image["height"],
            label=f"annotation {annotation_id}",
        )
        if category_id == 1:
            attributes = raw.get("attributes")
            if not isinstance(attributes, Mapping):
                raise EvaluationError(
                    f"Band annotation {annotation_id} lacks attributes."
                )
            if attributes.get("annotation_approved") is not True:
                raise EvaluationError(
                    f"Band annotation {annotation_id} is not human approved."
                )
            target = str(attributes.get("target", "")).strip()
            lane = str(attributes.get("lane", "")).strip()
            visibility = str(attributes.get("visibility", "")).strip()
            doublet_group = attributes.get("doublet_group")
            if target not in image["targets"] or lane not in image["lanes"]:
                raise EvaluationError(
                    f"Band annotation {annotation_id} has invalid target/lane."
                )
            if visibility not in {"clear", "faint"}:
                raise EvaluationError(
                    f"Band annotation {annotation_id} visibility must be clear/faint."
                )
            if doublet_group is not None:
                doublet_group = str(doublet_group).strip()
                if not doublet_group:
                    raise EvaluationError(
                        f"Band annotation {annotation_id} has empty doublet_group."
                    )
            ground_truth[image_id].append(
                {
                    "annotation_id": annotation_id,
                    "bbox": bbox,
                    "target": target,
                    "lane": lane,
                    "visibility": visibility,
                    "doublet_group": doublet_group,
                }
            )
        elif category_id in ARTIFACT_CATEGORIES:
            attributes = raw.get("attributes")
            if not isinstance(attributes, Mapping) or (
                attributes.get("annotation_approved") is not True
            ):
                raise EvaluationError(
                    f"Artifact annotation {annotation_id} is not human approved."
                )
            artifact_type = ARTIFACT_CATEGORIES[category_id]
            artifacts[image_id].append(
                {
                    "annotation_id": annotation_id,
                    "bbox": bbox,
                    "artifact_type": artifact_type,
                    "annotation_approved": True,
                }
            )
            artifact_counts[artifact_type] += 1
        else:
            raise EvaluationError(
                f"Unsupported holdout annotation category: {category_id}."
            )
    if any(count == 0 for count in artifact_counts.values()):
        missing = [name for name, count in artifact_counts.items() if count == 0]
        raise EvaluationError(
            "Holdout lacks required artifact regions: " + ", ".join(missing)
        )
    doublets: dict[tuple[int, str, str, str], int] = {}
    visibility_counts = {"clear": 0, "faint": 0}
    for image_id, records in ground_truth.items():
        for record in records:
            visibility_counts[record["visibility"]] += 1
            group = record.get("doublet_group")
            if group:
                key = (image_id, record["target"], record["lane"], str(group))
                doublets[key] = doublets.get(key, 0) + 1
    if any(count < 2 for count in doublets.values()):
        raise EvaluationError(
            "Every declared doublet_group must contain at least two bands."
        )
    if not doublets:
        raise EvaluationError("Holdout requires at least one reviewed doublet group.")
    if any(count == 0 for count in visibility_counts.values()):
        raise EvaluationError("Holdout requires both clear and faint reviewed bands.")
    return {
        "schema_version": 1,
        "annotations_sha256": annotations_sha256,
        "images": images,
        "ground_truth": ground_truth,
        "artifacts": artifacts,
        "study_groups": sorted(cohort_groups),
        "artifact_counts": artifact_counts,
        "seal": {
            "sealed_utc": str(seal["sealed_utc"]),
            "study_groups": sorted(seal_groups),
        },
    }


def _as_box(record: Mapping[str, Any], *, label: str) -> list[float]:
    raw = record.get("bbox", record.get("box"))
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise EvaluationError(f"{label} needs bbox=[x0,y0,x1,y1].")
    try:
        box = [float(value) for value in raw]
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError(f"{label} bbox is not numeric.") from exc
    if not all(math.isfinite(value) for value in box):
        raise EvaluationError(f"{label} bbox is non-finite.")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise EvaluationError(f"{label} bbox has non-positive area.")
    return box


def _integral_coordinate(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise EvaluationError(f"{label} must be an integral number.")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise EvaluationError(f"{label} must be an integral number.")
    return int(numeric)


def validate_sealed_alignment(
    alignment_applied: Any,
    expected_files: Iterable[str],
) -> dict[str, float]:
    """Require evaluator projects to retain the original sealed pixel frame."""
    expected = {str(filename) for filename in expected_files}
    if not isinstance(alignment_applied, Mapping):
        raise EvaluationError(
            "Reviewed evaluation alignment metadata must be an object."
        )
    recorded_files = set(alignment_applied)
    if recorded_files != expected:
        raise EvaluationError(
            "Reviewed evaluation alignment must cover exactly the sealed cohort."
        )
    normalized: dict[str, float] = {}
    for filename in sorted(expected):
        value = alignment_applied[filename]
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise EvaluationError(
                f"Reviewed evaluation alignment for {filename!r} is invalid."
            )
        angle = float(value)
        if not math.isfinite(angle) or abs(angle) > 1e-12:
            raise EvaluationError(
                "Reviewed evaluator projects must use the original sealed pixels; "
                f"alignment drift was recorded for {filename!r}."
            )
        normalized[filename] = 0.0
    return normalized


def _center_error(prediction: Sequence[float], truth: Sequence[float]) -> tuple[float, float]:
    pred_x = (prediction[0] + prediction[2]) / 2.0
    pred_y = (prediction[1] + prediction[3]) / 2.0
    truth_x = (truth[0] + truth[2]) / 2.0
    truth_y = (truth[1] + truth[3]) / 2.0
    distance = math.hypot(pred_x - truth_x, pred_y - truth_y)
    diagonal = math.hypot(truth[2] - truth[0], truth[3] - truth[1])
    return distance, distance / diagonal if diagonal > 0 else 0.0


def _prediction_artifacts(
    box: Sequence[float],
    artifacts: Sequence[Mapping[str, Any]],
) -> list[str]:
    artifact_boxes = [_as_box(item, label="artifact") for item in artifacts]
    if not artifact_boxes:
        return []
    overlaps = box_iou_matrix([box], artifact_boxes)[0]
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    labels: list[str] = []
    for index, artifact in enumerate(artifacts):
        artifact_box = artifact_boxes[index]
        inside = (
            artifact_box[0] <= cx <= artifact_box[2]
            and artifact_box[1] <= cy <= artifact_box[3]
        )
        if overlaps[index] > 0 or inside:
            labels.append(str(artifact["artifact_type"]))
    return sorted(set(labels))


def one_to_one_metrics(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]] = (),
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Match boxes deterministically and report detailed detector errors."""
    if not 0 < float(iou_threshold) <= 1:
        raise EvaluationError("iou_threshold must be in (0, 1].")
    normalized_predictions = []
    for index, raw in enumerate(predictions):
        score = float(raw.get("score", raw.get("confidence", 1.0)))
        if not math.isfinite(score):
            raise EvaluationError(f"prediction {index} has a non-finite score.")
        normalized_predictions.append(
            {
                **dict(raw),
                "bbox": _as_box(raw, label=f"prediction {index}"),
                "score": score,
                "_index": index,
            }
        )
    normalized_truth = [
        {
            **dict(raw),
            "bbox": _as_box(raw, label=f"ground truth {index}"),
            "_index": index,
        }
        for index, raw in enumerate(ground_truth)
    ]
    order = sorted(
        range(len(normalized_predictions)),
        key=lambda index: (
            -normalized_predictions[index]["score"],
            normalized_predictions[index]["_index"],
        ),
    )
    ious = box_iou_matrix(
        [record["bbox"] for record in normalized_predictions],
        [record["bbox"] for record in normalized_truth],
    )
    used_truth: set[int] = set()
    matches: list[dict[str, Any]] = []
    unmatched_predictions: list[int] = []
    for prediction_index in order:
        prediction = normalized_predictions[prediction_index]
        eligible = []
        for truth_index, truth in enumerate(normalized_truth):
            if truth_index in used_truth:
                continue
            if str(prediction.get("target", "")) != str(truth.get("target", "")):
                continue
            if str(prediction.get("lane", "")) != str(truth.get("lane", "")):
                continue
            eligible.append(truth_index)
        if not eligible:
            unmatched_predictions.append(prediction_index)
            continue
        truth_index = max(
            eligible,
            key=lambda index: (ious[prediction_index, index], -index),
        )
        matched_iou = float(ious[prediction_index, truth_index])
        if matched_iou < float(iou_threshold):
            unmatched_predictions.append(prediction_index)
            continue
        used_truth.add(truth_index)
        truth = normalized_truth[truth_index]
        center_px, center_normalized = _center_error(
            prediction["bbox"], truth["bbox"]
        )
        matches.append(
            {
                "prediction_index": prediction["_index"],
                "truth_index": truth["_index"],
                "annotation_id": truth.get("annotation_id"),
                "target": truth.get("target"),
                "lane": truth.get("lane"),
                "visibility": truth.get("visibility"),
                "doublet_group": truth.get("doublet_group"),
                "iou": matched_iou,
                "center_error_px": center_px,
                "center_error_normalized": center_normalized,
            }
        )
    false_positive_rows = []
    artifact_fp_counts = {name: 0 for name in ARTIFACT_CATEGORIES.values()}
    artifact_fp_counts["unattributed"] = 0
    for prediction_index in unmatched_predictions:
        prediction = normalized_predictions[prediction_index]
        labels = _prediction_artifacts(prediction["bbox"], artifacts)
        if not labels:
            labels = ["unattributed"]
        for label in labels:
            artifact_fp_counts[label] += 1
        false_positive_rows.append(
            {
                "prediction_index": prediction["_index"],
                "target": prediction.get("target"),
                "lane": prediction.get("lane"),
                "score": prediction["score"],
                "bbox": prediction["bbox"],
                "artifact_types": labels,
            }
        )
    true_positive = len(matches)
    false_positive = len(unmatched_predictions)
    false_negative = len(normalized_truth) - len(used_truth)
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
    matched_truth = {row["truth_index"] for row in matches}
    visibility_recall = {}
    visibility_counts = {}
    visibility_matched = {}
    for visibility in ("clear", "faint"):
        indices = {
            index
            for index, truth in enumerate(normalized_truth)
            if truth.get("visibility") == visibility
        }
        visibility_recall[visibility] = (
            len(indices & matched_truth) / len(indices) if indices else None
        )
        visibility_counts[visibility] = len(indices)
        visibility_matched[visibility] = len(indices & matched_truth)
    doublet_groups: dict[tuple[str, str, str], set[int]] = {}
    for index, truth in enumerate(normalized_truth):
        group = truth.get("doublet_group")
        if group:
            key = (str(truth.get("target")), str(truth.get("lane")), str(group))
            doublet_groups.setdefault(key, set()).add(index)
    recovered_groups = sum(
        bool(indices) and indices <= matched_truth
        for indices in doublet_groups.values()
    )
    doublet_indices = set().union(*doublet_groups.values()) if doublet_groups else set()
    matched_ious = [row["iou"] for row in matches]
    center_errors = [row["center_error_px"] for row in matches]
    normalized_errors = [row["center_error_normalized"] for row in matches]
    mean_matched_iou = float(np.mean(matched_ious)) if matched_ious else None
    mean_center_error_px = (
        float(np.mean(center_errors)) if center_errors else None
    )
    mean_center_error_normalized = (
        float(np.mean(normalized_errors)) if normalized_errors else None
    )
    doublet_bands = len(doublet_indices)
    doublet_bands_matched = len(doublet_indices & matched_truth)
    doublet_band_recall = (
        doublet_bands_matched / doublet_bands if doublet_bands else None
    )
    return {
        "summary": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "clear_recall": visibility_recall["clear"],
            "faint_recall": visibility_recall["faint"],
            "clear_bands": visibility_counts["clear"],
            "clear_bands_matched": visibility_matched["clear"],
            "faint_bands": visibility_counts["faint"],
            "faint_bands_matched": visibility_matched["faint"],
            "mean_matched_iou": mean_matched_iou,
            "micro_weighted_mean_matched_iou": mean_matched_iou,
            "matched_iou_sum": float(sum(matched_ious)),
            "matched_iou_count": len(matched_ious),
            "median_matched_iou": (
                float(np.median(matched_ious)) if matched_ious else None
            ),
            "mean_center_error_px": mean_center_error_px,
            "micro_weighted_mean_center_error_px": mean_center_error_px,
            "center_error_px_sum": float(sum(center_errors)),
            "center_error_px_count": len(center_errors),
            "median_center_error_px": (
                float(np.median(center_errors)) if center_errors else None
            ),
            "mean_center_error_normalized": mean_center_error_normalized,
            "micro_weighted_mean_center_error_normalized": (
                mean_center_error_normalized
            ),
            "center_error_normalized_sum": float(sum(normalized_errors)),
            "center_error_normalized_count": len(normalized_errors),
            "doublet_groups": len(doublet_groups),
            "doublet_groups_recovered": int(recovered_groups),
            "doublet_recovery": (
                recovered_groups / len(doublet_groups) if doublet_groups else None
            ),
            "doublet_bands": doublet_bands,
            "doublet_bands_matched": doublet_bands_matched,
            "doublet_band_recall": doublet_band_recall,
            "micro_doublet_band_recall": doublet_band_recall,
            "artifact_false_positives": artifact_fp_counts,
            "iou_threshold": float(iou_threshold),
        },
        "matches": matches,
        "false_positives": false_positive_rows,
        "matched_truth_indices": sorted(matched_truth),
    }


def _aggregate_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(int(item.get("true_positive", 0)) for item in summaries)
    fp = sum(int(item.get("false_positive", 0)) for item in summaries)
    fn = sum(int(item.get("false_negative", 0)) for item in summaries)
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    for visibility in ("clear", "faint"):
        total = sum(int(item.get(f"{visibility}_bands", 0)) for item in summaries)
        matched = sum(
            int(item.get(f"{visibility}_bands_matched", 0)) for item in summaries
        )
        result[f"{visibility}_bands"] = total
        result[f"{visibility}_bands_matched"] = matched
        result[f"{visibility}_recall"] = matched / total if total else None
    weighted_metrics = (
        (
            "mean_matched_iou",
            "micro_weighted_mean_matched_iou",
            "matched_iou_sum",
            "matched_iou_count",
        ),
        (
            "mean_center_error_px",
            "micro_weighted_mean_center_error_px",
            "center_error_px_sum",
            "center_error_px_count",
        ),
        (
            "mean_center_error_normalized",
            "micro_weighted_mean_center_error_normalized",
            "center_error_normalized_sum",
            "center_error_normalized_count",
        ),
    )
    for legacy_field, micro_field, sum_field, count_field in weighted_metrics:
        value_sum = sum(float(item.get(sum_field, 0.0)) for item in summaries)
        value_count = sum(int(item.get(count_field, 0)) for item in summaries)
        weighted_mean = value_sum / value_count if value_count else None
        result[sum_field] = value_sum
        result[count_field] = value_count
        result[legacy_field] = weighted_mean
        result[micro_field] = weighted_mean
    doublet_groups = sum(int(item.get("doublet_groups", 0)) for item in summaries)
    recovered_groups = sum(
        int(item.get("doublet_groups_recovered", 0)) for item in summaries
    )
    result["doublet_groups"] = doublet_groups
    result["doublet_groups_recovered"] = recovered_groups
    result["doublet_recovery"] = (
        recovered_groups / doublet_groups if doublet_groups else None
    )
    doublet_bands = sum(int(item.get("doublet_bands", 0)) for item in summaries)
    matched_doublet_bands = sum(
        int(item.get("doublet_bands_matched", 0)) for item in summaries
    )
    doublet_band_recall = (
        matched_doublet_bands / doublet_bands if doublet_bands else None
    )
    result["doublet_bands"] = doublet_bands
    result["doublet_bands_matched"] = matched_doublet_bands
    result["doublet_band_recall"] = doublet_band_recall
    result["micro_doublet_band_recall"] = doublet_band_recall
    artifact_counts = {
        **{name: 0 for name in ARTIFACT_CATEGORIES.values()},
        "unattributed": 0,
    }
    for summary in summaries:
        for key, value in summary.get("artifact_false_positives", {}).items():
            artifact_counts[str(key)] = artifact_counts.get(str(key), 0) + int(value)
    result["artifact_false_positives"] = artifact_counts
    return result


def stratified_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Aggregate per-image detector records across the required five strata."""
    rows: list[dict[str, Any]] = []
    method_phases = sorted(
        {
            (str(record["method"]), str(record["phase"]))
            for record in records
        }
    )
    for method, phase in method_phases:
        selected = [
            record
            for record in records
            if record["method"] == method and record["phase"] == phase
        ]
        for dimension in STRATA_DIMENSIONS:
            groups: dict[str, list[Mapping[str, Any]]] = {}
            if dimension == "target":
                targets = sorted(
                    {
                        str(item.get("target"))
                        for record in selected
                        for item in (
                            list(record.get("ground_truth", []))
                            + list(record.get("predictions", []))
                        )
                        if str(item.get("target", "")).strip()
                    }
                )
                for target in targets:
                    for record in selected:
                        narrowed = dict(record)
                        narrowed["ground_truth"] = [
                            item
                            for item in record.get("ground_truth", [])
                            if str(item.get("target")) == target
                        ]
                        narrowed["predictions"] = [
                            item
                            for item in record.get("predictions", [])
                            if str(item.get("target")) == target
                        ]
                        groups.setdefault(target, []).append(narrowed)
            else:
                for record in selected:
                    value = str(record.get("strata", {}).get(dimension, ""))
                    if value:
                        groups.setdefault(value, []).append(record)
            for value, group in sorted(groups.items()):
                summaries = [
                    one_to_one_metrics(
                        item.get("predictions", []),
                        item.get("ground_truth", []),
                        item.get("artifacts", []),
                        iou_threshold=iou_threshold,
                    )["summary"]
                    for item in group
                ]
                aggregate = _aggregate_summaries(summaries)
                rows.append(
                    {
                        "method": method,
                        "phase": phase,
                        "dimension": dimension,
                        "value": value,
                        "images": len(group),
                        **aggregate,
                    }
                )
    return rows


def validate_reviewed_projects(
    context: Mapping[str, Any],
    proposal_manifest: Mapping[str, Any],
    reviewed_projects: Mapping[str, Mapping[str, Any]],
    attestation: Mapping[str, Any],
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """Validate blinded review projects and return normalized active ROIs."""
    if proposal_manifest.get("schema_version") != 1:
        raise EvaluationError("Proposal manifest schema_version must be 1.")
    protocol_id = str(proposal_manifest.get("protocol_id", "")).strip()
    if not protocol_id:
        raise EvaluationError("Proposal manifest requires protocol_id.")
    run_nonce = str(proposal_manifest.get("run_nonce", "")).strip().lower()
    run_binding = str(
        proposal_manifest.get("run_binding_sha256", "")
    ).strip().lower()
    if not _is_sha256(run_nonce) or not _is_sha256(run_binding):
        raise EvaluationError(
            "Proposal manifest has invalid run binding provenance."
        )
    methods_payload = proposal_manifest.get("methods")
    if not isinstance(methods_payload, Mapping) or set(methods_payload) != set(
        METHODS
    ):
        raise EvaluationError(
            "Proposal manifest must contain exactly the three evaluation methods."
        )
    if attestation.get("schema_version") != 1:
        raise EvaluationError("Review attestation schema_version must be 1.")
    attested_protocol = str(attestation.get("protocol_id", "")).strip()
    if attested_protocol != protocol_id:
        raise EvaluationError(
            "Review attestation protocol does not match the proposal manifest."
        )
    reviewer = str(attestation.get("reviewer", "")).strip()
    if not reviewer:
        raise EvaluationError("Review attestation requires protocol_id and reviewer.")
    for key in ("blinded", "ground_truth_not_viewed", "whole_image_reviewed"):
        if attestation.get(key) is not True:
            raise EvaluationError(f"Review attestation requires {key}=true.")
    if set(reviewed_projects) != set(METHODS):
        raise EvaluationError("Reviewed projects must contain all three methods.")
    images = {int(key): value for key, value in context["images"].items()}
    normalized: dict[str, dict[int, list[dict[str, Any]]]] = {}
    seen_tasks: set[str] = set()
    expected_image_ids = {str(image_id) for image_id in images}
    for method in METHODS:
        project = reviewed_projects[method]
        if not isinstance(project, Mapping):
            raise EvaluationError(f"Reviewed {method} project must be an object.")
        proposal = methods_payload.get(method)
        if not isinstance(proposal, Mapping):
            raise EvaluationError(f"Proposal manifest lacks method {method}.")
        if proposal.get("schema_version") != 1:
            raise EvaluationError(
                f"Proposal method {method} schema_version must be 1."
            )
        proposal_images = proposal.get("images")
        if not isinstance(proposal_images, Mapping) or (
            set(str(key) for key in proposal_images) != expected_image_ids
        ):
            raise EvaluationError(
                f"Proposal method {method} images do not exactly match the cohort."
            )
        task_id = str(proposal.get("task_id", "")).strip()
        if not task_id or task_id in seen_tasks:
            raise EvaluationError("Proposal task ids must be non-empty and unique.")
        seen_tasks.add(task_id)
        declared_proposal_sha256 = str(
            proposal.get("proposal_sha256", "")
        ).strip().lower()
        actual_proposal_sha256 = proposal_images_sha256(proposal_images)
        if (
            not _is_sha256(declared_proposal_sha256)
            or declared_proposal_sha256 != actual_proposal_sha256
        ):
            raise EvaluationError(
                f"Proposal method {method} has an invalid proposal SHA-256."
            )
        declared_task_binding = str(
            proposal.get("task_binding_sha256", "")
        ).strip().lower()
        expected_task_binding = review_task_binding_sha256(
            run_nonce=run_nonce,
            run_binding_sha256=run_binding,
            protocol_id=protocol_id,
            method=method,
            task_id=task_id,
            proposal_sha256=declared_proposal_sha256,
        )
        if (
            not _is_sha256(declared_task_binding)
            or declared_task_binding != expected_task_binding
        ):
            raise EvaluationError(
                f"Proposal method {method} has an invalid task binding."
            )
        task = project.get("evaluation_task")
        if not isinstance(task, Mapping):
            raise EvaluationError(f"Reviewed {method} project lacks evaluation_task.")
        if task.get("schema_version") != 1:
            raise EvaluationError(
                f"Reviewed {method} evaluation_task schema_version must be 1."
            )
        if str(task.get("task_id")) != task_id:
            raise EvaluationError(f"Reviewed {method} task id does not match.")
        if str(task.get("protocol_id")) != protocol_id:
            raise EvaluationError(f"Reviewed {method} protocol does not match.")
        if (
            str(task.get("proposal_sha256", "")).lower()
            != declared_proposal_sha256
        ):
            raise EvaluationError(
                f"Reviewed {method} proposal SHA-256 does not match."
            )
        if (
            str(task.get("task_binding_sha256", "")).lower()
            != declared_task_binding
        ):
            raise EvaluationError(
                f"Reviewed {method} task binding does not match."
            )
        if task.get("blinded") is not True:
            raise EvaluationError(f"Reviewed {method} task is not blinded.")
        layouts = project.get("layouts")
        provenance = project.get("image_provenance")
        if not isinstance(layouts, Mapping) or not isinstance(provenance, Mapping):
            raise EvaluationError(f"Reviewed {method} project is incomplete.")
        expected_files = {image["file_name"] for image in images.values()}
        if set(str(key) for key in layouts) != expected_files:
            raise EvaluationError(f"Reviewed {method} images do not exactly match.")
        if set(str(key) for key in provenance) != expected_files:
            raise EvaluationError(f"Reviewed {method} provenance does not match.")
        validate_sealed_alignment(
            project.get("alignment_applied", {}),
            expected_files,
        )
        normalized[method] = {}
        by_filename = {image["file_name"]: image for image in images.values()}
        for filename in sorted(expected_files):
            image = by_filename[filename]
            recorded = provenance[filename]
            if not isinstance(recorded, Mapping) or (
                str(recorded.get("sha256", "")).lower() != image["sha256"]
            ):
                raise EvaluationError(
                    f"Reviewed {method} image hash differs for {filename!r}."
                )
            layout = layouts[filename]
            if not isinstance(layout, Mapping) or (
                layout.get("annotation_complete") is not True
            ):
                raise EvaluationError(
                    f"Reviewed {method} image {filename!r} is not complete."
                )
            rois = layout.get("rois")
            if not isinstance(rois, list):
                raise EvaluationError(
                    f"Reviewed {method} image {filename!r} has no ROI list."
                )
            output_rois = []
            expected_size = image["review_roi_size"]
            seen_roi_ids: set[str] = set()
            seen_boxes: set[tuple[int, int, int, int]] = set()
            for index, roi in enumerate(rois):
                if not isinstance(roi, Mapping):
                    raise EvaluationError("Reviewed ROI must be an object.")
                try:
                    x = _integral_coordinate(
                        roi["x"], label="Reviewed ROI x"
                    )
                    y = _integral_coordinate(
                        roi["y"], label="Reviewed ROI y"
                    )
                    w = _integral_coordinate(
                        roi["w"], label="Reviewed ROI width"
                    )
                    h = _integral_coordinate(
                        roi["h"], label="Reviewed ROI height"
                    )
                except KeyError as exc:
                    raise EvaluationError("Reviewed ROI geometry is invalid.") from exc
                roi_id = str(roi.get("roi_id", "")).strip()
                if not roi_id or roi_id in seen_roi_ids:
                    raise EvaluationError(
                        "Reviewed ROI ids must be non-empty and unique."
                    )
                seen_roi_ids.add(roi_id)
                box_key = (x, y, w, h)
                if box_key in seen_boxes:
                    raise EvaluationError(
                        "Reviewed ROI boxes must be unique within each image."
                    )
                seen_boxes.add(box_key)
                if w <= 0 or h <= 0:
                    raise EvaluationError(
                        "Reviewed ROI dimensions must be positive."
                    )
                if (w, h) != (
                    int(expected_size["w"]),
                    int(expected_size["h"]),
                ):
                    raise EvaluationError(
                        "Reviewed ROI size differs from the sealed common size."
                    )
                if (
                    x < 0
                    or y < 0
                    or x + w > int(image["width"])
                    or y + h > int(image["height"])
                ):
                    raise EvaluationError(
                        "Reviewed ROI bbox is outside the image bounds."
                    )
                use = roi.get("use", True)
                if not isinstance(use, bool):
                    raise EvaluationError(
                        "Reviewed ROI use must be a literal boolean."
                    )
                if not use:
                    continue
                if roi.get("annotation_approved") is not True:
                    raise EvaluationError(
                        f"Active reviewed ROI {method}/{filename}/{index} "
                        "is not approved."
                    )
                target = str(roi.get("target", "")).strip()
                lane = str(roi.get("lane", "")).strip()
                if target not in image["targets"] or lane not in image["lanes"]:
                    raise EvaluationError("Reviewed ROI target/lane is invalid.")
                output_rois.append(
                    {
                        "bbox": [float(x), float(y), float(x + w), float(y + h)],
                        "target": target,
                        "lane": lane,
                        "score": 1.0,
                        "roi_id": roi_id,
                    }
                )
            normalized[method][int(image["id"])] = output_rois
    return normalized


def _measure_boxes(
    loaded: image_io.LoadedImage,
    records: Sequence[Mapping[str, Any]],
    *,
    bands_are_dark: bool,
    config: Mapping[str, Any],
) -> dict[tuple[str, str], float]:
    signal = image_io.to_signal(loaded, bands_are_dark)
    method = str(config.get("background_method", "local"))
    margin = int(config.get("margin", 4))
    measurement_mode = str(config.get("measurement_mode", "rectangle"))
    bg_image = None
    bg_value = None
    if method == "rolling_ball":
        bg_image = densitometry.rolling_ball_background(
            signal, float(config.get("rolling_ball_radius", 50.0))
        )
    elif method == "fixed":
        bg_value = float(config.get("fixed_background", 0.0))
    elif method != "local":
        raise EvaluationError("Unsupported densitometry background_method.")
    values: dict[tuple[str, str], float] = {}
    for record in records:
        box = _as_box(record, label="measurement ROI")
        x0, y0, x1, y1 = box
        measurement = densitometry.measure_band(
            signal,
            str(record["target"]),
            str(record["lane"]),
            int(round(x0)),
            int(round(y0)),
            max(1, int(round(x1 - x0))),
            max(1, int(round(y1 - y0))),
            method=method,
            margin=margin,
            bg_image=bg_image,
            bg_value=bg_value,
            max_value=loaded.max_value,
            measurement_mode=measurement_mode,
        )
        key = (str(record["target"]), str(record["lane"]))
        values[key] = values.get(key, 0.0) + float(measurement.net_intden)
    return values


def _fixed_size_records(
    records: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    roi_width: int,
    roi_height: int,
) -> list[dict[str, Any]]:
    """Recenter records onto one sealed ROI size without changing band centres."""
    normalized = []
    fixed_width = min(int(roi_width), int(width))
    fixed_height = min(int(roi_height), int(height))
    for index, record in enumerate(records):
        x0, y0, x1, y1 = _as_box(record, label=f"fixed ROI {index}")
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        new_x0 = max(0.0, min(center_x - fixed_width / 2.0, width - fixed_width))
        new_y0 = max(
            0.0, min(center_y - fixed_height / 2.0, height - fixed_height)
        )
        normalized.append(
            {
                **dict(record),
                "bbox": [
                    new_x0,
                    new_y0,
                    new_x0 + fixed_width,
                    new_y0 + fixed_height,
                ],
            }
        )
    return normalized


def _finite_error(candidate: float | None, reference: float | None) -> dict[str, Any]:
    if candidate is None or reference is None:
        return {
            "status": "missing",
            "signed_error": None,
            "absolute_error": None,
            "relative_error": None,
        }
    if not math.isfinite(candidate) or not math.isfinite(reference):
        return {
            "status": "nonfinite",
            "signed_error": None,
            "absolute_error": None,
            "relative_error": None,
        }
    signed = candidate - reference
    return {
        "status": "ok" if reference != 0 else "zero_reference",
        "signed_error": signed,
        "absolute_error": abs(signed),
        "relative_error": abs(signed) / abs(reference) if reference != 0 else None,
    }


def _fold_changes(
    values: Mapping[tuple[str, str], float],
    *,
    targets: Sequence[str],
    lanes: Sequence[str],
    control_target: str,
    reference_lane: str,
) -> dict[tuple[str, str], float | None]:
    output: dict[tuple[str, str], float | None] = {}
    for target in targets:
        if target == control_target:
            continue
        target_reference = values.get((target, reference_lane))
        control_reference = values.get((control_target, reference_lane))
        reference_normalized = (
            target_reference / control_reference
            if target_reference is not None
            and control_reference is not None
            and control_reference > 0
            else None
        )
        for lane in lanes:
            target_value = values.get((target, lane))
            control_value = values.get((control_target, lane))
            normalized = (
                target_value / control_value
                if target_value is not None
                and control_value is not None
                and control_value > 0
                else None
            )
            output[(target, lane)] = (
                normalized / reference_normalized
                if normalized is not None
                and reference_normalized is not None
                and reference_normalized > 0
                else None
            )
    return output


def measure_reviewed_errors(
    context: Mapping[str, Any],
    image_root: str | os.PathLike[str],
    reviewed_by_method: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    plan: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Compare reviewed ROIs with sealed GT using identical native-pixel settings."""
    del image_root  # Context contains seal-validated absolute image paths.
    config = plan.get("densitometry", {})
    if not isinstance(config, Mapping):
        raise EvaluationError("Evaluation plan densitometry must be an object.")
    intensity_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    images = {int(key): value for key, value in context["images"].items()}
    truths = {int(key): value for key, value in context["ground_truth"].items()}
    for image_id, image in sorted(images.items()):
        loaded = image_io.load_image(image["path"], name=image["file_name"])
        reference_rois = _fixed_size_records(
            truths.get(image_id, []),
            width=int(image["width"]),
            height=int(image["height"]),
            roi_width=int(image["review_roi_size"]["w"]),
            roi_height=int(image["review_roi_size"]["h"]),
        )
        reference_values = _measure_boxes(
            loaded,
            reference_rois,
            bands_are_dark=bool(image["bands_are_dark"]),
            config=config,
        )
        method_values = {
            method: _measure_boxes(
                loaded,
                reviewed_by_method[method].get(image_id, []),
                bands_are_dark=bool(image["bands_are_dark"]),
                config=config,
            )
            for method in METHODS
        }
        for phase, values in [("ground_truth", reference_values), *method_values.items()]:
            for (target, lane), value in sorted(values.items()):
                intensity_rows.append(
                    {
                        "image_id": image_id,
                        "file_name": image["file_name"],
                        "phase": phase,
                        "target": target,
                        "lane": lane,
                        "net_intden": value,
                        **image["strata"],
                    }
                )
        all_keys = set(reference_values)
        for values in method_values.values():
            all_keys.update(values)
        for method, values in method_values.items():
            for target, lane in sorted(all_keys):
                reference = reference_values.get((target, lane))
                candidate = values.get((target, lane))
                error_rows.append(
                    {
                        "image_id": image_id,
                        "method": method,
                        "metric": "net_intden",
                        "target": target,
                        "lane": lane,
                        "reference": reference,
                        "candidate": candidate,
                        **_finite_error(candidate, reference),
                        **image["strata"],
                    }
                )
        reference_folds = _fold_changes(
            reference_values,
            targets=image["targets"],
            lanes=image["lanes"],
            control_target=image["control_target"],
            reference_lane=image["reference_lane"],
        )
        for method, values in method_values.items():
            candidate_folds = _fold_changes(
                values,
                targets=image["targets"],
                lanes=image["lanes"],
                control_target=image["control_target"],
                reference_lane=image["reference_lane"],
            )
            for target_lane in sorted(set(reference_folds) | set(candidate_folds)):
                target, lane = target_lane
                reference = reference_folds.get(target_lane)
                candidate = candidate_folds.get(target_lane)
                fold_rows.append(
                    {
                        "image_id": image_id,
                        "method": method,
                        "target": target,
                        "lane": lane,
                        "reference_fold_change": reference,
                        "candidate_fold_change": candidate,
                        **image["strata"],
                    }
                )
                error = _finite_error(candidate, reference)
                log2_error = None
                if (
                    candidate is not None
                    and reference is not None
                    and candidate > 0
                    and reference > 0
                ):
                    log2_error = math.log2(candidate) - math.log2(reference)
                error_rows.append(
                    {
                        "image_id": image_id,
                        "method": method,
                        "metric": "fold_change",
                        "target": target,
                        "lane": lane,
                        "reference": reference,
                        "candidate": candidate,
                        "log2_error": log2_error,
                        **error,
                        **image["strata"],
                    }
                )
    return {
        "intensities": intensity_rows,
        "fold_changes": fold_rows,
        "paired_errors": error_rows,
    }
