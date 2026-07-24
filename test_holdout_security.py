"""Fail-closed security and aggregation tests for sealed holdout evaluation."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from wblot.holdout_evaluation import (
    EvaluationError,
    proposal_images_sha256,
    review_task_binding_sha256,
    stratified_metrics,
    validate_holdout_inputs,
    validate_reviewed_projects,
)
from wblot.ml_training import PUBLICATION_METADATA_SCHEMA_VERSION


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_inputs(tmp_path: Path) -> dict:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "sealed.png"
    Image.fromarray(np.full((80, 100), 128, dtype=np.uint8)).save(image_path)
    image_sha256 = _sha256(image_path)
    metadata = {
        "study_group": "sealed-study",
        "annotation_complete": True,
        "original_sha256": image_sha256,
        "targets": ["Target", "Control"],
        "lanes": ["Reference", "Treated"],
        "lane_centers_hint": [20, 70],
        "target_centers_hint": [17, 53],
        "bands_are_dark": True,
        "control_target": "Control",
        "reference_lane": "Reference",
        "review_roi_size": {"w": 16, "h": 6},
        "strata": {
            "scanner": "flatbed-a",
            "exposure": "short",
            "background": "clean",
            "acquisition_style": "film",
        },
    }

    def band(
        annotation_id: int,
        bbox: list[int],
        target: str,
        lane: str,
        visibility: str,
        doublet_group: str | None = None,
    ) -> dict:
        return {
            "id": annotation_id,
            "image_id": 1,
            "category_id": 1,
            "bbox": bbox,
            "attributes": {
                "annotation_approved": True,
                "target": target,
                "lane": lane,
                "visibility": visibility,
                "doublet_group": doublet_group,
            },
        }

    annotations = [
        band(1, [12, 14, 16, 6], "Target", "Reference", "clear"),
        band(2, [62, 14, 16, 6], "Target", "Treated", "faint"),
        band(3, [12, 48, 16, 4], "Control", "Reference", "clear", "d1"),
        band(4, [12, 54, 16, 4], "Control", "Reference", "clear", "d1"),
    ]
    for annotation_id, category_id, bbox in (
        (5, 2, [0, 0, 7, 80]),
        (6, 3, [80, 0, 20, 12]),
        (7, 4, [36, 30, 18, 18]),
        (8, 5, [55, 32, 35, 16]),
    ):
        annotations.append(
            {
                "id": annotation_id,
                "image_id": 1,
                "category_id": category_id,
                "bbox": bbox,
                "attributes": {"annotation_approved": True},
            }
        )
    coco = {
        "images": [
            {
                "id": 1,
                "file_name": image_path.name,
                "width": 100,
                "height": 80,
                "blotgrid": metadata,
            }
        ],
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "band"},
            {"id": 2, "name": "border"},
            {"id": 3, "name": "text"},
            {"id": 4, "name": "smear"},
            {"id": 5, "name": "blank_cell"},
        ],
    }
    annotations_path = tmp_path / "holdout.json"
    _write_json(annotations_path, coco)
    seal = {
        "schema_version": 1,
        "sealed_utc": "2026-07-24T00:00:00Z",
        "annotations_sha256": _sha256(annotations_path),
        "image_sha256": {"1": image_sha256},
        "study_groups": ["sealed-study"],
    }
    ledger = {
        "schema_version": 1,
        "complete": True,
        "completeness_attested": True,
        "study_groups": ["development-study"],
        "image_sha256": ["a" * 64],
        "model_selection": [
            {
                "study_group": "development-study",
                "image_sha256": "a" * 64,
            }
        ],
    }
    manifest = [
        {
            "image_digest": "b" * 64,
            "sha256": "c" * 64,
            "study_group_sha256": "d" * 64,
            "original_sha256": "e" * 64,
            "license_id": 1,
            "source_id": "public-source-v1",
        }
    ]
    checkpoint = {
        "metadata_verified": True,
        "embedded_metadata_verified": True,
        "container_verified": True,
        "metadata": {
            "format_version": 2,
            "publication_metadata_schema_version": (
                PUBLICATION_METADATA_SCHEMA_VERSION
            ),
            "image_manifest": manifest,
            "image_manifest_sha256": hashlib.sha256(
                json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
        },
    }
    return {
        "image_root": image_root,
        "image_path": image_path,
        "annotations_path": annotations_path,
        "coco": coco,
        "seal": seal,
        "ledger": ledger,
        "checkpoint": checkpoint,
    }


def _validate(payload: dict) -> dict:
    return validate_holdout_inputs(
        payload["coco"],
        payload["annotations_path"],
        payload["image_root"],
        payload["seal"],
        payload["ledger"],
        payload["checkpoint"],
    )


def test_holdout_requires_flat_streamlit_restorable_filenames(tmp_path):
    payload = _strict_inputs(tmp_path)
    payload["coco"]["images"][0]["file_name"] = "nested/sealed.png"
    _write_json(payload["annotations_path"], payload["coco"])
    payload["seal"]["annotations_sha256"] = _sha256(
        payload["annotations_path"]
    )
    with pytest.raises(EvaluationError, match="flat basenames"):
        _validate(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda ledger: ledger.update(schema_version=2), "schema_version"),
        (lambda ledger: ledger.update(complete=False), "complete=true"),
        (
            lambda ledger: ledger.update(completeness_attested=False),
            "completeness_attested=true",
        ),
        (lambda ledger: ledger.update(model_selection=[]), "model_selection"),
        (
            lambda ledger: ledger["model_selection"][0].update(
                image_sha256="f" * 64
            ),
            "exactly match",
        ),
    ],
)
def test_development_ledger_is_complete_and_exact(
    tmp_path, mutation, message
):
    payload = _strict_inputs(tmp_path)
    mutation(payload["ledger"])
    with pytest.raises(EvaluationError, match=message):
        _validate(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda metadata: metadata.update(format_version=1),
            "format_version",
        ),
        (
            lambda metadata: metadata.update(
                publication_metadata_schema_version=1
            ),
            "publication schema_version",
        ),
        (
            lambda metadata: metadata.update(image_manifest=[]),
            "non-empty",
        ),
        (
            lambda metadata: metadata["image_manifest"][0].update(
                file_name="private.tif"
            ),
            "non-public",
        ),
        (
            lambda metadata: metadata["image_manifest"][0].update(
                image_digest="not-a-digest"
            ),
            "invalid digests",
        ),
    ],
)
def test_checkpoint_ancestry_is_verified_public_and_nonempty(
    tmp_path, mutation, message
):
    payload = _strict_inputs(tmp_path)
    mutation(payload["checkpoint"]["metadata"])
    with pytest.raises(EvaluationError, match=message):
        _validate(payload)


def test_duplicate_holdout_bytes_are_rejected(tmp_path):
    payload = _strict_inputs(tmp_path)
    duplicate_path = payload["image_root"] / "duplicate.png"
    duplicate_path.write_bytes(payload["image_path"].read_bytes())
    duplicate = copy.deepcopy(payload["coco"]["images"][0])
    duplicate["id"] = 2
    duplicate["file_name"] = duplicate_path.name
    payload["coco"]["images"].append(duplicate)
    payload["seal"]["image_sha256"]["2"] = _sha256(duplicate_path)
    _write_json(payload["annotations_path"], payload["coco"])
    payload["seal"]["annotations_sha256"] = _sha256(
        payload["annotations_path"]
    )
    with pytest.raises(EvaluationError, match="duplicate image bytes"):
        _validate(payload)


def test_checkpoint_sidecar_must_match_restricted_load_metadata(tmp_path):
    payload = _strict_inputs(tmp_path)
    payload["checkpoint"]["embedded_metadata_verified"] = False
    with pytest.raises(EvaluationError, match="restricted-load embedded metadata"):
        _validate(payload)


def test_checkpoint_container_must_match_generated_contract(tmp_path):
    payload = _strict_inputs(tmp_path)
    payload["checkpoint"]["container_verified"] = False
    with pytest.raises(EvaluationError, match="container contract"):
        _validate(payload)


def test_artifact_regions_require_explicit_human_approval(tmp_path):
    payload = _strict_inputs(tmp_path)
    payload["coco"]["annotations"][-1]["attributes"] = {
        "annotation_approved": False
    }
    _write_json(payload["annotations_path"], payload["coco"])
    payload["seal"]["annotations_sha256"] = _sha256(
        payload["annotations_path"]
    )
    with pytest.raises(EvaluationError, match="not human approved"):
        _validate(payload)


@pytest.mark.parametrize(
    "bbox",
    (
        [-1, 0, 7, 80],
        [94, 0, 7, 80],
        [0, -1, 7, 80],
        [0, 1, 7, 80],
    ),
)
@pytest.mark.parametrize("annotation_index", (0, 4))
def test_ground_truth_boxes_outside_image_are_rejected(
    tmp_path, bbox, annotation_index
):
    payload = _strict_inputs(tmp_path)
    payload["coco"]["annotations"][annotation_index]["bbox"] = bbox
    _write_json(payload["annotations_path"], payload["coco"])
    payload["seal"]["annotations_sha256"] = _sha256(
        payload["annotations_path"]
    )
    with pytest.raises(EvaluationError, match="outside the image bounds"):
        _validate(payload)


@pytest.mark.parametrize("annotation_index", (0, 4))
def test_ground_truth_box_may_touch_every_image_boundary(
    tmp_path, annotation_index
):
    payload = _strict_inputs(tmp_path)
    payload["coco"]["annotations"][annotation_index]["bbox"] = [0, 0, 100, 80]
    _write_json(payload["annotations_path"], payload["coco"])
    payload["seal"]["annotations_sha256"] = _sha256(
        payload["annotations_path"]
    )
    assert _validate(payload)["artifact_counts"]["border"] == 1


def _review_bundle(context: dict) -> tuple[dict, dict, dict]:
    protocol_id = "protocol-1"
    run_nonce = "1" * 64
    run_binding = "2" * 64
    proposals = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "run_nonce": run_nonce,
        "run_binding_sha256": run_binding,
        "methods": {},
    }
    projects = {}
    image = context["images"][1]
    for method_index, method in enumerate(
        ("classical", "gelgenie", "local")
    ):
        task_id = f"task-{method}"
        images = {"1": {"review_rois": []}}
        proposal_sha256 = proposal_images_sha256(images)
        task_binding = review_task_binding_sha256(
            run_nonce=run_nonce,
            run_binding_sha256=run_binding,
            protocol_id=protocol_id,
            method=method,
            task_id=task_id,
            proposal_sha256=proposal_sha256,
        )
        proposals["methods"][method] = {
            "schema_version": 1,
            "task_id": task_id,
            "images": images,
            "proposal_sha256": proposal_sha256,
            "task_binding_sha256": task_binding,
        }
        rois = [
            {
                "roi_id": f"{method}-roi-{index}",
                "x": 10 + 18 * index,
                "y": 10 + 8 * method_index,
                "w": 16,
                "h": 6,
                "target": "Target",
                "lane": "Reference",
                "annotation_approved": True,
                "use": True,
            }
            for index in range(2)
        ]
        projects[method] = {
            "evaluation_task": {
                "schema_version": 1,
                "task_id": task_id,
                "protocol_id": protocol_id,
                "proposal_sha256": proposal_sha256,
                "task_binding_sha256": task_binding,
                "blinded": True,
            },
            "image_provenance": {
                image["file_name"]: {"sha256": image["sha256"]}
            },
            "alignment_applied": {image["file_name"]: 0.0},
            "layouts": {
                image["file_name"]: {
                    "annotation_complete": True,
                    "rois": rois,
                }
            },
        }
    attestation = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "reviewer": "reviewer-1",
        "blinded": True,
        "ground_truth_not_viewed": True,
        "whole_image_reviewed": True,
    }
    return proposals, projects, attestation


def test_review_tasks_are_bound_to_exact_run_and_proposals(tmp_path):
    context = _validate(_strict_inputs(tmp_path))
    proposals, projects, attestation = _review_bundle(context)
    validate_reviewed_projects(context, proposals, projects, attestation)

    tampered = copy.deepcopy(proposals)
    tampered["methods"]["local"]["images"]["1"]["review_rois"].append({})
    with pytest.raises(EvaluationError, match="proposal SHA-256"):
        validate_reviewed_projects(context, tampered, projects, attestation)

    replayed = copy.deepcopy(projects)
    replayed["local"]["evaluation_task"]["task_binding_sha256"] = "f" * 64
    with pytest.raises(EvaluationError, match="task binding"):
        validate_reviewed_projects(context, proposals, replayed, attestation)


def test_reviewed_projects_must_remain_on_original_sealed_pixels(tmp_path):
    context = _validate(_strict_inputs(tmp_path))
    proposals, projects, attestation = _review_bundle(context)
    filename = context["images"][1]["file_name"]
    for project in projects.values():
        project["alignment_applied"] = {filename: 0.0}
    validate_reviewed_projects(context, proposals, projects, attestation)

    drifted = copy.deepcopy(projects)
    drifted["local"]["alignment_applied"][filename] = 0.1
    with pytest.raises(EvaluationError, match="original sealed pixels"):
        validate_reviewed_projects(
            context,
            proposals,
            drifted,
            attestation,
        )

    missing = copy.deepcopy(projects)
    del missing["local"]["alignment_applied"][filename]
    with pytest.raises(EvaluationError, match="exactly the sealed cohort"):
        validate_reviewed_projects(
            context,
            proposals,
            missing,
            attestation,
        )


def test_reviewed_roi_ids_are_unique_per_image_not_per_project(tmp_path):
    context = _validate(_strict_inputs(tmp_path))
    second_image = copy.deepcopy(context["images"][1])
    second_image.update(
        {
            "id": 2,
            "file_name": "sealed-second.png",
            "sha256": "9" * 64,
        }
    )
    context["images"][2] = second_image
    proposals, projects, attestation = _review_bundle(context)
    for method in ("classical", "gelgenie", "local"):
        proposal = proposals["methods"][method]
        proposal["images"]["2"] = {"review_rois": []}
        proposal_sha256 = proposal_images_sha256(proposal["images"])
        task_binding = review_task_binding_sha256(
            run_nonce=proposals["run_nonce"],
            run_binding_sha256=proposals["run_binding_sha256"],
            protocol_id=proposals["protocol_id"],
            method=method,
            task_id=proposal["task_id"],
            proposal_sha256=proposal_sha256,
        )
        proposal["proposal_sha256"] = proposal_sha256
        proposal["task_binding_sha256"] = task_binding
        task = projects[method]["evaluation_task"]
        task["proposal_sha256"] = proposal_sha256
        task["task_binding_sha256"] = task_binding
        first_filename = context["images"][1]["file_name"]
        projects[method]["layouts"][second_image["file_name"]] = copy.deepcopy(
            projects[method]["layouts"][first_filename]
        )
        projects[method]["image_provenance"][second_image["file_name"]] = {
            "sha256": second_image["sha256"]
        }
        projects[method]["alignment_applied"][second_image["file_name"]] = 0.0
    reviewed = validate_reviewed_projects(
        context, proposals, projects, attestation
    )
    assert all(set(images) == {1, 2} for images in reviewed.values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rois: rois[0].update(x=1.25), "integral"),
        (lambda rois: rois[0].update(x=95), "outside"),
        (
            lambda rois: rois[1].update(roi_id=rois[0]["roi_id"]),
            "ids must be",
        ),
        (
            lambda rois: rois[1].update(
                x=rois[0]["x"], y=rois[0]["y"]
            ),
            "boxes must be",
        ),
    ],
)
def test_reviewed_roi_geometry_is_integral_bounded_and_unique(
    tmp_path, mutation, message
):
    context = _validate(_strict_inputs(tmp_path))
    proposals, projects, attestation = _review_bundle(context)
    filename = context["images"][1]["file_name"]
    rois = projects["classical"]["layouts"][filename]["rois"]
    mutation(rois)
    with pytest.raises(EvaluationError, match=message):
        validate_reviewed_projects(context, proposals, projects, attestation)


def test_strata_expose_micro_weighted_localization_and_doublet_counts():
    exact_truth = [
        {
            "bbox": [0, 0, 10, 10],
            "target": "T",
            "lane": "L0",
            "visibility": "clear",
        }
    ]
    exact_predictions = [
        {"bbox": [0, 0, 10, 10], "target": "T", "lane": "L0"}
    ]
    shifted_truth = [
        {
            "bbox": [20 * index, 20, 20 * index + 10, 30],
            "target": "T",
            "lane": f"L{index + 1}",
            "visibility": "faint",
            "doublet_group": "d1" if index < 2 else None,
        }
        for index in range(3)
    ]
    shifted_predictions = [
        {
            "bbox": [20 * index + 1, 20, 20 * index + 11, 30],
            "target": "T",
            "lane": f"L{index + 1}",
        }
        for index in range(3)
    ]
    records = [
        {
            "method": "local",
            "phase": "post_review",
            "strata": {
                "scanner": "scanner-a",
                "exposure": "short",
                "background": "clean",
                "acquisition_style": "film",
            },
            "ground_truth": exact_truth,
            "predictions": exact_predictions,
        },
        {
            "method": "local",
            "phase": "post_review",
            "strata": {
                "scanner": "scanner-a",
                "exposure": "long",
                "background": "clean",
                "acquisition_style": "film",
            },
            "ground_truth": shifted_truth,
            "predictions": shifted_predictions,
        },
    ]
    rows = stratified_metrics(records, iou_threshold=0.5)
    scanner = next(
        row
        for row in rows
        if row["dimension"] == "scanner" and row["value"] == "scanner-a"
    )
    assert scanner["matched_iou_count"] == 4
    assert scanner["micro_weighted_mean_matched_iou"] == pytest.approx(
        scanner["matched_iou_sum"] / 4
    )
    assert scanner["center_error_px_count"] == 4
    assert scanner["micro_weighted_mean_center_error_px"] == pytest.approx(
        0.75
    )
    assert scanner["doublet_bands"] == 2
    assert scanner["doublet_bands_matched"] == 2
    assert scanner["micro_doublet_band_recall"] == pytest.approx(1.0)
