"""Synthetic tests for the sealed detector holdout workflow."""
from __future__ import annotations

from argparse import Namespace
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts import evaluate_detector
from wblot import detection
from wblot.holdout_evaluation import (
    EvaluationError,
    measure_reviewed_errors,
    one_to_one_metrics,
    proposal_images_sha256,
    review_task_binding_sha256,
    stratified_metrics,
    validate_holdout_inputs,
    validate_reviewed_projects,
)
from wblot.ml_training import PUBLICATION_METADATA_SCHEMA_VERSION


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _cohort(tmp_path: Path) -> dict:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "sealed.png"
    pixels = np.full((80, 100), 220, dtype=np.uint8)
    pixels[14:20, 12:28] = 20
    pixels[14:20, 62:78] = 95
    pixels[48:52, 12:28] = 30
    pixels[54:58, 12:28] = 35
    pixels[50:56, 62:78] = 25
    Image.fromarray(pixels).save(image_path)
    digest = _sha256(image_path)
    metadata = {
        "study_group": "sealed-study",
        "annotation_complete": True,
        "original_sha256": digest,
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
        band(5, [62, 50, 16, 6], "Control", "Treated", "clear"),
        {
            "id": 6,
            "image_id": 1,
            "category_id": 2,
            "bbox": [0, 0, 7, 80],
            "attributes": {"annotation_approved": True},
        },
        {
            "id": 7,
            "image_id": 1,
            "category_id": 3,
            "bbox": [80, 0, 20, 12],
            "attributes": {"annotation_approved": True},
        },
        {
            "id": 8,
            "image_id": 1,
            "category_id": 4,
            "bbox": [36, 30, 18, 18],
            "attributes": {"annotation_approved": True},
        },
        {
            "id": 9,
            "image_id": 1,
            "category_id": 5,
            "bbox": [55, 32, 35, 16],
            "attributes": {"annotation_approved": True},
        },
    ]
    coco = {
        "info": {"description": "synthetic sealed holdout"},
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
    annotation_path = tmp_path / "holdout.json"
    _write_json(annotation_path, coco)
    seal = {
        "schema_version": 1,
        "sealed_utc": "2026-07-24T00:00:00Z",
        "annotations_sha256": _sha256(annotation_path),
        "image_sha256": {"1": digest},
        "study_groups": ["sealed-study"],
    }
    ledger = {
        "schema_version": 1,
        "complete": True,
        "completeness_attested": True,
        "study_groups": ["development-study"],
        "image_sha256": ["f" * 64],
        "model_selection": [
            {
                "study_group": "development-study",
                "image_sha256": "f" * 64,
            }
        ],
    }
    checkpoint_manifest = [
        {
            "image_digest": "a" * 64,
            "sha256": "b" * 64,
            "study_group_sha256": "c" * 64,
            "original_sha256": "d" * 64,
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
            "image_manifest": checkpoint_manifest,
            "image_manifest_sha256": hashlib.sha256(
                json.dumps(
                    checkpoint_manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
        },
    }
    context = validate_holdout_inputs(
        coco,
        annotation_path,
        image_root,
        seal,
        ledger,
        checkpoint,
    )
    return {
        "image_root": image_root,
        "image_path": image_path,
        "annotations": annotation_path,
        "coco": coco,
        "seal": seal,
        "ledger": ledger,
        "checkpoint": checkpoint,
        "context": context,
    }


def _predictions_from_truth(context: dict) -> list[dict]:
    return [
        {
            "bbox": list(record["bbox"]),
            "target": record["target"],
            "lane": record["lane"],
            "score": 0.9,
        }
        for record in context["ground_truth"][1]
    ]


def _review_projects(context: dict, proposals: dict, protocol_id: str) -> dict:
    projects = {}
    image = context["images"][1]
    for method in ("classical", "gelgenie", "local"):
        task_id = proposals["methods"][method]["task_id"]
        rois = []
        for index, record in enumerate(_predictions_from_truth(context)):
            x0, y0, x1, y1 = record["bbox"]
            center_y = (y0 + y1) / 2
            height = image["review_roi_size"]["h"]
            y0 = max(0, min(center_y - height / 2, image["height"] - height))
            rois.append(
                {
                    "roi_id": f"{method}-{index}",
                    "x": int(round(x0)),
                    "y": int(round(y0)),
                    "w": image["review_roi_size"]["w"],
                    "h": height,
                    "target": record["target"],
                    "lane": record["lane"],
                    "annotation_approved": True,
                    "use": True,
                }
            )
        projects[method] = {
            "evaluation_task": {
                "schema_version": 1,
                "task_id": task_id,
                "protocol_id": protocol_id,
                "proposal_sha256": proposals["methods"][method][
                    "proposal_sha256"
                ],
                "task_binding_sha256": proposals["methods"][method][
                    "task_binding_sha256"
                ],
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
    return projects


def _bound_proposals(context: dict, protocol_id: str = "protocol-1") -> dict:
    run_nonce = "1" * 64
    run_binding = "2" * 64
    proposals = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "run_nonce": run_nonce,
        "run_binding_sha256": run_binding,
        "methods": {},
    }
    image_ids = {str(image_id): {"review_rois": []} for image_id in context["images"]}
    for method in ("classical", "gelgenie", "local"):
        task_id = f"task-{method}"
        method_images = copy.deepcopy(image_ids)
        proposal_sha256 = proposal_images_sha256(method_images)
        proposals["methods"][method] = {
            "schema_version": 1,
            "task_id": task_id,
            "images": method_images,
            "proposal_sha256": proposal_sha256,
            "task_binding_sha256": review_task_binding_sha256(
                run_nonce=run_nonce,
                run_binding_sha256=run_binding,
                protocol_id=protocol_id,
                method=method,
                task_id=task_id,
                proposal_sha256=proposal_sha256,
            ),
        }
    return proposals


def test_review_project_keeps_labels_and_zero_alignment_for_every_image(tmp_path):
    context = copy.deepcopy(_cohort(tmp_path)["context"])
    second = copy.deepcopy(context["images"][1])
    second.update(
        {
            "id": 2,
            "file_name": "second.png",
            "sha256": "9" * 64,
            "targets": ["Other target", "Control"],
            "lanes": ["Reference", "Other lane"],
        }
    )
    context["images"][2] = second
    project = evaluate_detector._review_project(
        "task-1",
        "protocol-1",
        "3" * 64,
        "4" * 64,
        context,
        {
            "1": {"review_rois": []},
            "2": {"review_rois": []},
        },
    )
    assert project["targets"] == ["Target", "Control", "Other target"]
    assert project["lanes"] == ["Reference", "Treated", "Other lane"]
    assert project["alignment_applied"] == {
        context["images"][1]["file_name"]: 0.0,
        "second.png": 0.0,
    }


def test_holdout_validation_rejects_leakage_and_incomplete_science(tmp_path):
    cohort = _cohort(tmp_path)
    leaked = copy.deepcopy(cohort["ledger"])
    leaked["study_groups"] = ["sealed-study"]
    leaked["model_selection"][0]["study_group"] = "sealed-study"
    with pytest.raises(EvaluationError, match="development ledger"):
        validate_holdout_inputs(
            cohort["coco"],
            cohort["annotations"],
            cohort["image_root"],
            cohort["seal"],
            leaked,
            cohort["checkpoint"],
        )

    invalid = copy.deepcopy(cohort["coco"])
    invalid["annotations"][1]["attributes"]["visibility"] = "unknown"
    invalid_path = tmp_path / "invalid.json"
    _write_json(invalid_path, invalid)
    invalid_seal = dict(
        cohort["seal"], annotations_sha256=_sha256(invalid_path)
    )
    with pytest.raises(EvaluationError, match="visibility"):
        validate_holdout_inputs(
            invalid,
            invalid_path,
            cohort["image_root"],
            invalid_seal,
            cohort["ledger"],
            cohort["checkpoint"],
        )

    overlapping_checkpoint = copy.deepcopy(cohort["checkpoint"])
    overlapping_checkpoint["metadata"]["image_manifest"][0][
        "sha256"
    ] = cohort["context"]["images"][1]["sha256"]
    overlapping_checkpoint["metadata"]["image_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            overlapping_checkpoint["metadata"]["image_manifest"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(EvaluationError, match="checkpoint"):
        validate_holdout_inputs(
            cohort["coco"],
            cohort["annotations"],
            cohort["image_root"],
            cohort["seal"],
            cohort["ledger"],
            overlapping_checkpoint,
        )


def test_detailed_metrics_cover_visibility_doublets_centres_and_artifacts(tmp_path):
    context = _cohort(tmp_path)["context"]
    predictions = _predictions_from_truth(context)
    predictions.pop(1)  # miss the faint band
    predictions.append(dict(predictions[0], score=0.8))  # duplicate -> FP
    predictions.append(
        {
            "bbox": [0, 25, 6, 35],
            "target": "Target",
            "lane": "Reference",
            "score": 0.7,
        }
    )
    result = one_to_one_metrics(
        predictions,
        context["ground_truth"][1],
        context["artifacts"][1],
        iou_threshold=0.5,
    )
    summary = result["summary"]
    assert summary["true_positive"] == 4
    assert summary["false_negative"] == 1
    assert summary["false_positive"] == 2
    assert summary["clear_recall"] == 1.0
    assert summary["faint_recall"] == 0.0
    assert summary["doublet_recovery"] == 1.0
    assert summary["mean_center_error_px"] == pytest.approx(0.0)
    assert summary["artifact_false_positives"]["border"] == 1
    assert summary["artifact_false_positives"]["unattributed"] == 1


def test_required_strata_reconcile_with_raw_counts(tmp_path):
    context = _cohort(tmp_path)["context"]
    records = []
    for method in ("classical", "gelgenie"):
        records.append(
            {
                "method": method,
                "phase": "direct",
                "predictions": _predictions_from_truth(context),
                "ground_truth": context["ground_truth"][1],
                "artifacts": context["artifacts"][1],
                "strata": context["images"][1]["strata"],
            }
        )
    rows = stratified_metrics(records, iou_threshold=0.5)
    dimensions = {row["dimension"] for row in rows}
    assert dimensions == {
        "scanner",
        "exposure",
        "target",
        "background",
        "acquisition_style",
    }
    scanner = next(
        row
        for row in rows
        if row["method"] == "classical" and row["dimension"] == "scanner"
    )
    assert scanner["true_positive"] == len(context["ground_truth"][1])
    assert scanner["false_positive"] == 0


def test_review_validation_and_native_downstream_errors(tmp_path):
    cohort = _cohort(tmp_path)
    context = cohort["context"]
    proposals = _bound_proposals(context)
    projects = _review_projects(context, proposals, "protocol-1")
    attestation = {
        "schema_version": 1,
        "protocol_id": "protocol-1",
        "reviewer": "reviewer-1",
        "blinded": True,
        "ground_truth_not_viewed": True,
        "whole_image_reviewed": True,
    }
    reviewed = validate_reviewed_projects(
        context, proposals, projects, attestation
    )
    downstream = measure_reviewed_errors(
        context,
        cohort["image_root"],
        reviewed,
        {
            "densitometry": {
                "background_method": "local",
                "margin": 3,
                "measurement_mode": "rectangle",
            }
        },
    )
    ok_errors = [
        row
        for row in downstream["paired_errors"]
        if row["status"] == "ok"
    ]
    assert ok_errors
    assert all(row["absolute_error"] == pytest.approx(0.0) for row in ok_errors)
    assert any(row["metric"] == "fold_change" for row in ok_errors)

    broken = copy.deepcopy(projects)
    filename = context["images"][1]["file_name"]
    broken["classical"]["layouts"][filename]["rois"][0][
        "annotation_approved"
    ] = False
    with pytest.raises(EvaluationError, match="not approved"):
        validate_reviewed_projects(context, proposals, broken, attestation)


def _plan() -> dict:
    return {
        "schema_version": 1,
        "protocol_id": "protocol-1",
        "seed": 19,
        "iou_threshold": 0.5,
        "classical": {"sensitivity": 0.65},
        "gelgenie": {
            "probability_threshold": 0.5,
            "review_threshold": 0.75,
            "device": "cpu",
            "normalization": "auto",
        },
        "local": {
            "score_threshold": 0.5,
            "review_threshold": 0.75,
            "device": "cpu",
        },
        "common": {"recovery_sensitivity": 0.65},
        "densitometry": {
            "background_method": "local",
            "margin": 3,
            "measurement_mode": "rectangle",
        },
    }


def _synthetic_result(context: dict) -> detection.DetectionResult:
    bands = []
    for record in context["ground_truth"][1]:
        x0, y0, x1, y1 = record["bbox"]
        bands.append(
            detection.DetectedBand(
                target=record["target"],
                lane=record["lane"],
                x=int(x0),
                y=int(y0),
                w=int(x1 - x0),
                h=int(y1 - y0),
                confidence=0.9,
            )
        )
    return detection.DetectionResult(
        bands=bands,
        lane_centers=[20, 70],
        target_centers=[17, 53],
        bands_are_dark=True,
        polarity_confidence=1.0,
    )


def _prepare_synthetic_run(
    tmp_path: Path,
    monkeypatch,
    *,
    name: str = "evaluation-1",
) -> dict:
    cohort = _cohort(tmp_path)
    repository = tmp_path / "repo"
    (repository / "runs").mkdir(parents=True)
    (repository / "data").mkdir()
    monkeypatch.setattr(evaluate_detector, "REPOSITORY_ROOT", repository)
    checkpoint = tmp_path / "detector.pt"
    checkpoint.write_bytes(b"checkpoint")
    gelgenie_model = tmp_path / "gelgenie.onnx"
    gelgenie_model.write_bytes(b"model")
    plan_path = tmp_path / "plan.json"
    seal_path = tmp_path / "seal.json"
    ledger_path = tmp_path / "ledger.json"
    _write_json(plan_path, _plan())
    _write_json(seal_path, cohort["seal"])
    _write_json(ledger_path, cohort["ledger"])
    checkpoint_provenance = copy.deepcopy(cohort["checkpoint"])
    checkpoint_provenance["sha256"] = _sha256(checkpoint)
    checkpoint_provenance["embedded_metadata_verified"] = True
    monkeypatch.setattr(
        evaluate_detector.ml_detector,
        "verified_checkpoint_provenance",
        lambda path: {
            **copy.deepcopy(checkpoint_provenance),
            "sha256": _sha256(Path(path)),
        },
    )
    monkeypatch.setattr(
        evaluate_detector.gelgenie,
        "model_provenance",
        lambda path: {
            "verified_official": True,
            "sha256": _sha256(Path(path)),
        },
    )
    monkeypatch.setattr(
        evaluate_detector,
        "_run_backends",
        lambda loaded, image, plan, checkpoint, gelgenie_model: {
            method: _synthetic_result(cohort["context"])
            for method in ("classical", "gelgenie", "local")
        },
    )
    output = repository / "runs" / name
    prepared = evaluate_detector.prepare_run(
        Namespace(
            annotations=cohort["annotations"],
            images=cohort["image_root"],
            seal=seal_path,
            development_ledger=ledger_path,
            plan=plan_path,
            checkpoint=checkpoint,
            gelgenie_model=gelgenie_model,
            output=output,
        )
    )
    return {
        "cohort": cohort,
        "repository": repository,
        "checkpoint": checkpoint,
        "gelgenie_model": gelgenie_model,
        "plan": plan_path,
        "seal": seal_path,
        "ledger": ledger_path,
        "output": output,
        "prepared": prepared,
    }


def _complete_review_packet(bundle: dict, tmp_path: Path) -> tuple[Path, Path]:
    output = bundle["output"]
    reviewed_directory = tmp_path / "reviewed"
    reviewed_directory.mkdir()
    for task_path in (
        output / "reviewer_packet" / "review_tasks"
    ).glob("*.json"):
        project = json.loads(task_path.read_text(encoding="utf-8"))
        assert project["alignment_applied"] == {
            filename: 0.0 for filename in project["layouts"]
        }
        for layout in project["layouts"].values():
            layout["annotation_complete"] = True
            for roi in layout["rois"]:
                if roi.get("use", True):
                    roi["annotation_approved"] = True
        _write_json(reviewed_directory / task_path.name, project)
    attestation_path = tmp_path / "attestation.json"
    _write_json(
        attestation_path,
        {
            "schema_version": 1,
            "protocol_id": "protocol-1",
            "reviewer": "reviewer-1",
            "blinded": True,
            "ground_truth_not_viewed": True,
            "whole_image_reviewed": True,
        },
    )
    return reviewed_directory, attestation_path


def test_prepare_and_score_write_only_machine_readable_ignored_outputs(
    tmp_path, monkeypatch
):
    bundle = _prepare_synthetic_run(tmp_path, monkeypatch)
    output = bundle["output"]
    prepared = bundle["prepared"]
    assert prepared["metrics_computed"] is False
    assert not (output / "summary.json").exists()
    assert (output / "reviewer_packet" / "review_order.csv").is_file()
    private_text = (
        output / "evaluator_private" / "blinding_key.json"
    ).read_text(encoding="utf-8")
    packet_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (output / "reviewer_packet").rglob("*")
        if path.is_file()
    )
    assert any(method in private_text for method in ("classical", "gelgenie", "local"))
    assert all(
        method not in packet_text for method in ("classical", "gelgenie", "local")
    )
    assert "ground_truth" not in packet_text
    blinding = json.loads(private_text)
    assert blinding["run_nonce"] not in packet_text

    reviewed_directory, attestation_path = _complete_review_packet(
        bundle, tmp_path
    )
    scored = evaluate_detector.score_run(
        Namespace(
            run=output,
            reviewed_projects=reviewed_directory,
            review_attestation=attestation_path,
        )
    )
    assert Path(scored["summary"]).is_file()
    assert (output / "score_complete.json").is_file()
    completed_manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert completed_manifest["metrics_computed"] is True
    assert completed_manifest["review_input_sha256"]["projects"]
    suffixes = {
        path.suffix
        for path in output.rglob("*")
        if path.is_file()
    }
    assert suffixes <= {".json", ".csv"}
    with pytest.raises(EvaluationError, match="already been scored"):
        evaluate_detector.score_run(
            Namespace(
                run=output,
                reviewed_projects=reviewed_directory,
                review_attestation=attestation_path,
            )
        )


def test_private_method_order_is_not_derived_from_plan_seed():
    run_nonce = "ab" * 32
    seed_orders = {
        seed: tuple(evaluate_detector._private_method_order(run_nonce))
        for seed in (1, 19, 999_999)
    }
    assert len(set(seed_orders.values())) == 1
    nonce_orders = {
        tuple(evaluate_detector._private_method_order(f"{value:064x}"))
        for value in range(1, 17)
    }
    assert len(nonce_orders) > 1
    assert set(seed_orders[19]) == {"classical", "gelgenie", "local"}


def test_every_frozen_input_and_internal_artifact_is_hash_checked(
    tmp_path, monkeypatch
):
    bundle = _prepare_synthetic_run(tmp_path, monkeypatch)
    output = bundle["output"]
    manifest = evaluate_detector.load_json(output / "run_manifest.json")
    committed_paths = [
        bundle["cohort"]["annotations"],
        bundle["seal"],
        bundle["ledger"],
        bundle["plan"],
        bundle["checkpoint"],
        bundle["gelgenie_model"],
        *[
            output / relative
            for relative in manifest["artifact_sha256"]
        ],
    ]
    for path in committed_paths:
        original = path.read_bytes()
        path.write_bytes(original + b"\nmutation")
        with pytest.raises(EvaluationError, match="changed"):
            evaluate_detector._verify_frozen_run(output, manifest)
        path.write_bytes(original)
    evaluate_detector._verify_frozen_run(output, manifest)


def test_scoring_failure_leaves_retryable_run_and_completion_is_last(
    tmp_path, monkeypatch
):
    bundle = _prepare_synthetic_run(tmp_path, monkeypatch)
    reviewed_directory, attestation_path = _complete_review_packet(
        bundle, tmp_path
    )
    original_atomic_csv = evaluate_detector.atomic_csv
    calls = {"count": 0}

    def fail_during_staging(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("simulated CSV failure")
        return original_atomic_csv(*args, **kwargs)

    monkeypatch.setattr(evaluate_detector, "atomic_csv", fail_during_staging)
    score_args = Namespace(
        run=bundle["output"],
        reviewed_projects=reviewed_directory,
        review_attestation=attestation_path,
    )
    with pytest.raises(RuntimeError, match="simulated CSV failure"):
        evaluate_detector.score_run(score_args)
    assert not (bundle["output"] / "score_complete.json").exists()
    assert not (bundle["output"] / "summary.json").exists()
    assert not list(bundle["output"].glob(".score-staging-*"))

    monkeypatch.setattr(evaluate_detector, "atomic_csv", original_atomic_csv)
    result = evaluate_detector.score_run(score_args)
    assert Path(result["summary"]).is_file()
    assert (bundle["output"] / "score_complete.json").is_file()


def test_summary_aggregate_uses_match_weighted_localization():
    summaries = [
        {
            "true_positive": 1,
            "false_positive": 0,
            "false_negative": 0,
            "matched_iou_sum": 1.0,
            "matched_iou_count": 1,
            "center_error_px_sum": 0.0,
            "center_error_px_count": 1,
            "center_error_normalized_sum": 0.0,
            "center_error_normalized_count": 1,
            "doublet_bands": 0,
            "doublet_bands_matched": 0,
            "doublet_groups": 0,
            "doublet_groups_recovered": 0,
            "artifact_false_positives": {},
        },
        {
            "true_positive": 3,
            "false_positive": 0,
            "false_negative": 0,
            "matched_iou_sum": 1.5,
            "matched_iou_count": 3,
            "center_error_px_sum": 6.0,
            "center_error_px_count": 3,
            "center_error_normalized_sum": 0.6,
            "center_error_normalized_count": 3,
            "doublet_bands": 2,
            "doublet_bands_matched": 1,
            "doublet_groups": 1,
            "doublet_groups_recovered": 0,
            "artifact_false_positives": {},
        },
    ]
    aggregate = evaluate_detector._aggregate(summaries)
    assert aggregate["matched_iou_count"] == 4
    assert aggregate["micro_weighted_mean_matched_iou"] == pytest.approx(0.625)
    assert aggregate["micro_weighted_mean_center_error_px"] == pytest.approx(1.5)
    assert aggregate["micro_doublet_band_recall"] == pytest.approx(0.5)


def test_cli_parser_and_output_restriction(tmp_path, monkeypatch):
    parser = evaluate_detector.build_parser()
    parsed = parser.parse_args(
        [
            "score",
            "--run",
            "runs/example",
            "--reviewed-projects",
            "runs/example/reviewed",
            "--review-attestation",
            "runs/example/attestation.json",
        ]
    )
    assert parsed.command == "score"
    repository = tmp_path / "repo"
    (repository / "runs").mkdir(parents=True)
    (repository / "data").mkdir()
    monkeypatch.setattr(evaluate_detector, "REPOSITORY_ROOT", repository)
    with pytest.raises(EvaluationError, match="runs/ or data/"):
        evaluate_detector._local_output_path(tmp_path / "public", must_exist=False)
