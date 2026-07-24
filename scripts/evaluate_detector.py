#!/usr/bin/env python3
"""Prepare and score a sealed, blinded BlotGrid detector comparison."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from wblot import detection, gelgenie, image_io, ml_detector  # noqa: E402
from wblot.holdout_evaluation import (  # noqa: E402
    METHODS,
    EvaluationError,
    atomic_csv,
    atomic_json,
    load_json,
    measure_reviewed_errors,
    one_to_one_metrics,
    sha256_file,
    stratified_metrics,
    validate_holdout_inputs,
    validate_reviewed_projects,
)


def _local_output_path(path: Path, *, must_exist: bool) -> Path:
    resolved = path.expanduser().resolve()
    allowed = (
        (REPOSITORY_ROOT / "runs").resolve(),
        (REPOSITORY_ROOT / "data").resolve(),
    )
    if not any(
        resolved == root or root in resolved.parents
        for root in allowed
    ):
        raise EvaluationError(
            "Evaluation output must stay below the repository's ignored "
            "runs/ or data/ directory."
        )
    if must_exist:
        if not resolved.is_dir():
            raise EvaluationError(f"Evaluation run directory does not exist: {resolved}")
    elif resolved.exists():
        raise EvaluationError(
            f"Evaluation output already exists; choose a new run id: {resolved}"
        )
    return resolved


def _number(
    container: Mapping[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(container[key])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError(f"Evaluation plan requires numeric {key!r}.") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise EvaluationError(
            f"Evaluation plan {key!r} must be between {minimum} and {maximum}."
        )
    return value


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema_version") != 1:
        raise EvaluationError("Evaluation plan schema_version must be 1.")
    protocol_id = str(plan.get("protocol_id", "")).strip()
    if not protocol_id:
        raise EvaluationError("Evaluation plan requires protocol_id.")
    try:
        seed = int(plan["seed"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError("Evaluation plan requires an integer seed.") from exc
    iou_threshold = _number(
        plan, "iou_threshold", minimum=0.01, maximum=1.0
    )
    classical = plan.get("classical")
    gelgenie_plan = plan.get("gelgenie")
    local = plan.get("local")
    common = plan.get("common")
    densitometry_plan = plan.get("densitometry")
    for label, value in (
        ("classical", classical),
        ("gelgenie", gelgenie_plan),
        ("local", local),
        ("common", common),
        ("densitometry", densitometry_plan),
    ):
        if not isinstance(value, Mapping):
            raise EvaluationError(f"Evaluation plan {label!r} must be an object.")
    normalized = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "seed": seed,
        "iou_threshold": iou_threshold,
        "classical": {
            "sensitivity": _number(
                classical, "sensitivity", minimum=0.0, maximum=1.0
            ),
        },
        "gelgenie": {
            "probability_threshold": _number(
                gelgenie_plan,
                "probability_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
            "review_threshold": _number(
                gelgenie_plan,
                "review_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
            "device": str(gelgenie_plan.get("device", "auto")),
            "normalization": str(gelgenie_plan.get("normalization", "auto")),
        },
        "local": {
            "score_threshold": _number(
                local, "score_threshold", minimum=0.0, maximum=1.0
            ),
            "review_threshold": _number(
                local, "review_threshold", minimum=0.0, maximum=1.0
            ),
            "device": str(local.get("device", "auto")),
        },
        "common": {
            "recovery_sensitivity": _number(
                common,
                "recovery_sensitivity",
                minimum=0.0,
                maximum=1.0,
            ),
        },
        "densitometry": {
            "background_method": str(
                densitometry_plan.get("background_method", "local")
            ),
            "margin": int(densitometry_plan.get("margin", 4)),
            "rolling_ball_radius": float(
                densitometry_plan.get("rolling_ball_radius", 50.0)
            ),
            "fixed_background": float(
                densitometry_plan.get("fixed_background", 0.0)
            ),
            "measurement_mode": str(
                densitometry_plan.get("measurement_mode", "rectangle")
            ),
            "multiband": "sum",
        },
    }
    if (
        normalized["gelgenie"]["review_threshold"]
        < normalized["gelgenie"]["probability_threshold"]
    ):
        raise EvaluationError(
            "GelGenie review_threshold cannot be below probability_threshold."
        )
    if (
        normalized["local"]["review_threshold"]
        < normalized["local"]["score_threshold"]
    ):
        raise EvaluationError(
            "Local review_threshold cannot be below score_threshold."
        )
    if normalized["densitometry"]["background_method"] not in {
        "local",
        "rolling_ball",
        "fixed",
    }:
        raise EvaluationError("Unsupported densitometry background method.")
    if normalized["densitometry"]["measurement_mode"] not in {
        "rectangle",
        "profile",
    }:
        raise EvaluationError("Unsupported densitometry measurement mode.")
    if normalized["densitometry"]["margin"] < 0:
        raise EvaluationError("Densitometry margin cannot be negative.")
    return normalized


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _empty_result(image: Mapping[str, Any], warning: str) -> detection.DetectionResult:
    return detection.DetectionResult(
        bands=[],
        lane_centers=list(image["lane_centers_hint"]),
        target_centers=list(image["target_centers_hint"]),
        bands_are_dark=bool(image["bands_are_dark"]),
        polarity_confidence=1.0,
        warnings=[warning],
    )


def _no_detection_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "no usable band components" in message
        or "no valid band predictions" in message
    )


def _band_record(band: detection.DetectedBand) -> dict[str, Any]:
    return {
        "bbox": [
            float(band.x),
            float(band.y),
            float(band.x + band.w),
            float(band.y + band.h),
        ],
        "target": str(band.target),
        "lane": str(band.lane),
        "score": float(band.confidence),
        "source": str(band.source),
        "status": str(band.status),
        "review_required": bool(band.review_required),
        "use": bool(band.use),
    }


def _active_predictions(result: detection.DetectionResult) -> list[dict[str, Any]]:
    return [_band_record(band) for band in result.bands if band.use]


def _common_review_result(
    result: detection.DetectionResult,
    image: Mapping[str, Any],
    gray,
    *,
    sensitivity: float,
) -> detection.DetectionResult:
    reviewed = copy.deepcopy(result)
    detection.ensure_complete_grid(
        reviewed,
        image["targets"],
        image["lanes"],
        tuple(gray.shape),
        gray=gray,
        sensitivity=sensitivity,
    )
    width = int(image["review_roi_size"]["w"])
    height = int(image["review_roi_size"]["h"])
    image_height, image_width = gray.shape
    for band in reviewed.bands:
        center_x = band.x + band.w / 2.0
        center_y = band.y + band.h / 2.0
        band.w = min(width, image_width)
        band.h = min(height, image_height)
        band.x = int(
            max(0, min(round(center_x - band.w / 2.0), image_width - band.w))
        )
        band.y = int(
            max(0, min(round(center_y - band.h / 2.0), image_height - band.h))
        )
    return reviewed


def _run_backends(
    loaded: image_io.LoadedImage,
    image: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    checkpoint: Path,
    gelgenie_model: Path,
) -> dict[str, detection.DetectionResult]:
    polarity = "dark" if image["bands_are_dark"] else "light"
    common = {
        "lane_centers_hint": image["lane_centers_hint"],
        "target_centers_hint": image["target_centers_hint"],
        "polarity": polarity,
    }
    results = {
        "classical": detection.detect_bands(
            loaded.gray,
            image["targets"],
            image["lanes"],
            sensitivity=plan["classical"]["sensitivity"],
            **common,
        )
    }
    try:
        results["gelgenie"] = gelgenie.predict_bands(
            loaded.gray,
            image["targets"],
            image["lanes"],
            gelgenie_model,
            max_value=loaded.max_value,
            probability_threshold=plan["gelgenie"]["probability_threshold"],
            review_threshold=plan["gelgenie"]["review_threshold"],
            device=plan["gelgenie"]["device"],
            normalization=plan["gelgenie"]["normalization"],
            **common,
        )
    except gelgenie.GelGenieError as exc:
        if not _no_detection_error(exc):
            raise
        results["gelgenie"] = _empty_result(image, str(exc))
    try:
        results["local"] = ml_detector.predict_bands(
            loaded.gray,
            image["targets"],
            image["lanes"],
            checkpoint,
            score_threshold=plan["local"]["score_threshold"],
            review_threshold=plan["local"]["review_threshold"],
            device=plan["local"]["device"],
            **common,
        )
    except ml_detector.MLDetectorError as exc:
        if not _no_detection_error(exc):
            raise
        results["local"] = _empty_result(image, str(exc))
    return results


def _random_task_id() -> str:
    return f"task-{secrets.token_hex(16)}"


def _private_method_order(run_nonce: str) -> list[str]:
    """Derive a pseudorandom method order from the evaluator-private nonce."""
    try:
        key = bytes.fromhex(run_nonce)
    except ValueError as exc:
        raise EvaluationError("Evaluator run_nonce is not valid hexadecimal.") from exc
    if len(key) != 32:
        raise EvaluationError("Evaluator run_nonce must contain 256 random bits.")
    return sorted(
        METHODS,
        key=lambda method: hmac.new(
            key,
            b"blotgrid-review-order-v1\0" + method.encode("utf-8"),
            hashlib.sha256,
        ).digest(),
    )


def _review_project(
    task_id: str,
    protocol_id: str,
    proposal_sha256: str,
    task_binding_sha256: str,
    context: Mapping[str, Any],
    proposal_images: Mapping[str, Any],
) -> dict[str, Any]:
    images = {int(key): value for key, value in context["images"].items()}
    first = images[min(images)]
    targets = list(
        dict.fromkeys(
            str(target)
            for _, image in sorted(images.items())
            for target in image["targets"]
        )
    )
    lanes = list(
        dict.fromkeys(
            str(lane)
            for _, image in sorted(images.items())
            for lane in image["lanes"]
        )
    )
    layouts = {}
    provenance = {}
    study_group = {}
    alignment_applied = {}
    for image_id, image in sorted(images.items()):
        rois = []
        for index, proposal in enumerate(
            proposal_images[str(image_id)]["review_rois"]
        ):
            x0, y0, x1, y1 = proposal["bbox"]
            rois.append(
                {
                    "roi_id": f"{task_id}-roi-{image_id}-{index}",
                    "x": int(round(x0)),
                    "y": int(round(y0)),
                    "w": int(round(x1 - x0)),
                    "h": int(round(y1 - y0)),
                    "target": proposal["target"],
                    "lane": proposal["lane"],
                    "confidence": None,
                    "source": "blinded proposal",
                    "status": "review required",
                    "review_required": True,
                    "annotation_approved": False,
                    "use": bool(proposal.get("use", True)),
                }
            )
        layouts[image["file_name"]] = {
            "mode": "sealed_holdout_review",
            "bands_dark": image["bands_are_dark"],
            "annotation_complete": False,
            "negative_image": False,
            "rois": rois,
        }
        provenance[image["file_name"]] = {"sha256": image["sha256"]}
        study_group[image["file_name"]] = image["study_group"]
        alignment_applied[image["file_name"]] = 0.0
    return {
        "evaluation_task": {
            "schema_version": 1,
            "task_id": task_id,
            "protocol_id": protocol_id,
            "proposal_sha256": proposal_sha256,
            "task_binding_sha256": task_binding_sha256,
            "blinded": True,
        },
        "targets": targets,
        "lanes": lanes,
        "control": first["control_target"],
        "image_provenance": provenance,
        "alignment_applied": alignment_applied,
        "study_group": study_group,
        "layouts": layouts,
    }


def _serializable_context(context: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(context, allow_nan=False))


def prepare_run(args: argparse.Namespace) -> dict[str, Any]:
    output = _local_output_path(args.output, must_exist=False)
    plan = validate_plan(load_json(args.plan))
    coco = load_json(args.annotations)
    seal = load_json(args.seal)
    ledger = load_json(args.development_ledger)
    checkpoint_provenance = ml_detector.verified_checkpoint_provenance(
        args.checkpoint
    )
    context = validate_holdout_inputs(
        coco,
        args.annotations,
        args.images,
        seal,
        ledger,
        checkpoint_provenance,
    )
    gelgenie_provenance = gelgenie.model_provenance(args.gelgenie_model)
    if not gelgenie_provenance.get("verified_official"):
        raise EvaluationError("GelGenie model is not the verified pinned artifact.")

    input_sha256 = {
        "annotations": sha256_file(args.annotations),
        "seal": sha256_file(args.seal),
        "development_ledger": sha256_file(args.development_ledger),
        "plan": sha256_file(args.plan),
        "checkpoint": checkpoint_provenance["sha256"],
        "checkpoint_metadata": _canonical_sha256(
            checkpoint_provenance.get("metadata", {})
        ),
        "gelgenie_model": gelgenie_provenance["sha256"],
        "gelgenie_provenance": _canonical_sha256(gelgenie_provenance),
    }
    run_nonce = secrets.token_hex(32)
    frozen_binding = {
        "schema_version": 1,
        "protocol_id": plan["protocol_id"],
        "input_sha256": input_sha256,
        "seal_verification": context["seal"],
    }
    proposals: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": plan["protocol_id"],
        "run_nonce": run_nonce,
        "run_binding_sha256": _canonical_sha256(frozen_binding),
        "methods": {},
    }
    images = {int(key): value for key, value in context["images"].items()}
    for method in METHODS:
        proposals["methods"][method] = {
            "schema_version": 1,
            "images": {},
        }
    for image_id, image in sorted(images.items()):
        loaded = image_io.load_image(image["path"], name=image["file_name"])
        results = _run_backends(
            loaded,
            image,
            plan,
            checkpoint=args.checkpoint,
            gelgenie_model=args.gelgenie_model,
        )
        for method, result in results.items():
            review_result = _common_review_result(
                result,
                image,
                loaded.gray,
                sensitivity=plan["common"]["recovery_sensitivity"],
            )
            proposals["methods"][method]["images"][str(image_id)] = {
                "direct": _active_predictions(result),
                "pre_review": _active_predictions(review_result),
                "review_rois": [_band_record(band) for band in review_result.bands],
                "warnings": list(result.warnings),
            }

    for method in METHODS:
        method_payload = proposals["methods"][method]
        task_id = _random_task_id()
        proposal_sha256 = _canonical_sha256(method_payload["images"])
        task_binding_sha256 = _canonical_sha256(
            {
                "schema_version": 1,
                "run_nonce": run_nonce,
                "run_binding_sha256": proposals["run_binding_sha256"],
                "task_id": task_id,
                "protocol_id": plan["protocol_id"],
                "method": method,
                "proposal_sha256": proposal_sha256,
            }
        )
        method_payload.update(
            {
                "task_id": task_id,
                "proposal_sha256": proposal_sha256,
                "task_binding_sha256": task_binding_sha256,
            }
        )

    order = _private_method_order(run_nonce)
    output.mkdir(parents=True)
    private = output / "evaluator_private"
    reviewer_packet = output / "reviewer_packet"
    review_tasks = reviewer_packet / "review_tasks"
    private.mkdir()
    review_tasks.mkdir(parents=True)
    blinding = {
        "schema_version": 1,
        "protocol_id": plan["protocol_id"],
        "run_nonce": run_nonce,
        "run_binding_sha256": proposals["run_binding_sha256"],
        "tasks": {},
    }
    review_rows = []
    for position, method in enumerate(order, start=1):
        method_payload = proposals["methods"][method]
        task_id = method_payload["task_id"]
        project = _review_project(
            task_id,
            plan["protocol_id"],
            method_payload["proposal_sha256"],
            method_payload["task_binding_sha256"],
            context,
            method_payload["images"],
        )
        atomic_json(review_tasks / f"{task_id}.json", project)
        blinding["tasks"][task_id] = method
        review_rows.append({"review_position": position, "task_id": task_id})
    atomic_json(private / "holdout_context.json", _serializable_context(context))
    atomic_json(private / "seal_verification.json", context["seal"])
    atomic_json(
        private / "detector_provenance.json",
        {
            "local": checkpoint_provenance,
            "gelgenie": gelgenie_provenance,
            "classical": {"backend": "BlotGrid classical detector"},
        },
    )
    atomic_json(private / "proposals.json", proposals)
    atomic_json(private / "blinding_key.json", blinding)
    atomic_csv(
        reviewer_packet / "review_order.csv",
        review_rows,
        fieldnames=("review_position", "task_id"),
    )
    frozen_artifacts = [
        private / "holdout_context.json",
        private / "seal_verification.json",
        private / "detector_provenance.json",
        private / "proposals.json",
        private / "blinding_key.json",
        reviewer_packet / "review_order.csv",
        *sorted(review_tasks.glob("*.json")),
    ]
    artifact_sha256 = {
        str(path.relative_to(output)): sha256_file(path)
        for path in frozen_artifacts
    }
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": plan["protocol_id"],
        "plan": plan,
        "inputs": {
            "annotations": str(args.annotations.expanduser().resolve()),
            "images": str(args.images.expanduser().resolve()),
            "seal": str(args.seal.expanduser().resolve()),
            "development_ledger": str(
                args.development_ledger.expanduser().resolve()
            ),
            "plan": str(args.plan.expanduser().resolve()),
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "gelgenie_model": str(args.gelgenie_model.expanduser().resolve()),
        },
        "input_sha256": input_sha256,
        "artifact_sha256": artifact_sha256,
        "run_binding_sha256": proposals["run_binding_sha256"],
        "reviewer_packet": "reviewer_packet",
        "metrics_computed": False,
    }
    atomic_json(output / "run_manifest.json", manifest)
    return {
        "output": str(output),
        "reviewer_packet": str(reviewer_packet),
        "images": len(images),
        "review_tasks": len(METHODS),
        "metrics_computed": False,
    }


def _load_reviewed_projects(
    directory: Path,
    blinding: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if not directory.is_dir():
        raise EvaluationError(f"Reviewed project directory is missing: {directory}")
    tasks = blinding.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks.values()) != set(METHODS):
        raise EvaluationError("Blinding key is invalid.")
    expected = {f"{task_id}.json" for task_id in tasks}
    actual = {path.name for path in directory.glob("*.json")}
    if actual != expected:
        raise EvaluationError(
            "Reviewed project directory must contain exactly one JSON per task."
        )
    return {
        str(method): load_json(directory / f"{task_id}.json")
        for task_id, method in tasks.items()
    }


def _verify_frozen_run(
    run: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    if manifest.get("schema_version") != 1:
        raise EvaluationError("Run manifest schema_version must be 1.")
    inputs = manifest.get("inputs")
    expected_hashes = manifest.get("input_sha256")
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(inputs, Mapping) or not isinstance(
        expected_hashes, Mapping
    ) or not isinstance(artifact_hashes, Mapping):
        raise EvaluationError("Run manifest lacks its frozen input commitments.")
    required_inputs = {
        "annotations",
        "images",
        "seal",
        "development_ledger",
        "plan",
        "checkpoint",
        "gelgenie_model",
    }
    if not required_inputs <= set(inputs):
        raise EvaluationError("Run manifest is missing a frozen input path.")
    paths = {
        key: Path(str(inputs[key])).expanduser().resolve()
        for key in required_inputs
    }
    for key in ("annotations", "seal", "development_ledger", "plan"):
        try:
            actual = sha256_file(paths[key])
        except OSError as exc:
            raise EvaluationError(f"Frozen input {key!r} is unreadable.") from exc
        if actual != expected_hashes.get(key):
            raise EvaluationError(f"Frozen input {key!r} changed after prepare.")

    checkpoint_provenance = ml_detector.verified_checkpoint_provenance(
        paths["checkpoint"]
    )
    if (
        checkpoint_provenance.get("sha256") != expected_hashes.get("checkpoint")
        or _canonical_sha256(checkpoint_provenance.get("metadata", {}))
        != expected_hashes.get("checkpoint_metadata")
    ):
        raise EvaluationError(
            "Frozen local checkpoint or its verified metadata changed after prepare."
        )
    gelgenie_provenance = gelgenie.model_provenance(paths["gelgenie_model"])
    if (
        not gelgenie_provenance.get("verified_official")
        or gelgenie_provenance.get("sha256")
        != expected_hashes.get("gelgenie_model")
        or _canonical_sha256(gelgenie_provenance)
        != expected_hashes.get("gelgenie_provenance")
    ):
        raise EvaluationError(
            "Frozen GelGenie artifact or provenance changed after prepare."
        )

    required_artifacts = {
        "evaluator_private/holdout_context.json",
        "evaluator_private/seal_verification.json",
        "evaluator_private/detector_provenance.json",
        "evaluator_private/proposals.json",
        "evaluator_private/blinding_key.json",
        "reviewer_packet/review_order.csv",
    }
    if not required_artifacts <= set(str(key) for key in artifact_hashes):
        raise EvaluationError("Run manifest lacks required artifact commitments.")
    for relative, expected in artifact_hashes.items():
        candidate = (run / str(relative)).resolve()
        try:
            candidate.relative_to(run)
        except ValueError as exc:
            raise EvaluationError("Frozen artifact path escapes the run.") from exc
        try:
            actual = sha256_file(candidate)
        except OSError as exc:
            raise EvaluationError(
                f"Frozen run artifact {relative!r} is unreadable."
            ) from exc
        if actual != expected:
            raise EvaluationError(
                f"Frozen run artifact {relative!r} changed after prepare."
            )
    return paths, checkpoint_provenance, gelgenie_provenance


def _aggregate(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(int(item["true_positive"]) for item in summaries)
    fp = sum(int(item["false_positive"]) for item in summaries)
    fn = sum(int(item["false_negative"]) for item in summaries)
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    result = {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
    }
    for visibility in ("clear", "faint"):
        total = sum(int(item.get(f"{visibility}_bands", 0)) for item in summaries)
        matched = sum(
            int(item.get(f"{visibility}_bands_matched", 0)) for item in summaries
        )
        result[f"{visibility}_bands"] = total
        result[f"{visibility}_bands_matched"] = matched
        result[f"{visibility}_recall"] = matched / total if total else None
    for legacy_field, micro_field, sum_field, count_field in (
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
    ):
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
    doublet_bands = sum(
        int(item.get("doublet_bands", 0)) for item in summaries
    )
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
    artifact_counts: dict[str, int] = {}
    for item in summaries:
        for label, count in item["artifact_false_positives"].items():
            artifact_counts[label] = artifact_counts.get(label, 0) + int(count)
    result["artifact_false_positives"] = artifact_counts
    return result


def _review_edits(
    proposals: Mapping[str, Any],
    reviewed: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for image_id, payload in proposals["methods"][method]["images"].items():
            before = payload["pre_review"]
            after = reviewed[method].get(int(image_id), [])
            rows.append(
                {
                    "method": method,
                    "image_id": int(image_id),
                    "active_proposals": len(before),
                    "reviewed_active_rois": len(after),
                    "net_active_change": len(after) - len(before),
                }
            )
    return rows


def score_run(args: argparse.Namespace) -> dict[str, Any]:
    run = _local_output_path(args.run, must_exist=True)
    if (run / "score_complete.json").exists():
        raise EvaluationError("This evaluation run has already been scored.")
    manifest = load_json(run / "run_manifest.json")
    paths, checkpoint_provenance, _ = _verify_frozen_run(run, manifest)
    private = run / "evaluator_private"
    proposals = load_json(private / "proposals.json")
    blinding = load_json(private / "blinding_key.json")
    attestation = load_json(args.review_attestation)
    annotations = paths["annotations"]
    image_root = paths["images"]
    seal_path = paths["seal"]
    ledger_path = paths["development_ledger"]
    plan_path = paths["plan"]
    plan = validate_plan(load_json(plan_path))
    if plan != manifest.get("plan"):
        raise EvaluationError("Run manifest plan differs from the frozen plan.")
    if any(
        payload.get("protocol_id") != plan["protocol_id"]
        for payload in (manifest, proposals, blinding, attestation)
    ):
        raise EvaluationError(
            "Plan, run, proposals, blinding key, and attestation protocol differ."
        )
    if (
        proposals.get("schema_version") != 1
        or blinding.get("schema_version") != 1
        or attestation.get("schema_version") != 1
        or proposals.get("run_binding_sha256")
        != manifest.get("run_binding_sha256")
        or blinding.get("run_binding_sha256")
        != manifest.get("run_binding_sha256")
    ):
        raise EvaluationError("Run protocol or binding metadata is invalid.")
    context = validate_holdout_inputs(
        load_json(annotations),
        annotations,
        image_root,
        load_json(seal_path),
        load_json(ledger_path),
        checkpoint_provenance,
    )
    if _serializable_context(context) != load_json(
        private / "holdout_context.json"
    ):
        raise EvaluationError(
            "Revalidated holdout context differs from the frozen prepared context."
        )
    reviewed_directory = args.reviewed_projects.expanduser().resolve()
    reviewed_projects = _load_reviewed_projects(
        reviewed_directory,
        blinding,
    )
    review_input_sha256 = {
        "attestation": sha256_file(args.review_attestation),
        "projects": {
            str(task_id): sha256_file(reviewed_directory / f"{task_id}.json")
            for task_id in blinding["tasks"]
        },
    }
    reviewed = validate_reviewed_projects(
        context,
        proposals,
        reviewed_projects,
        attestation,
    )
    images = {int(key): value for key, value in context["images"].items()}
    ground_truth = {
        int(key): value for key, value in context["ground_truth"].items()
    }
    artifacts = {int(key): value for key, value in context["artifacts"].items()}
    records = []
    per_image_rows = []
    match_rows = []
    false_positive_rows = []
    summaries: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        method: {phase: [] for phase in ("direct", "pre_review", "post_review")}
        for method in METHODS
    }
    for method in METHODS:
        for image_id, image in sorted(images.items()):
            phases = {
                "direct": proposals["methods"][method]["images"][str(image_id)][
                    "direct"
                ],
                "pre_review": proposals["methods"][method]["images"][str(image_id)][
                    "pre_review"
                ],
                "post_review": reviewed[method].get(image_id, []),
            }
            for phase, predictions in phases.items():
                result = one_to_one_metrics(
                    predictions,
                    ground_truth.get(image_id, []),
                    artifacts.get(image_id, []),
                    iou_threshold=plan["iou_threshold"],
                )
                summary = result["summary"]
                summaries[method][phase].append(summary)
                per_image_rows.append(
                    {
                        "method": method,
                        "phase": phase,
                        "image_id": image_id,
                        "file_name": image["file_name"],
                        **summary,
                        **image["strata"],
                    }
                )
                for row in result["matches"]:
                    match_rows.append(
                        {
                            "method": method,
                            "phase": phase,
                            "image_id": image_id,
                            **row,
                        }
                    )
                for row in result["false_positives"]:
                    false_positive_rows.append(
                        {
                            "method": method,
                            "phase": phase,
                            "image_id": image_id,
                            **row,
                        }
                    )
                records.append(
                    {
                        "method": method,
                        "phase": phase,
                        "image_id": image_id,
                        "predictions": predictions,
                        "ground_truth": ground_truth.get(image_id, []),
                        "artifacts": artifacts.get(image_id, []),
                        "strata": image["strata"],
                    }
                )
    summary = {
        "schema_version": 1,
        "protocol_id": plan["protocol_id"],
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "methods": {
            method: {
                phase: _aggregate(summaries[method][phase])
                for phase in summaries[method]
            }
            for method in METHODS
        },
        "paired_review_protocol": {
            "reviewer": attestation["reviewer"],
            "protocol_id": attestation["protocol_id"],
            "blinded": True,
        },
        "review_input_sha256": review_input_sha256,
    }
    strata_rows = stratified_metrics(
        records, iou_threshold=plan["iou_threshold"]
    )
    downstream = measure_reviewed_errors(
        context,
        image_root,
        reviewed,
        plan,
    )
    staging = Path(tempfile.mkdtemp(prefix=".score-staging-", dir=run))
    output_names = (
        "per_image_metrics.csv",
        "band_matches.csv",
        "false_positives.csv",
        "stratified_metrics.csv",
        "review_edits.csv",
        "intensities.csv",
        "fold_changes.csv",
        "paired_errors.csv",
        "review_input_hashes.json",
        "summary.json",
    )
    try:
        atomic_csv(staging / "per_image_metrics.csv", per_image_rows)
        atomic_csv(staging / "band_matches.csv", match_rows)
        atomic_csv(staging / "false_positives.csv", false_positive_rows)
        atomic_csv(staging / "stratified_metrics.csv", strata_rows)
        atomic_csv(
            staging / "review_edits.csv",
            _review_edits(proposals, reviewed),
        )
        atomic_csv(staging / "intensities.csv", downstream["intensities"])
        atomic_csv(staging / "fold_changes.csv", downstream["fold_changes"])
        atomic_csv(staging / "paired_errors.csv", downstream["paired_errors"])
        atomic_json(staging / "review_input_hashes.json", review_input_sha256)
        atomic_json(staging / "summary.json", summary)
        output_sha256 = {
            name: sha256_file(staging / name) for name in output_names
        }
        for name in output_names:
            os.replace(staging / name, run / name)
        completed_manifest = dict(manifest)
        completed_manifest.update(
            {
                "metrics_computed": True,
                "scored_utc": summary["scored_utc"],
                "review_input_sha256": review_input_sha256,
                "score_output_sha256": output_sha256,
            }
        )
        atomic_json(run / "run_manifest.json", completed_manifest)
        atomic_json(
            run / "score_complete.json",
            {
                "schema_version": 1,
                "protocol_id": plan["protocol_id"],
                "scored_utc": summary["scored_utc"],
                "run_manifest_sha256": sha256_file(run / "run_manifest.json"),
                "score_output_sha256": output_sha256,
            },
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "run": str(run),
        "images": len(images),
        "methods": len(METHODS),
        "summary": str(run / "summary.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare blinded detector-review tasks or score a completed sealed "
            "Western-blot holdout comparison."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="verify the seal and create blinded review tasks"
    )
    prepare.add_argument("--annotations", type=Path, required=True)
    prepare.add_argument("--images", type=Path, required=True)
    prepare.add_argument("--seal", type=Path, required=True)
    prepare.add_argument("--development-ledger", type=Path, required=True)
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--gelgenie-model", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(handler=prepare_run)

    score = subparsers.add_parser(
        "score", help="validate completed reviews and compute frozen metrics"
    )
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--reviewed-projects", type=Path, required=True)
    score.add_argument("--review-attestation", type=Path, required=True)
    score.set_defaults(handler=score_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except (
        EvaluationError,
        gelgenie.GelGenieError,
        ml_detector.MLDetectorError,
        OSError,
        ValueError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
