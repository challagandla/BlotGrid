"""Release-check integrity tests for detector checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import check_checkpoint_release
from wblot import ml_detector
from wblot.ml_detector import (
    sanitize_checkpoint_for_release,
    verified_checkpoint_provenance,
)
from wblot.ml_training import PUBLICATION_METADATA_SCHEMA_VERSION


@pytest.fixture
def tiny_detector_runtime(monkeypatch):
    torch = pytest.importorskip("torch")

    class TinyDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(
        ml_detector,
        "_build_model",
        lambda *args, **kwargs: TinyDetector(),
    )
    return torch, TinyDetector


def _write_sidecar(checkpoint: Path, metadata: dict) -> None:
    sidecar = {
        **metadata,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    checkpoint.with_name(checkpoint.name + ".metadata.json").write_text(
        json.dumps(sidecar, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _generated_metadata(*, optimizer_state_included: bool) -> dict:
    return {
        "format_version": 2,
        "architecture": "fasterrcnn_mobilenet_v3_large_fpn",
        "publication_metadata_schema_version": (
            PUBLICATION_METADATA_SCHEMA_VERSION
        ),
        "pretrained_transfer_learning": False,
        "trainable_backbone_layers": 6,
        "trainable_parameter_count": 1,
        "optimizer_state_included": optimizer_state_included,
        "training_config": {
            "optimizer": "SGD",
            "learning_rate": 0.005,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "trainable_backbone_layers": 6,
        },
    }


def _generated_payload(torch, model_type, *, include_optimizer: bool) -> dict:
    model = model_type()
    metadata = _generated_metadata(
        optimizer_state_included=include_optimizer
    )
    payload = {
        "format_version": 2,
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    if include_optimizer:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005,
        )
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        payload["state_dict"] = model.state_dict()
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def test_release_check_rejects_forged_sha_matching_sidecar(
    tmp_path,
    capsys,
    tiny_detector_runtime,
):
    torch, model_type = tiny_detector_runtime
    checkpoint = tmp_path / "detector.pt"
    payload = _generated_payload(torch, model_type, include_optimizer=False)
    embedded = payload["metadata"]
    torch.save(payload, checkpoint)
    forged = {
        **embedded,
        "architecture": "retinanet_resnet50_fpn_v2",
    }
    _write_sidecar(checkpoint, forged)

    assert check_checkpoint_release.main([str(checkpoint)]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["eligible_for_publication"] is False
    assert "does not exactly match" in output["error"]


def test_release_check_accepts_exact_embedded_sidecar_binding(
    tmp_path,
    tiny_detector_runtime,
):
    torch, model_type = tiny_detector_runtime
    checkpoint = tmp_path / "detector.pt"
    payload = _generated_payload(torch, model_type, include_optimizer=False)
    embedded = payload["metadata"]
    torch.save(payload, checkpoint)
    _write_sidecar(checkpoint, embedded)

    provenance = verified_checkpoint_provenance(checkpoint)
    assert provenance["metadata_verified"] is True
    assert provenance["embedded_metadata_verified"] is True
    assert provenance["container_verified"] is True
    assert provenance["verified_trainable_parameter_count"] == 1
    assert provenance["release_artifact_sanitized"] is False


def test_release_check_accepts_generated_sgd_optimizer_state(
    tmp_path,
    tiny_detector_runtime,
):
    torch, model_type = tiny_detector_runtime
    checkpoint = tmp_path / "detector.pt"
    payload = _generated_payload(torch, model_type, include_optimizer=True)
    torch.save(payload, checkpoint)
    _write_sidecar(checkpoint, payload["metadata"])

    provenance = verified_checkpoint_provenance(checkpoint)

    assert provenance["container_verified"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", True, "format"),
        (
            "publication_metadata_schema_version",
            True,
            "literal integer",
        ),
    ],
)
def test_container_schema_markers_require_literal_integers(
    tmp_path,
    tiny_detector_runtime,
    field,
    value,
    message,
):
    torch, model_type = tiny_detector_runtime
    checkpoint = tmp_path / f"{field}.pt"
    payload = _generated_payload(torch, model_type, include_optimizer=False)
    payload["metadata"][field] = value
    if field == "format_version":
        payload["format_version"] = value
    torch.save(payload, checkpoint)
    _write_sidecar(checkpoint, payload["metadata"])

    with pytest.raises(ml_detector.MLDetectorError, match=message):
        verified_checkpoint_provenance(checkpoint)


def test_release_cli_rejects_attribution_output_collision(
    tmp_path,
    capsys,
    tiny_detector_runtime,
):
    torch, model_type = tiny_detector_runtime
    checkpoint = tmp_path / "local.pt"
    payload = _generated_payload(torch, model_type, include_optimizer=False)
    torch.save(payload, checkpoint)
    _write_sidecar(checkpoint, payload["metadata"])
    release = tmp_path / "release.pt"

    assert (
        check_checkpoint_release.main(
            [
                str(checkpoint),
                "--release-output",
                str(release),
                "--attribution-output",
                str(release),
            ]
        )
        == 2
    )
    output = json.loads(capsys.readouterr().out)
    assert "must be distinct" in output["error"]
    assert not release.exists()


@pytest.mark.parametrize("collision", ["checkpoint", "sidecar"])
def test_release_cli_preserves_input_on_attribution_collision(
    tmp_path,
    capsys,
    tiny_detector_runtime,
    collision,
):
    torch, model_type = tiny_detector_runtime
    checkpoint = tmp_path / "local.pt"
    payload = _generated_payload(torch, model_type, include_optimizer=False)
    torch.save(payload, checkpoint)
    _write_sidecar(checkpoint, payload["metadata"])
    sidecar = checkpoint.with_name(checkpoint.name + ".metadata.json")
    before_checkpoint = checkpoint.read_bytes()
    before_sidecar = sidecar.read_bytes()
    attribution = checkpoint if collision == "checkpoint" else sidecar

    assert (
        check_checkpoint_release.main(
            [
                str(checkpoint),
                "--release-output",
                str(tmp_path / "release.pt"),
                "--attribution-output",
                str(attribution),
            ]
        )
        == 2
    )
    output = json.loads(capsys.readouterr().out)
    assert "must be distinct" in output["error"]
    assert checkpoint.read_bytes() == before_checkpoint
    assert sidecar.read_bytes() == before_sidecar


def test_release_cli_refuses_existing_attribution_output(
    tmp_path,
    capsys,
    tiny_detector_runtime,
):
    torch, model_type = tiny_detector_runtime
    checkpoint = tmp_path / "local.pt"
    payload = _generated_payload(torch, model_type, include_optimizer=False)
    torch.save(payload, checkpoint)
    _write_sidecar(checkpoint, payload["metadata"])
    attribution = tmp_path / "existing.json"
    attribution.write_text("preserve me", encoding="utf-8")

    assert (
        check_checkpoint_release.main(
            [
                str(checkpoint),
                "--release-output",
                str(tmp_path / "release.pt"),
                "--attribution-output",
                str(attribution),
            ]
        )
        == 2
    )
    output = json.loads(capsys.readouterr().out)
    assert "must not already exist" in output["error"]
    assert attribution.read_text(encoding="utf-8") == "preserve me"


def test_duplicate_json_sidecar_keys_are_quarantined(
    tmp_path,
    tiny_detector_runtime,
):
    torch, model_type = tiny_detector_runtime
    checkpoint = tmp_path / "duplicate-sidecar.pt"
    payload = _generated_payload(torch, model_type, include_optimizer=False)
    torch.save(payload, checkpoint)
    sidecar = {
        **payload["metadata"],
        "checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
    }
    encoded = json.dumps(sidecar)
    encoded = encoded.replace(
        '"architecture": "fasterrcnn_mobilenet_v3_large_fpn"',
        (
            '"architecture": "patient-123 secret cohort", '
            '"architecture": "fasterrcnn_mobilenet_v3_large_fpn"'
        ),
        1,
    )
    checkpoint.with_name(checkpoint.name + ".metadata.json").write_text(
        encoded,
        encoding="utf-8",
    )

    provenance = ml_detector.checkpoint_provenance(checkpoint)

    assert provenance["metadata_verified"] is not True
    assert provenance["metadata"] == {}
    assert "duplicate JSON object key" in provenance["warning"]


def test_container_rejects_tensor_with_oversized_backing_storage(
    tmp_path,
    tiny_detector_runtime,
):
    torch, model_type = tiny_detector_runtime
    checkpoint = tmp_path / "storage-slack.pt"
    payload = _generated_payload(torch, model_type, include_optimizer=False)
    payload["state_dict"]["weight"] = torch.arange(256.0)[:1]
    torch.save(payload, checkpoint)
    _write_sidecar(checkpoint, payload["metadata"])

    with pytest.raises(ml_detector.MLDetectorError, match="backing storage"):
        verified_checkpoint_provenance(checkpoint)


def test_release_sanitizer_strips_trailing_and_unreferenced_archive_data(
    tmp_path,
    tiny_detector_runtime,
):
    torch, model_type = tiny_detector_runtime
    checkpoint = tmp_path / "local.pt"
    payload = _generated_payload(torch, model_type, include_optimizer=False)
    torch.save(payload, checkpoint)
    with zipfile.ZipFile(checkpoint, "a") as archive:
        archive.writestr("local/private_patient.txt", b"PRIVATE-PATIENT-DATA")
    with checkpoint.open("ab") as handle:
        handle.write(b"PRIVATE-TRAILING-DATA")
    _write_sidecar(checkpoint, payload["metadata"])
    release = tmp_path / "release.pt"

    provenance = sanitize_checkpoint_for_release(checkpoint, release)

    assert provenance["release_artifact_sanitized"] is True
    assert b"PRIVATE-PATIENT-DATA" not in release.read_bytes()
    assert b"PRIVATE-TRAILING-DATA" not in release.read_bytes()
    with zipfile.ZipFile(release) as archive:
        assert all("private_patient" not in name for name in archive.namelist())


@pytest.mark.parametrize(
    ("forgery", "expected_error"),
    [
        ("patient_ids", "top-level fields"),
        ("raw_training_crop", "top-level fields"),
        ("extra_model_state", "model state"),
        ("optimizer_extra_field", "exactly state and param_groups"),
    ],
)
def test_release_cli_rejects_forged_data_bearing_container(
    tmp_path,
    capsys,
    tiny_detector_runtime,
    forgery,
    expected_error,
):
    torch, model_type = tiny_detector_runtime
    include_optimizer = forgery == "optimizer_extra_field"
    payload = _generated_payload(
        torch,
        model_type,
        include_optimizer=include_optimizer,
    )
    if forgery == "patient_ids":
        payload["patient_ids"] = ["patient-001", "patient-002"]
    elif forgery == "raw_training_crop":
        payload["raw_training_crop"] = torch.arange(64, dtype=torch.uint8)
    elif forgery == "extra_model_state":
        payload["state_dict"]["raw_training_crop"] = torch.ones(8)
    else:
        payload["optimizer_state_dict"]["patient_ids"] = ["patient-001"]

    checkpoint = tmp_path / f"{forgery}.pt"
    torch.save(payload, checkpoint)
    _write_sidecar(checkpoint, payload["metadata"])

    assert check_checkpoint_release.main([str(checkpoint)]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["eligible_for_publication"] is False
    assert expected_error in output["error"]
