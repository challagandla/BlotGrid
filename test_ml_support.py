"""Pure tests for optional ML support (no PyTorch installation required)."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from wblot.ml_detector import (
    CHECKPOINT_FORMAT_VERSION,
    DEFAULT_ARCHITECTURE,
    MAX_CHECKPOINT_METADATA_BYTES,
    MLDetectorError,
    MAX_TRAINABLE_BACKBONE_LAYERS,
    SUPPORTED_ARCHITECTURES,
    TRANSFER_TRAINABLE_BACKBONE_LAYERS,
    build_model,
    build_model_with_provenance,
    checkpoint_provenance,
    cluster_axis_centers,
    map_boxes_to_grid,
    normalize_grayscale,
    torchvision_initialization_provenance,
    validate_initialization_provenance,
)
from wblot import gelgenie, ml_detector
from scripts import download_pretrained_models
from wblot.ml_training import (
    CocoBandDataset,
    MLTrainingError,
    PUBLICATION_METADATA_SCHEMA_VERSION,
    _annotation_is_pseudo,
    _checkpoint_paths,
    _checkpoint_sidecar_preflight,
    _epoch_seed,
    _extract_dataset_license_lineage,
    _save_checkpoint,
    _training_environment,
    _validate_resume_metadata,
    _validated_resume_optimizer_state,
    box_iou_matrix,
    checkpoint_publication_assessment,
    detection_metrics,
    export_project_to_coco,
    image_leakage_groups,
    make_train_val_split,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _licensed_coco(filename: str = "dataset.png", sha256: str = "a" * 64) -> dict:
    attribution = "Example Dataset Authors (2026), doi:10.1234/example"
    return {
        "info": {
            "title": "Licensed test dataset",
            "description": "This description is deliberately not published.",
            "source_dois": ["10.1234/example"],
        },
        "licenses": [
            {
                "id": 7,
                "name": "Creative Commons Attribution 4.0",
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
        ],
        "blotgrid": {
            "license_lineage_schema_version": 1,
            "sources": [
                {
                    "id": "example_v1",
                    "doi": "10.1234/example",
                    "url": "https://doi.org/10.1234/example",
                    "license_id": 7,
                    "required_attribution": attribution,
                    "publication_assessment": {
                        "status": "cleared",
                        "redistribution_cleared": True,
                        "attribution_resolved": True,
                        "license_id": 7,
                    },
                }
            ],
        },
        "images": [
            {
                "id": 1,
                "file_name": filename,
                "width": 40,
                "height": 30,
                "license": 7,
                "blotgrid": {
                    "annotation_complete": True,
                    "negative_image": False,
                    "original_sha256": sha256,
                    "working_pixels_exported": False,
                    "source_id": "example_v1",
                    "source_doi": "10.1234/example",
                    "source_url": "https://doi.org/10.1234/example",
                    "required_attribution": attribution,
                    "source_archive": "example-v1.zip",
                    "source_image": "images/original.png",
                    "study_group": "Example study",
                },
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 4, 12, 8],
                "attributes": {
                    "annotation_approved": True,
                    "pseudo_label": False,
                    "review_required": False,
                },
            }
        ],
        "categories": [{"id": 1, "name": "band"}],
    }


def test_ml_modules_import_without_importing_torch():
    code = (
        "import sys; import wblot.ml_detector, wblot.ml_training; "
        "assert 'torch' not in sys.modules; print('ok')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "ok"


def test_new_ai_modules_keep_optional_runtimes_lazy():
    code = (
        "import sys; import wblot.gelgenie, wblot.sam_refiner; "
        "assert 'onnxruntime' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'sam2' not in sys.modules; print('ok')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "ok"


def test_gelgenie_manifest_and_preprocessing_contract():
    assert gelgenie.MODEL_REVISION == "fc36cc1ea468308b7336939a66c9884a26752d7e"
    assert gelgenie.MODEL_SIZE_BYTES == 57_279_486
    assert gelgenie.MODEL_SHA256 == (
        "10ea972a40625f03b5a2987e1a98b532c6d4ded1daae32f1eb91aa24db8d9c48"
    )
    image = np.zeros((32, 33), dtype=np.float32)
    image[8:12, 5:28] = 255.0
    tensor, padding, normalization = gelgenie.prepare_model_input(
        image,
        max_value=255.0,
        invert=True,
    )
    assert tensor.shape == (1, 1, 64, 64)
    assert padding == (16, 16, 15, 16)
    assert normalization == "dtype_max"
    restored = tensor[0, 0, 16:48, 15:48]
    assert restored[0, 0] == pytest.approx(1.0)
    assert restored[9, 10] == pytest.approx(0.0)

    sixteen_bit = np.arange(64 * 65, dtype=np.float32).reshape(64, 65)
    _, _, normalization = gelgenie.prepare_model_input(
        sixteen_bit,
        max_value=65535.0,
        invert=False,
    )
    assert normalization == "percentile"


def test_gelgenie_components_keep_vertical_doublets_separate():
    original_shape = (32, 33)
    padding = (16, 16, 15, 16)
    logits = np.zeros((1, 2, 64, 64), dtype=np.float32)
    top, _, left, _ = padding
    logits[0, 1, top + 5:top + 8, left + 5:left + 25] = 8.0
    logits[0, 1, top + 10:top + 13, left + 5:left + 25] = 8.0
    logits[0, 0] = 1.0
    boxes, scores, summary = gelgenie.segmentation_components(
        logits,
        original_shape=original_shape,
        padding=padding,
    )
    assert boxes == [[5.0, 5.0, 25.0, 8.0], [5.0, 10.0, 25.0, 13.0]]
    assert len(scores) == 2
    assert all(score > 0.99 for score in scores)
    assert summary["accepted_components"] == 2


def test_gelgenie_rejects_unverified_onnx(tmp_path):
    model = tmp_path / "unverified.onnx"
    model.write_bytes(b"x" * 2048)
    provenance = gelgenie.model_provenance(model)
    assert provenance["verified_official"] is False
    with pytest.raises(gelgenie.GelGenieError, match="unverified"):
        gelgenie._verified_model(model)


def test_pinned_downloader_is_bounded_verified_and_atomic(tmp_path, monkeypatch):
    content = b"verified model bytes"
    expected_sha256 = hashlib.sha256(content).hexdigest()

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        download_pretrained_models.urllib.request,
        "urlopen",
        lambda request, timeout: Response(content),
    )
    destination, status = download_pretrained_models._download_pinned(
        tmp_path / "model.onnx",
        url="https://models.example/verified",
        expected_size=len(content),
        expected_sha256=expected_sha256,
        label="test model",
        force=False,
    )
    assert status == "downloaded"
    assert destination.read_bytes() == content
    assert not (tmp_path / "model.onnx.part").exists()
    _, status = download_pretrained_models._download_pinned(
        destination,
        url="https://models.example/verified",
        expected_size=len(content),
        expected_sha256=expected_sha256,
        label="test model",
        force=False,
    )
    assert status == "already_verified"


def test_pinned_downloader_removes_truncated_partial(tmp_path, monkeypatch):
    content = b"short"

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        download_pretrained_models.urllib.request,
        "urlopen",
        lambda request, timeout: Response(content),
    )
    destination = tmp_path / "truncated.pt"
    with pytest.raises(RuntimeError, match="size"):
        download_pretrained_models._download_pinned(
            destination,
            url="https://models.example/truncated",
            expected_size=len(content) + 1,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            label="test model",
            force=False,
        )
    assert not destination.exists()
    assert not (tmp_path / "truncated.pt.part").exists()


@pytest.mark.parametrize(
    "architecture",
    ["fcos_resnet50_fpn", "retinanet_resnet50_fpn_v2"],
)
def test_optional_dense_detectors_have_two_class_logits(architecture):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    assert architecture in SUPPORTED_ARCHITECTURES
    model = build_model(architecture, pretrained=False)
    classification_head = model.head.classification_head
    anchors = model.anchor_generator.num_anchors_per_location()[0]
    assert classification_head.num_classes == 2
    assert classification_head.cls_logits.out_channels == anchors * 2


def test_transfer_and_scratch_backbone_trainability_are_explicit():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    transfer = build_model(
        DEFAULT_ARCHITECTURE,
        pretrained=False,
        trainable_backbone_layers=TRANSFER_TRAINABLE_BACKBONE_LAYERS,
        freeze_backbone_batch_norm=True,
    )
    scratch = build_model(
        DEFAULT_ARCHITECTURE,
        pretrained=False,
        trainable_backbone_layers=MAX_TRAINABLE_BACKBONE_LAYERS[
            DEFAULT_ARCHITECTURE
        ],
    )
    transfer_names = {
        name for name, parameter in transfer.named_parameters()
        if parameter.requires_grad
    }
    scratch_names = {
        name for name, parameter in scratch.named_parameters()
        if parameter.requires_grad
    }
    assert transfer_names < scratch_names
    assert len(transfer_names) == 70


def test_robust_grayscale_normalization_handles_nonfinite_and_rgb():
    image = np.zeros((12, 14, 3), dtype=np.float64)
    image[..., 0] = np.arange(14)
    image[0, 0, 0] = np.nan
    normalized = normalize_grayscale(image)
    assert normalized.shape == (12, 14)
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert 0.0 <= normalized.min() <= normalized.max() <= 1.0


def test_box_mapping_clusters_grid_and_keeps_doublets():
    boxes = [
        [8, 7, 22, 13],
        [78, 7, 92, 13],
        [8, 47, 22, 53],
        [78, 45, 92, 50],
        [78, 51, 92, 56],
    ]
    assignments, lanes, targets, warnings = map_boxes_to_grid(
        boxes,
        ["Top", "Bottom"],
        ["L1", "L2"],
        (70, 100),
    )
    assert lanes == [15, 85]
    assert targets[0] < targets[1]
    assert assignments.count(("Bottom", "L2")) == 2
    assert warnings == []
    fallback_centers, fallback = cluster_axis_centers([20], 3, 100)
    assert fallback is True
    assert len(fallback_centers) == 3


def test_box_iou_and_validation_metrics_are_one_to_one():
    ious = box_iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])
    assert ious.shape == (1, 1)
    assert ious[0, 0] == pytest.approx(1 / 3)
    metrics = detection_metrics(
        [
            {
                "pred_boxes": [[0, 0, 10, 10], [0, 0, 10, 10]],
                "pred_scores": [0.9, 0.8],
                "gt_boxes": [[0, 0, 10, 10]],
            }
        ]
    )
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_checkpoint_provenance_reads_only_safe_sidecar(tmp_path):
    checkpoint = tmp_path / "bands.pt"
    checkpoint.write_bytes(b"not a pickle and never loaded by provenance")
    digest = _sha256(checkpoint)
    sidecar = tmp_path / "bands.pt.metadata.json"
    sidecar.write_text(
        json.dumps(
            {
                "architecture": "fasterrcnn_mobilenet_v3_large_fpn",
                "epoch": 4,
                "checkpoint_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    provenance = checkpoint_provenance(checkpoint)
    assert provenance["sha256"] == digest
    assert provenance["architecture"] == "fasterrcnn_mobilenet_v3_large_fpn"
    assert provenance["metadata_verified"] is True
    assert provenance["size_bytes"] == checkpoint.stat().st_size

    sidecar.write_text(
        json.dumps(
            {
                "architecture": "fasterrcnn_resnet50_fpn_v2",
                "checkpoint_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    quarantined = checkpoint_provenance(checkpoint)
    assert quarantined["metadata_verified"] is False
    assert quarantined["metadata"] == {}
    assert "architecture" not in quarantined
    assert "quarantined" in quarantined["warning"]

    sidecar.write_text(
        json.dumps(
            {
                "architecture": "fasterrcnn_resnet50_fpn_v2",
                "epoch": 99,
            }
        ),
        encoding="utf-8",
    )
    unsigned = checkpoint_provenance(checkpoint)
    assert unsigned["metadata_verified"] is None
    assert unsigned["metadata"] == {}
    assert "architecture" not in unsigned
    assert "no checkpoint_sha256" in unsigned["warning"]


def _fake_torchvision() -> SimpleNamespace:
    detection = SimpleNamespace()
    for spec in ml_detector._TORCHVISION_WEIGHT_SPECS.values():
        weights = SimpleNamespace(
            name=spec["enum_member"],
            url=spec["url"],
            meta={"recipe": spec["recipe_url"]},
        )
        setattr(
            detection,
            spec["enum_class"],
            SimpleNamespace(**{spec["enum_member"]: weights}),
        )
    return SimpleNamespace(
        __version__="0.22.1+test",
        version=SimpleNamespace(
            git_version="59a3e1f9f78cfe44cb989877cc6f4ea77c8a75ca"
        ),
        models=SimpleNamespace(detection=detection),
    )


@pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
def test_torchvision_initialization_provenance_pins_concrete_coco_v1(architecture):
    spec = ml_detector._TORCHVISION_WEIGHT_SPECS[architecture]
    provenance = torchvision_initialization_provenance(
        architecture,
        pretrained=True,
        torchvision=_fake_torchvision(),
    )
    assert provenance["weights_enum"] == (
        f"{spec['enum_class']}.{spec['enum_member']}"
    )
    assert provenance["weights_revision"] == "COCO_V1"
    assert provenance["weights_url"] == spec["url"]
    assert provenance["url_sha256_prefix"] == spec["sha256_prefix"]
    assert provenance["recipe_url"] == spec["recipe_url"]
    assert provenance["torchvision_git_revision"].startswith("59a3e1f9")
    assert provenance["artifact_sha256"] is None
    assert provenance["license_assessment"]["status"] == "review_required"
    assert provenance["license_assessment"]["redistribution_cleared"] is False


def test_torchvision_artifact_requires_full_hash_and_exact_size(
    tmp_path, monkeypatch
):
    architecture = DEFAULT_ARCHITECTURE
    spec = ml_detector._TORCHVISION_WEIGHT_SPECS[architecture]
    prefix = str(spec["sha256_prefix"])
    exact_digest = prefix + "b" * (64 - len(prefix))
    artifact = tmp_path / "weights.pth"
    artifact.write_bytes(b"test weights")
    monkeypatch.setitem(spec, "sha256", exact_digest)
    monkeypatch.setitem(spec, "size_bytes", len(b"test weights"))
    monkeypatch.setattr(
        ml_detector,
        "_sha256_file",
        lambda path: exact_digest,
    )
    provenance = torchvision_initialization_provenance(
        architecture,
        pretrained=True,
        torchvision=_fake_torchvision(),
        artifact_path=artifact,
    )
    assert provenance["artifact_sha256"] == exact_digest
    assert provenance["artifact_size_bytes"] == len(b"test weights")
    assert provenance["artifact_verified"] is True

    monkeypatch.setattr(
        ml_detector,
        "_sha256_file",
        lambda path: prefix + "a" * (64 - len(prefix)),
    )
    with pytest.raises(MLDetectorError, match="full SHA-256 and byte size"):
        torchvision_initialization_provenance(
            architecture,
            pretrained=True,
            torchvision=_fake_torchvision(),
            artifact_path=artifact,
        )


def test_scratch_initialization_has_no_third_party_weight_lineage():
    provenance = torchvision_initialization_provenance(
        DEFAULT_ARCHITECTURE,
        pretrained=False,
        torchvision=_fake_torchvision(),
    )
    assert provenance["mode"] == "random_initialization"
    assert provenance["weights_enum"] is None
    assert provenance["weights_url"] is None
    assert provenance["artifact_sha256"] is None
    assert provenance["license_assessment"]["status"] == "not_applicable"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.update(
            initialization_scope="patient-123 secret cohort"
        ),
        lambda record: record["license_assessment"].update(
            reason="patient-123 secret cohort"
        ),
        lambda record: record.update(
            recipe_url="https://example.test/private-recipe"
        ),
    ],
)
def test_scratch_initialization_requires_exact_generated_record(mutation):
    record = torchvision_initialization_provenance(
        DEFAULT_ARCHITECTURE,
        pretrained=False,
        torchvision=_fake_torchvision(),
    )
    mutation(record)
    with pytest.raises(MLDetectorError):
        validate_initialization_provenance(
            record,
            DEFAULT_ARCHITECTURE,
            torchvision_version="0.22.1+test",
            torchvision_git_revision=(
                "59a3e1f9f78cfe44cb989877cc6f4ea77c8a75ca"
            ),
        )


def test_installed_torchvision_exports_every_pinned_enum_without_download():
    torchvision = pytest.importorskip("torchvision")
    for architecture in SUPPORTED_ARCHITECTURES:
        record = torchvision_initialization_provenance(
            architecture,
            pretrained=True,
            torchvision=torchvision,
        )
        validate_initialization_provenance(
            record,
            architecture,
            torchvision_version=torchvision.__version__,
            torchvision_git_revision=torchvision.version.git_version,
            require_verified_artifact=False,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("architecture", "fcos_resnet50_fpn"),
        ("weights_enum", "WrongWeights.COCO_V1"),
        ("weights_revision", "DEFAULT"),
        ("weights_url", "https://example.test/weights.pth"),
        ("url_sha256_prefix", "00000000"),
        ("expected_artifact_sha256", "0" * 64),
        ("expected_artifact_size_bytes", 1),
        ("artifact_sha256", "0" * 64),
        ("artifact_size_bytes", 1),
        ("torchvision_version", "0.0"),
        ("torchvision_git_revision", "0" * 40),
        ("recipe_url", "https://example.test/private-recipe"),
        ("training_dataset", "private cohort"),
        ("initialization_scope", "patient-123 secret cohort"),
        (
            "license_assessment",
            {"status": "cleared", "redistribution_cleared": True},
        ),
    ],
)
def test_pretrained_initialization_mutations_fail_closed(field, replacement):
    architecture = DEFAULT_ARCHITECTURE
    spec = ml_detector._TORCHVISION_WEIGHT_SPECS[architecture]
    record = torchvision_initialization_provenance(
        architecture,
        pretrained=True,
        torchvision=_fake_torchvision(),
    )
    record["artifact_sha256"] = spec["sha256"]
    record["artifact_size_bytes"] = spec["size_bytes"]
    record["artifact_verified"] = True
    record[field] = replacement
    with pytest.raises(MLDetectorError):
        validate_initialization_provenance(
            record,
            architecture,
            torchvision_version="0.22.1+test",
            torchvision_git_revision=(
                "59a3e1f9f78cfe44cb989877cc6f4ea77c8a75ca"
            ),
        )


def test_v1_checkpoint_remains_restricted_load_inference_compatible(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "v1.pt"
    torch.save(
        {
            "format_version": 1,
            "state_dict": {"weight": torch.tensor([1.0])},
            "metadata": {"architecture": DEFAULT_ARCHITECTURE},
        },
        checkpoint,
    )
    state, metadata, payload = ml_detector._load_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    assert payload["format_version"] == 1
    assert metadata["architecture"] == DEFAULT_ARCHITECTURE
    assert torch.equal(state["weight"], torch.tensor([1.0]))


def test_real_format2_scratch_checkpoint_strict_round_trip_and_inference(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    model, initialization = build_model_with_provenance(
        DEFAULT_ARCHITECTURE,
        pretrained=False,
    )
    checkpoint = tmp_path / "format2.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "state_dict": model.state_dict(),
            "metadata": {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "architecture": DEFAULT_ARCHITECTURE,
                "initialization_provenance": initialization,
            },
        },
        checkpoint,
    )
    state, metadata, payload = ml_detector._load_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert metadata["initialization_provenance"]["mode"] == (
        "random_initialization"
    )
    restored = build_model(DEFAULT_ARCHITECTURE, pretrained=False)
    restored.load_state_dict(state, strict=True)
    restored.eval()
    restored.transform.min_size = (64,)
    restored.transform.max_size = 64
    with torch.inference_mode():
        output = restored([torch.zeros((3, 64, 64), dtype=torch.float32)])
    assert len(output) == 1
    assert set(output[0]) >= {"boxes", "labels", "scores"}


def test_checkpoint_best_last_sidecars_round_trip_license_lineage(
    tmp_path, monkeypatch
):
    saved_payloads = []

    def fake_save(payload, destination):
        saved_payloads.append(payload)
        Path(destination).write_bytes(b"restricted checkpoint payload")

    fake_torch = SimpleNamespace(save=fake_save)
    monkeypatch.setattr(
        ml_detector,
        "_require_ml_runtime",
        lambda: (fake_torch, SimpleNamespace()),
    )
    lineage = _extract_dataset_license_lineage(_licensed_coco())
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": DEFAULT_ARCHITECTURE,
        "dataset_license_lineage": lineage,
        "initialization_provenance": torchvision_initialization_provenance(
            DEFAULT_ARCHITECTURE,
            pretrained=False,
            torchvision=_fake_torchvision(),
        ),
    }
    model = SimpleNamespace(state_dict=lambda: {"weight": 1})
    optimizer = SimpleNamespace(state_dict=lambda: {"momentum": 0.9})
    checkpoints = [tmp_path / "detector.pt", tmp_path / "detector.last.pt"]
    for checkpoint in checkpoints:
        _save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            metadata=metadata,
        )
        provenance = checkpoint_provenance(checkpoint)
        assert provenance["metadata_verified"] is True
        assert provenance["dataset_license_lineage"] == lineage
        assert provenance["initialization_provenance"]["mode"] == (
            "random_initialization"
        )
    assert [payload["format_version"] for payload in saved_payloads] == [
        CHECKPOINT_FORMAT_VERSION,
        CHECKPOINT_FORMAT_VERSION,
    ]
    assert all(
        payload["metadata"]["dataset_license_lineage"] == lineage
        for payload in saved_payloads
    )


def test_checkpoint_writer_rejects_unreadable_oversized_sidecar(
    tmp_path, monkeypatch
):
    fake_torch = SimpleNamespace(
        save=lambda payload, destination: Path(destination).write_bytes(b"unused")
    )
    monkeypatch.setattr(
        ml_detector,
        "_require_ml_runtime",
        lambda: (fake_torch, SimpleNamespace()),
    )
    checkpoint = tmp_path / "oversized.pt"
    with pytest.raises(MLTrainingError, match="1 MiB"):
        _save_checkpoint(
            checkpoint,
            model=SimpleNamespace(state_dict=lambda: {}),
            optimizer=SimpleNamespace(state_dict=lambda: {}),
            metadata={"oversized": "x" * MAX_CHECKPOINT_METADATA_BYTES},
        )
    assert not checkpoint.exists()
    assert not (tmp_path / "oversized.pt.metadata.json").exists()


def test_checkpoint_metadata_preflight_reserves_long_history_before_training():
    history = [
        {
            "epoch": index,
            "train_loss": 0.123456789,
            "true_positive": 1000,
            "false_positive": 100,
            "false_negative": 10,
            "precision": 0.9,
            "recall": 0.99,
            "f1": 0.94,
            "mean_matched_iou": 0.81,
        }
        for index in range(500)
    ]
    sidecar = _checkpoint_sidecar_preflight(
        {"history": history, "dataset": "x" * (256 * 1024)},
        include_optimizer=True,
    )
    assert len(sidecar["history"]) == 500
    with pytest.raises(MLTrainingError, match="1 MiB"):
        _checkpoint_sidecar_preflight(
            {"history": history, "dataset": "x" * MAX_CHECKPOINT_METADATA_BYTES},
            include_optimizer=True,
        )


def test_epoch_seeds_are_stable_and_resume_independent():
    uninterrupted = [_epoch_seed(1337, epoch) for epoch in range(3)]
    resumed_epoch_two = _epoch_seed(1337, 2)
    assert uninterrupted[2] == resumed_epoch_two
    assert len(set(uninterrupted)) == 3
    assert uninterrupted != [_epoch_seed(1338, epoch) for epoch in range(3)]


def test_explicit_split_rejects_image_leakage():
    images = {
        1: {"id": 1, "file_name": "a.tif"},
        2: {"id": 2, "file_name": "b.tif"},
    }
    with pytest.raises(MLTrainingError, match="leakage"):
        make_train_val_split(
            images,
            split={"train": [1, 2], "val": [2]},
        )
    train, val = make_train_val_split(
        images,
        split={"train": ["a.tif"], "val": ["b.tif"]},
    )
    assert train == [1]
    assert val == [2]


def test_automatic_split_requires_every_study_group():
    images = {
        1: {
            "id": 1,
            "file_name": "grouped.tif",
            "blotgrid": {"study_group": "Study A"},
        },
        2: {"id": 2, "file_name": "missing.tif", "blotgrid": {}},
    }
    with pytest.raises(MLTrainingError, match="non-empty study_group"):
        make_train_val_split(images)
    train, val = make_train_val_split(
        images,
        split={"train": ["grouped.tif"], "val": ["missing.tif"]},
    )
    assert train == [1]
    assert val == [2]


def test_checkpoint_paths_keep_best_and_last_distinct(tmp_path):
    best, last = _checkpoint_paths(tmp_path / "detector.pt")
    assert best == (tmp_path / "detector.pt").resolve()
    assert last == (tmp_path / "detector.last.pt").resolve()


def test_training_environment_records_runtime_and_source_hashes():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import torch
    import torchvision

    metadata = _training_environment(
        torch, torchvision, torch.device("cpu")
    )
    assert metadata["torch_version"] == torch.__version__
    assert metadata["torchvision_version"] == torchvision.__version__
    assert metadata["torchvision_git_revision"]
    assert metadata["device"] == "cpu"
    assert metadata["device_name"] == "CPU"
    assert metadata["platform"]["system"] == "Linux"
    assert set(metadata["platform"]) == {
        "system",
        "release",
        "machine",
        "libc_name",
        "libc_version",
    }
    assert metadata["deterministic_algorithms"] is True
    assert metadata["epoch_seed_scheme"].startswith("sha256(")
    assert "executable" not in metadata
    assert set(metadata["source_sha256"]) == {
        "wblot/ml_training.py",
        "wblot/ml_detector.py",
        "scripts/train_detector.py",
        "requirements-ml.txt",
    }
    assert all(
        len(digest) == 64 for digest in metadata["source_sha256"].values()
    )


def test_resume_contract_allows_only_epoch_extension():
    config = {
        "architecture": "fasterrcnn_mobilenet_v3_large_fpn",
        "epochs_requested": 2,
        "batch_size": 2,
        "learning_rate": 0.005,
        "pretrained_transfer_learning_requested": False,
    }
    environment = {
        "python_version": "3.10.12",
        "torch_version": "2.7.1+cu118",
        "torchvision_version": "0.22.1+cu118",
        "torchvision_git_revision": "59a3e1f",
        "cuda_build": "11.8",
        "device": "cuda",
        "source_sha256": {"wblot/ml_training.py": "a" * 64},
    }
    lineage = _extract_dataset_license_lineage(_licensed_coco())
    initialization = torchvision_initialization_provenance(
        DEFAULT_ARCHITECTURE,
        pretrained=False,
        torchvision=_fake_torchvision(),
    )
    initialization["torchvision_version"] = environment["torchvision_version"]
    initialization["torchvision_git_revision"] = environment[
        "torchvision_git_revision"
    ]
    metrics = {
        "images": 1,
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "mean_matched_iou": 1.0,
        "score_threshold": 0.5,
        "iou_threshold": 0.5,
    }
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "training_config": config,
        "training_environment": environment,
        "dataset_license_lineage": lineage,
        "initialization_provenance": initialization,
        "train_image_digests": [lineage["images"][0]["image_digest"]],
        "val_image_digests": [],
        "optimizer_state_included": True,
        "epoch": 0,
        "best_epoch": 0,
        "best_metrics": metrics,
        "current_metrics": metrics,
        "history": [{"epoch": 0, "train_loss": 0.1, **metrics}],
    }
    resumed = dict(config, epochs_requested=20)
    _validate_resume_metadata(
        metadata,
        resumed,
        environment,
        dataset_license_lineage=lineage,
    )
    with pytest.raises(MLTrainingError, match="learning_rate"):
        _validate_resume_metadata(
            metadata,
            dict(resumed, learning_rate=0.01),
            environment,
            dataset_license_lineage=lineage,
        )
    with pytest.raises(MLTrainingError, match="torch_version"):
        _validate_resume_metadata(
            metadata,
            resumed,
            dict(environment, torch_version="2.8.0"),
            dataset_license_lineage=lineage,
        )

    changed_coco = _licensed_coco()
    changed_coco["blotgrid"]["sources"][0]["required_attribution"] = "New credit"
    changed_coco["images"][0]["blotgrid"]["required_attribution"] = "New credit"
    with pytest.raises(MLTrainingError, match="licence lineage differs"):
        _validate_resume_metadata(
            metadata,
            resumed,
            environment,
            dataset_license_lineage=_extract_dataset_license_lineage(changed_coco),
        )
    with pytest.raises(MLTrainingError, match="inference-only"):
        _validate_resume_metadata(
            dict(metadata, format_version=1),
            resumed,
            environment,
            dataset_license_lineage=lineage,
        )
    without_initialization = dict(metadata)
    without_initialization.pop("initialization_provenance")
    with pytest.raises(MLTrainingError, match="initialization provenance"):
        _validate_resume_metadata(
            without_initialization,
            resumed,
            environment,
            dataset_license_lineage=lineage,
        )

    invalid_continuations = [
        (
            {key: value for key, value in metadata.items()
             if key != "optimizer_state_included"},
            "optimizer state",
        ),
        (
            {key: value for key, value in metadata.items() if key != "epoch"},
            "completed epoch",
        ),
        (dict(metadata, history=[]), "training history"),
        (
            dict(
                metadata,
                history=[{"epoch": 1, "train_loss": 0.1, **metrics}],
            ),
            "contiguous continuation",
        ),
        (
            {key: value for key, value in metadata.items()
             if key != "best_metrics"},
            "best validation metrics",
        ),
        (
            {
                key: value
                for key, value in metadata.items()
                if key != "best_epoch"
            },
            "best validation epoch",
        ),
        (
            dict(
                metadata,
                current_metrics={
                    **metrics,
                    "false_positive": 1,
                    "precision": 0.5,
                    "f1": 2.0 / 3.0,
                },
            ),
            "final history record",
        ),
    ]
    for invalid_metadata, message in invalid_continuations:
        with pytest.raises(MLTrainingError, match=message):
            _validate_resume_metadata(
                invalid_metadata,
                resumed,
                environment,
                dataset_license_lineage=lineage,
            )


def test_resume_optimizer_state_is_mandatory_and_structurally_complete():
    optimizer_state = {
        "state": {},
        "param_groups": [{"params": [0], "lr": 0.005}],
    }
    assert _validated_resume_optimizer_state(
        {"optimizer_state_dict": optimizer_state}
    ) is optimizer_state

    with pytest.raises(MLTrainingError, match="lacks optimizer state"):
        _validated_resume_optimizer_state({})
    with pytest.raises(MLTrainingError, match="incomplete or malformed"):
        _validated_resume_optimizer_state(
            {"optimizer_state_dict": {"state": {}}}
        )
    with pytest.raises(MLTrainingError, match="parameter groups"):
        _validated_resume_optimizer_state(
            {
                "optimizer_state_dict": {
                    "state": {},
                    "param_groups": [],
                }
            }
        )


def test_split_keeps_study_and_checksum_groups_together():
    images = {
        1: {
            "id": 1,
            "file_name": "a1.tif",
            "blotgrid": {"study_group": "Study A", "original_sha256": "a" * 64},
        },
        2: {
            "id": 2,
            "file_name": "a2.tif",
            "blotgrid": {"study_group": "Study A", "original_sha256": "b" * 64},
        },
        3: {
            "id": 3,
            "file_name": "b.tif",
            "blotgrid": {"study_group": "Study B", "original_sha256": "c" * 64},
        },
        4: {
            "id": 4,
            "file_name": "duplicate-b.tif",
            "blotgrid": {"study_group": "Study C", "original_sha256": "c" * 64},
        },
        5: {
            "id": 5,
            "file_name": "d.tif",
            "blotgrid": {"study_group": "Study D", "original_sha256": "d" * 64},
        },
    }
    assert image_leakage_groups(images) == [[1, 2], [3, 4], [5]]
    train, val = make_train_val_split(images, seed=0)
    assert ({1, 2} <= set(train)) or ({1, 2} <= set(val))
    assert ({3, 4} <= set(train)) or ({3, 4} <= set(val))
    with pytest.raises(MLTrainingError, match="study_group"):
        make_train_val_split(
            images,
            split={"train": [1, 3, 5], "val": [2, 4]},
        )
    train, val = make_train_val_split(
        images,
        seed=0,
        train_only_ids={1},
    )
    assert {1, 2} <= set(train)
    assert not ({1, 2} & set(val))


def _sample_project(filename: str, sha256: str, angle: float = 0.0) -> dict:
    return {
        "_saved_at": "2026-07-23T10:00:00",
        "targets": ["AKT"],
        "lanes": ["Control"],
        "control": "",
        "alignment_applied": {filename: angle},
        "image_provenance": {filename: {"sha256": sha256}},
        "study_group": {filename: "Study One"},
        "layouts": {
            filename: {
                "mode": "automatic",
                "bands_dark": True,
                "annotation_complete": True,
                "negative_image": False,
                "rois": [
                    {
                        "roi_id": "approved",
                        "use": True,
                        "annotation_approved": True,
                        "source": "detected",
                        "status": "accepted",
                        "target": "AKT",
                        "lane": "Control",
                        "x": 4,
                        "y": 5,
                        "w": 12,
                        "h": 5,
                    },
                    {
                        "roi_id": "pseudo",
                        "use": True,
                        "annotation_approved": False,
                        "source": "detected",
                        "status": "accepted",
                        "target": "AKT",
                        "lane": "Control",
                        "x": 4,
                        "y": 15,
                        "w": 12,
                        "h": 5,
                    },
                    {
                        "roi_id": "inactive",
                        "use": False,
                        "annotation_approved": True,
                        "source": "manual",
                        "target": "AKT",
                        "lane": "Control",
                        "x": 4,
                        "y": 22,
                        "w": 12,
                        "h": 5,
                    },
                    {
                        "roi_id": "fallback",
                        "use": True,
                        "annotation_approved": True,
                        "source": "fallback",
                        "target": "AKT",
                        "lane": "Control",
                        "x": 20,
                        "y": 5,
                        "w": 12,
                        "h": 5,
                    },
                ],
            }
        },
    }


def test_project_export_defaults_to_approved_nonfallback_active_boxes(tmp_path):
    image_path = tmp_path / "blot.tif"
    Image.fromarray(np.arange(40 * 50, dtype=np.uint16).reshape(40, 50)).save(
        image_path
    )
    project = _sample_project(image_path.name, _sha256(image_path))
    output = tmp_path / "bands.json"
    coco = export_project_to_coco(project, tmp_path, output)
    assert len(coco["images"]) == 1
    assert [item["attributes"]["roi_id"] for item in coco["annotations"]] == [
        "approved"
    ]
    summary = coco["blotgrid"]["export_summary"]
    assert summary["skipped_unreviewed"] == 1
    assert summary["skipped_inactive"] == 1
    assert summary["skipped_fallback"] == 1
    assert summary["dropped_unattested_empty_images"] == 0
    assert coco["images"][0]["blotgrid"]["study_group"] == "Study One"
    assert coco["blotgrid"]["annotation_policy"] == "human_approved_only"
    with pytest.raises(MLTrainingError, match="already exists"):
        export_project_to_coco(project, tmp_path, output)


def test_project_export_marks_opt_in_pseudo_labels(tmp_path):
    image_path = tmp_path / "blot.png"
    Image.fromarray(np.arange(40 * 50, dtype=np.uint8).reshape(40, 50)).save(
        image_path
    )
    project = _sample_project(image_path.name, _sha256(image_path))
    coco = export_project_to_coco(
        project,
        tmp_path,
        include_unreviewed=True,
    )
    assert len(coco["annotations"]) == 2
    pseudo = [
        item for item in coco["annotations"] if item["attributes"]["pseudo_label"]
    ]
    assert len(pseudo) == 1
    assert pseudo[0]["attributes"]["annotation_approved"] is False
    assert coco["blotgrid"]["pseudo_labels_included"] is True
    assert coco["blotgrid"]["export_summary"]["included_pseudo_labels"] == 1


def test_export_drops_unattested_empty_images_and_keeps_attested_negatives(tmp_path):
    image_path = tmp_path / "empty.png"
    Image.fromarray(np.zeros((24, 30), dtype=np.uint8)).save(image_path)
    project = _sample_project(image_path.name, _sha256(image_path))
    project["layouts"][image_path.name]["rois"] = []

    dropped = export_project_to_coco(project, tmp_path)
    assert dropped["images"] == []
    assert dropped["annotations"] == []
    assert (
        dropped["blotgrid"]["export_summary"]["dropped_unattested_empty_images"]
        == 1
    )

    layout = project["layouts"][image_path.name]
    layout["annotation_complete"] = True
    layout["negative_image"] = True
    negative = export_project_to_coco(project, tmp_path)
    assert len(negative["images"]) == 1
    assert negative["annotations"] == []
    assert negative["images"][0]["blotgrid"]["negative_image"] is True
    assert negative["blotgrid"]["export_summary"]["attested_negative_images"] == 1


def test_export_requires_saved_image_sha_by_default(tmp_path):
    image_path = tmp_path / "legacy.png"
    Image.fromarray(np.zeros((24, 30), dtype=np.uint8)).save(image_path)
    project = _sample_project(image_path.name, _sha256(image_path))
    project.pop("image_provenance")
    with pytest.raises(MLTrainingError, match="no saved SHA-256"):
        export_project_to_coco(project, tmp_path)
    coco = export_project_to_coco(
        project,
        tmp_path,
        allow_unverified=True,
    )
    assert len(coco["images"]) == 1
    assert coco["images"][0]["blotgrid"]["provenance_verified"] is False


def test_export_requires_complete_whole_image_review_for_positives(tmp_path):
    image_path = tmp_path / "incomplete.png"
    Image.fromarray(np.zeros((24, 30), dtype=np.uint8)).save(image_path)
    project = _sample_project(image_path.name, _sha256(image_path))
    project["layouts"][image_path.name]["annotation_complete"] = False
    with pytest.raises(MLTrainingError, match="annotation_complete"):
        export_project_to_coco(project, tmp_path)
    provisional = export_project_to_coco(
        project,
        tmp_path,
        allow_incomplete_annotations=True,
    )
    assert len(provisional["annotations"]) == 1
    assert (
        provisional["blotgrid"]["export_summary"]["incomplete_positive_images"]
        == 1
    )


def test_project_export_verifies_hash_and_recreates_aligned_pixels(tmp_path):
    image_path = tmp_path / "angled.tif"
    array = np.full((40, 50), 220, dtype=np.uint8)
    array[18:22, 8:42] = 20
    Image.fromarray(array).save(image_path)
    original_digest = _sha256(image_path)
    project = _sample_project(image_path.name, original_digest, angle=2.0)
    output = tmp_path / "aligned.json"
    coco = export_project_to_coco(project, tmp_path, output)
    exported = tmp_path / "aligned_images" / image_path.name
    assert exported.is_file()
    image_metadata = coco["images"][0]["blotgrid"]
    assert image_metadata["provenance_verified"] is True
    assert image_metadata["alignment_applied_degrees"] == 2.0
    assert image_metadata["original_sha256"] == original_digest
    assert image_metadata["exported_working_sha256"] == _sha256(exported)
    assert _sha256(image_path) == original_digest

    project["image_provenance"][image_path.name]["sha256"] = "0" * 64
    with pytest.raises(MLTrainingError, match="provenance mismatch"):
        export_project_to_coco(
            project,
            tmp_path,
            tmp_path / "mismatch.json",
        )


def test_coco_dataset_preserves_validated_deterministic_license_lineage(tmp_path):
    image_path = tmp_path / "dataset.png"
    Image.fromarray(np.zeros((30, 40), dtype=np.uint8)).save(image_path)
    payload = _licensed_coco(image_path.name, _sha256(image_path))
    dataset = CocoBandDataset(payload, tmp_path, require_checksums=True)
    lineage = dataset.license_lineage
    image_lineage = lineage["images"][0]

    assert lineage["dataset_info"] == {
        "title": payload["info"]["title"],
        "dois": payload["info"]["source_dois"],
    }
    assert lineage["licenses"] == payload["licenses"]
    assert lineage["sources"] == [
        {
            "id": "example_v1",
            "doi": "10.1234/example",
            "license_id": 7,
            "required_attribution": (
                "Example Dataset Authors (2026), doi:10.1234/example"
            ),
            "publication_assessment": {
                "status": "cleared",
                "redistribution_cleared": True,
                "attribution_resolved": True,
                "license_id": 7,
            },
        }
    ]
    assert image_lineage["license_id"] == 7
    assert image_lineage["source_id"] == "example_v1"
    assert image_lineage["content_sha256"] == _sha256(image_path)
    assert len(image_lineage["image_digest"]) == 64
    assert "file_name" not in image_lineage
    assert "source_identity" not in image_lineage
    assert lineage["sha256"] == _extract_dataset_license_lineage(payload)["sha256"]

    changed = json.loads(json.dumps(payload))
    changed["blotgrid"]["sources"][0]["required_attribution"] = "Changed credit"
    changed["images"][0]["blotgrid"]["required_attribution"] = "Changed credit"
    assert _extract_dataset_license_lineage(changed)["sha256"] != lineage["sha256"]


def test_coco_lineage_drops_private_and_unknown_metadata():
    payload = _licensed_coco()
    payload["info"].update(
        {
            "email": "private@example.test",
            "internal_note": "do not publish",
        }
    )
    payload["blotgrid"]["sources"][0].update(
        {
            "private_path": "/private/study",
            "access_token": "secret",
        }
    )
    payload["images"][0]["blotgrid"].update(
        {
            "source_mask": "/private/mask.png",
            "free_text_note": "patient-like identifier",
        }
    )
    lineage = _extract_dataset_license_lineage(payload)
    encoded = json.dumps(lineage)
    assert "This description is deliberately not published." not in encoded
    for forbidden in (
        "private@example.test",
        "do not publish",
        "/private/study",
        "secret",
        "/private/mask.png",
        "patient-like identifier",
        "dataset.png",
    ):
        assert forbidden not in encoded


@pytest.mark.parametrize("schema_marker", [2, True])
def test_coco_lineage_rejects_unknown_schema_marker(schema_marker):
    payload = _licensed_coco()
    payload["blotgrid"]["license_lineage_schema_version"] = schema_marker
    with pytest.raises(MLTrainingError, match="schema marker"):
        _extract_dataset_license_lineage(payload)


def _publication_provenance(*, pretrained: bool = False) -> dict:
    environment = {
        "python_version": "3.12.0",
        "python_implementation": "CPython",
        "platform": {
            "system": "Linux",
            "release": "6.8.0-test",
            "machine": "x86_64",
            "libc_name": "glibc",
            "libc_version": "2.39",
        },
        "torch_version": "2.7.1+test",
        "torchvision_version": "0.22.1+test",
        "torchvision_git_revision": (
            "59a3e1f9f78cfe44cb989877cc6f4ea77c8a75ca"
        ),
        "numpy_version": "2.2.0",
        "pillow_version": "11.0.0",
        "cuda_build": None,
        "cudnn_version": None,
        "device": "cpu",
        "device_name": "CPU",
        "device_capability": None,
        "device_total_memory": None,
        "deterministic_algorithms": True,
        "epoch_seed_scheme": "sha256(blotgrid-training-epoch-v1:seed:epoch)",
        "source_sha256": {
            "wblot/ml_training.py": "1" * 64,
            "wblot/ml_detector.py": "2" * 64,
            "scripts/train_detector.py": "3" * 64,
            "requirements-ml.txt": "4" * 64,
        },
    }
    initialization = torchvision_initialization_provenance(
        DEFAULT_ARCHITECTURE,
        pretrained=pretrained,
        torchvision=_fake_torchvision(),
    )
    if pretrained:
        spec = ml_detector._TORCHVISION_WEIGHT_SPECS[DEFAULT_ARCHITECTURE]
        initialization["artifact_sha256"] = spec["sha256"]
        initialization["artifact_size_bytes"] = spec["size_bytes"]
        initialization["artifact_verified"] = True
    licensed_coco = _licensed_coco()
    second_image = {
        **licensed_coco["images"][0],
        "id": 2,
        "file_name": "validation.png",
        "blotgrid": {
            **licensed_coco["images"][0]["blotgrid"],
            "original_sha256": "d" * 64,
            "study_group": "Example validation study",
        },
    }
    second_annotation = {
        **licensed_coco["annotations"][0],
        "id": 2,
        "image_id": 2,
    }
    licensed_coco["images"].append(second_image)
    licensed_coco["annotations"].append(second_annotation)
    lineage = _extract_dataset_license_lineage(licensed_coco)
    lineage_by_content = {
        record["content_sha256"]: record for record in lineage["images"]
    }
    train_lineage = lineage_by_content["a" * 64]
    validation_lineage = lineage_by_content["d" * 64]
    image_manifest = [
        {
            "image_digest": train_lineage["image_digest"],
            "sha256": "a" * 64,
            "study_group_sha256": "b" * 64,
            "original_sha256": "a" * 64,
            "license_id": 7,
            "source_id": "example_v1",
        },
        {
            "image_digest": validation_lineage["image_digest"],
            "sha256": "d" * 64,
            "study_group_sha256": "e" * 64,
            "original_sha256": "d" * 64,
            "license_id": 7,
            "source_id": "example_v1",
        },
    ]
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            image_manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    metrics = {
        "images": 1,
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "mean_matched_iou": 1.0,
        "score_threshold": 0.5,
        "iou_threshold": 0.5,
    }
    training_config = {
        "architecture": DEFAULT_ARCHITECTURE,
        "optimizer": "SGD",
        "epochs_requested": 1,
        "batch_size": 1,
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "validation_fraction": 0.2,
        "seed": 1337,
        "num_workers": 0,
        "score_threshold": 0.5,
        "iou_threshold": 0.5,
        "pretrained_transfer_learning_requested": pretrained,
        "trainable_backbone_layers": 3 if pretrained else 6,
        "backbone_normalization": (
            "FrozenBatchNorm2d" if pretrained else "BatchNorm2d"
        ),
        "allow_pseudo_labels": False,
        "allow_unverified_images": False,
    }
    return {
        "metadata_verified": True,
        "embedded_metadata_verified": True,
        "container_verified": True,
        "release_artifact_sanitized": True,
        "verified_trainable_parameter_count": 100,
        "sha256": "f" * 64,
        "metadata": {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "publication_metadata_schema_version": (
                PUBLICATION_METADATA_SCHEMA_VERSION
            ),
            "architecture": DEFAULT_ARCHITECTURE,
            "num_classes": 2,
            "categories": {"0": "background", "1": "band"},
            "epoch": 0,
            "epochs_requested": 1,
            "best_epoch": 0,
            "best_metrics": dict(metrics),
            "current_metrics": dict(metrics),
            "selection_metric": "validation_f1_then_mean_matched_iou",
            "history": [{"epoch": 0, "train_loss": 0.1, **metrics}],
            "training_config": training_config,
            "training_environment": environment,
            "initialization_provenance": initialization,
            "pretrained_transfer_learning": pretrained,
            "trainable_backbone_layers": 3 if pretrained else 6,
            "trainable_parameter_count": 100,
            "seed": 1337,
            "score_threshold": 0.5,
            "iou_threshold": 0.5,
            "dataset_license_lineage": lineage,
            "train_image_digests": [
                train_lineage["image_digest"]
            ],
            "val_image_digests": [
                validation_lineage["image_digest"]
            ],
            "annotation_sha256": "c" * 64,
            "annotation_complete_images": 2,
            "image_manifest": image_manifest,
            "image_manifest_sha256": manifest_sha256,
            "allow_unverified_images": False,
            "pseudo_label_annotations": 0,
            "allow_pseudo_labels": False,
            "created_utc": "2026-01-01T00:00:00+00:00",
            "torch_version": environment["torch_version"],
            "torchvision_version": environment["torchvision_version"],
            "torchvision_git_revision": environment[
                "torchvision_git_revision"
            ],
            "device": "cpu",
            "optimizer_state_included": True,
            "checkpoint_sha256": "f" * 64,
        },
    }


def test_checkpoint_publication_gate_emits_attribution_for_scratch_model():
    assessment = checkpoint_publication_assessment(_publication_provenance())
    assert assessment["eligible_for_publication"] is True
    assert assessment["blockers"] == []
    assert assessment["attribution_bundle"]["sources"][0][
        "required_attribution"
    ].startswith("Example Dataset")


def test_checkpoint_publication_gate_requires_embedded_metadata_binding():
    provenance = _publication_provenance()
    provenance["embedded_metadata_verified"] = False
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert "unverified_embedded_metadata" in {
        blocker["code"] for blocker in assessment["blockers"]
    }


def test_checkpoint_publication_gate_requires_verified_container():
    provenance = _publication_provenance()
    provenance["container_verified"] = False
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert "unverified_checkpoint_container" in {
        blocker["code"] for blocker in assessment["blockers"]
    }


def test_checkpoint_publication_gate_requires_sanitized_release_artifact():
    provenance = _publication_provenance()
    provenance["release_artifact_sanitized"] = False
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert "unsanitized_release_artifact" in {
        blocker["code"] for blocker in assessment["blockers"]
    }


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("publication_metadata_schema_version", True, "publication_schema"),
        ("num_classes", 2.0, "private_metadata"),
        ("trainable_parameter_count", 101, "unverified_checkpoint_container"),
        ("annotation_complete_images", 1, "whole_image_attestation"),
    ],
)
def test_checkpoint_publication_gate_requires_exact_scalar_types_and_binding(
    field,
    value,
    expected_code,
):
    provenance = _publication_provenance()
    provenance["metadata"][field] = value
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert expected_code in {
        blocker["code"] for blocker in assessment["blockers"]
    }


def test_checkpoint_publication_gate_rejects_previous_approval_schema():
    provenance = _publication_provenance()
    provenance["metadata"]["publication_metadata_schema_version"] = 1
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert "publication_schema" in {
        blocker["code"] for blocker in assessment["blockers"]
    }


def test_checkpoint_publication_gate_blocks_pretrained_and_private_metadata():
    provenance = _publication_provenance(pretrained=True)
    provenance["metadata"]["image_manifest"][0]["file_name"] = "private.tif"
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert {
        blocker["code"] for blocker in assessment["blockers"]
    } >= {"pretrained_redistribution_review", "private_metadata"}
    assert assessment["attribution_bundle"] is None


@pytest.mark.parametrize(
    ("container", "field", "expected_code"),
    [
        ("metadata", "patient_note", "private_metadata"),
        ("training_environment", "home_dir", "private_metadata"),
        ("training_config", "local_dataset_path", "private_metadata"),
        ("initialization_provenance", "private_note", "initializer_lineage"),
        ("best_metrics", "patient_count", "private_metadata"),
    ],
)
def test_checkpoint_publication_gate_rejects_unknown_nested_metadata(
    container,
    field,
    expected_code,
):
    provenance = _publication_provenance()
    target = (
        provenance["metadata"]
        if container == "metadata"
        else provenance["metadata"][container]
    )
    target[field] = "/private/patient-123"
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert expected_code in {
        blocker["code"] for blocker in assessment["blockers"]
    }


def _resign_lineage(lineage):
    lineage["sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in lineage.items() if key != "sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("policy", ["all_rights_reserved", "nc", "custom", "sa"])
def test_checkpoint_publication_gate_requires_reviewed_license_policy(policy):
    provenance = _publication_provenance()
    license_record = provenance["metadata"]["dataset_license_lineage"][
        "licenses"
    ][0]
    if policy == "all_rights_reserved":
        license_record.update(
            {
                "name": "All rights reserved - no redistribution",
                "url": "https://example.test/all-rights-reserved",
                "spdx_identifier": "LicenseRef-ARR",
            }
        )
    elif policy == "nc":
        license_record.update(
            {
                "name": "Creative Commons Attribution-NonCommercial 4.0",
                "url": "https://creativecommons.org/licenses/by-nc/4.0/",
                "spdx_identifier": "CC-BY-NC-4.0",
                "publication_assessment": {
                    "status": "review_required",
                    "redistribution_cleared": True,
                    "commercial_use_cleared": False,
                    "attribution_required": True,
                    "share_alike_required": False,
                    "obligations": ["attribution"],
                },
            }
        )
    elif policy == "custom":
        license_record.update(
            {
                "name": "Example custom data terms",
                "url": "https://example.test/custom",
                "spdx_identifier": "LicenseRef-Example",
            }
        )
    else:
        license_record.update(
            {
                "name": "Creative Commons Attribution-ShareAlike 2.1 Japan",
                "url": "https://creativecommons.org/licenses/by-sa/2.1/jp/",
                "spdx_identifier": "CC-BY-SA-2.1-JP",
                "publication_assessment": {
                    "status": "cleared",
                    "redistribution_cleared": True,
                    "commercial_use_cleared": True,
                    "attribution_required": True,
                    "share_alike_required": True,
                    "obligations": ["attribution", "share_alike"],
                },
            }
        )
    _resign_lineage(provenance["metadata"]["dataset_license_lineage"])
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    codes = {blocker["code"] for blocker in assessment["blockers"]}
    if policy == "sa":
        assert "share_alike_release_review" in codes
    else:
        assert "license_not_cleared" in codes


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("image_digest", "d" * 64, "image_manifest_lineage"),
        ("license_id", 999, "image_manifest_lineage"),
        ("source_id", "unknown-source", "image_manifest_lineage"),
        ("sha256", "not-a-digest", "image_manifest"),
    ],
)
def test_checkpoint_publication_gate_binds_manifest_to_lineage(
    field,
    value,
    expected_code,
):
    provenance = _publication_provenance()
    manifest = provenance["metadata"]["image_manifest"]
    manifest[0][field] = value
    provenance["metadata"]["image_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert expected_code in {
        blocker["code"] for blocker in assessment["blockers"]
    }


def test_checkpoint_publication_gate_requires_study_groups():
    provenance = _publication_provenance()
    manifest = provenance["metadata"]["image_manifest"]
    manifest[0]["study_group_sha256"] = None
    provenance["metadata"]["image_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert "image_manifest" in {
        blocker["code"] for blocker in assessment["blockers"]
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("allow_unverified_images", True),
        ("allow_pseudo_labels", True),
        ("pseudo_label_annotations", 1),
    ],
)
def test_checkpoint_publication_gate_rejects_non_primary_training(
    field,
    value,
):
    provenance = _publication_provenance()
    provenance["metadata"][field] = value
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda metadata: metadata["training_config"].update(
                allow_unverified_images=True
            ),
            "private_metadata",
        ),
        (
            lambda metadata: metadata["training_config"].update(
                allow_pseudo_labels=True
            ),
            "private_metadata",
        ),
        (
            lambda metadata: metadata["best_metrics"].update(
                f1="patient-123"
            ),
            "private_metadata",
        ),
        (
            lambda metadata: metadata["training_environment"].update(
                platform={
                    "system": "Linux",
                    "release": "patient-123 secret cohort",
                    "machine": "x86_64",
                    "libc_name": "glibc",
                    "libc_version": "2.39",
                }
            ),
            "private_metadata",
        ),
        (
            lambda metadata: (
                metadata.update(pretrained_transfer_learning=True),
                metadata["training_config"].update(
                    pretrained_transfer_learning_requested=True
                ),
            ),
            "initializer_lineage",
        ),
    ],
)
def test_checkpoint_publication_gate_validates_values_and_cross_fields(
    mutation,
    expected_code,
):
    provenance = _publication_provenance()
    mutation(provenance["metadata"])
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert expected_code in {
        blocker["code"] for blocker in assessment["blockers"]
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda metadata: metadata.update(epoch=999),
        lambda metadata: metadata["history"][-1].update(epoch=7),
        lambda metadata: metadata["current_metrics"].update(
            false_positive=1
        ),
        lambda metadata: (
            metadata["best_metrics"].update(precision=0.5),
            metadata["current_metrics"].update(precision=0.5),
            metadata["history"][-1].update(precision=0.5),
        ),
        lambda metadata: (
            metadata["best_metrics"].update(images=999),
            metadata["current_metrics"].update(images=999),
            metadata["history"][-1].update(images=999),
        ),
        lambda metadata: (
            metadata["best_metrics"].update(
                true_positive=0,
                false_negative=1,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                mean_matched_iou=1.0,
            ),
            metadata["current_metrics"].update(
                true_positive=0,
                false_negative=1,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                mean_matched_iou=1.0,
            ),
            metadata["history"][-1].update(
                true_positive=0,
                false_negative=1,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                mean_matched_iou=1.0,
            ),
        ),
        lambda metadata: (
            metadata["best_metrics"].update(mean_matched_iou=0.49),
            metadata["current_metrics"].update(mean_matched_iou=0.49),
            metadata["history"][-1].update(mean_matched_iou=0.49),
        ),
        lambda metadata: (
            metadata["current_metrics"].update(
                true_positive=0,
                false_negative=1,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                mean_matched_iou=0.0,
            ),
            metadata["history"][-1].update(
                true_positive=0,
                false_negative=1,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                mean_matched_iou=0.0,
            ),
        ),
        lambda metadata: metadata.update(val_image_digests=[]),
        lambda metadata: metadata["training_config"].update(
            validation_fraction=1.0
        ),
        lambda metadata: metadata["training_config"].update(
            backbone_normalization="FrozenBatchNorm2d"
        ),
        lambda metadata: (
            metadata.update(epoch=7, epochs_requested=8),
            metadata["training_config"].update(epochs_requested=8),
            metadata["history"][-1].update(epoch=7),
        ),
        lambda metadata: (
            metadata.update(iou_threshold=0.0),
            metadata["training_config"].update(iou_threshold=0.0),
            metadata["best_metrics"].update(iou_threshold=0.0),
            metadata["current_metrics"].update(iou_threshold=0.0),
            metadata["history"][-1].update(iou_threshold=0.0),
        ),
    ],
)
def test_checkpoint_publication_gate_enforces_training_continuity(mutation):
    provenance = _publication_provenance()
    mutation(provenance["metadata"])
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert "private_metadata" in {
        blocker["code"] for blocker in assessment["blockers"]
    }


@pytest.mark.parametrize(
    "shared_field",
    ["study_group_sha256", "original_sha256", "sha256"],
)
def test_checkpoint_publication_gate_rejects_split_group_leakage(
    shared_field,
):
    provenance = _publication_provenance()
    metadata = provenance["metadata"]
    lineage = metadata["dataset_license_lineage"]
    train_digest = metadata["train_image_digests"][0]
    validation_digest = metadata["val_image_digests"][0]
    manifest_by_digest = {
        record["image_digest"]: record for record in metadata["image_manifest"]
    }
    train_manifest = manifest_by_digest[train_digest]
    validation_manifest = manifest_by_digest[validation_digest]
    if shared_field == "original_sha256":
        validation_lineage = next(
            record
            for record in lineage["images"]
            if record["image_digest"] == validation_digest
        )
        validation_lineage["content_sha256"] = train_manifest[
            "original_sha256"
        ]
        _resign_lineage(lineage)
    validation_manifest[shared_field] = train_manifest[shared_field]
    metadata["image_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            metadata["image_manifest"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert "split_group_leakage" in {
        blocker["code"] for blocker in assessment["blockers"]
    }


def test_checkpoint_publication_gate_requires_doi_bound_attribution():
    provenance = _publication_provenance()
    lineage = provenance["metadata"]["dataset_license_lineage"]
    lineage["sources"][0]["required_attribution"] = "x"
    _resign_lineage(lineage)
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert "source_attribution" in {
        blocker["code"] for blocker in assessment["blockers"]
    }


@pytest.mark.parametrize(
    ("doi", "attribution", "expected_code"),
    [
        (
            "10.x/not-a-doi",
            "Example Dataset Authors (2026), doi:10.x/not-a-doi",
            "source_identity",
        ),
        (
            "10.1234/example",
            (
                "Example Dataset Authors (2026), "
                "doi:10.1234/example-private"
            ),
            "source_attribution",
        ),
    ],
)
def test_checkpoint_publication_gate_requires_valid_exact_doi_token(
    doi,
    attribution,
    expected_code,
):
    provenance = _publication_provenance()
    lineage = provenance["metadata"]["dataset_license_lineage"]
    lineage["sources"][0]["doi"] = doi
    lineage["sources"][0]["required_attribution"] = attribution
    _resign_lineage(lineage)
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert expected_code in {
        blocker["code"] for blocker in assessment["blockers"]
    }


@pytest.mark.parametrize(
    ("dataset_dois", "expected_code"),
    [
        (["patient-123 secret cohort"], "dataset_lineage"),
        (["10.9999/unrelated-dataset"], "dataset_source_identity"),
    ],
)
def test_checkpoint_publication_gate_binds_dataset_and_source_dois(
    dataset_dois,
    expected_code,
):
    provenance = _publication_provenance()
    lineage = provenance["metadata"]["dataset_license_lineage"]
    lineage["dataset_info"]["dois"] = dataset_dois
    _resign_lineage(lineage)
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert expected_code in {
        blocker["code"] for blocker in assessment["blockers"]
    }


def test_checkpoint_publication_gate_rejects_boolean_lineage_schema():
    provenance = _publication_provenance()
    lineage = provenance["metadata"]["dataset_license_lineage"]
    lineage["schema_version"] = True
    _resign_lineage(lineage)
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert "dataset_lineage" in {
        blocker["code"] for blocker in assessment["blockers"]
    }


def test_checkpoint_publication_gate_rejects_boolean_initializer_schema():
    provenance = _publication_provenance()
    provenance["metadata"]["initialization_provenance"]["schema_version"] = True
    assessment = checkpoint_publication_assessment(provenance)
    assert assessment["eligible_for_publication"] is False
    assert "initializer_lineage" in {
        blocker["code"] for blocker in assessment["blockers"]
    }


@pytest.mark.parametrize("invalid_reference", ["license", "source", "duplicate"])
def test_coco_license_lineage_rejects_invalid_references(invalid_reference):
    payload = _licensed_coco()
    if invalid_reference == "license":
        payload["images"][0]["license"] = 999
        message = "unknown licence"
    elif invalid_reference == "source":
        payload["images"][0]["blotgrid"]["source_id"] = "missing"
        message = "unknown source"
    else:
        payload["licenses"].append(dict(payload["licenses"][0]))
        message = "Duplicate COCO licence"
    with pytest.raises(MLTrainingError, match=message):
        _extract_dataset_license_lineage(payload)


def test_coco_dataset_verifies_checksums_pseudo_policy_and_clipped_boxes(tmp_path):
    image_path = tmp_path / "dataset.png"
    Image.fromarray(np.zeros((30, 40), dtype=np.uint8)).save(image_path)
    digest = _sha256(image_path)

    def payload(bbox, *, pseudo=False, sha=digest):
        return {
            "images": [
                {
                    "id": 1,
                    "file_name": image_path.name,
                    "width": 40,
                    "height": 30,
                    "blotgrid": {
                        "annotation_complete": True,
                        "negative_image": False,
                        "original_sha256": sha,
                        "working_pixels_exported": False,
                    },
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": bbox,
                    "attributes": {
                        "annotation_approved": not pseudo,
                        "pseudo_label": pseudo,
                        "review_required": False,
                    },
                }
            ],
            "categories": [{"id": 1, "name": "band"}],
        }

    dataset = CocoBandDataset(
        payload([-2, 4, 12, 8]),
        tmp_path,
        require_checksums=True,
    )
    assert dataset.valid_box_counts == {1: 1}
    assert dataset.image_checksums[1] == digest
    with pytest.raises(MLTrainingError, match="checksum mismatch"):
        CocoBandDataset(
            payload([2, 4, 12, 8], sha="0" * 64),
            tmp_path,
            require_checksums=True,
        )
    with pytest.raises(MLTrainingError, match="no usable"):
        CocoBandDataset(
            payload([100, 100, 5, 5]),
            tmp_path,
            require_checksums=True,
        )
    incomplete = payload([2, 4, 12, 8])
    incomplete["images"][0]["blotgrid"]["annotation_complete"] = False
    with pytest.raises(MLTrainingError, match="annotation_complete=true"):
        CocoBandDataset(
            incomplete,
            tmp_path,
            require_checksums=True,
        )
    with pytest.raises(MLTrainingError, match="no usable"):
        CocoBandDataset(
            payload([2, 4, 12, 8], pseudo=True),
            tmp_path,
            require_checksums=True,
        )
    pseudo_dataset = CocoBandDataset(
        payload([2, 4, 12, 8], pseudo=True),
        tmp_path,
        include_pseudo_labels=True,
        require_checksums=True,
    )
    assert pseudo_dataset.valid_box_counts == {1: 1}
    for invalid_marker in (1, "true", 0, "false"):
        malformed = payload([2, 4, 12, 8])
        malformed["annotations"][0]["attributes"][
            "pseudo_label"
        ] = invalid_marker
        with pytest.raises(MLTrainingError, match="literal boolean"):
            CocoBandDataset(
                malformed,
                tmp_path,
                require_checksums=True,
            )


@pytest.mark.parametrize(
    ("attributes", "is_pseudo"),
    [
        (None, True),
        ({}, True),
        ({"pseudo_label": False}, True),
        ({"annotation_approved": False}, True),
        (
            {
                "annotation_approved": True,
                "pseudo_label": False,
                "review_required": False,
            },
            False,
        ),
    ],
)
def test_annotation_approval_markers_fail_closed(attributes, is_pseudo):
    annotation = {}
    if attributes is not None:
        annotation["attributes"] = attributes
    assert _annotation_is_pseudo(annotation) is is_pseudo


def test_coco_json_rejects_duplicate_annotation_marker_keys(tmp_path):
    image_path = tmp_path / "dataset.png"
    Image.fromarray(np.zeros((30, 40), dtype=np.uint8)).save(image_path)
    payload = _licensed_coco(image_path.name, _sha256(image_path))
    encoded = json.dumps(payload)
    encoded = encoded.replace(
        '"annotation_approved": true',
        '"annotation_approved": false, "annotation_approved": true',
        1,
    )
    annotations = tmp_path / "duplicate.json"
    annotations.write_text(encoded, encoding="utf-8")

    with pytest.raises(MLTrainingError, match="duplicate JSON object key"):
        CocoBandDataset(
            annotations,
            tmp_path,
            require_checksums=True,
        )


@pytest.mark.parametrize("value", [1, 0, "true", "false"])
def test_annotation_approval_requires_literal_boolean(value):
    with pytest.raises(MLTrainingError, match="literal boolean"):
        _annotation_is_pseudo(
            {
                "attributes": {
                    "annotation_approved": value,
                    "pseudo_label": False,
                    "review_required": False,
                }
            }
        )


def test_predict_errors_cleanly_for_missing_checkpoint_without_torch_use(tmp_path):
    from wblot.ml_detector import predict_bands

    with pytest.raises(MLDetectorError):
        # Runtime availability is environment-dependent, so this only asserts
        # the public error type if the optional runtime is present.
        predict_bands(
            np.zeros((20, 20)),
            ["T"],
            ["L"],
            tmp_path / "missing.pt",
        )
