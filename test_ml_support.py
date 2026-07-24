"""Pure tests for optional ML support (no PyTorch installation required)."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from wblot.ml_detector import (
    DEFAULT_ARCHITECTURE,
    MLDetectorError,
    MAX_TRAINABLE_BACKBONE_LAYERS,
    SUPPORTED_ARCHITECTURES,
    TRANSFER_TRAINABLE_BACKBONE_LAYERS,
    build_model,
    checkpoint_provenance,
    cluster_axis_centers,
    map_boxes_to_grid,
    normalize_grayscale,
)
from wblot import gelgenie
from scripts import download_pretrained_models
from wblot.ml_training import (
    CocoBandDataset,
    MLTrainingError,
    _checkpoint_paths,
    _training_environment,
    _validate_resume_metadata,
    box_iou_matrix,
    detection_metrics,
    export_project_to_coco,
    image_leakage_groups,
    make_train_val_split,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    assert metadata["device"] == "cpu"
    assert metadata["device_name"] == "CPU"
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
    }
    environment = {
        "python_version": "3.10.12",
        "torch_version": "2.7.1+cu118",
        "torchvision_version": "0.22.1+cu118",
        "cuda_build": "11.8",
        "device": "cuda",
        "source_sha256": {"wblot/ml_training.py": "a" * 64},
    }
    resumed = dict(config, epochs_requested=20)
    _validate_resume_metadata(
        {
            "training_config": config,
            "training_environment": environment,
        },
        resumed,
        environment,
    )
    with pytest.raises(MLTrainingError, match="learning_rate"):
        _validate_resume_metadata(
            {
                "training_config": config,
                "training_environment": environment,
            },
            dict(resumed, learning_rate=0.01),
            environment,
        )
    with pytest.raises(MLTrainingError, match="torch_version"):
        _validate_resume_metadata(
            {
                "training_config": config,
                "training_environment": environment,
            },
            resumed,
            dict(environment, torch_version="2.8.0"),
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
