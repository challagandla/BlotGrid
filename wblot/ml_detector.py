"""Optional TorchVision detector inference for Western-blot bands.

This module is deliberately importable without PyTorch.  The optional runtime
is imported only by functions that actually build or run a model, so the
classical BlotGrid workflow remains usable in a small CPU-only environment.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np

from . import detection


DEFAULT_ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_fpn"
TRANSFER_TRAINABLE_BACKBONE_LAYERS = 3
MAX_TRAINABLE_BACKBONE_LAYERS = {
    "fasterrcnn_mobilenet_v3_large_fpn": 6,
    "fasterrcnn_resnet50_fpn_v2": 5,
    "fcos_resnet50_fpn": 5,
    "retinanet_resnet50_fpn_v2": 5,
}
SUPPORTED_ARCHITECTURES = (
    DEFAULT_ARCHITECTURE,
    "fasterrcnn_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
)
CHECKPOINT_FORMAT_VERSION = 2
_MAX_CHECKPOINT_BYTES = 8 * 1024**3
MAX_CHECKPOINT_METADATA_BYTES = 1024**2
_MAX_METADATA_BYTES = MAX_CHECKPOINT_METADATA_BYTES

_TORCHVISION_WEIGHT_SPECS: dict[str, dict[str, str | int]] = {
    "fasterrcnn_mobilenet_v3_large_fpn": {
        "enum_class": "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
        "enum_member": "COCO_V1",
        "url": (
            "https://download.pytorch.org/models/"
            "fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth"
        ),
        "sha256_prefix": "fb6a3cc7",
        "sha256": (
            "fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533"
        ),
        "size_bytes": 77844807,
        "recipe_url": (
            "https://github.com/pytorch/vision/tree/main/references/"
            "detection#faster-r-cnn-mobilenetv3-large-fpn"
        ),
    },
    "fasterrcnn_resnet50_fpn_v2": {
        "enum_class": "FasterRCNN_ResNet50_FPN_V2_Weights",
        "enum_member": "COCO_V1",
        "url": (
            "https://download.pytorch.org/models/"
            "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
        ),
        "sha256_prefix": "dd69338a",
        "sha256": (
            "dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf"
        ),
        "size_bytes": 175221657,
        "recipe_url": "https://github.com/pytorch/vision/pull/5763",
    },
    "fcos_resnet50_fpn": {
        "enum_class": "FCOS_ResNet50_FPN_Weights",
        "enum_member": "COCO_V1",
        "url": (
            "https://download.pytorch.org/models/"
            "fcos_resnet50_fpn_coco-99b0c9b7.pth"
        ),
        "sha256_prefix": "99b0c9b7",
        "sha256": (
            "99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7"
        ),
        "size_bytes": 129612099,
        "recipe_url": (
            "https://github.com/pytorch/vision/tree/main/references/"
            "detection#fcos-resnet-50-fpn"
        ),
    },
    "retinanet_resnet50_fpn_v2": {
        "enum_class": "RetinaNet_ResNet50_FPN_V2_Weights",
        "enum_member": "COCO_V1",
        "url": (
            "https://download.pytorch.org/models/"
            "retinanet_resnet50_fpn_v2_coco-5905b1c5.pth"
        ),
        "sha256_prefix": "5905b1c5",
        "sha256": (
            "5905b1c544219215e544dbe319720397bc4e68de61a733a59350d7976645b769"
        ),
        "size_bytes": 153130989,
        "recipe_url": "https://github.com/pytorch/vision/pull/5756",
    },
}

_TORCHVISION_WEIGHT_LICENSE_ASSESSMENT = {
    "status": "review_required",
    "redistribution_cleared": False,
    "torchvision_code_license": "BSD-3-Clause",
    "pretrained_weight_license": "not_stated_by_torchvision",
    "training_dataset": "COCO",
    "assessment": (
        "TorchVision states that pretrained models may have licences or terms "
        "derived from their training datasets. Do not treat the TorchVision "
        "source-code licence as clearance to redistribute a derived checkpoint."
    ),
    "evidence_url": "https://github.com/pytorch/vision#using-the-pre-trained-models",
}

_SCRATCH_INITIALIZATION_SCOPE = "all detector parameters initialized locally"
_SCRATCH_LICENSE_ASSESSMENT = {
    "status": "not_applicable",
    "redistribution_cleared": None,
    "reason": (
        "No third-party pretrained weights were loaded. Dataset lineage must "
        "still be assessed separately."
    ),
}
_PRETRAINED_INITIALIZATION_SCOPE = (
    "TorchVision detector weights loaded; BlotGrid replaces the COCO "
    "classification logits with a two-class background/band head."
)


class MLDetectorError(ValueError):
    """A recoverable configuration, checkpoint, or optional-runtime error."""


def _require_ml_runtime():
    """Import and return ``(torch, torchvision)`` only when ML is requested."""
    try:
        import torch
        import torchvision
    except Exception as exc:
        raise MLDetectorError(
            "AI band detection requires the optional PyTorch runtime. "
            "Install the matched torch/torchvision pair from requirements-ml.txt."
        ) from exc
    return torch, torchvision


def _safe_checkpoint_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a regular, local checkpoint without following a final symlink."""
    raw = os.fspath(path)
    if not raw.strip():
        raise MLDetectorError("A checkpoint path is required.")
    if "://" in raw:
        raise MLDetectorError("Only local checkpoint files are accepted.")
    candidate = Path(raw).expanduser()
    if candidate.is_symlink():
        raise MLDetectorError("Checkpoint symlinks are not accepted.")
    try:
        info = candidate.stat()
    except OSError as exc:
        raise MLDetectorError(f"Checkpoint is not readable: {candidate}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise MLDetectorError("The checkpoint path must name a regular file.")
    if info.st_size <= 0:
        raise MLDetectorError("The checkpoint file is empty.")
    if info.st_size > _MAX_CHECKPOINT_BYTES:
        raise MLDetectorError("The checkpoint is larger than the 8 GiB safety limit.")
    return candidate.resolve()


def _metadata_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".metadata.json")


def _safe_metadata(value: Any, *, depth: int = 0) -> Any:
    """Copy JSON-like metadata while dropping tensors and executable objects."""
    if depth > 8:
        return None
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item, depth=depth + 1) for item in value[:10000]]
    if isinstance(value, dict):
        copied: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 10000:
                break
            if isinstance(key, (str, int, float, bool)):
                copied[str(key)[:256]] = _safe_metadata(item, depth=depth + 1)
        return copied
    return None


def _reject_duplicate_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build a JSON object only when every key occurs exactly once."""
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON object key {key!r}")
        parsed[key] = value
    return parsed


def _loads_unique_json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_reject_duplicate_json_object)


def _sha256_file(path: Path, *, chunk_size: int = 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torchvision_source_revision(torchvision: Any) -> str:
    version_module = getattr(torchvision, "version", None)
    return str(getattr(version_module, "git_version", "") or "").strip()


def _resolve_torchvision_weights(
    torchvision: Any,
    architecture: str,
    *,
    pretrained: bool,
) -> Any:
    if architecture not in _TORCHVISION_WEIGHT_SPECS:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)
        raise MLDetectorError(
            f"Unsupported detector architecture {architecture!r}. "
            f"Supported architectures: {supported}."
        )
    if not pretrained:
        return None
    spec = _TORCHVISION_WEIGHT_SPECS[architecture]
    detection_models = torchvision.models.detection
    try:
        enum_class = getattr(detection_models, spec["enum_class"])
        weights = getattr(enum_class, spec["enum_member"])
    except AttributeError as exc:
        raise MLDetectorError(
            "The installed TorchVision runtime does not expose the pinned "
            f"{spec['enum_class']}.{spec['enum_member']} initialization."
        ) from exc
    if str(getattr(weights, "url", "")) != spec["url"]:
        raise MLDetectorError(
            "TorchVision's pretrained-weight URL differs from BlotGrid's pinned "
            f"provenance contract for {architecture}."
        )
    return weights


def _torchvision_cached_artifact(torch: Any, weights: Any) -> Path:
    filename = Path(urlparse(str(weights.url)).path).name
    if not filename:
        raise MLDetectorError("TorchVision's pretrained-weight URL has no filename.")
    candidate = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if candidate.is_symlink() or not candidate.is_file():
        raise MLDetectorError(
            "TorchVision initialized the detector but the exact cached weight "
            "artifact could not be verified."
        )
    return candidate


def torchvision_initialization_provenance(
    architecture: str,
    *,
    pretrained: bool,
    torchvision: Any,
    artifact_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return a bounded, exact initialization record without downloading weights."""
    weights = _resolve_torchvision_weights(
        torchvision,
        architecture,
        pretrained=bool(pretrained),
    )
    runtime_version = str(getattr(torchvision, "__version__", "") or "").strip()
    runtime_revision = _torchvision_source_revision(torchvision)
    if not pretrained:
        return {
            "schema_version": 1,
            "mode": "random_initialization",
            "architecture": architecture,
            "weights_enum": None,
            "weights_revision": None,
            "weights_url": None,
            "url_sha256_prefix": None,
            "expected_artifact_sha256": None,
            "expected_artifact_size_bytes": None,
            "artifact_sha256": None,
            "artifact_size_bytes": None,
            "artifact_verified": None,
            "torchvision_version": runtime_version or None,
            "torchvision_git_revision": runtime_revision or None,
            "initialization_scope": _SCRATCH_INITIALIZATION_SCOPE,
            "license_assessment": dict(_SCRATCH_LICENSE_ASSESSMENT),
        }

    if not runtime_version or not runtime_revision:
        raise MLDetectorError(
            "Exact TorchVision version and source revision are required when "
            "pretrained initialization is enabled."
        )
    spec = _TORCHVISION_WEIGHT_SPECS[architecture]
    enum_name = f"{spec['enum_class']}.{spec['enum_member']}"
    artifact_sha256 = None
    artifact_size = None
    artifact_verified = False
    if artifact_path is not None:
        artifact = Path(artifact_path)
        if artifact.is_symlink() or not artifact.is_file():
            raise MLDetectorError(
                "The TorchVision initialization artifact must be a regular file."
            )
        artifact_sha256 = _sha256_file(artifact)
        artifact_size = int(artifact.stat().st_size)
        if (
            artifact_sha256 != spec["sha256"]
            or artifact_size != spec["size_bytes"]
        ):
            raise MLDetectorError(
                "The cached TorchVision initialization artifact does not match "
                "BlotGrid's pinned full SHA-256 and byte size for "
                f"{architecture}."
            )
        artifact_verified = True
    recipe = getattr(weights, "meta", {}).get("recipe")
    return {
        "schema_version": 1,
        "mode": "torchvision_pretrained",
        "architecture": architecture,
        "weights_enum": enum_name,
        "weights_revision": spec["enum_member"],
        "weights_url": spec["url"],
        "url_sha256_prefix": spec["sha256_prefix"],
        "expected_artifact_sha256": spec["sha256"],
        "expected_artifact_size_bytes": spec["size_bytes"],
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": artifact_size,
        "artifact_verified": artifact_verified,
        "torchvision_version": runtime_version,
        "torchvision_git_revision": runtime_revision,
        "recipe_url": str(recipe) if recipe else None,
        "training_dataset": "COCO",
        "initialization_scope": _PRETRAINED_INITIALIZATION_SCOPE,
        "license_assessment": dict(_TORCHVISION_WEIGHT_LICENSE_ASSESSMENT),
    }


def validate_initialization_provenance(
    record: Any,
    architecture: str,
    *,
    torchvision_version: str,
    torchvision_git_revision: str,
    require_verified_artifact: bool = True,
) -> dict[str, Any]:
    """Validate a checkpoint initializer against the current pinned contract."""
    if architecture not in _TORCHVISION_WEIGHT_SPECS:
        raise MLDetectorError(
            f"Unsupported detector architecture {architecture!r}."
        )
    if not isinstance(record, dict):
        raise MLDetectorError("Initialization provenance must be an object.")
    copied = _safe_metadata(record)
    if (
        copied != record
        or type(copied.get("schema_version")) is not int
        or copied.get("schema_version") != 1
    ):
        raise MLDetectorError("Initialization provenance is not safe schema 1.")
    pretrained_fields = {
        "schema_version",
        "mode",
        "architecture",
        "weights_enum",
        "weights_revision",
        "weights_url",
        "url_sha256_prefix",
        "expected_artifact_sha256",
        "expected_artifact_size_bytes",
        "artifact_sha256",
        "artifact_size_bytes",
        "artifact_verified",
        "torchvision_version",
        "torchvision_git_revision",
        "recipe_url",
        "training_dataset",
        "initialization_scope",
        "license_assessment",
    }
    scratch_fields = pretrained_fields - {"recipe_url", "training_dataset"}
    mode = copied.get("mode")
    expected_fields = (
        scratch_fields
        if mode == "random_initialization"
        else pretrained_fields
        if mode == "torchvision_pretrained"
        else set()
    )
    if set(copied) != expected_fields:
        raise MLDetectorError(
            "Initialization provenance fields do not exactly match its mode."
        )
    if mode not in {"random_initialization", "torchvision_pretrained"}:
        raise MLDetectorError("Initialization provenance mode is invalid.")
    if copied.get("architecture") != architecture:
        raise MLDetectorError(
            "Initialization provenance architecture does not match the model."
        )
    if copied.get("torchvision_version") != (torchvision_version or None):
        raise MLDetectorError(
            "Initialization provenance TorchVision version changed."
        )
    if copied.get("torchvision_git_revision") != (
        torchvision_git_revision or None
    ):
        raise MLDetectorError(
            "Initialization provenance TorchVision source revision changed."
        )

    weight_fields = (
        "weights_enum",
        "weights_revision",
        "weights_url",
        "url_sha256_prefix",
        "expected_artifact_sha256",
        "expected_artifact_size_bytes",
        "artifact_sha256",
        "artifact_size_bytes",
    )
    assessment = copied.get("license_assessment")
    if mode == "random_initialization":
        if any(copied.get(field) is not None for field in weight_fields):
            raise MLDetectorError(
                "Scratch initialization must not claim third-party weights."
            )
        if copied.get("artifact_verified") is not None:
            raise MLDetectorError(
                "Scratch initialization cannot claim artifact verification."
            )
        if (
            copied.get("initialization_scope")
            != _SCRATCH_INITIALIZATION_SCOPE
            or assessment != _SCRATCH_LICENSE_ASSESSMENT
        ):
            raise MLDetectorError(
                "Scratch initialization provenance is not the generated record."
            )
        return copied

    spec = _TORCHVISION_WEIGHT_SPECS[architecture]
    expected = {
        "weights_enum": f"{spec['enum_class']}.{spec['enum_member']}",
        "weights_revision": spec["enum_member"],
        "weights_url": spec["url"],
        "url_sha256_prefix": spec["sha256_prefix"],
        "expected_artifact_sha256": spec["sha256"],
        "expected_artifact_size_bytes": spec["size_bytes"],
        "recipe_url": spec["recipe_url"],
        "training_dataset": "COCO",
        "initialization_scope": _PRETRAINED_INITIALIZATION_SCOPE,
    }
    for field, value in expected.items():
        if copied.get(field) != value:
            raise MLDetectorError(
                f"Initialization provenance field {field!r} is not pinned."
            )
    artifact_hash = copied.get("artifact_sha256")
    artifact_size = copied.get("artifact_size_bytes")
    artifact_verified = copied.get("artifact_verified")
    artifact_matches = (
        artifact_hash == spec["sha256"]
        and artifact_size == spec["size_bytes"]
        and artifact_verified is True
    )
    artifact_absent = (
        artifact_hash is None
        and artifact_size is None
        and artifact_verified is False
    )
    if (require_verified_artifact and not artifact_matches) or (
        not require_verified_artifact
        and not (artifact_matches or artifact_absent)
    ):
        raise MLDetectorError(
            "Initialization provenance lacks the verified pinned weight artifact."
        )
    if assessment != _TORCHVISION_WEIGHT_LICENSE_ASSESSMENT:
        raise MLDetectorError(
            "TorchVision initializer licence assessment is not conservative."
        )
    return copied


def checkpoint_provenance(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return safe checkpoint metadata and SHA-256 without importing PyTorch.

    Training writes a small ``.metadata.json`` sidecar next to each checkpoint.
    If a checkpoint came from elsewhere, provenance still includes its path,
    size, timestamp, and digest; model metadata is simply empty.
    """
    checkpoint = _safe_checkpoint_path(path)
    info = checkpoint.stat()
    digest = _sha256_file(checkpoint)
    metadata: dict[str, Any] = {}
    sidecar = _metadata_sidecar(checkpoint)
    metadata_warning = ""
    if sidecar.exists():
        try:
            if sidecar.is_symlink() or not sidecar.is_file():
                raise ValueError("metadata sidecar is not a regular file")
            if sidecar.stat().st_size > _MAX_METADATA_BYTES:
                raise ValueError("metadata sidecar exceeds 1 MiB")
            parsed = _loads_unique_json(sidecar.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("metadata sidecar must contain a JSON object")
            metadata = _safe_metadata(parsed)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            metadata_warning = f"Checkpoint metadata sidecar was ignored: {exc}"

    metadata_verified = (
        metadata.get("checkpoint_sha256") == digest
        if metadata.get("checkpoint_sha256")
        else None
    )
    exposed_metadata = metadata if metadata_verified is True else {}
    if metadata_verified is False:
        metadata_warning = (
            "Checkpoint metadata sidecar SHA-256 does not match the checkpoint; "
            "all sidecar claims were quarantined."
        )
    elif metadata and metadata_verified is None:
        metadata_warning = (
            "Checkpoint metadata sidecar has no checkpoint_sha256; all sidecar "
            "claims were quarantined."
        )
    result: dict[str, Any] = {
        "path": str(checkpoint),
        "filename": checkpoint.name,
        "size_bytes": int(info.st_size),
        "modified_utc": datetime.fromtimestamp(
            info.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": digest,
        "metadata": exposed_metadata,
        "metadata_verified": metadata_verified,
    }
    if metadata_warning:
        result["warning"] = metadata_warning
    for key, value in exposed_metadata.items():
        if key not in result:
            result[key] = value
    return result


def _build_model(
    architecture: str = DEFAULT_ARCHITECTURE,
    *,
    pretrained: bool = False,
    trainable_backbone_layers: int | None = None,
    freeze_backbone_batch_norm: bool = False,
    return_initialization_provenance: bool = False,
):
    """Build a two-class TorchVision detector (background + band)."""
    torch, torchvision = _require_ml_runtime()
    if architecture not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)
        raise MLDetectorError(
            f"Unsupported detector architecture {architecture!r}. "
            f"Supported architectures: {supported}."
        )

    try:
        detection_models = torchvision.models.detection
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        weights = _resolve_torchvision_weights(
            torchvision,
            architecture,
            pretrained=bool(pretrained),
        )
        if architecture == DEFAULT_ARCHITECTURE:
            model = detection_models.fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights,
                weights_backbone=None,
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        elif architecture == "fasterrcnn_resnet50_fpn_v2":
            model = detection_models.fasterrcnn_resnet50_fpn_v2(
                weights=weights,
                weights_backbone=None,
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        else:
            if architecture == "fcos_resnet50_fpn":
                model = detection_models.fcos_resnet50_fpn(
                    weights=weights,
                    weights_backbone=None,
                )
            else:
                model = detection_models.retinanet_resnet50_fpn_v2(
                    weights=weights,
                    weights_backbone=None,
                )
            # Retain the pretrained dense classification tower and replace only
            # its COCO logits with the two BlotGrid classes.
            classification_head = model.head.classification_head
            previous_logits = classification_head.cls_logits
            anchors = int(model.anchor_generator.num_anchors_per_location()[0])
            classification_head.cls_logits = torch.nn.Conv2d(
                int(previous_logits.in_channels),
                anchors * 2,
                kernel_size=previous_logits.kernel_size,
                stride=previous_logits.stride,
                padding=previous_logits.padding,
            )
            torch.nn.init.normal_(
                classification_head.cls_logits.weight,
                std=0.01,
            )
            prior_probability = 0.01
            torch.nn.init.constant_(
                classification_head.cls_logits.bias,
                -math.log((1.0 - prior_probability) / prior_probability),
            )
            classification_head.num_classes = 2
        if freeze_backbone_batch_norm:
            _freeze_backbone_batch_norm(model, torch, torchvision)
        if trainable_backbone_layers is not None:
            _set_trainable_backbone_layers(
                model,
                architecture,
                int(trainable_backbone_layers),
            )
        artifact_path = (
            _torchvision_cached_artifact(torch, weights) if pretrained else None
        )
        initialization_provenance = torchvision_initialization_provenance(
            architecture,
            pretrained=bool(pretrained),
            torchvision=torchvision,
            artifact_path=artifact_path,
        )
    except MLDetectorError:
        raise
    except Exception as exc:
        download_hint = (
            " Check network access to the official PyTorch weight host, or use "
            "--no-pretrained to start without transfer-learning weights."
            if pretrained
            else ""
        )
        raise MLDetectorError(
            f"Could not build the {architecture} detector.{download_hint}"
        ) from exc
    if return_initialization_provenance:
        return model, initialization_provenance
    return model


def _freeze_backbone_batch_norm(
    model: Any,
    torch: Any,
    torchvision: Any,
) -> None:
    """Use FrozenBatchNorm2d even when checkpoint weights avoid a download."""
    frozen_type = torchvision.ops.misc.FrozenBatchNorm2d

    def replace(parent: Any) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.BatchNorm2d):
                frozen = frozen_type(child.num_features, eps=child.eps)
                with torch.no_grad():
                    frozen.weight.copy_(child.weight.detach())
                    frozen.bias.copy_(child.bias.detach())
                    frozen.running_mean.copy_(child.running_mean.detach())
                    frozen.running_var.copy_(child.running_var.detach())
                setattr(parent, name, frozen)
            else:
                replace(child)

    replace(model.backbone.body)


def _set_trainable_backbone_layers(
    model: Any,
    architecture: str,
    trainable_layers: int,
) -> None:
    """Apply an explicit freeze policy independent of weight-loading mode."""
    maximum = MAX_TRAINABLE_BACKBONE_LAYERS[architecture]
    if trainable_layers < 0 or trainable_layers > maximum:
        raise MLDetectorError(
            f"trainable_backbone_layers for {architecture} must be between "
            f"0 and {maximum}, not {trainable_layers}."
        )
    body = model.backbone.body
    for parameter in body.parameters():
        parameter.requires_grad_(True)
    if architecture == DEFAULT_ARCHITECTURE:
        blocks = list(body.children())
        stage_indices = (
            [0]
            + [
                index
                for index, block in enumerate(blocks)
                if getattr(block, "_is_cn", False)
            ]
            + [len(blocks) - 1]
        )
        freeze_before = (
            len(blocks)
            if trainable_layers == 0
            else stage_indices[len(stage_indices) - trainable_layers]
        )
        for block in blocks[:freeze_before]:
            for parameter in block.parameters():
                parameter.requires_grad_(False)
        return
    layers_to_train = [
        "layer4",
        "layer3",
        "layer2",
        "layer1",
        "conv1",
    ][:trainable_layers]
    if trainable_layers == maximum:
        layers_to_train.append("bn1")
    for name, parameter in body.named_parameters():
        if not any(name.startswith(layer) for layer in layers_to_train):
            parameter.requires_grad_(False)


def build_model(
    architecture: str = DEFAULT_ARCHITECTURE,
    *,
    pretrained: bool = False,
    trainable_backbone_layers: int | None = None,
    freeze_backbone_batch_norm: bool = False,
):
    """Public model factory used by the training entry point."""
    return _build_model(
        architecture,
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        freeze_backbone_batch_norm=freeze_backbone_batch_norm,
    )


def build_model_with_provenance(
    architecture: str = DEFAULT_ARCHITECTURE,
    *,
    pretrained: bool = False,
    trainable_backbone_layers: int | None = None,
    freeze_backbone_batch_norm: bool = False,
):
    """Build a detector and return the exact initialization provenance."""
    return _build_model(
        architecture,
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        freeze_backbone_batch_norm=freeze_backbone_batch_norm,
        return_initialization_provenance=True,
    )


def _extract_checkpoint_payload(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise MLDetectorError("Checkpoint must contain a dictionary of model weights.")
    metadata = payload.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise MLDetectorError("Checkpoint metadata must be a dictionary.")
    metadata = _safe_metadata(metadata)

    state_dict = payload.get("state_dict")
    if state_dict is None:
        state_dict = payload.get("model_state_dict")
    if state_dict is None:
        # A raw state_dict is supported for interoperability.
        if payload and all(isinstance(key, str) for key in payload):
            state_dict = payload
        else:
            raise MLDetectorError("Checkpoint does not contain model weights.")
    if not isinstance(state_dict, dict) or not state_dict:
        raise MLDetectorError("Checkpoint model weights are missing or invalid.")
    return state_dict, metadata


def _load_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    device: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load tensor-only checkpoint content with PyTorch's restricted unpickler."""
    torch, _ = _require_ml_runtime()
    path = _safe_checkpoint_path(checkpoint_path)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise MLDetectorError(
            "This PyTorch version cannot perform restricted checkpoint loading. "
            "Upgrade to the version specified by requirements-ml.txt."
        ) from exc
    except Exception as exc:
        raise MLDetectorError(
            "The checkpoint could not be loaded safely. It may be corrupt, "
            "incompatible, or contain unsupported pickled objects."
        ) from exc
    state_dict, metadata = _extract_checkpoint_payload(payload)
    return state_dict, metadata, payload


def _canonical_metadata_bytes(value: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MLDetectorError(
            "Checkpoint metadata is not canonical JSON."
        ) from exc


def _strict_container_model(
    state_dict: dict[str, Any],
    metadata: dict[str, Any],
    *,
    torch: Any,
) -> Any:
    """Build the declared detector and strictly bind every model-state key."""
    if any(not isinstance(key, str) for key in state_dict):
        raise MLDetectorError("Checkpoint model-state keys must be strings.")
    for key, tensor in state_dict.items():
        _verify_exact_tensor_storage(
            tensor,
            torch=torch,
            field=f"model state {key!r}",
        )
    architecture = metadata.get("architecture")
    if not isinstance(architecture, str) or architecture not in SUPPORTED_ARCHITECTURES:
        raise MLDetectorError(
            "Checkpoint container does not declare a supported architecture."
        )
    trainable_layers = metadata.get("trainable_backbone_layers")
    if trainable_layers is not None and (
        isinstance(trainable_layers, bool) or not isinstance(trainable_layers, int)
    ):
        raise MLDetectorError(
            "Checkpoint trainable_backbone_layers must be an integer."
        )
    pretrained_transfer = metadata.get("pretrained_transfer_learning", False)
    if not isinstance(pretrained_transfer, bool):
        raise MLDetectorError(
            "Checkpoint pretrained_transfer_learning must be boolean."
        )
    model = _build_model(
        architecture,
        pretrained=False,
        trainable_backbone_layers=trainable_layers,
        freeze_backbone_batch_norm=pretrained_transfer,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise MLDetectorError(
            "Checkpoint model state does not exactly match the declared "
            "detector architecture."
        ) from exc
    return model


def _verify_exact_tensor_storage(
    tensor: Any,
    *,
    torch: Any,
    field: str,
) -> None:
    """Reject tensor views or storage slack that could conceal unrelated bytes."""
    if (
        type(tensor) is not torch.Tensor
        or tensor.layout != torch.strided
        or tensor.device.type != "cpu"
        or tensor.requires_grad
        or tensor.is_quantized
        or getattr(tensor, "is_nested", False)
        or not tensor.is_contiguous()
        or tensor.storage_offset() != 0
        or tensor.is_conj()
        or tensor.is_neg()
    ):
        raise MLDetectorError(
            f"Checkpoint {field} is not a plain contiguous CPU tensor."
        )
    expected_bytes = tensor.numel() * tensor.element_size()
    try:
        storage_bytes = tensor.untyped_storage().nbytes()
    except (AttributeError, RuntimeError) as exc:
        raise MLDetectorError(
            f"Checkpoint {field} backing storage could not be verified."
        ) from exc
    if storage_bytes != expected_bytes:
        raise MLDetectorError(
            f"Checkpoint {field} backing storage contains bytes outside its "
            "declared tensor."
        )


def _finite_optimizer_number(
    value: Any,
    *,
    field: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MLDetectorError(
            f"Checkpoint optimizer {field} must be a finite number."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or (positive and number == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise MLDetectorError(
            f"Checkpoint optimizer {field} must be finite and {qualifier}."
        )
    return number


def _verify_sgd_optimizer_state(
    optimizer_state: Any,
    *,
    metadata: dict[str, Any],
    model: Any,
    torch: Any,
) -> None:
    """Constrain optimizer payload to the exact generated one-group SGD schema."""
    if not isinstance(optimizer_state, dict) or set(optimizer_state) != {
        "state",
        "param_groups",
    }:
        raise MLDetectorError(
            "Checkpoint optimizer state must contain exactly state and param_groups."
        )
    training_config = metadata.get("training_config")
    if not isinstance(training_config, dict) or training_config.get("optimizer") != "SGD":
        raise MLDetectorError(
            "Checkpoint optimizer state is only accepted for declared SGD training."
        )
    learning_rate = _finite_optimizer_number(
        training_config.get("learning_rate"),
        field="learning_rate",
        positive=True,
    )
    momentum = _finite_optimizer_number(
        training_config.get("momentum"),
        field="momentum",
    )
    weight_decay = _finite_optimizer_number(
        training_config.get("weight_decay"),
        field="weight_decay",
    )
    trainable_layers = metadata.get("trainable_backbone_layers")
    configured_layers = training_config.get("trainable_backbone_layers")
    if (
        isinstance(configured_layers, bool)
        or not isinstance(configured_layers, int)
        or configured_layers != trainable_layers
    ):
        raise MLDetectorError(
            "Checkpoint optimizer trainable-backbone declaration is inconsistent."
        )

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise MLDetectorError(
            "Checkpoint optimizer has no declared trainable parameters."
        )
    reference_optimizer = torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    reference_group = reference_optimizer.state_dict()["param_groups"][0]
    parameter_groups = optimizer_state["param_groups"]
    if (
        not isinstance(parameter_groups, list)
        or len(parameter_groups) != 1
        or not isinstance(parameter_groups[0], dict)
    ):
        raise MLDetectorError(
            "Checkpoint optimizer must contain exactly one parameter group."
        )
    group = parameter_groups[0]
    if set(group) != set(reference_group):
        raise MLDetectorError(
            "Checkpoint optimizer parameter group contains unsupported fields."
        )
    expected_parameter_ids = reference_group["params"]
    if group.get("params") != expected_parameter_ids:
        raise MLDetectorError(
            "Checkpoint optimizer parameter IDs do not match the declared model."
        )
    for field, expected in reference_group.items():
        if field == "params":
            continue
        actual = group[field]
        if type(actual) is not type(expected) or actual != expected:
            raise MLDetectorError(
                f"Checkpoint optimizer parameter group {field!r} differs "
                "from the declared training configuration."
            )

    state = optimizer_state["state"]
    if not isinstance(state, dict):
        raise MLDetectorError("Checkpoint optimizer state entries must be an object.")
    expected_ids = set(expected_parameter_ids)
    if any(type(parameter_id) is not int for parameter_id in state) or not set(
        state
    ).issubset(expected_ids):
        raise MLDetectorError(
            "Checkpoint optimizer state contains an unknown parameter ID."
        )
    if momentum == 0.0 and state:
        raise MLDetectorError(
            "Checkpoint optimizer cannot contain momentum buffers when momentum is zero."
        )
    parameters_by_id = dict(zip(expected_parameter_ids, parameters))
    for parameter_id, parameter_state in state.items():
        if not isinstance(parameter_state, dict) or set(parameter_state) != {
            "momentum_buffer"
        }:
            raise MLDetectorError(
                "Checkpoint SGD state entries may contain only momentum_buffer."
            )
        buffer = parameter_state["momentum_buffer"]
        parameter = parameters_by_id[parameter_id]
        _verify_exact_tensor_storage(
            buffer,
            torch=torch,
            field=f"optimizer momentum buffer {parameter_id}",
        )
        if (
            buffer.dtype != parameter.dtype
            or tuple(buffer.shape) != tuple(parameter.shape)
        ):
            raise MLDetectorError(
                "Checkpoint SGD momentum buffer does not match its model parameter."
            )


def _verify_generated_checkpoint_container(
    payload: dict[str, Any],
    *,
    state_dict: dict[str, Any],
    metadata: dict[str, Any],
    torch: Any,
) -> int:
    """Verify the exact data-bearing contract written by format-2 training."""
    if (
        type(payload.get("format_version")) is not int
        or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION
    ):
        raise MLDetectorError(
            f"Checkpoint container must use format {CHECKPOINT_FORMAT_VERSION}."
        )
    if (
        type(metadata.get("format_version")) is not int
        or metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION
    ):
        raise MLDetectorError(
            f"Embedded checkpoint metadata must use format {CHECKPOINT_FORMAT_VERSION}."
        )
    if type(metadata.get("publication_metadata_schema_version")) is not int:
        raise MLDetectorError(
            "Checkpoint publication metadata schema must be a literal integer."
        )
    optimizer_included = metadata.get("optimizer_state_included")
    if not isinstance(optimizer_included, bool):
        raise MLDetectorError(
            "Checkpoint optimizer_state_included metadata must be boolean."
        )
    expected_keys = {"format_version", "state_dict", "metadata"}
    if optimizer_included:
        expected_keys.add("optimizer_state_dict")
    if set(payload) != expected_keys:
        raise MLDetectorError(
            "Checkpoint container top-level fields do not exactly match the "
            "generated format-2 schema."
        )

    model = _strict_container_model(state_dict, metadata, torch=torch)
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if (
        type(metadata.get("trainable_parameter_count")) is not int
        or metadata.get("trainable_parameter_count")
        != trainable_parameter_count
        or trainable_parameter_count <= 0
    ):
        raise MLDetectorError(
            "Checkpoint trainable_parameter_count does not match the strict "
            "declared model."
        )
    if optimizer_included:
        _verify_sgd_optimizer_state(
            payload["optimizer_state_dict"],
            metadata=metadata,
            model=model,
            torch=torch,
        )
    return trainable_parameter_count


def verified_checkpoint_provenance(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify sidecar claims against metadata inside a restricted-load checkpoint."""
    provenance = checkpoint_provenance(path)
    provenance["embedded_metadata_verified"] = False
    provenance["container_verified"] = False
    provenance["verified_trainable_parameter_count"] = None
    provenance["release_artifact_sanitized"] = False
    if provenance.get("metadata_verified") is not True:
        return provenance

    torch, _ = _require_ml_runtime()
    state_dict, embedded_metadata, payload = _load_checkpoint(
        path,
        device=torch.device("cpu"),
    )
    raw_embedded_metadata = payload.get("metadata")
    if not isinstance(raw_embedded_metadata, dict):
        raise MLDetectorError(
            "Restricted-load checkpoint metadata must be an object."
        )

    checkpoint = _safe_checkpoint_path(path)
    sidecar = _metadata_sidecar(checkpoint)
    try:
        raw_sidecar_metadata = _loads_unique_json(
            sidecar.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise MLDetectorError(
            "Checkpoint sidecar could not be re-read for exact comparison."
        ) from exc
    if not isinstance(raw_sidecar_metadata, dict):
        raise MLDetectorError(
            "Verified checkpoint sidecar metadata must be an object."
        )
    if raw_sidecar_metadata.get("checkpoint_sha256") != provenance["sha256"]:
        raise MLDetectorError(
            "Checkpoint sidecar SHA-256 changed during embedded verification."
        )
    raw_sidecar_metadata.pop("checkpoint_sha256", None)
    exposed_sidecar_metadata = dict(provenance["metadata"])
    exposed_sidecar_metadata.pop("checkpoint_sha256", None)
    if (
        _canonical_metadata_bytes(raw_sidecar_metadata)
        != _canonical_metadata_bytes(exposed_sidecar_metadata)
        or _canonical_metadata_bytes(raw_embedded_metadata)
        != _canonical_metadata_bytes(embedded_metadata)
    ):
        raise MLDetectorError(
            "Checkpoint metadata exceeds the safe exact-comparison contract."
        )
    if _canonical_metadata_bytes(
        raw_sidecar_metadata
    ) != _canonical_metadata_bytes(raw_embedded_metadata):
        raise MLDetectorError(
            "Checkpoint sidecar metadata does not exactly match the metadata "
            "embedded in the restricted-load checkpoint."
        )
    if payload.get("format_version") != embedded_metadata.get("format_version"):
        raise MLDetectorError(
            "Checkpoint container and embedded metadata format versions differ."
        )
    trainable_parameter_count = _verify_generated_checkpoint_container(
        payload,
        state_dict=state_dict,
        metadata=embedded_metadata,
        torch=torch,
    )
    if _sha256_file(checkpoint) != provenance["sha256"]:
        raise MLDetectorError(
            "Checkpoint changed during embedded metadata verification."
        )
    provenance["embedded_metadata_verified"] = True
    provenance["container_verified"] = True
    provenance[
        "verified_trainable_parameter_count"
    ] = trainable_parameter_count
    return provenance


def _clone_sanitized_optimizer_state(
    optimizer_state: dict[str, Any],
) -> dict[str, Any]:
    """Copy only the already-validated generated SGD primitives and tensors."""
    sanitized_state: dict[int, dict[str, Any]] = {}
    for parameter_id, parameter_state in optimizer_state["state"].items():
        buffer = parameter_state["momentum_buffer"]
        sanitized_state[int(parameter_id)] = {
            "momentum_buffer": buffer.detach().cpu().contiguous().clone()
        }
    sanitized_groups: list[dict[str, Any]] = []
    for group in optimizer_state["param_groups"]:
        sanitized_group: dict[str, Any] = {}
        for key, value in group.items():
            if key == "params":
                sanitized_group[key] = [int(item) for item in value]
            elif value is None or type(value) in {bool, int, float, str}:
                sanitized_group[key] = value
            else:
                raise MLDetectorError(
                    "Validated SGD state contained a non-primitive group value."
                )
        sanitized_groups.append(sanitized_group)
    return {"state": sanitized_state, "param_groups": sanitized_groups}


def sanitize_checkpoint_for_release(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> dict[str, Any]:
    """Write and verify a distinct release artifact with freshly owned tensors.

    The source archive is never certified for redistribution: PyTorch ZIP
    readers tolerate unreferenced members and trailing bytes. This function
    restricted-loads and validates the source, copies only declared logical
    values into new storage, writes a fresh archive and sidecar, and returns
    provenance for that newly generated artifact.
    """
    source_path = _safe_checkpoint_path(source)
    destination_path = Path(destination).expanduser().resolve()
    destination_sidecar = _metadata_sidecar(destination_path)
    if destination_path == source_path:
        raise MLDetectorError(
            "Release output must be distinct from the local training checkpoint."
        )
    if (
        destination_path.exists()
        or destination_path.is_symlink()
        or destination_sidecar.exists()
        or destination_sidecar.is_symlink()
    ):
        raise MLDetectorError(
            "Release output and sidecar must not already exist."
        )

    source_provenance = verified_checkpoint_provenance(source_path)
    if (
        source_provenance.get("embedded_metadata_verified") is not True
        or source_provenance.get("container_verified") is not True
    ):
        raise MLDetectorError(
            "Source checkpoint did not pass restricted container verification."
        )
    torch, _ = _require_ml_runtime()
    state_dict, metadata, payload = _load_checkpoint(
        source_path,
        device=torch.device("cpu"),
    )
    _verify_generated_checkpoint_container(
        payload,
        state_dict=state_dict,
        metadata=metadata,
        torch=torch,
    )
    sanitized_payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": {
            key: tensor.detach().cpu().contiguous().clone()
            for key, tensor in state_dict.items()
        },
        "metadata": dict(metadata),
    }
    if metadata["optimizer_state_included"]:
        sanitized_payload[
            "optimizer_state_dict"
        ] = _clone_sanitized_optimizer_state(
            payload["optimizer_state_dict"]
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_descriptor, checkpoint_temporary_name = tempfile.mkstemp(
        prefix=destination_path.name + ".",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    os.close(checkpoint_descriptor)
    checkpoint_temporary = Path(checkpoint_temporary_name)
    sidecar_descriptor, sidecar_temporary_name = tempfile.mkstemp(
        prefix=destination_sidecar.name + ".",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    os.close(sidecar_descriptor)
    sidecar_temporary = Path(sidecar_temporary_name)
    moved_checkpoint = False
    try:
        torch.save(sanitized_payload, checkpoint_temporary)
        release_sidecar = {
            **metadata,
            "checkpoint_sha256": _sha256_file(checkpoint_temporary),
        }
        with sidecar_temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                release_sidecar,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(checkpoint_temporary, destination_path)
        moved_checkpoint = True
        os.replace(sidecar_temporary, destination_sidecar)
    except Exception:
        if moved_checkpoint:
            destination_path.unlink(missing_ok=True)
        raise
    finally:
        checkpoint_temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)

    try:
        release_provenance = verified_checkpoint_provenance(destination_path)
    except Exception:
        destination_path.unlink(missing_ok=True)
        destination_sidecar.unlink(missing_ok=True)
        raise
    release_provenance["release_artifact_sanitized"] = True
    return release_provenance


def _resolve_device(requested: str):
    torch, _ = _require_ml_runtime()
    value = str(requested).strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise MLDetectorError(
            "CUDA was requested, but this PyTorch installation cannot access a CUDA GPU."
        )
    try:
        return torch.device(value)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise MLDetectorError(f"Invalid PyTorch device {requested!r}.") from exc


def normalize_grayscale(gray: np.ndarray) -> np.ndarray:
    """Robustly scale a grayscale image to float32 0..1."""
    arr = np.asarray(gray)
    if arr.ndim == 3:
        if arr.shape[2] < 3:
            arr = arr[..., 0]
        else:
            rgb = arr[..., :3].astype(np.float64)
            arr = (
                rgb[..., 0] * 0.2126
                + rgb[..., 1] * 0.7152
                + rgb[..., 2] * 0.0722
            )
    if arr.ndim != 2:
        raise MLDetectorError("AI band detection requires a two-dimensional image.")
    if min(arr.shape) < 2:
        raise MLDetectorError("The image is too small for AI band detection.")
    arr = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        raise MLDetectorError("The image contains no finite pixel values.")
    if not finite.all():
        arr = arr.copy()
        arr[~finite] = float(np.median(arr[finite]))
    lo, hi = np.percentile(arr, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.asarray(np.clip((arr - lo) / (hi - lo), 0.0, 1.0), dtype=np.float32)


def image_to_tensor(gray: np.ndarray):
    """Convert robustly normalized grayscale to a three-channel tensor."""
    torch, _ = _require_ml_runtime()
    normalized = normalize_grayscale(gray)
    rgb = np.repeat(normalized[None, :, :], 3, axis=0)
    return torch.from_numpy(np.ascontiguousarray(rgb))


def _validated_centers(
    values: Iterable[int | float] | None,
    *,
    count: int,
    limit: int,
    label: str,
) -> list[int] | None:
    if values is None:
        return None
    try:
        raw = list(values)
    except TypeError as exc:
        raise MLDetectorError(f"{label} centre hints must be a sequence.") from exc
    if not raw:
        return None
    if len(raw) != count:
        raise MLDetectorError(
            f"{label} centre hint count does not match the experiment setup."
        )
    try:
        centers = [int(round(float(value))) for value in raw]
    except (TypeError, ValueError, OverflowError) as exc:
        raise MLDetectorError(f"{label} centre hints must be finite pixels.") from exc
    if any(not math.isfinite(float(value)) for value in raw):
        raise MLDetectorError(f"{label} centre hints must be finite pixels.")
    if any(value < 0 or value >= limit for value in centers):
        raise MLDetectorError(f"{label} centre hints fall outside the image.")
    if any(right <= left for left, right in zip(centers[:-1], centers[1:])):
        raise MLDetectorError(f"{label} centre hints must be strictly ordered.")
    return centers


def cluster_axis_centers(
    values: Iterable[float],
    count: int,
    limit: int,
) -> tuple[list[int], bool]:
    """Cluster box centres on one axis; return centres and a fallback flag.

    The helper is pure NumPy so mapping logic can be tested without PyTorch.
    """
    if count < 1 or limit < 1:
        raise MLDetectorError("Grid dimensions must be positive.")
    raw = np.asarray(list(values), dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    raw = np.clip(raw, 0.0, max(limit - 1, 0))
    fallback = len(raw) < count or len(np.unique(np.round(raw, 6))) < count
    if fallback:
        if count == 1 and len(raw):
            return [int(np.clip(round(float(np.median(raw))), 0, limit - 1))], True
        centers = np.linspace(limit * 0.15, limit * 0.85, count)
        return [int(np.clip(round(value), 0, limit - 1)) for value in centers], True

    ordered = np.sort(raw)
    quantiles = (np.arange(count) + 0.5) / count
    centers = np.quantile(ordered, quantiles)
    for _ in range(50):
        labels = np.argmin(np.abs(raw[:, None] - centers[None, :]), axis=1)
        updated = centers.copy()
        for index in range(count):
            members = raw[labels == index]
            if len(members):
                updated[index] = float(np.median(members))
        updated.sort()
        if float(np.max(np.abs(updated - centers))) < 0.01:
            centers = updated
            break
        centers = updated
    rounded = [int(np.clip(round(value), 0, limit - 1)) for value in centers]
    if any(right <= left for left, right in zip(rounded[:-1], rounded[1:])):
        rounded = [
            int(np.clip(round(value), 0, limit - 1))
            for value in np.linspace(limit * 0.15, limit * 0.85, count)
        ]
        fallback = True
    return rounded, fallback


def map_boxes_to_grid(
    boxes: Iterable[Iterable[float]],
    targets: Iterable[str],
    lanes: Iterable[str],
    image_shape: tuple[int, int],
    *,
    lane_centers_hint: Iterable[int | float] | None = None,
    target_centers_hint: Iterable[int | float] | None = None,
) -> tuple[list[tuple[str, str]], list[int], list[int], list[str]]:
    """Assign every predicted box to its nearest target row and lane."""
    targets = [str(value) for value in targets]
    lanes = [str(value) for value in lanes]
    if not targets or not lanes:
        raise MLDetectorError("At least one target and one lane are required.")
    if len(image_shape) != 2:
        raise MLDetectorError("Image shape must be (height, width).")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height < 1 or width < 1:
        raise MLDetectorError("Image dimensions must be positive.")
    box_array = np.asarray(list(boxes), dtype=np.float64)
    if box_array.ndim != 2 or box_array.shape[1] != 4 or not len(box_array):
        raise MLDetectorError("No valid model boxes are available for grid mapping.")
    centers_x = (box_array[:, 0] + box_array[:, 2]) / 2.0
    centers_y = (box_array[:, 1] + box_array[:, 3]) / 2.0

    lane_centers = _validated_centers(
        lane_centers_hint, count=len(lanes), limit=width, label="Lane"
    )
    target_centers = _validated_centers(
        target_centers_hint, count=len(targets), limit=height, label="Target"
    )
    warnings: list[str] = []
    if lane_centers is None:
        lane_centers, used_fallback = cluster_axis_centers(
            centers_x, len(lanes), width
        )
        if used_fallback:
            warnings.append(
                "Too few distinct AI boxes were available to infer every lane; "
                "evenly spaced lane centres were used. Review assignments."
            )
    if target_centers is None:
        target_centers, used_fallback = cluster_axis_centers(
            centers_y, len(targets), height
        )
        if used_fallback:
            warnings.append(
                "Too few distinct AI boxes were available to infer every target row; "
                "evenly spaced target centres were used. Review assignments."
            )

    assignments: list[tuple[str, str]] = []
    lane_array = np.asarray(lane_centers, dtype=np.float64)
    target_array = np.asarray(target_centers, dtype=np.float64)
    for center_x, center_y in zip(centers_x, centers_y):
        lane_index = int(np.argmin(np.abs(lane_array - center_x)))
        target_index = int(np.argmin(np.abs(target_array - center_y)))
        assignments.append((targets[target_index], lanes[lane_index]))
    return assignments, lane_centers, target_centers, warnings


def predict_bands(
    gray: np.ndarray,
    targets: Iterable[str],
    lanes: Iterable[str],
    checkpoint_path: str | os.PathLike[str],
    *,
    score_threshold: float = 0.5,
    review_threshold: float = 0.75,
    device: str = "auto",
    lane_centers_hint: Iterable[int | float] | None = None,
    target_centers_hint: Iterable[int | float] | None = None,
    polarity: str = "auto",
) -> detection.DetectionResult:
    """Infer band boxes and map them to the nearest target-row/lane grid."""
    torch, _ = _require_ml_runtime()
    targets = [str(value) for value in targets]
    lanes = [str(value) for value in lanes]
    if not targets or not lanes:
        raise MLDetectorError("At least one target and one lane are required.")
    if polarity not in {"auto", "dark", "light"}:
        raise MLDetectorError("polarity must be 'auto', 'dark', or 'light'.")
    try:
        score_threshold = float(score_threshold)
        review_threshold = float(review_threshold)
    except (TypeError, ValueError) as exc:
        raise MLDetectorError("Confidence thresholds must be numeric.") from exc
    if not 0.0 <= score_threshold <= 1.0:
        raise MLDetectorError("score_threshold must be between 0 and 1.")
    if not score_threshold <= review_threshold <= 1.0:
        raise MLDetectorError(
            "review_threshold must be between score_threshold and 1."
        )

    normalized = normalize_grayscale(gray)
    height, width = normalized.shape
    resolved_device = _resolve_device(device)
    provenance = checkpoint_provenance(checkpoint_path)
    if provenance.get("metadata_verified") is False:
        raise MLDetectorError(
            "Checkpoint SHA-256 does not match its metadata sidecar. "
            "Refusing to run a modified checkpoint."
        )
    state_dict, metadata, _ = _load_checkpoint(
        checkpoint_path, device=torch.device("cpu")
    )
    architecture = str(metadata.get("architecture", DEFAULT_ARCHITECTURE))
    model = _build_model(architecture, pretrained=False)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise MLDetectorError(
            "Checkpoint weights do not match the declared detector architecture."
        ) from exc
    try:
        model.to(resolved_device)
        model.eval()
        image = torch.from_numpy(
            np.ascontiguousarray(np.repeat(normalized[None, :, :], 3, axis=0))
        ).to(resolved_device)
        with torch.inference_mode():
            output = model([image])[0]
        raw_boxes = output["boxes"].detach().cpu().numpy()
        raw_scores = output["scores"].detach().cpu().numpy()
        raw_labels = output["labels"].detach().cpu().numpy()
    except Exception as exc:
        raise MLDetectorError("AI detector inference failed for this image.") from exc

    valid: list[tuple[list[float], float]] = []
    for raw_box, raw_score, raw_label in zip(raw_boxes, raw_scores, raw_labels):
        score = float(raw_score)
        if int(raw_label) != 1 or not math.isfinite(score) or score < score_threshold:
            continue
        if len(raw_box) != 4 or not np.all(np.isfinite(raw_box)):
            continue
        x0 = float(np.clip(raw_box[0], 0.0, width))
        y0 = float(np.clip(raw_box[1], 0.0, height))
        x1 = float(np.clip(raw_box[2], 0.0, width))
        y1 = float(np.clip(raw_box[3], 0.0, height))
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            continue
        valid.append(([x0, y0, x1, y1], score))
    valid.sort(key=lambda item: item[1], reverse=True)
    valid = valid[:512]
    if not valid:
        raise MLDetectorError(
            "The AI model produced no valid band predictions at the selected "
            "score threshold."
        )

    assignments, lane_centers, target_centers, warnings = map_boxes_to_grid(
        [item[0] for item in valid],
        targets,
        lanes,
        (height, width),
        lane_centers_hint=lane_centers_hint,
        target_centers_hint=target_centers_hint,
    )
    bands: list[detection.DetectedBand] = []
    for (box, score), (target, lane) in zip(valid, assignments):
        x0, y0, x1, y1 = box
        x = max(0, int(math.floor(x0)))
        y = max(0, int(math.floor(y0)))
        x2 = min(width, int(math.ceil(x1)))
        y2 = min(height, int(math.ceil(y1)))
        needs_review = score < review_threshold
        bands.append(
            detection.DetectedBand(
                target=target,
                lane=lane,
                x=x,
                y=y,
                w=max(1, x2 - x),
                h=max(1, y2 - y),
                confidence=round(score, 4),
                source="ml",
                status="low confidence" if needs_review else "accepted",
                review_required=needs_review,
                component_count=1,
                profile_span=float(max(1, y2 - y)),
            )
        )
    detection._standardize_boxes(bands, (height, width))
    bands.sort(
        key=lambda band: (
            targets.index(band.target),
            lanes.index(band.lane),
            band.y,
            band.x,
        )
    )

    if polarity == "auto":
        bands_are_dark, polarity_confidence = detection.infer_band_polarity(
            normalized, len(lanes), len(targets)
        )
        if polarity_confidence < 0.08:
            warnings.append(
                "Band polarity is ambiguous; verify dark/light polarity before "
                "densitometry."
            )
    else:
        bands_are_dark = polarity == "dark"
        polarity_confidence = 1.0
    review_count = sum(band.review_required for band in bands)
    if review_count:
        warnings.append(
            f"{review_count} AI prediction(s) are below the review threshold "
            "and require visual confirmation."
        )
    warnings.append(
        "AI predictions are starting ROIs only; visually review every box before "
        "quantification."
    )
    return detection.DetectionResult(
        bands=bands,
        lane_centers=lane_centers,
        target_centers=target_centers,
        bands_are_dark=bool(bands_are_dark),
        polarity_confidence=float(polarity_confidence),
        warnings=warnings,
    )
