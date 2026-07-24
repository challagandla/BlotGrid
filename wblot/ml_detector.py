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
from typing import Any, Iterable

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
CHECKPOINT_FORMAT_VERSION = 1
_MAX_CHECKPOINT_BYTES = 8 * 1024**3
_MAX_METADATA_BYTES = 1024**2


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


def _sha256_file(path: Path, *, chunk_size: int = 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            parsed = json.loads(sidecar.read_text(encoding="utf-8"))
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

        if architecture == DEFAULT_ARCHITECTURE:
            weights = (
                detection_models.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
                if pretrained
                else None
            )
            model = detection_models.fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights,
                weights_backbone=None,
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        elif architecture == "fasterrcnn_resnet50_fpn_v2":
            weights = (
                detection_models.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                if pretrained
                else None
            )
            model = detection_models.fasterrcnn_resnet50_fpn_v2(
                weights=weights,
                weights_backbone=None,
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        else:
            if architecture == "fcos_resnet50_fpn":
                weights = (
                    detection_models.FCOS_ResNet50_FPN_Weights.DEFAULT
                    if pretrained
                    else None
                )
                model = detection_models.fcos_resnet50_fpn(
                    weights=weights,
                    weights_backbone=None,
                )
            else:
                weights = (
                    detection_models.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
                    if pretrained
                    else None
                )
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
