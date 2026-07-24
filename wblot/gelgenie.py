"""Verified GelGenie ONNX inference for editable BlotGrid band proposals.

GelGenie is used only to locate likely band pixels.  This module converts the
segmentation into rectangular proposals that flow through BlotGrid's existing
human-review and classical densitometry path; model-selected pixels are never
used directly for quantitative integration.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import stat
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import binary_closing, find_objects, label

from . import detection
from .ml_detector import map_boxes_to_grid


MODEL_ID = "mattaq/GelGenie-Universal-Dec-2023"
MODEL_REVISION = "fc36cc1ea468308b7336939a66c9884a26752d7e"
MODEL_FILENAME = "onnx_checkpoints/unet_dec_21_epoch_579.onnx"
MODEL_URL = (
    f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/"
    f"{MODEL_FILENAME}?download=true"
)
MODEL_SHA256 = "10ea972a40625f03b5a2987e1a98b532c6d4ded1daae32f1eb91aa24db8d9c48"
MODEL_SIZE_BYTES = 57_279_486
MODEL_LICENSE = "Apache-2.0"
MODEL_DOI = "10.1038/s41467-025-59189-0"
DEFAULT_MODEL_PATH = "models/gelgenie_universal.onnx"
_MAX_MODEL_BYTES = 128 * 1024**2
_MAX_INPUT_PIXELS = 16_777_216


class GelGenieError(ValueError):
    """A recoverable GelGenie model, image, or inference error."""


def _require_onnxruntime():
    try:
        import onnxruntime
    except Exception as exc:
        raise GelGenieError(
            "GelGenie inference requires ONNX Runtime. Re-run "
            "scripts/setup_ubuntu.sh or install requirements.txt."
        ) from exc
    return onnxruntime


def _safe_model_path(path: str | os.PathLike[str]) -> Path:
    raw = os.fspath(path)
    if not raw.strip():
        raise GelGenieError("A GelGenie ONNX model path is required.")
    if "://" in raw:
        raise GelGenieError(
            "Use a downloaded local ONNX model, not a URL. Run "
            "scripts/download_pretrained_models.py."
        )
    candidate = Path(raw).expanduser()
    if candidate.is_symlink():
        raise GelGenieError("GelGenie model symlinks are not accepted.")
    try:
        info = candidate.stat()
    except OSError as exc:
        raise GelGenieError(
            f"GelGenie model is not readable: {candidate}. Run "
            "scripts/download_pretrained_models.py."
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise GelGenieError("The GelGenie path must name a regular file.")
    if candidate.suffix.lower() != ".onnx":
        raise GelGenieError("The GelGenie model must be an .onnx file.")
    if info.st_size < 1024 or info.st_size > _MAX_MODEL_BYTES:
        raise GelGenieError("The GelGenie model file has an unsafe size.")
    return candidate.resolve()


def _sha256_file(path: Path, *, chunk_size: int = 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def model_provenance(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return immutable model identity and refuse unverified weights later."""
    model_path = _safe_model_path(path)
    digest = _sha256_file(model_path)
    return {
        "backend": "gelgenie_hybrid",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file": MODEL_FILENAME,
        "path": str(model_path),
        "sha256": digest,
        "size_bytes": int(model_path.stat().st_size),
        "verified_official": (
            digest == MODEL_SHA256
            and int(model_path.stat().st_size) == MODEL_SIZE_BYTES
        ),
        "expected_sha256": MODEL_SHA256,
        "license": MODEL_LICENSE,
        "doi": MODEL_DOI,
        "architecture": "SMP-UNet ResNet18",
        "onnx_opset": 12,
        "western_blot_validated": False,
        "human_review_required": True,
        "padding": "symmetric_zero_next_strict_multiple_32",
        "postprocessing": "two_class_softmax; connectivity=8; component_boxes",
    }


def _verified_model(path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    provenance = model_provenance(path)
    if not provenance["verified_official"]:
        raise GelGenieError(
            "GelGenie model SHA-256 or size does not match the pinned official "
            "Universal model. Refusing to execute unverified ONNX weights."
        )
    return Path(provenance["path"]), provenance


def available_providers() -> list[str]:
    """Return ONNX Runtime providers without importing it at module import time."""
    onnxruntime = _require_onnxruntime()
    return [str(provider) for provider in onnxruntime.get_available_providers()]


def runtime_provider(device: str = "auto") -> str:
    """Resolve the provider that would execute the pinned model."""
    return _providers_for_device(device, available_providers())[0]


def _providers_for_device(device: str, available: Iterable[str]) -> list[str]:
    requested = str(device).strip().lower()
    available = list(available)
    if requested not in {"auto", "cpu", "cuda"}:
        raise GelGenieError("GelGenie device must be auto, cpu, or cuda.")
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise GelGenieError(
                "CUDA was requested for GelGenie, but this ONNX Runtime build "
                "does not expose CUDAExecutionProvider. Use CPU or install a "
                "matched onnxruntime-gpu build."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if requested == "auto" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CPUExecutionProvider" not in available:
        raise GelGenieError("ONNX Runtime does not expose CPUExecutionProvider.")
    return ["CPUExecutionProvider"]


@lru_cache(maxsize=3)
def _load_session(
    path: str,
    digest: str,
    providers: tuple[str, ...],
):
    """Cache the immutable model session across Streamlit reruns."""
    del digest
    onnxruntime = _require_onnxruntime()
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    try:
        return onnxruntime.InferenceSession(
            path,
            sess_options=options,
            providers=list(providers),
        )
    except Exception as exc:
        raise GelGenieError("Could not initialize the GelGenie ONNX model.") from exc


def prepare_model_input(
    gray: np.ndarray,
    *,
    max_value: float,
    invert: bool,
    normalization: str = "auto",
) -> tuple[np.ndarray, tuple[int, int, int, int], str]:
    """Match GelGenie's 1-channel datatype scaling and divisible-by-32 padding."""
    arr = np.asarray(gray, dtype=np.float32)
    if arr.ndim != 2:
        raise GelGenieError(
            "GelGenie requires a two-dimensional grayscale image."
        )
    if min(arr.shape) < 8:
        raise GelGenieError("The image is too small for GelGenie inference.")
    if int(arr.shape[0]) * int(arr.shape[1]) > _MAX_INPUT_PIXELS:
        raise GelGenieError(
            "The image exceeds GelGenie's safe 16.8-megapixel inference limit. "
            "Crop the blot to the membrane region before detection."
        )
    finite = np.isfinite(arr)
    if not finite.any():
        raise GelGenieError("The image contains no finite pixels.")
    if not finite.all():
        arr = arr.copy()
        arr[~finite] = float(np.median(arr[finite]))
    try:
        scale = float(max_value)
    except (TypeError, ValueError) as exc:
        raise GelGenieError("Image max_value must be numeric.") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise GelGenieError("Image max_value must be positive and finite.")
    requested_normalization = str(normalization).strip().lower()
    if requested_normalization not in {"auto", "dtype_max", "percentile"}:
        raise GelGenieError(
            "GelGenie normalization must be auto, dtype_max, or percentile."
        )
    effective_normalization = requested_normalization
    if effective_normalization == "auto":
        effective_normalization = "percentile" if scale > 255.0 else "dtype_max"
    if effective_normalization == "percentile":
        low, high = np.percentile(arr, [0.1, 99.9])
        if math.isfinite(float(low)) and math.isfinite(float(high)) and high > low:
            normalized = np.clip(arr, low, high)
            normalized = (normalized - float(low)) / float(high - low)
        else:
            effective_normalization = "dtype_max"
            normalized = np.clip(arr, 0.0, scale) / scale
    else:
        normalized = np.clip(arr, 0.0, scale) / scale
    if invert:
        normalized = 1.0 - normalized

    height, width = normalized.shape
    padded_height = 32 * (height // 32 + 1)
    padded_width = 32 * (width // 32 + 1)
    top = (padded_height - height) // 2
    bottom = padded_height - height - top
    left = (padded_width - width) // 2
    right = padded_width - width - left
    padded = np.pad(
        normalized,
        ((top, bottom), (left, right)),
        mode="constant",
        constant_values=0.0,
    )
    tensor = np.ascontiguousarray(padded[None, None, :, :], dtype=np.float32)
    return tensor, (top, bottom, left, right), effective_normalization


def segmentation_components(
    logits: np.ndarray,
    *,
    original_shape: tuple[int, int],
    padding: tuple[int, int, int, int],
    probability_threshold: float = 0.5,
) -> tuple[list[list[float]], list[float], dict[str, int]]:
    """Convert two-class GelGenie logits into horizontal component boxes."""
    try:
        threshold = float(probability_threshold)
    except (TypeError, ValueError) as exc:
        raise GelGenieError("GelGenie probability threshold must be numeric.") from exc
    if not 0.05 <= threshold <= 0.95:
        raise GelGenieError(
            "GelGenie probability threshold must be between 0.05 and 0.95."
        )
    scores = np.asarray(logits, dtype=np.float32)
    if scores.ndim != 4 or scores.shape[0] != 1 or scores.shape[1] != 2:
        raise GelGenieError(
            "GelGenie returned an unexpected output shape; expected [1, 2, H, W]."
        )
    top, bottom, left, right = (int(value) for value in padding)
    height, width = (int(value) for value in original_shape)
    if min(height, width) < 1:
        raise GelGenieError("Original image dimensions must be positive.")
    expected_height = top + height + bottom
    expected_width = left + width + right
    if scores.shape[2:] != (expected_height, expected_width):
        raise GelGenieError(
            "GelGenie output dimensions do not match the padded model input."
        )

    # For two-class softmax, P(band) = sigmoid(band_logit - background_logit).
    delta = np.clip(scores[0, 1] - scores[0, 0], -80.0, 80.0)
    probability = 1.0 / (1.0 + np.exp(-delta))
    probability = probability[top:top + height, left:left + width]
    binary = probability >= threshold

    # Join tiny horizontal gaps without merging vertically adjacent doublets.
    closing_width = max(1, min(5, int(round(width / 600.0))))
    if closing_width > 1:
        binary = binary_closing(
            binary,
            structure=np.ones((1, closing_width), dtype=bool),
        )
    component_labels, component_count = label(
        binary,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    slices = find_objects(component_labels)
    minimum_area = max(4, int(round(height * width * 1e-6)))
    maximum_area = max(minimum_area + 1, int(round(height * width * 0.25)))
    boxes: list[list[float]] = []
    confidences: list[float] = []
    rejected_small = 0
    rejected_large = 0
    for component_id, component_slice in enumerate(slices, start=1):
        if component_slice is None:
            continue
        y_slice, x_slice = component_slice
        component = component_labels[y_slice, x_slice] == component_id
        area = int(np.count_nonzero(component))
        box_height = int(y_slice.stop - y_slice.start)
        box_width = int(x_slice.stop - x_slice.start)
        if area < minimum_area or box_width < 3 or box_height < 1:
            rejected_small += 1
            continue
        if area > maximum_area:
            rejected_large += 1
            continue
        component_probability = probability[y_slice, x_slice][component]
        confidence = float(np.mean(component_probability))
        boxes.append(
            [
                float(x_slice.start),
                float(y_slice.start),
                float(x_slice.stop),
                float(y_slice.stop),
            ]
        )
        confidences.append(confidence)

    ranked = sorted(
        zip(boxes, confidences),
        key=lambda item: item[1]
        * math.sqrt(max(1.0, (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))),
        reverse=True,
    )[:512]
    return (
        [item[0] for item in ranked],
        [item[1] for item in ranked],
        {
            "raw_components": int(component_count),
            "rejected_small": int(rejected_small),
            "rejected_large": int(rejected_large),
            "accepted_components": int(len(ranked)),
        },
    )


def predict_bands(
    gray: np.ndarray,
    targets: Iterable[str],
    lanes: Iterable[str],
    model_path: str | os.PathLike[str],
    *,
    max_value: float,
    probability_threshold: float = 0.5,
    review_threshold: float = 0.75,
    device: str = "auto",
    normalization: str = "auto",
    lane_centers_hint: Iterable[int | float] | None = None,
    target_centers_hint: Iterable[int | float] | None = None,
    polarity: str = "auto",
) -> detection.DetectionResult:
    """Segment likely bands and map component boxes to the experiment grid."""
    targets = [str(value) for value in targets]
    lanes = [str(value) for value in lanes]
    if not targets or not lanes:
        raise GelGenieError("At least one target and one lane are required.")
    if polarity not in {"auto", "dark", "light"}:
        raise GelGenieError("polarity must be auto, dark, or light.")
    try:
        probability_threshold = float(probability_threshold)
        review_threshold = float(review_threshold)
    except (TypeError, ValueError) as exc:
        raise GelGenieError("GelGenie thresholds must be numeric.") from exc
    if not probability_threshold <= review_threshold <= 1.0:
        raise GelGenieError(
            "GelGenie review threshold must be at least the probability threshold."
        )

    model, provenance = _verified_model(model_path)
    providers = _providers_for_device(device, available_providers())
    session = _load_session(
        str(model),
        str(provenance["sha256"]),
        tuple(providers),
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) < 1:
        raise GelGenieError("GelGenie ONNX input/output metadata is unexpected.")

    if polarity == "auto":
        bands_are_dark, polarity_confidence = detection.infer_band_polarity(
            gray, len(lanes), len(targets)
        )
    else:
        bands_are_dark = polarity == "dark"
        polarity_confidence = 1.0
    # The Universal model expects bright bands on a dark background. Invert
    # conventional dark-on-light film only on this disposable model copy.
    model_input, padding, effective_normalization = prepare_model_input(
        gray,
        max_value=max_value,
        invert=bands_are_dark,
        normalization=normalization,
    )
    try:
        output = session.run(
            [outputs[0].name],
            {inputs[0].name: model_input},
        )[0]
    except Exception as exc:
        raise GelGenieError("GelGenie ONNX inference failed for this image.") from exc

    boxes, confidences, component_summary = segmentation_components(
        output,
        original_shape=np.asarray(gray).shape,
        padding=padding,
        probability_threshold=probability_threshold,
    )
    if not boxes:
        raise GelGenieError(
            "GelGenie produced no usable band components at the selected "
            "probability threshold."
        )
    assignments, lane_centers, target_centers, warnings = map_boxes_to_grid(
        boxes,
        targets,
        lanes,
        np.asarray(gray).shape,
        lane_centers_hint=lane_centers_hint,
        target_centers_hint=target_centers_hint,
    )
    bands: list[detection.DetectedBand] = []
    height, width = np.asarray(gray).shape
    for box, confidence, (target, lane) in zip(
        boxes, confidences, assignments
    ):
        x0, y0, x1, y1 = box
        x = max(0, int(math.floor(x0)))
        y = max(0, int(math.floor(y0)))
        x2 = min(width, int(math.ceil(x1)))
        y2 = min(height, int(math.ceil(y1)))
        low_confidence = confidence < review_threshold
        bands.append(
            detection.DetectedBand(
                target=target,
                lane=lane,
                x=x,
                y=y,
                w=max(1, x2 - x),
                h=max(1, y2 - y),
                confidence=round(float(confidence), 4),
                source="gelgenie",
                status=(
                    "low confidence"
                    if low_confidence
                    else "pretrained proposal"
                ),
                # GelGenie was not trained or independently validated on
                # Western blots, so every proposal requires visual approval.
                review_required=True,
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

    if polarity_confidence < 0.08:
        warnings.append(
            "Band polarity is ambiguous; verify dark/light polarity before "
            "densitometry."
        )
    filtered_count = (
        component_summary["rejected_small"]
        + component_summary["rejected_large"]
    )
    if filtered_count:
        warnings.append(
            f"GelGenie filtered {filtered_count} implausibly small or large "
            "segmentation component(s)."
        )
    low_confidence_count = sum(
        band.status == "low confidence" for band in bands
    )
    if low_confidence_count:
        warnings.append(
            f"{low_confidence_count} GelGenie proposal(s) are below the "
            "review threshold."
        )
    warnings.extend(
        [
            f"GelGenie model input used {effective_normalization} normalization "
            f"and {'was' if bands_are_dark else 'was not'} inverted.",
            "GelGenie was pretrained on electrophoresis gels, not specifically "
            "Western blots; treat this result as an out-of-domain proposal.",
            "GelGenie masks locate candidate bands only. BlotGrid keeps shared "
            "editable rectangles and measures the original native-depth pixels.",
        ]
    )
    return detection.DetectionResult(
        bands=bands,
        lane_centers=lane_centers,
        target_centers=target_centers,
        bands_are_dark=bool(bands_are_dark),
        polarity_confidence=float(polarity_confidence),
        warnings=warnings,
    )
