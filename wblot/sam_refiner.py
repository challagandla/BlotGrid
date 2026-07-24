"""Optional SAM 2.1 assistant for recentering fixed-size BlotGrid ROIs.

SAM masks never define quantitative pixels here.  A verified local checkpoint
and an existing rectangular ROI are used to propose a better centre; BlotGrid
retains the globally locked ROI width/height and requires renewed human review.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import stat
from typing import Any

import numpy as np
from scipy.ndimage import label


BACKEND = "sam2.1_hiera_small"
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_small.pt"
)
CHECKPOINT_SHA256 = (
    "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
)
CHECKPOINT_SIZE_BYTES = 184_416_285
DEFAULT_CHECKPOINT_PATH = "models/sam2.1_hiera_small.pt"
CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_s.yaml"
LICENSE = "Apache-2.0"
UPSTREAM_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
_MAX_CHECKPOINT_BYTES = 384 * 1024**2


class SAMRefinerError(ValueError):
    """A recoverable optional-runtime, checkpoint, prompt, or mask error."""


@dataclass(frozen=True)
class RefinementResult:
    """Serializable evidence for one SAM position proposal."""

    backend: str
    checkpoint_sha256: str
    device: str
    crop_box_xyxy: tuple[int, int, int, int]
    prompt_box_xyxy: tuple[float, float, float, float]
    old_box_xywh: tuple[int, int, int, int]
    proposed_center_xy: tuple[float, float]
    mask_box_xyxy: tuple[int, int, int, int]
    mask_score: float
    mask_area: int
    displacement_pixels: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path, *, chunk_size: int = 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_checkpoint_path(path: str | os.PathLike[str]) -> Path:
    raw = os.fspath(path)
    if not raw.strip():
        raise SAMRefinerError("A SAM 2.1 checkpoint path is required.")
    if "://" in raw:
        raise SAMRefinerError(
            "Use a verified local checkpoint, not a URL. Run "
            "scripts/download_pretrained_models.py --model sam2.1-hiera-small."
        )
    candidate = Path(raw).expanduser()
    if candidate.is_symlink():
        raise SAMRefinerError("SAM checkpoint symlinks are not accepted.")
    try:
        info = candidate.stat()
    except OSError as exc:
        raise SAMRefinerError(
            f"SAM checkpoint is not readable: {candidate}. Run "
            "scripts/download_pretrained_models.py --model sam2.1-hiera-small."
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise SAMRefinerError("The SAM checkpoint path must be a regular file.")
    if info.st_size < 1024 or info.st_size > _MAX_CHECKPOINT_BYTES:
        raise SAMRefinerError("The SAM checkpoint file has an unsafe size.")
    return candidate.resolve()


def checkpoint_provenance(path: str | os.PathLike[str]) -> dict[str, Any]:
    checkpoint = _safe_checkpoint_path(path)
    digest = _sha256_file(checkpoint)
    return {
        "backend": BACKEND,
        "path": str(checkpoint),
        "sha256": digest,
        "size_bytes": int(checkpoint.stat().st_size),
        "verified_official": (
            digest == CHECKPOINT_SHA256
            and checkpoint.stat().st_size == CHECKPOINT_SIZE_BYTES
        ),
        "expected_sha256": CHECKPOINT_SHA256,
        "license": LICENSE,
        "upstream_revision": UPSTREAM_REVISION,
        "config": CONFIG_NAME,
        "western_blot_validated": False,
        "quantitative_mask_used": False,
        "human_review_required": True,
    }


def _verified_checkpoint(
    path: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any]]:
    provenance = checkpoint_provenance(path)
    if not provenance["verified_official"]:
        raise SAMRefinerError(
            "SAM checkpoint SHA-256 or size does not match the pinned official "
            "SAM 2.1 Hiera Small checkpoint. Refusing to load it."
        )
    return Path(provenance["path"]), provenance


def runtime_status(
    checkpoint: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Report optional package/checkpoint readiness without importing Torch."""
    package_available = importlib.util.find_spec("sam2") is not None
    result: dict[str, Any] = {
        "backend": BACKEND,
        "package_available": package_available,
        "checkpoint_available": False,
        "checkpoint_verified": False,
    }
    if checkpoint:
        candidate = Path(checkpoint).expanduser()
        result["checkpoint_available"] = candidate.is_file()
        if candidate.is_file():
            try:
                result["checkpoint_verified"] = bool(
                    checkpoint_provenance(candidate)["verified_official"]
                )
            except SAMRefinerError:
                pass
    return result


def _require_runtime():
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise SAMRefinerError(
            "SAM 2.1 refinement is optional. Install it with "
            "scripts/setup_ubuntu.sh --with-sam2."
        ) from exc
    return torch, build_sam2, SAM2ImagePredictor


def _resolve_device(requested: str) -> str:
    torch, _, _ = _require_runtime()
    value = str(requested).strip().lower()
    if value not in {"auto", "cpu", "cuda"}:
        raise SAMRefinerError("SAM device must be auto, cpu, or cuda.")
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise SAMRefinerError(
            "CUDA was requested for SAM, but PyTorch cannot access a CUDA GPU."
        )
    return value


@lru_cache(maxsize=2)
def _load_predictor(path: str, digest: str, device: str):
    del digest
    _, build_sam2, predictor_type = _require_runtime()
    try:
        model = build_sam2(
            CONFIG_NAME,
            path,
            device=device,
            apply_postprocessing=True,
        )
        return predictor_type(model)
    except Exception as exc:
        raise SAMRefinerError(
            "Could not initialize the pinned SAM 2.1 Hiera Small model."
        ) from exc


def _finite_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] != 3:
            arr = arr[:, :, 0]
    if arr.ndim not in {2, 3}:
        raise SAMRefinerError("SAM refinement requires a 2D or RGB image.")
    numeric = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(numeric)
    if not finite.any():
        raise SAMRefinerError("SAM refinement image has no finite pixels.")
    if not finite.all():
        numeric = numeric.copy()
        numeric[~finite] = float(np.median(numeric[finite]))
    low, high = np.percentile(numeric, [0.5, 99.5])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        raise SAMRefinerError("SAM refinement image range is invalid.")
    if high <= low:
        converted = np.zeros_like(numeric, dtype=np.uint8)
    else:
        converted = (
            np.clip((numeric - low) / (high - low), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    if converted.ndim == 2:
        converted = np.repeat(converted[:, :, None], 3, axis=2)
    return np.ascontiguousarray(converted)


def prepare_roi_crop(
    display_image: np.ndarray,
    roi: dict,
    *,
    context_scale: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Return local RGB context, crop-relative XYXY prompt, and crop bounds."""
    image = np.asarray(display_image)
    if image.ndim not in {2, 3}:
        raise SAMRefinerError("SAM refinement image must be 2D or RGB.")
    image_height, image_width = image.shape[:2]
    try:
        x = int(round(float(roi["x"])))
        y = int(round(float(roi["y"])))
        width = int(round(float(roi["w"])))
        height = int(round(float(roi["h"])))
        context_scale = float(context_scale)
    except (KeyError, TypeError, ValueError) as exc:
        raise SAMRefinerError("ROI geometry is incomplete or invalid.") from exc
    if width < 1 or height < 1 or image_width < 1 or image_height < 1:
        raise SAMRefinerError("ROI and image dimensions must be positive.")
    if not math.isfinite(context_scale) or not 2.0 <= context_scale <= 12.0:
        raise SAMRefinerError("SAM context scale must be between 2 and 12.")
    x = min(max(x, 0), max(0, image_width - width))
    y = min(max(y, 0), max(0, image_height - height))
    width = min(width, image_width)
    height = min(height, image_height)

    context_width = min(
        image_width,
        max(96, int(math.ceil(width * context_scale))),
    )
    context_height = min(
        image_height,
        max(96, int(math.ceil(height * context_scale * 1.5))),
    )
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    crop_x0 = int(round(center_x - context_width / 2.0))
    crop_y0 = int(round(center_y - context_height / 2.0))
    crop_x0 = min(max(crop_x0, 0), image_width - context_width)
    crop_y0 = min(max(crop_y0, 0), image_height - context_height)
    crop_x1 = crop_x0 + context_width
    crop_y1 = crop_y0 + context_height
    crop = _finite_uint8_rgb(image[crop_y0:crop_y1, crop_x0:crop_x1])
    prompt = np.asarray(
        [
            float(x - crop_x0),
            float(y - crop_y0),
            float(x + width - crop_x0),
            float(y + height - crop_y0),
        ],
        dtype=np.float32,
    )
    return crop, prompt, (crop_x0, crop_y0, crop_x1, crop_y1)


def _selected_component(
    mask: np.ndarray,
    *,
    prompt_center_xy: tuple[float, float],
    maximum_area: int,
) -> tuple[float, float, tuple[int, int, int, int], int]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise SAMRefinerError("SAM returned an empty or malformed mask.")
    labels, count = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count < 1:
        raise SAMRefinerError("SAM returned no connected mask component.")
    prompt_x, prompt_y = prompt_center_xy
    candidates: list[
        tuple[float, float, float, tuple[int, int, int, int], int]
    ] = []
    for component_id in range(1, count + 1):
        yy, xx = np.nonzero(labels == component_id)
        area = int(len(xx))
        if area < 3 or area > maximum_area:
            continue
        center_x = float(np.mean(xx))
        center_y = float(np.mean(yy))
        distance = math.hypot(center_x - prompt_x, center_y - prompt_y)
        candidates.append(
            (
                distance,
                center_x,
                center_y,
                (
                    int(np.min(xx)),
                    int(np.min(yy)),
                    int(np.max(xx)) + 1,
                    int(np.max(yy)) + 1,
                ),
                area,
            )
        )
    if not candidates:
        raise SAMRefinerError(
            "SAM mask components were empty or implausibly large."
        )
    _, center_x, center_y, box, area = min(candidates, key=lambda item: item[0])
    return center_x, center_y, box, area


def refine_box(
    display_image: np.ndarray,
    roi: dict,
    checkpoint_path: str | os.PathLike[str],
    *,
    device: str = "auto",
    context_scale: float = 4.0,
) -> RefinementResult:
    """Use one verified SAM box prompt to propose a new fixed-ROI centre."""
    checkpoint, provenance = _verified_checkpoint(checkpoint_path)
    resolved_device = _resolve_device(device)
    crop, prompt, crop_box = prepare_roi_crop(
        display_image,
        roi,
        context_scale=context_scale,
    )
    predictor = _load_predictor(
        str(checkpoint),
        str(provenance["sha256"]),
        resolved_device,
    )
    try:
        predictor.set_image(crop)
        masks, scores, _ = predictor.predict(
            box=prompt,
            multimask_output=False,
        )
    except Exception as exc:
        raise SAMRefinerError("SAM ROI refinement inference failed.") from exc
    masks = np.asarray(masks)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3 or not len(masks) or len(scores) != len(masks):
        raise SAMRefinerError("SAM returned malformed masks or scores.")
    if not np.isfinite(scores).all():
        raise SAMRefinerError("SAM returned a non-finite mask score.")
    best_index = int(np.argmax(scores))
    prompt_center_x = float((prompt[0] + prompt[2]) / 2.0)
    prompt_center_y = float((prompt[1] + prompt[3]) / 2.0)
    roi_area = max(1, int(round((prompt[2] - prompt[0]) * (prompt[3] - prompt[1]))))
    center_x, center_y, mask_box, mask_area = _selected_component(
        masks[best_index],
        prompt_center_xy=(prompt_center_x, prompt_center_y),
        maximum_area=max(24, roi_area * 12),
    )
    displacement = math.hypot(
        center_x - prompt_center_x,
        center_y - prompt_center_y,
    )
    old_width = int(round(float(roi["w"])))
    old_height = int(round(float(roi["h"])))
    maximum_displacement = max(8.0, old_width * 1.5, old_height * 3.0)
    if displacement > maximum_displacement:
        raise SAMRefinerError(
            "SAM mask is implausibly far from the original ROI; proposal rejected."
        )
    crop_x0, crop_y0, _, _ = crop_box
    return RefinementResult(
        backend=BACKEND,
        checkpoint_sha256=str(provenance["sha256"]),
        device=resolved_device,
        crop_box_xyxy=crop_box,
        prompt_box_xyxy=tuple(float(value) for value in prompt),
        old_box_xywh=(
            int(round(float(roi["x"]))),
            int(round(float(roi["y"]))),
            old_width,
            old_height,
        ),
        proposed_center_xy=(
            float(crop_x0 + center_x),
            float(crop_y0 + center_y),
        ),
        mask_box_xyxy=(
            int(crop_x0 + mask_box[0]),
            int(crop_y0 + mask_box[1]),
            int(crop_x0 + mask_box[2]),
            int(crop_y0 + mask_box[3]),
        ),
        mask_score=float(scores[best_index]),
        mask_area=int(mask_area),
        displacement_pixels=float(displacement),
    )


def recenter_fixed_roi(
    roi: dict,
    result: RefinementResult,
    shared_size: dict,
    image_shape: tuple[int, int],
) -> dict:
    """Apply only the SAM-proposed centre while preserving locked ROI size."""
    image_height, image_width = (int(value) for value in image_shape)
    try:
        width = min(max(1, int(shared_size["w"])), image_width)
        height = min(max(1, int(shared_size["h"])), image_height)
    except (KeyError, TypeError, ValueError) as exc:
        raise SAMRefinerError("Shared ROI size is invalid.") from exc
    center_x, center_y = result.proposed_center_xy
    x = min(max(int(round(center_x - width / 2.0)), 0), image_width - width)
    y = min(max(int(round(center_y - height / 2.0)), 0), image_height - height)
    updated = dict(roi)
    updated.update(
        {
            "x": x,
            "y": y,
            "w": width,
            "h": height,
            "annotation_approved": False,
            "review_required": True,
            "source": "sam2_refined",
            "status": "SAM-refined proposal",
            "confidence": round(float(result.mask_score), 4),
        }
    )
    return updated


def refinement_history_entry(
    roi_id: str,
    previous_roi: dict,
    updated_roi: dict,
    result: RefinementResult,
) -> dict[str, Any]:
    """Build the compact project-file provenance record for one refinement."""
    return {
        **result.as_dict(),
        "roi_id": str(roi_id),
        "previous_source": str(previous_roi.get("source", "")),
        "previous_status": str(previous_roi.get("status", "")),
        "new_box_xywh": [
            int(updated_roi["x"]),
            int(updated_roi["y"]),
            int(updated_roi["w"]),
            int(updated_roi["h"]),
        ],
        "quantitative_mask_used": False,
        "human_review_required": True,
    }
