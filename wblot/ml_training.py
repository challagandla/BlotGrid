"""Self-contained optional training and annotation-export support.

The module contains no top-level PyTorch imports.  COCO export and metric
helpers therefore remain usable in the base BlotGrid environment.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

from .ml_detector import (
    CHECKPOINT_FORMAT_VERSION,
    DEFAULT_ARCHITECTURE,
    MAX_TRAINABLE_BACKBONE_LAYERS,
    MLDetectorError,
    SUPPORTED_ARCHITECTURES,
    TRANSFER_TRAINABLE_BACKBONE_LAYERS,
    _load_checkpoint,
    _metadata_sidecar,
    _safe_metadata,
    _sha256_file,
    build_model,
    normalize_grayscale,
)


class MLTrainingError(MLDetectorError):
    """Invalid dataset, split, training option, or resume checkpoint."""


def _read_json_object(source: str | os.PathLike[str] | Mapping[str, Any]) -> dict:
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MLTrainingError(f"Could not read JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise MLTrainingError("JSON input must contain an object at its top level.")
    return payload


def _safe_image_path(root: Path, filename: str) -> Path:
    """Resolve a COCO/project filename beneath its declared image directory."""
    root = root.expanduser().resolve()
    normalized = str(filename).replace("\\", "/")
    candidates = [root / normalized, root / Path(normalized).name]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            return resolved
    raise MLTrainingError(
        f"Image {filename!r} was not found beneath {root}."
    )


def _load_image_array(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise MLTrainingError(f"Could not decode training image: {path}") from exc
    return array


def _xywh_to_xyxy(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, box_width, box_height = (float(value) for value in bbox)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        return None
    if box_width <= 0 or box_height <= 0:
        return None
    x0 = float(np.clip(x, 0.0, width))
    y0 = float(np.clip(y, 0.0, height))
    x1 = float(np.clip(x + box_width, 0.0, width))
    y1 = float(np.clip(y + box_height, 0.0, height))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return [x0, y0, x1, y1]


def _annotation_is_pseudo(annotation: Mapping[str, Any]) -> bool:
    attributes = annotation.get("attributes")
    if not isinstance(attributes, Mapping):
        return False
    if attributes.get("pseudo_label") is True:
        return True
    return (
        "annotation_approved" in attributes
        and attributes.get("annotation_approved") is not True
    )


def _is_attested_negative(image: Mapping[str, Any]) -> bool:
    metadata = image.get("blotgrid")
    return bool(
        isinstance(metadata, Mapping)
        and metadata.get("annotation_complete") is True
        and metadata.get("negative_image") is True
    )


class CocoBandDataset:
    """Minimal COCO bounding-box dataset for category ``1 = band``."""

    def __init__(
        self,
        annotations: str | os.PathLike[str] | Mapping[str, Any],
        image_root: str | os.PathLike[str],
        *,
        image_ids: Iterable[int] | None = None,
        training: bool = False,
        include_pseudo_labels: bool = False,
        require_checksums: bool = False,
        known_checksums: Mapping[int, str] | None = None,
    ) -> None:
        self.coco = _read_json_object(annotations)
        self.image_root = Path(image_root).expanduser().resolve()
        self.training = bool(training)
        self.include_pseudo_labels = bool(include_pseudo_labels)
        images = self.coco.get("images")
        annotations_list = self.coco.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations_list, list):
            raise MLTrainingError(
                "COCO JSON must contain 'images' and 'annotations' lists."
            )

        self.images_by_id: dict[int, dict[str, Any]] = {}
        for raw in images:
            if not isinstance(raw, dict) or "id" not in raw or "file_name" not in raw:
                raise MLTrainingError(
                    "Every COCO image requires an id and file_name."
                )
            try:
                image_id = int(raw["id"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise MLTrainingError("COCO image ids must be integers.") from exc
            if image_id in self.images_by_id:
                raise MLTrainingError(f"Duplicate COCO image id: {image_id}.")
            filename = str(raw["file_name"])
            self.images_by_id[image_id] = dict(raw)

        self.annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        seen_annotation_ids: set[int] = set()
        for raw in annotations_list:
            if not isinstance(raw, dict):
                continue
            try:
                image_id = int(raw.get("image_id"))
                category_id = int(raw.get("category_id"))
            except (TypeError, ValueError, OverflowError):
                continue
            if image_id not in self.images_by_id or category_id != 1:
                continue
            if _annotation_is_pseudo(raw) and not self.include_pseudo_labels:
                continue
            if "id" in raw:
                try:
                    annotation_id = int(raw["id"])
                except (TypeError, ValueError, OverflowError):
                    annotation_id = -1
                if annotation_id in seen_annotation_ids:
                    raise MLTrainingError(
                        f"Duplicate COCO annotation id: {annotation_id}."
                    )
                seen_annotation_ids.add(annotation_id)
            self.annotations_by_image[image_id].append(dict(raw))

        selected = (
            sorted(self.images_by_id)
            if image_ids is None
            else [int(value) for value in image_ids]
        )
        if len(selected) != len(set(selected)):
            raise MLTrainingError("Dataset image ids must be unique.")
        unknown = sorted(set(selected) - set(self.images_by_id))
        if unknown:
            raise MLTrainingError(f"Unknown COCO image ids: {unknown}.")
        self.image_ids = selected
        self.image_paths: dict[int, Path] = {}
        self.image_checksums: dict[int, str] = {}
        self.valid_box_counts: dict[int, int] = {}
        self.attested_negative_ids: set[int] = set()
        known = {int(key): str(value) for key, value in (known_checksums or {}).items()}
        for image_id in self.image_ids:
            image_record = self.images_by_id[image_id]
            filename = str(image_record["file_name"])
            image_path = _safe_image_path(self.image_root, filename)
            self.image_paths[image_id] = image_path
            actual_width, actual_height = _image_dimensions(image_path)
            try:
                declared_width = int(image_record.get("width", actual_width))
                declared_height = int(image_record.get("height", actual_height))
            except (TypeError, ValueError, OverflowError) as exc:
                raise MLTrainingError(
                    f"Invalid COCO dimensions for image {filename!r}."
                ) from exc
            if (declared_width, declared_height) != (actual_width, actual_height):
                raise MLTrainingError(
                    f"COCO dimensions for {filename!r} do not match the image file."
                )

            metadata = image_record.get("blotgrid")
            expected_sha256 = None
            if isinstance(metadata, Mapping):
                if metadata.get("working_pixels_exported") is True:
                    expected_sha256 = metadata.get("exported_working_sha256")
                if not expected_sha256:
                    expected_sha256 = metadata.get("original_sha256")
            if expected_sha256 is None and require_checksums:
                raise MLTrainingError(
                    f"COCO image {filename!r} has no verifiable SHA-256 metadata. "
                    "Use an explicit allow-unverified override only for trusted data."
                )
            actual_sha256 = known.get(image_id) or _sha256_file(image_path)
            if expected_sha256 is not None:
                expected = str(expected_sha256).strip().lower()
                if (
                    len(expected) != 64
                    or any(character not in "0123456789abcdef" for character in expected)
                ):
                    raise MLTrainingError(
                        f"COCO image {filename!r} has an invalid expected SHA-256."
                    )
                if actual_sha256.lower() != expected:
                    raise MLTrainingError(
                        f"Image checksum mismatch for {filename!r}; training pixels "
                        "do not match the exported annotation coordinate system."
                    )
            self.image_checksums[image_id] = actual_sha256

            valid_boxes, _ = self._boxes_for_image(
                image_id, width=actual_width, height=actual_height
            )
            self.valid_box_counts[image_id] = len(valid_boxes)
            if _is_attested_negative(image_record):
                self.attested_negative_ids.add(image_id)
            elif not len(valid_boxes):
                raise MLTrainingError(
                    f"Image {filename!r} has no usable category-1 boxes after "
                    "clipping and is not an attested complete negative."
                )

    def __len__(self) -> int:
        return len(self.image_ids)

    def _boxes_for_image(
        self,
        image_id: int,
        *,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        boxes: list[list[float]] = []
        crowds: list[int] = []
        for annotation in self.annotations_by_image.get(image_id, []):
            converted = _xywh_to_xyxy(
                annotation.get("bbox"), width=width, height=height
            )
            if converted is None:
                continue
            boxes.append(converted)
            crowds.append(int(bool(annotation.get("iscrowd", 0))))
        return (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(crowds, dtype=np.int64),
        )

    @staticmethod
    def _augment(
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply box-safe polarity, contrast, and spatial augmentations."""
        image = np.asarray(image, dtype=np.float32)
        boxes = np.asarray(boxes, dtype=np.float32).copy()
        height, width = image.shape
        if random.random() < 0.5:
            image = 1.0 - image
        contrast = random.uniform(0.72, 1.30)
        midpoint = float(np.median(image))
        image = np.clip((image - midpoint) * contrast + midpoint, 0.0, 1.0)
        gamma = random.uniform(0.82, 1.18)
        image = np.power(image, gamma, dtype=np.float32)
        if random.random() < 0.5:
            image = image[:, ::-1].copy()
            if len(boxes):
                old_x0 = boxes[:, 0].copy()
                old_x1 = boxes[:, 2].copy()
                boxes[:, 0] = width - old_x1
                boxes[:, 2] = width - old_x0
        if random.random() < 0.25:
            image = image[::-1, :].copy()
            if len(boxes):
                old_y0 = boxes[:, 1].copy()
                old_y1 = boxes[:, 3].copy()
                boxes[:, 1] = height - old_y1
                boxes[:, 3] = height - old_y0
        return np.ascontiguousarray(image), boxes

    def __getitem__(self, index: int):
        # Importing through the shared runtime guard keeps this module optional.
        from .ml_detector import _require_ml_runtime

        torch, _ = _require_ml_runtime()
        image_id = self.image_ids[index]
        raw = _load_image_array(self.image_paths[image_id])
        image = normalize_grayscale(raw)
        height, width = image.shape
        boxes, crowds = self._boxes_for_image(
            image_id, width=width, height=height
        )
        if self.training:
            image, boxes = self._augment(image, boxes)
        rgb = np.ascontiguousarray(np.repeat(image[None, :, :], 3, axis=0))
        image_tensor = torch.from_numpy(rgb)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        if len(boxes):
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )
        else:
            area = torch.zeros((0,), dtype=torch.float32)
        target = {
            "boxes": boxes_tensor,
            "labels": labels,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.as_tensor(crowds, dtype=torch.int64),
        }
        return image_tensor, target


def _collate_detection_batch(batch):
    return tuple(zip(*batch))


def _resolve_split_value(
    value: Any,
    images_by_id: Mapping[int, Mapping[str, Any]],
) -> int:
    filenames = {
        str(image["file_name"]): image_id
        for image_id, image in images_by_id.items()
    }
    basename_matches: dict[str, list[int]] = defaultdict(list)
    for image_id, image in images_by_id.items():
        filename = str(image["file_name"]).replace("\\", "/")
        basename_matches[Path(filename).name].append(image_id)
    if isinstance(value, str):
        if value in filenames:
            return filenames[value]
        normalized = value.replace("\\", "/")
        if normalized in filenames:
            return filenames[normalized]
        matches = basename_matches.get(Path(normalized).name, [])
        if len(matches) == 1:
            return matches[0]
    try:
        image_id = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MLTrainingError(f"Unknown split item: {value!r}.") from exc
    if image_id not in images_by_id:
        raise MLTrainingError(f"Unknown split image id: {image_id}.")
    return image_id


def image_leakage_groups(
    images_by_id: Mapping[int, Mapping[str, Any]],
) -> list[list[int]]:
    """Group images connected by a study id or original-image SHA-256."""
    image_ids = sorted(int(value) for value in images_by_id)
    parent = {image_id: image_id for image_id in image_ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    token_owner: dict[tuple[str, str], int] = {}
    for image_id in image_ids:
        metadata = images_by_id[image_id].get("blotgrid")
        if not isinstance(metadata, Mapping):
            continue
        tokens: list[tuple[str, str]] = []
        study_group = str(metadata.get("study_group") or "").strip()
        original_sha256 = str(metadata.get("original_sha256") or "").strip().lower()
        if study_group:
            tokens.append(("study_group", study_group))
        if original_sha256:
            tokens.append(("original_sha256", original_sha256))
        for token in tokens:
            if token in token_owner:
                union(image_id, token_owner[token])
            else:
                token_owner[token] = image_id

    groups: dict[int, list[int]] = defaultdict(list)
    for image_id in image_ids:
        groups[find(image_id)].append(image_id)
    return sorted(
        (sorted(group) for group in groups.values()),
        key=lambda group: (group[0], len(group)),
    )


def make_train_val_split(
    images_by_id: Mapping[int, Mapping[str, Any]],
    *,
    validation_fraction: float = 0.2,
    seed: int = 1337,
    split: str | os.PathLike[str] | Mapping[str, Any] | None = None,
    train_only_ids: Iterable[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Create or validate a disjoint image-level train/validation split."""
    all_ids = sorted(int(value) for value in images_by_id)
    if len(all_ids) < 2:
        raise MLTrainingError(
            "Training requires at least two images for a held-out validation split."
        )
    required_train = {int(value) for value in (train_only_ids or [])}
    unknown_train_only = sorted(required_train - set(all_ids))
    if unknown_train_only:
        raise MLTrainingError(
            f"Unknown train-only image ids: {unknown_train_only}."
        )
    leakage_groups = image_leakage_groups(images_by_id)
    if split is not None:
        payload = _read_json_object(split)
        if not isinstance(payload.get("train"), list) or not isinstance(
            payload.get("val"), list
        ):
            raise MLTrainingError(
                "Split JSON must contain 'train' and 'val' lists."
            )
        train_ids = [
            _resolve_split_value(value, images_by_id)
            for value in payload["train"]
        ]
        val_ids = [
            _resolve_split_value(value, images_by_id)
            for value in payload["val"]
        ]
        if not train_ids or not val_ids:
            raise MLTrainingError("Train and validation splits must both be non-empty.")
        if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
            raise MLTrainingError("Split JSON contains duplicate images.")
        overlap = set(train_ids) & set(val_ids)
        if overlap:
            raise MLTrainingError(
                f"Train/validation leakage: image ids appear in both splits: "
                f"{sorted(overlap)}."
            )
        for group in leakage_groups:
            group_set = set(group)
            if group_set & set(train_ids) and group_set & set(val_ids):
                raise MLTrainingError(
                    "Train/validation leakage: images sharing a study_group or "
                    f"original_sha256 were divided across splits: {group}."
                )
        pseudo_in_validation = sorted(required_train & set(val_ids))
        if pseudo_in_validation:
            raise MLTrainingError(
                "Pseudo-label images are train-only and cannot appear in validation: "
                f"{pseudo_in_validation}."
            )
        covered = set(train_ids) | set(val_ids)
        if covered != set(all_ids):
            missing = sorted(set(all_ids) - covered)
            unknown = sorted(covered - set(all_ids))
            raise MLTrainingError(
                f"Split JSON must assign every dataset image exactly once "
                f"(missing={missing}, unknown={unknown})."
            )
        return train_ids, val_ids

    missing_study_groups: list[int] = []
    for image_id in all_ids:
        metadata = images_by_id[image_id].get("blotgrid")
        study_group = (
            str(metadata.get("study_group") or "").strip()
            if isinstance(metadata, Mapping)
            else ""
        )
        if not study_group:
            missing_study_groups.append(image_id)
    if missing_study_groups:
        raise MLTrainingError(
            "Automatic splitting requires a non-empty study_group for every "
            "image. Add the missing groups or provide a deliberate --split-json "
            f"(missing image ids: {missing_study_groups})."
        )

    try:
        fraction = float(validation_fraction)
    except (TypeError, ValueError) as exc:
        raise MLTrainingError("validation_fraction must be numeric.") from exc
    if not 0.0 < fraction < 1.0:
        raise MLTrainingError("validation_fraction must be between 0 and 1.")
    forced_group_indexes = {
        index
        for index, group in enumerate(leakage_groups)
        if set(group) & required_train
    }
    candidate_groups = [
        group
        for index, group in enumerate(leakage_groups)
        if index not in forced_group_indexes
    ]
    if not candidate_groups:
        raise MLTrainingError(
            "No clean leakage group remains for validation; pseudo-label groups "
            "are train-only."
        )
    random.Random(int(seed)).shuffle(candidate_groups)
    maximum_validation_groups = (
        len(candidate_groups)
        if forced_group_indexes
        else len(candidate_groups) - 1
    )
    if maximum_validation_groups < 1:
        raise MLTrainingError(
            "At least two independent study/checksum groups are required for "
            "held-out validation."
        )
    target_validation_images = max(1, int(round(len(all_ids) * fraction)))
    chosen_validation_groups: list[list[int]] = []
    validation_images = 0
    for group in candidate_groups:
        if len(chosen_validation_groups) >= maximum_validation_groups:
            break
        if chosen_validation_groups and validation_images >= target_validation_images:
            break
        chosen_validation_groups.append(group)
        validation_images += len(group)
    val_ids = sorted(
        image_id
        for group in chosen_validation_groups
        for image_id in group
    )
    train_ids = sorted(set(all_ids) - set(val_ids))
    if not train_ids or not val_ids:
        raise MLTrainingError(
            "Could not create non-empty leakage-safe train and validation splits."
        )
    return train_ids, val_ids


def split_manifest(
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    images_by_id: Mapping[int, Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "seed": int(seed),
        "train": [int(value) for value in train_ids],
        "val": [int(value) for value in val_ids],
        "train_files": [
            str(images_by_id[int(value)]["file_name"]) for value in train_ids
        ],
        "val_files": [
            str(images_by_id[int(value)]["file_name"]) for value in val_ids
        ],
        "leakage_groups": image_leakage_groups(images_by_id),
    }


def box_iou_matrix(
    first: Iterable[Iterable[float]],
    second: Iterable[Iterable[float]],
) -> np.ndarray:
    """Return an ``N x M`` IoU matrix for ``[x0, y0, x1, y1]`` boxes."""
    boxes_a = np.asarray(list(first), dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(list(second), dtype=np.float64).reshape(-1, 4)
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    size = np.clip(bottom_right - top_left, 0.0, None)
    intersection = size[..., 0] * size[..., 1]
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * np.clip(
        boxes_a[:, 3] - boxes_a[:, 1], 0.0, None
    )
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * np.clip(
        boxes_b[:, 3] - boxes_b[:, 1], 0.0, None
    )
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def detection_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute greedy one-to-one box precision, recall, F1, and matched IoU."""
    if not 0.0 <= float(score_threshold) <= 1.0:
        raise MLTrainingError("score_threshold must be between 0 and 1.")
    if not 0.0 < float(iou_threshold) <= 1.0:
        raise MLTrainingError("iou_threshold must be in (0, 1].")
    true_positive = false_positive = false_negative = 0
    matched_ious: list[float] = []
    image_count = 0
    for record in records:
        image_count += 1
        predictions = np.asarray(
            record.get("pred_boxes", []), dtype=np.float64
        ).reshape(-1, 4)
        scores = np.asarray(
            record.get("pred_scores", []), dtype=np.float64
        ).reshape(-1)
        ground_truth = np.asarray(
            record.get("gt_boxes", []), dtype=np.float64
        ).reshape(-1, 4)
        if len(scores) != len(predictions):
            raise MLTrainingError("Prediction boxes and scores have different lengths.")
        keep = np.isfinite(scores) & (scores >= float(score_threshold))
        predictions = predictions[keep]
        scores = scores[keep]
        order = np.argsort(-scores)
        predictions = predictions[order]
        ious = box_iou_matrix(predictions, ground_truth)
        used_ground_truth: set[int] = set()
        for prediction_index in range(len(predictions)):
            if not len(ground_truth):
                false_positive += 1
                continue
            available = [
                index
                for index in range(len(ground_truth))
                if index not in used_ground_truth
            ]
            if not available:
                false_positive += 1
                continue
            best = max(available, key=lambda index: ious[prediction_index, index])
            best_iou = float(ious[prediction_index, best])
            if best_iou >= float(iou_threshold):
                true_positive += 1
                used_ground_truth.add(best)
                matched_ious.append(best_iou)
            else:
                false_positive += 1
        false_negative += len(ground_truth) - len(used_ground_truth)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else (1.0 if false_negative == 0 else 0.0)
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "images": int(image_count),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": (
            float(np.mean(matched_ious)) if matched_ious else 0.0
        ),
        "score_threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
    }


def _seed_everything(seed: int) -> None:
    from .ml_detector import _require_ml_runtime

    torch, _ = _require_ml_runtime()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def _seed_worker(worker_id: int) -> None:
    del worker_id
    from .ml_detector import _require_ml_runtime

    torch, _ = _require_ml_runtime()
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _resolve_training_device(requested: str):
    from .ml_detector import _resolve_device

    return _resolve_device(requested)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    metadata: Mapping[str, Any],
    include_optimizer: bool = True,
) -> None:
    from .ml_detector import _require_ml_runtime

    torch, _ = _require_ml_runtime()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    checkpoint_metadata = dict(metadata)
    checkpoint_metadata["optimizer_state_included"] = bool(include_optimizer)
    checkpoint_payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": model.state_dict(),
        "metadata": checkpoint_metadata,
    }
    if include_optimizer:
        checkpoint_payload["optimizer_state_dict"] = optimizer.state_dict()
    try:
        torch.save(checkpoint_payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = dict(_safe_metadata(checkpoint_metadata))
    sidecar["checkpoint_sha256"] = _sha256_file(path)
    _atomic_json_write(_metadata_sidecar(path), sidecar)


def _evaluate_model(
    model: Any,
    loader: Any,
    device: Any,
    *,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    from .ml_detector import _require_ml_runtime

    torch, _ = _require_ml_runtime()
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets):
                labels = output["labels"].detach().cpu().numpy()
                band_mask = labels == 1
                records.append(
                    {
                        "pred_boxes": output["boxes"].detach().cpu().numpy()[band_mask],
                        "pred_scores": output["scores"].detach().cpu().numpy()[band_mask],
                        "gt_boxes": target["boxes"].detach().cpu().numpy(),
                    }
                )
    return detection_metrics(
        records,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )


def _move_optimizer_to_device(optimizer: Any, device: Any) -> None:
    from .ml_detector import _require_ml_runtime

    torch, _ = _require_ml_runtime()
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _checkpoint_paths(
    output: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """Return the stable best and last-epoch checkpoint paths for one run."""
    output_path = Path(output).expanduser().resolve()
    last_path = output_path.with_name(
        output_path.stem + ".last" + (output_path.suffix or ".pt")
    )
    return output_path, last_path


def _training_environment(
    torch: Any,
    torchvision: Any,
    resolved_device: Any,
) -> dict[str, Any]:
    """Capture the runtime and exact training-source hashes."""
    source_root = Path(__file__).resolve().parents[1]
    source_sha256: dict[str, str] = {}
    for relative in (
        "wblot/ml_training.py",
        "wblot/ml_detector.py",
        "scripts/train_detector.py",
        "requirements-ml.txt",
    ):
        path = source_root / relative
        if path.is_file():
            source_sha256[relative] = _sha256_file(path)
    cudnn_version = None
    if hasattr(torch.backends, "cudnn"):
        try:
            cudnn_version = torch.backends.cudnn.version()
        except (AttributeError, RuntimeError):
            cudnn_version = None
    device_name = "CPU"
    device_capability = None
    device_total_memory = None
    if getattr(resolved_device, "type", "") == "cuda":
        try:
            device_name = str(torch.cuda.get_device_name(resolved_device))
            device_capability = list(
                torch.cuda.get_device_capability(resolved_device)
            )
            device_total_memory = int(
                torch.cuda.get_device_properties(resolved_device).total_memory
            )
        except (AssertionError, RuntimeError, TypeError, ValueError):
            device_name = str(resolved_device)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "numpy_version": str(np.__version__),
        "pillow_version": str(PILLOW_VERSION),
        "cuda_build": str(torch.version.cuda) if torch.version.cuda else None,
        "cudnn_version": int(cudnn_version) if cudnn_version else None,
        "device": str(resolved_device),
        "device_name": device_name,
        "device_capability": device_capability,
        "device_total_memory": device_total_memory,
        "deterministic_algorithms": True,
        "source_sha256": source_sha256,
    }


def _validate_resume_metadata(
    metadata: Mapping[str, Any],
    training_config: Mapping[str, Any],
    training_environment: Mapping[str, Any],
) -> None:
    """Reject a continuation whose data-independent run contract drifted."""
    saved_config = metadata.get("training_config")
    saved_environment = metadata.get("training_environment")
    if not isinstance(saved_config, Mapping) or not isinstance(
        saved_environment, Mapping
    ):
        raise MLTrainingError(
            "Resume checkpoint lacks the complete training contract. Start a "
            "new named run rather than continuing an older-format checkpoint."
        )
    for key, current_value in training_config.items():
        if key == "epochs_requested":
            continue
        if saved_config.get(key) != current_value:
            raise MLTrainingError(
                f"Resume training option {key!r} changed "
                f"({saved_config.get(key)!r} != {current_value!r}). "
                "Only the requested epoch count may change."
            )
    for key in (
        "python_version",
        "torch_version",
        "torchvision_version",
        "cuda_build",
        "cudnn_version",
        "device",
        "device_name",
        "device_capability",
        "source_sha256",
    ):
        if saved_environment.get(key) != training_environment.get(key):
            raise MLTrainingError(
                f"Resume environment field {key!r} changed. Start a new named "
                "run so the checkpoint remains reproducible."
            )


def train_detector(
    annotations: str | os.PathLike[str],
    image_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    architecture: str = DEFAULT_ARCHITECTURE,
    pretrained: bool = True,
    epochs: int = 20,
    batch_size: int = 2,
    learning_rate: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    validation_fraction: float = 0.2,
    seed: int = 1337,
    split_json: str | os.PathLike[str] | None = None,
    device: str = "auto",
    num_workers: int = 0,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    resume: str | os.PathLike[str] | None = None,
    allow_pseudo_labels: bool = False,
    allow_unverified_images: bool = False,
) -> dict[str, Any]:
    """Fine-tune a one-class TorchVision detector and save the best model."""
    from .ml_detector import _require_ml_runtime

    torch, torchvision = _require_ml_runtime()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise MLTrainingError(
            f"architecture must be one of: {', '.join(SUPPORTED_ARCHITECTURES)}"
        )
    output_path, last_path = _checkpoint_paths(output)
    if output_path.exists() and resume is None:
        raise MLTrainingError(
            f"Output checkpoint already exists: {output_path}. "
            "Choose a new path or use --resume."
        )
    if resume is not None:
        resume_path = Path(resume).expanduser().resolve()
        if resume_path not in {output_path, last_path}:
            raise MLTrainingError(
                "For safe checkpoint continuity, --resume must name either the "
                f"best checkpoint ({output_path.name}) or its last-epoch "
                f"checkpoint ({last_path.name})."
            )
    if int(epochs) < 1 or int(batch_size) < 1 or int(num_workers) < 0:
        raise MLTrainingError(
            "epochs and batch_size must be positive; num_workers cannot be negative."
        )
    if float(learning_rate) <= 0 or float(weight_decay) < 0:
        raise MLTrainingError(
            "learning_rate must be positive and weight_decay cannot be negative."
        )
    _seed_everything(int(seed))
    resolved_device = _resolve_training_device(device)
    trainable_backbone_layers = (
        TRANSFER_TRAINABLE_BACKBONE_LAYERS
        if pretrained
        else MAX_TRAINABLE_BACKBONE_LAYERS[architecture]
    )
    training_config = {
        "architecture": architecture,
        "optimizer": "SGD",
        "epochs_requested": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "momentum": float(momentum),
        "weight_decay": float(weight_decay),
        "validation_fraction": float(validation_fraction),
        "seed": int(seed),
        "num_workers": int(num_workers),
        "score_threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
        "pretrained_transfer_learning_requested": bool(pretrained),
        "trainable_backbone_layers": int(trainable_backbone_layers),
        "backbone_normalization": (
            "FrozenBatchNorm2d" if pretrained else "BatchNorm2d"
        ),
        "allow_pseudo_labels": bool(allow_pseudo_labels),
        "allow_unverified_images": bool(allow_unverified_images),
    }
    training_environment = _training_environment(
        torch, torchvision, resolved_device
    )

    coco_payload = _read_json_object(annotations)
    annotation_path = Path(annotations).expanduser().resolve()
    annotation_sha256 = _sha256_file(annotation_path)
    raw_annotations = coco_payload.get("annotations", [])
    pseudo_annotations = [
        annotation
        for annotation in raw_annotations
        if isinstance(annotation, Mapping) and _annotation_is_pseudo(annotation)
    ]
    if pseudo_annotations and not allow_pseudo_labels:
        raise MLTrainingError(
            f"COCO contains {len(pseudo_annotations)} pseudo-label annotation(s). "
            "Training refuses them by default; use the explicit pseudo-label "
            "override only for training, never validation."
        )
    pseudo_image_ids: set[int] = set()
    for annotation in pseudo_annotations:
        try:
            pseudo_image_ids.add(int(annotation["image_id"]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise MLTrainingError(
                "Pseudo-label annotation has an invalid image_id."
            ) from exc
    full_dataset = CocoBandDataset(
        coco_payload,
        image_root,
        training=False,
        include_pseudo_labels=bool(allow_pseudo_labels),
        require_checksums=not bool(allow_unverified_images),
    )
    if not full_dataset.image_ids:
        raise MLTrainingError("The COCO dataset contains no images.")
    annotation_count = sum(
        len(full_dataset.annotations_by_image.get(image_id, []))
        for image_id in full_dataset.image_ids
    )
    if annotation_count == 0:
        raise MLTrainingError("The COCO dataset contains no category-1 band boxes.")

    resume_state = resume_metadata = resume_payload = None
    if resume is not None:
        resume_state, resume_metadata, resume_payload = _load_checkpoint(
            resume, device=torch.device("cpu")
        )
        checkpoint_architecture = str(
            resume_metadata.get("architecture", DEFAULT_ARCHITECTURE)
        )
        if checkpoint_architecture != architecture:
            raise MLTrainingError(
                f"Resume checkpoint uses {checkpoint_architecture}, not {architecture}."
            )
        previous_annotation_sha256 = str(
            resume_metadata.get("annotation_sha256", "")
        ).lower()
        if previous_annotation_sha256 != annotation_sha256.lower():
            raise MLTrainingError(
                "Resume checkpoint annotation SHA-256 differs from the current "
                "COCO dataset; start a new training run for changed annotations."
            )
        _validate_resume_metadata(
            resume_metadata,
            training_config,
            training_environment,
        )

    split_source: str | os.PathLike[str] | Mapping[str, Any] | None = split_json
    if split_source is None and resume_metadata:
        saved_train = resume_metadata.get("train_image_ids")
        saved_val = resume_metadata.get("val_image_ids")
        if isinstance(saved_train, list) and isinstance(saved_val, list):
            split_source = {"train": saved_train, "val": saved_val}
    train_ids, val_ids = make_train_val_split(
        full_dataset.images_by_id,
        validation_fraction=validation_fraction,
        seed=seed,
        split=split_source,
        train_only_ids=pseudo_image_ids,
    )
    train_dataset = CocoBandDataset(
        full_dataset.coco,
        image_root,
        image_ids=train_ids,
        training=True,
        include_pseudo_labels=bool(allow_pseudo_labels),
        require_checksums=not bool(allow_unverified_images),
        known_checksums=full_dataset.image_checksums,
    )
    validation_dataset = CocoBandDataset(
        full_dataset.coco,
        image_root,
        image_ids=val_ids,
        training=False,
        include_pseudo_labels=False,
        require_checksums=not bool(allow_unverified_images),
        known_checksums=full_dataset.image_checksums,
    )
    if set(val_ids) & pseudo_image_ids:
        raise MLTrainingError("Pseudo-label images cannot be used for validation.")
    train_positive_boxes = sum(train_dataset.valid_box_counts.values())
    validation_positive_boxes = sum(validation_dataset.valid_box_counts.values())
    if train_positive_boxes < 1:
        raise MLTrainingError("The training split contains no usable positive boxes.")
    if validation_positive_boxes < 1:
        raise MLTrainingError(
            "The validation split contains no usable approved positive boxes."
        )

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader_options = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "collate_fn": _collate_detection_batch,
        "worker_init_fn": _seed_worker if int(num_workers) else None,
        "generator": generator,
        "pin_memory": resolved_device.type == "cuda",
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **loader_options
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )

    model = build_model(
        architecture,
        pretrained=bool(pretrained and resume is None),
        trainable_backbone_layers=trainable_backbone_layers,
        freeze_backbone_batch_norm=bool(pretrained),
    )
    model.to(resolved_device)
    if resume_state is not None:
        try:
            model.load_state_dict(resume_state, strict=True)
        except Exception as exc:
            raise MLTrainingError(
                "Resume checkpoint weights do not match the selected architecture."
            ) from exc
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameter_count = sum(parameter.numel() for parameter in parameters)
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(learning_rate),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    if resume_payload and isinstance(resume_payload.get("optimizer_state_dict"), dict):
        try:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            _move_optimizer_to_device(optimizer, resolved_device)
        except (ValueError, RuntimeError, KeyError) as exc:
            raise MLTrainingError(
                "Resume optimizer state is incompatible with this training setup."
            ) from exc
    transfer_learning_used = (
        bool(resume_metadata.get("pretrained_transfer_learning", pretrained))
        if resume_metadata
        else bool(pretrained)
    )

    start_epoch = (
        int(resume_metadata.get("epoch", -1)) + 1 if resume_metadata else 0
    )
    previous_best = (
        resume_metadata.get("best_metrics", {}) if resume_metadata else {}
    )
    best_f1 = float(previous_best.get("f1", -1.0))
    best_iou = float(previous_best.get("mean_matched_iou", -1.0))
    history = list(resume_metadata.get("history", [])) if resume_metadata else []
    if start_epoch >= int(epochs):
        raise MLTrainingError(
            f"Resume checkpoint is already at epoch {start_epoch}; "
            f"--epochs must be greater than {start_epoch}."
        )

    image_manifest: list[dict[str, Any]] = []
    for image_id in full_dataset.image_ids:
        image_record = full_dataset.images_by_id[image_id]
        blotgrid_metadata = image_record.get("blotgrid")
        if not isinstance(blotgrid_metadata, Mapping):
            blotgrid_metadata = {}
        image_manifest.append(
            {
                "image_id": int(image_id),
                "file_name": str(image_record["file_name"]),
                "sha256": full_dataset.image_checksums[image_id],
                "study_group": _safe_metadata(
                    blotgrid_metadata.get("study_group")
                ),
                "original_sha256": _safe_metadata(
                    blotgrid_metadata.get("original_sha256")
                ),
                "attested_negative": image_id
                in full_dataset.attested_negative_ids,
                "valid_box_count": int(
                    full_dataset.valid_box_counts.get(image_id, 0)
                ),
                "contains_pseudo_labels": image_id in pseudo_image_ids,
            }
        )
    manifest_bytes = json.dumps(
        image_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    image_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    split_path = output_path.with_name(output_path.name + ".split.json")
    manifest = split_manifest(
        train_ids, val_ids, full_dataset.images_by_id, seed=int(seed)
    )
    if split_json is not None:
        manifest["source_split_json"] = Path(split_json).name
    _atomic_json_write(split_path, manifest)

    best_metrics: dict[str, Any] = dict(previous_best)
    for epoch in range(start_epoch, int(epochs)):
        model.train()
        losses: list[float] = []
        for images, targets in train_loader:
            device_images = [image.to(resolved_device) for image in images]
            device_targets = [
                {
                    key: (
                        value.to(resolved_device)
                        if hasattr(value, "to")
                        else value
                    )
                    for key, value in target.items()
                }
                for target in targets
            ]
            loss_components = model(device_images, device_targets)
            loss = sum(loss_components.values())
            if not bool(torch.isfinite(loss)):
                raise MLTrainingError(
                    f"Non-finite training loss at epoch {epoch + 1}."
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = _evaluate_model(
            model,
            validation_loader,
            resolved_device,
            score_threshold=float(score_threshold),
            iou_threshold=float(iou_threshold),
        )
        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            **metrics,
        }
        history.append(epoch_record)
        # Keep sidecars compact for long runs without losing recent context.
        history = history[-500:]
        is_best = (
            metrics["f1"] > best_f1 + 1e-12
            or (
                abs(metrics["f1"] - best_f1) <= 1e-12
                and metrics["mean_matched_iou"] > best_iou
            )
        )
        if is_best:
            best_f1 = float(metrics["f1"])
            best_iou = float(metrics["mean_matched_iou"])
            best_metrics = dict(metrics)

        metadata = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "architecture": architecture,
            "num_classes": 2,
            "categories": {"0": "background", "1": "band"},
            "epoch": int(epoch),
            "epochs_requested": int(epochs),
            "best_metrics": best_metrics,
            "current_metrics": metrics,
            "selection_metric": "validation_f1_then_mean_matched_iou",
            "history": history,
            "training_config": training_config,
            "training_environment": training_environment,
            "pretrained_transfer_learning": transfer_learning_used,
            "trainable_backbone_layers": int(trainable_backbone_layers),
            "trainable_parameter_count": int(trainable_parameter_count),
            "seed": int(seed),
            "score_threshold": float(score_threshold),
            "iou_threshold": float(iou_threshold),
            "train_image_ids": [int(value) for value in train_ids],
            "val_image_ids": [int(value) for value in val_ids],
            "annotation_file": annotation_path.name,
            "annotation_sha256": annotation_sha256,
            "image_manifest": image_manifest,
            "image_manifest_sha256": image_manifest_sha256,
            "allow_unverified_images": bool(allow_unverified_images),
            "pseudo_label_annotations": len(pseudo_annotations),
            "allow_pseudo_labels": bool(allow_pseudo_labels),
            "split_file": split_path.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            # Retain these top-level fields for compatibility with older
            # provenance readers; the complete record is above.
            "torch_version": training_environment["torch_version"],
            "torchvision_version": training_environment["torchvision_version"],
            "device": training_environment["device"],
        }
        _save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            metadata=metadata,
        )
        if is_best:
            _save_checkpoint(
                output_path,
                model=model,
                optimizer=optimizer,
                metadata=metadata,
            )
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"loss={epoch_record['train_loss']:.4f} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} "
            f"iou={metrics['mean_matched_iou']:.3f}"
        )

    return {
        "best_checkpoint": str(output_path),
        "last_checkpoint": str(last_path),
        "split_json": str(split_path),
        "architecture": architecture,
        "device": str(resolved_device),
        "best_metrics": best_metrics,
        "epochs_completed": int(epochs - start_epoch),
        "train_images": len(train_ids),
        "validation_images": len(val_ids),
    }


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except (OSError, ValueError) as exc:
        raise MLTrainingError(f"Could not read image dimensions: {path}") from exc


def _project_image_value(container: Any, filename: str, default: Any) -> Any:
    if not isinstance(container, Mapping):
        return default
    if filename in container:
        return container[filename]
    normalized = str(filename).replace("\\", "/")
    basename = Path(normalized).name
    if normalized in container:
        return container[normalized]
    matches = [
        value
        for key, value in container.items()
        if Path(str(key).replace("\\", "/")).name == basename
    ]
    return matches[0] if len(matches) == 1 else default


def _verify_project_image_hash(
    image_path: Path,
    provenance: Any,
    *,
    filename: str,
    allow_unverified: bool,
) -> str:
    actual = _sha256_file(image_path)
    if not isinstance(provenance, Mapping) or not provenance.get("sha256"):
        if not allow_unverified:
            raise MLTrainingError(
                f"Project image {filename!r} has no saved SHA-256 provenance. "
                "Re-upload and save the project again, or use the explicit "
                "allow-unverified override for a trusted legacy project."
            )
        return actual
    expected = str(provenance["sha256"]).strip().lower()
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise MLTrainingError(
            f"Project image provenance for {filename!r} has an invalid SHA-256."
        )
    if actual.lower() != expected:
        raise MLTrainingError(
            f"Image provenance mismatch for {filename!r}: the selected source "
            "file is not the upload used to create these ROIs."
        )
    return actual


def _safe_relative_filename(filename: str) -> Path:
    normalized = str(filename).replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute() or ".." in candidate.parts:
        candidate = Path(candidate.name)
    if not candidate.name:
        raise MLTrainingError(f"Invalid project image filename: {filename!r}.")
    return candidate


def _write_working_image(
    source: Path,
    destination: Path,
    *,
    angle: float,
    force: bool,
) -> str:
    """Write the image whose coordinate system the reviewed ROIs use."""
    if destination.exists() and not force:
        raise MLTrainingError(
            f"Export image already exists: {destination}. Use --force to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if abs(angle) < 1e-9:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=destination.name + ".",
            suffix=destination.suffix + ".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return _sha256_file(destination)

    from . import alignment, image_io

    try:
        with Image.open(source) as original:
            image_format = original.format
        loaded = image_io.load_image(source, name=source.name)
        working = alignment.rotate_loaded(loaded, angle)
    except (OSError, ValueError) as exc:
        raise MLTrainingError(
            f"Could not recreate aligned working pixels for {source.name!r}."
        ) from exc
    if working.bit_depth <= 8:
        output_array = np.rint(
            np.clip(working.gray, 0.0, 255.0)
        ).astype(np.uint8)
    elif working.bit_depth <= 16:
        output_array = np.rint(
            np.clip(working.gray, 0.0, 65535.0)
        ).astype(np.uint16)
    elif image_format == "TIFF":
        output_array = np.asarray(working.gray, dtype=np.float32)
    else:
        # Pillow cannot write floating-point PNG/JPEG. Preserve the coordinate
        # system and monotonic contrast in a 16-bit representation.
        maximum = max(float(working.max_value), 1.0)
        output_array = np.rint(
            np.clip(working.gray / maximum, 0.0, 1.0) * 65535.0
        ).astype(np.uint16)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=destination.suffix + ".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(output_array).save(temporary, format=image_format)
        os.replace(temporary, destination)
    except (OSError, ValueError, KeyError) as exc:
        raise MLTrainingError(
            f"Could not write aligned training image: {destination}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(destination)


def export_project_to_coco(
    project: str | os.PathLike[str] | Mapping[str, Any],
    image_root: str | os.PathLike[str],
    output: str | os.PathLike[str] | None = None,
    *,
    include_inactive: bool = False,
    include_fallback: bool = False,
    include_unreviewed: bool = False,
    export_image_root: str | os.PathLike[str] | None = None,
    allow_unverified: bool = False,
    allow_incomplete_annotations: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Convert reviewed BlotGrid project ROIs to category-1 COCO boxes."""
    payload = _read_json_object(project)
    layouts = payload.get("layouts")
    if not isinstance(layouts, dict):
        raise MLTrainingError("BlotGrid project JSON has no 'layouts' object.")
    root = Path(image_root).expanduser().resolve()
    output_path = Path(output).expanduser().resolve() if output is not None else None
    if output_path is not None and output_path.exists() and not force:
        raise MLTrainingError(
            f"Output already exists: {output_path}. Use --force to replace it."
        )
    alignments = payload.get("alignment_applied", {})
    image_provenance = payload.get("image_provenance", {})
    study_groups = payload.get("study_group", {})
    alignment_angles: dict[str, float] = {}
    for filename in layouts:
        raw_angle = _project_image_value(alignments, str(filename), 0.0)
        try:
            angle = float(raw_angle)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MLTrainingError(
                f"Invalid saved alignment angle for {filename!r}."
            ) from exc
        if not math.isfinite(angle) or abs(angle) > 15.0:
            raise MLTrainingError(
                f"Saved alignment angle for {filename!r} is outside ±15 degrees."
            )
        alignment_angles[str(filename)] = angle
    has_alignment = any(abs(value) >= 1e-9 for value in alignment_angles.values())
    if export_image_root is not None:
        exported_root = Path(export_image_root).expanduser().resolve()
    elif has_alignment:
        if output_path is None:
            raise MLTrainingError(
                "Aligned project ROIs require output or export_image_root so the "
                "registered working images can be written."
            )
        exported_root = output_path.with_name(output_path.stem + "_images")
    else:
        exported_root = None
    if exported_root is not None:
        if exported_root == root:
            raise MLTrainingError(
                "The working-image export directory must differ from the original "
                "image directory; source uploads are never overwritten."
            )
        destinations: dict[Path, str] = {}
        for raw_filename in layouts:
            filename = str(raw_filename)
            destination = (exported_root / _safe_relative_filename(filename)).resolve()
            source = _safe_image_path(root, filename)
            if destination == source:
                raise MLTrainingError(
                    f"Working-image export would overwrite source image {filename!r}."
                )
            if destination in destinations:
                raise MLTrainingError(
                    f"Project filenames {destinations[destination]!r} and "
                    f"{filename!r} map to the same export path."
                )
            if destination.exists() and not force:
                raise MLTrainingError(
                    f"Export image already exists: {destination}. "
                    "Use --force to replace it."
                )
            destinations[destination] = filename
    source_name = (
        Path(project).name
        if isinstance(project, (str, os.PathLike))
        else "in-memory-project.json"
    )
    coco: dict[str, Any] = {
        "info": {
            "description": "Reviewed BlotGrid Western-blot band boxes",
            "version": "1.0",
            "date_created": datetime.now(timezone.utc).isoformat(),
            "source_project": source_name,
            "source_project_saved_at": payload.get("_saved_at"),
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "band", "supercategory": "western_blot"}
        ],
        "images": [],
        "annotations": [],
        "blotgrid": {
            "targets": _safe_metadata(payload.get("targets", [])),
            "lanes": _safe_metadata(payload.get("lanes", [])),
            "control": _safe_metadata(payload.get("control", "")),
            "filters": {
                "include_inactive": bool(include_inactive),
                "include_fallback": bool(include_fallback),
                "include_unreviewed": bool(include_unreviewed),
                "allow_unverified": bool(allow_unverified),
                "allow_incomplete_annotations": bool(
                    allow_incomplete_annotations
                ),
            },
            "annotation_policy": (
                "approved_and_pseudo_labels"
                if include_unreviewed
                else "human_approved_only"
            ),
            "pseudo_labels_included": bool(include_unreviewed),
            "image_root": (
                exported_root.name if exported_root is not None else root.name
            ),
            "source_metadata": {
                "polarity_mode": _safe_metadata(payload.get("polarity_mode")),
                "detection_sensitivity": _safe_metadata(
                    payload.get("detection_sensitivity")
                ),
                "measurement_mode": _safe_metadata(
                    payload.get("measurement_mode")
                ),
            },
            "warnings": [],
        },
    }
    annotation_id = 1
    skipped_inactive = skipped_fallback = skipped_unreviewed = 0
    skipped_invalid = review_count = pseudo_label_count = 0
    dropped_unattested_empty = attested_negative_count = 0
    incomplete_positive_count = 0
    for image_id, filename in enumerate(sorted(layouts, key=str), start=1):
        layout = layouts[filename]
        if not isinstance(layout, dict):
            coco["blotgrid"]["warnings"].append(
                f"Layout for {filename!r} was not an object and was skipped."
            )
            continue
        image_path = _safe_image_path(root, str(filename))
        provenance = _project_image_value(
            image_provenance, str(filename), {}
        )
        original_sha256 = _verify_project_image_hash(
            image_path,
            provenance,
            filename=str(filename),
            allow_unverified=bool(allow_unverified),
        )
        width, height = _image_dimensions(image_path)
        preview_rois = layout.get("rois", [])
        preview_has_positive = False
        if isinstance(preview_rois, list):
            for preview_roi in preview_rois:
                if not isinstance(preview_roi, dict):
                    continue
                if not bool(preview_roi.get("use", True)) and not include_inactive:
                    continue
                if (
                    str(preview_roi.get("source", "unknown")) == "fallback"
                    and not include_fallback
                ):
                    continue
                if (
                    preview_roi.get("annotation_approved") is not True
                    and not include_unreviewed
                ):
                    continue
                preview_box = _xywh_to_xyxy(
                    [
                        preview_roi.get("x"),
                        preview_roi.get("y"),
                        preview_roi.get("w"),
                        preview_roi.get("h"),
                    ],
                    width=width,
                    height=height,
                )
                if preview_box is not None:
                    preview_has_positive = True
                    break
        if (
            preview_has_positive
            and layout.get("annotation_complete") is not True
            and not allow_incomplete_annotations
        ):
            raise MLTrainingError(
                f"Layout for {filename!r} has positive band annotations but "
                "whole-image annotation_complete is not true. Complete and "
                "attest the image review before export, or use the explicit "
                "allow-incomplete override for trusted provisional data."
            )
        if (
            preview_has_positive
            and layout.get("annotation_complete") is True
            and layout.get("negative_image") is True
        ):
            raise MLTrainingError(
                f"Layout for {filename!r} is marked as a complete negative but "
                "also exports band annotations."
            )
        angle = alignment_angles[str(filename)]
        exported_sha256 = original_sha256
        if exported_root is not None:
            relative_name = _safe_relative_filename(str(filename))
            exported_path = exported_root / relative_name
            exported_sha256 = _write_working_image(
                image_path,
                exported_path,
                angle=angle,
                force=force,
            )
        image_record = {
            "id": image_id,
            "file_name": str(filename),
            "width": width,
            "height": height,
            "blotgrid": {
                "mode": _safe_metadata(layout.get("mode")),
                "bands_dark": _safe_metadata(layout.get("bands_dark")),
                "polarity_confidence": _safe_metadata(
                    layout.get("polarity_confidence")
                ),
                "detection_warnings": _safe_metadata(
                    layout.get("detection_warnings", [])
                ),
                "replicate": _safe_metadata(
                    payload.get("replicate", {}).get(filename)
                    if isinstance(payload.get("replicate"), dict)
                    else None
                ),
                "study_group": _safe_metadata(
                    _project_image_value(study_groups, str(filename), None)
                ),
                "original_sha256": original_sha256,
                "provenance_verified": bool(
                    isinstance(provenance, Mapping)
                    and provenance.get("sha256")
                ),
                "alignment_applied_degrees": angle,
                "exported_working_sha256": exported_sha256,
                "working_pixels_exported": exported_root is not None,
                "annotation_complete": layout.get("annotation_complete") is True,
                "negative_image": layout.get("negative_image") is True,
            },
        }
        coco["images"].append(image_record)
        annotation_start = len(coco["annotations"])
        rois = layout.get("rois", [])
        if not isinstance(rois, list):
            coco["blotgrid"]["warnings"].append(
                f"Layout for {filename!r} has no reviewed ROI list."
            )
            coco["images"].pop()
            dropped_unattested_empty += 1
            continue
        if not rois:
            coco["blotgrid"]["warnings"].append(
                f"Layout for {filename!r} contains no ROIs."
            )
        for roi in rois:
            if not isinstance(roi, dict):
                skipped_invalid += 1
                continue
            active = bool(roi.get("use", True))
            source = str(roi.get("source", "unknown"))
            annotation_approved = roi.get("annotation_approved") is True
            if not active and not include_inactive:
                skipped_inactive += 1
                continue
            if source == "fallback" and not include_fallback:
                skipped_fallback += 1
                continue
            if not annotation_approved and not include_unreviewed:
                skipped_unreviewed += 1
                continue
            converted = _xywh_to_xyxy(
                [
                    roi.get("x"),
                    roi.get("y"),
                    roi.get("w"),
                    roi.get("h"),
                ],
                width=width,
                height=height,
            )
            if converted is None:
                skipped_invalid += 1
                continue
            x0, y0, x1, y1 = converted
            box_width, box_height = x1 - x0, y1 - y0
            review_required = bool(roi.get("review_required", False))
            review_count += int(review_required)
            is_pseudo_label = not annotation_approved
            pseudo_label_count += int(is_pseudo_label)
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x0, y0, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "segmentation": [],
                    "attributes": {
                        "roi_id": _safe_metadata(roi.get("roi_id")),
                        "target": _safe_metadata(roi.get("target")),
                        "lane": _safe_metadata(roi.get("lane")),
                        "source": source,
                        "status": _safe_metadata(roi.get("status")),
                        "confidence": _safe_metadata(roi.get("confidence")),
                        "review_required": review_required,
                        "active": active,
                        "annotation_approved": annotation_approved,
                        "pseudo_label": is_pseudo_label,
                    },
                }
            )
            annotation_id += 1
        retained_count = len(coco["annotations"]) - annotation_start
        is_attested_negative = bool(
            layout.get("annotation_complete") is True
            and layout.get("negative_image") is True
        )
        if retained_count and is_attested_negative:
            raise MLTrainingError(
                f"Layout for {filename!r} is marked as a complete negative but "
                "also exports band annotations."
            )
        if (
            retained_count
            and layout.get("annotation_complete") is not True
        ):
            if not allow_incomplete_annotations:
                raise MLTrainingError(
                    f"Layout for {filename!r} has positive band annotations but "
                    "whole-image annotation_complete is not true. Complete and "
                    "attest the image review before export, or use the explicit "
                    "allow-incomplete override for trusted provisional data."
                )
            incomplete_positive_count += 1
            coco["blotgrid"]["warnings"].append(
                f"Image {filename!r} contains positive annotations without a "
                "complete whole-image review attestation."
            )
        if retained_count == 0:
            if is_attested_negative:
                attested_negative_count += 1
            else:
                coco["images"].pop()
                dropped_unattested_empty += 1
                coco["blotgrid"]["warnings"].append(
                    f"Image {filename!r} was dropped because no retained band "
                    "annotations remain and it lacks a complete-negative attestation."
                )

    coco["blotgrid"]["export_summary"] = {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "skipped_inactive": skipped_inactive,
        "skipped_fallback": skipped_fallback,
        "skipped_unreviewed": skipped_unreviewed,
        "skipped_invalid": skipped_invalid,
        "included_review_required": review_count,
        "included_pseudo_labels": pseudo_label_count,
        "attested_negative_images": attested_negative_count,
        "dropped_unattested_empty_images": dropped_unattested_empty,
        "incomplete_positive_images": incomplete_positive_count,
    }
    if review_count:
        coco["blotgrid"]["warnings"].append(
            f"{review_count} included ROI(s) still carry review_required=true; "
            "confirm these boxes before training."
        )

    if pseudo_label_count:
        coco["blotgrid"]["warnings"].append(
            f"{pseudo_label_count} unapproved ROI(s) were included as pseudo-labels. "
            "Do not treat pseudo-label validation metrics as independent evidence."
        )

    if output_path is not None:
        _atomic_json_write(output_path, coco)
    return coco
