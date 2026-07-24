"""Tests for the pinned public-data download and conversion pipeline."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import prepare_public_training_data as public_data


class _Response(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _source(name: str, payload: bytes) -> public_data.SourceFile:
    return public_data.SourceFile(
        name=name,
        url=f"https://data.example/{name}",
        size=len(payload),
        algorithm="md5",
        digest=hashlib.md5(payload).hexdigest(),  # noqa: S324 - upstream pin format
        collection="test",
        extract=name.endswith(".zip"),
    )


def test_official_source_manifest_is_complete_and_immutable():
    assert len(public_data.SOURCES) == 9
    by_name = {source.name: source for source in public_data.SOURCES}
    assert by_name["matthew_gels.zip"].size == 756_367_174
    assert by_name["matthew_gels.zip"].digest == (
        "bfe792d8afd10bbeb2f6024f45e512ef"
    )
    assert by_name["readme.txt"].size == 2_729
    rgp = by_name["rgp_caps_electrophoresis_image.zip"]
    assert rgp.size == 30_105_023
    assert rgp.algorithm == "sha256"
    assert rgp.digest == (
        "b0a8ee39d8db7eec745ac5aaec970e78eef65b04a7a0fd964960eb639b511d46"
    )


def test_download_is_bounded_verified_atomic_and_idempotent(tmp_path, monkeypatch):
    payload = b"immutable training data"
    source = _source("tiny.zip", payload)
    monkeypatch.setattr(
        public_data.urllib.request,
        "urlopen",
        lambda request, timeout: _Response(payload),
    )
    destination = tmp_path / "tiny.zip"
    assert public_data.download_source(source, destination) == "downloaded"
    assert destination.read_bytes() == payload
    assert not (tmp_path / "tiny.zip.part").exists()
    assert public_data.download_source(source, destination) == "already_verified"


def test_download_removes_truncated_partial(tmp_path, monkeypatch):
    expected = b"complete"
    source = _source("tiny.zip", expected)
    monkeypatch.setattr(
        public_data.urllib.request,
        "urlopen",
        lambda request, timeout: _Response(expected[:-1]),
    )
    destination = tmp_path / "tiny.zip"
    with pytest.raises(public_data.PreparationError, match="server size changed"):
        public_data.download_source(source, destination)
    assert not destination.exists()
    assert not (tmp_path / "tiny.zip.part").exists()


@pytest.mark.parametrize("unsafe_name", ["../escape.txt", "/absolute.txt"])
def test_safe_extract_rejects_path_traversal(tmp_path, unsafe_name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(unsafe_name, b"no")
    with pytest.raises(public_data.PreparationError, match="Unsafe ZIP"):
        public_data.safe_extract(
            archive,
            tmp_path / "raw",
            archive_sha256=public_data._digest(archive),
        )
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_rejects_symbolic_links(tmp_path):
    archive = tmp_path / "link.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(info, "target")
    with pytest.raises(public_data.PreparationError, match="symbolic link"):
        public_data.safe_extract(
            archive,
            tmp_path / "raw",
            archive_sha256=public_data._digest(archive),
        )


def test_safe_extract_detects_corruption_and_force_repairs(tmp_path):
    archive = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("dataset/image.bin", b"verified bytes")
    archive_sha256 = public_data._digest(archive)
    destination = tmp_path / "raw"
    assert (
        public_data.safe_extract(
            archive,
            destination,
            archive_sha256=archive_sha256,
        )
        == "extracted"
    )
    (destination / "dataset/image.bin").write_bytes(b"corruptd bytes")
    with pytest.raises(public_data.PreparationError, match="missing or corrupt"):
        public_data.safe_extract(
            archive,
            destination,
            archive_sha256=archive_sha256,
        )
    assert (
        public_data.safe_extract(
            archive,
            destination,
            archive_sha256=archive_sha256,
            force=True,
        )
        == "extracted"
    )
    assert (destination / "dataset/image.bin").read_bytes() == b"verified bytes"


def test_mask_components_use_eight_connectivity(tmp_path):
    mask = np.zeros((8, 9), dtype=np.uint8)
    mask[2, 3] = 1
    mask[3, 4] = 1
    path = tmp_path / "mask.tif"
    Image.fromarray(mask).save(path)
    boxes, size = public_data._component_boxes(path)
    assert size == (9, 8)
    assert boxes == [[3.0, 2.0, 2.0, 2.0]]


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")


def _toy_archive(tmp_path: Path) -> Path:
    tree = tmp_path / "tree"
    base_image = np.arange(12 * 16, dtype=np.uint8).reshape(12, 16)
    other_image = np.flipud(base_image).copy()
    val_image = np.fliplr(base_image).copy()
    test_image = np.full((12, 16), 80, dtype=np.uint8)
    mask_a = np.zeros((12, 16), dtype=np.uint8)
    mask_a[2:4, 3:9] = 1
    mask_b = np.zeros((12, 16), dtype=np.uint8)
    mask_b[6:8, 4:13] = 1
    mask_val = np.zeros((12, 16), dtype=np.uint8)
    mask_val[4:7, 2:10] = 1

    # An identical image/mask pair within train is retained only once.
    for name in ("duplicate_a", "duplicate_b"):
        _write_png(tree / "dataset/images" / f"{name}.png", base_image)
        _write_png(tree / "dataset/masks" / f"{name}.png", mask_a)

    # This byte-identical image spans train/val and has conflicting labels:
    # the entire SHA group must be quarantined.
    _write_png(tree / "dataset/images/leak.png", other_image)
    _write_png(tree / "dataset/masks/leak.png", mask_a)
    _write_png(tree / "dataset/val_images/leak.png", other_image)
    _write_png(tree / "dataset/val_masks/leak.png", mask_b)

    _write_png(tree / "dataset/images/good_train.png", test_image)
    _write_png(tree / "dataset/masks/good_train.png", mask_b)
    _write_png(tree / "dataset/val_images/good_val.png", val_image)
    _write_png(tree / "dataset/val_masks/good_val.png", mask_val)
    _write_png(tree / "dataset/test_images/test_only.png", test_image + 1)
    _write_png(tree / "dataset/test_masks/test_only.png", mask_a)

    archive = tmp_path / "nathan_gels.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(tree))
    return archive


def test_prepare_quarantines_leakage_and_writes_verified_coco(tmp_path):
    archive = _toy_archive(tmp_path)
    payload = archive.read_bytes()
    source = _source("nathan_gels.zip", payload)
    output = tmp_path / "output"
    downloads = output / "downloads"
    downloads.mkdir(parents=True)
    shutil.copyfile(archive, downloads / source.name)

    manifest = public_data.prepare(output, sources=(source,))
    assert manifest["counts"] == {
        "usable_images": 3,
        "train_images": 2,
        "validation_images": 1,
        "band_boxes": 3,
        "heldout_records": 4,
        "deduplicated_records": 1,
        "quarantined_image_groups": 1,
        "quarantined_records": 2,
    }
    annotations = json.loads((output / "annotations.json").read_text())
    split = json.loads((output / "split.json").read_text())
    heldout = json.loads((output / "heldout.json").read_text())
    assert set(split["train"] + split["val"]) == {
        image["id"] for image in annotations["images"]
    }
    assert not set(split["train"]) & set(split["val"])
    reasons = [record["reason"] for record in heldout["records"]]
    assert reasons.count(
        "cross_partition_duplicate_or_conflicting_labels"
    ) == 2
    assert (
        reasons.count(
            "deduplicated_identical_image_and_mask_within_partition"
        )
        == 1
    )
    assert reasons.count("upstream_test_held_out") == 1
    for image in annotations["images"]:
        generated = output / "images" / image["file_name"]
        assert public_data._digest(generated) == image["blotgrid"]["original_sha256"]
        assert image["blotgrid"]["out_of_domain"] is True

    first_hashes = {
        name: public_data._digest(output / name)
        for name in (
            "annotations.json",
            "split.json",
            "heldout.json",
            "dataset_manifest.json",
        )
    }
    public_data.prepare(output, sources=(source,))
    assert first_hashes == {
        name: public_data._digest(output / name) for name in first_hashes
    }
    generated_image = next((output / "images").iterdir())
    generated_image.write_bytes(b"corrupt")
    with pytest.raises(public_data.PreparationError, match="unexpected content"):
        public_data.prepare(output, sources=(source,))
    public_data.prepare(output, sources=(source,), force=True)
    annotations = json.loads((output / "annotations.json").read_text())
    expected_by_name = {
        image["file_name"]: image["blotgrid"]["original_sha256"]
        for image in annotations["images"]
    }
    assert public_data._digest(generated_image) == expected_by_name[
        generated_image.name
    ]
    stale = output / "images/stale.png"
    stale.write_bytes(b"stale")
    with pytest.raises(public_data.PreparationError, match="stale/unexpected"):
        public_data.prepare(output, sources=(source,))
    public_data.prepare(output, sources=(source,), force=True)
    assert not stale.exists()


def test_rgp_origin_gets_share_alike_image_license(tmp_path):
    train_image = tmp_path / "rgp.jpg"
    val_image = tmp_path / "gelgenie.png"
    _write_png(train_image, np.full((8, 10), 30, dtype=np.uint8))
    _write_png(val_image, np.full((8, 10), 60, dtype=np.uint8))

    def record(
        image: Path,
        *,
        source_archive: str,
        image_source_archive: str,
        partition: str,
    ):
        return {
            "source_archive": source_archive,
            "image_source_archive": image_source_archive,
            "partition": partition,
            "image_path": image,
            "mask_path": image,
            "image": image.name,
            "mask": image.name,
            "image_sha256": public_data._digest(image),
            "mask_sha256": public_data._digest(image),
            "width": 10,
            "height": 8,
            "boxes": [[1.0, 1.0, 5.0, 2.0]],
        }

    output = tmp_path / "output"
    public_data.build_outputs(
        output,
        [
            record(
                train_image,
                source_archive="lsdb_gels.zip",
                image_source_archive="rgp_caps_electrophoresis_image.zip",
                partition="train",
            ),
            record(
                val_image,
                source_archive="nathan_gels.zip",
                image_source_archive="nathan_gels.zip",
                partition="val",
            ),
        ],
        [],
        {},
        [],
        [],
    )
    annotations = json.loads((output / "annotations.json").read_text())
    licenses = {image["blotgrid"]["image_source_archive"]: image["license"]
                for image in annotations["images"]}
    assert licenses["rgp_caps_electrophoresis_image.zip"] == 2
    assert licenses["nathan_gels.zip"] == 1
    assert annotations["info"]["license"].startswith("Mixed:")
