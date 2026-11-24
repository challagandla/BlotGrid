"""Focused tests for public Western acquisition and review-project import."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import stat
import struct
import zipfile

import numpy as np
from PIL import Image
import pytest
import tifffile

from wblot import detection, gelgenie, review_sources


class _Response(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        declared: int | None = None,
        content_type: str = "image/jpeg",
    ):
        super().__init__(payload)
        self.headers = {
            "Content-Length": str(
                len(payload) if declared is None else declared
            ),
            "Content-Type": content_type,
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_verified_download_is_atomic_bounded_and_idempotent(tmp_path):
    payload = b"pinned public source"
    spec = review_sources.DownloadSpec(
        key="mendeley:zz3ktsyjds-v2.zip",
        url=(
            "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1."
            "amazonaws.com/zz3ktsyjds-2.zip"
        ),
        destination=tmp_path / "source.bin",
        size=len(payload),
        algorithm="sha256",
        digest=hashlib.sha256(payload).hexdigest(),
    )
    first = review_sources.download_verified(
        spec,
        opener=lambda _request, timeout: _Response(payload),
    )
    assert first["status"] == "downloaded"
    assert spec.destination.read_bytes() == payload
    assert not (tmp_path / "source.bin.part").exists()
    second = review_sources.download_verified(
        spec,
        opener=lambda *_args, **_kwargs: pytest.fail("unexpected network call"),
    )
    assert second["status"] == "already_verified"
    assert second["sha256"] == hashlib.sha256(payload).hexdigest()


def test_unreceipted_hpa_jpeg_is_not_silently_adopted(tmp_path):
    path = tmp_path / "13294_medium.jpg"
    Image.fromarray(np.arange(80, dtype=np.uint8).reshape(8, 10)).save(
        path,
        format="JPEG",
    )
    spec = review_sources.DownloadSpec(
        key="hpa:13294",
        url=(
            "https://v25.proteinatlas.org/"
            "images_western_blot/13294_medium.jpg"
        ),
        destination=path,
        max_bytes=1024 * 1024,
    )
    assert review_sources._existing_download_verified(spec) is False


def test_fresh_hpa_download_must_decode_before_promotion(tmp_path):
    spec = review_sources.DownloadSpec(
        key="hpa:13294",
        url=(
            "https://v25.proteinatlas.org/"
            "images_western_blot/13294_medium.jpg"
        ),
        destination=tmp_path / "13294_medium.jpg",
        max_bytes=1024 * 1024,
    )
    with pytest.raises(
        review_sources.ReviewSourceError,
        match="single-frame JPEG",
    ):
        review_sources.download_verified(
            spec,
            attempts=1,
            opener=lambda _request, timeout: _Response(b"not a jpeg"),
        )
    assert not spec.destination.exists()
    assert not (tmp_path / "13294_medium.jpg.part").exists()


def test_unpinned_download_must_match_declared_content_length(tmp_path):
    payload = b"short response"
    spec = review_sources.DownloadSpec(
        key="S-BIAD2849:test.tif",
        url=(
            "https://www.ebi.ac.uk/biostudies/files/"
            "S-BIAD2849/test.tif"
        ),
        destination=tmp_path / "source.bin",
        max_bytes=1024,
    )
    with pytest.raises(
        review_sources.ReviewSourceError,
        match="response length changed",
    ):
        review_sources.download_verified(
            spec,
            attempts=1,
            opener=lambda _request, timeout: _Response(
                payload,
                declared=len(payload) + 10,
            ),
        )
    assert not spec.destination.exists()


def test_unreceipted_checksumless_payload_is_not_adopted_by_size(tmp_path):
    path = tmp_path / "test.tif"
    path.write_bytes(b"same-sized-untrusted")
    spec = review_sources.DownloadSpec(
        key="S-BIAD2849:test.tif",
        url=(
            "https://www.ebi.ac.uk/biostudies/files/"
            "S-BIAD2849/test.tif"
        ),
        destination=path,
        size=path.stat().st_size,
    )
    assert review_sources._existing_download_verified(spec) is False


def test_biostudies_official_fire_mirror_redirect_is_exact():
    key = "S-BIAD2306:Fig. 1B a-tubulin.zip"
    assert review_sources._validate_download_url(
        key,
        (
            "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/306/"
            "S-BIAD2306/Files/Fig.%201B%20a-tubulin.zip"
        ),
    )
    with pytest.raises(review_sources.ReviewSourceError):
        review_sources._validate_download_url(
            key,
            (
                "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/306/"
                "S-BIAD2306/Files/other.zip"
            ),
        )
    assert review_sources._validate_download_url(
        "S-BIAD2849:folder/image.tif",
        (
            "https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/"
            "849/S-BIAD2849/Files/folder/image.tif"
        ),
    )


@pytest.mark.parametrize("name", ["../escape.tif", "/tmp/escape.tif", "a/b.tif"])
def test_manifest_download_filename_must_be_a_safe_basename(name):
    with pytest.raises(review_sources.ReviewSourceError):
        review_sources._safe_basename(name, label="test filename")


@pytest.mark.parametrize(
    "name",
    ["../escape.png", "/absolute.png", r"folder\image.png", "bad\x00.png"],
)
def test_zip_member_path_rejects_traversal_backslashes_and_nul(name):
    with pytest.raises(review_sources.ReviewSourceError, match="Unsafe ZIP"):
        review_sources._safe_member_path(name)


def test_safe_zip_rejects_duplicate_and_symbolic_link(tmp_path):
    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as bundle:
        bundle.writestr("image.png", b"one")
        bundle.writestr("image.png", b"two")
    with pytest.raises(review_sources.ReviewSourceError, match="duplicate"):
        review_sources.safe_extract_zip(duplicate, tmp_path / "duplicate")

    link = tmp_path / "link.zip"
    info = zipfile.ZipInfo("link.png")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(link, "w") as bundle:
        bundle.writestr(info, "target")
    with pytest.raises(review_sources.ReviewSourceError, match="symbolic link"):
        review_sources.safe_extract_zip(link, tmp_path / "link")


def test_safe_zip_rejects_encrypted_flag_and_ratio_bomb(tmp_path):
    encrypted = tmp_path / "encrypted.zip"
    with zipfile.ZipFile(encrypted, "w") as bundle:
        bundle.writestr("image.png", b"not actually encrypted")
    payload = bytearray(encrypted.read_bytes())
    local = payload.index(b"PK\x03\x04")
    central = payload.index(b"PK\x01\x02")
    local_flags = struct.unpack_from("<H", payload, local + 6)[0] | 1
    central_flags = struct.unpack_from("<H", payload, central + 8)[0] | 1
    struct.pack_into("<H", payload, local + 6, local_flags)
    struct.pack_into("<H", payload, central + 8, central_flags)
    encrypted.write_bytes(payload)
    with pytest.raises(review_sources.ReviewSourceError, match="encrypted"):
        review_sources.safe_extract_zip(encrypted, tmp_path / "encrypted")

    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(
        bomb,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as bundle:
        bundle.writestr("zeros.bin", b"\0" * (2 * 1024 * 1024))
    with pytest.raises(
        review_sources.ReviewSourceError,
        match="compression-ratio bomb",
    ):
        review_sources.safe_extract_zip(bomb, tmp_path / "bomb")


def test_safe_zip_extracts_selected_files_and_rechecks_crc(tmp_path):
    archive = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("images/a.png", b"image")
        bundle.writestr("notes/readme.txt", b"not selected")
    destination = tmp_path / "out"
    first = review_sources.safe_extract_zip(
        archive,
        destination,
        select=lambda name: name.endswith(".png"),
    )
    assert first["status"] == "extracted"
    assert (destination / "images/a.png").read_bytes() == b"image"
    assert not (destination / "notes/readme.txt").exists()
    second = review_sources.safe_extract_zip(
        archive,
        destination,
        select=lambda name: name.endswith(".png"),
    )
    assert second["status"] == "already_verified"


def test_hpa_stream_parser_filters_scope_and_deduplicates_urls(
    tmp_path,
    monkeypatch,
):
    xml = b"""<?xml version="1.0"?>
<proteinAtlas>
  <entry version="25" url="https://v25.proteinatlas.org/ENSG1">
    <name>GENE1</name><identifier id="ENSG1" db="Ensembl"/>
    <antibody id="HPA000001">
      <westernBlot source="HPA" technology="WB">
        <verification description="supported evidence">supported</verification>
        <image imageType="assay"><imageUrl>https://v25.proteinatlas.org/images_western_blot/1_medium.jpg</imageUrl></image>
        <blotLanes><lane laneId="1" laneContent="Marker"/></blotLanes>
      </westernBlot>
      <westernBlot source="External" technology="WB">
        <image><imageUrl>https://example.org/external.jpg</imageUrl></image>
      </westernBlot>
      <westernBlot source="HPA" technology="WBsiRNA">
        <image imageType="assay"><imageUrl>https://v25.proteinatlas.org/images_western_blot/1_medium.jpg</imageUrl></image>
      </westernBlot>
      <westernBlot source="HPA" technology="IHC">
        <image><imageUrl>https://example.org/ihc.jpg</imageUrl></image>
      </westernBlot>
    </antibody>
  </entry>
</proteinAtlas>"""
    path = tmp_path / "hpa.xml.gz"
    import gzip

    with gzip.open(path, "wb") as handle:
        handle.write(xml)
    monkeypatch.setattr(
        review_sources,
        "HPA_XML_SHA256",
        review_sources.file_digest(path),
    )
    records = review_sources.stream_hpa_western_metadata(path)
    assert len(records) == 1
    assert records[0]["image_url"].endswith("/1_medium.jpg")
    assert records[0]["source_row_count"] == 2
    assert records[0]["gene_name"] == "GENE1"
    assert review_sources.stream_hpa_western_urls(path) == {
        (
            "https://v25.proteinatlas.org/"
            "images_western_blot/1_medium.jpg"
        )
    }


def test_single_series_reduced_image_pyramid_uses_full_resolution(tmp_path):
    source = tmp_path / "licor.TIF"
    full = np.linspace(-2.0, 7.0, 24 * 32, dtype=np.float32).reshape(24, 32)
    with tifffile.TiffWriter(source) as bundle:
        bundle.write(
            full,
            subifds=2,
            software="Image Studio 5.5.4",
        )
        bundle.write(full[::2, ::2], subfiletype=1)
        bundle.write(full[::4, ::4], subfiletype=1)
    candidate = {
        "path": source,
        "source_id": "bia_s_biad2306",
        "upstream_path": "figure.zip#licor.TIF",
    }
    derived, metadata, conversion = review_sources._prepare_image(
        candidate,
        tmp_path / "derived",
    )
    assert derived.suffix == ".png"
    assert metadata["width"] == 32
    assert metadata["height"] == 24
    assert metadata["frames"] == 1
    assert metadata["bit_depth"] == 16
    assert conversion is not None
    assert conversion["pyramid_levels"] == 3
    assert conversion["selected_level"] == 0
    assert conversion["source_software"] == "Image Studio 5.5.4"
    assert conversion["method"] == "finite_minmax_linear_to_uint16_v1"
    assert conversion["derived_sha256"] == review_sources.file_digest(derived)


def test_identical_rgb_tiff_is_losslessly_collapsed(tmp_path):
    source = tmp_path / "grayscale_rgb.tif"
    gray = np.arange(15 * 17, dtype=np.uint8).reshape(15, 17)
    tifffile.imwrite(source, np.repeat(gray[..., None], 3, axis=2))
    candidate = {
        "path": source,
        "source_id": "zenodo_5810203",
        "upstream_path": "grayscale_rgb.tif",
    }
    derived, metadata, conversion = review_sources._prepare_image(
        candidate,
        tmp_path / "derived",
    )
    assert metadata["width"] == 17
    assert metadata["height"] == 15
    assert conversion is not None
    assert conversion["color_collapse"] == {
        "method": "identical_rgb_channels_to_grayscale_v1",
        "source_samples": 3,
        "channels_bit_identical": True,
        "alpha_fully_opaque": False,
    }
    with Image.open(derived) as image:
        converted = np.asarray(image)
    assert np.array_equal(converted, gray.astype(np.uint16) * 257)


def test_nonidentical_rgb_tiff_is_preserved_without_grayscale_collapse(tmp_path):
    source = tmp_path / "color.tif"
    color = np.zeros((10, 12, 3), dtype=np.uint8)
    color[..., 1] = 1
    tifffile.imwrite(source, color)
    candidate = {
        "path": source,
        "source_id": "zenodo_5810203",
        "upstream_path": "color.tif",
    }
    prepared, metadata, conversion = review_sources._prepare_image(
        candidate,
        tmp_path / "derived",
    )
    assert prepared == source
    assert metadata["mode"] == "RGB"
    assert conversion is None


def test_tiff_decode_is_bounded_before_array_materialization(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "bounded.tif"
    tifffile.imwrite(source, np.zeros((20, 20), dtype=np.uint16))
    monkeypatch.setattr(review_sources, "MAX_IMAGE_PIXELS", 100)
    candidate = {
        "path": source,
        "source_id": "zenodo_5810203",
        "upstream_path": "bounded.tif",
    }
    with pytest.raises(
        review_sources.ReviewSourceError,
        match="tiff_resource_limit",
    ):
        review_sources._prepare_image(candidate, tmp_path / "derived")


def test_genuinely_ambiguous_multiseries_tiff_is_quarantined(tmp_path):
    source = tmp_path / "ambiguous.tif"
    with tifffile.TiffWriter(source) as bundle:
        bundle.write(np.zeros((12, 14), dtype=np.uint16))
        bundle.write(np.zeros((7, 9), dtype=np.uint16))
    candidate = {
        "path": source,
        "source_id": "bia_s_biad2849",
        "license_id": 2,
        "source_doi": "10.6019/S-BIAD2849",
        "source_url": "https://example.org/study",
        "required_attribution": "citation",
        "source_archive": None,
        "source_image": "ambiguous.tif",
        "upstream_url": "https://example.org/ambiguous.tif",
        "upstream_path": "ambiguous.tif",
        "study_group": "study",
    }
    images, quarantined = review_sources.canonicalize_candidates(
        tmp_path,
        [candidate],
    )
    assert images == []
    assert len(quarantined) == 1
    assert "ambiguous_tiff_series:2" in quarantined[0]["reason"]


def test_verified_review_copy_has_independent_inode(tmp_path):
    source = tmp_path / "source.png"
    destination = tmp_path / "review" / "image.png"
    Image.fromarray(np.zeros((8, 9), dtype=np.uint8)).save(source)
    digest = review_sources.file_digest(source)
    review_sources._link_or_copy_verified(source, destination, digest)
    assert review_sources.file_digest(destination) == digest
    assert source.stat().st_ino != destination.stat().st_ino
    destination.write_bytes(b"changed review copy")
    assert review_sources.file_digest(source) == digest


def test_exact_content_connects_original_study_groups():
    images = [
        {"study_group": "study-a", "sha256": "a" * 64},
        {"study_group": "study-b", "sha256": "a" * 64},
        {"study_group": "study-b", "sha256": "b" * 64},
        {"study_group": "study-c", "sha256": "c" * 64},
    ]
    review_sources._assign_leakage_components(images)
    assert images[0]["study_group"] == images[1]["study_group"]
    assert images[1]["study_group"] == images[2]["study_group"]
    assert images[3]["study_group"] == "study-c"
    assert images[0]["exact_content_count"] == 2
    assert images[2]["exact_content_count"] == 1


def test_corpus_commit_replaces_stale_generated_state(tmp_path):
    root = tmp_path / "corpus"
    staging = tmp_path / "staging"
    root.mkdir()
    staging.mkdir()
    for relative in review_sources._GENERATED_CORPUS_PATHS:
        old = root / relative
        new = staging / relative
        if Path(relative).suffix:
            old.write_text("old", encoding="utf-8")
            new.write_text("new", encoding="utf-8")
        else:
            old.mkdir()
            new.mkdir()
            (old / "stale").write_text("stale", encoding="utf-8")
            (new / "current").write_text("current", encoding="utf-8")
    review_sources._commit_staged_corpus(root, staging)
    for relative in review_sources._GENERATED_CORPUS_PATHS:
        installed = root / relative
        if Path(relative).suffix:
            assert installed.read_text(encoding="utf-8") == "new"
        else:
            assert not (installed / "stale").exists()
            assert (installed / "current").is_file()


def test_project_proposals_are_never_preapproved(tmp_path, monkeypatch):
    image_path = tmp_path / "canonical" / "images" / "image.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((20, 30), dtype=np.uint8)).save(image_path)
    image = {
        "project_filename": "image.png",
        "canonical_path": "canonical/images/image.png",
        "source_id": "zenodo_5810203",
        "license_id": 1,
        "source_doi": "10.5281/zenodo.5810203",
        "source_url": "https://zenodo.org/records/5810203",
        "required_attribution": review_sources.ZENODO_CITATION,
        "source_image": "image.png",
        "upstream_url": "https://example.org/image.png",
        "upstream_path": "image.png",
        "study_group": "zenodo_5810203:Figure_1",
        "sha256": review_sources.file_digest(image_path),
        "width": 30,
        "height": 20,
        "bit_depth": 8,
    }
    malicious_layout = review_sources._empty_layout(image, "test")
    malicious_layout["annotation_complete"] = True
    malicious_layout["rois"] = [
        {
            "roi_id": "roi_0",
            "use": True,
            "annotation_approved": True,
            "review_required": False,
            "target": "Band",
            "lane": "Unassigned",
            "x": 2,
            "y": 3,
            "w": 8,
            "h": 4,
        }
    ]
    monkeypatch.setattr(
        review_sources,
        "_gelgenie_layout",
        lambda *_args, **_kwargs: malicious_layout,
    )
    project = review_sources.build_project(
        tmp_path,
        [image],
        batch_id="batch_0001",
        gelgenie_model=tmp_path / "model.onnx",
    )
    layout = project["layouts"]["image.png"]
    assert layout["annotation_complete"] is False
    assert project["annotation_complete"]["image.png"] is False
    assert layout["fallback_count"] == 0
    assert layout["rois"][0]["annotation_approved"] is False
    assert layout["rois"][0]["review_required"] is True
    assert layout["rois"][0]["pseudo_label"] is True
    lineage = project["source_lineage"]
    assert lineage["images"]["image.png"]["source_id"] == "zenodo_5810203"
    assert lineage["licenses"][0]["spdx_identifier"] == "CC-BY-4.0"
    assert lineage["licenses"][0]["publication_assessment"]["status"] == "cleared"


def test_large_review_proposal_is_resized_and_mapped_to_native_pixels(
    tmp_path,
    monkeypatch,
):
    image_path = tmp_path / "canonical" / "images" / "large.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((200, 300), dtype=np.uint8)).save(image_path)
    image = {
        "project_filename": "large.png",
        "canonical_path": "canonical/images/large.png",
        "width": 300,
        "height": 200,
    }
    observed = {}

    def fake_predict(gray, *_args, **_kwargs):
        observed["shape"] = gray.shape
        return detection.DetectionResult(
            bands=[
                detection.DetectedBand(
                    target="Band",
                    lane="Unassigned",
                    x=10,
                    y=5,
                    w=20,
                    h=10,
                    confidence=0.9,
                )
            ],
            lane_centers=[20],
            target_centers=[10],
            bands_are_dark=True,
            polarity_confidence=0.8,
            warnings=[],
        )

    monkeypatch.setattr(
        review_sources,
        "MAX_REVIEW_PROPOSAL_PIXELS",
        15_000,
    )
    monkeypatch.setattr(gelgenie, "predict_bands", fake_predict)
    monkeypatch.setattr(
        gelgenie,
        "model_provenance",
        lambda _path: {"sha256": "a" * 64, "verified_official": True},
    )
    monkeypatch.setattr(
        gelgenie,
        "runtime_provider",
        lambda _device: "CPUExecutionProvider",
    )
    monkeypatch.setattr(
        gelgenie,
        "available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    layout = review_sources._gelgenie_layout(
        tmp_path,
        image,
        tmp_path / "model.onnx",
    )
    assert observed["shape"][0] * observed["shape"][1] <= 15_000
    proposal_input = layout["detector_provenance"]["proposal_input"]
    assert proposal_input["resized"] is True
    assert proposal_input["native_width"] == 300
    assert proposal_input["native_height"] == 200
    assert layout["rois"][0]["x"] > 10
    assert layout["rois"][0]["y"] > 5
    assert layout["rois"][0]["annotation_approved"] is False


def test_live_pinned_plan_has_exact_source_counts():
    root = Path("data/public_western")
    if not (root / "downloads" / "metadata").is_dir():
        pytest.skip("local pinned public-Western metadata is unavailable")
    plan = review_sources.plan_sources(root.resolve())
    assert plan["downloads"] == {
        "hpa_full_index_and_images": 11_968,
        "hpa_first_review_subset": 256,
        "mendeley_png": 186,
        "S-BIAD2849_raw_tiff": 42,
        "S-BIAD2306_wb_archives": 38,
        "zenodo_original_tiff": 20,
    }
