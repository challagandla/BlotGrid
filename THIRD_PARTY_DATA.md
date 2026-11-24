# Third-party public training data

BlotGrid can prepare an optional public **bootstrap** dataset from the sources
below. Downloaded archives, extracted images, generated COCO annotations, and
trained checkpoints stay under the git-ignored `data/` and `models/`
directories. They are not redistributed by this repository.

## GelGenie labelled gel images

- Dataset: *Segmented Gel Electrophoresis Images used for Training GelGenie
  Models*
- Immutable release: Zenodo v3,
  [doi:10.5281/zenodo.14641949](https://doi.org/10.5281/zenodo.14641949)
- Concept DOI:
  [doi:10.5281/zenodo.13218468](https://doi.org/10.5281/zenodo.13218468)
- License: Creative Commons Attribution 4.0
- Contents: gel-electrophoresis images with hand-labelled foreground masks
- Creators: Matthew Aquilina, Nathan Wu, Kiros Kwan, Filip Bušić, James Dodd,
  Laura Nicolás-Sáenz, Alan O'Callaghan, Peter Bankhead, and Katherine Dunn
- Data collectors: Siyuan Stella Wang, Yichen Zhao, Thomas Mayer, Huangchen
  Cui, Joana Reis, and Ricarda Törner

Suggested dataset citation:

> Aquilina, Matthew; Wu, Nathan; Kwan, Kiros; Bušić, Filip; Dodd, James;
> Nicolás-Sáenz, Laura; O'Callaghan, Alan; Bankhead, Peter; Dunn, Katherine
> (2025). *Segmented Gel Electrophoresis Images used for Training GelGenie
> Models* (v3). Zenodo. https://doi.org/10.5281/zenodo.14641949

The preparation script verifies every v3 archive against its published byte
count, MD5 checksum, and published per-archive labelled-image count before
accepting the corpus. It converts connected foreground
components to bounding boxes but retains the upstream image/mask identities,
partitions, checksums, authorship, DOI, license, converter hash, and software
environment in the generated manifest.

These are electrophoresis gels, not a Western-blot validation cohort. A detector
trained from them must be named and reported as an out-of-domain public
bootstrap model. It cannot support claims about Western-blot accuracy and does
not replace the reviewed, study-grouped annotation workflow.

## RGP caps electrophoresis images

The GelGenie `lsdb_gels` archive contains derived masks but directs users to the
original RGP caps database for the corresponding source images.

- Dataset: *RGP: 332 PCR-based genetic markers on rice chromosomes*
- Database DOI:
  [doi:10.18908/lsdba.nbdc00318-05-000](https://doi.org/10.18908/lsdba.nbdc00318-05-000)
- Electrophoresis-image collection DOI:
  [doi:10.18908/lsdba.nbdc00318-05-002](https://doi.org/10.18908/lsdba.nbdc00318-05-002)
- Provider: National Institute of Agrobiological Sciences / LSDB Archive
- License: Creative Commons Attribution-ShareAlike 2.1 Japan
- Required attribution: “RGP: 332 PCR-based genetic markers on rice
  chromosomes © National Institute of Agrobiological Sciences licensed under
  CC Attribution-Share Alike 2.1 Japan;
  doi:10.18908/lsdba.nbdc00318-05-002”

The preparation script pins and SHA-256 verifies the archive used for this
training run. Any redistributed derivative based on the RGP data remains
subject to its attribution and share-alike terms.

## Western-blot review sources

The following corpora may be used to create an annotation-review queue. They
are not treated as labelled training data on download.

### Human Protein Atlas v25.1

- Source: official v25.1 bulk metadata, restricted to Western-blot records with
  `source="HPA"` and their versioned HPA image URLs
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Primary reference:
  [doi:10.1126/science.1260419](https://doi.org/10.1126/science.1260419)
- Required per-image credit: “Image credit: Human Protein Atlas”, the specific
  versioned HPA image or gene URL, and the primary reference DOI above

These are processed medium-size panels, not raw instrument scans. Their
appearance, compression, source annotations, and HPA validation categories do
not establish band-box ground truth or detector accuracy. Only images hosted by
HPA and enumerated from the official bulk metadata are in scope; external
third-party image sources are excluded.

### Mendeley Data `zz3ktsyjds` v2

- Dataset: *An enzyme that selectively S-nitrosylates proteins to regulate
  insulin signaling*
- Immutable version:
  [doi:10.17632/zz3ktsyjds.2](https://doi.org/10.17632/zz3ktsyjds.2)
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Review payload: 186 raw PNG images from version 2

Attribution must identify the dataset title, creators, immutable version DOI,
and CC BY 4.0 license.

### BioImage Archive

- Study [S-BIAD2849](https://doi.org/10.6019/S-BIAD2849): 42 raw Western-blot
  TIFF payloads selected from 52 Western-blot-related entries
- Study [S-BIAD2306](https://doi.org/10.6019/S-BIAD2306): 38 Western-blot
  source ZIP archives
- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

For `S-BIAD2849`, only entries classified as raw Western-blot images enter the
review queue. For `S-BIAD2306`, `6A.zip`, `6D.zip`, `6G.zip`, `S9A.zip`,
`S9C.zip`, and `S9E.zip` are microscopy payloads and are excluded; its README
archive is retained only as provenance. The study accessions and DOIs remain
attached to derived records even though CC0 does not require attribution.

### Zenodo record 5810203

- Dataset: *Western blot images related to "A macrophage-hepatocyte
  glucocorticoid receptor axis coordinates fasting ketogenesis"*
- Immutable record:
  [doi:10.5281/zenodo.5810203](https://doi.org/10.5281/zenodo.5810203)
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Review payload: 20 original TIFF images; associated PDFs and RTF material are
  provenance documents only

Attribution must identify the dataset title, creators, immutable record DOI,
and CC BY 4.0 license.

Downloaded archives, extracted images, generated manifests, review projects,
and proposals remain in git-ignored local data directories and are not
redistributed. Detector-generated boxes are unapproved review aids, never
ground truth. None of these sources supplies complete band-level bounding-box
labels, so model fitting and checkpoint release remain fail-closed until a
human has corrected and approved the relevant boxes and completed the
whole-image review attestation.

The source-specific importer pins the five selection manifests in code,
restricts every download to its accession/version namespace, validates images
and scientific TIFF structure under explicit resource limits, and builds the
review corpus transactionally:

```bash
.venv/bin/python scripts/prepare_review_sources.py plan \
  --root data/public_western
.venv/bin/python scripts/prepare_review_sources.py fetch \
  --root data/public_western --workers 8
.venv/bin/python scripts/prepare_review_sources.py build \
  --root data/public_western --batch-size 20 \
  --gelgenie-model models/gelgenie_universal.onnx
```

The generated `data/public_western/MANUAL_LABELING.md` is the labeling
protocol; `REVIEW_QUEUE.tsv` is the batch ledger. A rebuilt corpus replaces the
entire previous generated corpus, so stale batches cannot silently remain in
the queue.

## Leakage and label-conflict policy

Public archives are not trusted merely because they are published. The
preparation step fingerprints each image and mask. Byte-identical images found
in different upstream partitions are either quarantined by the existing
labelled-data pipeline or joined into one leakage component by the Western
review importer. The latter preserves both the original study group and exact
rendered-image SHA-256. BlotGrid's split validator independently unions
`study_group` and `original_sha256`, so exact copies and related experiments
cannot cross automatic train/validation partitions.
