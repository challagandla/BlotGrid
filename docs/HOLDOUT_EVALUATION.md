# Sealed detector evaluation

This workflow compares BlotGrid's classical detector, the verified GelGenie
adapter, and one local format-2 checkpoint carrying publication metadata schema
2 without exposing method identity or ground truth to the reviewer. It does not
create a validation cohort. A person must annotate and approve the untouched
holdout before sealing it.

## Required holdout COCO fields

Use the usual COCO `images`, `annotations`, and `categories` collections.
Category 1 is `band`; categories 2–5 are `border`, `text`, `smear`, and
`blank_cell`.

Every image needs this additional metadata:

```json
{
  "id": 1,
  "file_name": "study-07.png",
  "width": 1600,
  "height": 900,
  "blotgrid": {
    "study_group": "study-07",
    "annotation_complete": true,
    "original_sha256": "<64 lowercase hex characters>",
    "targets": ["Protein", "Loading control"],
    "lanes": ["Reference", "Treatment"],
    "lane_centers_hint": [400, 1200],
    "target_centers_hint": [300, 650],
    "bands_are_dark": true,
    "control_target": "Loading control",
    "reference_lane": "Reference",
    "review_roi_size": {"w": 120, "h": 40},
    "strata": {
      "scanner": "scanner-a",
      "exposure": "short",
      "background": "clean",
      "acquisition_style": "film"
    }
  }
}
```

Each band annotation must be human-approved and identify its grid cell and
visibility:

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [340, 280, 120, 40],
  "attributes": {
    "annotation_approved": true,
    "target": "Protein",
    "lane": "Reference",
    "visibility": "clear",
    "doublet_group": null
  }
}
```

Use the same non-empty `doublet_group` on two or more bands belonging to one
target/lane doublet. Every artifact region also requires
`"attributes": {"annotation_approved": true}`. The complete cohort must include
at least one clear band, one faint band, one reviewed doublet group, and one
region of each artifact type. Duplicate image bytes are rejected.

## Seal

Create the seal only after the COCO file and images are final:

```json
{
  "schema_version": 1,
  "sealed_utc": "2026-07-24T12:00:00Z",
  "annotations_sha256": "<SHA-256 of the final COCO JSON file>",
  "image_sha256": {
    "1": "<SHA-256 of study-07.png>"
  },
  "study_groups": ["study-07"]
}
```

The image mapping keys must be exactly all COCO image IDs or exactly all
filenames. Its groups must exactly match the cohort.

## Development and model-selection ledger

The ledger must enumerate every image and study group used for development,
threshold tuning, informal debugging, training, validation, or model
selection. Both completeness flags are mandatory:

```json
{
  "schema_version": 1,
  "complete": true,
  "completeness_attested": true,
  "study_groups": ["study-01", "study-02"],
  "image_sha256": [
    "<development image SHA-256>",
    "<model-selection image SHA-256>"
  ],
  "model_selection": [
    {
      "study_group": "study-01",
      "image_sha256": "<development image SHA-256>"
    },
    {
      "study_group": "study-02",
      "image_sha256": "<model-selection image SHA-256>"
    }
  ]
}
```

The inventory unions must exactly equal the top-level groups and hashes.
Overlap with the sealed cohort or the checkpoint's training ancestry aborts
the run.

The local detector must be a format-2 checkpoint with publication metadata
schema 2 and its exact generated metadata sidecar. Both `prepare` and `score`
restricted-load the checkpoint, require the sidecar metadata to match the
embedded metadata exactly, and validate the generated model/optimizer container
before trusting its training ancestry.

## Frozen plan

Choose all thresholds and densitometry settings before preparing the run:

```json
{
  "schema_version": 1,
  "protocol_id": "western-holdout-v1",
  "seed": 1337,
  "iou_threshold": 0.5,
  "classical": {"sensitivity": 0.65},
  "gelgenie": {
    "probability_threshold": 0.5,
    "review_threshold": 0.75,
    "device": "auto",
    "normalization": "auto"
  },
  "local": {
    "score_threshold": 0.5,
    "review_threshold": 0.75,
    "device": "auto"
  },
  "common": {"recovery_sensitivity": 0.65},
  "densitometry": {
    "background_method": "local",
    "margin": 4,
    "rolling_ball_radius": 50.0,
    "fixed_background": 0.0,
    "measurement_mode": "rectangle"
  }
}
```

Run `scripts/evaluate_detector.py prepare` as shown in the
[README](../README.md#sealed-detector-comparison). The run keeps image binaries
out of its JSON/CSV-only outputs. Give the reviewer:

```text
reviewer_packet/
  review_tasks/      blinded <task-id>.json projects
  review_order.csv   randomized task order
```

Also give them read-only access to the exact sealed `--images` directory through
a separate channel. Never provide the annotations, seal, or
`evaluator_private/`. The packet contains neither method identities nor COCO
ground truth. For each row in `review_order.csv`, the reviewer restores the
matching project JSON in BlotGrid and then re-uploads the sealed images; project
JSON stores image hashes and ROI geometry, not pixels. They review every whole
image, move/add/disable fixed-size ROIs as needed, approve every active ROI, and
save the completed project in a separate directory using the original
`<task-id>.json` filename. Resize handles and global size controls are locked
for evaluator tasks. The scorer requires exactly those three filenames.

## Review attestation and scoring

The reviewer supplies:

```json
{
  "schema_version": 1,
  "protocol_id": "western-holdout-v1",
  "reviewer": "reviewer identifier",
  "blinded": true,
  "ground_truth_not_viewed": true,
  "whole_image_reviewed": true
}
```

Run the README's `score` command. Scoring re-verifies every frozen input,
proposal binding, project image hash, integral in-bounds fixed-size ROI, and
attestation. It stages all results before publication and writes
`score_complete.json` last, so an interrupted run remains safely retryable.
