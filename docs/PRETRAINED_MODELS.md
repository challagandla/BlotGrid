# Pretrained model workflow

BlotGrid supports two verified open-source pretrained models without changing
its quantitative contract:

- **GelGenie Universal ONNX** proposes band locations from a full blot.
- **SAM 2.1 Hiera Small** optionally recenters one selected ROI from a box
  prompt.

Both outputs require human review. Measurement continues to use one locked
rectangular size across the blot, the original native-depth pixels, and
BlotGrid's classical background correction.

## Install the models

The ordinary install already includes CPU ONNX Runtime. Download the pinned
GelGenie model:

```bash
.venv/bin/python scripts/download_pretrained_models.py \
  --model gelgenie-universal
```

SAM is a separate optional dependency:

```bash
./scripts/setup_ubuntu.sh --with-sam2 --ml-runtime cu118
.venv/bin/python scripts/download_pretrained_models.py \
  --model sam2.1-hiera-small
```

The second setup command is intended for a normal Ubuntu terminal where
`nvidia-smi` can see the GPU. A container or managed coding sandbox may hide
`/dev/nvidia*`; that is not evidence that the host driver should be reinstalled.

Verify both local inference paths:

```bash
.venv/bin/python scripts/smoke_pretrained_models.py --device auto
```

The installer and smoke check print model revision, size, SHA-256, runtime
provider/device, and whether the artifact matched the allowlist. Local model
files and metadata sidecars stay in the ignored `models/` directory.

## Use GelGenie

1. In the sidebar, select **GelGenie pretrained ONNX + classical recovery**.
2. Keep `models/gelgenie_universal.onnx`, or choose another path containing the
   exact verified official artifact.
3. Use automatic normalization unless a validation experiment supports another
   setting.
4. In **2 · ROIs**, choose **Detect all bands**.
5. Inspect the whole image, add missed bands, disable false proposals, and
   correct target/lane assignments.

GelGenie is an out-of-domain starting point: it was trained on electrophoresis
gels rather than Western blots. BlotGrid converts its foreground components to
shared-size rectangles, marks all of them for review, and uses the classical
cell-local recovery pass when a grid cell has no GelGenie proposal.

## Use SAM refinement

1. Install the optional package and verified checkpoint as shown above.
2. Open **Experimental SAM ROI refinement** in the sidebar and select `auto` or
   `cuda`.
3. In **Band review & intensity profile**, select an ROI.
4. Choose **Refine this ROI position with SAM 2.1**.
5. Inspect the recentered orange proposal and approve it only if correct.

SAM sees a locally contrast-stretched RGB crop and the existing ROI as an XYXY
box prompt. It chooses the connected mask component nearest the original
centre. Implausibly large, empty, or displaced masks are rejected. Only ROI
position may change; width and height remain locked. The project records the
checkpoint hash, prompt, mask score, old/new geometry, and the fact that no mask
pixels were used for quantification.

See [third-party model notices](../THIRD_PARTY_MODELS.md) for immutable
artifact and license details.
