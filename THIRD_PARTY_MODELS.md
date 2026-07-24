# Third-party pretrained models

BlotGrid does not commit model weights to this repository. The allowlisted
installer downloads GelGenie and SAM artifacts from their official hosts, pins
immutable revisions, verifies exact byte counts and SHA-256 values, and writes
local metadata sidecars under the ignored `models/` directory. Optional detector
initializers are downloaded into TorchVision's cache by TorchVision itself;
BlotGrid separately verifies their pinned full digest and size before training.

## TorchVision detector initialization

Optional detector training uses TorchVision `0.22.1` and pins concrete COCO v1
weight enums rather than the mutable `DEFAULT` aliases:

- `FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1`:
  `https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth`
  - SHA-256: `fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533`
  - Size: 77,844,807 bytes
- `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`:
  `https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`
  - SHA-256: `dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf`
  - Size: 175,221,657 bytes
- `FCOS_ResNet50_FPN_Weights.COCO_V1`:
  `https://download.pytorch.org/models/fcos_resnet50_fpn_coco-99b0c9b7.pth`
  - SHA-256: `99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7`
  - Size: 129,612,099 bytes
- `RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1`:
  `https://download.pytorch.org/models/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth`
  - SHA-256: `5905b1c544219215e544dbe319720397bc4e68de61a733a59350d7976645b769`
  - Size: 153,130,989 bytes

Training records the installed TorchVision source revision, enum, URL, URL
SHA-256 prefix, pinned full SHA-256 and byte size, and the exact cached artifact
actually loaded. A correct eight-character filename prefix is not sufficient.
TorchVision source code is BSD-3-Clause, but TorchVision states that pretrained
models may have separate licences or terms inherited from their training
datasets. BlotGrid therefore records these COCO initializers as
`review_required`, not as cleared for checkpoint redistribution. Use
`--no-pretrained` when a release candidate must avoid this transfer-weight
lineage.

Before publishing any locally trained checkpoint, run:

```bash
.venv/bin/python scripts/check_checkpoint_release.py models/detector.pt \
  --release-output models/detector.release.pt \
  --attribution-output models/detector.attribution.json
```

The local training archive is never published directly: PyTorch ZIP readers can
ignore unreferenced members and trailing bytes. The command restricted-loads
the checkpoint, requires its SHA-verified sidecar to match the embedded metadata
exactly, strict-loads the declared model state, verifies exact tensor backing
storage, and permits only the generated model/SGD fields. It then copies only
those validated logical values into the distinct `--release-output` archive and
re-verifies that artifact. The release output is written only if dataset
sources and licences, public metadata, annotation approval, and initializer
redistribution checks all pass. Local training remains available when
publication clearance is incomplete.

Checkpoint format 2 adds the public-safe dataset and initializer lineage
contract. Publication metadata schema 2 additionally requires literal human
approval markers on every band annotation; schema-1 release candidates must be
retrained under that fail-closed policy. Format-1 checkpoints remain available
for restricted-load inference, but cannot be resumed: their exact initializer
and licence ancestry cannot be reconstructed safely. Start a new format-2
training run instead.

## GelGenie Universal

- Project: [GelGenie](https://github.com/mattaq31/GelGenie)
- Model card:
  [mattaq/GelGenie-Universal-Dec-2023](https://huggingface.co/mattaq/GelGenie-Universal-Dec-2023)
- Revision: `fc36cc1ea468308b7336939a66c9884a26752d7e`
- Artifact: `onnx_checkpoints/unet_dec_21_epoch_579.onnx`
- SHA-256: `10ea972a40625f03b5a2987e1a98b532c6d4ded1daae32f1eb91aa24db8d9c48`
- Size: 57,279,486 bytes
- License: Apache License 2.0
- Citation: Aquilina et al., “GelGenie: an AI-powered framework for gel
  electrophoresis image analysis,” *Nature Communications* 16, 4087 (2025),
  [doi:10.1038/s41467-025-59189-0](https://doi.org/10.1038/s41467-025-59189-0).

GelGenie was trained for gel-electrophoresis segmentation, not independently
validated Western-blot quantification. BlotGrid therefore treats every result
as an unapproved ROI proposal and never substitutes its segmentation mask for
native-pixel densitometry.

## Segment Anything 2.1 Hiera Small

- Project: [Segment Anything 2](https://github.com/facebookresearch/sam2)
- Source revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Checkpoint: `sam2.1_hiera_small.pt`
- SHA-256: `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`
- Size: 184,416,285 bytes
- License: Apache License 2.0

SAM 2.1 is an optional, experimental annotation assistant. BlotGrid gives it an
existing ROI as a box prompt and may use the selected mask only to propose a new
ROI centre. The shared ROI width and height remain locked, the original image
is still measured, and human approval is cleared after every refinement.

The use of these projects does not imply endorsement by their authors,
institutions, Meta, or the University of Edinburgh.
