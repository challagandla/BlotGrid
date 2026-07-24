# Third-party pretrained models

BlotGrid does not commit model weights to this repository. The allowlisted
installer downloads each artifact from its official host, pins an immutable
revision, verifies the exact byte count and SHA-256, and writes a local metadata
sidecar under the ignored `models/` directory.

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
