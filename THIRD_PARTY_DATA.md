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
- DOI:
  [doi:10.18908/lsdba.nbdc00318-05-000](https://doi.org/10.18908/lsdba.nbdc00318-05-000)
- Provider: National Institute of Agrobiological Sciences / LSDB Archive
- License: Creative Commons Attribution-ShareAlike 2.1 Japan
- Required attribution: “RGP: 332 PCR-based genetic markers on rice
  chromosomes © National Institute of Agrobiological Sciences licensed under
  CC Attribution-Share Alike 2.1 Japan”

The preparation script pins and SHA-256 verifies the archive used for this
training run. Any redistributed derivative based on the RGP data remains
subject to its attribution and share-alike terms.

## Leakage and label-conflict policy

Public archives are not trusted merely because they are published. The
preparation step fingerprints each image and mask. Byte-identical images found
in different upstream partitions, or identical images carrying conflicting
masks, are quarantined from model fitting and recorded in the held-out
manifest. Clean training and validation records keep their actual source-image
SHA-256, so BlotGrid's split validator can still detect leakage.
