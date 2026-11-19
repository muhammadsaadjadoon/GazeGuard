# Dataset and large-file guidance
## What stays out of Git
- Raw video and audio recordings, private landmark CSVs, trained models from real people, and third-party datasets must not be committed to this repository.
- Large public datasets can remain locally at `datasets/public_external/` — do not push that folder to GitHub.
- Only small metadata files, schemas, and documentation belong in `datasets/` in the repository.
## Before using an external dataset
- Obtain it directly from its official source and follow that dataset owner's terms and licence.
- Third-party dataset rights are separate from this project's code licence, and downloading a dataset does not transfer any rights.
- The training workflow in this repository expects the project's own labelled landmark CSV format; dropping in an unrelated dataset will not train it.
## Private data
- Landmark observations are biometric-related information even when no raw images are stored. Collect them only with explicit, informed consent, keep them local, restrict access, and delete them when no longer needed.
