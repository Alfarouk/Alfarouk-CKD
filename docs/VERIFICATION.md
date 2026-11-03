# Verification

Run the complete local suite with:

```bash
python -m pip install -e '.[ml,test]'
python -m pytest
```

The suite currently contains 41 tests. They cover label preservation, mask validation, calibration, anisotropic
spacing, empty masks, overlap conventions, patient/duplicate separation,
portable image handling, export files, CLI failures, local HTTP requests and a
small offline Swin V2 training/checkpoint/inference round trip. The small model
and generated samples are engineering fixtures, not clinical experiments.

To make a human-readable example:

```bash
ckd demo --calibrated --output results/demo
ckd serve
```

The demo exports actual PNG and JSON artifacts and the browser exercises the
same analysis function. Upload both an image and an integer mask to inspect a
different pair. Files stay local and are not saved by the web server.

Latest resolved package versions are recorded in `requirements.lock`. The
direct versions were checked against the official PyPI registry during the
November 2026 recovery. To reproduce the resolved environment, install `requirements.lock`, then
install the repository with `python -m pip install -e . --no-deps`. The ordinary
UI-only install stays small and does not require PyTorch. Upgrades still require
this suite; a newer version is
not itself proof of compatibility.

The source's original trained inference cannot be reproduced without its
missing checkpoint, annotation contract and dataset. A passing software suite
must not be described as proof of clinical accuracy or original model results.
