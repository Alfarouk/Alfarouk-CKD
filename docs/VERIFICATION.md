# Verification

```bash
python -m pip install -e '.[ml,test]'
python -m pytest
# Include real private-asset HTTP/CLI checks after fetching the models:
CKD_MODEL_DIR="$PWD/models" python -m pytest
```

The suite checks masks, calibration, empty regions, overlap conventions,
patient/image split separation, safe exports, CLI and HTTP behavior, offline
model architectures, hash failures, explicit input profiles, four/five schema
separation, restricted imports and private asset download safeguards. A small
five-label training/save/load/inference round trip remains included.

All three supplied originals were statically inspected before restricted CPU
loads. Converted safetensors strictly matched every expected tensor and passed
finite-weight checks. Actual supplied weights were then run 24 times, one model
at a time, across three generated non-patient inputs and their named profiles.
Every output was finite and matched its shape contract. These are executable
recovery and sensitivity checks, not medical accuracy tests.

To reproduce the normal flows:

```bash
ckd fetch-models --models-dir models
ckd demo --output results/demo
ckd run-model --id source-swin-four --profile source-training-256 \
  --image results/demo/engineering-fixture.image.png --models-dir models --output results/source-check
ckd run-model --id biopsy-efficientnet --profile research-rgb-224 \
  --image results/demo/engineering-fixture.image.png --models-dir models --output results/classifier-check
ckd serve --models-dir models
```

In the browser, check the generated sample, manual label upload and each
supplied model. Confirm the source badge, four-class legend or two raw indices,
input choice, image/overlay toggle and JSON download. A generated input remains
a generated input even when a real checkpoint runs on it.

`requirements.lock` records resolved package versions. Reproduce it, then
install the repository with `pip install -e . --no-deps`. The UI-only install
stays small and does not require PyTorch. Package upgrades require the suite;
a newer version is not itself evidence of compatibility.

No held-out patient dataset, original classifier class order or independently
reproduced scientific accuracy has been recovered. Software checks must not be
reported as clinical validation.

## Installed-package browser checks for 1.1.0

The built wheel served the interface with its packaged registry and static files.
All three supplied models ran from the browser on a generated engineering image.
Both source Swin input recipes completed. Model identity and input settings
stayed attached to the current result when the next selection changed.
Controls locked during inference. All three JSON downloads were inspected.

Source-four mask upload and a supplied 0.4 × 0.5 mm pixel scale also passed.
An incomplete scale produced a clear error and preserved the previous result.
The current screenshots show an actual supplied-model run and uploaded-mask
analysis. Their input is generated test material, not a patient scan or accuracy
benchmark.
