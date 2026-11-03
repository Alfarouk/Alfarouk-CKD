# Testing

```bash
python -m pip install -e '.[ml,test]'
python -m pytest
# Include private-model integration tests after fetching the assets:
CKD_MODEL_DIR="$PWD/models" python -m pytest
```

The suite covers masks, calibration, empty regions, overlap conventions,
patient/image split separation, exports, CLI and HTTP behavior, offline model
architectures, hash failures, input profiles, label-schema separation, restricted
checkpoint import and asset-download safeguards. A small five-label
training/save/load/prediction round trip is included.

## Run the main flows

```bash
ckd fetch-models --models-dir models
ckd demo --output results/demo
ckd run-model --id source-swin-four --profile source-training-256 \
  --image results/demo/engineering-fixture.image.png --models-dir models --output results/swin-check
ckd run-model --id biopsy-efficientnet --profile research-rgb-224 \
  --image results/demo/engineering-fixture.image.png --models-dir models --output results/classifier-check
ckd serve --models-dir models
```

In the browser, check the sample, mask upload and each model. Confirm the result’s
model name, input profile, four-class legend or two output indices, view controls
and JSON download. Changing the next selection must not relabel the current
result. Model controls lock during inference. An incomplete scale must show an
error while preserving the previous result.

## Test coverage

All 70 local tests passed with the private assets. The three models completed
24 sequential CPU runs across generated inputs and named profiles, with finite
outputs matching their shape contracts. The built wheel’s resources and
browser flows were also checked, including all three JSON exports, both Swin
profiles, source-four mask upload and anisotropic calibration.

Fresh Linux CI checks a clean dependency install and tests without private
assets. The four private-model integration checks run locally when
`CKD_MODEL_DIR` points to verified files. Generated inputs test the software;
they are not an accuracy benchmark or clinical validation.

`requirements.lock` records resolved versions. Install them, then install the
project with `pip install -e . --no-deps`. The UI-only install needs no PyTorch.
Dependency upgrades require the appropriate tests and model checks.
