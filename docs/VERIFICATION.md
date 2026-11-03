# Testing

```bash
python -m pip install -e '.[ml,test]'
python -m pytest
# Include private-model integration tests after fetching the assets:
CKD_MODEL_DIR="$PWD/models" python -m pytest
```

The tests check masks, calibration, empty regions and overlap calculations;
patient and image separation between splits; exports and CLI/HTTP behavior;
offline model architectures; and checkpoint import and download checks. They
also cover hash failures, input profiles and the two label schemas. A small
five-label model is trained, saved, loaded and used for prediction.

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

All 70 local tests passed with the private assets. Separate model checks ran
the three models 24 times on the CPU, using generated images and the named
input profiles. Each output was finite and had the expected shape. Package and
browser checks covered all three JSON exports, both Swin profiles, four-class
mask upload and different row/column pixel scales.

Fresh Linux CI checks a clean dependency install and tests without private
assets. The four private-model integration checks run locally when
`CKD_MODEL_DIR` points to verified files. Generated inputs test the software;
they are not an accuracy benchmark or clinical validation.

`requirements.lock` records resolved versions. Install them, then install the
project with `pip install -e . --no-deps`. The UI-only install needs no PyTorch.
Dependency upgrades require the appropriate tests and model checks.
