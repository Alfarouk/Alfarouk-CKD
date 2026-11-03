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

In the browser, check the saved figures, mask upload and each model. Confirm the result’s
model name, input profile, four-class legend or two output indices, view controls
and JSON download. Changing the next selection must not relabel the current
result. Model controls lock during inference. An incomplete scale must show an
error while preserving the previous result.

## Test coverage

The suite includes 79 tests. Model checks use generated images and registered
input profiles to check output shapes and finite values. Browser checks cover
JSON exports, both Swin profiles, four-class mask upload and different
row/column pixel scales. Run the private-model checks with the verified assets
when changing a model or input profile.

Fresh Linux CI checks a clean dependency install and tests without private
assets. The four private-model integration checks run locally when
`CKD_MODEL_DIR` points to verified files. Generated inputs test the software;
they are not an accuracy benchmark or clinical validation.

`requirements.lock` records resolved versions. Install them, then install the
project with `pip install -e . --no-deps`. The UI-only install needs no PyTorch.
Dependency upgrades require the appropriate tests and model checks.

## Example gallery

Opening the app should show the saved ultrasound figure without calling the
generated-demo or inference endpoint. Each selector must show the full matching
figure and update its full-size link. Check all three examples on a narrow and
wide viewport. Switching to a model or mask analysis must keep saved figures
separate from computed results. The default page should not display generated
test imagery or source-figure diagnoses as a new result.
