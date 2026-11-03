# Kidney Segmentation Lab

A local research workbench for inspecting model outputs and labelled masks.
It now runs three supplied checkpoints: a four-class Swin V2 segmenter and two
classifiers. It also keeps the corrected five-label training experiment,
calibrated geometry, data checks, batch exports and generated sample.

**Research software; no diagnosis or disease staging.** Running the weights
proves a software path. It does not recover missing class meanings or establish
accuracy on a patient image. The classifiers return two neutral indices. Their
original image recipes and class order have not been recovered.

The original custom code was found inside a bundled Windows environment. This
repository separates it from installed packages and makes it portable. The
November 2026 recovery, UI, tests and documentation were developed with AI
assistance. Those additions are distinct from the original academic work.

![Local research workbench with a generated labelled sample](docs/images/workbench.jpg)

## Start locally

Use Python 3.12 or newer. Python 3.12 on macOS is the verified local environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml,test]'
ckd fetch-models --models-dir models
ckd serve --models-dir models
```

Open **http://127.0.0.1:8782**. On Windows, activate with
`.venv\Scripts\activate`. Stop the server with Ctrl+C.

`fetch-models` uses your existing GitHub CLI sign-in and access to this private
repository. It downloads the three safetensors assets from release `v1.1.0`,
checks every SHA256 and refuses to overwrite different existing files. No
credentials are stored in the repository. Original `.pth` files are also kept
in the private release; add `--include-originals` to retrieve them.

For the generated sample and manual mask analysis alone, `pip install -e .`
remains enough. They work without PyTorch or downloaded model files. The server
stays on localhost and sends no images to an external service.

## Run a supplied model

Choose **Run a model**, select a model, explicitly choose an input profile and
provide an image. The JSON export records the actual weight hash, input choice,
runtime configuration and outputs. Only one model is loaded at a time.

```bash
ckd models --models-dir models
ckd run-model --id source-swin-four --profile source-training-256 \
  --image image.png --models-dir models --output results/source-swin
ckd run-model --id source-swin-four --profile source-clahe-256 \
  --image image.png --models-dir models --output results/source-swin-clahe
ckd run-model --id biopsy-efficientnet --profile research-rgb-224 \
  --image image.png --models-dir models --output results/biopsy
ckd run-model --id sclerosis-convnext --profile research-rgb-224 \
  --image image.png --models-dir models --output results/convnext
```

The two recovered Swin recipes differ: the training code uses grayscale,
Pillow bicubic resize and division by 255; the inference script adds CLAHE and
uses OpenCV linear resize. Both are available by name. Neither is silently
substituted for the other.

Classifier options are **research input choices**, not recovered training
settings. RGB224 with ImageNet normalization, RGB380 with that normalization,
and RGB224 divided by 255 without normalization can be compared. Their results
may differ. A larger index score is not proof of the right recipe or a disease.
See the [model card](docs/MODEL_CARD.md).

If you already have the exact supplied originals, convert them locally:

```bash
ckd import-models --source-dir /path/to/originals --models-dir models
```

Conversion accepts only the recorded source hashes. It uses restricted
`weights_only=True` loading, checks all tensor names/shapes and finite values,
and writes safetensors. The Swin file needs a narrowly scoped, reviewed NumPy
scalar/dtype allowlist for its saved metric. No custom Python class is trusted,
no unrestricted pickle is used, and the originals are unchanged.

## Analyze labels and geometry

```bash
ckd demo --output results/demo
ckd analyze --image image.png --mask labels.png --output results/analysis
ckd analyze --image image.png --mask labels.png --spacing 0.25 0.30 --output results/calibrated
ckd analyze --image image.png --mask source-labels.png --label-schema source-four --output results/source-labels
ckd check-data --manifest data/manifest.csv
ckd batch --manifest data/manifest.csv --output results/batch
```

Exports contain the original image, mask when present, overlay when present and
JSON. Batch analysis also writes CSV. Spacing means **row mm/pixel, column
mm/pixel for the original image**. Otherwise geometry stays in pixels. Volume
requires both real spacing and an independently measured third dimension via
`--depth-mm`. No anatomy or disease is inferred from a coloured region. Use separate output
folders for segmentation and classifier results; cross-task reuse is rejected
before writing, so an old mask cannot accompany a new classifier report.

![Generated engineering fixture; not a patient image or model prediction](docs/demo/engineering-fixture.overlay.png)

## Keep the five-label experiment separate

```bash
ckd train --manifest data/manifest.csv --output checkpoints/experiment --epochs 10
ckd predict --model checkpoints/experiment/model.json --image image.png --output results/prediction
python -m pytest
```

This experiment preserves the Swin V2 backbone and five-stage decoder while
correcting the original mask-combination problem. It has **five labels**; the
supplied checkpoint has **four**. They use separate model classes, manifests,
result schemas and legends. The supplied weights are never relabelled as a
five-label trained model.

Use the [data guide](docs/DATA.md) first. Patients or identical image content may
not cross splits. Training defaults to random initialization with no download.
`--pretrained-backbone microsoft/swinv2-tiny-patch4-window8-256` explicitly opts
into fetching a published backbone; that backbone alone is not a trained kidney
segmenter. Focal loss and cross-entropy are supported.

## Evidence and limits

All three architectures strictly accept their supplied tensors. Real CPU runs
and input-profile probes are recorded separately from clinical evaluation.
The Swin file stores `epoch=40` and `val_dice=0.8047522725330458`; this is saved
metadata, not reproduced accuracy. Its original metric includes background and
scores classes absent in both masks as 1. Its annotation and split history still
need independent evidence.

The source mask-combination code overwrites region IDs wherever the capsule is
nonzero. This defect was reproduced on public masks from a matching candidate
dataset format. That supports the defect finding; it does not prove the
checkpoint used that exact dataset or establish medical label meanings.

Read [architecture](docs/ARCHITECTURE.md), [model card](docs/MODEL_CARD.md),
[data](docs/DATA.md), [provenance](docs/PROVENANCE.md) and
[verification](docs/VERIFICATION.md).

## Ownership and credit

Prepared for Alfarouk Salah Elkashef to maintain and explain. The upstream
Swin V2, EfficientNet and ConvNeXt architectures and their libraries remain
credited to their authors. No medical dataset, webcam image, bundled Windows
environment or unrelated YOLO checkout is redistributed.

The custom source did not state a license. The repository and release assets
remain private; no claim of sole authorship, collaborator rights or permission
for public redistribution is invented.
