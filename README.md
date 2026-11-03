# Kidney Segmentation Lab

Kidney Segmentation Lab runs Swin V2, EfficientNet-B4 and ConvNeXt models and
measures regions from label masks. The browser and CLI use the same model and
analysis code. Both can export the image, output and settings for each run. The Examples tab
opens saved ultrasound and biopsy output figures.

![Ultrasound and biopsy examples in the workbench](docs/images/workbench.jpg)

## Quick start

Use Python 3.12 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
ckd serve
```

Open **http://127.0.0.1:8782**. In Windows PowerShell, activate the environment
with `.\.venv\Scripts\Activate.ps1`. Stop the server with Ctrl+C.

The app opens with three saved output figures: an ultrasound scan with its
segmentation overlay and report, plus biopsy examples labelled Normal and
Sclerosed. Select a figure to see it in full, or open the full-size image.
These are saved examples; switching between them does not run a model.

![Saved biopsy patch and score chart](docs/images/biopsy-example.jpg)

**Run a model** and **Use your labels** show new results separately. Mask analysis
works with the core install. For inference, add the model dependencies and files:

```bash
python -m pip install -e '.[ml,test]'
ckd fetch-models --models-dir models
ckd serve --models-dir models
```

Downloads require the [GitHub CLI](https://cli.github.com/) and access to this
private repository. The downloader uses the existing sign-in, checks each
SHA256 and preserves existing files. Model assets are pinned to release
`v1.1.0`, independently of the application version. No credentials or model
binaries are stored in Git.

## Run inference

In **Run a model**, choose a model, input profile and image. Each result keeps its
own model name and input settings, even when you change the next selection.
The downloaded report includes the weight hash, model configuration, input
settings and output. The workbench loads one model at a time.

| Model | Output | Input profiles |
| --- | --- | --- |
| Swin V2 tiny | Four pixel-label IDs | Grayscale 256; grayscale 256 with CLAHE |
| EfficientNet-B4 | Two output indices | RGB 224 or RGB 380 with ImageNet normalization; RGB 224 unit range |
| ConvNeXt tiny | Two output indices | RGB 224 or RGB 380 with ImageNet normalization; RGB 224 unit range |

```bash
ckd models --models-dir models
ckd run-model --id source-swin-four --profile source-training-256 \
  --image image.png --models-dir models --output results/swin
ckd run-model --id source-swin-four --profile source-clahe-256 \
  --image image.png --models-dir models --output results/swin-clahe
ckd run-model --id biopsy-efficientnet --profile research-rgb-224 \
  --image image.png --models-dir models --output results/efficientnet
ckd run-model --id sclerosis-convnext --profile research-rgb-224 \
  --image image.png --models-dir models --output results/convnext
```

The two Swin profiles use different grayscale decoding and resize methods.
The CLAHE profile also changes local contrast. For the classifiers, the
profiles compare image sizes and normalization settings. Their original
training recipe and class order are undocumented. See the
[model card](docs/MODEL_CARD.md) before interpreting results.

If you have the checkpoint files listed in the registry, convert them locally:

```bash
ckd import-models --source-dir /path/to/checkpoints --models-dir models
```

Import accepts only files whose hashes match the registry. It loads tensors
with restricted loading, checks their names, shapes and finite values, then
writes safetensors files. It leaves the input files unchanged. The private
asset release also contains the `.pth` files; use
`ckd fetch-models --models-dir models --include-originals` to download them.

## Analyze masks and geometry

```bash
ckd demo --output results/demo
ckd analyze --image image.png --mask labels.png --output results/analysis
ckd analyze --image image.png --mask labels.png --spacing 0.25 0.30 --output results/calibrated
ckd analyze --image image.png --mask labels.png --label-schema source-four --output results/four-class
ckd check-data --manifest data/manifest.csv
ckd batch --manifest data/manifest.csv --output results/batch
```

Exports contain the image, JSON and, for segmentation, the mask and overlay.
Batch analysis also writes CSV. Spacing is **row mm/pixel, column mm/pixel**
for the input image. Without calibration, measurements stay in pixels. Volume
also requires an independently measured third dimension via `--depth-mm`.
Separate output folders are required for segmentation and classification.

## Train a five-label segmenter

```bash
ckd train --manifest data/manifest.csv --output checkpoints/experiment --epochs 10
ckd predict --model checkpoints/experiment/model.json --image image.png --output results/prediction
python -m pytest
```

The training pipeline preserves region IDs and uses five labels. It stays
separate from the four-output checkpoint, with its own model class, manifest,
result schema and legend. The code never relabels the four-output weights.

Use the [data guide](docs/DATA.md) to prepare patient-separated manifests.
Duplicate image content across splits is rejected. Training starts with random
weights unless `--pretrained-backbone microsoft/swinv2-tiny-patch4-window8-256`
is selected. That option downloads a backbone, not a trained kidney segmenter.
Focal loss and cross-entropy are available.

## Scope

The workbench is for research on image models and labelled regions. It does
not diagnose disease or assign stages. The checkpoint class meanings and
training-data identity are unverified, and index scores are not calibrated
disease probabilities. Assessing accuracy requires licensed labelled data and
independent evaluation. The generated CLI/test fixtures check software behavior only.

Read the [architecture](docs/ARCHITECTURE.md), [model card](docs/MODEL_CARD.md),
[data guide](docs/DATA.md), [test guide](docs/VERIFICATION.md) and
[attribution and use notes](docs/PROVENANCE.md).
