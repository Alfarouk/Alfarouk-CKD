# Kidney Segmentation Lab

A small research workbench for kidney-ultrasound segmentation experiments. It
loads labelled masks, preserves region IDs, makes overlays, checks data splits
and runs a documented Swin V2 training and inference path.

The supplied project had useful model and analysis code hidden inside a Windows
virtual environment. This repository separates that work from installed
packages and makes the software portable and testable.

**This is research software, not a diagnostic system.** It does not classify a
person's kidney health or stage CKD. The original trained kidney checkpoint and
source dataset were not included in the archive. The included sample is a
generated engineering fixture, not a patient scan or a model prediction.

![The running local research workbench showing a generated labelled sample](docs/images/workbench.jpg)

## Try the workbench

Use Python 3.12 or newer. The tested environment uses Python 3.12 on macOS.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
ckd serve
```

Open **http://127.0.0.1:8782**. On Windows, activate with
`.venv\Scripts\activate`. The workbench stays on localhost; it does not send
images to a cloud service. Stop it with Ctrl+C.

The page can load an image and its integer label mask, switch between image and
overlay, add known row/column spacing, and download the analysis as JSON. It
does not load a model automatically or pretend the sample proves accuracy.

## Example output

![Generated engineering fixture with region overlay](docs/demo/engineering-fixture.overlay.png)

This is the exported generated fixture. It is not an ultrasound patient scan or
a trained segmentation result. The JSON beside it records that distinction.

## Use the command line

```bash
ckd demo --output results/demo
ckd analyze --image image.png --mask labels.png --output results/analysis
ckd analyze --image image.png --mask labels.png --spacing 0.25 0.30 --output results/calibrated
ckd analyze --image image.png --mask labels.png --reference reference.png --output results/evaluation
ckd check-data --manifest data/manifest.csv
ckd batch --manifest data/manifest.csv --output results/batch
```

The analysis exports the original grayscale image, integer mask, overlay and
JSON report. Batch analysis also writes CSV. Spacing is **row mm/pixel, column
mm/pixel for the original image**. Without it, results stay in pixels. Volume
is available only when both a real scale and an independently measured third
dimension are supplied with `--depth-mm`. No anatomy or disease meaning is
inferred from a coloured region.

## Train or use a model

```bash
python -m pip install -e '.[ml,test]'
ckd train --manifest data/manifest.csv --output checkpoints/experiment --epochs 10
ckd predict --model checkpoints/experiment/model.json --image image.png --output results/prediction
python -m pytest
```

Training preserves the source Swin V2 tiny backbone, five-stage decoder and
focal-loss option. Use `--loss cross-entropy` for a cross-entropy baseline. `--pretrained-backbone microsoft/swinv2-tiny-patch4-window8-256` explicitly
opts into downloading the published backbone. That backbone alone is **not** a
trained kidney segmenter. The default uses random initialization without a
network download; it needs adequate research data and training.

Use the [data guide](docs/DATA.md) before training. Each sample has an explicit
patient/group ID and split. Repeated patients or identical image content across
splits fail validation. The corrected task has five neutral labels, so a
source-era four-class model must not be silently relabelled or presented as
compatible. Checkpoint loading uses safetensors plus a hash-bound manifest.

## What was repaired

- Region labels are no longer overwritten by a whole-kidney capsule mask.
- Training and prediction share preprocessing and a clear label contract.
- Data splits are explicit instead of silently splitting an already split set.
- Physical measurements require calibration; an assumed scan depth is gone.
- Paths are portable. Missing files and invalid masks produce useful errors.
- Model loading no longer falls back to unrestricted Python pickle execution.

See [architecture](docs/ARCHITECTURE.md), [model card](docs/MODEL_CARD.md),
[provenance](docs/PROVENANCE.md) and [verification](docs/VERIFICATION.md).

## Ownership and credit

Prepared for Alfarouk Salah Elkashef to maintain and explain. The supplied
academic code is the starting point; the November 2026 recovery, tests, UI and
documentation were developed with AI assistance. PyTorch, Transformers and
Swin V2 remain credited to their authors. No medical dataset, personal webcam
image, bundled Windows environment or unrelated YOLO checkout is redistributed.

The supplied custom files did not state a license. This repository does not
claim ownership of collaborators' work or grant rights they have not granted.
Confirm the source contributors and intended license before public release.
