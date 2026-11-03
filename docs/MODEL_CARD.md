# Model card

Kidney Segmentation Lab runs the three image models below and measures labelled
regions. Their outputs support image-model research, not diagnosis or disease
staging.

| Model asset | Runtime | Output | Input contract |
| --- | --- | --- | --- |
| best_swinv2_kidney.pth | Swin V2 tiny with five decoder blocks | Four pixel-label IDs | Grayscale 256 and grayscale 256/CLAHE profiles |
| biopsy_pathologist.pth | TorchVision EfficientNet-B4 with nested linear head | Two indices | Three explicit RGB research profiles |
| kidney_sclerosis_convnext.pth | Transformers ConvNeXt tiny | Two indices | Three explicit RGB research profiles |

The registry pins checkpoint and safetensors hashes. Inference uses the
safetensors assets in the private model release. The loader checks every
tensor name, shape and finite value. A matching state dictionary means the
weights fit the configured model. It does not identify
undocumented operations or class meanings, or establish clinical performance.

## Swin V2

The four-class configuration uses a 256 × 256 input, batch size 8 and learning
rate 0.0002. Its checkpoint records zero-based `epoch=40` and a plan of 50 epochs.
The stored `val_dice=0.8047522725330458` is historical metadata, not independently
reproduced accuracy. That metric includes background and scores a class absent
from both masks as 1. It cannot be directly compared with foreground macro Dice.

Two input profiles are available:

- `source-training-256`: Pillow grayscale, bicubic resize to 256 × 256 and /255.
- `source-clahe-256`: OpenCV grayscale, CLAHE (2.0, 8 × 8), linear resize to 256 × 256 and /255.

Changing the input profile can change a prediction. Neither profile identifies
the original dataset or its annotation rules. The four output IDs remain
neutral. Their anatomical meanings are unverified. Keep them separate from the five-label
training pipeline.

## Classifiers

The classifier checkpoints contain tensors without class names, output order,
input recipe, dataset, split, training history or accuracy. Swapping label names
would leave the same weights intact, so tensor values alone cannot identify a class.

EfficientNet uses the B4 backbone and a nested Identity→Linear head. ConvNeXt
uses tiny widths/depths, GELU and layer-normalization epsilon 1e-12. These are
explicit inference configurations. The weights do not uniquely identify
operations that have no saved parameters: Identity and ReLU head variants
accept the same biopsy weights but give different outputs on generated inputs.

The profiles are RGB 224/ImageNet normalization, RGB 380/ImageNet normalization
and RGB 224/unit range. Each report records the selected profile, raw logits and
softmax index scores. The training recipe is undocumented. Scores are not
calibrated clinical probabilities and cannot validate a profile or label meaning.
The workbench produces no cross-model ensemble or disease verdict.

## Five-label training

The separate training pipeline preserves all region IDs. Use patient-separated
manifests and a held-out test set. Each run records loss and foreground macro
Dice. A class absent from both masks is null and excluded from that average.
Generated samples test software behavior only.

Physical units require image calibration. Volume also requires a measured third
dimension. Scientific performance claims require labelled data, its annotation
and split history, and independent evaluation.

## References

- [PyTorch restricted serialization](https://docs.pytorch.org/docs/2.15/notes/serialization.html)
- [Transformers Swin V2](https://huggingface.co/docs/transformers/model_doc/swinv2)
- [TorchVision EfficientNet-B4](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b4.html)
- [Transformers ConvNeXt](https://huggingface.co/docs/transformers/model_doc/convnext)
- [kidneyUS annotation format and access terms](https://github.com/rsingla92/kidneyUS)
