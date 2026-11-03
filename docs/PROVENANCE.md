# Attribution and use

The workbench uses the following third-party architectures and libraries:

- [Swin V2](https://huggingface.co/docs/transformers/model_doc/swinv2), through
  Hugging Face Transformers.
- [EfficientNet-B4](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b4.html),
  through TorchVision.
- [ConvNeXt](https://huggingface.co/docs/transformers/model_doc/convnext), through
  Hugging Face Transformers.
- PyTorch, safetensors, NumPy, Pillow and OpenCV for tensors, storage and images.

These projects retain their authorship and respective licenses. Installing the
workbench does not change the terms of its dependencies.

## Model assets

`model_registry.json` identifies each checkpoint and operational safetensors
file by its exact SHA256. Assets are stored in this private repository’s model
release. Runtime compatibility does not establish training-data rights,
clinical validity or permission to redistribute a model.

The classifier files contain tensors without class names or training input
settings. Their runtime assumptions are documented in the [model card](MODEL_CARD.md).
The four-output Swin checkpoint is distinct from the five-label training pipeline.

## Data and distribution

Three rendered project-output figures are included in the Examples tab. They
show an ultrasound overlay/report and two biopsy-score plots. They are saved
figures, not results computed when the page opens. Image metadata has been
removed without changing the pixels. The raw training dataset and original
DICOM files are not included. The CLI/test fixture is generated test data.

The [kidneyUS project](https://github.com/rsingla92/kidneyUS) is a reference for a
related capsule/region annotation format, not a confirmed training-data source
for these checkpoints. Its images have their own access conditions and
non-commercial terms; none are redistributed here.

This repository has no open-source license grant. Keep code, model and dataset
permissions separate when planning public or commercial distribution.
