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

No patient dataset or private camera image is bundled. The generated sample is
synthetic test material. The [kidneyUS project](https://github.com/rsingla92/kidneyUS)
is a reference for a related capsule/region annotation format, not a confirmed
training-data source for these checkpoints. Its images have their own access
conditions and non-commercial terms; none are redistributed here.

This repository has no open-source license grant. Keep code, model and dataset
permissions separate when planning public or commercial distribution.
