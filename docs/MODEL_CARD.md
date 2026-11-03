# Supplied model card

These models support local engineering and research inspection. There is no
diagnostic indication, disease staging or clinical deployment claim.

| Artifact | Verified tensor-compatible runtime | Outputs | Input knowledge |
| --- | --- | --- | --- |
| best_swinv2_kidney.pth | Swin V2 tiny + five decoder blocks | Four pixel indices | Two differing recipes recovered from source code |
| biopsy_pathologist.pth | TorchVision EfficientNet-B4 + nested linear head | Two raw indices | Original resize/crop/channels/normalization unknown |
| kidney_sclerosis_convnext.pth | Transformers ConvNeXt tiny | Two raw indices | Original image recipe and class order unknown |

The registry pins original and converted hashes. Operational files are
safetensors in a private release, outside Git. Strict tensor loading and finite
checks passed for all three. This establishes compatible names/shapes, not the
original parameterless operations, complete training graph or clinical value.

## Four-class source Swin

The supplied checkpoint matches the recovered source model tensors. It records
`epoch=40`, four classes, a 256 × 256 target, batch size 8, learning rate 0.0002 and 50
planned epochs. The source saves its loop index, so 40 is reported as stored,
not rewritten as 40 completed epochs.

Its stored validation Dice is 0.8047522725330458. The original metric averages
all four classes including background and assigns 1 when a class is absent from
both prediction and target. This value has not been independently reproduced
and must not be compared directly with the corrected foreground-macro metric.

The source training recipe is Pillow grayscale, bicubic 256 resize and /255.
The source inference recipe is OpenCV grayscale, CLAHE (2.0, 8 × 8), linear 256
resize and /255. Both are explicit choices in the workbench; neither is called
a proven original training match for every use of these supplied weights.

Four neutral source IDs remain separate from the five corrected labels. The
original scripts disagree about anatomy. Their combination step can replace
internal region labels with capsule class 3. This was reproduced on the public
masks in a closely matching candidate dataset, whose original images require
access approval. Exact checkpoint-dataset identity remains unverified.

Six synthetic-input runs across both recipes show that the input choice changes
outputs. Class 2 never won a pixel in those probes, while class 1 did. This is not
a clinical experiment and does not prove a particular training collapse.

## Classifiers

Each supplied classifier stores only tensors. Neither file records class
names/order, input recipe, dataset, split, training history or accuracy.
Reassigning the two output labels leaves the same weight tensors intact, so
weights alone cannot determine which semantic label belongs to an index.

The selected runtime is explicit. EfficientNet uses the library B4 backbone
and a nested Identity→Linear head. The Identity operation is an assumption:
parameterless head operations are not recoverable from state-dictionary keys.
ConvNeXt uses the recorded tiny widths/depths with GELU and layer-normalization
epsilon 1e-12. These choices are documented research settings, not hidden claims
about the original training code.

A controlled recovery check loaded the same biopsy tensors into Identity and
ReLU head variants. Both matched every tensor strictly, yet produced different
outputs on generated inputs. This shows why those stored tensors cannot uniquely
identify the missing parameterless operation; it does not select an original variant.

RGB224/ImageNet normalization, RGB380/ImageNet normalization and RGB224/unit
range are named experiment profiles. Every result stores the actual settings,
raw logits and softmax index scores. Those scores are neither calibrated
clinical probabilities nor proof of the correct original profile. No cross-model
ensemble or disease verdict is produced.

## Corrected five-label training

The separate experiment keeps five labels and preserves the input region IDs.
Use patient-separated manifests and a held-out test set. It writes this run’s
loss and foreground macro Dice. An absent class in both masks is null and
excluded from that macro average. Generated samples test software only.

Physical units require actual image calibration. No assumed scan depth,
centre-to-cortex heuristic or image-dimension disease rules are restored.
Original labelled data, annotation/split history and independent evaluation are
still needed for scientific performance claims.

## Primary technical references

- [PyTorch restricted serialization](https://docs.pytorch.org/docs/2.15/notes/serialization.html)
- [Transformers Swin V2](https://huggingface.co/docs/transformers/model_doc/swinv2)
- [TorchVision EfficientNet-B4](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b4.html)
- [Transformers ConvNeXt](https://huggingface.co/docs/transformers/model_doc/convnext)
- [Candidate kidneyUS source and access conditions](https://github.com/rsingla92/kidneyUS)
