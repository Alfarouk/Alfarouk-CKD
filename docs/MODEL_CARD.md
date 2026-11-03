# Model card

**Purpose:** engineering and research experiments on labelled kidney ultrasound
images. No diagnostic indication or clinical deployment is claimed.

**Architecture:** Swin Transformer V2 tiny backbone, followed by five learned
upsampling blocks and a pixel classifier. This preserves the source design;
Swin V2 is a third-party architecture, not a new architecture invented here.
The corrected task has five labels described in the data guide.

**Available weights:** no trained kidney model is included. The supplied archive
contains a file called `sam_b.pt`, but that is not the `best_swinv2_kidney.pth`
the inference code expects. It was not unpickled, trusted or published as a
kidney model. Neither an ImageNet backbone nor random weights establish kidney
segmentation performance.

**Training and evaluation:** use patient-separated manifests and a held-out test
set. Training writes this run's loss and foreground macro Dice, with per-class
overlap available for labelled predictions. An absent class in both masks has
a null score and is excluded from the macro average. A single synthetic run
checks the program; it cannot provide scientific accuracy or generalization.

**Measurements:** physical units require actual image calibration. The workbench
does not infer pixel spacing from an assumed scan depth, assign anatomy names
to unverified IDs, estimate cortex thickness from distance to its inner edge,
or infer disease severity from image dimensions. Any future medical use needs
the appropriate clinical, data-quality and regulatory validation.

**Known missing source assets:** original labelled dataset and its annotation
guide, trained kidney checkpoint, data split/patient identifiers and any
source-era training history. No performance figures are reconstructed from
screenshots or invented to fill these gaps.

## Primary technical references

- [Transformers Swin V2 documentation](https://huggingface.co/docs/transformers/model_doc/swinv2)
- [Microsoft's published Swin V2 tiny backbone](https://huggingface.co/microsoft/swinv2-tiny-patch4-window8-256)
- [PyTorch serialization guidance](https://docs.pytorch.org/docs/stable/notes/serialization)
- [Open Kidney US authors' access and license information](https://github.com/rsingla92/kidneyUS)
