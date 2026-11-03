# Project provenance

The source was supplied in `CKD.zip`. The useful custom files were inside
`.venv/Scripts`, beside installed Windows programs:

- `kidney_swin_model.py`: Swin V2 model, three-file dataset, training, metrics,
  measurements and inference.
- `examples.py`: a second model/inference/report path with a different anatomy
  interpretation and preprocessing.
- `utils.py`: image loading, mask quality checks, overlap and overlays.
- `config.yaml`: local segmentation settings and paths.

The location did not make these files third-party dependencies. They were
identified separately from installed-package RECORD manifests before recovery.

The archive also contained Ultralytics YOLOv5 at upstream commit
`2540fd4c1c2d9186126a71b3eb681d3a0a11861e`. All 146 non-environment files were
compared with its Git object database. Code matches after line-ending
normalization; only a training-command argument in two README files differs.
This does not establish an original YOLO implementation by Alfarouk. YOLOv5 is
not copied into the clean kidney workbench.

## Source-era work and this recovery

The archive is associated with Alfarouk's academic work. His family believes
he implemented the custom CKD project; that is not a verified sole-authorship
history. The upstream models and libraries remain credited to their authors.
The November 2026 restoration adds portable packaging, explicit contracts,
tests, data validation, safe checkpoint manifests, a local UI and documentation
with AI assistance. These new additions must not be backdated as original work.

The source had no stated license for the custom files. No blanket license or
copyright ownership over collaborators' code is invented. Original contributors
and release rights should be established before publishing the source openly.

The clean repo excludes bundled environments, bytecode, IDE paths, unrelated
images, a private webcam photo, patient-derived analysis plots and model blobs.
Original archive hashes and the detailed recovery evidence are retained in the
private LifeFlow profile rather than exposed in this public-facing workbench.


## Separately supplied model files

On 3 November 2026 three model files were supplied separately from the archive:
`best_swinv2_kidney.pth`, `biopsy_pathologist.pth` and
`kidney_sclerosis_convnext.pth`. The first has the source Swin tensor layout
and saved training configuration. The other two strictly fit the documented
research runtimes but contain only tensors, with no labels or input recipe.

These files establish supplied artifacts, not who independently trained each
model or its clinical accuracy. Exact originals and converted safetensors are
kept in the private repository’s release assets. Source licensing and dataset
rights remain separate from the ability to execute a tensor-compatible model.

A controlled test also found that different parameterless classifier-head
operations can accept exactly the same tensors while giving different outputs.
The documented runtime assumption is therefore kept separate from original
training provenance.
