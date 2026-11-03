# Data and label contracts

## Five-label training data

The importer accepts an image, a binary capsule mask and a region mask for each
sample. Region masks contain integer IDs 0–3. The capsule is a filled area,
not a thin outer rim. The importer preserves region IDs and uses ID 4 only for
capsule pixels with no region label.

| ID | Meaning |
| --- | --- |
| 0 | Background |
| 1 | Region 1 |
| 2 | Region 2 |
| 3 | Region 3 |
| 4 | Capsule interior without a region label |

Anatomical names require the dataset’s matching annotation guide. Region IDs
remain neutral until that contract is established.

```bash
ckd import-triplets --images source/images --capsules source/masks/capsule \
  --regions source/masks/regions --output data/imported
```

The importer copies images to PNG without resizing or overwriting input files.
It rejects region pixels outside the capsule, missing masks and shape mismatches.
Create a manifest beside the imported folders:

```csv
split,patient_id,image,mask
train,patient-001,images/sample-001.png,masks/sample-001.png
val,patient-002,images/sample-002.png,masks/sample-002.png
test,patient-003,images/sample-003.png,masks/sample-003.png
```

Paths must stay inside the manifest directory. Split at patient or source-group
level so related frames cannot cross splits. Duplicate image content across
splits is rejected. Training uses train/val rows; test rows remain held out.
Use de-identified patient IDs and integer PNG masks, not RGB heatmaps. Apply any
EXIF rotation to image and mask together before import.

## Four-class inference

The registered Swin checkpoint has four output channels and uses `source-four`.
It cannot use the five-label training manifest or inherit those anatomical meanings.
The stored training configuration names `usdata/US_Dataset_80_20`, but it does
not identify a verified dataset release or split membership.

A capsule/region import must preserve region IDs before filling unlabelled
capsule pixels. Assigning a capsule class across the whole capsule afterward
would erase the internal labels. This overwrite was demonstrated on public
masks in the related [kidneyUS annotation format](https://github.com/rsingla92/kidneyUS).
That comparison does not establish checkpoint-dataset identity.

No training dataset is bundled. kidneyUS requires registration and review for
image access and has non-commercial use terms. Use the dataset’s own annotation
guide and permissions if you work with it. Do not transfer its anatomical names
to an unrelated checkpoint based on a similar file structure.

## Classifiers and samples

Classifier checkpoints do not record dataset identity, class names, output order
or training input settings. Their outputs remain neutral indices. Generated
images exercise import, geometry, inference and exports offline; they cannot
establish accuracy or label meaning.
