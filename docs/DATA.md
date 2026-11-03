# Data and label contract

The recovered project expected images plus separate `capsule` and `regions`
masks. The capsule array in the saved analysis output covers the whole kidney;
it is not a thin outer rim. The regions array contains IDs 0, 1, 2 and 3.

The old combination code first wrote region labels, then replaced every pixel
inside the capsule with class 3. A supplied analysis plot shows the resulting
combined mask containing only 0 and 3. The current import preserves region IDs
and fills only the remaining capsule interior with ID 4:

| ID | Meaning in this repository |
| --- | --- |
| 0 | Background |
| 1 | Original region 1 |
| 2 | Original region 2 |
| 3 | Original region 3 |
| 4 | Capsule interior without a region label |

Names stay neutral until the dataset's actual annotation guide establishes
them. The two original scripts disagreed about whether a region meant sinus,
parenchyma or cortex. An attractive legend is not evidence that a label is right.

```bash
ckd import-triplets --images source/images --capsules source/masks/capsule \
  --regions source/masks/regions --output data/imported
```

This copies images to PNG and writes combined masks without resizing or
overwriting source files. Region IDs outside the capsule, missing masks and
mismatched shapes fail. Make a manifest beside the imported folders:

```csv
split,patient_id,image,mask
train,patient-001,images/sample-001.png,masks/sample-001.png
val,patient-002,images/sample-002.png,masks/sample-002.png
test,patient-003,images/sample-003.png,masks/sample-003.png
```

Paths must remain inside the manifest directory. Choose splits at patient or
source-group level before experiments; do not put related frames on both sides.
Duplicate image content across splits is also rejected. Training uses only the
listed train/val rows; test rows are held out. Keep patient IDs de-identified.
PNG masks must hold integer IDs, not a rendered RGB heatmap. Normalize any
EXIF orientation for the image and mask together first; the workbench rejects
implicit image-only rotations that could misalign a pair.

## Original data recovery

No original image dataset or source annotation CSV was included. The trained
four-class checkpoint was supplied separately on 3 November 2026. The Windows config refers to a local `Normal` folder and the
training script to a generic Kaggle `US_Dataset_prepared` path. Those paths do
not identify a verified dataset release.

The structure resembles the authors' [Open Kidney Ultrasound Data Set](https://github.com/rsingla92/kidneyUS).
That is a possible lead, **not a confirmed attribution**. Its official page
requires registration and manual review before image access and restricts use
to non-commercial purposes. No gated images or mirror downloads are bundled
here. Its published benchmark weights are not the separately supplied custom Swin model.
Use the authors' annotation guide and permissions if that source is confirmed.

The generated sample has no patients and no scientific validation value. It
exists to exercise the import, geometry, export, UI and test flows offline.


## Supplied checkpoint versus candidate annotations

The separately supplied Swin file has four outputs and a saved Kaggle path
`usdata/US_Dataset_80_20`. This is a stronger source-code link; it still does not
identify the creator, exact images or split membership.

The candidate kidneyUS source fills the capsule as a binary mask and encodes
regions as CEC 1, medulla 2 and cortex 3. The recovered combination logic overwrites
almost all those internal pixels when run on public mask pairs from that
format. This confirms a defect on that matching format; it does not prove the
checkpoint used those particular masks. The supplied model’s four IDs remain
neutral rather than borrowing the candidate’s anatomical names.

Classifier weights contain no dataset or class-name metadata. No gated image
source is bypassed or redistributed. Generated profile probes remain software
experiments; an unlabelled input cannot establish class order or accuracy.
