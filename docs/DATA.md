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

No original image dataset, source annotation CSV or trained kidney checkpoint
was included. The Windows config refers to a local `Normal` folder and the
training script to a generic Kaggle `US_Dataset_prepared` path. Those paths do
not identify a verified dataset release.

The structure resembles the authors' [Open Kidney Ultrasound Data Set](https://github.com/rsingla92/kidneyUS).
That is a possible lead, **not a confirmed attribution**. Its official page
requires registration and manual review before image access and restricts use
to non-commercial purposes. No gated images or mirror downloads are bundled
here. Its published benchmark weights are not the missing custom Swin model.
Use the authors' annotation guide and permissions if that source is confirmed.

The generated sample has no patients and no scientific validation value. It
exists to exercise the import, geometry, export, UI and test flows offline.
