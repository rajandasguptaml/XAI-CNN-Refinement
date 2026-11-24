# Data card

No images are stored in this repository. The two datasets are public on Kaggle; download them and place them under
`data/images/` (or point `data.image_root` in `configs/default.yaml` elsewhere):

```
data/images/
├── msoud/Training/{glioma,meningioma,notumor,pituitary}/*.jpg
├── msoud/Testing/...
├── sartaj/Training/{glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor}/*.jpg
└── sartaj/Testing/...
```

| Name in the paper | Kaggle dataset | Images | Licence |
|---|---|---|---|
| Dataset-1 (Msoud) | [masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | 7,200 files; 6,997 after excluding 203 `*aug*` files added after the paper (the paper reports 7,023) | CC BY 4.0 |
| Dataset-2 (SARTAJ / NeuroMRI) | [sartajbhuvaji/brain-tumor-classification-mri](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) | 3,264 | MIT |

Msoud's own card states that it was assembled from the figshare dataset (Jun Cheng), SARTAJ and Br35H.

## `image_index.csv`

One row per image of both datasets (10,464 rows), taken from the
[Brain Tumor MRI Duplicate & Leakage Audit](https://www.kaggle.com/datasets/rajandasguptaml/brain-tumor-mri-duplicate-and-leakage-audit)
(CC BY 4.0), which hashes every image four ways and links duplicates.

| Column | Meaning |
|---|---|
| `image_id` | `<dataset>/<Split>/<class folder>/<file>`, the path under `data/images/` |
| `dataset`, `split`, `class_dir`, `label`, `filename` | origin of the image; `label` is one of glioma, meningioma, notumor, pituitary |
| `md5_pixels` | MD5 of the grayscale pixels (identical images share it) |
| `group_phash4` | duplicate group: images within perceptual-hash distance 4, i.e. the same scan re-saved, re-cropped or rescaled |
| `recommended_split` | group-aware 80/10/10 split of each dataset; no group straddles train, validation and test |
| `clean_test_original` | image of the original Testing folder with no duplicate in its dataset's Training folder |

## Why the leak-free sets matter

The two datasets overlap heavily, and each contains near-duplicate images within itself:

| Finding (audit, perceptual-hash distance <= 4) | Value |
|---|---|
| Msoud Testing images that also appear in Msoud Training | 714 of 1,600 (44.6%) |
| SARTAJ images that appear in Msoud Training | 76.1% |
| Distinct images across both datasets | 5,203 of 10,464 |

Applied to this repository's splits:

| Protocol | Dataset-1 test | of which unseen | Dataset-2 | of which unseen |
|---|---|---|---|---|
| `paper` (random 80/10/10 by image, as in the paper) | 699 | 367 | 3,264 | 653 |
| `clean` (group-aware 80/10/10) | 705 | 705 | 3,264 | 787 |

"Unseen" means no image of the same duplicate group, and no pixel-identical image, is in the training or validation
partition. The paper evaluated on the full sets, so its Dataset-2 result treats images that the model was trained on
as an unseen domain. This repository reports every number on both the full and the unseen sets. The unseen part of
Dataset-2 is small and dominated by glioma, so its macro F1 has wide uncertainty.

## Preprocessing

`xairefine/preprocess.py` follows Sec. 3.2: grayscale, Gaussian blur, threshold at 45, two erosions and two
dilations, largest contour, crop at its extreme points with a 5-pixel margin, rescale intensities to [0, 255], resize
to 224 x 224. All images are preprocessed once into `outputs/cache/images224.npy` (about 0.5 GB).
