Printed Defect Inspection – Classification & (Optional) Siamese Localization

---

## Overview

This project provides:

* **Image-level classification** of seven print defects: `banding`, `blur`, `color-shift`, `dropout`, `ghosting`, `misregistration`, `scratch`.
* A minimal **desktop GUI** for quick checks.
* **Synthetic data generation** and **augmentation** utilities to build training sets.


---

## Structure

```
.
├─ model/                      # core classifier package
│  ├─ model.py                 # Net
│  ├─ dataset.py               # DefectDataset, CLASS_NAMES
│  ├─ preprocess.py            # I/O + transforms
│  ├─ train.py                 # training entry
│  ├─ train_type.py            # training different models
│  └─ print_matrice.py         # confusion matrix / report
│
├─ GUI/
│  └─ gui.py                   # desktop GUI (Tkinter + optional Pillow)
│
├─ generate_defect/                      # (optional) data generation utilities
│  ├─ generate.py              # orchestration (7 classes)
│  ├─ banding.py  ...          # per-defect synthesizers
│  └─ generate_aug.py          # common augmentations
│
├─ dataset/                    
└─ README.md
