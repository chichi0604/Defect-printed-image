Printed Defect Inspection – Classification & (Optional) Siamese Localization

> Drop-in README you can adapt after you paste your repo. All paths/flags are examples—replace with your actual ones.

---

## Overview

This project provides:

* **Image-level classification** of seven print defects: `banding`, `blur`, `color-shift`, `dropout`, `ghosting`, `misregistration`, `scratch`.
* A minimal **desktop GUI** for quick checks.
* **Synthetic data generation** and **augmentation** utilities to build training sets.
* (Optional) hooks for a **Siamese (gold vs. test) localization + multi-label** model if you later switch from pure classification.

Key ideas implemented in code:

* **Domain-partitioned evaluation** (appearance-cluster split) to reduce leakage.
* **Class rebalancing** via sampling/weights.
* **Label smoothing** for better calibration.
* **TTA (hflip)** at test time.

---

## Repo Structure (example)

```
.
├─ model9/                     # core classifier package
│  ├─ __init__.py
│  ├─ model.py                 # Net
│  ├─ dataset.py               # DefectDataset, CLASS_NAMES
│  ├─ preprocess.py            # I/O + transforms
│  ├─ train.py                 # training entry
│  ├─ test.py                  # batched inference
│  └─ print_matrice.py         # confusion matrix / report
│
├─ GUI/
│  └─ gui_check.py             # desktop GUI (Tkinter + optional Pillow)
│
├─ synth/                      # (optional) data generation utilities
│  ├─ generate.py              # orchestration (7 classes)
│  ├─ banding.py  ...          # per-defect synthesizers
│  └─ augment_print_env.py     # common augmentations
│
├─ dataset/                    # your actual data lives elsewhere; keep small samples here
├─ requirements.txt
└─ README.md
```

> If your tree differs, update paths below. Make sure `model9/` has an `__init__.py`.

---

## Installation

Tested with **Python 3.10/3.11**, **Windows 10/11**.

```powershell
# 1) Create env
conda create -n defect python=3.11 -y
conda activate defect

# 2) Install PyTorch (pick CUDA that matches your driver)
# Example (CUDA 12.x):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) Core deps
pip install -r requirements.txt
# If you don't have a requirements.txt yet:
pip install opencv-python pillow tqdm scikit-learn matplotlib pandas
```

---

## Data Preparation

**Flat directory layout** (no per-class subfolders). Labels are parsed from filename **prefixes** like `classprefix-xxxxx-*.png`.

Example:

```
G:\defect\dataset\dataset_resized\train\
  banding-000123-a.png
  blur-001245-x.png
  color-shift-000888.png
...
G:\defect\dataset\dataset_resized\val\
G:\defect\dataset\dataset_resized\test\
```

Adjust the parsing rule in `model9/dataset.py` if your naming differs.

> For synthetic data, see `synth/generate.py` (outputs into something like `G:\defect\dataset\output_box`).

---

## Quick Start

### 1) Train (image-level classifier)

Run the module so that relative imports work:

```powershell
# Example paths – change to yours
set TRAIN=G:\defect\dataset\dataset_resized\train
set VAL=G:\defect\dataset\dataset_resized\val

python -m model9.train --train_dir "%TRAIN%" --val_dir "%VAL%" \
  --epochs 60 --batch_size 64 --lr 3e-4 --weight_decay 1e-4 \
  --workers 8 --amp --save_dir G:\defect\model9\model
```

Common flags you might expose (edit to match your script):

* `--class_weights auto` or `--sampler weighted` (rebalancing)
* `--label_smoothing 0.02`
* `--seed 42`

### 2) Evaluate on test set

```powershell
set TEST=G:\defect\dataset\dataset_resized\test
python -m model9.test --data_dir "%TEST%" \
  --ckpt G:\defect\model9\model\best_cls.pth --tta hflip

# Confusion matrix / report
python -m model9.print_matrice --data_dir "%TEST%" --ckpt G:\defect\model9\model\best_cls.pth
```

### 3) Desktop GUI (quick check)

```powershell
python GUI\gui_check.py --ckpt G:\defect\model9\model\best_cls.pth
```

If Pillow is missing, the GUI still works but thumbnails/background are disabled.

---

## Synthetic Data (optional)

`/synth/generate.py` orchestrates 7-class synthesis. Tweak:

* `ORIGINAL_DIR` – source clean images ("gold" bases)
* `OUT_DIR` – output folder
* `PER_IMAGE_TOTAL` – how many samples per base image (evenly split per class)
* `SAVE_MASKS` – set `True` to save binary masks if available

Run:

```powershell
python synth\generate.py
```

---

## Troubleshooting

**Relative import error** like:

```
ImportError: attempted relative import with no known parent package
```

Fix by running **as a module** from the project root:

```powershell
python -m model9.print_matrice ...
```

Or ensure `model9/` contains `__init__.py` and your **working directory is the repo root**. As a last resort, set `PYTHONPATH` to the repo root before running.

**Checkpoint not found** in GUI

* Pass `--ckpt` explicitly or hardcode the path in the script (search for `best_cls.pth`).

**Flat labels don’t parse**

* Confirm your filename prefixes match `CLASS_NAMES` in `model9/dataset.py` and the regex used for parsing.

---

## Reproducibility

* Use a fixed random `--seed` and deterministic data splits.
* Keep a copy of the `val/test` splits used for all reported metrics.
* Document whether **TTA** was applied.

---

## Results (placeholder)

| Metric           | Val | Test |
| ---------------- | --- | ---- |
| Accuracy         |     |      |
| Macro-F1         |     |      |
| Blur R\@1        |     |      |
| Color-shift R\@1 |     |      |

Add confusion matrices/plots from `print_matrice.py` into the repo’s `docs/` or this README.

---

## Configuration Reference (example)

```yaml
# config.yaml (optional if you prefer config files)
seed: 42
num_classes: 7
img_size: 256
optimizer:
  name: AdamW
  lr: 3.0e-4
  weight_decay: 1.0e-4
scheduler:
  name: CosineAnnealing
  min_lr_ratio: 0.05
sampler:
  type: weighted  # or none
label_smoothing: 0.02
tta: hflip        # test-time only
```

---

## FAQ

**Q: Can this handle unseen defect types?**
A: The classifier is closed-set. Open-set handling needs an explicit rejection mechanism.

**Q: Can I switch to Siamese (gold vs. test) with localization?**
A: Yes—organize pairs and masks, then replace the head with a UNet-style decoder for masks and a multi-label head for types. Keep the training targets and losses consistent.

---

## License

Choose a license (e.g., MIT) and place it as `LICENSE` in the repo root.

---

## Citation

If this work supports a paper/thesis, add BibTeX here.

---

## Contact

* Maintainer: \<your name / email>
* Issues: please include OS, Python, PyTorch versions, exact command, and full traceback.

---

## Checklist (delete after editing)

* [ ] Paths updated (`G:\defect\...`)
* [ ] Confirmed `CLASS_NAMES` and filename prefixes
* [ ] Added exact train/eval commands and flags that your scripts actually support
* [ ] Inserted metrics/plots
* [ ] Picked a license
