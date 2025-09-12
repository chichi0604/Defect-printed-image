import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import DefectDataset, build_preprocess, CLASS_NAMES
from model.model import Net

# Path
DIR_OLD = r"G:\defect\dataset\dataset_resized\test"
CKPT    = r"G:\defect\model9\default\best_cls.pth"

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH   = 64
NUM_WORKERS = 4

# Save console output to txt; set to None if not needed
SAVE_LOG = None

class Tee:
    def __init__(self, fname):
        self.file = open(fname, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, x):
        self.stdout.write(x); self.file.write(x)
    def flush(self):
        self.stdout.flush(); self.file.flush()
    def close(self):
        try: self.file.close()
        except: pass

def _loader_for(root: str) -> DataLoader:
    ds = DefectDataset(root, split='val')
    return DataLoader(ds, batch_size=BATCH, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True), ds

@torch.no_grad()
def _infer_all(model: Net, loader: DataLoader):
    model.eval()
    all_logits = []
    all_labels = []
    all_paths  = []
    for x,y,paths in tqdm(loader, desc='[Test]', ncols=100):
        x = x.to(DEVICE, non_blocking=True).float()
        # Enable TTA
        logits1 = model(x)
        x2 = torch.flip(x, dims=[-1])
        logits = (logits1 + model(x2)) / 2.0
        # Disable TTA
        # logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y)
        all_paths += list(paths)
    return torch.cat(all_logits,0), torch.cat(all_labels,0), all_paths

# Calculate Macro-F1 from confusion matrix
def macro_f1_from_cm(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    # Safe division: set to 0 when denominator is 0
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec  = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1   = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    return float(f1.mean())

def _ensure_outdir() -> str:
    base_dir = os.path.dirname(CKPT) if os.path.isfile(CKPT) else os.getcwd()
    out_dir = os.path.join(base_dir, "TTA_visualisation")
    # out_dir = os.path.join(base_dir, "NO_TTA_visualisation")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _save_cm_csv(cm: np.ndarray, class_names, out_path: str):
    # First column: row labels; First row: column labels
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(class_names) + "\n")
        for i, cname in enumerate(class_names):
            row = [str(cm[i, j]) for j in range(len(class_names))]
            f.write(cname + "," + ",".join(row) + "\n")

def _plot_cm(cm: np.ndarray, class_names, title: str, out_path: str, normalize: bool=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if normalize:
        # Row normalization (proportion of each true class)
        row_sums = cm.sum(axis=1, keepdims=True)
        M = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums>0)
        fmt = ".2f"
    else:
        M = cm
        fmt = "d"

    K = len(class_names)
    # Adaptive figure size based on number of classes
    fig_w = max(8, 0.9 * K)
    fig_h = max(6, 0.9 * K)
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(M, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(K)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Annotate values in cells
    thresh = (M.max() / 2.0) if M.size > 0 else 0.5
    for i in range(K):
        for j in range(K):
            val = M[i, j]
            text_val = f"{val:{fmt}}"
            plt.text(j, i, text_val,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if val > thresh else "black",
                     fontsize=9)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _report(logits: torch.Tensor, labels: torch.Tensor, title: str):
    pred = logits.argmax(dim=1)
    correct = (pred == labels).sum().item()
    acc = correct / max(1, labels.numel())

    # Confusion matrix
    K = len(CLASS_NAMES)
    cm = np.zeros((K,K), dtype=np.int32)
    for t,p in zip(labels.tolist(), pred.tolist()):
        cm[t,p] += 1

    macro_f1 = macro_f1_from_cm(cm)

    print(f"[{title}] Accuracy: {acc:.4f}")
    print(f"[{title}] Macro-F1 (Scheme A): {macro_f1:.4f}\n")

    colw = max(14, max(len(c) for c in CLASS_NAMES)+2)
    header = " "*(colw)
    for c in CLASS_NAMES:
        header += f"{c:<{colw}}"
    print("Confusion Matrix ")
    print(header)
    for i, cname in enumerate(CLASS_NAMES):
        row = f"{cname:<{colw}}"
        for j in range(K):
            row += f"{cm[i,j]:<{colw}}"
        print(row)
    print("")

    # Save confusion matrix and heatmap
    out_dir = _ensure_outdir()
    csv_path = os.path.join(out_dir, f"cm_{title}.csv".replace(" ", "_"))
    png_raw  = os.path.join(out_dir, f"cm_heatmap_{title}.png".replace(" ", "_"))
    png_norm = os.path.join(out_dir, f"cm_heatmap_norm_{title}.png".replace(" ", "_"))

    _save_cm_csv(cm, CLASS_NAMES, csv_path)

    _plot_cm(cm, CLASS_NAMES, "Confusion Matrix", png_raw, normalize=False)
    _plot_cm(cm, CLASS_NAMES, "Row-normalized Confusion Matrix", png_norm, normalize=True)

    print(f"[Saved] Confusion matrix CSV: {csv_path}")
    print(f"[Saved] Heatmap (raw counts): {png_raw}")
    print(f"[Saved] Heatmap (row-normalized): {png_norm}\n")

def main():
    # Load model
    model = Net(pretrained=False).to(DEVICE)
    assert os.path.isfile(CKPT), f"Weight file not found: {CKPT}"
    sd = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(sd, strict=True)

    # Run test set
    if os.path.isdir(DIR_OLD):
        loader, ds = _loader_for(DIR_OLD)
        print(f"\n File count: {len(ds)} | Directory: {DIR_OLD}")
        print(f"[Class order] {CLASS_NAMES}\n")
        logits, labels, _ = _infer_all(model, loader)
        _report(logits, labels, "Test Set")

if __name__ == "__main__":
    _tee = None
    try:
        if SAVE_LOG:
            _tee = Tee(SAVE_LOG)
            sys.stdout = _tee
        main()
    finally:
        if _tee:
            sys.stdout = _tee.stdout
            _tee.close()