import os, random, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import DefectDataset, CLASS_NAMES
from model import Net

EXP_MODE = "CE"      # Change to "CE+W" to switch to weighted cross entropy

# Path
TRAIN_DIR = r"G:\defect\dataset\dataset_resized\train"
VAL_DIR   = r"G:\defect\dataset\dataset_resized\val"
SAVE_ROOT = r"G:\defect\model9\type_runs"
SAVE_DIR  = os.path.join(SAVE_ROOT, EXP_MODE.replace("+","_"))
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
SEED         = 2025
BATCH_SIZE   = 64
EPOCHS       = 40
LR           = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
PIN_MEMORY   = True
PATIENCE     = 8

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def _labels_from_ds(ds):
    if hasattr(ds, "samples"):
        return np.asarray([int(y) for _, y, _ in ds.samples], dtype=np.int64)
    ys = []
    for i in range(len(ds)):
        _, y, _ = ds[i]
        ys.append(int(y))
    return np.asarray(ys, dtype=np.int64)

def compute_class_weights(train_ds, K):
    labels = _labels_from_ds(train_ds)
    cnt = np.bincount(labels, minlength=K).astype(np.float32)

    # 1/sqrt(n_c)
    w = 1.0 / np.sqrt(np.maximum(cnt, 1.0))

    # Median normalization (does not change overall loss scale)
    med = np.median(w[w > 0])
    if med > 0:
        w = w / med

    # Clip range
    w = np.clip(w, 0.5, 4.0)

    return w, cnt

def build_loaders():
    train_ds = DefectDataset(TRAIN_DIR, split='train')
    val_ds   = DefectDataset(VAL_DIR,   split='val')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              drop_last=False, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              drop_last=False, persistent_workers=True)

    # Count class distribution & weights (only for CE+W)
    w, cnt = compute_class_weights(train_ds, K=len(CLASS_NAMES))

    print(f"[Data] train={len(train_ds)} | val={len(val_ds)} | classes={CLASS_NAMES}")
    print(f"[Data] train class counts = {cnt.tolist()}")
    if EXP_MODE == "CE":
        print(f"[Data] class weights (1/sqrt(n), clip[0.5,4.0]) = {w.round(3).tolist()}")

    return train_loader, val_loader, w

@torch.no_grad()
def evaluate(model, loader, device, ce_weight=None):
    model.eval()
    tot, cor, loss_sum = 0, 0, 0.0
    for x, y, _ in tqdm(loader, desc="[Val  ]", ncols=98):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        logits = model(x)
        # Validation without weighting for objective evaluation
        loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        cor += (pred == y).sum().item()
        tot += x.size(0)
    return loss_sum / max(tot, 1), (cor / max(tot, 1))

def train_one_epoch(model, loader, optimizer, scaler, device, ce_weight=None):
    model.train()
    tot, cor, loss_sum = 0, 0, 0.0
    for x, y, _ in tqdm(loader, desc="[Train]", ncols=98):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(True):
            logits = model(x)

            loss = F.cross_entropy(logits, y, weight=ce_weight)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        cor += (pred == y).sum().item()
        tot += x.size(0)

    return loss_sum / max(tot, 1), (cor / max(tot, 1))

def main():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    print(f"[Mode] {EXP_MODE}")

    train_loader, val_loader, w_np = build_loaders()
    ce_weight = torch.tensor(w_np, dtype=torch.float32, device=device) if EXP_MODE == "CE+W" else None

    model = Net(pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.05)
    scaler = torch.cuda.amp.GradScaler()

    best_acc, best_ep, bad = 0.0, -1, 0

    with open(os.path.join(SAVE_DIR, "class_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES))

    for epoch in range(1, EPOCHS+1):
        print(f"\n Epoch {epoch:02d}/{EPOCHS} ")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, ce_weight)
        va_loss, va_acc = evaluate(model, val_loader, device)

        def fmt(l, a): return f"loss={l:.4f} | acc={a*100:.2f}%"
        print(f"[Train] {fmt(tr_loss, tr_acc)}")
        print(f"[Val  ] {fmt(va_loss, va_acc)}")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last_cls.pth"))
        if va_acc > best_acc:
            best_acc, best_ep, bad = va_acc, epoch, 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_cls.pth"))
            print(f" saved best (val_acc={best_acc*100:.2f}% @ epoch {best_ep})")
        else:
            bad += 1
            print(f" no improvement ({bad}/{PATIENCE}) | best={best_acc*100:.2f}% @ {best_ep}")

        scheduler.step()
        if bad >= PATIENCE:
            print(f"[early-stop] best={best_acc*100:.2f}% @ epoch {best_ep}")
            break

    print(f"[done] {EXP_MODE} | best={best_acc*100:.2f}% @ epoch {best_ep} | -> {SAVE_DIR}")

if __name__ == "__main__":
    main()