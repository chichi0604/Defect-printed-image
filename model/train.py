import os, random, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import DefectDataset, CLASS_NAMES
from model import Net

# Path
TRAIN_DIR = r"G:\defect\dataset\dataset_resized\train"
VAL_DIR   = r"G:\defect\dataset\dataset_resized\val"
SAVE_DIR  = r"/model/default"

# Hyperparameters
SEED         = 2025
BATCH_SIZE   = 64
EPOCHS       = 40
LR           = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
PIN_MEMORY   = True
LABEL_SMOOTH = 0.02
PATIENCE     = 8

os.makedirs(SAVE_DIR, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _class_counts(ds):
    cnt = np.zeros((len(CLASS_NAMES),), dtype=np.int64)
    for _, y, _ in ds.samples:
        cnt[y] += 1
    return cnt

def build_loaders():
    train_ds = DefectDataset(TRAIN_DIR, split='train')
    val_ds   = DefectDataset(VAL_DIR,   split='val')

    # Simple class frequency weighted sampling
    counts = _class_counts(train_ds).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    w = inv / np.median(inv[inv > 0])
    w = np.clip(w, 0.5, 4.0)
    weights = np.array([w[y] for _, y, _ in train_ds.samples], dtype=np.float32)

    sampler = WeightedRandomSampler(weights=torch.from_numpy(weights),
                                    num_samples=len(weights),
                                    replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False, persistent_workers=True)

    print(f"[Data] train={len(train_ds)} | val={len(val_ds)} | classes={CLASS_NAMES}")
    print(f"[Data] train class counts = {counts.tolist()}")
    return train_loader, val_loader

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, cor, loss_sum = 0, 0, 0.0
    for x, y, _ in tqdm(loader, desc="[Val  ]", ncols=98):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        cor += (pred == y).sum().item()
        tot += x.size(0)
    return loss_sum / max(tot, 1), (cor / max(tot, 1))

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    tot, cor, loss_sum = 0, 0, 0.0
    for x, y, _ in tqdm(loader, desc="[Train]", ncols=98):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(True):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTH)

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

    train_loader, val_loader = build_loaders()


    model = Net(pretrained=True).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.05)
    scaler = torch.cuda.amp.GradScaler()

    best_acc, best_ep, bad = 0.0, -1, 0

    with open(os.path.join(SAVE_DIR, "class_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES))

    for epoch in range(1, EPOCHS + 1):
        print(f"\n Epoch {epoch:02d}/{EPOCHS}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
        va_loss, va_acc = evaluate(model, val_loader, device)

        def fmt(l, a): return f"loss={l:.4f} | acc={a*100:.2f}%"
        print(f"[Train] {fmt(tr_loss, tr_acc)}")
        print(f"[Val  ] {fmt(va_loss, va_acc)}")

        # 保存
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last_cls.pth"))
        if va_acc > best_acc:
            best_acc, best_ep, bad = va_acc, epoch, 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_cls.pth"))
            print(f" saved best (val_acc={best_acc*100:.2f}% @ epoch {best_ep})")
        else:
            bad += 1
            print(f" no improve ({bad}/{PATIENCE}) | best={best_acc*100:.2f}% @ {best_ep}")

        scheduler.step()
        if bad >= PATIENCE:
            print(f"[early-stop] best={best_acc*100:.2f}% @ epoch {best_ep}")
            break

    print(f"[done] best={best_acc*100:.2f}% @ epoch {best_ep} | weights -> {SAVE_DIR}")

if __name__ == "__main__":
    main()
