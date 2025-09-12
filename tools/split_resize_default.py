from pathlib import Path
import cv2, numpy as np, random
from collections import Counter, defaultdict

SEED = 42
random.seed(SEED); np.random.seed(SEED)

SRC_DIR  = r"/dataset/output_box_aug"
DST_ROOT = r"/dataset/dataset_resized"

CLASSES = ["banding","blur","color-shift","dropout","ghosting","misregistration","scratch"]
IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"}

K_DOMAINS    = 20
KMEANS_ITERS = 20
BATCH        = 4096

OUT_W, OUT_H = 256, 256  # Stretch to 256 x 256

def _canon_name(name: str) -> str:
    return name.lower().replace("_","-").strip()

def _starts_with_any(base: str, stem: str) -> bool:
    return base.startswith(stem+"-") or base.startswith(stem+"_") or base.startswith(stem+".")

def parse_class_by_any(path: Path):
    base = _canon_name(path.name)
    for cls in sorted(CLASSES, key=len, reverse=True):
        if _starts_with_any(base, cls):
            return cls
    parts = [_canon_name(p) for p in path.parts]
    for cls in CLASSES:
        if cls in parts or cls.replace("-","_") in parts:
            return cls
    tok = base.split("-")[0].split("_")[0].split(".")[0]
    tok = _canon_name(tok)
    return tok if tok in CLASSES else None

def is_img(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS

def read_bgr_cv(fp: str):
    return cv2.imread(fp, cv2.IMREAD_COLOR)

def write_png_cv(fp: str, img):
    Path(fp).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(fp, img)  # output extension guaranteed to be .png

# Fingerprint: strong smoothing + dHash → 0/1 vector
def dhash_bits(img_bgr, size=16, blur_ks=7):
    if blur_ks>0:
        img_bgr = cv2.GaussianBlur(img_bgr, (blur_ks|1, blur_ks|1), 0)
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size+1, size), interpolation=cv2.INTER_AREA)
    diff = g[:,1:] > g[:,:-1]
    return diff.astype(np.uint8).reshape(-1)


def collect_images(root_dir: str):
    p = Path(root_dir); assert p.exists(), f"{root_dir} does not exist"
    return [x for x in p.rglob("*") if is_img(x)]

# Kmeans (binary vector; Euclidean ≈ Hamming)
def kmeans_binary(X, K, iters=20, batch=BATCH, seed=SEED):
    N, D = X.shape
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, K, replace=False)
    C = X[idx].astype(np.float32)
    labels = np.zeros(N, dtype=np.int32)
    for _ in range(iters):
        for s in range(0, N, batch):
            Xb = X[s:s+batch].astype(np.float32)
            x2 = (Xb*Xb).sum(1, keepdims=True)
            c2 = (C*C).sum(1)[None,:]
            d2 = x2 - 2.0*(Xb @ C.T) + c2
            labels[s:s+batch] = d2.argmin(1)
        newC = np.zeros_like(C)
        for k in range(K):
            idxk = np.where(labels==k)[0]
            if len(idxk)==0:
                j = rng.randint(0, N); newC[k] = X[j]
            else:
                newC[k] = X[idxk].mean(0)
        C = newC
    return labels, C

# Whole domains not split + class balance (greedy + swap fine-tuning)
def balanced_assign(cluster_cls_counts, caps, targets, seed=SEED):
    K, C = cluster_cls_counts.shape
    S = len(caps)
    def cost(alloc):
        diff = alloc - targets
        return float((diff*diff).sum())
    order = np.argsort(-cluster_cls_counts.sum(axis=1))
    assign = -np.ones(K, dtype=int)
    alloc  = np.zeros((S, C), dtype=np.int64)
    used   = np.zeros(S, dtype=int)
    for k in order:
        best_s, best_cost = None, None
        for s in range(S):
            if used[s] >= caps[s]: continue
            a_try = alloc.copy()
            a_try[s] += cluster_cls_counts[k]
            c = cost(a_try)
            if best_cost is None or c < best_cost:
                best_cost, best_s = c, s
        if best_s is None:
            best_s = int(np.argmin(used - np.array(caps)))
            best_s = max(0, min(S-1, best_s))
        assign[k] = best_s
        alloc[best_s] += cluster_cls_counts[k]
        used[best_s]  += 1
    improved = True
    while improved:
        improved = False
        base_cost = cost(alloc)
        for i in range(K):
            for j in range(i+1, K):
                si, sj = assign[i], assign[j]
                if si == sj: continue
                a_try = alloc.copy()
                a_try[si] -= cluster_cls_counts[i]; a_try[sj] -= cluster_cls_counts[j]
                a_try[si] += cluster_cls_counts[j]; a_try[sj] += cluster_cls_counts[i]
                if cost(a_try) + 1e-9 < base_cost:
                    alloc = a_try
                    assign[i], assign[j] = sj, si
                    base_cost = cost(alloc)
                    improved = True
    return assign, alloc

def main():
    # Scan + class parsing + fingerprint
    files = collect_images(SRC_DIR)
    print(f"[INFO] Scanned {len(files)} images")
    feats, labels_cls, kept_files = [], [], []
    skipped = 0
    for p in files:
        cls = parse_class_by_any(p)
        if cls is None:
            skipped += 1; continue
        img = read_bgr_cv(str(p))
        if img is None:
            skipped += 1; continue
        feats.append(dhash_bits(img, size=16, blur_ks=7))
        labels_cls.append(cls)
        kept_files.append(str(p))
    if skipped:
        print(f"[WARN] Skipped {skipped} images (unrecognized class or read failure)")

    X = np.asarray(feats, dtype=np.uint8)
    print(f"[INFO] Feature extraction complete: N={X.shape[0]}, D={X.shape[1]}")

    labs, _ = kmeans_binary(X, K=K_DOMAINS, iters=KMEANS_ITERS, batch=BATCH, seed=SEED)
    cnt = Counter(labs.tolist())
    print(f"[INFO] Feature extraction complete: N={X.shape[0]}, D={X.shape[1]}")
    for k in range(K_DOMAINS):
        print(f"  cluster#{k:02d}: {cnt[k]}")

    # Enumeration of domains × classes
    cls_index = {c:i for i,c in enumerate(CLASSES)}
    K = K_DOMAINS; C = len(CLASSES)
    cluster_cls_counts = np.zeros((K, C), dtype=np.int64)
    for i, c in enumerate(labels_cls):
        cluster_cls_counts[labs[i], cls_index[c]] += 1

    total_per_cls = cluster_cls_counts.sum(axis=0)
    tgt_train = np.rint(total_per_cls * 0.80).astype(np.int64)
    tgt_val   = np.rint(total_per_cls * 0.10).astype(np.int64)
    tgt_test  = total_per_cls - tgt_train - tgt_val
    targets = np.stack([tgt_train, tgt_val, tgt_test], axis=0)

    caps = np.array([16,2,2], dtype=int)
    assert caps.sum() == K

    assign, alloc = balanced_assign(cluster_cls_counts, caps, targets, seed=SEED)

    print("\n Target quotas (three rows=train/val/test; columns=seven classes)")
    print(targets)
    print("\n Actual allocation (by class count)")
    print(alloc)
    print("\n Deviation (alloc - targets)")
    print(alloc - targets)


    for sp in ("train", "val", "test"):
        (Path(DST_ROOT) / sp).mkdir(parents=True, exist_ok=True)

    def split_of_cluster(k):
        return ("train", "val", "test")[assign[k]]


    def ensure_prefixed_name(c: str, base_name: str) -> str:
        stem = Path(base_name).stem
        cn = c
        if stem.lower().startswith((cn + "-", cn + "_", cn + ".")):
            new_stem = stem
        else:
            new_stem = f"{c}-{stem}"
        return new_stem + ".png"

    for i, fp in enumerate(kept_files):
        c = labels_cls[i]
        k = int(labs[i])
        sp = split_of_cluster(k)

        img = read_bgr_cv(fp)
        if img is None:
            print(f"[READ FAIL] {fp}")
            continue
        img_resized = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

        out_name = ensure_prefixed_name(c, Path(fp).name)
        dst = str(Path(DST_ROOT) / sp / out_name)
        write_png_cv(dst, img_resized)


    per_split_cls = {"train": Counter(), "val": Counter(), "test": Counter()}
    for i, fp in enumerate(kept_files):
        sp = split_of_cluster(int(labs[i]))
        per_split_cls[sp][labels_cls[i]] += 1

    print("\n Image counts by class in each split ")
    for sp in ("train", "val", "test"):
        print(sp, dict(per_split_cls[sp]))

    inv = defaultdict(set)
    with open(Path(DST_ROOT) / "manifest.csv", "w", encoding="utf-8") as f:
        f.write("path,class,cluster,split\n")
        for i, fp in enumerate(kept_files):
            c = labels_cls[i];
            k = int(labs[i]);
            sp = split_of_cluster(k)
            inv[k].add(sp)
            f.write(f"{fp},{c},{k},{sp}\n")
    bad = [k for k, v in inv.items() if len(v) > 1]
    print("\nCross-split clusters:", bad)
    print("\n Done ->", DST_ROOT)



if __name__ == "__main__":
    main()
