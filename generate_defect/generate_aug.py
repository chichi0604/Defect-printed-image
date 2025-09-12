from pathlib import Path
import cv2, random, numpy as np, re
from tqdm import tqdm

IN_DIR  = Path(r"G:\defect\dataset\output_box")
OUT_DIR = Path(r"G:\defect\dataset\output_box_aug")
SEED = 42
random.seed(SEED); np.random.seed(SEED)

AUG_PER_IMAGE = 2
COPY_ORIGINAL = True


CLASSES = ["banding","blur","color-shift","dropout","ghosting","misregistration","scratch"]

ALIASES = {
    "banding":         ["banding"],
    "blur":            ["blur"],
    "color-shift":     ["color-shift","color_shift"],
    "dropout":         ["dropout"],
    "ghosting":        ["ghosting"],
    "misregistration": ["misregistration"],
    "scratch":         ["scratch"],
}

def _list_src_files_for_class(cls: str):
    files = []
    for alias in ALIASES.get(cls,[cls]):
        d = IN_DIR/alias/"images"
        if d.exists():
            files += sorted(d.glob("*.png"))
    return files

# Horizontal flip with 50% probability
def aug_flip(img):
    if random.random() < 0.5:
        return cv2.flip(img, 1)
    return img

# Mild brightness/contrast jitter
def aug_bc(img):
    alpha = random.uniform(0.92, 1.10)
    beta  = random.randint(-8, 8)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# Light JPEG re-encode to introduce mild compression artifacts
def aug_jpeg(img):
    q = random.randint(90, 100)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img

AUG_FUNCS = [aug_flip, aug_bc, aug_jpeg]

_num_re = re.compile(r".*[-_](\d+)(?:[-_].*)?\.png$", re.I)

def next_idx(dst_dir: Path) -> int:
    mx = -1
    if dst_dir.exists():
        for p in dst_dir.glob("*.png"):
            m = _num_re.match(p.name)
            if m:
                mx = max(mx, int(m.group(1)))
    return mx + 1

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        src_files = _list_src_files_for_class(cls)
        if not src_files:
            print(f"[{cls}] No source images found (neither new nor legacy naming). Skipped.")
            continue

        dst_dir = OUT_DIR/cls/"images"     # Write into the new (hyphen) naming directory
        dst_dir.mkdir(parents=True, exist_ok=True)
        cur = next_idx(dst_dir)

        for p in tqdm(src_files, desc=cls, leave=False):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue

            if COPY_ORIGINAL:
                fname = f"{cls}-{cur:06d}-ori.png"; cur += 1
                cv2.imwrite(str(dst_dir/fname), img)

            k = min(AUG_PER_IMAGE, len(AUG_FUNCS))
            for f in random.sample(AUG_FUNCS, k=k):
                out = f(img)
                fname = f"{cls}-{cur:06d}-aug.png"; cur += 1
                cv2.imwrite(str(dst_dir/fname), out)

        print(f"[{cls}] Processed {len(src_files)} images -> now {len(list(dst_dir.glob('*.png')))} files in folder")

    print(" Aug done ->", OUT_DIR)

if __name__ == "__main__":
    main()