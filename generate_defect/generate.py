from pathlib import Path
import re, cv2, random
from tqdm import tqdm

from banding import banding
from blur import blur
from color_shift import color_shift
from dropout import dropout
from ghosting import ghosting
from misregistration import misregistration
from scratch import scratch

# Paths
ORIGINAL_DIR = Path(r"G:\defect\dataset\original_1024")
OUT_DIR      = Path(r"G:\defect\dataset\output_box")
SEED = 42
PER_IMAGE_TOTAL = 350
RESET_THIS_RUN  = False
SAVE_MASKS      = False

random.seed(SEED)


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


DEFECTS = {
    "banding":         banding,
    "blur":            blur,
    "color-shift":     color_shift,
    "dropout":         dropout,
    "ghosting":        ghosting,
    "misregistration": misregistration,
    "scratch":         scratch,
}

_num_re = re.compile(r".*[-_](\d+)\.png$", re.I)

def ensure_dirs():
    for cls in CLASSES:
        (OUT_DIR/cls/"images").mkdir(parents=True, exist_ok=True)
        if SAVE_MASKS:
            (OUT_DIR/cls/"masks").mkdir(parents=True, exist_ok=True)

def _variant_dirs_for_class(cls: str, kind: str):
    dirs = []
    for alias in ALIASES.get(cls, [cls]):
        dirs.append(OUT_DIR/alias/kind)
    return dirs

def next_idx_for_class(cls: str):
    mx = -1
    for d in _variant_dirs_for_class(cls, "images"):
        if not d.exists():
            continue
        for p in d.glob("*.png"):
            m = _num_re.match(p.name)
            if m:
                mx = max(mx, int(m.group(1)))
    return mx + 1

def per_class_quota(total, ncls):
    base = total // ncls
    rest = total - base * ncls
    q = [base]*ncls
    for i in range(rest):
        q[i] += 1
    return q

def _list_images(root: Path):
    imgs = []
    for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp"):
        imgs += list(root.glob(ext))
    return sorted(imgs)

def main():
    imgs = _list_images(ORIGINAL_DIR)
    assert len(imgs) > 0, f"original directory is empty: {ORIGINAL_DIR}"

    ensure_dirs()
    if RESET_THIS_RUN:
        for cls in CLASSES:
            for p in (OUT_DIR/cls/"images").glob("*.png"): p.unlink()
            if SAVE_MASKS:
                for p in (OUT_DIR/cls/"masks").glob("*.png"): p.unlink()

    next_id = {cls: next_idx_for_class(cls) for cls in CLASSES}
    quotas  = per_class_quota(PER_IMAGE_TOTAL, len(CLASSES))

    total_to_gen = len(imgs) * PER_IMAGE_TOTAL
    bar = tqdm(total=total_to_gen, desc="Gen", unit="img", dynamic_ncols=True,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

    for p in imgs:
        src = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if src is None:
            tqdm.write(f"[WARN] Failed to read: {p}")
            continue

        for cls, cnt in zip(CLASSES, quotas):
            func = DEFECTS[cls]
            for i in range(cnt):
                try:
                    img_def, mask, _ = func(src)
                    idx = next_id[cls]

                    # Output file: use hyphen naming for both directory and filename
                    fname = f"{cls}-{idx:06d}.png"
                    cv2.imwrite(str((OUT_DIR/cls/"images"/fname)), img_def)
                    if SAVE_MASKS:
                        cv2.imwrite(str((OUT_DIR/cls/"masks"/fname)), mask)
                    next_id[cls] += 1
                except Exception as e:
                    tqdm.write(f"[WARN] {cls} on {p.name} failed: {e}")
                finally:
                    bar.set_postfix_str(f"{p.name} | {cls} {i+1}/{cnt}")
                    bar.update(1)

    bar.close()
    tqdm.write(f"Done -> {OUT_DIR}")

if __name__ == "__main__":
    main()
