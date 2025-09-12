from pathlib import Path
import shutil

SRC_ROOT = Path(r"G:\defect\dataset\dataset_resized")
SRC_DIRS = ["test", "train", "val"]
DST_DIR  = Path(r"G:\defect\dataset\resized")

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DST_DIR.mkdir(parents=True, exist_ok=True)

def unique_target(dst_dir: Path, name: str) -> Path:
    base, ext = Path(name).stem, Path(name).suffix
    candidate = dst_dir / f"{base}{ext}"
    i = 1
    while candidate.exists():
        candidate = dst_dir / f"{base}__dup{i}{ext}"
        i += 1
    return candidate

copied = 0
skipped = 0
for d in SRC_DIRS:
    src = SRC_ROOT / d
    if not src.exists():
        continue
    for f in src.rglob("*"):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTS:
            target = unique_target(DST_DIR, f.name)
            shutil.copy2(f, target)
            copied += 1
        else:
            skipped += 1

print(f"Done: copied {copied} image files to {DST_DIR}")
