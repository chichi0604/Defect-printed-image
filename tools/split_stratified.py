from pathlib import Path
import re, shutil, random
from collections import defaultdict, Counter

SRC_DIR   = Path(r"G:\defect\dataset\resized")
OUT_ROOT  = Path(r"G:\defect\dataset\dataset_resized_Stratified")
TRAIN_DIR = OUT_ROOT / "train"
VAL_DIR   = OUT_ROOT / "val"
TEST_DIR  = OUT_ROOT / "test"

GROUP_SIZE = 150
SEED       = 2025
COPY_MODE  = True       
STRICT_CHECK = True     

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PAT = re.compile(r"^(?P<cls>.+)-(?P<num>\d{6})(?P<suffix>-[^.]+)(?P<ext>\.\w+)$")

for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
    d.mkdir(parents=True, exist_ok=True)

def base_id_from_number(n: int) -> int:
    return n // GROUP_SIZE

def put_file(src: Path, dst_dir: Path):
    dst = dst_dir / src.name
    if dst.exists():
        dst.unlink()
    if COPY_MODE:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))

groups = defaultdict(list)
skipped = 0
bad_examples = []

for f in SRC_DIR.rglob("*"):            
    if not f.is_file() or f.suffix.lower() not in ALLOWED_EXTS:
        continue
    m = PAT.match(f.name)
    if not m:
        skipped += 1
        if len(bad_examples) < 5:
            bad_examples.append(f.name)
        continue
    cls = m.group("cls")
    num = int(m.group("num"))
    bid = base_id_from_number(num)
    groups[(cls, bid)].append(f)

print(f"Found {len(groups)} (class, base image) groups; skipped {skipped} unmatched files.")
if bad_examples:
    print("Example unmatched files (max 5):", bad_examples)

rng = random.Random(SEED)
stats_split = Counter()
stats_cls_split = defaultdict(Counter)
bad_groups = []

for (cls, bid), files in sorted(groups.items()):
    if STRICT_CHECK and len(files) != GROUP_SIZE:
        bad_groups.append(((cls, bid), len(files)))
        continue

    rng.shuffle(files)
    train_files = files[:120]
    val_files   = files[120:135]
    test_files  = files[135:150]

    for p in train_files:
        put_file(p, TRAIN_DIR)
        stats_split["train"] += 1
        stats_cls_split["train"][cls] += 1
    for p in val_files:
        put_file(p, VAL_DIR)
        stats_split["val"] += 1
        stats_cls_split["val"][cls] += 1
    for p in test_files:
        put_file(p, TEST_DIR)
        stats_split["test"] += 1
        stats_cls_split["test"][cls] += 1

print("\n Distribution Complete ")
print(f"train: {stats_split['train']}  val: {stats_split['val']}  test: {stats_split['test']}")
for split in ("train","val","test"):
    print(f"{split} per class count:", dict(stats_cls_split[split]))

if bad_groups:
    print("\n[Warning] The following (class, base image) groups have sample count ≠150, skipped:")
    for (cls, bid), n in bad_groups:
        print(f"  ({cls}, base {bid}) -> {n}")
