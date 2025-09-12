from pathlib import Path
import re, shutil, random
from collections import defaultdict, Counter

SRC_DIR   = Path(r"G:\defect\dataset\resized")
OUT_ROOT  = Path(r"G:\defect\dataset\dataset_resized_template")
TRAIN_DIR = OUT_ROOT / "train"
VAL_DIR   = OUT_ROOT / "val"
TEST_DIR  = OUT_ROOT / "test"

GROUP_SIZE  = 150
SEED        = 2025
COPY_MODE   = False
CLEAR_DEST  = True
STRICT_150  = True

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PAT = re.compile(r"^(?P<cls>.+)-(?P<num>\d{6})(?P<suffix>-[^.]+)(?P<ext>\.\w+)$")

for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
    d.mkdir(parents=True, exist_ok=True)
if CLEAR_DEST:
    for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        for f in d.glob("*"):
            if f.is_file():
                f.unlink()

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

bases_all_files = defaultdict(list)
per_cls_base = defaultdict(lambda: defaultdict(list))
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
    bases_all_files[bid].append(f)
    per_cls_base[cls][bid].append(f)

if skipped:
    print(f"[Info] Unmatched/ignored files: {skipped}, examples: {bad_examples}")

base_ids = sorted(bases_all_files.keys())
print(f"Detected number of bases: {len(base_ids)} (expected 20) -> {base_ids}")

if STRICT_150:
    bad = []
    for cls, m in per_cls_base.items():
        for bid, files in m.items():
            if len(files) != GROUP_SIZE:
                bad.append((cls, bid, len(files)))
    if bad:
        print("\n[Warning] The following (class, base) groups have count ≠ 150; continuing, but distribution may be uneven:")
        for cls, bid, n in bad[:20]:
            print(f"  ({cls}, base {bid}) -> {n}")
        if len(bad) > 20:
            print(f"  ... {len(bad)-20} more groups omitted")

rng = random.Random(SEED)
if len(base_ids) < 20:
    print(f"[Warning] Only detected {len(base_ids)} bases; sampling from available ones.")

train_ids = set(rng.sample(base_ids, min(16, len(base_ids))))
remain = [b for b in base_ids if b not in train_ids]
val_ids = set(rng.sample(remain, min(2, len(remain))))
remain = [b for b in remain if b not in val_ids]
test_ids = set(rng.sample(remain, min(2, len(remain))))

print(f"\nBase assignment:\n  train: {sorted(train_ids)}\n  val  : {sorted(val_ids)}\n  test : {sorted(test_ids)}")

stats_split = Counter()
stats_cls_split = defaultdict(Counter)

for bid, files in bases_all_files.items():
    if bid in train_ids:
        out = TRAIN_DIR
        split = "train"
    elif bid in val_ids:
        out = VAL_DIR
        split = "val"
    elif bid in test_ids:
        out = TEST_DIR
        split = "test"
    else:
        continue
    for p in files:
        put_file(p, out)
        stats_split[split] += 1
        cls = p.name.split("-", 1)[0]
        stats_cls_split[split][cls] += 1

print("\n=== Done ===")
print(f"train: {stats_split['train']}  val: {stats_split['val']}  test: {stats_split['test']}")
for split in ("train","val","test"):
    if stats_split[split]:
        print(f"{split} per-class counts (should be roughly equal):", dict(stats_cls_split[split]))
