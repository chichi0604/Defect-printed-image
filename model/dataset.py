import os, glob
from typing import List, Tuple
from torch.utils.data import Dataset

CLASS_NAMES = [
    'banding', 'blur', 'color-shift', 'dropout',
    'ghosting', 'misregistration', 'scratch'
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

from model.preprocess import imread_bgr, preprocess_bgr_to_tensor

def build_preprocess(split: str):

    return None

def _is_img(p: str) -> bool:
    return os.path.splitext(p)[-1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}

def _scan_by_subfolders(root: str) -> List[Tuple[str, int]]:
    pairs = []
    for cname in CLASS_NAMES:
        d = os.path.join(root, cname)
        if not os.path.isdir(d):
            continue
        for ext in ('*.png','*.jpg','*.jpeg','*.bmp','*.webp','*.tif','*.tiff'):
            for p in glob.glob(os.path.join(d, ext)):
                if _is_img(p):
                    pairs.append((p, CLASS_TO_IDX[cname]))
    return pairs

def _scan_by_prefix(root: str) -> List[Tuple[str, int]]:
    pairs = []
    for ext in ('*.png','*.jpg','*.jpeg','*.bmp','*.webp','*.tif','*.tiff'):
        for p in glob.glob(os.path.join(root, ext)):
            if not _is_img(p):
                continue
            name = os.path.basename(p).lower()
            for cname in CLASS_NAMES:
                if name.startswith(cname + '-') or name.startswith(cname + '_') or name.startswith(cname + '.'):
                    pairs.append((p, CLASS_TO_IDX[cname]))
                    break
    return pairs

def scan_pairs_auto(root: str) -> List[Tuple[str, int]]:
    ps = _scan_by_subfolders(root)
    return ps if ps else _scan_by_prefix(root)

class DefectDataset(Dataset):
    def __init__(self, root: str, split: str = 'train'):
        self.root = root
        self.split = split
        pairs = scan_pairs_auto(root)
        if len(pairs) == 0:
            raise RuntimeError(
                f"[{split}] No 7-class samples found in: {root}\n"
                f"Class set: {CLASS_NAMES}\n"
                f"Supported formats: subfolders (root/class/*.png) or flat prefixes (root/class-*.png)"
            )
        self.samples = [(p, y, i) for i, (p, y) in enumerate(sorted(pairs))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, y, _ = self.samples[idx]
        x = preprocess_bgr_to_tensor(imread_bgr(p), size=256, normalize=True)
        return x, y, p
