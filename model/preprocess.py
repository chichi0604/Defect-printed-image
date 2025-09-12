import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def imread_bgr(path: str):
    # robust to non-ascii paths
    arr = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img  # BGR, uint8

def _resize_locked(img_bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    interp = cv2.INTER_CUBIC if min(h, w) < size else cv2.INTER_AREA
    return cv2.resize(img_bgr, (size, size), interpolation=interp)

def preprocess_bgr_to_tensor(img_bgr: np.ndarray, size: int = 256, normalize: bool = True) -> torch.Tensor:
    img = _resize_locked(img_bgr, size)                 # (H,W,3) BGR
    img = img[:, :, ::-1].astype(np.float32) / 255.0    # -> RGB, float
    if normalize:
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))                  # HWC -> CHW
    return torch.from_numpy(img)                         # float32
