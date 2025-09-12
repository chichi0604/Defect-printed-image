import cv2, numpy as np, random

def _fbm_like(h, w, octaves=3):
    # Fractal-like noise: multi-scale bicubic upsampling + amplitude decay, then normalize via np.ptp()
    hh, ww = max(2, h//8+1), max(2, w//8+1)
    base = cv2.resize(np.random.rand(hh, ww).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    acc, amp = np.zeros((h, w), np.float32), 1.0
    for o in range(octaves):
        hh = max(2, h//(8>>o)+1)
        ww = max(2, w//(8>>o)+1)
        r = cv2.resize(np.random.rand(hh, ww).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        acc += amp * r
        amp *= 0.5
    rng = float(np.ptp(acc))
    if rng < 1e-6:
        return np.zeros((h, w), np.float32)
    acc = (acc - float(acc.min())) / (rng + 1e-6)
    return acc

def dropout(img, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    h, w = img.shape[:2]
    im = img.astype(np.float32) / 255.0

    # Soft elliptical ROI
    cx, cy = random.uniform(0.3,0.7)*w, random.uniform(0.3,0.7)*h
    rx, ry = random.uniform(0.18,0.32)*w, random.uniform(0.18,0.32)*h
    y, x = np.ogrid[:h,:w]
    d = ((x-cx)/max(1e-6,rx))**2 + ((y-cy)/max(1e-6,ry))**2
    roi = np.exp(-d*2.8).astype(np.float32)
    roi = cv2.GaussianBlur(roi,(0,0), 6.0)

    # Paper texture
    paper = _fbm_like(h, w, octaves=3)
    paper = (0.85 + 0.15*paper).astype(np.float32)
    paper3 = np.dstack([paper]*3)

    # Dark rim to mimic “feathered/pressed edges”
    rim = cv2.Laplacian(roi, cv2.CV_32F, ksize=3)
    rim = np.clip(-rim*2.0, 0, 1)

    # Stronger “show-through” at center
    alpha = (0.5 + 0.5*roi)[..., None]
    out = im*(1-alpha) + paper3*alpha
    out = np.clip(out - 0.08*rim[..., None], 0, 1)

    mask = (np.clip(roi*255,0,255)).astype(np.uint8)
    meta = {"type":"dropout"}
    return (out*255).astype(np.uint8), mask, meta
