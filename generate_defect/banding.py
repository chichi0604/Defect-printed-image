import cv2, numpy as np, random, math

def _set_seed(seed):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

def _px(v, h, w, base=512):
    # Convert a reference pixel value scaled to current image size
    return max(1, int(round(v * min(h, w) / float(base))))

def _soft_mask(mask, k=3, sigma=0):
    # Light Gaussian blur to soften a binary/float mask
    if k <= 1:
        return mask
    return cv2.GaussianBlur(mask, (k | 1, k | 1), sigma)

def _area_ok(mask, min_frac=0.30, max_frac=0.95):
    # Validate that mask coverage is within a target fraction range
    a = float((mask > 0).sum()) / (mask.shape[0] * mask.shape[1] + 1e-6)
    return (a >= min_frac) and (a <= max_frac)

def banding(img, seed=None):
    _set_seed(seed)
    h, w = img.shape[:2]
    out = img.astype(np.float32).copy()

    # Low-frequency multiplicative gain along Y (horizontal stripes)
    y = np.arange(h, dtype=np.float32)
    baseT = _px(random.uniform(36, 96), h, w)  # period in pixels
    # Mixture of 2–3 low-frequency cosines + very low-freq drift (avoid perfectly uniform spacing)
    g = (0.08*np.cos(2*np.pi*y/baseT + random.uniform(0, 2*math.pi))
         + 0.04*np.cos(2*np.pi*y/(baseT*0.6) + random.uniform(0, 2*math.pi))
         + 0.03*np.cos(2*np.pi*y/(baseT*1.8) + random.uniform(0, 2*math.pi)))
    drift = 0.05*np.sin(np.linspace(0, 2*np.pi, h, dtype=np.float32))
    gain1 = 1.0 + (g + drift)  # shape: (h,)

    # Expand to full image (horizontal only, no rotation)
    gain = np.repeat(gain1[:, None], w, axis=1).astype(np.float32)  # (h,w)

    # Optional piecewise amplitude change (still horizontal)
    if random.random() < 0.35:
        cut = random.randint(h//6, h - h//6)
        gain[cut:] *= random.uniform(0.8, 1.2)

    # Apply multiplicative gain
    out = np.clip(out * gain[..., None], 0, 255).astype(np.uint8)

    # Mask by thresholding |gain-1| with a quantile (avoid .ptp())
    dev = np.abs(gain - 1.0).astype(np.float32)
    tau = float(np.quantile(dev, 0.7))
    mask = (dev > tau).astype(np.uint8) * 255
    mask = _soft_mask(mask, k=3)

    # Banding is a “large-area” defect: relax coverage bounds; if not enough, lower threshold iteratively
    for _ in range(4):
        if _area_ok(mask, 0.30, 0.95):
            break
        tau *= 0.9
        mask = (dev > tau).astype(np.uint8) * 255
        mask = _soft_mask(mask, k=3)

    return out, mask, {"period_px": float(baseT)}
