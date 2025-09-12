import cv2, numpy as np, random

MIN_FRAC = 0.005   # 0.5%
MAX_FRAC = 0.12    # 12%

def _set_seed(seed):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

def _pick_seed(gray):
    h, w = gray.shape
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    score = np.abs(lap)
    th = np.quantile(score, 0.35)
    ys, xs = np.where(score <= th)
    if len(xs) == 0:
        return w//2, h//2
    i = random.randrange(len(xs))
    return int(xs[i]), int(ys[i])

def _area_ok(mask):
    a = float((mask>0).sum()) / (mask.shape[0]*mask.shape[1] + 1e-6)
    return (a >= MIN_FRAC) and (a <= MAX_FRAC), a

def _soft(mask_u8, k=3, sigma=0):
    if k<=1: return mask_u8
    return cv2.GaussianBlur(mask_u8, (k|1, k|1), sigma)

def color_shift(img, seed=None):
    _set_seed(seed)
    h, w = img.shape[:2]

    # Auto downscale: floodFill on a smaller image if >~1.2MP, then upsample the mask
    DOWNSCALE = 2 if (h*w > 1200*1000) else 1
    small = cv2.resize(img, (w//DOWNSCALE, h//DOWNSCALE), interpolation=cv2.INTER_AREA) if DOWNSCALE>1 else img

    # Lab & seed
    lab_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sx, sy = _pick_seed(gray_small)

    mask = np.zeros((lab_small.shape[0]+2, lab_small.shape[1]+2), np.uint8)

    # One of three common casts (modify a*/b*; L* only slightly perturbed)
    mode = random.choice(["yellow","magenta","cyan"])
    # Initial tolerance (Lab channels): a*/b* mainly control area
    lo = (6, 6, 6); hi = (6, 6, 6)

    # Adaptive tolerance: up to 6 tries to push area into [MIN_FRAC, MAX_FRAC]
    filled_ok = False
    for _ in range(6):
        mask[:] = 0
        flags = (4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY)
        # floodFill requires newVal; with MASK_ONLY the image is not changed
        _ = cv2.floodFill(lab_small, mask, (sx, sy), newVal=(0,0,0),
                          loDiff=lo, upDiff=hi, flags=flags)
        m_small = (mask[1:-1, 1:-1] > 0).astype(np.uint8)*255
        ok, frac = _area_ok(m_small)
        if ok:
            filled_ok = True
            break
        # Too small -> enlarge tol; too big -> shrink
        scale = 1.35 if frac < MIN_FRAC else 0.75
        lo = tuple(max(1, int(v*scale)) for v in lo)
        hi = tuple(max(1, int(v*scale)) for v in hi)

    # Fallback tiny blob if flood fails (rare)
    if not filled_ok:
        if m_small.sum() == 0:
            m_small = np.zeros_like(m_small); cv2.circle(m_small,(sx,sy), max(4, min(h,w)//100), 255, -1)

    # Upsample mask to original size
    if DOWNSCALE>1:
        mask_u8 = cv2.resize(m_small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_u8 = m_small

    # Smooth + morphological cleanup
    mask_u8 = _soft(mask_u8, k=5)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Apply cast in Lab: adjust a*/b*; L* only ±2
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = (mask_u8.astype(np.float32)/255.0)[...,None]

    if mode == "yellow":   da, db = 0.0,  random.uniform(8,16)
    elif mode == "magenta":da, db =  random.uniform(8,16), 0.0
    else:                  da, db = -random.uniform(8,16), 0.0

    dL = random.uniform(-2.0, 2.0)  # slight illumination perturbation
    lab[...,0] = np.clip(lab[...,0] + dL*m[...,0], 0, 255)
    lab[...,1] = np.clip(lab[...,1] + da*m[...,0], 0, 255)
    lab[...,2] = np.clip(lab[...,2] + db*m[...,0], 0, 255)

    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out, mask_u8, {"mode": mode, "downscale": DOWNSCALE, "tol_lo": lo, "tol_hi": hi}
