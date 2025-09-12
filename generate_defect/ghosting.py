import cv2, numpy as np, random

def _edge_mask(gray):
    # Softened edge map to confine echoes near edges
    e = cv2.Canny(gray, 60, 150)
    e = cv2.dilate(e, np.ones((3,3), np.uint8), iterations=1)
    e = cv2.GaussianBlur(e, (0,0), 1.0)
    return e

def ghosting(img, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    h, w = img.shape[:2]
    im = img.astype(np.float32)/255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    emask = _edge_mask(gray).astype(np.float32)/255.0
    emask3 = np.dstack([emask]*3)

    # Echo period and decay
    period = max(8, int(round(random.uniform(0.04, 0.12)*h)))
    necho  = random.randint(1, 3)
    alpha0 = random.uniform(0.12, 0.25)
    decay  = random.uniform(0.45, 0.75)

    base = im.copy()
    for k in range(1, necho+1):
        shift = k * period
        M = np.float32([[1,0,0],[0,1, shift]])
        echo = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # Make echoes low-contrast + slightly desaturated
        echo = np.clip(echo*random.uniform(0.6, 0.85), 0, 1)
        a = alpha0 * (decay ** (k-1))
        im = im*(1 - a*emask3) + echo*(a*emask3)

    im = np.clip(im, 0, 1)
    m = (emask*255).astype(np.uint8)
    meta = {"type":"ghosting","period_px":period,"echo":necho}
    return (im*255).astype(np.uint8), m, meta
