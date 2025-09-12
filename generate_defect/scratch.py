import cv2, numpy as np, random, math

def _one_walk(h,w):
    # Random polyline: several directed segments with clamped boundaries
    nseg = random.randint(6, 12)
    pts = []
    x,y = random.randint(0,w-1), random.randint(0,h-1)
    for _ in range(nseg):
        ang = random.uniform(0, 2*math.pi)
        step = random.randint(int(0.04*min(h,w)), int(0.12*min(h,w)))
        x = np.clip(int(x + step*math.cos(ang)), 0, w-1)
        y = np.clip(int(y + step*math.sin(ang)), 0, h-1)
        pts.append((x,y))
    return pts

def scratch(img, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    h, w = img.shape[:2]
    im = img.astype(np.float32)/255.0

    nline = random.randint(3, 8)
    layer = np.zeros((h,w), np.float32)

    for _ in range(nline):
        pts = _one_walk(h,w)
        for i in range(len(pts)-1):
            width = random.uniform(0.8, 1.8)
            cv2.line(layer, pts[i], pts[i+1], 1.0, int(round(width)))
    layer = cv2.GaussianBlur(layer, (0,0), 0.8)

    # Bright core + soft dark halo
    bright = np.clip(layer*0.6, 0, 1)
    dark   = cv2.GaussianBlur(layer, (0,0), 2.0)*0.25

    out = im + bright[...,None] - dark[...,None]
    out = np.clip(out, 0, 1)

    mask = (np.clip(layer*255,0,255)).astype(np.uint8)
    meta={"type":"scratch","count":nline}
    return (out*255).astype(np.uint8), mask, meta
