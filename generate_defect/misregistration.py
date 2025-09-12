import cv2
import numpy as np

def _edge(gray):
    # Soft edge map for where channel shifts are visible
    e = cv2.Canny(gray, 60, 150)
    e = cv2.dilate(e, np.ones((3,3), np.uint8), iterations=1)
    e = cv2.GaussianBlur(e, (0,0), 1.0)
    return e/255.0

def _remap_shift(ch, dx_field, dy_field):
    # Per-pixel displacement field remap (bilinear)
    h, w = ch.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mapx = xx + dx_field
    mapy = yy + dy_field
    return cv2.remap(ch, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def misregistration(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    im = img.copy()
    h, w = im.shape[:2]

    # Edge mask
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    em = _edge(gray)
    em3 = np.dstack([em]*3)

    # Build smooth displacement fields (vertical or horizontal bias)
    amp_px = np.random.uniform(0.25, 1.25)  # in pixels
    vertical = np.random.rand() < 0.5

    y = np.linspace(0, 2*np.pi, h, dtype=np.float32)
    x = np.linspace(0, 2*np.pi, w, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    if vertical:
        dx0 = amp_px * (0.5*np.sin(2.0*Y) + 0.25*np.sin(3.3*Y + 0.5))
        dy0 = np.zeros_like(dx0)
    else:
        dy0 = amp_px * (0.5*np.sin(2.0*X) + 0.25*np.sin(3.3*X + 0.5))
        dx0 = np.zeros_like(dy0)

    # Slight per-channel random offsets
    out_ch = []
    for c in cv2.split(im):
        bx = (np.random.rand(h, w).astype(np.float32)-0.5)*0.15
        by = (np.random.rand(h, w).astype(np.float32)-0.5)*0.15
        dx = dx0 + bx
        dy = dy0 + by
        out_ch.append(_remap_shift(c, dx, dy))
    mis = cv2.merge(out_ch)

    # Show misregistration only near edges; elsewhere mix back original to avoid global cast
    imf = im.astype(np.float32)/255.0
    k = 0.85  # intensity retention near edges
    out = imf*(1-em3) + (k*mis + (1-k)*imf)*em3
    out = np.clip(out, 0, 1)

    mask = (em*255).astype(np.uint8)
    meta={"type":"misregistration","vertical":bool(vertical),"amp_px":float(amp_px)}
    return (out*255).astype(np.uint8), mask, meta
