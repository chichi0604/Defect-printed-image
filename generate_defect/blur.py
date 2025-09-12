import cv2, numpy as np, random

def _soft_roi(h, w):
    # Soft elliptical ROI via distance decay (centered at a random location)
    cx, cy = random.uniform(0.25,0.75)*w, random.uniform(0.25,0.75)*h
    rx, ry = random.uniform(0.2,0.35)*w, random.uniform(0.2,0.35)*h
    y, x = np.ogrid[:h,:w]
    d = ((x-cx)/rx)**2 + ((y-cy)/ry)**2
    m = np.exp(-d*2.5).astype(np.float32)  # 0..1
    m = cv2.GaussianBlur(m, (0,0), 7.0)
    return (m / m.max()).astype(np.float32)

def blur(img, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    h,w = img.shape[:2]
    im = img.astype(np.float32)/255.0

    roi = _soft_roi(h,w)[...,None]  # HxWx1

    # Choose Gaussian blur or a simple horizontal line kernel (motion-like)
    if random.random()<0.6:
        k = random.choice([3,5,7,9,11])
        blurred = cv2.GaussianBlur(im, (k,k), random.uniform(0.8,2.5))
    else:
        k = random.randint(5,15)
        kern = np.zeros((k,k), np.float32)
        kern[k//2,:] = 1.0/k
        blurred = cv2.filter2D(im, -1, kern)

    # Strength decays outside the ROI
    alpha = (0.35 + 0.45*roi).astype(np.float32)
    out = im*(1-alpha*roi) + blurred*(alpha*roi)
    out = np.clip(out, 0, 1)

    mask = (np.clip(roi*255,0,255)).astype(np.uint8)[...,0]
    meta={"type":"blur","kernel":k}
    return (out*255).astype(np.uint8), mask, meta
