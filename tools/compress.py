from pathlib import Path
import cv2

SRC = Path(r"G:\defect\dataset\original")
DST = Path(r"G:\defect\dataset\original_1024")
DST.mkdir(parents=True, exist_ok=True)

MAX_SIDE = 1024
PNG_PARAM = [cv2.IMWRITE_PNG_COMPRESSION, 3]

ex = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp")
files = []
for e in ex: files += list(SRC.glob(e))
for p in sorted(files):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        print("skip:", p); continue
    h,w = img.shape[:2]
    if max(h,w) > MAX_SIDE:
        s = MAX_SIDE / max(h,w)
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    out = DST / (p.stem + ".png")
    cv2.imwrite(str(out), img, PNG_PARAM)

print("done ->", DST)
