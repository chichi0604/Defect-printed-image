import os
import threading
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import cv2

import tkinter as tk
from tkinter import filedialog, messagebox
try:
    from PIL import Image, ImageTk
except Exception:
    Image = ImageTk = None  # Works without Pillow; thumbnails/background won't render

# project modules
CUR_DIR   = Path(__file__).resolve().parent
PROJ_ROOT = CUR_DIR.parent

from model.model import Net
from model.preprocess import preprocess_bgr_to_tensor
from model.dataset import CLASS_NAMES as DEFAULT_CLASS_NAMES

def _find_ckpt():
    env_p = os.environ.get("CKPT_PATH", "").strip()
    if env_p and Path(env_p).is_file():
        return env_p
    candidates = [
        CUR_DIR / "default" / "best_cls.pth",
        CUR_DIR / "best_cls.pth",
        PROJ_ROOT / "model" / "default" / "best_cls.pth",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return ""

CKPT_PATH = _find_ckpt()

def load_class_names(ckpt_path: str):
    names = list(DEFAULT_CLASS_NAMES)
    if ckpt_path:
        txt = Path(ckpt_path).with_name("class_names.txt")
        if txt.is_file():
            try:
                lines = [x.strip() for x in txt.read_text(encoding="utf-8").splitlines() if x.strip()]
                if lines:
                    names = lines
            except Exception:
                pass
    return names

CLASS_NAMES = load_class_names(CKPT_PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def imread_bgr_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img  # BGR

def bgr_to_tk(img_bgr, max_w, max_h):
    if Image is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    pil.thumbnail((max_w, max_h))
    return ImageTk.PhotoImage(pil)

def _find_bg():
    bg_path = r"G:\defect\GUI\background\tmp4081.png"  # Change the background

    if os.path.isfile(bg_path):
        return bg_path

    return None

# Model
_MODEL = None
_MODEL_LOCK = threading.Lock()

def load_model_once():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not CKPT_PATH:
        raise FileNotFoundError(
            "Checkpoint not found. Set CKPT_PATH or place best_cls.pth in current folder or model_cls_ft_fix/."
        )
    model = Net(pretrained=False).to(DEVICE)
    sd = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(sd, strict=True)
    model.eval()
    _MODEL = model
    return _MODEL

# Inference
def infer_one(test_bgr: np.ndarray):
    model = load_model_once()
    x = preprocess_bgr_to_tensor(test_bgr, size=256, normalize=True).unsqueeze(0).to(DEVICE).float()
    with torch.no_grad():
        logits1 = model(x)
        logits2 = model(torch.flip(x, dims=[-1]))  # hflip TTA
        logits = (logits1 + logits2) / 2.0
        probs = F.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    pred_name = CLASS_NAMES[pred_idx]
    k = min(3, probs.numel())
    vals, idxs = torch.topk(probs, k=k)
    top3 = [(CLASS_NAMES[int(i)], float(v)) for v, i in zip(vals.tolist(), idxs.tolist())]
    return pred_idx, pred_name, top3

# Parameters
COLORS = {
    "text":      "#000000",
    "muted":     "#a9b4c9",
    "line":      "#000000",
    "bar_bg":    "#222222",
    "accent":    "#3b82f6",
}

# GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Defect Inspector")
        self.geometry("860x520")
        self.minsize(820, 480)
        self.configure(bg="#000000")

        # State
        self.gold_path = None
        self.test_path = None
        self.gold_img_bgr = None
        self.test_img_bgr = None
        self.gold_tk = None
        self.test_tk = None
        self.pred_name = "—"
        self.top3 = []
        self.clock_item = None

        # Background
        self.bg_path = _find_bg()
        self.bg_pil = None
        self.bg_tk = None
        if Image is not None and self.bg_path:
            try:
                self.bg_pil = Image.open(self.bg_path).convert("RGB")
            except Exception:
                self.bg_pil = None

        # Layout params (match original vertical order: header -> main(L/R) -> ops -> result)
        self.margin = 16
        self.header_h = 54
        self.ops_h = 56
        self.result_h = 110
        self.col_gap = 8

        # Canvas
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        # Bindings
        self.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_click)

        # Shortcuts
        self.bind("<Return>", lambda e: self._run_threaded())
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Control-g>", lambda e: self._open_gold())
        self.bind("<Control-o>", lambda e: self._open_test())

        # Initial draw + clock
        self._redraw()
        self._tick_clock()

    # Layout (same zones as original)
    def _zones(self, W, H):
        m = self.margin
        header = (m, m, W - m, m + self.header_h)

        # main area height
        main_top = header[3] + 8
        main_bottom = H - m - self.ops_h - self.result_h - 8
        main_h = max(120, main_bottom - main_top)

        # split main into left/right
        left_w = (W - 2*m - self.col_gap) // 2
        left = (m, main_top, m + left_w, main_top + main_h)
        right = (left[2] + self.col_gap, main_top, W - m, main_top + main_h)

        ops = (m, H - m - self.ops_h - self.result_h - 8, W - m, H - m - self.result_h - 8)
        result = (m, H - m - self.result_h, W - m, H - m)
        return header, left, right, ops, result

    #  Drawing helpers
    def _draw_line_rect(self, rect, width=2, color=None):
        if color is None: color = COLORS["line"]
        x1, y1, x2, y2 = rect
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, fill="")

    def _draw_text(self, x, y, text, size=12, weight="normal", anchor="nw", color=None):
        if color is None: color = COLORS["text"]
        font = ("Microsoft YaHei UI", size, weight)
        return self.canvas.create_text(x, y, text=text, fill=color, font=font, anchor=anchor)

    def _draw_button(self, rect, text, tag):
        x1, y1, x2, y2 = rect
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=COLORS["line"], width=2, fill="", tags=(tag, f"{tag}_box"))
        tid = self._draw_text((x1+x2)//2, (y1+y2)//2, text, size=12, weight="bold", anchor="c", color=COLORS["text"])
        self.canvas.addtag_withtag(tag, tid)

    def _scale_bg(self, W, H):
        if Image is None or self.bg_pil is None:
            return
        bg = self.bg_pil
        bg_ratio = bg.width / bg.height
        win_ratio = W / H
        if win_ratio > bg_ratio:
            new_w, new_h = W, int(W / bg_ratio)
        else:
            new_h, new_w = H, int(H * bg_ratio)
        bg_resized = bg.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - W) // 2
        top  = (new_h - H) // 2
        self.bg_tk = ImageTk.PhotoImage(bg_resized.crop((left, top, left + W, top + H)))

    # Redraw everything
    def _redraw(self):
        c = self.canvas
        c.delete("all")
        W = max(1, self.winfo_width())
        H = max(1, self.winfo_height())

        # Background
        if self.bg_pil is not None and Image is not None:
            self._scale_bg(W, H)
            c.create_image(0, 0, anchor="nw", image=self.bg_tk)
        else:
            # fallback gradient
            for i in range(H):
                g = int(12 + (i / max(1, H)) * 24)
                c.create_line(0, i, W, i, fill=f"#{g:02x}{g:02x}{g:02x}")

        header, left, right, ops, result = self._zones(W, H)

        # Lines only (no filled panels)
        self._draw_line_rect(left, width=2)     # Gold area
        self._draw_line_rect(right, width=2)    # Test area
        self._draw_line_rect(result, width=2)   # Result area
        self._draw_line_rect(ops, width=2)      # Ops bar outline

        # Header: title (left) + device/classes + clock (right)
        self._draw_text(header[0]+4, header[1]+6,
                        "Defect Inspector", size=14, weight="bold")
        if self.clock_item is None:
            self.clock_item = self._draw_text(header[2]-6, header[1]+6, "", size=12, weight="normal", anchor="ne", color=COLORS["muted"])

        # Left panel: title + choose button + filename + thumbnail
        self._draw_text(left[0]+10, left[1]+8, "① Select Gold", size=12, weight="bold", color=COLORS["muted"])
        self._draw_button((left[0]+10, left[1]+32, left[0]+140, left[1]+64), "Choose Image", "open_gold")
        gold_name = Path(self.gold_path).name if self.gold_path else "No file selected"
        self._draw_text(left[0]+160, left[1]+38, gold_name, size=11, color=COLORS["muted"])

        # Thumbnail area for Gold
        self._draw_thumbnail(self.gold_tk, left, top_offset=80)

        # Right panel: title + choose button + filename + thumbnail
        self._draw_text(right[0]+10, right[1]+8, "② Select Test", size=12, weight="bold", color=COLORS["muted"])
        self._draw_button((right[0]+10, right[1]+32, right[0]+140, right[1]+64), "Choose Image", "open_test")
        test_name = Path(self.test_path).name if self.test_path else "No file selected"
        self._draw_text(right[0]+160, right[1]+38, test_name, size=11, color=COLORS["muted"])
        self._draw_thumbnail(self.test_tk, right, top_offset=80)

        # Ops bar (buttons left, status right)
        bx = ops[0] + 10
        by = ops[1] + (ops[3]-ops[1]-34)//2
        self._draw_button((bx, by, bx+110, by+34), "▶ Run (Enter)", "run")
        bx += 120
        self._draw_button((bx, by, bx+90, by+34), "Clear", "clear")
        bx += 100
        self._draw_button((bx, by, bx+90, by+34), "Exit (Esc)", "exit")

        # Result area: Top-1 to Top-3 (bars responsive to width)
        self._draw_text(result[0]+10, result[1]+10, f"Defect: {self.pred_name}", size=14, weight="bold")
        self._draw_top3_bars(result)

    def _draw_thumbnail(self, tkimg, rect, top_offset=80):
        if tkimg is None:
            return
        x1, y1, x2, y2 = rect
        # center area within rect, leaving space for controls above
        avail_w = max(10, (x2 - x1) - 20)
        avail_h = max(10, (y2 - (y1+top_offset)) - 20)
        # image already thumbnailed to fit — just center draw
        cx = x1 + 10 + avail_w // 2
        cy = y1 + top_offset + 10 + avail_h // 2
        self.canvas.create_image(cx, cy, image=tkimg)

    def _draw_top3_bars(self, rect):
        x1, y1, x2, y2 = rect
        # If no data, nothing to draw
        if not self.top3:
            return

        pad_l, pad_r, pad_t = 10, 14, 38
        label_w, pct_w = 130, 56
        bar_h, gap = 18, 6

        bar_left  = x1 + pad_l + label_w
        bar_right = x2 - pad_r - pct_w
        bar_width = max(40, bar_right - bar_left)

        y = y1 + pad_t
        for name, prob in self.top3:
            pct = max(0.0, min(1.0, float(prob)))
            filled = int(bar_width * pct)

            # Label
            self._draw_text(x1 + pad_l, y + bar_h/2, f"{name}", size=11, weight="bold", anchor="w", color=COLORS["text"])
            # Bar background
            self.canvas.create_rectangle(bar_left, y, bar_right, y+bar_h, outline=COLORS["bar_bg"], width=1, fill="")
            # Bar filled part
            self.canvas.create_rectangle(bar_left, y, bar_left + filled, y+bar_h, outline="", width=0, fill=COLORS["accent"])
            # Percentage
            self._draw_text(x2 - pad_r, y + bar_h/2, f"{pct*100:.1f}%", size=11, weight="normal", anchor="e", color=COLORS["muted"])
            y += bar_h + gap

    def _on_click(self, event):
        x, y = event.x, event.y
        items = self.canvas.find_overlapping(x, y, x, y)
        tags = set()
        for it in items:
            tags.update(self.canvas.gettags(it))
        if "open_gold" in tags:
            self._open_gold()
        elif "open_test" in tags:
            self._open_test()
        elif "run" in tags:
            self._run_threaded()
        elif "clear" in tags:
            self._clear_state()
        elif "exit" in tags:
            self.destroy()

    def _on_resize(self, event):
        # On resize, regenerate thumbnails to fit their zones and redraw
        W, H = self.winfo_width(), self.winfo_height()
        _, left, right, _, _ = self._zones(W, H)
        if self.gold_img_bgr is not None and Image is not None:
            self.gold_tk = bgr_to_tk(self.gold_img_bgr,
                                     max_w=max(10, (left[2]-left[0]) - 20),
                                     max_h=max(10, (left[3]-left[1]) - 100))
        if self.test_img_bgr is not None and Image is not None:
            self.test_tk = bgr_to_tk(self.test_img_bgr,
                                     max_w=max(10, (right[2]-right[0]) - 20),
                                     max_h=max(10, (right[3]-right[1]) - 100))
        self._redraw()

    def _tick_clock(self):
        if self.clock_item is not None:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.canvas.itemconfig(self.clock_item, text=now)
        self.after(1000, self._tick_clock)

    def _open_gold(self):
        path = filedialog.askopenfilename(
            title="Choose Gold Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff")]
        )
        if not path:
            return
        try:
            img = imread_bgr_unicode(path)
            self.gold_img_bgr = img
            self.gold_path = path
            W, H = self.winfo_width(), self.winfo_height()
            _, left, _, _, _ = self._zones(W, H)
            self.gold_tk = bgr_to_tk(img,
                                     max_w=max(10, (left[2]-left[0]) - 20),
                                     max_h=max(10, (left[3]-left[1]) - 100))
            self._redraw()
        except Exception as e:
            messagebox.showerror("Read Error", f"{e}")

    def _open_test(self):
        path = filedialog.askopenfilename(
            title="Choose Test Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff")]
        )
        if not path:
            return
        try:
            img = imread_bgr_unicode(path)
            self.test_img_bgr = img
            self.test_path = path
            W, H = self.winfo_width(), self.winfo_height()
            _, _, right, _, _ = self._zones(W, H)
            self.test_tk = bgr_to_tk(img,
                                     max_w=max(10, (right[2]-right[0]) - 20),
                                     max_h=max(10, (right[3]-right[1]) - 100))
            self._redraw()
        except Exception as e:
            messagebox.showerror("Read Error", f"{e}")

    def _run_threaded(self):
        if not self.test_path:
            messagebox.showwarning("Notice", "Please choose the Test image first.")
            return
        t = threading.Thread(target=self._run_infer, daemon=True)
        t.start()

    def _run_infer(self):
        try:
            test_bgr = self.test_img_bgr if self.test_img_bgr is not None else imread_bgr_unicode(self.test_path)
            _, pred_name, top3 = infer_one(test_bgr)
            self.pred_name = pred_name
            self.top3 = top3
            self._redraw()
        except Exception as e:
            err = "".join(traceback.format_exception_only(type(e), e)).strip()
            messagebox.showerror(
                "Inference Error",
                f"{err}\n\nIf this persists, please check:\n"
                f"1) Checkpoint path: {CKPT_PATH or '(missing)'}\n"
                f"2) model.py / preprocess.py match your training code\n"
                f"3) Class count matches (class_names.txt)"
            )

    def _clear_state(self):
        self.gold_path = None
        self.test_path = None
        self.gold_img_bgr = None
        self.test_img_bgr = None
        self.gold_tk = None
        self.test_tk = None
        self.pred_name = "—"
        self.top3 = []
        self._redraw()

if __name__ == "__main__":
    try:
        load_model_once()
    except Exception as e:
        print(f"[Warn] Model preload failed: {e}")
    app = App()
    app.mainloop()
