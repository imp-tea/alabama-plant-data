# pixel_art_gui_v2.py
import os, math, threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# ----------------------------- Pillow constants (10+ compatible) -----------------------------
try:
    from PIL.Image import Quantize, Resampling, Dither
    MEDIANCUT = Quantize.MEDIANCUT
    NEAREST = Resampling.NEAREST
    DITHER_NONE = Dither.NONE
    DITHER_FS = Dither.FLOYDSTEINBERG
except Exception:
    MEDIANCUT = Image.MEDIANCUT
    NEAREST = Image.NEAREST
    DITHER_NONE = Image.Dither.NONE
    DITHER_FS = Image.Dither.FLOYDSTEINBERG

# ----------------------------- Pipeline (from your script) -----------------------------
def flatten_transparency_threshold(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        rgba = img.convert("RGBA")
        arr = np.array(rgba, dtype=np.uint8)
        rgb, a = arr[..., :3], arr[..., 3]
        transparent_mask = a <= 64
        if transparent_mask.any():
            opaque_mask = a > 64
            near_black_threshold = 16
            near_black = np.all(rgb <= near_black_threshold, axis=-1)
            opaque_near_black_exists = bool(np.any(opaque_mask & near_black))
            fill_value = 255 if opaque_near_black_exists else 0
            rgb[transparent_mask] = fill_value
        return Image.fromarray(rgb, mode="RGB")
    else:
        return img.convert("RGB")

def _srgb_to_linear(c):
    a = 0.055
    return np.where(c <= 0.04045, c/12.92, ((c + a)/(1 + a))**2.4)

def _rgb_to_lab(rgb8):
    rgb = rgb8.astype(np.float64) / 255.0
    rgb = _srgb_to_linear(rgb)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = rgb @ M.T
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[:,0]/Xn; y = xyz[:,1]/Yn; z = xyz[:,2]/Zn
    eps = 216/24389; kappa = 24389/27
    fx = np.where(x > eps, np.cbrt(x), (kappa*x + 16)/116)
    fy = np.where(y > eps, np.cbrt(y), (kappa*y + 16)/116)
    fz = np.where(z > eps, np.cbrt(z), (kappa*z + 16)/116)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.stack([L, a, b], axis=1)

def _deltaE76(lab1, lab2):
    d = lab1 - lab2
    return np.sqrt(np.sum(d*d, axis=-1))

def _sample_uniform_grid(img_np, step=4):
    h, w, _ = img_np.shape
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pts = img_np[yy, xx]
    return pts.reshape(-1, 3)

def _dedup_colors(rgb, round_to=16):
    q = (rgb // round_to) * round_to + round_to//2
    uq, idx = np.unique(q, axis=0, return_index=True)
    return uq[idx.argsort()]

def _farthest_point_palette(cand_lab, K, seed_idx=None, dist_fn=_deltaE76):
    N = cand_lab.shape[0]
    if seed_idx is None:
        chroma = np.sqrt(cand_lab[:,1]**2 + cand_lab[:,2]**2)
        seed_idx = int(np.argmax(chroma))
    chosen = [seed_idx]
    dmin = dist_fn(cand_lab, cand_lab[seed_idx:seed_idx+1]).reshape(N)
    for _ in range(1, K):
        idx = int(np.argmax(dmin))
        chosen.append(idx)
        dmin = np.minimum(dmin, dist_fn(cand_lab, cand_lab[idx:idx+1]).reshape(N))
    return np.array(chosen, dtype=int)

def _kmedoids_one_iter(cand_lab, palette_idx, dist_fn=_deltaE76):
    palette = cand_lab[palette_idx]
    D = np.stack([dist_fn(cand_lab, p[None,:]) for p in palette], axis=1)
    assign = np.argmin(D, axis=1)
    new_idx = []
    for k in range(palette.shape[0]):
        members = np.where(assign == k)[0]
        if len(members) == 0:
            new_idx.append(palette_idx[k]); continue
        sub = cand_lab[members]
        SD = np.sum(np.sqrt(((sub[:,None,:]-sub[None,:,:])**2).sum(-1)), axis=1)
        new_idx.append(members[int(np.argmin(SD))])
    return np.array(new_idx, dtype=int)

def posterize_diverse(img_rgb, colors=10, sample_step=4, bucket=16, refine=True, map_chunk=400_000):
    img_np = np.array(img_rgb, dtype=np.uint8)
    cand_rgb = _sample_uniform_grid(img_np, step=sample_step)
    cand_rgb = _dedup_colors(cand_rgb, round_to=bucket)
    cand_lab = _rgb_to_lab(cand_rgb)
    K = max(2, int(colors))
    pal_idx = _farthest_point_palette(cand_lab, K)
    if refine:
        pal_idx = _kmedoids_one_iter(cand_lab, pal_idx)
    palette_rgb = cand_rgb[pal_idx]
    H, W, _ = img_np.shape
    flat_rgb = img_np.reshape(-1, 3)
    lab_pal = _rgb_to_lab(palette_rgb)
    out_idx = np.empty((flat_rgb.shape[0],), dtype=np.uint8)

    start = 0
    while start < flat_rgb.shape[0]:
        end = min(start + map_chunk, flat_rgb.shape[0])
        lab_chunk = _rgb_to_lab(flat_rgb[start:end])
        D = np.stack([_deltaE76(lab_chunk, p[None,:]) for p in lab_pal], axis=1)
        out_idx[start:end] = np.argmin(D, axis=1).astype(np.uint8)
        start = end

    paletted = Image.fromarray(out_idx.reshape(H, W), mode="P")
    pal_list = palette_rgb.astype(np.uint8).reshape(-1).tolist()
    paletted.putpalette(pal_list + [0]*(768 - len(pal_list)))
    return paletted

def _best_period(strength, lo, hi):
    best_s, best_score = None, -1.0
    strength = strength.astype(np.float64)
    strength = (strength - strength.mean()) / (strength.std() + 1e-6)
    n = len(strength)
    for s in range(lo, min(hi, n) + 1):
        best_for_s = -1.0
        for o in range(s):
            vals = strength[o::s]
            if len(vals) <= 1:
                continue
            score = np.mean(np.abs(vals))
            if score > best_for_s:
                best_for_s = score
        score = best_for_s / math.sqrt(s)
        if score > best_score:
            best_score, best_s = score, s
    return best_s or 1

def estimate_grid_step_from_edges(pal_img, smin=2, smax=64):
    arr = np.array(pal_img)
    edges_x = (np.diff(arr, axis=1) != 0).astype(np.uint8)
    edges_y = (np.diff(arr, axis=0) != 0).astype(np.uint8)
    col_strength = edges_x.sum(axis=0)
    row_strength = edges_y.sum(axis=1)
    sx = _best_period(col_strength, smin, smax)
    sy = _best_period(row_strength, smin, smax)
    return sy, sx

def downscale_by_mode(pal_img, sy, sx):
    arr = np.array(pal_img)
    H, W = arr.shape
    H2, W2 = (H // sy) * sy, (W // sx) * sx
    arr = arr[:H2, :W2]
    hb, wb = H2 // sy, W2 // sx
    blocks = arr.reshape(hb, sy, wb, sx).swapaxes(1, 2).reshape(hb * wb, sy * sx)
    out_flat = np.empty(hb * wb, dtype=np.uint8)
    for i in range(blocks.shape[0]):
        bc = np.bincount(blocks[i], minlength=256)
        out_flat[i] = np.argmax(bc)
    out = out_flat.reshape(hb, wb)
    out_img = Image.fromarray(out, mode="P")
    out_img.putpalette(pal_img.getpalette())
    return out_img

def fix_pixel_art(img, colors=10, smin=2, smax=64, sample_step=4, bucket=16, refine=True, override=None):
    img_prep = flatten_transparency_threshold(img)
    pal = posterize_diverse(img_prep, colors=colors, sample_step=sample_step,
                            bucket=bucket, refine=refine)
    if override is None:
        sy, sx = estimate_grid_step_from_edges(pal, smin=smin, smax=smax)
        if sx < sy:
            sx = sy
        else:
            sy = sx
    else:
        sy, sx = override
    fixed = downscale_by_mode(pal, sy, sx)
    return pal, (sy, sx), fixed

# ----------------------------- GUI -----------------------------
class PixelArtGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Art Fixer (GUI)")
        self.root.geometry("1000x600")          # fixed initial size; no grow from image content
        self.root.minsize(800, 500)

        self.input_path = None
        self.orig_img = None
        self.fixed_img = None
        self.block_size = (None, None)
        self.process_lock = threading.Lock()

        self._build_layout()
        self._build_controls()

    # Always two equal halves via canvases that resize with window
    def _build_layout(self):
        self.main = ttk.Frame(self.root, padding=6)
        self.main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.image_frame = ttk.Frame(self.main)
        self.image_frame.grid(row=0, column=0, sticky="nsew")
        self.main.rowconfigure(0, weight=1)
        self.main.columnconfigure(0, weight=1)

        # Two canvases occupy exactly half each
        self.left_canvas  = tk.Canvas(self.image_frame, bg="#111", highlightthickness=0)
        self.right_canvas = tk.Canvas(self.image_frame, bg="#111", highlightthickness=0)
        self.left_canvas.grid(row=0, column=0, sticky="nsew", padx=(0,3))
        self.right_canvas.grid(row=0, column=1, sticky="nsew", padx=(3,0))
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.columnconfigure(1, weight=1)
        self.image_frame.rowconfigure(0, weight=1)

        # Keep a reference to images drawn on canvases
        self._left_tk = None
        self._right_tk = None

        # On any resize, resize canvases and redraw
        self.image_frame.bind("<Configure>", self._on_frame_configure)

        self.ctrl = ttk.Frame(self.main)
        self.ctrl.grid(row=1, column=0, sticky="ew", pady=(6,0))
        for i in range(10):
            self.ctrl.columnconfigure(i, weight=1)

    def _build_controls(self):
        # File ops
        self.btn_load = ttk.Button(self.ctrl, text="Load Image", command=self.load_image)
        self.btn_load.grid(row=0, column=0, sticky="w")
        self.btn_save = ttk.Button(self.ctrl, text="Save Image", command=self.save_image, state="disabled")
        self.btn_save.grid(row=0, column=1, sticky="w", padx=(6,0))
        self.btn_process = ttk.Button(self.ctrl, text="Process Now", command=self._trigger_reprocess)
        self.btn_process.grid(row=0, column=9, sticky="e")

        # Parameters (text entries with clamping)
        # colors
        ttk.Label(self.ctrl, text="Colors (2..128)").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.colors_var = tk.StringVar(value="64")
        ttk.Entry(self.ctrl, width=6, textvariable=self.colors_var).grid(row=1, column=1, sticky="w", pady=(6,0))

        # smin/smax
        ttk.Label(self.ctrl, text="smin (2..128)").grid(row=1, column=2, sticky="e", pady=(6,0))
        self.smin_var = tk.StringVar(value="12")
        ttk.Entry(self.ctrl, width=6, textvariable=self.smin_var).grid(row=1, column=3, sticky="w", pady=(6,0))

        ttk.Label(self.ctrl, text="smax (2..256)").grid(row=1, column=4, sticky="e", pady=(6,0))
        self.smax_var = tk.StringVar(value="32")
        ttk.Entry(self.ctrl, width=6, textvariable=self.smax_var).grid(row=1, column=5, sticky="w", pady=(6,0))

        # sample step / bucket / refine
        ttk.Label(self.ctrl, text="Sample step (1..32)").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.sample_var = tk.StringVar(value="4")
        ttk.Entry(self.ctrl, width=6, textvariable=self.sample_var).grid(row=2, column=1, sticky="w", pady=(6,0))

        ttk.Label(self.ctrl, text="Bucket").grid(row=2, column=2, sticky="e", pady=(6,0))
        self.bucket_var = tk.StringVar(value="16")
        bucket_box = ttk.Combobox(self.ctrl, textvariable=self.bucket_var, width=6,
                                  values=["8","16","32"], state="readonly")
        bucket_box.grid(row=2, column=3, sticky="w", pady=(6,0))

        self.refine_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.ctrl, text="Refine (1-iter k-medoids)", variable=self.refine_var).grid(
            row=2, column=4, columnspan=2, sticky="w", pady=(6,0)
        )

        # Override
        self.override_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.ctrl, text="Override (sy, sx)", variable=self.override_var).grid(
            row=2, column=6, sticky="e", padx=(12,3), pady=(6,0)
        )
        self.sy_var = tk.StringVar(value="24")
        self.sx_var = tk.StringVar(value="24")
        ttk.Entry(self.ctrl, width=6, textvariable=self.sy_var).grid(row=2, column=7, sticky="w", pady=(6,0))
        ttk.Entry(self.ctrl, width=6, textvariable=self.sx_var).grid(row=2, column=8, sticky="w", padx=(6,0), pady=(6,0))

    # ----------------------------- Helpers -----------------------------
    @staticmethod
    def _clamp_int(txt, lo, hi, default):
        try:
            v = int(float(txt))
        except Exception:
            v = default
        v = max(lo, min(hi, v))
        return v

    def _parse_params(self):
        colors = self._clamp_int(self.colors_var.get(), 2, 128, 64)
        self.colors_var.set(str(colors))

        smin = self._clamp_int(self.smin_var.get(), 2, 128, 12)
        smax = self._clamp_int(self.smax_var.get(), 2, 256, 32)
        if smax < smin:
            smax = smin
        self.smin_var.set(str(smin)); self.smax_var.set(str(smax))

        sample_step = self._clamp_int(self.sample_var.get(), 1, 32, 4)
        self.sample_var.set(str(sample_step))

        bucket = 16
        try:
            bucket = int(self.bucket_var.get())
            if bucket not in (8, 16, 32):
                bucket = 16
        except Exception:
            bucket = 16
        self.bucket_var.set(str(bucket))

        refine = bool(self.refine_var.get())

        override = None
        if self.override_var.get():
            sy = self._clamp_int(self.sy_var.get(), 1, 1_000_000, 24)
            sx = self._clamp_int(self.sx_var.get(), 1, 1_000_000, 24)
            self.sy_var.set(str(sy)); self.sx_var.set(str(sx))
            override = (sy, sx)

        return dict(colors=colors, smin=smin, smax=smax,
                    sample_step=sample_step, bucket=bucket, refine=refine, override=override)

    # ----------------------------- File ops -----------------------------
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            img = Image.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")
            return
        self.input_path = path
        self.orig_img = img.convert("RGBA")
        self._redraw_canvases()  # draw original immediately
        self.btn_save.config(state="disabled")
        self.fixed_img = None
        self.block_size = (None, None)

    def save_image(self):
        if self.fixed_img is None or self.input_path is None or self.block_size[0] is None:
            return
        base, _ = os.path.splitext(self.input_path)
        sy, sx = self.block_size
        out = base + f".pixel_{sy}x{sx}.png"
        try:
            self.fixed_img.save(out)
            messagebox.showinfo("Saved", f"Saved: {out}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")

    # ----------------------------- Processing -----------------------------
    def _trigger_reprocess(self):
        if self.orig_img is None:
            messagebox.showwarning("No image", "Load an image first.")
            return
        params = self._parse_params()
        threading.Thread(
            target=self._process_worker,
            args=(self.orig_img.copy(), params),
            daemon=True
        ).start()

    def _process_worker(self, img, params):
        with self.process_lock:
            try:
                _, (sy, sx), fixed = fix_pixel_art(
                    img,
                    colors=params["colors"],
                    smin=params["smin"], smax=params["smax"],
                    sample_step=params["sample_step"], bucket=params["bucket"], refine=params["refine"],
                    override=params["override"]
                )
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Processing error", str(e)))
                return
            self.fixed_img = fixed
            self.block_size = (sy, sx)
            self.root.after(0, self._redraw_canvases)
            self.root.after(0, lambda: self.btn_save.config(state="normal"))

    # ----------------------------- Rendering -----------------------------
    def _on_frame_configure(self, event):
        # Keep canvases equal halves
        w = max(2, event.width)
        h = max(2, event.height)
        half = (w - 6) // 2  # account for padx
        for cv in (self.left_canvas, self.right_canvas):
            cv.config(width=half, height=h)
        self._redraw_canvases()

    def _fit_and_nearest(self, img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        if img is None or target_w <= 1 or target_h <= 1:
            return None
        w, h = img.width, img.height
        scale = min(target_w / w, target_h / h)
        if scale <= 0:
            scale = 1.0
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return img.resize((new_w, new_h), resample=NEAREST)

    def _draw_on_canvas(self, canvas: tk.Canvas, pil_img: Image.Image, caption: str = ""):
        canvas.delete("all")
        cw = int(canvas.winfo_width()) or 1
        ch = int(canvas.winfo_height()) or 1
        if pil_img is None:
            # Caption only
            canvas.create_text(cw//2, ch//2, text=caption, fill="#aaa")
            return
        disp = self._fit_and_nearest(pil_img, cw, ch)
        if disp is None:
            canvas.create_text(cw//2, ch//2, text=caption, fill="#aaa")
            return
        tk_img = ImageTk.PhotoImage(disp)
        # Centered
        canvas.create_image(cw//2, ch//2, image=tk_img, anchor="center")
        # Keep reference
        if canvas is self.left_canvas:
            self._left_tk = tk_img
        else:
            self._right_tk = tk_img
        if caption:
            canvas.create_text(cw//2, ch-12, text=caption, fill="#ddd")

    def _redraw_canvases(self):
        # Left: original (flattened for consistent preview)
        left_src = flatten_transparency_threshold(self.orig_img) if self.orig_img is not None else None
        self._draw_on_canvas(self.left_canvas, left_src, "Original" if left_src is None else "")

        # Right: processed
        cap = ""
        if self.block_size[0] is not None:
            cap = f"Detected {self.block_size[0]}x{self.block_size[1]}"
        self._draw_on_canvas(self.right_canvas,
                             self.fixed_img.convert("RGB") if self.fixed_img is not None else None,
                             "Processed" if self.fixed_img is None else cap)

def main():
    root = tk.Tk()
    try:
        if hasattr(root, 'tk_call'):
            root.tk.call('tk', 'scaling', 1.0)
    except Exception:
        pass
    app = PixelArtGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
