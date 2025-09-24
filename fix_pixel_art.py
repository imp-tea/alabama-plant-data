#!/usr/bin/env python3
"""
Recover true pixel art from upscaled, messy images.

Pipeline:
  0) Transparency rule:
       - alpha <= 127  → RGB=(0,0,0), alpha := 255
       - alpha >= 128  → keep RGB,  alpha := 255
     Then drop alpha (work in RGB).
  1) Quantize with a diversity-first palette in Lab space (farthest-point sampling,
     optional 1-iter k-medoids refinement). Preserves rare accent colors.
  2) Estimate logical pixel size (sy, sx) by grid-period search on color-change boundaries.
  3) Downscale by (sy, sx) using block-mode (most frequent color) pooling.

Usage:
  python fix_pixel_art.py input.png
  python fix_pixel_art.py input.png --colors 10 --preview 8
  python fix_pixel_art.py input.png --override 24 24 --preview 6
"""

import argparse, math
from PIL import Image
import numpy as np

# Pillow 10+ names, with fallback for older Pillow
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


# ----------------------------- Transparency handling -----------------------------

def flatten_transparency_threshold(img: Image.Image) -> Image.Image:
    """
    Flatten transparency by checking for opaque near-black pixels first:
      - Transparent pixels become black only when no opaque near-black pixels exist.
      - Otherwise transparent pixels become white.
    Then drop alpha and return an RGB image.
    """
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

# ----------------------------- Color / Lab utilities -----------------------------

def _srgb_to_linear(c):
    a = 0.055
    return np.where(c <= 0.04045, c/12.92, ((c + a)/(1 + a))**2.4)

def _rgb_to_lab(rgb8):  # rgb8: (N,3) uint8 in [0,255]
    # sRGB -> XYZ (D65) -> Lab (D65)
    rgb = rgb8.astype(np.float64) / 255.0
    rgb = _srgb_to_linear(rgb)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = rgb @ M.T
    # Normalize by white point (D65)
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


# ----------------------------- Diversity-first posterize -----------------------------

def _sample_uniform_grid(img_np, step=4):
    """Uniform spatial sampling to avoid large regions dominating."""
    h, w, _ = img_np.shape
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pts = img_np[yy, xx]  # (len(ys), len(xs), 3)
    return pts.reshape(-1, 3)

def _dedup_colors(rgb, round_to=16):
    """Bucket RGB to reduce near-duplicates (round_to ∈ {8,16,32})."""
    q = (rgb // round_to) * round_to + round_to//2
    uq, idx = np.unique(q, axis=0, return_index=True)
    return uq[idx.argsort()]

def _farthest_point_palette(cand_lab, K, seed_idx=None, dist_fn=_deltaE76):
    """Greedy k-center: pick farthest new color each step to maximize coverage."""
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
    """One k-medoids update to reduce error without collapsing diversity."""
    palette = cand_lab[palette_idx]
    D = np.stack([dist_fn(cand_lab, p[None,:]) for p in palette], axis=1)  # (N,K)
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
    """
    Quantize with a palette chosen for diversity (broad gamut). Returns a 'P' image.
    """
    img_np = np.array(img_rgb, dtype=np.uint8)

    cand_rgb = _sample_uniform_grid(img_np, step=sample_step)
    cand_rgb = _dedup_colors(cand_rgb, round_to=bucket)
    cand_lab = _rgb_to_lab(cand_rgb)

    K = max(2, int(colors))
    pal_idx = _farthest_point_palette(cand_lab, K)
    if refine:
        pal_idx = _kmedoids_one_iter(cand_lab, pal_idx)
    palette_rgb = cand_rgb[pal_idx]  # (K,3)

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
    paletted.putpalette(pal_list + [0]*(768 - len(pal_list)))  # pad to 256*3
    return paletted


# ----------------------------- Pixel-size estimation & downscale -----------------------------

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
    arr = np.array(pal_img)  # palette indices
    edges_x = (np.diff(arr, axis=1) != 0).astype(np.uint8)  # H x (W-1)
    edges_y = (np.diff(arr, axis=0) != 0).astype(np.uint8)  # (H-1) x W
    col_strength = edges_x.sum(axis=0)  # length W-1
    row_strength = edges_y.sum(axis=1)  # length H-1
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


# ----------------------------- End-to-end pipeline -----------------------------

def fix_pixel_art(img,
                  colors=10,
                  smin=2, smax=64,
                  sample_step=4, bucket=16, refine=True,
                  override=None):
    """
    Full pipeline with transparency thresholding.
    """
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


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Recover true pixel art from upscaled images.")
    ap.add_argument("input", help="Input image (PNG/JPG).")
    ap.add_argument("-o", "--output", default=None, help="Output path for the 1:1 pixel PNG.")
    ap.add_argument("--colors", type=int, default=64, help="Palette size.")
    ap.add_argument("--range", type=int, nargs=2, metavar=("SMIN","SMAX"), default=(12,32),
                    help="Search range (inclusive) for logical pixel size.")
    ap.add_argument("--override", type=int, nargs=2, metavar=("SY","SX"),
                    help="Force block size (height width) instead of detecting.")
    ap.add_argument("--sample-step", type=int, default=4, help="Spatial sampling stride for palette candidates.")
    ap.add_argument("--bucket", type=int, default=16, choices=[8,16,32],
                    help="Pre-bucketing step (higher reduces near-duplicate candidates more).")
    ap.add_argument("--no-refine", action="store_true", help="Disable 1-iter k-medoids refinement.")
    ap.add_argument("--preview", type=int, default=0, help="If >0, also save an Nx preview (nearest).")
    args = ap.parse_args()

    img = Image.open(args.input)

    pal, (sy, sx), fixed = fix_pixel_art(
        img,
        colors=args.colors,
        smin=args.range[0], smax=args.range[1],
        sample_step=args.sample_step, bucket=args.bucket, refine=not args.no_refine,
        override=tuple(args.override) if args.override else None
    )

    out = args.output or (args.input.rsplit(".",1)[0] + f".pixel_{sy}x{sx}.png")
    fixed.save(out)
    print(f"Detected block size: {sy}x{sx} px; saved: {out}")

    if args.preview and args.preview > 0:
        prev = fixed.resize((fixed.width*args.preview, fixed.height*args.preview), resample=NEAREST)
        prev_path = out.rsplit(".",1)[0] + f"_x{args.preview}.png"
        prev.save(prev_path)
        print(f"Preview saved: {prev_path}")

if __name__ == "__main__":
    main()

