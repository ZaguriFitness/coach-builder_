"""Anatomical surface texture from the muscle-ID colour map.

The source model's colour map paints every muscle in its own pastel colour, with light grey / ivory for
tendons, fascia and bone. This turns it into an anatomy-atlas look: muscles in deep red with a slightly
different shade per muscle (so neighbours stay apart), a darker groove along every muscle's edge (the
separation between muscles), fine fibre grain, and ivory tendons and fascia.

    python3 texture.py ../../model/anatomy_color.jpg ../../model/anatomy_real.jpg

Needs numpy and Pillow.
"""
import sys
import numpy as np
from PIL import Image, ImageFilter


def smooth(x, e0, e1):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def main(src, out):
    c = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx, mn = c.max(-1), c.min(-1)
    sat = mx - mn
    used = mx > 0.12                                   # the black outside the UV islands stays black
    yellow = (r > b + 0.08) & (g > b + 0.06) & (np.abs(r - g) < 0.14)   # ivory: tendons, bone
    muscle = smooth(sat, 0.06, 0.15) * (~yellow) * used

    # hue → a per-muscle shade, so two neighbouring muscles never look like one
    h = np.zeros_like(r)
    d = np.where(sat > 1e-4, sat, 1)
    h = np.where(mx == r, ((g - b) / d) % 6, h)
    h = np.where(mx == g, (b - r) / d + 2, h)
    h = np.where(mx == b, (r - g) / d + 4, h)
    h = h / 6
    shade = 0.84 + 0.16 * np.sin(h * 2 * np.pi * 3 + 0.6)
    warm = 0.5 + 0.5 * np.cos(h * 2 * np.pi * 2)

    # separation: distance into the muscle from its edge (eroded masks), darker in the groove
    m8 = Image.fromarray((muscle * 255).astype(np.uint8))
    inner = np.zeros_like(r)
    for k in (3, 7, 11, 17):
        inner += np.asarray(m8.filter(ImageFilter.MinFilter(k)), dtype=np.float32) / 255
    inner /= 4
    # edges between two touching muscles (no grey seam drawn between them): colour change in the map
    blur = np.asarray(Image.fromarray((c * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), dtype=np.float32) / 255
    seam = smooth(np.abs(c - blur).sum(-1), 0.04, 0.16)

    # fine fibre grain
    rng = np.random.default_rng(7)
    noise = rng.random(r.shape).astype(np.float32)
    grain = np.asarray(Image.fromarray((noise * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32) / 255

    red = np.stack([0.60 + 0.06 * warm, 0.11 + 0.03 * warm, 0.10 + 0.02 * (1 - warm)], -1)
    lit = (0.62 + 0.38 * inner) * shade * (1 - 0.35 * seam) * (0.93 + 0.14 * grain)
    mus = red * lit[..., None]
    tendon = np.stack([0.88, 0.84, 0.77]) * (0.94 + 0.08 * grain)[..., None] * (0.9 + 0.1 * (mx - 0.6)[..., None])
    rgb = mus * muscle[..., None] + tendon * (1 - muscle)[..., None]
    # pad the islands outward over the unused black, so mipmapping never pulls black into the seams
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    filled = used.copy()
    out_rgb = np.asarray(img, dtype=np.float32)
    for _ in range(6):
        grown = np.asarray(Image.fromarray((out_rgb * filled[..., None]).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)), dtype=np.float32)
        cover = np.asarray(Image.fromarray((filled * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))) > 0
        new = cover & ~filled
        out_rgb[new] = grown[new]
        filled = filled | cover
    rgb = out_rgb / 255
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out, quality=88, optimize=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
