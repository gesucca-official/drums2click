"""Draw the GUI graphics and write them to assets/:

- icon.ico / icon.png / icon_48.png: a metronome with click waves
- check_{on,off}_{light,dark}.png, radio_{on,off}_{light,dark}.png: checkbox and radio indicators

No image library needed: shapes are rendered from signed distance fields with numpy, and the
PNG/ICO files are encoded by hand.

    python tools/make_assets.py
"""
import struct
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ICON_SIZES = [256, 128, 64, 48, 32, 24, 16]
# indicator colors, matching gui.py's LIGHT / DARK palettes
THEMES = {"light": dict(field="#f6f6f9", border="#b9bcc9", accent="#f0602c", mark="#ffffff"),
          "dark": dict(field="#2a2d38", border="#5b5f70", accent="#ff7a3d", mark="#1a1206")}


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255


class Canvas:
    """RGBA float image with signed-distance-field shapes, in pixels of the render size."""

    def __init__(self, size):
        self.size = size
        self.Y, self.X = np.mgrid[0:size, 0:size].astype(float) + 0.5
        self.img = np.zeros((size, size, 4))

    def paint(self, d, color):
        a = np.clip(0.5 - d, 0, 1)[..., None]
        self.img[..., :3] = self.img[..., :3] * (1 - a) + color * a
        self.img[..., 3:] = self.img[..., 3:] * (1 - a) + a

    def round_box(self, cx, cy, hw, hh, r):
        qx, qy = np.abs(self.X - cx) - hw + r, np.abs(self.Y - cy) - hh + r
        return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r

    def convex(self, points, r=0.0):
        """Convex polygon (clockwise in screen coordinates), corners rounded by r."""
        d = np.full(self.X.shape, -np.inf)
        pts = np.array(points, dtype=float)
        for (x0, y0), (x1, y1) in zip(pts, np.roll(pts, -1, axis=0)):
            ex, ey = x1 - x0, y1 - y0
            d = np.maximum(d, ((self.X - x0) * ey - (self.Y - y0) * ex) / np.hypot(ex, ey))
        return d - r

    def segment(self, x0, y0, x1, y1, r):
        px, py, ex, ey = self.X - x0, self.Y - y0, x1 - x0, y1 - y0
        h = np.clip((px * ex + py * ey) / (ex * ex + ey * ey), 0, 1)
        return np.hypot(px - ex * h, py - ey * h) - r

    def circle(self, cx, cy, r):
        return np.hypot(self.X - cx, self.Y - cy) - r

    def arc(self, cx, cy, radius, width, a0, a1):
        """Arc between angles a0..a1 (degrees, 0 = right, counter-clockwise), round caps."""
        ang = np.degrees(np.arctan2(cy - self.Y, self.X - cx))
        ring = np.abs(np.hypot(self.X - cx, self.Y - cy) - radius) - width / 2
        caps = [self.circle(cx + radius * np.cos(np.radians(a)), cy - radius * np.sin(np.radians(a)), width / 2)
                for a in (a0, a1)]
        return np.where((ang >= a0) & (ang <= a1), ring, np.minimum(*caps))

    def downsample(self, size):
        f = self.size // size
        pre = self.img.copy()
        pre[..., :3] *= pre[..., 3:]
        small = pre.reshape(size, f, size, f, 4).mean(axis=(1, 3))
        a = small[..., 3:]
        small[..., :3] = np.where(a > 0, small[..., :3] / np.maximum(a, 1e-9), 0)
        return (small.clip(0, 1) * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- icon

def draw_icon():
    big = 1536  # a multiple of every icon size
    c = Canvas(big)
    s = big / 256  # designed on a 256 grid
    t = (c.Y / big)[..., None]
    c.paint(c.round_box(128 * s, 128 * s, 120 * s, 120 * s, 52 * s),
            hex_rgb("#40357a") * (1 - t) + hex_rgb("#1a1838") * t)

    t = ((c.Y / s - 50) / 156).clip(0, 1)[..., None]
    body = [(98 * s, 50 * s), (158 * s, 50 * s), (190 * s, 206 * s), (66 * s, 206 * s)]
    c.paint(c.convex(body, 10 * s), hex_rgb("#ffb347") * (1 - t) + hex_rgb("#ff6a3d") * t)
    face = [(108 * s, 70 * s), (148 * s, 70 * s), (166 * s, 170 * s), (90 * s, 170 * s)]
    c.paint(c.convex(face, 6 * s), hex_rgb("#2a2350") * 0.9 + 0.02)
    c.paint(c.round_box(128 * s, 196 * s, 70 * s, 12 * s, 10 * s), hex_rgb("#e0552f"))

    px, py, tx, ty = 128 * s, 168 * s, 160 * s, 64 * s
    c.paint(c.segment(px, py, tx, ty, 6 * s), hex_rgb("#ffffff"))
    wx, wy = px + (tx - px) * 0.62, py + (ty - py) * 0.62
    c.paint(c.round_box(wx, wy, 13 * s, 9 * s, 5 * s), hex_rgb("#ffd27a"))
    c.paint(c.circle(px, py, 9 * s), hex_rgb("#ffffff"))
    for radius in (26, 46):
        c.paint(c.arc(tx, ty, radius * s, 9 * s, -40, 40), hex_rgb("#ffffff"))
    return c


# ---------------------------------------------------------------- indicators

IND = 18   # indicator size in the GUI
GAP = 8    # transparent space between indicator and label


def draw_indicator(kind, on, colors):
    f = 8
    c = Canvas(IND * f)
    s = f
    mid = IND * f / 2
    if kind == "check":
        if on:
            c.paint(c.round_box(mid, mid, 8 * s, 8 * s, 4 * s), hex_rgb(colors["accent"]))
            mark = hex_rgb(colors["mark"])
            c.paint(np.minimum(c.segment(4.8 * s, 9.4 * s, 7.6 * s, 12.2 * s, 1.1 * s),
                               c.segment(7.6 * s, 12.2 * s, 13.2 * s, 6.0 * s, 1.1 * s)), mark)
        else:
            c.paint(c.round_box(mid, mid, 8 * s, 8 * s, 4 * s), hex_rgb(colors["border"]))
            c.paint(c.round_box(mid, mid, 6.6 * s, 6.6 * s, 3 * s), hex_rgb(colors["field"]))
    else:
        if on:
            c.paint(c.circle(mid, mid, 8 * s), hex_rgb(colors["accent"]))
            c.paint(c.circle(mid, mid, 3.2 * s), hex_rgb(colors["mark"]))
        else:
            c.paint(c.circle(mid, mid, 8 * s), hex_rgb(colors["border"]))
            c.paint(c.circle(mid, mid, 6.6 * s), hex_rgb(colors["field"]))
    small = c.downsample(IND)
    out = np.zeros((IND, IND + GAP, 4), dtype=np.uint8)
    out[:, :IND] = small
    return out


# ---------------------------------------------------------------- encoding

def png_bytes(rgba):
    h, w = rgba.shape[:2]
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def ico_bytes(images):
    """ICO with PNG-compressed entries (supported since Windows Vista)."""
    header = struct.pack("<HHH", 0, 1, len(images))
    entries, data = b"", b""
    offset = 6 + 16 * len(images)
    for size, png in images:
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
    return header + entries + data


def main():
    ASSETS.mkdir(exist_ok=True)
    icon = draw_icon()
    pngs = {n: png_bytes(icon.downsample(n)) for n in ICON_SIZES}
    (ASSETS / "icon.png").write_bytes(pngs[256])
    (ASSETS / "icon_48.png").write_bytes(pngs[48])
    (ASSETS / "icon.ico").write_bytes(ico_bytes(list(pngs.items())))
    for theme, colors in THEMES.items():
        for kind in ("check", "radio"):
            for on in (True, False):
                name = f"{kind}_{'on' if on else 'off'}_{theme}.png"
                (ASSETS / name).write_bytes(png_bytes(draw_indicator(kind, on, colors)))
    print("Wrote", len(list(ASSETS.glob("*.png"))) + 1, "files to", ASSETS.relative_to(ROOT))


if __name__ == "__main__":
    main()
