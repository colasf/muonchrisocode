"""Muon Bloom FX engine - a small additive CPU renderer for thin-line motion graphics.

Everything is light added on black. Geometry is split in two scalar layers,
"w" (white) and "r" (red), accumulated with bilinear splats so lines and dots
are anti-aliased at sub-pixel precision. Filled rectangles (barcodes, data
blocks) are pixel-snapped. Text is drawn with PIL on matching 8-bit layers.
`Frame.finish()` merges the layers, adds a multi-scale bloom and tonemaps to an
RGB uint8 image (optionally inverted: white field, black lines, red stays red).

All coordinates are given in *design space* (2978 x 1400, the output that covers
the wall); the frame rescales them when rendering smaller previews. A 2D view
transform (zoom around a point) and a screen clip rectangle can be set for
"zoom" cuts.

`with frame.build(age, rect):` makes whatever is drawn inside construct itself
(lines drawn by a pen, marks thrown out, bars growing, text decoded ...) instead
of simply appearing - see build.py. Data never fades in.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import build as _B

DESIGN_W, DESIGN_H = 2978, 1400
LAYERS = ("w", "r")
RED = np.array([1.0, 0.045, 0.035], np.float32)
CHAR_W = 0.61                                 # Space Mono advance, in em

_FONT_DIRS = [
    Path(__file__).resolve().parents[2] / "touchdesigner" / "muonchristo" / "font",
    Path(__file__).resolve().parent,
]


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "SpaceMono-Bold.ttf" if bold else "SpaceMono-Regular.ttf"
    for d in _FONT_DIRS:
        if (d / name).exists():
            return ImageFont.truetype(str(d / name), size)
    return ImageFont.load_default()


def text_w(s: str, size: float) -> float:
    """Width of a string in design px (the font is monospaced)."""
    return len(s) * size * CHAR_W


# ----------------------------------------------------------------------------
# small math helpers
# ----------------------------------------------------------------------------

def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, np.float32) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def periodic_noise(t, T, rng_or_seed, n=6, base=1):
    """Smooth random signal of t that loops exactly with period T."""
    rng = np.random.default_rng(rng_or_seed) if not isinstance(rng_or_seed, np.random.Generator) else rng_or_seed
    k = np.arange(base, base + n)
    amp = rng.uniform(0.4, 1.0, n) / k ** 0.8
    ph = rng.uniform(0, 2 * np.pi, n)
    t = np.asarray(t, np.float64)[..., None]
    return (amp * np.sin(2 * np.pi * k * t / T + ph)).sum(-1) / amp.sum()


def hash01(*keys):
    """Deterministic pseudo-random float in [0, 1) from integers (vectorised)."""
    h = np.uint64(1469598103934665603)
    out = np.full(np.broadcast(*[np.asarray(k) for k in keys]).shape, h, np.uint64)
    for k in keys:
        out ^= np.asarray(k).astype(np.uint64)
        out *= np.uint64(1099511628211)
        out ^= out >> np.uint64(29)
    return (out % np.uint64(1 << 24)).astype(np.float64) / float(1 << 24)


def _clip_param(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    """Vectorised Liang-Barsky: parametric [t0, t1] of each segment inside the rect."""
    dx = x1 - x0
    dy = y1 - y0
    t0 = np.zeros_like(x0)
    t1 = np.ones_like(x0)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
            r = q / p
            t0 = np.where(p < 0, np.maximum(t0, r), t0)
            t1 = np.where(p > 0, np.minimum(t1, r), t1)
            t1 = np.where((p == 0) & (q < 0), -1.0, t1)
    return t0, t1


@lru_cache(maxsize=64)
def _disc(r_q: float):
    """Sub-pixel sample offsets + weights covering an anti-aliased disc of radius r (px)."""
    step = 0.5 if r_q > 1.2 else 0.34
    R = r_q + 0.75
    g = np.arange(-R, R + 1e-6, step, dtype=np.float32)
    ox, oy = np.meshgrid(g, g)
    d = np.hypot(ox, oy)
    cov = np.clip(r_q + 0.5 - d, 0.0, 1.0)
    m = cov > 0
    return ox[m].copy(), oy[m].copy(), (cov[m] * step * step).astype(np.float32)


# ----------------------------------------------------------------------------
# bloom
# ----------------------------------------------------------------------------

def _blur(a):
    """Separable binomial [1 4 6 4 1]/16 blur on the last two axes (edge clamp)."""
    p = np.concatenate([a[..., :1, :], a[..., :1, :], a, a[..., -1:, :], a[..., -1:, :]], axis=-2)
    a = (p[..., 0:-4, :] + p[..., 4:, :] + 4.0 * (p[..., 1:-3, :] + p[..., 3:-1, :]) + 6.0 * p[..., 2:-2, :]) * (1 / 16)
    p = np.concatenate([a[..., :1], a[..., :1], a, a[..., -1:], a[..., -1:]], axis=-1)
    return (p[..., 0:-4] + p[..., 4:] + 4.0 * (p[..., 1:-3] + p[..., 3:-1]) + 6.0 * p[..., 2:-2]) * (1 / 16)


def _down2(a):
    if a.shape[-2] % 2:
        a = np.concatenate([a, a[..., -1:, :]], axis=-2)
    if a.shape[-1] % 2:
        a = np.concatenate([a, a[..., -1:]], axis=-1)
    return 0.25 * (a[..., 0::2, 0::2] + a[..., 1::2, 0::2] + a[..., 0::2, 1::2] + a[..., 1::2, 1::2])


def _up2(a, shape):
    """Bilinear 2x upsample (half-pixel centres), cropped to `shape` (H, W)."""
    prev = np.concatenate([a[..., :1, :], a[..., :-1, :]], axis=-2)
    nxt = np.concatenate([a[..., 1:, :], a[..., -1:, :]], axis=-2)
    a = np.stack([0.75 * a + 0.25 * prev, 0.75 * a + 0.25 * nxt], axis=-2)
    a = a.reshape(*a.shape[:-3], a.shape[-3] * 2, a.shape[-1])
    prev = np.concatenate([a[..., :1], a[..., :-1]], axis=-1)
    nxt = np.concatenate([a[..., 1:], a[..., -1:]], axis=-1)
    a = np.stack([0.75 * a + 0.25 * prev, 0.75 * a + 0.25 * nxt], axis=-1)
    a = a.reshape(*a.shape[:-2], a.shape[-2] * 2)
    return a[..., : shape[0], : shape[1]]


def half_boxes(boxes):
    """Pixel boxes -> the boxes of the half-size level of the bloom that cover them."""
    return [(max(x0, 0) // 2, max(y0, 0) // 2, (max(x1, 0) + 1) // 2, (max(y1, 0) + 1) // 2) for (x0, y0, x1, y1) in boxes]


def bloom(stack, weights, mute=()):
    """Multi-scale glow. `weights[k]` scales pyramid level k+1 (1/2, 1/4, ... res).
    mute = boxes of the half-size level (half_boxes) that have NO GLOW: what is in them is taken out of the
    source of the glow (level 1, before the smaller levels are made of it), and the glow summed at level 1 is
    put out in them before it goes back to full size - nothing in the box glows, nothing glows into it."""
    levels = [stack]
    cur = stack
    for k in range(len(weights)):
        cur = _down2(cur)
        if k == 0:
            for (x0, y0, x1, y1) in mute:
                cur[..., y0:y1, x0:x1] = 0.0
        levels.append(cur)
    acc = None
    for k in range(len(weights), 0, -1):
        lv = _blur(levels[k]) * weights[k - 1]
        acc = lv if acc is None else _up2(acc, lv.shape[-2:]) + lv
    for (x0, y0, x1, y1) in mute:
        acc[..., y0:y1, x0:x1] = 0.0
    return _up2(acc, stack.shape[-2:])


def _tonemap(x, knee=0.72):
    return np.where(x < knee, x, knee + (1 - knee) * (1 - np.exp(-(np.maximum(x, knee) - knee) / (1 - knee))))


# ----------------------------------------------------------------------------
# frame
# ----------------------------------------------------------------------------

class Frame:
    def __init__(self, W: int, H: int):
        self.W, self.H = W, H
        self.s = W / DESIGN_W
        self.acc = {k: np.zeros(W * H, np.float32) for k in LAYERS}
        self._pend = {k: [] for k in LAYERS}
        self._diff = {}
        self._txt = {}
        self._txt_draw = {}
        self._occl = []
        self.post = []              # callables(base, frame) applied to the light layers before the bloom
        self.invert_rects = []      # design-space rects shown inverted (white field, black lines)
        self.noglow_rects = []      # design-space rects without glow (see bloom): the subtitle box
        self._bld = None            # the block being constructed (see build()), or build.MUTE: draw nothing
        self.set_view()
        self.set_clip()

    # -- view / clip ---------------------------------------------------------
    def set_view(self, zoom=1.0, cx=DESIGN_W / 2, cy=DESIGN_H / 2, sx=None, sy=None):
        """2D zoom: design point (cx, cy) lands on screen (sx, sy), scaled by zoom. Identity by default."""
        self.vz, self.vcx, self.vcy = float(zoom), float(cx), float(cy)
        self.vsx = float(cx if sx is None and zoom == 1.0 else (DESIGN_W / 2 if sx is None else sx))
        self.vsy = float(cy if sy is None and zoom == 1.0 else (DESIGN_H / 2 if sy is None else sy))

    def set_clip(self, x0=None, y0=None, x1=None, y1=None):
        """Screen clip rectangle in design pixels (None = whole frame)."""
        if x0 is None:
            self.clip = (-3.0, -3.0, self.W + 2.0, self.H + 2.0)
        else:
            s = self.s
            self.clip = (x0 * s, y0 * s, x1 * s, y1 * s)

    def tx(self, x):
        return ((np.asarray(x, np.float32) - self.vcx) * self.vz + self.vsx) * self.s

    def ty(self, y):
        return ((np.asarray(y, np.float32) - self.vcy) * self.vz + self.vsy) * self.s

    def to_screen(self, x, y):
        """Design-space point -> design screen position under the current view."""
        return ((x - self.vcx) * self.vz + self.vsx, (y - self.vcy) * self.vz + self.vsy)

    def to_world(self, X, Y):
        """Inverse of to_screen."""
        return ((X - self.vsx) / self.vz + self.vcx, (Y - self.vsy) / self.vz + self.vcy)

    def _in_clip(self, px, py):
        c = self.clip
        return (px >= c[0]) & (px <= c[2]) & (py >= c[1]) & (py <= c[3])

    # -- construction ----------------------------------------------------------
    def build(self, age, rect, **kw):
        """Context manager: what is drawn inside constructs itself instead of simply appearing.
            with f.build(t - t_appears, (x0, y0, x1, y1)):
                ...the usual drawing calls...
        age < 0: nothing is drawn (it is not there yet); None or large: drawn as usual. See build.Block for
        the options (wave, flow, line, cps, bars, marks, commit, key) and build.io for blocks that leave."""
        return _B.Block(self, age, rect, **kw)

    @property
    def muted(self):
        """True inside a block that is not there yet (nothing is drawn)."""
        return self._bld is _B.MUTE

    # -- low level ---------------------------------------------------------
    def points(self, layer, x, y, w):
        """Raw splats in *pixel* space."""
        if np.size(x):
            self._pend[layer].append((np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(w, np.float32)))

    def flush(self):
        W, H = self.W, self.H
        for k in LAYERS:
            items = self._pend[k]
            if not items:
                continue
            x = np.concatenate([i[0] for i in items])
            y = np.concatenate([i[1] for i in items])
            w = np.concatenate([i[2] for i in items])
            self._pend[k] = []
            fx0 = np.floor(x)
            fy0 = np.floor(y)
            ix = fx0.astype(np.int64)
            iy = fy0.astype(np.int64)
            m = (ix >= 0) & (ix < W - 1) & (iy >= 0) & (iy < H - 1) & (w != 0)
            if not m.all():
                ix, iy, x, y, w, fx0, fy0 = ix[m], iy[m], x[m], y[m], w[m], fx0[m], fy0[m]
            if not len(ix):
                continue
            fx = x - fx0
            fy = y - fy0
            gx = 1.0 - fx
            gy = 1.0 - fy
            idx = iy * W + ix
            ii = np.concatenate([idx, idx + 1, idx + W, idx + W + 1])
            ww = np.concatenate([w * gx * gy, w * fx * gy, w * gx * fy, w * fx * fy])
            self.acc[k] += np.bincount(ii, weights=ww, minlength=W * H).astype(np.float32)
        for k, D in self._diff.items():
            self.acc[k] += D.cumsum(0).cumsum(1)[: self.H, : self.W].ravel()
        self._diff = {}

    def scale_rect(self, layer, x0, y0, x1, y1, factor):
        """Multiply already-flushed light inside a rect (design space) - used to 'occlude'."""
        if self._bld is _B.MUTE:
            return
        self.flush()
        a = self.acc[layer].reshape(self.H, self.W)
        X0, X1 = int(self.tx(x0)), int(math.ceil(float(self.tx(x1))))
        Y0, Y1 = int(self.ty(y0)), int(math.ceil(float(self.ty(y1))))
        a[max(Y0, 0): max(Y1, 0), max(X0, 0): max(X1, 0)] *= factor

    def occlude(self, x0, y0, x1, y1):
        """Black out everything drawn so far under a rect (design space); later drawing stays."""
        if self._bld is _B.MUTE:
            return
        self.flush()
        X0, Y0 = max(int(self.tx(x0)), 0), max(int(self.ty(y0)), 0)
        X1, Y1 = int(math.ceil(float(self.tx(x1)))), int(math.ceil(float(self.ty(y1))))
        for k in LAYERS:
            self.acc[k].reshape(self.H, self.W)[Y0:max(Y1, 0), X0:max(X1, 0)] = 0.0
        for img in self._txt.values():
            ImageDraw.Draw(img).rectangle((X0, Y0, X1, Y1), fill=0)

    # -- primitives (design space) ------------------------------------------
    def segments(self, layer, x0, y0, x1, y1, i0, i1=None, width=1.0, spacing=0.5):
        """Anti-aliased line segments. i0/i1 = intensity (per px of length) at each end.
        Inside a block being built: long lines are drawn by a pen (bright head), short marks are thrown out."""
        b = self._bld
        if b is None:
            return self._segments(layer, x0, y0, x1, y1, i0, i1, width, spacing)
        if b is _B.MUTE:
            return
        out = b.segs(x0, y0, x1, y1, i0, i1, width)
        if out is None:
            return
        self._segments(layer, *out[:7], spacing)
        if len(out[7]):
            self._dots("w", out[7], out[8], 3.0, 1.7)

    def _segments(self, layer, x0, y0, x1, y1, i0, i1=None, width=1.0, spacing=0.5):
        s = self.s
        x0 = np.atleast_1d(self.tx(x0))
        y0 = np.atleast_1d(self.ty(y0))
        x1 = np.atleast_1d(self.tx(x1))
        y1 = np.atleast_1d(self.ty(y1))
        shape = np.broadcast_shapes(x0.shape, y0.shape, x1.shape, y1.shape, np.shape(i0),
                                    np.shape(i1) if i1 is not None else (), np.shape(width))
        if len(shape) != 1 or shape[0] == 0:
            return
        n = shape[0]
        i0 = np.broadcast_to(np.asarray(i0, np.float32), (n,))
        i1 = i0 if i1 is None else np.broadcast_to(np.asarray(i1, np.float32), (n,))
        x0, y0, x1, y1 = (np.broadcast_to(a, (n,)) for a in (x0, y0, x1, y1))
        width = np.broadcast_to(np.asarray(width, np.float32), (n,))

        live = (i0 > 1e-4) | (i1 > 1e-4)
        c = self.clip
        t0, t1 = _clip_param(x0, y0, x1, y1, c[0], c[1], c[2], c[3])
        live &= t0 <= t1
        if not live.any():
            return
        x0, y0, x1, y1, i0, i1, t0, t1, width = (a[live] for a in (x0, y0, x1, y1, i0, i1, t0, t1, width))
        dx = x1 - x0
        dy = y1 - y0
        cx0, cy0 = x0 + dx * t0, y0 + dy * t0
        cx1, cy1 = x0 + dx * t1, y0 + dy * t1
        ci0, ci1 = i0 + (i1 - i0) * t0, i0 + (i1 - i0) * t1
        dx = cx1 - cx0
        dy = cy1 - cy0
        L = np.hypot(dx, dy)
        cnt = np.maximum(1, np.ceil(L / spacing)).astype(np.int64)
        seg = np.repeat(np.arange(len(cnt)), cnt)
        start = np.repeat(np.cumsum(cnt) - cnt, cnt)
        cntf = cnt.astype(np.float32)
        t = ((np.arange(cnt.sum()) - start).astype(np.float32) + 0.5) / cntf[seg]
        px = cx0[seg] + dx[seg] * t
        py = cy0[seg] + dy[seg] * t
        pw = (ci0[seg] + (ci1 - ci0)[seg] * t) * (L / cntf)[seg] * s ** 0.35
        wide = width[seg] > 1.25
        if wide.any():
            # thick lines: extra passes offset along the normal
            nx = -dy / np.maximum(L, 1e-6)
            ny = dx / np.maximum(L, 1e-6)
            ws = width[seg][wide] * s
            reps = np.maximum(2, np.ceil(ws / 0.6)).astype(np.int64)
            idx = np.repeat(np.nonzero(wide)[0], reps)
            rstart = np.repeat(np.cumsum(reps) - reps, reps)
            k = (np.arange(reps.sum()) - rstart).astype(np.float32)
            rr = np.repeat(reps, reps).astype(np.float32)
            wrep = np.repeat(ws, reps)
            off = (k / (rr - 1) - 0.5) * (wrep - 1.0)
            sg = seg[idx]
            ox = px[idx] + nx[sg] * off
            oy = py[idx] + ny[sg] * off
            ow = pw[idx] * (wrep / rr)
            self.points(layer, ox, oy, ow)
            px, py, pw = px[~wide], py[~wide], pw[~wide]
        self.points(layer, px, py, pw)

    def polyline(self, layer, xs, ys, i, width=1.0, closed=False, i_end=None):
        xs = np.asarray(xs, np.float32)
        ys = np.asarray(ys, np.float32)
        if closed:
            xs = np.append(xs, xs[0])
            ys = np.append(ys, ys[0])
        if len(xs) < 2:
            return
        ia = np.broadcast_to(np.asarray(i, np.float32), (len(xs),))
        if i_end is not None:
            ia = np.linspace(i, i_end, len(xs)).astype(np.float32)
        b = self._bld
        if b is not None:                       # being built: the line is traced from its first point
            if b is _B.MUTE:
                return
            p = b.prog(xs[0], ys[0], 1.25)
            if p <= 0.0:
                return
            if p < 1.0:
                cum = np.r_[0.0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
                target = p * cum[-1]
                k = int(np.clip(np.searchsorted(cum, target, side="right"), 1, len(xs) - 1))
                fr = float((target - cum[k - 1]) / max(cum[k] - cum[k - 1], 1e-6))
                xe, ye = xs[k - 1] + (xs[k] - xs[k - 1]) * fr, ys[k - 1] + (ys[k] - ys[k - 1]) * fr
                xs, ys = np.r_[xs[:k], xe].astype(np.float32), np.r_[ys[:k], ye].astype(np.float32)
                ia = np.r_[ia[:k], ia[k - 1] + (ia[k] - ia[k - 1]) * fr].astype(np.float32)
                self._dots("w", [xe], [ye], 3.0, 1.7)
        self._segments(layer, xs[:-1], ys[:-1], xs[1:], ys[1:], ia[:-1], ia[1:], width=width)

    def rect(self, layer, x0, y0, x1, y1, i, width=1.0):
        if self._bld is not None:               # being built: two pens leave the top-left corner
            self.polyline(layer, [x0, x1, x1], [y0, y0, y1], i, width=width)
            self.polyline(layer, [x0, x0, x1], [y0, y1, y1], i, width=width)
            return
        self.polyline(layer, [x0, x1, x1, x0], [y0, y0, y1, y1], i, width=width, closed=True)

    def crosses(self, layer, cx, cy, half, i, width=1.0):
        cx = np.asarray(cx, np.float32)
        cy = np.asarray(cy, np.float32)
        self.segments(layer, np.r_[cx - half, cx], np.r_[cy, cy - half], np.r_[cx + half, cx], np.r_[cy, cy + half],
                      np.r_[np.broadcast_to(i, cx.shape), np.broadcast_to(i, cx.shape)], width=width)

    def dots(self, layer, x, y, r, i):
        """Filled anti-aliased discs; r in design px (scalar or array), i = brightness."""
        b = self._bld
        if b is None:
            return self._dots(layer, x, y, r, i)
        if b is _B.MUTE:
            return
        out = b.pops(x, y, r, i)
        if out is not None:
            self._dots(layer, *out)

    def _dots(self, layer, x, y, r, i):
        s = self.s
        x = np.atleast_1d(self.tx(x))
        y = np.atleast_1d(self.ty(y))
        n = len(x)
        r = np.broadcast_to(np.asarray(r, np.float32) * s * self.vz ** 0.5, (n,))
        i = np.broadcast_to(np.asarray(i, np.float32), (n,))
        m = (i > 1e-4) & self._in_clip(x, y)
        if not m.any():
            return
        x, y, r, i = x[m], y[m], r[m], i[m]
        rq = np.maximum(0.5, np.round(r * 4) / 4)
        for rv in np.unique(rq):
            sel = rq == rv
            ox, oy, ow = _disc(float(rv))
            xs = (x[sel][:, None] + ox[None, :]).ravel()
            ys = (y[sel][:, None] + oy[None, :]).ravel()
            ws = (i[sel][:, None] * ow[None, :]).ravel()
            self.points(layer, xs, ys, ws)

    def pixels(self, layer, x, y, i, snap=False):
        """Single-pixel points (crisp dot lattices, point clouds)."""
        b = self._bld
        if b is not None:
            if b is _B.MUTE:
                return
            x, y = np.atleast_1d(np.asarray(x, np.float32)), np.atleast_1d(np.asarray(y, np.float32))
            m = b.there(x, y)
            x, y, i = x[m], y[m], (np.broadcast_to(np.asarray(i, np.float32), x.shape)[m] if np.ndim(i) else i)
        x = np.atleast_1d(self.tx(x))
        y = np.atleast_1d(self.ty(y))
        if snap:
            x = np.round(x)
            y = np.round(y)
        i = np.broadcast_to(np.asarray(i, np.float32), x.shape)
        m = self._in_clip(x, y) & (i > 1e-4)
        self.points(layer, x[m], y[m], i[m])

    def rects(self, layer, x0, y0, x1, y1, i):
        """Filled, pixel-snapped rectangles (barcodes, data blocks). Clipped to the clip rect."""
        b = self._bld
        if b is None:
            return self._rects(layer, x0, y0, x1, y1, i)
        if b is _B.MUTE:
            return
        out = b.grow(x0, y0, x1, y1, i)
        if out is not None:
            self._rects(layer, *out)

    def _rects(self, layer, x0, y0, x1, y1, i):
        c = self.clip
        X0 = np.clip(np.round(np.atleast_1d(self.tx(x0))), max(c[0], 0), min(c[2], self.W))
        X1 = np.clip(np.round(np.atleast_1d(self.tx(x1))), max(c[0], 0), min(c[2], self.W))
        Y0 = np.clip(np.round(np.atleast_1d(self.ty(y0))), max(c[1], 0), min(c[3], self.H))
        Y1 = np.clip(np.round(np.atleast_1d(self.ty(y1))), max(c[1], 0), min(c[3], self.H))
        shape = np.broadcast_shapes(X0.shape, X1.shape, Y0.shape, Y1.shape, np.shape(i))
        X0, X1, Y0, Y1 = (np.broadcast_to(a, shape).astype(np.int64) for a in (X0, X1, Y0, Y1))
        i = np.broadcast_to(np.asarray(i, np.float32), shape)
        m = (X1 > X0) & (Y1 > Y0) & (i != 0)
        if not m.any():
            return
        X0, X1, Y0, Y1, i = X0[m], X1[m], Y0[m], Y1[m], i[m]
        D = self._diff.get(layer)
        if D is None:
            D = self._diff[layer] = np.zeros((self.H + 1, self.W + 1), np.float32)
        np.add.at(D, (Y0, X0), i)
        np.add.at(D, (Y0, X1), -i)
        np.add.at(D, (Y1, X0), -i)
        np.add.at(D, (Y1, X1), i)

    def rings(self, layer, cx, cy, radius, i, spacing=0.5, width=1.0):
        """Circles (outline). Arrays of centres/radii."""
        cx = np.atleast_1d(np.asarray(cx, np.float32))
        cy = np.atleast_1d(np.asarray(cy, np.float32))
        radius = np.broadcast_to(np.asarray(radius, np.float32), cx.shape)
        i = np.broadcast_to(np.asarray(i, np.float32), cx.shape)
        segs = np.maximum(24, (radius * self.s * self.vz * 0.5).astype(int))
        b = self._bld
        if b is _B.MUTE:
            return
        xs0, ys0, xs1, ys1, ii, hx, hy = [], [], [], [], [], [], []
        for c_x, c_y, rr, nn, iv in zip(cx, cy, radius, segs, i):
            a = np.linspace(0, 2 * np.pi, nn + 1)
            if b is not None:                   # being built: the circle is traced from its top
                p = b.prog(c_x, c_y, 1.25)
                if p <= 0.0:
                    continue
                if p < 1.0:
                    a = -0.5 * np.pi + np.linspace(0, 2 * np.pi * p, max(2, int(nn * p) + 1))
                    hx.append(c_x + rr * np.cos(a[-1])); hy.append(c_y + rr * np.sin(a[-1]))
            px = c_x + rr * np.cos(a)
            py = c_y + rr * np.sin(a)
            xs0.append(px[:-1]); ys0.append(py[:-1]); xs1.append(px[1:]); ys1.append(py[1:])
            ii.append(np.full(len(a) - 1, iv, np.float32))
        if xs0:
            self._segments(layer, np.concatenate(xs0), np.concatenate(ys0), np.concatenate(xs1), np.concatenate(ys1),
                           np.concatenate(ii), width=width, spacing=spacing)
        if hx and len(hx) <= _B.Block.HEADS:
            self._dots("w", hx, hy, 3.0, 1.7)

    # -- text ----------------------------------------------------------------
    def _text_layer(self, layer):
        if layer not in self._txt:
            self._txt[layer] = Image.new("L", (self.W, self.H), 0)
            self._txt_draw[layer] = ImageDraw.Draw(self._txt[layer])
        return self._txt[layer], self._txt_draw[layer]

    def text(self, layer, x, y, s, size=22, alpha=1.0, anchor="ls", bold=False):
        """Text (Space Mono). Inside a block being built it is decoded out of noise."""
        b = self._bld
        if b is not None and s:
            if b is _B.MUTE:
                return
            w = text_w(s, size)
            s = b.text(s, x - (0.0 if anchor[0] == "l" else w if anchor[0] == "r" else 0.5 * w), y, anchor[0] != "l")
            if not s.strip():
                return
        self._text(layer, x, y, s, size, alpha, anchor, bold)

    def _text(self, layer, x, y, s, size=22, alpha=1.0, anchor="ls", bold=False):
        if alpha <= 0.004 or not s:
            return
        X, Y = float(self.tx(x)), float(self.ty(y))
        if not self._in_clip(X, Y):
            return
        _, d = self._text_layer(layer)
        d.text((X, Y), s, font=font(max(6, int(round(size * self.s))), bold), fill=int(255 * min(alpha, 1.0)),
               anchor=anchor)

    def text_lines(self, layer, x, y, lines, size=14, lead=1.25, alpha=1.0, anchor="ls"):
        for k, ln in enumerate(lines):
            self.text(layer, x, y + k * size * lead / max(self.vz, 1e-6), ln, size=size, alpha=alpha, anchor=anchor)

    def dim(self, x0, y0, x1, y1, factor):
        """Multiply everything drawn so far (both layers and their text) inside a rect (design space)."""
        if self._bld is _B.MUTE:
            return
        self.flush()
        X0, Y0 = max(int(self.tx(x0)), 0), max(int(self.ty(y0)), 0)
        X1, Y1 = max(int(math.ceil(float(self.tx(x1)))), 0), max(int(math.ceil(float(self.ty(y1)))), 0)
        for k in LAYERS:
            self.acc[k].reshape(self.H, self.W)[Y0:Y1, X0:X1] *= factor
        for img in self._txt.values():
            box = (X0, Y0, min(X1, self.W), min(Y1, self.H))
            if box[2] > box[0] and box[3] > box[1]:
                img.paste(img.crop(box).point(lambda v: int(v * factor)), box)

    def tag(self, layer, x, y, s, size=18, alpha=1.0, anchor="ls", pad=5, bold=False, ref=None, wipe=1.0):
        """Inverted label: solid box with the text cut out (Ikeda-style data tag).
        ref = the string that sizes the box (default: `s`): a tag being written keeps its final box while its
        letters change (`s` must then have the length of `ref`, or be left-anchored); wipe < 1 draws only
        that fraction of the box, from the left. Returns the box (design px on screen), or None.
        Inside a block being built the box is pushed out and the letters are decoded behind its edge."""
        b = self._bld
        if b is not None and s and ref is None:
            if b is _B.MUTE:
                return None
            w = text_w(s, size)
            txt, wp = b.tag(s, x - (0.0 if anchor[0] == "l" else w if anchor[0] == "r" else 0.5 * w), y)
            return self._tag(layer, x, y, txt, size, alpha, anchor, pad, bold, ref=s, wipe=min(wp, wipe))
        if b is _B.MUTE:
            return None
        return self._tag(layer, x, y, s, size, alpha, anchor, pad, bold, ref, wipe)

    def _tag(self, layer, x, y, s, size=18, alpha=1.0, anchor="ls", pad=5, bold=False, ref=None, wipe=1.0):
        if alpha <= 0.004 or not (s or ref) or wipe <= 0.0:
            return None
        X, Y = float(self.tx(x)), float(self.ty(y))
        if not self._in_clip(X, Y):
            return None
        _, d = self._text_layer(layer)
        f = font(max(6, int(round(size * self.s))), bold)
        l, t, r, b = d.textbbox((X, Y), ref or s, font=f, anchor=anchor)
        p = pad * self.s
        box = (l - p, t - p, l - p + (r - l + 2 * p) * min(wipe, 1.0), b + p)
        d.rectangle(box, fill=int(255 * min(alpha, 1.0)))
        if s:
            d.text((X, Y), s, font=f, fill=0, anchor=anchor)
        other = "w" if layer == "r" else "r"
        if other in self._txt:
            self._txt_draw[other].rectangle(box, fill=0)
        self._occl.append((int(box[0]), int(box[1]), int(math.ceil(box[2])), int(math.ceil(box[3]))))
        return tuple(v / self.s for v in box)

    def text_vertical(self, layer, x, y, s, size=22, alpha=1.0):
        """Text rotated 90 deg counter-clockwise, (x, y) = bottom-left corner of the rotated text."""
        b = self._bld
        if b is not None and s:
            if b is _B.MUTE:
                return
            s = b.text(s, x, y, False)
        if alpha <= 0.004 or not s.strip():
            return
        img, _ = self._text_layer(layer)
        sc = self.s
        f = font(max(6, int(round(size * sc))))
        l, t, r, b = f.getbbox(s)
        tmp = Image.new("L", (r - l + 2, b - t + 2), 0)
        ImageDraw.Draw(tmp).text((-l + 1, -t + 1), s, font=f, fill=int(255 * min(alpha, 1.0)))
        tmp = tmp.rotate(90, expand=True)
        img.paste(255, (int(self.tx(x)), int(self.ty(y)) - tmp.size[1]), tmp)

    # -- output --------------------------------------------------------------
    def pixel_boxes(self, rects):
        """Design-space rects -> the pixel boxes that cover them, cut to the picture."""
        s = self.s
        return [(max(int(x0 * s), 0), max(int(y0 * s), 0), min(int(math.ceil(x1 * s)), self.W), min(int(math.ceil(y1 * s)), self.H))
                for (x0, y0, x1, y1) in rects]

    def finish(self, bloom_weights=(0.5, 0.45, 0.38, 0.34, 0.3, 0.28, 0.24, 0.2), bloom_gain=1.0,
               text_gain=1.0, exposure=1.0, invert=False, invert_rect=None, dither_seed=3, palette=None,
               palette_mix=1.0):
        """`palette` = (white_rgb, red_rgb) recolours the two layers, each channel tonemapped on its own so hot
        spots burn to white; `palette_mix` fades from the classic white / red look (0) to it (1)."""
        self.flush()
        H, W = self.H, self.W
        base = np.stack([self.acc[k].reshape(H, W) for k in LAYERS]) * exposure
        for (x0, y0, x1, y1) in self._occl:
            base[:, max(y0, 0): max(y1, 0), max(x0, 0): max(x1, 0)] = 0.0
        for j, k in enumerate(LAYERS):
            if k in self._txt:
                base[j] += np.asarray(self._txt[k], np.float32) * (text_gain / 255.0)
        for fn in self.post:                    # glitch-type effects: fn(base (2, H, W) float32, frame) -> base
            base = fn(base, self)
        mute = half_boxes(self.pixel_boxes(self.noglow_rects))
        light = base + bloom(base, bloom_weights, mute) * bloom_gain if bloom_gain else base
        wt = _tonemap(light[0])
        rt = _tonemap(light[1])
        rgb = np.empty((H, W, 3), np.float32)
        for c in range(3):
            rgb[..., c] = wt + rt * RED[c]
        if palette is not None and palette_mix > 0:
            wcol, rcol = palette
            for c in range(3):
                pc = _tonemap(light[0] * wcol[c] + light[1] * rcol[c])
                rgb[..., c] += (pc - rgb[..., c]) * min(float(palette_mix), 1.0)
        rects = list(self.invert_rects)
        if invert:
            rects.append(invert_rect if invert_rect else (0, 0, DESIGN_W, DESIGN_H))
        for (x0, y0, x1, y1) in rects:
            # white field, black lines, red stays red (inside a design-space rect)
            ys = slice(max(int(y0 * self.s), 0), min(int(math.ceil(y1 * self.s)), H))
            xs = slice(max(int(x0 * self.s), 0), min(int(math.ceil(x1 * self.s)), W))
            w_, r_ = wt[ys, xs], rt[ys, xs]
            rgb[ys, xs, 0] = 1.0 - w_
            rgb[ys, xs, 1] = 1.0 - w_ - r_
            rgb[ys, xs, 2] = 1.0 - w_ - r_
        np.clip(rgb, 0.0, 1.0, out=rgb)
        rgb *= 255.0
        rgb += _dither(H, W, dither_seed)
        np.clip(rgb, 0, 255, out=rgb)
        return rgb.astype(np.uint8)


@lru_cache(maxsize=4)
def _dither(H, W, seed):
    rng = np.random.default_rng(seed)
    return (rng.random((H, W, 1), dtype=np.float32) - 0.5)


# ----------------------------------------------------------------------------
# cameras
# ----------------------------------------------------------------------------

class Camera:
    def __init__(self, pos, target, fov_deg=40.0, up=(0.0, 1.0, 0.0), W=DESIGN_W, H=DESIGN_H, roll_deg=0.0,
                 screen_center=None):
        pos = np.asarray(pos, np.float64)
        target = np.asarray(target, np.float64)
        f = target - pos
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(up, np.float64))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        if roll_deg:
            a = math.radians(roll_deg)
            r, u = r * math.cos(a) + u * math.sin(a), u * math.cos(a) - r * math.sin(a)
        self.pos = pos
        self.R = np.stack([r, u, f]).astype(np.float32)
        self.W, self.H = W, H
        self.focal = (H / 2) / math.tan(math.radians(fov_deg) / 2)
        self.cx, self.cy = screen_center if screen_center else (W / 2, H / 2)
        self.near = 0.05
        self.ortho = False

    def project(self, P):
        """P (..., 3) world -> (sx, sy, depth, ok) in design pixels."""
        d = (np.asarray(P, np.float32) - self.pos.astype(np.float32)) @ self.R.T
        z = d[..., 2]
        ok = z > self.near
        zz = np.where(ok, z, 1.0)
        sx = self.cx + self.focal * d[..., 0] / zz
        sy = self.cy - self.focal * d[..., 1] / zz
        return sx.astype(np.float32), sy.astype(np.float32), z.astype(np.float32), ok


class OrthoCamera:
    """Orthographic view: `scale` design px per world unit, centred on `center` (screen cx, cy)."""

    def __init__(self, pos, target, scale, up=(0.0, 1.0, 0.0), W=DESIGN_W, H=DESIGN_H, screen_center=None,
                 fake_depth=35.0, roll_deg=0.0):
        pos = np.asarray(pos, np.float64)
        target = np.asarray(target, np.float64)
        f = target - pos
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(up, np.float64))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        if roll_deg:
            a = math.radians(roll_deg)
            r, u = r * math.cos(a) + u * math.sin(a), u * math.cos(a) - r * math.sin(a)
        self.pos = target.astype(np.float64)
        self.R = np.stack([r, u, f]).astype(np.float32)
        self.scale = scale
        self.cx, self.cy = screen_center if screen_center else (W / 2, H / 2)
        self.fake_depth = fake_depth
        self.ortho = True

    def project(self, P):
        d = (np.asarray(P, np.float32) - self.pos.astype(np.float32)) @ self.R.T
        sx = self.cx + self.scale * d[..., 0]
        sy = self.cy - self.scale * d[..., 1]
        z = np.full(sx.shape, self.fake_depth, np.float32) + 0.0 * d[..., 2]
        ok = np.ones(sx.shape, bool)
        return sx.astype(np.float32), sy.astype(np.float32), z, ok
