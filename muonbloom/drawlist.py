"""Draw list: a Frame that records what the scenes ask it to draw, instead of rasterising it.

The realtime engine (engine/) is a GPU renderer for these lists: the scenes stay in Python, run every frame
on a `DrawList` instead of an `engine.Frame`, and what they drew is handed over as one flat binary blob.

    with recording():                    # show.Frame -> DrawList for the time of the block
        dl = Show().render(t)            # the scene, the towers, the HUD: everything, nothing rasterised
    blob = dl.pack()                     # bytes: the format below, the same in a file and in shared memory

`DrawList` overrides exactly the raw primitives of engine.Frame (engine/BRIEF.md, section 5). Everything
above them - build-ups, polylines, rings, crosses, the pixel lattices - still runs in engine.py / build.py,
so a tweak of the look that does not touch the raw primitives needs nothing here.

What is recorded, and what is left to the renderer:

    segments, dots, rects    design-space values + the view / clip they were drawn under (a `state`); the
                             renderer transforms, clips and rasterises them
    splats                   `Frame.points`: single bilinear splats, already in pixel space (the lattices and
                             point clouds of `Frame.pixels` end here)
    light ops                occlude / dim / scale_rect: a pixel rect that multiplies what was drawn BEFORE it.
                             Each one remembers how many segments, dots, splats and rects came before it; in
                             between, the geometry is purely additive and is drawn in one go
    text ops                 ordered list of 'paints' on the two 8-bit text layers: a solid rect, or a run of
                             glyphs.  dst = dst + (colour - dst) * factor  per layer, as PIL does. Text is
                             snapped to whole pixels exactly as PIL snaps it, so the glyphs are PIL's own
                             bitmaps: every glyph a frame uses for the first time travels with it (`glyphs`)
    occl                     the boxes of the tags: geometry under them is zeroed at the very end
    invert, post, options    what Frame.finish needs

`replay(blob)` rasterises a blob on the CPU with the code of engine.Frame: it is the specification of the
format, and the check that a recorded frame is the frame (engine/tools/check_drawlist.py).

File of several frames: the blobs one after the other (each one starts with its size), see `write_frames` /
`read_frames`.
"""
from __future__ import annotations

import contextlib
import inspect
import math
import struct

import numpy as np

from . import build as _B
from . import engine as _E

MAGIC = b"MBDL"
VERSION = 1
HEADER = 256
_HEAD = struct.Struct("<4sIIIdIIffff8fII")        # magic, version, bytes, flags, t, W, H, s, exposure, bloom_gain,
#                                                   text_gain, bloom weights (8), dither seed, frame number
_SEC0 = 96                                          # the section table starts here: (offset, count, bytes) each
(S_STATES, S_SEGS, S_DOTS, S_SPLATS, S_RECTS, S_LIGHTOPS, S_OCCL, S_TEXTOPS, S_CHARS, S_GLYPHS, S_INVERT,
 S_POSTOPS) = range(12)
NSEC = 12

FLAG_POST_UNKNOWN = 1          # the frame had a `post` callable the recorder does not know: it is not in the list
FLAG_PALETTE = 2               # finish() was asked for a palette (not used by the show): ignored

STATE_F, SEG_F, DOT_F, SPLAT_F, RECT_F = 12, 9, 5, 4, 6       # float32 per record
# state : vz, vcx, vcy, vsx, vsy, 0, clip x0, y0, x1, y1 (pixels), 0, 0
# seg   : x0, y0, x1, y1, i0, i1, width, spacing, meta   meta = 2 * state + layer (0 = "w", 1 = "r")
#         (spacing = distance between the splats of the reference, in pixels. Up to 0.5, the default, the
#          renderer integrates the line; above, it draws the same splats one by one: their grain shows)
# dot   : x, y, r, i, meta
# splat : x, y, w, layer                                (pixel space)
# rect  : x0, y0, x1, y1, i, meta
LIGHTOP = np.dtype([("x0", "<i4"), ("y0", "<i4"), ("x1", "<i4"), ("y1", "<i4"), ("fw", "<f4"), ("fr", "<f4"),
                    ("nseg", "<u4"), ("ndot", "<u4"), ("nsplat", "<u4"), ("nrect", "<u4")])
TEXTOP = np.dtype([("kind", "<u4"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4"), ("d", "<i4"),
                   ("cw", "<f4"), ("cr", "<f4"), ("fw", "<f4"), ("fr", "<f4"),
                   ("off", "<u4"), ("cnt", "<u4"), ("pad", "<u4")])
T_RECT, T_RUN, T_VRUN = 0, 1, 2
# T_RECT : pixels [a, c) x [b, d)
# T_RUN  : glyph k of the run has its origin (left, baseline) at pixel (a + k * c, b); c = advance, d = font key
# T_VRUN : the same run turned 90 degrees counter-clockwise: texel (gx, gy) of glyph k, whose bitmap sits at
#          (ox, oy) from its origin, lands on pixel (a + oy + gy, b - 1 - k * c - ox - gx)
# cw, cr = colour painted on the white / red text layer (0..1), fw, fr = how much of it (times the glyph)
BOX = np.dtype([("x0", "<i4"), ("y0", "<i4"), ("x1", "<i4"), ("y1", "<i4")])
_GLYPH = struct.Struct("<IIiiiII")                  # font key, code point, advance, ox, oy, w, h  + w * h bytes
POSTOP = np.dtype([("kind", "<u4"), ("x0", "<i4"), ("y0", "<i4"), ("x1", "<i4"), ("y1", "<i4"),
                   ("p0", "<f4"), ("p1", "<f4"), ("p2", "<f4")])
(P_ROLL_X, P_REPEAT, P_SMEAR_Y, P_SMEAR_X, P_TO_RED, P_TO_WHITE, P_ROLL_Y) = range(1, 8)
# P_ROLL_X  : rect rolled by p0 pixels along x (wraps inside the rect)
# P_REPEAT  : the rect [x0, x1) x [y0, y1) is stamped p0 times more, p1 * its width apart (p1 = +-1), each
#             copy times p2 ** k; stops at the frame edge
# P_SMEAR_Y : pixels brighter than p0 drag down, decaying p2 per pixel, gain p1 (max with what is there)
# P_SMEAR_X : the same along x; p1 < 0: towards the left (gain -p1)
# P_TO_RED / P_TO_WHITE : one layer of the rect is added to the other, and emptied
# P_ROLL_Y  : rect rolled by p0 pixels along y

LAYER = {"w": 0, "r": 1}


# ----------------------------------------------------------------------------
# fonts: PIL's glyphs and metrics, asked once per glyph
# ----------------------------------------------------------------------------

class Font:
    """Space Mono at one pixel size: the bitmaps and the metrics PIL (FreeType) gives, one glyph at a time.
    A string set by PIL is its glyphs side by side, `adv` pixels apart (checked in check_drawlist.py)."""

    def __init__(self, px, bold):
        self.px, self.bold = int(px), bool(bold)
        self.key = self.px | (0x8000 if bold else 0)
        self.pil = _E.font(self.px, self.bold)
        self.ascent = self.pil.getmetrics()[0]
        self.g = {}                                 # char -> (adv, ox, oy, w, h, bitmap bytes, bbox l, t, r, b)
        self.mono = {"0"}                           # the chars whose advance is `adv` (all of them, in principle)
        self.adv = None
        self.adv = self.glyph("0")[0]               # the advance of the font: Space Mono has only one
        self._bbox = {}
        self._dy = {"s": 0, "a": self.ascent, "d": -self.pil.getmetrics()[1],
                    "m": self.pil.getmask2("A", "L", anchor="lm")[1][1] - self.pil.getmask2("A", "L", anchor="ls")[1][1]}

    def dy(self, v, s):
        """Pixels between the baseline and the point a vertical anchor refers to (PIL: a, t, m, s, b, d)."""
        d = self._dy.get(v)
        if d is None:                               # t / b: the top / bottom of the ink of this string
            bb = self.bbox(s)
            d = -bb[1] if v == "t" else -bb[3]
        return d

    def glyph(self, ch):
        g = self.g.get(ch)
        if g is None:
            mask, (ox, oy) = self.pil.getmask2(ch, "L", anchor="ls")
            w, h = mask.size
            l, t, r, b = self.pil.getbbox(ch, anchor="ls")
            adv = int(round(self.pil.getlength(ch)))
            g = self.g[ch] = (adv, int(ox), int(oy), w, h, bytes(mask) if w and h else b"", l, t, r, b)
            if adv == self.adv:
                self.mono.add(ch)
        return g

    def has(self, s):
        """True if every char of s has the advance of the font (and makes sure its glyphs are known)."""
        if self.mono.issuperset(s):
            return True
        for ch in set(s):
            self.glyph(ch)
        return self.mono.issuperset(s)

    def bbox(self, s):
        """PIL's getbbox(s, anchor='ls'), from the origin of the first glyph, and the advance of the whole
        string: (left, top, right, bottom, advance)."""
        bb = self._bbox.get(s)
        if bb is None:
            x = l = t = r = b = 0
            for ch in s:
                g = self.glyph(ch)
                l, t, r, b = min(l, x + g[6]), min(t, g[7]), max(r, x + g[8]), max(b, g[9])
                x += g[0]
            if len(self._bbox) > 20000:
                self._bbox.clear()
            bb = self._bbox[s] = (l, t, r, b, x)
        return bb


_FONTS = {}


def get_font(px, bold=False):
    f = _FONTS.get((px, bold))
    if f is None:
        f = _FONTS[(px, bold)] = Font(px, bold)
    return f


def snap(X, Y):
    """The pixel PIL puts the origin of a text on, for a position given in floats: whole pixels only (the
    fraction is rounded in 1/64 px, x to the nearest and y half down - FreeType's y axis points up)."""
    xi, yi = int(X), int(Y)
    return xi + ((int(round((X - xi) * 64.0)) + 32) >> 6), yi - ((32 - int(round((Y - yi) * 64.0))) >> 6)


# ----------------------------------------------------------------------------
# the recorder
# ----------------------------------------------------------------------------

_POOL = []                      # buffers of the frames that are gone, reused by the next ones


class DrawList(_E.Frame):
    """engine.Frame without a pixel buffer: the raw primitives are recorded. `finish()` returns the frame
    itself; `pack()` gives the blob."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.s = W / _E.DESIGN_W
        self.acc, self._pend, self._diff, self._txt, self._txt_draw = {}, {k: [] for k in _E.LAYERS}, {}, {}, {}
        self._occl = []
        self.post = []
        self.invert_rects = []
        self._bld = None
        b = _POOL.pop() if _POOL else dict(seg=np.empty((1 << 15, SEG_F), np.float32), dot=np.empty((1 << 14, DOT_F), np.float32),
                                           splat=np.empty((1 << 15, SPLAT_F), np.float32), rect=np.empty((1 << 12, RECT_F), np.float32))
        self._buf = b
        self._seg, self._dot, self._splat, self._rect = b["seg"], b["dot"], b["splat"], b["rect"]
        self._nseg = self._ndot = self._nsplat = self._nrect = 0
        self._states = []
        self._sid = -1                  # index of the state the next primitive is drawn under (-1: to be made)
        self._lightops = []
        self._textops = []
        self._strs = []
        self._nchar = 0
        self._fonts = {}                # font key -> (Font, set of the chars used by this frame)
        self.opt = None
        self.t = 0.0
        self.set_view()
        self.set_clip()

    def __del__(self):
        b = self.__dict__.get("_buf")
        if b is not None and len(_POOL) < 4:
            b.update(seg=self._seg, dot=self._dot, splat=self._splat, rect=self._rect)
            _POOL.append(b)

    # -- state -----------------------------------------------------------------
    def set_view(self, *a, **k):
        super().set_view(*a, **k)
        self._sid = -1

    def set_clip(self, *a, **k):
        super().set_clip(*a, **k)
        self._sid = -1

    def _meta(self, layer):
        if self._sid < 0:
            c = self.clip
            st = (self.vz, self.vcx, self.vcy, self.vsx, self.vsy, 0.0, c[0], c[1], c[2], c[3], 0.0, 0.0)
            if not self._states or self._states[-1] != st:
                self._states.append(st)
            self._sid = len(self._states) - 1
        return 2 * self._sid + LAYER[layer]

    @staticmethod
    def _n(*vals):
        n = 1
        for v in vals:
            if isinstance(v, np.ndarray):
                if v.ndim > 1:
                    return -1
                m = v.size if v.ndim else 1
            elif isinstance(v, (list, tuple)):
                m = len(v)
            else:
                continue
            if m != 1:
                if m == 0:
                    return 0
                n = m
        return n

    def _room(self, name, n, need):
        a = getattr(self, name)
        if n + need > len(a):
            b = np.empty((max(2 * len(a), n + need), a.shape[1]), np.float32)
            b[:n] = a[:n]
            setattr(self, name, b)
            return b
        return a

    # -- raw primitives -----------------------------------------------------------
    def flush(self):
        pass

    def _segments(self, layer, x0, y0, x1, y1, i0, i1=None, width=1.0, spacing=0.5):
        n = self._n(x0, y0, x1, y1, i0, i1, width)
        if n <= 0:
            return
        k = self._nseg
        r = self._room("_seg", k, n)[k: k + n]
        r[:, 0] = x0
        r[:, 1] = y0
        r[:, 2] = x1
        r[:, 3] = y1
        r[:, 4] = i0
        r[:, 5] = i0 if i1 is None else i1
        r[:, 6] = width
        r[:, 7] = spacing
        r[:, 8] = self._meta(layer)
        self._nseg = k + n

    def _dots(self, layer, x, y, r, i):
        n = self._n(x, y, r, i)
        if n <= 0:
            return
        k = self._ndot
        a = self._room("_dot", k, n)[k: k + n]
        a[:, 0] = x
        a[:, 1] = y
        a[:, 2] = r
        a[:, 3] = i
        a[:, 4] = self._meta(layer)
        self._ndot = k + n

    def points(self, layer, x, y, w):
        n = np.size(x)
        if not n:
            return
        k = self._nsplat
        a = self._room("_splat", k, n)[k: k + n]
        a[:, 0] = x
        a[:, 1] = y
        a[:, 2] = w
        a[:, 3] = LAYER[layer]
        self._nsplat = k + n

    def _rects(self, layer, x0, y0, x1, y1, i):
        n = self._n(x0, y0, x1, y1, i)
        if n <= 0:
            return
        k = self._nrect
        a = self._room("_rect", k, n)[k: k + n]
        a[:, 0] = x0
        a[:, 1] = y0
        a[:, 2] = x1
        a[:, 3] = y1
        a[:, 4] = i
        a[:, 5] = self._meta(layer)
        self._nrect = k + n

    # -- what multiplies the light drawn so far ----------------------------------------
    def _lightop(self, X0, Y0, X1, Y1, fw, fr):
        self._lightops.append((X0, Y0, X1, Y1, fw, fr, self._nseg, self._ndot, self._nsplat, self._nrect))

    def scale_rect(self, layer, x0, y0, x1, y1, factor):
        if self._bld is _B.MUTE:
            return
        X0, X1 = int(self.tx(x0)), int(math.ceil(float(self.tx(x1))))
        Y0, Y1 = int(self.ty(y0)), int(math.ceil(float(self.ty(y1))))
        f = float(factor)
        self._lightop(max(X0, 0), max(Y0, 0), max(X1, 0), max(Y1, 0), f if layer == "w" else 1.0, f if layer == "r" else 1.0)

    def occlude(self, x0, y0, x1, y1):
        if self._bld is _B.MUTE:
            return
        X0, Y0 = max(int(self.tx(x0)), 0), max(int(self.ty(y0)), 0)
        X1, Y1 = int(math.ceil(float(self.tx(x1)))), int(math.ceil(float(self.ty(y1))))
        self._lightop(X0, Y0, max(X1, 0), max(Y1, 0), 0.0, 0.0)
        self._textops.append((T_RECT, X0, Y0, X1 + 1, Y1 + 1, 0.0, 0.0, 1.0, 1.0, 0, 0, 0))     # PIL's rectangle: inclusive

    def dim(self, x0, y0, x1, y1, factor):
        if self._bld is _B.MUTE:
            return
        X0, Y0 = max(int(self.tx(x0)), 0), max(int(self.ty(y0)), 0)
        X1, Y1 = max(int(math.ceil(float(self.tx(x1)))), 0), max(int(math.ceil(float(self.ty(y1)))), 0)
        f = float(factor)
        self._lightop(X0, Y0, X1, Y1, f, f)
        self._textops.append((T_RECT, X0, Y0, X1, Y1, 0.0, 0.0, 1.0 - f, 1.0 - f, 0, 0, 0))

    # -- text ------------------------------------------------------------------------
    def _run(self, kind, layer, fnt, x, y, s, colour, amount):
        """Glyphs of `s`, first origin at pixel (x, y), painted on one text layer."""
        used = self._fonts.get(fnt.key)
        if used is None:
            used = self._fonts[fnt.key] = (fnt, set())
        used[1].update(s)
        c = (colour, 0.0, amount, 0.0) if layer == "w" else (0.0, colour, 0.0, amount)
        if fnt.has(s):
            self._textops.append((kind, x, y, fnt.adv, fnt.key, *c, self._nchar, len(s), 0))
            self._strs.append(s)
            self._nchar += len(s)
            return
        for ch in s:                                # a char with another advance (not in Space Mono): one run each
            self._textops.append((kind, x, y, fnt.adv, fnt.key, *c, self._nchar, 1, 0))
            self._strs.append(ch)
            self._nchar += 1
            step = fnt.glyph(ch)[0]
            x, y = (x + step, y) if kind == T_RUN else (x, y - step)

    def _text(self, layer, x, y, s, size=22, alpha=1.0, anchor="ls", bold=False):
        if alpha <= 0.004 or not s:
            return
        X, Y = float(self.tx(x)), float(self.ty(y))
        if not self._in_clip(X, Y):
            return
        fnt = get_font(max(6, int(round(size * self.s))), bold)
        xi, yi = self._origin(fnt, X, Y, s, anchor)
        self._run(T_RUN, layer, fnt, xi, yi, s, int(255 * min(alpha, 1.0)) / 255.0, 1.0)

    @staticmethod
    def _origin(fnt, X, Y, s, anchor):
        """Pixel of the origin (left, baseline) of the first glyph of a text anchored at (X, Y)."""
        xi, yi = snap(X, Y)
        h, v = anchor[0], anchor[1]
        if h != "l":
            adv = fnt.bbox(s)[4]
            xi -= adv if h == "r" else (adv + 1) // 2
        return xi, yi + (fnt.dy(v, s) if v != "s" else 0)

    def _tag(self, layer, x, y, s, size=18, alpha=1.0, anchor="ls", pad=5, bold=False, ref=None, wipe=1.0):
        if alpha <= 0.004 or not (s or ref) or wipe <= 0.0:
            return None
        X, Y = float(self.tx(x)), float(self.ty(y))
        if not self._in_clip(X, Y):
            return None
        fnt = get_font(max(6, int(round(size * self.s))), bold)
        rs = ref or s
        l, t, r, b, adv = fnt.bbox(rs)
        if anchor[0] != "l":
            sh = adv if anchor[0] == "r" else (adv + 1) // 2
            l, r = l - sh, r - sh
        if anchor[1] != "s":
            dy = fnt.dy(anchor[1], rs)
            t, b = t + dy, b + dy
        l, t, r, b = l + X, t + Y, r + X, b + Y
        p = pad * self.s
        box = (l - p, t - p, l - p + (r - l + 2 * p) * min(wipe, 1.0), b + p)
        a = int(255 * min(alpha, 1.0)) / 255.0
        # the solid box on its layer; the other layer's text is cleared under it
        cw, cr = (a, 0.0) if layer == "w" else (0.0, a)
        self._textops.append((T_RECT, int(box[0]), int(box[1]), int(box[2]) + 1, int(box[3]) + 1, cw, cr, 1.0, 1.0, 0, 0, 0))
        if s:
            xi, yi = self._origin(fnt, X, Y, s, anchor)
            self._run(T_RUN, layer, fnt, xi, yi, s, 0.0, 1.0)
        self._occl.append((int(box[0]), int(box[1]), int(math.ceil(box[2])), int(math.ceil(box[3]))))
        return tuple(v / self.s for v in box)

    def text_vertical(self, layer, x, y, s, size=22, alpha=1.0):
        b = self._bld
        if b is not None and s:
            if b is _B.MUTE:
                return
            s = b.text(s, x, y, False)
        if alpha <= 0.004 or not s.strip():
            return
        fnt = get_font(max(6, int(round(size * self.s))), False)
        l, t = fnt.bbox(s)[:2]
        # the text is set in a box of its own, 1 px in from its edges, turned, and pasted with its bottom left on (x, y)
        U, V = 1 - l, 1 - t
        self._run(T_VRUN, layer, fnt, int(self.tx(x)) + V, int(self.ty(y)) - U, s, 1.0, int(255 * min(alpha, 1.0)) / 255.0)

    # -- output ------------------------------------------------------------------------
    def finish(self, bloom_weights=(0.5, 0.45, 0.38, 0.34, 0.3, 0.28, 0.24, 0.2), bloom_gain=1.0,
               text_gain=1.0, exposure=1.0, invert=False, invert_rect=None, dither_seed=3, palette=None,
               palette_mix=1.0):
        w = tuple(float(v) for v in bloom_weights)
        if len(w) != 8:
            raise ValueError("the renderer has an 8-level bloom")
        rects = list(self.invert_rects)
        if invert:
            rects.append(invert_rect if invert_rect else (0, 0, _E.DESIGN_W, _E.DESIGN_H))
        flags = FLAG_PALETTE if (palette is not None and palette_mix > 0) else 0
        postops = []
        for fn in self.post:
            ops = post_ops(fn, self)
            if ops is None:
                flags |= FLAG_POST_UNKNOWN
            else:
                postops += ops
        s, W, H = self.s, self.W, self.H
        inv = [(max(int(x0 * s), 0), max(int(y0 * s), 0), min(int(math.ceil(x1 * s)), W), min(int(math.ceil(y1 * s)), H))
               for (x0, y0, x1, y1) in rects]
        self.opt = dict(bloom_weights=w, bloom_gain=float(bloom_gain or 0.0), text_gain=float(text_gain),
                        exposure=float(exposure), dither_seed=int(dither_seed), flags=flags, invert=inv, postops=postops)
        return self

    def counts(self):
        return dict(states=len(self._states), segments=self._nseg, dots=self._ndot, splats=self._nsplat,
                    rects=self._nrect, lightops=len(self._lightops), textops=len(self._textops), chars=self._nchar,
                    tags=len(self._occl))

    def pack(self, known=None, t=None, frame=0, out=None):
        """The frame as one blob. known = dict font key -> set of the chars whose glyphs the reader already has
        (updated): only the others are put in the blob; None = all the glyphs the frame uses (a blob that
        stands alone). out = a writable buffer to pack into (shared memory); returns the bytes used, else the
        blob as a bytearray."""
        if self.opt is None:
            raise RuntimeError("pack() before finish()")
        o = self.opt
        glyphs = []
        ng = 0
        for key, (fnt, used) in self._fonts.items():
            if known is not None:
                have = known.get(key)
                if have is None:
                    have = known[key] = set()
                new = used - have
                have |= new
            else:
                new = used
            for ch in sorted(new):
                adv, ox, oy, w, h, bits = fnt.glyph(ch)[:6]
                glyphs.append(_GLYPH.pack(key, ord(ch), adv, ox, oy, w, h))
                glyphs.append(bits + b"\0" * (-len(bits) % 4))
                ng += 1
        glyphs = b"".join(glyphs)
        chars = "".join(self._strs).encode("utf-32-le")
        secs = [
            (np.asarray(self._states, "<f4").reshape(-1, STATE_F), len(self._states)),
            (self._seg[: self._nseg], self._nseg),
            (self._dot[: self._ndot], self._ndot),
            (self._splat[: self._nsplat], self._nsplat),
            (self._rect[: self._nrect], self._nrect),
            (np.array(self._lightops, LIGHTOP), len(self._lightops)),
            (np.array(self._occl, BOX), len(self._occl)),
            (np.array(self._textops, TEXTOP), len(self._textops)),
            (chars, self._nchar),
            (glyphs, ng),
            (np.array(o["invert"], BOX), len(o["invert"])),
            (np.array(o["postops"], POSTOP), len(o["postops"])),
        ]
        sizes = [len(d) if isinstance(d, bytes) else d.nbytes for d, _ in secs]
        total = HEADER + sum(sizes)
        if out is None:
            buf = bytearray(total)
        else:
            buf = out
            if len(buf) < total:
                raise BufferError(f"draw list of {total} bytes does not fit in {len(buf)}")
        mv = memoryview(buf)
        _HEAD.pack_into(buf, 0, MAGIC, VERSION, total, o["flags"], self.t if t is None else t, self.W, self.H, self.s,
                        o["exposure"], o["bloom_gain"], o["text_gain"], *o["bloom_weights"], o["dither_seed"], frame)
        off = HEADER
        for k, ((d, n), nb) in enumerate(zip(secs, sizes)):
            struct.pack_into("<III", buf, _SEC0 + 12 * k, off, n, nb)
            if nb:
                if isinstance(d, bytes):
                    mv[off: off + nb] = d
                else:
                    np.frombuffer(mv, np.uint8, nb, off)[:] = np.ascontiguousarray(d).view(np.uint8).reshape(-1)
            off += nb
        return buf if out is None else total


@contextlib.contextmanager
def recording():
    """Inside the block, Show.render records: it returns a DrawList instead of an image."""
    from . import show as show_mod
    prev = show_mod.Frame
    show_mod.Frame = DrawList
    try:
        yield
    finally:
        show_mod.Frame = prev


def record(show, t, W=None, H=None, look=None):
    """The DrawList of the show at time t."""
    from . import layout as L
    with recording():
        dl = show.render(t, W or L.W, H or L.H, look=look)
    dl.t = float(t)
    return dl


# ----------------------------------------------------------------------------
# the one pixel-space effect: scenes/glitch.py make_post, as a list of operations
# ----------------------------------------------------------------------------

def post_ops(fn, fr):
    """Operations of a `Frame.post` callable (list of POSTOP tuples), or None if it is not one this module
    knows. The only one is the closure made by scenes/glitch.py make_post: its random draws do not depend on
    the picture, so they are replayed here (same generators, same order) to get the operations as data.
    check_drawlist.py compares the two on every glitch frame it renders: if glitch.py changes, it says so."""
    try:
        cv = inspect.getclosurevars(fn).nonlocals
        step, beat, sub, amt, vertical, view = (cv[k] for k in ("step", "beat", "sub", "amt", "vertical", "view"))
    except Exception:
        return None
    if getattr(fn, "__qualname__", "") != "make_post.<locals>.post":
        return None
    s, W, H = fr.s, fr.W, fr.H
    ops = []
    ya, yb = int(view[1] * s), int(view[3] * s)
    xa, xb = max(0, int(view[0] * s)), min(W, int(view[2] * s))
    # 1 - slices pushed sideways
    rng = np.random.default_rng((11, step))
    for i in range(int(28 * amt + 0.5)):
        h = max(2, int(rng.choice([4, 6, 10, 16, 28, 48, 90, 150]) * s))
        full = rng.random() < 0.18 * amt
        y0 = int(rng.integers(0 if full else ya, (H if full else yb) - h))
        dx = int(rng.normal(0.0, 1.0) * (30 + 520 * amt) * s * (0.85 + 0.3 * sub))
        if dx:
            ops.append((P_ROLL_X, 0, y0, W, y0 + h, dx, 0.0, 0.0))
    # 2 - beat repeat
    rng = np.random.default_rng((12, step))
    for i in range(int(6 * amt + 0.5)):
        h = max(4, int(rng.integers(24, 240) * s))
        y0 = int(rng.integers(ya, yb - h))
        w = max(4, int(rng.choice([16, 32, 64, 128, 256, 512]) * s))
        x0 = int(rng.integers(0, W - w))
        reps = int(rng.integers(3, 16))
        sgn = 1 if rng.random() < 0.5 else -1
        ops.append((P_REPEAT, x0, y0, x0 + w, y0 + h, reps - 1, sgn, 0.92))
    # 3 - pixel-sorted smears
    rng = np.random.default_rng((13, beat))
    for i in range(int(4 * amt + 5 * vertical + 0.5)):
        if rng.random() < 0.2 + 0.8 * vertical:
            w = max(4, int(rng.integers(14, 150) * s))
            x0 = int(rng.integers(xa, max(xa + 1, xb - w)))
            ops.append((P_SMEAR_Y, x0, ya, x0 + w, yb, 0.35, 0.6, 0.990 ** (1.0 / s)))
        else:
            h = max(3, int(rng.integers(8, 80) * s))
            y0 = int(rng.integers(ya, yb - h))
            flip = rng.random() < 0.5
            ops.append((P_SMEAR_X, 0, y0, W, y0 + h, 0.4, -0.75 if flip else 0.75, 0.9955 ** (1.0 / s)))
    # 4 - bands turning red, others losing their red
    rng = np.random.default_rng((14, beat))
    for i in range(int(4 * amt + 0.5)):
        h = max(2, int(rng.integers(6, 110) * s))
        y0 = int(rng.integers(ya, yb - h))
        ops.append((P_TO_RED if rng.random() < 0.7 else P_TO_WHITE, 0, y0, W, y0 + h, 0.0, 0.0, 0.0))
    # 5 - vertical hold
    rng = np.random.default_rng((15, step))
    if rng.random() < 0.5 * amt + 0.6 * vertical:
        w = max(8, int(rng.integers(60, 500) * s))
        x0 = int(rng.integers(0, W - w))
        ops.append((P_ROLL_Y, x0, ya, x0 + w, yb, int(rng.integers(20, 400) * s), 0.0, 0.0))
    return ops


def apply_post(base, ops):
    """The operations of post_ops on the light stack `base` (2, H, W): what the renderer's shaders do."""
    H, W = base.shape[1:]
    for kind, x0, y0, x1, y1, p0, p1, p2 in ops:
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        if kind == P_ROLL_X:
            base[:, y0:y1, x0:x1] = np.roll(base[:, y0:y1, x0:x1], int(p0), axis=2)
        elif kind == P_ROLL_Y:
            base[:, y0:y1, x0:x1] = np.roll(base[:, y0:y1, x0:x1], int(p0), axis=1)
        elif kind == P_REPEAT:
            w = x1 - x0
            tile = base[:, y0:y1, x0:x1].copy()
            for r in range(1, int(p0) + 1):
                x = x0 + int(p1) * r * w
                if x < 0 or x + w > W:
                    break
                base[:, y0:y1, x:x + w] = tile * (float(p2) ** r)
        elif kind in (P_SMEAR_X, P_SMEAR_Y):
            ax = 2 if kind == P_SMEAR_X else 1
            band = base[:, y0:y1, x0:x1].astype(np.float64)
            flip = p1 < 0
            if flip:
                band = band[:, :, ::-1]
            n = band.shape[ax]
            g = float(p2) ** np.arange(n)
            g = g[None, None, :] if ax == 2 else g[None, :, None]
            src = np.where(band > p0, band, 0.0)
            out = np.maximum(band, abs(p1) * np.maximum.accumulate(src / g, axis=ax) * g)
            base[:, y0:y1, x0:x1] = out[:, :, ::-1] if flip else out
        elif kind in (P_TO_RED, P_TO_WHITE):
            a, b = (1, 0) if kind == P_TO_RED else (0, 1)
            base[a, y0:y1, x0:x1] += base[b, y0:y1, x0:x1]
            base[b, y0:y1, x0:x1] = 0.0
    return base


# ----------------------------------------------------------------------------
# reading a blob, and rasterising it with the reference code
# ----------------------------------------------------------------------------

class Blob:
    """A packed frame, read back (numpy views on the buffer)."""

    def __init__(self, buf):
        mv = memoryview(buf)
        h = _HEAD.unpack_from(mv, 0)
        if h[0] != MAGIC or h[1] != VERSION:
            raise ValueError("not a draw list (or another version of the format)")
        (self.nbytes, self.flags, self.t, self.W, self.H, self.s, self.exposure, self.bloom_gain, self.text_gain) = h[2:11]
        self.bloom_weights, self.dither_seed, self.frame = h[11:19], h[19], h[20]
        sec = [struct.unpack_from("<III", mv, _SEC0 + 12 * k) for k in range(NSEC)]
        self.sec = sec

        def f32(k, w):
            return np.frombuffer(mv, "<f4", sec[k][1] * w, sec[k][0]).reshape(-1, w)

        def rec(k, dt):
            return np.frombuffer(mv, dt, sec[k][1], sec[k][0])
        self.states, self.segs, self.dots = f32(S_STATES, STATE_F), f32(S_SEGS, SEG_F), f32(S_DOTS, DOT_F)
        self.splats, self.rects = f32(S_SPLATS, SPLAT_F), f32(S_RECTS, RECT_F)
        self.lightops, self.occl, self.textops = rec(S_LIGHTOPS, LIGHTOP), rec(S_OCCL, BOX), rec(S_TEXTOPS, TEXTOP)
        self.chars = np.frombuffer(mv, "<u4", sec[S_CHARS][1], sec[S_CHARS][0])
        self.invert, self.postops = rec(S_INVERT, BOX), rec(S_POSTOPS, POSTOP)
        self.glyphs = {}                            # (font key, code point) -> (adv, ox, oy, bitmap (h, w) uint8)
        off = sec[S_GLYPHS][0]
        for _ in range(sec[S_GLYPHS][1]):
            key, cp, adv, ox, oy, w, h_ = _GLYPH.unpack_from(mv, off)
            off += _GLYPH.size
            self.glyphs[(key, cp)] = (adv, ox, oy, np.frombuffer(mv, np.uint8, w * h_, off).reshape(h_, w))
            off += (w * h_ + 3) & ~3

    def counts(self):
        return dict(states=len(self.states), segments=len(self.segs), dots=len(self.dots), splats=len(self.splats),
                    rects=len(self.rects), lightops=len(self.lightops), textops=len(self.textops), chars=len(self.chars),
                    tags=len(self.occl), glyphs=len(self.glyphs), bytes=self.nbytes)


def _paint(dst, x0, y0, src, colour, amount):
    """dst (H, W) float 0..255 <- dst + (255 * colour - dst) * amount * src / 255 on the part of src inside."""
    H, W = dst.shape
    h, w = src.shape
    a0, b0 = max(0, -x0), max(0, -y0)
    a1, b1 = min(w, W - x0), min(h, H - y0)
    if a1 <= a0 or b1 <= b0:
        return
    d = dst[y0 + b0: y0 + b1, x0 + a0: x0 + a1]
    d += (255.0 * colour - d) * (amount / 255.0) * src[b0:b1, a0:a1]


def replay(blob, glyphs=None, as_float=False, fine=False):
    """Rasterise a blob with the code of engine.Frame -> RGB uint8 (H, W, 3).
    glyphs = dict of the glyphs known so far (for blobs packed with `known`); updated.
    as_float: the picture before the dither and the rounding to 8 bits, float32 0..1 (what the renderer
    gives with --float).
    fine: the lines drawn with the default spacing (a splat every half pixel) are sampled four times finer.
    The renderer integrates those lines exactly, so this is the picture it must give, to a fraction of a
    level (engine/tools/compare.py). Lines with a wider spacing keep it: their grain is part of the look."""
    from PIL import Image
    b = blob if isinstance(blob, Blob) else Blob(blob)
    gl = b.glyphs if glyphs is None else glyphs
    if glyphs is not None:
        glyphs.update(b.glyphs)
    fr = _E.Frame(b.W, b.H)
    H, W = b.H, b.W

    def use(meta):
        st = b.states[int(meta) >> 1]
        fr.vz, fr.vcx, fr.vcy, fr.vsx, fr.vsy = (float(v) for v in st[:5])
        fr.clip = tuple(float(v) for v in st[6:10])
        return _E.LAYERS[int(meta) & 1]

    def runs(a, col, lo, hi):
        if hi <= lo:
            return
        m = a[lo:hi, col]
        cut = np.flatnonzero(m[1:] != m[:-1]) + 1
        for i, j in zip(np.r_[0, cut], np.r_[cut, hi - lo]):
            yield lo + int(i), lo + int(j)

    def draw(c0, c1):
        for i, j in runs(b.segs, 8, c0[0], c1[0]):
            v = b.segs[i:j]
            for sp in np.unique(v[:, 7]):
                u = v[v[:, 7] == sp]
                fr._segments(use(u[0, 8]), u[:, 0], u[:, 1], u[:, 2], u[:, 3], u[:, 4], u[:, 5], u[:, 6],
                             0.125 if fine and sp <= 0.5001 else float(sp))
        for i, j in runs(b.dots, 4, c0[1], c1[1]):
            v = b.dots[i:j]
            fr._dots(use(v[0, 4]), v[:, 0], v[:, 1], v[:, 2], v[:, 3])
        for i, j in runs(b.splats, 3, c0[2], c1[2]):
            v = b.splats[i:j]
            fr.points(_E.LAYERS[int(v[0, 3])], v[:, 0], v[:, 1], v[:, 2])
        for i, j in runs(b.rects, 5, c0[3], c1[3]):
            v = b.rects[i:j]
            fr._rects(use(v[0, 5]), v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4])
        fr.flush()

    done = (0, 0, 0, 0)
    for op in b.lightops:
        upto = (int(op["nseg"]), int(op["ndot"]), int(op["nsplat"]), int(op["nrect"]))
        draw(done, upto)
        done = upto
        for k, f in ((0, op["fw"]), (1, op["fr"])):
            fr.acc[_E.LAYERS[k]].reshape(H, W)[max(op["y0"], 0): max(op["y1"], 0), max(op["x0"], 0): max(op["x1"], 0)] *= f
    draw(done, (len(b.segs), len(b.dots), len(b.splats), len(b.rects)))

    txt = np.zeros((2, H, W), np.float64)
    for op in b.textops:
        col, amt = (op["cw"], op["cr"]), (op["fw"], op["fr"])
        if op["kind"] == T_RECT:
            x0, y0, x1, y1 = max(op["a"], 0), max(op["b"], 0), min(op["c"], W), min(op["d"], H)
            if x1 > x0 and y1 > y0:
                for k in (0, 1):
                    d = txt[k, y0:y1, x0:x1]
                    d += (255.0 * col[k] - d) * amt[k]
            continue
        for n, cp in enumerate(b.chars[op["off"]: op["off"] + op["cnt"]]):
            adv, ox, oy, bits = gl[(int(op["d"]), int(cp))]
            if not bits.size:
                continue
            for k in (0, 1):
                if amt[k] == 0.0:
                    continue
                if op["kind"] == T_RUN:
                    _paint(txt[k], op["a"] + n * op["c"] + ox, op["b"] + oy, bits, col[k], amt[k])
                else:
                    rot = bits.T[::-1]              # texel (gx, gy) -> (gy, w - 1 - gx)
                    _paint(txt[k], op["a"] + oy, op["b"] - n * op["c"] - ox - bits.shape[1], rot, col[k], amt[k])
    for k in (0, 1):
        if txt[k].any():
            fr._txt[_E.LAYERS[k]] = Image.fromarray(np.clip(np.round(txt[k]), 0, 255).astype(np.uint8))
    fr._occl = [tuple(int(v) for v in r) for r in b.occl]
    if len(b.postops):
        ops = [tuple(r) for r in b.postops.tolist()]
        fr.post.append(lambda base, f: apply_post(base, ops))
    return _finish(fr, b, as_float)


def _finish(fr, b, as_float):
    """Frame.finish, with the inverted rects given in pixels (and the picture in floats if asked)."""
    base = np.stack([fr.acc[k].reshape(b.H, b.W) for k in _E.LAYERS]) * b.exposure
    for (x0, y0, x1, y1) in fr._occl:
        base[:, max(y0, 0): max(y1, 0), max(x0, 0): max(x1, 0)] = 0.0
    for j, k in enumerate(_E.LAYERS):
        if k in fr._txt:
            base[j] += np.asarray(fr._txt[k], np.float32) * (b.text_gain / 255.0)
    for fn in fr.post:
        base = fn(base, fr)
    light = base + _E.bloom(base, b.bloom_weights) * b.bloom_gain if b.bloom_gain else base
    wt, rt = _E._tonemap(light[0]), _E._tonemap(light[1])
    rgb = np.empty((b.H, b.W, 3), np.float32)
    for c in range(3):
        rgb[..., c] = wt + rt * _E.RED[c]
    for (x0, y0, x1, y1) in b.invert:
        ys, xs = slice(y0, y1), slice(x0, x1)
        rgb[ys, xs, 0] = 1.0 - wt[ys, xs]
        rgb[ys, xs, 1] = 1.0 - wt[ys, xs] - rt[ys, xs]
        rgb[ys, xs, 2] = 1.0 - wt[ys, xs] - rt[ys, xs]
    np.clip(rgb, 0.0, 1.0, out=rgb)
    if as_float:
        return rgb
    rgb *= 255.0
    rgb += _E._dither(b.H, b.W, b.dither_seed)
    np.clip(rgb, 0, 255, out=rgb)
    return rgb.astype(np.uint8)


# ----------------------------------------------------------------------------
# files
# ----------------------------------------------------------------------------

def write_frames(path, blobs):
    """One file for one frame or for a time range: the blobs one after the other."""
    with open(path, "wb") as fh:
        for b in blobs:
            fh.write(b)


def read_frames(path):
    """The blobs of a file (list of memoryviews)."""
    data = memoryview(open(path, "rb").read())
    out, off = [], 0
    while off < len(data):
        n = struct.unpack_from("<I", data, off + 8)[0]
        out.append(data[off: off + n])
        off += n
    return out
