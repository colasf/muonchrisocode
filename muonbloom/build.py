"""Build-up animations: data never fades in, it is constructed.

Every data element of the show (strip, panel, counter, card, note, ruler, bar) is written on by a short
choreography made of the same five moves, so the whole instrument switches on in one language:

  locate    registration marks say where it will be (corner brackets, a block cursor)
  trace     its lines are drawn by a pen with a bright head (rules, outlines, axes)
  cascade   repeated marks arrive one after the other and overshoot before they settle (ticks, bars, rows)
  decode    text resolves out of random glyphs, numbers spin before they lock
  commit    a two-frame inverted flash when it is complete; the registration marks go away

Everything is a pure function of the AGE of the element (seconds since its build started): no state, so it
ports to the realtime app as it is. A large age means 'built': every helper then draws the finished thing.

Two ways to use it:

  1. `with f.build(age, rect):` - the general way (engine.Frame.build -> Block below). Whatever is drawn
     inside the block is constructed automatically, primitive by primitive, in a wave that crosses the rect:
         long lines            are drawn by a pen with a bright head
         short marks           (ticks, crosses, dashes) are thrown out long and fall back to their length
         polylines, outlines, rings   are traced
         filled rects          (bars, blocks, rules) grow from their base; the small ones overshoot
         dots                  pop
         text                  is decoded out of noise, its figures spin before they lock
         tags                  are pushed out, their letters cut behind the edge
     with registration brackets around the rect while it is being made. A negative age draws nothing at all
     (the block is not there yet); a large age, or None, draws it as usual. io() gives the age of a block
     that also leaves: it is then taken apart the way it was made.

  2. The helpers of this module (pen, tag, open_box, decode, roll ...) - for a bespoke choreography, as in
     the first seconds of the show.
"""
from __future__ import annotations

import math

import numpy as np

from . import layout as L

RATE = 30.0                                  # the scramble and the blinks change 30 times a second
_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGIT = "0123456789"


# ----------------------------------------------------------------------------
# time
# ----------------------------------------------------------------------------

def lin(age, t0, t1):
    """Progress 0..1 of the step that runs from t0 to t1 (seconds of the element's age)."""
    return float(np.clip((age - t0) / max(t1 - t0, 1e-6), 0.0, 1.0))


def ease(x):
    """Ease-out: a fast start and a soft landing (a pen that brakes)."""
    x = np.clip(x, 0.0, 1.0)
    return 1.0 - (1.0 - x) ** 3


def spring(x):
    """0 -> 1 with an overshoot of about 20 %: a mark that arrives too far, then settles."""
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return np.where(x >= 1.0, 1.0, 1.0 - np.exp(-5.0 * x) * np.cos(10.0 * x))


def cascade(age, order, t0, wave, dur):
    """Progress 0..1 of marks that arrive one after the other. order = a count (0 .. n-1, in sequence) or an
    array of ranks 0..1: the first mark starts at t0, the last at t0 + wave, each one takes `dur`."""
    rank = np.arange(order) / max(order - 1, 1) if np.isscalar(order) else np.asarray(order, np.float64)
    return np.clip((age - t0 - wave * rank) / dur, 0.0, 1.0)


def frame_no(age):
    """Index of the 1/30 s step the element is in (what the scramble and the blinks are keyed on)."""
    return max(int(math.floor(age * RATE + 1e-6)), 0)


def io(age, left=None, out=0.35, span=1.3):
    """Age to give Frame.build for an element that also LEAVES: `age` = seconds since it appeared, `left` =
    seconds before it must be gone. While it is there the age simply runs; during its last `out` seconds it
    runs backwards from `span` to 0, so the element is taken apart the way it was made, only faster.
    Negative once it is gone: the block then draws nothing."""
    if left is None:
        return age
    if left <= 0.0:
        return -1.0
    if left < out:
        return min(age, span * left / out)
    return age


def rnd(n, key, frame):
    """n pseudo-random floats in [0, 1): one per item, all new at every frame (engine.hash01 hardly mixes its
    last key into its high bits, so a digit drawn from it would not change from one frame to the next)."""
    seed = (int(key) * 0xC2B2AE3D27D4EB4F + int(frame) * 0x165667B19E3779F9 + 0x27D4EB2F165667C5) & 0xFFFFFFFFFFFFFFFF
    x = (np.asarray(n, np.uint64) if np.ndim(n) else np.arange(int(n), dtype=np.uint64)) + np.uint64(1)
    x = x * np.uint64(0x9E3779B97F4A7C15) ^ np.uint64(seed)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xFF51AFD7ED558CCD)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xC4CEB9FE1A85EC53)
    x ^= x >> np.uint64(33)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def marks_on(age, until, blinks=2):
    """Registration marks: there from the start of the build until `until`, blinking as they arrive."""
    if age < 0.0 or age >= until:
        return False
    fr = frame_no(age)
    return fr >= 2 * blinks or fr % 2 == 0


# ----------------------------------------------------------------------------
# text
# ----------------------------------------------------------------------------

def _noise(c, h):
    if c.isdigit():
        return _DIGIT[int(h * 10)]
    if c.isalpha():
        return _ALPHA[int(h * 26)]
    return c


def decode(s, age, cps=80.0, delay=0.0, key=0, band=4, pad=False):
    """Text resolving out of noise. It is written left to right at `cps` characters a second; the last `band`
    characters written are still random (digits stay digits, letters stay letters, signs and spaces are
    already right), the others are final. pad=True keeps the full length with spaces, so that tags and
    right-anchored text do not move while they are written."""
    if not s:
        return s
    n = (age - delay) * cps
    if n >= len(s) + band:
        return s
    if n <= 0.0:
        return " " * len(s) if pad else ""
    h = rnd(len(s), key, frame_no(age))
    out = []
    for k, c in enumerate(s):
        if k >= n:
            if not pad:
                break
            out.append(" ")
        elif k < n - band:
            out.append(c)
        else:
            out.append(_noise(c, h[k]))
    return "".join(out)


def roll(s, age, dur=0.45, delay=0.0, key=0):
    """A read-out acquiring its value: its digits spin (a new random digit 30 times a second) and lock one
    after the other, left to right, over `dur`. Letters and signs do not move. Nothing before `delay`."""
    a = age - delay
    if a < 0.0:
        return ""
    if a >= dur or not s:
        return s
    idx = [k for k, c in enumerate(s) if c.isdigit()]
    if not idx:
        return s
    locked = int(len(idx) * a / dur)
    h = rnd(len(s), key + 77, frame_no(age))
    out = list(s)
    for k in idx[locked:]:
        out[k] = _DIGIT[int(h[k] * 10)]
    return "".join(out)


def resolve(s, age, cps=90.0, delay=0.0, key=0, pad=False, spin=0.3):
    """decode + roll: the line is written out of noise, and its digits keep spinning a little longer."""
    a = age - delay
    if a < 0.0:
        return " " * len(s) if pad else ""
    return roll(decode(s, a, cps=cps, key=key, band=5, pad=pad), a, dur=len(s) / cps + spin, key=key)


def flash(f, rect, age, at, frames=2):
    """Commit: the rect is shown inverted (white field, black lines) for `frames` thirtieths of a second."""
    if rect is not None and at <= age < at + frames / RATE:
        f.invert_rects.append(tuple(rect))


def tag_state(s, a, cps=60.0, wipe=0.1, lead=6, key=0):
    """(letters, fraction of the box) of a tag `a` seconds after it started to be made: the box is `lead`
    letters ahead of the writing, and takes at least `wipe` seconds."""
    w = min(1.0, a / wipe, (a * cps + lead) / len(s))
    txt = decode(s, a, cps=cps, key=key, pad=True)
    if w < 1.0:                                          # only the letters that fit in the part of the box that is there
        txt = txt[: int(w * len(s))].ljust(len(s))
    return (txt if txt.strip() else ""), w


def tag(f, layer, x, y, s, age, t0=0.0, size=18, pad=5, bold=False, anchor="ls", alpha=1.0, wipe=0.1, cps=60.0,
        key=0, commit=False, lead=6):
    """An inverted tag being made: its solid box is pushed out from the left, `lead` letters ahead of the
    writing (and never faster than `wipe` seconds for the whole box); the letters are cut out of it behind
    that edge, resolving out of random glyphs. commit=True: a two-frame inverted flash when the last letter
    is right. Returns the box of the tag (None before it starts)."""
    a = age - t0
    if a < 0.0 or not s:
        return None
    txt, w = tag_state(s, a, cps=cps, wipe=wipe, lead=lead, key=key)
    box = f._tag(layer, x, y, txt, size=size, alpha=alpha, anchor=anchor, pad=pad, bold=bold, ref=s, wipe=w)
    if commit:
        flash(f, box, a, (len(s) + 4) / cps)
    return box


# ----------------------------------------------------------------------------
# lines
# ----------------------------------------------------------------------------

def pen(f, layer, x0, y0, x1, y1, p, inten=0.9, width=L.LW, head=3.2):
    """A line being drawn from (x0, y0) towards (x1, y1): p = 0..1 of its length, with a bright point at the
    pen while it moves. Returns the position of the pen."""
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0:
        return x0, y0
    xe, ye = x0 + (x1 - x0) * p, y0 + (y1 - y0) * p
    f._segments(layer, [x0], [y0], [xe], [ye], inten, width=width)
    if p < 1.0 and head > 0.0:
        f._dots("w", [xe], [ye], head, 1.7)
    return xe, ye


def brackets(f, rect, size=14.0, inten=0.9, layer="w", width=L.LW):
    """Registration marks: the four corners of a rect."""
    x0, y0, x1, y1 = rect
    s = min(size, 0.5 * (x1 - x0), 0.5 * (y1 - y0))
    ax, ay, bx, by = [], [], [], []
    for cx, sx in ((x0, 1.0), (x1, -1.0)):
        for cy, sy in ((y0, 1.0), (y1, -1.0)):
            ax += [cx, cx]
            ay += [cy, cy]
            bx += [cx + sx * s, cx]
            by += [cy, cy + sy * s]
    f._segments(layer, ax, ay, bx, by, inten, width=width)


def open_box(f, rect, age, t_line=(0.0, 0.2), t_open=(0.18, 0.42), from_right=False, inten=0.95, width=L.LW_FRAME):
    """A framed box that opens like a plate: a line is drawn across its middle (from the left edge, or from
    the right one), then it splits into the top and bottom edges, which move apart; what was behind is cut
    out as it opens. Returns True once it is fully open."""
    x0, y0, x1, y1 = rect
    yc = 0.5 * (y0 + y1)
    pl = float(ease(lin(age, *t_line)))
    po = float(ease(lin(age, *t_open)))
    if pl <= 0.0:
        return False
    xa, xb = (x1 - (x1 - x0) * pl, x1) if from_right else (x0, x0 + (x1 - x0) * pl)
    if po <= 0.0:
        f._segments("w", [xa], [yc], [xb], [yc], inten, width=width)
        if pl < 1.0:
            f._dots("w", [xa if from_right else xb], [yc], 3.6, 1.7)
        return False
    hh = 0.5 * (y1 - y0) * po
    f.occlude(xa, yc - hh, xb, yc + hh)
    f._segments("w", [xa, xb, xb, xa], [yc - hh, yc - hh, yc + hh, yc + hh], [xb, xb, xa, xa],
                [yc - hh, yc + hh, yc + hh, yc - hh], inten, width=width)
    return po >= 1.0


# ----------------------------------------------------------------------------
# the general way: a block of HUD that constructs itself (Frame.build)
# ----------------------------------------------------------------------------

MUTE = object()                               # Frame._bld while a block is not there yet: nothing is drawn


class Block:
    """A piece of the HUD being constructed. Made by Frame.build(age, rect, ...) and used as a context
    manager; the primitives of the frame ask it what to draw while it is active.

        age      seconds since the block started to build (< 0: not there, nothing is drawn; None or large:
                 built, everything is drawn as usual). Use io() for a block that also leaves.
        rect     the block on the wall (design px): the wave crosses it, the registration marks frame it
        wave     seconds the wave takes to cross the rect; flow = its direction: 'diag' (from the top left),
                 'lr', 'rl', 'tb', 'bt', or 'out' (away from `origin`, default the centre of the rect)
        line     seconds a pen takes to draw a line once the wave has reached its start
        cps      characters decoded per second
        bars     'auto' (tall rects rise from their bottom, wide ones grow from their left), 'down' (tall
                 rects hang from their top), 'centre' (every rect opens from its middle line)
        marks    registration brackets around the rect while it is being made
        commit   a two-frame inverted flash of the rect when the block is complete
    """
    LONG = 24.0                 # px: a longer line is drawn by a pen, a shorter one is a mark thrown out
    HOLD = 1.7                  # s after the wave has crossed: the block is built, the frame draws as usual
    HEADS = 6                   # more pens than this at work in one call: no heads (it is a grid, a texture)

    def __init__(self, f, age, rect, key=0, wave=0.4, flow="diag", line=0.3, cps=110.0, marks=True, origin=None,
                 bars="auto", commit=False):
        self.f, self.age = f, age
        self.rect = tuple(float(v) for v in rect)
        self.key, self.wave, self.flow, self.line, self.cps = int(key), float(wave), flow, float(line), float(cps)
        self.marks, self.bars, self.commit = marks, bars, commit
        x0, y0, x1, y1 = self.rect
        self.origin = origin if origin is not None else (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        self.rmax = max(max(math.hypot(x - self.origin[0], y - self.origin[1]) for x in (x0, x1) for y in (y0, y1)), 1.0)
        self.prev = None

    # -- context -----------------------------------------------------------------
    def __enter__(self):
        f = self.f
        self.prev = f._bld
        if self.age is None or self.prev is MUTE:        # no age: as the frame is; inside a muted block: muted
            return self
        if self.age < 0.0:
            f._bld = MUTE
        elif self.age < self.wave + self.HOLD:
            f._bld = self
            if self.marks and marks_on(self.age, self.wave + 0.45):
                brackets(f, self.rect)
            if self.commit:
                flash(f, self.rect, self.age, self.wave + 0.85)
        else:
            f._bld = None
        return self

    def __exit__(self, *exc):
        self.f._bld = self.prev
        return False

    # -- when the wave reaches a point ---------------------------------------------
    def la(self, x, y):
        """Local age at (x, y): the wave gets there `wave * u` seconds after the block started."""
        x0, y0, x1, y1 = self.rect
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        fl = self.flow
        if fl == "lr":
            u = (x - x0) / w
        elif fl == "rl":
            u = (x1 - x) / w
        elif fl == "tb":
            u = (y - y0) / h
        elif fl == "bt":
            u = (y1 - y) / h
        elif fl == "out":
            u = np.hypot(x - self.origin[0], y - self.origin[1]) / self.rmax
        else:
            u = 0.65 * (x - x0) / w + 0.35 * (y - y0) / h
        return self.age - self.wave * np.clip(u, 0.0, 1.0)

    def prog(self, x, y, k=1.0):
        """Progress 0..1 of a pen that starts at (x, y)."""
        return float(ease(self.la(x, y) / (self.line * k)))

    # -- primitives ------------------------------------------------------------------
    def segs(self, x0, y0, x1, y1, i0, i1, width):
        """Segments being made: (x0, y0, x1, y1, i0, i1, width, head x, head y), or None if none is there yet."""
        x0, y0, x1, y1 = (np.atleast_1d(np.asarray(v, np.float32)) for v in (x0, y0, x1, y1))
        shape = np.broadcast_shapes(x0.shape, y0.shape, x1.shape, y1.shape, np.shape(i0),
                                    np.shape(i1) if i1 is not None else (), np.shape(width))
        if len(shape) != 1 or shape[0] == 0:
            return None
        n = shape[0]
        x0, y0, x1, y1 = (np.broadcast_to(a, (n,)) for a in (x0, y0, x1, y1))
        i0 = np.broadcast_to(np.asarray(i0, np.float32), (n,))
        i1 = i0 if i1 is None else np.broadcast_to(np.asarray(i1, np.float32), (n,))
        width = np.broadcast_to(np.asarray(width, np.float32), (n,))
        la = self.la(x0, y0)
        long = np.hypot(x1 - x0, y1 - y0) > self.LONG
        p = np.where(long, ease(la / self.line), np.where(la >= 0.0, 1.0 + 1.6 * np.exp(-np.maximum(la, 0.0) / 0.07), 0.0))
        keep = p > 0.0
        if not keep.any():
            return None
        p = p.astype(np.float32)
        xe, ye = x0 + (x1 - x0) * p, y0 + (y1 - y0) * p
        ie = i0 + (i1 - i0) * np.minimum(p, 1.0)
        head = keep & long & (p < 1.0)
        if head.sum() > self.HEADS:
            head = np.zeros(n, bool)
        return x0[keep], y0[keep], xe[keep], ye[keep], i0[keep], ie[keep], width[keep], xe[head], ye[head]

    def grow(self, x0, y0, x1, y1, i):
        """Filled rects being made: (x0, y0, x1, y1, i), or None."""
        x0, y0, x1, y1 = (np.atleast_1d(np.asarray(v, np.float32)) for v in (x0, y0, x1, y1))
        shape = np.broadcast_shapes(x0.shape, y0.shape, x1.shape, y1.shape, np.shape(i))
        x0, y0, x1, y1 = (np.broadcast_to(a, shape).astype(np.float32) for a in (x0, y0, x1, y1))
        i = np.broadcast_to(np.asarray(i, np.float32), shape)
        w, h = x1 - x0, y1 - y0
        la = self.la(x0, 0.5 * (y0 + y1))
        keep = la > 0.0
        if not keep.any():
            return None
        x = la / 0.3
        tall = np.abs(h) > 1.15 * np.abs(w)
        g = np.where(np.where(tall, np.abs(h), np.abs(w)) <= 70.0, spring(x), ease(x)).astype(np.float32)
        if self.bars == "centre":
            ym = 0.5 * (y0 + y1)
            y0, y1 = ym - 0.5 * h * g, ym + 0.5 * h * g
        elif self.bars == "down":
            y1 = np.where(tall, y0 + h * g, y1)
            x1 = np.where(tall, x1, x0 + w * g)
        else:
            y0 = np.where(tall, y1 - h * g, y0)
            x1 = np.where(tall, x1, x0 + w * g)
        return x0[keep], y0[keep], x1[keep], y1[keep], i[keep]

    def pops(self, x, y, r, i):
        """Dots being made: (x, y, r, i) - they pop, a little too large at first - or None."""
        x, y = np.atleast_1d(np.asarray(x, np.float32)), np.atleast_1d(np.asarray(y, np.float32))
        n = len(x)
        r = np.broadcast_to(np.asarray(r, np.float32), (n,))
        i = np.broadcast_to(np.asarray(i, np.float32), (n,))
        la = self.la(x, y)
        keep = la >= 0.0
        if not keep.any():
            return None
        return x[keep], y[keep], (r * spring(la / 0.22))[keep].astype(np.float32), i[keep]

    def there(self, x, y):
        """Mask of the points the wave has reached (pixels)."""
        return self.la(x, y) >= 0.0

    def text(self, s, xl, y, pad):
        """A string being decoded; xl = its left edge."""
        return resolve(s, float(self.la(xl, y)), cps=self.cps, key=(int(xl) * 7919 + int(y) * 104729 + self.key) & 0xFFFFF,
                       pad=pad)

    def tag(self, s, xl, y):
        """(letters, fraction of the box) of a tag being made; xl = its left edge."""
        a = float(self.la(xl, y))
        if a <= 0.0:
            return "", 0.0
        return tag_state(s, a, key=(int(xl) * 7919 + int(y) * 104729 + self.key) & 0xFFFFF)
