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


def tag(f, layer, x, y, s, age, t0=0.0, size=18, pad=5, bold=False, anchor="ls", alpha=1.0, wipe=0.1, cps=60.0,
        key=0, commit=False, lead=6):
    """An inverted tag being made: its solid box is pushed out from the left, `lead` letters ahead of the
    writing (and never faster than `wipe` seconds for the whole box); the letters are cut out of it behind
    that edge, resolving out of random glyphs. commit=True: a two-frame inverted flash when the last letter
    is right. Returns the box of the tag (None before it starts)."""
    a = age - t0
    if a < 0.0 or not s:
        return None
    w = min(1.0, a / wipe, (a * cps + lead) / len(s))
    if w >= 1.0:
        txt = decode(s, a, cps=cps, key=key, pad=True)
    else:
        n = int(w * len(s))                              # letters that fit in the part of the box that is there
        txt = decode(s, a, cps=cps, key=key, pad=True)[:n].ljust(len(s))
    box = f.tag(layer, x, y, txt if txt.strip() else "", size=size, alpha=alpha, anchor=anchor, pad=pad, bold=bold,
                ref=s, wipe=w)
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
    f.segments(layer, [x0], [y0], [xe], [ye], inten, width=width)
    if p < 1.0 and head > 0.0:
        f.dots("w", [xe], [ye], head, 1.7)
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
    f.segments(layer, ax, ay, bx, by, inten, width=width)


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
        f.segments("w", [xa], [yc], [xb], [yc], inten, width=width)
        if pl < 1.0:
            f.dots("w", [xa if from_right else xb], [yc], 3.6, 1.7)
        return False
    hh = 0.5 * (y1 - y0) * po
    f.occlude(xa, yc - hh, xb, yc + hh)
    f.rect("w", xa, yc - hh, xb, yc + hh, inten, width=width)
    return po >= 1.0
