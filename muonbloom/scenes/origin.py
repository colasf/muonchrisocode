"""ORIGIN + STAR - the opening.   Sheet 1.0 + 1.1, 00:00 - 00:44.

The TouchDesigner intro (lattice of crosses, red dot, expanding rings, the rotating spiral of streaks,
the star, the rays of its collapse) kept as it is and given its data:

  00:00  BLACK       nothing at all: no counter, no ticks, no lattice
  00:01  BORDER      the white frame fades in
  00:03  LOADING     top left: MUON BLOOM // LOADING, a bar that creeps and jumps on the low thumps
  00:06  LOADED      the lattice of crosses draws itself row by row, then the red dot and its crosshair:
                     at 00:07 everything is there, the dot pulses with the low thumps
  00:11  HERE / NOW  "something is passing through you": rings leave the dot (light-time shells around the
                     muon), the arrivals barcode starts in the score strip - one tick per muon through one
                     spectator, 63 per second - and the wall counter runs (about 60 800 per second)
  00:19  ALWAYS      the spiral field of streaks grows out of the dot and turns
  00:24  NOT FELT    a 'felt' trace stays flat under the frantic barcode
  00:28  MINISCULE   the dot collapses to a point, a scale ruler dives through eighteen powers of ten
  00:31  STAR        "billions of years ago ... a star collapsed, immense": the point is the core of a star;
                     progenitor data, shells, a countdown
  00:40  COLLAPSE    the disc implodes, bounces, and the rays burst out (3D, the ring plane tilts): the
                     messengers. One ray is tagged: the one that will reach us (-> messenger.py).

THE TOWERS stand in front of the wall for the whole show and nobody knows yet where: nothing here has a
fixed x. `Lay` reads the wall from ctx: the one centre of the image (ctx.focus, the middle of the best bay),
a column for the scene card, a column for the notes that point at the centre, the free panels of the
bottom band. Heroes (dot, star, bounce, the tagged ray) and every piece of text stay clear of the towers;
rings, rays, the streak field and leader lines simply pass behind them.

NO FADE-IN for anything that shows data (so far applied to 00:00 - 00:16 only): the loading block, the edge
meters, the arrivals strip, the HERE note, the scale, the four panels, the card and the two counter cells are
CONSTRUCTED, each by its own choreography of the moves in muonbloom/build.py (registration marks, lines
drawn by a pen, marks arriving in a wave and overshooting, text decoded out of noise, figures spinning
before they lock, a two-frame inverted flash when the element is complete). Check a build frame by frame
with tools/filmstrip.py.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import Camera, hash01, smoothstep, text_w
from ..show import Scene

FOV = 40.0
FOCAL = (L.H / 2.0) / math.tan(math.radians(FOV) / 2.0)

T_HERE = sd.said("Here, Right now", 11.17)
T_ALWAYS = sd.said("It always has", 19.43)
T_FEEL = sd.said("You cannot feel it", 24.13)
T_NEVER = sd.said("You never could", 26.33)
T_MINI = sd.said("Miniscule", 28.22)
T_STAR = sd.said("Billions of years ago", 31.23)
T_BLOOMED = sd.said("From that collapse", 43.5)
T_FRAME = 1.0                                    # the white border starts to fade in on black
T_LOAD = 3.15                                    # MUON BLOOM // LOADING comes on (first sound event)
T_LOADED = 6.0                                   # 100 %: the lattice of crosses draws itself, row by row
T_BOOT = 6.7                                     # the crosses are drawn: the red dot appears
T_READY = 7.0                                    # everything is there (dot, crosshair, lattice, edge ticks)
LOAD_STEPS = ["CANVAS 2978 X 1400", "AUDIO 13:22", "DETECTORS L C R // OSC", "MUON FLUX 1 /CM2/MIN"]
LOAD_N = 50                                      # blocks in the loading bar
T_FIELD = 17.6                                   # the streak field starts to grow
T_IMPLODE = 39.25                                # the disc gives way
T_X = 40.0                                       # bounce: the rays leave
T_END = 44.0

RATE_YOU = 63.0                                  # muons / s through one spectator
RATE_WALL = 60800.0                              # 28 m x 13 m at 1 / cm2 / min
YEAR0 = 4.8e9                                    # years ago ("perhaps before the Earth existed")
CARD_Y = 300.0
FRAME_CLIP = (L.FX0, L.FY0, L.FX1, L.FY1)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ----------------------------------------------------------------------------
# the wall, as these scenes use it (shared with messenger.py)
# ----------------------------------------------------------------------------

class Lay:
    """Layout derived from ctx - the towers can stand anywhere, so no x is fixed.
         C          the one centre of the image (ctx.focus)
         half       free half-width around it inside its bay (heroes stay within it)
         card       column of the scene card (the leftmost usable one), or None
         note       column of the notes that point at the centre (next to the focus bay), or None
         slots      free panels of the bottom band, left to right (wide panels are split)
         cell_r     bottom-right counter cell"""

    def __init__(self, ctx):
        self.C = (float(ctx.focus[0]), float(ctx.focus[1]))
        cols = [(float(a), float(b)) for a, b in ctx.cols]
        bay = ctx.focus_bay
        inside = [c for c in cols if c[0] <= self.C[0] <= c[1]]
        self.focus_col = inside[0] if inside else (bay[0] + 28.0, bay[1] - 28.0)
        self.half = min(self.C[0] - self.focus_col[0], self.focus_col[1] - self.C[0])
        others = [c for c in cols if c != self.focus_col]
        self.card = next((c for c in others if c[1] - c[0] >= 240.0), None)
        wide = [c for c in others if c != self.card and c[1] - c[0] >= 300.0]
        right = [c for c in wide if c[0] >= self.focus_col[1]]
        left = [c for c in wide if c[1] <= self.focus_col[0]]
        self.note = right[0] if right else (left[-1] if left else None)
        self.note_side = 1.0 if (self.note is None or self.note[0] >= self.focus_col[1]) else -1.0
        self.towers = [(t.x0, t.top, t.x1, t.bot) for t in ctx.towers.values()]
        self.cell_r = tuple(float(v) for v in ctx.cell_r)
        self.far = max(math.hypot(x - self.C[0], y - self.C[1]) for x in (L.FX0, L.FX1) for y in (L.FY0, L.FY1))
        self.r_star = max(90.0, min(212.0, self.half - 70.0))
        self.slots = self._bottom(ctx)
        self.avoid = []                 # rects taken by the card / the notes in the frame being drawn

    def _bottom(self, ctx, want=4, min_w=300.0, gap=30.0):
        lim = self.cell_r[0] - 26.0
        free = []
        for a, b in ctx.slots_pre["panels"]:
            b = min(float(b), lim)
            if b - a >= 240.0:
                free.append((float(a), b))
        n = [1] * len(free)
        while sum(n) < want:
            best, bw = None, 0.0
            for i, (a, b) in enumerate(free):
                w = (b - a - gap * n[i]) / (n[i] + 1)
                if w >= min_w and w > bw:
                    best, bw = i, w
            if best is None:
                break
            n[best] += 1
        out = []
        for (a, b), k in zip(free, n):
            w = (b - a - gap * (k - 1)) / k
            out += [(a + j * (w + gap), a + j * (w + gap) + w) for j in range(k)]
        return out

    def free(self, x0, y0, x1, y1, pad=16.0):
        """True if a text rect is clear of every tower, of the header and bottom bands and of the blocks
        already placed in this frame."""
        if x0 < L.COL_X0 or x1 > L.COL_X1 or y0 < L.HEAD_Y + 10.0 or y1 > L.VIEW[3]:
            return False
        for a, top, b, bot in self.towers:
            if x1 > a - pad and x0 < b + pad and y1 > top - pad and y0 < bot + pad:
                return False
        for a, top, b, bot in self.avoid:
            if x1 > a - 8.0 and x0 < b + 8.0 and y1 > top - 8.0 and y0 < bot + 8.0:
                return False
        return True

    def take(self, rect):
        self.avoid.append(rect)


def header_gap(f, ctx, t):
    """The subtitle box can stand off the strip (it steps aside for a tower): while it is on, the slice of
    header band left between the two is kept dark, so the image does not show through as a stray patch."""
    x0, x1 = L.STRIP[2], float(ctx.sub[0])
    a = hud.subtitle_box_alpha(t)
    if x1 - x0 > 2.0 and a > 0.01:
        f.dim(x0, L.FY0, x1, L.HEAD_Y, 1.0 - a)


def title_fit(options, width, size=L.T_MICRO):
    """The longest of the given titles that fits the width."""
    for s in options:
        if text_w(s, size) + 10.0 <= width:
            return s
    return ""


CARD_BUILD = 1.6                                 # seconds a card takes to build (card(..., build=True))


def card(f, lay, title, rows, age, red_title=False, red_rows=(), dim_rows=(), cps=70.0, build=False):
    """Scene card in the card column: title tag + typed data rows. A row is a string or a (long, short)
    pair: the short form is used in a narrow column, a row that still does not fit is dropped.
    build=True: the card is constructed instead of typed - its plate opens downwards, a spine is drawn along
    the rows, the title tag is made and flashes, every row is decoded out of noise from its tick on the
    spine (its figures spin before they lock), and the spine leaves."""
    col = lay.card
    if col is None:
        return
    x, y = col[0] + 12.0, CARD_Y
    w = col[1] - col[0] - 16.0
    out = []
    for k, r in enumerate(rows):
        long_, short = (r, None) if isinstance(r, str) else r
        if text_w(long_, L.T_SMALL) <= w:
            out.append((k, long_))
        elif short and text_w(short, L.T_SMALL) <= w:
            out.append((k, short))
    tsize = 34 if text_w(title, 34) + 18.0 <= w else 26
    wmax = max([text_w(s, L.T_SMALL) for _, s in out] + [text_w(title, tsize) + 16.0])
    rect = (x - 10.0, y - 46.0, x + wmax + 14.0, y + 36.0 + len(out) * 26.0)
    if build:
        f.occlude(rect[0], rect[1], rect[2], rect[1] + (rect[3] - rect[1]) * float(B.ease(B.lin(age, 0.0, 0.3))))
        lay.take(rect)
        ys0, ys1 = y + 14.0, y + 50.0 + (len(out) - 1) * 26.0 + 6.0
        pa = float(B.ease(B.lin(age, 0.05, 0.4)))                          # the spine comes down ...
        pz = float(B.ease(B.lin(age, CARD_BUILD - 0.25, CARD_BUILD)))      # ... and leaves from the top
        ya, yz = ys0 + (ys1 - ys0) * pz, ys0 + (ys1 - ys0) * pa
        if out and pz < 1.0 and pa > 0.0:
            f.segments("w", [x - 8.0], [ya], [x - 8.0], [yz], 0.8, width=1.3)
            if pa < 1.0:
                f.dots("w", [x - 8.0], [yz], 3.0, 1.7)
        B.tag(f, "r" if red_title else "w", x, y, title, age, t0=0.05, size=tsize, pad=8, bold=True, cps=50.0,
              commit=True)
        for j, (k, s) in enumerate(out):
            a = age - 0.3 - 0.07 * j
            if a < 0.0:
                continue
            yr = y + 50.0 + j * 26.0
            if pz < 1.0 and yr - 6.0 >= ya:
                f.segments("w", [x - 8.0], [yr - 6.0], [x - 2.0], [yr - 6.0], 0.8, width=1.3)
            f.text("r" if k in red_rows else "w", x, yr, B.resolve(s, a, cps=110.0, key=j), size=L.T_SMALL,
                   alpha=0.6 if k in dim_rows else 0.9)
        return
    f.occlude(*rect)
    lay.take(rect)
    f.tag("r" if red_title else "w", x, y, title[: int(age * 24) + 1], size=tsize, pad=8, bold=True)
    for j, (k, s) in enumerate(out):
        f.text("r" if k in red_rows else "w", x, y + 50.0 + j * 26.0, hud.typed(s, age, cps, 0.25 + 0.1 * j),
               size=L.T_SMALL, alpha=0.6 if k in dim_rows else 0.9)


def _note_col(lay):
    """Notes column and its side; without one, the right part of the focus bay if it is wide enough."""
    if lay.note is not None:
        return lay.note, lay.note_side
    x0 = lay.C[0] + lay.r_star + 60.0                # beside the star, inside the focus bay
    if lay.focus_col[1] - x0 >= 300.0:
        return (x0, lay.focus_col[1]), 1.0
    return None, 1.0


def axis_note(f, lay, title, below=(), above=(), title2=None, age=9.0, age2=None, alpha=1.0, size=30, red=True,
              build=False):
    """Read-out of the centre, set on the horizontal axis that runs through it (the axis is its leader and
    passes behind whatever tower stands in between): tag just above the axis at the near edge of the notes
    column, data lines above / below it, an optional second tag under the axis.
    build=True: constructed instead of typed - a pen draws the bold piece of axis, the plate opens from the
    axis, each tag is made and flashes, the lines are decoded out of noise and their figures spin, then lock."""
    col, side = _note_col(lay)
    if col is None or alpha <= 0.01:
        return
    cy = lay.C[1]
    xe = col[0] if side > 0 else col[1]
    tx = xe + side * 30.0
    anc = "ls" if side > 0 else "rs"
    width = col[1] - col[0] - 34.0
    la = [s for s in above if text_w(s, L.T_SMALL) <= width]
    lb = [s for s in below if text_w(s, L.T_SMALL) <= width]
    if text_w(title, size) + 14.0 > width:
        size = 22
    wmax = max([text_w(title, size) + 14.0] + ([text_w(title2, size) + 14.0] if title2 else [])
               + [text_w(s, L.T_SMALL) for s in la + lb])
    x0 = tx - 8.0 if side > 0 else tx - wmax - 8.0
    top = cy - 22.0 - size - 12.0 - 25.0 * len(la) - (8.0 if la else 0.0)
    a2 = age if age2 is None else age2
    low = a2 > 0 and bool(title2 or lb)
    bot = cy + 7.0 + ((size + 30.0 if title2 else 16.0) + 25.0 * len(lb) + 6.0 if low else 0.0)
    rect = (x0, top, x0 + wmax + 16.0, bot)
    lay_r = "r" if red else "w"
    if build:
        pad = side < 0                                   # right-anchored text keeps its length while it is written
        po = float(B.ease(B.lin(age, 0.08, 0.34)))       # the plate opens upwards from the axis ...
        pl = float(B.ease(B.lin(a2, 0.0, 0.25))) if low else 0.0             # ... and downwards for the second tag
        f.occlude(rect[0], cy - (cy - top) * po, rect[2], cy + 7.0 + (bot - cy - 7.0) * pl)
        lay.take(rect)
        B.tag(f, lay_r, tx, cy - 22.0, title, age, t0=0.12, size=size, pad=7, alpha=alpha, bold=True, anchor=anc,
              cps=40.0, key=1, commit=True)
        for k, s in enumerate(la):
            f.text("w", tx, cy - 22.0 - size - 14.0 - 25.0 * k, B.resolve(s, age, 70.0, 0.35 + 0.2 * k, key=k, pad=pad,
                                                                         spin=0.45),
                   size=L.T_SMALL, alpha=(0.9 if k == 0 else 0.7) * alpha, anchor=anc)
        if low:
            yb = cy + 34.0
            if title2:
                B.tag(f, lay_r, tx, cy + 14.0 + size, title2, a2, t0=0.05, size=size, pad=7, alpha=alpha, bold=True,
                      anchor=anc, cps=40.0, key=2, commit=True)
                yb = cy + 14.0 + size + 36.0
            for k, s in enumerate(lb):
                f.text("w", tx, yb + 25.0 * k, B.resolve(s, a2, 70.0, 0.3 + 0.2 * k, key=4 + k, pad=pad, spin=0.45),
                       size=L.T_SMALL, alpha=(0.9 if k == 0 else 0.7) * alpha, anchor=anc)
        B.pen(f, lay_r, xe + side * 2.0, cy, xe + side * (wmax + 38.0), cy, B.ease(B.lin(age, 0.0, 0.25)),
              0.95 * alpha, width=L.LW_BOLD, head=3.8)
        return
    f.occlude(*rect)
    lay.take(rect)
    f.tag(lay_r, tx, cy - 22.0, title[: int(age * 22) + 1], size=size, pad=7, alpha=alpha, bold=True, anchor=anc)
    for k, s in enumerate(la):
        f.text("w", tx, cy - 22.0 - size - 14.0 - 25.0 * k, hud.typed(s, age, 60, 0.3 + 0.2 * k), size=L.T_SMALL,
               alpha=(0.9 if k == 0 else 0.7) * alpha, anchor=anc)
    if low:
        yb = cy + 34.0
        if title2:
            f.tag(lay_r, tx, cy + 14.0 + size, title2[: int(a2 * 22) + 1], size=size, pad=7, alpha=alpha, bold=True,
                  anchor=anc)
            yb = cy + 14.0 + size + 36.0
        for k, s in enumerate(lb):
            f.text("w", tx, yb + 25.0 * k, hud.typed(s, a2, 60, 0.3 + 0.2 * k), size=L.T_SMALL,
                   alpha=(0.9 if k == 0 else 0.7) * alpha, anchor=anc)
    f.segments(lay_r, [xe + side * 2.0], [cy], [xe + side * (wmax + 38.0)], [cy], 0.95 * alpha, width=L.LW_BOLD)


def note(f, lay, y, title, lines=(), anchor=None, red=False, age=9.0, alpha=1.0, size=L.T_TAG, big=None,
         big_size=46, big_red=False):
    """A read-out in the notes column at height y (baseline of its tag): tag, an optional big figure, typed
    lines. With an anchor, a leader runs back to that point of the image: 45 degrees to the level of the tag,
    then straight to the column (behind a tower if one is there)."""
    col, side = _note_col(lay)
    if col is None or alpha <= 0.01:
        return False
    xe = col[0] if side > 0 else col[1]
    tx = xe + side * 46.0
    anc = "ls" if side > 0 else "rs"
    width = col[1] - col[0] - 50.0
    ln = [s for s in lines if text_w(s, L.T_SMALL) <= width]
    bs = int(min(big_size, width / text_w(big, 1.0))) if big else 0
    wmax = max([text_w(title, size) + 12.0] + [text_w(s, L.T_SMALL) for s in ln] + ([text_w(big, bs)] if big else []))
    x0 = tx - 8.0 if side > 0 else tx - wmax - 8.0
    y_l = y + (bs + 26.0 if big else 0.0)             # where the data lines hang from
    rect = (x0, y - size - 10.0, x0 + wmax + 16.0, y_l + 16.0 + 25.0 * len(ln))
    f.occlude(*rect)
    lay.take(rect)
    lay_r = "r" if red else "w"
    f.tag(lay_r, tx, y, title[: int(age * 30) + 1], size=size, pad=5, alpha=alpha, anchor=anc)
    if big:
        f.text("r" if big_red else "w", tx - 2.0 * side, y + 16.0 + bs * 0.86, hud.typed(big, age, 40, 0.1), size=bs,
               alpha=alpha, anchor=anc)
    for k, s in enumerate(ln):
        f.text("w", tx, y_l + 34.0 + 25.0 * k, hud.typed(s, age, 70, 0.15 + 0.1 * k), size=L.T_SMALL, alpha=0.9 * alpha,
               anchor=anc)
    if anchor is not None:
        ax, ay = anchor
        ym = y - size * 0.36
        hx = xe + side * 36.0
        ex = ax + side * abs(ym - ay)
        if (ex - hx) * side > 0:
            ex = hx
        f.segments(lay_r, [ax, ex], [ay, ym], [ex, hx], [ym, ym], 0.8 * alpha, width=L.LW)
        f.dots(lay_r, [ax], [ay], 2.8, 1.3 * alpha)
    return True


def cell_right(f, lay, label, value, sub=None, red=False, alpha=1.0, short=None, value_short=None, age=None):
    """Counter in the bottom-right cell; shortens itself if a tower squeezes the cell, vanishes if there is
    no room at all. age = seconds since it started to build (None = built), see hud.counter_cell."""
    x0, y0, x1, y1 = lay.cell_r
    w = x1 - x0
    if w < 210.0 or alpha <= 0.01:
        return
    sub_w = (text_w(sub, L.T_SMALL) + 24.0) if sub else 0.0
    if text_w(label, L.T_SMALL) + 46.0 + sub_w > w:
        if text_w(label, L.T_SMALL) + 46.0 <= w:
            sub = None
        else:
            label, sub = (short or ""), None
    if text_w(value, 38) + 46.0 > w and value_short:
        value = value_short
    hud.counter_cell(f, label, value, rect=lay.cell_r, alpha=alpha, sub=sub, red=red, age=age, from_right=True)


# ----------------------------------------------------------------------------
# the spiral field of streaks (signature texture of the intro, returns in the outro)
# ----------------------------------------------------------------------------

class StreakField:
    """Thousands of short strokes riding logarithmic spiral arms around a centre. Closed form: every stroke
    is a function of time (it drifts outward along its arm, the whole field turns)."""

    def __init__(self, n=42000, arms=9, pitch=1.28, r_max=1780.0, r_min=58.0, seed=5):
        rng = np.random.default_rng(seed)
        self.n, self.arms, self.pitch, self.r_max, self.r_min = n, arms, pitch, r_max, r_min
        self.arm = rng.integers(0, arms, n)
        self.u = rng.random(n)
        self.j = rng.normal(0.0, 1.0, n)                     # position across the arm
        self.len = rng.uniform(0.75, 1.3, n)
        self.b = rng.uniform(0.45, 1.0, n)
        self.tilt = rng.normal(0.0, 0.16, n)
        self.spd = rng.uniform(0.7, 1.3, n)
        self.bend = rng.normal(0.0, 0.22, n)
        self.grain = rng.random(n) < 0.12
        self.arm_ph = rng.uniform(-0.12, 0.12, arms)

    def draw(self, f, t, cx, cy, front=1e9, gain=1.0, omega=0.11, drift=0.014, squeeze=1.0, layer="w",
             r_scale=1.0, twist=0.0):
        """front = radius the field has reached; squeeze < 1 pulls every stroke towards the centre."""
        q = (self.u + drift * self.spd * t) % 1.0
        r = (self.r_min + (self.r_max - self.r_min) * q ** 0.62) * r_scale
        width = 0.105 + 0.07 * (r / self.r_max)
        phi = (2 * np.pi * self.arm / self.arms + self.arm_ph[self.arm] + self.pitch * np.log(r / self.r_min)
               + omega * t + width * self.j + twist * (1.0 - q))
        r = r * squeeze
        x = cx + r * np.cos(phi)
        y = cy + r * np.sin(phi)
        vis = (r < front) & (x > L.FX0 - 30) & (x < L.FX1 + 30) & (y > L.FY0 - 30) & (y < L.FY1 + 30)
        if not vis.any():
            return
        edge = np.exp(-((front - r) / 110.0) ** 2) if front < 5000 else 0.0
        birth = np.minimum(1.0, q / 0.03) * np.minimum(1.0, (1.0 - q) / 0.05)
        inten = gain * self.b * birth * (0.34 + 0.52 * np.exp(-0.5 * self.j ** 2)) * (1.0 + 1.2 * edge)
        psi = math.atan(self.pitch) + 0.55 * np.tanh(self.j) + self.tilt     # barbs of a feather
        ang = phi + psi
        ln = self.len * (11.0 + 15.0 * (r / self.r_max) ** 0.5)
        b = self.bend
        m = vis & (inten > 0.01)
        x, y, ang, ln, b, inten = x[m], y[m], ang[m], ln[m], b[m], inten[m]
        ax, ay = x - 0.5 * ln * np.cos(ang - b), y - 0.5 * ln * np.sin(ang - b)
        bx, by = x + 0.5 * ln * np.cos(ang + b), y + 0.5 * ln * np.sin(ang + b)
        f.segments(layer, np.r_[ax, x], np.r_[ay, y], np.r_[x, bx], np.r_[y, by],
                   np.r_[inten * 0.5, inten], np.r_[inten, inten * 0.7], width=1.5, spacing=0.8)
        g = self.grain[m]
        f.dots(layer, bx[g], by[g], 1.4, 1.5 * inten[g])


# ----------------------------------------------------------------------------
# the rings
# ----------------------------------------------------------------------------

class Ripples:
    """Rings leaving the dot: each settles on its own radius and keeps breathing there (irregular spacing,
    doubles and triples, slightly off-centre: the TouchDesigner look). The widest goes first; every sound
    event of the music throws one more ring that travels out and fades."""

    def __init__(self, t0, onsets, far=1780.0, seed=11):
        rng = np.random.default_rng(seed)
        self.far = far
        fam = np.sort(np.r_[rng.uniform(70, 780, 9), rng.uniform(780, max(far - 60.0, 900.0), 7)])[::-1]
        R, tau, bold = [], [], []
        self.pulses = np.array([t for t in onsets if t0 + 2.0 < t < T_STAR])
        for k, rk in enumerate(fam):
            n = int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25]))
            ts = t0 + 0.04 + 0.17 * k + rng.uniform(0, 0.08)
            for j in range(n):
                R.append(rk + rng.normal(0, 4.0) + j * rng.uniform(7, 19))
                tau.append(ts + 0.09 * j)
                bold.append(rng.random() < 0.3)
        self.R = np.array(R)
        self.tau = np.array(tau)
        self.bold = np.array(bold)
        n = len(R)
        self.T = rng.uniform(0.85, 1.15, n) * (0.55 + 0.5 * self.R / 900.0)
        self.off = rng.normal(0, 3.6, (n, 2))
        self.fq = rng.uniform(0.07, 0.19, n)
        self.ph = rng.uniform(0, 2 * np.pi, n)
        self.i = rng.uniform(0.5, 0.95, n)

    def radii(self, t, kick=0.0):
        a = np.maximum(t - self.tau, 0.0)
        r = self.R * (1.0 - np.exp(-a / self.T)) * (1.0 + 0.016 * np.sin(2 * np.pi * self.fq * t + self.ph))
        return r + 7.0 * kick * (0.3 + 0.7 * hash01(np.arange(len(r)), 3))

    def draw(self, f, t, cx, cy, gain=1.0, kick=0.0, scale=1.0, r_min=30.0):
        r = self.radii(t, kick) * scale
        age = t - self.tau
        inten = gain * self.i * np.clip(age / 0.25, 0, 1) * (0.55 + 0.45 * np.exp(-r / 900.0))
        inten = inten * (1.0 + 1.2 * np.exp(-np.maximum(age, 0) / 0.5))
        m = (age > 0) & (r > r_min)
        for sel, w in ((m & self.bold, L.LW_BOLD), (m & ~self.bold, 1.3)):
            if sel.any():
                f.rings("w", cx + self.off[sel, 0], cy + self.off[sel, 1], r[sel], inten[sel], width=w)
        a = t - self.pulses
        pm = (a > 0) & (a < 4.5)
        if pm.any() and scale == 1.0:
            pr = (self.far + 30.0) * (1.0 - np.exp(-a[pm] / 1.35))
            f.rings("w", np.full(pm.sum(), cx), np.full(pm.sum(), cy), pr, gain * 0.8 * np.exp(-a[pm] / 1.5), width=1.3)


# ----------------------------------------------------------------------------
# the collapse: rays in 3D (shared with messenger.py, which follows one of them)
# ----------------------------------------------------------------------------

class Nova:
    N_RAYS = 120

    def __init__(self, lay, seed=23):
        rng = np.random.default_rng(seed)
        self.C, self.half = lay.C, lay.half
        n = self.N_RAYS
        d = _unit(rng.normal(size=(n, 3)))
        speed = rng.uniform(0.35, 1.0, n) ** 0.8
        delay = np.where(rng.random(n) < 0.6, 0.0, rng.choice([0.64, 1.14, 1.64, 2.14, 2.64, 3.13], n))
        bright = rng.uniform(0.55, 1.0, n)
        bold = rng.random(n) < 0.45
        fade_t = rng.uniform(T_END + 0.8, 47.9, n)           # when the other messengers leave the picture
        kind = rng.choice(["P+", "P+", "P+", "P+", "HE", "HE", "C", "O", "FE"], n)
        self._aim()
        away = (d @ self.hero_dir) < 0.93                    # keep the other rays out of the hero's way
        self.dir, self.speed, self.delay, self.bright = d[away], speed[away], delay[away], bright[away]
        self.bold, self.fade_t, self.kind = bold[away], fade_t[away], kind[away]
        self.n = len(self.dir)
        R = math.ceil((lay.far + 40.0) / 64.0) * 64.0
        g = np.arange(-R, R + 0.01, 64.0)
        X, Y = np.meshgrid(g, g)
        keep = np.hypot(X, Y) < R + 20
        self.lattice = np.stack([X[keep], Y[keep], np.zeros(keep.sum())], 1).astype(np.float32)

    # -- camera --------------------------------------------------------------
    def camera(self, t):
        """Frontal until the bounce (world x, y = design px around the centre), then it drops and circles."""
        u = float(smoothstep(T_X, T_END + 0.5, t))
        eps = math.radians(58.0) * u ** 0.8
        alp = math.radians(-24.0 + 46.0 * u)
        D = FOCAL * (1.0 + 0.5 * u)
        pos = D * np.array([math.sin(eps) * math.sin(alp), -math.sin(eps) * math.cos(alp), math.cos(eps)])
        up = (0.0, 1.0, 0.0) if eps < 1e-4 else (0.0, math.cos(eps), math.sin(eps))
        return Camera(pos, (0.0, 0.0, 0.0), fov_deg=FOV, up=up, screen_center=self.C)

    # -- the hero ray ----------------------------------------------------------
    def _set_hero(self, at, depth):
        cam = self.camera(T_END)
        d = np.array([(at[0] - self.C[0]) / FOCAL, -(at[1] - self.C[1]) / FOCAL, 1.0]) * depth
        self.hero_at = at
        self.hero_tip_end = cam.pos + d @ cam.R.astype(np.float64)
        self.hero_dir = _unit(self.hero_tip_end)
        self.hero_len_end = float(np.linalg.norm(self.hero_tip_end))

    def _aim(self):
        """Direction of the ray that will reach us: its tip must stay inside the focus bay - never behind a
        tower - all the way while the camera drops and circles (40 - 44 s)."""
        ts = np.arange(T_X + 0.2, T_END + 0.01, 0.2)
        lim = self.half - 46.0
        for fx in (0.55, 0.45, 0.35, 0.25, 0.15, 0.05):
            for depth in (1000.0, 800.0, 1300.0):
                self._set_hero((self.C[0] + fx * self.half, self.C[1] + 20.0), depth)
                xy = np.array([self.tip_screen(t) for t in ts])
                if (np.abs(xy[:, 0] - self.C[0]).max() < lim and xy[:, 1].max() < L.VIEW[3] - 70.0
                        and xy[:, 1].min() > L.HEAD_Y + 40.0):
                    return

    def hero_len(self, t):
        u = np.clip((t - T_X) / (T_END - T_X), 0.0, None)
        return self.hero_len_end * u ** 1.35

    def hero_tip(self, t):
        return self.hero_dir * self.hero_len(t)

    def tip_screen(self, t):
        x, y, _, _ = self.camera(t).project(self.hero_tip(t)[None].astype(np.float32))
        return float(x[0]), float(y[0])

    # -- state ---------------------------------------------------------------
    def lengths(self, t):
        a = np.maximum(t - T_X - self.delay, 0.0)
        return 1500.0 * self.speed * a ** 1.25 + 60.0 * np.minimum(a * 8.0, 1.0)

    @staticmethod
    def _clip_near(cam, A, B, near=40.0):
        """Clip 3D segments A->B to the half-space in front of the camera; returns projected ends + mask."""
        R = cam.R.astype(np.float64)
        za = (A - cam.pos) @ R[2]
        zb = (B - cam.pos) @ R[2]
        ok = (za > near) | (zb > near)
        sa = np.where(za < near, (near - za) / np.where(zb != za, zb - za, 1.0), 0.0)
        sb = np.where(zb < near, (near - za) / np.where(zb != za, zb - za, 1.0), 1.0)
        A2 = A + (B - A) * sa[:, None]
        B2 = A + (B - A) * sb[:, None]
        ax, ay, az, _ = cam.project(A2)
        bx, by, bz, _ = cam.project(B2)
        return ax, ay, np.maximum(az, near), bx, by, np.maximum(bz, near), ok, sb >= 1.0

    def draw_plane(self, f, cam, rip, t, gain=1.0):
        """The ring plane and its lattice, seen by the camera."""
        P = self.lattice
        sx, sy, z, ok = cam.project(P)
        sc = FOCAL / np.maximum(z, 1.0)
        m = ok & (sx > L.FX0 - 20) & (sx < L.FX1 + 20) & (sy > L.FY0 - 20) & (sy < L.FY1 + 20)
        f.crosses("w", sx[m], sy[m], 8.0 * sc[m].mean() if m.any() else 8.0, 0.22 * gain)
        r = rip.radii(t)
        a = np.linspace(0, 2 * np.pi, 181)
        sel = np.nonzero((t - rip.tau > 0) & (r > 60))[0]
        shock = 260.0 + 1250.0 * (1.0 - math.exp(-(t - T_X) / 1.6))
        for k in sel:
            rr = r[k] * (1.0 + 0.3 * math.exp(-((r[k] - shock) / 180.0) ** 2))
            P = np.stack([rr * np.cos(a) + rip.off[k, 0], rr * np.sin(a) + rip.off[k, 1], np.zeros_like(a)], 1)
            px, py, pz, pok = cam.project(P.astype(np.float32))
            if pok.all():
                f.polyline("w", px, py, gain * rip.i[k] * 0.6, width=L.LW_BOLD if rip.bold[k] else 1.2)
        P = np.stack([shock * np.cos(a), shock * np.sin(a), np.zeros_like(a)], 1)
        px, py, pz, pok = cam.project(P.astype(np.float32))
        if pok.all():
            f.polyline("w", px, py, gain * 1.2 * math.exp(-(t - T_X) / 2.2), width=3.2)

    def draw_rays(self, f, cam, t, gain=1.0):
        Lr = self.lengths(t)
        live = Lr > 1.0
        fade = np.clip((self.fade_t - t) / 0.7, 0.0, 1.0)
        A = np.zeros((self.n, 3))
        B = self.dir * Lr[:, None]
        ax, ay, az, bx, by, bz, ok, tip_in = self._clip_near(cam, A, B)
        m = live & ok & (fade > 0)
        near = np.clip(FOCAL / bz, 0.6, 3.0)
        inten = gain * self.bright * fade * (0.75 + 0.5 * np.clip(near - 1.0, 0, 2))
        for sel, w in ((m & self.bold, 3.4), (m & ~self.bold, 2.0)):
            if sel.any():
                f.segments("w", ax[sel], ay[sel], bx[sel], by[sel], inten[sel] * 1.15, inten[sel] * 0.8, width=w)
        tm = m & tip_in
        f.dots("w", bx[tm], by[tm], np.clip(4.2 * near[tm], 3.0, 9.0), 1.5 * inten[tm])

    def draw_hero(self, f, cam, t, gain=1.0):
        """The ray that will reach us: a red hair inside the white, a red point at its tip. Returns the tip."""
        if t <= T_X + 0.05:
            return None
        tip = self.hero_tip(t)[None]
        ax, ay, az, bx, by, bz, ok, tip_in = self._clip_near(cam, np.zeros((1, 3)), tip)
        if not ok[0]:
            return None
        f.segments("w", ax, ay, bx, by, 1.2 * gain, 1.0 * gain, width=3.4)
        f.segments("r", ax, ay, bx, by, 0.5 * gain, 1.3 * gain, width=2.0)
        x, y = float(bx[0]), float(by[0])
        f.dots("w", [x], [y], 7.5, 1.6 * gain)
        f.dots("r", [x], [y], 13.0, 0.9 * gain)
        return x, y


def hero_note(f, lay, tip, age, alpha=1.0):
    """The tag of the ray that will reach us: in the notes column, a red leader to its tip."""
    if tip is None or age <= 0 or alpha <= 0.01:
        return
    ok = lay.note is not None and note(f, lay, lay.C[1] + 150.0, "P+ // THE ONE THAT WILL REACH US",
                                       ["PRIMARY COSMIC RAY // PROTON", "E 3.2E15 EV", "HEADING: HERE"],
                                       anchor=(tip[0] + 12.0 * lay.note_side, tip[1] + 10.0), red=True, age=age, alpha=alpha)
    if not ok:                                       # no notes column: a small tag on the ray itself
        f.occlude(tip[0] + 20.0, tip[1] + 16.0, tip[0] + 70.0, tip[1] + 50.0)
        f.tag("r", tip[0] + 26.0, tip[1] + 42.0, "P+", size=L.T_TAG, pad=5, alpha=alpha)


# ----------------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------------

class Origin(Scene):
    name = "origin"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.lay = Lay(ctx)
        self.C = self.lay.C
        far = self.lay.far
        self.field = StreakField(n=int(min(70000, 42000 * ((far + 80.0) / 1780.0) ** 2)), r_max=far + 80.0)
        self.rip = Ripples(T_HERE, list(ctx.cues.onset_t), far)
        self.nova = Nova(self.lay)
        rng = np.random.default_rng(3)
        # arrival times of the muons through one spectator, 63 / s (Poisson), and their energies
        gaps = rng.exponential(1.0 / RATE_YOU, int(RATE_YOU * 50))
        self.arr_t = np.cumsum(gaps)
        self.arr_e = np.exp(rng.normal(1.2, 0.8, len(gaps)))                 # GeV
        # loading bar: it creeps, and jumps on every low thump of the opening
        kt, ka = ctx.cues.kicks(T_LOAD, T_LOADED - 0.12)
        self.load_t = np.asarray(kt, np.float64)
        self.load_w = 0.7 * np.asarray(ka, np.float64) / max(float(np.sum(ka)), 1e-6)
        # ... and when each block of the bar opens
        ts = np.r_[np.arange(T_LOAD, T_LOADED, 1.0 / 240.0), T_LOADED]
        ps = np.array([self._load_p(float(v)) for v in ts])
        self.load_block_t = np.array([float(ts[np.argmax(ps >= (k + 0.5) / LOAD_N)]) for k in range(LOAD_N)])
        # star granulation
        n = 5200
        r = np.sqrt(rng.random(n))
        a = rng.uniform(0, 2 * np.pi, n)
        self.gran = np.stack([r * np.cos(a), r * np.sin(a)], 1)
        self.gran_r = rng.uniform(3.5, 8.5, n)
        self.gran_b = rng.uniform(0.25, 0.8, n)
        self.gran_ph = rng.uniform(0, 2 * np.pi, n)

    # ------------------------------------------------------------------ draw
    def draw(self, f, t, ctx):
        self.lay.avoid = []
        if t < T_STAR + 0.05:
            return self._origin(f, t, ctx)
        elif t < T_X:
            self._star(f, t, ctx)
        else:
            return self._nova(f, t, ctx)
        return {}

    def _panels(self, f, fns, alpha=1.0):
        """Bottom band: the panels, by priority, in the free slots between the towers (extra ones dropped)."""
        y0, y1 = L.BOT[1], L.BOT[3]
        for (x0, x1), fn in zip(self.lay.slots, fns):
            f.occlude(x0 - 10.0, y0 - 26.0, x1 + 10.0, L.FY1 - 3.0)
            fn(f, x0, x1, y0, y1, alpha)

    def _panels_build(self, f, fns, age, lag=0.15):
        """The same panels, building up instead of fading in, one after the other (age = seconds since the
        first one started): registration brackets say where a panel will be, its plate opens downwards,
        then the panel constructs itself (each fn gets the age of its panel instead of an alpha)."""
        y0, y1 = L.BOT[1], L.BOT[3]
        for k, ((x0, x1), fn) in enumerate(zip(self.lay.slots, fns)):
            a = age - lag * k
            if a < 0.0:
                continue
            top, bot = y0 - 26.0, L.FY1 - 3.0
            f.occlude(x0 - 10.0, top, x1 + 10.0, top + (bot - top) * float(B.ease(B.lin(a, 0.0, 0.25))))
            if B.marks_on(a, 0.95):
                B.brackets(f, (x0 - 8.0, y0 - 24.0, x1 + 8.0, y1 + 8.0))
            fn(f, x0, x1, y0, y1, a)

    # --- 00:00 - 00:31 ---------------------------------------------------------
    def _boot(self, t):
        """Finishing options of the opening: black, the border fades in, the furniture builds up once loaded
        (the counter with 'something is passing through you'). Data never fades in: it is constructed."""
        fa = float(smoothstep(T_FRAME, T_FRAME + 2.2, t))
        ca = t - (T_HERE + 1.6)
        return {"frame_alpha": fa, "tower_outline": 0.16 * fa, "edge_ticks": t >= T_LOADED,
                "edge_kw": {"reveal": B.lin(t, T_LOADED, T_READY)}, "cell": ca > 0.0, "cell_age": ca}

    def _load_p(self, t):
        creep = 0.3 * float(np.clip((t - T_LOAD) / (T_LOADED - T_LOAD), 0.0, 1.0))
        jumps = float(np.sum(self.load_w * smoothstep(self.load_t, self.load_t + 0.12, t)))
        return 1.0 if t >= T_LOADED else min(creep + jumps, 0.99)

    def _loading(self, f, t):
        """Top left, on black: MUON BLOOM // LOADING. No fade, it is constructed: a block cursor blinks, the
        title tag is made, two pens trace the bar, the figure spins and locks; then each block opens from the
        centre line as the bar reaches it (it creeps, and jumps on the low thumps). At 100 % the whole thing
        flashes once, and it is taken apart - backwards - while the crosses are drawn."""
        age = t - T_LOAD
        out = B.lin(t, T_LOADED + 0.3, T_BOOT + 0.05)          # 0 -> 1: taken apart
        if age < 0.0 or out >= 1.0:
            return
        fr = B.frame_no(age)
        p = self._load_p(t)
        done = t >= T_LOADED
        name = "MUON BLOOM"
        x0, y0 = L.FX0 + 44.0, L.FY0 + 40.0
        x1 = x0 + 700.0
        ty, by0, by1 = y0 + 34.0, y0 + 58.0, y0 + 86.0
        bm, hh = 0.5 * (by0 + by1), 0.5 * (by1 - by0)
        gone = (lambda s, key: hud.erode(s, min(out * 2.0, 1.0), key, fr)) if out > 0.0 else (lambda s, key: s)
        if out > 0.0:
            pc, px, pr = 1.0 - B.lin(out, 0.9, 1.0), 1.0 - float(B.ease(B.lin(out, 0.3, 0.9))), 1.0 - B.lin(out, 0.2, 0.3)
            wipe = 1.0 - B.lin(out, 0.5, 1.0)
        else:
            pc, px, pr = B.lin(age, 0.35, 0.42), float(B.ease(B.lin(age, 0.42, 0.74))), B.lin(age, 0.74, 0.8)
            wipe = 1.0
        xe = x0 + (x1 - x0) * px
        if done:
            # the crosses are being written behind it: it hides them, and uncovers them as it is taken apart
            cover = max(xe + 8.0, x0 + 21.0 + text_w(name, 30) * wipe) if out > 0.0 else x1 + 14.0
            f.occlude(x0 - 14.0, y0 - 6.0, cover, by1 + 42.0)
        # 1 - a block cursor blinks twice where the title will be, then the title tag is made
        if age < 0.2:
            if fr % 4 < 2:
                f.rects("w", x0, ty - 27.0, x0 + 18.0, ty + 7.0, 0.95)
            return
        if out > 0.0:
            f.tag("w", x0 + 7.0, ty, gone(name, 1), size=30, pad=7, bold=True, ref=name, wipe=wipe)
        else:
            B.tag(f, "w", x0 + 7.0, ty, name, age, t0=0.2, size=30, pad=7, bold=True, cps=45.0, key=1)
        if done:
            word = B.decode("LOADED", t - T_LOADED, cps=60.0, key=2)
        else:
            word = B.decode("LOADING", age, cps=45.0, delay=0.5, key=2) + ("." * (int(age * 2.5) % 4) if age > 0.9 else "")
        f.text("r" if done else "w", x0 + 29.0 + text_w(name, 30), ty, gone(word, 2), size=30)
        f.text("w", x1, ty, gone(B.roll(f"{int(100 * p):03d} %", age, 0.3, 0.75, key=3), 3), size=30, anchor="rs")
        # 2 - the bar: two pens leave the middle of its left end, run along its top and bottom, and close it
        if pc > 0.0:
            f.segments("w", [x0], [bm - hh * pc], [x0], [bm + hh * pc], 0.9, width=L.LW)
        if px > 0.0:
            f.segments("w", [x0, x0], [by0, by1], [xe, xe], [by0, by1], 0.9, width=L.LW)
            if px < 1.0 and out <= 0.0:
                f.dots("w", [xe, xe], [by0, by1], 3.2, 1.7)
        if pr > 0.0:
            f.segments("w", [x1, x1], [by0, by1], [x1, x1], [by0 + hh * pr, by1 - hh * pr], 0.9, width=L.LW)
        # 3 - the blocks: each one opens from the centre line when the bar reaches it, and overshoots
        k = np.arange(LOAD_N)
        pitch = (x1 - x0 - 7.0) / LOAD_N
        xs = x0 + 5.0 + k * pitch
        g = B.spring((t - self.load_block_t) / 0.2)
        if out > 0.0:                                   # ... and they close again, from the right
            g = g * (1.0 - B.ease((out - 0.5 * (LOAD_N - 1 - k) / LOAD_N) / 0.3))
        on = (t >= self.load_block_t) & (xs + pitch - 3.0 <= xe) & (g > 0.02)
        f.rects("w", xs[on], bm - (hh - 5.0) * g[on], xs[on] + pitch - 3.0, bm + (hh - 5.0) * g[on], 0.95)
        nxt = int(on.sum())
        if not done and px >= 1.0 and nxt < LOAD_N and fr % 4 < 2:        # the next block, as a blinking outline
            f.rect("w", xs[nxt], bm - hh + 5.0, xs[nxt] + pitch - 3.0, bm + hh - 5.0, 0.9)
        # 4 - what is being loaded: one line after the other, each decoded as it comes up
        if done:
            s = B.decode("READY", t - T_LOADED, cps=60.0, key=9)
        else:
            slot = (T_LOADED - 0.2 - (T_LOAD + 0.8)) / len(LOAD_STEPS)
            ks = int(np.clip((age - 0.8) / slot, 0, len(LOAD_STEPS) - 1))
            s = B.decode(f"{ks + 1:02d}/{len(LOAD_STEPS):02d}  {LOAD_STEPS[ks]}", age - 0.8 - ks * slot, cps=140.0,
                         key=10 + ks)
        f.text("w", x0, by1 + 30.0, gone(s, 4), size=L.T_SMALL, alpha=0.7)
        B.flash(f, (x0 - 8.0, y0 - 2.0, x1 + 8.0, by1 + 8.0), t - T_LOADED, 0.0)

    def _lattice(self, f, t, gain=1.0):
        a = gain
        if t < T_LOADED or a <= 0.01:
            return
        C = self.C
        x0, y0, x1, y1 = L.FRAME
        kx = np.arange(math.ceil((x0 + 14 - C[0]) / 64.0), math.floor((x1 - 14 - C[0]) / 64.0) + 1)
        ky = np.arange(math.ceil((y0 + 14 - C[1]) / 64.0), math.floor((y1 - 14 - C[1]) / 64.0) + 1)
        KX, KY = np.meshgrid(kx, ky)
        X, Y = (C[0] + KX * 64.0).ravel(), (C[1] + KY * 64.0).ravel()
        major = ((KX % 4 == 0) & (KY % 4 == 0)).ravel()
        # drawn row by row, top to bottom, between LOADED and the red dot; the row being written is brighter
        row = (KY - ky[0]).ravel()
        head = (t - T_LOADED) / (T_BOOT - 0.05 - T_LOADED) * len(ky)
        on = row < head
        f.crosses("w", X[on & ~major], Y[on & ~major], 7.0, 0.2 * a)
        f.crosses("w", X[on & major], Y[on & major], 10.0, 0.42 * a, width=1.3)
        new = on & (row >= head - 1.0)
        if new.any() and head < len(ky) + 1.0:
            f.crosses("w", X[new], Y[new], 9.0, 0.9 * a, width=1.3)

    def _pulse(self, t, ctx):
        return min(1.5, ctx.cues.kick(t, 0.28) * 1.6 + 0.5 * ctx.cues.onset(t, 0.3))

    def _dot(self, f, t, ctx, size=1.0):
        """The red dot and its crosshair (the axes of the whole image). Returns its radius."""
        C = self.C
        born = float(smoothstep(T_BOOT, T_BOOT + 0.15, t))
        if born <= 0:
            return 0.0
        p = self._pulse(t, ctx)
        r = (16.0 + 4.0 * math.sin(2 * math.pi * t / 2.9) + 13.0 * p) * born * size
        r = max(r, 2.6)
        f.dots("r", [C[0]], [C[1]], r, 1.25)
        if r > 9:
            f.rings("w", [C[0]], [C[1]], [r + 3.0], 0.75, width=1.5)
        grow = float(smoothstep(T_BOOT + 0.05, T_READY, t))
        if grow > 0:
            g = r + 14
            xl, xr = C[0] - g - (C[0] - g - L.FX0) * grow, C[0] + g + (L.FX1 - C[0] - g) * grow
            yt, yb = C[1] - g - (C[1] - g - L.FY0) * grow, C[1] + g + (L.FY1 - C[1] - g) * grow
            f.segments("r", [xl, C[0] + g, C[0], C[0]], [C[1], C[1], yt, C[1] + g],
                       [C[0] - g, xr, C[0], C[0]], [C[1], C[1], C[1] - g, yb], 0.42, width=1.2)
            k = np.arange(1, 30) * 100.0
            for sgn in (-1.0, 1.0):
                xs = C[0] + sgn * k
                xs = xs[(xs > xl) & (xs < xr)]
                f.segments("r", xs, np.full_like(xs, C[1] - 6), xs, np.full_like(xs, C[1] + 6), 0.6)
                ys = C[1] + sgn * k
                ys = ys[(ys > yt) & (ys < yb)]
                f.segments("r", np.full_like(ys, C[0] - 6), ys, np.full_like(ys, C[0] + 6), ys, 0.6)
        return r

    def _origin(self, f, t, ctx):
        lay, C = self.lay, self.C
        opt = self._boot(t)
        if t < T_LOADED:
            self._loading(f, t)
            return opt
        mini = float(smoothstep(T_MINI, T_MINI + 1.1, t))
        self._lattice(f, t, gain=1.0 - 0.45 * float(smoothstep(T_FIELD, T_FIELD + 6, t)))
        self._loading(f, t)                 # on top of the lattice: it hides the crosses written under it
        kick = self._pulse(t, ctx)
        f.set_clip(*FRAME_CLIP)
        if t > T_FIELD:
            front = 60.0 + (lay.far + 120.0) * float(smoothstep(T_FIELD, T_FIELD + 8.5, t)) ** 1.25
            self.field.draw(f, t, C[0], C[1], front=front, gain=0.9 + 0.25 * kick)
        if t > T_HERE:
            self.rip.draw(f, t, C[0], C[1], kick=kick)
        f.set_clip()
        self._dot(f, t, ctx, size=1.0 - 0.94 * mini)
        if t <= T_HERE - 0.6:
            return opt
        self._strip_arrivals(f, t, ctx)
        a = t - (T_HERE + 1.2)
        if a > 0:
            card(f, lay, "SOMETHING", [("MU FLUX        1 /CM2/MIN", "FLUX  1 /CM2/MIN"),
                                       ("SEA LEVEL      ~170 /M2/S", "~170 /M2/S"),
                                       ("THROUGH YOU    ~63 /S", "YOU   ~63 /S"),
                                       ("THROUGH WALL   ~60 800 /S", "WALL  ~60 800 /S"),
                                       ("MEAN ENERGY    4 GEV", "MEAN  4 GEV"), ("SPEED          0.9997 C", "SPEED 0.9997 C"),
                                       ("CHARGE         + OR -", None), ("LIFETIME       2.197 US", "LIFE  2.197 US")],
                 a, red_rows=(2,), build=True)
        self._notes(f, t, mini)
        self._dial(f, t)
        if t > T_HERE + 0.8:
            self._panels_build(f, [lambda f, x0, x1, y0, y1, a_: self._p_felt(f, t, x0, x1, y0, y1, a_),
                                   lambda f, x0, x1, y0, y1, a_: self._p_arrivals(f, t, x0, x1, y0, y1, a_),
                                   lambda f, x0, x1, y0, y1, a_: self._p_scale(f, t, x0, x1, y0, y1, a_),
                                   lambda f, x0, x1, y0, y1, a_: self._p_energy(f, t, x0, x1, y0, y1, a_)],
                               t - (T_HERE + 0.8))
        cell_right(f, lay, "THROUGH THIS WALL // SINCE 00:00", f"{int(RATE_WALL * t):,}".replace(",", " "),
                   sub="364 M2", age=t - (T_HERE + 1.8), short="THROUGH THIS WALL")
        return opt

    def _notes(self, f, t, mini):
        """What the voice says about the dot, written on its axis in the notes column."""
        lay, C = self.lay, self.C
        if T_HERE <= t < T_ALWAYS:
            a = t - T_HERE
            al = 1.0 - float(smoothstep(T_ALWAYS - 0.8, T_ALWAYS - 0.1, t))
            axis_note(f, lay, "HERE", above=["CINCINNATI // 147 M ASL", "39.103 N  084.512 W"], title2="RIGHT NOW",
                      below=[f"T {sd.tc(t)}", "1 MUON EVERY 16 MS"], age=a, age2=a - 0.75, alpha=al, build=True)
        elif T_ALWAYS <= t < T_MINI:
            al = 1.0 - float(smoothstep(T_MINI - 0.8, T_MINI - 0.1, t))
            axis_note(f, lay, "ALWAYS", above=["DAY AND NIGHT // INDOORS AND OUT"],
                      below=["2E9 THROUGH YOU EVERY YEAR", "1.6E11 IN A LIFETIME", "NOT ONE OF THEM FELT"][: 2 + (t >= T_FEEL)],
                      age=t - T_ALWAYS, alpha=al, red=False)
        if mini > 0:
            g = 150.0 - 118.0 * mini
            for sx in (-1, 1):
                for sy in (-1, 1):
                    x, y = C[0] + sx * g, C[1] + sy * g
                    f.segments("r", [x, x], [y, y], [x - sx * 26, x], [y, y - sy * 26], 1.0 * mini, width=L.LW_BOLD)
            a = t - T_MINI - 0.5
            if a > 0:
                axis_note(f, lay, "MU // POINT-LIKE", below=["SIZE   < 1E-18 M", "MASS   1.88E-28 KG", "CHARGE -1 E",
                                                           "NO STRUCTURE FOUND"], age=a)

    def _dial(self, f, t):
        """A fixed scale among the moving rings: light-time shells, 1 px = 1 cm on the wall.
        Built, not faded in: a pen shoots out of the dot and brakes; every tick is thrown out long as the pen
        passes it and falls back to its length; its label is decoded out of noise."""
        age = t - (T_HERE + 1.2)
        if age <= 0.0:
            return
        lay, C = self.lay, self.C
        ang = math.radians(-27.0) if lay.note_side > 0 else math.radians(-153.0)
        ca, sa = math.cos(ang), math.sin(ang)
        rmax = min(lay.far, (C[1] - L.HEAD_Y - 30.0) / abs(sa))
        grow = 1.5
        u = float(B.ease(age / grow))
        reach = rmax * u
        rr = np.arange(100.0, rmax, 100.0)
        tr = grow * (1.0 - (1.0 - rr / rmax) ** (1.0 / 3.0))        # when the pen passes each tick
        on = age >= tr
        if reach > 60.0:
            f.segments("w", [C[0] + 60 * ca], [C[1] + 60 * sa], [C[0] + reach * ca], [C[1] + reach * sa], 0.6, width=1.3)
            if u < 1.0:
                f.dots("w", [C[0] + reach * ca], [C[1] + reach * sa], 3.2, 1.7)
        hl = 9.0 + 17.0 * np.exp(-(age - tr[on]) / 0.1)
        f.segments("w", C[0] + rr[on] * ca + hl * sa, C[1] + rr[on] * sa - hl * ca, C[0] + rr[on] * ca - hl * sa,
                   C[1] + rr[on] * sa + hl * ca, 0.9, width=1.3)
        right = lay.note_side > 0
        for k, r in enumerate(rr):
            if k % 2 == 0 or not on[k]:
                continue
            x, y = C[0] + r * ca, C[1] + r * sa
            s = f"{r / 100:.0f} M  {r / 100 / 0.2998:04.1f} NS"
            w = text_w(s, L.T_SMALL)
            rect = (x + 8.0, y + 8.0, x + 16.0 + w, y + 32.0) if right else (x - 16.0 - w, y + 8.0, x - 8.0, y + 32.0)
            txt = B.resolve(s, age - tr[k], 70.0, 0.06, key=k, pad=not right)
            if not txt.strip() or not lay.free(*rect):
                continue
            f.occlude(*rect)
            f.text("w", x + 12.0 if right else x - 12.0, y + 26.0, txt, size=L.T_SMALL, alpha=0.85,
                   anchor="ls" if right else "rs")

    def _strip_arrivals(self, f, t, ctx):
        """Score strip: every muon through one spectator as a tick (63 / s), the last five seconds.
        It builds up in a good second, without a fade: the red band shoots across, the rules and their ticks
        follow it (hud.strip_base); then a scan head prints the arrivals - close to the head they are still
        noise and settle behind it; the cursor drops, its tag is made and its count spins before it locks."""
        age = t - (T_HERE - 0.6)
        x0, y0, x1, y1 = L.STRIP
        f.occlude(x0, y0, x0 + (x1 - x0) * float(B.ease(B.lin(age, 0.0, 0.36))), y1)
        span, ahead = 5.0, 0.9
        ta, tb = t - span, t + ahead
        ix0, iy0, ix1, iy1, yb = hud.strip_base(f, title="ARRIVALS // MUONS THROUGH ONE SPECTATOR // ~63 PER SECOND",
                                                ticks=(ta, tb, 0.1, 1.0), age=age)
        X = lambda v: ix0 + (np.asarray(v) - ta) / (tb - ta) * (ix1 - ix0)
        i0, i1 = np.searchsorted(self.arr_t, [max(ta, 0.0), t])
        tt, ee = self.arr_t[i0:i1], self.arr_e[i0:i1]
        xs = X(tt)
        hh = np.clip(9.0 + 13.0 * np.log1p(ee), 6, 50)
        ps = B.lin(age, 0.4, 1.0)                           # the scan head that prints the arrivals
        if ps < 1.0:
            xh = ix0 + (ix1 - ix0) * ps
            m = (xs <= xh) & (ps > 0.0)
            near = np.clip((xh - xs[m]) / 110.0, 0.0, 1.0)             # 0 under the head -> 1 settled
            noise = 6.0 + 44.0 * B.rnd(np.arange(i0, i1)[m], 21, B.frame_no(age))
            xs, tt, ee, hh = xs[m], tt[m], ee[m], hh[m] * near + noise * (1.0 - near)
            if ps > 0.0:
                f.segments("w", [xh], [iy0 - 4], [xh], [iy1 + 4], 1.3, width=1.3)
        fresh = np.exp(-(t - tt) / 0.25)
        f.rects("w", xs, iy0 + 1, xs + 2, iy0 + 1 + hh, 0.7 + 0.3 * fresh)
        hot = ee > 9.0
        f.rects("r", xs[hot], iy1 - 1 - hh[hot] * 0.9, xs[hot] + 3, iy1 - 1, 0.95)
        lo = ~hot
        f.rects("w", xs[lo], iy1 - 5, xs[lo] + 2, iy1 - 1, 0.5)
        xc = float(X(t))
        pc = B.lin(age, 0.92, 1.02)                         # the cursor drops once the head has passed it
        if pc > 0.0:
            f.segments("r", [xc], [iy0 - 4], [xc], [iy0 - 4 + (iy1 - iy0 + 8) * pc], 1.2, width=L.LW)
            s = f"N {int(RATE_YOU * t):06d}"
            f.tag("r", xc + 8, yb + 34, B.roll(s, age, 0.35, 1.12, key=6), size=L.T_SMALL, pad=4, bold=True, ref=s,
                  wipe=B.lin(age, 1.0, 1.12))
        f.text("w", ix1, y0 + 26, B.decode("RED = ABOVE 9 GEV", age, 80.0, 1.0, key=8, pad=True), size=L.T_MICRO,
               alpha=0.7, anchor="rs")
        header_gap(f, ctx, t)

    # bottom panels of the origin ---------------------------------------------------
    # They take the AGE of their panel (seconds since it started to build) instead of an alpha: nothing here
    # fades in. Header first (hud.panel_header), then each panel constructs its own content.
    def _p_arrivals(self, f, t, x0, x1, y0, y1, age):
        hud.panel_header(f, x0, x1, y0, "ARRIVALS / 100 MS", age=age)
        n = int(np.clip((x1 - x0) / 7.0, 30, 90))
        edges = t - n * 0.1 + np.arange(n + 1) * 0.1
        cnt = np.searchsorted(self.arr_t, edges[1:]) - np.searchsorted(self.arr_t, edges[:-1])
        bw = (x1 - x0) / n
        xs = x0 + np.arange(n) * bw
        hh = np.clip(cnt / 14.0, 0, 1) * (y1 - y0 - 34)
        hh = np.minimum(hh * B.spring(B.cascade(age, n, 0.3, 0.42, 0.3)), y1 - y0 - 30)   # the bars rise in a wave
        ok = (edges[:-1] >= 0) & (hh > 0.5)
        f.rects("w", xs[ok], y1 - 6 - hh[ok], xs[ok] + bw - 2, y1 - 6, 0.9)
        f.text("w", x1, y0 + 30, B.resolve("MEAN 6.3", age, 60.0, 0.75, key=1, pad=True), size=L.T_MICRO, alpha=0.75,
               anchor="rs")

    def _p_felt(self, f, t, x0, x1, y0, y1, age):
        b = float(smoothstep(T_FEEL, T_FEEL + 0.5, t))
        hud.panel_header(f, x0, x1, y0, title_fit(["FELT // WHAT YOUR NERVES REPORT", "FELT"], x1 - x0), age=age)
        ym = (y0 + y1) / 2 + 12
        B.pen(f, "w", x0, ym, x1, ym, B.ease(B.lin(age, 0.3, 0.62)), 0.9, width=L.LW)       # the flat trace
        if age > 0.4:
            hud.ruler(f, x0, x1, y1 - 4, 0, 6, 0.1, 1.0, down=False, inten=0.6, reveal=B.lin(age, 0.4, 0.85))
        f.text("r" if b > 0 else "w", x1, y0 + 36, B.roll("0.000", age, 0.45, 0.55, key=2), size=28, anchor="rs")
        if b > 0:
            f.tag("r", x0 + 4, ym - 18, "NOTHING"[: int((t - T_FEEL) * 20) + 1], size=L.T_LABEL, pad=4, alpha=b, bold=True)
            s = title_fit(["ENERGY LEFT IN YOU  ~2 MEV/CM  //  6E-10 W", "~2 MEV/CM  //  6E-10 W", "6E-10 W"], x1 - x0 - 8)
            f.text("w", x0 + 4, ym + 26, hud.typed(s, t - T_FEEL, 60, 0.4), size=L.T_MICRO, alpha=0.8)

    def _p_scale(self, f, t, x0, x1, y0, y1, age):
        m = t - T_MINI
        hud.panel_header(f, x0, x1, y0, "SCALE // METRES", layer="r" if m > 0 else "w", age=age)
        ya = y1 - 46
        B.pen(f, "w", x0, ya, x1, ya, B.ease(B.lin(age, 0.3, 0.55)), 0.8, width=1.0)
        dec = np.arange(0, 19)
        xs = x0 + 8 + dec / 18.0 * (x1 - x0 - 16)
        g = B.spring(B.cascade(age, 19, 0.42, 0.4, 0.25))             # the decades, one after the other
        on = g > 0.0
        f.segments("w", xs[on], np.full(on.sum(), ya), xs[on], ya + (np.where(dec % 3 == 0, 14.0, 7.0) * g)[on], 0.9)
        for d, name in ((0, "M"), (3, "MM"), (6, "UM"), (9, "NM"), (12, "PM"), (15, "FM"), (18, "AM")):
            if on[d]:
                f.text("w", float(xs[d]), ya + 38, name, size=L.T_MICRO, alpha=0.75, anchor="ms")
        pos = 18.0 * float(smoothstep(0.0, 2.2, m)) if m > 0 else 0.0
        xc = x0 + 8 + pos / 18.0 * (x1 - x0 - 16)
        f.rects("r", x0 + 8, ya - 10, xc, ya - 4, 1.0 if m > 0 else 0.0)
        pc = B.lin(age, 0.82, 0.92)                                   # the cursor drops, then its tag is made
        if pc > 0.0:
            f.segments("r", [xc], [ya - 26], [xc], [ya - 26 + 42 * pc], 1.2, width=L.LW)
        lab = "YOU 1.7E0" if m <= 0 else (f"1E-{int(pos):02d}" if pos < 17.9 else "MUON < 1E-18")
        w = text_w(lab, L.T_SMALL) + 12
        B.tag(f, "r" if m > 0 else "w", xc + 8 if xc + 8 + w < x1 else xc - 8 - w, ya - 22, lab, age, t0=0.9,
              size=L.T_SMALL, pad=4, bold=True, cps=70.0, key=3)

    def _p_energy(self, f, t, x0, x1, y0, y1, age):
        hud.panel_header(f, x0, x1, y0, title_fit(["ENERGY OF THE LAST 10 S // GEV", "ENERGY // GEV"], x1 - x0), age=age)
        i0, i1 = np.searchsorted(self.arr_t, [max(t - 10.0, 0.0), t])
        nb = int(np.clip((x1 - x0) / 13.0, 16, 40))
        edges = np.geomspace(0.3, 100.0, nb + 1)
        cnt, _ = np.histogram(self.arr_e[i0:i1], edges)
        yb = y1 - 24
        hh = np.sqrt(cnt / max(cnt.max(), 1)) * (yb - y0 - 22)
        k0 = int(np.searchsorted(edges, 3.3)) - 1                     # the bars rise from the mode outwards
        rank = np.abs(np.arange(nb) - k0) / max(k0, nb - 1 - k0, 1)
        hh = np.minimum(hh * B.spring(B.cascade(age, rank, 0.3, 0.4, 0.3)), yb - y0 - 14)
        bw = (x1 - x0) / nb
        xs = x0 + np.arange(nb) * bw
        hot = edges[:-1] >= 9.0
        up = hh > 0.5
        f.rects("w", xs[~hot & up], yb - hh[~hot & up], xs[~hot & up] + bw - 2, yb, 0.9)
        f.rects("r", xs[hot & up], yb - hh[hot & up], xs[hot & up] + bw - 2, yb, 0.95)
        B.pen(f, "w", x0, yb + 1, x1, yb + 1, B.ease(B.lin(age, 0.25, 0.5)), 0.6, width=1.0)
        for j, v in enumerate((1, 10, 100)):
            if age < 0.5 + 0.09 * j:
                continue
            xv = x0 + math.log(v / 0.3) / math.log(100.0 / 0.3) * (x1 - x0)
            f.segments("w", [xv], [yb], [xv], [yb + 7 + 9 * math.exp(-(age - 0.5 - 0.09 * j) / 0.08)], 0.8)
            f.text("w", xv - 4 if v == 100 else xv + 4, yb + 20, f"{v}", size=L.T_MICRO, alpha=0.65,
                   anchor="rs" if v == 100 else "ls")

    # --- 00:31 - 00:40 ---------------------------------------------------------
    def _star_disc(self, f, t, R, gain=1.0, hole=0.0):
        if R < 3:
            return
        C = self.C
        P = self.gran
        flick = 0.75 + 0.25 * np.sin(self.gran_ph + t * 2.3)
        out = np.hypot(P[:, 0], P[:, 1]) * R > hole + 5.0
        f.dots("w", C[0] + P[out, 0] * R * 0.985, C[1] + P[out, 1] * R * 0.985,
               self.gran_r[out] * max(R / 212.0, 0.35), gain * self.gran_b[out] * flick[out])
        if hole > 4:
            f.rings("w", [C[0]], [C[1]], [hole + 3.0], 0.9 * gain, width=1.6)
        f.rings("w", [C[0], C[0]], [C[1], C[1]], [R, R - 3.0], [1.3 * gain, 0.9 * gain], width=3.0)

    def _star(self, f, t, ctx):
        lay, C = self.lay, self.C
        a = t - T_STAR
        grow = float(smoothstep(0.0, 2.3, a))
        implode = float(smoothstep(T_IMPLODE, T_X, t)) ** 2.2
        R = lay.r_star * (1.0 - math.exp(-a / 0.5)) * (1.0 - 0.9 * implode)
        self._lattice(f, t, gain=0.75)
        f.set_clip(*FRAME_CLIP)
        if grow < 1.0:                  # the field of the opening falls back into the star (time runs backwards)
            self.field.draw(f, t, C[0], C[1], gain=1.1 * (1.0 - grow), squeeze=1.0 - 0.88 * grow ** 0.7,
                            omega=0.11 + 2.2 * grow, twist=-5.0 * grow)
        suck = 1.0 - 0.42 * implode
        self.rip.draw(f, t, C[0], C[1], gain=1.0, kick=self._pulse(t, ctx), scale=suck, r_min=R + 10)
        f.set_clip()
        core = (24.0 * (1.0 - 0.55 * implode) + 5.0 * self._pulse(t, ctx)) * min(1.0, R / 60.0)
        self._star_disc(f, t, R, gain=1.0 + 0.8 * implode, hole=core + 8.0)
        f.dots("r", [C[0]], [C[1]], core, 1.4)
        self._strip_star(f, t, ctx)
        left = max(T_X - t, 0.0)
        card(f, lay, "PROGENITOR", [("CLASS     RED SUPERGIANT", "RED SUPERGIANT"), ("MASS      25 M_SUN", "M  25 M_SUN"),
                                    ("RADIUS    1 000 R_SUN", "R  1 000 R_SUN"), ("AGE       7.1E6 YR", "AGE 7.1E6 YR"),
                                    ("CORE      FE  1.4 M_SUN", "CORE FE 1.4 M_SUN"), ("T_CORE    5.0E9 K", "T_CORE 5.0E9 K"),
                                    ("RHO_CORE  1E10 G/CM3", None),
                                    (f"COLLAPSE  T-{left:06.3f} S", f"T-{left:06.3f} S")], a, red_rows=(7,))
        self._star_notes(f, t, R, core, a, implode)
        al = float(smoothstep(0.3, 1.0, a))
        self._panels(f, [lambda f, x0, x1, y0, y1, a_: self._p_core_mass(f, t, x0, x1, y0, y1, a_, a, implode),
                         lambda f, x0, x1, y0, y1, a_: self._p_shells(f, t, x0, x1, y0, y1, a_, a, implode),
                         lambda f, x0, x1, y0, y1, a_: self._p_pressure(f, t, ctx, x0, x1, y0, y1, a_),
                         lambda f, x0, x1, y0, y1, a_: self._p_core_radius(f, t, x0, x1, y0, y1, a_, implode)], al)
        cell_right(f, lay, "YEAR // BEFORE NOW", hud.typed(f"-{YEAR0:,.0f}".replace(",", " "), a, 40, 0.1),
                   sub="LOOKING BACK", red=True, value_short=f"-{YEAR0:.1E}".replace("E+0", "E"))

    def _star_notes(self, f, t, R, core, a, implode):
        """Three read-outs in the notes column with leaders to the star; the core sits on the axis."""
        lay, C = self.lay, self.C
        if a < 1.2 or implode > 0.6 or _note_col(lay)[0] is None:
            return
        al = 1.0 - implode / 0.6
        side = lay.note_side
        col, _ = _note_col(lay)
        xe = col[0] if side > 0 else col[1]
        wpl = min(col[1] - col[0] - 30.0, 400.0)
        xp = xe + 30.0 if side > 0 else xe - 30.0 - wpl
        f.occlude(xp, C[1] - 290.0, xp + wpl, C[1] - 9.0)
        if a > 2.6:
            # the axis from the core: a slit is cut through the disc so the red leader shows
            x_edge = C[0] + side * (R + 4.0)
            f.occlude(min(C[0] + side * (core + 12.0), x_edge), C[1] - 3.5, max(C[0] + side * (core + 12.0), x_edge), C[1] + 3.5)
            f.segments("r", [C[0] + side * (core + 4.0)], [C[1]], [xe], [C[1]], 0.95 * al, width=L.LW)
            axis_note(f, lay, "IRON CORE", below=["1.4 M_SUN  //  R 1 500 KM", "T 5.0E9 K", "NOTHING LEFT TO BURN"],
                      age=a - 2.6, alpha=al)
        ang = math.radians(-50.0)
        note(f, lay, C[1] - 252.0, "PHOTOSPHERE", ["R 1 000 R_SUN  //  T 3 600 K"],
             anchor=(C[0] + side * R * math.cos(ang), C[1] + R * math.sin(ang)), age=a - 1.2, alpha=al)
        if a > 1.8:
            ang = math.radians(-22.0)
            note(f, lay, C[1] - 142.0, "HYDROGEN ENVELOPE", ["16 M_SUN  //  THE REST HAS BURNED"],
                 anchor=(C[0] + side * (R + 40.0) * math.cos(ang), C[1] + (R + 40.0) * math.sin(ang)), age=a - 1.8,
                 alpha=al)

    def _strip_star(self, f, t, ctx):
        x0, y0, x1, y1 = L.STRIP
        f.occlude(x0, y0, x1, y1)
        marks = [(T_STAR, "STAR"), (T_IMPLODE, "COLLAPSE"), (T_X, "BOUNCE"), (T_BLOOMED, "BLOOM")]
        hud.show_strip(f, t, ctx, "STAR // THE LAST SECONDS", 30.0, 48.0, marks)
        header_gap(f, ctx, t)

    # bottom panels of the star -------------------------------------------------------
    def _p_shells(self, f, t, x0, x1, y0, y1, al, a, implode):
        hud.panel_header(f, x0, x1, y0, title_fit(["BURNED // SHELL BY SHELL", "BURNED"], x1 - x0), alpha=al)
        shells = [("H", "7E6 Y"), ("HE", "7E5 Y"), ("C", "600 Y"), ("NE", "1 Y"), ("O", "6 MO"), ("SI", "1 D"),
                  ("FE", "INERT")]
        cw = (x1 - x0) / len(shells)
        for k, (el, dur) in enumerate(shells):
            xx = x0 + k * cw
            if a <= 0.5 + 0.35 * k:
                continue
            last = k == len(shells) - 1
            f.rect("w", xx + 3, y0 + 20, xx + cw - 6, y0 + 62, 0.7 * al)
            if not last:
                f.rects("w", xx + 7, y0 + 24, xx + cw - 10, y0 + 58, 0.85 * al)
            else:
                f.rects("r", xx + 7, y0 + 24, xx + cw - 10, y0 + 58, 0.6 + 0.4 * implode)
            f.text("r" if last else "w", xx + 5, y0 + 88, el, size=L.T_LABEL, alpha=0.95 * al)
            if text_w(dur, L.T_MICRO) <= cw - 6:
                f.text("r" if last else "w", xx + 5, y0 + 112, dur, size=L.T_MICRO, alpha=0.7 * al)

    def _p_core_mass(self, f, t, x0, x1, y0, y1, al, a, implode):
        hud.panel_header(f, x0, x1, y0, title_fit(["CORE MASS // CHANDRASEKHAR LIMIT 1.44", "CORE MASS // LIMIT 1.44",
                                                    "CORE MASS"], x1 - x0), alpha=al, layer="r" if implode > 0 else "w")
        m_core = 1.10 + 0.34 * float(smoothstep(0.0, T_IMPLODE - T_STAR, a))
        xa, xb = x0 + 8, x1 - 8
        ya = y0 + 70
        hud.ruler(f, xa, xb, ya, 1.0, 1.5, 0.01, 0.1, inten=0.8 * al)
        for v in (1.0, 1.1, 1.2, 1.3, 1.4):
            f.text("w", xa + (v - 1.0) / 0.5 * (xb - xa) + 4, ya + 30, f"{v:.1f}", size=L.T_MICRO, alpha=0.75 * al)
        xm = xa + (m_core - 1.0) / 0.5 * (xb - xa)
        f.rects("w", xa, ya - 16, xm, ya - 6, 0.95 * al)
        xl = xa + 0.44 / 0.5 * (xb - xa)
        f.segments("r", [xl], [ya - 34], [xl], [ya + 18], 1.2 * al, width=L.LW)
        f.text("r" if m_core >= 1.438 else "w", x1, y0 + 36, f"{m_core:.3f} M_SUN", size=L.T_LABEL, alpha=al, anchor="rs")

    def _p_pressure(self, f, t, ctx, x0, x1, y0, y1, al):
        hud.panel_header(f, x0, x1, y0, title_fit(["PRESSURE // GRAVITY"], x1 - x0), alpha=al)
        n = 90
        lv = ctx.cues.loud_curve(t - 6.0, t, n)
        xs = np.linspace(x0 + 4, x1 - 4, n)
        f.polyline("w", xs, y1 - 8 - lv * (y1 - y0 - 40), 0.9 * al, width=L.LW)
        g = 0.45 + 0.4 * np.clip((t - 6.0 + np.arange(n) / n * 6.0 - T_STAR) / (T_X - T_STAR), 0, 1) ** 2
        f.polyline("r", xs, y1 - 8 - g * (y1 - y0 - 40), 0.95 * al, width=L.LW)

    def _p_core_radius(self, f, t, x0, x1, y0, y1, al, implode):
        hud.panel_header(f, x0, x1, y0, "CORE RADIUS // KM", alpha=al, layer="r" if implode > 0 else "w")
        rk = 1500.0 * (30.0 / 1500.0) ** implode                  # 1 500 km -> 30 km in the last instant
        f.text("r" if implode > 0 else "w", x0 + 2, y0 + 76, f"{rk:,.0f}".replace(",", " "), size=46, alpha=al)
        xa, xb, ya = x0 + 6, x1 - 6, y1 - 24
        frac = math.log10(rk / 10.0) / math.log10(300.0)
        f.rects("r" if implode > 0 else "w", xa, ya - 12, xa + (xb - xa) * frac, ya - 5, 0.95 * al)
        f.segments("w", [xa], [ya], [xb], [ya], 0.7 * al)
        for v, lab in ((10.0, "10"), (100.0, "100"), (1000.0, "1 000")):
            xv = xa + math.log10(v / 10.0) / math.log10(300.0) * (xb - xa)
            f.segments("w", [xv], [ya], [xv], [ya + 7], 0.8 * al)
            f.text("w", xv + 4, ya + 20, lab, size=L.T_MICRO, alpha=0.65 * al)
        s = title_fit(["NEUTRON STAR AT 30", "-> 30"], x1 - x0 - 150)
        f.text("w", x1, y0 + 36, s, size=L.T_MICRO, alpha=0.75 * al, anchor="rs")

    # --- 00:40 - 00:44 ---------------------------------------------------------
    def _nova(self, f, t, ctx):
        lay, nova = self.lay, self.nova
        cam = nova.camera(t)
        a = t - T_X
        f.set_clip(*FRAME_CLIP)
        nova.draw_plane(f, cam, self.rip, t, gain=1.0)
        cx, cy, cz, _ = cam.project(np.zeros((1, 3), np.float32))
        X, Y = float(cx[0]), float(cy[0])
        flash = math.exp(-a / 0.5)
        beat = ctx.cues.onset(t, 0.14) * 1.6 + ctx.cues.kick(t, 0.14)
        f.dots("w", [X], [Y], 36.0 + 60.0 * flash + 14.0 * beat, 1.6)
        f.dots("w", [X], [Y], 120.0 + 260.0 * flash, 0.10 + 0.5 * flash)
        nova.draw_rays(f, cam, t, gain=1.0)
        tip = nova.draw_hero(f, cam, t)
        f.dots("r", [X], [Y], 9.0, 1.4)
        f.set_clip()
        self._strip_star(f, t, ctx)
        nu = 1e58 * (1.0 - math.exp(-a / 1.1))
        card(f, lay, "SUPERNOVA", [("TYPE      II // CORE COLLAPSE", "CORE COLLAPSE"), ("E_TOTAL   3E46 J", "E  3E46 J"),
                                   ("NEUTRINOS 99 %", "NU  99 %"), (f"N_NU      {nu:.2E}".replace("E+", "E"),
                                                                    f"N_NU {nu:.1E}".replace("E+", "E")),
                                   ("SHOCK     10 000 KM/S", "10 000 KM/S"), ("REMNANT   NEUTRON STAR", "NEUTRON STAR"),
                                   ("EJECTA    P+ 89  HE 10  Z>2 1 %", None), ("FERMI ACCELERATION  ->  1E15 EV", "-> 1E15 EV")],
             a, red_title=True, red_rows=(3,), cps=80.0)
        hero_note(f, lay, tip, a - 1.2)
        f.set_clip(*FRAME_CLIP)
        self._ray_tags(f, cam, t)
        f.set_clip()
        self._panels(f, [lambda f, x0, x1, y0, y1, a_: self._p_neutrino(f, t, x0, x1, y0, y1),
                         lambda f, x0, x1, y0, y1, a_: self._p_messengers(f, t, x0, x1, y0, y1, a),
                         lambda f, x0, x1, y0, y1, a_: self._p_light(f, t, x0, x1, y0, y1),
                         lambda f, x0, x1, y0, y1, a_: self._p_ejecta(f, t, x0, x1, y0, y1, a)])
        cell_right(f, lay, "YEAR // BEFORE NOW", f"-{YEAR0:,.0f}".replace(",", " "), sub="T+" + f"{a:05.2f} S", red=True,
                   value_short=f"-{YEAR0:.1E}".replace("E+0", "E"))
        return {"exposure": 1.0 + 0.5 * flash}

    def _ray_tags(self, f, cam, t):
        """Names on a few rays, only where the wall is free (no tower, no card, no note)."""
        nova, lay = self.nova, self.lay
        Lr = nova.lengths(t)
        A = np.zeros((nova.n, 3))
        ax, ay, az, bx, by, bz, ok, tip_in = nova._clip_near(cam, A, nova.dir * Lr[:, None])
        fade = np.clip((nova.fade_t - t) / 0.7, 0.0, 1.0)
        ox, oy, _, _ = cam.project(np.zeros((1, 3), np.float32))
        shown = 0
        for k in np.nonzero((Lr > 1.0) & ok & tip_in & (fade > 0))[0]:
            if k % 4:
                continue
            x, y = float(bx[k]), float(by[k])
            if math.hypot(x - float(ox[0]), y - float(oy[0])) < 300.0:
                continue
            s = f"{nova.kind[k]} {0.3 * 10 ** (2 + 4 * hash01(k, 7)):.1E} GEV".replace("E+0", "E")
            rect = (x + 8.0, y - 34.0, x + 22.0 + text_w(s, L.T_SMALL), y - 4.0)
            txt = hud.typed(s, t - T_X - nova.delay[k], 60, 0.4)
            if not txt or not lay.free(*rect):
                continue
            lay.take(rect)
            f.occlude(*rect)
            f.text("w", x + 14.0, y - 12.0, txt, size=L.T_SMALL, alpha=0.9 * float(fade[k]))
            shown += 1
            if shown >= 5:
                break

    # bottom panels of the supernova ----------------------------------------------------
    def _p_messengers(self, f, t, x0, x1, y0, y1, a):
        nova = self.nova
        n_out = int((nova.lengths(t) > 1).sum()) + 1
        hud.panel_header(f, x0, x1, y0, title_fit([f"MESSENGERS OUT // {n_out:03d} DRAWN OF 1E53", f"MESSENGERS OUT // {n_out:03d}",
                                                    "MESSENGERS"], x1 - x0))
        Lr = nova.lengths(t)
        order = np.argsort(-nova.speed)
        bw = (x1 - x0) / nova.n
        hh = np.clip(Lr[order] / 3200.0, 0, 1) * (y1 - y0 - 30)
        xs = x0 + np.arange(nova.n) * bw
        f.rects("w", xs, y1 - 6 - hh, xs + max(bw - 2, 1.5), y1 - 6, 0.9)
        hx = x0 + (nova.n // 3) * bw
        f.rects("r", hx, y1 - 6 - (y1 - y0 - 30) * min(1.0, a / 4.0), hx + max(bw - 1, 2), y1 - 6, 1.0)

    def _p_neutrino(self, f, t, x0, x1, y0, y1):
        hud.panel_header(f, x0, x1, y0, title_fit(["NEUTRINO BURST // 10 S", "NEUTRINOS"], x1 - x0), layer="r")
        cols = int(np.clip((x1 - x0) / 5.2, 50, 150))
        dt = 4.0 / cols
        kk = math.floor((t - 4.0) / dt) + np.arange(cols)
        ts = kk * dt
        dens = np.where(ts >= T_X, 0.08 + 0.9 * np.exp(-(ts - T_X) / 1.3), 0.02)
        hud.barcode_lanes(f, x0, x1, y0 + 12, y1, dens, kk, lanes=3, seed=4)

    def _p_light(self, f, t, x0, x1, y0, y1):
        hud.panel_header(f, x0, x1, y0, title_fit(["LIGHT CURVE // LOG L", "LIGHT CURVE"], x1 - x0))
        n = 120
        tt = T_X - 1.0 + np.arange(n) / n * 6.0
        lum = np.where(tt < T_X, 0.12, 0.12 + 0.85 * (1 - np.exp(-(tt - T_X) / 0.25)) * np.exp(-(tt - T_X) / 9.0))
        xs = np.linspace(x0 + 4, x1 - 4, n)
        seen = tt <= t
        if seen.sum() > 1:
            f.polyline("w", xs[seen], y1 - 8 - lum[seen] * (y1 - y0 - 40), 0.95, width=L.LW)
            f.dots("r", [xs[seen][-1]], [y1 - 8 - lum[seen][-1] * (y1 - y0 - 40)], 4.0, 1.5)
        f.text("w", x1, y0 + 30, "1E9 L_SUN", size=L.T_MICRO, alpha=0.75, anchor="rs")

    def _p_ejecta(self, f, t, x0, x1, y0, y1, a):
        hud.panel_header(f, x0, x1, y0, title_fit(["COSMIC RAYS // WHAT THEY ARE", "COSMIC RAYS"], x1 - x0))
        rows = [("P+", 0.89, "89 %  PROTONS"), ("HE", 0.10, "10 %  HELIUM"), ("Z>2", 0.01, " 1 %  HEAVIER")]
        wbar = x1 - x0 - 70 - 180
        for k, (name, frac, lab) in enumerate(rows):
            y = y0 + 34 + k * 32
            g = min(1.0, max(0.0, (a - 0.3 * k) / 0.5))
            f.text("r" if k == 0 else "w", x0 + 4, y + 10, name, size=L.T_SMALL, alpha=0.95)
            f.rects("r" if k == 0 else "w", x0 + 60, y - 6, x0 + 60 + max(3.0, wbar * frac * g), y + 10, 0.95)
            f.text("w", x1, y + 10, lab, size=L.T_SMALL, alpha=0.8, anchor="rs")
