"""ORIGIN + STAR - the opening.   Sheet 1.0 + 1.1, 00:00 - 00:44.

The TouchDesigner intro (lattice of crosses, red dot, expanding rings, the rotating spiral of streaks,
the star, the rays of its collapse) kept as it is and given its data:

  00:00  BLACK       nothing at all: no counter, no ticks, no lattice
  00:01  BORDER      the white frame fades in
  00:03  LOADING     top left: MUON BLOOM // LOADING, a bar that creeps and jumps on the low thumps
  00:06  LOADED      the lattice of crosses draws itself row by row, then the red dot and its crosshair:
                     at 00:07 everything is there, the dot pulses with the low thumps
  00:07:47  HUD      on the sound: the arrivals barcode starts in the score strip - one tick per muon through
                     one spectator, 70 per second (no card, no bottom panels, no counters here: the
                     counter of the show is built with the star)
  00:11  HERE / NOW  "something is passing through you": rings leave the dot (light-time shells around the
                     muon), HERE / RIGHT NOW is written on its axis: white, in one border
  00:19  ALWAYS      the spiral field grows out of the dot and turns: particles leave the dot one after the
                     other, each one drawing the line of its own path behind it (the trails of the
                     TouchDesigner galaxy, scene3.0.mov); no line is switched on
  00:28  MINISCULE   the dot collapses to a point: MU // POINT-LIKE
  00:31  STAR        "billions of years ago ... a star collapsed, immense": the field falls straight into the
                     point (no spin), which is the core of a star - a bright body made of detail (the burning
                     shells as rings of radial bars, a boiling limb, a crown of rays); progenitor data, a
                     countdown. The star GROWS WITHOUT A STEP: it swells out of the point in T_GROW (a slow
                     start, a soft landing), its shells come out of the core by getting longer, and it follows
                     the music through a smooth envelope (swell: it rises before a hit and relaxes after it) -
                     driven by the raw kicks it jumped on every hit, and grew in stairs when four came in a row
  00:40  COLLAPSE    the star implodes, bounces - a glare of rays, not a white disc - and the rays burst out
                     (3D, the ring plane tilts): the messengers. The bounce throws the glare out of the imploded
                     core in T_BURST (a few frames), it is not switched on. One ray is tagged: the one that will
                     reach us (-> messenger.py); a few others are named where and when the wall is free
                     (_plan_ray_tags: decided once, so a name never comes on written nor flickers).

THE TOWERS stand in front of the wall for the whole show and nobody knows yet where: nothing here has a
fixed x. `Lay` reads the wall from ctx: the one centre of the image (ctx.focus, the middle of the best bay),
a column for the scene card, a column for the notes that point at the centre, the free panels of the
bottom band. Heroes (dot, star, bounce, the tagged ray) and every piece of text stay clear of the towers;
rings, rays, the streak field and leader lines simply pass behind them.

NO FADE for anything that shows data: every strip, card, note, panel, counter and label is CONSTRUCTED when
it appears and taken apart when it leaves. In the first 16 seconds each element has its own choreography
(the helpers of muonbloom/build.py: registration marks, lines drawn by a pen, marks arriving in a wave and
overshooting, text decoded out of noise, figures spinning before they lock, a two-frame inverted flash when
the element is complete); from there on the general mechanism does it (`with f.build(age, rect):`, and
B.io for what leaves). Check a build frame by frame with tools/filmstrip.py.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import hud
from .. import layout as L
from .. import showdata as sd
from .. import engine as E
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
T_HUD = 7.0 + 47.0 / 60.0                        # 00:07:47, on the sound: the HUD builds up (the arrivals strip)
LOAD_STEPS = ["CANVAS 2978 X 1400", f"AUDIO {sd.mmss(sd.SHOW_END)}", "DETECTORS L C R // OSC", "MUON FLUX 1 /CM2/MIN"]
LOAD_N = 50                                      # blocks in the loading bar
T_FIELD = 17.0                                   # the field starts to leave the dot (first lines seen 1.4 s later,
#                                                  all in place after 9.3 s)
T_FALL = 3.3                                     # seconds the field takes to fall into the star, from T_STAR
T_GROW = 1.9                                     # seconds the star takes to swell out of the point (ease in, ease out)
T_IMPLODE = 39.25                                # the disc gives way
T_X = 40.0                                       # bounce: the rays leave
T_BURST = 0.13                                   # seconds the bounce takes to throw the glare out of the core
T_END = 44.0
SUCK = 0.58                                      # the rings around the star at the end of the implosion (x their radius)
RAY_TAG_MIN = 0.7                                # a ray of the supernova is named if its name can stay that long (s)
RAY_TAG_MAX = 5                                  # names on the rays at a time

RATE_YOU = sd.RATE_YOU                           # muons / s through one spectator
RATE_WALL = 60800.0                              # 28 m x 13 m at 1 / cm2 / min
YEAR0 = 4.8e9                                    # years ago ("perhaps before the Earth existed")
CARD_Y = 300.0
FRAME_CLIP = (L.FX0, L.FY0, L.FX1, L.FY1)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def swell(times, amps, t, rise=0.14, tau=0.2):
    """Smooth envelope of the sound events at `times` (kicks, onsets) with the levels `amps`: every event is a
    bump that starts to rise `rise` seconds BEFORE it (an S curve), is at its top on the event, and relaxes
    after it, leaving flat (time constant tau). No jump and no kink anywhere: a body driven by it swells and
    breathes with the music, where the raw pulses of Cues.kick / Cues.onset (instant attack) make it jump
    on every hit - on the star, that read as a growth in steps."""
    if not len(times):
        return 0.0
    i0, i1 = np.searchsorted(times, [t - 12.0 * tau, t + rise])
    if i1 <= i0:
        return 0.0
    x = t - np.asarray(times[i0:i1], np.float64)
    up = np.clip((x + rise) / rise, 0.0, 1.0)
    xp = np.maximum(x, 0.0) / tau
    k = np.where(x < 0.0, up * up * (3.0 - 2.0 * up), (1.0 + xp) * np.exp(-xp))
    return float((np.asarray(amps[i0:i1], np.float64) * k).sum())


_RING_N = 144                                    # sides of a ring drawn by ring()
_RING_C, _RING_S = np.cos(np.linspace(0.0, 2.0 * np.pi, _RING_N + 1)), np.sin(np.linspace(0.0, 2.0 * np.pi, _RING_N + 1))


def ring(f, layer, cx, cy, radii, inten, width=1.0):
    """Circles for rings that grow or breathe: their vertices stay where they are on the circle whatever the
    radius (Frame.rings takes a number of sides that follows the radius, so its corners move as it grows)."""
    r = np.atleast_1d(np.asarray(radii, np.float64))
    i = np.broadcast_to(np.asarray(inten, np.float64), r.shape)
    m = (r > 0.5) & (i > 1e-4)
    if not m.any():
        return
    r, i = r[m][:, None], i[m]
    x, y = cx + r * _RING_C[None, :], cy + r * _RING_S[None, :]
    f.segments(layer, x[:, :-1].ravel(), y[:, :-1].ravel(), x[:, 1:].ravel(), y[:, 1:].ravel(), np.repeat(i, _RING_N),
               width=width)


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
            f.segments("w", [x - 8.0], [ya], [x - 8.0], [yz], E.wl(0.8), width=E.ww(1.3))
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
                f.segments("w", [x - 8.0], [yr - 6.0], [x - 2.0], [yr - 6.0], E.wl(0.8), width=E.ww(1.3))
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


def box_note(f, lay, lines, age):
    """The minimal read-out of the centre: the four corners of ONE white border on the horizontal axis, at the near edge of the
    notes column, and white lines of text in it - no red tag, no data lines, no bold piece of axis.
    lines = [(text, age)] or [(text, age, size[, layer])] (layer "r" = red): each line is decoded out of noise from its own age; the
    first ones are the words of the voice (size 30), smaller ones are the figures behind them. A line too
    wide for the column is left out. The border is drawn by two
    pens (Frame.build) while its plate opens from the axis; give age=B.io(...) to have it taken apart."""
    col, side = _note_col(lay)
    if col is None or age < 0.0:
        return
    padx, pady, gap = 22.0, 18.0, 24.0
    rows = [(s, a, (z[0] if z else 30), (z[1] if len(z) > 1 else "w")) for s, a, *z in lines]
    rows = [r for r in rows if text_w(r[0], r[2]) + 2.0 * padx + 20.0 <= col[1] - col[0]]
    if not rows:
        return
    cy = lay.C[1]
    w = max(text_w(r[0], r[2]) for r in rows) + 2.0 * padx
    h = sum(0.7 * r[2] for r in rows) + (len(rows) - 1) * gap + 2.0 * pady
    x0 = col[0] + 10.0 if side > 0 else col[1] - 10.0 - w
    rect = (x0, cy - h / 2.0, x0 + w, cy + h / 2.0)
    lay.take(rect)                               # (no plate: the rings show through between its corners)
    f.noglow_rects.append((rect[0] - hud.NOGLOW_PAD, rect[1] - hud.NOGLOW_PAD, rect[2] + hud.NOGLOW_PAD,
                           rect[3] + hud.NOGLOW_PAD))          # no glow, like the subtitle
    B.brackets(f, rect, size=16.0 * float(B.ease(B.lin(age, 0.0, 0.25))), inten=0.95)     # only its four corners
    y = rect[1] + pady
    for k, (s, mine, z, layer) in enumerate(rows):
        y += 0.7 * z
        f.text(layer, x0 + padx, y, B.decode(s, min(age, mine) - 0.12, 45.0 * 30.0 / z, key=k, pad=True), size=z, bold=True)
        y += gap


def axis_note(f, lay, title, below=(), above=(), title2=None, age=9.0, age2=None, alpha=1.0, size=30, red=True,
              build=False, commit=True):
    """Read-out of the centre, set on the horizontal axis that runs through it (the axis is its leader and
    passes behind whatever tower stands in between): tag just above the axis at the near edge of the notes
    column, data lines above / below it, an optional second tag under the axis.
    build=True: constructed instead of typed - a pen draws the bold piece of axis, the plate opens from the
    axis, each tag is made and flashes (commit), the lines are decoded out of noise and their figures spin,
    then lock. Give age=B.io(...) to have it taken apart when it leaves (commit=False then); a line given as
    (text, age) is written from its own age (a line that comes later)."""
    col, side = _note_col(lay)
    if col is None or alpha <= 0.01 or (build and age < 0.0):
        return
    cy = lay.C[1]
    xe = col[0] if side > 0 else col[1]
    tx = xe + side * 30.0
    anc = "ls" if side > 0 else "rs"
    width = col[1] - col[0] - 34.0
    own = lambda item: item if isinstance(item, tuple) else (item, None)
    la = [own(s) for s in above if text_w(own(s)[0], L.T_SMALL) <= width]
    lb = [own(s) for s in below if text_w(own(s)[0], L.T_SMALL) <= width]
    if text_w(title, size) + 14.0 > width:
        size = 22
    wmax = max([text_w(title, size) + 14.0] + ([text_w(title2, size) + 14.0] if title2 else [])
               + [text_w(s, L.T_SMALL) for s, _ in la + lb])
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
              cps=40.0, key=1, commit=commit)
        for k, (s, mine) in enumerate(la):
            txt = (B.resolve(s, age, 70.0, 0.35 + 0.2 * k, key=k, pad=pad, spin=0.45) if mine is None else
                   B.resolve(s, mine, 70.0, key=k, pad=pad, spin=0.45))
            f.text("w", tx, cy - 22.0 - size - 14.0 - 25.0 * k, txt, size=L.T_SMALL,
                   alpha=(0.9 if k == 0 else 0.7) * alpha, anchor=anc)
        if low:
            yb = cy + 34.0
            if title2:
                B.tag(f, lay_r, tx, cy + 14.0 + size, title2, a2, t0=0.05, size=size, pad=7, alpha=alpha, bold=True,
                      anchor=anc, cps=40.0, key=2, commit=commit)
                yb = cy + 14.0 + size + 36.0
            for k, (s, mine) in enumerate(lb):
                txt = (B.resolve(s, a2, 70.0, 0.3 + 0.2 * k, key=4 + k, pad=pad, spin=0.45) if mine is None else
                       B.resolve(s, mine, 70.0, key=4 + k, pad=pad, spin=0.45))
                f.text("w", tx, yb + 25.0 * k, txt, size=L.T_SMALL, alpha=(0.9 if k == 0 else 0.7) * alpha, anchor=anc)
        B.pen(f, lay_r, xe + side * 2.0, cy, xe + side * (wmax + 38.0), cy, B.ease(B.lin(age, 0.0, 0.25)),
              0.95 * alpha, width=L.LW_BOLD, head=3.8)
        return
    f.occlude(*rect)
    lay.take(rect)
    f.tag(lay_r, tx, cy - 22.0, title[: int(age * 22) + 1], size=size, pad=7, alpha=alpha, bold=True, anchor=anc)
    for k, (s, _) in enumerate(la):
        f.text("w", tx, cy - 22.0 - size - 14.0 - 25.0 * k, hud.typed(s, age, 60, 0.3 + 0.2 * k), size=L.T_SMALL,
               alpha=(0.9 if k == 0 else 0.7) * alpha, anchor=anc)
    if low:
        yb = cy + 34.0
        if title2:
            f.tag(lay_r, tx, cy + 14.0 + size, title2[: int(a2 * 22) + 1], size=size, pad=7, alpha=alpha, bold=True,
                  anchor=anc)
            yb = cy + 14.0 + size + 36.0
        for k, (s, _) in enumerate(lb):
            f.text("w", tx, yb + 25.0 * k, hud.typed(s, a2, 60, 0.3 + 0.2 * k), size=L.T_SMALL,
                   alpha=(0.9 if k == 0 else 0.7) * alpha, anchor=anc)
    f.segments(lay_r, [xe + side * 2.0], [cy], [xe + side * (wmax + 38.0)], [cy], 0.95 * alpha, width=L.LW_BOLD)


def note(f, lay, y, title, lines=(), anchor=None, red=False, age=9.0, alpha=1.0, size=L.T_TAG, big=None,
         big_size=46, big_red=False, build=None, plate=1.0):
    """A read-out in the notes column at height y (baseline of its tag): tag, an optional big figure, typed
    lines. With an anchor, a leader runs back to that point of the image: 45 degrees to the level of the tag,
    then straight to the column (behind a tower if one is there).
    build = seconds since the note appeared (B.io for one that also leaves): it is then constructed from its
    anchor outwards - leader drawn by a pen, tag pushed out, figures and lines decoded - instead of typed.
    Negative = not there. A note never fades in. A line given as (text, age) is decoded from its own age
    (a line that is added later). plate = how far its dark plate is open (0..1, from its top): with `build`
    it opens while the note is made and closes when it is taken apart, it is not put there in one frame."""
    lines = [s if isinstance(s, str) else B.resolve(s[0], s[1], 70.0, key=k).ljust(len(s[0])) for k, s in enumerate(lines)]
    col, side = _note_col(lay)
    if col is None or alpha <= 0.01:
        return False
    if build is not None:
        if build < 0.0:
            return True
        xe = col[0] if side > 0 else col[1]
        x_far = xe + side * (col[1] - col[0])
        ys = [y - size - 16.0, y + (big_size + 26.0 if big else 0.0) + 22.0 + 25.0 * len(lines)] + ([anchor[1]] if anchor else [])
        xs = [xe, x_far] + ([anchor[0]] if anchor else [])
        with f.build(build, (min(xs) - 6.0, min(ys), max(xs) + 6.0, max(ys)), flow="out",
                     origin=anchor if anchor else (xe, y), wave=0.35, line=0.22, marks=False, key=int(y)):
            return note(f, lay, y, title, lines, anchor, red, 9.0, alpha, size, big, big_size, big_red,
                        plate=float(B.ease(B.lin(build, 0.08, 0.4))))
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
    if plate > 0.0:
        f.occlude(rect[0], rect[1], rect[2], rect[1] + (rect[3] - rect[1]) * min(plate, 1.0))
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
    """Particles riding logarithmic spiral arms around a centre, each one trailing the line of its own path
    (the trails of the TouchDesigner galaxy, scene3.0.mov): a thin curve that is bright at the particle and
    dies away behind it.
    Closed form: where a particle is, is a function of time alone, so its trail is simply where it was during
    the last seconds - a line is never switched on, it is drawn by its particle. Three regimes in that one
    function:
      emission (t_emit)   every particle leaves the centre in its turn - the nearest places first - and
                          travels out along its arm to its place: the field grows out of the dot, line by
                          line, each one from its first point
      settled             it drifts outward along its arm while the whole field turns
      fall (t_fall)       it falls straight to the centre, faster and faster, its line pulled in behind it,
                          and is swallowed at `absorb` (the limb of the star). No spin."""

    M = 200                                                  # segments of a line (closer together near the dot)

    def __init__(self, n=4600, arms=9, pitch=1.28, r_max=1780.0, r_min=58.0, seed=5):
        rng = np.random.default_rng(seed)
        self.n, self.arms, self.pitch, self.r_max, self.r_min = n, arms, pitch, r_max, r_min
        self.arm = rng.integers(0, arms, n)
        self.u = rng.random(n)
        self.j = rng.normal(0.0, 1.0, n)                     # position across the arm
        self.span = 7.0 + 11.0 * rng.random(n)               # how far back its trail goes (s): long lines
        self.b = rng.uniform(0.45, 1.0, n)
        self.spd = rng.uniform(0.92, 1.08, n)                # (nearly one speed and one pitch: the lines of the
        self.wob = np.zeros(n)                               # TouchDesigner galaxy run side by side, they do not cross)
        self.bold = rng.random(n) < 0.16
        self.arm_ph = rng.uniform(-0.12, 0.12, arms)
        self.late = rng.random(n)                            # its turn in the emission (0 = first)
        self.lag = rng.random(n)                             # ... and in the fall
        self.red = np.zeros(n, bool)                         # two of the lines are red (bold ones: red is thinner to the eye)
        self.red[np.random.default_rng(seed + 21).choice(n, min(2, n), replace=False)] = True

    def draw(self, f, t, cx, cy, gain=1.0, omega=0.11, drift=0.014, layer="w", t_emit=None, t_fall=None,
             absorb=0.0, emit=(5.4, 0.5, 1.2, 2.2, 0.7), fall=(0.1, 1.5, 1.2)):
        """t_emit = when the field starts to leave the centre (None: it has always been there);
        emit = (when the farthest leaves, jitter of the starts, travel time of the nearest, extra time of the
        farthest, length of the trail of a particle that travels), in seconds.
        t_fall = when it starts to fall back; fall = (jitter of the starts, fall time of the nearest, extra
        time of the farthest): one ordered movement, the nearest are swallowed first;
        absorb = radius at which a falling particle is swallowed."""
        M = self.M
        dr = self.r_max - self.r_min
        q0 = self.u + drift * self.spd * t
        base = np.floor(q0)
        qh = q0 - base                                        # its place on the way out (0..1)
        # how far back the trail goes: never before the particle was born or left the centre. A particle that
        # travels to its place is a comet (all of them cross the centre: full lines would pile up to white),
        # its line grows once it is there; a particle that falls pulls its line in behind it.
        span = np.full(self.n, 1e3)                           # every line goes back to the dot it left
        if t_emit is not None:
            reach = ((self.u + drift * self.spd * (t_emit + 4.0)) % 1.0) ** 0.62     # how far its place is (0..1)
            tau = t_emit + emit[0] * reach ** 1.15 + emit[1] * self.late
            fly = emit[2] + emit[3] * reach
            span = np.minimum(span, t - tau)
        if t_fall is not None and t > t_fall:
            fall_t = fall[1] + fall[2] * qh ** 0.62 + 0.15 * self.late
            ta = t - t_fall - fall[0] * self.lag
            xh = np.clip(ta / fall_t, 0.0, 1.0)
            gone = np.clip((ta - fall_t) / 0.45, 0.0, 1.0)    # swallowed: what is left of its line goes in after it
            # (its old arc is reeled in first; the stub that is left when it arrives shrinks to nothing - it used
            # to stay on the limb until the field was no longer drawn, and went out with it in one frame)
            span = span * (1.0 - xh) ** 3.5 + 0.3 * xh * (1.0 - gone * gone * (3.0 - 2.0 * gone))
        span = np.minimum(span, qh / (drift * self.spd))
        m = span > 0.02
        if not m.any():
            return
        ts = t - (1.0 - np.linspace(1.0, 0.0, M + 1) ** 2)[None, :] * span[m][:, None]
        qs = np.clip(self.u[m][:, None] + drift * self.spd[m][:, None] * ts - base[m][:, None], 0.0, 1.0)
        rs = dr * qs ** 0.62                                  # how far from the centre zone its place is
        if t_emit is not None:                                # on its way there: a soft start, a soft landing
            x = np.clip((ts - tau[m][:, None]) / fly[m][:, None], 0.0, 1.0)
            rs = rs * (x * x * (3.0 - 2.0 * x))
        r = self.r_min + rs
        # (the angle of a point depends on its radius alone, and on the turn of the whole field at t: every line is
        # a piece of one family of spirals, and two spirals of a family never cross)
        arm, j = self.arm[m], self.j[m]
        phi = (2 * np.pi * arm[:, None] / self.arms + self.arm_ph[arm][:, None]
               + (self.pitch + self.wob[m][:, None]) * np.log(r / self.r_min) + omega * t
               + 0.28 * j[:, None])
        # dim where it leaves the centre and where it reaches the edge of the field
        lvl = np.minimum(1.0, (1.0 - qs) / 0.05)
        if t_fall is not None:                                # the fall: along its own radius, no turn
            x = np.clip((t - t_fall - fall[0] * self.lag[m][:, None]) / fall_t[m][:, None], 0.0, 1.0)     # the whole line falls
            r = r * (1.0 - x ** 2.2)
        head_on = r[:, 0] > absorb + 1.0
        r = np.maximum(r, absorb)
        x = cx + r * np.cos(phi)
        y = cy + r * np.sin(phi)
        vis = ((x.max(1) > L.FX0 - 30) & (x.min(1) < L.FX1 + 30) & (y.max(1) > L.FY0 - 30) & (y.min(1) < L.FY1 + 30))
        if not vis.any():
            return
        inten = (gain * self.b[m] * (0.5 + 0.5 * np.exp(-0.5 * j ** 2)))[vis]
        x, y, lvl, bold, head_on, red = x[vis], y[vis], lvl[vis], self.bold[m][vis], head_on[vis], self.red[m][vis]
        ii = np.broadcast_to(np.where(red, 1.3, E.wl(inten * 0.85))[:, None], lvl.shape) * (lvl > 0.5)       # one level from end to end: no fade
        wb, wt = (2.0, E.WALL_LINE) if E.WALL else (1.5, 1.05)      # (the wall rule: no hairline, no grey line)
        # ... and, as every line is now white: one thin line in two (a static choice), or the field is a white mass
        kp = ((np.nonzero(m)[0][vis] % 2 == 0) | bold | red) if E.WALL else np.ones(len(bold), bool)
        head_on = head_on & kp
        for sel, w, lay_ in ((bold & ~red, wb, layer), (~bold & ~red & kp, wt, layer), (red, 1.9, "r")):
            if sel.any():
                f.segments(lay_, x[sel, :-1].ravel(), y[sel, :-1].ravel(), x[sel, 1:].ravel(), y[sel, 1:].ravel(),
                           ii[sel, :-1].ravel(), ii[sel, 1:].ravel(), width=w)
        for sel, lay_ in ((head_on & ~red, layer), (head_on & red, "r")):
            f.dots(lay_, x[sel, 0], y[sel, 0], np.where(bold[sel] | red[sel], 2.6, 1.8), (E.wl(1.5 * inten) * (lvl[:, 0] > 0.5))[sel])


# ----------------------------------------------------------------------------
# the rings
# ----------------------------------------------------------------------------

class Ripples:
    """Rings leaving the dot: each settles on its own radius and keeps breathing there (irregular spacing,
    doubles and triples, slightly off-centre: the TouchDesigner look). The widest goes first; every sound
    event of the music throws one more ring that travels out and fades."""

    DOT_GAP = 15.0                                   # px between the dots of a dotted ring

    def __init__(self, t0, onsets, far=1780.0, seed=11):
        rng = np.random.default_rng(seed)
        self.far = far
        fam = np.sort(np.r_[rng.uniform(70, 780, 9), rng.uniform(780, max(far - 60.0, 900.0), 7)])[::-1]
        R, tau, bold, first = [], [], [], []
        self.pulses = np.array([t for t in onsets if t0 + 2.0 < t < T_STAR])
        for k, rk in enumerate(fam):
            n = int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25]))
            ts = t0 + 0.04 + 0.17 * k + rng.uniform(0, 0.08)
            for j in range(n):
                R.append(rk + rng.normal(0, 4.0) + j * rng.uniform(7, 19))
                tau.append(ts + 0.09 * j)
                bold.append(rng.random() < 0.3)
                first.append(j == 0)
        self.R = np.array(R)
        self.tau = np.array(tau)
        self.bold = np.array(bold)
        n = len(R)
        # the wall rule: every ring that is drawn is white and at least 1.6 px - so fewer of them: the first of
        # each family and the bold ones (a static choice), the doubles and triples of hairlines are left out
        self.keep = (np.array(first) | self.bold) if E.WALL else np.ones(n, bool)
        self.T = rng.uniform(0.85, 1.15, n) * (0.55 + 0.5 * self.R / 900.0)
        self.off = rng.normal(0, 3.6, (n, 2))
        self.fq = rng.uniform(0.07, 0.19, n)
        self.ph = rng.uniform(0, 2 * np.pi, n)
        self.i = rng.uniform(0.5, 0.95, n)
        # a few rings are made of dots instead of a line: a fixed number of them for each (DOT_GAP apart once the
        # ring has settled), so that a ring that grows spreads its dots and none comes or goes
        rd = np.random.default_rng(seed + 60)
        self.dotted = rd.random(n) < 0.22
        self.spin = rd.uniform(0.05, 0.14, n) * rd.choice([-1.0, 1.0], n)       # they turn: rad / s, one way or the other
        self.dot_a = {int(k): np.linspace(0.0, 2 * np.pi, max(24, int(round(2 * np.pi * self.R[k] / self.DOT_GAP))), endpoint=False)
                      for k in np.nonzero(self.dotted)[0]}

    def radii(self, t, kick=0.0):
        a = np.maximum(t - self.tau, 0.0)
        r = self.R * (1.0 - np.exp(-a / self.T)) * (1.0 + 0.016 * np.sin(2 * np.pi * self.fq * t + self.ph))
        return r + 7.0 * kick * (0.3 + 0.7 * hash01(np.arange(len(r)), 3))

    def draw(self, f, t, cx, cy, gain=1.0, kick=0.0, scale=1.0, r_min=30.0, soft=0.0):
        """r_min = radius under which a ring is not drawn (the dot, the star); soft = over how many px above it
        a ring dims on its way in: a body that grows swallows the rings, they do not go out in one frame."""
        r = self.radii(t, kick) * scale
        age = t - self.tau
        inten = gain * self.i * np.clip(age / 0.25, 0, 1) * (0.55 + 0.45 * np.exp(-r / 900.0))
        inten = inten * (1.0 + 1.2 * np.exp(-np.maximum(age, 0) / 0.5))
        if soft > 0.0:
            inten = inten * np.clip((r - r_min) / soft, 0.0, 1.0)
        m = (age > 0) & (r > r_min) & (self.keep | self.dotted)
        inten = E.wl(inten)
        for sel, w in ((m & self.bold & ~self.dotted, L.LW_BOLD), (m & ~self.bold & ~self.dotted, E.ww(1.3))):
            if sel.any():
                f.rings("w", cx + self.off[sel, 0], cy + self.off[sel, 1], r[sel], inten[sel], width=w)
        for k in np.nonzero(m & self.dotted)[0]:
            a = self.dot_a[int(k)] + self.spin[k] * t
            f.dots("w", cx + self.off[k, 0] + r[k] * np.cos(a), cy + self.off[k, 1] + r[k] * np.sin(a),
                   (2.2 if self.bold[k] else 1.8) if E.WALL else (1.5 if self.bold[k] else 1.1), 1.3 * inten[k])
        a = t - self.pulses
        pm = (a > 0) & (a < 4.5)
        if pm.any() and scale == 1.0:
            pr = (self.far + 30.0) * (1.0 - np.exp(-a[pm] / 1.35))
            f.rings("w", np.full(pm.sum(), cx), np.full(pm.sum(), cy), pr, gain * (1.0 if E.WALL else 0.8) * np.exp(-a[pm] / 1.5),
                    width=E.ww(1.3))


# ----------------------------------------------------------------------------
# the collapse: rays in 3D (shared with messenger.py, which follows one of them)
# ----------------------------------------------------------------------------

class Nova:
    N_RAYS = 120

    def __init__(self, lay, seed=23):
        rng = np.random.default_rng(seed)
        self.C, self.half = lay.C, lay.half
        self.core_in = 0.1 * lay.r_star + 1.0                # radius of the imploded star: what the bounce starts from
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
        # the crown of the star, which the bounce throws out as the glare of the supernova: rays around the
        # centre, short and dense at the limb, a few long ones
        rng = np.random.default_rng(seed + 100)
        n = 1500
        self.ray_a = rng.uniform(0, 2 * np.pi, n)            # angle, length (radii of the star), level, phase, rate
        self.ray_len = np.where(rng.random(n) < 0.16, rng.uniform(0.7, 1.9, n), 0.05 + rng.exponential(0.2, n))
        self.ray_b = rng.uniform(0.45, 1.0, n)
        self.ray_ph = rng.uniform(0, 2 * np.pi, n)
        self.ray_w = rng.uniform(0.6, 2.4, n)
        self.ray_u = rng.random(n)                           # the wind: a point running out along a long ray
        self.ray_v = rng.uniform(0.12, 0.4, n)

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
        if E.WALL:                                      # heavier crosses: one in three each way on the plane
            m = m & (np.round(P[:, 0] / 64.0) % 3 == 0) & (np.round(P[:, 1] / 64.0) % 3 == 0)
        f.crosses("w", sx[m], sy[m], 8.0 * sc[m].mean() if m.any() else 8.0, 0.22 * gain)
        # the implosion had pulled the rings in (Origin._star: suck): the bounce throws them back out, and the
        # shock leaves the core - nothing is put in place in one frame
        back = SUCK + (1.0 - SUCK) * float(smoothstep(0.0, 0.35, t - T_X))
        r = rip.radii(t) * back
        a = np.linspace(0, 2 * np.pi, 181)
        sel = np.nonzero((t - rip.tau > 0) & (r > 60.0 * back) & (rip.keep | rip.dotted))[0]
        shock = 260.0 * (1.0 - math.exp(-(t - T_X) / 0.09)) + 1250.0 * (1.0 - math.exp(-(t - T_X) / 1.6))
        for k in sel:
            rr = r[k] * (1.0 + 0.3 * math.exp(-((r[k] - shock) / 180.0) ** 2))
            ak = rip.dot_a[int(k)] + rip.spin[k] * t if rip.dotted[k] else a
            P = np.stack([rr * np.cos(ak) + rip.off[k, 0], rr * np.sin(ak) + rip.off[k, 1], np.zeros_like(ak)], 1)
            px, py, pz, pok = cam.project(P.astype(np.float32))
            if rip.dotted[k]:
                f.dots("w", px[pok], py[pok], (2.2 if rip.bold[k] else 1.8) if E.WALL else (1.5 if rip.bold[k] else 1.1),
                       E.wl(0.8 * gain * rip.i[k]))
            elif pok.all():
                f.polyline("w", px, py, E.wl(gain * rip.i[k] * 0.6), width=L.LW_BOLD if rip.bold[k] else E.ww(1.2))
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

    def draw_core(self, f, x, y, t, flash=0.0, beat=0.0, scale=1.0, gain=1.0, out=1.0):
        """The supernova itself: a core hotter than white (it glows) and its glare - the crown of the star thrown
        out by the bounce: rays, not a flat white disc. flash = 1 at the bounce, then it dies away; scale and
        gain shrink and dim the glare of the star that is left behind (messenger.py).
        out = 0..1, the burst itself: at 0 the core and the crown are those of the imploded star (core_in), the
        bounce throws them out to their size - the glare leaves the core, it is not switched on."""
        c_in = self.core_in
        r_core = (26.0 + 30.0 * flash + 12.0 * beat) * scale + 5.0
        f.dots("w", [x], [y], c_in + (r_core - c_in) * out, 1.6 + 1.6 * gain)
        if gain <= 0.01:
            return
        ln = self.ray_len * (150.0 + 520.0 * flash) * scale * (0.8 + 0.2 * np.sin(self.ray_ph + self.ray_w * t) + 0.35 * beat)
        ln = self.ray_len * c_in + (ln - self.ray_len * c_in) * out
        r_in = (30.0 + 30.0 * flash) * scale + 5.0
        r_in = c_in + (r_in - c_in) * out
        ca, sa = np.cos(self.ray_a), np.sin(self.ray_a)
        f.segments("w", x + r_in * ca, y + r_in * sa, x + (r_in + ln) * ca, y + (r_in + ln) * sa,
                   (0.9 + 1.6 * flash) * gain * self.ray_b, 0.0, width=E.ww(1.2))

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


def hero_note(f, lay, tip, age, left=None):
    """The tag of the ray that will reach us: in the notes column, a red leader to its tip. It is constructed
    from the tip of the ray (age = seconds since it appeared) and taken apart `left` seconds before it goes."""
    age = B.io(age, left, out=0.4)
    if tip is None or age <= 0:
        return
    ok = lay.note is not None and note(f, lay, lay.C[1] + 150.0, "P+ // THE ONE THAT WILL REACH US",
                                       ["PRIMARY COSMIC RAY // PROTON", "E 3.2E15 EV", "HEADING: HERE"],
                                       anchor=(tip[0] + 12.0 * lay.note_side, tip[1] + 10.0), red=True, build=age)
    if not ok:                                       # no notes column: a small tag on the ray itself
        f.occlude(tip[0] + 20.0, tip[1] + 16.0, tip[0] + 70.0, tip[1] + 50.0)
        B.tag(f, "r", tip[0] + 26.0, tip[1] + 42.0, "P+", age, size=L.T_TAG, pad=5)


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
        self.field = StreakField(n=120, r_max=far + 80.0, r_min=18.0)
        self.field.red[117] = True                           # a third red line, picked on the frame (the long one to the right)
        self.rip = Ripples(T_HERE, list(ctx.cues.onset_t), far)
        self.nova = Nova(self.lay)
        rng = np.random.default_rng(3)
        # arrival times of the muons through one spectator, RATE_YOU / s (Poisson), and their energies
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
        # the star (00:31 - 00:40): a bright body that keeps its detail. The burning shells are rings of radial
        # bars - a barcode wrapped around the core, each ring turning on its own -, a sparse granulation, a limb
        # that boils, and a crown of rays with the wind running out along the long ones
        sr = self.SHELL_R
        ang, e0, e1, lvl, band, run = [], [], [], [], [], []
        n_run = 0
        for k in range(len(sr) - 1):
            n = int(2 * np.pi * 0.5 * (sr[k] + sr[k + 1]) * 212.0 / 2.6)       # one line every 2.6 px of the ring
            i = 0
            while i < n:
                w = int(rng.integers(2, 13))                                    # a bar (a run of lines) or a gap
                if rng.random() < 0.62:
                    idx = np.arange(i, min(i + w, n))
                    a, b = rng.choice([0.0, 0.0, 0.0, 0.3, 0.55], 2)            # some bars stop short of the ring
                    if a + b > 0.6:
                        b = 0.0
                    ang.append((idx + rng.uniform(-0.15, 0.15, len(idx))) / n)
                    e0.append(np.full(len(idx), a))
                    e1.append(np.full(len(idx), b))
                    lvl.append(np.full(len(idx), rng.uniform(0.5, 1.0)))
                    band.append(np.full(len(idx), k))
                    run.append(np.full(len(idx), n_run))
                    n_run += 1
                i += w
        self.bar_a, self.bar_e0, self.bar_e1 = np.concatenate(ang), np.concatenate(e0), np.concatenate(e1)
        self.bar_b, self.bar_band, self.bar_run = np.concatenate(lvl), np.concatenate(band), np.concatenate(run)
        self.bar_turn = rng.uniform(0.006, 0.02, len(sr) - 1) * np.where(np.arange(len(sr) - 1) % 2, -1.0, 1.0)
        self.bar_blink = rng.random(n_run) < 0.15                             # these bars breathe, each at its own pace
        n = 800
        r = np.sqrt(rng.random(n))
        a = rng.uniform(0, 2 * np.pi, n)
        self.gran = np.stack([r * np.cos(a), r * np.sin(a)], 1)
        self.gran_r = rng.uniform(1.2, 2.8, n)
        self.gran_b = rng.uniform(0.5, 1.3, n)
        self.gran_ph = rng.uniform(0, 2 * np.pi, n)
        self.limb_k = rng.integers(3, 23, 7)                 # harmonics of the boiling limb
        self.limb_ph = rng.uniform(0, 2 * np.pi, 7)
        self.limb_w = rng.uniform(-1.3, 1.3, 7)
        self.blink_ph = rng.uniform(0, 2 * np.pi, n_run)     # the bars that breathe: phase, rate (a period of 0.7 - 1.8 s)
        self.blink_w = 2 * np.pi / rng.uniform(0.7, 1.8, n_run)
        self.ray_plan = self._plan_ray_tags()                # which rays of the supernova are named, and when

    # ------------------------------------------------------------------ draw
    def draw(self, f, t, ctx):
        self.lay.avoid = []
        if t < T_STAR + 0.05:
            return self._origin(f, t, ctx)
        elif t < T_X:
            self._star(f, t, ctx)
        else:
            return self._nova(f, t, ctx)
        return {"cell": False}           # no bottom band under the star: the counter is built with the supernova

    def _panels(self, f, fns, alpha=1.0, age=None, lag=0.12):
        """Bottom band: the panels, by priority, in the free slots between the towers (extra ones dropped).
        age = seconds since they appeared: each panel then constructs itself (Frame.build), one after the
        other; a panel never fades in."""
        y0, y1 = L.BOT[1], L.BOT[3]
        for k, ((x0, x1), fn) in enumerate(zip(self.lay.slots, fns)):
            with f.build(None if age is None else age - lag * k, (x0 - 8.0, y0 - 24.0, x1 + 8.0, y1 + 8.0), key=30 + k):
                f.occlude(x0 - 10.0, y0 - 26.0, x1 + 10.0, L.FY1 - 3.0)
                fn(f, x0, x1, y0, y1, alpha)

    # --- 00:00 - 00:31 ---------------------------------------------------------
    def _boot(self, t):
        """Finishing options of the opening: black, the border fades in, the furniture builds up once loaded
        (no counter in the origin: it is built with the star, see draw). Data never fades in: it is constructed."""
        fa = float(smoothstep(T_FRAME, T_FRAME + 2.2, t))
        return {"frame_alpha": fa, "tower_outline": 0.16 * fa, "edge_ticks": t >= T_LOADED,
                "edge_kw": {"reveal": B.lin(t, T_LOADED, T_READY)}, "cell": False}

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

    def _lattice(self, f, t, gain=1.0, hole=0.0):
        """The lattice of crosses; hole = radius around the centre left free (the star): the crosses dim as the
        limb comes close to them, they are not taken off one ring at a time."""
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
        w = np.ones(len(X))
        if hole > 0.0:
            g = min(hole / 10.0, 1.0)                   # a hole that opens from nothing: no cross goes in one frame
            w = 1.0 - g + g * np.clip((np.hypot(X - C[0], Y - C[1]) - hole) / min(70.0, 4.0 * hole), 0.0, 1.0)
            on = on & (w > 0.0)
        minor = on & ~major
        if E.WALL:                                      # heavier crosses: one in two each way, or they are a wallpaper
            minor = minor & ((KX % 2 == 0) & (KY % 2 == 0)).ravel()
        f.crosses("w", X[minor], Y[minor], 7.0, (0.2 * a * w)[minor])
        f.crosses("w", X[on & major], Y[on & major], 10.0, (0.42 * a * w)[on & major], width=1.3)
        new = on & (row >= head - 1.0)
        if new.any() and head < len(ky) + 1.0:
            f.crosses("w", X[new], Y[new], 9.0, 0.9 * a, width=1.3)

    def _pulse(self, t, ctx):
        """The low thumps as they hit: it jumps on each of them (the red dot of the opening)."""
        return min(1.5, ctx.cues.kick(t, 0.28) * 1.6 + 0.5 * ctx.cues.onset(t, 0.3))

    def _swell(self, t, ctx):
        """The same thumps as a smooth envelope (swell): what the star is driven by. Its crown, its shells and
        its core swell before a hit and relax after it; with _pulse they jumped on every hit, and four hits in a
        row made the star grow in four steps."""
        c = ctx.cues
        v = 1.6 * swell(c.kick_t, c.kick_a, t) + 0.5 * swell(c.onset_t, c.onset_a, t)
        return 1.5 * math.tanh(v / 1.5)

    def _drive(self, t, ctx):
        """What the field and the rings follow: the thumps as they hit during the opening, handed over to the
        smooth envelope while the dot shrinks to a point (00:28 - 00:31), so that nothing changes at the star."""
        w = float(smoothstep(T_MINI, T_STAR - 0.3, t))
        return (self._pulse(t, ctx) * (1.0 - w) if w < 1.0 else 0.0) + (self._swell(t, ctx) * w if w > 0.0 else 0.0)

    def _crosshair(self, f, r, grow):
        """The axes of the whole image, from the dot (radius r) to the frame; grow = 0..1 of their length."""
        C = self.C
        g = r + 14
        xl, xr = C[0] - g - (C[0] - g - L.FX0) * grow, C[0] + g + (L.FX1 - C[0] - g) * grow
        yt, yb = C[1] - g - (C[1] - g - L.FY0) * grow, C[1] + g + (L.FY1 - C[1] - g) * grow
        f.segments("r", [xl, C[0] + g, C[0], C[0]], [C[1], C[1], yt, C[1] + g],
                   [C[0] - g, xr, C[0], C[0]], [C[1], C[1], C[1] - g, yb], 0.8 if E.WALL else 0.42, width=E.ww(1.2))
        k = np.arange(1, 30) * 100.0
        for sgn in (-1.0, 1.0):
            xs = C[0] + sgn * k
            xs = xs[(xs > xl) & (xs < xr)]
            f.segments("r", xs, np.full_like(xs, C[1] - 6), xs, np.full_like(xs, C[1] + 6), E.wl(0.6), width=E.ww(1.0))
            ys = C[1] + sgn * k
            ys = ys[(ys > yt) & (ys < yb)]
            f.segments("r", np.full_like(ys, C[0] - 6), ys, np.full_like(ys, C[0] + 6), ys, E.wl(0.6), width=E.ww(1.0))

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
            self._crosshair(f, r, grow)
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
        kick = self._drive(t, ctx)
        f.set_clip(*FRAME_CLIP)
        if t > T_FIELD:                 # the particles leave the dot one after the other, each drawing its line
            self.field.draw(f, t, C[0], C[1], gain=0.9 + 0.25 * kick, t_emit=T_FIELD,
                            t_fall=T_STAR if t > T_STAR else None, absorb=6.0)       # (its fall starts on the word)
        if t > T_HERE:
            self.rip.draw(f, t, C[0], C[1], kick=kick)
        f.set_clip()
        self._dot(f, t, ctx, size=1.0 - 0.94 * mini)
        if t <= T_HUD:
            return opt
        self._strip_arrivals(f, t, ctx)
        self._notes(f, t, mini)
        self._dial(f, t)
        return opt

    def _notes(self, f, t, mini):
        """What the voice says about the dot, written on its axis in the notes column."""
        lay, C = self.lay, self.C
        if T_HERE <= t < T_ALWAYS:                  # each note is taken apart before the next one is made
            left = T_ALWAYS - 0.1 - t
            a = B.io(t - T_HERE, left, out=0.45)
            axis_note(f, lay, "HERE", above=["CINCINNATI // 147 M ASL", "39.103 N  084.512 W"], title2="RIGHT NOW",
                      below=[f"T {sd.tc(t)}", "1 MUON EVERY 16 MS"], age=a, age2=a - 0.75, build=True, commit=left > 0.45)
        if mini > 0:
            g = 150.0 - 118.0 * mini
            for sx in (-1, 1):
                for sy in (-1, 1):
                    x, y = C[0] + sx * g, C[1] + sy * g
                    f.segments("r", [x, x], [y, y], [x - sx * 26, x], [y, y - sy * 26], 1.0, width=L.LW_BOLD)
            left = T_STAR - 0.05 - t                # it is taken apart before the star is born out of the point
            a = B.io(t - T_MINI - 0.5, left, out=0.4)
            if a > 0:
                axis_note(f, lay, "MU // POINT-LIKE", below=["SIZE   < 1E-18 M", "MASS   1.88E-28 KG", "CHARGE -1 E",
                                                           "NO STRUCTURE FOUND"], age=a, build=True, commit=left > 0.4)

    def _dial(self, f, t):
        """A fixed scale among the moving rings: light-time shells, 1 px = 1 cm on the wall.
        Built, not faded in: a pen shoots out of the dot and brakes; every tick is thrown out long as the pen
        passes it and falls back to its length; its label is decoded out of noise. It is taken apart the same
        way, backwards, just before the star is born (it used to go out with the cut, on the star's first frame)."""
        grow = 1.5
        age = B.io(t - (T_HUD + 1.8), T_STAR - 0.05 - t, out=0.45, span=grow)
        if age <= 0.0:
            return
        lay, C = self.lay, self.C
        ang = math.radians(-27.0) if lay.note_side > 0 else math.radians(-153.0)
        ca, sa = math.cos(ang), math.sin(ang)
        rmax = min(lay.far, (C[1] - L.HEAD_Y - 30.0) / abs(sa))
        u = float(B.ease(age / grow))
        reach = rmax * u
        rr = np.arange(100.0, rmax, 100.0)
        tr = grow * (1.0 - (1.0 - rr / rmax) ** (1.0 / 3.0))        # when the pen passes each tick
        on = age >= tr
        if reach > 60.0:
            f.segments("w", [C[0] + 60 * ca], [C[1] + 60 * sa], [C[0] + reach * ca], [C[1] + reach * sa], E.wl(0.6), width=E.ww(1.3))
            if u < 1.0:
                f.dots("w", [C[0] + reach * ca], [C[1] + reach * sa], 3.2, 1.7)
        hl = 9.0 + 17.0 * np.exp(-(age - tr[on]) / 0.1)
        f.segments("w", C[0] + rr[on] * ca + hl * sa, C[1] + rr[on] * sa - hl * ca, C[0] + rr[on] * ca - hl * sa,
                   C[1] + rr[on] * sa + hl * ca, 0.9, width=E.ww(1.3))
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
        """Score strip: every muon through one spectator as a tick (RATE_YOU / s), the last five seconds.
        It builds up in a good second, without a fade: the red band shoots across, the rules and their ticks
        follow it (hud.strip_base); then a scan head prints the arrivals - close to the head they are still
        noise and settle behind it; the cursor drops, its tag is made and its count spins before it locks."""
        age = t - T_HUD
        x0, y0, x1, y1 = L.STRIP
        f.occlude(x0, y0, x0 + (x1 - x0) * float(B.ease(B.lin(age, 0.0, 0.36))), y1)
        span, ahead = 5.0, 0.9
        ta, tb = t - span, t + ahead
        ix0, iy0, ix1, iy1, yb = hud.strip_base(f, ticks=(ta, tb, 0.1, 1.0), age=age)
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
        f.rects("w", xs[lo], iy1 - 5, xs[lo] + 2, iy1 - 1, E.wl(0.5))
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

    # --- 00:31 - 00:40 ---------------------------------------------------------
    # limits of the burning shells, in radii of the star: the silicon around the iron core ... the hydrogen envelope
    SHELL_R = (0.13, 0.25, 0.35, 0.46, 0.58, 0.72, 0.86, 1.0)

    def _star_disc(self, f, t, R, gain=1.0, hole=0.0, pulse=0.0):
        """The star, as a stylised supernova: bright, and made of detail instead of a flat white disc.
        The body is the burning shells - rings of radial bars, a barcode wrapped around the core, each ring
        turning on its own, some bars flickering - over a dim fill, with a sparse granulation; the limb is a
        bold ring that boils; around it a crown of rays (dense and short at the limb, a few long ones) with the
        wind running out along the long ones. Lines are drawn hotter than white where the star must glow.

        Everything here is a continuous function of R, of the hole and of the pulse, so the star can grow, breathe
        and give way without a step: a shell comes out of the core by getting longer, its ring and its grains
        come up as they leave the core, a bar that breathes does it on its own slow period (nothing is switched
        on or off), the limb and the rings keep their vertices in place while they grow (ring)."""
        if R < 0.5:
            return
        C = self.C
        k = max(R / 212.0, 0.25)
        gain = gain * float(smoothstep(0.5, 9.0, R))              # it comes out of the point: no first frame
        wall = E.WALL       # the wall rule: no grey in the body - black between full white bars, one line in two
        if not wall:
            fill = np.arange(hole + 6.0, R - 3.0, 1.0 / f.s)      # a dim fill, as rings: the red core stays red
            f.rings("w", np.full(len(fill), C[0]), np.full(len(fill), C[1]), fill, 0.14 * gain)
        # the shells: rings of radial bars
        sr = np.asarray(self.SHELL_R)
        ra, rb = sr[self.bar_band], sr[self.bar_band + 1]
        h = rb - ra
        r0 = np.maximum((ra + 0.07 * h + self.bar_e0 * h) * R, hole + 5.0)
        r1 = (rb - 0.07 * h - self.bar_e1 * h) * R
        a = 2 * np.pi * (self.bar_a + self.bar_turn[self.bar_band] * t)
        br = 0.5 + 0.5 * np.sin(self.blink_ph + self.blink_w * t)
        breath = np.where(self.bar_blink, 0.1 + 0.9 * br * br * (3.0 - 2.0 * br), 1.0)[self.bar_run]
        if wall:            # a bar that breathes gets shorter instead of dimmer; its level is the white of the wall
            r1 = r0 + (r1 - r0) * breath
            on = (r1 > r0 + 1.0) & (np.arange(len(r0)) % 2 == 0)
            lvl = np.full(len(r0), min(gain, 1.0))
        else:
            on = r1 > r0 + 1.0
            # (the lines keep one width: a line of 1.25 px or less is drawn in one pass and would change level there)
            lvl = gain * self.bar_b * breath * (1.0 + 0.5 * pulse) * (0.67 + 0.33 * float(smoothstep(0.4, 0.6, k)))
        f.segments("w", C[0] + r0[on] * np.cos(a[on]), C[1] + r0[on] * np.sin(a[on]), C[0] + r1[on] * np.cos(a[on]),
                   C[1] + r1[on] * np.sin(a[on]), lvl[on], width=2.2 if wall else 1.5)
        rr = sr[1:-1] * R
        ring(f, "w", C[0], C[1], rr, E.wl(0.7 * gain * np.clip((rr - hole - 8.0) / 14.0, 0.0, 1.0)), width=E.ww(1.3))
        # granulation: points that flicker
        P = self.gran
        rg = np.hypot(P[:, 0], P[:, 1]) * R
        edge = np.clip((rg - hole - 5.0) / 10.0, 0.0, 1.0) * np.clip((R - 5.0 - rg) / 8.0, 0.0, 1.0)
        m = edge > 0.0
        flick = 0.6 + 0.4 * np.sin(self.gran_ph + t * 2.3)
        if wall:            # the grains flicker in size, at full level
            f.dots("w", C[0] + P[m, 0] * R, C[1] + P[m, 1] * R, self.gran_r[m] * max(k, 0.6) * flick[m] * edge[m],
                   min(gain, 1.0))
        else:
            f.dots("w", C[0] + P[m, 0] * R, C[1] + P[m, 1] * R, self.gran_r[m] * max(k, 0.6),
                   gain * self.gran_b[m] * flick[m] * edge[m])
        ring(f, "w", C[0], C[1], [hole + 3.0], 1.1 * gain * float(smoothstep(hole + 5.0, hole + 24.0, R)), width=1.6)
        # the limb: a bold ring, hotter than white so that it glows, and a line that boils around it
        ring(f, "w", C[0], C[1], [R, R - 6.0 * k], [2.4 * gain, 0.9 * gain], width=3.0)
        th = np.linspace(0.0, 2 * np.pi, 361)
        boil = sum(np.sin(kk * th + ph + w * t) / (1.0 + 0.12 * kk) for kk, ph, w in zip(self.limb_k, self.limb_ph, self.limb_w))
        rb_ = R + (9.0 + 3.6 * boil + 5.0 * pulse) * k
        f.polyline("w", C[0] + rb_ * np.cos(th), C[1] + rb_ * np.sin(th), 0.9 * gain, width=E.ww(1.3))
        # the crown: rays that breathe and swell with the music; the wind runs out along the long ones
        nv = self.nova
        ln = nv.ray_len * R * (0.78 + 0.22 * np.sin(nv.ray_ph + nv.ray_w * t) + 0.3 * pulse)
        r_in = R + 15.0 * k
        ca, sa = np.cos(nv.ray_a), np.sin(nv.ray_a)
        f.segments("w", C[0] + r_in * ca, C[1] + r_in * sa, C[0] + (r_in + ln) * ca, C[1] + (r_in + ln) * sa,
                   1.15 * gain * nv.ray_b, 0.0, width=E.ww(1.2))
        far = nv.ray_len > 0.6
        pw = (nv.ray_u[far] + nv.ray_v[far] * t) % 1.0
        rw = r_in + ln[far] * pw
        f.dots("w", C[0] + rw * ca[far], C[1] + rw * sa[far], 1.9 * max(k, 0.6),
               1.4 * gain * nv.ray_b[far] * (1.0 - pw) ** 0.7 * np.minimum(1.0, pw / 0.06))

    def _star(self, f, t, ctx):
        lay, C = self.lay, self.C
        a = t - T_STAR
        sw = self._swell(t, ctx)        # the music, as a smooth envelope: the star swells, it does not jump
        implode = float(smoothstep(T_IMPLODE, T_X, t)) ** 2.2
        # it swells out of the point the dot had shrunk to: a slow start, a soft landing (it used to leave at
        # full speed and to be there, whole, on the first frame)
        R = lay.r_star * float(smoothstep(0.0, T_GROW, a)) * (1.0 - 0.9 * implode)
        self._lattice(f, t, gain=0.55 + 0.2 * float(smoothstep(0.0, 1.0, a)), hole=R + 14.0 * float(smoothstep(0.0, 0.4, a)))
        f.set_clip(*FRAME_CLIP)
        if a < T_FALL + 0.3:            # the field of the opening falls into the star: straight in, no spin
            self.field.draw(f, t, C[0], C[1], gain=0.9 + 0.25 * sw, t_emit=T_FIELD, t_fall=T_STAR, absorb=R + 6.0)
        suck = 1.0 - (1.0 - SUCK) * implode
        self.rip.draw(f, t, C[0], C[1], gain=1.0, kick=sw, scale=suck, r_min=max(R + 10.0, 30.0),
                      soft=40.0 * min(1.0, R / 40.0))
        f.set_clip()
        core = max((24.0 * (1.0 - 0.55 * implode) + 5.0 * sw) * min(1.0, R / 60.0), 2.6)
        if a < 0.5:                     # the axes and the brackets of the opening go back into the dot
            u = float(smoothstep(0.05, 0.45, a))
            if u < 1.0:
                self._crosshair(f, core, 1.0 - u)
                g, arm = 32.0 - 22.0 * u, 26.0 * (1.0 - u)
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        x, y = C[0] + sx * g, C[1] + sy * g
                        f.segments("r", [x, x], [y, y], [x - sx * arm, x], [y, y - sy * arm], 1.0, width=L.LW_BOLD)
        self._star_disc(f, t, R, gain=1.0 + 0.8 * implode, hole=core + 8.0 * min(1.0, R / 40.0), pulse=sw)
        f.dots("r", [C[0]], [C[1]], core, 1.4)
        self._strip_star(f, t, ctx)
        left = max(T_X - t, 0.0)
        card(f, lay, "PROGENITOR", [("CLASS     RED SUPERGIANT", "RED SUPERGIANT"), ("MASS      25 M_SUN", "M  25 M_SUN"),
                                    ("RADIUS    1 000 R_SUN", "R  1 000 R_SUN"), ("AGE       7.1E6 YR", "AGE 7.1E6 YR"),
                                    ("CORE      FE  1.4 M_SUN", "CORE FE 1.4 M_SUN"), ("T_CORE    5.0E9 K", "T_CORE 5.0E9 K"),
                                    ("RHO_CORE  1E10 G/CM3", None),
                                    (f"COLLAPSE  T-{left:06.3f} S", f"T-{left:06.3f} S")], a, red_rows=(7,), build=True)
        self._star_notes(f, t, R, core, a, implode)

    def _star_notes(self, f, t, R, core, a, implode):
        """Three read-outs in the notes column with leaders to the star; the core sits on the axis. Each one is
        constructed from the star outwards; all are taken apart when the disc gives way."""
        lay, C = self.lay, self.C
        left = T_IMPLODE + 0.5 - t                  # seconds before they must be gone
        t_n = T_GROW - 0.3                          # they come when the star has nearly its size
        if a < t_n or left <= 0.0 or _note_col(lay)[0] is None:
            return
        side = lay.note_side
        col, _ = _note_col(lay)
        xe = col[0] if side > 0 else col[1]
        wpl = min(col[1] - col[0] - 30.0, 400.0)
        xp = xe + 30.0 if side > 0 else xe - 30.0 - wpl
        pp = float(B.ease(B.lin(B.io(a - t_n, left, out=0.4), 0.0, 0.3)))         # their plate opens downwards
        f.occlude(xp, C[1] - 290.0, xp + wpl, C[1] - 290.0 + 281.0 * pp)
        ac = B.io(a - t_n - 1.4, left, out=0.4)
        if ac > 0.0:
            # the axis from the core: a slit is cut through the disc so the red leader shows; the pen that draws
            # the leader cuts it (it is not there before the pen)
            pc = float(B.ease(B.lin(ac, 0.0, 0.3)))
            xs = C[0] + side * (core + 4.0)
            cut = min(R + 4.0, core + 4.0 + abs(xe - xs) * pc + 6.0)
            if cut > core + 12.0:
                f.occlude(min(C[0] + side * (core + 12.0), C[0] + side * cut), C[1] - 3.5,
                          max(C[0] + side * (core + 12.0), C[0] + side * cut), C[1] + 3.5)
            B.pen(f, "r", xs, C[1], xe, C[1], pc, 0.95, width=L.LW)
            axis_note(f, lay, "IRON CORE", below=["1.4 M_SUN  //  R 1 500 KM", "T 5.0E9 K", "NOTHING LEFT TO BURN"],
                      age=ac - 0.2, build=True, commit=left > 0.4)
        ang = math.radians(-50.0)
        note(f, lay, C[1] - 252.0, "PHOTOSPHERE", ["R 1 000 R_SUN  //  T 3 600 K"],
             anchor=(C[0] + side * R * math.cos(ang), C[1] + R * math.sin(ang)), build=B.io(a - t_n, left, out=0.4))
        ang = math.radians(-22.0)
        note(f, lay, C[1] - 142.0, "HYDROGEN ENVELOPE", ["16 M_SUN  //  THE REST HAS BURNED"],
             anchor=(C[0] + side * (R + 40.0) * math.cos(ang), C[1] + (R + 40.0) * math.sin(ang)),
             build=B.io(a - t_n - 0.6, left, out=0.4))

    def _strip_star(self, f, t, ctx):
        x0, y0, x1, y1 = L.STRIP
        f.occlude(x0, y0, x1, y1)
        marks = [(T_STAR, "STAR"), (T_IMPLODE, "COLLAPSE"), (T_X, "BOUNCE"), (T_BLOOMED, "BLOOM")]
        hud.show_strip(f, t, ctx, "STAR // THE LAST SECONDS", 30.0, 48.0, marks, age=t - T_STAR - 0.05)
        header_gap(f, ctx, t)

    # bottom panels of the star -------------------------------------------------------
    def _p_shells(self, f, t, x0, x1, y0, y1, al, a, implode):
        hud.panel_header(f, x0, x1, y0, title_fit(["BURNED // SHELL BY SHELL", "BURNED"], x1 - x0), alpha=al)
        shells = [("H", "7E6 Y"), ("HE", "7E5 Y"), ("C", "600 Y"), ("NE", "1 Y"), ("O", "6 MO"), ("SI", "1 D"),
                  ("FE", "INERT")]
        cw = (x1 - x0) / len(shells)
        for k, (el, dur) in enumerate(shells):
            xx = x0 + k * cw
            last = k == len(shells) - 1
            # one shell after the other, each one constructed when its turn comes (outline traced, block grown,
            # name decoded): a shell is never just there
            with f.build(a - 0.5 - 0.35 * k, (xx + 1, y0 + 16, xx + cw - 4, y0 + 118), wave=0.12, line=0.2, marks=False,
                         key=70 + k):
                f.rect("w", xx + 3, y0 + 20, xx + cw - 6, y0 + 62, E.wl(0.7 * al), width=E.ww(1.0))
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
        f.segments("w", [xa], [ya], [xb], [ya], E.wl(0.7 * al), width=E.ww(1.0))
        for v, lab in ((10.0, "10"), (100.0, "100"), (1000.0, "1 000")):
            xv = xa + math.log10(v / 10.0) / math.log10(300.0) * (xb - xa)
            f.segments("w", [xv], [ya], [xv], [ya + 7], E.wl(0.8 * al), width=E.ww(1.0))
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
        # the bounce throws the glare out of the imploded core in T_BURST (an accelerating burst, a few frames):
        # it is not there, whole, on the first frame. The flash is at its top when the glare is out.
        out = float(smoothstep(0.0, T_BURST, a))
        flash = math.exp(-max(a - T_BURST, 0.0) / 0.5)
        burst = math.exp(-max(a - T_BURST, 0.0) / 0.16)  # the first instant of the bounce
        c = ctx.cues                                     # the music swells the glare, it does not make it jump
        beat = 1.6 * swell(c.onset_t, c.onset_a, t, 0.1, 0.12) + swell(c.kick_t, c.kick_a, t, 0.1, 0.12)
        nova.draw_core(f, X, Y, t, flash=flash, beat=beat, out=out)
        nova.draw_rays(f, cam, t, gain=1.35)
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
             a, red_title=True, red_rows=(3,), cps=80.0, build=True)
        hero_note(f, lay, tip, a - 1.2)
        f.set_clip(*FRAME_CLIP)
        self._ray_tags(f, cam, t)
        f.set_clip()
        with f.build(a - 0.1, lay.cell_r, marks=False, key=41):
            cell_right(f, lay, "YEAR // BEFORE NOW", f"-{YEAR0:,.0f}".replace(",", " "), sub="T+" + f"{a:05.2f} S", red=True,
                       value_short=f"-{YEAR0:.1E}".replace("E+0", "E"))
        # no counter of the show under the supernova: it is built at the cut to the messenger
        return {"exposure": 1.0 + (0.5 * flash + 1.2 * burst) * out, "bloom_gain": 0.75 + 0.5 * flash * out,
                "cell": False}

    def _ray_text(self, k):
        return f"{self.nova.kind[k]} {0.3 * 10 ** (2 + 4 * hash01(k, 7)):.1E} GEV".replace("E+0", "E")

    def _plan_ray_tags(self):
        """{ray: (when its name is written, when it is gone)} for the few rays of the supernova that are named.
        A name hangs from the tip of its ray, which sweeps the wall while the camera drops and circles: where
        the wall is free under it, and for how long, is known in advance, so it is decided once, here. A name is
        written when its way is clear for RAY_TAG_MIN seconds at least, and taken apart before it would meet a
        tower, the card, the note of the hero ray, another name or the edge of the view. (Asked at every frame,
        'is the wall free here?' made the names come on already written, and go off and on as their rays moved.)"""
        nova, lay = self.nova, self.lay
        dt = 1.0 / 30.0
        ts = np.arange(T_X + 0.2, T_END + 1e-9, dt)
        zero = np.zeros((nova.n, 3))
        tips = np.zeros((len(ts), nova.n, 2))
        vis = np.zeros((len(ts), nova.n), bool)
        taken = []
        col, _ = _note_col(lay)
        for i, t in enumerate(ts):
            t = float(t)
            cam = nova.camera(t)
            Lr = nova.lengths(t)
            ax, ay, az, bx, by, bz, ok, tip_in = nova._clip_near(cam, zero, nova.dir * Lr[:, None])
            ox, oy, _, _ = cam.project(np.zeros((1, 3), np.float32))
            tips[i, :, 0], tips[i, :, 1] = bx, by
            vis[i] = ((Lr > 1.0) & ok & tip_in & (nova.fade_t - 0.2 > t)
                      & (np.hypot(bx - float(ox[0]), by - float(oy[0])) >= 300.0))
            hx, hy = nova.tip_screen(t)
            obst = [(hx - 50.0, hy - 50.0, hx + 50.0, hy + 50.0)]          # the tip of the hero ray
            if lay.card is not None:                                        # the card of the scene
                obst.append((lay.card[0] + 2.0, CARD_Y - 46.0, lay.card[1] + 10.0, CARD_Y + 36.0 + 8 * 26.0))
            if col is not None and t >= T_X + 1.2:                          # the note of the hero ray (hero_note)
                obst.append((col[0], lay.C[1] + 116.0, col[1], lay.C[1] + 245.0))
            taken.append(obst)
        hit = lambda r, o, pad: r[2] > o[0] - pad and r[0] < o[2] + pad and r[3] > o[1] - pad and r[1] < o[3] + pad
        lay.avoid = []
        plan, rects = {}, {}
        for k in range(0, nova.n, 4):
            x, y = tips[:, k, 0], tips[:, k, 1]
            R = np.stack([x + 8.0, y - 34.0, x + 22.0 + text_w(self._ray_text(k), L.T_SMALL), y - 4.0], 1)
            ok = np.zeros(len(ts), bool)
            for i in np.nonzero(vis[:, k])[0]:
                r = R[i]
                if not lay.free(*r) or any(hit(r, o, 8.0) for o in taken[i]):
                    continue
                others = [rects[j][i] for j, (i0, i1) in plan.items() if i0 <= i <= i1]
                ok[i] = len(others) < RAY_TAG_MAX and not any(hit(r, o, 8.0) for o in others)
            i0 = n = i = 0
            while i < len(ok):                              # the longest stretch during which its way is clear
                j = i
                while j < len(ok) and ok[j]:
                    j += 1
                if j - i > n:
                    i0, n = i, j - i
                i = j + 1
            if (n - 1) * dt >= RAY_TAG_MIN:
                plan[k], rects[k] = (i0, i0 + n - 1), R
        return {k: (float(ts[i0]), float(ts[i1])) for k, (i0, i1) in plan.items()}

    def _ray_tags(self, f, cam, t):
        """Names on a few rays, where and when the wall is free (_plan_ray_tags): each is decoded when its way
        is clear and taken apart before it would meet something; none comes on written, none flickers."""
        nova, lay = self.nova, self.lay
        live = [k for k, (t0, t1) in self.ray_plan.items() if t0 <= t < t1]
        if not live:
            return
        Lr = nova.lengths(t)
        ax, ay, az, bx, by, bz, ok, tip_in = nova._clip_near(cam, np.zeros((nova.n, 3)), nova.dir * Lr[:, None])
        for k in live:
            t0, t1 = self.ray_plan[k]
            s = self._ray_text(k)
            txt = B.resolve(s, B.io(t - t0, t1 - t, out=0.3, span=0.6), 60.0, key=int(k))
            if not txt.strip():
                continue
            x, y = float(bx[k]), float(by[k])
            lay.take((x + 8.0, y - 34.0, x + 22.0 + text_w(s, L.T_SMALL), y - 4.0))
            f.occlude(x + 8.0, y - 34.0, x + 22.0 + text_w(txt, L.T_SMALL), y - 4.0)     # its plate, as far as it is written
            f.text("w", x + 14.0, y - 12.0, txt, size=L.T_SMALL, alpha=0.9)

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
        kk, frac, dt = hud.barcode_keys(t, 4.0, cols)
        ts = kk * dt
        dens = np.where(ts >= T_X, 0.08 + 0.9 * np.exp(-(ts - T_X) / 1.3), 0.02)
        hud.barcode_lanes(f, x0, x1, y0 + 12, y1, dens, kk, lanes=3, seed=4, frac=frac)

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
