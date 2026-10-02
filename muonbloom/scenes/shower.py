"""AIR SHOWER - "Through the atmosphere above this city. Through..."   Sheet 1.2, 01:07.0 - 01:12.4.

The liked air-shower study, ported to the wall canvas and put on the voice:
  01:07.55  "Through the atmosphere above this city."  the primary (red) meets the air: first interaction
            ORTHO_SIDE chart: altitude rules every km, the cascade feathers out, a red line tracks the front
  01:11.12  "Through..."                               the front lands: ORTHO_TOP, the footprint blooms over
                                                        the dot lattice of the city
  01:12.40  cut to YOU
One muon of this shower is tagged from its birth (MU- 0001): it is the one that goes through the body in
the next scene. In the plan view it lands exactly on ctx.focus, where the figure of YOU will stand.
When a second bay is wide enough, a second elevation (Z) of the same shower runs in it, without tags. HUD: longitudinal profile on the 0-16 km ruler (strip), particle stream (column),
time to ground, birth-rate barcode, particle counters (bottom band).

The towers stand in front of the wall for the whole show (dark before the detectors are revealed), and
their position / size is not known yet. So nothing here has a fixed x: the shower axis sits on
ctx.focus (the middle of the bay that hosts the one-centre compositions), the particle column takes
an edge column of ctx.cols if one is wide enough, the bottom blocks flow into ctx.slots_pre, and every
text / tag / callout is tested against the tower rectangles (`hidden`) and moved or dropped. Cascades,
rules, rings and long lines are allowed to pass behind the towers.

This module also holds the shower *world* (cascade model + drawing) and the layout helpers that the
DANCE and GLITCH scenes reuse.
"""
from __future__ import annotations

import math
import os
import pickle

import numpy as np

from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import OrthoCamera, hash01, smoothstep, text_w
from ..show import Scene

V_PRIMARY = 12.0                # km / s of the primary on screen
V_RAIN = 7.5
TOP = 27.0                      # km, where tracks are born
ALT_MAX = 16.0                  # km, top of the rulers
CACHE_VERSION = 7

K_E, K_G, K_H, K_MU, K_P, K_RAIN = 0, 1, 2, 3, 4, 5
K_NAME = ["E-", "GAMMA", "PI+", "MU-", "P", "MU+"]
#                 e      gamma  hadron  muon   primary rain
K_INT = np.array([0.36, 0.14, 0.95, 0.8, 1.5, 0.9], np.float32)
K_TAU = np.array([1.35, 0.8, 2.0, 1.6, 3.0, 0.28], np.float32)
K_W = np.array([1.0, 1.0, 1.5, 1.4, 2.6, 1.4], np.float32)
K_RED = np.array([0, 0, 0, 1, 1, 1], np.int8)
K_HEAD = np.array([1.2, 0.0, 2.4, 2.8, 4.4, 2.6], np.float32)
K_HEAD_I = np.array([0.5, 0.0, 1.3, 1.4, 1.6, 1.3], np.float32)

CITY = "39.103N 084.512W"                       # Cincinnati
COL_W = 390.0                                   # width of the particle stream column
VIEW_Y0, VIEW_Y1 = 232.0, 1196.0                # main view, between the header and the bottom band
X_MIN, X_MAX = L.COL_X0, L.COL_X1               # text stays inside the edge ticks


# ----------------------------------------------------------------------------
# layout helpers: everything is placed from the towers
# ----------------------------------------------------------------------------

def hidden(ctx, x0, y0, x1, y1, pad=12.0):
    """True if a rect is (partly) behind a tower, or right against one."""
    for tw in ctx.towers.values():
        if x1 > tw.x0 - pad and x0 < tw.x1 + pad and y1 > tw.top - pad and y0 < tw.bot:
            return True
    return False


def tower_boxes(ctx, pad=14.0):
    return [(tw.x0 - pad, tw.top - pad, tw.x1 + pad, tw.bot) for tw in ctx.towers.values()]


def inner_bays(ctx):
    """The bays between neighbouring towers, left to right."""
    tws = sorted(ctx.towers.values(), key=lambda t: t.x0)
    return [(a.x1, b.x0) for a, b in zip(tws[:-1], tws[1:])]


def col_of(ctx, x):
    """The usable text column (ctx.cols) that holds x, else the nearest one."""
    cols = ctx.cols
    if not cols:
        return (X_MIN, X_MAX)
    for c in cols:
        if c[0] <= x <= c[1]:
            return c
    return min(cols, key=lambda c: min(abs(c[0] - x), abs(c[1] - x)))


def tbox(x, y, s, size, anchor="ls", pad=0.0):
    """Bounding box of a line of text set at (x, y)."""
    w = text_w(s, size)
    x0 = x if anchor[0] == "l" else (x - w if anchor[0] == "r" else x - w / 2)
    return (x0 - pad, y - 0.95 * size - pad, x0 + w + pad, y + 0.3 * size + pad)


def put_text(f, ctx, layer, x, y, s, size=L.T_SMALL, alpha=1.0, anchor="ls", bold=False):
    """Text, only if no tower stands in front of it. Returns True if drawn."""
    if not s or hidden(ctx, *tbox(x, y, s, size, anchor)):
        return False
    f.text(layer, x, y, s, size=size, alpha=alpha, anchor=anchor, bold=bold)
    return True


def put_tag(f, ctx, layer, x, y, s, size=L.T_TAG, alpha=1.0, anchor="ls", pad=5, bold=False):
    if not s or hidden(ctx, *tbox(x, y, s, size, anchor, pad)):
        return False
    f.tag(layer, x, y, s, size=size, alpha=alpha, anchor=anchor, pad=pad, bold=bold)
    return True


def put_first(f, ctx, layer, places, y, s, tag=True, **kw):
    """Tag / text at the first of `places` = [(x, anchor), ...] that is not behind a tower."""
    for x, anchor in places:
        if (put_tag if tag else put_text)(f, ctx, layer, x, y, s, anchor=anchor, **kw):
            return True
    return False


def auto_callout(f, ctx, view, x, y, title, lines=(), red=False, age=9.0, alpha=1.0, avoid=(), prefer=(1, -1),
                 dx=56.0, dy=50.0, size=L.T_TAG, lsize=L.T_SMALL):
    """hud.callout on the side where its text is inside the view and clear of the towers (and of the `avoid`
    boxes). Falls back to the title alone, then to nothing. Returns the box it used, or None."""
    xmin, xmax = max(view[0] + 6, X_MIN), min(view[2] - 6, X_MAX)
    for ls in (tuple(lines), ()):
        w = max([text_w(title, size) + 12] + [text_w(s, lsize) for s in ls])
        for sx, sy in ((prefer[0], prefer[1]), (prefer[0], -prefer[1]), (-prefer[0], prefer[1]),
                       (-prefer[0], -prefer[1])):
            ex, ey = x + sx * dx, y + sy * dy
            tx = ex + sx * 44
            top = ey - 0.75 * size - 6
            bot = ey + 0.36 * size + len(ls) * lsize * 1.45 + 8 + 0.3 * lsize
            box = (tx - 6, top, tx + w + 6, bot) if sx > 0 else (tx - w - 6, top, tx + 6, bot)
            if box[0] < xmin or box[2] > xmax or box[1] < view[1] + 6 or box[3] > view[3] - 6:
                continue
            if hidden(ctx, *box) or any(box[0] < o[2] and o[0] < box[2] and box[1] < o[3] and o[1] < box[3]
                                        for o in avoid):
                continue
            hud.callout(f, x, y, sx * dx, sy * dy, title, ls, red=red, alpha=alpha, age=age, size=size, lsize=lsize)
            return box
        if not lines:
            break
    return None


def flow(panels, blocks, gap=36.0):
    """Lay blocks out in the free panels of the bottom band, in priority order.
    block = (name, minimum width, preferred width[, flex]). A block takes its preferred width in the
    tightest free panel that holds it, else what the tightest panel holding its minimum offers; a flex
    block takes all of the widest panel left. A block that fits nowhere is dropped. Returns {name: (x0, x1)}."""
    free = [[float(a), float(b)] for a, b in panels]
    out = {}
    for blk in blocks:
        name, w_min, w_pref = blk[:3]
        if len(blk) > 3 and blk[3]:
            ok = [iv for iv in free if iv[1] - iv[0] >= w_min]
            if not ok:
                continue
            iv = max(ok, key=lambda v: v[1] - v[0])
            w = iv[1] - iv[0]
        else:
            fit = [iv for iv in free if iv[1] - iv[0] >= w_pref]
            ok = fit or [iv for iv in free if iv[1] - iv[0] >= w_min]
            if not ok:
                continue
            iv = min(ok, key=lambda v: v[1] - v[0])
            w = min(w_pref, iv[1] - iv[0])
        out[name] = (iv[0], iv[0] + w)
        iv[0] += w + gap
    return out


def chord(cx, cy, r, y):
    """x-interval a circle covers at height y (None if it does not reach it)."""
    d = abs(y - cy)
    if d >= r:
        return None
    h = math.sqrt(r * r - d * d)
    return (cx - h, cx + h)


def view_bays(ctx, view, min_w=420.0):
    """Bays for a pair of views inside `view`, left to right, and the x where the wall is split between them:
    the two widest free bays; else one very wide bay shared in two halves; else a single bay (split = None)."""
    cl = [(max(a, view[0]), min(b, view[2])) for a, b in ctx.bays]
    cl = [b for b in cl if b[1] - b[0] > 1.0]
    ok = [b for b in cl if b[1] - b[0] >= min_w]
    if len(ok) >= 2:
        two = sorted(sorted(ok, key=lambda b: b[0] - b[1])[:2])
        return two, 0.5 * (two[0][1] + two[1][0])
    widest = max(cl, key=lambda b: b[1] - b[0]) if cl else (view[0], view[2])
    if widest[1] - widest[0] >= 2.1 * min_w:
        m = 0.5 * (widest[0] + widest[1])
        return [(widest[0], m), (m, widest[1])], m
    return [widest], None


def stage_for(ctx, x, room=380.0):
    """A stage whose particle column leaves `room` px on both sides of the axis at x: column on the left,
    else on the right, else no column."""
    st = None
    for side in ("L", "R", None):
        st = Stage(ctx, side)
        if st.col is None or min(x - st.view[0], st.view[2] - x) >= room:
            break
    return st


def bottom_panels(ctx, pre=False):
    """Free panels of the bottom band, left to right (pre = before the detectors are revealed: no scopes).
    The counter cell is kept clear where the show really draws it (ctx.slots['cell'], also before the reveal)."""
    sl = ctx.slots_pre if pre else ctx.slots
    cx0, _, cx1, _ = ctx.slots["cell"]
    out = []
    for a, b in sl["panels"]:
        if cx1 - cx0 > 1.0 and a < cx1 + 26.0 and cx0 - 26.0 < b:
            if cx0 - 26.0 - a > 120.0:
                out.append((a, cx0 - 26.0))
            if b - (cx1 + 26.0) > 120.0:
                out.append((cx1 + 26.0, b))
        else:
            out.append((a, b))
    return out


def zone(ctx, x, hw, bay=None, snap=100.0):
    """x-interval a cascade occupies around its axis: clipped to its bay, and snapped onto the edges of the
    bay when it nearly fills it (so that texts line up on the towers)."""
    a, b = x - hw, x + hw
    if bay is None:
        bay = next(((b0, b1) for b0, b1 in ctx.bays if b0 <= x <= b1), None)
    if bay:
        a, b = max(a, bay[0]), min(b, bay[1])
        if a - bay[0] < snap:
            a = bay[0]
        if bay[1] - b < snap:
            b = bay[1]
    return (a, b)


def _hits(box, boxes, pad=8.0, pady=5.0):
    return any(box[0] < o[2] + pad and o[0] - pad < box[2] and box[1] < o[3] + pady and o[1] - pady < box[3]
               for o in boxes)


def _free(ctx, st, box, zones=(), taken=(), pad=16.0):
    """True if a text box is inside the view, clear of the towers, of the cascade zones and of other boxes."""
    x0, y0, x1, y1 = box
    if x0 < st.tx0 - 8 or x1 > st.tx1 + 8:
        return False
    if hidden(ctx, x0, y0, x1, y1, pad=pad):
        return False
    if any(x0 < b and a < x1 for a, b in zones):
        return False
    return not _hits(box, taken, pad)


def anchors(ctx, st, side):
    """Candidate x anchors of a text block: the edge of the view, then the column edges at the towers.
    'l' = left edges, from the left; 'r' = right edges, from the right."""
    tws = sorted(ctx.towers.values(), key=lambda t: t.x0)
    if side == "l":
        return [st.tx0] + [t.x1 + 28.0 for t in tws if st.tx0 < t.x1 + 28.0 < st.tx1]
    return [st.tx1] + [t.x0 - 28.0 for t in reversed(tws) if st.tx0 < t.x0 - 28.0 < st.tx1]


def info_layout(ctx, st, label, cam, id_lines=(), id_short=(), zones_top=(), zones_full=(), within=None,
                unit="KM", ratio=1e5):
    """Where the fixed texts of a view go for this tower placement and this view:
         a      the view tag (+ scale bar under it), top left
         b      the scene id lines, top right: the full version, else the short one, else nothing
         xr     the right-hand column of the chart (slant depth labels, FRONT / GROUND tags), full height
    zones_top / zones_full = x-intervals taken by the cascades at the top of the view / over its whole height.
    Every candidate is an edge of the view or of a tower, so the blocks line up with the furniture."""
    y0 = st.view[1] + 44
    lay = dict(y0=y0, label=label, cam=cam, a=None, scale=None, note=None, b=None, xr=None, boxes=[],
               zones_top=tuple(zones_top), zones_full=tuple(zones_full))
    wA = text_w(label, L.T_LABEL) + 12
    for zs in (zones_top, ()):
        for x in anchors(ctx, st, "l"):
            box = (x - 6, y0 - 28, x + wA, y0 + 12)
            if _free(ctx, st, box, zs):
                lay["a"] = x
                lay["boxes"].append(box)
                break
        if lay["a"] is not None:
            break
    for lines in (id_lines, id_short):
        if not lines or lay["b"]:
            continue
        w = max(text_w(ln, L.T_SMALL) for ln, _ in lines)
        for x in anchors(ctx, st, "r"):
            box = (x - w - 6, y0 - 20, x + 6, y0 + (len(lines) - 1) * 26 + 8)
            if _free(ctx, st, box, zones_top, lay["boxes"]):
                lay["b"] = (x, lines)
                lay["boxes"].append(box)
                break
    if lay["a"] is not None:
        x = lay["a"]
        others = lay["boxes"][1:]
        if getattr(cam, "ortho", False):
            sc = cam.scale
            cands = [n for n in (5, 2, 1) if n * sc <= 320.0] or [1]
            for zs, ns in ((zones_top, cands), ((), cands[-1:])):     # the longest short bar that fits
                for n in ns:
                    note = f"{n} {unit}   1:{int(ratio / sc):d}"
                    box = (x - 2, y0 + 22, x + n * sc + 14 + text_w(note, L.T_SMALL), y0 + 52)
                    if _free(ctx, st, box, zs, others):
                        lay["scale"] = (n, note)
                        lay["boxes"].append(box)
                        break
                if lay["scale"]:
                    break
        else:
            note = f"F {getattr(cam, 'fov', 44.0):.1f} DEG   CAM ORBIT"
            box = (x - 2, y0 + 22, x + text_w(note, L.T_SMALL), y0 + 48)
            if _free(ctx, st, box, (), others):
                lay["note"] = note
                lay["boxes"].append(box)
    v = st.view
    for x in anchors(ctx, st, "r"):
        if within is not None and not (within[0] + 200 <= x <= within[1]):
            continue
        if _free(ctx, st, (x - 200, v[1] + 120, x, v[3] - 10), zones_full):
            lay["xr"] = x
            break
    return lay


def draw_info(f, ctx, lay, alpha=1.0):
    y0 = lay["y0"]
    if lay["a"] is not None:
        x = lay["a"]
        f.tag("w", x, y0, lay["label"], size=L.T_LABEL, pad=5, alpha=alpha)
        if lay["scale"]:
            n, note = lay["scale"]
            sc = lay["cam"].scale
            f.segments("w", [x, x, x + n * sc], [y0 + 38, y0 + 31, y0 + 31], [x + n * sc, x, x + n * sc],
                       [y0 + 38, y0 + 45, y0 + 45], 0.9 * alpha)
            step = 1.0 if n > 2 else 0.25
            for k in np.arange(step, n - 1e-6, step):
                major = abs(k - round(k)) < 1e-6
                f.segments("w", [x + k * sc], [y0 + 38], [x + k * sc], [y0 + (29 if major else 33)], 0.9 * alpha)
            f.text("w", x + n * sc + 12, y0 + 44, note, size=L.T_SMALL, alpha=0.8 * alpha)
        elif lay["note"]:
            f.text("w", x, y0 + 40, lay["note"], size=L.T_SMALL, alpha=0.7 * alpha)
    if lay["b"]:
        x, lines = lay["b"]
        for k, (ln, al) in enumerate(lines):
            f.text("w", x, y0 + k * 26, ln, size=L.T_SMALL, alpha=al * alpha, anchor="rs")


def put_right(f, ctx, st, lay, layer, y, s, size=L.T_SMALL, pad=4, alpha=1.0):
    """A tag in the right-hand column of the chart (lay['xr']); when the placement leaves no such column,
    at the first right anchor that is free at this height; else it is dropped."""
    if lay["xr"] is not None:
        xs, check = [lay["xr"]], False
    else:
        xs, check = anchors(ctx, st, "r"), True
    for x in xs:
        box = tbox(x, y, s, size, "rs", pad)
        if hidden(ctx, *box) or (check and not _free(ctx, st, box, lay["zones_full"], lay["boxes"])):
            continue
        f.tag(layer, x, y, s, size=size, alpha=alpha, anchor="rs", pad=pad)
        return True
    return False


class Stage:
    """Where things go for a tower placement: the particle column (an edge column of ctx.cols, on the
    preferred side if it is wide enough, else on the other side, else none), and the main view next to it.
    side = None: no column at all, the view takes the whole width."""

    def __init__(self, ctx, side="L"):
        cols = ctx.cols
        left = cols[0] if cols and cols[0][0] <= X_MIN + 1.0 else None
        right = cols[-1] if cols and cols[-1][1] >= X_MAX - 1.0 else None
        self.col, self.side = None, None
        for name in ((side, "L" if side == "R" else "R") if side else ()):
            c = left if name == "L" else right
            if c is not None and c[1] - c[0] >= COL_W - 1.0:
                self.side = name
                self.col = ((c[0], 240.0, c[0] + COL_W, 1190.0) if name == "L" else
                            (c[1] - COL_W, 240.0, c[1], 1190.0))
                break
        x0, x1 = L.FX0 + 12.0, L.FX1 - 12.0
        if self.side == "L":
            x0 = self.col[2] + 16.0
        elif self.side == "R":
            x1 = self.col[0] - 16.0
        self.view = (x0, VIEW_Y0, x1, VIEW_Y1)
        self.tx0 = max(x0 + 24.0, X_MIN)            # text limits inside the view
        self.tx1 = min(x1 - 24.0, X_MAX)
        self.cy = 0.5 * (VIEW_Y0 + VIEW_Y1)
        self.inner = inner_bays(ctx)
        self.focus_col = col_of(ctx, ctx.focus[0])


# ----------------------------------------------------------------------------
# cascade model
# ----------------------------------------------------------------------------

def _perp_basis(d):
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def _deflect(d, theta, rng):
    u, w = _perp_basis(d)
    phi = rng.uniform(0, 2 * math.pi)
    nd = math.cos(theta) * d + math.sin(theta) * (math.cos(phi) * u + math.sin(phi) * w)
    return nd / np.linalg.norm(nd)


class _Cascade:
    def __init__(self, rng, max_branches, speed):
        self.rng = rng
        self.max_branches = max_branches
        self.speed = speed
        self.branches = []      # (pts (k,3), times (k,), kind, energy)

    def add(self, p0, d, length, t0, kind, e, curv=0.0, speed=None):
        rng = self.rng
        speed = speed or self.speed
        k = int(np.clip(math.ceil(length * 5), 2, 10)) if curv else 2
        s = np.linspace(0.0, length, k)
        pts = p0[None, :] + d[None, :] * s[:, None]
        tang = d.copy()
        if curv:
            u, w = _perp_basis(d)
            phi = rng.uniform(0, 2 * math.pi)
            n = math.cos(phi) * u + math.sin(phi) * w
            pts = pts + n[None, :] * (0.5 * curv * s * s)[:, None]
            tang = d + n * curv * length
            tang /= np.linalg.norm(tang)
        hit = False
        if pts[-1, 1] < 0.0:
            below = np.nonzero(pts[:, 1] < 0.0)[0][0]
            a, b = pts[below - 1], pts[below]
            f = a[1] / (a[1] - b[1])
            end = a + (b - a) * f
            pts = np.vstack([pts[:below], end[None, :]])
            s = np.append(s[:below], s[below - 1] + (s[below] - s[below - 1]) * f)
            hit = True
        times = t0 + s / speed
        self.branches.append((pts.astype(np.float32), times.astype(np.float32), kind, e))
        return pts[-1], tang, times[-1], hit

    def full(self):
        return len(self.branches) >= self.max_branches


def _lam_h(h):
    return 0.85 * math.exp(max(h, 0.0) / 9.0)


def _lam_e(h):
    return 0.30 * math.exp(max(h, 0.0) / 8.4)


def build_shower(rng, ground, h1, zen_deg, az_deg, E0, speed, Ec=1.4e-4, max_branches=9000):
    """Primary -> hadronic interactions -> pi0 -> photons -> e+e- / bremsstrahlung sub-showers, and
    pi+- -> muons (long straight tracks to the ground). Times are relative to the birth of the primary."""
    zen, az = math.radians(zen_deg), math.radians(az_deg)
    d0 = np.array([math.sin(zen) * math.cos(az), -math.cos(zen), math.sin(zen) * math.sin(az)])
    G = np.array([ground[0], 0.0, ground[1]])
    P1 = G - d0 * (h1 / math.cos(zen))
    Ptop = G - d0 * (TOP / math.cos(zen))
    c = _Cascade(rng, max_branches, speed)
    _, _, t1, _ = c.add(Ptop, d0, float(np.linalg.norm(P1 - Ptop)), 0.0, K_P, E0, speed=V_PRIMARY)
    info = dict(P1=P1, t1=t1, d0=d0, E0=E0, zen=zen_deg, h1=h1, G=G, t_ground=t1 + (h1 / math.cos(zen)) / speed)
    stack = [(P1, d0, E0, K_H, t1, 0)]
    while stack:
        p, d, e, kind, t, gen = stack.pop()
        if kind == K_H:
            Ln = _lam_h(p[1]) * rng.exponential(1.0) * (0.4 if gen == 0 else 1.0)
            end, tang, tend, hit = c.add(p, d, Ln, t, K_H, e)
            if hit:
                continue
            m = int(3 + rng.poisson(2.5 + 3.0 * e ** 0.3))
            frac = rng.dirichlet(np.ones(m) * 0.7) * e * 0.55
            lead = e * 0.45
            if lead > 0.012:
                stack.append((end, _deflect(tang, 0.01 * rng.exponential(), rng), lead, K_H, tend, gen + 1))
            for ei in frac:
                th = min(0.9, 0.03 / math.sqrt(max(ei, 1e-6)) * (0.35 + rng.exponential()))
                nd = _deflect(tang, th, rng)
                if rng.random() < 0.66:
                    if ei > 0.05 and not c.full():
                        stack.append((end, nd, ei, K_H, tend, gen + 1))
                    else:
                        ld = rng.exponential(0.35 + 4.0 * ei)
                        pe, ptang, pt, phit = c.add(end, nd, ld, tend, K_H, ei)
                        if not phit:
                            md = _deflect(ptang, 0.02 * rng.exponential(), rng)
                            if float(np.dot(md, d0)) < math.cos(0.36):
                                md = _deflect(d0, rng.uniform(0.05, 0.36), rng)
                            c.add(pe, md, 60.0, pt, K_MU, ei * 0.8)
                else:
                    share = 0.5 + 0.3 * (rng.random() - 0.5)
                    for ef in (ei * share, ei * (1 - share)):
                        stack.append((end, _deflect(nd, 0.02 * rng.exponential(), rng), ef, K_G, tend, gen + 1))
        else:
            lam = _lam_e(p[1])
            if e < Ec or c.full():
                Ln = min(lam * rng.exponential(0.55), 1.3)
                c.add(p, d, Ln, t, kind, e, curv=0.9 * rng.exponential() if kind == K_E else 0.0)
                continue
            Ln = min(lam * rng.exponential(0.75), 2.2)
            curv = min(1.2, 0.35 * (Ec / e) ** 0.6) if kind == K_E else 0.0
            end, tang, tend, hit = c.add(p, d, Ln, t, kind, e, curv=curv)
            if hit:
                continue
            u = rng.uniform(0.12, 0.88)
            kids = (K_E, K_E) if kind == K_G else (K_E, K_G)
            for kk, ef in zip(kids, (e * u, e * (1 - u))):
                th = min(1.2, 0.42 * math.sqrt(Ec / max(ef, 1e-9)) * (0.25 + rng.exponential()))
                stack.append((end, _deflect(tang, th, rng), ef, kk, tend, gen + 1))
    return c.branches, info


# ----------------------------------------------------------------------------
# world: a list of showers (+ single 'rain' muons) on absolute show time
# ----------------------------------------------------------------------------

class World:
    """showers: dicts(t_int, ground=(x, z), E0, zen, az, h1, travel[, aim=detector index, you=(x, z)])
       rain:    dicts(t_land, target=(x, alt, z)[, det=index])   one straight muon landing at t_land
       dets:    [(x, alt, z)] world position of the detector heads (muons aimed at one pass through it)
    All geometry in km, y up, ground at y = 0."""

    def __init__(self, showers, rain=(), dets=(), seed=4, cache=None):
        self.dets = [tuple(map(float, d)) for d in dets]
        path = self._cache_path(cache)
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    self.__dict__.update(pickle.load(fh))
                return
            except Exception:
                pass
        self._build(showers, rain, seed)
        if path:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "wb") as fh:
                    pickle.dump(self.__dict__, fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
            except Exception:
                pass

    @staticmethod
    def _cache_path(cache):
        if not cache:
            return None
        return str(L.DATA / "cache" / f"shower_world_{cache}_v{CACHE_VERSION}.pkl")

    # ---------------------------------------------------------------- build
    def _build(self, showers, rain, seed):
        self.events = []
        B = []
        self.n_showers = len(showers)
        self.crossings = []                 # (event, detector, age at the crossing, energy)
        self.you = None                     # the tagged muon of a shower: dict(ev, a, b, t0, t1, E)
        for k, sp in enumerate(showers):
            rng = np.random.default_rng(seed * 1000 + 17 * k)
            zen = sp["zen"]
            speed = (sp["h1"] / math.cos(math.radians(zen))) / sp["travel"]
            br, info = build_shower(rng, sp["ground"], sp["h1"], zen, sp["az"], sp["E0"], speed,
                                    max_branches=int(9000 + 14000 * sp["E0"]))
            aim = sp.get("aim")
            if aim is not None and self.dets:
                br = self._aim_muons(br, aim, rng, speed, k, count=sp.get("n_aim", 3))
            if sp.get("you") is not None:
                br = self._tag_muon(br, sp["you"], rng, speed, k)
            info.update(t0=sp["t_int"] - info["t1"], kind="shower", aim=aim, speed=speed, index=k)
            self.events.append(info)
            B.append(br)
        rng = np.random.default_rng(seed * 1000 + 999)
        for rp in rain:
            x, alt, z = rp["target"]
            zen = math.radians(abs(rng.normal(0, 9 if rp.get("det") is not None else 14)))
            az = rng.uniform(0, 2 * math.pi)
            d = np.array([math.sin(zen) * math.cos(az), -math.cos(zen), math.sin(zen) * math.sin(az)])
            Pd = np.array([x, alt, z])
            start = Pd - d * ((TOP - alt) / math.cos(zen))
            G = Pd + d * (alt / math.cos(zen))
            t_det = float(np.linalg.norm(Pd - start)) / V_RAIN
            t_gnd = t_det + float(np.linalg.norm(G - Pd)) / V_RAIN
            ev = len(self.events)
            self.events.append(dict(t0=rp["t_land"] - t_det, kind="rain"))
            if alt > 1e-3:
                pts = np.stack([start, Pd, G]).astype(np.float32)
                tt = np.array([0.0, t_det, t_gnd], np.float32)
                if rp.get("det") is not None:
                    self.crossings.append((ev, int(rp["det"]), t_det, 0.3))
            else:
                pts = np.stack([start, G]).astype(np.float32)
                tt = np.array([0.0, t_gnd], np.float32)
            B.append([(pts, tt, K_RAIN, 0.3)])
        self._pack(B)
        self._build_static()

    def _aim_muons(self, branches, det, rng, speed, ev, count=3):
        """Send the muons that land closest to a detector straight through its head."""
        dx, dalt, dz = self.dets[det]
        mu = [i for i, b in enumerate(branches) if b[2] == K_MU and b[0][-1, 1] <= 1e-3 and b[0][0, 1] > dalt + 1.5]
        if not mu:
            return branches
        dist = [math.hypot(branches[i][0][-1, 0] - dx, branches[i][0][-1, 2] - dz) for i in mu]
        out = list(branches)
        for n, i in enumerate([mu[i] for i in np.argsort(dist)[:count]]):
            pts, times, kind, e = branches[i]
            start = pts[0].astype(np.float64)
            off = 0.0 if n == 0 else 0.22
            Pd = np.array([dx + rng.uniform(-off, off), dalt, dz + rng.uniform(-off, off)])
            d = (Pd - start) / np.linalg.norm(Pd - start)
            G = Pd + d * (dalt / max(-d[1], 1e-3))
            t_det = float(times[0]) + float(np.linalg.norm(Pd - start)) / speed
            t_gnd = t_det + float(np.linalg.norm(G - Pd)) / speed
            out[i] = (np.stack([start, Pd, G]).astype(np.float32), np.array([times[0], t_det, t_gnd], np.float32),
                      kind, e)
            self.crossings.append((ev, det, t_det, float(e)))
        return out

    def _tag_muon(self, branches, target, rng, speed, ev):
        """One muon of the shower is THE muon: born high, it lands on `target` with the front."""
        mu = [i for i, b in enumerate(branches) if b[2] == K_MU and b[0][-1, 1] <= 1e-3]
        if not mu:
            return branches
        G = np.array([target[0], 0.0, target[1]])
        best = max(mu, key=lambda i: branches[i][0][0, 1] - 0.6 * math.hypot(branches[i][0][-1, 0] - G[0],
                                                                         branches[i][0][-1, 2] - G[2]))
        pts, times, kind, e = branches[best]
        start = pts[0].astype(np.float64)
        Ln = float(np.linalg.norm(G - start))
        out = list(branches)
        out[best] = (np.stack([start, G]).astype(np.float32), np.array([times[0], times[0] + Ln / speed], np.float32),
                     kind, e)
        self.you = dict(ev=ev, a=start.astype(np.float32), b=G.astype(np.float32), t0=float(times[0]),
                        t1=float(times[0] + Ln / speed), E=4.213)
        return out

    def _pack(self, B):
        V_, VT, SE, SK, SA, EN = [], [], [], [], [], []
        births, ranges = [], []
        off = seg = 0
        for ev, brs in enumerate(B):
            bt, bk, be, bp = [], [], [], []
            s0 = seg
            for pts, times, kind, e in brs:
                m = len(pts)
                V_.append(pts)
                VT.append(times)
                SA.append(np.arange(off, off + m - 1, dtype=np.int64))
                SE.append(np.full(m - 1, ev, np.int32))
                SK.append(np.full(m - 1, kind, np.int8))
                EN.append(np.full(m - 1, e, np.float32))
                off += m
                seg += m - 1
                bt.append(times[0]); bk.append(kind); be.append(e); bp.append(pts[0])
            ranges.append((s0, seg))
            o = np.argsort(bt)
            births.append(dict(t=np.asarray(bt, np.float32)[o], k=np.asarray(bk, np.int8)[o],
                               e=np.asarray(be, np.float32)[o], p=np.asarray(bp, np.float32)[o]))
        self.births = births
        self.ranges = ranges
        self.V = np.concatenate(V_).astype(np.float32)
        self.VT = np.concatenate(VT).astype(np.float32)
        self.SA = np.concatenate(SA)
        self.SE = np.concatenate(SE)
        self.SK = np.concatenate(SK)
        self.SEn = np.concatenate(EN)
        n_ev = len(B)
        self.ev_t0 = np.array([e["t0"] for e in self.events], np.float64)
        last = np.zeros(n_ev, np.float32)
        np.maximum.at(last, self.SE, self.VT[self.SA + 1])
        self.ev_life = (last + 4.5).astype(np.float64)
        rng = np.random.default_rng(99)
        self.SVar = rng.uniform(0.6, 1.35, len(self.SA)).astype(np.float32)
        self.SGain = np.where(self.SK <= K_G,
                              np.clip(0.75 + 0.12 * np.log10(np.maximum(self.SEn, 1e-6) / 1.4e-4), 0.7, 1.5),
                              1.0).astype(np.float32)
        # ground hits
        hits = []
        for ev, brs in enumerate(B):
            for pts, times, kind, e in brs:
                if kind not in (K_MU, K_H, K_RAIN) or pts[-1, 1] > 1e-3:
                    continue
                hits.append((ev, times[-1], pts[-1, 0], pts[-1, 2], K_MU if kind == K_RAIN else kind))
        self.hits = np.array(hits, np.float32).reshape(-1, 5)
        # floating labels
        labels = []
        rng = np.random.default_rng(17)
        for ev, brs in enumerate(B):
            if self.events[ev]["kind"] != "shower":
                continue
            cand = [(pts[-1], times[-1], kind, e) for pts, times, kind, e in brs
                    if kind in (K_E, K_H, K_MU) and pts[-1, 1] > 0.6]
            pick = rng.choice(len(cand), size=min(44, len(cand)), replace=False)
            for i in pick:
                p, tt, kind, e = cand[i]
                if kind == K_H:
                    word = "PI0" if rng.random() < 0.3 else "PI+"
                elif kind == K_MU:
                    word = "MU+" if rng.random() < 0.55 else "MU-"
                else:
                    word = None
                num = f"{rng.uniform(0.0005, 0.02):.10f}"
                labels.append((ev, float(tt), p.astype(np.float32), word, num, kind == K_MU))
        self.labels = labels

    def _build_static(self):
        g = np.arange(-30.0, 30.01, 0.5, dtype=np.float32)
        X, Z = np.meshgrid(g, g)
        self.lattice = np.stack([X.ravel(), np.zeros(X.size, np.float32), Z.ravel()], 1)
        self.lat_major = ((np.abs(X) % 4 < 1e-3) & (np.abs(Z) % 4 < 1e-3)).ravel()

    # ---------------------------------------------------------------- state
    def state(self, t):
        age = t - self.ev_t0
        alive = (age >= 0.0) & (age < self.ev_life)
        env = 1.0 - smoothstep(self.ev_life - 3.2, self.ev_life, age)
        return age, alive, np.asarray(env, np.float64)

    def front(self, k, a):
        """Altitude (km) of the front of shower k at age a (the primary before the first interaction)."""
        e = self.events[k]
        c = math.cos(math.radians(e["zen"]))
        if a < e["t1"]:
            return e["h1"] + (e["t1"] - a) * V_PRIMARY * c
        return max(0.0, e["h1"] - (a - e["t1"]) * e["speed"] * c)

    @staticmethod
    def fog(z):
        return np.clip(1.35 - z / 70.0, 0.25, 1.0).astype(np.float32)

    # ---------------------------------------------------------------- drawing
    def draw_ground(self, f, cam, kind, view, gain=1.0):
        """kind: 'persp' / 'top' = dot lattice + red baseline, 'side' / 'front' = ground line with km ticks."""
        if kind in ("side", "front"):
            xs = np.arange(-34.0, 34.01, 1.0, dtype=np.float32)
            P = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
            ax, ay, _, _ = cam.project(P)
            f.segments("w", [view[0]], [ay[0]], [view[2]], [ay[0]], 0.9 * gain, width=L.LW)
            f.segments("w", ax, ay, ax, ay - np.where(np.arange(len(xs)) % 5 == 0, 16, 7), 0.7 * gain)
            return
        sx, sy, z, ok = cam.project(self.lattice)
        inten = np.where(self.lat_major, 0.95, 0.5) * self.fog(z) * gain
        if kind == "top":
            f.pixels("w", sx[ok], sy[ok], inten[ok] * 1.25)
            mj = self.lat_major & ok
            f.crosses("w", sx[mj], sy[mj], 6.0, 0.6 * gain)
        else:
            f.pixels("w", sx[ok], sy[ok], inten[ok])
        xs = np.linspace(-30, 30, 61, dtype=np.float32)
        P = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
        lx, ly, lz, lok = cam.project(P)
        m = lok[:-1] & lok[1:]
        f.segments("r", lx[:-1][m], ly[:-1][m], lx[1:][m], ly[1:][m], 0.5 * self.fog(lz[:-1][m]) * gain)

    def draw_cascades(self, f, cam, age, alive, env, gain=1.0):
        live = np.nonzero(alive)[0]
        if not len(live):
            return
        idx = np.concatenate([np.arange(*self.ranges[e]) for e in live])
        sa, ev = self.SA[idx], self.SE[idx]
        G = age[ev].astype(np.float32)
        ta = self.VT[sa]
        vis = G >= ta
        if not vis.any():
            return
        idx, sa, ev, G, ta = idx[vis], sa[vis], ev[vis], G[vis], ta[vis]
        kind = self.SK[idx]
        var = self.SVar[idx] * self.SGain[idx]
        tb = self.VT[sa + 1]
        pa = self.V[sa]
        pb = self.V[sa + 1].copy()
        part = G < tb
        frac = np.clip((G[part] - ta[part]) / np.maximum(tb[part] - ta[part], 1e-6), 0, 1)
        pb[part] = pa[part] + (pb[part] - pa[part]) * frac[:, None]
        tau = K_TAU[kind]
        base = K_INT[kind] * var * env[ev].astype(np.float32) * gain
        flash = np.where(kind <= K_G, 0.5, 1.3).astype(np.float32)

        def inten(a):
            return base * (0.72 * np.exp(-a / tau) + 0.28 * np.exp(-a / (4.5 * tau)) + flash * np.exp(-a / 0.06))

        ia, ib = inten(G - ta), inten(np.where(part, 0.0, G - tb))
        ax, ay, az, aok = cam.project(pa)
        bx, by, bz, bok = cam.project(pb)
        ok = aok & bok
        ia = ia * self.fog(az)
        ib = ib * self.fog(bz)
        red = K_RED[kind] == 1
        width = K_W[kind]
        for m, name in ((ok & ~red, "w"), (ok & red, "r")):
            if m.any():
                f.segments(name, ax[m], ay[m], bx[m], by[m], ia[m], ib[m], width=width[m])
        hm = part & ok & (K_HEAD[kind] > 0)
        if hm.any():
            hk = kind[hm]
            fog = self.fog(bz[hm])
            hr = K_HEAD[hk] * np.clip(40.0 / np.maximum(bz[hm], 1.0), 0.6, 1.6)
            hred = K_RED[hk] == 1
            hi = K_HEAD_I[hk] * fog * env[ev[hm]].astype(np.float32) * gain
            if (~hred).any():
                f.dots("w", bx[hm][~hred], by[hm][~hred], hr[~hred], hi[~hred])
            if hred.any():
                f.dots("r", bx[hm][hred], by[hm][hred], hr[hred], hi[hred])
                f.dots("w", bx[hm][hred], by[hm][hred], hr[hred] * 0.4, 0.8 * hi[hred])

    def draw_hits(self, f, cam, age, alive, gain=1.0):
        """Rings where muons / hadrons reach the ground."""
        if not len(self.hits):
            return
        ev = self.hits[:, 0].astype(int)
        a = age[ev] - self.hits[:, 1]
        m = alive[ev] & (a >= 0) & (a < 1.6)
        if not m.any():
            return
        h, a = self.hits[m], a[m].astype(np.float32)
        u = a / 1.6
        mu = h[:, 4] == K_MU
        r = np.where(mu, 0.12 + 0.75 * (1 - (1 - u) ** 3), 0.1 + 0.4 * (1 - (1 - u) ** 3))
        n = 28
        ang = np.linspace(0, 2 * np.pi, n + 1, dtype=np.float32)
        P = np.stack([h[:, 2:3] + r[:, None] * np.cos(ang)[None], np.zeros((len(h), n + 1), np.float32),
                      h[:, 3:4] + r[:, None] * np.sin(ang)[None]], -1)
        sx, sy, z, ok = cam.project(P.reshape(-1, 3))
        sx, sy, z, ok = (v.reshape(len(h), n + 1) for v in (sx, sy, z, ok))
        inten = ((1 - u) ** 2.4 * np.where(mu, 0.85, 0.5))[:, None] * self.fog(z) * gain
        for sel, name in ((mu, "r"), (~mu, "w")):
            if sel.any():
                okm = ok[sel][:, :-1] & ok[sel][:, 1:]
                f.segments(name, sx[sel][:, :-1][okm], sy[sel][:, :-1][okm], sx[sel][:, 1:][okm],
                           sy[sel][:, 1:][okm], inten[sel][:, :-1][okm], width=L.LW)
        cx, cy, cz, cok = cam.project(np.stack([h[:, 2], np.zeros(len(h)), h[:, 3]], 1))
        f.dots("w", cx[cok], cy[cok], 2.4, (1.4 * (1 - u) ** 3)[cok] * gain)

    def draw_splash(self, f, cam, age, gain=1.0):
        """The front reaches the ground: two rings of dots run out over the lattice."""
        for k in range(self.n_showers):
            e = self.events[k]
            a = age[k] - e["t_ground"]
            if not (0 <= a < 2.6):
                continue
            u = a / 2.6
            for j, (rmax, lay, rad) in enumerate(((11.0, "r", 2.5), (6.5, "w", 1.8))):
                r = 0.3 + rmax * (1 - (1 - u) ** 2.2)
                n = int(90 + 26 * r)
                ang = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float32) + j * 0.05
                P = np.stack([e["G"][0] + r * np.cos(ang), np.zeros(n, np.float32), e["G"][2] + r * np.sin(ang)], 1)
                sx, sy, z, ok = cam.project(P.astype(np.float32))
                f.dots(lay, sx[ok], sy[ok], rad * np.clip(36.0 / z[ok], 0.5, 1.6),
                       (1.1 * (1 - u) ** 1.4) * self.fog(z[ok]) * gain)

    def draw_interaction(self, f, cam, age, view, ctx=None, tags=True, avoid=()):
        """First interaction of the primary: rays, a red ring, a red cross-hair, and (tags) its data on the
        side that is clear of the towers and of the `avoid` boxes. Returns the boxes used by the text."""
        rng = np.random.default_rng(3)
        boxes = []
        for k in range(self.n_showers):
            e = self.events[k]
            a = age[k] - e["t1"]
            if not (0 <= a < 1.9):
                continue
            px, py, pz, ok = cam.project(e["P1"][None].astype(np.float32))
            if not ok[0]:
                continue
            cx, cy = float(px[0]), float(py[0])
            u = a / 1.9
            n = 72
            ang = rng.uniform(0, 2 * np.pi, n)
            ln = rng.uniform(60, 280, n) * (1 - (1 - min(1, a / 0.4)) ** 3)
            r0 = 16 + 40 * u
            fade = (1 - u) ** 1.6
            f.segments("w", cx + np.cos(ang) * r0, cy + np.sin(ang) * r0, cx + np.cos(ang) * (r0 + ln),
                       cy + np.sin(ang) * (r0 + ln), 0.6 * fade)
            f.dots("w", cx + np.cos(ang) * (r0 + ln), cy + np.sin(ang) * (r0 + ln), 3.0, 1.2 * fade)
            f.rings("r", [cx], [cy], [20 + 230 * (1 - (1 - u) ** 2)], fade, width=L.LW_BOLD)
            big = 1e5
            f.segments("r", [cx - big, cx], [cy, cy - big], [cx + big, cx], [cy, cy + big], 0.55 * fade)
            if not tags or ctx is None:
                continue
            title = "FIRST_INTERACTION"
            lines = [f"H {e['h1']:.3f} KM", f"E0 {e['E0'] * 3.2:.2f}E15 EV"]
            w = max(text_w(title, L.T_TAG) + 12, max(text_w(s, L.T_SMALL) for s in lines))
            xmin, xmax = max(view[0] + 6, X_MIN), min(view[2] - 6, X_MAX)
            for sgn in (1, -1):
                bx0 = cx + 36 if sgn > 0 else cx - 36 - w
                box = (bx0 - 6, cy - 58, bx0 + w + 6, cy + 40)
                if box[0] < xmin or box[2] > xmax or box[1] < view[1] + 4 or box[3] > view[3] - 4 or hidden(ctx, *box):
                    continue
                if _hits(box, avoid):
                    continue
                anchor = "ls" if sgn > 0 else "rs"
                f.tag("r", cx + sgn * 36, cy - 30, title, size=L.T_TAG, alpha=fade, pad=5, anchor=anchor)
                for j, s in enumerate(lines):
                    f.text("w", cx + sgn * 36, cy + 4 + j * 26, s, size=L.T_SMALL, alpha=fade, anchor=anchor)
                boxes.append(box)
                break
            for tx_, ty_, txt in ((max(view[0] + 14, X_MIN), cy + 6, f"{e['h1']:06.3f}"),       # its altitude / its x,
                                  (cx + 4, view[1] + 24, f"{float(e['P1'][0]):+07.3f}")):   # on the cross-hair
                if not _hits(tbox(tx_, ty_, txt, L.T_MICRO, pad=3), avoid):
                    put_tag(f, ctx, "r", tx_, ty_, txt, size=L.T_MICRO, pad=3, alpha=fade)
        return boxes

    def draw_labels(self, f, cam, age, alive, view, avoid=(), limit=8):
        """Floating data tags on the ends of the tracks (PI+, MU-, energies), clear of the `avoid` boxes
        (the towers, the callouts)."""
        cands = []
        for ev, tt, p, word, num, red in self.labels:
            if not alive[ev]:
                continue
            a = age[ev] - tt
            if 0 <= a < 2.6:
                cands.append((a, p, word, num, red, float(smoothstep(0, 0.08, a) * (1 - smoothstep(1.6, 2.6, a)))))
        cands.sort(key=lambda c: c[0])
        boxes = [tuple(b) for b in avoid]
        xmin, xmax = max(view[0] + 30, X_MIN), min(view[2] - 20, X_MAX)
        n = 0
        for a, p, word, num, red, alpha in cands:
            px, py, pz, ok = cam.project(p[None])
            if not ok[0]:
                continue
            x, y = float(px[0]), float(py[0])
            bx = (x - 6, y - 50, x + 40 + 10.4 * (len(num) + (len(word) + 2 if word else 0)), y + 8)
            if not (xmin < bx[0] and bx[2] < xmax and view[1] + 60 < y < view[3] - 20):
                continue
            if any(bx[0] < o[2] and o[0] < bx[2] and bx[1] < o[3] and o[1] < bx[3] for o in boxes):
                continue
            if n >= limit:
                break
            n += 1
            boxes.append(bx)
            f.dots("w", [x], [y], 2.2, alpha)
            f.segments("w", [x], [y], [x + 20], [y - 20], 0.5 * alpha)
            m = int(min(len(num), a * 80))
            if word:
                f.tag("r" if red else "w", x + 24, y - 24, word, size=L.T_SMALL, alpha=alpha, pad=3)
                f.text("w", x + 24 + 10.4 * len(word) + 16, y - 24, num[:m], size=L.T_SMALL, alpha=alpha)
            else:
                f.text("w", x + 24, y - 24, num[:m], size=L.T_SMALL, alpha=alpha)

    # ---------------------------------------------------------------- HUD
    def draw_column(self, f, k, a, rect, title="PARTICLE_STREAM", alpha=1.0):
        """Scrolling list of the particles born so far in shower k (age a): id, kind, energy (GeV), x y z (km).
        `rect` = None when the placement leaves no column for it: nothing is drawn."""
        b = self.births[k]
        n_now = int(np.searchsorted(b["t"], a))
        if rect is None or alpha <= 0.01:
            return n_now
        x0, y0, x1, y1 = rect
        f.rects("w", x0, y0, x1, y0 + 5, 0.95 * alpha)
        f.tag("w", x0 + 4, y0 + 32, title, size=L.T_MICRO, pad=3, alpha=alpha)
        f.segments("w", [x1, x0], [y0, y1], [x1, x1], [y1, y1], 0.6 * alpha)
        pitch = 19.0
        n_rows = int((y1 - y0 - 96) / pitch)
        idx = np.arange(max(0, n_now - n_rows), n_now)[::-1]
        yy = y0 + 62
        for r, i in enumerate(idx):
            kk = int(b["k"][i])
            p = b["p"][i]
            name = K_NAME[kk] if kk != K_MU else ("MU-" if i % 2 else "MU+")
            line = f"{i:05d} {name:<5} {b['e'][i] * 3.2e6:09.2f} {p[0]:+06.2f} {p[1]:05.2f} {p[2]:+06.2f}"
            f.text("r" if K_RED[kk] == 1 else "w", x0 + 8, yy + r * pitch, line, size=L.T_MICRO,
                   alpha=(0.95 if r < 3 else 0.7) * alpha)
        f.text("w", x0 + 8, y1 - 10, f"N {n_now:06d}", size=L.T_SMALL, alpha=0.9 * alpha)
        return n_now

    def draw_strip(self, f, k, a, label=None, alpha=1.0, pulse=0.0):
        """Score strip: the longitudinal profile of shower k on a 0-16 km ruler, built live (births per 100 m),
        H1 and XMAX tags on the red band, a red cursor on the front."""
        x0, y0, x1, y1, yb = hud.strip_base(f, title=label or f"LONGITUDINAL_PROFILE // SHOWER {k + 1:02d}",
                                            alpha=alpha)
        e = self.events[k]

        def X(h):
            return x0 + (ALT_MAX - np.asarray(h, np.float64)) / ALT_MAX * (x1 - x0)

        hs = np.arange(0, ALT_MAX + 0.01, 0.1)
        xs = X(hs)
        kk = np.round(hs * 10).astype(int)
        ln = np.where(kk % 10 == 0, 16.0, np.where(kk % 5 == 0, 9.0, 4.0))
        f.segments("w", xs, np.full_like(xs, y0), xs, y0 + ln, 0.8 * alpha)
        f.segments("w", xs, np.full_like(xs, y1), xs, y1 - ln, 0.8 * alpha)
        for h in range(0, int(ALT_MAX) + 1, 2):
            f.text("w", float(X(h)) + 6, y1 - 20, f"{h:02d} KM", size=L.T_MICRO, alpha=0.7 * alpha)
        b = self.births[k]
        n_now = int(np.searchsorted(b["t"], a))
        if n_now:
            alt = b["p"][:n_now, 1]
            cnt, _ = np.histogram(alt, bins=160, range=(0.0, ALT_MAX))
            norm = max(cnt.max(), 1)
            hh = (50.0 + 8.0 * pulse) * np.sqrt(cnt / norm)
            bx = X(np.arange(160) * 0.1 + 0.1)
            m = cnt > 0
            f.rects("w", bx[m], y0 + 1, bx[m] + 6, y0 + 1 + hh[m], 0.95 * alpha)
            hb = 18.0 * np.sqrt(cnt / norm) * hash01(np.arange(160), k)
            f.rects("w", bx[m], y1 - 38 - hb[m], bx[m] + 6, y1 - 38, 0.55 * alpha)
            imax = int(np.argmax(cnt))
            if cnt[imax] > 30:
                f.tag("w", float(X(imax * 0.1 + 0.05)), yb + 7, f"XMAX {imax * 0.1:04.1f} KM", size=L.T_MICRO,
                      pad=4, alpha=alpha)
        if a >= e["t1"] - 0.05:
            f.tag("w", float(X(e["h1"])), yb + 7, f"H1 {e['h1']:.1f} KM", size=L.T_MICRO, pad=4, alpha=alpha)
        front = min(ALT_MAX, self.front(k, a))
        xc = float(X(front))
        f.segments("r", [xc], [y0 - 4], [xc], [y1 + 4], 1.2 * alpha, width=L.LW)
        anchor = "ls" if xc < x1 - 260 else "rs"
        ty = y0 - 9 if xc > x0 + 560 else y0 + 44           # keep clear of the title tag on the left
        f.tag("r", xc + (6 if anchor == "ls" else -6), ty, f"FRONT {front:06.3f} KM", size=L.T_MICRO, pad=3,
              alpha=alpha, anchor=anchor)
        return front

    def counts(self, k, a):
        b = self.births[k]
        kinds = b["k"][: int(np.searchsorted(b["t"], a))]
        return [("E+-", int((kinds == K_E).sum()), "w"), ("GAMMA", int((kinds == K_G).sum()), "w"),
                ("HADRON", int((kinds == K_H).sum()), "w"), ("MU+-", int((kinds == K_MU).sum()), "r")]

    def draw_counters(self, f, k, a, x0, x1, y0, title="PARTICLES", alpha=1.0):
        """Counters per particle kind: 2 x 2 when the panel is wide enough, else one narrow column."""
        hud.panel_header(f, x0, x1, y0, title, alpha=alpha)
        rows = self.counts(k, a)
        if x1 - x0 >= 430:
            cw = (x1 - x0) / 2
            for r, (lab, n, lay) in enumerate(rows):
                xx = x0 + (r % 2) * cw
                yy = y0 + 46 + (r // 2) * 44
                f.tag(lay, xx + 4, yy, lab, size=L.T_SMALL, pad=3, alpha=alpha)
                f.text(lay, xx + 104, yy + 2, f"{n:06d}", size=28, alpha=0.95 * alpha)
        else:
            for r, (lab, n, lay) in enumerate(rows):
                yy = y0 + 38 + r * 27
                f.tag(lay, x0 + 4, yy, lab, size=L.T_MICRO, pad=3, alpha=alpha)
                f.text(lay, x0 + 92, yy + 2, f"{n:06d}", size=22, alpha=0.95 * alpha)

    def draw_barcode(self, f, t, x0, x1, y0, y1, title="BIRTH_RATE >> BARCODE", span=3.0, alpha=1.0, boost=0.0):
        """Scrolling barcode: one column per ~13 ms, lit by the number of particles born in it."""
        hud.panel_header(f, x0, x1, y0, title if x1 - x0 > 260 else "BIRTH_RATE", alpha=alpha)
        cols = max(8, int((x1 - x0) / 4.0))
        dt = span / cols
        k_first = math.floor((t - span) / dt)
        kk = k_first + np.arange(cols)
        rate = np.zeros(cols, np.float32)
        t0s = self.ev_t0[: self.n_showers]
        for ev in np.nonzero((t0s < t) & (t0s > t - 30.0))[0]:
            b = self.births[ev]
            tt = kk * dt - self.ev_t0[ev]
            rate += (np.searchsorted(b["t"], tt + dt) - np.searchsorted(b["t"], tt))
        dens = np.clip(0.06 + 0.9 * np.tanh(rate / 60.0) + boost, 0.0, 1.0)
        cw = (x1 - x0) / cols
        frac = (t - span) / dt - k_first
        xs = x0 + (np.arange(cols) - frac) * cw
        lane_h = (y1 - y0 - 14) / 3
        for ln in range(3):
            on = hash01(kk, ln + 13) < dens * (1.0 - 0.2 * ln)
            m = on & (xs >= x0) & (xs + cw <= x1)
            ly0 = y0 + 12 + ln * lane_h
            f.rects("w", xs[m], ly0, xs[m] + cw, ly0 + lane_h - 3, 0.95 * alpha)
        f.segments("r", [x1 - 2], [y0 + 7], [x1 - 2], [y1], 1.2 * alpha, width=L.LW)


def altitude_rules(f, ctx, st, cam, x_ref, z_ref, lay=None, hmax=17, label_x=None, gain=1.0, depth=True, span=None,
                   front=None):
    """Horizontal rule every km of altitude across the view, or across `span` = (x0, x1) (rules may pass
    behind the towers). Km labels at the left of the focus column, slant depth of the air in the right-hand
    column of the layout - where no tower and no other block hides them."""
    view = st.view
    x0, x1 = span if span else (view[0], view[2])
    P = np.stack([np.full(hmax, x_ref), np.arange(hmax, dtype=np.float32), np.full(hmax, z_ref)], 1).astype(np.float32)
    _, py, _, _ = cam.project(P)
    lx = label_x if label_x is not None else st.focus_col[0] + 4
    taken = lay["boxes"] if lay else ()
    xr = lay["xr"] if (lay and depth) else None
    y_front = None                       # the FRONT tag sits on the red front line: keep its row free
    if front is not None:
        y_front = float(cam.project(np.array([[x_ref, front, z_ref]], np.float32))[1][0])
    for h in range(hmax):
        y = float(py[h])
        if not (view[1] + 8 < y < view[3] - 4):
            continue
        f.segments("w", [x0], [y], [x1], [y], (0.13 if h % 5 else 0.3) * gain)
        if y < view[1] + 40:
            continue
        txt = f"{h:02d} KM"
        if not _hits(tbox(lx, y - 7, txt, L.T_SMALL), taken):
            put_text(f, ctx, "w", lx, y - 7, txt, size=L.T_SMALL, alpha=0.75 * gain)
        if xr is not None and not (y_front is not None and abs(y - y_front) < 30.0):
            txt = f"X {1030.0 * math.exp(-h / 8.4):06.1f} G/CM2"      # slant depth of the air above this altitude
            if not _hits(tbox(xr, y - 7, txt, L.T_MICRO, "rs"), taken):
                put_text(f, ctx, "w", xr, y - 7, txt, size=L.T_MICRO, alpha=0.5 * gain, anchor="rs")


# ----------------------------------------------------------------------------
# scene: the hero shower of the intro
# ----------------------------------------------------------------------------

class Shower(Scene):
    name = "shower"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.t_in, self.t_out = 67.0, 72.4
        for sec in sd.SECTIONS:
            if sec[4] == "shower":
                self.t_in, self.t_out = sec[2], sec[3]
        self.t_int = sd.said("Through the atmosphere above this city", 67.55)
        self.t_land = sd.said("Through...", 71.117)
        self.you_xz = (1.15, 0.6)
        spec = dict(t_int=self.t_int, ground=(0.0, 0.0), E0=1.0, zen=6.0, az=128.0, h1=14.6,
                    travel=self.t_land - self.t_int, you=self.you_xz)
        # a few lone muons on the kicks: the rain that never stops
        kt, ka = ctx.cues.kicks(self.t_in + 0.6, self.t_out)
        rng = np.random.default_rng(12)
        rain = []
        for tk in kt[np.argsort(-ka)[:7]]:
            rain.append(dict(t_land=float(tk), target=(float(rng.uniform(-11, 11)), 0.0, float(rng.uniform(-4, 4)))))
        self.world = World([spec], rain, seed=4, cache=f"hero_{int(self.t_int * 1000)}_{int(self.t_land * 1000)}")
        # --- layout from the towers ---
        # view 1 (the hero: tags, MU- 0001, the footprint) lives in the focus bay, on ctx.focus;
        # view 2 (a second elevation, no tags) takes the widest other bay, if there is one worth it.
        st = self.st = stage_for(ctx, ctx.focus[0])
        view = st.view
        fb = ctx.focus_bay
        cands = []
        for b in ctx.bays:
            if abs(b[0] - fb[0]) < 1.0 and abs(b[1] - fb[1]) < 1.0:
                continue
            a0, a1 = max(b[0], view[0]), min(b[1], view[2])          # what the particle column leaves of it
            if a1 - a0 >= 550.0:                                     # wide enough for a scale of 75 px per km
                cands.append((a0, a1))
        self.bay2 = max(cands, key=lambda b: b[1] - b[0]) if cands else None
        w1 = fb[1] - fb[0]
        w2 = (self.bay2[1] - self.bay2[0]) if self.bay2 else w1
        self.s_side = float(np.clip((min(w1, w2) / 2 - 20) / 3.4, 60.0, 96.0))     # px per km, elevations
        self.s_top = float(np.clip(w1 / 8.0, 70.0, 110.0))                         # px per km, plan
        if self.bay2 is None:
            self.clip1, self.clip2 = view, None
        elif self.bay2[0] >= fb[1]:
            split = 0.5 * (fb[1] + self.bay2[0])
            self.clip1, self.clip2 = (view[0], view[1], split, view[3]), (split, view[1], view[2], view[3])
        else:
            split = 0.5 * (self.bay2[1] + fb[0])
            self.clip1, self.clip2 = (split, view[1], view[2], view[3]), (view[0], view[1], split, view[3])
        self.x2 = 0.5 * (self.bay2[0] + self.bay2[1]) if self.bay2 else None

    # ------------------------------------------------------------------ cameras
    def _cam_side(self, a, x_screen, yaw_deg):
        """Elevation tracking the front: the ground ends just above the bottom band."""
        w = self.world
        e = w.events[0]
        gx, gz = float(e["G"][0]), float(e["G"][2])
        half = 0.5 * (VIEW_Y1 - VIEW_Y0)
        c_lo = (half - 30.0) / self.s_side
        c_hi = max(c_lo, e["h1"] - (half - 130.0) / self.s_side)       # first interaction under the header
        cy = float(np.clip(w.front(0, a) + 1.2, c_lo, c_hi))
        yaw = math.radians(yaw_deg)
        return OrthoCamera((gx + 60 * math.sin(yaw), cy, gz + 60 * math.cos(yaw)), (gx, cy, gz), scale=self.s_side,
                           screen_center=(x_screen, self.st.cy))

    def _cam_top(self, t, focus):
        """Plan view centred on the point where the tagged muon lands: MU- 0001 sits exactly on ctx.focus,
        where the figure of the next scene (YOU) stands; the core of the shower is 1.3 km away from it."""
        y = self.world.you
        c = y["b"] if y is not None else self.world.events[0]["G"]
        cx, cz = float(c[0]), float(c[2])
        u = (t - self.t_land) / max(self.t_out - self.t_land, 1e-3)
        return OrthoCamera((cx, 40.0, cz + 1e-3), (cx, 0.0, cz), scale=self.s_top * (1.0 + 0.14 * u),
                           up=(0.0, 0.0, -1.0), screen_center=(float(focus[0]), float(focus[1])),
                           roll_deg=6.0 + 10.0 * u)

    # ------------------------------------------------------------------ draw
    def draw(self, f, t, ctx):
        w, st = self.world, self.st
        view = st.view
        e = w.events[0]
        gx, gz = float(e["G"][0]), float(e["G"][2])
        age, alive, env = w.state(t)
        a = float(age[0])
        top = t >= self.t_land
        two = self.clip2 is not None
        kick = min(1.5, ctx.cues.kick(t))
        fx = ctx.focus[0]
        cam1 = self._cam_top(t, ctx.focus) if top else self._cam_side(a, fx, 8.0)
        views = [("top" if top else "side", cam1, self.clip1)]
        hw = 3.4 * self.s_side
        if top:                                   # the footprint: its dense part stays clear of the texts
            gp = cam1.project(e["G"][None].astype(np.float32))
            xc, yc = float(gp[0][0]), float(gp[1][0])
            ch = chord(xc, yc, 2.4 * cam1.scale, view[1] + 100.0)
            z_top = [ch] if ch else []
            z_full = [zone(ctx, xc, 3.7 * cam1.scale, ctx.focus_bay)]
        else:
            z_top = [(fx - 150.0, fx + 150.0)]
            z_full = [zone(ctx, fx, hw, ctx.focus_bay)]
        if two:
            views.append(("side", self._cam_side(a, self.x2, 98.0), self.clip2))
            z_top.append((self.x2 - 150.0, self.x2 + 150.0))
            z_full.append(zone(ctx, self.x2, hw, self.bay2))
        name = "ORTHO_TOP" if top else ("ORTHO_SIDE X / Z" if two else "ORTHO_SIDE")
        lay = info_layout(ctx, st, f"VIEW {2 if top else 1:02d} // {name}", cam1,
                          [("AIR_SHOWER // MUON BLOOM // SHOWER 01", 0.85),
                           (f"E0 {e['E0'] * 3.2:.2f}E15 EV   ZENITH {e['zen']:.1f} DEG", 0.6),
                           (f"GROUND {CITY}", 0.6)],
                          [("AIR_SHOWER // SHOWER 01", 0.85), (f"E0 {e['E0'] * 3.2:.2f}E15 EV", 0.6),
                           (f"ZENITH {e['zen']:.1f} DEG", 0.6), (CITY, 0.6)],
                          z_top, z_full, within=(self.clip2[0], self.clip2[2]) if (top and two) else None)
        # altitude rules: across the wall while both views are elevations, then only under the second one
        f.set_clip(*view)
        if not top:
            altitude_rules(f, ctx, st, cam1, gx, gz, lay, front=w.front(0, a))
        elif two:
            altitude_rules(f, ctx, st, views[1][1], gx, gz, lay, span=(self.clip2[0], self.clip2[2]),
                           label_x=col_of(ctx, self.x2)[0] + 4)
        fixed = tower_boxes(ctx) + lay["boxes"]
        for j, (kind, cam, clip) in enumerate(views):
            f.set_clip(*clip)
            w.draw_ground(f, cam, kind, clip)
            w.draw_cascades(f, cam, age, alive, env, gain=1.0 + 0.12 * kick)
            w.draw_hits(f, cam, age, alive)
            w.draw_splash(f, cam, age)
            if j == 0:
                self._overlay(f, ctx, lay, cam, kind, t, a, clip)
                boxes = w.draw_interaction(f, cam, age, clip, ctx, avoid=lay["boxes"])
                box = self._the_muon(f, ctx, cam, kind, t, a, boxes + lay["boxes"], clip)
                if box:
                    boxes.append(box)
                w.draw_labels(f, cam, age, alive, clip, avoid=fixed + boxes, limit=7)
            else:
                w.draw_interaction(f, cam, age, clip, tags=False)
                self._second(f, ctx, cam, t, a, clip)
                w.draw_labels(f, cam, age, alive, clip, avoid=fixed, limit=4)
        f.set_clip()
        w.draw_column(f, 0, a, st.col)
        w.draw_strip(f, 0, a, label="LONGITUDINAL_PROFILE // SHOWER 01 // ABOVE THIS CITY", pulse=kick)
        self._bottom(f, ctx, t, a)
        draw_info(f, ctx, lay)
        return {"invert": 0.0 <= t - self.t_int < 0.05, "invert_rect": view}

    def _overlay(self, f, ctx, lay, cam, kind, t, a, clip):
        w, st = self.world, self.st
        view = st.view
        e = w.events[0]
        front = w.front(0, a)
        if kind == "side":
            f.set_clip(*view)
            if 0 < front < e["h1"]:                                   # the front: one red line across the wall
                P = np.array([[float(e["G"][0]), front, float(e["G"][2])]], np.float32)
                y = float(cam.project(P)[1][0])
                f.segments("r", [view[0]], [y], [view[2]], [y], 0.9, width=L.LW)
                put_right(f, ctx, st, lay, "r", y - 9, f"FRONT {front:06.3f} KM", size=L.T_LABEL, pad=5)
            gy = float(cam.project(e["G"][None].astype(np.float32))[1][0])
            if view[1] < gy < view[3]:
                put_right(f, ctx, st, lay, "w", gy - 9, f"GROUND // {CITY}", size=L.T_SMALL, pad=4)
            f.set_clip(*clip)
            ax, ay, _, _ = cam.project(np.stack([e["G"], e["P1"]]).astype(np.float32))
            f.segments("r", [ax[0]], [ay[0]], [ax[1]], [ay[1] - 60], 0.5)
            if self.clip2 is not None and view[1] < float(ay[0]) < view[3]:
                put_text(f, ctx, "w", float(ax[0]) + 44, float(ay[0]) - 14, "ELEVATION X", size=L.T_SMALL, alpha=0.7)
        else:
            gx, gy, _, _ = cam.project(e["G"][None].astype(np.float32))
            X, Y = float(gx[0]), float(gy[0])
            big = 1e5
            f.segments("r", [X - big, X], [Y, Y - big], [X + big, X], [Y, Y + big], 0.45)
            sc = cam.scale
            f.rings("w", [X] * 4, [Y] * 4, [sc * r for r in (1, 2, 4, 8)], 0.25)
            for r in (1, 2, 4, 8):
                if X + sc * r + 70 < min(clip[2], st.tx1):
                    put_text(f, ctx, "w", X + sc * r + 7, Y - 8, f"{r} KM", size=L.T_SMALL, alpha=0.6)
            put_tag(f, ctx, "r", X - 12, Y - 14, "CORE", size=L.T_SMALL, pad=4, anchor="rs")

    def _second(self, f, ctx, cam, t, a, clip):
        """The second elevation: no tags, only its name and the tagged muon as a bolder track."""
        e = self.world.events[0]
        ax, ay, _, _ = cam.project(np.stack([e["G"], e["P1"]]).astype(np.float32))
        f.segments("r", [ax[0]], [ay[0]], [ax[1]], [ay[1] - 60], 0.5)
        if clip[1] < float(ay[0]) < clip[3]:
            put_text(f, ctx, "w", float(ax[0]) + 44, float(ay[0]) - 14, "ELEVATION Z", size=L.T_SMALL, alpha=0.7)
        y = self.world.you
        if y is None or a < y["t0"]:
            return
        prog = float(np.clip((a - y["t0"]) / (y["t1"] - y["t0"]), 0.0, 1.0))
        A, B = y["a"], y["b"]
        sx, sy, _, ok = cam.project(np.stack([A, A + (B - A) * prog]).astype(np.float32))
        if ok.all():
            f.segments("r", sx[:1], sy[:1], sx[1:], sy[1:], 1.0, width=L.LW_BOLD)
            if prog < 1.0:
                f.rings("r", sx[1:], sy[1:], [13.0], 0.9, width=L.LW)

    def _the_muon(self, f, ctx, cam, kind, t, a, avoid, clip):
        """The tagged muon: a brighter red track, a tag that follows its head, a red target where it lands."""
        y = self.world.you
        if y is None or a < y["t0"]:
            return None
        prog = float(np.clip((a - y["t0"]) / (y["t1"] - y["t0"]), 0.0, 1.0))
        A, B = y["a"], y["b"]
        head = A + (B - A) * prog
        P = np.stack([A, head, B]).astype(np.float32)
        sx, sy, _, ok = cam.project(P)
        if not ok.all():
            return None
        f.segments("r", sx[:1], sy[:1], sx[1:2], sy[1:2], 1.2, width=L.LW_BOLD)
        hx, hy = float(sx[1]), float(sy[1])
        gam = y["E"] / 0.10566
        t_lab = float(np.linalg.norm(B - A)) / 299792.458 * 1e6          # us
        if kind == "side":
            tx_, ty_ = float(sx[2]), float(sy[2])                        # where it is going: a red target
            if clip[1] < ty_ < clip[3]:
                f.rings("r", [tx_], [ty_], [16.0], 0.9, width=L.LW)
                f.crosses("r", [tx_], [ty_], 26.0, 0.9)
            if not (clip[1] + 30 < hy < clip[3] - 10):
                return None
            f.dots("r", [hx], [hy], 5.0, 1.8)
            f.dots("w", [hx], [hy], 2.0, 1.2)
            f.rings("r", [hx], [hy], [15.0], 0.9, width=L.LW)
            return auto_callout(f, ctx, clip, hx, hy, "MU- 0001",
                                [f"E {y['E']:.3f} GEV", f"GAMMA {gam:.1f}", f"T-{max(0.0, self.t_land - t):05.3f} S"],
                                red=True, age=a - y["t0"], avoid=avoid, prefer=(1, -1))
        al = max(0.0, t - self.t_land)
        u = min(1.0, al / 0.9)
        f.rings("r", [hx, hx], [hy, hy], [14.0, 14.0 + 90.0 * (1 - (1 - u) ** 3)], [1.0, 0.9 * (1 - u)],
                width=L.LW_BOLD)
        f.crosses("r", [hx], [hy], 34.0, 1.0, width=L.LW)
        f.dots("w", [hx], [hy], 3.4, 1.5)
        return auto_callout(f, ctx, clip, hx, hy, "MU- 0001",
                            ["ARRIVED", f"T {t_lab:.1f} US", f"TAU' {t_lab / gam:.2f} US", ">> THROUGH..."],
                            red=True, age=al, avoid=avoid, prefer=(1, 1), dx=40.0, dy=64.0)

    def _bottom(self, f, ctx, t, a):
        """Bottom band, before the detectors are revealed: the blocks flow into the panels between the towers."""
        w = self.world
        sl = ctx.slots_pre
        y0, y1 = sl["y0"], sl["y1"]
        place = flow(bottom_panels(ctx, pre=True),
                     [("time", 400.0, 610.0), ("count", 200.0, 470.0), ("bar", 160.0, 0.0, True)])
        if "time" in place:
            x0, x1 = place["time"]
            wide = x1 - x0 >= 560
            size = min(64.0, (x1 - x0 - (200 if wide else 10)) / 6.1)
            hud.panel_header(f, x0, x1, y0, "FRONT // TIME TO GROUND")
            rem = self.t_land - t
            if rem > 0:
                f.text("w", x0 + 2, y0 + 96, f"T-{rem:06.3f} S", size=size)
                if wide:
                    f.text("w", x1 - 4, y0 + 46, f"ALT {min(w.front(0, a), 99.0):06.3f} KM", size=L.T_SMALL, alpha=0.8,
                           anchor="rs")
                    f.text("w", x1 - 4, y0 + 72, "V 0.9998 C", size=L.T_SMALL, alpha=0.6, anchor="rs")
                    slow = (self.t_land - self.t_int) / (w.events[0]["h1"] / 299792.458)
                    f.text("w", x1 - 4, y0 + 98, f"SLOWED x {slow:,.0f}".replace(",", " "), size=L.T_SMALL, alpha=0.6,
                           anchor="rs")
            else:
                f.tag("r", x0 + 6, y0 + 96, "GROUND", size=size, pad=8, bold=True)
                if wide:
                    f.text("r", x1 - 4, y0 + 46, f"T+{-rem:05.3f} S", size=L.T_SMALL, anchor="rs")
                    f.text("w", x1 - 4, y0 + 72, CITY, size=L.T_SMALL, alpha=0.7, anchor="rs")
        if "count" in place:
            w.draw_counters(f, 0, a, place["count"][0], place["count"][1], y0)
        if "bar" in place:
            w.draw_barcode(f, t, place["bar"][0], place["bar"][1], y0, y1)
