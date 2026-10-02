"""YOU - muons pass through you.   Sheet 1.3, 01:12.4 - 01:41.0 (the "hero WOW moment").

About one muon per square centimetre per minute reaches the ground: ~63 go through a spectator
every second, ~3 800 a minute, more than fifty thousand during the show (the voice of scene 5).

A 1.80 m body built from lofted ellipse sections: a rim-lit point cloud with CT-like slice
contours over a dot-lattice floor. Cosmic muons (red) rain through it on the 16th-note grid of
the music (90 BPM here); each crossing is hit-tested: the part lights up, the slice at that
height turns red, the energy left behind is logged.

Linear, cut on the bars of the music (2.667 s):
  01:12.48  "YOU"        the muon tagged in the air shower comes straight down through the HEART;
                         on the drum roll the body assembles, slice by slice, around that hit
  01:14.64  the drop     PERSPECTIVE orbit, the whole instrument switches on, the rain starts
  01:19.97               ORTHO_FRONT / ORTHO_SIDE, a red scan slice sweeping
  01:25.30  "An echo..." THORAX close-up, a second muon through the heart: the one we follow
  01:27.97  "...through your bones, your cells, ancient, indifferent": powers of ten along that
            track, one per bar: BONE (cm) - CELLS (um) - DNA (nm) - WATER (0.1 nm)
  01:38.64  "It is called...": nothing left but the straight red track and what was measured

LAYOUT. The three towers stand in front of the wall for the whole show (dark until 01:44) and
nobody knows yet where: nothing here has a fixed x. `Plan` reads the tower placement from ctx:
  focus bay   the figure, the heart hit, the track we follow and their tags (ctx.focus)
  title col   YOU, the parameters, the hit log         (leftmost free column)
  side col    the second view (ORTHO_SIDE) / the read-outs of the zoom   (widest free column)
  data col    energy left in each part of the body     (what remains)
  bottom      counts on every time scale, hit barcode  (ctx.slots_pre panels)
Only textures (floor lattice, rain, tissue, cells, molecules) run behind the towers. A column
that does not exist with a given placement simply drops its block.
"""
from __future__ import annotations

import math

import numpy as np

from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import Camera, OrthoCamera, hash01, smoothstep, text_w
from ..show import Scene

# -- music grid of this section (found on the kicks of the music stem) --------------------
BAR = 8.0 / 3.0                 # 2.667 s: 4 beats at 90 BPM
STEP = BAR / 16.0
BAR0 = 74.64                    # the drop
T_IN, T_OUT = 72.4, 101.0
CUTS = [BAR0 + 2 * BAR, BAR0 + 4 * BAR, BAR0 + 5 * BAR, BAR0 + 6 * BAR, BAR0 + 7 * BAR, BAR0 + 8 * BAR,
        BAR0 + 9 * BAR]         # ortho, thorax, bone, cells, dna, atoms, track

# -- vertical layout (x comes from the towers, see Plan) -------------------------------------
Y_TOP, Y_BOT = 240.0, 1190.0
Y_MID = 715.0
WALL = (L.FX0 + 4.0, Y_TOP, L.FX1 - 4.0, Y_BOT)

DEDX = 0.2                      # GeV per metre of tissue (~2 MeV/cm)
RATE = 63.0                     # muons / s through one spectator (ctx.RATE_YOU)

_L = [  # name, ys, cx, cz, rx, rz   (figure faces +z, its left side is +x)
    ("FOOT_L", [0.0, 0.03, 0.075], [0.12, 0.12, 0.115], [0.05, 0.045, 0.01], [0.042, 0.045, 0.036],
     [0.115, 0.12, 0.055]),
    ("LEG_L", [0.075, 0.12, 0.33, 0.47, 0.52, 0.72, 0.9], [0.115, 0.115, 0.112, 0.108, 0.106, 0.1, 0.095],
     [0.0, 0.0, -0.005, 0.005, 0.005, 0.005, 0.0], [0.036, 0.04, 0.058, 0.048, 0.052, 0.075, 0.088],
     [0.045, 0.048, 0.062, 0.05, 0.055, 0.08, 0.09]),
    ("TORSO", [0.86, 0.92, 1.0, 1.08, 1.2, 1.3, 1.4, 1.46], [0.0] * 8, [0.0, 0.0, 0.005, 0.01, 0.02, 0.02, 0.01, 0.0],
     [0.165, 0.178, 0.16, 0.145, 0.16, 0.178, 0.19, 0.12], [0.105, 0.115, 0.105, 0.1, 0.11, 0.12, 0.105, 0.08]),
    ("NECK", [1.44, 1.5, 1.565], [0.0] * 3, [0.0, 0.0, 0.005], [0.058, 0.052, 0.055], [0.06, 0.055, 0.06]),
    ("HEAD", [1.555, 1.585, 1.63, 1.68, 1.73, 1.77, 1.795, 1.8], [0.0] * 8, [0.02] * 8,
     [0.055, 0.07, 0.08, 0.083, 0.078, 0.062, 0.04, 0.01], [0.07, 0.085, 0.095, 0.1, 0.097, 0.08, 0.05, 0.012]),
    ("ARM_L", [0.84, 0.96, 1.1, 1.14, 1.3, 1.43], [0.335, 0.322, 0.3, 0.295, 0.262, 0.225],
     [0.02, 0.015, 0.01, 0.005, 0.0, 0.0], [0.028, 0.034, 0.04, 0.042, 0.05, 0.056],
     [0.03, 0.036, 0.042, 0.044, 0.052, 0.058]),
    ("HAND_L", [0.64, 0.68, 0.74, 0.8, 0.845], [0.35, 0.35, 0.348, 0.343, 0.338], [0.02, 0.024, 0.025, 0.023, 0.02],
     [0.012, 0.02, 0.022, 0.024, 0.026], [0.03, 0.046, 0.05, 0.045, 0.03]),
]
LOFTS = []
for _name, _ys, _cx, _cz, _rx, _rz in _L:
    LOFTS.append((_name, *(np.asarray(v, np.float64) for v in (_ys, _cx, _cz, _rx, _rz))))
    if _name.endswith("_L"):
        LOFTS.append((_name[:-2] + "_R", np.asarray(_ys), -np.asarray(_cx), np.asarray(_cz), np.asarray(_rx),
                      np.asarray(_rz)))
PART_NAMES = [l[0] for l in LOFTS]
HEART = np.array([0.045, 1.27, 0.04])


def _loft_at(l, y):
    name, ys, cx, cz, rx, rz = l
    return (np.interp(y, ys, cx), np.interp(y, ys, cz), np.interp(y, ys, rx), np.interp(y, ys, rz))


def spaced(n):
    """12 345 678 (thin thousands separator, as on the counter cell)."""
    return f"{int(n):,}".replace(",", " ")


def width(c):
    return c[1] - c[0]


# ----------------------------------------------------------------------------
# where things go, for whatever tower placement
# ----------------------------------------------------------------------------

class Plan:
    """Columns of this look, all derived from the tower placement in ctx."""

    def __init__(self, ctx, panels="slots_pre"):
        self.fx, self.fy = float(ctx.focus[0]), Y_MID
        self.bays = [tuple(b) for b in ctx.bays]
        self.fbay = tuple(ctx.focus_bay)
        cols = [tuple(c) for c in ctx.cols]
        inside = [c for c in cols if c[0] - 1 <= self.fx <= c[1] + 1]
        self.fcol = inside[0] if inside else (self.fbay[0] + 28.0, self.fbay[1] - 28.0)
        self.half = min(self.fx - self.fbay[0], self.fbay[1] - self.fx)      # free half-width around the focus
        others = [c for c in cols if c != self.fcol]
        cand = [c for c in others if width(c) >= 255.0]
        self.title = min(cand, key=lambda c: c[0]) if cand else None
        rest = [c for c in others if c != self.title]
        wide = [c for c in rest if width(c) >= 300.0]
        self.side = max(wide, key=lambda c: width(c) - 0.3 * abs(0.5 * (c[0] + c[1]) - self.fx)) if wide else None
        rest = [c for c in rest if c != self.side]
        cand = [c for c in rest if width(c) >= 250.0]
        self.data = max(cand, key=width) if cand else None
        sl = getattr(ctx, panels)
        self.panels = sorted([tuple(p) for p in sl["panels"]], key=width, reverse=True)
        self.py0, self.py1 = sl["y0"], sl["y1"]
        # the field window: focus bay + side bay (and the tower between them). Textures live there only.
        sb = self.bay_of(self.side) if self.side else self.fbay
        self.win = (min(self.fbay[0], sb[0]), max(self.fbay[1], sb[1]))

    def plates(self, f, y0=Y_TOP - 4.0, y1=Y_BOT + 8.0):
        """Black out what is not the field window: the title and data columns are read on black."""
        for a, b in ((L.FX0 + 2.0, self.win[0]), (self.win[1], L.FX1 - 2.0)):
            if b - a > 2.0:
                f.occlude(a, y0, b, y1)
        for col in (self.title, self.data):
            if col is not None and self.win[0] < 0.5 * (col[0] + col[1]) < self.win[1]:
                b = self.bay_of(col)
                f.occlude(b[0], y0, b[1], y1)

    def bay_of(self, col):
        c = 0.5 * (col[0] + col[1])
        for b in self.bays:
            if b[0] <= c <= b[1]:
                return b
        return col


def callout(f, x, y, title, lines=(), col=None, prefer=1, dy=-50.0, red=False, alpha=1.0, age=9.0, size=L.T_TAG,
            lsize=L.T_SMALL, elbow=40.0):
    """hud.callout on a black plate, kept inside `col`. It goes on the `prefer` side (outwards, away from
    what it points at); when the block does not fit there it first drops its data lines, then shortens
    its leader, and only then changes side."""
    if alpha <= 0.01 or age < 0:
        return
    lo, hi = col if col else (L.COL_X0, L.COL_X1)

    def block(ls):
        return max([text_w(title, size) + 12] + [text_w(ln, lsize) for ln in ls]) + 10

    def fits(side, ls, el):
        return (x + el + 44 + block(ls) <= hi) if side > 0 else (x - el - 44 - block(ls) >= lo)

    lines = list(lines)
    tries = [(prefer, lines, elbow), (prefer, [], elbow), (prefer, [], 14.0), (-prefer, lines, elbow),
             (-prefer, [], elbow), (-prefer, [], 14.0)]
    side, lines, elbow = next((tr for tr in tries if fits(*tr)), (1 if hi - x >= x - lo else -1, [], 14.0))
    w = block(lines)
    tx = x + side * (elbow + 44)
    y0 = y + dy - size * 0.78
    y1 = y + dy + size * 0.36 + len(lines) * lsize * 1.45 + (16 if lines else 8)
    f.occlude(min(tx, tx + side * w) - 8, y0, max(tx, tx + side * w) + 8, y1)
    hud.callout(f, x, y, side * elbow, dy, title, lines, red=red, alpha=alpha, age=age, size=size, lsize=lsize,
                side=side)


def fit_text(options, room, size):
    """First of `options` (long to short) that fits in `room` px at `size`."""
    for s in options:
        if text_w(s, size) + 14 <= room:
            return s
    return options[-1]


# ----------------------------------------------------------------------------
# the body (shared with the flood scene)
# ----------------------------------------------------------------------------

class Body:
    """Point cloud + slice contours of a standing figure, with an inside test for the muons."""

    def __init__(self, seed=33, density=16000.0, floor=3.2):
        rng = np.random.default_rng(seed)
        P, Nn, part = [], [], []
        for li, l in enumerate(LOFTS):
            ys = l[1]
            y = np.arange(ys[0], ys[-1], 0.004)
            cx, cz, rx, rz = _loft_at(l, y)
            per = math.pi * (3 * (rx + rz) - np.sqrt((3 * rx + rz) * (rx + 3 * rz)))
            cnt = rng.poisson(per * 0.004 * density)
            yy = np.repeat(y, cnt) + rng.uniform(0, 0.004, cnt.sum())
            th = rng.uniform(0, 2 * np.pi, cnt.sum())
            cx, cz, rx, rz = _loft_at(l, yy)
            n = np.stack([np.cos(th) / rx, np.zeros_like(th), np.sin(th) / rz], 1)
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            P.append(np.stack([cx + rx * np.cos(th), yy, cz + rz * np.sin(th)], 1))
            Nn.append(n)
            part.append(np.full(len(yy), li, np.int16))
        self.pts = np.concatenate(P).astype(np.float32)
        self.nrm = np.concatenate(Nn).astype(np.float32)
        self.pts_part = np.concatenate(part)
        self.rnd = rng.random(len(self.pts)).astype(np.float32)
        self.drift = rng.normal(0, 1, (len(self.pts), 3)).astype(np.float32)
        # CT-like slice contours every 3 cm
        self.levels = np.arange(0.015, 1.8, 0.03)
        segs_a, segs_b, lev = [], [], []
        ang = np.linspace(0, 2 * np.pi, 41)
        for k, y in enumerate(self.levels):
            for l in LOFTS:
                if l[1][0] <= y <= l[1][-1]:
                    cx, cz, rx, rz = _loft_at(l, y)
                    e = np.stack([cx + rx * np.cos(ang), np.full_like(ang, y), cz + rz * np.sin(ang)], 1)
                    segs_a.append(e[:-1])
                    segs_b.append(e[1:])
                    lev.append(np.full(40, k, np.int32))
        self.ca = np.concatenate(segs_a).astype(np.float32)
        self.cb = np.concatenate(segs_b).astype(np.float32)
        self.clev = np.concatenate(lev)
        g = np.arange(-floor, floor + 1e-3, 0.1)
        X, Z = np.meshgrid(g, g)
        self.lattice = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1).astype(np.float32)
        self.lat_major = ((np.abs(np.round(X * 10)) % 5 == 0) & (np.abs(np.round(Z * 10)) % 5 == 0)).ravel()

    def inside(self, p):
        """Which loft contains each point (N, 3) -> index or -1."""
        out = np.full(len(p), -1, np.int32)
        for li, l in enumerate(LOFTS):
            ys = l[1]
            m = (p[:, 1] >= ys[0]) & (p[:, 1] <= ys[-1]) & (out < 0)
            if not m.any():
                continue
            cx, cz, rx, rz = _loft_at(l, p[m, 1])
            ins = ((p[m, 0] - cx) / rx) ** 2 + ((p[m, 2] - cz) / rz) ** 2 <= 1.0
            out[np.nonzero(m)[0][ins]] = li
        return out

    def target_in(self, name, rng):
        if name == "HEART":
            return HEART + rng.normal(0, 0.012, 3)
        l = LOFTS[PART_NAMES.index(name)]
        ys = l[1]
        y = rng.uniform(ys[0] + 0.25 * (ys[-1] - ys[0]), ys[-1] - 0.25 * (ys[-1] - ys[0]))
        cx, cz, rx, rz = _loft_at(l, y)
        a = rng.uniform(0, 2 * np.pi)
        rr = rng.uniform(0, 0.55)
        return np.array([cx + rx * rr * math.cos(a), y, cz + rz * rr * math.sin(a)])

    def muon(self, rng, t_hit, tgt, hero, zen_sigma=0.38, speed=22.0, zen=None, top=2.9):
        """A straight track through `tgt`; t_hit = time its head reaches the target."""
        th = min(1.0, abs(rng.normal(0, zen_sigma))) if zen is None else zen
        ph = rng.uniform(0, 2 * np.pi)
        d = np.array([math.sin(th) * math.cos(ph), -math.cos(th), math.sin(th) * math.sin(ph)])
        s_top = (top - tgt[1]) / math.cos(th)
        s_bot = tgt[1] / math.cos(th)
        a = tgt - d * s_top
        b = tgt + d * s_bot
        Ln = s_top + s_bot
        u = np.linspace(0, 1, 700)
        pts = a[None] + (b - a)[None] * u[:, None]
        ins = self.inside(pts)
        inside = ins >= 0
        entry = int(np.argmax(inside)) if inside.any() else -1
        part, hit, dE, u_in = None, None, 0.0, None
        if entry >= 0:
            dE = inside.sum() * (Ln / 700) * DEDX
            if hero:        # tag the organ it was sent through, where it crosses it
                hit = tgt
                u_in = s_top / Ln
                k = self.inside(tgt[None])[0]
                part = "HEART" if np.linalg.norm(tgt - HEART) < 0.05 else (PART_NAMES[k] if k >= 0 else "TORSO")
            else:
                hit = pts[entry]
                part = PART_NAMES[ins[entry]]
                if np.linalg.norm(hit - HEART) < 0.085:
                    part = "HEART"
                u_in = u[entry]
        edges = np.diff(np.r_[0, inside.astype(int), 0])
        starts = np.nonzero(edges == 1)[0]
        ends = np.nonzero(edges == -1)[0] - 1
        runs = list(zip(u[starts], u[ends]))
        dur = Ln / speed
        t0 = t_hit - dur * (u_in if u_in is not None else s_top / Ln)
        return dict(t=t0, a=a.astype(np.float32), b=b.astype(np.float32), L=Ln, dur=dur, part=part, hit=hit,
                    u_in=u_in, dE=dE, E=float(np.exp(rng.normal(1.2, 0.7))), runs=runs, hero=hero,
                    charge=rng.choice(["+", "-"]), t_hit=t_hit if hit is not None else None)

    # ------------------------------------------------------------------ drawing
    def draw_floor(self, f, cam, gain=1.0):
        sx, sy, z, ok = cam.project(self.lattice)
        fog = np.clip(1.4 - z / 9.0, 0.3, 1.0) if not cam.ortho else 1.0
        f.pixels("w", sx[ok], sy[ok], (np.where(self.lat_major, 0.9, 0.4) * fog * gain)[ok])
        mj = self.lat_major & ok
        f.crosses("w", sx[mj], sy[mj], 4.5, 0.5 * gain)

    def draw(self, f, cam, hot_pts=(), hot_lev=None, gain=1.0, cloud=1.0, slices=1.0, top=False, reveal=None,
             dissolve=0.0, t=0.0, outline=0.0):
        """hot_pts: [(xyz, strength)] red glow on the cloud; hot_lev: per-level red amount (len(levels));
        reveal: (y_centre, half-height) only the slices inside are there (the body assembling);
        dissolve 0..1: the cloud drifts away and thins out; outline: dotted silhouette left behind."""
        if cloud > 0.01:
            P = self.pts
            keep = np.ones(len(P), bool)
            if dissolve > 0:
                keep = self.rnd > dissolve ** 0.7
                P = P + self.drift * (0.55 * dissolve ** 1.5) + np.array([0.0, 0.25 * dissolve ** 2, 0.0], np.float32)
            px, py, pz, pok = cam.project(P)
            if cam.ortho:
                facing = np.abs(self.nrm @ (-cam.R[2]))
            else:
                v = cam.pos.astype(np.float32)[None] - self.pts
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                facing = np.abs((self.nrm * v).sum(1))
            inten = (0.16 + 0.78 * (1 - facing) ** 2) * cloud * gain * (1.0 - 0.6 * dissolve)
            if top:
                inten = inten * 0.14
            red = np.zeros(len(P), np.float32)
            for hp, strength in hot_pts:
                d = np.linalg.norm(self.pts - np.asarray(hp, np.float32)[None], axis=1)
                red = np.maximum(red, np.exp(-(d / 0.07) ** 2) * strength)
            m = pok & keep
            f.pixels("w", px[m], py[m], (inten * (1 - 0.8 * red))[m])
            rm = m & (red > 0.05)
            if rm.any():
                f.pixels("r", px[rm], py[rm], 2.4 * red[rm] * gain)
        if slices > 0.01:
            ax, ay, az, aok = cam.project(self.ca)
            bx, by, bz, bok = cam.project(self.cb)
            okc = aok & bok
            ci = np.full(len(self.ca), (0.27 if not top else 0.09) * slices * gain, np.float32)
            if reveal is not None:
                yc, hh = reveal
                ylev = self.levels[self.clev]
                edge = np.clip((hh - np.abs(ylev - yc)) / 0.06, 0.0, 1.0)
                ci = ci * edge * (1.0 + 1.6 * np.clip(1.0 - (hh - np.abs(ylev - yc)) / 0.12, 0.0, 1.0))
            if dissolve > 0:
                ci = ci * np.where(hash01(self.clev, np.arange(len(self.clev)) // 5, 3) > dissolve * 1.15, 1.0, 0.0)
            hot = np.zeros(len(self.ca), np.float32) if hot_lev is None else np.asarray(hot_lev, np.float32)[self.clev]
            f.segments("w", ax[okc], ay[okc], bx[okc], by[okc], (ci * (1 - hot))[okc], width=1.3)
            hm = okc & (hot > 0.02) & (ci > 0.001)
            if hm.any():
                f.segments("r", ax[hm], ay[hm], bx[hm], by[hm], 1.4 * hot[hm] * gain, width=1.8)
        if outline > 0.01 and cam.ortho:
            self._outline(f, cam, outline)

    def _outline(self, f, cam, alpha):
        """Dotted silhouette: for every height the extreme screen x of each limb (ortho views)."""
        for l in LOFTS:
            ys = np.arange(l[1][0], l[1][-1], 0.012)
            cx, cz, rx, rz = _loft_at(l, ys)
            r_, u_ = cam.R[0].astype(np.float64), cam.R[1].astype(np.float64)
            ext = np.sqrt((rx * r_[0]) ** 2 + (rz * r_[2]) ** 2)
            c = np.stack([cx, ys, cz], 1) - cam.pos[None]
            sx = cam.cx + cam.scale * (c @ r_)
            sy = cam.cy - cam.scale * (c @ u_)
            for sgn in (-1.0, 1.0):
                f.dots("w", sx + sgn * cam.scale * ext, sy, 1.5, 0.75 * alpha)


def draw_muon(f, cam, m, a, col=None, gain=1.0, tag=True, lines=True):
    """One muon track at age `a` (s since it entered the view): head, ionisation runs, hit ring, data tag."""
    prog = min(1.0, a / m["dur"])
    fade = 1.0 if a < m["dur"] else math.exp(-(a - m["dur"]) / 0.3)
    A, B = m["a"], m["b"]
    head = A + (B - A) * prog
    sx, sy, z, ok = cam.project(np.stack([A, head]).astype(np.float32))
    if not ok.all():
        return
    hero = m["hero"]
    f.segments("r", sx[:1], sy[:1], sx[1:], sy[1:], (0.8 if hero else 0.55) * fade * gain, width=1.9 if hero else 1.3)
    for (u0, u1) in m["runs"]:
        if u0 >= prog:
            continue
        u1 = min(u1, prog)
        qx, qy, _, _ = cam.project(np.stack([A + (B - A) * u0, A + (B - A) * u1]).astype(np.float32))
        f.segments("r", qx[:1], qy[:1], qx[1:], qy[1:], 1.3 * fade * gain, width=3.0 if hero else 2.0)
    if prog < 1.0:
        f.dots("r", sx[1:], sy[1:], 3.6, 1.6 * gain)
        f.dots("w", sx[1:], sy[1:], 1.4, 1.0 * gain)
    if m["hit"] is None:
        return
    ah = a - m["dur"] * m["u_in"]
    if 0 <= ah < 0.8:
        hx, hy, _, _ = cam.project(m["hit"][None].astype(np.float32))
        u = ah / 0.8
        f.rings("r", [hx[0]], [hy[0]], [8 + 78 * (1 - (1 - u) ** 3)], 1.0 * (1 - u) ** 1.5 * gain, width=1.8)
        f.dots("w", [hx[0]], [hy[0]], 3.4 * (1 - u) + 1.2, 1.6 * (1 - u) * gain)
    if tag and hero and 0 <= ah < 2.4:
        hx, hy, _, _ = cam.project(m["hit"][None].astype(np.float32))
        x, y = float(hx[0]), float(hy[0])
        alpha = 1 - float(smoothstep(1.8, 2.4, ah))
        callout(f, x, y, m["part"], [f"MU{m['charge']} {m['E']:.3f} GEV", f"DE {m['dE'] * 1000:.1f} MEV"] if lines
                else [], col=col, prefer=1 if x >= cam.cx else -1, red=m["part"] == "HEART", alpha=alpha, age=ah)


# ----------------------------------------------------------------------------
# small 2D helpers for the micro views
# ----------------------------------------------------------------------------

def _jit(i, j, seed, amp):
    return (hash01(i, j, seed) - 0.5) * 2 * amp, (hash01(i, j, seed + 1) - 0.5) * 2 * amp


def _circ(ax, ay, bx, by, cx, cy):
    """Circumcentres of triangles (vectorised)."""
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy


class You(Scene):
    name = "you"

    def __init__(self, ctx, seed=33):
        super().__init__(ctx)
        self.P = Plan(ctx)
        self.t_you = sd.said("YOU", 72.483)
        self.t_echo = sd.said("An echo", 85.2)
        self.t_called = sd.said("It is called", 97.883)
        self.body = Body(seed)
        rng = np.random.default_rng(seed + 1)
        self._build_muons(rng)
        self._build_micro(np.random.default_rng(seed + 2))

    # ----------------------------------------------------------------- muons
    def _build_muons(self, rng):
        b = self.body
        mus = []
        # the one from the shower: straight down through the heart on the word
        mus.append(b.muon(rng, self.t_you, HEART.copy(), True, zen=0.05, speed=30.0))
        heroes = [(BAR0 + 0.06, "HEAD"), (BAR0 + 1.06, "HAND_L"), (BAR0 + BAR, "TORSO"), (BAR0 + BAR + 1.0, "LEG_R"),
                  (CUTS[0] + 0.1, "HEART"), (CUTS[0] + 1.0, "NECK"), (CUTS[0] + BAR, "HAND_R"),
                  (CUTS[0] + BAR + 1.0, "FOOT_L")]
        for th, name in heroes:
            mus.append(b.muon(rng, th, b.target_in(name, rng), True))
        # the one we follow down the scales: through the heart, on the bar of "An echo, from a distant past"
        self.echo = b.muon(rng, CUTS[1] + 0.12, HEART + np.array([0.0, 0.0, 0.0]), True, zen=0.16, speed=14.0)
        mus.append(self.echo)
        hero_t = np.array([m["t_hit"] for m in mus])
        k = 0
        while BAR0 + k * STEP < T_OUT:
            tk = BAR0 + k * STEP
            k += 1
            dens = 0.58 if tk < CUTS[2] else 0.4
            if rng.random() > dens or np.abs(hero_t - tk).min() < 0.09:
                continue
            if rng.random() < 0.55:
                tgt = b.target_in(PART_NAMES[rng.integers(0, len(PART_NAMES))], rng)
            else:
                a, r = rng.uniform(0, 2 * np.pi), 0.9 * math.sqrt(rng.random())
                tgt = np.array([r * math.cos(a), rng.uniform(0.3, 1.6), r * math.sin(a)])
            mus.append(b.muon(rng, tk, tgt, False))
        self.mus = mus
        self.mu_t = np.array([m["t"] for m in mus])
        # the rest of the rain: it falls on the whole wall, towers included (not hit-tested, just seen)
        n = int(95 * (T_OUT - BAR0))
        tr = np.sort(rng.uniform(BAR0, T_OUT, n))
        rad, az = 0.42 + 2.9 * np.sqrt(rng.random(n)), rng.uniform(0, 2 * np.pi, n)
        G = np.stack([rad * np.cos(az), np.zeros(n), rad * np.sin(az)], 1)
        th, ph = np.minimum(np.abs(rng.normal(0, 0.3, n)), 0.9), rng.uniform(0, 2 * np.pi, n)
        d = np.stack([np.sin(th) * np.cos(ph), -np.cos(th), np.sin(th) * np.sin(ph)], 1)
        Ln = 2.9 / np.cos(th)
        self.rain = dict(t=tr, a=(G - d * Ln[:, None]).astype(np.float32), b=G.astype(np.float32), dur=Ln / 22.0)
        self.hit_t = np.sort([m["t_hit"] for m in mus if m["hit"] is not None])

    # ----------------------------------------------------------------- views
    def _view(self, t):
        """(kind, name, index, progress in the view)."""
        if t < BAR0:
            return "build", "PERSPECTIVE // ACQUIRING", 0, (t - T_IN) / (BAR0 - T_IN)
        names = [("persp", "PERSPECTIVE"), ("ortho", "ORTHO_FRONT / ORTHO_SIDE"), ("thorax", "THORAX / ORTHO_TOP"),
                 ("bone", "BONE // 1E-2 M"), ("cells", "CELLS // 1E-5 M"), ("dna", "DNA // 1E-9 M"),
                 ("atoms", "WATER // 1E-10 M"), ("track", "TRACK // MU")]
        edges = [BAR0] + CUTS + [T_OUT]
        k = int(np.searchsorted(edges, t, side="right")) - 1
        k = min(max(k, 0), len(names) - 1)
        return names[k][0], names[k][1], k + 1, (t - edges[k]) / (edges[k + 1] - edges[k])

    def _cams(self, kind, t, u):
        """[(camera, clip rect, column for its tags, label)] - every view centred in a free bay."""
        P = self.P
        if kind in ("build", "persp"):
            yaw = math.radians(-26.0 + 11.0 * (t - T_IN))
            D = 4.95 - 0.2 * smoothstep(BAR0, CUTS[0], t)
            cam = Camera((D * math.sin(yaw), 1.22, D * math.cos(yaw)), (0.0, 0.92, 0.0), fov_deg=34.0,
                         screen_center=(P.fx, Y_MID + 12))
            return [(cam, WALL, P.fcol, None)]
        if kind == "ortho":
            h = Y_MID + 12
            sc = 455.0
            cf = OrthoCamera((0.0, 0.92, 10.0), (0.0, 0.92, 0.0), scale=sc, screen_center=(P.fx, h))
            if P.side is None:          # one bay only: front on the first bar, side on the second
                if (t - CUTS[0]) < BAR:
                    return [(cf, (P.fbay[0] + 4, Y_TOP, P.fbay[1] - 4, Y_BOT), P.fcol, "FRONT")]
                cs = OrthoCamera((10.0, 0.92, 0.0), (0.0, 0.92, 0.0), scale=sc, screen_center=(P.fx, h))
                return [(cs, (P.fbay[0] + 4, Y_TOP, P.fbay[1] - 4, Y_BOT), P.fcol, "SIDE")]
            sb = P.bay_of(P.side)
            cs = OrthoCamera((10.0, 0.92, 0.0), (0.0, 0.92, 0.0), scale=sc,
                             screen_center=(0.5 * (P.side[0] + P.side[1]), h))
            return [(cf, (P.fbay[0] + 4, Y_TOP, P.fbay[1] - 4, Y_BOT), P.fcol, "FRONT"),
                    (cs, (max(sb[0], L.FX0) + 4, Y_TOP, min(sb[1], L.FX1) - 4, Y_BOT), P.side, "SIDE")]
        fit = min(1.0, (2 * P.half - 120.0) / 1000.0)       # the thorax has to fit between two towers
        yaw = math.radians(-22.0 + 20.0 * u)
        D = (1.5 - 0.14 * u) / fit
        cam = Camera((D * math.sin(yaw) + 0.03, 1.36, D * math.cos(yaw)), (0.03, 1.24, 0.0), fov_deg=40.0,
                     screen_center=(P.fx, Y_MID))
        if P.side is None:
            return [(cam, WALL, P.fcol, None)]
        # the neighbouring bay: the same moment seen from above, the slices stacked into a target
        sb = P.bay_of(P.side)
        sc = min(0.5 * width(P.side) - 36.0, 0.5 * (Y_BOT - Y_TOP) - 86.0)          # px per metre
        ct = OrthoCamera((0.0, 10.0, 1e-3), (0.0, 0.0, 0.0), scale=sc, up=(0.0, 0.0, -1.0),
                         screen_center=(0.5 * (P.side[0] + P.side[1]), Y_MID + 16), roll_deg=18.0 * u)
        return [(cam, (P.fbay[0] + 4, Y_TOP, P.fbay[1] - 4, Y_BOT), P.fcol, None),
                (ct, (max(sb[0], L.FX0) + 4, Y_TOP, min(sb[1], L.FX1) - 4, Y_BOT), P.side, "TOP")]

    # ----------------------------------------------------------------- render
    def draw(self, f, t, ctx):
        t = float(np.clip(t, T_IN, T_OUT - 1e-3))
        P = self.P
        kind, vname, vidx, u = self._view(t)
        ages = t - self.mu_t
        opt = {}
        micro = kind in ("bone", "cells", "dna", "atoms", "track")
        if not micro:
            scan_y = None
            if kind == "ortho":
                w = ((t - CUTS[0]) / BAR) % 2.0
                scan_y = 0.03 + 1.76 * (1 - w if w < 1 else w - 1)
            for cam, clip, col, label in self._cams(kind, t, u):
                f.set_clip(*clip)
                self._draw_world(f, cam, t, ages, kind, scan_y, col, clip, label)
                if label:
                    f.text("w", col[1] - 6, Y_BOT - 14, label, size=L.T_SMALL, alpha=0.7, anchor="rs")
            f.set_clip()
        else:
            f.set_clip(*WALL)
            getattr(self, f"_micro_{kind}")(f, t, u)
            f.set_clip()
        if t >= BAR0:
            self._draw_title(f, t, ages, kind)
            self._draw_data(f, t, ages)
            self._draw_view_tag(f, kind, vname, vidx, t)
            if micro:
                self._draw_scale_strip(f, t, kind, u)
            else:
                self._draw_strip(f, t)
            self._draw_bottom(f, t, ages)
        else:
            self._draw_build_hud(f, t)
            opt["edge_alpha"] = 0.6
        # the drop: one short inverted flash
        if BAR0 <= t < BAR0 + 0.07:
            opt["invert"] = True
        return opt

    # ------------------------------------------------------------------ world
    def _hot(self, t, ages):
        """Red glow sources on the cloud + red amount per slice level, from the recent hits."""
        hot_pts = []
        hot_lev = np.zeros(len(self.body.levels), np.float32)
        for m, a in zip(self.mus, ages):
            if m["hit"] is None:
                continue
            ah = a - m["dur"] * m["u_in"]
            if 0 <= ah < 0.9:
                hot_pts.append((m["hit"], math.exp(-ah / 0.3)))
            if 0 <= ah < 0.6:
                k = int(np.argmin(np.abs(self.body.levels - m["hit"][1])))
                hot_lev[max(0, k - 1): k + 2] = np.maximum(hot_lev[max(0, k - 1): k + 2], math.exp(-ah / 0.25))
        return hot_pts, hot_lev

    def _draw_world(self, f, cam, t, ages, kind, scan_y, col, clip, label=None):
        b = self.body
        top = label == "TOP"
        hot_pts, hot_lev = self._hot(t, ages)
        if kind == "build":
            # drum roll: the slices come on around the heart, a step per 16th note
            k = max(0.0, (t - self.t_you) / STEP)
            hh = 0.05 + 0.1 * (math.floor(k) + min(1.0, (k % 1.0) / 0.25))
            b.draw(f, cam, hot_pts=hot_pts, hot_lev=hot_lev, cloud=0.0, slices=1.25 if t >= self.t_you else 0.0,
                   reveal=(HEART[1], hh))
            m0 = self.mus[0]
            a0 = t - m0["t"]
            if a0 >= 0:
                self._draw_persistent(f, cam, m0, a0, t, col)
            return
        b.draw_floor(f, cam)
        self.P.plates(f)            # the floor is the only thing that would run under the title / data columns
        if scan_y is not None:
            k = int(np.argmin(np.abs(b.levels - scan_y)))
            hot_lev[k] = 1.0
        b.draw(f, cam, hot_pts=hot_pts, hot_lev=hot_lev, top=top)
        if top:                     # range rings around you
            o = cam.project(np.zeros((1, 3), np.float32))
            X, Y = float(o[0][0]), float(o[1][0])
            rr = [r for r in (0.25, 0.5, 0.75, 1.0) if cam.scale * r < 0.5 * width(col) + 10]
            f.rings("w", [X] * len(rr), [Y] * len(rr), [cam.scale * r for r in rr], 0.32)
            for r in rr:
                f.text("w", X + cam.scale * r * 0.7071 + 6, Y + cam.scale * r * 0.7071 + 14, f"{r:.2f} M",
                       size=L.T_MICRO, alpha=0.7)
            f.segments("r", [clip[0], X], [Y, clip[1]], [clip[2], X], [Y, clip[3]], 0.3)
        self._draw_rain(f, cam, t)
        for m, a in zip(self.mus, ages):
            if a < 0 or a > m["dur"] + (2.6 if m["hero"] else 0.9) or m is self.echo:
                continue
            # between two close towers only the heart keeps its tag (the others would sit on the figure)
            draw_muon(f, cam, m, a, col=col, tag=not top and (self.P.half >= 300.0 or m["part"] == "HEART"),
                      lines=self.P.half >= 400.0)
        if kind == "thorax":
            a = t - self.echo["t"]
            if a >= 0:
                self._draw_persistent(f, cam, self.echo, a, t, col, lines=not top and self.P.half >= 400.0)
        if scan_y is not None:
            p = cam.project(np.array([[0.0, scan_y, 0.0]], np.float32))
            y = float(p[1][0])
            f.segments("r", [clip[0]], [y], [clip[2]], [y], 0.8, width=1.5)
            f.tag("r", col[0] + 6, y - 9, f"SLICE Y {scan_y:.3f} M", size=L.T_SMALL, pad=4)

    def _draw_rain(self, f, cam, t):
        r = self.rain
        i0, i1 = np.searchsorted(r["t"], t - 0.5), np.searchsorted(r["t"], t)
        if i1 <= i0:
            return
        age = t - r["t"][i0:i1]
        dur = r["dur"][i0:i1]
        prog = np.minimum(1.0, age / dur)
        A, B = r["a"][i0:i1], r["b"][i0:i1]
        Hd = A + (B - A) * prog[:, None].astype(np.float32)
        ax, ay, _, aok = cam.project(A)
        hx, hy, _, hok = cam.project(Hd)
        fade = np.where(age < dur, 1.0, np.exp(-(age - dur) / 0.1))
        ok = aok & hok
        f.segments("r", ax[ok], ay[ok], hx[ok], hy[ok], 0.1 * fade[ok], 0.42 * fade[ok])
        fl = ok & (age < dur)
        f.dots("r", hx[fl], hy[fl], 2.0, 1.1)
        gd = ok & (age >= dur) & (age < dur + 0.3)
        if gd.any():
            f.dots("r", hx[gd], hy[gd], 1.6, 1.0 * (1 - (age[gd] - dur[gd]) / 0.3))

    def _draw_persistent(self, f, cam, m, a, t, col, lines=True):
        """A hero track that stays on screen once it has passed (the one of 'YOU', the one of 'an echo')."""
        prog = min(1.0, a / m["dur"])
        A, B = m["a"], m["b"]
        sx, sy, _, ok = cam.project(np.stack([A, A + (B - A) * prog]).astype(np.float32))
        if not ok.all():
            return
        hold = 0.62 + 0.38 * math.exp(-max(0.0, a - m["dur"]) / 0.5)
        f.segments("r", sx[:1], sy[:1], sx[1:], sy[1:], 0.9 * hold, width=2.2)
        for (u0, u1) in m["runs"]:
            if u0 >= prog:
                continue
            qx, qy, _, _ = cam.project(np.stack([A + (B - A) * u0, A + (B - A) * min(u1, prog)]).astype(np.float32))
            f.segments("r", qx[:1], qy[:1], qx[1:], qy[1:], 1.5 * hold, width=3.4)
        if prog < 1.0:
            f.dots("r", sx[1:], sy[1:], 4.2, 1.7)
            f.dots("w", sx[1:], sy[1:], 1.6, 1.0)
        ah = a - m["dur"] * m["u_in"]
        if ah < 0:
            return
        hx, hy, _, _ = cam.project(m["hit"][None].astype(np.float32))
        x, y = float(hx[0]), float(hy[0])
        if ah < 1.0:
            uu = ah / 1.0
            f.rings("r", [x], [y], [10 + 120 * (1 - (1 - uu) ** 3)], (1 - uu) ** 1.5, width=2.0)
        pulse = 0.55 + 0.45 * math.exp(-self.ctx.cues.since_kick(t) / 0.1)
        f.rings("r", [x], [y], [15.0], 0.9 * pulse, width=1.8)
        f.dots("w", [x], [y], 2.6, 1.4)
        callout(f, x, y, "HEART", [f"MU{m['charge']} {m['E']:.3f} GEV", f"DE {m['dE'] * 1000:.1f} MEV",
                                   "FROM 15.2 KM UP"] if lines else [], col=col, dy=-56.0, red=True, age=ah,
                elbow=46.0)

    # ------------------------------------------------------------------ HUD
    def _draw_build_hud(self, f, t):
        """Before the drop: almost nothing. A line of status under the figure, typed on the roll."""
        a = t - self.t_you
        if a < 0:
            return
        P = self.P
        k = int(a / STEP)
        y = Y_BOT - 8
        msg = fit_text(["ACQUIRING // SUBJECT 01 // 1.80 M", "ACQUIRING // SUBJECT 01", "ACQUIRING"], width(P.fcol),
                       L.T_LABEL)
        f.text("w", P.fx, y, hud.typed(msg, a, cps=40), size=L.T_LABEL, alpha=0.85, anchor="ms")
        n = 14
        pitch = min(26.0, (width(P.fcol) - 20) / n)
        x0 = P.fx - n * pitch / 2
        xs = x0 + np.arange(n) * pitch
        f.rect("w", x0 - 6, y + 16, x0 + n * pitch - 2, y + 40, 0.6)
        on = np.arange(n) <= k
        f.rects("w", xs[on], y + 21, xs[on] + pitch - 8, y + 35, 0.95)

    def _draw_view_tag(self, f, kind, name, idx, t):
        P = self.P
        y0 = Y_TOP + 36
        opts = [f"VIEW {idx:02d} // {name}"]
        for sep in (" / ", " // "):
            if sep in name:
                opts.append(f"VIEW {idx:02d} // {name.split(sep)[0]}")
        opts.append(f"VIEW {idx:02d}")
        if kind in ("bone", "cells", "dna", "atoms", "track"):
            # the track leaves the top of the bay on the left of its centre: the tag goes on the right
            room = P.fcol[1] - (P.fx - 50)
            f.tag("w", P.fcol[1] - 6, y0, fit_text(opts, room, L.T_LABEL), size=L.T_LABEL, pad=5, anchor="rs")
        else:
            room = width(P.fcol) if kind == "ortho" else (P.fx - 80) - P.fcol[0]
            f.tag("w", P.fcol[0] + 6, y0, fit_text(opts, room, L.T_LABEL), size=L.T_LABEL, pad=5)
            if kind == "ortho":
                s = 455.0
                x0 = P.fcol[0] + 6
                if P.fx - 200 - x0 > 0.5 * s + 70:
                    f.segments("w", [x0, x0, x0 + 0.5 * s], [y0 + 34, y0 + 26, y0 + 26],
                               [x0 + 0.5 * s, x0, x0 + 0.5 * s], [y0 + 34, y0 + 42, y0 + 42], 0.9, width=L.LW)
                    f.text("w", x0 + 0.5 * s + 12, y0 + 40, "0.5 M", size=L.T_SMALL, alpha=0.8)
        # what the neighbouring bay is: the same rain, the same floor
        if P.side is not None and kind in ("persp", "ortho", "thorax"):
            info = fit_text(["YOU // MUON BLOOM // 1.80 M // EFFECTIVE AREA 0.38 M2", "YOU // 1.80 M // AREA 0.38 M2",
                             "YOU // 1.80 M"], width(P.side), L.T_SMALL)
            x1 = P.side[1] - 6
            f.occlude(x1 - text_w(info, L.T_SMALL) - 10, y0 - 22, x1 + 6, y0 + 34)
            f.text("w", x1, y0, info, size=L.T_SMALL, alpha=0.85, anchor="rs")
            f.text("w", x1, y0 + 26, f"{sd.tc(t)}  //  1 MUON /CM2 /MIN", size=L.T_MICRO, alpha=0.6, anchor="rs")

    def _draw_title(self, f, t, ages, kind):
        """YOU, the parameters of the view, the hit log (leftmost free column)."""
        P = self.P
        if P.title is None:
            return
        x0, x1 = P.title
        w = x1 - x0
        y0 = Y_TOP + 10
        ts = min(112.0, (w - 40) / (3 * 0.61))
        f.tag("w", x0 + 10, y0 + 0.93 * ts, "YOU", size=ts, pad=10, bold=True)
        if kind in self.MEDIA:
            rows_ = self.MEDIA[kind]
        else:
            rows_ = ["MU FLUX    1 /CM2/MIN", "THROUGH YOU   ~63 /S", "HEIGHT       1.800 M", "AREA_EFF     0.38 M2",
                     "DE/DX     2.0 MEV/CM"]
        yr = y0 + ts + 64
        hud.rows(f, x0, yr, rows_, size=L.T_SMALL, lead=1.5)
        yl = yr + 5 * 25.5 + 30
        full = w >= 300
        f.tag("w", x0 + 4, yl, "HIT_LOG", size=L.T_MICRO, pad=3)
        if full:
            f.text("w", x0 + 96, yl, "PART   X     Y    Z     MEV", size=L.T_MICRO, alpha=0.5)
        hits = [(a - m["dur"] * m["u_in"], m) for m, a in zip(self.mus, ages) if m["hit"] is not None]
        n_rows = int((Y_BOT - yl - 40) / 21)
        hits = sorted([h for h in hits if h[0] >= 0], key=lambda h: h[0])[:n_rows]
        for k, (ah, m) in enumerate(hits):
            hp = m["hit"]
            if full:
                line = f"{m['part']:<7}{hp[0]:+.2f} {hp[1]:.2f} {hp[2]:+.2f} {m['dE'] * 1000:5.1f}"
            else:
                line = f"{m['part']:<7} {hp[1]:.2f} M {m['dE'] * 1000:5.1f} MEV"
            f.text("r" if m["part"] == "HEART" or k == 0 else "w", x0, yl + 32 + k * 21, line, size=L.T_MICRO,
                   alpha=0.95 if k < 3 else 0.65)

    def _energy(self, ages):
        dep = {}
        for m, a in zip(self.mus, ages):
            if m["hit"] is None:
                continue
            ah = a - m["dur"] * m["u_in"]
            if ah >= 0:
                dep[m["part"]] = dep.get(m["part"], 0.0) + m["dE"] * 1000 * math.exp(-ah / 5.0)
        return dep

    PARTS = ["HEAD", "HEART", "TORSO", "ARM_L", "ARM_R", "HAND_L", "HAND_R", "LEG_L", "LEG_R", "FOOT_L"]

    def _draw_data(self, f, t, ages):
        """Energy left in each part of the body + what falls on the wall (the remaining column)."""
        P = self.P
        if P.data is None:
            return
        x0, x1 = P.data
        y = Y_TOP + 44
        hud.panel_header(f, x0, x1, y, "ENERGY LEFT IN YOU // MEV")
        dep = self._energy(ages)
        bw = x1 - x0 - 92 - 74
        for k, p in enumerate(self.PARTS):
            yy = y + 30 + k * 36
            v = dep.get(p, 0.0)
            f.text("r" if p == "HEART" and v > 1 else "w", x0 + 2, yy + 16, f"{p:<7}", size=L.T_SMALL, alpha=0.85)
            f.rects("r" if p == "HEART" else "w", x0 + 92, yy + 2, x0 + 92 + min(bw, v * bw / 260.0), yy + 17, 0.95)
            f.text("w", x1, yy + 16, f"{v:5.1f}", size=L.T_SMALL, alpha=0.8, anchor="rs")
        y2 = y + 30 + len(self.PARTS) * 36 + 56
        hud.panel_header(f, x0, x1, y2, "MEANWHILE // EVERY SQUARE METRE")
        s = min(64.0, (x1 - x0 - 10) / (6 * 0.61))
        f.text("w", x0, y2 + 34 + s, "167 /S", size=s, alpha=0.97)
        f.text("w", x0 + 2, y2 + 68 + s, "MUONS, DAY AND NIGHT", size=L.T_SMALL, alpha=0.8)
        f.text("w", x0 + 2, y2 + 94 + s, "1 /CM2 /MIN AT THE GROUND", size=L.T_MICRO, alpha=0.6)
        f.text("w", x0 + 2, y2 + 116 + s, "ROOFS AND WALLS DO NOT STOP THEM", size=L.T_MICRO, alpha=0.6)

    def _draw_strip(self, f, t):
        ta, tb = t - 5.5, t + 2.0
        x0, y0, x1, y1, yb = hud.strip_base(f, title="HIT_TIMELINE // WHITE = MUON  RED = ENERGY LEFT IN YOU",
                                            ticks=(ta - BAR0, tb - BAR0, STEP, BAR))
        X = lambda tt: x0 + (np.asarray(tt) - ta) / (tb - ta) * (x1 - x0)
        placed = []
        for m in self.mus:
            tt = m["t_hit"] if m["t_hit"] is not None else m["t"]
            if not (ta - 0.2 <= tt <= tb + 0.2) or tt < T_IN:
                continue
            xe = float(X(tt))
            key = int(m["t"] * 1000)
            n = 2 + int(hash01(key, 2) * 6)
            bx = xe + np.arange(n) * 4.0
            ok = (bx > x0) & (bx < x1 - 3)
            hh = (8 + 10 * math.log1p(m["E"])) * (0.3 + 0.7 * hash01(key, np.arange(n)))
            f.rects("w", bx[ok], y0 + 1, bx[ok] + 2, y0 + 1 + hh[ok], 0.95 if tt <= t else 0.4)
            if m["part"]:
                hb = (6 + 60 * m["dE"]) * (0.3 + 0.7 * hash01(key, np.arange(n) + 9))
                f.rects("r", bx[ok], y1 - hb[ok], bx[ok] + 2, y1 - 1, 0.9 if tt <= t else 0.4)
                if m["hero"] and x0 + 10 < xe < x1 - 110:
                    row = 1 if any(abs(p - xe) < 120 for p in placed) else 0
                    placed.append(xe)
                    f.tag("r" if tt <= t else "w", xe, yb + 26 + row * 24, m["part"], size=L.T_MICRO, pad=4,
                          alpha=1.0 if tt <= t else 0.5)
        hud.strip_cursor(f, float(X(t)), y0, y1, f"T {sd.tc(t)}")

    SCALES = {"bone": -2.0, "cells": -4.7, "dna": -8.7, "atoms": -9.5, "track": -15.0}
    MARKS = [(0.26, "YOU 1.8 M"), (-2.0, "RIB 1 CM"), (-4.7, "CELL 20 UM"), (-8.7, "DNA 2 NM"), (-9.5, "H2O 0.3 NM"),
             (-15.0, "NUCLEUS"), (-18.0, "MUON < 1E-18 M")]
    MEDIA = {
        "bone": ["MEDIUM  CORTICAL BONE", "RHO        1.92 G/CM3", "DE/DX     3.4 MEV/CM", "FIELD        ~10 CM",
                 "DEPTH       0.062 M"],
        "cells": ["MEDIUM      MYOCARDIUM", "CELL          ~20 UM", "DE/DX     0.20 KEV/UM", "FIELD         650 UM",
                  "ION PAIRS   ~6.6 /UM"],
        "dna": ["MEDIUM       CHROMATIN", "HELIX         2.0 NM", "TURN          3.4 NM", "FIELD         130 NM",
                "MEAN FREE    ~150 NM"],
        "atoms": ["MEDIUM   WATER  H2O", "O-H        0.096 NM", "IONISATION   12.6 EV", "FIELD          10 NM",
                  "MUON      POINT-LIKE"],
        "track": ["MEDIUM            ---", "CHARGE          -1 E", "SPIN             1/2", "SIZE     < 1E-18 M",
                  "NAME             ___"],
    }

    def _draw_scale_strip(self, f, t, kind, u):
        x0, y0, x1, y1, yb = hud.strip_base(f, title="SCALE // POWERS OF TEN ALONG THE TRACK // METRES",
                                            ticks=(1.0, -19.0, 0.2, 1.0))
        X = lambda e: x0 + (1.0 - np.asarray(e, np.float64)) / 20.0 * (x1 - x0)
        for e in range(0, -19, -3):
            f.text("w", float(X(e)) + 5, y0 + 32, f"1E{e:+03d}", size=L.T_MICRO, alpha=0.75)
        cur = self.SCALES[kind] - 0.1 * u
        for e, word in self.MARKS:
            xv = float(X(e))
            last = e <= -17.5
            red = last or abs(e - self.SCALES[kind]) < 0.3
            f.segments("w", [xv], [yb - 12], [xv], [yb + 12], 0.9)
            f.tag("r" if red else "w", xv - (text_w(word, L.T_MICRO) + 8 if last else -4),
                  yb + (48 if e in (-9.5, -15.0) else 26), word, size=L.T_MICRO, pad=3,
                  alpha=1.0 if e >= cur - 0.3 or last else 0.45)
        # every decade we went through leaves a block on the top comb
        dec = np.arange(0.0, cur, -0.2)
        xd = X(dec)
        f.rects("w", xd, y0 + 1, xd + 3, y0 + 14 + 16 * hash01(np.arange(len(dec)), 4), 0.9)
        hud.strip_cursor(f, float(X(cur)), y0, y1, f"FIELD 1E{cur:+06.2f} M")

    def _draw_bottom(self, f, t, ages):
        """Counts on every time scale, hit barcode (+ energy if there was no column for it): the free
        panels of the bottom band, widest first."""
        P = self.P
        y0, y1 = P.py0, P.py1
        panels = [p for p in P.panels if width(p) >= 200.0]
        if panels:
            cx0, cx1 = panels[0]
            hud.panel_header(f, cx0, cx1, y0, "MUONS THROUGH YOU")
            cols = [("PER SECOND", "63"), ("PER MINUTE", "3 780"), ("PER DAY", "5.4 M"), ("IN A LIFE", "159 BN")]
            n = int(min(4, max(1, (cx1 - cx0) // 180)))
            cols = {4: cols, 3: [cols[0], cols[1], cols[3]], 2: [cols[0], cols[3]], 1: [cols[0]]}[n]
            cw = (cx1 - cx0) / n
            fs = min(52.0, (cw - 12) / (6 * 0.61))
            for k, (lab, val) in enumerate(cols):
                xx = cx0 + k * cw
                f.text("w", xx + 4, y0 + 36, lab, size=L.T_MICRO, alpha=0.75)
                f.text("r" if lab == "IN A LIFE" else "w", xx + 2, y0 + 100, val, size=fs, alpha=0.97)
        if len(panels) > 1:
            bx0, bx1 = panels[1]
            hud.panel_header(f, bx0, bx1, y0, "HIT_BARCODE")
            n = int(np.clip((bx1 - bx0) / 3.9, 40, 200))
            dt = 3.0 / n
            kf = math.floor((t - 3.0) / dt)
            kk = kf + np.arange(n)
            tt = kk * dt
            lo = np.searchsorted(self.hit_t, tt)
            hi = np.searchsorted(self.hit_t, tt + dt * 3)
            dens = np.where(tt >= self.t_you - 0.1, 0.05 + 0.9 * np.tanh((hi - lo) / 1.5), 0.0)
            hud.barcode_lanes(f, bx0, bx1, y0 + 12, y1, dens, kk, lanes=3, seed=7)
        if len(panels) > 2 and P.data is not None:
            self._draw_sequence(f, panels[2], t)
        if len(panels) > 2 and P.data is None and width(panels[2]) >= 330:
            ex0, ex1 = panels[2]
            hud.panel_header(f, ex0, ex1, y0, "ENERGY LEFT IN YOU // MEV")
            dep = self._energy(ages)
            ncol = 2 if ex1 - ex0 >= 620 else 1
            cw = (ex1 - ex0) / ncol
            for k, p in enumerate(self.PARTS[: 5 * ncol]):
                c, r = k // 5, k % 5
                xx, yy = ex0 + c * cw, y0 + 20 + r * 22
                v = dep.get(p, 0.0)
                f.text("r" if p == "HEART" and v > 1 else "w", xx + 2, yy + 14, f"{p:<7}", size=L.T_MICRO, alpha=0.85)
                f.rects("r" if p == "HEART" else "w", xx + 78, yy + 3, xx + 78 + min(cw - 150, v * 1.2), yy + 14, 0.95)
                f.text("w", xx + cw - 14, yy + 14, f"{v:5.1f}", size=L.T_MICRO, alpha=0.8, anchor="rs")

    SEQ = ["PERSP", "ORTHO", "THRX", "BONE", "CELL", "DNA", "H2O", "TRACK"]

    def _draw_sequence(self, f, panel, t):
        """Where we are in the zoom: one box per view, the current one red."""
        x0, x1 = panel
        hud.panel_header(f, x0, x1, self.P.py0, "SEQUENCE // 1.8 M -> 1E-18 M")
        _, _, idx, u = self._view(t)
        n = len(self.SEQ)
        w = (x1 - x0) / n
        y = self.P.py0 + 24
        for k, name in enumerate(self.SEQ):
            xx = x0 + k * w
            f.rect("w", xx + 2, y, xx + w - 5, y + 40, 0.7)
            if k + 1 < idx:
                f.rects("w", xx + 6, y + 4, xx + w - 9, y + 36, 0.9)
            elif k + 1 == idx:
                f.rects("r", xx + 6, y + 4, xx + 6 + (w - 15) * max(0.08, u), y + 36, 0.95)
            if w >= 46:
                f.text("r" if k + 1 == idx else "w", xx + 3, y + 64, name, size=L.T_MICRO,
                       alpha=0.95 if k + 1 <= idx else 0.5)
        f.text("w", x0 + 3, y + 98, f"VIEW {idx:02d} / {n:02d}", size=L.T_SMALL, alpha=0.8)

    # ------------------------------------------------------------------------------
    # micro views: the same straight track, the matter around it at smaller and smaller scales.
    # The field fills the wall (it runs behind the towers); the track, its ticks and its tags stay in the
    # focus bay; the read-outs sit on plates in the side column.
    # ------------------------------------------------------------------------------
    TILT = math.radians(11.0)

    def _build_micro(self, rng):
        self.fib_ph = rng.uniform(0, 2 * np.pi, (110, 3))
        self.fib_y = np.sort(rng.uniform(-1.02, 0.95, 110))
        self.skin = rng.uniform(0, 1, (3200, 2))
        self.marrow = rng.uniform(-1, 1, (1500, 2))
        self.water = rng.uniform(0, 1, (1400, 2))
        self.helix = [(-34.0, -7.0, 0.42, 0.3), (10.0, 9.0, -0.3, 1.9), (40.0, -12.0, 1.25, 4.0),
                      (-16.0, 13.0, 2.6, 2.2), (24.0, -2.0, -1.2, 0.9), (-52.0, 4.0, -0.9, 3.1),
                      (58.0, 10.0, 0.2, 5.0)]                                  # x, y (nm), angle, phase
        self.delta = np.cumsum(rng.normal(0, 1, (26, 2)) * np.array([1.0, 0.6]) + np.array([0.55, -0.3]), 0)

    def _track_pt(self, s):
        """Screen point at distance s (px) along the track from the centre of the view."""
        return self.P.fx + math.sin(self.TILT) * s, Y_MID + math.cos(self.TILT) * s

    def _track(self, f, t, tick=None, gain=1.0, width_=2.2, layer="r"):
        """The red line across the view, its ticks, and a pulse running down it on every beat."""
        x0, y0 = self._track_pt(-560.0)
        x1, y1 = self._track_pt(560.0)
        f.segments(layer, [x0], [y0], [x1], [y1], 1.0 * gain, width=width_)
        if tick:
            s = np.arange(-552.0, 552.0, tick)
            px, py = self._track_pt(s)
            nx, ny = math.cos(self.TILT), -math.sin(self.TILT)
            ln = np.where(np.arange(len(s)) % 5 == 0, 11.0, 6.0)
            f.segments(layer, px - nx * ln, py - ny * ln, px + nx * ln, py + ny * ln, 0.85 * gain)
        beat = BAR / 4
        ph = ((t - BAR0) % beat) / beat
        px, py = self._track_pt(-560.0 + 1120.0 * ph)
        f.dots("r", [px], [py], 5.0, 1.8 * gain)
        f.dots("w", [px], [py], 1.8, 1.2 * gain)

    def _scale_bar(self, f, ppu, length, label):
        """Bottom left of the focus bay (the track leaves it on the right)."""
        x0, y = self.P.fcol[0] + 12 + text_w(label, L.T_SMALL) + 14, Y_BOT - 40
        x1 = x0 + length * ppu
        f.occlude(self.P.fcol[0] + 4, y - 22, x1 + 14, y + 22)
        f.segments("w", [x0, x0, x1], [y, y - 8, y - 8], [x1, x0, x1], [y, y + 8, y + 8], 0.95, width=L.LW)
        f.text("w", x0 - 12, y + 6, label, size=L.T_SMALL, alpha=0.9, anchor="rs")

    def _lattice(self, f):
        """Faint measuring lattice over every micro view. Called once the field is drawn: it also lays the
        black plates outside the field window."""
        gx = np.arange(WALL[0] + 56, WALL[2] - 20, 120.0)
        gy = np.arange(WALL[1] + 55, WALL[3] - 20, 120.0)
        X, Y = np.meshgrid(gx, gy)
        f.crosses("w", X.ravel(), Y.ravel(), 5.0, 0.22)
        self.P.plates(f)

    def _readouts(self, f, entries, y=None):
        """Read-out list in the side column, on ONE outlined plate: entries = [(title, lines, red, age)]."""
        P = self.P
        if P.side is None:
            return
        x0, x1 = P.side
        size, lsize = L.T_TAG, L.T_SMALL
        y = y if y is not None else Y_TOP + 84
        cpl = int((x1 - x0 - 34) / (lsize * 0.61))
        hs = [size * 1.5 + len(ls) * lsize * 1.5 + 28 for _, ls, _, _ in entries]
        w = min(x1 - x0, max(text_w(ln[:cpl], lsize) for _, ls, _, _ in entries for ln in ls) + 44)
        f.occlude(x0, y - 46, x0 + w, y - 46 + sum(hs) + 14)
        f.rect("w", x0, y - 46, x0 + w, y - 46 + sum(hs) + 14, 0.4)
        for (title, lines, red, age), h in zip(entries, hs):
            f.tag("r" if red else "w", x0 + 18, y, title, size=size, pad=5)
            for k, ln in enumerate(lines):
                f.text("w", x0 + 14, y + size * 0.5 + (k + 1) * lsize * 1.5,
                       hud.typed(ln[:cpl], max(age, 0.0), cps=160, delay=0.05 + 0.05 * k), size=lsize, alpha=0.9)
            y += h

    # -- bone ------------------------------------------------------------------------
    def _micro_bone(self, f, t, u):
        P = self.P
        fit = min(1.0, (P.half - 30.0) / 361.0)              # the rib has to fit between two towers
        ppu = 300.0 * fit * (1.0 + 0.16 * u)                 # px per cm
        cx, cy = P.fx, Y_MID
        X = lambda wx: cx + np.asarray(wx) * ppu
        Y = lambda wy: cy + np.asarray(wy) * ppu
        wx = np.linspace((WALL[0] - cx) / ppu, (WALL[2] - cx) / ppu, 320)

        def wavy(y, k=0):
            return y + 0.028 * np.sin(1.3 * wx + k) + 0.012 * np.sin(3.7 * wx + 2.1 * k)

        ex, ey, ea, eb = 0.0, -0.06, 0.98, 0.6    # the rib
        inside = lambda x, y, s=1.0: ((x - ex) / (ea * s)) ** 2 + ((y - ey) / (eb * s)) ** 2 < 1.0
        # skin
        f.polyline("w", X(wx), Y(wavy(-1.5, 0)), 0.95, width=L.LW_BOLD)
        f.polyline("w", X(wx), Y(wavy(-1.34, 1)), 0.6, width=L.LW)
        sx_ = wx[0] + self.skin[:, 0] * (wx[-1] - wx[0])
        f.pixels("w", X(sx_), Y(-1.49 + 0.14 * self.skin[:, 1]), 0.55)
        # fat: lobules
        ni = int((wx[-1] - wx[0]) / 0.125 / 2) + 2
        i, j = np.meshgrid(np.arange(-ni, ni + 1), np.arange(0, 3))
        jx, jy = _jit(i, j, 5, 0.035)
        fx_, fy_ = i * 0.125 + (j % 2) * 0.06 + jx, -1.27 + j * 0.105 + jy
        f.rings("w", X(fx_.ravel()), Y(fy_.ravel()), 0.05 * ppu, 0.3)
        f.polyline("w", X(wx), Y(wavy(-1.02, 2)), 0.6, width=L.LW)
        # muscle fibres, parting around the bone
        for k, y0 in enumerate(self.fib_y):
            ph = self.fib_ph[k]
            yy = y0 + 0.02 * np.sin(2.2 * wx + ph[0]) + 0.008 * np.sin(7.0 * wx + ph[1])
            push = np.exp(-((wx - ex) / (ea * 1.25)) ** 2)
            yy = yy + np.sign(y0 - ey + 1e-6) * push * np.clip(eb * 1.12 - np.abs(y0 - ey), 0, None)
            m = ~inside(wx, yy, 1.08)
            xs, ys = X(wx), Y(yy)
            seg = m[:-1] & m[1:]
            f.segments("w", xs[:-1][seg], ys[:-1][seg], xs[1:][seg], ys[1:][seg], 0.2 + 0.16 * hash01(k, 3))
        # bone: cortical shell, trabecular sponge, marrow
        a = np.linspace(0, 2 * np.pi, 200)
        f.polyline("w", X(ex + ea * np.cos(a)), Y(ey + eb * np.sin(a)), 1.0, width=L.LW_BOLD + 0.6)
        f.polyline("w", X(ex + ea * 0.8 * np.cos(a)), Y(ey + eb * 0.74 * np.sin(a)), 0.75, width=L.LW)
        ah = np.linspace(0, 2 * np.pi, 150, endpoint=False)
        f.segments("w", X(ex + ea * 0.97 * np.cos(ah)), Y(ey + eb * 0.97 * np.sin(ah)), X(ex + ea * 0.83 * np.cos(ah)),
                   Y(ey + eb * 0.77 * np.sin(ah)), 0.5)
        i, j = np.meshgrid(np.arange(-9, 10), np.arange(-6, 7))
        jx, jy = _jit(i, j, 21, 0.03)
        nx_, ny_ = ex + i * 0.085 + jx, ey + j * 0.085 + jy
        ok = inside(nx_, ny_, 0.76)
        for di, dj, sd_ in ((1, 0, 31), (0, 1, 32), (1, 1, 33)):
            a_ = (slice(None, -dj or None), slice(None, -di or None))
            b_ = (slice(dj, None), slice(di, None))
            link = ok[a_] & ok[b_] & (hash01(i[a_], j[a_], sd_) < (0.72 if sd_ < 33 else 0.3))
            f.segments("w", X(nx_[a_][link]), Y(ny_[a_][link]), X(nx_[b_][link]), Y(ny_[b_][link]), 0.6, width=1.3)
        f.dots("w", X(nx_[ok]), Y(ny_[ok]), 1.8, 0.8)
        mk = inside(ex + self.marrow[:, 0] * ea, ey + self.marrow[:, 1] * eb, 0.74)
        f.pixels("w", X(ex + self.marrow[mk, 0] * ea), Y(ey + self.marrow[mk, 1] * eb), 0.45)
        # pleura + lung
        f.polyline("w", X(wx), Y(wavy(0.95, 3)), 0.8, width=L.LW)
        f.polyline("w", X(wx), Y(wavy(1.0, 3)), 0.55)
        ni = int((wx[-1] - wx[0]) / 0.09 / 2) + 2
        i, j = np.meshgrid(np.arange(-ni, ni + 1), np.arange(0, 7))
        jx, jy = _jit(i, j, 9, 0.02)
        lx, ly = i * 0.09 + (j % 2) * 0.045 + jx, 1.07 + j * 0.08 + jy
        vis = (Y(ly) < Y_BOT + 20).ravel()
        f.rings("w", X(lx.ravel()[vis]), Y(ly.ravel()[vis]), 0.036 * ppu, 0.26)
        # the track
        self._lattice(f)
        self._track(f, t, tick=15.0)
        x0_, y0_ = self._track_pt(-0.63 * ppu)
        x1_, y1_ = self._track_pt(0.52 * ppu)
        f.rings("r", [x0_, x1_], [y0_, y1_], [11.0, 11.0], 1.0, width=1.8)
        f.segments("r", [x0_], [y0_], [x1_], [y1_], 1.2, width=4.0)
        age = t - CUTS[2]
        xe, ye = self._track_pt(0.82 * ppu)
        callout(f, xe, ye, "RIB_3", ["PATH 1.15 CM", "DE 3.9 MEV", "~130 000 IONS"], col=P.fcol, dy=44.0, red=True,
                age=age)
        # tissues on the track + their energy loss: rows at the height of each tissue, in the side column
        if P.side is not None:
            sx0, sx1 = P.side
            f.tag("w", sx0 + 6, Y_TOP + 36, "ON THE TRACK // RHO G/CM3 // DE/DX MEV/CM", size=L.T_MICRO, pad=3)
            bw = max(60.0, sx1 - sx0 - 330.0)
            for y_, name, rho, de in ((-1.18, "FAT", "0.95", 1.8), (-0.5, "MUSCLE", "1.05", 2.1), (ey, "BONE", "1.92", 3.4),
                                      (1.3, "LUNG", "0.26", 0.5)):
                yy = float(Y(y_))
                if not (Y_TOP + 70 < yy < Y_BOT - 30):
                    continue
                f.occlude(sx0 - 10, yy - 24, sx0 + 266 + de / 3.4 * bw, yy + 24)
                f.segments("w", [sx0 - 8], [yy], [sx0 + 10], [yy], 0.9, width=L.LW)
                f.tag("r" if name == "BONE" else "w", sx0 + 20, yy + 8, name, size=L.T_LABEL, pad=4)
                f.text("w", sx0 + 130, yy + 7, rho, size=L.T_SMALL, alpha=0.8)
                f.rects("r" if name == "BONE" else "w", sx0 + 200, yy - 7, sx0 + 200 + de / 3.4 * bw, yy + 7, 0.95)
                f.text("w", sx0 + 210 + de / 3.4 * bw, yy + 7, f"{de:.1f}", size=L.T_SMALL, alpha=0.85)
        self._scale_bar(f, ppu, 1.0, "1 CM")

    # -- cells -----------------------------------------------------------------------
    def _micro_cells(self, f, t, u):
        P = self.P
        ppu = 4.6 * (1.0 + 0.16 * u)              # px per um
        a = 20.0
        cx, cy = P.fx, Y_MID
        half_w = max(cx - WALL[0], WALL[2] - cx)
        nj = int((WALL[3] - WALL[1]) / ppu / (a * 0.866) / 2) + 3
        ni = int(half_w / ppu / a) + nj // 2 + 4
        i, j = np.meshgrid(np.arange(-ni, ni + 1), np.arange(-nj, nj + 1))
        jx, jy = _jit(i, j, 41, 0.2 * a)
        px = (i + 0.5 * j) * a + jx
        py = j * a * 0.866 + jy
        # Voronoi from the (jittered) triangular lattice: circumcentres of its two triangle families
        A = (slice(None, -1), slice(None, -1))
        Bx, By = px[:-1, 1:], py[:-1, 1:]           # P(i+1, j)
        Cx, Cy = px[1:, :-1], py[1:, :-1]           # P(i, j+1)
        Dx, Dy = px[1:, 1:], py[1:, 1:]             # P(i+1, j+1)
        c1x, c1y = _circ(px[A], py[A], Bx, By, Cx, Cy)
        c2x, c2y = _circ(Bx, By, Dx, Dy, Cx, Cy)
        S = lambda wx, wy: (cx + wx * ppu, cy + wy * ppu)
        for x0, y0, x1, y1 in ((c1x, c1y, c2x, c2y), (c2x[:, :-1], c2y[:, :-1], c1x[:, 1:], c1y[:, 1:]),
                               (c2x[:-1, :], c2y[:-1, :], c1x[1:, :], c1y[1:, :])):
            X0, Y0 = S(x0.ravel(), y0.ravel())
            X1, Y1 = S(x1.ravel(), y1.ravel())
            f.segments("w", X0, Y0, X1, Y1, 0.5, width=1.3)
        NX, NY = S(px.ravel(), py.ravel())
        nr = (2.6 + 1.2 * hash01(i, j, 43)).ravel() * ppu
        vis = (NX > WALL[0] - 30) & (NX < WALL[2] + 30) & (NY > WALL[1] - 30) & (NY < WALL[3] + 30)
        ox, oy = _jit(i, j, 47, 2.2)
        f.rings("w", NX[vis] + ox.ravel()[vis] * ppu, NY[vis] + oy.ravel()[vis] * ppu, nr[vis], 0.42)
        f.dots("w", NX[vis] + ox.ravel()[vis] * ppu, NY[vis] + oy.ravel()[vis] * ppu, 1.8, 0.7)
        # cells the track goes through = nearest seed of the points of the track
        s = np.arange(-560.0, 560.0, 4.0)
        tx_, ty_ = self._track_pt(s)
        near_ok = np.nonzero(np.abs(NX - cx) < 420)[0]
        d2 = (tx_[:, None] - NX[None, near_ok]) ** 2 + (ty_[:, None] - NY[None, near_ok]) ** 2
        near = near_ok[np.argmin(d2, 1)]
        order, first = [], {}
        for k, n_ in enumerate(near):
            if n_ not in first:
                first[n_] = k
                order.append(n_)
        path = {n_: int((near == n_).sum()) * 4.0 / ppu for n_ in order}
        self._lattice(f)
        beat = BAR / 4
        lead = -560.0 + 1120.0 * (((t - BAR0) % beat) / beat)
        ii, jj = i.ravel(), j.ravel()
        I0, J0 = int(i.min()), int(j.min())
        tags = []
        for rank, n_ in enumerate(order):
            r, c = jj[n_] - J0, ii[n_] - I0
            if not (1 <= r < c1x.shape[0] and 1 <= c < c1x.shape[1]):
                continue
            vx = np.array([c1x[r, c], c2x[r, c - 1], c1x[r, c - 1], c2x[r - 1, c - 1], c1x[r - 1, c], c2x[r - 1, c]])
            vy = np.array([c1y[r, c], c2y[r, c - 1], c1y[r, c - 1], c2y[r - 1, c - 1], c1y[r - 1, c], c2y[r - 1, c]])
            VX, VY = S(vx, vy)
            s_mid = s[first[n_]] + path[n_] * ppu / 2
            glow = math.exp(-max(0.0, lead - s_mid) / 260.0) if lead >= s_mid - 30 else 0.25
            f.polyline("r", VX, VY, 0.5 + 0.9 * glow, width=2.4, closed=True)
            f.rings("r", [NX[n_] + ox.ravel()[n_] * ppu], [NY[n_] + oy.ravel()[n_] * ppu], [nr[n_]], 0.5 + 0.6 * glow)
            if rank in (1, 5, 9) and Y_TOP + 130 < NY[n_] < Y_BOT - 150:
                tags.append((rank, n_))
        self._track(f, t, width_=2.6)
        for q, (rank, n_) in enumerate(tags):
            side = 1 if q % 2 == 0 else -1
            callout(f, float(NX[n_]) + side * 34, float(NY[n_]), f"CELL {rank + 1:02d}",
                    [f"PATH {path[n_]:.1f} UM", f"DE {path[n_] * 0.2:.2f} KEV", f"{int(path[n_] * 6.6):d} ION PAIRS"],
                    col=P.fcol, prefer=side, dy=-46.0 * side, red=True, age=t - CUTS[3] - 0.15 * q, elbow=36.0)
        # ion pairs per crossed cell, in the order the muon met them (side column)
        if P.side is not None:
            sx0, sx1 = P.side
            yt = Y_TOP + 96
            n_show = min(len(order), 22)
            f.occlude(sx0, yt - 58, sx1, yt + n_show * 17 + 136)
            f.rect("w", sx0, yt - 58, sx1, yt + n_show * 17 + 136, 0.4)
            sx0, sx1 = sx0 + 16, sx1 - 16
            f.tag("w", sx0 + 6, yt - 26, fit_text(["ION PAIRS PER CELL // IN THE ORDER IT MET THEM",
                                                   "ION PAIRS PER CELL"], sx1 - sx0, L.T_MICRO), size=L.T_MICRO, pad=3)
            bw = sx1 - sx0 - 60
            for rank, n_ in enumerate(order[:n_show]):
                v = path[n_] * 6.6
                f.rects("r", sx0 + 44, yt + rank * 17, sx0 + 44 + min(bw, v * bw / 190.0), yt + rank * 17 + 10, 0.95)
                f.text("w", sx0 + 34, yt + rank * 17 + 11, f"{rank + 1:02d}", size=L.T_MICRO, alpha=0.7, anchor="rs")
            yn = yt + n_show * 17 + 44
            for k, ln in enumerate([f"{len(order)} CELLS ON THE TRACK", f"{sum(path.values()) * 6.6:.0f} ION PAIRS",
                                    "NO CELL NOTICED"]):
                f.text("r" if k == 2 else "w", sx0 + 6, yn + k * 32, ln, size=L.T_TAG, alpha=0.92)
        self._scale_bar(f, ppu, 50.0, "50 UM")

    # -- dna ---------------------------------------------------------------------------
    def _micro_dna(self, f, t, u):
        P = self.P
        ppu = 23.0 * (1.0 + 0.16 * u)             # px per nm
        cx, cy = P.fx, Y_MID
        f.pixels("w", WALL[0] + self.water[:, 0] * (WALL[2] - WALL[0]),
                 WALL[1] + self.water[:, 1] * (WALL[3] - WALL[1]), 0.4)
        for k, (hx, hy, ang, ph) in enumerate(self.helix):
            s = np.arange(-110.0, 110.0, 0.085)                  # nm along the axis
            th = 2 * np.pi * s / 3.4 + ph + 0.5 * t * (1 if k % 2 else -1)
            ca, sa = math.cos(ang), math.sin(ang)
            axx, axy = hx + ca * s, hy + sa * s
            vis = (np.abs(cy + axy * ppu - Y_MID) < 520) & (cx + axx * ppu > WALL[0] - 40) & (cx + axx * ppu < WALL[2] + 40)
            if vis.sum() < 4:
                continue
            s, th, axx, axy = s[vis], th[vis], axx[vis], axy[vis]
            brk = np.diff(s) > 0.2
            for dphi in (0.0, 2.44):
                off = np.cos(th + dphi)
                depth = 0.55 + 0.45 * np.sin(th + dphi)
                X_ = cx + (axx - sa * off) * ppu
                Y_ = cy + (axy + ca * off) * ppu
                ok = ~brk
                f.segments("w", X_[:-1][ok], Y_[:-1][ok], X_[1:][ok], Y_[1:][ok], 0.85 * depth[:-1][ok],
                           0.85 * depth[1:][ok], width=1.8)
            r = np.arange(0, len(s), 4)
            o1, o2 = np.cos(th[r]), np.cos(th[r] + 2.44)
            f.segments("w", cx + (axx[r] - sa * o1) * ppu, cy + (axy[r] + ca * o1) * ppu,
                       cx + (axx[r] - sa * o2) * ppu, cy + (axy[r] + ca * o2) * ppu, 0.42)
        self._lattice(f)
        age = t - CUTS[4]
        self._track(f, t, width_=2.0)
        # one ionisation in the field: on average they are 150 nm apart
        px, py = self._track_pt(150.0)
        a = age % (BAR / 2)
        uu = min(1.0, a / 0.9)
        f.rings("r", [px], [py], [9 + 60 * (1 - (1 - uu) ** 3)], (1 - uu) ** 1.4, width=1.8)
        f.dots("r", [px], [py], 5.0, 1.6)
        n = int(min(len(self.delta), 3 + a * 40))
        dx_, dy_ = px + self.delta[:n, 0] * 9.0, py + self.delta[:n, 1] * 9.0
        f.polyline("r", np.r_[px, dx_], np.r_[py, dy_], 0.9, width=1.6)
        f.dots("r", dx_[-1:], dy_[-1:], 3.0, 1.5)
        hxp, hyp = cx + (10.0 - 6.0 * math.cos(-0.3)) * ppu, cy + (9.0 - 6.0 * math.sin(-0.3) + 1.0) * ppu
        solo = P.side is None           # no list beside the view: the tags carry the data themselves
        callout(f, hxp, hyp, "DNA", ["HELIX 2.0 NM", "3.4 NM A TURN"] if solo else [], col=P.fcol, prefer=1, dy=70.0,
                age=age)
        callout(f, px, py, "ION PAIR", ["~33 EV", "NEXT +148 NM"] if solo else [], col=P.fcol, prefer=-1, dy=56.0,
                red=True, age=age - 0.3)
        qx, qy = self._track_pt(-330.0)
        callout(f, qx, qy, "MU-", ["BETWEEN THE TURNS", "MFP ~150 NM"] if solo else [], col=P.fcol, prefer=1, dy=-40.0,
                red=True, age=age - 0.6)
        self._readouts(f, [
            ("DNA", ["DOUBLE HELIX // 2.0 NM WIDE", "3.4 NM PER TURN // 10 BASE PAIRS", "2 M OF IT IN EVERY CELL"],
             False, age),
            ("ION PAIR", ["ONE ELECTRON SET FREE // ~33 EV", "THE NEXT ONE: 148 NM FURTHER"], True, age - 0.3),
            ("MU-", ["PASSES BETWEEN THE TURNS", "MEAN FREE PATH ~150 NM", "THE HELIX DOES NOT NOTICE"], True,
             age - 0.6)])
        self._scale_bar(f, ppu, 10.0, "10 NM")

    # -- atoms -------------------------------------------------------------------------
    def _micro_atoms(self, f, t, u):
        P = self.P
        ppu = 290.0 * (1.0 + 0.16 * u)            # px per nm
        cx, cy = P.fx, Y_MID
        a = 0.31
        half_w = max(cx - WALL[0], WALL[2] - cx)
        ni, nj = int(half_w / ppu / a) + 2, int((WALL[3] - WALL[1]) / ppu / a / 2) + 2
        i, j = np.meshgrid(np.arange(-ni, ni + 1), np.arange(-nj, nj + 1))
        jx, jy = _jit(i, j, 71, 0.085)
        wob = 0.012 * np.sin(t * 9.0 + 6.28 * hash01(i, j, 73))
        ox_, oy_ = (i + 0.5 * (j % 2)) * a + jx + wob, j * a * 0.9 + jy + wob[::-1]
        ang = 2 * np.pi * hash01(i, j, 75) + 0.5 * np.sin(t * 2.0 + 6.28 * hash01(i, j, 77))
        OX, OY = cx + ox_.ravel() * ppu, cy + oy_.ravel() * ppu
        ang = ang.ravel()
        # the molecule on the track gets ionised
        tpx, tpy = self._track_pt(110.0)
        hit = int(np.argmin((OX - tpx - 34) ** 2 + (OY - tpy) ** 2))
        for sgn in (-1.0, 1.0):
            hx_ = OX + 0.096 * ppu * np.cos(ang + sgn * 0.912)
            hy_ = OY + 0.096 * ppu * np.sin(ang + sgn * 0.912)
            f.segments("w", OX, OY, hx_, hy_, 0.55, width=1.3)
            f.dots("w", hx_, hy_, 3.2, 0.85)
        f.dots("w", OX, OY, 6.0, 0.9)
        f.rings("w", OX, OY, 0.14 * ppu, 0.13)
        self._lattice(f)
        self._track(f, t, width_=1.8)
        tage = t - CUTS[5]
        age = tage % BAR
        uu = min(1.0, age / 1.4)
        f.rings("r", [OX[hit]], [OY[hit]], [0.14 * ppu], 1.0, width=2.0)
        f.rings("r", [OX[hit]], [OY[hit]], [0.14 * ppu + 150 * (1 - (1 - uu) ** 3)], (1 - uu) ** 1.5, width=1.6)
        f.dots("r", [OX[hit]], [OY[hit]], 6.5, 1.5)
        n = int(min(len(self.delta), 2 + age * 22))
        room = P.fcol[1] - OX[hit] - 210
        sc = min(22.0, room / max(1.0, float(np.abs(self.delta[:, 0]).max())))
        ex_ = OX[hit] + self.delta[:n, 0] * sc
        ey_ = OY[hit] - np.abs(self.delta[:n, 1]) * 17.0
        f.polyline("r", np.r_[OX[hit], ex_], np.r_[OY[hit], ey_], 0.9, width=1.6)
        f.dots("r", ex_[-1:], ey_[-1:], 4.0, 1.7)
        solo = P.side is None
        callout(f, float(ex_[-1]), float(ey_[-1]), "E-", ["KNOCKED OUT", "12.6 EV"] if solo else [], col=P.fcol,
                prefer=1, dy=-50.0, red=True, age=age - 0.2, elbow=30.0)
        callout(f, float(OX[hit]), float(OY[hit]) + 0.14 * ppu, "H2O+", ["1 MOLECULE", "IN ~500"] if solo else [],
                col=P.fcol, prefer=-1, dy=110.0, red=True, age=age - 0.1)
        qx, qy = self._track_pt(-300.0)
        callout(f, qx, qy, "MU-", ["NO SIZE MEASURED", "< 1E-18 M"] if solo else [], col=P.fcol, prefer=1, dy=-40.0,
                red=True, age=tage - 0.5)
        self._readouts(f, [
            ("WATER", ["H2O // O-H 0.096 NM // 104.5 DEG", "70 % OF YOU"], False, tage),
            ("H2O+ / E-", ["ONE MOLECULE IN ~500 ON THE TRACK", "ITS ELECTRON KNOCKED OUT // 12.6 EV"], True,
             tage - 0.3),
            ("MU-", ["NO SIZE EVER MEASURED", "< 1E-18 M // POINT-LIKE", "SMALLER THAN ANYTHING IT MEETS"], True,
             tage - 0.6)])
        self._scale_bar(f, ppu, 1.0, "1 NM")

    # -- the track alone -----------------------------------------------------------------
    def _micro_track(self, f, t, u):
        P = self.P
        hud.cross_grid(f, WALL, step=96.0, inten=0.24, origin=(P.fx, Y_MID))
        P.plates(f)
        self._track(f, t, tick=28.0, width_=2.6)
        age = t - CUTS[6]
        x, y = self._track_pt(-150.0)
        lines = ["MOMENTUM   4.02 GEV/C", "SPEED      0.99965 C", "GAMMA      38.1", "DE/DX      2.0 MEV/CM",
                 "CHARGE     -1 E", "MASS       105.658 MEV/C2", "BORN       15.2 KM UP", "AGE        2.6 US  ITS OWN CLOCK"]
        col = P.side if P.side is not None else (P.fx + 150.0, P.fcol[1])
        cw = col[1] - col[0]
        if cw < 345.0:              # no room for the list: the tag carries the essential, the name stays open
            callout(f, x, y, "TRACK 0001", ["4.02 GEV/C", "0.99965 C", "NAME ___"], col=P.fcol, prefer=1, dy=-40.0,
                    red=True, age=age)
            return
        callout(f, x, y, "TRACK 0001", [], col=P.fcol, prefer=1, dy=-40.0, red=True, age=age)
        fs = float(np.clip((cw - 50) / (32 * 0.61), 15.0, 28.0))
        xl, yl = col[0] + 22, Y_TOP + 150
        f.occlude(col[0], yl - 2.6 * fs, col[1], yl + (len(lines) + 5.4) * fs * 1.45)
        f.rect("w", col[0], yl - 2.6 * fs, col[1], yl + (len(lines) + 5.4) * fs * 1.45, 0.4)
        f.tag("r", xl, yl - fs * 1.2, "MEASURED ON THE TRACK", size=L.T_MICRO, pad=3)
        for k, ln in enumerate(lines):
            f.text("w", xl, yl + 8 + k * fs * 1.45, hud.typed(ln, age, cps=120, delay=0.07 * k), size=fs, alpha=0.92)
        yn = yl + 8 + len(lines) * fs * 1.45 + 0.9 * fs
        f.text("w", xl, yn, "NAME", size=fs, alpha=0.92)
        blink = int(t * 4) % 2 == 0
        f.rects("r", xl + 11 * fs * 0.61, yn - fs * 1.2, xl + 11 * fs * 0.61 + fs * (11 if blink else 10.2),
                yn + fs * 0.3, 1.0)
        note = ["DID NOT STOP //", "CONTINUES ~8 M INTO THE", "GROUND BELOW YOU"] if cw < 660 else \
               ["DID NOT STOP //", "CONTINUES ~8 M INTO THE GROUND BELOW YOU"]
        for k, ln in enumerate(note):
            f.text("r", xl, yn + 2.6 * fs + k * 30, hud.typed(ln, age, cps=80, delay=0.5 + 0.2 * k), size=L.T_LABEL,
                   alpha=0.9)
