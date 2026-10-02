"""GALAXY - where they come from.   Sheet scene 4 DATA ON / AMBIENT, 03:00 - 03:56.

The TouchDesigner galaxy scene (perspective grid floor with red axes, a red dot, then a spiral of
thousands of thin particle trails with floating 0.00xxxxxxx labels, slowly turning, seen at an
angle), re-anchored on the towers and given the Muon Bloom data layer:

  * 03:00 cut: the grid floor, its red axes and the red dot - which sits exactly on the CENTRE
    detector. The galaxy grows continuously out of the dot (a tiny spiral after 2.5 s, most of the
    wall after 12 s, complete after 15-18 s): its tilted disc sweeps the whole wall behind the
    three towers, then turns slowly and calmly until the cut at 03:56. Nothing thins: the core
    keeps feeding two arms with new trails while the old ones wind up into the disc;
  * LIVE ("data on"): every detector onset is traced back to its source - a red burst of trails
    leaves that tower's detector, joins the rotation and cools to white, and a line is added to
    the trace log (echoes dimmer). A strong hit also makes the core answer: a red shock ring on
    the floor, a flash through the whole disc, a fast new generation of trails;
  * detail: labelled range rings, trace log, orbit log of tracked trails, arm-density histogram,
    rotation curve, counters, the three detector streams as a barcode.

Nothing depends on the drums. The model is closed-form in time (tables of radius / angle against
age, built once per span), so any frame can be drawn on its own - the realtime app can integrate
the same equations per particle.

The towers stand in front of the wall and nobody knows yet where: no fixed x here. The core follows
the centre detector, the two data columns are the outermost free columns of ctx.cols (dropped or
shortened when the towers leave no room), the world dissolves towards them, the floating numbers
and the tags keep clear of every tower. Only trails, rings and the floor pass behind the towers.

NOTHING THAT SHOWS DATA FADES IN OR POPS IN (muonbloom/build.py). At the cut (03:00) the two data columns
and the bottom panels construct themselves, block after block; the label of a range ring is decoded as
its ring draws itself; a floating number is decoded when it lands on its trail and taken apart when it
leaves; the TRACE tag of a detection is constructed on its tower and taken apart; a new line of the trace
log is decoded as it comes in. The galaxy itself (floor, trails, rings, bursts, flashes) is the image.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import CHAR_W, Camera, hash01, smoothstep
from ..show import Scene

ELEV = math.radians(27.0)       # camera elevation above the disc
DIST = 2.6                      # camera distance to the core (disc radius = 1)
FOCAL = 3250.0
Y_CLIP = 1196.0                 # the world stops above the bottom data band
KPC = 16.0                      # disc radius 1.0 = 16 kpc
V0, RC = 0.11, 0.15             # rotation: omega(r) = V0 / (r + RC)  (flat rotation curve), slow and calm
DT = 0.1                        # age step of the tables (s)
WAVE_V = 0.42                   # speed of the ring a low hit of the music sends through the arms (radii / s)
T_DELAY = 0.6                   # the dot alone, before the first trails leave it
N_FOUND, GROW = 1700, 11.0      # founders: all born in the first GROW seconds, they never fade
RATE, YOUNG = 60.0, (20.0, 28.0)   # then the core keeps emitting RATE trails / s, each fading between these ages
PATTERN = 0.05                  # the two arms the core feeds turn at this speed (rad / s)
STRONG = 0.65                   # a hit at least this strong makes the core answer with a new generation
GEN_LIFE = (14.0, 20.0)
RED_LIFE = (26.0, 34.0)
N_BINS = 60
RINGS = (4, 8, 12, 16)          # range rings (kpc); ring k switches on 0.4 + 1.6 k seconds after the cut


COL_W = 403.0                   # width a data column takes when the bay gives it
EDGE = 60.0                     # px / s: the red number of a detection is decoded as it gets clear of an obstacle
LAB_PERIOD = 6.0                # a floating number rides its trail for one turn of this length (s)
LAB_BACK, LAB_AHEAD = 20, 14    # frames looked back / ahead to know when a floating number came on / will go


def omega(r):
    return V0 / (r + RC)


def fit(size, n_chars, width):
    """Largest type size <= size that sets n_chars inside width."""
    return max(10.0, min(float(size), width / (max(n_chars, 1) * CHAR_W)))


def _gap(box, boxes):
    """Smallest gap (px) between a box and the boxes already placed (<= 0: it touches one of them)."""
    g = 1e9
    for o in boxes:
        g = min(g, max(o[0] - box[2], box[0] - o[2], o[1] - box[3], box[1] - o[3]))
    return g


def _cum(w):
    return np.concatenate([np.zeros((len(w), 1)), np.cumsum(0.5 * (w[:, 1:] + w[:, :-1]) * DT, axis=1)], 1)


class Model:
    """Particles of the galaxy: founders + continuous emission + one generation per strong hit (white),
    and one traced burst per detector onset (red, cooling to white)."""

    def __init__(self, t0, t1, hits, inj, seed=5):
        """hits = [(time, tower key, energy, echo)], inj = {key: (r, azimuth)}: where the towers sit on the disc."""
        rng = np.random.default_rng(seed)
        span = t1 - t0
        ages = np.arange(0.0, span + 10.0 + DT, DT)
        self.strong = [(a, k, e) for a, k, e, ec in hits if not ec and e >= STRONG and a < t1]
        blocks = []

        def block(n, tb, size, tau, th0, fade, kind):
            bulge = rng.random(n) < 0.14
            u = rng.random(n)
            R = np.where(bulge, 0.02 + 0.2 * u, 0.14 + 1.3 * u ** 0.8) * size
            tau_ = tau[0] + (tau[1] - tau[0]) * rng.random(n)
            tau_ = tau_ * (1.0 + 0.2 * R / (1.5 * size))
            loose = rng.random(n) < 0.1
            th = np.where(loose, rng.uniform(0, 2 * np.pi, n), th0)
            blocks.append(dict(tb=tb, R=R, tau=tau_, th0=th, bulge=bulge, f0=np.full(n, fade[0]),
                               f1=np.full(n, fade[1]), kind=np.full(n, kind)))

        # founders: the galaxy growing out of the dot
        n = N_FOUND
        block(n, t0 + T_DELAY + GROW * rng.random(n) ** 1.6, 1.0, (7.0, 11.0),
              rng.integers(0, 2, n) * np.pi + rng.normal(0, 0.36, n), (1e9, 2e9), 0)
        # the core keeps feeding two slowly turning arms
        n = int(RATE * span)
        tb = t0 + T_DELAY + rng.random(n) * span
        block(n, tb, 0.9, (6.0, 9.0), rng.integers(0, 2, n) * np.pi + PATTERN * (tb - t0) + rng.normal(0, 0.3, n),
              YOUNG, 1)
        # a strong hit: the core answers
        for k, (th_, key, e) in enumerate(self.strong):
            n = int(350 + 500 * e)
            block(n, th_ + rng.uniform(0.0, 0.35, n) ** 2, 0.95, (1.8, 3.4),
                  rng.integers(0, 3, n) * (2 * np.pi / 3) + 0.7 * k + rng.normal(0, 0.4, n), GEN_LIFE, 2)
        cat = lambda key: np.concatenate([b[key] for b in blocks])
        self.tb, self.R, tau, th0 = cat("tb"), cat("R"), cat("tau"), cat("th0")
        self.bulge, self.kind = cat("bulge"), cat("kind")
        n = self.n = len(self.tb)
        life = rng.uniform(0.85, 1.15, n)                                     # personal clock: who fades first
        self.f0, self.f1 = (cat("f0") * life).astype(np.float32), (cat("f1") * life).astype(np.float32)
        r = self.R[:, None] * (1.0 - np.exp(-ages[None, :] / tau[:, None])) + 0.0035 * self.R[:, None] * ages[None, :]
        self.r = r.astype(np.float32)
        self.th = (th0[:, None] + _cum(omega(r))).astype(np.float32)
        self.h = (rng.normal(0, 0.012, n) * np.where(self.bulge, 2.5, 1.0)).astype(np.float32)
        self.len = (2.0 + 6.0 * rng.random(n) ** 2).astype(np.float32)        # trail length (s)
        self.b = (0.22 + 0.78 * rng.random(n) ** 2.2).astype(np.float32)
        self.ecc = (0.05 * rng.random(n)).astype(np.float32)
        self.eph = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        # ---- red traced bursts (one per detector onset) ------------------------
        rb, thb, tbb, hb, lb, bb = [], [], [], [], [], []
        self.events = []
        first = 0
        strong_t = {s[0] for s in self.strong}
        for th_, key, e, echo in hits:
            stage = th_ in strong_t
            m = 110 if stage else int((9 if echo else 30) * (0.45 + e))
            r_i, a_i = inj[key]
            if r_i < 0.03:                    # the centre tower sits on the core: a burst in every direction
                Rk = (0.08 + 0.62 * rng.random(m) ** 0.9 * (0.5 + 0.6 * e)) * (1.5 if stage else 1.0)
                tk = (1.2 + 1.4 * rng.random(m)) if stage else (2.2 + 1.5 * rng.random(m))
                rr = Rk[:, None] * (1.0 - np.exp(-ages[None, :] / tk[:, None]))
                a0, dth = rng.uniform(0, 2 * np.pi, m), np.zeros(m)
            else:
                dr = rng.normal(0.03, 0.11, m) * (0.6 + 0.6 * e) * (1.5 if stage else 1.0)
                rr = np.maximum(r_i + dr[:, None] * (1.0 - np.exp(-ages[None, :] / 2.4)), 0.02)
                a0, dth = np.full(m, a_i), rng.normal(0, 0.48 if stage else 0.3, m)
            rb.append(rr)
            thb.append(a0[:, None] + dth[:, None] * (1.0 - np.exp(-ages[None, :] / 1.1)) + _cum(omega(rr)))
            tbb.append(np.full(m, th_))
            hb.append(rng.normal(0, 0.01, m))
            lb.append(1.4 + 2.6 * rng.random(m))
            bb.append((0.45 + 0.55 * rng.random(m)) * (0.55 if echo else 1.0) * (0.5 + 0.6 * e))
            self.events.append(dict(t=float(th_), key=key, e=float(e), echo=bool(echo), stage=stage, r=r_i,
                                    a=a_i, first=first, gl=float(rng.uniform(0, 360)),
                                    gb=float(rng.normal(0, 4.0)), d=float(rng.uniform(0.4, 14.0))))
            first += m
        if rb:
            self.rr = np.concatenate(rb).astype(np.float32)
            self.rth = np.concatenate(thb).astype(np.float32)
            self.rtb = np.concatenate(tbb).astype(np.float64)
            self.rh = np.concatenate(hb).astype(np.float32)
            self.rlen = np.concatenate(lb).astype(np.float32)
            self.rb = np.concatenate(bb).astype(np.float32)
        else:
            self.rr = self.rth = np.zeros((0, len(ages)), np.float32)
            self.rtb = np.zeros(0)
            self.rh = self.rlen = self.rb = np.zeros(0, np.float32)

    @staticmethod
    def _interp(tab, age):
        """tab (N, n_age), age (N, M) in seconds -> values (N, M)."""
        x = np.clip(age / DT, 0.0, tab.shape[1] - 1.001)
        i = x.astype(np.int64)
        fr = (x - i).astype(np.float32)
        rows = np.arange(tab.shape[0])[:, None]
        return tab[rows, i] * (1 - fr) + tab[rows, i + 1] * fr

    def trails(self, t, m=15, red=False):
        """World positions of the trail samples at time t. Returns (P (n, m, 3), age (n,), idx)."""
        if red:
            tb, tab_r, tab_t, h, ln = self.rtb, self.rr, self.rth, self.rh, self.rlen
            a = t - tb
            idx = np.nonzero((a > 0.0) & (a < RED_LIFE[1]))[0]
        else:
            tb, tab_r, tab_t, h, ln = self.tb, self.r, self.th, self.h, self.len
            a = t - tb
            idx = np.nonzero((a > 0.0) & (a < self.f1))[0]
        if not len(idx):
            return np.zeros((0, m, 3), np.float32), a[idx], idx
        a = a[idx]
        j = np.arange(m, dtype=np.float32) / (m - 1)
        age = np.maximum(a[:, None] - j[None, :] * ln[idx][:, None], 0.0)
        r = self._interp(tab_r[idx], age)
        th = self._interp(tab_t[idx], age)
        if not red:
            r = r * (1.0 + self.ecc[idx][:, None] * np.cos(2 * (th - self.eph[idx][:, None])))
        P = np.stack([r * np.cos(th), h[idx][:, None] * np.minimum(r / 0.4, 1.0), r * np.sin(th)], -1)
        return P.astype(np.float32), a, idx


class Galaxy(Scene):
    name = "galaxy"
    towers = "auto"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.core = ctx.towers["C"].det
        fov = math.degrees(2 * math.atan((L.H / 2) / FOCAL))
        self.cam = Camera((0.0, DIST * math.sin(ELEV), DIST * math.cos(ELEV)), (0.0, 0.0, 0.0), fov_deg=fov,
                          screen_center=self.core)
        self.inj = {k: self._unproject(*ctx.towers[k].det) for k in L.ORDER}
        self.inj["C"] = (0.0, 0.0)
        self.span = None
        self._build_floor()
        # the two data columns: the outermost free columns, unless the core itself sits there
        cols = list(ctx.cols)
        cx = self.core[0]
        self.col_l = self.col_r = None
        if cols:
            a, b = cols[0]
            if b - a >= 200 and cx - min(b, a + COL_W) > 420:
                self.col_l = (a, min(b, a + COL_W))
            a, b = cols[-1]
            if b - a >= 200 and max(a, b - COL_W) - cx > 420 and (len(cols) > 1 or self.col_l is None):
                self.col_r = (max(a, b - COL_W), b)
        self.x_lo = self.col_l[1] if self.col_l else L.FX0      # the world dissolves towards the columns
        self.x_hi = self.col_r[0] if self.col_r else L.FX1
        # where the label of each range ring goes: on the near side, wherever no tower stands (None: nowhere)
        self._ring_xy = []
        for rk in RINGS:
            xy = None
            for a_lab in (1.12, 0.62, 2.0, 2.52, 0.3, 2.84):
                lab = np.array([[rk / KPC * math.cos(a_lab), 0.0, rk / KPC * math.sin(a_lab)]], np.float32)
                lx, ly, _, _ = self.cam.project(lab)
                x, y = float(lx[0]), float(ly[0])
                if y < Y_CLIP - 30 and self._clear(x - 4, x + 90):
                    xy = (x, y)
                    break
            self._ring_xy.append(xy)

    def _clear(self, xa, xb, pad=16.0):
        """Is the x-interval between the data columns and clear of every tower?"""
        if xa < self.x_lo + 20 or xb > self.x_hi - 20:
            return False
        return not any(xa < tw.x1 + pad and tw.x0 - pad < xb for tw in self.ctx.towers.values())

    # ------------------------------------------------------------------ build
    def _build(self, t, ctx):
        """Everything that depends on where the look sits in the show (its span, the detector onsets in it).
        Rebuilt only if the look is auditioned somewhere else."""
        _, sec, _ = sd.section_at(t)
        t0, t1 = sd.look_span(sec[4], t) if sec[4] == self.name else (sec[2], sec[3])
        if self.span == (t0, t1):
            return
        self.span = (t0, t1)
        hits = []
        for key in sd.KEYS:
            tt, ee, ec = ctx.det.hits(key, t0, t1 + 1.0)
            hits += [(float(a), key, float(b), bool(c)) for a, b, c in zip(tt, ee, ec)]
        hits.sort()
        self.model = m = Model(t0, t1, hits, self.inj)
        rng = np.random.default_rng(77)
        cand = np.nonzero(~m.bulge & (m.R > 0.22) & (m.R < 1.25) & (m.b > 0.45))[0]
        self.lab = rng.choice(cand, size=min(260, len(cand)), replace=False)
        self.lab_val = 0.0004 + 0.02 * rng.random(len(self.lab)) ** 2
        self.lab_dig = rng.integers(8, 11, len(self.lab))
        self.lab_ph = rng.uniform(0, 6.0, len(self.lab))
        self.lab_str = [f"{v:.{int(d)}f}" for v, d in zip(self.lab_val, self.lab_dig)]
        self.lab_r, self.lab_th = m.r[self.lab], m.th[self.lab]      # their tables (read some thirty times a frame)
        self.track = np.sort(rng.choice(cand, size=min(90, len(cand)), replace=False))

    def _unproject(self, sx, sy):
        """Screen point -> (r, azimuth) on the disc plane."""
        cam = self.cam
        d = np.array([(sx - cam.cx) / cam.focal, -(sy - cam.cy) / cam.focal, 1.0])
        w = d @ cam.R.astype(np.float64)
        s = -cam.pos[1] / w[1]
        p = cam.pos + s * w
        return float(math.hypot(p[0], p[2])), float(math.atan2(p[2], p[0]))

    def _build_floor(self):
        g = np.arange(-3.0, 3.001, 0.25)
        u = np.linspace(-3.0, 3.0, 61)
        a, b = [], []
        for v in g:
            p = np.stack([u, np.zeros_like(u), np.full_like(u, v)], 1)
            q = np.stack([np.full_like(u, v), np.zeros_like(u), u], 1)
            for line in (p, q):
                a.append(line[:-1]); b.append(line[1:])
        self.fa = np.concatenate(a).astype(np.float32)
        self.fb = np.concatenate(b).astype(np.float32)
        mid = 0.5 * (self.fa + self.fb)
        self.f_rho = np.hypot(mid[:, 0], mid[:, 2])
        self.f_axis = ((np.abs(self.fa[:, 0]) < 1e-6) & (np.abs(self.fb[:, 0]) < 1e-6)) | \
                      ((np.abs(self.fa[:, 2]) < 1e-6) & (np.abs(self.fb[:, 2]) < 1e-6))

    def _fade(self, x, y):
        """The world dissolves (no hard edge) under the two data columns and above the bottom band."""
        return (smoothstep(self.x_lo - 25.0, self.x_lo + 145.0, x)
                * (1.0 - smoothstep(self.x_hi - 150.0, self.x_hi + 20.0, x))
                * (1.0 - smoothstep(1080.0, Y_CLIP, y)))

    def _wave(self, r, t, ctx):
        """Brightness boost at radius r from the rings the last low hits of the music sent out."""
        kt, ka = ctx.cues.kicks(max(self.span[0], t - 3.2), t + 1e-6)
        out = np.zeros_like(r)
        for tk, a in zip(kt, ka):
            age = t - float(tk)
            out += min(float(a), 2.0) * np.exp(-((r - WAVE_V * age) / 0.05) ** 2) * math.exp(-age / 1.5)
        return np.minimum(out, 2.5)

    def _flash(self, t):
        """Every strong hit lights the whole disc for a moment."""
        return sum(math.exp(-(t - s[0]) / 0.6) for s in self.model.strong if 0.0 <= t - s[0] < 4.0)

    # ------------------------------------------------------------------ draw
    def draw(self, f, t, ctx):
        self._build(t, ctx)
        f.set_clip(L.FX0 + 2, L.FY0 + 2, L.FX1 - 2, Y_CLIP)
        self._floor(f, t)
        n_alive, n_red, hist, hot = self._particles(f, t, ctx)
        self._links(f, t)
        self._core(f, t, ctx)
        self._labels(f, t)
        self._events(f, t)
        f.set_clip()
        self._trace_tag(f, t, ctx)
        self._left(f, t, ctx, n_alive)
        self._right(f, t, ctx, hist, hot)
        self._bottom(f, t, ctx, n_alive)
        return {"tower_dim": 0.34, "burst_size": 0.85}

    def _floor(self, f, t):
        cam = self.cam
        t0 = self.span[0]
        self._ring_labels = self._ring_labels_at(t)
        ax, ay, az, aok = cam.project(self.fa)
        bx, by, bz, bok = cam.project(self.fb)
        ok = aok & bok
        fog = np.exp(-(self.f_rho / 2.3) ** 2) * np.clip(2.9 / np.maximum(az, 0.3), 0.3, 1.5)
        fog = fog * (0.12 + 0.88 * self._fade(0.5 * (ax + bx), 0.5 * (ay + by)))
        m = ok & ~self.f_axis
        f.segments("w", ax[m], ay[m], bx[m], by[m], 0.24 * fog[m])
        m = ok & self.f_axis
        f.segments("r", ax[m], ay[m], bx[m], by[m], 0.7 * fog[m], width=L.LW)
        # range rings every 4 kpc, dotted, labelled on the near side: each one switches on as the galaxy reaches it
        for k, rk in enumerate(RINGS):
            age = t - (t0 + 0.4 + 1.6 * k)
            if age <= 0:
                continue
            rr = rk / KPC
            n = int(260 * rr) + 60
            ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
            show = ang < 2 * np.pi * min(1.0, age / 0.8)              # it draws itself around
            P = np.stack([rr * np.cos(ang), np.zeros(n), rr * np.sin(ang)], 1).astype(np.float32)
            sx, sy, sz, ok = cam.project(P)
            ok = ok & show
            f.dots("w", sx[ok], sy[ok], 1.4, 0.55 * (0.12 + 0.88 * self._fade(sx[ok], sy[ok])))
            if self._ring_xy[k] is not None:                      # its label is decoded while it draws itself
                x, y = self._ring_xy[k]
                f.text("w", x + 10, y + 22, B.resolve(f"{rk:02d} KPC", age, 30.0, 0.1, key=k), size=L.T_SMALL, alpha=0.75)

    def _ring_labels_at(self, t):
        """Places of the range-ring labels that are on at time t."""
        return [xy for k, xy in enumerate(self._ring_xy) if xy is not None and t - (self.span[0] + 0.4 + 1.6 * k) > 0]

    def _particles(self, f, t, ctx):
        cam = self.cam
        m = self.model
        kick = min(1.0, ctx.cues.kick(t, 0.14))
        hist = np.zeros(N_BINS)
        hot = np.zeros(N_BINS)
        n_alive = n_red = 0
        empty = (np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0, bool), np.zeros(0), np.zeros(0), np.zeros(0))
        self._heads = self._rheads = empty
        gain = 1.0 + 0.6 * self._flash(t)
        # ---- white primaries
        M = 15
        P, age, idx = m.trails(t, M)
        if len(idx):
            sx, sy, sz, ok = cam.project(P.reshape(-1, 3))
            sx, sy, sz, ok = (v.reshape(-1, M) for v in (sx, sy, sz, ok))
            r_head = np.hypot(P[:, 0, 0], P[:, 0, 2])
            wave = self._wave(r_head, t, ctx)
            near = np.clip(DIST / np.maximum(sz[:, 0], 0.4), 0.55, 1.9) ** 0.9
            born = np.clip(age / 0.5, 0.0, 1.0)
            fade = 1.0 - smoothstep(m.f0[idx], m.f1[idx], age)
            n_alive = int((fade > 0.5).sum())
            b = m.b[idx] * near * born * fade * gain * (0.8 + 0.25 * kick) * (1.0 + 1.5 * wave)
            b = b * (0.06 + 0.94 * self._fade(sx[:, 0], sy[:, 0]))
            b = b * (0.38 + 0.62 * smoothstep(0.03, 0.3, r_head))        # keep the structure of the core readable
            fall = (1.0 - np.arange(M) / (M - 1)) ** 1.25
            i0 = b[:, None] * fall[None, :-1]
            i1 = b[:, None] * fall[None, 1:]
            okk = ok[:, :-1] & ok[:, 1:]
            wide = (m.b[idx] > 0.5)[:, None] & okk
            thin = okk & ~wide
            f.segments("w", sx[:, :-1][thin], sy[:, :-1][thin], sx[:, 1:][thin], sy[:, 1:][thin],
                       0.85 * i0[thin], 0.85 * i1[thin])
            f.segments("w", sx[:, :-1][wide], sy[:, :-1][wide], sx[:, 1:][wide], sy[:, 1:][wide],
                       0.8 * i0[wide], 0.8 * i1[wide], width=L.LW)
            hd = ok[:, 0]
            f.dots("w", sx[hd, 0], sy[hd, 0], (1.0 + 1.6 * m.b[idx][hd]) * near[hd] ** 0.6, 1.3 * b[hd])
            th_head = np.arctan2(P[:, 0, 2], P[:, 0, 0])
            sel = r_head > 0.1
            hist += np.histogram(th_head[sel] % (2 * np.pi), bins=N_BINS, range=(0, 2 * np.pi), weights=fade[sel])[0]
            self._heads = (idx, sx[:, 0], sy[:, 0], ok[:, 0], age, r_head, th_head)
        # ---- red traced bursts
        M = 11
        P, age, idx = m.trails(t, M, red=True)
        if len(idx):
            n_red = len(idx)
            sx, sy, sz, ok = cam.project(P.reshape(-1, 3))
            sx, sy, sz, ok = (v.reshape(-1, M) for v in (sx, sy, sz, ok))
            near = np.clip(DIST / np.maximum(sz[:, 0], 0.4), 0.55, 1.9) ** 0.9
            cool = smoothstep(3.5, 9.0, age)
            life = np.clip(age / 0.12, 0.0, 1.0) * (1.0 - smoothstep(RED_LIFE[0], RED_LIFE[1], age))
            b = m.rb[idx] * near * life * (0.25 + 0.75 * self._fade(sx[:, 0], sy[:, 0]))
            fall = (1.0 - np.arange(M) / (M - 1)) ** 1.1
            okk = ok[:, :-1] & ok[:, 1:]
            for lay, wgt in (("r", (1.0 - cool) * 1.5), ("w", cool * 0.7)):
                i0 = (b * wgt)[:, None] * fall[None, :-1]
                i1 = (b * wgt)[:, None] * fall[None, 1:]
                f.segments(lay, sx[:, :-1][okk], sy[:, :-1][okk], sx[:, 1:][okk], sy[:, 1:][okk], i0[okk], i1[okk],
                           width=L.LW if lay == "r" else 1.0)
            hd = ok[:, 0]
            f.dots("r", sx[hd, 0], sy[hd, 0], 2.0 * near[hd] ** 0.6, (1.6 * b * (1.0 - cool))[hd])
            f.dots("w", sx[hd, 0], sy[hd, 0], 1.0 * near[hd] ** 0.6, (1.1 * b)[hd])
            th_head = np.arctan2(P[:, 0, 2], P[:, 0, 0]) % (2 * np.pi)
            hot += np.histogram(th_head, bins=N_BINS, range=(0, 2 * np.pi), weights=(1.0 - cool) * life)[0]
            hist += np.histogram(th_head, bins=N_BINS, range=(0, 2 * np.pi), weights=life)[0]
            self._rheads = (idx, sx[:, 0], sy[:, 0], ok[:, 0], age, None, None)
        return n_alive, n_red, hist, hot

    def _links(self, f, t):
        """A strong hit on a side tower: a red arc runs along the floor from that detector to the core."""
        for ev in self.model.events:
            a = t - ev["t"]
            if not ev["stage"] or ev["r"] < 0.03 or not (0.0 <= a < 1.6):
                continue
            s = np.linspace(0.0, 1.0, 90)
            r = ev["r"] * (1.0 - s)
            th = ev["a"] + 1.25 * s ** 1.3
            P = np.stack([r * np.cos(th), np.zeros(90), r * np.sin(th)], 1).astype(np.float32)
            sx, sy, _, ok = self.cam.project(P)
            reach = min(1.0, a / 0.16)                    # it takes 160 ms to get there
            k = int(max(2, reach * 90))
            f.polyline("r", sx[:k], sy[:k], 1.9 * math.exp(-a / 0.4), width=L.LW_BOLD)
            if reach < 1.0:
                f.dots("r", [sx[k - 1]], [sy[k - 1]], 5.0, 1.8)
                f.dots("w", [sx[k - 1]], [sy[k - 1]], 2.0, 1.2)

    def _core(self, f, t, ctx):
        cx, cy = self.core
        a0 = t - (self.span[0] + T_DELAY)
        kick = min(1.0, ctx.cues.kick(t, 0.16))
        fl = min(1.5, self._flash(t))
        pulse = 0.6 + 0.4 * math.sin(2 * math.pi * t * 0.45) ** 2
        g = float(np.clip(a0 / 4.0, 0.0, 1.0))
        if g > 0:
            f.dots("w", [cx], [cy], 20.0 + 6.0 * kick + 10.0 * fl, (0.2 + 0.15 * kick + 0.3 * fl) * g)
        f.dots("r", [cx], [cy], 8.5 + 2.0 * pulse * (1 - g) + 2.0 * kick + 4.0 * fl, 1.6)
        f.dots("w", [cx], [cy], 2.6, 1.3)
        if g < 1.0:                 # the dot alone at the cut: its ring opens and lets go
            f.rings("r", [cx], [cy], [26.0 + 8.0 * pulse + 60.0 * g], 0.6 * (1 - g), width=L.LW)
        for k, (tg, key, e) in enumerate(self.model.strong):    # a strong hit: red shock ring on the floor + rays
            a = t - tg
            if not (0 <= a < 2.6):
                continue
            u = a / 2.6
            rr = 0.02 + 1.25 * (0.4 + e) * (1 - (1 - u) ** 2.4)
            ang = np.linspace(0, 2 * np.pi, 420, endpoint=False)
            P = np.stack([rr * np.cos(ang), np.zeros(420), rr * np.sin(ang)], 1).astype(np.float32)
            sx, sy, _, ok = self.cam.project(P)
            f.dots("r", sx[ok], sy[ok], 2.3, 1.5 * (1 - u) ** 1.5 * (0.2 + 0.8 * self._fade(sx[ok], sy[ok])))
            n = 70
            j = np.arange(n)
            an = hash01(j, 3, int(tg * 10)) * 2 * np.pi
            ln = (50 + 520 * e * hash01(j, 4, int(tg * 10))) * (1 - (1 - min(1.0, a / 0.35)) ** 3)
            f.segments("w", cx + 14 * np.cos(an), cy + 7 * np.sin(an), cx + (14 + ln) * np.cos(an),
                       cy + (14 + ln) * np.sin(an) * 0.5, 0.95 * (1 - u) ** 2, 0.0, width=L.LW)

    def _margin(self, x, y, rings=None):
        """How far (px) a floating label at (x, y) is inside the free part of the wall: clear of the columns,
        the bottom band, the core, the towers and their tags, the range-ring labels (`rings`, default: the
        ones on now). <= 0: it is not free. Scalars or arrays."""
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        m = np.minimum(np.minimum(x - (self.x_lo + 65), (self.x_hi - 280) - x),
                       np.minimum(y - (L.FY0 + 40), (Y_CLIP - 60) - y))
        cx, cy = self.core
        m = np.minimum(m, np.maximum(np.abs(x - cx + 90) - 330, np.abs(y - cy) - 130))
        for tw in self.ctx.towers.values():
            m = np.minimum(m, np.maximum(np.maximum(tw.x0 - 250 - x, x - (tw.x1 + 12)), (tw.top - 30) - y))
            m = np.minimum(m, np.maximum(np.maximum(tw.x0 - 480 - x, x - (tw.x1 + 260)),
                                         np.maximum((tw.top - 40) - y, y - (tw.top + tw.det_h + 150))))
        for bx, by in (self._ring_labels if rings is None else rings):
            m = np.minimum(m, np.maximum(np.maximum(bx - 240 - x, x - (bx + 110)), np.maximum(by - 30 - y, y - (by + 40))))
        return m

    def _label_heads(self, t):
        """Head of every labelled trail at time t: (screen x, screen y, age of the particle, alive and in view)."""
        m, k = self.model, self.lab
        a = t - m.tb[k]
        ok = (a > 0.0) & (a < m.f1[k])
        age = np.maximum(a, 0.0)[:, None]
        r = m._interp(self.lab_r, age)[:, 0]
        th = m._interp(self.lab_th, age)[:, 0]
        r = r * (1.0 + m.ecc[k] * np.cos(2 * (th - m.eph[k])))
        P = np.stack([r * np.cos(th), m.h[k] * np.minimum(r / 0.4, 1.0), r * np.sin(th)], -1).astype(np.float32)
        sx, sy, _, pok = self.cam.project(P)
        return sx.astype(np.float64), sy.astype(np.float64), a, ok & pok

    def _label_set(self, t):
        """The floating numbers that are on at time t -> ({label: (x, y)}, their boxes). A number rides the head
        of its trail for one turn out of 2.5 (LAB_PERIOD each), once its particle has settled, while it is
        clear of the towers, the core, the columns and the ring labels, and of the numbers placed before it;
        twelve at most."""
        x, y, age, ok = self._label_heads(t)
        k = self.lab
        ph = np.floor((t + self.lab_ph) / LAB_PERIOD).astype(np.int64)
        on = ok & (age > 2.0) & (age < self.model.f0[k] - 1.0) & (hash01(np.arange(len(k)), ph) <= 0.4)
        on[on] = self._margin(x[on], y[on], self._ring_labels_at(t)) > 0
        out, boxes = {}, []
        for q in np.nonzero(on)[0]:
            xq, yq = float(x[q]), float(y[q])
            bx = (xq - 6, yq - 26, xq + 26 + 11 * len(self.lab_str[q]), yq + 12)
            if _gap(bx, boxes) <= 0:
                continue
            boxes.append(bx)
            out[int(q)] = (xq, yq)
            if len(boxes) >= 12:
                break
        return out, boxes

    def _labels(self, f, t):
        """The floating numbers of the TouchDesigner scene, riding on particle heads. They never pop in or
        out: a number is decoded when it comes on (its turn starts, its trail carries it clear of a tower or
        of another number ...) and taken apart before it goes. When that happens is found by applying the
        rules a little back and ahead in time: the model is closed-form, so this stays a pure function of t
        (the realtime app would simply keep a timer per number)."""
        now, self._boxes = self._label_set(t)
        if not now:
            return
        dt = 1.0 / 30.0
        since, left = {}, {}
        pend = set(now)
        for n in range(1, LAB_BACK + 1):                # when did each one come on?
            if not pend:
                break
            prev = self._label_set(t - n * dt)[0]
            for q in [q for q in pend if q not in prev]:
                since[q] = (n - 1) * dt
                pend.discard(q)
        pend = set(now)
        for n in range(1, LAB_AHEAD + 1):               # when will it go?
            if not pend:
                break
            nxt = self._label_set(t + n * dt)[0]
            for q in [q for q in pend if q not in nxt]:
                left[q] = n * dt
                pend.discard(q)
        cps = 55.0
        for q, (x, y) in now.items():
            s = self.lab_str[q]
            a = B.io(since.get(q, 9.0), left.get(q), out=0.45, span=len(s) / cps + 0.35)
            if a <= 0.0:
                continue
            f.dots("w", [x], [y], 2.8 * float(B.spring(a / 0.22)), 1.3)
            f.text("w", x + 12, y + 6, B.resolve(s, a, cps, 0.06, key=q), size=L.T_LABEL, alpha=0.88)

    def _events(self, f, t):
        """Each detection traced back: a ring on the floor where the trails leave and a red number riding
        the leading trail (decoded when it lands on it, taken apart before it leaves it, and as it comes up to
        an obstacle or to another number: see EDGE)."""
        m = self.model
        ridx, rx, ry, rok, rage = self._rheads[:5]
        pos = {int(k): j for j, k in enumerate(ridx)}
        for ev in m.events:
            a = t - ev["t"]
            if a < 0 or a > 6.0:
                continue
            if a < 1.6:                   # ring on the disc plane
                u = a / 1.6
                big = 0.3 if ev["stage"] else (0.22 if not ev["echo"] else 0.1)
                rr = 0.02 + big * (0.5 + ev["e"]) * (1 - (1 - u) ** 2)
                ang = np.linspace(0, 2 * np.pi, 120, endpoint=False)
                c = np.array([ev["r"] * math.cos(ev["a"]), 0.0, ev["r"] * math.sin(ev["a"])])
                P = (c[None] + np.stack([rr * np.cos(ang), np.zeros(120), rr * np.sin(ang)], 1)).astype(np.float32)
                sx, sy, _, ok = self.cam.project(P)
                f.dots("r", sx[ok], sy[ok], 1.8, 1.3 * (1 - u) ** 1.4 * (0.5 if ev["echo"] else 1.0))
            if not ev["echo"] and 0.25 < a < 4.5:
                for jj in range(3 if ev["stage"] else 1):       # a strong hit: three of its trails carry a number
                    j = pos.get(ev["first"] + 7 * jj)
                    if j is None or not rok[j]:
                        continue
                    x, y = float(rx[j]), float(ry[j])
                    bx = (x - 6, y - 26, x + 150, y + 12)
                    mg, gap = float(self._margin(x, y)), _gap(bx, self._boxes)
                    if mg > 0 and gap > 0:
                        self._boxes.append(bx)
                        s = f"{ev['e'] * 0.01 + 0.0001 * hash01(int(ev['t'] * 100), jj):.10f}"
                        ab = B.io(a - 0.25, 4.5 - a, out=0.5, span=len(s) / 45.0 + 0.35)
                        ab = min(ab, mg / EDGE, gap / EDGE)
                        f.text("r", x + 12, y + 6, B.resolve(s, ab, 45.0, key=40 + jj), size=L.T_LABEL)

    def _trace_tag(self, f, t, ctx):
        """Tag on every tower whose detection is being traced back right now. It never fades: its black
        plate opens, the TRACE tag is made, the three lines are decoded; a new detection of the same tower
        while the tag is up is decoded in place; when its time is over the whole thing is taken apart."""
        up = {}                                     # tower -> (when its tag came up, event shown, its number, end)
        count = 0
        for e in self.model.events:
            if e["echo"] or e["t"] > t:
                continue
            count += 1
            prev = up.get(e["key"])
            start = prev[0] if prev and prev[3] > e["t"] else e["t"]
            up[e["key"]] = (start, e, count, e["t"] + (4.0 if e["stage"] else 2.6))
        tws = sorted(ctx.towers.values(), key=lambda v: v.x0)
        for key, (start, e, n, end) in up.items():
            if t >= end:
                continue
            tw = ctx.towers[key]
            i = tws.index(tw)
            room_r = (tws[i + 1].x0 if i + 1 < len(tws) else self.x_hi) - tw.x1
            room_l = tw.x0 - (tws[i - 1].x1 if i > 0 else self.x_lo)
            if max(room_r, room_l) < 290:
                continue                        # no room beside this tower: the trace log has it anyway
            side = 1 if room_r >= 330 or room_r >= room_l else -1
            x = tw.x1 + 30 if side > 0 else tw.x0 - 30
            y = tw.top + tw.det_h + 44          # beside the tower body, clear of the bloom at its head
            anchor = "ls" if side > 0 else "rs"
            lines = [f"{L.NAMES[key]}  E {e['e']:.3f}", f"GL {e['gl']:05.1f}  GB {e['gb']:+05.1f}",
                     f"D {e['d']:05.2f} KPC"]
            w = max(len(s) for s in lines) * L.T_SMALL * 0.61
            bx0 = x - 6 if side > 0 else x - w - 6
            ab = B.io(t - start, end - t, out=0.4, span=0.9)
            a = min(ab, t - e["t"])                 # a new detection of this tower is decoded in place
            f.occlude(bx0, y + 12, bx0 + w + 12, y + 12 + (4 + len(lines) * 25) * float(B.ease(B.lin(ab, 0.0, 0.2))))
            title = f"TRACE {n:03d}"
            if a < ab:                              # ... its number is decoded in the tag, which stays
                f.tag("r", x, y, B.decode(title, a, cps=60.0, key=130, pad=True), size=L.T_TAG, pad=5, anchor=anchor,
                      ref=title)
            else:
                B.tag(f, "r", x, y, title, a, size=L.T_TAG, pad=5, anchor=anchor, cps=60.0, key=130)
            for k, ln in enumerate(lines):
                f.text("w", x, y + 36 + k * 25, B.resolve(ln, a, 90.0, 0.1 + 0.08 * k, key=131 + k, pad=side < 0),
                       size=L.T_SMALL, alpha=0.92, anchor=anchor)

    # ------------------------------------------------------------------ HUD
    def _left(self, f, t, ctx, n_alive):
        if self.col_l is None:
            return
        x0, x1 = self.col_l
        w = x1 - x0
        _, sec, _ = sd.section_at(t)
        head = f"{sec[0]} // {sec[1]}"
        age = t - self.span[0]                    # the column is constructed at the cut, block after block
        ts = fit(76, 6, w - 30)
        with f.build(age - 0.1, (x0 - 8, 246, x1 + 4, 360 + ts), key=140, wave=0.35):
            f.tag("w", x0, 272, head, size=fit(L.T_SMALL, len(head), w - 10), pad=4)
            f.tag("w", x0 + 4, 296 + ts, "GALAXY", size=ts, pad=ts * 0.13)
            sub = "COSMIC RAY SOURCE MAP // MILKY WAY" if w >= 360 else "COSMIC RAY SOURCE MAP"
            f.text("w", x0, 348 + ts, sub, size=fit(L.T_SMALL, len(sub), w), alpha=0.8)
        m = self.model
        n_tr = sum(1 for e in m.events if e["t"] <= t)
        lines = [f"EXPOSURE  {t - self.span[0]:06.2f} S",
                 f"PRIMARIES {n_alive:05d}",
                 f"TRACED    {n_tr:05d}",
                 f"EMISSION  {RATE:03.0f} /S",
                 f"DISC R    {KPC:.1f} KPC",
                 "B_FIELD   3.0 UG",
                 "RESIDENCE 15 MYR",
                 "V_ROT     220 KM/S",
                 f"INCL      {90 - math.degrees(ELEV):.0f} DEG"]
        with f.build(age - 0.3, (x0 - 8, 376 + ts, x1 + 4, 398 + ts + 8 * 25.5 + 12), flow="tb", wave=0.6, key=141):
            hud.rows(f, x0, 398 + ts, lines, size=L.T_SMALL, lead=1.5, red=(2,))
        full = w >= 345
        y = 652.0 + ts
        ev = [e for e in m.events if e["t"] <= t][::-1]
        n_rows = 7
        with f.build(age - 0.55, (x0 - 8, y - 22, x1 + 4, y + 54 + n_rows * 21), flow="tb", wave=0.4, key=142):
            f.tag("w", x0, y, "TRACE_LOG // DETECTION -> SOURCE" if w >= 290 else "TRACE_LOG", size=L.T_MICRO, pad=3)
            f.text("w", x0, y + 30, "T        DET   E     GL    GB    D_KPC" if full else "T        DET   E",
                   size=L.T_MICRO, alpha=0.55)
            for k, e in enumerate(ev[:n_rows]):
                line = f"{sd.tc(e['t'])[:8]} DET_{e['key']} {e['e']:.3f}"
                if full:
                    line += f" {e['gl']:05.1f} {e['gb']:+05.1f} {e['d']:05.2f}"
                if k == 0:                  # the detection that just came in is decoded on top of the log
                    line = B.resolve(line, t - e["t"], 120.0, 0.15 if len(ev) == 1 else 0.0, key=143)
                fresh = t - e["t"] < 2.5
                f.text("r" if fresh else "w", x0, y + 54 + k * 21, line, size=L.T_MICRO,
                       alpha=(0.95 if k < 2 else 0.62) * (0.6 if e["echo"] else 1.0))
            wait = "DATA ON // WAITING FOR A MUON" if w >= 265 else "WAITING FOR A MUON"
            if not ev:
                f.tag("r", x0, y + 60, wait, size=L.T_MICRO, pad=3, alpha=0.5 + 0.5 * math.sin(t * 9) ** 2)
            elif t - ev[-1]["t"] < 0.15:    # the first muon: the tag is wiped away, the log starts under it
                f.tag("r", x0, y + 60, "", size=L.T_MICRO, pad=3, ref=wait, wipe=1.0 - B.lin(t - ev[-1]["t"], 0.0, 0.15))
        # orbit log: a few tracked primaries, their numbers changing as they turn
        y2 = y + 54 + n_rows * 21 + 30
        idx, _, _, _, p_age, r_head, th_head = self._heads
        full = w >= 300
        with f.build(age - 0.8, (x0 - 8, y2 - 22, x1 + 4, y2 + 54 + 8 * 21), flow="tb", wave=0.4, key=144):
            f.tag("w", x0, y2, "ORBIT_LOG // TRACKED PRIMARIES" if w >= 275 else "ORBIT_LOG", size=L.T_MICRO, pad=3)
            f.text("w", x0, y2 + 30, "ID    R_KPC  AZ_DEG  V_KM/S  AGE_S" if full else "ID    R_KPC  AZ_DEG",
                   size=L.T_MICRO, alpha=0.55)
            if len(idx):
                pos = np.searchsorted(idx, self.track)
                okk = (pos < len(idx)) & (idx[np.minimum(pos, len(idx) - 1)] == self.track)
                sel = pos[okk]
                sel = sel[p_age[sel] > 2.0]
                sel = sel[np.argsort(p_age[sel])][:8]                # the latest to settle first: a slow ticker
                for k, j in enumerate(sel):
                    r = float(r_head[j])
                    line = f"{int(idx[j]):04d}  {r * KPC:05.2f}  {math.degrees(float(th_head[j])) % 360:06.2f}"
                    if full:
                        line += f"  {220.0 * r / (r + RC):06.2f}  {float(p_age[j]):05.2f}"
                    if k == 0:              # a primary that has just settled into its orbit enters the log
                        line = B.resolve(line, float(p_age[j]) - 2.0, 120.0, key=145)
                    f.text("w", x0, y2 + 54 + k * 21, line, size=L.T_MICRO, alpha=0.9 if k < 2 else 0.6)

    def _right(self, f, t, ctx, hist, hot):
        if self.col_r is None:
            return
        x0, x1 = self.col_r
        w = x1 - x0
        age = t - self.span[0]                    # the column is constructed at the cut: histogram, then curve
        top, bot = 300.0, 828.0
        ys = top + np.arange(N_BINS) * (bot - top) / N_BINS
        xl = x0 + 58
        wmax = x1 - xl - 6
        v = hist / max(hist.max(), 1.0)
        hm = hot > 0.3 * np.maximum(hist, 1.0)
        with f.build(age - 0.4, (x0 - 8, 248, x1 + 4, bot + 10), flow="tb", wave=0.7, key=150):
            f.tag("w", x0, 272, f"ARM_DENSITY // {N_BINS} BINS // AZIMUTH" if w >= 300 else "ARM_DENSITY",
                  size=L.T_MICRO, pad=3)
            f.rects("w", xl, ys[~hm], xl + wmax * v[~hm], ys[~hm] + 4, 0.9)
            f.rects("r", xl, ys[hm], xl + wmax * v[hm], ys[hm] + 4, 1.0)
            f.segments("w", [xl - 6], [top - 4], [xl - 6], [bot + 2], 0.6)
            for k in range(0, N_BINS, N_BINS // 6):
                f.segments("w", [xl - 13], [ys[k] + 2], [xl - 6], [ys[k] + 2], 0.8)
                f.text("w", x0, ys[k] + 8, f"{k * 360 // N_BINS:03d}", size=L.T_MICRO, alpha=0.7)
        # rotation curve: speed against radius; the ring of every strong hit travels along it
        y0, y1 = 902.0, 1150.0
        px0, px1, py0, py1 = x0 + 8, x1 - 8, y0 + 14, y1 - 30
        with f.build(age - 0.9, (x0 - 8, y0 - 40, x1 + 4, y1 + 14), key=151, wave=0.4):
            f.tag("w", x0, y0 - 16, "ROTATION_CURVE // V(R)", size=L.T_MICRO, pad=3)
            f.segments("w", [px0, px0], [py0, py1], [px0, px1], [py1, py1], 0.7)
            hud.ruler(f, px0, px1, py1, 0.0, 24.0, 1.0, 4.0 if w >= 300 else 8.0, fmt=lambda vv: f"{vv:.0f}", inten=0.6,
                      lab_dy=26)
            r = np.linspace(0.0, 1.5, 90)
            vr = omega(r) * r / V0
            f.polyline("w", px0 + r / 1.5 * (px1 - px0), py1 - vr * (py1 - py0) * 0.9, 0.95, width=L.LW)
            f.text("w", px1, py0 + 6, "220 KM/S", size=L.T_MICRO, alpha=0.7, anchor="rs")
            f.text("w", px1, y1 + 6, "KPC", size=L.T_MICRO, alpha=0.6, anchor="rs")
        for tg, key, e in self.model.strong:
            a = t - tg
            if 0 <= a < 2.6:            # the marker of a strong hit: a line that is drawn, travels, and is withdrawn
                rw = 1.25 * (0.4 + e) * (1 - (1 - a / 2.6) ** 2.4)
                xx = px0 + min(rw, 1.5) / 1.5 * (px1 - px0)
                B.pen(f, "r", xx, py1, xx, py0, min(B.lin(a, 0.0, 0.15), 1.0 - B.lin(a, 2.25, 2.6)), 1.1, width=L.LW,
                      head=0.0)

    def _bottom(self, f, t, ctx, n_alive):
        """Bottom band: the free panels between the scopes and the towers, taken by need (they change with
        the placement of the towers; what does not fit is dropped)."""
        panels = sorted(ctx.slots["panels"], key=lambda q: q[0])
        y0, y1 = ctx.slots["y0"], ctx.slots["y1"]
        main = next((q for q in panels if q[1] - q[0] >= 330.0), None)
        rest = [q for q in panels if q is not main]
        code = next((q for q in rest if q[1] - q[0] >= 200.0), None)
        expo = next((q for q in rest if q is not code and q[1] - q[0] >= 125.0), None)
        a_in = t - self.span[0]                  # the panels are constructed at the cut, one after the other
        if main:
            x0, x1 = main
            w = x1 - x0
            with f.build(a_in - 0.2, (x0 - 8, y0 - 24, x1 + 8, y1 + 8), key=160):
                hud.panel_header(f, x0, x1, y0, "PRIMARIES IN FLIGHT")
                size = fit(58, 5, min(w * 0.5, 250.0) - 24)
                f.text("w", x0, y0 + 96, f"{n_alive:,}".replace(",", " "), size=size)
                xm = x0 + min(250.0, w * 0.55)
                f.text("w", xm, y0 + 44, "TRACED BACK", size=L.T_MICRO, alpha=0.75)
                n_tr = sum(1 for e in self.model.events if e["t"] <= t)
                f.text("r", xm, y0 + 96, f"{n_tr:03d}", size=size)
                cap = "EACH TRAIL = ONE COSMIC RAY IN THE GALACTIC FIELD" if w >= 415 else "ONE TRAIL = ONE COSMIC RAY"
                f.text("w", x0, y0 + 122, cap, size=L.T_MICRO, alpha=0.6)
        if code:                          # data on: the three detector streams of the last seconds, as a barcode
            x0, x1 = code
            w = x1 - x0
            with f.build(a_in - 0.4, (x0 - 8, y0 - 24, x1 + 8, y1 + 8), key=161, flow="lr", wave=0.5):
                hud.panel_header(f, x0, x1, y0, "DATA ON // DETECTOR STREAMS >> BARCODE" if w >= 335 else
                                 "DETECTOR STREAMS")
                cols = max(40, int((w - 34) / 4.0))
                dt = 6.0 / 110
                kk = math.floor(t / dt) - cols + np.arange(cols) + 1
                ts = kk * dt
                bx0 = x0 + 34
                cw = (x1 - bx0) / cols
                xs = bx0 + np.arange(cols) * cw
                lane = (y1 - y0 - 14) / 3
                for i, key in enumerate(L.ORDER):
                    v = ctx.det.value(key, ts)
                    on = hash01(kk, i + 3) < 0.04 + 1.7 * v
                    ya = y0 + 14 + i * lane
                    f.rects("w", xs[on], ya, xs[on] + cw, ya + lane - 4, 0.95)
                    age, _ = ctx.det.last(key, t, echoes=True)
                    f.text("r" if age < 1.0 else "w", x0, ya + lane - 10, key, size=L.T_LABEL, alpha=0.9)
        if expo:
            x0, x1 = expo
            with f.build(a_in - 0.6, (x0 - 8, y0 - 24, x1 + 8, y1 + 8), key=162):
                hud.panel_header(f, x0, x1, y0, "EXPOSURE")
                f.text("w", x0, y0 + 96, f"{t - self.span[0]:04.1f}", size=fit(50, 4, x1 - x0))
