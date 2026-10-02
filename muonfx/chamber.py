"""CLOUD CHAMBER - Ryoji Ikeda edition.

Charged particle tracks in a magnetic field, presented as precise data:
  * the chamber: log-spiral e+e- pairs, red cosmic muons, a muon that stops and
    decays (mu- -> e- nu nu), hadronic stars, Compton curls. Tracks zip in, glow,
    then condense into dotted droplet trails that evaporate;
  * a supersymmetry-style score strip on top: rules, 16th-note ticks, comb
    histograms and word tags on a red band scrolling toward a red cursor;
  * a bottom band: track log (dense rows), dE/dx converted into a 3-lane
    barcode (test pattern), momentum spectrum;
  * red tracking crosshairs with coordinate readouts, inverted data tags,
    a dot lattice instead of a grid.
Rhythm (128 BPM, 4-bar phrases): bars 1-2 observe, bar 3 hard-cuts to a x3 zoom
on the phrase's hero event, bar 4 returns to the full chamber. Every phrase
opens with a 2-frame inversion; the last beat of the loop is a barcode burst.
Loops seamlessly every T = 16 bars (30 s).
"""
from __future__ import annotations

import math

import numpy as np

from .engine import Frame, hash01, smoothstep

BPM = 128.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
PHRASE = 4 * BAR

STRIP = (40.0, 338.0, 2960.0, 470.0)          # top score strip
CH = (40.0, 505.0, 2960.0, 1455.0)            # chamber window
BOT = (40.0, 1482.0, 2960.0, 1612.0)          # bottom band
BOX = (CH[0] + 2, CH[1] + 2, CH[2] - 2, CH[3] - 2)
CH_C = ((CH[0] + CH[2]) / 2, (CH[1] + CH[3]) / 2)
E_SIGN = 1.0


def _inside(x, y, box=BOX):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _spiral(p0, ang, R0, sign, k, rng, Rmin=2.5, scatter=0.0, max_len=9000.0, box=BOX):
    """Charged track: circle of radius R that shrinks by k per px of path (log spiral)."""
    x, y = p0
    a = ang
    R = R0
    pts = [(x, y)]
    L = 0.0
    while R > Rmin and L < max_len:
        ds = min(max(0.07 * R, 0.6), 30.0)
        a += sign * ds / R
        if scatter:
            a += scatter * rng.normal() * math.sqrt(ds) / math.sqrt(R)
        x += math.cos(a) * ds
        y += math.sin(a) * ds
        L += ds
        R -= k * ds
        pts.append((x, y))
        if not _inside(x, y, box):
            break
    return np.asarray(pts, np.float32)


def _stopping(p0, ang, R0, sign, L_stop, rng, R_end=18.0, scatter=0.0, box=BOX):
    """Heavy particle ranging out: curvature tightens sharply near the end."""
    x, y = p0
    a = ang
    pts = [(x, y)]
    L = 0.0
    while L < L_stop:
        u = L / L_stop
        R = R_end + (R0 - R_end) * (1 - u) ** 1.8
        ds = min(max(0.07 * R, 0.6), 20.0)
        a += sign * ds / R
        if scatter:
            a += scatter * rng.normal() * math.sqrt(ds) / math.sqrt(R)
        x += math.cos(a) * ds
        y += math.sin(a) * ds
        L += ds
        pts.append((x, y))
        if not _inside(x, y, box):
            break
    return np.asarray(pts, np.float32)


def _arclen(p):
    d = np.hypot(*np.diff(p, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(d)]).astype(np.float32)


def _edge_hit(x, y, dx, dy, box=BOX):
    ts = []
    if dx > 1e-6:
        ts.append((box[2] - x) / dx)
    if dx < -1e-6:
        ts.append((box[0] - x) / dx)
    if dy > 1e-6:
        ts.append((box[3] - y) / dy)
    if dy < -1e-6:
        ts.append((box[1] - y) / dy)
    return min(ts) if ts else 0.0


class Chamber:
    name = "chamber"

    def __init__(self, T=30.0, seed=11):
        self.T = T
        self.rng = np.random.default_rng(seed)
        self._tracks, self._tags, self._flashes, self._dashes = [], [], [], []
        self.events = []        # (t, kind, word, amp, momentum, x, y)
        self.heroes = []        # zoom targets per phrase
        self._build()
        self._pack()
        self._build_static()

    # ================================================================== build
    def _track(self, pts, t0, layer, inten, tau, times=None, speed=2500.0, width=1.0, head_r=2.0,
               drop_gap=8.0, drop_r=0.95, drop_i=0.9, drop_tau=3.0, flash=1.4):
        if len(pts) < 2:
            return None
        s = _arclen(pts)
        tv = s / speed if times is None else np.asarray(times, np.float32)
        self._tracks.append(dict(p=pts, s=s, tv=tv.astype(np.float32), t0=t0 % self.T, layer=layer, inten=inten,
                                 tau=tau, width=width, head_r=head_r, drop_gap=drop_gap, drop_r=drop_r,
                                 drop_i=drop_i, drop_tau=drop_tau, flash=flash))
        return self._tracks[-1]

    def _event(self, t, kind, word, amp, p, x, y):
        self.events.append((t % self.T, kind, word, amp, p, x, y))

    @staticmethod
    def _tag_box(x, y, side, rise, word, lines, size=17, dsize=13):
        w = max(len(word) * size * 0.6 + 12, max((len(l) for l in lines), default=0) * dsize * 0.6) + 50
        ty = y - rise
        h = size + 10 + len(lines) * dsize * 1.3
        x0, x1 = (x, x + w) if side > 0 else (x - w, x)
        return x0, min(y, ty - size - 6) - 4, x1, max(y, ty + h) + 4

    def _tag(self, x, y, t, word, lines, red=False, dur=4.2, cands=()):
        T = self.T
        for (cx, cy, ct) in [(x, y, t), *cands]:
            if not (BOX[1] + 60 < cy < BOX[3] - 60):
                continue
            for sd in ([1, -1] if cx < BOX[2] - 600 else [-1, 1]):
                for rise in (34, 76, -30, 118, -72):
                    bx = self._tag_box(cx, cy, sd, rise, word, lines)
                    if bx[0] < BOX[0] + 10 or bx[2] > BOX[2] - 10 or bx[1] < BOX[1] + 70 or bx[3] > BOX[3] - 10:
                        continue
                    clash = False
                    for o in self._tags:
                        d = (ct % T - o["t"]) % T
                        if d >= o["dur"] and (T - d) >= dur:
                            continue
                        ob = o["box"]
                        if bx[0] < ob[2] and ob[0] < bx[2] and bx[1] < ob[3] and ob[1] < bx[3]:
                            clash = True
                            break
                    if not clash:
                        self._tags.append(dict(x=cx, y=cy, t=ct % T, word=word, lines=lines, red=red, dur=dur,
                                               side=sd, rise=rise, box=bx))
                        return

    def _flash(self, x, y, t, layer=1, rmax=70.0, dur=0.8):
        self._flashes.append((x, y, t % self.T, layer, rmax, dur))

    # --- event types ---------------------------------------------------------
    def _muon(self, t):
        rng = self.rng
        charge = 1.0 if rng.random() < 0.56 else -1.0
        theta = float(np.clip(rng.normal(0, 0.36), -0.85, 0.85))
        x0 = rng.uniform(BOX[0] + 120, BOX[2] - 120)
        ang = math.pi / 2 + theta
        p = float(np.exp(rng.normal(1.2, 0.7)))
        pts = _spiral((x0, BOX[1] + 1), ang, R0=5200.0 * p, sign=-charge * E_SIGN, k=0.0, rng=rng, max_len=4000)
        tr = self._track(pts, t, layer=1, inten=1.0, tau=1.2, speed=6000.0, width=1.25, head_r=2.6, drop_gap=7.0,
                         drop_r=1.1, drop_i=1.0, drop_tau=3.2, flash=1.8)
        for _ in range(rng.integers(1, 4)):
            j = int(rng.integers(len(pts) // 5, max(len(pts) // 5 + 1, len(pts) - 2)))
            sx, sy = pts[j]
            d = pts[min(j + 1, len(pts) - 1)] - pts[j]
            da = math.atan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.9, 1.7)
            dp = _spiral((sx, sy), da, R0=rng.uniform(14, 55), sign=E_SIGN, k=rng.uniform(0.07, 0.16), rng=rng,
                         scatter=0.12)
            self._track(dp, t + tr["tv"][j], layer=0, inten=0.8, tau=0.8, speed=700.0, head_r=1.4, drop_gap=5.5)
        word = f"MU{'+' if charge > 0 else '-'}"
        js = (len(pts) * np.array([0.35, 0.5, 0.22, 0.62, 0.72])).astype(int)
        cands = [(pts[j][0], pts[j][1], t + tr["tv"][j]) for j in js]
        self._tag(*cands[0], word, [f"P {p:.4f} GEV/C", f"THETA {math.degrees(theta):+.2f} DEG"], red=True,
                  cands=cands[1:])
        self._event(t, "mu", word, 1.0, p, float(x0), float(BOX[1]))

    def _decay(self, t, where=None):
        rng = self.rng
        charge = 1.0 if rng.random() < 0.5 else -1.0
        theta = float(np.clip(rng.normal(0, 0.3), -0.6, 0.6))
        L_stop = rng.uniform(560, 820)
        if where is None:
            x0 = rng.uniform(BOX[0] + 400, BOX[2] - 400)
        else:   # aim the stop point near `where`
            x0 = where[0] - math.cos(math.pi / 2 + theta) * (where[1] - BOX[1])
            x0 = float(np.clip(x0, BOX[0] + 200, BOX[2] - 200))
            L_stop = float(np.clip((where[1] - BOX[1]) / max(math.cos(theta), 0.5), 380, 900))
        pts = _stopping((x0, BOX[1] + 1), math.pi / 2 + theta, R0=rng.uniform(2600, 5200), sign=-charge * E_SIGN,
                        L_stop=L_stop, rng=rng, scatter=0.25)
        s = _arclen(pts)
        dur = 1.05
        tv = dur * (1.0 - np.sqrt(np.clip(1.0 - s / s[-1], 0, 1)))
        self._track(pts, t, layer=1, inten=1.05, tau=1.6, times=tv, width=1.5, head_r=2.8, drop_gap=6.0,
                    drop_r=1.2, drop_i=1.0, drop_tau=4.0, flash=1.8)
        ex, ey = float(pts[-1][0]), float(pts[-1][1])
        t_dec = t + dur + 0.35
        self._flash(ex, ey, t + dur, layer=1, rmax=40, dur=0.5)
        self._flash(ex, ey, t_dec, layer=0, rmax=120, dur=0.9)
        word = f"MU{'+' if charge > 0 else '-'}"
        self._tag(ex, ey, t + dur, f"{word} STOP", ["TAU 2.197E-6 S", f"X {ex:07.2f}  Y {ey:07.2f}"], red=True,
                  dur=5.0)
        e_pts = _spiral((ex, ey), rng.uniform(0, 2 * math.pi), R0=rng.uniform(190, 400), sign=-charge * E_SIGN,
                        k=rng.uniform(0.06, 0.1), rng=rng, scatter=0.06)
        self._track(e_pts, t_dec, layer=0, inten=0.95, tau=1.0, speed=3200.0, width=1.1, head_r=2.4, drop_gap=7.0)
        for _ in range(2):
            na = rng.uniform(0, 2 * math.pi)
            ln = _edge_hit(ex, ey, math.cos(na), math.sin(na))
            self._dashes.append((ex, ey, ex + math.cos(na) * ln, ey + math.sin(na) * ln, t_dec % self.T,
                                 max(2, int(ln / 22))))
        E = rng.uniform(12, 52)
        self._tag(ex + 1, ey + 1, t_dec, "DECAY", ["MU -> E NU NU", f"E {E:.2f} MEV"], dur=4.6)
        self._event(t, "mu", word, 1.0, 0.11, ex, float(BOX[1]))
        self._event(t_dec, "decay", "DECAY", 0.9, 0.035, ex, ey)
        return (ex, ey), t_dec

    def _pair(self, t, where=None):
        rng = self.rng
        if where is None:
            vx = rng.uniform(BOX[0] + 350, BOX[2] - 350)
            vy = rng.uniform(BOX[1] + 150, BOX[3] - 150)
        else:
            vx, vy = where
        g_ang = rng.uniform(0.35, 2.8)
        back = _edge_hit(vx, vy, -math.cos(g_ang), -math.sin(g_ang))
        sx, sy = vx - math.cos(g_ang) * back, vy - math.sin(g_ang) * back
        self._dashes.append((sx, sy, vx, vy, t % self.T, max(2, int(back / 22))))
        t_v = t + 0.22
        E = float(np.exp(rng.normal(-2.6, 0.7))) if where is None else rng.uniform(0.09, 0.16)
        share = rng.uniform(0.3, 0.7)
        for q, frac in ((-1.0, share), (1.0, 1 - share)):
            pts = _spiral((vx, vy), g_ang + q * rng.uniform(0.02, 0.1), R0=3400.0 * E * frac + 70.0,
                          sign=-q * E_SIGN, k=rng.uniform(0.07, 0.13), rng=rng, scatter=0.06)
            self._track(pts, t_v, layer=0, inten=0.95, tau=0.9, speed=2700.0, width=1.1, head_r=2.2)
        self._flash(vx, vy, t_v, layer=0, rmax=40, dur=0.5)
        self._tag(vx, vy, t_v, "E+E-", [f"GAMMA {E:.4f} GEV", f"X {vx:07.2f}  Y {vy:07.2f}"])
        self._event(t_v, "pair", "E+E-", 0.65, E, vx, vy)
        return (vx, vy), t_v

    def _star(self, t, where=None):
        rng = self.rng
        if where is None:
            vx = rng.uniform(BOX[0] + 450, BOX[2] - 450)
            vy = rng.uniform(BOX[1] + 250, BOX[3] - 220)
        else:
            vx, vy = where
        ia = rng.uniform(1.25, 1.9)
        back = _edge_hit(vx, vy, -math.cos(ia), -math.sin(ia))
        inc = _spiral((vx - math.cos(ia) * back, vy - math.sin(ia) * back), ia, R0=18000, sign=1.0, k=0.0, rng=rng,
                      max_len=back)
        tr = self._track(inc, t, layer=0, inten=1.0, tau=1.2, speed=6000.0, width=1.3, head_r=2.6)
        t_v = t + float(tr["tv"][-1])
        n = int(rng.integers(5, 9))
        E = rng.uniform(0.8, 3.2)
        for _ in range(n):
            a = rng.uniform(0, 2 * math.pi)
            if rng.random() < 0.35:
                pts = _stopping((vx, vy), a, R0=rng.uniform(900, 4000), sign=rng.choice([-1, 1]),
                                L_stop=rng.uniform(70, 240), rng=rng, R_end=60)
                self._track(pts, t_v, layer=0, inten=1.1, tau=1.5, speed=1400.0, width=2.4, head_r=2.6,
                            drop_gap=4.5, drop_r=1.2, drop_tau=3.6)
            else:
                pts = _spiral((vx, vy), a, R0=rng.uniform(500, 9000), sign=rng.choice([-1, 1]),
                              k=rng.uniform(0.0, 0.6), rng=rng, max_len=rng.uniform(160, 900), scatter=0.3)
                self._track(pts, t_v, layer=0, inten=0.9, tau=1.1, speed=3300.0, width=1.1, head_r=2.2)
                if rng.random() < 0.4 and len(pts) > 3:
                    kx, ky = pts[-1]
                    d = pts[-1] - pts[-2]
                    ka = math.atan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.4, 1.2)
                    kp = _spiral((kx, ky), ka, R0=rng.uniform(300, 1600), sign=E_SIGN, k=rng.uniform(0.05, 0.3),
                                 rng=rng, max_len=rng.uniform(150, 600), scatter=0.4)
                    self._track(kp, t_v + _arclen(pts)[-1] / 3300.0, layer=0, inten=0.8, tau=0.9, speed=3000.0)
        self._flash(vx, vy, t_v, layer=0, rmax=100, dur=0.8)
        self._tag(vx, vy, t_v, f"STAR N={n}", [f"E {E:.3f} GEV", f"X {vx:07.2f}  Y {vy:07.2f}"], dur=4.6)
        self._event(t_v, "star", "STAR", 0.9, E, vx, vy)
        return (vx, vy), t_v

    def _curl(self, t, faint=False):
        rng = self.rng
        x = rng.uniform(BOX[0] + 60, BOX[2] - 60)
        y = rng.uniform(BOX[1] + 60, BOX[3] - 40)
        pts = _spiral((x, y), rng.uniform(0, 2 * math.pi), R0=rng.uniform(16, 95), sign=E_SIGN,
                      k=rng.uniform(0.07, 0.17), rng=rng, scatter=0.15)
        self._track(pts, t, layer=0, inten=0.5 if faint else 0.75, tau=0.8, speed=820.0, head_r=1.3, drop_gap=6.0,
                    drop_r=0.8, drop_i=0.65 if faint else 0.8, drop_tau=2.4)
        if not faint and rng.random() < 0.25:
            self._tag(x, y, t, "E-", [f"{rng.uniform(0.0001, 0.009):.10f}"], dur=2.6)
        self._event(t, "e", "E-", 0.18, 0.004, x, y)

    def _alpha(self, t):
        rng = self.rng
        x = rng.uniform(BOX[0] + 80, BOX[2] - 80)
        y = rng.uniform(BOX[1] + 80, BOX[3] - 60)
        pts = _spiral((x, y), rng.uniform(0, 2 * math.pi), R0=40000, sign=1.0, k=0.0, rng=rng,
                      max_len=rng.uniform(60, 170))
        self._track(pts, t, layer=0, inten=0.9, tau=1.3, speed=900.0, width=3.6, head_r=2.6, drop_gap=3.0,
                    drop_r=1.3, drop_i=0.95, drop_tau=3.0)
        if rng.random() < 0.5:
            self._tag(x, y, t, "ALPHA", [f"{rng.uniform(4.2, 5.8):.3f} MEV"], dur=3.0)
        self._event(t, "alpha", "ALPHA", 1.1, 0.005, x, y)

    def _build(self):
        rng = self.rng
        T = self.T
        n_bars = int(round(T / BAR))
        n_beats = n_bars * 4
        # one hero event per phrase, peaking just before the zoom bar (bar 3 of 4)
        hero_kinds = ["pair", "decay", "star", "decay"]
        for p in range(n_bars // 4):
            kind = hero_kinds[p % len(hero_kinds)]
            where = (rng.uniform(900, 2100), rng.uniform(760, 1200))
            zoom_start = (4 * p + 2) * BAR
            n0 = len(self._tracks)
            if kind == "pair":
                v, tv = self._pair(zoom_start - 0.55 - 0.22, where)
            elif kind == "star":
                v, tv = self._star(zoom_start - 1.2, where)
            else:
                v, tv = self._decay(zoom_start - 1.05 - 0.35 - 0.3, where)
            # frame the products of the event (skip the incoming track for decays / stars)
            prod = self._tracks[n0 + (0 if kind == "pair" else 1):]
            pts = np.concatenate([d["p"] for d in prod] + [np.array([v], np.float32)])
            lo, hi = pts.min(0), pts.max(0)
            cx, cy = (lo + hi) / 2
            bw, bh = max(hi[0] - lo[0], 120.0), max(hi[1] - lo[1], 120.0)
            z = float(np.clip(min((CH[2] - CH[0]) / (bw * 1.3), (CH[3] - CH[1]) / (bh * 1.3)), 1.8, 3.6))
            self.heroes.append(dict(x=float(cx), y=float(cy), z=z, t=tv, kind=kind))
        # muons on (most) downbeats + a few syncopated ones
        for b in range(n_bars):
            if b % 4 != 2:
                self._muon(b * BAR + rng.choice([0.0, 0.0, 0.5 * BEAT]))
        for b in rng.choice(np.arange(n_beats), size=n_beats // 7, replace=False):
            if b % 4:
                self._muon(b * BEAT)
        for b in rng.choice(np.arange(n_beats), size=n_beats // 6, replace=False):
            self._pair((b + 0.5) * BEAT)
        for b in rng.choice(np.arange(n_beats), size=2, replace=False):
            self._star(b * BEAT + 0.25 * BEAT)
        for _ in range(int(n_beats * 1.1)):
            self._curl(rng.integers(0, n_beats * 4) * BEAT / 4, faint=rng.random() < 0.5)
        for _ in range(n_beats // 6):
            self._alpha(rng.integers(0, n_beats * 2) * BEAT / 2)

    # =================================================================== pack
    def _pack(self):
        rng = self.rng
        tr = self._tracks
        self.t0 = np.array([d["t0"] for d in tr], np.float32)
        self.layer = np.array([1 if d["layer"] else 0 for d in tr], np.int8)
        self.inten = np.array([d["inten"] for d in tr], np.float32)
        self.tau = np.array([d["tau"] for d in tr], np.float32)
        self.width = np.array([d["width"] for d in tr], np.float32)
        self.head_r = np.array([d["head_r"] for d in tr], np.float32)
        self.flashk = np.array([d["flash"] for d in tr], np.float32)
        self.dur = np.array([d["tv"][-1] for d in tr], np.float32)
        self.length = np.array([d["s"][-1] for d in tr], np.float32)
        self.drop_tau = np.array([d["drop_tau"] for d in tr], np.float32)
        self.life = self.dur + self.drop_tau * 3.2 + 1.0
        P, TV, TID, SA = [], [], [], []
        off = 0
        for i, d in enumerate(tr):
            m = len(d["p"])
            P.append(d["p"]); TV.append(d["tv"]); TID.append(np.full(m, i, np.int32))
            SA.append(np.arange(off, off + m - 1, dtype=np.int64))
            off += m
        self.P = np.concatenate(P).astype(np.float32)
        self.TV = np.concatenate(TV).astype(np.float32)
        self.TID = np.concatenate(TID)
        self.SA = np.concatenate(SA)
        DP, DT, DI, DU, DD, DR, DA = [], [], [], [], [], [], []
        for i, d in enumerate(tr):
            s = d["s"]
            if s[-1] < 2:
                continue
            cnt = max(1, int(s[-1] / d["drop_gap"]))
            sd = np.sort(rng.uniform(0, s[-1], cnt)).astype(np.float32)
            sd = 0.5 * sd + 0.5 * (np.arange(cnt) + 0.5) * (s[-1] / cnt)
            DP.append(np.stack([np.interp(sd, s, d["p"][:, 0]), np.interp(sd, s, d["p"][:, 1])], 1)
                      + rng.normal(0, 0.5, (cnt, 2)))
            DT.append(np.interp(sd, s, d["tv"]))
            DI.append(np.full(cnt, i, np.int32))
            DU.append(rng.uniform(0.35, 1.5, cnt))
            ang = rng.uniform(0, 2 * np.pi, cnt)
            DD.append(np.stack([np.cos(ang), np.sin(ang)], 1) * rng.uniform(0.6, 2.2, (cnt, 1)))
            DR.append(d["drop_r"] * rng.uniform(0.75, 1.25, cnt))
            DA.append(d["drop_i"] * rng.uniform(0.6, 1.2, cnt))
        self.DP = np.concatenate(DP).astype(np.float32)
        self.DT = np.concatenate(DT).astype(np.float32)
        self.DI = np.concatenate(DI)
        self.DU = np.concatenate(DU).astype(np.float32)
        self.DD = np.concatenate(DD).astype(np.float32)
        self.DR = np.concatenate(DR).astype(np.float32)
        self.DA = np.concatenate(DA).astype(np.float32)
        ev = sorted(self.events, key=lambda e: e[0])
        self.events = ev
        self.ev_t = np.array([e[0] for e in ev], np.float32)
        self.ev_amp = np.array([e[3] for e in ev], np.float32)
        self.ev_p = np.array([e[4] for e in ev], np.float32)
        self.ev_kind = [e[1] for e in ev]
        self.ev_word = [e[2] for e in ev]

    def _build_static(self):
        gx = np.arange(CH[0] + 20, CH[2], 40.0)
        gy = np.arange(CH[1] + 15, CH[3], 40.0)
        X, Y = np.meshgrid(gx, gy)
        self.lat = np.stack([X.ravel(), Y.ravel()], 1).astype(np.float32)
        major = (np.round((X - CH[0] - 20) / 40) % 5 == 0) & (np.round((Y - CH[1] - 15) / 40) % 5 == 0)
        self.lat_major = major.ravel()

    # ================================================================= render
    def render(self, t, W=3000, H=1688):
        T = self.T
        t = t % T
        f = Frame(W, H)
        bar = int(t // BAR) % int(round(T / BAR))
        phrase = bar // 4
        in_bar = t - bar * BAR
        zoom_mode = bar % 4 == 2
        burst = t >= T - BEAT
        invert = (t % PHRASE) < 0.06

        g = (t - self.t0) % T
        alive = g < self.life

        f.set_clip(*CH)
        if zoom_mode:
            h = self.heroes[phrase % len(self.heroes)]
            zoom = h["z"] * (0.9 + 0.2 * float(smoothstep(0.0, BAR, in_bar)))
            half_w = (CH[2] - CH[0]) / 2 / zoom
            half_h = (CH[3] - CH[1]) / 2 / zoom
            cx = float(np.clip(h["x"], CH[0] + half_w, CH[2] - half_w))
            cy = float(np.clip(h["y"], CH[1] + half_h, CH[3] - half_h))
            f.set_view(zoom, cx, cy, CH_C[0], CH_C[1])
        if burst:
            self._draw_barcode_burst(f, t)
        else:
            self._draw_lattice(f)
            self._draw_dashes(f, t)
            heads = self._draw_tracks(f, g, alive)
            self._draw_drops(f, g, alive)
            self._draw_flashes(f, t)
            self._draw_crosshairs(f, heads)
            self._draw_tags(f, t)
        view = (f.vz, f.vcx, f.vcy, f.vsx, f.vsy)
        f.set_view()
        f.set_clip()
        self._draw_frame(f, view)
        self._draw_strip(f, t)
        self._draw_bottom(f, t, g)
        self._draw_header(f, t, zoom_mode, view)
        return f.finish(bloom_weights=(0.3, 0.26, 0.2, 0.16, 0.13, 0.1, 0.08, 0.06), bloom_gain=0.75,
                        invert=invert, invert_rect=(0, 318, 3000, 1688))

    # --- helpers to draw in screen space while a zoom view is active ------------
    @staticmethod
    def _push(f):
        v = (f.vz, f.vcx, f.vcy, f.vsx, f.vsy)
        f.set_view()
        return v

    @staticmethod
    def _pop(f, v):
        f.vz, f.vcx, f.vcy, f.vsx, f.vsy = v

    # --- chamber content -------------------------------------------------------
    def _draw_lattice(self, f):
        p = self.lat
        mn = ~self.lat_major
        f.pixels("w", p[mn, 0], p[mn, 1], 0.55, snap=f.vz == 1.0)
        m = p[self.lat_major]
        f.crosses("w", m[:, 0], m[:, 1], 4.0 / f.vz, 0.5)

    def _draw_dashes(self, f, t):
        for x0, y0, x1, y1, t0, n in self._dashes:
            a = (t - t0) % self.T
            if a > 2.4:
                continue
            reveal = min(1.0, a / 0.2)
            inten = 0.3 * math.exp(-a / 0.9)
            k = np.arange(n)
            u0 = k / n
            u1 = np.minimum((k + 0.45) / n, reveal)
            keep = u0 <= reveal
            u0, u1 = u0[keep], u1[keep]
            if len(u0):
                f.segments("w", x0 + (x1 - x0) * u0, y0 + (y1 - y0) * u0, x0 + (x1 - x0) * u1, y0 + (y1 - y0) * u1,
                           inten)

    def _intensity(self, a, tid):
        return self.inten[tid] * (np.exp(-a / self.tau[tid]) + self.flashk[tid] * np.exp(-a / 0.06))

    def _draw_tracks(self, f, g, alive):
        sa = self.SA
        tid = self.TID[sa]
        G = g[tid]
        ta = self.TV[sa]
        vis = alive[tid] & (G >= ta)
        sa, tid, G, ta = sa[vis], tid[vis], G[vis], ta[vis]
        tb = self.TV[sa + 1]
        pa = self.P[sa]
        pb = self.P[sa + 1].copy()
        part = G < tb
        frac = np.clip((G[part] - ta[part]) / np.maximum(tb[part] - ta[part], 1e-6), 0, 1)
        pb[part] = pa[part] + (pb[part] - pa[part]) * frac[:, None]
        ia = self._intensity(G - ta, tid)
        ib = self._intensity(np.where(part, 0.0, G - tb), tid)
        lay = self.layer[tid]
        for L, name in ((0, "w"), (1, "r")):
            m = lay == L
            if m.any():
                f.segments(name, pa[m, 0], pa[m, 1], pb[m, 0], pb[m, 1], ia[m], ib[m], width=self.width[tid[m]])
        hp, ht = pb[part], tid[part]
        for L, name in ((0, "w"), (1, "r")):
            m = self.layer[ht] == L
            if m.any():
                f.dots(name, hp[m, 0], hp[m, 1], self.head_r[ht[m]], 1.4)
        return hp, self.layer[ht]

    def _draw_drops(self, f, g, alive):
        tid = self.DI
        a = g[tid] - self.DT
        vis = alive[tid] & (a > 0.04)
        if not vis.any():
            return
        a, tid = a[vis], tid[vis]
        tau = self.drop_tau[tid] * self.DU[vis]
        inten = 1.3 * self.DA[vis] * smoothstep(0.05, 0.5, a) * np.exp(-np.maximum(a - 0.5, 0) / tau)
        inten *= 1.0 - smoothstep(tau * 1.6, tau * 2.4, a)
        pos = self.DP[vis] + self.DD[vis] * np.sqrt(a)[:, None] * 1.5
        lay = self.layer[tid]
        for L, name in ((0, "w"), (1, "r")):
            m = (lay == L) & (inten > 0.01)
            if m.any():
                f.dots(name, pos[m, 0], pos[m, 1], self.DR[vis][m] * 1.15, inten[m])

    def _draw_flashes(self, f, t):
        for x, y, t0, layer, rmax, dur in self._flashes:
            a = (t - t0) % self.T
            if a > dur:
                continue
            u = a / dur
            name = "r" if layer else "w"
            f.rings(name, [x], [y], [max(rmax * (1 - (1 - u) ** 3), 1.0)], 0.9 * (1 - u) ** 1.5)
            f.dots(name, [x], [y], 3.0 * (1 - u) + 1.2, 1.5 * (1 - u))

    def _draw_crosshairs(self, f, heads):
        hp, hl = heads
        red = hl == 1
        if not red.any():
            return
        big = 1e5
        for x, y in hp[red][:3]:
            x, y = float(x), float(y)
            f.segments("r", [-big, x], [y, -big], [big, x], [y, big], 0.5)
            X, Y = f.to_screen(x, y)
            v = self._push(f)
            f.rings("r", [X], [Y], [14.0], 0.9)
            f.text("r", X + 22, Y - 14, f"X {x:07.2f}", size=15)
            f.text("r", X + 22, Y + 4, f"Y {y:07.2f}", size=15)
            f.tag("r", CH[0] + 10, Y + 6, f"{y:07.2f}", size=13, pad=3)
            f.tag("r", X + 4, CH[1] + 22, f"{x:07.2f}", size=13, pad=3)
            self._pop(f, v)

    def _draw_tags(self, f, t):
        for tg in self._tags:
            a = (t - tg["t"]) % self.T
            if a > tg["dur"]:
                continue
            alpha = 1.0 - float(smoothstep(tg["dur"] - 0.9, tg["dur"], a))
            side = tg["side"]
            X, Y = f.to_screen(tg["x"], tg["y"])
            ex, ey = X + side * 30, Y - tg["rise"]
            hx = ex + side * 22
            grow = float(smoothstep(0.0, 0.1, a))
            v = self._push(f)
            f.segments("w", [X, ex], [Y, ey], [X + (ex - X) * grow, ex + (hx - ex) * grow], [Y + (ey - Y) * grow, ey],
                       0.6 * alpha)
            f.dots("w", [X], [Y], 2.2, alpha)
            if a > 0.06:
                lay = "r" if tg["red"] else "w"
                anchor = "ls" if side > 0 else "rs"
                tx = hx + 8 * side
                f.tag(lay, tx, ey + 6, tg["word"], size=18, alpha=alpha, anchor=anchor)
                for k, ln in enumerate(tg["lines"]):
                    n = int(min(len(ln), max(0.0, (a - 0.12 - 0.08 * k) * 90)))
                    if n:
                        txt = ln[:n] if side > 0 else ln[len(ln) - n:]
                        f.text("w", tx, ey + 32 + k * 19, txt, size=15, alpha=0.9 * alpha, anchor=anchor)
            self._pop(f, v)

    def _draw_barcode_burst(self, f, t):
        """Last beat of the loop: the chamber turns into a test-pattern barcode."""
        u = (t - (self.T - BEAT)) / BEAT
        fr = int(round(t * 30))
        x0, y0, x1, y1 = CH
        lanes = 7
        lh = (y1 - y0) / lanes
        span = x1 - x0
        for ln in range(lanes):
            n = 520
            k = np.arange(n)
            wdt = 1 + (hash01(k, ln + 11) * 9).astype(int)
            on = hash01(k, ln + 5, 7) < 0.4 + 0.45 * u
            xs = np.cumsum(wdt) - wdt
            shift = fr * (13 + 7 * ln) * (1 if ln % 2 else -1)
            xs = (xs + shift) % span + x0
            f.rects("w", xs[on], y0 + ln * lh + 2, xs[on] + wdt[on], y0 + (ln + 1) * lh - 2, 1.0)
        f.rects("r", x0, (y0 + y1) / 2 - 3, x1, (y0 + y1) / 2 + 3, 1.0)

    # --- HUD ---------------------------------------------------------------------
    def _draw_frame(self, f, view):
        x0, y0, x1, y1 = CH
        f.rect("w", x0, y0, x1, y1, 0.85)
        vz, vcx, vcy, vsx, vsy = view
        # live rulers in chamber coordinates: they follow the zoom
        wy0 = (y0 - vsy) / vz + vcy
        wy1 = (y1 - vsy) / vz + vcy
        vals = np.arange(math.ceil(wy0 / 10.0) * 10.0, wy1, 10.0)
        ys = (vals - vcy) * vz + vsy
        ln = np.where(vals % 250 == 0, 16.0, np.where(vals % 50 == 0, 8.0, 3.0))
        f.segments("w", np.full_like(ys, x0), ys, x0 + ln, ys, 0.7)
        f.segments("w", np.full_like(ys, x1), ys, x1 - ln, ys, 0.7)
        for v, yy in zip(vals[vals % 250 == 0], ys[vals % 250 == 0]):
            f.text("w", x0 + 22, yy + 5, f"{int(v):04d}", size=12, alpha=0.7)
        wx0 = (x0 - vsx) / vz + vcx
        wx1 = (x1 - vsx) / vz + vcx
        vals = np.arange(math.ceil(wx0 / 10.0) * 10.0, wx1, 10.0)
        xs = (vals - vcx) * vz + vsx
        ln = np.where(vals % 250 == 0, 16.0, np.where(vals % 50 == 0, 8.0, 3.0))
        f.segments("w", xs, np.full_like(xs, y1), xs, y1 - ln, 0.7)
        for v, xx in zip(vals[vals % 250 == 0], xs[vals % 250 == 0]):
            f.text("w", xx + 4, y1 - 22, f"{int(v):04d}", size=12, alpha=0.7)

    def _draw_strip(self, f, t):
        x0, y0, x1, y1 = STRIP
        ta, tb = t - 5.5, t + 2.0
        span = tb - ta

        def X(tt):
            return x0 + (tt - ta) / span * (x1 - x0)

        f.segments("w", [x0, x0], [y0, y1], [x1, x1], [y0, y1], 0.9)
        k0 = math.ceil(ta / (BEAT / 4))
        k1 = math.floor(tb / (BEAT / 4))
        ks = np.arange(k0, k1 + 1)
        xs = X(ks * BEAT / 4)
        ln = np.where(ks % 16 == 0, 16.0, np.where(ks % 4 == 0, 9.0, 4.0))
        f.segments("w", xs, np.full_like(xs, y0), xs, y0 + ln, 0.8)
        f.segments("w", xs, np.full_like(xs, y1), xs, y1 - ln, 0.8)
        for k, xx in zip(ks[ks % 16 == 0], xs[ks % 16 == 0]):
            bar = int(k // 16) % 16
            f.text("w", xx + 5, y0 + 30, f"{bar + 1:02d}", size=13, alpha=0.85)
            f.text("w", xx + 5, y1 - 22, f"{(k * BEAT / 4) % self.T:06.3f}", size=12, alpha=0.6)
        yb = (y0 + y1) / 2 - 8
        f.rects("r", x0, yb - 3, x1, yb + 3, 1.0)
        placed = []
        for (te, kind, word, amp, p, ex, ey) in self.events:
            for tt in (te - self.T, te, te + self.T):
                if not (ta - 0.2 <= tt <= tb + 0.2):
                    continue
                xe = X(tt)
                key = int(te * 1000)
                n = 3 + int(hash01(key, 1) * (4 + 10 * amp))
                bx = xe + np.arange(n) * 4.0
                hh = (6 + 40 * amp) * (0.25 + 0.75 * hash01(key, np.arange(n)))
                f.rects("w", bx, y0 + 1, bx + 2, y0 + 1 + hh, 0.95)
                if kind in ("mu", "pair", "decay", "star", "alpha"):
                    hb = (4 + 24 * amp) * (0.2 + 0.8 * hash01(key, np.arange(n) + 50))
                    f.rects("w", bx, y1 - hb, bx + 2, y1 - 1, 0.7)
                    past = tt <= t
                    near = abs(tt - t) < 0.12
                    row = 1 if any(abs(r - xe) < 120 for r in placed) else 0
                    placed.append(xe)
                    lay = "r" if (near or (kind in ("mu", "decay") and past)) else "w"
                    f.tag(lay, xe, yb + 6 + row * 26, word, size=14, pad=4, alpha=1.0 if past else 0.5)
        xc = X(t)
        f.segments("r", [xc], [y0 - 4], [xc], [y1 + 4], 1.2, width=1.6)
        f.tag("r", xc + 6, y1 + 22, f"T {t:06.3f}", size=13, pad=3)

    def _signal(self, tt):
        out = np.zeros_like(tt, dtype=np.float32)
        for te, amp in zip(self.ev_t, self.ev_amp):
            a = (tt - te) % self.T
            m = a < 2.0
            out[m] += amp * np.exp(-a[m] / 0.2) * (1 - np.exp(-a[m] / 0.015))
        return out

    def _draw_bottom(self, f, t, g):
        x0, y0, x1, y1 = BOT
        # --- track log ---------------------------------------------------------
        lx0, lx1 = x0, 980.0
        f.rects("w", lx0, y0, lx1, y0 + 5, 0.95)
        f.tag("w", lx0 + 4, y0 - 10, "TRACK_LOG", size=12, pad=3)
        order = np.argsort(g)
        rows = [i for i in order if g[i] < 12.0][:24]
        for r, i in enumerate(rows):
            yy = y0 + 12 + r * 4.6
            L = min(1.0, math.log10(1 + self.length[i]) / 3.6) * (lx1 - lx0 - 60)
            prog = min(1.0, g[i] / max(self.dur[i], 1e-3))
            lay = "r" if self.layer[i] else "w"
            f.rects(lay, lx0 + 40, yy, lx0 + 40 + L * prog, yy + 1.6, 0.9 * math.exp(-g[i] / 6.0) + 0.1)
            f.rects("w", lx0 + 4, yy, lx0 + 4 + 28 * hash01(i, 3), yy + 1.6, 0.5)
        cur = lx0 + (lx1 - lx0) * ((t % BAR) / BAR)
        f.segments("r", [cur], [y0 + 7], [cur], [y1], 1.0)
        # --- dE/dx barcode (3 lanes) ------------------------------------------------
        bx0, bx1 = 1020.0, 1980.0
        f.rects("w", bx0, y0, bx1, y0 + 5, 0.95)
        f.tag("w", bx0 + 4, y0 - 10, "DE/DX >> BARCODE", size=12, pad=3)
        cols = 240
        dt = 4.0 / cols
        k_first = math.floor((t - 4.0) / dt)
        kk = k_first + np.arange(cols)
        sig = self._signal(kk * dt)
        cw = (bx1 - bx0) / cols
        frac = (t - 4.0) / dt - k_first
        xs = bx0 + (np.arange(cols) - frac) * cw
        lane_h = (y1 - y0 - 12) / 3
        for ln in range(3):
            dens = 0.08 + 0.85 * np.tanh(1.6 * sig * (1.0 - 0.25 * ln))
            on = hash01(kk, ln + 3) < dens
            ly0 = y0 + 10 + ln * lane_h
            m = on & (xs >= bx0) & (xs + cw <= bx1)
            f.rects("w", xs[m], ly0, xs[m] + cw, ly0 + lane_h - 3, 0.95)
        f.segments("r", [bx1 - 2], [y0 + 7], [bx1 - 2], [y1], 1.2)
        # --- spectrum --------------------------------------------------------------
        sx0, sx1 = 2020.0, 2960.0
        f.rects("w", sx0, y0, sx1, y0 + 5, 0.95)
        f.tag("w", sx0 + 4, y0 - 10, "SPECTRUM P [GEV/C]", size=12, pad=3)
        nb = 48
        edges = np.linspace(math.log10(0.002), math.log10(40.0), nb + 1)
        a = (t - self.ev_t) % self.T
        w = np.where(a < 12.0, np.exp(-a / 4.0) * (1 - np.exp(-a / 0.05)), 0.0)
        b = np.clip(np.searchsorted(edges, np.log10(np.maximum(self.ev_p, 1e-4))) - 1, 0, nb - 1)
        h = np.bincount(b, weights=w, minlength=nb)[:nb]
        bw = (sx1 - sx0) / nb
        hh = (y1 - y0 - 18) * np.tanh(0.7 * h)
        xs = sx0 + np.arange(nb) * bw
        f.rects("w", xs + 2, y1 - hh, xs + bw - 2, y1, 0.9)
        mu = [i for i, k in enumerate(self.ev_kind) if k == "mu" and a[i] < 12.0]
        if mu:
            j = min(mu, key=lambda i: a[i])
            xb = sx0 + b[j] * bw
            f.rects("r", xb + 2, y0 + 10, xb + bw - 2, y1, 1.0)
            f.text("r", xb + bw + 6, y0 + 26, f"{self.ev_p[j]:.4f}", size=13)

    def _draw_header(self, f, t, zoom_mode, view):
        x0, y0, x1, y1 = CH
        a = (t - self.ev_t) % self.T
        rate = float((a < 8.0).sum()) / 8.0
        mu_rate = float(sum(1 for i, k in enumerate(self.ev_kind) if k == "mu" and a[i] < 8.0)) / 8.0
        f.text("w", x0 + 34, y0 + 30, "CLOUD_CHAMBER_01", size=16, alpha=0.95)
        f.text("w", x0 + 34, y0 + 50, "B 1.50 T // 128 BPM // SENSITIVE 30 MS", size=12, alpha=0.6)
        f.text("w", x1 - 34, y0 + 30, f"TRACKS/S {rate:05.2f}", size=16, alpha=0.95, anchor="rs")
        f.text("r", x1 - 34, y0 + 50, f"MU/S {mu_rate:05.2f}", size=12, alpha=0.95, anchor="rs")
        if zoom_mode:
            vz, vcx, vcy, vsx, vsy = view
            f.tag("w", x0 + 34, y0 + 90, f"ZOOM X{vz:.2f}", size=16, pad=4)
            f.text("w", x0 + 34, y0 + 120, f"CENTRE {vcx:07.2f} {vcy:07.2f}", size=12, alpha=0.8)
