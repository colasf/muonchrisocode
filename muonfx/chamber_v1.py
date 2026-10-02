"""CLOUD CHAMBER - charged particle tracks in a magnetic field.

Flat HUD in the language of Scene2 (white frame, '+' grid, red-grid scopes,
Space Mono labels). Tracks are drawn by a moving head, glow as a solid line,
then condense into dotted droplet trails that diffuse and evaporate.

Physics flavour (B field into the wall):
  * cosmic muons (red) cross from the top, almost straight, knocking out small
    delta-ray electron curls;
  * some muons stop and decay (mu- -> e- nu nu): a red hook, a flash, then an
    electron spiral;
  * photon conversions (gamma -> e+ e-) draw opposite logarithmic spirals;
  * hadronic "stars", Compton curls and short fat alpha tracks fill the field.
Events sit on a 128 BPM grid; the whole piece loops seamlessly every T seconds.
"""
from __future__ import annotations

import math

import numpy as np

from .engine import Frame, smoothstep, periodic_noise

BPM = 128.0
BEAT = 60.0 / BPM

FRAME_BOX = (35.0, 310.0, 2940.0, 1620.0)      # same frame as Scene2 (fits the wall)
BOX = (38.0, 313.0, 2937.0, 1617.0)            # tracks live inside the frame
PANEL_Y0, PANEL_Y1 = 1452.0, 1592.0
PANELS = [(175.0, 505.0), (1335.0, 1665.0), (2495.0, 2825.0)]

E_SIGN = 1.0          # curl direction of negative particles (e-, mu-)


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
    """Distance from an inside point along (dx, dy) to the box border."""
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


class ChamberV1:
    name = "chamber_v1"

    def __init__(self, T=30.0, seed=11):
        self.T = T
        self.rng = np.random.default_rng(seed)
        self._tracks = []
        self._labels = []
        self._flashes = []
        self._pulses = []       # (t, amplitude, kind, momentum, y)
        self._build()
        self._pack()
        self._build_static()

    # ------------------------------------------------------------------ build
    def _track(self, pts, t0, layer, inten, tau, times=None, speed=2500.0, width=1.0, head_r=2.2,
               drop_gap=9.0, drop_r=1.25, drop_i=0.85, drop_tau=3.2, flash=1.6):
        if len(pts) < 2:
            return None
        s = _arclen(pts)
        tv = s / speed if times is None else np.asarray(times, np.float32)
        self._tracks.append(dict(p=pts, s=s, tv=tv.astype(np.float32), t0=t0 % self.T, layer=layer, inten=inten,
                                 tau=tau, width=width, head_r=head_r, drop_gap=drop_gap, drop_r=drop_r,
                                 drop_i=drop_i, drop_tau=drop_tau, flash=flash))
        return self._tracks[-1]

    @staticmethod
    def _label_box(x, y, side, rise, text, size):
        w = 56 + len(text) * size * 0.6
        ey = y - rise
        x0, x1 = (x, x + w) if side > 0 else (x - w, x)
        return x0, min(y, ey - size) - 4, x1, max(y, ey + 8) + 4

    def _label(self, x, y, text, t, layer=0, dur=4.5, side=None, size=22, cands=()):
        """Callout label; tries a few anchors / sides / leader heights to avoid overlapping live labels."""
        T = self.T
        first = None
        for (cx, cy, ct) in [(x, y, t), *cands]:
            if not (BOX[1] + 110 < cy < PANEL_Y0 - 30):
                continue
            sides = [side] if side else ([1, -1] if cx < BOX[2] - 560 else [-1, 1])
            for sd in sides:
                for rise in (26, 64, -22, 102, -60):
                    bx = self._label_box(cx, cy, sd, rise, text, size)
                    if bx[0] < BOX[0] + 12 or bx[2] > BOX[2] - 12 or bx[1] < BOX[1] + 100 or bx[3] > PANEL_Y0 - 44:
                        continue
                    cand = dict(x=cx, y=cy, text=text, t=ct % T, layer=layer, dur=dur, side=sd, size=size, rise=rise,
                                box=bx)
                    if first is None:
                        first = cand
                    clash = False
                    for o in self._labels:
                        d = (cand["t"] - o["t"]) % T
                        if d >= o["dur"] and (T - d) >= dur:
                            continue
                        ob = o["box"]
                        if bx[0] < ob[2] and ob[0] < bx[2] and bx[1] < ob[3] and ob[1] < bx[3]:
                            clash = True
                            break
                    if not clash:
                        self._labels.append(cand)
                        return
        if first is not None and len(self._labels) % 3 == 0:
            self._labels.append(first)          # keep a few even if crowded

    def _flash(self, x, y, t, layer=1, rmax=70.0, dur=0.8):
        self._flashes.append((x, y, t % self.T, layer, rmax, dur))

    def _muon(self, t, stop=False):
        rng = self.rng
        charge = 1.0 if rng.random() < 0.56 else -1.0          # mu+ excess
        theta = float(np.clip(rng.normal(0, 0.38), -0.9, 0.9))   # zenith angle
        x0 = rng.uniform(BOX[0] + 120, BOX[2] - 120)
        ang = math.pi / 2 + theta                                # heading down the wall
        p = float(np.exp(rng.normal(1.2, 0.7)))                  # GeV/c
        sign = -charge * E_SIGN
        if not stop:
            pts = _spiral((x0, BOX[1] + 1), ang, R0=5200.0 * p, sign=sign, k=0.0, rng=rng, max_len=4000)
            tr = self._track(pts, t, layer=1, inten=1.05, tau=1.5, speed=7200.0, width=1.6, head_r=3.0,
                             drop_gap=8.0, drop_r=1.6, drop_i=1.0, drop_tau=3.6, flash=2.0)
            # delta rays knocked out along the way
            for _ in range(rng.integers(1, 4)):
                j = rng.integers(len(pts) // 5, max(len(pts) // 5 + 1, len(pts) - 2))
                sx, sy = pts[j]
                if sy > PANEL_Y0 - 30:
                    continue
                d = pts[min(j + 1, len(pts) - 1)] - pts[j]
                base = math.atan2(d[1], d[0])
                side = rng.choice([-1, 1])
                da = base + side * rng.uniform(0.9, 1.7)
                dp = _spiral((sx, sy), da, R0=rng.uniform(14, 55), sign=E_SIGN, k=rng.uniform(0.07, 0.16),
                             rng=rng, scatter=0.12)
                self._track(dp, t + tr["tv"][j], layer=0, inten=0.8, tau=0.9, speed=700.0, head_r=1.6,
                            drop_gap=6.0, drop_r=1.0, drop_i=0.8)
            mid = pts[len(pts) // 3]
            cands = [(pts[j][0] + 14, pts[j][1], t + tr["tv"][j]) for j in
                     (len(pts) * np.array([0.33, 0.5, 0.22, 0.62, 0.42, 0.72])).astype(int)]
            self._label(*cands[0][:2], f"MU{'+' if charge > 0 else '-'} {p:.4f} GEV/C", cands[0][2], layer=1,
                        cands=cands[1:])
            self._pulses.append((t, 1.0, "mu", p, float(mid[1])))
            return
        # stopping muon + decay
        x0 = rng.uniform(BOX[0] + 300, BOX[2] - 300)
        L_stop = rng.uniform(520, 900)
        pts = _stopping((x0, BOX[1] + 1), ang, R0=rng.uniform(2600, 5200), sign=sign, L_stop=L_stop, rng=rng,
                        scatter=0.25)
        s = _arclen(pts)
        dur = 1.05
        tv = dur * (1.0 - np.sqrt(np.clip(1.0 - s / s[-1], 0, 1)))
        tr = self._track(pts, t, layer=1, inten=1.1, tau=1.8, times=tv, width=1.9, head_r=3.2, drop_gap=7.0,
                         drop_r=1.7, drop_i=1.0, drop_tau=4.2, flash=2.0)
        ex, ey = pts[-1]
        t_dec = t + dur + 0.35
        self._flash(ex, ey, t + dur, layer=1, rmax=46, dur=0.6)
        self._flash(ex, ey, t_dec, layer=0, rmax=110, dur=0.9)
        self._label(ex + 16, ey - 8, f"MU{'+' if charge > 0 else '-'} STOP  2.197E-6 S", t + dur, layer=1, dur=5.5)
        # decay electron: same charge as the muon
        e_ang = rng.uniform(0, 2 * math.pi)
        e_pts = _spiral((ex, ey), e_ang, R0=rng.uniform(190, 420), sign=-charge * E_SIGN, k=rng.uniform(0.06, 0.1),
                        rng=rng, scatter=0.06)
        self._track(e_pts, t_dec, layer=0, inten=0.95, tau=1.1, speed=3400.0, width=1.3, head_r=2.6, drop_gap=8.0,
                    drop_r=1.3, drop_i=0.9, drop_tau=3.6)
        # the two neutrinos: invisible in a real chamber, hinted as faint dashes
        for _ in range(2):
            na = rng.uniform(0, 2 * math.pi)
            ln = _edge_hit(ex, ey, math.cos(na), math.sin(na))
            n = max(2, int(ln / 22))
            self._dashes.append((ex, ey, ex + math.cos(na) * ln, ey + math.sin(na) * ln, t_dec % self.T, n))
        self._label(ex + 16, ey + 30, f"DECAY MU->E NU NU  E {rng.uniform(12, 52):.1f} MEV", t_dec, layer=0, dur=5.0)
        self._pulses.append((t, 1.0, "mu", 0.11, float(ey)))
        self._pulses.append((t_dec, 0.8, "e", 0.035, float(ey)))

    def _pair(self, t):
        rng = self.rng
        vx = rng.uniform(BOX[0] + 350, BOX[2] - 350)
        vy = rng.uniform(BOX[1] + 180, PANEL_Y0 - 160)
        g_ang = rng.uniform(0.35, 2.8)                      # photon heading (mostly downward)
        back = _edge_hit(vx, vy, -math.cos(g_ang), -math.sin(g_ang))
        sx, sy = vx - math.cos(g_ang) * back, vy - math.sin(g_ang) * back
        self._dashes.append((sx, sy, vx, vy, t % self.T, max(2, int(back / 22))))
        t_v = t + 0.22
        E = float(np.exp(rng.normal(-2.6, 0.7)))              # GeV
        share = rng.uniform(0.25, 0.75)
        for q, frac in ((-1.0, share), (1.0, 1 - share)):
            R0 = 3400.0 * E * frac + 70.0
            pts = _spiral((vx, vy), g_ang + q * rng.uniform(0.02, 0.1), R0=R0, sign=-q * E_SIGN,
                          k=rng.uniform(0.07, 0.13), rng=rng, scatter=0.06)
            self._track(pts, t_v, layer=0, inten=0.95, tau=1.0, speed=2900.0, width=1.25, head_r=2.4)
        self._flash(vx, vy, t_v, layer=0, rmax=40, dur=0.5)
        self._label(vx + 14, vy - 12, f"GAMMA->E+E-  {E:.4f} GEV", t_v, layer=0)
        self._pulses.append((t_v, 0.65, "pair", E, float(vy)))

    def _curl(self, t, faint=False):
        rng = self.rng
        x = rng.uniform(BOX[0] + 60, BOX[2] - 60)
        y = rng.uniform(BOX[1] + 60, PANEL_Y0 - 30)
        pts = _spiral((x, y), rng.uniform(0, 2 * math.pi), R0=rng.uniform(16, 95), sign=E_SIGN,
                      k=rng.uniform(0.07, 0.17), rng=rng, scatter=0.15)
        self._track(pts, t, layer=0, inten=0.55 if faint else 0.8, tau=0.9, speed=820.0, head_r=1.6, drop_gap=6.5,
                    drop_r=1.0, drop_i=0.7 if faint else 0.85, drop_tau=2.6)
        if not faint and rng.random() < 0.3:
            self._label(x + 10, y - 10, f"{rng.uniform(0.0001, 0.009):.10f}", t, layer=0, dur=3.0, size=20)
        self._pulses.append((t, 0.18, "e", 0.004, float(y)))

    def _star(self, t):
        rng = self.rng
        vx = rng.uniform(BOX[0] + 450, BOX[2] - 450)
        vy = rng.uniform(BOX[1] + 250, PANEL_Y0 - 220)
        ia = rng.uniform(1.2, 1.95)
        back = _edge_hit(vx, vy, -math.cos(ia), -math.sin(ia))
        inc = _spiral((vx - math.cos(ia) * back, vy - math.sin(ia) * back), ia, R0=18000, sign=1.0, k=0.0, rng=rng,
                      max_len=back)
        tr = self._track(inc, t, layer=0, inten=1.0, tau=1.3, speed=6500.0, width=1.5, head_r=2.8)
        t_v = t + float(tr["tv"][-1])
        n = int(rng.integers(4, 8))
        E = rng.uniform(0.8, 3.2)
        for j in range(n):
            a = rng.uniform(0, 2 * math.pi)
            heavy = rng.random() < 0.35
            if heavy:     # slow proton: short, thick, ranging out
                pts = _stopping((vx, vy), a, R0=rng.uniform(900, 4000), sign=rng.choice([-1, 1]),
                                L_stop=rng.uniform(70, 240), rng=rng, R_end=60)
                self._track(pts, t_v, layer=0, inten=1.1, tau=1.6, speed=1400.0, width=2.6, head_r=2.8,
                            drop_gap=5.0, drop_r=1.7, drop_i=1.0, drop_tau=3.8)
            else:
                pts = _spiral((vx, vy), a, R0=rng.uniform(500, 9000), sign=rng.choice([-1, 1]),
                              k=rng.uniform(0.0, 0.6), rng=rng, max_len=rng.uniform(160, 900), scatter=0.3)
                self._track(pts, t_v, layer=0, inten=0.9, tau=1.2, speed=3300.0, width=1.3, head_r=2.4)
                if rng.random() < 0.4 and len(pts) > 3:   # kink: in-flight decay
                    kx, ky = pts[-1]
                    d = pts[-1] - pts[-2]
                    ka = math.atan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.4, 1.2)
                    kp = _spiral((kx, ky), ka, R0=rng.uniform(300, 1600), sign=E_SIGN, k=rng.uniform(0.05, 0.3),
                                 rng=rng, max_len=rng.uniform(150, 600), scatter=0.4)
                    t_k = t_v + _arclen(pts)[-1] / 3300.0
                    self._track(kp, t_k, layer=0, inten=0.8, tau=1.0, speed=3000.0, width=1.1, head_r=2.0)
        self._flash(vx, vy, t_v, layer=0, rmax=90, dur=0.8)
        self._label(vx + 16, vy - 14, f"STAR N={n}  {E:.2f} GEV", t_v, layer=0, dur=5.0)
        self._pulses.append((t_v, 0.9, "star", E, float(vy)))

    def _alpha(self, t):
        rng = self.rng
        x = rng.uniform(BOX[0] + 80, BOX[2] - 80)
        y = rng.uniform(BOX[1] + 80, PANEL_Y0 - 60)
        a = rng.uniform(0, 2 * math.pi)
        L = rng.uniform(60, 170)
        pts = _spiral((x, y), a, R0=40000, sign=1.0, k=0.0, rng=rng, max_len=L)
        self._track(pts, t, layer=0, inten=0.95, tau=1.4, speed=900.0, width=4.2, head_r=3.0, drop_gap=3.2,
                    drop_r=1.9, drop_i=0.95, drop_tau=3.2)
        if rng.random() < 0.5:
            self._label(x + 12, y - 10, f"ALPHA {rng.uniform(4.2, 5.8):.2f} MEV", t, layer=0, dur=3.2, size=20)
        self._pulses.append((t, 1.2, "alpha", 0.005, float(y)))

    def _build(self):
        rng = self.rng
        self._dashes = []
        T = self.T
        n_beats = int(round(T / BEAT))
        bars = n_beats // 4
        decay_bars = set(rng.choice(np.arange(1, bars), size=max(1, bars // 4), replace=False).tolist())
        for b in range(bars):
            self._muon(b * 4 * BEAT, stop=b in decay_bars)
        for b in rng.choice(np.arange(n_beats), size=n_beats // 6, replace=False):
            if b % 4:
                self._muon(b * BEAT + rng.choice([0.0, 0.5]) * BEAT)
        for b in rng.choice(np.arange(n_beats), size=n_beats // 4, replace=False):
            self._pair((b + 0.5) * BEAT)
        for b in rng.choice(np.arange(n_beats), size=max(2, n_beats // 20), replace=False):
            self._star(b * BEAT + 0.25 * BEAT)
        for _ in range(int(n_beats * 1.1)):
            self._curl(rng.integers(0, n_beats * 4) * BEAT / 4, faint=rng.random() < 0.5)
        for _ in range(n_beats // 5):
            self._alpha(rng.integers(0, n_beats * 2) * BEAT / 2)

    # ------------------------------------------------------------------ pack
    def _pack(self):
        rng = self.rng
        tr = self._tracks
        n = len(tr)
        self.t0 = np.array([d["t0"] for d in tr], np.float32)
        self.layer = np.array([1 if d["layer"] else 0 for d in tr], np.int8)
        self.inten = np.array([d["inten"] for d in tr], np.float32)
        self.tau = np.array([d["tau"] for d in tr], np.float32)
        self.width = np.array([d["width"] for d in tr], np.float32)
        self.head_r = np.array([d["head_r"] for d in tr], np.float32)
        self.flashk = np.array([d["flash"] for d in tr], np.float32)
        self.dur = np.array([d["tv"][-1] for d in tr], np.float32)
        self.life = self.dur + np.array([d["drop_tau"] for d in tr], np.float32) * 3.2 + 1.0

        P, TV, TID, SA = [], [], [], []
        off = 0
        for i, d in enumerate(tr):
            m = len(d["p"])
            P.append(d["p"])
            TV.append(d["tv"])
            TID.append(np.full(m, i, np.int32))
            SA.append(np.arange(off, off + m - 1, dtype=np.int64))
            off += m
        self.P = np.concatenate(P).astype(np.float32)
        self.TV = np.concatenate(TV).astype(np.float32)
        self.TID = np.concatenate(TID)
        self.SA = np.concatenate(SA)

        # condensation droplets along each track
        DP, DT, DI, DU, DD, DR, DA = [], [], [], [], [], [], []
        for i, d in enumerate(tr):
            s = d["s"]
            if s[-1] < 2:
                continue
            gap = d["drop_gap"]
            cnt = max(1, int(s[-1] / gap))
            sd = np.sort(rng.uniform(0, s[-1], cnt)).astype(np.float32)
            sd = 0.5 * sd + 0.5 * (np.arange(cnt) + 0.5) * (s[-1] / cnt)     # semi-regular spacing
            px = np.interp(sd, s, d["p"][:, 0])
            py = np.interp(sd, s, d["p"][:, 1])
            DP.append(np.stack([px, py], 1) + rng.normal(0, 0.55, (cnt, 2)))
            DT.append(np.interp(sd, s, d["tv"]))
            DI.append(np.full(cnt, i, np.int32))
            DU.append(rng.uniform(0.35, 1.5, cnt))
            ang = rng.uniform(0, 2 * np.pi, cnt)
            DD.append(np.stack([np.cos(ang), np.sin(ang)], 1) * rng.uniform(0.6, 2.4, (cnt, 1)))
            DR.append(d["drop_r"] * rng.uniform(0.7, 1.3, cnt))
            DA.append(np.full(cnt, d["drop_i"], np.float32) * rng.uniform(0.6, 1.2, cnt))
        self.DP = np.concatenate(DP).astype(np.float32)
        self.DT = np.concatenate(DT).astype(np.float32)
        self.DI = np.concatenate(DI)
        self.DU = np.concatenate(DU).astype(np.float32)
        self.DD = np.concatenate(DD).astype(np.float32)
        self.DR = np.concatenate(DR).astype(np.float32)
        self.DA = np.concatenate(DA).astype(np.float32)
        self.drop_tau = np.array([d["drop_tau"] for d in tr], np.float32)

        self.pulses = np.array([(p[0] % self.T, p[1], p[3], p[4]) for p in self._pulses], np.float32)
        self.pulse_kind = [p[2] for p in self._pulses]

    def _build_static(self):
        rng = np.random.default_rng(5)
        gx = np.arange(FRAME_BOX[0] + 72, FRAME_BOX[2] - 20, 115.0)
        gy = np.arange(FRAME_BOX[1] + 62, FRAME_BOX[3] - 20, 115.0)
        X, Y = np.meshgrid(gx, gy)
        self.cross_xy = np.stack([X.ravel(), Y.ravel()], 1).astype(np.float32)
        # background mist: slow drifting specks
        n = 520
        self.mist = rng.uniform([BOX[0], BOX[1]], [BOX[2], PANEL_Y0], (n, 2)).astype(np.float32)
        self.mist_amp = rng.uniform(6, 30, (n, 2)).astype(np.float32)
        self.mist_ph = rng.uniform(0, 2 * np.pi, (n, 2)).astype(np.float32)
        self.mist_k = rng.integers(1, 3, (n, 2)).astype(np.float32)
        self.mist_i = rng.uniform(0.05, 0.3, n).astype(np.float32)
        self.mist_tw = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        self.mist_tk = rng.integers(1, 6, n).astype(np.float32)
        # edge ruler ticks
        self.tick_y = np.arange(FRAME_BOX[1] + 14, FRAME_BOX[3] - 10, 13.0, dtype=np.float32)

    # ------------------------------------------------------------------ render
    def _age(self, t, t0):
        return (t - t0) % self.T

    def render(self, t, W=3000, H=1688):
        f = Frame(W, H)
        T = self.T
        g = self._age(t, self.t0)                       # age of every track
        alive = g < self.life
        self._draw_mist(f, t)
        self._draw_crosses(f, t)
        self._draw_dashes(f, t)
        self._draw_tracks(f, g, alive)
        self._draw_drops(f, g, alive)
        self._draw_flashes(f, t)
        self._draw_labels(f, t)
        self._draw_hud(f, t)
        return f.finish(bloom_weights=(0.55, 0.5, 0.42, 0.36, 0.32, 0.3, 0.26, 0.22), bloom_gain=0.85)

    def _draw_mist(self, f, t):
        ph = 2 * np.pi * t / self.T
        p = self.mist + self.mist_amp * np.sin(self.mist_k * ph + self.mist_ph)
        tw = 0.55 + 0.45 * np.sin(self.mist_tk * ph + self.mist_tw)
        f.dots("w", p[:, 0], p[:, 1], 1.0, self.mist_i * tw)

    def _event_glow(self, t, x, y, sigma=170.0):
        """How much recent events light up points (x, y) - used by the '+' grid."""
        out = np.zeros(len(x), np.float32)
        for fx, fy, ft, _, _, dur in self._flashes:
            a = (t - ft) % self.T
            if a < 2.0:
                amp = math.exp(-a / 0.45)
                out += amp * np.exp(-((x - fx) ** 2 + (y - fy) ** 2) / (2 * sigma ** 2))
        return out

    def _draw_crosses(self, f, t):
        c = self.cross_xy
        glow = self._event_glow(t, c[:, 0], c[:, 1])
        f.crosses("w", c[:, 0], c[:, 1], 16, 0.16 + 0.55 * glow)

    def _draw_dashes(self, f, t):
        for x0, y0, x1, y1, t0, n in self._dashes:
            a = (t - t0) % self.T
            if a > 2.4:
                continue
            reveal = min(1.0, a / 0.2)
            inten = 0.28 * math.exp(-a / 0.9)
            k = np.arange(n)
            u0 = k / n
            u1 = (k + 0.5) / n
            keep = u0 <= reveal
            u1 = np.minimum(u1, reveal)
            u0, u1 = u0[keep], u1[keep]
            if not len(u0):
                continue
            f.segments("w", x0 + (x1 - x0) * u0, y0 + (y1 - y0) * u0, x0 + (x1 - x0) * u1, y0 + (y1 - y0) * u1, inten)

    def _intensity(self, a, tid):
        return self.inten[tid] * (np.exp(-a / self.tau[tid]) + self.flashk[tid] * np.exp(-a / 0.07))

    def _draw_tracks(self, f, g, alive):
        sa = self.SA
        tid = self.TID[sa]
        G = g[tid]
        ta = self.TV[sa]
        tb = self.TV[sa + 1]
        vis = alive[tid] & (G >= ta)
        sa, tid, G, ta, tb = sa[vis], tid[vis], G[vis], ta[vis], tb[vis]
        pa = self.P[sa]
        pb = self.P[sa + 1].copy()
        part = G < tb
        frac = np.clip((G[part] - ta[part]) / np.maximum(tb[part] - ta[part], 1e-6), 0, 1)
        pb[part] = pa[part] + (pb[part] - pa[part]) * frac[:, None]
        aa = G - ta
        ab = np.where(part, 0.0, G - tb)
        ia = self._intensity(aa, tid)
        ib = self._intensity(ab, tid)
        lay = self.layer[tid]
        for L, name in ((0, "w"), (1, "r")):
            m = lay == L
            if m.any():
                f.segments(name, pa[m, 0], pa[m, 1], pb[m, 0], pb[m, 1], ia[m], ib[m], width=self.width[tid[m]])
        # moving heads
        self._heads = (pb[part], self.layer[tid[part]])
        if part.any():
            ht = tid[part]
            hp = pb[part]
            for L, name in ((0, "w"), (1, "r")):
                m = self.layer[ht] == L
                if m.any():
                    f.dots(name, hp[m, 0], hp[m, 1], self.head_r[ht[m]], 1.5)
                    if L == 1:
                        f.dots("w", hp[m, 0], hp[m, 1], self.head_r[ht[m]] * 0.45, 0.9)

    def _draw_drops(self, f, g, alive):
        tid = self.DI
        G = g[tid]
        a = G - self.DT
        vis = alive[tid] & (a > 0.04)
        if not vis.any():
            return
        a = a[vis]
        tid = tid[vis]
        u = self.DU[vis]
        tau = self.drop_tau[tid] * u
        inten = self.DA[vis] * smoothstep(0.05, 0.55, a) * np.exp(-np.maximum(a - 0.55, 0) / tau)
        # drops also wink out one by one near the end of their life
        inten *= 1.0 - smoothstep(tau * 1.6, tau * 2.4, a)
        pos = self.DP[vis] + self.DD[vis] * np.sqrt(a)[:, None] * 1.6
        r = self.DR[vis] * (0.75 + 0.35 * smoothstep(0.0, 2.5, a))
        lay = self.layer[tid]
        for L, name in ((0, "w"), (1, "r")):
            m = (lay == L) & (inten > 0.01)
            if m.any():
                f.dots(name, pos[m, 0], pos[m, 1], r[m], inten[m])

    def _draw_flashes(self, f, t):
        for x, y, t0, layer, rmax, dur in self._flashes:
            a = (t - t0) % self.T
            if a > dur:
                continue
            u = a / dur
            r = rmax * (1 - (1 - u) ** 3)
            name = "r" if layer else "w"
            f.rings(name, [x], [y], [max(r, 1.0)], 0.9 * (1 - u) ** 1.5, width=1.4)
            f.dots(name, [x], [y], 3.5 * (1 - u) + 1.5, 1.6 * (1 - u))

    def _draw_labels(self, f, t):
        for lb in self._labels:
            a = (t - lb["t"]) % self.T
            if a > lb["dur"]:
                continue
            alpha = smoothstep(0.0, 0.1, a) * (1.0 - smoothstep(lb["dur"] - 1.2, lb["dur"], a))
            name = "r" if lb["layer"] else "w"
            x, y, side = lb["x"], lb["y"], lb["side"]
            # callout: short diagonal + horizontal leader
            ex, ey = x + side * 26, y - lb["rise"]
            hx = ex + side * 22
            grow = smoothstep(0.0, 0.12, a)
            f.segments("w", [x, ex], [y, ey], [x + (ex - x) * grow, ex + (hx - ex) * grow],
                       [y + (ey - y) * grow, ey], 0.55 * alpha)
            f.dots("w", [x], [y], 2.0, 0.9 * alpha)
            n = int(min(len(lb["text"]), max(0.0, (a - 0.08) * 70)))
            if n:
                txt = lb["text"][:n]
                if side > 0:
                    f.text(name, hx + 8, ey + 7, txt, size=lb["size"], alpha=alpha, anchor="ls")
                else:
                    f.text(name, hx - 8, ey + 7, lb["text"][-n:], size=lb["size"], alpha=alpha, anchor="rs")

    # ------------------------------------------------------------------ HUD
    def _signal(self, tt, kinds=None):
        """dE/dx-like response: sum of fast-attack pulses at event times (loop-safe)."""
        out = np.zeros_like(tt, dtype=np.float32)
        for (pt, amp, _, _), kind in zip(self.pulses, self.pulse_kind):
            if kinds and kind not in kinds:
                continue
            a = (tt - pt) % self.T
            m = a < 2.5
            out[m] += amp * np.exp(-a[m] / 0.22) * (1 - np.exp(-a[m] / 0.015))
        return out

    def _rate(self, t, window=8.0, kinds=None):
        n = 0
        for (pt, _, _, _), kind in zip(self.pulses, self.pulse_kind):
            if kinds and kind not in kinds:
                continue
            if (t - pt) % self.T < window:
                n += 1
        return n / window

    def _draw_hud(self, f, t):
        T = self.T
        x0, y0, x1, y1 = FRAME_BOX
        # occlude tracks under the panels
        for px0, px1 in PANELS:
            for name in ("w", "r"):
                f.scale_rect(name, px0 - 6, PANEL_Y0 - 36, px1 + 6, PANEL_Y1 + 6, 0.0)
        f.rect("w", x0, y0, x1, y1, 0.95, width=3.0)

        # rulers on the inner left / right edges: static ticks + live markers following the heads
        ty = self.tick_y
        k = np.arange(len(ty))
        base = np.where(k % 5 == 0, 16.0, 7.0)
        heads, hl = getattr(self, "_heads", (np.zeros((0, 2), np.float32), np.zeros(0, np.int8)))
        hist_w = np.zeros_like(ty)
        hist_r = np.zeros_like(ty)
        for (pt, amp, _, py), kind in zip(self.pulses, self.pulse_kind):
            a = (t - pt) % T
            if a < 4.0 and kind != "mu":
                hist_w += 0.5 * amp * math.exp(-a / 1.2) * np.exp(-((ty - py) / 30.0) ** 2)
        if len(heads):
            dy = ty[None, :] - heads[:, 1:2]
            bump = np.exp(-(dy / 9.0) ** 2)
            hist_w += bump[hl == 0].sum(0)
            hist_r += bump[hl == 1].sum(0)
        len_l = base + 64 * np.tanh(hist_w)
        len_r = base + 90 * np.tanh(1.5 * hist_r)
        f.segments("w", np.full_like(ty, x0), ty, x0 + len_l, ty, 0.42 + 0.5 * np.tanh(hist_w))
        f.segments("w", np.full_like(ty, x1), ty, x1 - base, ty, 0.42)
        live = hist_r > 0.02
        if live.any():
            f.segments("r", np.full(live.sum(), x1), ty[live], x1 - len_r[live], ty[live], 1.1 * np.tanh(2 * hist_r[live]))

        # header
        tr_rate = self._rate(t)
        mu_rate = self._rate(t, kinds=("mu",))
        f.text("w", x0 + 28, y0 + 44, "CLOUD_CHAMBER_01", size=22, alpha=0.95)
        f.text("w", x0 + 28, y0 + 74, "B_FIELD 1.50 T  //  SENSITIVE 30 MS  //  128 BPM", size=18, alpha=0.6)
        f.text("w", x1 - 28, y0 + 44, f"TRACKS/S {tr_rate:05.2f}", size=22, alpha=0.95, anchor="rs")
        f.text("r", x1 - 28, y0 + 74, f"MU/S {mu_rate:05.2f}", size=18, alpha=0.95, anchor="rs")

        # scopes
        self._scope_dedx(f, t, *PANELS[0])
        self._scope_trigger(f, t, *PANELS[1])
        self._scope_spectrum(f, t, *PANELS[2])

    def _panel_frame(self, f, px0, px1, title, vlabel):
        py0, py1 = PANEL_Y0, PANEL_Y1
        gx = np.linspace(px0, px1, 11)
        gy = np.linspace(py0, py1, 6)
        f.segments("r", gx, np.full_like(gx, py0), gx, np.full_like(gx, py1), 0.32)
        f.segments("r", np.full_like(gy, px0), gy, np.full_like(gy, px1), gy, 0.32)
        f.rect("w", px0, py0, px1, py1, 0.75, width=1.2)
        f.text("w", px1, py0 - 10, title, size=19, alpha=0.95, anchor="rs")
        f.text_vertical("w", px0 - 28, py1, vlabel, size=17, alpha=0.9)

    def _scope_dedx(self, f, t, px0, px1):
        n = 240
        tt = t - 5.0 + np.linspace(0, 5.0, n)
        sig = self._signal(tt) + 0.06 * (1 + periodic_noise(tt * 3.0, self.T * 3.0, 31, n=8, base=20))
        cur = float(sig[-1])
        self._panel_frame(f, px0, px1, f"dE/dx :{cur:.2f}", "Chamber 1")
        xs = np.linspace(px0 + 2, px1 - 2, n)
        ys = PANEL_Y1 - 10 - (PANEL_Y1 - PANEL_Y0 - 22) * np.tanh(0.95 * sig)
        f.polyline("w", xs, ys, 0.85, width=1.3)
        f.dots("w", [xs[-1]], [ys[-1]], 2.6, 1.3)

    def _scope_trigger(self, f, t, px0, px1):
        n = 360
        tt = t - 7.5 + np.linspace(0, 7.5, n)
        hi = np.zeros(n, bool)
        for (pt, _, _, _), kind in zip(self.pulses, self.pulse_kind):
            if kind == "mu":
                hi |= ((tt - pt) % self.T) < 0.16
        cur = int(hi[-1])
        self._panel_frame(f, px0, px1, f"Trigger :{cur}", "Trigger")
        xs = np.linspace(px0 + 2, px1 - 2, n)
        lo_y, hi_y = PANEL_Y1 - 16, PANEL_Y0 + 22
        ys = np.where(hi, hi_y, lo_y)
        # square wave: duplicate points at the transitions
        X = np.repeat(xs, 2)[1:]
        Y = np.repeat(ys, 2)[:-1]
        f.polyline("w", X, Y, 0.85, width=1.3)
        f.polyline("r", xs, np.full(n, lo_y + 6), 0.0)
        if cur:
            f.dots("r", [xs[-1]], [hi_y], 3.2, 1.6)

    def _scope_spectrum(self, f, t, px0, px1):
        nb = 16
        edges = np.linspace(np.log10(0.002), np.log10(40.0), nb + 1)
        h = np.zeros(nb, np.float32)
        last_p, last_age = 0.0, 1e9
        for (pt, amp, p, _), kind in zip(self.pulses, self.pulse_kind):
            a = (t - pt) % self.T
            if a < 12.0:
                b = int(np.clip(np.searchsorted(edges, math.log10(max(p, 1e-4))) - 1, 0, nb - 1))
                h[b] += math.exp(-a / 4.0) * (1.0 - math.exp(-a / 0.08))
                if kind == "mu" and a < last_age:
                    last_p, last_age = p, a
        self._panel_frame(f, px0, px1, f"p :{last_p:.2f} GeV", "Spectrum")
        bw = (px1 - px0) / nb
        cols_x, cols_y, cols_i = [], [], []
        for b in range(nb):
            height = (PANEL_Y1 - PANEL_Y0 - 16) * math.tanh(0.8 * h[b])
            k = int(height / 7.0)
            if k <= 0:
                continue
            cx = px0 + (b + 0.5) * bw
            ys = PANEL_Y1 - 6 - np.arange(k) * 7.0
            for dx in (-5.0, 0.0, 5.0):
                cols_x.append(np.full(k, cx + dx))
                cols_y.append(ys)
                cols_i.append(np.full(k, 0.95))
        if cols_x:
            f.dots("w", np.concatenate(cols_x), np.concatenate(cols_y), 1.7, np.concatenate(cols_i))
