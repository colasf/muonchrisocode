"""AIR SHOWER - Ryoji Ikeda edition.

A cosmic ray hits the upper atmosphere; a cascade of particles rains onto three
detectors. One shower per 4-bar phrase (128 BPM), hard cuts on every bar:

  bar 1  PERSPECTIVE   first interaction on the downbeat (2-frame inversion)
  bar 2  ORTHO_SIDE    scientific chart tracking the front: altitude rules, red front line
  bar 3  ORTHO_TOP     the front lands on the downbeat: the footprint blooms over a dot lattice
  bar 4  GROUND        low view among the detectors, the aftermath

The cascade is physics-flavoured: primary (red) -> hadronic interactions ->
pi0 -> photons -> e+e- / bremsstrahlung sub-showers (white, feathery, denser as
the air thickens) and pi+- -> muons (red, long straight tracks to the ground).
HUD: a supersymmetry-like longitudinal-profile strip that builds live, a
scrolling particle data column, detector slab readouts, a birth-rate barcode.
Loops seamlessly every 16 bars (30 s); the last beat is a barcode burst.
"""
from __future__ import annotations

import math

import numpy as np

from .engine import Camera, Frame, OrthoCamera, hash01, smoothstep

BPM = 128.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
PHRASE = 4 * BAR

STRIP = (40.0, 338.0, 2960.0, 470.0)
MAIN = (40.0, 490.0, 2960.0, 1462.0)
BOT = (40.0, 1482.0, 2960.0, 1612.0)
MAIN_C = ((MAIN[0] + MAIN[2]) / 2, (MAIN[1] + MAIN[3]) / 2)
COL = (40.0, 505.0, 372.0, 1447.0)             # particle data column

V_PRIMARY = 12.0
TOP = 27.0
DETECTORS = [(-9.0, 0.0), (0.0, 0.0), (9.0, 0.0)]
DET_NAMES = ["DET_L", "DET_C", "DET_R"]
DET_HALF = 0.55
DET_H = 2.7
SLABS = np.linspace(0.25, DET_H - 0.2, 9)

K_E, K_G, K_H, K_MU, K_P, K_RAIN = 0, 1, 2, 3, 4, 5
K_NAME = ["E-", "GAMMA", "PI+", "MU-", "P", "MU+"]
#                 e      gamma  hadron  muon   primary rain
K_INT = np.array([0.34, 0.13, 0.95, 0.72, 1.5, 0.85], np.float32)
K_TAU = np.array([1.35, 0.8, 2.0, 1.6, 3.0, 0.28], np.float32)
K_W = np.array([1.0, 1.0, 1.3, 1.0, 2.2, 1.1], np.float32)
K_RED = np.array([0, 0, 0, 1, 1, 1], np.int8)
K_HEAD = np.array([1.1, 0.0, 2.2, 2.4, 4.0, 2.2], np.float32)
K_HEAD_I = np.array([0.5, 0.0, 1.3, 1.4, 1.6, 1.3], np.float32)

# (ground x, z, E0, zenith, azimuth, h1, aimed detector)
PLAN = [
    (-8.6, 0.4, 0.75, 9.0, 40.0, 14.2, 0),
    (0.3, 0.2, 1.0, 5.0, 130.0, 14.8, 1),
    (4.8, -4.6, 0.6, 17.0, 320.0, 13.6, None),
    (8.8, 0.6, 0.85, 11.0, 250.0, 14.0, 2),
]


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
            L = _lam_h(p[1]) * rng.exponential(1.0) * (0.4 if gen == 0 else 1.0)
            end, tang, tend, hit = c.add(p, d, L, t, K_H, e)
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
                L = min(lam * rng.exponential(0.55), 1.3)
                c.add(p, d, L, t, kind, e, curv=0.9 * rng.exponential() if kind == K_E else 0.0)
                continue
            L = min(lam * rng.exponential(0.75), 2.2)
            curv = min(1.2, 0.35 * (Ec / e) ** 0.6) if kind == K_E else 0.0
            end, tang, tend, hit = c.add(p, d, L, t, kind, e, curv=curv)
            if hit:
                continue
            u = rng.uniform(0.12, 0.88)
            kids = (K_E, K_E) if kind == K_G else (K_E, K_G)
            for kk, ef in zip(kids, (e * u, e * (1 - u))):
                th = min(1.2, 0.42 * math.sqrt(Ec / max(ef, 1e-9)) * (0.25 + rng.exponential()))
                stack.append((end, _deflect(tang, th, rng), ef, kk, tend, gen + 1))
    return c.branches, info


# ----------------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------------

class Shower:
    name = "shower"

    def __init__(self, T=30.0, seed=4):
        self.T = T
        rng = np.random.default_rng(seed)
        self.events = []
        B = []
        n_phr = int(round(T / PHRASE))
        for k in range(n_phr):
            gx, gz, E0, zen, az, h1, aim = PLAN[k % len(PLAN)]
            path = h1 / math.cos(math.radians(zen))
            speed = path / (2 * BAR)                  # the front lands on the downbeat of bar 3
            br, info = build_shower(rng, (gx, gz), h1, zen, az, E0, speed,
                                    max_branches=int(9000 + 14000 * E0))
            if aim is not None:
                br = self._aim_muons(br, aim, rng, speed, count=2)
            t_int = k * PHRASE                        # first interaction on the phrase downbeat
            info.update(t0=(t_int - info["t1"]) % T, kind="shower", aim=aim, phrase=k, speed=speed)
            self.events.append(info)
            B.append(br)
        # background muon rain, on the beat
        n_beats = int(round(T / BEAT))
        beats = rng.choice(np.arange(n_beats), size=int(n_beats * 0.4), replace=False)
        for j, b in enumerate(beats):
            if j % 3 == 0:
                dx, dz = DETECTORS[rng.integers(0, 3)]
                gx, gz = dx + rng.uniform(-0.3, 0.3), dz + rng.uniform(-0.3, 0.3)
            else:
                gx, gz = rng.uniform(-20, 20), rng.uniform(-16, 12)
            zen = math.radians(abs(rng.normal(0, 14)))
            az = rng.uniform(0, 2 * math.pi)
            d = np.array([math.sin(zen) * math.cos(az), -math.cos(zen), math.sin(zen) * math.sin(az)])
            G = np.array([gx, 0.0, gz])
            start = G - d * (TOP / math.cos(zen))
            t_travel = (TOP / math.cos(zen)) / 7.5
            pts = np.stack([start, G]).astype(np.float32)
            self.events.append(dict(t0=(b * BEAT - t_travel) % T, kind="rain"))
            B.append([(pts, np.array([0.0, t_travel], np.float32), K_RAIN, 0.3)])
        self._pack(B)
        self._build_static()

    # ---------------------------------------------------------------- build
    @staticmethod
    def _aim_muons(branches, det, rng, speed, count=2):
        dx, dz = DETECTORS[det]
        mu = [i for i, b in enumerate(branches) if b[2] == K_MU and b[0][-1, 1] <= 1e-3]
        if not mu:
            return branches
        dist = [math.hypot(branches[i][0][-1, 0] - dx, branches[i][0][-1, 2] - dz) for i in mu]
        out = list(branches)
        for i in [mu[i] for i in np.argsort(dist)[:count]]:
            pts, times, kind, e = branches[i]
            start = pts[0].astype(np.float64)
            G = np.array([dx + rng.uniform(-0.35, 0.35), 0.0, dz + rng.uniform(-0.35, 0.35)])
            L = float(np.linalg.norm(G - start))
            out[i] = (np.stack([start, G]).astype(np.float32), np.array([times[0], times[0] + L / speed], np.float32),
                      kind, e)
        return out

    def _pack(self, B):
        V_, VT, SE, SK, SA, EN = [], [], [], [], [], []
        births = []
        off = 0
        for ev, brs in enumerate(B):
            bt, bk, be, bp = [], [], [], []
            for pts, times, kind, e in brs:
                m = len(pts)
                V_.append(pts)
                VT.append(times)
                SA.append(np.arange(off, off + m - 1, dtype=np.int64))
                SE.append(np.full(m - 1, ev, np.int32))
                SK.append(np.full(m - 1, kind, np.int8))
                EN.append(np.full(m - 1, e, np.float32))
                off += m
                bt.append(times[0]); bk.append(kind); be.append(e); bp.append(pts[0])
            o = np.argsort(bt)
            births.append(dict(t=np.asarray(bt, np.float32)[o], k=np.asarray(bk, np.int8)[o],
                               e=np.asarray(be, np.float32)[o], p=np.asarray(bp, np.float32)[o]))
        self.births = births
        self.V = np.concatenate(V_).astype(np.float32)
        self.VT = np.concatenate(VT).astype(np.float32)
        self.SA = np.concatenate(SA)
        self.SE = np.concatenate(SE)
        self.SK = np.concatenate(SK)
        self.SEn = np.concatenate(EN)
        n_ev = len(B)
        self.ev_t0 = np.array([e["t0"] for e in self.events], np.float32)
        last = np.zeros(n_ev, np.float32)
        np.maximum.at(last, self.SE, self.VT[self.SA + 1])
        self.ev_life = last + 4.5
        rng = np.random.default_rng(99)
        self.SVar = rng.uniform(0.6, 1.35, len(self.SA)).astype(np.float32)
        self.SGain = np.where(self.SK <= K_G,
                              np.clip(0.75 + 0.12 * np.log10(np.maximum(self.SEn, 1e-6) / 1.4e-4), 0.7, 1.5),
                              1.0).astype(np.float32)
        # ground hits + detector crossings
        hits, dets = [], []
        for ev, brs in enumerate(B):
            for pts, times, kind, e in brs:
                if kind not in (K_MU, K_H, K_RAIN) or pts[-1, 1] > 1e-3:
                    continue
                kk = K_MU if kind == K_RAIN else kind
                hits.append((ev, times[-1], pts[-1, 0], pts[-1, 2], kk))
                if kk != K_MU:
                    continue
                a, b = pts[-2].astype(np.float64), pts[-1].astype(np.float64)
                ta, tb = float(times[-2]), float(times[-1])
                for di, (dx, dz) in enumerate(DETECTORS):
                    ys = np.append(SLABS, DET_H)
                    f = np.clip((a[1] - ys) / max(a[1] - b[1], 1e-6), 0, 1)
                    xs = a[0] + (b[0] - a[0]) * f
                    zs = a[2] + (b[2] - a[2]) * f
                    inside = (np.abs(xs - dx) < DET_HALF) & (np.abs(zs - dz) < DET_HALF)
                    if inside.sum() >= 3:
                        tp = ta + (tb - ta) * f
                        dets.append(dict(ev=ev, det=di, slab_t=tp[:-1].astype(np.float32), slab_on=inside[:-1],
                                         t_hit=float(tp[inside].min())))
        self.hits = np.array(hits, np.float32)
        self.det_hits = dets
        rng = np.random.default_rng(7)
        for d in dets:
            d["label"] = f"{DET_NAMES[d['det']]} HIT {rng.uniform(0.001, 0.009):.10f}"
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
        g = np.arange(-24.0, 24.01, 0.5, dtype=np.float32)
        X, Z = np.meshgrid(g, g)
        self.lattice = np.stack([X.ravel(), np.zeros(X.size, np.float32), Z.ravel()], 1)
        self.lat_major = ((np.abs(X) % 4 < 1e-3) & (np.abs(Z) % 4 < 1e-3)).ravel()
        lines, owner, slab = [], [], []
        for di, (dx, dz) in enumerate(DETECTORS):
            c = [(dx - DET_HALF, dz - DET_HALF), (dx + DET_HALF, dz - DET_HALF), (dx + DET_HALF, dz + DET_HALF),
                 (dx - DET_HALF, dz + DET_HALF)]
            for (x, z) in c:
                lines.append(((x, 0.0, z), (x, DET_H, z)))
                owner.append(di)
                slab.append(-1)
            for si, y in enumerate([*SLABS, DET_H, 0.0]):
                for j in range(4):
                    (xa, za), (xb, zb) = c[j], c[(j + 1) % 4]
                    lines.append(((xa, y, za), (xb, y, zb)))
                    owner.append(di)
                    slab.append(si if si < len(SLABS) else -1)
        self.det_lines = np.array(lines, np.float32)
        self.det_owner = np.array(owner, np.int32)
        self.det_slab = np.array(slab, np.int32)

    # ---------------------------------------------------------------- views
    def view(self, t):
        T = self.T
        bar = int(t // BAR) % int(round(T / BAR))
        phrase = bar // 4
        kind = ("persp", "side", "top", "ground")[bar % 4]
        e = self.events[phrase]
        u = (t - bar * BAR) / BAR                     # progress inside the bar
        gx, gz = float(e["G"][0]), float(e["G"][2])
        sc = MAIN_C
        if kind == "persp":
            yaw = math.radians(-24.0 + 47.0 * phrase + 9.0 * u)
            D = 26.0 - 1.5 * u
            pos = (gx + D * math.sin(yaw), 8.0, gz + D * math.cos(yaw))
            cam = Camera(pos, (gx, 8.4, gz), fov_deg=44.0, screen_center=sc)
            cam.fov = 44.0
        elif kind == "side":
            yaw = math.radians(8.0 * phrase)
            a = (t - e["t0"]) % T
            front = e["h1"] - max(0.0, a - e["t1"]) * e["speed"] * math.cos(math.radians(e["zen"]))
            cy = float(np.clip(front + 1.2, 4.6, 11.0))
            pos = (gx + 60 * math.sin(yaw), cy, gz + 60 * math.cos(yaw))
            cam = OrthoCamera(pos, (gx, cy, gz), scale=92.0, screen_center=sc)
        elif kind == "ground":
            aim = e.get("aim")
            tx, tz = (DETECTORS[aim] if aim is not None else (gx, gz))
            mx, mz = (gx + tx) / 2, (gz + tz) / 2
            yaw = math.radians(22.0 - 15.0 * phrase + 6.0 * u)
            D = 17.0
            pos = (mx + D * math.sin(yaw), 1.6, mz + D * math.cos(yaw))
            cam = Camera(pos, (mx, 4.6, mz), fov_deg=52.0, screen_center=sc)
            cam.fov = 52.0
        else:
            roll = 8.0 * phrase + 12.0 * u
            cam = OrthoCamera((gx, 40.0, gz + 1e-3), (gx, 0.0, gz), scale=108.0 + 14.0 * u, up=(0.0, 0.0, -1.0),
                              screen_center=sc, roll_deg=roll)
        return kind, cam, phrase, u

    @staticmethod
    def _fog(z):
        return np.clip(1.35 - z / 70.0, 0.25, 1.0).astype(np.float32)

    # ---------------------------------------------------------------- render
    def render(self, t, W=3000, H=1688):
        T = self.T
        t = t % T
        f = Frame(W, H)
        kind, cam, phrase, u = self.view(t)
        age = (t - self.ev_t0) % T
        alive = age < self.ev_life
        env = 1.0 - smoothstep(self.ev_life - 3.2, self.ev_life, age)
        invert = (t % PHRASE) < 0.06
        burst = t >= T - BEAT

        f.set_clip(*MAIN)
        if burst:
            self._draw_barcode_burst(f, t)
        else:
            self._draw_ground(f, cam, kind)
            det_glow, det_slabs = self._detector_state(age, alive)
            self._draw_detectors(f, cam, det_glow, det_slabs, kind)
            self._draw_cascades(f, cam, age, alive, env)
            self._draw_hits(f, cam, age, alive)
            self._draw_splash(f, cam, age)
            self._draw_view_overlay(f, cam, kind, phrase, age)
            self._draw_interaction(f, cam, age)
            self._draw_labels(f, cam, age, alive)
        f.set_clip()
        det_glow, det_slabs = self._detector_state(age, alive)
        self._draw_column(f, t, phrase, age)
        self._draw_strip(f, t, phrase, age)
        self._draw_bottom(f, t, age, det_glow, det_slabs, phrase)
        self._draw_view_tag(f, kind, cam, phrase)
        return f.finish(bloom_weights=(0.34, 0.3, 0.24, 0.2, 0.16, 0.13, 0.1, 0.08), bloom_gain=0.7,
                        invert=invert, invert_rect=(0, 318, 3000, 1688))

    # --- world ----------------------------------------------------------------
    def _draw_ground(self, f, cam, kind):
        p = self.lattice
        sx, sy, z, ok = cam.project(p)
        if kind == "side":
            xs = np.arange(-30.0, 30.01, 1.0, dtype=np.float32)
            P = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
            ax, ay, _, _ = cam.project(P)
            f.segments("w", [MAIN[0]], [ay[0]], [MAIN[2]], [ay[0]], 0.9)
            f.segments("w", ax, ay, ax, ay - np.where(np.arange(len(xs)) % 5 == 0, 14, 6), 0.7)
            return
        inten = np.where(self.lat_major, 0.9, 0.45) * self._fog(z)
        if kind == "top":
            f.pixels("w", sx[ok], sy[ok], inten[ok] * 1.2)
            mj = self.lat_major & ok
            f.crosses("w", sx[mj], sy[mj], 5.0, 0.6)
        else:
            f.pixels("w", sx[ok], sy[ok], inten[ok])
        # red baseline through the detectors
        xs = np.linspace(-24, 24, 49, dtype=np.float32)
        L = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
        lx, ly, lz, lok = cam.project(L)
        m = lok[:-1] & lok[1:]
        f.segments("r", lx[:-1][m], ly[:-1][m], lx[1:][m], ly[1:][m], 0.5 * self._fog(lz[:-1][m]))

    def _detector_state(self, age, alive):
        glow = np.zeros(len(DETECTORS), np.float32)
        slabs = np.zeros((len(DETECTORS), len(SLABS)), np.float32)
        for d in self.det_hits:
            ev = d["ev"]
            if not alive[ev]:
                continue
            a = age[ev] - d["t_hit"]
            if 0 <= a < 3.0:
                glow[d["det"]] = max(glow[d["det"]], math.exp(-a / 0.7))
            sa = age[ev] - d["slab_t"]
            on = d["slab_on"] & (sa >= 0)
            slabs[d["det"]] = np.maximum(slabs[d["det"]], np.where(on, np.exp(-np.maximum(sa, 0) / 0.45), 0.0))
        return glow, slabs

    def _draw_detectors(self, f, cam, glow, slabs, kind):
        a, b = self.det_lines[:, 0], self.det_lines[:, 1]
        ax, ay, az, aok = cam.project(a)
        bx, by, bz, bok = cam.project(b)
        own, sl = self.det_owner, self.det_slab
        w_i = 0.6 + 0.6 * glow[own]
        r_i = np.zeros(len(a), np.float32)
        m = sl >= 0
        r_i[m] = 1.6 * slabs[own[m], sl[m]]
        w_i = np.where(r_i > 0.05, w_i * 0.4, w_i)
        ok = aok & bok
        f.segments("w", ax[ok], ay[ok], bx[ok], by[ok], w_i[ok], width=np.where(sl < 0, 1.3, 1.0)[ok])
        f.segments("r", ax[ok], ay[ok], bx[ok], by[ok], r_i[ok], width=1.6)
        if kind == "top":   # detectors seen from above: fill when hit
            for di, (dx, dz) in enumerate(DETECTORS):
                if glow[di] > 0.05:
                    P = np.array([[dx - DET_HALF, 0, dz - DET_HALF], [dx + DET_HALF, 0, dz + DET_HALF]], np.float32)
                    px, py, _, _ = cam.project(P)
                    f.rects("r", min(px), min(py), max(px), max(py), 0.4 * glow[di])

    def _draw_cascades(self, f, cam, age, alive, env):
        sa, ev = self.SA, self.SE
        G = age[ev]
        ta = self.VT[sa]
        vis = alive[ev] & (G >= ta)
        if not vis.any():
            return
        sa, ev, G, ta = sa[vis], ev[vis], G[vis], ta[vis]
        kind = self.SK[vis]
        var = self.SVar[vis] * self.SGain[vis]
        tb = self.VT[sa + 1]
        pa = self.V[sa]
        pb = self.V[sa + 1].copy()
        part = G < tb
        frac = np.clip((G[part] - ta[part]) / np.maximum(tb[part] - ta[part], 1e-6), 0, 1)
        pb[part] = pa[part] + (pb[part] - pa[part]) * frac[:, None]
        tau = K_TAU[kind]
        base = K_INT[kind] * var * env[ev]
        flash = np.where(kind <= K_G, 0.5, 1.3).astype(np.float32)

        def inten(a):
            return base * (0.72 * np.exp(-a / tau) + 0.28 * np.exp(-a / (4.5 * tau)) + flash * np.exp(-a / 0.06))

        ia, ib = inten(G - ta), inten(np.where(part, 0.0, G - tb))
        ax, ay, az, aok = cam.project(pa)
        bx, by, bz, bok = cam.project(pb)
        ok = aok & bok
        ia = ia * self._fog(az)
        ib = ib * self._fog(bz)
        red = K_RED[kind] == 1
        width = K_W[kind]
        for m, name in ((ok & ~red, "w"), (ok & red, "r")):
            if m.any():
                f.segments(name, ax[m], ay[m], bx[m], by[m], ia[m], ib[m], width=width[m])
        hm = part & ok & (K_HEAD[kind] > 0)
        if hm.any():
            hk = kind[hm]
            fog = self._fog(bz[hm])
            hr = K_HEAD[hk] * np.clip(40.0 / np.maximum(bz[hm], 1.0), 0.6, 1.6)
            hred = K_RED[hk] == 1
            hi = K_HEAD_I[hk] * fog * env[ev[hm]]
            if (~hred).any():
                f.dots("w", bx[hm][~hred], by[hm][~hred], hr[~hred], hi[~hred])
            if hred.any():
                f.dots("r", bx[hm][hred], by[hm][hred], hr[hred], hi[hred])
                f.dots("w", bx[hm][hred], by[hm][hred], hr[hred] * 0.4, 0.8 * hi[hred])

    def _draw_hits(self, f, cam, age, alive):
        ev = self.hits[:, 0].astype(int)
        a = age[ev] - self.hits[:, 1]
        m = alive[ev] & (a >= 0) & (a < 1.6)
        if not m.any():
            return
        h, a = self.hits[m], a[m]
        u = a / 1.6
        mu = h[:, 4] == K_MU
        r = np.where(mu, 0.15 + 1.1 * (1 - (1 - u) ** 3), 0.1 + 0.5 * (1 - (1 - u) ** 3))
        n = 28
        ang = np.linspace(0, 2 * np.pi, n + 1, dtype=np.float32)
        P = np.stack([h[:, 2:3] + r[:, None] * np.cos(ang)[None], np.zeros((len(h), n + 1), np.float32),
                      h[:, 3:4] + r[:, None] * np.sin(ang)[None]], -1)
        sx, sy, z, ok = cam.project(P.reshape(-1, 3))
        sx, sy, z, ok = (v.reshape(len(h), n + 1) for v in (sx, sy, z, ok))
        inten = ((1 - u) ** 2 * np.where(mu, 1.0, 0.5))[:, None] * self._fog(z)
        for sel, name in ((mu, "r"), (~mu, "w")):
            if sel.any():
                okm = ok[sel][:, :-1] & ok[sel][:, 1:]
                f.segments(name, sx[sel][:, :-1][okm], sy[sel][:, :-1][okm], sx[sel][:, 1:][okm],
                           sy[sel][:, 1:][okm], inten[sel][:, :-1][okm])
        cx, cy, cz, cok = cam.project(np.stack([h[:, 2], np.zeros(len(h)), h[:, 3]], 1))
        f.dots("w", cx[cok], cy[cok], 2.0, (1.4 * (1 - u) ** 3)[cok])

    def _draw_splash(self, f, cam, age):
        for k, e in enumerate(self.events):
            if e["kind"] != "shower":
                continue
            a = age[k] - e["t_ground"]
            if not (0 <= a < 2.6):
                continue
            u = a / 2.6
            for j, (rmax, lay, rad) in enumerate(((11.0, "r", 2.2), (6.5, "w", 1.5))):
                r = 0.3 + rmax * (1 - (1 - u) ** 2.2)
                n = int(90 + 26 * r)
                ang = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float32) + j * 0.05
                P = np.stack([e["G"][0] + r * np.cos(ang), np.zeros(n, np.float32), e["G"][2] + r * np.sin(ang)], 1)
                sx, sy, z, ok = cam.project(P.astype(np.float32))
                f.dots(lay, sx[ok], sy[ok], rad * np.clip(36.0 / z[ok], 0.5, 1.6), (1.1 * (1 - u) ** 1.4) *
                       self._fog(z[ok]))

    def _draw_interaction(self, f, cam, age):
        rng = np.random.default_rng(3)
        for k, e in enumerate(self.events):
            if e["kind"] != "shower":
                continue
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
            f.rings("r", [cx], [cy], [20 + 230 * (1 - (1 - u) ** 2)], fade, width=1.5)
            big = 1e5
            f.segments("r", [cx - big, cx], [cy, cy - big], [cx + big, cx], [cy, cy + big], 0.55 * fade)
            f.tag("r", cx + 34, cy - 26, "FIRST_INTERACTION", size=16, alpha=fade, pad=4)
            f.text("w", cx + 34, cy + 4, f"H {e['h1']:.3f} KM   E0 {e['E0'] * 3.2:.2f}E15 EV", size=15, alpha=fade)
            f.tag("r", MAIN[0] + 346, cy + 6, f"{e['h1']:06.3f}", size=12, pad=3, alpha=fade)
            f.tag("r", cx + 4, MAIN[1] + 20, f"{float(e['P1'][0]):+07.3f}", size=12, pad=3, alpha=fade)

    def _draw_view_overlay(self, f, cam, kind, phrase, age):
        e = self.events[phrase]
        a = age[phrase]
        speed = e["speed"]
        front = e["h1"] - max(0.0, a - e["t1"]) * speed * math.cos(math.radians(e["zen"]))
        if kind == "side":
            # altitude rules every km with labels, red line at the shower front, red axis
            for h in range(0, 17):
                P = np.array([[0.0, h, 0.0]], np.float32) + np.array([[float(e["G"][0]), 0, float(e["G"][2])]],
                                                                     np.float32)
                px, py, _, _ = cam.project(P)
                y = float(py[0])
                f.segments("w", [MAIN[0] + 340], [y], [MAIN[2]], [y], 0.13 if h % 5 else 0.3)
                f.text("w", MAIN[0] + 350, y - 6, f"{h:02d} KM", size=13, alpha=0.75)
            if 0 < front < e["h1"]:
                P = np.array([[float(e["G"][0]), front, float(e["G"][2])]], np.float32)
                px, py, _, _ = cam.project(P)
                y = float(py[0])
                f.segments("r", [MAIN[0] + 340], [y], [MAIN[2]], [y], 0.9, width=1.4)
                f.tag("r", MAIN[2] - 12, y - 8, f"FRONT {front:06.3f} KM", size=14, pad=4, anchor="rs")
            ax, ay, _, _ = cam.project(np.stack([e["G"], e["P1"]]).astype(np.float32))
            f.segments("r", [ax[0]], [ay[0]], [ax[1]], [ay[1] - 60], 0.5)
        elif kind == "top":
            gx, gy, _, _ = cam.project(e["G"][None].astype(np.float32))
            X, Y = float(gx[0]), float(gy[0])
            big = 1e5
            f.segments("r", [X - big, X], [Y, Y - big], [X + big, X], [Y, Y + big], 0.45)
            sc = cam.scale
            f.rings("w", [X] * 4, [Y] * 4, [sc * r for r in (1, 2, 4, 8)], 0.25)
            for r in (1, 2, 4, 8):
                f.text("w", X + sc * r + 6, Y - 6, f"{r} KM", size=12, alpha=0.6)
            f.tag("r", X + 12, Y + 26, f"CORE {float(e['G'][0]):+07.3f} {float(e['G'][2]):+07.3f}", size=13, pad=3)

    def _draw_labels(self, f, cam, age, alive):
        cands = []
        for ev, tt, p, word, num, red in self.labels:
            if not alive[ev]:
                continue
            a = age[ev] - tt
            if 0 <= a < 2.6:
                cands.append((a, p, word, num, red, float(smoothstep(0, 0.08, a) * (1 - smoothstep(1.6, 2.6, a)))))
        cands.sort(key=lambda c: c[0])
        boxes = []
        for a, p, word, num, red, alpha in cands:
            px, py, pz, ok = cam.project(p[None])
            if not ok[0]:
                continue
            x, y = float(px[0]), float(py[0])
            if not (MAIN[0] + 360 < x < MAIN[2] - 200 and MAIN[1] + 40 < y < MAIN[3] - 20):
                continue
            bx = (x - 4, y - 44, x + 30 + 10 * len(num), y + 6)
            if any(bx[0] < o[2] and o[0] < bx[2] and bx[1] < o[3] and o[1] < bx[3] for o in boxes):
                continue
            if len(boxes) >= 8:
                break
            boxes.append(bx)
            f.dots("w", [x], [y], 2.0, alpha)
            f.segments("w", [x], [y], [x + 18], [y - 18], 0.5 * alpha)
            n = int(min(len(num), a * 80))
            if word:
                f.tag("r" if red else "w", x + 22, y - 22, word, size=14, alpha=alpha, pad=3)
                f.text("w", x + 22 + 11 * len(word) + 14, y - 22, num[:n], size=14, alpha=alpha)
            else:
                f.text("w", x + 22, y - 22, num[:n], size=14, alpha=alpha)

    def _draw_barcode_burst(self, f, t):
        fr = int(round(t * 30))
        u = (t - (self.T - BEAT)) / BEAT
        x0, y0, x1, y1 = MAIN
        lanes = 9
        lh = (y1 - y0) / lanes
        span = x1 - x0
        for ln in range(lanes):
            k = np.arange(520)
            wdt = 1 + (hash01(k, ln + 21) * 9).astype(int)
            on = hash01(k, ln + 9, 3) < 0.4 + 0.45 * u
            xs = np.cumsum(wdt) - wdt
            xs = (xs + fr * (11 + 6 * ln) * (1 if ln % 2 else -1)) % span + x0
            f.rects("w", xs[on], y0 + ln * lh + 2, xs[on] + wdt[on], y0 + (ln + 1) * lh - 2, 1.0)
        f.rects("r", x0, (y0 + y1) / 2 - 3, x1, (y0 + y1) / 2 + 3, 1.0)

    # --- HUD ---------------------------------------------------------------------
    def _draw_column(self, f, t, phrase, age):
        x0, y0, x1, y1 = COL
        f.occlude(x0, y0, x1, y1)
        f.rects("w", x0, y0, x1, y0 + 5, 0.95)
        f.tag("w", x0 + 4, y0 + 30, "PARTICLE_STREAM", size=13, pad=3)
        f.segments("w", [x1, x0], [y0, y1], [x1, x1], [y1, y1], 0.6)
        b = self.births[phrase]
        a = age[phrase]
        n_now = int(np.searchsorted(b["t"], a))
        rows = 52
        idx = np.arange(max(0, n_now - rows), n_now)[::-1]
        yy = y0 + 58
        for r, i in enumerate(idx):
            k = int(b["k"][i])
            p = b["p"][i]
            name = K_NAME[k] if k != K_MU else ("MU-" if i % 2 else "MU+")
            line = f"{i:05d} {name:<5} {b['e'][i] * 3.2e6:010.3f} {p[0]:+06.2f} {p[1]:05.2f} {p[2]:+06.2f}"
            red = K_RED[k] == 1
            f.text("r" if red else "w", x0 + 8, yy + r * 17, line, size=12, alpha=0.95 if r < 3 else 0.7)
        f.text("w", x0 + 8, y1 - 8, f"N {n_now:06d}", size=13, alpha=0.9)

    def _draw_strip(self, f, t, phrase, age):
        x0, y0, x1, y1 = STRIP
        e = self.events[phrase]
        a = age[phrase]
        f.segments("w", [x0, x0], [y0, y1], [x1, x1], [y0, y1], 0.9)

        def X(h):
            return x0 + (16.0 - h) / 16.0 * (x1 - x0)

        hs = np.arange(0, 16.01, 0.1)
        xs = X(hs)
        k = np.round(hs * 10).astype(int)
        ln = np.where(k % 10 == 0, 16.0, np.where(k % 5 == 0, 9.0, 4.0))
        f.segments("w", xs, np.full_like(xs, y0), xs, y0 + ln, 0.8)
        f.segments("w", xs, np.full_like(xs, y1), xs, y1 - ln, 0.8)
        for h in range(0, 17):
            f.text("w", X(h) + 5, y1 - 22, f"{h:02d} KM", size=12, alpha=0.7)
        yb = (y0 + y1) / 2 - 8
        f.rects("r", x0, yb - 3, x1, yb + 3, 1.0)
        # longitudinal profile of the current shower: births per 0.1 km, built live
        b = self.births[phrase]
        n_now = int(np.searchsorted(b["t"], a))
        if n_now:
            alt = b["p"][:n_now, 1]
            cnt, _ = np.histogram(alt, bins=160, range=(0.0, 16.0))
            hh = 52.0 * np.sqrt(cnt / max(cnt.max(), 1))
            bx = X(np.arange(160) * 0.1 + 0.1)
            m = cnt > 0
            f.rects("w", bx[m], y0 + 1, bx[m] + 7, y0 + 1 + hh[m], 0.95)
            hb = 30.0 * np.sqrt(cnt / max(cnt.max(), 1)) * hash01(np.arange(160), phrase)
            f.rects("w", bx[m], y1 - hb[m], bx[m] + 7, y1 - 1, 0.6)
            imax = int(np.argmax(cnt))
            if cnt[imax] > 30:
                f.tag("w", X(imax * 0.1 + 0.05), yb + 6, f"XMAX {imax * 0.1:04.1f} KM", size=13, pad=4)
        if a >= e["t1"] - 0.05:
            f.tag("w", X(e["h1"]), yb + 6, f"H1 {e['h1']:.1f} KM", size=13, pad=4)
        front = e["h1"] - max(0.0, a - e["t1"]) * e["speed"] * math.cos(math.radians(e["zen"]))
        if a < e["t1"]:
            front = min(16.0, e["h1"] + (e["t1"] - a) * V_PRIMARY)
        front = max(front, 0.0)
        xc = X(front)
        f.segments("r", [xc], [y0 - 4], [xc], [y1 + 4], 1.2, width=1.6)
        f.tag("r", xc + 6, y1 + 22, f"FRONT {front:06.3f} KM", size=13, pad=3)
        f.tag("w", x0 + 4, y0 - 10, f"LONGITUDINAL_PROFILE // SHOWER {phrase + 1:02d}", size=12, pad=3)

    def _draw_bottom(self, f, t, age, glow, slabs, phrase):
        x0, y0, x1, y1 = BOT
        # detector readouts
        for di, name in enumerate(DET_NAMES):
            px0 = x0 + di * 330
            f.rects("w", px0, y0, px0 + 300, y0 + 5, 0.95)
            f.tag("r" if glow[di] > 0.2 else "w", px0 + 4, y0 + 30, name, size=14, pad=3)
            for s in range(len(SLABS)):
                sx = px0 + 90 + s * 22
                f.rect("w", sx, y0 + 18, sx + 16, y0 + 34, 0.7)
                v = slabs[di, len(SLABS) - 1 - s]
                if v > 0.05:
                    f.rects("r", sx + 2, y0 + 20, sx + 14, y0 + 32, min(1.0, v * 1.4))
            n = sum(1 for d in self.det_hits if d["det"] == di and 0 <= (age[d["ev"]] - d["t_hit"]) < 10.0)
            f.text("w", px0 + 4, y0 + 66, f"HITS/10S {n:02d}", size=13, alpha=0.85)
            last = [age[d["ev"]] - d["t_hit"] for d in self.det_hits
                    if d["det"] == di and age[d["ev"]] - d["t_hit"] >= 0]
            if last:
                f.text("w", px0 + 4, y0 + 88, f"LAST +{min(last):07.3f} S", size=13, alpha=0.6)
            gl = float(glow[di])
            f.rects("r", px0 + 4, y1 - 14, px0 + 4 + 290 * gl, y1 - 8, 1.0)
        # birth-rate barcode
        bx0, bx1 = 1060.0, 1980.0
        f.rects("w", bx0, y0, bx1, y0 + 5, 0.95)
        f.tag("w", bx0 + 4, y0 - 10, "BIRTH_RATE >> BARCODE", size=12, pad=3)
        cols = 230
        dt = 3.0 / cols
        k_first = math.floor((t - 3.0) / dt)
        kk = k_first + np.arange(cols)
        rate = np.zeros(cols, np.float32)
        for ev, e in enumerate(self.events[: int(round(self.T / PHRASE))]):
            b = self.births[ev]
            tt = (kk * dt - e["t0"]) % self.T
            lo = np.searchsorted(b["t"], tt)
            hi = np.searchsorted(b["t"], tt + dt)
            rate += (hi - lo)
        dens = 0.06 + 0.9 * np.tanh(rate / 60.0)
        cw = (bx1 - bx0) / cols
        frac = (t - 3.0) / dt - k_first
        xs = bx0 + (np.arange(cols) - frac) * cw
        lane_h = (y1 - y0 - 12) / 3
        for ln in range(3):
            on = hash01(kk, ln + 13) < dens * (1.0 - 0.2 * ln)
            m = on & (xs >= bx0) & (xs + cw <= bx1)
            ly0 = y0 + 10 + ln * lane_h
            f.rects("w", xs[m], ly0, xs[m] + cw, ly0 + lane_h - 3, 0.95)
        f.segments("r", [bx1 - 2], [y0 + 7], [bx1 - 2], [y1], 1.2)
        # counters
        cx0 = 2020.0
        f.rects("w", cx0, y0, x1, y0 + 5, 0.95)
        b = self.births[phrase]
        n_now = int(np.searchsorted(b["t"], age[phrase]))
        kinds = b["k"][:n_now]
        rows = [("E+-", int((kinds == K_E).sum()), "w"), ("GAMMA", int((kinds == K_G).sum()), "w"),
                ("HADRON", int((kinds == K_H).sum()), "w"), ("MU+-", int((kinds == K_MU).sum()), "r")]
        for r, (lab, n, lay) in enumerate(rows):
            xx = cx0 + (r % 2) * 470
            yy = y0 + 42 + (r // 2) * 40
            f.tag(lay, xx + 4, yy, lab, size=15, pad=3)
            f.text(lay, xx + 120, yy, f"{n:06d}", size=24, alpha=0.95)

    def _draw_view_tag(self, f, kind, cam, phrase):
        names = {"persp": "01 // PERSPECTIVE", "side": "02 // ORTHO_SIDE", "ground": "03 // GROUND",
                 "top": "04 // ORTHO_TOP"}
        x0, y0 = MAIN[0] + 370, MAIN[1] + 36
        f.tag("w", x0, y0, f"VIEW {names[kind]}", size=16, pad=4)
        if getattr(cam, "ortho", False):
            s = cam.scale
            f.segments("w", [x0, x0, x0 + 5 * s], [y0 + 34, y0 + 28, y0 + 28], [x0 + 5 * s, x0, x0 + 5 * s],
                       [y0 + 34, y0 + 40, y0 + 40], 0.9)
            for k in range(6):
                f.segments("w", [x0 + k * s], [y0 + 34], [x0 + k * s], [y0 + 26], 0.9)
            f.text("w", x0 + 5 * s + 10, y0 + 40, f"5 KM   1:{int(1e5 / s):d}", size=13, alpha=0.8)
        else:
            f.text("w", x0, y0 + 36, f"F {getattr(cam, 'fov', 44.0):.1f} DEG   CAM ORBIT", size=13, alpha=0.7)
        f.text("w", MAIN[2] - 20, y0, f"AIR_SHOWER // MUON BLOOM // SHOWER {phrase + 1:02d}/04", size=14,
               alpha=0.85, anchor="rs")
        e = self.events[phrase]
        f.text("w", MAIN[2] - 20, y0 + 22, f"E0 {e['E0'] * 3.2:.2f}E15 EV   ZENITH {e['zen']:.1f} DEG", size=13,
               alpha=0.6, anchor="rs")
