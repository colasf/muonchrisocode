"""AIR SHOWER - a cosmic ray hits the upper atmosphere and rains particles onto the detectors.

3D, slowly swaying camera (the sphere / galaxy language): perspective ground grid
with '+' marks, a red altitude axis, three wireframe detector towers.

Each shower is a small physics-flavoured cascade:
  primary (red) -> hadronic interactions -> pi0 -> photons -> e+e- / bremsstrahlung
  sub-showers (white, feathery, getting denser as the air thickens) and
  pi+- -> muons (red, long straight tracks reaching the ground).
The shower front sweeps down as a band of bright heads; trails fade to a ghost of
the structure. Muons that cross a detector light its slabs and pop a readout.
Background muon "rain" keeps the detectors ticking on a 128 BPM grid.
Everything loops seamlessly every T seconds.
"""
from __future__ import annotations

import math

import numpy as np

from .engine import Camera, Frame, smoothstep

BPM = 128.0
BEAT = 60.0 / BPM

V = 5.2                   # visual speed of the shower front (units / s, 1 unit ~ 1 km)
V_PRIMARY = 11.0
TOP = 27.0                # primaries enter from this altitude (above the frame)
DETECTORS = [(-9.0, 0.0), (0.0, 0.0), (9.0, 0.0)]
DET_NAMES = ["DET_L", "DET_C", "DET_R"]
DET_HALF = 0.55
DET_H = 2.7
SLABS = np.linspace(0.25, DET_H - 0.2, 9)

# kind codes
K_E, K_G, K_H, K_MU, K_P, K_RAIN = 0, 1, 2, 3, 4, 5
#                 e      gamma  hadron  muon   primary rain-muon
K_INT = np.array([0.30, 0.12, 0.95, 0.95, 1.5, 0.9], np.float32)
K_TAU = np.array([1.0, 0.7, 2.0, 1.7, 3.0, 0.28], np.float32)
K_W = np.array([1.0, 1.0, 1.35, 1.25, 2.4, 1.1], np.float32)
K_RED = np.array([0, 0, 0, 1, 1, 1], np.int8)
K_HEAD = np.array([1.1, 0.0, 2.4, 2.6, 4.2, 2.4], np.float32)
K_HEAD_I = np.array([0.45, 0.0, 1.3, 1.4, 1.6, 1.3], np.float32)


def _perp_basis(d):
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    return u, w


def _deflect(d, theta, rng):
    u, w = _perp_basis(d)
    phi = rng.uniform(0, 2 * math.pi)
    nd = math.cos(theta) * d + math.sin(theta) * (math.cos(phi) * u + math.sin(phi) * w)
    return nd / np.linalg.norm(nd)


class _Cascade:
    """Builds branches (polylines with arrival times) for one shower."""

    def __init__(self, rng, max_branches):
        self.rng = rng
        self.max_branches = max_branches
        self.branches = []      # (pts (k,3), times (k,), kind, energy)

    def add(self, p0, d, length, t0, kind, e, curv=0.0, speed=V, ground_stop=True):
        rng = self.rng
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
        if ground_stop and pts[-1, 1] < 0.0:
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


def build_shower(rng, ground, h1, zen_deg, az_deg, E0, Ec=1.4e-4, max_branches=9000):
    zen, az = math.radians(zen_deg), math.radians(az_deg)
    d0 = np.array([math.sin(zen) * math.cos(az), -math.cos(zen), math.sin(zen) * math.sin(az)])
    G = np.array([ground[0], 0.0, ground[1]])
    P1 = G - d0 * (h1 / math.cos(zen))
    Ptop = G - d0 * (TOP / math.cos(zen))
    c = _Cascade(rng, max_branches)
    _, _, t1, _ = c.add(Ptop, d0, float(np.linalg.norm(P1 - Ptop)), 0.0, K_P, E0, speed=V_PRIMARY)
    info = dict(P1=P1, t1=t1, d0=d0, E0=E0, zen=zen_deg, h1=h1, G=G,
                t_ground=t1 + (h1 / math.cos(zen)) / V)
    stack = [(P1, d0, E0, K_H, t1, 0)]
    em_first = []
    while stack:
        p, d, e, kind, t, gen = stack.pop()
        if kind == K_H:
            L = _lam_h(p[1]) * rng.exponential(1.0) * (0.4 if gen == 0 else 1.0)
            end, tang, tend, hit = c.add(p, d, L, t, K_H, e)
            if hit:
                continue
            # hadronic interaction: leading particle + pions
            m = int(3 + rng.poisson(2.5 + 3.0 * e ** 0.3))
            frac = rng.dirichlet(np.ones(m) * 0.7) * e * 0.55
            lead = e * 0.45
            if lead > 0.012:
                stack.append((end, _deflect(tang, 0.01 * rng.exponential(), rng), lead, K_H, tend, gen + 1))
            for ei in frac:
                th = min(0.9, 0.03 / math.sqrt(max(ei, 1e-6)) * (0.35 + rng.exponential()))
                nd = _deflect(tang, th, rng)
                r = rng.random()
                if r < 0.66:          # charged pion
                    if ei > 0.05 and not c.full():
                        stack.append((end, nd, ei, K_H, tend, gen + 1))
                    else:             # decays into a muon
                        ld = rng.exponential(0.35 + 4.0 * ei)
                        pe, ptang, pt, phit = c.add(end, nd, ld, tend, K_H, ei)
                        if not phit:
                            md = _deflect(ptang, 0.02 * rng.exponential(), rng)
                            cosang = float(np.dot(md, d0))
                            if cosang < math.cos(0.36):
                                md = _deflect(d0, rng.uniform(0.05, 0.36), rng)
                            c.add(pe, md, 60.0, pt, K_MU, ei * 0.8)
                else:                 # neutral pion -> two photons
                    for share in (0.5 + 0.3 * (rng.random() - 0.5),):
                        for ef in (ei * share, ei * (1 - share)):
                            gd = _deflect(nd, 0.02 * rng.exponential(), rng)
                            stack.append((end, gd, ef, K_G, tend, gen + 1))
        else:  # electromagnetic: e (brems) / gamma (pair)
            lam = _lam_e(p[1])
            if e < Ec or c.full():
                L = min(lam * rng.exponential(0.55), 1.3)
                curv = 0.9 * rng.exponential() if kind == K_E else 0.0
                c.add(p, d, L, t, kind, e, curv=curv)
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


class ShowerV1:
    name = "shower_v1"

    def __init__(self, T=30.0, seed=4):
        self.T = T
        rng = np.random.default_rng(seed)
        self.rng = rng
        # (t0, ground x, z, E0, zenith, azimuth, h1, aim detector)
        plan = [
            (0.25 * T / 30 * 30 * 0 + 0.2, -9.0, 0.6, 0.75, 9.0, 40.0, 14.2, 0),
            (6.2, 7.0, -3.0, 0.5, 16.0, 200.0, 13.2, None),
            (12.0, 0.4, 0.3, 1.0, 5.0, 130.0, 14.8, 1),
            (18.3, -4.0, -5.5, 0.55, 20.0, 320.0, 13.6, None),
            (23.9, 9.0, 0.8, 0.8, 11.0, 250.0, 14.0, 2),
        ]
        self.events = []
        B = []   # (branches, event index)
        for k, (t0, gx, gz, E0, zen, az, h1, aim) in enumerate(plan):
            br, info = build_shower(rng, (gx, gz), h1, zen, az, E0, max_branches=int(9000 + 14000 * E0))
            if aim is not None:
                br = self._aim_muons(br, aim, rng, count=2)
            info.update(t0=t0 % T, kind="shower", aim=aim)
            self.events.append(info)
            B.append(br)
        # background muon rain, on the beat
        n_beats = int(round(T / BEAT))
        beats = rng.choice(np.arange(n_beats), size=int(n_beats * 0.45), replace=False)
        for j, b in enumerate(beats):
            aim = j % 3 == 0
            if aim:
                dx, dz = DETECTORS[rng.integers(0, 3)]
                gx, gz = dx + rng.uniform(-0.3, 0.3), dz + rng.uniform(-0.3, 0.3)
            else:
                gx, gz = rng.uniform(-20, 20), rng.uniform(-16, 12)
            zen = math.radians(abs(rng.normal(0, 14)))
            az = rng.uniform(0, 2 * math.pi)
            d = np.array([math.sin(zen) * math.cos(az), -math.cos(zen), math.sin(zen) * math.sin(az)])
            G = np.array([gx, 0.0, gz])
            start = G - d * (TOP / math.cos(zen))
            # arrive on the beat
            t_travel = (TOP / math.cos(zen)) / (V * 1.6)
            t0 = (b * BEAT - t_travel) % T
            pts = np.stack([start, G]).astype(np.float32)
            times = np.array([0.0, t_travel], np.float32)
            self.events.append(dict(t0=t0, kind="rain", aim=aim))
            B.append([(pts, times, K_RAIN, 0.3)])
        self._pack(B)
        self._build_static()

    # ---------------------------------------------------------------- build
    @staticmethod
    def _aim_muons(branches, det, rng, count=4):
        """Nudge the muons that land closest to a detector so they cross it."""
        dx, dz = DETECTORS[det]
        mu = [i for i, b in enumerate(branches) if b[2] == K_MU and b[0][-1, 1] <= 1e-3]
        if not mu:
            return branches
        dist = [math.hypot(branches[i][0][-1, 0] - dx, branches[i][0][-1, 2] - dz) for i in mu]
        order = [mu[i] for i in np.argsort(dist)[:count]]
        out = list(branches)
        for i in order:
            pts, times, kind, e = branches[i]
            start = pts[0].astype(np.float64)
            G = np.array([dx + rng.uniform(-0.35, 0.35), 0.0, dz + rng.uniform(-0.35, 0.35)])
            L = float(np.linalg.norm(G - start))
            npts = np.stack([start, G]).astype(np.float32)
            ntimes = np.array([times[0], times[0] + L / V], np.float32)
            out[i] = (npts, ntimes, kind, e)
        return out

    def _pack(self, B):
        V_, VT, SE, SK, SA, EN = [], [], [], [], [], []
        off = 0
        birth = []
        for ev, brs in enumerate(B):
            kinds_times = {K_E: [], K_G: [], K_H: [], K_MU: [], K_P: [], K_RAIN: []}
            for pts, times, kind, e in brs:
                m = len(pts)
                V_.append(pts)
                VT.append(times)
                SA.append(np.arange(off, off + m - 1, dtype=np.int64))
                SE.append(np.full(m - 1, ev, np.int32))
                SK.append(np.full(m - 1, kind, np.int8))
                EN.append(np.full(m - 1, e, np.float32))
                off += m
                kinds_times[kind].append(times[0])
            birth.append({k: np.sort(np.asarray(v, np.float32)) for k, v in kinds_times.items()})
        self.V = np.concatenate(V_).astype(np.float32)
        self.VT = np.concatenate(VT).astype(np.float32)
        self.SA = np.concatenate(SA)
        self.SE = np.concatenate(SE)
        self.SK = np.concatenate(SK)
        self.SEn = np.concatenate(EN)
        self.birth = birth
        n_ev = len(B)
        self.ev_t0 = np.array([e["t0"] for e in self.events], np.float32)
        last = np.zeros(n_ev, np.float32)
        np.maximum.at(last, self.SE, self.VT[self.SA + 1])
        self.ev_end = last
        self.ev_life = last + 4.5
        # a little per-segment variation so the cascade does not look uniform
        rng = np.random.default_rng(99)
        self.SVar = rng.uniform(0.6, 1.35, len(self.SA)).astype(np.float32)
        # EM brightness grows a bit with energy
        self.SGain = np.where(self.SK <= K_G, np.clip(0.75 + 0.12 * np.log10(np.maximum(self.SEn, 1e-6) / 2.6e-4), 0.7, 1.5),
                              1.0).astype(np.float32)

        # ground hits of muons / hadrons (ring ripples) and detector crossings
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
                    f = (a[1] - ys) / max(a[1] - b[1], 1e-6)
                    f = np.clip(f, 0, 1)
                    xs = a[0] + (b[0] - a[0]) * f
                    zs = a[2] + (b[2] - a[2]) * f
                    inside = (np.abs(xs - dx) < DET_HALF) & (np.abs(zs - dz) < DET_HALF)
                    if inside.sum() >= 3:
                        tp = ta + (tb - ta) * f
                        dets.append(dict(ev=ev, det=di, slab_t=tp[:-1].astype(np.float32),
                                         slab_on=inside[:-1], t_top=float(tp[-1]), t_hit=float(tp[inside].min())))
        self.hits = np.array([(h[0], h[1], h[2], h[3], h[4]) for h in hits], np.float32)
        self.det_hits = dets
        rng = np.random.default_rng(7)
        for d in dets:
            d["label"] = f"{DET_NAMES[d['det']]} HIT {rng.uniform(0.001, 0.009):.10f}"

        # floating numbers near vertices of the bigger branches
        labels = []
        rng = np.random.default_rng(17)
        for ev, brs in enumerate(B):
            if self.events[ev]["kind"] != "shower":
                continue
            cand = [(pts[-1], times[-1], kind, e) for pts, times, kind, e in brs
                    if kind in (K_E, K_H, K_MU) and pts[-1, 1] > 0.6]
            if not cand:
                continue
            pick = rng.choice(len(cand), size=min(40, len(cand)), replace=False)
            for i in pick:
                p, tt, kind, e = cand[i]
                red = kind == K_MU or rng.random() < 0.18
                txt = f"{rng.uniform(0.0005, 0.02):.10f}" if kind != K_MU else f"MU {rng.uniform(1, 9):.4f} GEV"
                labels.append((ev, float(tt), p.astype(np.float32), txt, 1 if red else 0))
        self.labels = labels

    def _build_static(self):
        g = np.arange(-24.0, 24.01, 2.0)
        self.grid_lines = []
        for v in g:
            self.grid_lines.append(((v, 0.0, -24.0), (v, 0.0, 24.0)))
            self.grid_lines.append(((-24.0, 0.0, v), (24.0, 0.0, v)))
        self.grid_lines = np.array(self.grid_lines, np.float32)
        X, Z = np.meshgrid(g, g)
        self.grid_pts = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1).astype(np.float32)
        # detector wireframes: vertical edges + slab rectangles
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

    # ---------------------------------------------------------------- camera
    def camera(self, t):
        ph = 2 * math.pi * t / self.T
        yaw = math.radians(15.0 * math.sin(ph) + 3.0 * math.sin(2 * ph + 0.7))
        dist = 35.0 + 1.8 * math.sin(ph + 1.3)
        h = 8.6 + 0.9 * math.sin(2 * ph + 0.4)
        pos = (dist * math.sin(yaw), h, dist * math.cos(yaw))
        target = (0.6 * math.sin(ph + 0.5), 6.9, 0.0)
        return Camera(pos, target, fov_deg=43.0, roll_deg=0.8 * math.sin(ph + 2.0))

    @staticmethod
    def _fog(z):
        return np.clip(1.35 - z / 70.0, 0.25, 1.0).astype(np.float32)

    # ---------------------------------------------------------------- render
    def render(self, t, W=3000, H=1688):
        f = Frame(W, H)
        cam = self.camera(t)
        T = self.T
        age = (t - self.ev_t0) % T
        alive = age < self.ev_life
        env = 1.0 - smoothstep(self.ev_life - 3.2, self.ev_life, age)

        self._draw_ground(f, cam, t)
        self._draw_axis(f, cam)
        det_glow, det_slabs = self._detector_state(age, alive)
        self._draw_detectors(f, cam, det_glow, det_slabs)
        self._draw_cascades(f, cam, age, alive, env)
        self._draw_hits(f, cam, age, alive)
        self._draw_splash(f, cam, age)
        self._draw_interactions(f, cam, age)
        self._draw_labels(f, cam, age, alive, det_glow)
        self._draw_hud(f, t, age, det_glow)
        return f.finish(bloom_weights=(0.5, 0.5, 0.45, 0.4, 0.36, 0.32, 0.28, 0.24), bloom_gain=0.9)

    def _draw_ground(self, f, cam, t):
        a = self.grid_lines[:, 0]
        b = self.grid_lines[:, 1]
        # split the long lines so the fog can vary along them
        n = 12
        u = np.linspace(0, 1, n + 1, dtype=np.float32)
        P = a[:, None, :] + (b - a)[:, None, :] * u[None, :, None]
        sx, sy, z, ok = cam.project(P.reshape(-1, 3))
        sx, sy, z, ok = (v.reshape(-1, n + 1) for v in (sx, sy, z, ok))
        inten = 0.075 * self._fog(z)
        m = ok[:, :-1] & ok[:, 1:]
        f.segments("w", sx[:, :-1][m], sy[:, :-1][m], sx[:, 1:][m], sy[:, 1:][m], inten[:, :-1][m], inten[:, 1:][m])
        # '+' marks on the intersections
        p = self.grid_pts
        h = 0.22
        ends = np.concatenate([p - [h, 0, 0], p + [h, 0, 0], p - [0, 0, h], p + [0, 0, h]]).astype(np.float32)
        sx, sy, z, ok = cam.project(ends)
        k = len(p)
        ia = 0.42 * self._fog(z[:k])
        for j0, j1 in ((0, k), (2 * k, 3 * k)):
            s0 = slice(j0, j0 + k)
            s1 = slice(j1 - k + k, j1 + k) if False else slice(j0 + k, j0 + 2 * k)
            m = ok[s0] & ok[s1]
            f.segments("w", sx[s0][m], sy[s0][m], sx[s1][m], sy[s1][m], ia[m])
        # red baseline through the detectors
        P = np.array([[-24, 0, 0], [24, 0, 0]], np.float32)
        u = np.linspace(0, 1, 25, dtype=np.float32)
        L = P[0][None] + (P[1] - P[0])[None] * u[:, None]
        sx, sy, z, ok = cam.project(L)
        f.segments("r", sx[:-1], sy[:-1], sx[1:], sy[1:], 0.55 * self._fog(z[:-1]))

    def _draw_axis(self, f, cam):
        base = np.array([-15.0, 0.0, -4.0], np.float32)
        top = base + [0, 18.0, 0]
        sx, sy, z, ok = cam.project(np.stack([base, top]))
        f.segments("r", sx[:1], sy[:1], sx[1:], sy[1:], 0.9, width=1.3)
        ys = np.arange(0.0, 18.01, 0.5, dtype=np.float32)
        ln = np.where(ys % 2 == 0, 0.55, 0.22).astype(np.float32)
        a = np.stack([np.full_like(ys, base[0]), ys, np.full_like(ys, base[2])], 1)
        b = a + np.stack([ln, np.zeros_like(ys), np.zeros_like(ys)], 1)
        ax, ay, _, _ = cam.project(a)
        bx, by, _, _ = cam.project(b)
        f.segments("r", ax, ay, bx, by, 0.85)
        for y in range(2, 18, 2):
            px, py, _, _ = cam.project(np.array([[base[0] + 0.75, y, base[2]]], np.float32))
            f.text("w", float(px[0]), float(py[0]) + 7, f"{y:02d}.0 KM", size=19, alpha=0.7)

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
            val = np.where(on, np.exp(-np.maximum(sa, 0) / 0.45), 0.0)
            slabs[d["det"]] = np.maximum(slabs[d["det"]], val)
        return glow, slabs

    def _draw_detectors(self, f, cam, glow, slabs):
        a = self.det_lines[:, 0]
        b = self.det_lines[:, 1]
        ax, ay, az, aok = cam.project(a)
        bx, by, bz, bok = cam.project(b)
        own = self.det_owner
        sl = self.det_slab
        w_i = 0.55 + 0.6 * glow[own]
        r_i = np.zeros(len(a), np.float32)
        m = sl >= 0
        r_i[m] = 1.6 * slabs[own[m], sl[m]]
        w_i = np.where(r_i > 0.05, w_i * 0.4, w_i)
        f.segments("w", ax, ay, bx, by, w_i, width=np.where(sl < 0, 1.4, 1.0))
        f.segments("r", ax, ay, bx, by, r_i, width=1.6)

    def _draw_cascades(self, f, cam, age, alive, env):
        sa = self.SA
        ev = self.SE
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
        aa = G - ta
        ab = np.where(part, 0.0, G - tb)
        tau = K_TAU[kind]
        base = K_INT[kind] * var * env[ev]
        flash = np.where(kind <= K_G, 0.5, 1.3).astype(np.float32)

        def inten(a):
            return base * (0.72 * np.exp(-a / tau) + 0.28 * np.exp(-a / (4.5 * tau)) + flash * np.exp(-a / 0.06))

        ia, ib = inten(aa), inten(ab)
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
        # heads of the moving front
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
        if not len(self.hits):
            return
        ev = self.hits[:, 0].astype(int)
        a = age[ev] - self.hits[:, 1]
        m = alive[ev] & (a >= 0) & (a < 1.6)
        if not m.any():
            return
        h = self.hits[m]
        a = a[m]
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
                f.segments(name, sx[sel][:, :-1][okm], sy[sel][:, :-1][okm], sx[sel][:, 1:][okm], sy[sel][:, 1:][okm],
                           inten[sel][:, :-1][okm])
        cx, cy, cz, cok = cam.project(np.stack([h[:, 2], np.zeros(len(h)), h[:, 3]], 1))
        f.dots("w", cx[cok], cy[cok], 2.2, (1.4 * (1 - u) ** 3)[cok])

    def _draw_splash(self, f, cam, age):
        for k, e in enumerate(self.events):
            if e["kind"] != "shower":
                continue
            a = age[k] - e["t_ground"]
            if not (0 <= a < 2.6):
                continue
            u = a / 2.6
            for j, (rmax, lay, rad) in enumerate(((11.0, "r", 2.3), (6.5, "w", 1.6))):
                r = 0.3 + rmax * (1 - (1 - u) ** 2.2)
                n = int(90 + 26 * r)
                ang = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float32) + j * 0.05
                P = np.stack([e["G"][0] + r * np.cos(ang), np.zeros(n, np.float32), e["G"][2] + r * np.sin(ang)], 1)
                sx, sy, z, ok = cam.project(P.astype(np.float32))
                f.dots(lay, sx[ok], sy[ok], rad * np.clip(36.0 / z[ok], 0.5, 1.6), (1.1 * (1 - u) ** 1.4) * self._fog(z[ok]))

    def _draw_interactions(self, f, cam, age):
        rng = np.random.default_rng(3)
        for k, e in enumerate(self.events):
            if e["kind"] != "shower":
                continue
            a = age[k] - e["t1"]
            if not (0 <= a < 2.4):
                continue
            px, py, pz, ok = cam.project(e["P1"][None].astype(np.float32))
            if not ok[0]:
                continue
            cx, cy = float(px[0]), float(py[0])
            u = a / 2.4
            # burst: radial pins with dots (the sphere's language) + an expanding ring
            n = 64
            ang = rng.uniform(0, 2 * np.pi, n)
            ln = rng.uniform(60, 260, n) * (1 - (1 - min(1, a / 0.5)) ** 3)
            r0 = 18 + 40 * u
            fade = (1 - u) ** 1.6
            x0, y0 = cx + np.cos(ang) * r0, cy + np.sin(ang) * r0
            x1, y1 = cx + np.cos(ang) * (r0 + ln), cy + np.sin(ang) * (r0 + ln)
            f.segments("w", x0, y0, x1, y1, 0.55 * fade)
            f.dots("w", x1, y1, 3.2, 1.2 * fade)
            f.rings("r", [cx], [cy], [20 + 230 * (1 - (1 - u) ** 2)], 1.0 * fade, width=1.5)
            f.dots("r", [cx], [cy], 6.0, 2.0 * fade)
            f.text("w", cx + 30, cy - 22, f"FIRST_INTERACTION  H {e['h1']:.2f} KM", size=21, alpha=fade)
            f.text("r", cx + 30, cy + 6, f"E0 {e['E0'] * 3.2:.2f}E15 EV", size=21, alpha=fade)

    def _draw_labels(self, f, cam, age, alive, det_glow):
        cands = []
        for ev, tt, p, txt, red in self.labels:
            if not alive[ev]:
                continue
            a = age[ev] - tt
            if not (0 <= a < 2.6):
                continue
            alpha = smoothstep(0, 0.08, a) * (1 - smoothstep(1.6, 2.6, a))
            cands.append((a, ev, tt, p, txt, red, float(alpha)))
        cands.sort(key=lambda c: c[0])                   # oldest first: they keep their place
        boxes = []
        for a, ev, tt, p, txt, red, alpha in cands:
            px, py, pz, ok = cam.project(p[None])
            if not ok[0]:
                continue
            x, y = float(px[0]), float(py[0])
            bx = (x - 4, y - 32, x + 16 + 12.2 * len(txt), y + 4)
            if any(bx[0] < o[2] and o[0] < bx[2] and bx[1] < o[3] and o[1] < bx[3] for o in boxes):
                continue
            if len(boxes) >= 9:
                break
            boxes.append(bx)
            f.dots("w", [x], [y], 2.0, alpha)
            n = int(min(len(txt), a * 80))
            f.text("r" if red else "w", x + 10, y - 8, txt[:n], size=20, alpha=alpha)
        for d in self.det_hits:
            ev = d["ev"]
            if not alive[ev]:
                continue
            a = age[ev] - d["t_hit"]
            if not (0 <= a < 2.8):
                continue
            alpha = smoothstep(0, 0.06, a) * (1 - smoothstep(1.8, 2.8, a))
            dx, dz = DETECTORS[d["det"]]
            px, py, _, ok = cam.project(np.array([[dx + DET_HALF, DET_H + 0.35, dz]], np.float32))
            if ok[0]:
                n = int(min(len(d["label"]), a * 90))
                f.text("r", float(px[0]) + 12, float(py[0]), d["label"][:n], size=21, alpha=alpha)

    def _draw_hud(self, f, t, age, det_glow):
        T = self.T
        showers = [k for k, e in enumerate(self.events) if e["kind"] == "shower"]
        cur = min(showers, key=lambda k: age[k])
        e = self.events[cur]
        a = age[cur]
        x0, y0 = 60, 350
        f.text("w", x0, y0, "AIR_SHOWER // MUON BLOOM", size=22, alpha=0.95)
        f.text("w", x0, y0 + 30, f"EVENT 0x3F2A-{cur:02d}   E0 {e['E0'] * 3.2:.2f}E15 EV   ZENITH {e['zen']:.1f} DEG",
               size=18, alpha=0.65)
        # live counters for the current shower
        b = self.birth[cur]
        cnt = {k: int(np.searchsorted(b[k], a)) for k in b}
        f.text("w", x0, y0 + 60, f"E+- {cnt[K_E]:05d}   GAMMA {cnt[K_G]:05d}   HADRON {cnt[K_H]:04d}", size=18,
               alpha=0.65)
        f.text("r", x0, y0 + 88, f"MU+- {cnt[K_MU]:04d}", size=18, alpha=0.95)
        # front altitude + scaled time
        tc = max(0.0, a - e["t1"])
        front = max(0.0, e["h1"] - V * tc * math.cos(math.radians(e["zen"])))
        if a < e["t1"]:
            front = e["h1"] + (e["t1"] - a) * V_PRIMARY
        x1 = 2940
        f.text("w", x1, y0, f"FRONT_ALT {min(front, 99.99):05.2f} KM", size=22, alpha=0.95, anchor="rs")
        f.text("w", x1, y0 + 30, f"T+{tc * 1.6e-5:.9f} S", size=18, alpha=0.65, anchor="rs")
        # altitude gauge on the right
        gx, gy0, gy1 = 2905.0, 470.0, 1350.0
        ys = np.linspace(gy1, gy0, 17)
        ln = np.where(np.arange(17) % 4 == 0, 22.0, 10.0)
        f.segments("w", np.full(17, gx), ys, gx - ln, ys, 0.55)
        f.segments("w", [gx], [gy0], [gx], [gy1], 0.35)
        for k in range(0, 17, 4):
            f.text("w", gx - 30, ys[k] + 6, f"{k:02d}", size=17, alpha=0.6, anchor="rs")
        my = gy1 + (gy0 - gy1) * min(front, 16.0) / 16.0
        f.segments("r", [gx - 44], [my], [gx + 8], [my], 1.3, width=1.6)
        f.dots("r", [gx - 50], [my], 3.5, 1.6)
        # detector readouts
        by = 1590
        for i, name in enumerate(DET_NAMES):
            n = sum(1 for d in self.det_hits if d["det"] == i and 0 <= (age[d["ev"]] - d["t_hit"]) < 10.0)
            xx = 60 + i * 300
            f.text("r" if det_glow[i] > 0.2 else "w", xx, by, f"{name} {n:02d}", size=20, alpha=0.95 if det_glow[i] > 0.2 else 0.7)
