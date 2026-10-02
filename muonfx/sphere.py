"""SPHERE - muon scattering tomography of a lumpy body. Ryoji Ikeda edition.

The plexus-sphere still (ouptut/newscene.png) redrawn in the Muon Bloom language:
  * the body: a lumpy sphere drawn as a rim-lit surface net, crossed by
    reconstructed cosmic-muon tracks. Tracks through the hidden dense core kink
    at their point of closest approach (PoCA): the dots gather on the core and
    give it away. Nuclear 'stars' fan prongs out of single vertices;
  * the lollipop halo becomes a polar opacity histogram: 144 bins on a precise
    base circle, its peak pointing at the core, flagged bins in red, unrolled
    again in the right column;
  * one hero muon per phrase goes straight through the core (red, data tag).
HUD: track timeline strip, MUOGRAPHY column with the PoCA track log, counts,
scatter barcode, scattering-angle spectrum. The camera makes one turn per loop.
"""
from __future__ import annotations

import math

import numpy as np

from .engine import DESIGN_H, Camera, Frame, hash01, periodic_noise
from .hud import (BAR, BEAT, BOT, MAIN, PHRASE, STRIP, barcode_burst, barcode_lanes, finish, panel_header,
                  ruler, timing)

C = (1500.0, (MAIN[1] + MAIN[3]) / 2)
LEFT_COL = (40.0, 505.0, 372.0, 1447.0)
RIGHT_COL = (2560.0, 505.0, 2960.0, 1447.0)
VIEW = (400.0, MAIN[1], 2530.0, MAIN[3])
R_PX = 288.0                  # projected radius of the unit sphere (design px)
CAM_D, CAM_H = 7.4, 1.9
N_BINS = 144
RING_R0 = 340.0               # base circle of the opacity histogram
RING_L = (14.0, 116.0)        # spoke length for opacity 0 .. 1
RING_OUT = RING_R0 + RING_L[1]
HOT = 0.78                    # flagged bins
CORE = np.array([0.30, -0.14, 0.22])
CORE_R = 0.30
RATE = 524.0                  # muons / s through a 1 m radius sphere (1 /cm2/min x pi m2)
EXPO0 = 2537.0                # exposure (s) at t = 0
MRAD = 350.0                  # drawn kink angle (rad) -> displayed scattering angle (mrad)

_DIST = math.hypot(CAM_D, CAM_H)
_FOCAL = R_PX / math.tan(math.asin(1.0 / _DIST))
FOV = math.degrees(2 * math.atan((DESIGN_H / 2) / _FOCAL))
PX_PER_M = _FOCAL / _DIST


def _mwe(v):
    return 1.2 + 6.8 * np.asarray(v)


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------

def _radius_field(rng, n=18, amp=0.026):
    """Smooth random radius over the unit sphere: rho(u) = 1 + sum of plane waves (rms ~ amp)."""
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    fr = rng.uniform(3.5, 11.0, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    a = rng.uniform(0.4, 1.0, n) / fr ** 0.5
    a *= amp / math.sqrt((a ** 2).sum() / 2)

    def rho(u):
        return 1.0 + (a * np.sin((u @ d.T) * fr + ph)).sum(-1)

    return rho


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _perp(d):
    a = np.where(np.abs(d[:, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    e1 = _unit(np.cross(d, a))
    return e1, np.cross(d, e1)


def _fib(n, rng, jitter=0.3):
    k = np.arange(n) + 0.5
    y = 1 - 2 * k / n
    r = np.sqrt(1 - y * y)
    th = math.pi * (3 - math.sqrt(5)) * k
    u = np.stack([r * np.cos(th), y, r * np.sin(th)], 1)
    u += rng.normal(0, jitter * math.sqrt(4 * math.pi / n), u.shape)
    return _unit(u)


def _surf(u, rho):
    return u * rho(u)[..., None]


def _normals(u, rho, e=2e-3):
    t1, t2 = _perp(u)
    p = _surf(u, rho)
    n = _unit(np.cross(_surf(_unit(u + e * t1), rho) - p, _surf(_unit(u + e * t2), rho) - p))
    return n * np.sign((n * u).sum(1))[:, None]


def _net(P, rng, k=3, extra=0.3, reach=14):
    """Surface net: every vertex to its k nearest, plus a few longer links for the tangle."""
    P = P.astype(np.float32)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :reach]
    i = np.repeat(np.arange(len(P)), k)
    j = nn[:, :k].ravel()
    i2 = np.nonzero(rng.random(len(P)) < extra)[0]
    j2 = nn[i2, rng.integers(k, reach, len(i2))]
    I, J = np.r_[i, i2], np.r_[j, j2]
    key = np.unique(np.minimum(I, J) * len(P) + np.maximum(I, J))
    return key // len(P), key % len(P)


def _march(O, D, rho, s0, s1, n):
    """Signed distance to the surface (positive inside) sampled along rays O + s D."""
    s = np.linspace(s0, s1, n)
    P = O[:, None, :] + s[None, :, None] * D[:, None, :]
    r = np.linalg.norm(P, axis=-1)
    return s, rho(P / np.maximum(r, 1e-9)[..., None]) - r


def _crossings(s, g):
    """First entry / last exit of each ray, linearly interpolated on the sign change."""
    ins = g > 0
    n = g.shape[1]
    rows = np.arange(len(g))
    i0 = np.argmax(ins, 1)
    i1 = n - 1 - np.argmax(ins[:, ::-1], 1)
    j0 = np.maximum(i0 - 1, 0)
    j1 = np.minimum(i1 + 1, n - 1)
    ga, gb = g[rows, j0], g[rows, i0]
    s_in = np.where(i0 > 0, s[j0] + (s[i0] - s[j0]) * ga / np.where(ga != gb, ga - gb, 1.0), s[i0])
    ga, gb = g[rows, i1], g[rows, j1]
    s_out = np.where(i1 < n - 1, s[i1] + (s[j1] - s[i1]) * ga / np.where(ga != gb, ga - gb, 1.0), s[i1])
    return s_in, s_out, ins.any(1)


def _slice(rho, y, n=240):
    """Contour of the body cut by the horizontal plane at height y."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    uy = np.full(n, y)
    for _ in range(5):
        uy = np.clip(uy, -0.999, 0.999)
        r = np.sqrt(1 - uy ** 2)
        u = np.stack([r * np.cos(a), uy, r * np.sin(a)], 1)
        uy = y / rho(u)
    return _surf(u, rho)


def _kink(d, theta, psi):
    e1, e2 = _perp(d)
    side = e1 * np.cos(psi)[:, None] + e2 * np.sin(psi)[:, None]
    return d * np.cos(theta)[:, None] + side * np.sin(theta)[:, None]


# ----------------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------------

class Sphere:
    name = "sphere"

    def __init__(self, T=30.0, seed=41):
        self.T = T
        rng = np.random.default_rng(seed)
        self.rho = _radius_field(rng)
        self._build_surface(rng)
        self._build_tracks(rng)
        self._build_stars(rng)
        self._build_heroes(rng)
        self.slice_y = float(CORE[1])
        self.slice = _slice(self.rho, self.slice_y).astype(np.float32)
        self.ring_seed = int(rng.integers(1 << 30))
        self.ring_jit = hash01(np.arange(N_BINS), 77) - 0.5

    # ------------------------------------------------------------------ build
    def _build_surface(self, rng, n=1500):
        u = _fib(n, rng)
        self.sp = _surf(u, self.rho).astype(np.float32)
        self.sn = _normals(u, self.rho).astype(np.float32)
        self.ea, self.eb = _net(self.sp, rng)

    def _build_tracks(self, rng, n=360, n_core=36):
        N = n + n_core
        th = np.r_[np.minimum(np.abs(rng.normal(0, 0.95, n)), 1.5), np.arccos(rng.uniform(-1, 1, n_core))]
        ph = rng.uniform(0, 2 * np.pi, N)
        d = np.stack([np.sin(th) * np.cos(ph), -np.cos(th), np.sin(th) * np.sin(ph)], 1)
        e1, e2 = _perp(d)
        r = np.r_[1.12 * np.sqrt(rng.random(n)), 1.1 * CORE_R * np.sqrt(rng.random(n_core))]
        a = rng.uniform(0, 2 * np.pi, N)
        b = np.zeros((N, 3))
        b[n:] = CORE                        # the last n_core are aimed at the core
        b += e1 * (r * np.cos(a))[:, None] + e2 * (r * np.sin(a))[:, None]
        s, g = _march(b, d, self.rho, -1.8, 1.8, 480)
        s_in, s_out, hit = _crossings(s, g)
        keep = hit & (s_out - s_in > 0.08)
        b, d, s_in, s_out = b[keep], d[keep], s_in[keep], s_out[keep]
        N = len(b)
        s_c = ((CORE[None] - b) * d).sum(1)
        dist = np.linalg.norm(CORE[None] - (b + s_c[:, None] * d), axis=1)
        through = (dist < CORE_R) & (s_c > s_in) & (s_c < s_out)
        s_k = np.where(through, s_c, s_in + (s_out - s_in) * rng.uniform(0.15, 0.85, N))
        K = b + s_k[:, None] * d
        theta = np.where(through, np.abs(rng.normal(0, 0.1, N)) * (0.5 + (1 - dist / CORE_R)) + 0.02,
                         np.abs(rng.normal(0, 0.006, N)))
        d2 = _kink(d, theta, rng.uniform(0, 2 * np.pi, N))
        s2, g2 = _march(K, d2, self.rho, 0.0, 2.4, 480)
        _, s_ex, _ = _crossings(s2, g2)
        self.tA = (b + s_in[:, None] * d).astype(np.float32)
        self.tK = K.astype(np.float32)
        self.tB = (K + s_ex[:, None] * d2).astype(np.float32)
        self.through = through
        self.mrad = theta * MRAD
        self.p = np.exp(rng.normal(1.2, 0.7, N))
        self.tt = rng.uniform(0, self.T, N)
        self.zen = np.degrees(np.arccos(np.clip(-d[:, 1], -1, 1)))
        self.azi = np.degrees(np.arctan2(d[:, 2], d[:, 0])) % 360

    def _build_stars(self, rng):
        spots = [CORE + np.array([0.06, 0.12, -0.05]), np.array([-0.46, 0.34, 0.12]),
                 np.array([-0.12, -0.52, -0.3]), np.array([0.36, 0.42, -0.36])]
        self.stars = []
        for k, v in enumerate(spots):
            n = int(rng.integers(7, 13))
            dd = _unit(rng.normal(size=(n, 3)))
            s, g = _march(np.repeat(v[None], n, 0), dd, self.rho, 0.0, 2.2, 440)
            _, L, _ = _crossings(s, g)
            heavy = rng.random(n) < 0.35
            L = np.where(heavy, L * rng.uniform(0.1, 0.4, n), L)
            self.stars.append(dict(v=v.astype(np.float32), ends=(v + dd * L[:, None]).astype(np.float32),
                                   heavy=heavy, n=n, t=k * PHRASE + BAR + 0.12, E=float(rng.uniform(0.8, 2.4))))

    def _build_heroes(self, rng):
        """One muon per phrase, on the downbeat, straight through the core (diagonal on screen)."""
        self.heroes = []
        for k in range(int(round(self.T / PHRASE))):
            t0 = k * PHRASE + 0.12
            R = self._camera(t0 + 1.0).R.astype(np.float64)
            tilt = math.radians(rng.uniform(24, 36)) * (1 if k % 2 == 0 else -1)
            d = _unit(-R[1] * math.cos(tilt) + R[0] * math.sin(tilt) + R[2] * rng.uniform(-0.25, 0.25))
            K = CORE + rng.normal(0, 0.04, 3)
            ang = float(rng.uniform(0.1, 0.15))
            d2 = _kink(d[None], np.array([ang]), np.array([rng.uniform(0, 2 * np.pi)]))[0]
            s, g = _march(np.stack([K, K]), np.stack([-d, d2]), self.rho, 0.0, 2.4, 480)
            _, L, _ = _crossings(s, g)
            path = np.stack([K - d * 6.0, K - d * L[0], K, K + d2 * L[1], K + d2 * 6.0]).astype(np.float32)
            self.heroes.append(dict(t=t0, path=path, K=K, p=float(rng.uniform(2.5, 6.5)), mrad=ang * MRAD,
                                    charge="+" if k % 2 == 0 else "-"))

    def _camera(self, t):
        yaw = 2 * math.pi * t / self.T + 0.7
        return Camera((CAM_D * math.sin(yaw), CAM_H, CAM_D * math.cos(yaw)), (0.0, 0.0, 0.0), fov_deg=FOV,
                      screen_center=C)

    # ----------------------------------------------------------------- render
    def render(self, t, W=3000, H=1688):
        tm = timing(t, self.T)
        t = tm["t"]
        f = Frame(W, H)
        cam = self._camera(t)
        phi, v, phc = self._ring_values(cam)
        f.set_clip(*VIEW)
        if tm["burst"]:
            barcode_burst(f, VIEW, t, self.T, seed=13)
        else:
            self._draw_field(f)
            self._draw_body(f, cam, t)
            self._draw_ring(f, phi, v)
            self._draw_callouts(f, cam, t, phi, v, phc)
        f.set_clip()
        self._draw_view_tag(f, t)
        self._draw_left(f, t)
        self._draw_right(f, phi, v)
        self._draw_strip(f, t)
        self._draw_bottom(f, t)
        return finish(f, tm["invert"])

    def _ring_values(self, cam):
        """Opacity per polar bin (screen angle, 0 = right, counter-clockwise), peaking toward the core."""
        cx, cy, _, _ = cam.project(CORE[None].astype(np.float32))
        phc = math.atan2(-(float(cy[0]) - C[1]), float(cx[0]) - C[0])
        phi = 2 * np.pi * (np.arange(N_BINS) + 0.5) / N_BINS
        base = periodic_noise(phi, 2 * np.pi, self.ring_seed, n=7, base=2)
        peak = np.exp(-(np.angle(np.exp(1j * (phi - phc))) / 0.22) ** 2)
        v = np.clip(0.34 + 0.2 * base + 0.52 * peak + 0.08 * self.ring_jit, 0.04, 1.0)
        return phi, v, phc

    # ------------------------------------------------------------------ world
    def _draw_field(self, f):
        x0, y0, x1, y1 = VIEW
        step = 48.0
        kx = np.arange(math.ceil((x0 + 16 - C[0]) / step), math.floor((x1 - 16 - C[0]) / step) + 1)
        ky = np.arange(math.ceil((y0 + 16 - C[1]) / step), math.floor((y1 - 16 - C[1]) / step) + 1)
        KX, KY = np.meshgrid(kx, ky)
        X, Y = C[0] + KX * step, C[1] + KY * step
        far = np.hypot(X - C[0], Y - C[1]) > RING_OUT + 30
        major = (KX % 4 == 0) & (KY % 4 == 0)
        f.pixels("w", X[far], Y[far], np.where(major, 0.9, 0.42)[far])
        m = far & major
        f.crosses("w", X[m], Y[m], 4.0, 0.4)
        # horizontal axis through the centre, in metres at the centre depth
        y = C[1]
        va, vb = (x0 - C[0]) / PX_PER_M, (x1 - C[0]) / PX_PER_M
        gap = (RING_OUT + 60) / PX_PER_M
        for lo, hi, xa, xb in ((va, -gap, x0 + 12, C[0] - RING_OUT - 60), (gap, vb, C[0] + RING_OUT + 60, x1 - 12)):
            f.segments("w", [xa], [y], [xb], [y], 0.32)
            ruler(f, x0, x1, y, va, vb, 0.1, 0.5, fmt=lambda v: f"{v:+.1f} M", inten=0.55, size=11, lo=lo, hi=hi)

    def _draw_body(self, f, cam, t):
        # surface net, rim lit: bright where the surface turns away, dimmer on the far side
        sx, sy, _, _ = cam.project(self.sp)
        view = _unit(cam.pos.astype(np.float32)[None] - self.sp)
        nv = (self.sn * view).sum(1)
        rim = (0.08 + 0.75 * (1 - np.abs(nv)) ** 2.2) * np.where(nv > 0, 1.0, 0.45)
        ea, eb = self.ea, self.eb
        f.segments("w", sx[ea], sy[ea], sx[eb], sy[eb], rim[ea], rim[eb])
        f.pixels("w", sx, sy, 0.25 + 0.6 * rim)
        # tracks: entry -> PoCA -> exit, the fresh ones brighter
        ax, ay, _, _ = cam.project(self.tA)
        kx, ky, _, _ = cam.project(self.tK)
        bx, by, _, _ = cam.project(self.tB)
        age = (t - self.tt) % self.T
        inten = np.where(self.through, 0.24, 0.2) * (1.0 + 2.5 * np.exp(-age / 0.35))
        f.segments("w", np.r_[ax, kx], np.r_[ay, ky], np.r_[kx, bx], np.r_[ky, by], np.r_[inten, inten])
        m = self.through
        f.dots("w", kx[m], ky[m], 1.6, 1.1)
        hot = m & (self.mrad > 45)
        f.dots("r", kx[hot], ky[hot], 2.4, 1.5)
        # nuclear stars
        for s in self.stars:
            vx, vy, _, _ = cam.project(s["v"][None])
            ex, ey, _, _ = cam.project(s["ends"])
            n = s["n"]
            fresh = math.exp(-((t - s["t"]) % self.T) / 0.6)
            w = np.where(s["heavy"], 1.6, 1.0)
            i = np.where(s["heavy"], 0.9, 0.55)
            f.segments("w", np.repeat(vx, n), np.repeat(vy, n), ex, ey, i, i * 0.35, width=w)
            f.dots("w", vx, vy, 2.6, 1.5)
            if fresh > 0.02:
                f.segments("r", np.repeat(vx, n), np.repeat(vy, n), ex, ey, 1.4 * fresh, 0.5 * fresh, width=w)
                f.rings("r", vx, vy, [12 + 70 * (1 - fresh)], fresh, width=1.2)
        # the core, as the scattering gives it away
        cx, cy, cz, _ = cam.project(CORE[None].astype(np.float32))
        rc = CORE_R * cam.focal / float(cz[0])
        f.rings("r", cx, cy, [rc], 0.45)
        f.crosses("r", cx, cy, 9.0, 0.9)
        # horizontal slice through the core: front half bright, back half faint
        lx, ly, lz, _ = cam.project(self.slice)
        front = lz < np.median(lz)
        li = np.where(front, 0.8, 0.2)
        f.segments("w", lx, ly, np.roll(lx, -1), np.roll(ly, -1), li, np.roll(li, -1), width=1.2)
        # centre mark
        f.crosses("w", [C[0]], [C[1]], 7.0, 0.8)

    def _draw_ring(self, f, phi, v):
        cx, cy = C
        L = RING_L[0] + (RING_L[1] - RING_L[0]) * v
        c, s = np.cos(phi), -np.sin(phi)
        x0, y0 = cx + RING_R0 * c, cy + RING_R0 * s
        x1, y1 = cx + (RING_R0 + L) * c, cy + (RING_R0 + L) * s
        hot = v > HOT
        f.segments("w", x0[~hot], y0[~hot], x1[~hot], y1[~hot], 0.75)
        f.segments("r", x0[hot], y0[hot], x1[hot], y1[hot], 1.1, width=1.3)
        f.dots("w", x1[~hot], y1[~hot], 3.0, 1.3)
        f.dots("r", x1[hot], y1[hot], 3.6, 1.8)
        f.rings("w", [cx], [cy], [RING_R0], 0.55)
        # dotted scale circles at 2 / 4.6 / 8 MWE
        for frac, inten in ((0.0, 0.3), (0.5, 0.35), (1.0, 0.6)):
            r = RING_R0 + RING_L[0] + (RING_L[1] - RING_L[0]) * frac
            a = np.linspace(0, 2 * np.pi, int(2 * np.pi * r / 6), endpoint=False)
            f.pixels("w", cx + r * np.cos(a), cy + r * np.sin(a), inten)
        # inward ticks every 10 deg, long ones every 45 deg
        a = np.radians(np.arange(0, 360, 10))
        ln = np.where(np.arange(36) % 9 == 0, 14.0, np.where(np.arange(36) % 3 == 0, 8.0, 4.0))
        f.segments("w", cx + RING_R0 * np.cos(a), cy - RING_R0 * np.sin(a), cx + (RING_R0 - ln) * np.cos(a),
                   cy - (RING_R0 - ln) * np.sin(a), 0.8)
        # angle labels on the axis, value at the peak
        for deg, anchor in ((0, "lm"), (180, "rm")):
            a = math.radians(deg)
            f.text("w", cx + (RING_OUT + 22) * math.cos(a), cy + 5, f"{deg:03d}", size=12, alpha=0.8, anchor=anchor)
        k = int(np.argmax(v))
        r = RING_R0 + L[k] + 16
        f.text("r", cx + r * c[k], cy + r * s[k], f"{_mwe(v[k]):.2f}", size=13, alpha=0.95,
               anchor="lm" if c[k] >= 0 else "rm")

    def _callout(self, f, x, y, ang, title, lines, red=False, alpha=1.0, age=9.0, r_out=RING_OUT + 40):
        """Leader from a point in the body out past the ring, then a data tag and typed lines."""
        side = 1 if math.cos(ang) >= 0 else -1
        a = max(-0.8, min(0.8, math.atan2(math.sin(ang), abs(math.cos(ang)))))
        ex, ey = C[0] + side * r_out * math.cos(a), C[1] - r_out * math.sin(a)
        hx = ex + side * 36
        f.segments("w", [x, ex], [y, ey], [ex, hx], [ey, ey], 0.75 * alpha)
        f.dots("w", [x], [y], 2.2, 1.3 * alpha)
        anchor = "ls" if side > 0 else "rs"
        tx = hx + side * 10
        f.tag("r" if red else "w", tx, ey + 7, title, size=18, alpha=alpha, pad=4, anchor=anchor)
        if lines:
            wl = max(len(ln) for ln in lines) * 0.6 * 14
            bx = tx if side > 0 else tx - wl
            f.occlude(bx - 5, ey + 18, bx + wl + 5, ey + 40 + (len(lines) - 1) * 19)
        for k, ln in enumerate(lines):
            n = int(min(len(ln), max(0, (age - 0.1 - 0.08 * k) * 100)))
            f.text("w", tx, ey + 34 + k * 19, ln[:n], size=14, alpha=0.9 * alpha, anchor=anchor)

    def _draw_callouts(self, f, cam, t, phi, v, phc):
        # anomaly: through the peak of the opacity ring
        cx, cy, _, _ = cam.project(CORE[None].astype(np.float32))
        n_sc = int(self.through.sum())
        self._callout(f, float(cx[0]), float(cy[0]), phc, "ANOMALY",
                      ["RHO 11.3 G/CM3   Z~82",
                       f"R {CORE_R:.2f} M   DEPTH {1 - np.linalg.norm(CORE):.2f} M",
                       f"POCA {n_sc:3d}   LAMBDA 9.8 MRAD2/CM"], red=True)
        # the hero muon of this phrase
        h = self.heroes[int(t // PHRASE) % len(self.heroes)]
        age = (t - h["t"]) % self.T
        px, py, _, ok = cam.project(h["path"])
        if not ok.all():
            return
        fade = 0.5 + 0.5 * math.exp(-age / 1.6)
        prog = min(1.0, age / 0.4)
        seg = np.hypot(np.diff(px), np.diff(py))
        cum = np.r_[0, np.cumsum(seg)]
        reach = prog * cum[-1]
        inside = np.array([False, True, True, False])
        xe, ye = px[0], py[0]
        for k in range(4):
            if cum[k] >= reach:
                break
            u = min(1.0, (reach - cum[k]) / max(seg[k], 1e-6))
            xe, ye = px[k] + (px[k + 1] - px[k]) * u, py[k] + (py[k + 1] - py[k]) * u
            f.segments("r", [px[k]], [py[k]], [xe], [ye], (1.3 if inside[k] else 0.55) * fade,
                       width=2.0 if inside[k] else 1.2)
        for k in (0, 3):                    # detector hits along the track outside the body
            L = seg[k]
            if L < 1:
                continue
            ux, uy = (px[k + 1] - px[k]) / L, (py[k + 1] - py[k]) / L
            d = np.arange(18.0, L - 10, 34.0)
            d = d[cum[k] + d <= reach]
            hx0, hy0 = px[k] + ux * d, py[k] + uy * d
            f.segments("r", hx0 - uy * 6, hy0 + ux * 6, hx0 + uy * 6, hy0 - ux * 6, 0.8 * fade)
        if prog < 1.0:
            hx, hy = xe, ye
            f.dots("r", [hx], [hy], 3.4, 1.8)
            f.dots("w", [hx], [hy], 1.3, 1.0)
        if age > 0.15:
            f.rings("r", px[[1, 3]], py[[1, 3]], [8.0, 8.0], 0.9 * fade)
            f.rings("r", px[[2]], py[[2]], [16.0], 1.0 * fade, width=1.2)
        if age < 4.5:
            alpha = 1.0 - max(0.0, (age - 3.8) / 0.7)
            ang = math.atan2(-(py[1] - C[1]), px[1] - C[0])
            K = h["K"]
            self._callout(f, float(px[1]), float(py[1]), ang, f"MU{h['charge']}",
                          [f"P {h['p']:.3f} GEV/C",
                           f"THETA_SC {h['mrad']:.1f} MRAD",
                           f"POCA {K[0]:+.3f} {K[1]:+.3f} {K[2]:+.3f}"], red=True, alpha=alpha, age=age)
        # the star of this phrase
        s = self.stars[int(t // PHRASE) % len(self.stars)]
        sa = (t - s["t"]) % self.T
        if sa < 3.0:
            vx, vy, _, _ = cam.project(s["v"][None])
            ang = math.atan2(-(float(vy[0]) - C[1]), float(vx[0]) - C[0]) + math.pi
            self._callout(f, float(vx[0]), float(vy[0]), ang, f"STAR N={s['n']}",
                          [f"E {s['E']:.3f} GEV   BLACK {int(s['heavy'].sum())}"],
                          alpha=1.0 - max(0.0, (sa - 2.4) / 0.6), age=sa)

    # ------------------------------------------------------------------ HUD
    def _draw_view_tag(self, f, t):
        x0, y0 = VIEW[0] + 20, MAIN[1] + 36
        f.tag("w", x0, y0, "VIEW 01 // SCATTERING TOMOGRAPHY", size=16, pad=4)
        yaw = (math.degrees(2 * math.pi * t / self.T + 0.7)) % 360
        f.text("w", x0, y0 + 30, f"ORBIT {yaw:05.1f} DEG   ELEV {math.degrees(math.atan2(CAM_H, CAM_D)):.1f} DEG",
               size=13, alpha=0.75)
        f.text("w", VIEW[2] - 20, y0, f"SPHERE_01 // MUON BLOOM // R 1.000 M // {N_BINS} BINS", size=14,
               alpha=0.85, anchor="rs")
        # slice label on the right edge of the body
        f.text("w", VIEW[2] - 20, y0 + 26, f"SLICE Y {self.slice_y:+.3f} M   CORE {CORE[0]:+.2f} {CORE[1]:+.2f} "
               f"{CORE[2]:+.2f}", size=12, alpha=0.65, anchor="rs")

    def _draw_left(self, f, t):
        x0, y0, x1, y1 = LEFT_COL
        f.tag("w", x0 + 8, y0 + 64, "MUOGRAPHY", size=50, pad=8)
        expo = EXPO0 + t
        hh, mm, ss = int(expo // 3600), int(expo % 3600 // 60), int(expo % 60)
        rows = ["TARGET    SPHERE_01", "RADIUS    1.000 M", f"RATE      ~{RATE:.0f} MU/S",
                f"EXPOSURE  {hh:02d}:{mm:02d}:{ss:02d}", f"TRACKS    {int(RATE * expo):,}",
                "RHO_BODY  2.65 G/CM3", "RHO_CORE  11.3 G/CM3"]
        for k, r in enumerate(rows):
            f.text("r" if k == 6 else "w", x0 + 8, y0 + 120 + k * 22, r, size=14, alpha=0.85)
        f.tag("w", x0 + 8, y0 + 300, "TRACK_LOG // POCA", size=12, pad=3)
        f.text("w", x0 + 8, y0 + 330, "ID   ZEN   AZI   MRAD  X     Y     Z", size=12, alpha=0.55)
        age = (t - self.tt) % self.T
        for row, k in enumerate(np.argsort(age)[:30]):
            K = self.tK[k]
            line = (f"{k:04d} {self.zen[k]:4.1f} {self.azi[k]:5.1f} {self.mrad[k]:6.1f} "
                    f"{K[0]:+.2f} {K[1]:+.2f} {K[2]:+.2f}")
            red = self.through[k] and self.mrad[k] > 30
            f.text("r" if red else "w", x0 + 8, y0 + 352 + row * 19, line, size=12,
                   alpha=0.95 if row < 3 or red else 0.62)

    def _draw_right(self, f, phi, v):
        x0, y0, x1, y1 = RIGHT_COL
        f.tag("w", x0 + 4, y0 + 36, f"OPACITY_PROFILE // MWE // {N_BINS} BINS", size=12, pad=3)
        top, bot = y0 + 70, y1 - 34
        ys = top + np.arange(N_BINS) * (bot - top) / N_BINS
        xl = x0 + 64
        wmax = x1 - xl - 8
        w = v * wmax
        hot = v > HOT
        f.rects("w", xl, ys[~hot], xl + w[~hot], ys[~hot] + 2, 0.9)
        f.rects("r", xl, ys[hot], xl + w[hot], ys[hot] + 2, 1.0)
        f.segments("w", [xl - 6], [top - 6], [xl - 6], [bot + 4], 0.6)
        for k in range(0, N_BINS, N_BINS // 8):
            f.segments("w", [xl - 12], [ys[k] + 1], [xl - 6], [ys[k] + 1], 0.8)
            f.text("w", x0 + 4, ys[k] + 6, f"{k * 360 // N_BINS:03d}", size=11, alpha=0.7)
        xt = xl + HOT * wmax
        f.segments("r", [xt], [top - 12], [xt], [bot + 6], 0.55)
        f.text("r", xt - 4, bot + 24, f"FLAG > {_mwe(HOT):.2f} MWE", size=12, alpha=0.9, anchor="rs")
        for frac in (0.0, 0.5, 1.0):
            f.text("w", xl + frac * wmax, top - 14, f"{_mwe(frac):.1f}", size=11, alpha=0.6,
                   anchor="ls" if frac < 1 else "rs")

    def _draw_strip(self, f, t):
        x0, y0, x1, y1 = STRIP
        ta, tb = t - 5.5, t + 2.0
        X = lambda tt: x0 + (np.asarray(tt) - ta) / (tb - ta) * (x1 - x0)
        f.segments("w", [x0, x0], [y0, y1], [x1, x1], [y0, y1], 0.9)
        ruler(f, x0, x1, y0, ta, tb, BEAT / 4, BAR, down=True)
        ruler(f, x0, x1, y1, ta, tb, BEAT / 4, BAR, down=False)
        yb = (y0 + y1) / 2 - 6
        f.rects("r", x0, yb - 3, x1, yb + 3, 1.0)
        tt = np.r_[self.tt - self.T, self.tt, self.tt + self.T]
        idx = np.tile(np.arange(len(self.tt)), 3)
        m = (tt >= ta - 0.1) & (tt <= tb + 0.1)
        k = idx[m]
        xe = X(tt[m])
        f.rects("w", xe, y0 + 1, xe + 2, y0 + 1 + 8 + 14 * np.log1p(self.p[k]), 0.95)
        f.rects("r", xe, y1 - np.minimum(52.0, 3 + 0.6 * self.mrad[k]), xe + 2, y1 - 1, 0.9)
        events = [(h["t"], f"MU{h['charge']} // CORE", True) for h in self.heroes]
        events += [(s["t"], f"STAR N={s['n']}", False) for s in self.stars]
        for te, label, red in events:
            for tv in (te - self.T, te, te + self.T):
                if ta - 0.3 <= tv <= tb + 0.3:
                    past = tv <= t
                    f.tag("r" if red and past else "w", float(X(tv)), yb + 6, label, size=14, pad=4,
                          alpha=1.0 if past else 0.5)
        xc = float(X(t))
        f.segments("r", [xc], [y0 - 4], [xc], [y1 + 4], 1.2, width=1.6)
        f.tag("r", xc + 6, y1 + 22, f"T {t:06.3f}", size=13, pad=3)
        f.tag("w", x0 + 4, y0 - 10, "TRACK_TIMELINE // WHITE = MOMENTUM  RED = SCATTERING ANGLE", size=12, pad=3)

    def _draw_bottom(self, f, t):
        x0, y0, x1, y1 = BOT
        expo = EXPO0 + t
        n_all = int(RATE * expo)
        cx0, cx1 = x0, 1180.0
        panel_header(f, cx0, cx1, y0, "RECONSTRUCTED")
        cols = [("TRACKS", f"{n_all:,}"), ("SCATTERED", f"{int(n_all * 0.0191):,}"),
                ("STARS", f"{int(expo * 0.37):,}"), ("RHO_CORE", "11.3")]
        cw = (cx1 - cx0) / 4
        for k, (lab, val) in enumerate(cols):
            xx = cx0 + k * cw
            f.text("w", xx + 4, y0 + 34, lab, size=13, alpha=0.75)
            f.text("r" if k == 3 else "w", xx + 2, y0 + 96, val, size=44 if k == 0 else 48, alpha=0.97)
        # scatter barcode: one column per 15 ms, lit by the scattering of the tracks inside it
        bx0, bx1 = 1220.0, 2000.0
        panel_header(f, bx0, bx1, y0, "SCATTER_BARCODE")
        cols_n = 200
        dt = 3.0 / cols_n
        kk = math.floor((t - 3.0) / dt) + np.arange(cols_n)
        tt = (kk * dt) % self.T
        order = np.argsort(self.tt)
        ts, ms = self.tt[order], self.mrad[order]
        lo, hi = np.searchsorted(ts, tt), np.searchsorted(ts, tt + dt * 3)
        csum = np.r_[0, np.cumsum(ms)]
        dens = 0.06 + 0.9 * np.tanh((hi - lo) / 2.5 + (csum[hi] - csum[lo]) / 40.0)
        barcode_lanes(f, bx0, bx1, y0 + 10, y1, dens, kk, lanes=3, seed=5)
        # scattering-angle spectrum
        sx0 = 2040.0
        panel_header(f, sx0, x1, y0, "THETA_SCATTER // MRAD")
        edges = np.linspace(0, 90, 73)
        cnt, _ = np.histogram(self.mrad, edges)
        hgt = 88 * np.sqrt(cnt / max(cnt.max(), 1))
        bw = (x1 - sx0 - 20) / 72
        xs = sx0 + 10 + np.arange(72) * bw
        tail = edges[:-1] >= 30
        yb = y1 - 14
        f.rects("w", xs[~tail], yb - hgt[~tail], xs[~tail] + bw - 3, yb, 0.95)
        f.rects("r", xs[tail], yb - hgt[tail], xs[tail] + bw - 3, yb, 0.95)
        ruler(f, sx0 + 10, sx0 + 10 + 72 * bw, yb + 2, 0, 90, 5, 10, fmt=lambda v: f"{v:.0f}", inten=0.6, size=11,
              lab_dy=24)
