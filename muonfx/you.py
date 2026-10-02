"""YOU - muons pass through you. Ryoji Ikeda edition.

About one muon per square centimetre per minute reaches sea level: roughly
17 go through a standing person every second, ~42 billion in a lifetime.

A 1.80 m body, built from lofted ellipse sections, drawn as a rim-lit point
cloud with CT-like slice contours over a dot-lattice floor. Cosmic muons
(red) rain through it on a 16th-note grid; each crossing is hit-tested
against the body: the part lights up, the slice at that height turns red,
the energy deposit is logged. Every bar downbeat sends a "hero" muon through
a named part (heart, head, hands...) with a data tag.

One 4-bar phrase (128 BPM), hard cuts on every bar:
  bar 1  PERSPECTIVE      orbit around the figure
  bar 2  ORTHO_FRONT/SIDE two orthographic panels, a red scan slice sweeping
  bar 3  THORAX           close-up, the hero muon crosses the heart
  bar 4  ORTHO_TOP        seen from above: slices stack into a target
HUD: hit timeline strip, YOU column with hit log, per-time-scale counts,
hit barcode, energy deposit per body part. Loops seamlessly every 30 s.
"""
from __future__ import annotations

import math

import numpy as np

from .engine import Camera, Frame, OrthoCamera, hash01, smoothstep
from .hud import (BAR, BEAT, BOT, MAIN, PHRASE, STRIP, barcode_burst, barcode_lanes, finish, panel_header,
                  ruler, timing)

MAIN_C = ((MAIN[0] + MAIN[2]) / 2 + 180, (MAIN[1] + MAIN[3]) / 2)
LEFT_COL = (40.0, 505.0, 372.0, 1447.0)
VIEW_X0 = 400.0

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
for name, ys, cx, cz, rx, rz in _L:
    LOFTS.append((name, *(np.asarray(v, np.float64) for v in (ys, cx, cz, rx, rz))))
    if name.endswith("_L"):
        LOFTS.append((name[:-2] + "_R", np.asarray(ys), -np.asarray(cx), np.asarray(cz), np.asarray(rx),
                      np.asarray(rz)))
PART_NAMES = [l[0] for l in LOFTS]
HEART = np.array([0.045, 1.27, 0.04])
HERO_PARTS = ["HEART", "HEAD", "HAND_L", "TORSO", "LEG_R", "HAND_R", "NECK", "FOOT_L",
              "HEART", "ARM_R", "HEAD", "TORSO", "LEG_L", "HAND_L", "HEART", "FOOT_R"]
DEDX = 0.2      # GeV per metre of tissue (~2 MeV/cm)


def _loft_at(l, y):
    name, ys, cx, cz, rx, rz = l
    return (np.interp(y, ys, cx), np.interp(y, ys, cz), np.interp(y, ys, rx), np.interp(y, ys, rz))


class You:
    name = "you"

    def __init__(self, T=30.0, seed=33):
        self.T = T
        rng = np.random.default_rng(seed)
        self._build_body(rng)
        self._build_muons(rng)
        self._build_static()

    # ------------------------------------------------------------------ body
    def _build_body(self, rng):
        P, Nn, part = [], [], []
        for li, l in enumerate(LOFTS):
            ys = l[1]
            y = np.arange(ys[0], ys[-1], 0.004)
            cx, cz, rx, rz = _loft_at(l, y)
            per = math.pi * (3 * (rx + rz) - np.sqrt((3 * rx + rz) * (rx + 3 * rz)))
            cnt = rng.poisson(per * 0.004 * 16000)
            yy = np.repeat(y, cnt) + rng.uniform(0, 0.004, cnt.sum())
            th = rng.uniform(0, 2 * np.pi, cnt.sum())
            cx, cz, rx, rz = _loft_at(l, yy)
            x = cx + rx * np.cos(th)
            z = cz + rz * np.sin(th)
            n = np.stack([np.cos(th) / rx, np.zeros_like(th), np.sin(th) / rz], 1)
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            P.append(np.stack([x, yy, z], 1))
            Nn.append(n)
            part.append(np.full(len(x), li, np.int16))
        self.pts = np.concatenate(P).astype(np.float32)
        self.nrm = np.concatenate(Nn).astype(np.float32)
        self.pts_part = np.concatenate(part)
        # CT-like slice contours every 3 cm
        self.levels = np.arange(0.015, 1.8, 0.03)
        segs_a, segs_b, lev = [], [], []
        ang = np.linspace(0, 2 * np.pi, 41)
        for k, y in enumerate(self.levels):
            for l in LOFTS:
                if l[1][0] <= y <= l[1][-1]:
                    cx, cz, rx, rz = _loft_at(l, y)
                    e = np.stack([cx + rx * np.cos(ang), np.full_like(ang, y), cz + rz * np.sin(ang)], 1)
                    segs_a.append(e[:-1]); segs_b.append(e[1:]); lev.append(np.full(40, k, np.int32))
        self.ca = np.concatenate(segs_a).astype(np.float32)
        self.cb = np.concatenate(segs_b).astype(np.float32)
        self.clev = np.concatenate(lev)

    def _inside(self, p):
        """Which loft contains each point (N,3) -> index or -1."""
        out = np.full(len(p), -1, np.int32)
        for li, l in enumerate(LOFTS):
            ys = l[1]
            m = (p[:, 1] >= ys[0]) & (p[:, 1] <= ys[-1]) & (out < 0)
            if not m.any():
                continue
            cx, cz, rx, rz = _loft_at(l, p[m, 1])
            ins = ((p[m, 0] - cx) / rx) ** 2 + ((p[m, 2] - cz) / rz) ** 2 <= 1.0
            idx = np.nonzero(m)[0][ins]
            out[idx] = li
        return out

    def _target_in(self, name, rng):
        if name == "HEART":
            return HEART + rng.normal(0, 0.012, 3)
        l = LOFTS[PART_NAMES.index(name)]
        ys = l[1]
        y = rng.uniform(ys[0] + 0.25 * (ys[-1] - ys[0]), ys[-1] - 0.25 * (ys[-1] - ys[0]))
        cx, cz, rx, rz = _loft_at(l, y)
        a = rng.uniform(0, 2 * np.pi)
        rr = rng.uniform(0, 0.55)
        return np.array([cx + rx * rr * math.cos(a), y, cz + rz * rr * math.sin(a)])

    # ----------------------------------------------------------------- muons
    def _build_muons(self, rng):
        T = self.T
        n16 = int(round(T / (BEAT / 4)))
        mus = []
        n_bars = int(round(T / BAR))
        for b in range(n_bars):            # hero muons on the downbeats
            name = HERO_PARTS[b % len(HERO_PARTS)]
            if b % 4 == 2:
                name = "HEART"
            mus.append(self._muon(rng, b * BAR + 0.12, self._target_in(name, rng), hero=True))
        for k in range(n16):
            if k % 16 == 0 or rng.random() > 0.42:
                continue
            if rng.random() < 0.55:
                name = PART_NAMES[rng.integers(0, len(PART_NAMES))]
                tgt = self._target_in(name, rng)
            else:
                a, r = rng.uniform(0, 2 * np.pi), 0.9 * math.sqrt(rng.random())
                tgt = np.array([r * math.cos(a), rng.uniform(0.3, 1.6), r * math.sin(a)])
            mus.append(self._muon(rng, k * BEAT / 4, tgt, hero=False))
        self.mus = mus
        self.mu_t = np.array([m["t"] for m in mus], np.float32)

    def _muon(self, rng, t, tgt, hero):
        th = min(1.0, abs(rng.normal(0, 0.38)))
        ph = rng.uniform(0, 2 * np.pi)
        d = np.array([math.sin(th) * math.cos(ph), -math.cos(th), math.sin(th) * math.sin(ph)])
        s_top = (2.9 - tgt[1]) / math.cos(th)
        s_bot = tgt[1] / math.cos(th)
        a = tgt - d * s_top
        b = tgt + d * s_bot
        L = s_top + s_bot
        u = np.linspace(0, 1, 700)
        pts = a[None] + (b - a)[None] * u[:, None]
        ins = self._inside(pts)
        inside = ins >= 0
        entry = int(np.argmax(inside)) if inside.any() else -1
        part, hit, dE, u_in = None, None, 0.0, None
        if entry >= 0:
            dE = inside.sum() * (L / 700) * DEDX
            if hero:        # tag the organ it was sent through, where it crosses it
                hit = tgt
                u_in = s_top / L
                k = self._inside(tgt[None])[0]
                part = "HEART" if np.linalg.norm(tgt - HEART) < 0.05 else (PART_NAMES[k] if k >= 0 else "TORSO")
            else:
                hit = pts[entry]
                part = PART_NAMES[ins[entry]]
                if np.linalg.norm(hit - HEART) < 0.085:
                    part = "HEART"
                u_in = u[entry]
        # inside intervals (for brighter ionisation segments)
        edges = np.diff(np.r_[0, inside.astype(int), 0])
        starts = np.nonzero(edges == 1)[0]
        ends = np.nonzero(edges == -1)[0] - 1
        runs = list(zip(u[starts], u[ends]))
        E = float(np.exp(rng.normal(1.2, 0.7)))
        speed = 22.0
        return dict(t=t % self.T, a=a.astype(np.float32), b=b.astype(np.float32), L=L, dur=L / speed, part=part,
                    hit=hit, u_in=u_in, dE=dE, E=E, runs=runs, hero=hero, charge=rng.choice(["+", "-"]))

    def _build_static(self):
        g = np.arange(-2.0, 2.001, 0.1)
        X, Z = np.meshgrid(g, g)
        self.lattice = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1).astype(np.float32)
        self.lat_major = ((np.abs(np.round(X * 10)) % 5 == 0) & (np.abs(np.round(Z * 10)) % 5 == 0)).ravel()

    # ----------------------------------------------------------------- views
    def _views(self, tm):
        bip, u, phrase = tm["bip"], tm["u"], tm["phrase"]
        if bip == 0:
            yaw = math.radians(phrase * 90 + 20 + 38 * u)
            D = 3.7
            cam = Camera((D * math.sin(yaw), 1.25, D * math.cos(yaw)), (0.0, 0.93, 0.0), fov_deg=34.0,
                         screen_center=MAIN_C)
            return "PERSPECTIVE", [(cam, (VIEW_X0, MAIN[1], MAIN[2], MAIN[3]))]
        if bip == 1:
            h = (MAIN[1] + MAIN[3]) / 2
            w0 = (VIEW_X0 + MAIN[2]) / 2
            cf = OrthoCamera((0.0, 0.93, 10.0), (0.0, 0.93, 0.0), scale=470.0, screen_center=((VIEW_X0 + w0) / 2, h))
            cs = OrthoCamera((10.0, 0.93, 0.0), (0.0, 0.93, 0.0), scale=470.0, screen_center=((w0 + MAIN[2]) / 2, h))
            return "ORTHO_FRONT / ORTHO_SIDE", [(cf, (VIEW_X0, MAIN[1], w0 - 6, MAIN[3])),
                                               (cs, (w0 + 6, MAIN[1], MAIN[2], MAIN[3]))]
        if bip == 2:
            yaw = math.radians(phrase * 90 - 25 + 22 * u)
            D = 1.45 - 0.12 * u
            cam = Camera((D * math.sin(yaw) + 0.03, 1.36, D * math.cos(yaw)), (0.03, 1.24, 0.0), fov_deg=40.0,
                         screen_center=MAIN_C)
            return "THORAX // CLOSE_UP", [(cam, (VIEW_X0, MAIN[1], MAIN[2], MAIN[3]))]
        cam = OrthoCamera((0.0, 10.0, 1e-3), (0.0, 0.0, 0.0), scale=860.0 + 90 * u, up=(0.0, 0.0, -1.0),
                          screen_center=MAIN_C, roll_deg=phrase * 30 + 25 * u)
        return "ORTHO_TOP", [(cam, (VIEW_X0, MAIN[1], MAIN[2], MAIN[3]))]

    # ----------------------------------------------------------------- render
    def render(self, t, W=3000, H=1688):
        tm = timing(t, self.T)
        t = tm["t"]
        f = Frame(W, H)
        name, views = self._views(tm)
        ages = (t - self.mu_t) % self.T
        scan_y = None
        if tm["bip"] == 1:
            uu = tm["u"]
            scan_y = 0.03 + 1.76 * (uu if tm["phrase"] % 2 == 0 else 1 - uu)
        if tm["burst"]:
            f.set_clip(VIEW_X0, MAIN[1], MAIN[2], MAIN[3])
            barcode_burst(f, (VIEW_X0, MAIN[1], MAIN[2], MAIN[3]), t, self.T, seed=9)
        else:
            for cam, clip in views:
                f.set_clip(*clip)
                self._draw_world(f, cam, ages, scan_y, name)
            if tm["bip"] == 1:
                w0 = (VIEW_X0 + MAIN[2]) / 2
                f.set_clip()
                f.segments("w", [w0], [MAIN[1] + 10], [w0], [MAIN[3] - 10], 0.6)
        f.set_clip()
        self._draw_view_tag(f, name, views, tm)
        self._draw_left(f, t, ages)
        self._draw_strip(f, t)
        self._draw_bottom(f, t, ages)
        return finish(f, tm["invert"])

    def _draw_world(self, f, cam, ages, scan_y, vname):
        # floor lattice
        sx, sy, z, ok = cam.project(self.lattice)
        fog = np.clip(1.4 - z / 9.0, 0.3, 1.0) if not cam.ortho else 1.0
        f.pixels("w", sx[ok], sy[ok], (np.where(self.lat_major, 0.9, 0.4) * fog)[ok])
        mj = self.lat_major & ok
        f.crosses("w", sx[mj], sy[mj], 4.0, 0.45)
        # body point cloud, rim lit (brighter at the silhouette)
        px, py, pz, pok = cam.project(self.pts)
        if cam.ortho:
            view = -cam.R[2]
            facing = np.abs(self.nrm @ view)
        else:
            v = cam.pos.astype(np.float32)[None] - self.pts
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            facing = np.abs((self.nrm * v).sum(1))
        inten = 0.12 + 0.62 * (1 - facing) ** 2
        if vname == "ORTHO_TOP":
            inten = inten * 0.12
        # parts hit recently glow red
        red = np.zeros(len(self.pts), np.float32)
        for m, a in zip(self.mus, ages):
            if m["hit"] is None:
                continue
            ah = a - m["dur"] * m["u_in"]
            if 0 <= ah < 0.9:
                d = np.linalg.norm(self.pts - m["hit"][None].astype(np.float32), axis=1)
                red = np.maximum(red, np.exp(-(d / 0.07) ** 2) * math.exp(-ah / 0.3))
        f.pixels("w", px[pok], py[pok], (inten * (1 - 0.8 * red))[pok])
        rm = pok & (red > 0.05)
        if rm.any():
            f.pixels("r", px[rm], py[rm], 2.2 * red[rm])
        # slice contours
        ax, ay, az, aok = cam.project(self.ca)
        bx, by, bz, bok = cam.project(self.cb)
        okc = aok & bok
        ci = np.full(len(self.ca), 0.2 if vname != "ORTHO_TOP" else 0.075, np.float32)
        hot = np.zeros(len(self.ca), np.float32)
        for m, a in zip(self.mus, ages):
            if m["hit"] is None:
                continue
            ah = a - m["dur"] * m["u_in"]
            if 0 <= ah < 0.6:
                k = int(np.argmin(np.abs(self.levels - m["hit"][1])))
                hot = np.maximum(hot, np.where(np.abs(self.clev - k) <= 1, math.exp(-ah / 0.25), 0.0))
        if scan_y is not None:
            k = int(np.argmin(np.abs(self.levels - scan_y)))
            hot = np.maximum(hot, np.where(self.clev == k, 1.0, 0.0))
        f.segments("w", ax[okc], ay[okc], bx[okc], by[okc], (ci * (1 - hot))[okc])
        hm = okc & (hot > 0.02)
        if hm.any():
            f.segments("r", ax[hm], ay[hm], bx[hm], by[hm], 1.3 * hot[hm], width=1.3)
        if vname == "ORTHO_TOP":      # range rings around you
            o = cam.project(np.zeros((1, 3), np.float32))
            X, Y = float(o[0][0]), float(o[1][0])
            sc = cam.scale
            rr = [0.25, 0.5, 0.75, 1.0, 1.5]
            f.rings("w", [X] * len(rr), [Y] * len(rr), [sc * r for r in rr], 0.3)
            for r in rr:
                f.text("w", X + sc * r + 6, Y - 6, f"{r:.2f} M", size=12, alpha=0.7)
            big = 1e5
            f.segments("r", [X - big, X], [Y, Y - big], [X + big, X], [Y, Y + big], 0.35)
        # muons
        for m, a in zip(self.mus, ages):
            if a > m["dur"] + 0.9:
                continue
            self._draw_muon(f, cam, m, a)
        if scan_y is not None:
            p = cam.project(np.array([[0.0, scan_y, 0.0]], np.float32))
            y = float(p[1][0])
            f.segments("r", [f.clip[0] / f.s], [y], [f.clip[2] / f.s], [y], 0.8, width=1.2)
            f.tag("r", f.clip[0] / f.s + 12, y - 8, f"SLICE Y {scan_y:.3f} M", size=13, pad=3)

    def _draw_muon(self, f, cam, m, a):
        prog = min(1.0, a / m["dur"])
        fade = 1.0 if a < m["dur"] else math.exp(-(a - m["dur"]) / 0.3)
        A, B = m["a"], m["b"]
        head = A + (B - A) * prog
        P = np.stack([A, head]).astype(np.float32)
        sx, sy, z, ok = cam.project(P)
        if not ok.all():
            return
        w = 1.0 if not m["hero"] else 1.4
        f.segments("r", sx[:1], sy[:1], sx[1:], sy[1:], (0.55 if not m["hero"] else 0.8) * fade, width=w)
        for (u0, u1) in m["runs"]:
            if u0 >= prog:
                continue
            u1 = min(u1, prog)
            Q = np.stack([A + (B - A) * u0, A + (B - A) * u1]).astype(np.float32)
            qx, qy, _, _ = cam.project(Q)
            f.segments("r", qx[:1], qy[:1], qx[1:], qy[1:], 1.3 * fade, width=2.2 if m["hero"] else 1.6)
        if prog < 1.0:
            f.dots("r", sx[1:], sy[1:], 3.0, 1.6)
            f.dots("w", sx[1:], sy[1:], 1.2, 1.0)
        if m["hit"] is not None:
            ah = a - m["dur"] * m["u_in"]
            if 0 <= ah < 0.8:
                hx, hy, _, _ = cam.project(m["hit"][None].astype(np.float32))
                u = ah / 0.8
                f.rings("r", [hx[0]], [hy[0]], [8 + 70 * (1 - (1 - u) ** 3)], 1.0 * (1 - u) ** 1.5, width=1.3)
                f.dots("w", [hx[0]], [hy[0]], 3.2 * (1 - u) + 1.0, 1.6 * (1 - u))
            if m["hero"] and 0 <= ah < 2.2:
                hx, hy, _, _ = cam.project(m["hit"][None].astype(np.float32))
                x, y = float(hx[0]), float(hy[0])
                alpha = 1 - float(smoothstep(1.6, 2.2, ah))
                ex, ey = x + 40, y - 46
                f.segments("w", [x, ex], [y, ey], [ex, ex + 30], [ey, ey], 0.7 * alpha)
                f.tag("r" if m["part"] == "HEART" else "w", ex + 38, ey + 7, m["part"], size=20, alpha=alpha, pad=5)
                hp = m["hit"]
                lines = [f"X {hp[0]:+.3f}  Y {hp[1]:.3f}  Z {hp[2]:+.3f} M",
                         f"MU{m['charge']} {m['E']:.3f} GEV   DE {m['dE'] * 1000:.1f} MEV"]
                for k, ln in enumerate(lines):
                    n = int(min(len(ln), max(0, (ah - 0.1 - 0.08 * k) * 100)))
                    f.text("w", ex + 38, ey + 34 + k * 20, ln[:n], size=15, alpha=0.9 * alpha)

    # ------------------------------------------------------------------ HUD
    def _draw_view_tag(self, f, name, views, tm):
        x0, y0 = VIEW_X0 + 20, MAIN[1] + 36
        f.tag("w", x0, y0, f"VIEW {tm['bip'] + 1:02d} // {name}", size=16, pad=4)
        cam = views[0][0]
        if cam.ortho:
            s = cam.scale
            f.segments("w", [x0, x0 + 0.5 * s], [y0 + 30, y0 + 30], [x0 + 0.5 * s, x0 + 0.5 * s], [y0 + 30, y0 + 24],
                       0.9)
            f.segments("w", [x0], [y0 + 24], [x0], [y0 + 30], 0.9)
            f.text("w", x0 + 0.5 * s + 10, y0 + 36, "0.5 M", size=13, alpha=0.8)
        f.text("w", MAIN[2] - 20, y0, "YOU // MUON BLOOM // 1.80 M // 0.10 M2 CROSS SECTION", size=14, alpha=0.85,
               anchor="rs")

    def _draw_left(self, f, t, ages):
        x0, y0, x1, y1 = LEFT_COL
        f.tag("w", x0 + 8, y0 + 110, "YOU", size=110, pad=10)
        rows = ["MU FLUX  1 /CM2/MIN", "THROUGH YOU ~17 /S", "HEIGHT   1.800 M", "AREA     0.100 M2",
                "DE/DX    2.0 MEV/CM"]
        for k, r in enumerate(rows):
            f.text("w", x0 + 8, y0 + 170 + k * 22, r, size=14, alpha=0.85)
        f.tag("w", x0 + 8, y0 + 310, "HIT_LOG", size=12, pad=3)
        hits = [(a - m["dur"] * m["u_in"], m) for m, a in zip(self.mus, ages) if m["hit"] is not None]
        hits = sorted([h for h in hits if h[0] >= 0], key=lambda h: h[0])[:30]
        for k, (ah, m) in enumerate(hits):
            hp = m["hit"]
            line = f"{m['part']:<7}{hp[0]:+.2f} {hp[1]:.2f} {hp[2]:+.2f} {m['dE'] * 1000:5.1f}"
            f.text("r" if m["part"] == "HEART" or k == 0 else "w", x0 + 8, y0 + 340 + k * 19, line, size=12,
                   alpha=0.95 if k < 3 else 0.65)

    def _draw_strip(self, f, t):
        x0, y0, x1, y1 = STRIP
        ta, tb = t - 5.5, t + 2.0
        X = lambda tt: x0 + (np.asarray(tt) - ta) / (tb - ta) * (x1 - x0)
        f.segments("w", [x0, x0], [y0, y1], [x1, x1], [y0, y1], 0.9)
        ruler(f, x0, x1, y0, ta, tb, BEAT / 4, BAR, down=True)
        ruler(f, x0, x1, y1, ta, tb, BEAT / 4, BAR, down=False)
        yb = (y0 + y1) / 2 - 6
        f.rects("r", x0, yb - 3, x1, yb + 3, 1.0)
        placed = []
        for m in self.mus:
            for tt in (m["t"] - self.T, m["t"], m["t"] + self.T):
                if not (ta - 0.3 <= tt <= tb + 0.3):
                    continue
                xe = float(X(tt))
                key = int(m["t"] * 1000)
                n = 2 + int(hash01(key, 2) * 6)
                bx = xe + np.arange(n) * 4.0
                hh = (8 + 10 * math.log1p(m["E"])) * (0.3 + 0.7 * hash01(key, np.arange(n)))
                f.rects("w", bx, y0 + 1, bx + 2, y0 + 1 + hh, 0.95)
                if m["part"]:
                    hb = (6 + 60 * m["dE"]) * (0.3 + 0.7 * hash01(key, np.arange(n) + 9))
                    f.rects("r", bx, y1 - hb, bx + 2, y1 - 1, 0.9)
                    if m["hero"]:
                        row = 1 if any(abs(p - xe) < 150 for p in placed) else 0
                        placed.append(xe)
                        f.tag("r" if tt <= t else "w", xe, yb + 6 + row * 26, m["part"], size=14, pad=4,
                              alpha=1.0 if tt <= t else 0.5)
        xc = float(X(t))
        f.segments("r", [xc], [y0 - 4], [xc], [y1 + 4], 1.2, width=1.6)
        f.tag("r", xc + 6, y1 + 22, f"T {t:06.3f}", size=13, pad=3)
        f.tag("w", x0 + 4, y0 - 10, "HIT_TIMELINE // WHITE = MUON  RED = ENERGY LEFT IN YOU", size=12, pad=3)

    def _draw_bottom(self, f, t, ages):
        x0, y0, x1, y1 = BOT
        # counts on every time scale
        cx0, cx1 = x0, 1180.0
        panel_header(f, cx0, cx1, y0, "MUONS THROUGH YOU")
        cols = [("PER SECOND", "17"), ("PER MINUTE", "1,000"), ("PER DAY", "1.4 M"), ("LIFETIME", "42 BN")]
        cw = (cx1 - cx0) / 4
        for k, (lab, val) in enumerate(cols):
            xx = cx0 + k * cw
            f.text("w", xx + 4, y0 + 34, lab, size=13, alpha=0.75)
            f.text("r" if k == 3 else "w", xx + 2, y0 + 96, val, size=48, alpha=0.97)
        # hit barcode
        bx0, bx1 = 1220.0, 2000.0
        panel_header(f, bx0, bx1, y0, "HIT_BARCODE")
        cols_n = 200
        dt = 3.0 / cols_n
        kf = math.floor((t - 3.0) / dt)
        kk = kf + np.arange(cols_n)
        tt = (kk * dt) % self.T
        ht = np.sort([(m["t"] + m["dur"] * (m["u_in"] or 0)) % self.T for m in self.mus if m["hit"] is not None])
        lo = np.searchsorted(ht, tt)
        hi = np.searchsorted(ht, tt + dt * 3)
        dens = 0.05 + 0.9 * np.tanh((hi - lo) / 1.5)
        barcode_lanes(f, bx0, bx1, y0 + 10, y1, dens, kk, lanes=3, seed=7)
        # energy left in each part (decaying)
        ex0 = 2040.0
        panel_header(f, ex0, x1, y0, "ENERGY LEFT IN YOU // MEV")
        dep = {}
        for m, a in zip(self.mus, ages):
            if m["hit"] is None:
                continue
            ah = a - m["dur"] * m["u_in"]
            if ah >= 0:
                dep[m["part"]] = dep.get(m["part"], 0.0) + m["dE"] * 1000 * math.exp(-ah / 5.0)
        parts = ["HEAD", "HEART", "TORSO", "ARM_L", "ARM_R", "HAND_L", "HAND_R", "LEG_L", "LEG_R", "FOOT_L"]
        for k, p in enumerate(parts):
            col, row = k // 5, k % 5
            xx = ex0 + col * 460
            yy = y0 + 22 + row * 21
            v = dep.get(p, 0.0)
            f.text("w", xx + 4, yy + 12, f"{p:<7}", size=12, alpha=0.8)
            w = min(300.0, v * 2.2)
            f.rects("r" if p == "HEART" else "w", xx + 90, yy + 2, xx + 90 + w, yy + 12, 0.95)
            f.text("w", xx + 400, yy + 12, f"{v:6.1f}", size=12, alpha=0.8)
