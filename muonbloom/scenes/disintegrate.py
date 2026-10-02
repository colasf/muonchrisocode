"""DISINTEGRATE - everything returns to almost nothing.   Sheet scene 10, 10:06 - 10:52.

The sphere of COSMIC GROOVE (scenes/sphere.py) comes back with the picture it had built - the same
94 tracks: the PoCA dots gathered on the core, the last ghost lines - and falls apart, the way a
muon does: mu -> e + nu + nu.
  10:06 - 10:28  loud, dense: every detector hit sends a muon that stops on the body and decays
                 there (a three-prong star: one electron, two neutrinos); the chunk under it breaks
                 off - vertices drift away, edges erode into dots, halo sticks drop. Echo repeats
                 of a hit keep biting at the same wound. The old tracks decay one by one: the core
                 sparkles, then its picture is gone.
  10:28 - 10:52  thinning out: what is left follows the decay law N/N0 = exp(-t/tau); the HUD text
                 erodes, the counters run backwards, the log empties. At 10:52 a few drifting dots
                 and the lattice: the next scene opens on "nothing chose when...".
Everything is a closed-form function of t: the release time of every vertex / stick / track is
fixed at start-up from the hit list (the realtime app does the same from the live stream).
The layout is the one of the sphere scene (sphere.Lay): every block is dealt out from the columns
the towers leave free, nothing has a fixed x.
"""
from __future__ import annotations

import math

import numpy as np

from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import hash01, smoothstep
from ..show import Scene
from .sphere import (BUS_Y, HOT, N_BINS, R_FAR, Y_BASE, Y_LOW, Body, Lay, Sphere, er, fade, ring_values,
                     sphere_tracks, unit)

T0, T1 = 606.0, 652.0
TAU = 16.0                        # s: decay constant of the body
EDGE_LIFE, STRUT_LIFE, DOT_TAU = 1.6, 2.2, 3.4
G = 760.0                         # px / s2: the halo sticks fall
KEEP = 7                          # vertices that never go: "almost" nothing


def remaining(t):
    """Fraction of the body still there at t."""
    t = np.asarray(t, np.float64)
    return np.exp(-np.clip(t - T0, 0, None) / TAU) * (1.0 - smoothstep(634.0, 652.0, t) ** 1.5)


def michel(x):
    """Energy spectrum of the electron of a muon decay (x = E / 52.8 MeV)."""
    return x * x * (3.0 - 2.0 * x)


class Disintegrate(Sphere):
    name = "disintegrate"
    towers = "auto"

    def __init__(self, ctx):
        Scene.__init__(self, ctx)
        self.lay = Lay(ctx)
        self.body = Body()
        self.ring_jit = hash01(np.arange(N_BINS), 77) - 0.5
        self.scan_order = np.zeros(len(self.body.ea))
        self.t_first = T0
        # the picture it comes back with: the tracks of COSMIC GROOVE. Echo lines were gone long before the
        # end of that scene (their PoCA dot is what is left), primary lines stand as ghosts, the newest
        # ones clearest. Every track gets the moment it decays here.
        self.old = sphere_tracks(ctx.det)
        rng = np.random.default_rng(1010)
        prim = [tr for tr in self.old if not tr["echo"]]
        rank = {tr["id"]: k for k, tr in enumerate(prim[::-1])}        # 0 = the newest primary
        for tr in self.old:
            if tr["echo"]:
                tr["ghost"] = 0.0
                tr["decay"] = float(min(T0 + 0.8 + rng.exponential(9.0), 641.0))
            else:
                k = rank[tr["id"]]
                tr["ghost"] = 1.0 if k < 8 else (0.5 if k < 16 else 0.25)
                tr["decay"] = float(min(T0 + 1.5 + rng.exponential(11.0), 645.0))
            tr["star"] = self._star_dirs(rng)
            tr["ee"] = float(self._michel_sample(rng))
        self._schedule(ctx, np.random.default_rng(2020))

    # ------------------------------------------------------------------ build
    @staticmethod
    def _star_dirs(rng):
        """Three directions in a random plane, roughly 120 deg apart: electron, neutrino, neutrino."""
        a = unit(rng.normal(size=3))
        b = unit(np.cross(a, rng.normal(size=3)))
        th = rng.uniform(0, 2 * np.pi) + np.array([0.0, 2.1 + rng.normal(0, 0.25), 4.2 + rng.normal(0, 0.25)])
        return (np.cos(th)[:, None] * a[None] + np.sin(th)[:, None] * b[None]).astype(np.float32)

    @staticmethod
    def _michel_sample(rng):
        while True:
            x = rng.random()
            if rng.random() < michel(x):
                return x

    def _schedule(self, ctx, rng):
        b = self.body
        n = b.n
        u = b.u
        ev = []
        for key in sd.KEYS:
            tt, ee, ec = ctx.det.hits(key, T0, T1)
            ev += [(float(a), key, float(e), bool(c)) for a, e, c in zip(tt, ee, ec)]
        ev += [(float(tq), "", 0.0, False) for tq in np.arange(T0 + 1.0, T1, 0.25)]      # the slow trickle
        ev.sort()
        rel = np.full(n, np.inf)
        ragged = 4.5 * (b.rho(u, 0.0) - 1.0)          # smooth noise: the wounds get ragged edges
        vel = np.zeros((n, 3))
        free = np.ones(n, bool)
        released = 0
        chunks = []
        last_c = {}
        for th, key, e, echo in ev:
            want = int(n * (1.0 - float(remaining(th + 0.5)))) - released
            room = n - KEEP - released
            if room <= 0:
                break
            if key == "":                       # trickle: single nodes let go when the decay law runs ahead
                if want > 14:
                    idx = rng.choice(np.nonzero(free)[0], size=min(3, room), replace=False)
                    rel[idx] = th + rng.uniform(0, 0.25, len(idx))
                    vel[idx] = u[idx] * rng.uniform(0.08, 0.2, (len(idx), 1)) + rng.normal(0, 0.03, (len(idx), 3))
                    free[idx] = False
                    released += len(idx)
                continue
            # the bites follow the decay law: a hit takes what the law is owed (an echo a part of it); when
            # the body is ahead of the law it only chips a node or two
            base = int((4 if echo else 14) * (0.4 + e) * (1.5 if key == "C" else 1.0))
            if want <= 0:
                k = 1 if echo else 3
            elif echo:
                k = min(want, max(2, base))
            else:
                k = max(want, min(base, want + 6))
            k = int(min(k, room, 150))
            if k <= 0:
                continue
            cand = np.nonzero(free)[0]
            if echo and key in last_c:
                c = unit(last_c[key] + rng.normal(0, 0.16, 3))
            else:                               # a new crater, where the body is still whole (more often on top)
                w = 1.3 + u[cand, 1]
                c = unit(u[rng.choice(cand, p=w / w.sum())] + rng.normal(0, 0.1, 3))
            last_c[key] = c
            order = cand[np.argsort(-(u[cand] @ c + ragged[cand]))[:k]]
            ang = np.arccos(np.clip(u[order] @ c, -1, 1))
            rel[order] = th + 0.02 + 0.22 * ang
            vel[order] = (c[None] * (0.14 + 0.2 * e) + u[order] * rng.uniform(0.05, 0.2, (k, 1))
                          + rng.normal(0, 0.04, (k, 3)))
            free[order] = False
            released += k
            chunks.append(dict(t=th, key=key, e=e, echo=echo, c=c, n=k, star=self._star_dirs(rng),
                               ee=float(self._michel_sample(rng)), id=len(chunks) + 1))
        self.rel, self.vel = rel, vel.astype(np.float32)
        self.chunks = chunks
        self.rel_sorted = np.sort(rel[np.isfinite(rel)])
        # halo sticks: a chunk takes the sticks on its side of the limb with it; the rest follow the decay law
        drop = np.full(N_BINS, np.inf)
        phi = 2 * np.pi * (np.arange(N_BINS) + 0.5) / N_BINS
        for ch in chunks:
            cam, _ = self._cam(ch["t"])
            px, py, _, _ = cam.project(np.stack([np.zeros(3), ch["c"]]).astype(np.float32))
            pa = math.atan2(-(float(py[1]) - float(py[0])), float(px[1]) - float(px[0]))
            lim = math.hypot(float(px[1]) - float(px[0]), float(py[1]) - float(py[0])) / self.lay.R   # 1 = on the limb
            half = (0.06 + 0.5 * ch["n"] / 230.0) * (0.35 + 0.65 * lim)
            d = np.abs(np.angle(np.exp(1j * (phi - pa))))
            hit = (d < half) & np.isinf(drop)
            drop[hit] = ch["t"] + 0.08 + 0.5 * d[hit] / max(half, 1e-3) * 0.4
        rest = np.nonzero(np.isinf(drop))[0]
        rng.shuffle(rest)
        tgrid = np.linspace(T0, T1 - 0.6, 600)
        frac = 1.0 - remaining(tgrid)
        have = N_BINS - len(rest)
        for j, bi in enumerate(rest[: len(rest) - 2]):
            drop[bi] = float(np.interp((have + j + 1) / N_BINS, frac, tgrid))
        self.drop = drop
        self.stick_v = rng.normal(0, 60.0, (N_BINS, 2)) + np.array([0.0, -40.0])
        self.stick_w = rng.normal(0, 2.4, N_BINS)

    def _cam(self, t):
        return self.lay.camera(t, T0, 92.0, phase=2.2)

    # ------------------------------------------------------------------ state
    def _counts(self, t):
        b = self.body
        nodes = int((self.rel > t).sum())
        edges = int((np.minimum(self.rel[b.ea], self.rel[b.eb]) > t).sum())
        struts = int((np.minimum(self.rel[b.sa], self.rel[b.sb]) > t).sum())
        sticks = int((self.drop > t).sum())
        tracks = sum(1 for tr in self.old if tr["decay"] > t)
        return nodes, edges, struts, sticks, tracks

    def chunks_before(self, t):
        return [ch for ch in self.chunks if ch["t"] <= t]

    # ----------------------------------------------------------------- render
    def draw(self, f, t, ctx):
        lay = self.lay
        cam, yaw = self._cam(t)
        e_txt = float(smoothstep(628.0, 651.0, t)) ** 1.2
        frac = float(remaining(t))
        alive = [tr for tr in self.old if tr["decay"] > t]
        thr = [tr for tr in alive if tr["through"]]
        phi, v, phc = ring_values(lay, cam, 7, 9, 9, self.ring_jit)
        v = v * (0.35 + 0.65 * frac ** 0.5)
        f.set_clip(*lay.clip)
        self._lattice(f)
        self._eroding_body(f, cam, t)
        self._ring(f, phi, v, 1.0, alive=self.drop > t, gain=1.0)
        self._falling_sticks(f, phi, v, t)
        for tr in self.old:                               # the old picture, decaying
            a = t - tr["decay"]
            if a < 0:
                self._draw_track(f, cam, tr, 60.0, ctx, persist=tr["ghost"], leader=False)
            elif a < 0.6:
                self._draw_track(f, cam, tr, 60.0, ctx, persist=tr["ghost"] * (1.0 - a / 0.6), leader=False)
            if 0 <= a < (1.6 if tr["echo"] else 3.2):
                self._star(f, cam, tr["K"], tr["star"], a, tr["e"], big=not tr["echo"], label=False)
        f.set_clip()
        self._core(f, cam, t, thr, gain=max(0.0, min(1.0, frac * 3.0)))
        self._impacts(f, cam, t, ctx)
        self._callouts_d(f, t, e_txt, frac, yaw)
        e_tomo = float(np.clip(1.0 - frac ** 0.6, 0.0, 1.0))
        self._tomogram(f, t, alive, thr, yaw, erode=e_tomo, alive=hash01(np.arange(200), 21) > e_tomo,
                       title="TOMOGRAM // TOP VIEW // LOSING THE PICTURE", erode_txt=e_txt)
        self._integrity(f, t, ctx, e_txt)
        self._left_d(f, t, e_txt)
        self._right(f, t, ctx, alive, title="DECAY", erode=e_txt)
        self._michel(f, t, e_txt)
        self._strip(f, t, ctx, title="INTEGRITY_TIMELINE // 10.0 DISINTEGRATE // ONE TICK = ONE HIT", t0=T0, t1=T1,
                    erode=e_txt, curve=lambda tt: remaining(tt))
        self._bottom_d(f, t, ctx, e_txt, frac)
        return {"burst_gain": 0.6, "burst_size": 0.7}       # the hits come in swarms: the towers reply lower

    # ------------------------------------------------------------------ world
    def _eroding_body(self, f, cam, t):
        b = self.body
        P0 = b.surface(t).astype(np.float32)
        N = b.normals(t).astype(np.float32)
        a = (t - self.rel).astype(np.float32)
        gone = a >= 0
        aa = np.where(gone, a, 0.0)[:, None]
        drift = 3.0 * (1.0 - np.exp(-aa / 3.0))                     # the pieces slow down and float
        P = P0 + self.vel * drift + np.array([0.0, -0.012, 0.0], np.float32) * aa * aa
        sx, sy, _, ok = cam.project(P)
        view = unit(cam.pos.astype(np.float32)[None] - P0)
        nv = (N * view).sum(1)
        rim = (0.1 + 0.8 * (1 - np.abs(nv)) ** 2.2) * np.where(nv > 0, 1.0, 0.45)
        for (ia, ib, var, life, nd, base) in ((b.ea, b.eb, b.e_var, EDGE_LIFE, 5, None),
                                              (b.sa, b.sb, b.s_var, STRUT_LIFE, 8, 0.1)):
            te = np.minimum(self.rel[ia], self.rel[ib])
            ae = t - te
            whole = ae < 0
            if base is None:
                i0, i1 = rim[ia] * var, rim[ib] * var
            else:
                i0 = i1 = (0.085 + base * np.maximum(rim[ia], rim[ib])) * var
            f.segments("w", sx[ia][whole], sy[ia][whole], sx[ib][whole], sy[ib][whole], i0[whole], i1[whole])
            erd = (ae >= 0) & (ae < life)
            if erd.any():               # the edge is now a row of dots that thins out
                ua = (ae[erd] / life)[:, None]
                s = (np.arange(nd) + 0.5)[None, :] / nd
                keep = hash01(np.nonzero(erd)[0][:, None], np.arange(nd)[None, :], 31) > ua
                X = sx[ia][erd][:, None] * (1 - s) + sx[ib][erd][:, None] * s
                Y = sy[ia][erd][:, None] * (1 - s) + sy[ib][erd][:, None] * s
                I = (0.5 * (i0[erd] + i1[erd]))[:, None] * (1.0 - ua) * 2.2 + 0.0 * s
                f.pixels("w", X[keep], Y[keep], np.minimum(I[keep] + 0.25 * (1 - ua + 0 * s)[keep], 1.6))
        whole = ~gone
        f.pixels("w", sx[whole], sy[whole], 0.3 + 0.65 * rim[whole])
        fl = gone & ok
        if fl.any():
            fd = np.exp(-a[fl] / DOT_TAU)
            fresh = np.exp(-a[fl] / 0.25)
            f.dots("w", sx[fl], sy[fl], 1.5, 0.25 + 0.9 * fd + 0.8 * fresh)

    def _falling_sticks(self, f, phi, v, t):
        lay = self.lay
        a = t - self.drop
        m = (a >= 0) & (a < 1.5)
        if not m.any():
            return
        Ln = lay.l0 + (lay.l1 - lay.l0) * v[m]
        c, s = np.cos(phi[m]), -np.sin(phi[m])
        am = a[m]
        mx = lay.cx + (lay.r0 + Ln / 2) * c + self.stick_v[m, 0] * lay.s * am
        my = lay.cy + (lay.r0 + Ln / 2) * s + self.stick_v[m, 1] * lay.s * am + 0.5 * G * lay.s * am * am
        ang = np.arctan2(s, c) + self.stick_w[m] * am
        hx, hy = np.cos(ang) * Ln / 2, np.sin(ang) * Ln / 2
        fd = (1.0 - am / 1.5) ** 1.2
        hot = v[m] > HOT
        for sel, layer in ((~hot, "w"), (hot, "r")):
            if sel.any():
                f.segments(layer, (mx - hx)[sel], (my - hy)[sel], (mx + hx)[sel], (my + hy)[sel], 0.9 * fd[sel],
                           width=L.LW)
                f.dots(layer, (mx + hx)[sel], (my + hy)[sel], 3.2, 1.4 * fd[sel])

    def _star(self, f, cam, pos, dirs, a, e, big=True, label=False):
        """mu -> e + nu + nu at `pos`: a solid prong (the electron) and two dotted ones (the neutrinos)."""
        if a < 0 or a > 3.2:
            return
        grow = 1.0 - math.exp(-a / 0.16)
        fd = math.exp(-a / (1.1 if big else 0.45))
        sc = (0.55 if big else 0.24) * (0.6 + 0.6 * e)
        P = np.stack([pos, pos + dirs[0] * 0.5 * sc * grow, pos + dirs[1] * (1.1 * sc * grow + 0.25 * a),
                      pos + dirs[2] * (1.1 * sc * grow + 0.25 * a)]).astype(np.float32)
        px, py, _, ok = cam.project(P)
        if not ok.all():
            return
        f.segments("w", [px[0]], [py[0]], [px[1]], [py[1]], 1.3 * fd, 0.5 * fd, width=L.LW_BOLD if big else L.LW)
        f.dots("w", px[1:2], py[1:2], 2.4 if big else 1.6, 1.3 * fd)
        for k in (2, 3):                    # neutrinos: dotted, straight, leaving
            n = int(max(4, math.hypot(px[k] - px[0], py[k] - py[0]) / 9.0))
            s = (np.arange(n) + 0.5) / n
            f.pixels("w", px[0] + (px[k] - px[0]) * s, py[0] + (py[k] - py[0]) * s, 1.3 * fd * (1 - 0.6 * s))
        f.dots("r", px[0:1], py[0:1], 3.4 if big else 2.0, (1.6 if big else 1.2) * fd)
        if big:
            f.rings("r", px[0:1], py[0:1], [8 + 54 * (1 - math.exp(-a / 0.5))], 0.9 * math.exp(-a / 0.45), width=L.LW)
        if label and a < 2.2:               # prong labels, only where they stay whole inside the body column
            al = min(1.0, 3 * fd)
            x0, x1 = self.lay.body
            for k, (word, g) in enumerate((("E-", 1.0), ("NU", 0.7), ("NU", 0.7)), start=1):
                if x0 <= float(px[k]) + 8 <= x1 - 20:
                    f.text("w", float(px[k]) + 8, float(py[k]) + 5, word, size=L.T_MICRO, alpha=g * al)

    def _impacts(self, f, cam, t, ctx):
        """The muons of this scene: leader from the detector, a red track that stops on the body, a star."""
        b = self.body
        clip = self.lay.clip
        last_prim = None
        for ch in self.chunks:
            a = t - ch["t"]
            if a < 0:
                break
            if not ch["echo"]:
                last_prim = ch
            if a > 3.2:
                continue
            S = (ch["c"] * b.rho(ch["c"][None], t)[0]).astype(np.float32)
            d = unit(-ch["c"] * 0.55 + np.array([0.0, -1.0, 0.0]))            # it came from above
            far = S - d * 0.2
            # walk back to the far radius
            bq = float(far @ (-d))
            far = far + (-d) * (-bq + math.sqrt(max(bq * bq - (float(far @ far) - R_FAR ** 2), 0.0)))
            px, py, _, ok = cam.project(np.stack([far, S]).astype(np.float32))
            if not ok.all():
                continue
            g = (0.4 if ch["echo"] else 1.0) * (0.5 + 0.6 * ch["e"])
            fl = math.exp(-a / 0.5)
            f.set_clip(*clip)
            f.segments("r", [px[0]], [py[0]], [px[1]], [py[1]], 1.3 * g * fl + 0.1 * g, width=2.6 if ch["key"] == "C" else 1.5)
            self._star(f, cam, S, ch["star"], a, ch["e"], big=not ch["echo"], label=not ch["echo"] and ch is last_prim)
            f.set_clip()
            if a < 1.7:                     # leader: detector -> bus -> down into the track (may pass behind a tower)
                tw = ctx.towers[ch["key"]]
                ox, oy = tw.det
                yb = BUS_Y[ch["key"]]
                al = (0.4 if ch["echo"] else 1.0) * math.exp(-a / 0.5) * min(1.0, a / 0.1 + 0.2)
                f.polyline("r", [ox, ox, float(px[0]), float(px[0])], [oy - 14, yb, yb, float(py[0])], 1.1 * al, width=L.LW)
                f.dots("r", [float(px[0])], [yb], 3.0, 1.4 * al)

    # ------------------------------------------------------------------- HUD
    def _callouts_d(self, f, t, e_txt, frac, yaw):
        fr = int(t * 30)
        self._view_block(f, "VIEW 01 // BODY // LOSING MASS",
                         [f"ORBIT {math.degrees(yaw) % 360:05.1f} DEG", f" N/N0 {frac:.3f}"], e_txt, fr)
        prim = [ch for ch in self.chunks if ch["t"] <= t and not ch["echo"]]
        if prim:
            ch = prim[-1]
            age = t - ch["t"]
            if age < 4.0:
                al = 1.0 - float(smoothstep(3.2, 4.0, age))
                lines = [f"{L.NAMES[ch['key']]}  E {ch['e']:.3f}", f"E_E {ch['ee'] * 52.8:5.2f} MEV", f"CHUNK -{ch['n']:03d} NODES"]
                self._corner(f, 1, er("MU- > E- NU NU", e_txt, 93, fr),
                             [(er(hud.typed(ln, age, delay=0.1 + 0.1 * k), e_txt, 94 + k, fr), "w") for k, ln in enumerate(lines)],
                             alpha=al, tag_alpha=al * fade(e_txt), tag_size=L.T_SMALL)
        self._corner(f, -1, er("DISINTEGRATING", e_txt, 96, fr),
                     [(er(f"HITS {len(self.chunks_before(t)):03d}", e_txt, 97, fr), "w"),
                      (er(f"REMAINING {100 * frac:05.2f} %", e_txt, 98, fr), "w")], tag_alpha=fade(e_txt))

    def _integrity(self, f, t, ctx, e_txt):
        """N/N0 against time, under the tomogram: the decay law (dotted) and what is really left of the body
        (staircase)."""
        lay = self.lay
        if not lay.tomo:
            return
        fr = int(t * 30)
        x0, x1 = self._low_panel(f, lay.tomo, "INTEGRITY // N/N0 // DECAY LAW EXP(-T/TAU)", e_txt, 101, fr)
        px0, px1, py0, py1 = x0 + 60, x1 - 8, Y_LOW + 34, Y_BASE
        g = 1.0 - e_txt
        X = lambda tt: px0 + (np.asarray(tt, np.float64) - T0) / (T1 - T0) * (px1 - px0)
        Y = lambda q: py1 - np.asarray(q, np.float64) * (py1 - py0)
        f.segments("w", [px0, px0], [py0, py1], [px0, px1], [py1, py1], 0.65 * g)
        for q in (0.0, 0.5, 1.0):
            f.segments("w", [px0 - 8], [float(Y(q))], [px0], [float(Y(q))], 0.8 * g)
            f.text("w", x0, float(Y(q)) + 5, er(f"{q:.1f}", e_txt, 102, fr), size=L.T_MICRO, alpha=0.7)
        hud.ruler(f, px0, px1, py1 + 2, T0, T1, 1.0, 10.0, inten=0.6 * g)
        for tv in np.arange(610.0, T1, 10.0 if px1 - px0 >= 420 else 20.0):
            f.text("w", float(X(tv)) + 3, py1 + 28, er(sd.tc(tv)[:5], e_txt, 103, fr), size=L.T_MICRO, alpha=0.7)
        tt = np.linspace(T0, T1, 150)
        keep = hash01(np.arange(150), 105) > e_txt
        f.dots("w", X(tt)[keep], Y(remaining(tt))[keep], 1.3, 0.7)
        # what is really left
        n = self.body.n
        ts = np.linspace(T0, min(t, T1), max(2, int((min(t, T1) - T0) * 12)))
        left = 1.0 - np.searchsorted(self.rel_sorted, ts, side="right") / n
        if e_txt < 0.97:
            f.polyline("w", X(ts), Y(left), 1.0 * (1 - 0.6 * e_txt), width=L.LW_BOLD)
        for ch in self.chunks:
            if ch["t"] > t:
                break
            if not ch["echo"] and hash01(ch["id"], 106) > e_txt:
                f.rects("r", float(X(ch["t"])), py1 - 4 - 22 * ch["e"], float(X(ch["t"])) + 3, py1, 0.95)
        xc, yc = float(X(min(t, T1))), float(Y(left[-1]))
        f.segments("r", [xc], [py0 - 4], [xc], [py1 + 4], 1.1, width=L.LW)
        f.dots("r", [xc], [yc], 4.0, 1.6)
        f.tag("r", max(min(xc + 10, px1 - 112), px0 + 4), max(yc - 12, py0 + 22), f"N/N0 {left[-1]:.3f}", size=L.T_MICRO, pad=3)
        f.text("w", px1, py0 + 12, er(f"TAU {TAU:.1f} S", e_txt, 108, fr), size=L.T_MICRO, alpha=0.8, anchor="rs")

    def _left_d(self, f, t, e_txt):
        if not self.lay.info:
            return
        fr = int(t * 30)
        b = self.body
        nodes, edges, struts, sticks, tracks = self._counts(t)
        rows_ = [f"NODES     {nodes:04d} / {b.n:04d}", f"EDGES     {edges:04d} / {len(b.ea):04d}",
                 f"STRUTS    {struts:04d} / {len(b.sa):04d}", f"STICKS    {sticks:04d} / {N_BINS:04d}",
                 f"TRACKS    {tracks:04d} / {len(self.old):04d}", "MU- -> E- + NU + NU", "TAU_MU    2.197 US"]
        x, w, y = self._info_block(f, "DISINTEGRATE", rows_, red=(5,), erode=e_txt, fr=fr, key=111)
        past = self.chunks_before(t)[::-1]
        n_rows = int(round(24 * (1.0 - float(smoothstep(634.0, 650.5, t)))))
        lines = []
        for row, ch in enumerate(past[:n_rows]):
            m = int(ch["t"] // 60)
            age = t - ch["t"]
            red = age < 1.0 or (not ch["echo"] and ch["n"] > 80)
            fields = [f"{m:02d}:{ch['t'] - 60 * m:04.1f}", f"{ch['key']}{'e' if ch['echo'] else ' '}", f"{ch['e']:.2f}",
                      f"{ch['ee'] * 52.8:7.2f}", f" -{ch['n']:03d}"]
            lines.append((fields, "r" if red else "w", 0.95 if row < 2 or red else 0.62, age))
        self._log_block(f, x, w, y, "DECAY_LOG // LIVE", ["TIME   ", "D ", "E   ", "E_E MEV", "NODES"], lines,
                        erode=e_txt, fr=fr, key=120)

    def _michel(self, f, t, e_txt):
        """Spectrum of the decay electrons seen so far, against the Michel shape (under the detectors)."""
        if not self.lay.spec:
            return
        fr = int(t * 30)
        x0, x1 = self._low_panel(f, self.lay.spec, "E_ELECTRON // MICHEL // MEV", e_txt, 141, fr, cap=460.0)
        es = [ch["ee"] for ch in self.chunks if ch["t"] <= t] + [tr["ee"] for tr in self.old if tr["decay"] <= t]
        edges = np.linspace(0, 1, 23)
        cnt, _ = np.histogram(np.array(es), edges) if es else (np.zeros(22), None)
        bw = (x1 - x0 - 8) / 22
        xs = x0 + 4 + np.arange(22) * bw
        yb, hmax = Y_BASE, 150.0
        top = max(float(np.max(cnt)), 1.0)
        keep = (cnt > 0) & (hash01(np.arange(22), 142) > e_txt)
        f.rects("w", xs[keep], yb - hmax * cnt[keep] / top, xs[keep] + bw - 4, yb, 0.95)
        xx = np.linspace(0, 1, 60)
        kk = hash01(np.arange(60), 143) > e_txt
        f.dots("r", (x0 + 4 + xx * 22 * bw)[kk], (yb - hmax * michel(xx))[kk], 1.5, 1.1)
        hud.ruler(f, x0 + 4, x0 + 4 + 22 * bw, yb + 2, 0, 52.8, 2.4, 12.0 if x1 - x0 >= 300 else 24.0,
                  fmt=lambda v: er(f"{v:.0f}", e_txt, 144, fr), inten=0.6 * (1 - e_txt), lab_dy=26)

    def _bottom_d(self, f, t, ctx, e_txt, frac):
        fr = int(t * 30)
        nodes, edges, struts, sticks, tracks = self._counts(t)
        self._numbers_panel(f, "REMAINING", [("NODES", f"{nodes:04d}", "w"), ("EDGES", f"{edges:04d}", "w"),
                                             ("STICKS", f"{sticks:03d}", "w")], erode=e_txt, fr=fr)
        self._barcode_panel(f, t, ctx, erode=e_txt, fr=fr)
        left = nodes / self.body.n
        self._single_panel(f, "N/N0", "1.0" if left >= 1 else (f"{left:.3f}"[1:] if left < 0.1 else f"{left:.2f}"[1:]),
                           "r", erode=e_txt, fr=fr)
