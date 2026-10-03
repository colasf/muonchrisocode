"""DISINTEGRATE - everything returns to almost nothing.   Sheet 9.2, 09:35 - 09:48: the ending of scene 9.

Scene 10 of the sheet (10:06 - 10:52 in the V7 audio) was cut in the scene 10 edit of the audio: its best
part became the ending of scene 9, and so did this, condensed from 46 s to 13 s.
The sphere of COSMIC GROOVE (scenes/sphere.py) comes back with the picture it had built - the same
tracks: the PoCA dots gathered on the core, the last ghost lines - and falls apart, the way a
muon does: mu -> e + nu + nu.
  09:35 - 09:45  loud, dense (the groove of the rise goes on): every detector hit sends a muon that stops
                 on the body and decays there (a three-prong star: one electron, two neutrinos); the chunk
                 under it breaks off - vertices drift away, edges erode into dots, halo sticks drop. Echo
                 repeats of a hit keep biting at the same wound. The old tracks decay one by one: the core
                 sparkles, then its picture is gone. What is left follows the decay law N/N0 = exp(-t/tau);
                 the HUD text erodes, the counters run backwards, the log empties.
  09:45 - 09:48  the music drops into the pulse of scene 11: a few drifting dots and the lattice; the next
                 scene opens on "nothing chose when...".
Everything is a closed-form function of t: the release time of every vertex / stick / track is
fixed at start-up from the hit list (the realtime app does the same from the live stream).
The layout is the one of the sphere scene (sphere.Lay): every block is dealt out from the columns
the towers leave free, nothing has a fixed x.

NOTHING THAT SHOWS DATA FADES, in or out. The picture comes back with a cut, and its furniture is
constructed in the first two seconds like at any scene start (sphere.BLK, Sphere._blk). The loss is
then shown without a single alpha ramp:
  - the lettering erodes (sphere.er: characters flicker through random glyphs and drop out);
  - the tags break up with their text (sphere.etag): the characters that are gone leave holes in the
    box, what is left is a row of fragments, then nothing;
  - the picture thins out (hash masks on the lattice, the contour, the voxels, the bars);
  - the rules - header rules, frames, axes, ticks, the strip - are taken apart the way they were made,
    in slow motion, while their lettering erodes (Disintegrate._age: the build of the block is run
    backwards: pens retreat, ticks stretch and let go);
  - the estimate and the core of the tomogram, the red circle of the core, the labels of the halo are
    taken apart when the tracks that carried them have decayed; the tag of the last decay is made on the
    hit and taken apart four seconds later (or when the next hit comes); the labels of a star are made
    with it and taken apart.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import hash01, smoothstep
from ..show import Scene
from .sphere import (BLK, BUS_Y, FOLLOW, HOT, N_BINS, R_FAR, Y_BASE, Y_LOW, Body, Lay, Sphere, draw_leader, er, fade,
                     one_tag, pop, ring, ring_values, sphere_tracks, unit)

T0, T1 = 575.0, 588.0
T_ALMOST = 585.3                  # the music drops into the pulse of scene 11 (first heavy kick 584.9): almost nothing
TAU = 3.6                         # s: decay constant of the body
EDGE_LIFE, STRUT_LIFE, DOT_TAU = 0.9, 1.2, 1.8
G = 760.0                         # px / s2: the halo sticks fall
KEEP = 7                          # vertices that never go: "almost" nothing
UNMAKE = 0.4                      # s of build a line / a rule needs once the wave of its block has reached it


def remaining(t):
    """Fraction of the body still there at t."""
    t = np.asarray(t, np.float64)
    return np.exp(-np.clip(t - T0, 0, None) / TAU) * (1.0 - smoothstep(T0 + 5.5, T_ALMOST, t) ** 1.5)


def michel(x):
    """Energy spectrum of the electron of a muon decay (x = E / 52.8 MeV)."""
    return x * x * (3.0 - 2.0 * x)


def e_text(t):
    """How far the lettering of the HUD has eroded at t (0..1)."""
    return float(smoothstep(T0 + 4.5, T_ALMOST + 1.6, t)) ** 1.2


def e_picture(t):
    """How far the tomogram has lost its picture at t (0..1)."""
    return float(np.clip(1.0 - float(remaining(t)) ** 0.6, 0.0, 1.0))


def _when(fn, level, n=2760):
    """First time of the scene at which fn reaches `level` (T1 if it never does)."""
    ts = np.linspace(T0, T1, n + 1)
    hit = np.nonzero(np.array([fn(float(v)) for v in ts]) >= level)[0]
    return float(ts[hit[0]]) if len(hit) else T1


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
                tr["decay"] = float(min(T0 + 0.25 + rng.exponential(2.4), T0 + 8.0))
            else:
                k = rank[tr["id"]]
                tr["ghost"] = 1.0 if k < 8 else (0.5 if k < 16 else 0.25)
                tr["decay"] = float(min(T0 + 0.45 + rng.exponential(3.0), T0 + 9.0))
            tr["star"] = self._star_dirs(rng)
            tr["ee"] = float(self._michel_sample(rng))
        self._schedule(ctx, np.random.default_rng(2020))
        # the body still moves (sphere.Body.rho): the noise on the clock of the music, a ring from every bite.
        # A piece that breaks off keeps the radius it had when it let go
        self._init_motion(ctx, T0, T1)
        self.waves = [(ch["t"], (0.3 if ch["echo"] else 1.0) * (0.55 + 0.6 * ch["e"]), np.asarray(ch["c"], np.float64))
                      for ch in self.chunks]
        b = self.body
        self.rho_rel = np.ones(b.n)
        for i in np.nonzero(np.isfinite(self.rel))[0]:
            self.rho_rel[i] = float(b.rho(b.u[i:i + 1], self._pose(float(self.rel[i]), ctx))[0])
        # the tomogram loses its picture: when the erosion reaches each hash (a piece of the contour, a voxel,
        # a point lets go then), and where the estimate stands after every decay of a scattered track
        self._e_t = np.linspace(T0, T1, 1301)
        self._e_v = np.maximum.accumulate(np.array([e_picture(float(v)) for v in self._e_t]))
        thr_all = [tr for tr in self.old if tr["through"]]
        self.est = self._est_steps([(T0 - 1.0, thr_all)] + [(d, [q for q in thr_all if q["decay"] > d])
                                                              for d in sorted(q["decay"] for q in thr_all)])
        self.decays = np.sort(np.array([tr["decay"] for tr in self.old]))
        # the hits that get the tag of the bottom-right corner and the labels of their star
        self.prim = [ch for ch in self.chunks if not ch["echo"]]
        self.prim_t = np.array([ch["t"] for ch in self.prim])
        for ch, nxt in zip(self.prim, list(self.prim_t[1:]) + [1e9]):
            ch["next"] = float(nxt)
        # when things of the HUD have to go (they are taken apart just before): the labels of the halo with
        # its first stick; the estimate and the dotted core of the tomogram, the red circle of the core and
        # its cross when too few scattered tracks are left to give them (or the picture is too far gone);
        # the staircase of the integrity panel with the last of its lettering
        self.t_drop0 = float(np.min(self.drop))
        dec = sorted(tr["decay"] for tr in self.old if tr["through"])
        self.t_est_gone = min(dec[-2] if len(dec) >= 2 else T0, _when(e_picture, 0.9))
        self.t_ring_gone = dec[-3] if len(dec) >= 3 else T0
        self.t_core_gone = min(self.t_ring_gone, _when(e_picture, 0.6))

        def conf(t):                    # confidence of the core, as Sphere._core computes it in this scene
            n = sum(1 for d in dec if d > t)
            return (min(1.0, n / 7.0) if n >= 3 else 0.0) * max(0.0, min(1.0, float(remaining(t)) * 3.0))
        self.t_cross_gone = _when(lambda t: 0.6 - conf(t), 0.0)
        self.t_line_gone = _when(e_text, 0.97)

    T_BUILD = T0                      # the furniture is constructed when the picture comes back

    def _age(self, name, t, erode=0.0):
        """The boxes and rules of a block of furniture (see Sphere._age): constructed when the scene starts,
        like everything; then, while the lettering erodes, taken apart the way they were made - the build of
        the block run backwards, in slow motion (it lasts as long as the erosion). Nothing fades: the
        instrument is unmade. The strip outlasts the panels a little."""
        a = self._age0(name, t)
        left = float(1.0 - smoothstep(0.5, 0.96, erode)) if name == "strip" else fade(erode)
        return a if left >= 1.0 else min(a, (BLK[name]["wave"] + UNMAKE) * left - 0.02)

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
        ev += [(float(tq), "", 0.0, False) for tq in np.arange(T0 + 0.3, T1, 0.07)]      # the trickle
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
                    idx = rng.choice(np.nonzero(free)[0], size=min(max(3, want // 3), room), replace=False)
                    rel[idx] = th + rng.uniform(0, 0.07, len(idx))
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

    def _lost(self, t):
        """h -> seconds since the erosion of the picture reached the hash h (negative: not yet). What the
        tomogram loses lets go at that moment and is given the time to leave (Sphere._tomogram)."""
        def since(h):
            i = np.searchsorted(self._e_v, np.asarray(h, np.float64), side="left")
            return t - np.where(i < len(self._e_t), self._e_t[np.minimum(i, len(self._e_t) - 1)], T1 + 9.0)
        return since

    # ----------------------------------------------------------------- render
    def draw(self, f, t, ctx):
        lay = self.lay
        cam, yaw = self._cam(t)
        self.pose, self._sl = self._pose(t, ctx), None
        e_txt = e_text(t)
        frac = float(remaining(t))
        alive = [tr for tr in self.old if tr["decay"] > t]
        thr = [tr for tr in alive if tr["through"]]
        phi, v, phc = ring_values(lay, cam, 7, 9, 9, self.ring_jit)
        v = v * (0.35 + 0.65 * frac ** 0.5)
        follow = FOLLOW * lay.R * (self.body.outline(cam, phi, self.pose) - 1.0)    # the halo follows the outline of the body
        f.set_clip(*lay.clip)
        self._lattice(f)
        self._eroding_body(f, cam, t)
        self._ring(f, phi, v, 1.0, alive=self.drop > t, gain=1.0,          # its labels go with its first stick
                   label_age=B.io(t - T0 - 0.4, self.t_drop0 - t, out=0.3, span=0.45), follow=follow)
        self._falling_sticks(f, phi, v, t, follow)
        for tr in self.old:                               # the old picture, decaying
            a = t - tr["decay"]
            if a < 0:
                self._draw_track(f, cam, tr, 60.0, ctx, persist=tr["ghost"], leader=False)
            elif a < 0.6:
                self._draw_track(f, cam, tr, 60.0, ctx, persist=tr["ghost"] * (1.0 - a / 0.6), leader=False)
            if 0 <= a < (1.6 if tr["echo"] else 3.2):
                self._star(f, cam, tr["K"], tr["star"], a, tr["e"], big=not tr["echo"])
        f.set_clip()
        self._core(f, cam, t, thr, gain=max(0.0, min(1.0, frac * 3.0)),
                   ring_age=B.io(60.0, self.t_ring_gone - t, out=0.5, span=0.5),
                   cross_age=B.io(60.0, self.t_cross_gone - t, out=0.3, span=0.3))
        self._impacts(f, cam, t, ctx)
        self._callouts_d(f, t, e_txt, frac, yaw)
        e_tomo = e_picture(t)
        a_tomo = self._age0("tomo", t)
        k_dec = int(np.searchsorted(self.decays, t, side="right"))
        self._tomogram(f, t, alive, thr, yaw, erode=e_tomo, lost=self._lost(t),
                       title="TOMOGRAM // TOP VIEW // LOSING THE PICTURE", erode_txt=e_txt,
                       est_age=B.io(a_tomo - 0.45, self.t_est_gone - t, out=0.4),
                       core_age=B.io(a_tomo - 0.6, self.t_core_gone - t, out=0.5, span=0.5),
                       gone=[(tr["K"][0], tr["K"][2], t - tr["decay"]) for tr in self.old if 0.0 <= t - tr["decay"] < 0.5],
                       loose=2.5 * e_tomo, count_age=t - float(self.decays[k_dec - 1]) if k_dec else None)
        self._integrity(f, t, ctx, e_txt)
        self._left_d(f, t, e_txt)
        self._right(f, t, ctx, alive, title="DECAY", erode=e_txt)
        self._michel(f, t, e_txt)
        self._strip(f, t, ctx, title="INTEGRITY_TIMELINE // 9.2 RISE ENDING // ONE TICK = ONE HIT", t0=T0, t1=T1,
                    erode=e_txt, curve=lambda tt: remaining(tt))
        self._bottom_d(f, t, ctx, e_txt, frac)
        return {"burst_gain": 0.6, "burst_size": 0.7}       # the hits come in swarms: the towers reply lower

    # ------------------------------------------------------------------ world
    def _eroding_body(self, f, cam, t):
        b = self.body
        a = (t - self.rel).astype(np.float32)
        gone = a >= 0
        rho, N = b.shape(self.pose)
        P0 = (b.u * np.where(gone, self.rho_rel, rho)[:, None]).astype(np.float32)      # (a piece that let go keeps its radius)
        N = N.astype(np.float32)
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

    def _falling_sticks(self, f, phi, v, t, follow=None):
        lay = self.lay
        a = t - self.drop
        m = (a >= 0) & (a < 1.5)
        if not m.any():
            return
        Ln = lay.l0 + (lay.l1 - lay.l0) * v[m]
        if follow is not None:                  # (the length it had on the halo: Sphere._ring)
            Ln = np.maximum(Ln + follow[m], 5.0 * lay.s)
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

    def _star(self, f, cam, pos, dirs, a, e, big=True, label_left=None):
        """mu -> e + nu + nu at `pos`: a solid prong (the electron) and two dotted ones (the neutrinos).
        label_left = seconds its prong labels still have (None = no labels): they are made with the star and
        taken apart - they do not fade with it."""
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
        if big:                             # (a ring that grows: always the same vertices, sphere.ring)
            ring(f, "r", float(px[0]), float(py[0]), 8 + 54 * (1 - math.exp(-a / 0.5)), 0.9 * math.exp(-a / 0.45), n=40)
        if label_left is not None and label_left > 0.0:     # prong labels, only where they stay whole inside the body column
            x0, x1 = self.lay.body
            for k, (word, g) in enumerate((("E-", 1.0), ("NU", 0.7), ("NU", 0.7)), start=1):
                x, y = float(px[k]) + 8, float(py[k]) + 5
                if x0 <= x <= x1 - 20:
                    with f.build(B.io(a - 0.08, label_left, out=0.2, span=0.12), (x - 2.0, y - 14.0, x + 24.0, y + 4.0),
                                 wave=0.03, marks=False, cps=30.0, key=k):
                        f.text("w", x, y, word, size=L.T_MICRO, alpha=g)

    def _impacts(self, f, cam, t, ctx):
        """The muons of this scene: leader from the detector, a red track that stops on the body, a star."""
        b = self.body
        clip = self.lay.clip
        for ch in self.chunks:
            a = t - ch["t"]
            if a < 0:
                break
            if a > 3.2:
                continue
            S = (ch["c"] * b.rho(ch["c"][None], self.pose)[0]).astype(np.float32)
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
            # the labels of the star of a hit: 2.2 s, less when the next hit comes (then they are taken apart)
            self._star(f, cam, S, ch["star"], a, ch["e"], big=not ch["echo"],
                       label_left=None if ch["echo"] else min(2.2 - a, ch["next"] + 0.2 - t))
            f.set_clip()
            if a < 1.7:                     # leader: detector -> bus -> down into the track (may pass behind a tower)
                tw = ctx.towers[ch["key"]]
                ox, oy = tw.det
                yb = BUS_Y[ch["key"]]
                al = (0.4 if ch["echo"] else 1.0) * math.exp(-a / 0.5)
                if draw_leader(f, "r", [ox, ox, float(px[0]), float(px[0])], [oy - 14, yb, yb, float(py[0])], 1.1 * al,
                               a / 0.1) >= 3:                           # drawn in 0.1 s, from the detector
                    f.dots("r", [float(px[0])], [yb], 3.0, 1.4 * al)

    # ------------------------------------------------------------------- HUD
    def _callouts_d(self, f, t, e_txt, frac, yaw):
        fr = int(t * 30)
        self._view_block(f, "VIEW 01 // BODY // LOSING MASS",
                         [f"ORBIT {math.degrees(yaw) % 360:05.1f} DEG", f" N/N0 {frac:.3f}"], e_txt, fr, t=t)
        # bottom right: the last decay - its tag is made on the hit and taken apart four seconds later (or as
        # soon as the next hit comes)
        got = one_tag(self.prim_t, t, 4.0)
        if got:
            ch = self.prim[got[0]]
            lines = [f"{L.NAMES[ch['key']]}  E {ch['e']:.3f}", f"E_E {ch['ee'] * 52.8:5.2f} MEV", f"CHUNK -{ch['n']:03d} NODES"]
            self._corner(f, 1, "MU- > E- NU NU", [(er(ln, e_txt, 94 + k, fr), "w") for k, ln in enumerate(lines)],
                         tag_size=L.T_SMALL, age=got[1], erode=e_txt, key=93, fr=fr)
        self._corner(f, -1, "DISINTEGRATING",
                     [(er(f"HITS {len(self.chunks_before(t)):03d}", e_txt, 97, fr), "w"),
                      (er(f"REMAINING {100 * frac:05.2f} %", e_txt, 98, fr), "w")],
                     age=self._age0("corner", t), tag_age=self._age("corner", t, e_txt), erode=e_txt, key=96, fr=fr)

    def _integrity(self, f, t, ctx, e_txt):
        """N/N0 against time, under the tomogram: the decay law (dotted) and what is really left of the body
        (staircase)."""
        lay = self.lay
        if not lay.tomo:
            return
        fr = int(t * 30)
        x0, x1 = lay.tomo
        px0, px1, py0, py1 = x0 + 60, x1 - 8, Y_LOW + 34, Y_BASE
        X = lambda tt: px0 + (np.asarray(tt, np.float64) - T0) / (T1 - T0) * (px1 - px0)
        Y = lambda q: py1 - np.asarray(q, np.float64) * (py1 - py0)
        rect = (x0, Y_LOW - 26.0, x1, Y_BASE + 34.0)
        with self._blk(f, "profile", t, rect, erode=e_txt):             # its boxes and rules: header, axes, ticks
            self._low_panel(f, lay.tomo, "INTEGRITY // N/N0 // DECAY LAW EXP(-T/TAU)", e_txt, 101, fr)
            f.segments("w", [px0, px0], [py1, py1], [px0, px1], [py0, py1], 0.65)
            for q in (0.0, 0.5, 1.0):
                f.segments("w", [px0], [float(Y(q))], [px0 - 8], [float(Y(q))], 0.8)
            hud.ruler(f, px0, px1, py1 + 2, T0, T1, 0.5, 2.0, inten=0.6)
        # what is really left
        n = self.body.n
        ts = np.linspace(T0, min(t, T1), max(2, int((min(t, T1) - T0) * 12)))
        left = 1.0 - np.searchsorted(self.rel_sorted, ts, side="right") / n
        xc, yc = float(X(min(t, T1))), float(Y(left[-1]))
        with self._blk(f, "profile", t, rect):                          # its lettering, the decay law, the hits, the cursor
            for q in (0.0, 0.5, 1.0):
                f.text("w", x0, float(Y(q)) + 5, er(f"{q:.1f}", e_txt, 102, fr), size=L.T_MICRO, alpha=0.7)
            for tv in np.arange(T0 + 1.0, T1, 2.0 if px1 - px0 >= 420 else 4.0):
                f.text("w", float(X(tv)) + 3, py1 + 28, er(sd.tc(tv)[:5], e_txt, 103, fr), size=L.T_MICRO, alpha=0.7)
            tt = np.linspace(T0, T1, 150)
            keep = hash01(np.arange(150), 105) > e_txt
            f.dots("w", X(tt)[keep], Y(remaining(tt))[keep], 1.3, 0.7)
            for ch in self.chunks:
                if ch["t"] > t:
                    break
                if not ch["echo"] and hash01(ch["id"], 106) > e_txt:      # (the tick of a new hit grows)
                    f.rects("r", float(X(ch["t"])), py1 - (4 + 22 * ch["e"]) * float(pop(t - ch["t"])),
                            float(X(ch["t"])) + 3, py1, 0.95)
            f.segments("r", [xc], [py0 - 4], [xc], [py1 + 4], 1.1, width=L.LW)
            f.dots("r", [xc], [yc], 4.0, 1.6)
            f.tag("r", max(min(xc + 10, px1 - 112), px0 + 4), max(yc - 12, py0 + 22), f"N/N0 {left[-1]:.3f}", size=L.T_MICRO, pad=3)
            f.text("w", px1, py0 + 12, er(f"TAU {TAU:.1f} S", e_txt, 108, fr), size=L.T_MICRO, alpha=0.8, anchor="rs")
        # the staircase: traced when the panel is made, un-traced when the lettering is all but gone
        with f.build(B.io(self._age0("profile", t) - 0.3, self.t_line_gone - t, out=0.6, span=0.45), rect, flow="lr",
                     wave=0.0, line=0.45, marks=False, key=101):
            f.polyline("w", X(ts), Y(left), 1.0, width=L.LW_BOLD)

    def _left_d(self, f, t, e_txt):
        if not self.lay.info:
            return
        fr = int(t * 30)
        b = self.body
        nodes, edges, struts, sticks, tracks = self._counts(t)
        rows_ = [f"NODES     {nodes:04d} / {b.n:04d}", f"EDGES     {edges:04d} / {len(b.ea):04d}",
                 f"STRUTS    {struts:04d} / {len(b.sa):04d}", f"STICKS    {sticks:04d} / {N_BINS:04d}",
                 f"TRACKS    {tracks:04d} / {len(self.old):04d}", "MU- -> E- + NU + NU", "TAU_MU    2.197 US"]
        x, w, y = self._info_block(f, "DISINTEGRATE", rows_, red=(5,), erode=e_txt, fr=fr, key=111, t=t)
        past = self.chunks_before(t)[::-1]
        nr = 24.0 * (1.0 - float(smoothstep(T0 + 6.0, T_ALMOST + 0.8, t)))      # the log empties, from the bottom
        n_rows = int(math.ceil(nr))
        lines = []
        for row, ch in enumerate(past[:n_rows]):
            m = int(ch["t"] // 60)
            age = t - ch["t"]
            red = age < 1.0 or (not ch["echo"] and ch["n"] > 80)
            fields = [f"{m:02d}:{ch['t'] - 60 * m:04.1f}", f"{ch['key']}{'e' if ch['echo'] else ' '}", f"{ch['e']:.2f}",
                      f"{ch['ee'] * 52.8:7.2f}", f" -{ch['n']:03d}"]
            going = 1.0 - (nr - row) if row == n_rows - 1 else 0.0  # its last row falls apart before it is gone
            lines.append((fields, "r" if red else "w", 0.95 if row < 2 or red else 0.62, age, going))
        self._log_block(f, x, w, y, "DECAY_LOG // LIVE", ["TIME   ", "D ", "E   ", "E_E MEV", "NODES"], lines,
                        erode=e_txt, fr=fr, key=120, t=t)

    def _michel(self, f, t, e_txt):
        """Spectrum of the decay electrons seen so far, against the Michel shape (under the detectors)."""
        if not self.lay.spec:
            return
        fr = int(t * 30)
        x0, x1 = self.lay.spec
        x1 = min(x1, x0 + 460.0)
        es = [ch["ee"] for ch in self.chunks if ch["t"] <= t] + [tr["ee"] for tr in self.old if tr["decay"] <= t]
        edges = np.linspace(0, 1, 23)
        cnt, _ = np.histogram(np.array(es), edges) if es else (np.zeros(22), None)
        bw = (x1 - x0 - 8) / 22
        xs = x0 + 4 + np.arange(22) * bw
        yb, hmax = Y_BASE, 150.0
        top = max(float(np.max(cnt)), 1.0)
        rect = (x0, Y_LOW - 26.0, x1, Y_BASE + 34.0)
        with self._blk(f, "spec", t, rect, erode=e_txt):                # its boxes and rules
            self._low_panel(f, self.lay.spec, "E_ELECTRON // MICHEL // MEV", e_txt, 141, fr, cap=460.0)
            hud.ruler(f, x0 + 4, x0 + 4 + 22 * bw, yb + 2, 0, 52.8, 2.4, 12.0 if x1 - x0 >= 300 else 24.0,
                      fmt=lambda v: er(f"{v:.0f}", e_txt, 144, fr), inten=0.6, lab_dy=26)
        with self._blk(f, "spec", t, rect):                             # its bars and the Michel shape
            keep = (cnt > 0) & (hash01(np.arange(22), 142) > e_txt)
            f.rects("w", xs[keep], yb - hmax * cnt[keep] / top, xs[keep] + bw - 4, yb, 0.95)
            xx = np.linspace(0, 1, 60)
            kk = hash01(np.arange(60), 143) > e_txt
            f.dots("r", (x0 + 4 + xx * 22 * bw)[kk], (yb - hmax * michel(xx))[kk], 1.5, 1.1)

    def _bottom_d(self, f, t, ctx, e_txt, frac):
        fr = int(t * 30)
        nodes, edges, struts, sticks, tracks = self._counts(t)
        left = nodes / self.body.n          # (0.995 and more reads 1.0: ".2f" would round it to "1.00", shown as ".00")
        frac_ = "1.0" if left >= 0.995 else (f"{left:.3f}"[1:] if left < 0.1 else f"{left:.2f}"[1:])
        cols = [("NODES", f"{nodes:04d}", "w"), ("EDGES", f"{edges:04d}", "w"), ("STICKS", f"{sticks:03d}", "w")]
        # (N/N0 has the smallest panel of the band; with only two panels it is the fourth counter of the first)
        self._numbers_panel(f, "REMAINING", cols if self.lay.p_one else cols + [("N/N0", frac_, "r")], erode=e_txt, fr=fr,
                            t=t)
        self._barcode_panel(f, t, ctx, erode=e_txt, fr=fr)
        self._single_panel(f, "N/N0", frac_, "r", erode=e_txt, fr=fr, t=t)
