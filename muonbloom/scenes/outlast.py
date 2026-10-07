"""OUTLAST - time dilation, the reason muons reach the ground.   Scene 8, 07:34 - 08:32.

A muon lives 2.197 us. Even at almost the speed of light it should decay after ~0.66 km, yet
muons born 15 km up reach the ground: their clocks run slow (time dilation, factor gamma) - or,
in their own frame, the atmosphere is contracted by gamma.

The study the user liked, re-laid vertically for the wall and its towers: altitude is the
vertical axis, 15 km at the top of the view, the ground just above the bottom band. The muons
FALL down the wall, in two blocks of 64 lanes:
  CLASSICAL (white)     no dilation: every run dies within the first kilometres
  RELATIVISTIC (red)    the same 64 muons with dilated clocks: each run gets further
Four runs, gamma = 2, 6, 14, 29 (a slow crescendo, 13 s each). Every run opens on a MICRO view
of the first 2.5 km (where the classical population dies) and zooms out to the whole atmosphere.
The last run starts on "...a million worlds to reach this one" and lands exactly on
"Perhaps, it Outlasted the star that made it."  ->  OUTLAST; then the survivors keep raining,
faster and faster, into scene 9.

Nothing has a fixed x. The towers stand in front of the wall and nobody knows yet where, how wide
or how tall: everything is placed from ctx.cols (the usable columns between the towers):
  * the two lane blocks take the two widest columns (a very wide bay is split and hosts both),
    their altitude rulers on their outer sides;
  * the run card (Lorentz factor, parameters, decay stream) and the muon-frame card (proper-time
    clock against its 2.197 us lifetime, contracted atmosphere, survival curves) take what is left,
    scaled to the width they get, and are dropped rather than put behind a tower;
  * the ledger of who reaches the ground gets a column of its own if one is left, else it sits in
    the void the classical muons leave;
  * arrival grid / decay barcode / counts go to ctx.slots["panels"], widest first.

Nothing that shows data fades in or pops in (build.py): the furniture of the scene - strip, run card,
muon-frame card, the rulers and labels of the two lane blocks, the bottom panels - is CONSTRUCTED during the
first second and a half; what belongs to a run (its number, its Lorentz factor and parameters, the decay
lengths, the contracted atmosphere, the theoretical curves) is rewritten when the run starts; what comes
and goes with the zoom of a run (the ledger in the void of the classical block, the ground line, the MICRO
tag, the read-outs of the classical deaths) is built when it comes and taken apart when it goes. The lanes,
the muons, the red front and the rain are the image: they keep their own intensities.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import hud
from .. import layout as L
from .. import showdata as sd
from .. import engine as E
from ..engine import CHAR_W, hash01, smoothstep
from ..show import Scene

C = 299792458.0
TAU = 2.197e-6
ATM = 15000.0
GAMMAS = [2.0, 6.0, 14.0, 29.0]
N = 64
M_MU = 0.10566                                # GeV

T0, T1 = 454.0, 512.0                         # the scene on the sheet
T_TRAVEL = 9.0                                # s for the front to fall 15 km
RUN_GAP = 13.0                                # s between two launches
ACC = 1.6                                     # accelerando exponent of the front
MICRO = 2500.0                                # micro view range (m)
ZOOM_IN = (2.6, 4.4)                          # run time: the view opens from the micro view to the whole atmosphere
ZOOM_BACK = (RUN_GAP - 1.1, RUN_GAP - 0.1)    # ... and comes back to the first kilometres for the next run
Y_TOP, Y_GND = 300.0, 1150.0                  # 15 km .. ground on the wall
CARD_W = 430.0                                # a data card never gets wider than this


def _f(x):
    return np.clip(x, 0.0, 1.0) ** ACC


def _finv(y):
    return np.clip(y, 0.0, 1.0) ** (1.0 / ACC)


def _ss_inv(v):
    """u in 0..1 with smoothstep(0, 1, u) = v."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if m * m * (3.0 - 2.0 * m) < v:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def _fit(options, width, size, pad=8.0):
    """First of the strings that fits `width` px at `size` (the last one if none does)."""
    for s in options:
        if len(s) * size * CHAR_W + pad <= width:
            return s
    return options[-1]


def _subcols(cols, unit=620.0, gutter=44.0, min_w=200.0):
    """Usable columns cut into blocks: a bay much wider than `unit` hosts two or three of them."""
    out = []
    for a, b in cols:
        w = b - a
        if w < min_w:
            continue
        n = int(np.clip(round(w / unit), 1, 3))
        sw = (w - (n - 1) * gutter) / n
        out += [(a + k * (sw + gutter), a + k * (sw + gutter) + sw) for k in range(n)]
    return out


class Outlast(Scene):
    name = "outlast"
    towers = "auto"

    def __init__(self, ctx, seed=21):
        super().__init__(ctx)
        rng = np.random.default_rng(seed)
        self.runs = []
        for g in GAMMAS:
            b = math.sqrt(1.0 - 1.0 / g ** 2)
            Lc = b * C * TAU
            Lr = g * Lc
            u = rng.uniform(1e-9, 1.0, N)
            dc = -np.log(u) * Lc
            dr = -np.log(u) * Lr                  # the same muons, with dilated clocks
            spd = rng.uniform(0.975, 1.025, N)
            self.runs.append(dict(g=g, b=b, Lc=Lc, Lr=Lr, dc=dc, dr=dr, spd=spd,
                                  ang=rng.uniform(-1.0, 1.0, (N, 3)), t_lab=ATM / (b * C)))
        # air strata: density ~ exp(-h / 8.4 km), denser towards the ground
        uu = rng.uniform(0, 1, 340)
        h = -8400.0 * np.log(1 - uu * (1 - math.exp(-ATM / 8400.0)))
        self.strata = np.sort(ATM - h)
        # the last run lands on the voice
        self.t_out = sd.said("Perhaps, it Outlasted", 502.433)
        self.starts = [self.t_out - T_TRAVEL - (len(GAMMAS) - 1 - p) * RUN_GAP for p in range(len(GAMMAS))]
        self.final = [(self._state(run, 1e9, 0), self._state(run, 1e9, 1)) for run in self.runs]
        self.rain_h = hash01(np.arange(N), 77)
        # run times at which things come into view with the zoom-out of a run, and leave with its zoom-back:
        # the ground line (it has to be inside the view) and the ledger in the void of the classical block
        zg = math.log((ATM / MICRO) / (1.0 + 12.0 / (Y_GND - Y_TOP))) / math.log(ATM / MICRO)
        at = lambda v: (ZOOM_IN[0] + (ZOOM_IN[1] - ZOOM_IN[0]) * _ss_inv(v),
                        ZOOM_BACK[0] + (ZOOM_BACK[1] - ZOOM_BACK[0]) * _ss_inv(1.0 - v))
        self.r_gnd = at(zg)
        self.r_led = at(0.3)
        self._layout(ctx)

    # ------------------------------------------------------------------ layout
    def _layout(self, ctx):
        """Place everything from the usable columns between the towers (see the module docstring)."""
        cols = [c for c in getattr(ctx, "cols", [])] or [(L.FX0 + 64.0, L.FX1 - 64.0)]
        sub = _subcols(cols) or [max(cols, key=lambda c: c[1] - c[0])]
        while len(sub) < 2:                       # one usable bay only: it hosts both lane blocks
            k = int(np.argmax([b - a for a, b in sub]))
            a, b = sub.pop(k)
            m = 0.5 * (a + b)
            sub[k:k] = [(a, m - 22.0), (m + 22.0, b)]
        best, pair = -1.0, (0, 1)
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                score = min(sub[i][1] - sub[i][0], sub[j][1] - sub[j][0]) + (60.0 if j == i + 1 else 0.0)
                if score > best:
                    best, pair = score, (i, j)
        self.blk_c, self.blk_r = sub[pair[0]], sub[pair[1]]
        rest = [s for k, s in enumerate(sub) if k not in pair]
        run = next((s for s in rest if s[1] - s[0] >= 240.0), None)
        frame = next((s for s in rest[::-1] if s is not run and s[1] - s[0] >= 300.0), None)
        led = [s for s in rest if s is not run and s is not frame and s[1] - s[0] >= 420.0]
        self.col_run = (run[0], min(run[1], run[0] + CARD_W)) if run else None
        self.col_frame = (frame[0], min(frame[1], frame[0] + CARD_W)) if frame else None
        self.col_led = max(led, key=lambda s: s[1] - s[0]) if led else None
        # lanes: the classical block keeps its ruler on its left, the relativistic one on its right
        a, b = self.blk_c
        lab = 108.0 if b - a >= 520.0 else 94.0
        self.rul_c = a + 14.0
        self.xc = np.linspace(a + 14.0 + lab, b - 22.0, N)
        a, b = self.blk_r
        self.rul_r = b - 14.0
        self.xr = np.linspace(a + 22.0, b - 14.0 - lab, N)
        self.pitch = float(min(self.xc[1] - self.xc[0], self.xr[1] - self.xr[0]))
        # bottom panels, widest first: arrival grid, decay barcode, counts
        panels = sorted(ctx.slots["panels"], key=lambda p: p[0] - p[1])
        self.pan_grid = panels[0] if panels and panels[0][1] - panels[0][0] >= 280.0 else None
        rest_p = panels[1:] if self.pan_grid else panels
        if len(rest_p) >= 2:
            self.pan_bar, self.pan_cnt = rest_p[0], rest_p[1]
        elif len(rest_p) == 1:
            self.pan_bar, self.pan_cnt = None, rest_p[0]
        else:
            self.pan_bar = self.pan_cnt = None
        if self.pan_cnt and self.pan_cnt[1] - self.pan_cnt[0] < 120.0:
            self.pan_cnt = None

    def _clip(self, f, which, readouts=False):
        """Clip to one lane block (never over the towers next to it)."""
        blk, xs = (self.blk_c, self.xc) if which == 0 else (self.blk_r, self.xr)
        f.set_clip(max(xs[0] - 14.0, blk[0] - 4.0), Y_TOP - 6.0, min(xs[-1] + 14.0, blk[1] + 4.0), Y_GND + 26.0)

    # ------------------------------------------------------------------ state
    def _state(self, run, r, which):
        """Per-lane head distance, decay flags and times at run time r."""
        dec = run["dc"] if which == 0 else run["dr"]
        spd = run["spd"]
        D = ATM * _f(r * spd / T_TRAVEL)
        r_dec = T_TRAVEL / spd * _finv(dec / ATM)
        r_gnd = T_TRAVEL / spd
        dead = (D >= dec) & (dec < ATM)
        ground = (D >= ATM) & (dec >= ATM)
        head = np.minimum(D, dec)
        return dict(D=D, dec=dec, head=head, dead=dead, ground=ground, r_dec=r_dec, r_gnd=r_gnd)

    def _run_at(self, t):
        p = int(np.clip(np.searchsorted(self.starts, t, side="right") - 1, 0, len(self.runs) - 1))
        return p, t - self.starts[p]

    def _zoom(self, p, r):
        """0 = micro view of the first 2.5 km, 1 = the whole atmosphere."""
        z = float(smoothstep(ZOOM_IN[0], ZOOM_IN[1], r))
        if p < len(self.runs) - 1:
            z *= 1.0 - float(smoothstep(ZOOM_BACK[0], ZOOM_BACK[1], r))
        return z

    # ------------------------------------------------------------------ draw
    def draw(self, f, t, ctx):
        t = float(np.clip(t, T0 - 2.0, T1 + 4.0))
        p, r = self._run_at(t)
        run = self.runs[p]
        z = self._zoom(p, r)
        dmax = MICRO * (ATM / MICRO) ** z
        st = [self._state(run, r, 0), self._state(run, r, 1)]
        Y = lambda d: Y_TOP + np.asarray(d, np.float64) / dmax * (Y_GND - Y_TOP)

        for which, xs in ((0, self.xc), (1, self.xr)):
            self._clip(f, which)
            self._draw_strata(f, xs, Y, dmax)
        self._draw_lanes(f, run, r, st, Y, dmax, z, t)
        f.set_clip()
        self._draw_lane_hud(f, t, run, r, st, Y, dmax, z, p)
        self._draw_ledger(f, t, p, r, st, z)
        self._draw_run_card(f, t, run, r, p, st)
        self._draw_muon_frame(f, t, run, r, p)
        self._draw_survival(f, t, run, r, p, st)
        self._draw_strip(f, t, p, r)
        self._draw_bottom(f, t, ctx, run, r, st, p)
        return {}

    # --- the lanes ------------------------------------------------------------------
    def _draw_strata(self, f, xs, Y, dmax):
        s = self.strata[self.strata <= dmax]
        if E.WALL:          # the wall rule: the strata are not drawn. Faint, they do not land; as full white dashes
            return          # they hid the red lanes coming down from the top (the user, 2026-10-07: no line gradients)
        ys = Y(s)
        f.segments("w", np.full_like(ys, xs[0] - 8), ys, np.full_like(ys, xs[-1] + 8), ys, 0.05)

    def _draw_lanes(self, f, run, r, st, Y, dmax, z, t):
        micro = 1.0 - z
        yg = float(Y(ATM))
        bar = float(np.clip(0.36 * self.pitch, 1.5, 4.0))
        for which, lay, xs in ((0, "w", self.xc), (1, "r", self.xr)):
            self._clip(f, which)
            S = st[which]
            head, dead = S["head"], S["dead"]
            a = r - S["r_dec"]
            base = 0.5 if which else 0.45
            inten = np.where(dead, 0.3 + 0.4 * np.exp(-np.maximum(a, 0) / 0.5), base)
            inten = np.where(S["ground"], 0.8 + 0.6 * np.exp(-np.maximum(r - S["r_gnd"], 0) / 0.6), inten)
            vis = head > 1.0
            n = int(vis.sum())
            if n:
                if which == 0:              # (the wall rule: the white lanes at the level and weight that land)
                    inten = E.wl(inten)
                f.segments(lay, xs[vis], np.full(n, Y_TOP), xs[vis], Y(head[vis]), inten[vis],
                           width=E.ww(1.3) if which == 0 else 1.3)
            dm = dead & vis                         # where each one died: a small bar stays
            if dm.any():
                yd = Y(head[dm])
                f.segments(lay, xs[dm] - bar, yd, xs[dm] + bar, yd, 0.75, width=L.LW)
            alive = ~dead & ~S["ground"] & vis & (head < dmax)
            f.dots(lay, xs[alive], Y(head[alive]), 2.0 + 0.7 * micro, 1.5)
            if which == 1:
                f.dots("w", xs[alive], Y(head[alive]), 0.8, 0.8)
            # decay bursts: e (solid) + two neutrinos (faint)
            bm = dead & (a >= 0) & (a < 1.8)
            if bm.any():
                grow = np.clip(a[bm] / 0.16, 0, 1)
                fade = np.exp(-a[bm] / 0.7)
                cx, cy = xs[bm], Y(head[bm])
                ang = run["ang"][bm]
                scale = 1.0 + 1.6 * micro
                for k, (ln, it) in enumerate(((20.0, 0.95), (13.0, 0.35), (13.0, 0.35))):
                    th = ang[:, k] * (1.1 if k == 0 else 2.6) + (0 if k == 0 else math.pi * (k - 1.5))
                    Ln = ln * scale * grow
                    f.segments("w", cx, cy, cx + np.sin(th) * Ln, cy + np.cos(th) * Ln, E.wl(it) * fade, width=E.ww(1.2))
                f.dots(lay, cx, cy, 1.7 + 0.8 * micro, 1.5 * fade)
            # arrivals at the ground: a comb growing under the ground line
            gm = S["ground"]
            if gm.any():
                ng = int(gm.sum())
                ag = r - S["r_gnd"][gm]
                f.segments(lay, xs[gm], np.full(ng, yg + 4), xs[gm], np.full(ng, yg + 20), 1.0, width=1.6)
                fl = ag < 0.6
                if fl.any():
                    f.dots("w", xs[gm][fl], np.full(int(fl.sum()), yg), 3.2, 1.6 * (1 - ag[fl] / 0.6))
        # per-lane readouts of the classical deaths (micro view), kept inside the block: each one is decoded
        # when its muon dies, and they are taken apart when the view starts to open (no fade with the zoom)
        r_out = 0.5 * (ZOOM_IN[0] + ZOOM_IN[1])
        if r < r_out:
            self._clip(f, 0)
            S = st[0]
            step = max(7, int(math.ceil(82.0 / max(self.pitch, 1.0))))
            y_tag = float(Y(run["Lc"])) + 18.0            # the decay-length tag sits there, on the right
            for i in range(0, N, step):
                a = r - S["r_dec"][i] - 0.05
                if S["dead"][i] and a > 0:
                    x, y = float(self.xc[i]), float(Y(S["head"][i]))
                    if x + 13 + 70 > self.blk_c[1]:       # no room on its right: no readout (never over a tower)
                        continue
                    if abs(y - y_tag) < 24.0 and x + 90 > self.xc[-1] - 240.0:
                        continue
                    f.text("w", x + 13, y + 5, B.resolve(f"{S['head'][i] / 1000:.3f} KM", B.io(a, r_out - r, out=0.3, span=0.45),
                                                         cps=70.0, key=i, spin=0.25), size=L.T_MICRO, alpha=0.8)
        # after OUTLAST: the survivors keep raining, faster and faster (the build into scene 9)
        tau = t - (self.t_out + 1.0)
        if tau > 0:
            self._clip(f, 1)
            gm = self.final[-1][1]["ground"]
            k = (1.9 - 0.3) / 8.5
            ph = (self.rain_h[gm] + 0.3 * tau + 0.5 * k * tau * tau) % 1.0
            amt = float(smoothstep(0.0, 1.5, tau))
            y = Y_TOP + ph ** 1.4 * (Y_GND - Y_TOP)
            x = self.xr[gm]
            tail = 40.0 + 90.0 * ph
            f.segments("r", x, y - tail, x, y, 0.0, 1.2 * amt, width=1.8)
            f.dots("r", x, y, 2.6, 1.6 * amt)
            f.dots("w", x, y, 1.0, 0.9 * amt)
            land = np.exp(-ph / 0.05)
            f.dots("w", x, np.full(len(x), Y_GND), 3.4, 1.5 * amt * land)

    def _micro_age(self, p, r, sa):
        """Build age of the MICRO tag: made with the scene, taken apart when a run zooms out, made again when
        the view comes back to the first kilometres for the next run."""
        if p < len(self.runs) - 1 and r >= ZOOM_BACK[0] + 0.1:
            return r - (ZOOM_BACK[0] + 0.1)
        return B.io((sa - 0.6) if p == 0 else 99.0, ZOOM_IN[0] + 1.35 - r, out=0.4, span=0.5)

    def _draw_lane_hud(self, f, t, run, r, st, Y, dmax, z, p):
        sa = t - T0                                           # the rulers and labels are built with the scene
        own = None if p == 0 else r                           # what belongs to a run is rewritten when it starts
        last = p == len(self.runs) - 1
        yg = float(Y(ATM))
        alt_bot = (ATM - dmax) / 1000.0
        ppk = (Y_GND - Y_TOP) / (dmax / 1000.0)              # px per km
        if ppk > 220:
            minor, major, fmt = 0.05, 0.25, (lambda v: f"{v:05.2f} KM")
        elif ppk > 100:
            minor, major, fmt = 0.05, 0.5, (lambda v: f"{v:04.1f} KM")
        else:
            minor, major, fmt = 0.1, 1.0, (lambda v: f"{v:02.0f} KM")
        wc, wr = self.xc[-1] - self.xc[0], self.xr[-1] - self.xr[0]
        Lc, Lr, g = run["Lc"], run["Lr"], run["g"]
        yt = Y_TOP - 46
        label = (_fit(["CLASSICAL // NO TIME DILATION", "CLASSICAL", "CLS"], wc + 20, L.T_LABEL),
                 _fit([f"RELATIVISTIC // CLOCKS SLOWED x{g:.1f}", f"RELATIVISTIC x{g:.1f}", f"REL x{g:.1f}"], wr + 20,
                      L.T_LABEL))
        decay = ((Lc, [f"BETA*C*TAU {Lc / 1000:.3f} KM", f"L {Lc / 1000:.3f} KM"]),
                 (Lr, [f"GAMMA*BETA*C*TAU {Lr / 1000:.2f} KM", f"L {Lr / 1000:.2f} KM"]))
        for which, xs, xr_, blk, right, lay in ((0, self.xc, self.rul_c, self.blk_c, True, "w"),
                                                (1, self.xr, self.rul_r, self.blk_r, False, "r")):
            wb = xs[-1] - xs[0]
            rect = (blk[0] - 2.0, Y_TOP - 74.0, blk[1] + 2.0, Y_GND + 12.0)
            with f.build(sa - 0.25 - 0.15 * which, rect, flow="tb", wave=0.45, key=30 + which):
                # the altitude ruler of the block, its top rule, its counters
                f.set_clip(blk[0] - 2, Y_TOP - 8, blk[1] + 2, Y_GND + 10)
                f.segments("w", [xr_], [Y_TOP], [xr_], [Y_GND], E.wl(0.7), width=E.ww(1.0))
                hud.vruler(f, xr_, Y_TOP, Y_GND, 15.0, alt_bot, minor, major, fmt=fmt, right=right, inten=0.8,
                           size=L.T_MICRO)
                f.set_clip()
                f.segments("w", [xs[0] - 10], [Y_TOP], [xs[-1] + 10], [Y_TOP], E.wl(0.75), width=L.LW)
                S = st[which]
                alive = int((~S["dead"]).sum())
                gnd = int(S["ground"].sum())
                txt = _fit([f"ALIVE {alive:03d}/{N:03d}   GROUND {gnd:03d}", f"ALIVE {alive:03d}  GND {gnd:03d}",
                            f"{alive:03d} / {gnd:03d}"], wb + 20, L.T_SMALL)
                f.text(lay, xs[0] - 10, Y_TOP - 12, txt, size=L.T_SMALL, alpha=0.95)
                if which == 0:
                    f.tag("w", xs[0] - 10, yt, label[0], size=L.T_LABEL, pad=4)
                with f.build(own, rect, flow="tb", wave=0.3, marks=False, key=40 + which):
                    if which == 1:                            # the label carries the Lorentz factor of the run
                        f.tag("r", xs[0] - 10, yt, label[1], size=L.T_LABEL, pad=4)
                    Ld, opts = decay[which]                   # decay-length marker of the run
                    if Ld < dmax:
                        y = float(Y(Ld))
                        k = np.arange(xs[0] - 8, xs[-1] + 8, 14.0)
                        f.segments(lay, k, np.full_like(k, y), k + 7, np.full_like(k, y), 0.95, width=L.LW)
                        f.tag(lay, xs[-1] + 8, y + 26, _fit(opts, wb, L.T_MICRO), size=L.T_MICRO, pad=3, anchor="rs")
                    else:
                        tail = " > GROUND" if dmax >= ATM - 1 else " v"
                        f.tag(lay, xs[-1] + 8, Y_GND - 14, _fit([o + tail for o in opts], wb, L.T_MICRO), size=L.T_MICRO,
                              pad=3, anchor="rs")
        # ground line: it comes into view at the end of the zoom-out - drawn by a pen, its tags made - and is
        # taken apart before the view goes back to the first kilometres
        if yg < Y_GND + 12:
            ag = (r - self.r_gnd[0]) if last else B.io(r - self.r_gnd[0], self.r_gnd[1] - r, out=0.25, span=0.7)
            with f.build(ag, (self.xc[0] - 130.0, yg - 10.0, self.xr[-1] + 130.0, yg + 46.0), flow="lr", wave=0.3,
                         line=0.25, marks=False, key=46):
                for xs in (self.xc, self.xr):
                    f.segments("w", [xs[0] - 12], [yg], [xs[-1] + 12], [yg], 1.0, width=L.LW_BOLD)
                f.tag("w", self.xc[0] - 16, yg + 36, "GROUND 0 M", size=L.T_MICRO, pad=3, anchor="rs")
                f.tag("w", self.xr[-1] + 16, yg + 36, "GROUND 0 M", size=L.T_MICRO, pad=3)
        # the front (image): its altitude and its lab time ride the two rulers - two tags made when the run
        # starts and taken apart when the front reaches the ground
        frac = float(_f(min(r, T_TRAVEL) / T_TRAVEL))
        dfront = ATM * frac
        if 0 < r < T_TRAVEL and dfront <= dmax:
            y = float(Y(dfront))
            for xs in (self.xc, self.xr):
                f.segments("r", [xs[0] - 12], [y], [xs[-1] + 12], [y], 1.1, width=L.LW)
            af = B.io(r, T_TRAVEL - r, out=0.25, span=0.4)
            B.tag(f, "r", self.xr[-1] + 16, y + 6, f"{frac * run['t_lab'] * 1e6:06.3f} US", af, size=L.T_MICRO, pad=3,
                  cps=120.0, key=47)
            B.tag(f, "r", self.xc[0] - 16, y + 6, f"{(ATM - dfront) / 1000:06.3f} KM", af, size=L.T_MICRO, pad=3,
                  anchor="rs", cps=120.0, key=48)
        # the MICRO tag comes and goes with the zoom: made and taken apart, never faded
        full = f"MICRO // FIRST {MICRO / 1000:.1f} KM  x{ATM / MICRO:.0f}"
        room = wc + 20 - 26 * L.T_SMALL * CHAR_W - 30
        B.tag(f, "w", self.xc[-1] + 10, Y_TOP - 12, _fit([full, f"MICRO x{ATM / MICRO:.0f}"], room, L.T_MICRO),
              self._micro_age(p, r, sa), size=L.T_MICRO, pad=3, anchor="rs", cps=90.0, key=49)

    # --- who reaches the ground: a ledger ----------------------------------------------
    def _draw_ledger(self, f, t, p, r, st, z):
        own = self.col_led is not None
        last = p == len(self.runs) - 1
        if own:                                   # a column of its own: built with the scene, always there
            x0, x1 = self.col_led
            y0, age = 276.0, t - T0 - 0.5
        else:                                     # in the void the classical muons leave: constructed when the run
            if z < 0.05:                          # zooms out, taken apart when the view goes back (never faded)
                return
            x0, x1 = float(self.xc[0]) + 2, float(self.xc[-1])
            y0 = 706.0
            age = (r - self.r_led[0]) if last else B.io(r - self.r_led[0], self.r_led[1] - r, out=0.45)
            # its dark plate over the lanes opens downwards (and closes when the ledger is taken apart): no fade
            po = float(B.ease(B.lin(age, 0.0, 0.3)))
            f.dim(self.xc[0] - 8, y0 - 30, self.xc[-1] + 8, y0 - 30 + (Y_GND - 8 - (y0 - 30)) * po, 0.0 if E.WALL else 0.2)
        w = x1 - x0
        compact = w < 0.7 * 652.0
        if compact:
            s = min(1.0, w / 392.0)
            cols = ((0, "RUN"), (84, "GAMMA"), (240, "CLS"), (330, "REL"))
        else:
            s = min(1.0, w / 652.0)
            cols = ((0, "RUN"), (84, "GAMMA"), (200, "L_DECAY KM"), (372, "CLASSICAL"), (524, "RELATIVISTIC"))
        if s < 0.45:
            return                                # no room: the arrival grid and the counts still tell it
        fs, fh, pitch = 30 * s, max(13.0, 16 * s), 52 * s
        x = x0
        cx = [c[0] * s for c in cols]
        with f.build(age, (x0 - 18.0, y0 - 24.0, x1 + 6.0, Y_GND - 8.0), flow="tb", wave=0.45, key=70):
            if own:
                self._draw_combs(f, x0, x1, y0 + 316 * s + 120, Y_GND - 6, p, r, st)
            f.tag("w", x, y0, _fit(["AT GROUND // 64 + 64 MUONS LAUNCHED AT 15 KM", "AT GROUND // 64 + 64 MUONS",
                                    "AT GROUND"], w, L.T_MICRO), size=L.T_MICRO, pad=3)
            for dx, name in cols:
                f.text("w", x + dx * s, y0 + 40 * s, name, size=fh, alpha=0.6)
            f.segments("w", [x], [y0 + 52 * s], [x + (392 if compact else 652) * s], [y0 + 52 * s], E.wl(0.5), width=E.ww(1.0))
            for q, run in enumerate(self.runs):
                y = y0 + 98 * s + q * pitch
                cls_x, rel_x = x + cx[-2], x + cx[-1]
                if q > p:
                    f.text("w", x, y, f"{q + 1:02d}", size=fs, alpha=0.25)
                    f.text("w", x + cx[1], y, f"{run['g']:05.2f}", size=fs, alpha=0.25)
                    f.text("w", cls_x, y, "---" if compact else "---/064", size=fs, alpha=0.2)
                    f.text("w", rel_x, y, "---" if compact else "---/064", size=fs, alpha=0.2)
                    continue
                S0, S1 = (st[0], st[1]) if q == p else self.final[q]
                gc, gr = int(S0["ground"].sum()), int(S1["ground"].sum())
                done = q < p or r >= T_TRAVEL
                hot = q == len(self.runs) - 1 and done
                lay = "r" if hot else "w"
                a_row = 1.0 if q == p else 0.7
                f.text(lay, x, y, f"{q + 1:02d}", size=fs, alpha=a_row)
                f.text(lay, x + cx[1], y, f"{run['g']:05.2f}", size=fs, alpha=a_row)
                if not compact:
                    f.text(lay, x + cx[2], y, f"{run['Lr'] / 1000:05.2f}", size=fs, alpha=a_row)
                f.text("w", cls_x, y, f"{gc:03d}" if compact else f"{gc:03d}/064", size=fs, alpha=a_row)
                f.text("r", rel_x, y, f"{gr:03d}" if compact else f"{gr:03d}/064", size=fs,
                       alpha=1.0 if q == p or hot else 0.8)
                if q == p and not done:
                    f.rects("r", x - 14, y - 22 * s, x - 8, y + 2, 1.0)
        # the verdict, on the voice: the tag is made (box pushed out, letters decoded, a two-frame flash),
        # the sentence is decoded beside it
        a = t - self.t_out
        if a >= 0:
            gr = int(self.final[-1][1]["ground"].sum())
            y = y0 + 98 * s + 4 * pitch + 62 * s
            ts = max(40.0, 74 * s)
            B.tag(f, "r", x + 4, y, "OUTLAST", a, size=ts, pad=12 * s, bold=True, cps=30.0, key=71, commit=True)
            l1, l2 = B.resolve(f"{gr} OF {N} OUTLIVE", a, 40.0, 0.3, key=72), B.resolve("THEIR OWN LIFETIME", a, 40.0, 0.7, key=73)
            tx = x + 7 * ts * CHAR_W + 60 * s
            if tx + 18 * L.T_LABEL * CHAR_W <= x1 + 12:           # the sentence beside the tag ...
                f.text("w", tx, y - 34 * s, l1, size=L.T_LABEL, alpha=0.95)
                f.text("w", tx, y - 6 * s, l2, size=L.T_LABEL, alpha=0.95)
                yn = y + 44
            else:                                                 # ... or under it when the block is narrow
                f.text("w", x + 4, y + 40, l1, size=L.T_SMALL, alpha=0.95)
                f.text("w", x + 4, y + 66, l2, size=L.T_SMALL, alpha=0.95)
                yn = y + 96
            tau = a - 1.0
            if tau > 0 and yn < Y_GND - 8:
                k = (1.9 - 0.3) / 8.5
                more = int(np.floor(self.rain_h[self.final[-1][1]["ground"]] + 0.3 * tau + 0.5 * k * tau * tau).sum())
                f.text("r", x + 4, yn, B.resolve(f"STILL ARRIVING  +{more:04d}", tau, 60.0, key=74), size=L.T_SMALL,
                       alpha=0.9)

    def _draw_combs(self, f, x0, x1, y_top, y_bot, p, r, st):
        """One comb per run: how far each of its 64 relativistic muons got (sorted). Red = reached the ground."""
        if y_bot - y_top < 200:
            return
        rh = (y_bot - y_top) / len(self.runs)
        bw = (x1 - x0) / N
        for q, run in enumerate(self.runs):
            yb = y_top + (q + 1) * rh - 10
            hmax = rh - 52
            f.segments("w", [x0], [yb], [x1], [yb], E.wl(0.5 if q <= p else 0.25), width=E.ww(1.0))
            f.tag("r" if q == p else "w", x0, yb - hmax - 12, f"RUN {q + 1:02d} // GAMMA {run['g']:04.1f}",
                  size=L.T_MICRO, pad=3, alpha=1.0 if q <= p else 0.4)
            if q > p:
                continue
            order = np.argsort(-run["dr"])
            S = st[1] if q == p else self.final[q][1]
            done = (S["dead"] | S["ground"])[order]
            gnd = (run["dr"] >= ATM)[order]
            hh = np.clip(run["dr"][order] / ATM, 0.0, 1.0) * hmax
            bx = x0 + np.arange(N) * bw
            for m, lay in ((done & ~gnd, "w"), (done & gnd, "r")):
                if m.any():
                    f.rects(lay, bx[m], yb - np.maximum(hh[m], 2.0), bx[m] + max(bw - 3.0, 2.0), yb, 0.9)
            n_g = int((done & gnd).sum())
            f.text("r" if n_g else "w", x1, yb - hmax - 8, f"{n_g:03d}/064 AT GROUND", size=L.T_MICRO, alpha=0.9,
                   anchor="rs")

    # --- the run card ----------------------------------------------------------------------
    def _draw_run_card(self, f, t, run, r, p, st):
        if self.col_run is None:
            return
        x0, x1 = self.col_run
        w = x1 - x0
        g, b = run["g"], run["b"]
        sa = t - T0                               # the card is built with the scene ...
        own = None if p == 0 else r               # ... and what belongs to a run is rewritten when the run starts
        rect = (x0 - 8.0, 238.0, x1 + 8.0, 676.0)
        hot = "r" if p == len(self.runs) - 1 else "w"
        big = float(np.clip((w - 6) / (5 * CHAR_W), 48.0, 92.0))
        rows = [f"BETA      {b:.6f}", f"E_MU      {g * M_MU:.3f} GEV", "TAU       2.197 US",
                f"GAMMA*TAU {g * 2.197:.2f} US", f"L_DECAY   {run['Lr'] / 1000:.2f} KM",
                f"T_15KM    {run['t_lab'] * 1e6:.2f} US", f"T_PROPER  {run['t_lab'] * 1e6 / g:.3f} US",
                f"ATM*      {ATM / g:07.1f} M"]
        with f.build(sa - 0.15, rect, flow="tb", wave=0.4, key=80):
            if w >= 300:
                f.text("w", x1, 262, "OUTLAST // 8.0", size=L.T_MICRO, alpha=0.6, anchor="rs")
            f.tag(hot, x0, 312 - 14, _fit(["LORENTZ FACTOR GAMMA", "GAMMA"], w, L.T_MICRO), size=L.T_MICRO, pad=3)
            with f.build(own, rect, flow="tb", wave=0.25, marks=False, key=81):
                f.tag("w", x0, 262, f"RUN {p + 1:02d}/{len(self.runs):02d}", size=L.T_LABEL, pad=4)
                f.text(hot, x0 - 2, 312 + big, f"{g:05.2f}", size=big, alpha=0.97)
                hud.rows(f, x0, 456, rows, size=L.T_SMALL, lead=1.62, alpha=0.9)
        # the deaths of the relativistic lanes, in order: every line is decoded as it arrives
        yy = 716.0
        with f.build(sa - 0.35, (x0 - 8.0, yy - 50.0, x1 + 8.0, 1152.0), flow="tb", wave=0.4, key=82):
            hud.panel_header(f, x0, x1, yy - 26, _fit(["DECAY_STREAM // RELATIVISTIC", "DECAY_STREAM"], w, L.T_MICRO))
            f.text("w", x0, yy + 12, "LANE  DIST KM   T_RUN S", size=L.T_MICRO, alpha=0.55)
            S = st[1]
            order = np.argsort(S["r_dec"])
            done = [i for i in order if S["dead"][i] and r >= S["r_dec"][i]]
            for k, i in enumerate(done[::-1][:19]):
                a = r - S["r_dec"][i]
                f.text("r", x0, yy + 36 + k * 21,
                       B.resolve(f"L{i:03d}  {S['dec'][i] / 1000:7.3f}   {S['r_dec'][i]:6.3f}", a, 170.0, key=int(i), spin=0.2),
                       size=L.T_MICRO, alpha=1.0 if a < 0.5 else 0.8 if k < 3 else 0.6)
            if w >= 290:
                f.text("w", x1, yy + 12, f"N {len(done):03d}", size=L.T_MICRO, alpha=0.8, anchor="rs")

    # --- the muon's own frame + survival ------------------------------------------------------
    def _draw_muon_frame(self, f, t, run, r, p):
        if self.col_frame is None:
            return
        x0, x1 = self.col_frame
        w = x1 - x0
        g = run["g"]
        sa = t - T0
        own = None if p == 0 else r
        frac = float(_f(min(max(r, 0.0), T_TRAVEL) / T_TRAVEL))
        t_lab = frac * run["t_lab"] * 1e6
        t_prop = t_lab / g
        ax0, ax1 = x0 + 6, x1 - 6
        ya, yb = 348.0, 478.0
        with f.build(sa - 0.3, (x0 - 8.0, 238.0, x1 + 8.0, 532.0), wave=0.4, key=83):
            hud.panel_header(f, x0, x1, 262, _fit(["MUON_FRAME // PROPER TIME", "MUON_FRAME"], w, L.T_MICRO))
            # proper-time axis, 0..4 us
            f.text("w", ax0, ya - 34, f"T_PROPER {t_prop:07.4f} US", size=L.T_SMALL, alpha=0.95)
            hud.ruler(f, ax0, ax1, ya, 0.0, 4.0, 0.1, 1.0, fmt=lambda v: f"{v:.0f}", down=True, lab_dy=30)
            xp = float(ax0 + min(t_prop, 4.0) / 4.0 * (ax1 - ax0))
            f.rects("w", ax0, ya - 16, xp, ya - 6, 0.95)
            xt = ax0 + 2.197 / 4.0 * (ax1 - ax0)
            f.segments("r", [xt], [ya - 30], [xt], [ya + 16], 1.2, width=L.LW)
            f.tag("r", xt + 7, ya + 62, "TAU 2.197 US", size=L.T_MICRO, pad=3)
            f.segments("r", [xt], [ya + 16], [xt], [ya + 62], 0.6, width=E.ww(1.0))
            # lab-time axis, 0..60 us
            f.text("w", ax0, yb - 34, f"T_LAB    {t_lab:07.3f} US", size=L.T_SMALL, alpha=0.8)
            hud.ruler(f, ax0, ax1, yb, 0.0, 60.0, 2.0, 20.0, fmt=lambda v: f"{v:.0f}", down=True, lab_dy=30)
            xl = float(ax0 + min(t_lab, 60.0) / 60.0 * (ax1 - ax0))
            f.rects("w", ax0, yb - 16, xl, yb - 6, E.wl(0.55))
        if run["Lr"] < ATM:                       # the clock passes the lifetime: the tag is made at that moment,
            r_past = T_TRAVEL * float(_finv(run["Lr"] / ATM))             # and taken apart before the next run resets the clock
            B.tag(f, "r", x1, ya - 34, _fit(["PAST ITS LIFETIME", "> TAU"], w - 19 * L.T_SMALL * CHAR_W - 16, L.T_MICRO),
                  B.io(r - r_past, RUN_GAP - r, out=0.3, span=0.4), size=L.T_MICRO, pad=3, anchor="rs", cps=90.0, key=91)
        # the atmosphere: 15 km for us, 15 km / gamma for the muon (length contraction)
        rect = (x0 - 8.0, 542.0, x1 + 8.0, 700.0)
        full = ax1 - ax0
        with f.build(sa - 0.45, rect, wave=0.4, key=84):
            hud.panel_header(f, x0, x1, 566, _fit(["ATMOSPHERE // LAB FRAME VS MUON FRAME", "ATMOSPHERE // LAB VS MUON",
                                                   "ATMOSPHERE"], w, L.T_MICRO))
            for k, (yc0, wdt, lay, opts) in enumerate((
                    (590.0, full, "w", ["LAB   15000.0 M"]),
                    (646.0, full / g, "r", [f"MUON  {ATM / g:07.1f} M  = 15 KM / {g:.1f}", f"MUON  {ATM / g:07.1f} M"]))):
                yc1 = yc0 + 18
                # the contracted atmosphere is the one of the run: redrawn when the run starts
                with f.build(own if k else None, rect, wave=0.3, marks=False, key=92):
                    if E.WALL:              # the wall rule: one stratum in four, full white, black between them
                        s_ = ax0 + self.strata[::4 * (int(math.ceil(g)) if k else 1)] / ATM * wdt
                        f.segments("w", s_, np.full(len(s_), yc0 + 2), s_, np.full(len(s_), yc1 - 2), 1.0, width=L.LW)
                        f.rect("w", ax0, yc0, ax0 + wdt, yc1, 1.0, width=L.LW)
                    else:
                        s_ = ax0 + self.strata[::(int(math.ceil(g)) if k else 1)] / ATM * wdt
                        f.segments("w", s_, np.full(len(s_), yc0 + 2), s_, np.full(len(s_), yc1 - 2), 0.3 if k == 0 else 0.5)
                        f.rect("w", ax0, yc0, ax0 + wdt, yc1, 0.8 if k else 0.45, width=L.LW if k else 1.0)
                    f.text(lay, ax0, yc1 + 24, _fit(opts, full, L.T_MICRO), size=L.T_MICRO, alpha=0.9)
                xm = ax0 + frac * wdt
                f.segments(lay, [xm], [yc0 - 6], [xm], [yc1 + 6], 1.2, width=L.LW)
                f.dots(lay, [xm], [(yc0 + yc1) / 2], 3.6, 1.5)

    def _draw_survival(self, f, t, run, r, p, st):
        if self.col_frame is None:
            return
        x0, x1 = self.col_frame
        w = x1 - x0
        sa = t - T0
        own = None if p == 0 else r
        px0, px1, py0, py1 = x0 + 34, x1 - 6, 790.0, 1128.0
        rect = (x0 - 8.0, 724.0, x1 + 8.0, 1170.0)
        with f.build(sa - 0.6, rect, wave=0.45, key=85):
            hud.panel_header(f, x0, x1, 748, _fit(["SURVIVAL // N/N0 VS DISTANCE", "SURVIVAL // N/N0"], w, L.T_MICRO))
            f.segments("w", [px0, px0], [py0, py1], [px0, px1], [py1, py1], E.wl(0.7), width=L.LW)
            hud.ruler(f, px0, px1, py1, 0.0, 15.0, 1.0, 5.0, fmt=lambda v: f"{v:.0f} KM" if v < 14 else "", down=True,
                      lab_dy=32)
            f.text("w", px1, py1 + 32, "15", size=L.T_MICRO, alpha=0.75, anchor="rs")
            for v in (0.25, 0.5, 0.75, 1.0):
                y = py1 - v * (py1 - py0)
                f.segments("w", [px0 - 9], [y], [px0], [y], E.wl(0.7), width=E.ww(1.0))
                f.text("w", px0 - 12, y + 5, f"{v:.2f}"[1:] if v < 1 else "1", size=L.T_MICRO, alpha=0.6, anchor="rs")
            # the two theoretical curves and the two probabilities are the ones of the run: rewritten at its start
            d = np.linspace(0, ATM, 220)
            xs = px0 + d / ATM * (px1 - px0)
            with f.build(own, (px0, py0, px1, py1), flow="lr", wave=0.5, marks=False, key=86):
                for Ld, lay in ((run["Lc"], "w"), (run["Lr"], "r")):
                    yv = py1 - np.exp(-d / Ld) * (py1 - py0)
                    if E.WALL:              # a dotted curve that lands: fewer, larger, full level
                        f.dots(lay, xs[::8], yv[::8], 2.0, 1.0)
                    else:
                        f.dots(lay, xs[::3], yv[::3], 1.3, 0.8)
                pc = math.exp(-ATM / run["Lc"])
                pr = math.exp(-ATM / run["Lr"])
                room = px1 - px0 - 30
                f.text("w", px1, py0 + 14, _fit([f"P_GROUND CLASSICAL {pc:.1E}", f"P_CLS {pc:.1E}"], room, L.T_MICRO),
                       size=L.T_MICRO, anchor="rs", alpha=0.9)
                f.text("r", px1, py0 + 38, _fit([f"P_GROUND RELATIVISTIC {pr:.3f}", f"P_REL {pr:.3f}"], room, L.T_MICRO),
                       size=L.T_MICRO, anchor="rs")
            frac = float(_f(min(max(r, 0.0), T_TRAVEL) / T_TRAVEL))
            dn = d[d <= ATM * frac + 1]
            for which, lay in ((0, "w"), (1, "r")):
                dec = st[which]["dec"]
                if len(dn) > 1:
                    alive = (dec[None, :] > dn[:, None]).mean(1)
                    f.polyline(lay, px0 + dn / ATM * (px1 - px0), py1 - alive * (py1 - py0), 1.0, width=L.LW_BOLD)

    # --- score strip -----------------------------------------------------------------------
    def _draw_strip(self, f, t, p_now, r_now):
        sa = t - T0                               # the strip is built with the scene: band, rules, then its content
        x0, y0, x1, y1, yb = hud.strip_base(
            f, title="OUTLAST // FOUR RUNS, RISING GAMMA // EACH COMB = ONE SURVIVAL CURVE", ticks=(T0, T1, 1.0, 10.0),
            age=sa)
        X = lambda tt: x0 + (np.asarray(tt, np.float64) - T0) / (T1 - T0) * (x1 - x0)
        with f.build(sa - 0.3, L.STRIP, flow="lr", wave=0.5, marks=False, bars="centre", key=87):
            for p, run in enumerate(self.runs):
                ts = self.starts[p]
                xs0 = float(X(ts))
                span = float(X(ts + T_TRAVEL)) - xs0
                order = np.argsort(-run["dr"])
                resolved = np.ones(N, bool)
                if p == p_now:
                    S = self._state(run, r_now, 1)
                    resolved = S["dead"] | S["ground"]
                elif p > p_now:
                    resolved[:] = False
                bx = xs0 + np.arange(N) * span / N
                m = resolved[order]
                hh = np.clip(run["dr"][order] / ATM, 0, 1) * 44.0
                f.rects("w", bx[m], y0 + 1, bx[m] + 2, y0 + 2 + hh[m], 0.95)
                hc = np.maximum(np.clip(run["dc"][order] / ATM, 0, 1) * 30.0, 2.0)
                f.rects("w", bx[m], y1 - 1 - hc[m], bx[m] + 2, y1 - 1, E.wl(0.7))
                lay = "r" if p == p_now else "w"
                f.segments("w", [xs0], [yb - 14], [xs0], [yb + 14], 0.8, width=E.ww(1.0))
                f.tag(lay, xs0 + 5, yb + 28, f"RUN {p + 1:02d} GAMMA {run['g']:04.1f}", size=L.T_MICRO, pad=3,
                      alpha=1.0 if p <= p_now else 0.45)
            xo = float(X(self.t_out))
            f.segments("r", [xo], [y0], [xo], [y1], 0.9 if t >= self.t_out else 0.45, width=L.LW)
            xc = float(X(np.clip(t, T0, T1)))
            f.segments("r", [xc], [y0 - 4], [xc], [y1 + 4], 1.2, width=L.LW)
            if xc < x1 - 130:
                f.tag("r", xc + 6, y1 - 4, sd.tc(t), size=L.T_MICRO, pad=3)
            else:
                f.tag("r", xc - 6, y1 - 4, sd.tc(t), size=L.T_MICRO, pad=3, anchor="rs")
        B.tag(f, "r", xo + 5, yb - 14, "OUTLAST", t - self.t_out, size=L.T_MICRO, pad=3, cps=50.0, key=93)   # on the voice

    # --- bottom band ---------------------------------------------------------------------------
    def _draw_bottom(self, f, t, ctx, run, r, st, p):
        y0, y1 = ctx.slots["y0"], ctx.slots["y1"]
        sa = t - T0                               # the three panels are built with the scene, one after the other
        box = lambda a, b: (a - 8.0, y0 - 24.0, b + 8.0, y1 + 8.0)
        if self.pan_grid:                         # arrival grid: one cell per muon
            x0, x1 = self.pan_grid
            with f.build(sa - 0.5, box(x0, x1), key=88):
                hud.panel_header(f, x0, x1, y0, f"ARRIVAL_GRID // {N} + {N}")
                cw = (x1 - x0) / (N // 2)
                for which, yy0 in ((0, y0 + 18), (1, y0 + 70)):
                    S = st[which]
                    for row in range(2):
                        ids = np.arange(N // 2) + row * (N // 2)
                        cx = x0 + np.arange(N // 2) * cw
                        cy = yy0 + row * 24
                        gnd = S["ground"][ids]
                        dead = S["dead"][ids]
                        f.rects("r" if which else "w", cx[gnd] + 2, cy, cx[gnd] + cw - 2, cy + 18, 1.0)
                        live = ~gnd & ~dead
                        f.rects("w", cx[live] + 4, cy + 7, cx[live] + cw - 4, cy + 11, E.wl(0.7))
                        nd = int(dead.sum())
                        if E.WALL:          # a dead one: a short full white tick in the middle of its cell
                            xm = cx[dead] + 0.5 * cw
                            f.segments("w", xm - 2.5, np.full(nd, cy + 9), xm + 2.5, np.full(nd, cy + 9), 1.0, width=L.LW)
                        else:
                            f.segments("w", cx[dead] + 3, np.full(nd, cy + 9), cx[dead] + cw - 3, np.full(nd, cy + 9), 0.3)
        if self.pan_bar:                          # decay barcode (last 6 s)
            x0, x1 = self.pan_bar
            with f.build(sa - 0.65, box(x0, x1), key=89):
                hud.panel_header(f, x0, x1, y0, "DECAY_BARCODE")
                cols = int(np.clip((x1 - x0) / 3.2, 40, 150))
                kk, frac, dt = hud.barcode_keys(t, 6.0, cols)
                tt = kk * dt
                dens = np.zeros(len(kk))
                for q in range(max(0, p - 1), p + 1):
                    rr = tt - self.starts[q]
                    for which in (0, 1):
                        S = self.final[q][which]
                        rd = np.sort(S["r_dec"][S["dec"] < ATM])
                        dens += np.where(rr >= 0, np.searchsorted(rd, rr + dt) - np.searchsorted(rd, rr), 0)
                dens = 0.05 + 0.9 * np.tanh(dens / 2.0)
                hud.barcode_lanes(f, x0, x1, y0 + 12, y1, dens, kk, lanes=3, seed=5, frac=frac)
        if self.pan_cnt:                          # totals
            x0, x1 = self.pan_cnt
            with f.build(sa - 0.8, box(x0, x1), key=90):
                hud.panel_header(f, x0, x1, y0, "AT GROUND")
                gc = int(st[0]["ground"].sum())
                gr = int(st[1]["ground"].sum())
                f.text("w", x0 + 2, y0 + 34, "CLASSICAL", size=L.T_MICRO, alpha=0.8)
                f.text("w", x0, y0 + 70, f"{gc:03d}", size=36)
                f.text("r", x0 + 2, y0 + 92, "RELATIVISTIC", size=L.T_MICRO, alpha=0.9)
                f.text("r", x0, y0 + 126, f"{gr:03d}", size=36)
