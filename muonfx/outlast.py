"""OUTLAST - time dilation, the reason muons reach the ground. Ryoji Ikeda edition.

A muon lives 2.197 us. Even at almost the speed of light it should decay after
~0.66 km, yet muons born 15 km up reach the ground: their clocks run slow
(time dilation, factor gamma), or in their own frame the atmosphere is
contracted by gamma.

Four runs, one per 4-bar phrase (128 BPM), at gamma = 2, 6, 14, 29 (a
crescendo). Each run launches 96 + 96 muons from 15 km towards the ground:
  top lanes (white)   classical physics, no dilation: all die within ~1 km
  lower lanes (red)   relativity: each run goes further; at gamma 29 most outlast
The front accelerates (accelerando). Bar 1 = MICRO view of the first 2.5 km
(the classical population dies on the downbeat), bars 2-3 = full atmosphere,
bar 4 = result. Panels: the muon's own frame (proper-time clock against its
2.197 us lifetime, the atmosphere contracted by gamma), survival curves N/N0,
arrival grid, a score strip where every run leaves its survival-curve comb.
Loops seamlessly every 16 bars (30 s).
"""
from __future__ import annotations

import math

import numpy as np

from .engine import Frame, hash01, smoothstep
from .hud import (BAR, BEAT, BOT, MAIN, PHRASE, STRIP, barcode_burst, barcode_lanes, big_number, finish,
                  panel_header, ruler, timing)

C = 299792458.0
TAU = 2.197e-6
ATM = 15000.0
GAMMAS = [2.0, 6.0, 14.0, 29.0]
N = 64
M_MU = 0.10566                                # GeV

LX0, LX1 = 440.0, 2920.0                     # lanes: 15 km altitude -> ground
CL = (575.0, 815.0)                          # classical lanes (y range)
RL = (885.0, 1125.0)                         # relativistic lanes
LANES_CLIP = (LX0 - 4, 522.0, LX1 + 34, 1172.0)
T_TRAVEL = 3 * BAR
ACC = 1.6                                     # accelerando exponent of the front
MICRO = 2500.0                                # micro view range (m)


def _f(x):
    return np.clip(x, 0.0, 1.0) ** ACC


def _finv(y):
    return np.clip(y, 0.0, 1.0) ** (1.0 / ACC)


class Outlast:
    name = "outlast"

    def __init__(self, T=30.0, seed=21):
        self.T = T
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
        self.lane_y = {0: np.linspace(CL[0], CL[1], N), 1: np.linspace(RL[0], RL[1], N)}

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

    # ------------------------------------------------------------------ render
    def render(self, t, W=3000, H=1688):
        tm = timing(t, self.T)
        t = tm["t"]
        f = Frame(W, H)
        p = tm["phrase"] % len(self.runs)
        run = self.runs[p]
        r = t - p * PHRASE
        micro = tm["bip"] == 0
        dmax = MICRO if micro else ATM
        st = [self._state(run, r, 0), self._state(run, r, 1)]

        if tm["burst"]:
            f.set_clip(*MAIN)
            barcode_burst(f, MAIN, t, self.T, seed=3)
            f.set_clip()
        else:
            f.set_clip(*LANES_CLIP)
            self._draw_lanes(f, run, r, st, dmax, micro)
            f.set_clip()
            self._draw_lane_hud(f, run, r, st, dmax, micro, tm)
            self._draw_muon_frame(f, run, r)
            self._draw_survival(f, run, r, st)
        self._draw_left(f, run, r, p, st)
        self._draw_strip(f, t, p, r)
        self._draw_bottom(f, t, run, r, st, p, tm)
        return finish(f, tm["invert"])

    def _x(self, d, dmax):
        return LX0 + np.asarray(d) / dmax * (LX1 - LX0)

    def _draw_lanes(self, f, run, r, st, dmax, micro):
        X = lambda d: self._x(d, dmax)
        # air strata backdrop
        s = self.strata[self.strata <= dmax]
        xs = X(s)
        f.segments("w", xs, np.full_like(xs, CL[0] - 14), xs, np.full_like(xs, RL[1] + 14), 0.05)
        for which, lay in ((0, "w"), (1, "r")):
            S = st[which]
            ys = self.lane_y[which]
            head = S["head"]
            dead = S["dead"]
            a = r - S["r_dec"]
            base = 0.5 if which else 0.45
            inten = np.where(dead, 0.1 + 0.5 * np.exp(-np.maximum(a, 0) / 0.5), base)
            vis = head > 1.0
            f.segments(lay, np.full(vis.sum(), LX0), ys[vis], X(head[vis]), ys[vis], inten[vis], width=1.0)
            alive = ~dead & ~S["ground"] & (head > 1.0) & (head < dmax)
            f.dots(lay, X(head[alive]), ys[alive], 1.9 if not micro else 2.5, 1.5)
            if which == 1:
                f.dots("w", X(head[alive]), ys[alive], 0.7, 0.8)
            # decay bursts: e (solid) + two neutrinos (faint)
            bm = dead & (a >= 0) & (a < 1.4)
            if bm.any():
                grow = np.clip(a[bm] / 0.14, 0, 1)
                fade = np.exp(-a[bm] / 0.55)
                cx, cy = X(head[bm]), ys[bm]
                ang = run["ang"][bm]
                scale = 2.6 if micro else 1.0
                for k, (ln, it) in enumerate(((20.0, 0.95), (13.0, 0.35), (13.0, 0.35))):
                    th = ang[:, k] * (1.1 if k == 0 else 2.6) + (0 if k == 0 else math.pi * (k - 1.5))
                    L = ln * scale * grow
                    f.segments("w", cx, cy, cx + np.cos(th) * L, cy + np.sin(th) * L, it * fade)
                f.dots(lay, cx, cy, 2.4 if micro else 1.6, 1.5 * fade)
            # arrivals at the ground: a comb growing past the ground line
            gm = S["ground"]
            if gm.any() and dmax >= ATM:
                ag = r - S["r_gnd"][gm]
                f.segments(lay, np.full(gm.sum(), LX1 + 3), ys[gm], np.full(gm.sum(), LX1 + 18), ys[gm], 1.0)
                fl = ag < 0.5
                if fl.any():
                    f.dots("w", np.full(fl.sum(), LX1), ys[gm][fl], 3.0, 1.6 * (1 - ag[fl] / 0.5))
        if micro:   # per-lane readouts of the classical deaths
            S = st[0]
            ys = self.lane_y[0]
            for i in range(0, N, 7):
                if S["dead"][i] and r - S["r_dec"][i] > 0.05:
                    x = float(X(S["head"][i]))
                    n = int(min(12, (r - S["r_dec"][i]) * 90))
                    f.text("w", x + 26, float(ys[i]) + 5, f"{S['head'][i] / 1000:.3f} KM"[:n], size=13, alpha=0.8)

    def _draw_lane_hud(self, f, run, r, st, dmax, micro, tm):
        X = lambda d: self._x(d, dmax)
        fmt = lambda d: f"{(ATM - d) / 1000:04.1f}"
        minor, major = (20.0, 250.0) if micro else (100.0, 1000.0)
        ruler(f, LX0, LX1, CL[0] - 26, 0.0, dmax, minor, major, fmt=None, down=False)
        for d in np.arange(0.0, dmax + 1, major * (1 if micro else 1)):
            f.text("w", float(X(d)) + 4, CL[0] - 34, f"{(ATM - d) / 1000:05.2f} KM" if micro else fmt(d) + " KM",
                   size=12, alpha=0.75)
        ruler(f, LX0, LX1, RL[1] + 24, 0.0, dmax, minor, major, fmt=None, down=True)
        # decay-length markers
        Lc, Lr = run["Lc"], run["Lr"]
        for L, (y0, y1), lay, txt in ((Lc, CL, "w", f"BETA*C*TAU {Lc / 1000:.3f} KM"),
                                      (Lr, RL, "r", f"GAMMA*BETA*C*TAU {Lr / 1000:.2f} KM")):
            if L < dmax:
                x = float(X(L))
                k = np.arange(y0 - 8, y1 + 8, 12.0)
                f.segments(lay, np.full_like(k, x), k, np.full_like(k, x), k + 6, 0.9)
                f.tag(lay, x + 6, y1 + 22 if lay == "w" else y0 - 12, txt, size=13, pad=3)
            else:
                f.tag(lay, LX1 - 6, y0 - 12, txt + " >", size=13, pad=3, anchor="rs")
        # ground line
        if dmax >= ATM:
            f.segments("w", [LX1], [CL[0] - 20], [LX1], [RL[1] + 20], 1.0, width=2.0)
            f.tag("w", LX1 - 6, RL[1] + 52, "GROUND 0 M", size=13, pad=3, anchor="rs")
        # the front
        frac = float(_f(min(r, T_TRAVEL) / T_TRAVEL))
        dfront = ATM * frac
        if r < T_TRAVEL and dfront <= dmax:
            x = float(X(dfront))
            f.segments("r", [x], [CL[0] - 18], [x], [RL[1] + 18], 1.1, width=1.4)
            f.tag("r", x + 6, RL[1] + 52, f"T_LAB {frac * run['t_lab'] * 1e6:07.3f} US", size=13, pad=3)
        # block labels + counters
        f.tag("w", LX0, CL[0] - 60, "CLASSICAL // NO TIME DILATION", size=14, pad=4)
        f.tag("r", LX0, RL[0] - 44, f"RELATIVISTIC // CLOCKS SLOWED x{run['g']:.1f}", size=14, pad=4)
        for which, (y0, y1) in ((0, CL), (1, RL)):
            S = st[which]
            alive = int((~S["dead"]).sum())
            gnd = int(S["ground"].sum())
            lay = "r" if which else "w"
            yy = y0 - 60 if which == 0 else y0 - 44
            f.text(lay, LX1 - 4, yy, f"ALIVE {alive:03d}/{N:03d}   GROUND {gnd:03d}", size=15, anchor="rs")
        if micro:
            f.tag("w", LX0 + 470, CL[0] - 60, f"MICRO // FIRST {MICRO / 1000:.1f} KM  x{ATM / MICRO:.1f}", size=14,
                  pad=4)

    def _draw_muon_frame(self, f, run, r):
        x0, x1, y0 = 440.0, 1640.0, 1200.0
        panel_header(f, x0, x1, y0, "MUON_FRAME // PROPER TIME")
        g = run["g"]
        frac = float(_f(min(r, T_TRAVEL) / T_TRAVEL))
        t_lab = frac * run["t_lab"] * 1e6
        t_prop = t_lab / g
        ax0, ax1 = x0 + 20, x1 - 20
        # proper-time axis, 0..4 us
        ya = 1262.0
        ruler(f, ax0, ax1, ya, 0.0, 4.0, 0.05, 0.5, fmt=lambda v: f"{v:.1f}", down=True)
        xp = float(ax0 + min(t_prop, 4.0) / 4.0 * (ax1 - ax0))
        f.rects("w", ax0, ya - 14, xp, ya - 6, 0.95)
        xt = ax0 + 2.197 / 4.0 * (ax1 - ax0)
        f.segments("r", [xt], [ya - 30], [xt], [ya + 14], 1.2, width=1.6)
        f.tag("r", xt + 6, ya - 20, "TAU 2.197 US", size=12, pad=3)
        f.text("w", ax0, ya - 26, f"T_PROPER {t_prop:07.4f} US", size=14, alpha=0.95)
        if t_prop > 2.197:
            f.tag("r", min(xp, ax1) + 6, ya - 8, "PAST ITS LIFETIME", size=11, pad=3)
        # lab-time axis, 0..60 us
        yb = 1330.0
        ruler(f, ax0, ax1, yb, 0.0, 60.0, 1.0, 10.0, fmt=lambda v: f"{v:.0f}", down=True)
        xl = float(ax0 + min(t_lab, 60.0) / 60.0 * (ax1 - ax0))
        f.rects("w", ax0, yb - 14, xl, yb - 6, 0.55)
        f.text("w", ax0, yb - 26, f"T_LAB {t_lab:07.3f} US", size=14, alpha=0.8)
        # the atmosphere seen by the muon: 15 km / gamma, sliding past it
        yc0, yc1 = 1392.0, 1428.0
        full = ax1 - ax0
        Lp = full / g
        xm = ax0 + 30
        left = xm - frac * Lp
        f.rect("w", ax0, yc0, ax1, yc1, 0.18)
        s = self.strata / ATM * Lp + left
        m = (s >= ax0) & (s <= ax1)
        f.segments("w", s[m], np.full(m.sum(), yc0 + 2), s[m], np.full(m.sum(), yc1 - 2), 0.55)
        f.segments("w", [max(left, ax0), max(left, ax0)], [yc0, yc1], [min(left + Lp, ax1), min(left + Lp, ax1)],
                   [yc0, yc1], 0.9)
        f.dots("r", [xm], [(yc0 + yc1) / 2], 4.0, 1.6)
        f.text("w", ax0, yc1 + 16, f"ATMOSPHERE IN MUON FRAME  15000 M / {g:.1f} = {ATM / g:06.1f} M", size=13,
               alpha=0.85)

    def _draw_survival(self, f, run, r, st):
        x0, x1, y0 = 1720.0, 2920.0, 1200.0
        panel_header(f, x0, x1, y0, "SURVIVAL // N/N0 VS DISTANCE")
        px0, px1, py0, py1 = x0 + 20, x1 - 20, 1236.0, 1440.0
        f.segments("w", [px0, px0], [py0, py1], [px0, px1], [py1, py1], 0.7)
        ruler(f, px0, px1, py1, 0.0, 15.0, 0.5, 5.0, fmt=lambda v: f"{v:.0f} KM", down=True, lab_dy=-6)
        for v in (0.25, 0.5, 0.75, 1.0):
            y = py1 - v * (py1 - py0)
            f.segments("w", [px0 - 8], [y], [px0], [y], 0.7)
        d = np.linspace(0, ATM, 220)
        xs = px0 + d / ATM * (px1 - px0)
        for L, lay in ((run["Lc"], "w"), (run["Lr"], "r")):
            yv = py1 - np.exp(-d / L) * (py1 - py0)
            f.dots(lay, xs[::2], yv[::2], 1.2, 0.8)
        frac = float(_f(min(r, T_TRAVEL) / T_TRAVEL))
        dn = d[d <= ATM * frac + 1]
        for which, lay in ((0, "w"), (1, "r")):
            dec = st[which]["dec"]
            alive = (dec[None, :] > dn[:, None]).mean(1)
            if len(dn) > 1:
                f.polyline(lay, px0 + dn / ATM * (px1 - px0), py1 - alive * (py1 - py0), 1.0, width=1.4)
        pc = math.exp(-ATM / run["Lc"])
        pr = math.exp(-ATM / run["Lr"])
        f.text("w", px1, py0 + 16, f"P_GROUND CLASSICAL {pc:.1E}", size=14, anchor="rs")
        f.text("r", px1, py0 + 38, f"P_GROUND RELATIVISTIC {pr:.3f}", size=14, anchor="rs")

    def _draw_left(self, f, run, r, p, st):
        x = 60.0
        g, b = run["g"], run["b"]
        f.tag("w", x, 540, f"RUN {p + 1:02d}/{len(self.runs):02d}", size=16, pad=4)
        big_number(f, x, 668, "LORENTZ FACTOR GAMMA", f"{g:05.2f}", size=78, layer="r" if p == len(self.runs) - 1
                   else "w")
        rows = [f"BETA      {b:.6f}", f"E_MU      {g * M_MU:.3f} GEV", "TAU       2.197 US",
                f"GAMMA*TAU {g * 2.197:.2f} US", f"L_DECAY   {run['Lr'] / 1000:.2f} KM",
                f"T_15KM    {run['t_lab'] * 1e6:.2f} US", f"T_PROPER  {run['t_lab'] * 1e6 / g:.3f} US",
                f"ATM*      {ATM / g:07.1f} M"]
        for k, row in enumerate(rows):
            f.text("w", x, 740 + k * 26, row, size=15, alpha=0.9)
        # tiny event stream of the deaths
        S = st[1]
        order = np.argsort(S["r_dec"])
        yy = 1000.0
        f.tag("w", x, yy, "DECAY_STREAM", size=12, pad=3)
        k = 0
        for i in order:
            if not S["dead"][i] or r < S["r_dec"][i]:
                continue
            if k >= 16:
                break
            f.text("r", x, yy + 30 + k * 17, f"L{i:03d} {S['dec'][i] / 1000:6.3f} KM {S['r_dec'][i]:.3f}", size=12,
                   alpha=0.85)
            k += 1

    def _draw_strip(self, f, t, p_now, r_now):
        x0, y0, x1, y1 = STRIP
        T = self.T
        X = lambda tt: x0 + np.asarray(tt) / T * (x1 - x0)
        f.segments("w", [x0, x0], [y0, y1], [x1, x1], [y0, y1], 0.9)
        ruler(f, x0, x1, y0, 0.0, T, BEAT, BAR, fmt=None, down=True)
        ruler(f, x0, x1, y1, 0.0, T, BEAT, BAR, fmt=None, down=False)
        for b in range(int(round(T / BAR))):
            f.text("w", float(X(b * BAR)) + 5, y0 + 30, f"{b + 1:02d}", size=13, alpha=0.8)
        yb = (y0 + y1) / 2 - 6
        f.rects("r", x0, yb - 3, x1, yb + 3, 1.0)
        for p, run in enumerate(self.runs):
            ts = p * PHRASE
            done = p < p_now or (p == p_now)
            xs0 = float(X(ts))
            span = float(X(ts + T_TRAVEL)) - xs0
            order = np.argsort(-run["dr"])
            resolved = np.ones(N, bool)
            if p == p_now:
                S = self._state(run, r_now, 1)
                resolved = S["dead"] | S["ground"]
            elif p > p_now:
                resolved[:] = False
            for which, dd in ((1, run["dr"]), (0, run["dc"])):
                hh = np.clip(dd[order] / ATM, 0, 1) * (46.0 if which else 30.0)
                bx = xs0 + np.arange(N) * span / N
                m = resolved[order]
                if which:
                    f.rects("w", bx[m], y0 + 1, bx[m] + 2, y0 + 1 + hh[m], 0.95)
                else:
                    f.rects("w", bx[m], y1 - 1 - np.maximum(hh[m], 1.5), bx[m] + 2, y1 - 1, 0.7)
            lay = "r" if p == p_now else "w"
            f.tag(lay, xs0 + 4, yb + 6, f"RUN {p + 1:02d} GAMMA {run['g']:04.1f}", size=13, pad=3,
                  alpha=1.0 if done else 0.45)
        xc = float(X(t))
        f.segments("r", [xc], [y0 - 4], [xc], [y1 + 4], 1.2, width=1.6)
        f.tag("r", xc + 6, y1 + 22, f"T {t:06.3f}", size=13, pad=3)
        f.tag("w", x0 + 4, y0 - 10, "OUTLAST // FOUR RUNS, RISING GAMMA // EACH COMB = ONE SURVIVAL CURVE",
              size=12, pad=3)

    def _draw_bottom(self, f, t, run, r, st, p, tm):
        x0, y0, x1, y1 = BOT
        # decay barcode (last 3 s)
        bx0, bx1 = x0, 980.0
        panel_header(f, bx0, bx1, y0, "DECAY_BARCODE")
        cols = 240
        dt = 3.0 / cols
        kf = math.floor((t - 3.0) / dt)
        kk = kf + np.arange(cols)
        tt = kk * dt
        rr = tt - p * PHRASE
        dens = np.zeros(cols)
        for which in (0, 1):
            rd = np.sort(self._state(run, 1e9, which)["r_dec"][self._state(run, 1e9, which)["dec"] < ATM])
            lo = np.searchsorted(rd, rr)
            hi = np.searchsorted(rd, rr + dt)
            dens += (hi - lo)
        dens = np.where(rr >= 0, 0.05 + 0.9 * np.tanh(dens / 3.0), 0.05)
        barcode_lanes(f, bx0, bx1, y0 + 10, y1, dens, kk, lanes=3, seed=5)
        # arrival grid: one cell per relativistic muon
        gx0, gx1 = 1020.0, 1980.0
        panel_header(f, gx0, gx1, y0, f"ARRIVAL_GRID // {N} + {N}")
        cw = (gx1 - gx0) / (N // 2)
        for which, yy0 in ((0, y0 + 16), (1, y0 + 66)):
            S = st[which]
            for row in range(2):
                ids = np.arange(N // 2) + row * (N // 2)
                cx = gx0 + np.arange(N // 2) * cw
                cy = yy0 + row * 24
                gnd = S["ground"][ids]
                dead = S["dead"][ids]
                f.rects("r" if which else "w", cx[gnd] + 2, cy, cx[gnd] + cw - 2, cy + 18, 1.0)
                live = ~gnd & ~dead
                f.rects("w", cx[live] + 8, cy + 7, cx[live] + cw - 8, cy + 11, 0.7)
                f.segments("w", cx[dead] + 3, np.full(dead.sum(), cy + 9), cx[dead] + cw - 3,
                           np.full(dead.sum(), cy + 9), 0.3)
        # totals
        tx0 = 2020.0
        panel_header(f, tx0, x1, y0, "AT GROUND")
        gc = int(st[0]["ground"].sum())
        gr = int(st[1]["ground"].sum())
        f.tag("w", tx0 + 4, y0 + 44, "CLASSICAL", size=14, pad=3)
        f.text("w", tx0 + 160, y0 + 48, f"{gc:03d}", size=40)
        f.tag("r", tx0 + 4, y0 + 104, "RELATIVISTIC", size=14, pad=3)
        f.text("r", tx0 + 160, y0 + 108, f"{gr:03d}", size=40)
        if r >= T_TRAVEL:
            word = "OUTLAST" if p == len(self.runs) - 1 else f"{gr:03d} SURVIVED"
            f.tag("r" if p == len(self.runs) - 1 else "w", x1 - 10, y0 + 104, word, size=40, pad=8, anchor="rs")
