"""FLOOD - "more than fifty thousand of them will flood through your skin".
Sheet 5.0 (voice) + 5.1 Cosmic Break, 04:36 - 05:22.

The figure of YOU comes back, between the left and the centre tower, and this time the rain is
shown at its real rate: 70 muons a second through one body (showdata.RATE_YOU).
  04:36.9  "In the time you spend here,"        the show as a bar, the count so far
  04:41.2  "more than fifty thousand of them"   the count projected to the end of the show
  04:44.9  "will flood through your skin."      the rain goes from 1 in 5 to all of them: real time
  04:48.5  "as they have, every minute of your life."   the same number on every time scale
  04:53.7  "as they will,"
  04:56.7  "into the space / you leave behind"  the body dissolves. Its dotted outline stays, the
           rain does not change at all, the entry map (seen from above) loses the silhouette
  04:59 - 05:22  cosmic break: the empty outline, the unchanged rain, the counter still running
The towers keep answering their own (live) hits on top of this.

LAYOUT. Nothing has a fixed x: the towers can stand anywhere (see you.Plan).
  focus bay   the figure and the rain (the only texture: it is clipped to that bay)
  side col    the count: this show as a bar, so far, by the end, every time scale
  title col   FLOOD, the parameters, the entry log
  data col    the entry map seen from above, entries per part of the body, the energy of the muons
  bottom      ctx.slots panels (between the scopes of the towers)
A column that does not exist with a placement drops its block; type sizes follow the widths. The
count never drops: without a side column it shares the focus bay with the figure, or takes the
title column.

NOTHING FADES OR POPS in the read-outs. The furniture (strip, tags, columns, map, bottom panels) is
constructed during the first second and a half of the scene (Frame.build, staggered); what the voice
brings (the projection to the end of the show, the time scales, AFTER YOU) is made on its word: tags
pushed out, lines decoded, figures spinning before they lock; when the target changes state its
read-outs are decoded again; in the break the lines that leave fall apart letter by letter and their
tag and rule are pulled back. The body, its outline and the rain are the image: they keep their ramps.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import hud
from .. import engine as EN
from .. import human
from .. import layout as L
from .. import showdata as sd
from ..engine import OrthoCamera, hash01, smoothstep, text_w
from ..show import Scene
from .you import HEART, PART_NAMES, Body, Plan, fit_text, spaced, width

T0, T1 = 276.0, 322.0
RATE = sd.RATE_YOU
SHOW = sd.SHOW_END
Y_TOP, Y_BOT = 240.0, 1190.0
GROUPS = [("HEAD", ("HEAD", "NECK")), ("TORSO", ("TORSO",)), ("ARMS", ("ARM_L", "ARM_R")),
          ("HANDS", ("HAND_L", "HAND_R")), ("LEGS", ("LEG_L", "LEG_R")), ("FEET", ("FOOT_L", "FOOT_R"))]
# The energy of the muons at the ground (sea level, vertical): Gaisser's formula with the low-energy correction
# of Guan et al. 2015,  dN/dE ~ (E + 3.64)^-2.7 [1 / (1 + 1.1 E / 115) + 0.054 / (1 + 1.1 E / 850)],  E in GeV.
# Its mean is a little over 4 GeV and half of the muons are under 2 GeV: the histogram of the data column.
E_LO, E_HI, E_BINS = 0.1, 100.0, 24             # three decades on a log axis, eight bins each
E_FALL, E_SPAN = 0.3, 10.0                      # s an entry takes to fall into its bin / s it stays in the count


def _spectrum(E):
    return (E + 3.64) ** -2.7 * (1.0 / (1.0 + 1.1 * E / 115.0) + 0.054 / (1.0 + 1.1 * E / 850.0))


def _energy_law():
    """(energies, their cumulative distribution above E_LO, expected share of every bin of the histogram)."""
    E = np.geomspace(E_LO, 1000.0, 4000)
    sp = _spectrum(E)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (sp[1:] + sp[:-1]) * np.diff(E))])
    c /= c[-1]
    share = np.diff(np.interp(np.geomspace(E_LO, E_HI, E_BINS + 1), E, c))
    share[-1] += 1.0 - float(np.interp(E_HI, E, c))             # the few above 100 GeV count in the last bin
    return E, c, share


class Flood(Scene):
    name = "flood"

    def __init__(self, ctx, seed=55):
        super().__init__(ctx)
        self.t_time = sd.said("In the time you spend here", 276.93)
        self.t_fifty = sd.said("more than fifty thousand", 281.15)
        self.t_flood = sd.said("will flood through your skin", 284.92)
        self.t_every = sd.said("as they have", 288.5)
        self.t_will = sd.said("as they will", 293.68)
        self.t_space = sd.said("into the space", 296.68)
        self.t_leave = sd.said("you leave behind", 298.17)
        self.t_gone = self.t_leave + 4.6
        # when the read-outs change state (gone = smoothstep(t_space, t_gone): PRESENT < 0.05, ABSENT >= 0.98)
        ts = np.linspace(self.t_space, self.t_gone, 4001)
        g = np.asarray(smoothstep(self.t_space, self.t_gone, ts))
        self.t_leaving = float(ts[np.argmax(g >= 0.05)])
        self.t_half = float(ts[np.argmax(g > 0.5)])
        self.t_absent = float(ts[np.argmax(g >= 0.98)])
        self.body = Body(33)
        self.P = P = Plan(ctx, panels="slots")
        # who gets which column. The count is what the voice says: it must always have a place.
        self.fig_x, self.fig_half, self.fig_col = P.fx, P.half, P.fcol
        self.fig_clip = (P.fbay[0] + 6.0, P.fbay[1] - 6.0)
        self.count_col, self.title_col, self.map_col = P.side, P.title, P.data
        if self.count_col is None:
            w = width(P.fcol)
            if w >= 1000.0:                 # one big bay: the figure on its left, the count on its right
                split = P.fcol[0] + 0.56 * w
                self.fig_col = (P.fcol[0], split - 40.0)
                self.fig_x = 0.5 * (self.fig_col[0] + self.fig_col[1])
                self.fig_half = 0.5 * width(self.fig_col)
                self.fig_clip = (P.fbay[0] + 6.0, split - 24.0)
                self.count_col = (split, P.fcol[1])
            elif self.title_col is not None:
                self.count_col, self.title_col = self.title_col, None
            elif self.map_col is not None:
                self.count_col, self.map_col = self.map_col, None
        self._build_rain(np.random.default_rng(seed))
        # the energy of every muon of the rain, drawn from the spectrum at the ground (its own generator: the
        # rain itself does not change)
        self.e_grid, self.e_cdf, self.e_share = _energy_law()
        self.h_E = np.interp(np.random.default_rng(seed + 101).random(len(self.h_t)), self.e_cdf, self.e_grid)

    # ------------------------------------------------------------------ rain
    def _build_rain(self, rng, K=170):
        b = self.body
        # arrival times: 1 in 5 before the word "flood", every one of them after
        tt, t = [], T0 + 0.05
        while t < T1:
            r = RATE * (0.2 + 0.8 * float(smoothstep(self.t_flood - 0.1, self.t_flood + 1.0, t)))
            t += rng.exponential(1.0 / r)
            tt.append(t)
        th = np.array(tt)
        n_hit = len(th)
        # targets anywhere inside the body
        tg, _ = human.random_inside(rng, n_hit)
        # the ones that miss, around the figure (same rate again)
        n_miss = n_hit
        tm = np.sort(rng.choice(th, n_miss) + rng.uniform(-0.2, 0.2, n_miss))
        rad, az = 0.3 + 0.75 * np.sqrt(rng.random(n_miss)), rng.uniform(0, 2 * np.pi, n_miss)
        gm = np.stack([rad * np.cos(az), rng.uniform(0.2, 1.7, n_miss), rad * np.sin(az)], 1)
        t_all = np.r_[th, tm]
        P = np.vstack([tg, gm])
        N = len(t_all)
        zen = np.minimum(np.abs(rng.normal(0, 0.28, N)), 0.85)
        phi = rng.uniform(0, 2 * np.pi, N)
        d = np.stack([np.sin(zen) * np.cos(phi), -np.cos(zen), np.sin(zen) * np.sin(phi)], 1)
        top = 2.25
        A = P - d * ((top - P[:, 1]) / np.cos(zen))[:, None]
        B = P + d * (P[:, 1] / np.cos(zen))[:, None]
        u = np.linspace(0, 1, K)
        pts = A[:, None, :] + (B - A)[:, None, :] * u[None, :, None]
        ins = b.inside(pts.reshape(-1, 3)).reshape(N, K)
        inside = ins >= 0
        has = inside.any(1)
        first = np.argmax(inside, 1)
        last = K - 1 - np.argmax(inside[:, ::-1], 1)
        entry = pts[np.arange(N), first]
        Ln = np.linalg.norm(B - A, axis=1)
        dur = Ln / 20.0
        u_in = u[first]
        order = np.argsort(t_all)
        self.r_t = (t_all - dur * np.where(has, u_in, 0.5))[order]      # time the muon enters the view
        self.r_hit_t = t_all[order]
        self.r_a, self.r_b = A[order].astype(np.float32), B[order].astype(np.float32)
        self.r_dur = dur[order]
        self.r_has = has[order]
        self.r_u0, self.r_u1 = u[first][order], u[last][order]
        self.r_entry = entry[order].astype(np.float32)
        self.r_part = np.where(has, ins[np.arange(N), first], -1)[order]
        self.r_frac = (inside.sum(1) / K * Ln)[order]                    # metres of body crossed
        z13 = (top - 1.3) / np.maximum(top - 0.0, 1e-6)
        self.r_mid = (A + (B - A) * z13)[order].astype(np.float32)      # where it crosses shoulder height
        o2 = np.argsort(self.r_hit_t)
        self.h_t = self.r_hit_t[o2]
        self.h_has = self.r_has[o2]
        self.h_entry = self.r_entry[o2]
        self.h_mid = self.r_mid[o2]
        self.h_part = self.r_part[o2]
        self.h_frac = self.r_frac[o2]
        gi = np.full(len(PART_NAMES), -1)
        for k, (_, names) in enumerate(GROUPS):
            for nm in names:
                gi[PART_NAMES.index(nm)] = k
        self.h_group = np.where(self.h_part >= 0, gi[np.maximum(self.h_part, 0)], -1)

    # ------------------------------------------------------------------ draw
    def draw(self, f, t, ctx):
        t = float(np.clip(t, T0, T1 - 1e-3))
        P = self.P
        gone = float(smoothstep(self.t_space, self.t_gone, t))           # 0 = the body is there, 1 = only its place
        there = 1.0 - float(smoothstep(self.t_space, self.t_leave + 1.2, t))
        scale = min(448.0, (2 * self.fig_half - 60.0) / 0.78)            # the figure fits between two towers
        cam = OrthoCamera((0.0, 0.92, 10.0), (0.0, 0.92, 0.0), scale=scale, screen_center=(self.fig_x, 716.0))
        f.set_clip(self.fig_clip[0], Y_TOP, self.fig_clip[1], Y_BOT)
        self._draw_figure(f, cam, t, gone, there)
        f.set_clip()
        self._draw_figure_tags(f, cam, t, gone)
        if self.count_col is not None:
            self._draw_count(f, t, ctx, self.count_col, gone)
        if self.title_col is not None:
            self._draw_left(f, t, self.title_col, gone)
        if self.map_col is not None:
            self._draw_map(f, t, self.map_col, gone, there)
        marks = [(self.t_time, "IN THE TIME"), (self.t_fifty, "50 000"), (self.t_flood, "FLOOD"),
                 (self.t_every, "EVERY MINUTE"), (self.t_will, "AS THEY WILL"), (self.t_leave, "LEAVE BEHIND"),
                 (299.0, "5.1 COSMIC BREAK")]
        hud.show_strip(f, t, ctx, f"FLOOD // {RATE:.0f} MUONS /S THROUGH ONE BODY // REAL TIME", T0, T1, marks,
                       age=t - T0)
        self._draw_bottom(f, t, ctx)
        return {"burst_size": 0.8}

    def _active(self, t, tail=0.45):
        i0, i1 = np.searchsorted(self.r_t, t - 0.75), np.searchsorted(self.r_t, t)
        sl = slice(i0, i1)
        age = t - self.r_t[sl]
        m = age < self.r_dur[sl] + tail
        return sl, age, m

    def _draw_figure(self, f, cam, t, gone, there):
        b = self.body
        b.draw_floor(f, cam, gain=0.9)
        sl, age, m = self._active(t)
        dur, has = self.r_dur[sl], self.r_has[sl]
        # hits of the last moments: red on the skin, red on the slices
        hot_pts, hot_lev = [], np.zeros(len(b.levels), np.float32)
        if there > 0.02:
            ah = t - self.r_hit_t[sl]
            sel = np.nonzero(has & (ah >= 0) & (ah < 0.35))[0]
            for k in sel[-26:]:
                e = self.r_entry[sl][k]
                s_ = math.exp(-ah[k] / 0.14) * there
                hot_pts.append((e, s_))
                lv = int(np.argmin(np.abs(b.levels - e[1])))
                hot_lev[lv] = max(hot_lev[lv], 0.85 * s_)
        b.draw(f, cam, hot_pts=hot_pts, hot_lev=hot_lev, dissolve=gone, t=t,
               outline=float(smoothstep(self.t_leave - 0.3, self.t_leave + 2.5, t)) * (0.95 - 0.25 * gone))
        if not m.any():
            return
        A, B = self.r_a[sl][m], self.r_b[sl][m]
        age, dur, has = age[m], dur[m], has[m]
        prog = np.minimum(1.0, age / dur)
        Hd = A + (B - A) * prog[:, None].astype(np.float32)
        ax, ay, _, _ = cam.project(A)
        hx, hy, _, _ = cam.project(Hd)
        fade = np.where(age < dur, 1.0, np.exp(-(age - dur) / 0.15))
        base = EN.wl(0.3 + 0.2 * there)      # (the wall: the head of a track is a full red line, its tail still runs out)
        f.segments("r", ax, ay, hx, hy, 0.12 * fade, base * fade, width=EN.ww(1.2))
        fl = age < dur
        f.dots("r", hx[fl], hy[fl], 2.4, 1.3)
        if there > 0.02:            # ionisation inside the body: the bright part of each track
            u0, u1 = self.r_u0[sl][m], np.minimum(self.r_u1[sl][m], prog)
            run = has & (u1 > u0)
            if run.any():
                P0 = A[run] + (B[run] - A[run]) * u0[run, None].astype(np.float32)
                P1 = A[run] + (B[run] - A[run]) * u1[run, None].astype(np.float32)
                x0, y0, _, _ = cam.project(P0)
                x1, y1, _, _ = cam.project(P1)
                f.segments("r", x0, y0, x1, y1, 1.15 * fade[run] * there, width=2.2)
                ah = (age - dur * u0)[run]
                fresh = (ah >= 0) & (ah < 0.22)
                f.dots("w", x0[fresh], y0[fresh], 2.6, 1.5 * (1 - ah[fresh] / 0.22) * there)

    def _draw_figure_tags(self, f, cam, t, gone):
        col = self.fig_col
        x0 = col[0] + 6
        y = Y_TOP + 36
        state = "PRESENT" if gone < 0.05 else ("LEAVING" if gone < 0.98 else "ABSENT")
        tag = fit_text([f"VIEW // ORTHO_FRONT // TARGET // {state}", f"TARGET // {state}", state], width(col) - 12,
                       L.T_LABEL)
        if gone < 0.05:                 # made when the scene starts ...
            B.tag(f, "w", x0, y, tag, t - T0 - 0.05, size=L.T_LABEL, pad=5, cps=90.0, key=1)
        else:                           # ... its letters are decoded again when the target changes state
            f.tag("r" if gone >= 0.98 else "w", x0, y,
                  B.decode(tag, t - (self.t_leaving if gone < 0.98 else self.t_absent), cps=90.0, key=1, pad=True),
                  size=L.T_LABEL, pad=5, ref=tag)
        shown = "1 IN 5" if t < self.t_flood else "ALL OF THEM // REAL TIME"
        txt = fit_text([f"SHOWING {shown}", shown, "1 IN 5" if t < self.t_flood else "REAL TIME"], width(col) - 12, L.T_SMALL)
        f.text("w", x0, y + 34, B.resolve(txt, t - (T0 + 0.25 if t < self.t_flood else self.t_flood), 80.0, key=2),
               size=L.T_SMALL, alpha=0.8)
        # height ruler on the left of the figure: its ticks are thrown out from the ground up
        xr = self.fig_x - min(self.fig_half - 96.0, 300.0)
        yb = cam.cy + cam.scale * 0.92
        if xr > col[0] + 70:
            with f.build(t - T0 - 0.4, (xr - 90.0, yb - cam.scale * 1.8 - 12.0, xr + 4.0, yb + 12.0), flow="bt", wave=0.4,
                         marks=False, key=3):
                hud.vruler(f, xr, yb, yb - cam.scale * 1.8, 0.0, 1.8, 0.1, 0.5, fmt=lambda v: f"{v:.1f} M", right=False)
        if gone > 0.5:                  # where the heart was: a red cross is thrown out, the volume of air is decoded
            yh = cam.cy - cam.scale * (HEART[1] - 0.92)
            xh = self.fig_x + cam.scale * HEART[0]
            with f.build(t - self.t_half, (xh - 14.0, yh - 14.0, xh + 14.0, yh + 14.0), flow="out", wave=0.05,
                         marks=False, key=4):
                f.crosses("r", [xh], [yh], 12.0, 0.9, width=L.LW)
            msg = fit_text([f"1.80 M  x  {sd.AREA_YOU:.2f} M2  OF AIR", "1.80 M OF AIR", "AIR"], width(col), L.T_LABEL)
            f.text("w", self.fig_x, Y_BOT - 14, B.resolve(msg, t - self.t_gone + 1.6, 30.0, key=5, pad=True),
                   size=L.T_LABEL, alpha=0.85, anchor="ms")

    # ------------------------------------------------------------------ the count
    def _draw_count(self, f, t, ctx, col, gone):
        x0, x1 = col[0] + 16, col[1] - 6
        w = x1 - x0
        k = min(1.0, w / 740.0)                     # everything in this column scales with its width
        n_now = ctx.through_you(t)
        n_end = ctx.through_you(SHOW)
        fr = int(t * 30)
        age = t - T0 - 0.2                  # the column is constructed when the scene starts
        # the show as a bar
        y = Y_TOP + 36
        yb0, yb1 = y + 26, y + 58
        X = lambda tt: x0 + np.asarray(tt, np.float64) / SHOW * (x1 - x0)
        proj = float(smoothstep(self.t_fifty, self.t_fifty + 1.2, t))
        with f.build(age, (x0 - 8, y - 26, x1 + 8, yb1 + 40), flow="lr", wave=0.4, key=10):
            f.tag("w", x0, y, f"THIS SHOW // {sd.mmss(SHOW)}", size=L.T_LABEL, pad=5)
            f.rect("w", x0, yb0, x1, yb1, 0.8, width=L.LW)
            f.rects("w", x0 + 3, yb0 + 4, float(X(t)), yb1 - 4, 0.9)
            mins = np.arange(0, SHOW, 60.0)
            f.segments("w", X(mins), np.full(len(mins), yb1), X(mins), np.full(len(mins), yb1 + 9.0), EN.wl(0.8), width=EN.ww(1.0))
            if proj > 0:                    # "by the end": a red bar runs from now to the end of the show
                xe = float(X(t)) + (x1 - 3 - float(X(t))) * proj
                f.rects("r", float(X(t)), yb0 + 11, xe, yb1 - 11, 0.95)
            f.segments("r", [float(X(t))], [yb0 - 6], [float(X(t))], [yb1 + 12], 1.2, width=L.LW)
            f.text("r", float(X(t)) + 6, yb1 + 30, sd.tc(t)[:5], size=L.T_SMALL, alpha=0.95)
            f.text("w", x1, yb1 + 30, sd.mmss(SHOW), size=L.T_SMALL, alpha=0.7, anchor="rs")
        # so far
        y1_ = y + 142
        lab = "THROUGH YOU SO FAR" if gone < 0.5 else fit_text(["THROUGH THE SPACE YOU LEFT", "THROUGH THAT SPACE"], w,
                                                               L.T_TAG)
        big = float(np.clip((w - 10) / (6 * 0.61), 70.0, 176.0))
        yn = y1_ + 20 + 0.84 * big
        with f.build(age - 0.25, (x0 - 8, y1_ - 28, x1 + 8, yn + 46), wave=0.35, key=11):
            f.tag("w", x0, y1_, hud.erode(lab, 1.0 - abs(2 * gone - 1.0) ** 0.5 if 0 < gone < 1 else 0.0, 3, fr),
                  size=L.T_TAG, pad=5)
            f.text("w", x0 - 6 * k, yn, spaced(n_now), size=big, alpha=1.0, bold=True)
            f.text("w", x0, yn + 36, fit_text([f"+{RATE:.0f} EVERY SECOND  //  1 MUON /CM2 /MIN  //  {sd.AREA_YOU:.2f} M2",
                                               f"+{RATE:.0f} EVERY SECOND // 1 /CM2 /MIN", f"+{RATE:.0f} EVERY SECOND"],
                                              w, L.T_SMALL), size=L.T_SMALL, alpha=0.75)
        # the break: only the count stays. What leaves does not fade: its letters fall apart (1 - calm of them
        # are gone), then its tag and its rule are pulled back (`back` runs 1 -> 0 at the very end)
        calm = 1.0 - float(smoothstep(self.t_gone - 0.5, self.t_gone + 3.0, t))
        back = float(np.clip((calm - 0.01) / 0.15, 0.0, 1.0))
        # by the end
        y2 = yn + 106
        mid = 0.705 * big
        if proj > 0 and calm > 0.01:
            a2 = t - self.t_fifty           # made on "more than fifty thousand": tag pushed out, the count runs up
            s2 = "BY THE END OF THIS SHOW"
            txt, wp = B.tag_state(s2, a2, cps=70.0, key=12)
            f.tag("r", x0, y2, txt if calm >= 1.0 else hud.erode(s2, 1 - calm, 12, fr), size=L.T_TAG, pad=5, ref=s2,
                  wipe=min(wp, back))
            n_run = int(n_now + (n_end - n_now) * proj)
            f.text("r", x0 - 4 * k, y2 + 16 + 0.82 * mid, hud.erode(B.decode(spaced(n_run), a2, cps=50.0, key=13),
                                                                 1 - calm, 7, fr), size=mid, bold=True)
            xg = x0 + text_w(spaced(n_end), mid) + 26
            if proj >= 1.0 and xg + text_w("> 50 000", 34) <= x1:
                f.text("w", xg, y2 + 16 + 0.82 * mid, hud.erode(B.resolve("> 50 000", a2 - 1.2, 40.0, key=14), 1 - calm,
                                                                8, fr), size=34, alpha=0.9)
        # every time scale
        a = t - self.t_every
        if a > 0:
            rs = float(np.clip(w / 24.0, 18.0, 30.0))
            pitch = 1.47 * rs
            y3 = y2 + 16 + mid + 60
            rows = [("EVERY SECOND", sd.through_you_in(1)), ("EVERY MINUTE", sd.through_you_in(60)),
                    ("EVERY DAY", sd.through_you_in(86400)), ("EVERY YEAR", sd.through_you_in(sd.YEAR)),
                    ("IN 80 YEARS", sd.through_you_in(80 * sd.YEAR))]
            n_fit = int((Y_BOT - 6 - y3) / pitch)                # rows that fit above the bottom band (one is kept
            rows = rows[: max(0, n_fit - 1)]                     # for AFTER YOU)
            # the rule over the rows: drawn by a pen on "as they have", pulled back in the break
            B.pen(f, "w", x0, y3 - 1.33 * rs, x1, y3 - 1.33 * rs, float(B.ease(B.lin(a, 0.0, 0.3))) * back, 0.7,
                  width=L.LW, head=3.0 if a < 0.3 else 0.0)
            for q, (lab_, val) in enumerate(rows):          # a row every 0.22 s: decoded, its figure spins and locks
                ak = a - 0.22 * q
                if ak < 0:
                    continue
                yy = y3 + q * pitch
                hot = q == 1 and a < 4.6
                f.text("w", x0, yy, hud.erode(B.decode(lab_, ak, cps=60.0, key=20 + q), 1 - calm, 20 + q, fr), size=rs,
                       alpha=0.7)
                f.text("r" if hot else "w", x1, yy,
                       hud.erode(B.resolve(val, ak, 40.0, key=30 + q, pad=True, spin=0.4), 1 - calm, 30 + q, fr),
                       size=1.13 * rs, alpha=0.97, anchor="rs")
            aw = t - self.t_will
            if aw > 0 and n_fit >= 1:
                yy = y3 + len(rows) * pitch
                keep = max(calm, 0.9)                          # this line stays through the break
                f.text("w", x0, yy, B.decode("AFTER YOU", aw, cps=60.0, key=40), size=rs, alpha=0.7 * keep)
                f.text("r", x1, yy, B.decode("UNCHANGED", aw, cps=40.0, delay=0.2, key=41, pad=True), size=1.13 * rs,
                       alpha=0.97 * keep, anchor="rs")

    # ------------------------------------------------------------------ left column
    def _draw_left(self, f, t, col, gone):
        x0, x1 = col
        w = x1 - x0
        ts = min(100.0, (w - 40) / (5 * 0.61))
        age = t - T0 - 0.1                  # the column is constructed when the scene starts
        B.tag(f, "w", x0 + 10, Y_TOP + 28 + 0.93 * ts, "FLOOD", age, size=ts, pad=10, bold=True, wipe=0.2, cps=22.0,
              key=50)
        rate = RATE * (0.2 + 0.8 * float(smoothstep(self.t_flood - 0.1, self.t_flood + 1.0, t)))
        state = "     PRESENT" if gone < 0.05 else "      ABSENT" if gone > 0.98 else "     LEAVING"
        if gone >= 0.05:                    # the state of the target is decoded again when it changes
            state = B.decode(state, t - (self.t_absent if gone > 0.98 else self.t_leaving), cps=80.0, key=53, pad=True)
        rows = ["MU FLUX    1 /CM2/MIN", f"THROUGH YOU   {RATE:.0f} /S", f"ON SCREEN   {rate:4.1f} /S",
                "SKIN          1.9 M2", "TARGET   " + state]
        yr = Y_TOP + ts + 96
        with f.build(age - 0.15, (x0 - 6, yr - 24, x1, yr + 5 * 25.5), flow="tb", wave=0.25, marks=False, key=51):
            hud.rows(f, x0, yr, rows, size=L.T_SMALL, lead=1.5, red=(4,) if gone > 0.98 else ())
        yl = yr + 5 * 25.5 + 30
        full = w >= 300
        with f.build(age - 0.35, (x0, yl - 20, x1, yl + 8), flow="lr", wave=0.15, marks=False, key=52):
            f.tag("w", x0 + 4, yl, "ENTRY_LOG", size=L.T_MICRO, pad=3)
            if full:
                f.text("w", x0 + 110, yl, "PART   X     Y    Z    CM", size=L.T_MICRO, alpha=0.5)
        fr = int(t * 30)
        t_stop = self.t_leave + 0.6
        i1s = int(np.searchsorted(self.h_t, min(t, t_stop)))
        n_rows = int((Y_BOT - yl - 40) / 21)
        idx = [k for k in range(i1s - 1, max(i1s - 400, -1), -1) if self.h_has[k]][:n_rows]
        for row, k in enumerate(idx):
            e = self.h_entry[k]
            if full:
                line = f"{PART_NAMES[self.h_part[k]]:<7}{e[0]:+.2f} {e[1]:.2f} {e[2]:+.2f} {self.h_frac[k] * 100:5.1f}"
            else:
                line = f"{PART_NAMES[self.h_part[k]]:<7} {e[1]:.2f} M {self.h_frac[k] * 100:5.1f} CM"
            # a line is decoded when its muon enters (and the first ones with the log, when the scene starts)
            line = B.resolve(line, min(t - float(self.h_t[k]), age - 0.45 - 0.03 * row), 170.0, key=k & 0xFFFF)
            if gone > 0:
                line = hud.erode(line, gone * 1.05, 100 + row, fr)
            f.text("r" if row == 0 and gone < 0.3 else "w", x0, yl + 32 + row * 21, line, size=L.T_MICRO,
                   alpha=0.95 if row < 3 else 0.65)
        if gone > 0.98:
            f.text("r", x0, yl + 32, B.decode("NO TARGET", t - self.t_absent, cps=50.0, key=54), size=L.T_MICRO, alpha=0.95)
            n_through = int(np.searchsorted(self.h_t, t)) - int(np.searchsorted(self.h_t, self.t_gone))
            f.text("w", x0, yl + 53, B.decode(f"PASSED ANYWAY  {n_through:05d}", t - self.t_absent, cps=70.0, delay=0.2,
                                              key=55), size=L.T_MICRO, alpha=0.8)

    # ------------------------------------------------------------------ entry map
    def _draw_map(self, f, t, col, gone, there):
        xa, xb = col
        half = min((xb - xa) / 2 - 8, 200.0)
        cx, cy = xa + half + 8, Y_TOP + 62.0 + half
        sc = half / 0.46
        age0 = t - T0 - 0.35                # the map and the list are constructed when the scene starts
        with f.build(age0, (xa - 6, Y_TOP + 14, xa + 2 * half + 22, cy + half + 6), wave=0.4, key=60):
            f.tag("w", xa, Y_TOP + 36, "ENTRY_MAP // FROM ABOVE", size=L.T_MICRO, pad=3)
            f.rect("w", cx - half, cy - half, cx + half, cy + half, EN.wl(0.5), width=EN.ww(1.0))
            if EN.WALL:                      # the wall: one full ring and four ticks on the frame, no grey cross-hair
                f.rings("w", [cx], [cy], [sc * 0.2], 1.0, width=EN.WALL_LINE)
                f.segments("w", [cx - half, cx + half - 12.0, cx, cx], [cy, cy, cy - half, cy + half - 12.0],
                           [cx - half + 12.0, cx + half, cx, cx], [cy, cy, cy - half + 12.0, cy + half], 1.0,
                           width=EN.WALL_LINE)
            else:
                f.rings("w", [cx, cx], [cy, cy], [sc * 0.2, sc * 0.4], 0.3)
                f.segments("w", [cx - half, cx], [cy, cy - half], [cx + half, cx], [cy, cy + half], 0.22)
            f.text("w", cx + sc * 0.2 + 4, cy - 5, "0.2 M", size=L.T_MICRO, alpha=0.6)
            xb = min(xb, xa + 2 * half + 16)
            span = 13.0
            i0, i1 = np.searchsorted(self.h_t, t - span), np.searchsorted(self.h_t, t)
            if i1 > i0:
                th = self.h_t[i0:i1]
                has = self.h_has[i0:i1]
                late = th > self.t_leave                       # no body any more: every muon marks where it passed
                P = np.where((has & ~late)[:, None], self.h_entry[i0:i1], self.h_mid[i0:i1])
                use = (has & ~late) | late
                x, y = cx + P[:, 0] * sc, cy - P[:, 2] * sc
                ok = use & (np.abs(x - cx) < half - 3) & (np.abs(y - cy) < half - 3)
                age = t - th
                f.dots("r", x[ok], y[ok], 2.3, (0.25 + 0.95 * np.exp(-age[ok] / 3.5)))
                new = ok & (age < 0.12)
                f.dots("w", x[new], y[new], 2.0, 1.2)
        # entries per part of the body, last 10 s
        yb = cy + half + 62
        with f.build(age0 - 0.3, (xa - 6, yb - 44, xb + 6, yb + 10 + len(GROUPS) * 34 + 4), flow="tb", wave=0.3, key=61):
            f.tag("w", xa, yb - 22, "ENTRIES // LAST 10 S", size=L.T_MICRO, pad=3)
            j0 = np.searchsorted(self.h_t, t - 10.0)
            j1 = np.searchsorted(self.h_t, min(t, self.t_leave + 0.6))
            g = self.h_group[j0:max(j1, j0)]
            for k, (name, _) in enumerate(GROUPS):
                n = int((g == k).sum())
                yy = yb + 10 + k * 34
                f.text("w", xa, yy + 16, name, size=L.T_SMALL, alpha=0.85)
                f.rects("r" if name == "TORSO" else "w", xa + 84, yy + 2,
                        xa + 84 + min(xb - xa - 150, n * (xb - xa - 150) / 240.0), yy + 18, 0.95)
                f.text("w", xb, yy + 16, f"{n:03d}", size=L.T_SMALL, alpha=0.85, anchor="rs")
        if gone > 0.98:
            f.text("r", xa, yb + 10 + len(GROUPS) * 34 + 22, B.decode("NOTHING IN THE WAY", t - self.t_absent, cps=50.0,
                                                                      key=62), size=L.T_SMALL, alpha=0.9)
        # the energy of the muons, under the list (dropped when the towers leave no room for it)
        y_e = yb + 10 + len(GROUPS) * 34 + 56
        if Y_BOT - y_e >= 126.0 and xb - xa >= 300.0:
            self._draw_energy(f, t, xa, xb, y_e, age0 - 0.6)

    def _draw_energy(self, f, t, xa, xb, y_t, age):
        """MUON_ENERGY: the energy of every muon of the last ten seconds, as a histogram on a log axis.
        An entry falls into its bin as a red drop and its bar takes it; the stepped line is what the law
        gives (the spectrum of the muons at the ground: see _spectrum), and the bars settle on it as the count
        grows. It does not change when the body leaves: the rain is the same."""
        j0, j1 = np.searchsorted(self.h_t, t - E_SPAN - 0.3), np.searchsorted(self.h_t, t)
        sel = j0 + np.nonzero(self.h_has[j0:j1])[0]
        a = t - self.h_t[sel]
        E = self.h_E[sel]
        le = np.clip(np.log10(E / E_LO) / math.log10(E_HI / E_LO), 0.0, 0.9999)      # place on the axis, 0..1
        b = (le * E_BINS).astype(np.int64)
        w = np.asarray(smoothstep(E_FALL - 0.04, E_FALL + 0.1, a) * (1.0 - smoothstep(E_SPAN, E_SPAN + 0.3, a)), np.float64)
        n_eff = float(w.sum())
        top = float(self.e_share.max()) * 1.35                  # share of the tallest bin of the law = 74 % of the plot
        h = np.clip(np.bincount(b, weights=w, minlength=E_BINS) / math.hypot(n_eff, 40.0) / top, 0.0, 1.0)
        x0, x1 = xa + 2.0, xb - 2.0
        bw = (x1 - x0) / E_BINS
        y0 = y_t + 16.0
        y1 = min(Y_BOT - 30.0, y0 + 150.0)                      # it takes the room the column leaves, up to 150 px
        H = y1 - y0
        xs = x0 + np.arange(E_BINS) * bw
        yt = y1 - h * H                                         # top of every bar
        with f.build(age, (xa - 6.0, y_t - 22.0, xb + 6.0, y1 + 28.0), flow="tb", wave=0.3, key=63):
            f.tag("w", xa, y_t, fit_text(["MUON_ENERGY // GEV // LAST 10 S", "MUON_ENERGY // GEV", "ENERGY // GEV"],
                                         xb - xa - 12.0, L.T_MICRO), size=L.T_MICRO, pad=3)
            f.rects("w", xs + 1.5, yt, xs + bw - 1.5, y1, 0.92)
            # the law, as the outline of the histogram it predicts
            yl = y1 - self.e_share / top * H
            f.segments("w", xs, yl, xs + bw, yl, EN.wl(0.5), width=EN.ww(1.0))
            f.segments("w", xs[1:], yl[:-1], xs[1:], yl[1:], EN.wl(0.5), width=EN.ww(1.0))
            # log axis: 0.1 - 1 - 10 - 100 GeV
            f.segments("w", [x0], [y1 + 1.0], [x1], [y1 + 1.0], EN.wl(0.7), width=EN.ww(1.0))
            dec = np.arange(4) / 3.0
            mn = (np.arange(3)[:, None] + np.log10(np.arange(2, 10))[None, :]).ravel() / 3.0
            f.segments("w", x0 + dec * (x1 - x0), np.full(4, y1 + 1.0), x0 + dec * (x1 - x0), np.full(4, y1 + 10.0), EN.wl(0.8), width=EN.ww(1.0))
            f.segments("w", x0 + mn * (x1 - x0), np.full(len(mn), y1 + 1.0), x0 + mn * (x1 - x0), np.full(len(mn), y1 + 5.0), EN.wl(0.6), width=EN.ww(1.0))
            for k, (lab, anc) in enumerate((("0.1", "ls"), ("1", "ms"), ("10", "ms"), ("100", "rs"))):
                f.text("w", x0 + dec[k] * (x1 - x0), y1 + 26.0, lab, size=L.T_MICRO, alpha=0.7, anchor=anc)
            # read-outs, over the high energies (their bars are low)
            if len(sel):
                f.text("r", x1, y0 + 12.0, f"LAST {float(E[-1]):6.2f} GEV", size=L.T_MICRO, alpha=0.95, anchor="rs")
            if n_eff > 0.5:
                f.text("w", x1, y0 + 30.0, f"MEAN {float((w * E).sum() / n_eff):6.2f} GEV", size=L.T_MICRO, alpha=0.85,
                       anchor="rs")
            f.text("w", x1, y0 + 48.0, f"N {int(round(n_eff)):04d}", size=L.T_MICRO, alpha=0.6, anchor="rs")
        if age < 0.3:
            return
        # every entry: a red drop falling on its bin (it speeds up on the way), a red cap where it lands
        fall = a < E_FALL
        if fall.any():
            af = a[fall]
            xd = x0 + le[fall] * (x1 - x0)
            yd = (y0 + 10.0) + (yt[b[fall]] - y0 - 10.0) * (af / E_FALL) ** 2
            f.segments("r", xd, yd - 10.0 * np.minimum(af / 0.06, 1.0), xd, yd, 0.15, 1.3, width=1.6)
        hot = np.bincount(b, weights=np.where(fall, 0.0, np.exp(-(a - E_FALL) / 0.12)), minlength=E_BINS)
        m = hot > 0.03
        f.rects("r", xs[m] + 1.5, yt[m] - 3.0, xs[m] + bw - 1.5, yt[m], np.minimum(hot[m], 1.0))

    # ------------------------------------------------------------------ bottom band
    def _draw_bottom(self, f, t, ctx):
        P = self.P
        y0, y1 = P.py0, P.py1
        panels = [p for p in P.panels if width(p) >= 130.0]
        wide = [p for p in panels if width(p) >= 300.0]
        small = [p for p in panels if width(p) < 300.0]
        # the panels are constructed when the scene starts, one after the other
        block = lambda p, k: f.build(t - T0 - 0.3 - 0.15 * k, (p[0] - 8, y0 - 24, p[1] + 8, y1 + 8), wave=0.4, key=70 + k)
        if wide:
            x0, x1 = wide[0]
            with block(wide[0], 0):
                hud.panel_header(f, x0, x1, y0, "MUONS THROUGH YOU")
                cols = [("PER SECOND", sd.through_you_in(1)), ("PER MINUTE", sd.through_you_in(60)),
                        ("IN A LIFE", sd.through_you_in(80 * sd.YEAR))]
                n = int(min(3, max(1, (x1 - x0) // 150)))
                cols = cols if n == 3 else [cols[0], cols[2]] if n == 2 else [cols[1]]
                cw = (x1 - x0) / n
                for k, (lab, val) in enumerate(cols):
                    f.text("w", x0 + k * cw + 4, y0 + 36, lab, size=L.T_MICRO, alpha=0.75)
                    f.text("r" if lab == "IN A LIFE" else "w", x0 + k * cw + 2, y0 + 96, val, size=40, alpha=0.97)
        bar = wide[1] if len(wide) > 1 else None
        rest = small + wide[2:]
        if not rest and bar and width(bar) >= 520.0:        # two panels only: the rate takes the end of the barcode's
            bar, rest = (bar[0], bar[1] - 196.0), [(bar[1] - 170.0, bar[1])]
        if bar:
            x0, x1 = bar
            with block(bar, 1):
                hud.panel_header(f, x0, x1, y0, "ENTRY_BARCODE // 3 S")
                n = int(np.clip((x1 - x0) / 3.2, 40, 200))
                kk, frac, dt = hud.barcode_keys(t, 3.0, n)
                tt = kk * dt
                lo, hi = np.searchsorted(self.h_t, tt), np.searchsorted(self.h_t, tt + dt)
                dens = 0.04 + 0.92 * np.tanh((hi - lo) / 3.2 * (150.0 / n))
                quiet = float(smoothstep(self.t_leave, self.t_gone, t))
                hud.barcode_lanes(f, x0, x1, y0 + 12, y1, dens, kk, lanes=3, seed=5, inten=EN.wl(0.95 - 0.5 * quiet),
                                  frac=frac)
        if rest:
            x0, x1 = rest[0]
            with block(rest[0], 2):
                hud.panel_header(f, x0, x1, y0, "RATE")
                f.text("w", x0 + 2, y0 + 78, f"{RATE:.0f}", size=58)
                f.text("w", x0 + 78, y0 + 78, "/S", size=24, alpha=0.8)
                f.text("w", x0 + 2, y0 + 110, "UNCHANGED", size=L.T_MICRO, alpha=0.75)
