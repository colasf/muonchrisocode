"""DANCE - the air-shower groove.   Sheet scene 6, 05:22 - 06:55 (the loud section, 125 BPM).

The liked air-shower loop put on the grid of the track and built around the three towers:
one shower per 4-bar phrase (7.68 s, 12 phrases from the drop at 05:22.83), hard cut on every bar:

  bar 1  PERSPECTIVE   first interaction on the downbeat (2-frame inversion of the view), slow push-in
  bar 2  ORTHO_SIDE    two elevations (X and Z), one in each of the two widest bays the towers leave free,
                       tracking the front: altitude rules, slant depth, red front line across the wall
  bar 3  ORTHO_TOP     the front lands on the downbeat. Plan view locked on the towers: seen from
                       above, the head of each tower IS its detector, the footprint blooms around it
  bar 4  ORTHO_FRONT   elevation locked on the towers (1 km = the same px in x and y): the wall is the
                       air above the detectors; what reached the ground stands on it as a comb, read by
                       a scan line; lone muons land on the beat

Showers fall in turn on DET_L / DET_C / DET_R (or between them): their muons go *through* the head of
the tower (white tick). The towers themselves only answer to the live detector streams (faces, blooms
and scopes are drawn by the show); every live hit also gets its red muon track here. Energy rises over
the 12 phrases, barcode bursts close the phrases more and more often. 05:22.0 - 05:22.83 is the pickup:
the primary alone, coming down. The GLITCH scene keeps this grid running (phrases 13-17) and corrupts it.

Nothing has a fixed x: the mapping km <-> wall, the bays of the two elevations, the particle column
(an edge column of ctx.cols, on the side away from the shower when both are wide enough), the bottom
panels (ctx.slots) and every label are derived from the tower rectangles; text that a tower would hide
is moved or dropped (see the helpers in shower.py).
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from .. import hud
from .. import layout as L
from ..engine import Camera, OrthoCamera, hash01
from ..show import Scene
from .shower import (K_RED, VIEW_Y0, VIEW_Y1, Stage, World, _free, altitude_rules, bottom_panels, chord, draw_info,
                     flow, info_layout, put_right, put_tag, put_text, tbox, tower_boxes, view_bays, zone)

T0 = 322.826                    # downbeat of the drop (the biggest kick around 05:22)
BEAT = 0.48                     # 125 BPM
BAR = 4 * BEAT
PHRASE = 4 * BAR
N_PHRASES = 12                  # scene 6: 05:22.83 - 06:54.99
N_BUILD = 17                    # ... and 5 more phrases for scene 7 (glitch), to 07:33
T_END = T0 + N_PHRASES * PHRASE
AIMS = [0, 1, None, 2, 1, 0, None, 2, 1, 0, 2, 1, None, 1, 0, 2, 1]        # detector index (L, C, R) or a bay
BURSTS = {3, 5, 7, 8, 9, 10, 11, 13, 14, 15}                               # phrases closed by a barcode burst
KINDS = ("persp", "side", "top", "front")
NAMES = {"persp": "PERSPECTIVE", "side": "ORTHO_SIDE", "top": "ORTHO_TOP", "front": "ORTHO_FRONT"}
Y_GROUND = 1190.0               # screen y of altitude 0 in the front elevation
LEGEND = ("LATERAL_DISTRIBUTION // N PER 125 M // AT GROUND {n:05d}", "LATERAL // AT GROUND {n:05d}")
PANEL_BLOCKS = [("count", 200.0, 470.0), ("shower", 130.0, 146.0), ("bar", 160.0, 0.0, True)]     # bottom band

_WORLDS = {}


class Geo:
    """World <-> wall mapping derived from the tower placement: L and R stand 18 km apart (the scale is kept
    between 70 and 125 px per km, so towers standing close together are simply less than 18 km apart)."""

    def __init__(self, towers):
        tl, tc, tr = towers["L"], towers["C"], towers["R"]
        self.S = float(np.clip((tr.cx - tl.cx) / 18.0, 70.0, 125.0))
        self.x_mid = 0.5 * (tl.cx + tr.cx)
        self.y_plan = 0.5 * (tl.top + tr.top)
        self.dets = [((t.cx - self.x_mid) / self.S, (Y_GROUND - t.top) / self.S, (t.top - self.y_plan) / self.S)
                     for t in (tl, tc, tr)]
        sig = [round(v, 2) for t in (tl, tc, tr) for v in (t.x0, t.x1, t.top, t.bot)]
        self.key = zlib.crc32(repr(sig).encode()) % 100000

    def plan_xz(self, sx, sy):
        return (sx - self.x_mid) / self.S, (sy - self.y_plan) / self.S


def get_world(ctx):
    """The shower world of the groove, shared by Dance and Glitch (built once per process and per tower
    placement, cached on disk: about 35 s the first time)."""
    geo = Geo(ctx.towers)
    if geo.key in _WORLDS:
        return _WORLDS[geo.key], geo
    rng = np.random.default_rng(606)
    wide = sorted(sorted(ctx.bays, key=lambda b: b[0] - b[1])[:2])             # the two widest free bays
    y_free = 0.5 * (max(t.top for t in ctx.towers.values()) + Y_GROUND)        # plan view: a row clear of the heads
    showers = []
    for k in range(N_BUILD):
        aim = AIMS[k]
        if aim is None:
            b = wide[0] if (k // 2) % 2 == 0 else wide[-1]
            spread = min(90.0, 0.2 * (b[1] - b[0]))
            gx, gz = geo.plan_xz(0.5 * (b[0] + b[1]) + rng.uniform(-spread, spread), y_free + rng.uniform(-60, 60))
            zen = rng.uniform(11.0, 16.0)
        else:
            gx, gz = geo.dets[aim][0] + rng.uniform(-0.1, 0.1), geo.dets[aim][2] + rng.uniform(-0.1, 0.1)
            zen = rng.uniform(3.0, 6.0)
        showers.append(dict(t_int=T0 + k * PHRASE, ground=(gx, gz), E0=min(1.0, 0.5 + 0.5 * k / (N_PHRASES - 1)),
                            zen=zen, az=rng.uniform(0, 360), h1=rng.uniform(13.6, 14.8), travel=2 * BAR, aim=aim,
                            n_aim=3))
    x_lo, x_hi = geo.plan_xz(L.FX0 + 80.0, 0.0)[0], geo.plan_xz(L.FX1 - 80.0, 0.0)[0]
    rain = []
    for b in range(N_BUILD * 16):                 # lone muons on the beat, more of them in bars 3-4 and later on
        k, bar = b // 16, (b // 4) % 4
        level = min(1.0, k / (N_PHRASES - 1))
        if bar < 2:
            n = int(rng.random() < 0.25 + 0.3 * level)
        else:
            n = 1 + int(rng.random() < 0.3 + 0.5 * level) + int(rng.random() < 0.35 * level)
        for j in range(n):
            tl = T0 + b * BEAT + (0.0 if j == 0 else 0.5 * BEAT * (j % 2) + 0.25 * BEAT * (j // 2))
            if rng.random() < 0.15:
                d = int(rng.integers(0, 3))
                x, alt, z = geo.dets[d]
                rain.append(dict(t_land=tl, target=(x + rng.uniform(-0.12, 0.12), alt, z + rng.uniform(-0.12, 0.12)),
                                 det=d))
            else:
                rain.append(dict(t_land=tl, target=(rng.uniform(x_lo, x_hi), 0.0, rng.uniform(-4.0, 4.6))))
    w = World(showers, rain, dets=geo.dets, seed=6, cache=f"dance_{geo.key}_{int(T0 * 1000)}")
    _WORLDS[geo.key] = w
    return w, geo


def grid(t):
    """(phrase, bar in phrase, progress in the bar, time since the phrase downbeat) on the 125 BPM grid."""
    if t < T0:
        return 0, 0, 0.0, t - T0
    r = t - T0
    p = min(int(r // PHRASE), N_BUILD - 1)
    rp = r - p * PHRASE
    bar = min(int(rp // BAR), 3)
    return p, bar, (rp - bar * BAR) / BAR, rp


class Dance(Scene):
    name = "dance"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.world, self.geo = get_world(ctx)
        self._arr = {}
        self._stages = {"L": Stage(ctx, "L"), "R": Stage(ctx, "R"), "N": Stage(ctx, None)}
        self._elev = {}

    # ------------------------------------------------------------------ layout of a phrase
    def stage(self, p):
        """Particle column away from the shower: on the right while it falls on the left tower. When the
        placement only leaves a column right next to the tower the shower falls on, the column is dropped
        for that phrase: the footprint matters more than the list."""
        aim = AIMS[p]
        st = self._stages["R" if aim == 0 else "L"]
        if st.col is not None and aim is not None:
            tw = self.ctx.towers[L.ORDER[aim]]
            gap = (st.col[0] - tw.x1) if st.side == "R" else (tw.x0 - st.col[2])
            if gap < 420.0:
                return self._stages["N"]
        return st

    def elev(self, st):
        """Bays, split x and scale (px per km) of the elevations of bar 2 on a stage: the two widest free
        bays of its view (or the two halves of one very wide bay, or a single bay)."""
        if st.side not in self._elev:
            bays, split = view_bays(self.ctx, st.view)
            w_min = min(b[1] - b[0] for b in bays)
            self._elev[st.side] = (bays, split, float(np.clip((w_min / 2 - 10) / 3.4, 60.0, 92.0)))
        return self._elev[st.side]

    def _stage_x(self, p, st):
        """Screen x of the shower axis in the perspective view: the middle of one of the elevation bays."""
        bays = self.elev(st)[0]
        aim = AIMS[p]
        b = bays[0] if aim == 0 else bays[-1] if aim == 2 else bays[0 if p % 2 == 0 else -1]
        return 0.5 * (b[0] + b[1])

    def zones(self, ctx, st, p, kind):
        """x-intervals taken by the cascade(s) of the current bar: (at the top of the view, over its height)."""
        geo = self.geo
        if kind == "persp":
            x = self._stage_x(p, st)
            return [(x - 150.0, x + 150.0)], [zone(ctx, x, 330.0)]
        if kind == "side":
            bays, _, s_side = self.elev(st)
            xs = [0.5 * (b[0] + b[1]) for b in bays]
            return [(x - 150.0, x + 150.0) for x in xs], [zone(ctx, x, 3.4 * s_side, b) for x, b in zip(xs, bays)]
        e = self.world.events[p]
        x = geo.x_mid + float(e["G"][0]) * geo.S
        if kind == "top":                         # the dense part of the footprint, where it reaches the top
            ch = chord(x, geo.y_plan + float(e["G"][2]) * geo.S, 2.4 * geo.S, VIEW_Y0 + 100.0)
            return ([ch] if ch else []), [zone(ctx, x, 4.0 * geo.S)]
        return [(x - 1.5 * geo.S, x + 1.5 * geo.S)], [zone(ctx, x, 2.5 * geo.S)]

    def cameras(self, p, bar, u, t, st):
        """[(camera, clip rect)] of the current bar: one view, or the two elevations of the side bar."""
        w, geo = self.world, self.geo
        view = st.view
        e = w.events[p]
        gx, gz = float(e["G"][0]), float(e["G"][2])
        kind = KINDS[bar]
        if kind == "persp":
            yaw = math.radians(-24.0 + 47.0 * p + 16.0 * u)
            D = 27.0 - 5.0 * u
            ty = 9.4 - 2.0 * u
            cam = Camera((gx + D * math.sin(yaw), ty - 0.4, gz + D * math.cos(yaw)), (gx, ty, gz), fov_deg=44.0,
                         screen_center=(self._stage_x(p, st), st.cy))
            cam.fov = 44.0
            return kind, [(cam, view)]
        if kind == "side":
            bays, split, s_side = self.elev(st)
            half = 0.5 * (VIEW_Y1 - VIEW_Y0)
            c_lo = (half - 30.0) / s_side
            c_hi = max(c_lo, e["h1"] - (half - 130.0) / s_side)
            cy = float(np.clip(w.front(p, t - e["t0"]) + 1.2, c_lo, c_hi))
            out = []
            for j, bay in enumerate(bays):
                yaw = math.radians(90.0 * j + 6.0 * (p % 3 - 1))
                cam = OrthoCamera((gx + 60 * math.sin(yaw), cy, gz + 60 * math.cos(yaw)), (gx, cy, gz),
                                  scale=s_side, screen_center=(0.5 * (bay[0] + bay[1]), st.cy))
                clip = view if split is None else ((view[0], view[1], split, view[3]) if j == 0 else
                                                   (split, view[1], view[2], view[3]))
                out.append((cam, clip))
            return kind, out
        if kind == "top":
            cam = OrthoCamera((0.0, 40.0, 1e-3), (0.0, 0.0, 0.0), scale=geo.S, up=(0.0, 0.0, -1.0),
                              screen_center=(geo.x_mid, geo.y_plan))
            return kind, [(cam, view)]
        cam = OrthoCamera((0.0, 0.0, 60.0), (0.0, 0.0, 0.0), scale=geo.S, screen_center=(geo.x_mid, Y_GROUND))
        return kind, [(cam, view)]

    # ------------------------------------------------------------------ draw
    def draw(self, f, t, ctx, gain=1.0, hud_alpha=1.0, flash=True, bursts=True, hide=(), extended=False):
        """`gain` scales the world, `hud_alpha` the HUD; `hide` = shower indices not drawn and `extended` =
        keep the grid running after phrase 12 (both used by GLITCH)."""
        w = self.world
        if not extended:
            t = min(t, T_END - 1e-3)
        p, bar, u, rp = grid(t)
        e = w.events[p]
        a = t - e["t0"]
        age, alive, env = w.state(t)
        for k in hide:
            alive[k] = False
        st = self.stage(p)
        view = st.view
        kind, cams = self.cameras(p, bar, u, t, st)
        kick = min(1.5, ctx.cues.kick(t))
        level = min(1.0, p / (N_PHRASES - 1))
        pick = t < T0                                     # the pickup before the drop: almost nothing yet
        if pick:
            gain, hud_alpha = gain * 0.45, hud_alpha * 0.4
        burst = self.burst(p, rp) if bursts else None
        aim = AIMS[p]
        on = L.NAMES[L.ORDER[aim]] if aim is not None else None
        z_top, z_full = self.zones(ctx, st, p, kind)
        name = NAMES[kind] + (" X / Z" if kind == "side" and len(cams) > 1 else "")
        lay = info_layout(ctx, st, f"VIEW {bar + 1:02d} // {name}", cams[0][0],
                          [(f"AIR_SHOWER // MUON BLOOM // SHOWER {p + 1:02d}/{N_PHRASES:02d}", 0.85),
                           (f"E0 {e['E0'] * 3.2:.2f}E15 EV   ZENITH {e['zen']:04.1f} DEG", 0.6),
                           (f"FALLS ON {on}" if on else "FALLS BETWEEN THE DETECTORS", 0.6)],
                          [(f"AIR_SHOWER {p + 1:02d}/{N_PHRASES:02d}", 0.85), (f"E0 {e['E0'] * 3.2:.2f}E15 EV", 0.6),
                           (f"ZENITH {e['zen']:04.1f} DEG", 0.6), (f"ON {on}" if on else "BETWEEN DETECTORS", 0.6)],
                          z_top, z_full)
        if kind == "front" and lay["a"] is not None:          # room for the legend of the comb, under the view tag
            for k, txt in enumerate(LEGEND):
                box = tbox(lay["a"], lay["y0"] + 92, txt.format(n=0), L.T_SMALL)
                if _free(ctx, st, box, lay["zones_top"], lay["boxes"][1:]):
                    lay["legend"] = k
                    lay["boxes"].append(box)
                    break
        f.set_clip(*view)
        if burst is not None:
            hud.barcode_burst(f, view, t, burst, lanes=9, seed=p)
        else:
            avoid = tower_boxes(ctx) + lay["boxes"]
            for j, (cam, clip) in enumerate(cams):
                if j == 0:
                    f.set_clip(*view)
                    w.draw_ground(f, cam, kind, view, gain=gain * (1.0 + 0.45 * kick) * (2.6 if kind == "persp" else 1.0))
                    if kind in ("side", "front"):
                        altitude_rules(f, ctx, st, cam, float(e["G"][0]), float(e["G"][2]), lay, gain=gain,
                                       front=w.front(p, a) if kind == "side" else 0.0)
                f.set_clip(*clip)
                g = gain * (1.0 + 0.16 * kick) * (1.7 if kind == "front" else 1.0)
                w.draw_cascades(f, cam, age, alive, env, gain=g)
                w.draw_hits(f, cam, age, alive, gain=gain)
                w.draw_splash(f, cam, age, gain=gain)
                self._overlay(f, ctx, st, lay, cam, kind, p, a, clip, j, gain)
                boxes = w.draw_interaction(f, cam, age, clip, ctx, tags=(j == 0), avoid=lay["boxes"])
                if kind == "front":
                    self._lateral(f, ctx, st, lay, cam, p, a, u, gain)
                if kind in ("top", "front"):
                    self._crossings(f, cam, age, alive)
                w.draw_labels(f, cam, age, alive, clip, avoid=avoid + boxes,
                              limit=(6 + int(3 * level)) // len(cams) + 1)
            f.set_clip(*view)
            if kind != "top":
                self._strikes(f, t, ctx, st)
        if not pick and gain > 0.3:
            self._aim_mark(f, ctx, st, p, t)
        f.set_clip()
        w.draw_column(f, p, a, st.col, alpha=hud_alpha)
        w.draw_strip(f, p, a, label=f"LONGITUDINAL_PROFILE // SHOWER {p + 1:02d}/{N_PHRASES:02d} // 125.0 BPM",
                     alpha=hud_alpha, pulse=kick)
        self._panels(f, t, p, bar, a, ctx, kick, hud_alpha)
        if hud_alpha > 0.05 and burst is None:
            draw_info(f, ctx, lay, alpha=hud_alpha)
        inv = flash and t >= T0 and 0.0 <= rp < 0.05
        return {"invert": bool(inv), "invert_rect": view}

    @staticmethod
    def burst(p, rp):
        """Progress 0..1 of the barcode burst closing phrase p, or None."""
        if p not in BURSTS:
            return None
        n = 2 if p == N_PHRASES - 1 else 1
        start = PHRASE - n * BEAT
        if rp >= start:
            return float(min(1.0, (rp - start) / (n * BEAT)))
        return None

    def _overlay(self, f, ctx, st, lay, cam, kind, p, a, clip, j, gain):
        w = self.world
        view = st.view
        e = w.events[p]
        front = w.front(p, a)
        al = min(1.0, gain)
        if kind == "side":
            if j == 0 and 0 < front < e["h1"]:
                P = np.array([[float(e["G"][0]), front, float(e["G"][2])]], np.float32)
                _, py, _, _ = cam.project(P)
                y = float(py[0])
                f.set_clip(*view)
                f.segments("r", [view[0]], [y], [view[2]], [y], 0.9 * gain, width=L.LW)
                put_right(f, ctx, st, lay, "r", y - 9, f"FRONT {front:06.3f} KM", size=L.T_LABEL, pad=5, alpha=al)
                f.set_clip(*clip)
            ax, ay, _, _ = cam.project(np.stack([e["G"], e["P1"]]).astype(np.float32))
            f.segments("r", [ax[0]], [ay[0]], [ax[1]], [ay[1] - 60], 0.5 * gain)
            if self.elev(st)[1] is not None:
                put_text(f, ctx, "w", float(ax[0]) + 14, min(float(ay[0]) - 12, view[3] - 30),
                         "ELEVATION X" if j == 0 else "ELEVATION Z", size=L.T_SMALL, alpha=0.7 * al)
        elif kind == "top":
            gx, gy, _, _ = cam.project(e["G"][None].astype(np.float32))
            X, Y = float(gx[0]), float(gy[0])
            big = 1e5
            f.segments("r", [X - big, X], [Y, Y - big], [X + big, X], [Y, Y + big], 0.45 * gain)
            sc = cam.scale
            f.rings("w", [X] * 4, [Y] * 4, [sc * r for r in (1, 2, 4, 8)], 0.25 * gain)
            for r in (1, 2, 4, 8):
                if view[0] + 30 < X + sc * r + 7 < st.tx1 - 60:
                    put_text(f, ctx, "w", X + sc * r + 7, Y - 8, f"{r} KM", size=L.T_SMALL, alpha=0.6 * al)
            aim = AIMS[p]
            xr = X + 64 if aim is None else max(X + 64, ctx.towers[L.ORDER[aim]].x1 + 22)
            xl = X - 64 if aim is None else min(X - 64, ctx.towers[L.ORDER[aim]].x0 - 22)
            txt = f"CORE {float(e['G'][0]):+07.3f} {float(e['G'][2]):+07.3f}"
            for x, anchor in ((xr, "ls"), (xl, "rs")):
                if _free(ctx, st, tbox(x, Y + 150, txt, L.T_SMALL, anchor, 4), (), lay["boxes"], pad=4.0):
                    f.tag("r", x, Y + 150, txt, size=L.T_SMALL, pad=4, alpha=al, anchor=anchor)
                    break
        elif kind == "front":
            ax, ay, _, _ = cam.project(np.stack([e["G"], e["P1"]]).astype(np.float32))
            f.segments("r", [ax[0]], [ay[0]], [ax[1]], [ay[1]], 0.4 * gain)
            put_right(f, ctx, st, lay, "w", Y_GROUND - 9, "GROUND 00 KM", size=L.T_SMALL, pad=4, alpha=al)

    def _arrivals(self, k):
        """Everything of shower k that reached the ground: (x in km, age at arrival, is a muon)."""
        if k not in self._arr:
            w = self.world
            s0, s1 = w.ranges[k]
            end = w.SA[s0:s1] + 1
            m = w.V[end, 1] < 1e-3
            self._arr[k] = (w.V[end[m], 0].astype(np.float64), w.VT[end[m]].astype(np.float64),
                            K_RED[w.SK[s0:s1][m]] == 1)
        return self._arr[k]

    def _lateral(self, f, ctx, st, lay, cam, p, a, u, gain):
        """Bar 4: what reached the ground, as a comb standing on the ground line (count per 125 m, muons in
        red), read by a scan line that crosses the wall in one bar. When the shower fell on a tower, the comb
        opens around it (the two halves start at the edges of the tower) so its peak is not hidden."""
        view = st.view
        gx, gt, mu = self._arrivals(p)
        S = cam.scale
        bw = 0.125
        core = float(self.world.events[p]["G"][0])
        aim = AIMS[p]
        shift = 0.0 if aim is None else (0.5 * ctx.towers[L.ORDER[aim]].w + 14.0)
        lo = math.floor(((view[0] - cam.cx) / S - core) / bw) * bw
        edges = np.arange(lo, (view[2] - cam.cx) / S - core + bw, bw)            # distance to the core (km)
        m = gt <= a
        c_all, _ = np.histogram(gx[m] - core, edges)
        c_mu, _ = np.histogram(gx[m & mu] - core, edges)
        norm = max(int(c_all.max()), 1)
        grow = 1.0 - (1.0 - min(1.0, u / 0.2)) ** 3
        h = 330.0 * np.sqrt(c_all / norm) * grow
        hm = 330.0 * np.sqrt(c_mu / norm) * grow
        mid = edges[:-1] + bw / 2
        xs = cam.cx + (core + edges[:-1]) * S + np.sign(mid) * shift
        w_ = bw * S
        k = c_all > 0
        f.rects("w", xs[k] + 2, Y_GROUND - h[k], xs[k] + w_ - 3, Y_GROUND - hm[k] - 1, 0.62 * gain)
        k = c_mu > 0
        f.rects("r", xs[k] + 2, Y_GROUND - hm[k], xs[k] + w_ - 3, Y_GROUND - 1, 0.95 * gain)
        xc = view[0] + u * (view[2] - view[0])
        i = int(np.argmin(np.abs(xs + w_ / 2 - xc)))
        f.segments("r", [xc], [Y_GROUND - 372], [xc], [Y_GROUND], 0.9 * gain, width=L.LW)
        anchor = "ls" if xc < view[2] - 440 else "rs"
        put_tag(f, ctx, "r", xc + (8 if anchor == "ls" else -8), Y_GROUND - 356,
                f"R {mid[i]:+06.2f} KM  N {int(c_all[i]):04d}  MU {int(c_mu[i]):03d}", size=L.T_SMALL, pad=4,
                anchor=anchor, alpha=min(1.0, gain))
        if "legend" in lay:                   # its legend, under the view tag
            f.text("w", lay["a"], lay["y0"] + 92, LEGEND[lay["legend"]].format(n=int(m.sum())), size=L.T_SMALL,
                   alpha=0.75 * min(1.0, gain))

    def _crossings(self, f, cam, age, alive):
        """Muons of the showers (and of the beat rain) going through the head of a tower: a white tick."""
        w = self.world
        for ev, det, ta, e in w.crossings:
            if not alive[ev]:
                continue
            a = age[ev] - ta
            if not (0.0 <= a < 0.8):
                continue
            px, py, _, _ = cam.project(np.array([w.dets[det]], np.float32))
            x, y = float(px[0]), float(py[0])
            u = a / 0.8
            f.rings("w", [x], [y], [10.0 + 54.0 * (1 - (1 - u) ** 3)], 0.9 * (1 - u) ** 1.5, width=L.LW)
            f.dots("w", [x], [y], 3.6 * (1 - u) + 1.0, 1.6 * (1 - u))

    def _strikes(self, f, t, ctx, st):
        """Every live hit of a detector is a muon: its track, straight down into the head of the tower (the
        latest one carries its energy, on the side of the track where there is room)."""
        view = st.view
        for key in L.ORDER:
            if not ctx.det.online(key, t):
                continue
            tw = ctx.towers[key]
            tt, ee, ec = ctx.det.hits(key, t - 0.9, t + 1e-6, echoes=False)
            for k, (th, e) in enumerate(zip(tt, ee)):
                a = t - th
                fade = math.exp(-a / 0.28)
                if fade < 0.03:
                    continue
                ang = (float(hash01(int(th * 1000), 3)) - 0.5) * 0.5
                x1, y1 = tw.det
                y0 = view[1] + 2
                x0 = x1 + math.tan(ang) * (y1 - y0)
                f.segments("r", [x0], [y0], [x1], [y1 - 4], (0.7 + 0.9 * e) * fade, width=L.LW_BOLD)
                if a < 0.5 and k == len(tt) - 1:
                    txt = f"MU {e * 9.9:.1f} GEV"
                    xm, ym = x1 + 0.3 * (x0 - x1), y1 - 0.3 * (y1 - y0)
                    for x, anchor in ((xm + 14, "ls"), (xm - 14, "rs")):
                        if _free(ctx, st, tbox(x, ym, txt, L.T_MICRO, anchor, 3), pad=4.0):
                            f.tag("r", x, ym, txt, size=L.T_MICRO, pad=3, alpha=min(1.0, 2.5 * fade), anchor=anchor)
                            break

    def _aim_mark(self, f, ctx, st, p, t):
        """Red corner brackets around the head of the tower the current shower falls on."""
        aim = AIMS[p]
        if aim is None:
            return
        tw = ctx.towers[L.ORDER[aim]]
        cx, cy = tw.cx, tw.top + 0.5 * tw.det_h
        hw, hh, c = tw.w / 2 + 26, tw.det_h / 2 + 26, 22.0
        blink = 0.7 + 0.3 * math.sin(2 * math.pi * (t - T0) / BEAT)
        for sx in (-1, 1):
            for sy in (-1, 1):
                x, y = cx + sx * hw, cy + sy * hh
                f.segments("r", [x, x], [y, y], [x - sx * c, x], [y, y - sy * c], 0.95 * blink, width=L.LW_BOLD)
        for x, anchor, label in ((cx + hw + 12, "ls", f"SHOWER {p + 1:02d} >>"),
                                 (cx - hw - 12, "rs", f"<< SHOWER {p + 1:02d}")):
            if _free(ctx, st, tbox(x, cy + hh + 6, label, L.T_SMALL, anchor), pad=4.0):
                f.text("r", x, cy + hh + 6, label, size=L.T_SMALL, alpha=0.95, anchor=anchor)
                break

    def _panels(self, f, t, p, bar, a, ctx, kick, alpha):
        """Bottom band: the blocks flow into the free panels between the scopes and the towers."""
        if alpha <= 0.01:
            return
        w = self.world
        y0, y1 = ctx.slots["y0"], ctx.slots["y1"]
        place = flow(bottom_panels(ctx), PANEL_BLOCKS)
        if "count" in place:
            w.draw_counters(f, p, a, place["count"][0], place["count"][1], y0, alpha=alpha)
        if "bar" in place:
            w.draw_barcode(f, t, place["bar"][0], place["bar"][1], y0, y1, boost=0.1 * kick, alpha=alpha)
        if "shower" in place:
            x0, x1 = place["shower"]
            hud.panel_header(f, x0, x1, y0, "SHOWER", alpha=alpha)
            f.text("r" if p >= N_PHRASES else "w", x0, y0 + 74, f"{p + 1:02d}/{N_PHRASES:02d}", size=44, alpha=alpha)
            bw = (x1 - x0 - 3 * 6) / 4
            for k in range(4):                    # the four bars of the phrase
                bx = x0 + k * (bw + 6)
                f.rect("w", bx, y0 + 92, bx + bw, y0 + 114, 0.7 * alpha)
                if k == bar:
                    f.rects("r", bx + 3, y0 + 95, bx + bw - 3, y0 + 111, 0.95 * alpha)
                elif k < bar:
                    f.rects("w", bx + 3, y0 + 95, bx + bw - 3, y0 + 111, 0.55 * alpha)
