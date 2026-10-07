"""DANCE - the air-shower groove.   Sheet scene 6, 05:22 - 06:55 (the loud section, 125 BPM).

The liked air-shower loop put on the grid of the track and built around the three towers: one shower per
4-bar phrase (7.68 s, 12 phrases from the drop at 05:22.83), hard cut on every bar. The shower keeps its
clock in every phrase - first interaction on the downbeat, two bars of fall, the front on the ground on the
downbeat of bar 3, one bar and a half of what is left - but no two phrases look at it the same way (PLAN).

The four views of the first phrase are the grammar:

  PERSPECTIVE   first interaction on the downbeat (2-frame inversion of the view), slow push-in
  ORTHO_SIDE    two elevations (X and Z), one in each of the two widest bays the towers leave free,
                tracking the front: altitude rules, slant depth, red front line across the wall
  ORTHO_TOP     plan view locked on the towers: seen from above, the head of each tower IS its detector,
                the footprint blooms around it
  ORTHO_FRONT   elevation locked on the towers (1 km = the same px in x and y): the wall is the air above
                the detectors; what reached the ground stands on it as a comb, read by a scan line: every
                tooth the line passes over springs, lights up and gives its count; lone muons land on the beat

and the sequence moves on from there, towards the place where all of it lands:

  ZENITH        the same shower seen from the ground, looking straight up: a chart of the sky (zenith angle,
                compass), the burst overhead, the tracks running out to the horizon as the front comes down
  DESCENT       the camera rides down behind the front, on the axis of the shower: the cascade opens around
                downtown Cincinnati, which grows under it (its streets first, its buildings under 9 km)
  CITY_PLAN     the street plan around 9th Street x Vine Street, where the wall is: the front draws the map
                as it runs out over the ground, the muons that reach it are marked where they land (and three
                of them say on what: a building, a street, the river)
  CITY          a low perspective of the skyline, the tracks coming down through it
  SITE          the corner itself: the wall, its three towers, the muons through the place, YOU ARE HERE

The city is muonbloom/city.py (downtown baked from OpenStreetMap; its credit line is set in every view that
shows it); without it the plan falls back on the four classic views, bar for bar. In the city views the
corner stands under the detector the shower falls on (or 0.9 km from the core, when it falls between the
detectors), in the middle of a free bay. What is written about the place is what is known: the corner of
9th Street and Vine Street; the wall is drawn where the model puts it (an inference, see city.py), with its
width and its three detectors, without an address.

Showers fall in turn on DET_L / DET_C / DET_R (or between them): their muons go *through* the head of
the tower (white tick). The towers themselves only answer to the live detector streams (faces, blooms
and scopes are drawn by the show); every live hit also gets its red muon track here. Energy rises over
the 12 phrases. Nothing covers the picture at the end of a phrase: the next primary is announced on the
tower it aims at (INCOMING, half a bar before the downbeat). 05:22.0 - 05:22.83 is the pickup: the primary
alone, coming down. The break of GLITCH (06:55 - 07:02.7) holds the furniture of this scene.

Nothing has a fixed x: the mapping km <-> wall, the bays of the two elevations, the particle column
(an edge column of ctx.cols, on the side away from the shower when both are wide enough), the bottom
panels (ctx.slots) and every label are derived from the tower rectangles; text that a tower would hide
is moved or dropped (see the helpers in shower.py).

Nothing that shows data fades in or pops in. The furniture (strip, particle column, bottom panels) is
CONSTRUCTED during the pickup (05:22.0 - 05:22.83) and is there, complete, on the drop; it is not rebuilt
afterwards. What a bar brings - its view tag, its rules, its front line, its plan marks, its comb, its chart
of the sky, its map - is constructed on the cut of that bar (short builds: a bar lasts 1.92 s); what a phrase
brings - the title of the strip, the id lines, the shower number, the aim mark - on the downbeat of the
phrase. The particle column is made again when a phrase moves it to the other side. Labels on events (first
interaction, track ends, strikes, the counts of the comb) are made on the event and taken apart, never faded.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import zlib

import numpy as np

from .. import build as B
from .. import engine as E
from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import Camera, OrthoCamera, hash01
from ..show import Scene
from .shower import (K_RED, VIEW_Y0, VIEW_Y1, Stage, World, _free, _hits, altitude_rules, anchors, auto_callout,
                     bottom_panels, chord, draw_info, flow, hidden, info_layout, put_right, put_tag, put_text, tbox,
                     tower_boxes, view_bays, zone)

_city = None                            # downtown Cincinnati, baked from OpenStreetMap (muonbloom/city.py)
if importlib.util.find_spec("muonbloom.city") is not None:
    try:
        from .. import city as _city
    except Exception as _err:           # it is there and cannot be loaded: say so, and keep to the classic views
        print(f"dance: muonbloom/city.py cannot be imported ({_err}): the city views are left out", file=sys.stderr)
        _city = None

T0 = 322.826                    # downbeat of the drop (the biggest kick around 05:22)
BEAT = 0.48                     # 125 BPM
BAR = 4 * BEAT
PHRASE = 4 * BAR
N_PHRASES = 12                  # scene 6: 05:22.83 - 06:54.99
N_BUILD = 17                    # ... and 5 more showers in the world (the grid can run on after the scene)
T_END = T0 + N_PHRASES * PHRASE
T_SCENE = next((s[2] for s in sd.SECTIONS if s[4] == "dance"), T0 - 0.826)     # 05:22.0: the pickup starts
AIMS = [0, 1, None, 2, 1, 0, None, 2, 1, 0, 2, 1, None, 1, 0, 2, 1]        # detector index (L, C, R) or a bay
KINDS = ("persp", "side", "top", "front")                                  # the four classic views, bar by bar
# what each bar of each phrase looks at: the grammar first, then the sky from the ground, the city, the wall;
# the last shower goes all the way down, from its first interaction to the corner where the audience stands
PLAN = (("persp", "side", "top", "front"),          # 01  DET_L   the grammar
        ("side", "zenith", "top", "front"),         # 02  DET_C   the sky from the ground
        ("zenith", "persp", "front", "top"),        # 03  between: the comb on the landing, then the plan
        ("persp", "side", "city", "front"),         # 04  DET_R   the footprint lands on a street plan
        ("dive", "dive", "skyline", "city"),        # 05  DET_C   down with the front onto downtown
        ("zenith", "side", "top", "front"),         # 06  DET_L
        ("persp", "dive", "skyline", "front"),      # 07  between
        ("dive", "dive", "city", "site"),           # 08  DET_R   the block, the wall: YOU ARE HERE
        ("persp", "zenith", "front", "top"),        # 09  DET_C
        ("side", "zenith", "city", "front"),        # 10  DET_L
        ("zenith", "zenith", "site", "city"),       # 11  DET_R   the sky falls on the corner
        ("persp", "dive", "skyline", "site"))       # 12  DET_C   from the first interaction down to the wall
VIEW_NO = {"persp": 1, "side": 2, "top": 3, "front": 4, "zenith": 5, "dive": 6, "city": 7, "skyline": 8, "site": 9}
NAMES = {"persp": "PERSPECTIVE", "side": "ORTHO_SIDE", "top": "ORTHO_TOP", "front": "ORTHO_FRONT",
         "zenith": "ZENITH // THE SKY FROM THE GROUND", "dive": "DESCENT // WITH THE FRONT", "city": "CITY_PLAN",
         "skyline": "CITY // PERSPECTIVE", "site": "SITE // THE WALL"}
TOWN = ("dive", "city", "skyline", "site")          # the views that need the model of the city
CHASE = {(2, 1), (6, 0)}        # (phrase, bar): the perspectives that stay close on the front instead of the wide orbit
SKY_AZ = {4: -38.0, 6: 42.0, 11: 172.0}             # phrase: where the camera of the skyline stands (degrees from the south)
FOV_SKY = 84.0                  # the chart of the sky: 30 degrees of zenith angle fit the height of the view
DIVE_BACK = (2.4, 0.22)         # the camera of the descent stays this far behind the front: km + km per km of altitude
DIVE_LOW = 9.0                  # km: under this altitude the descent shows the blocks and the buildings of the city
S_CITY = 520.0                  # px per km of the street plan: the basin, from the interstates to the river
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
    """The shower world of the groove (built once per process and per tower placement, cached on disk: about
    35 s the first time)."""
    geo = Geo(ctx.towers)
    if geo.key in _WORLDS:
        return _WORLDS[geo.key], geo
    rng = np.random.default_rng(606)
    wide = sorted(sorted(ctx.bays, key=lambda b: b[0] - b[1])[:2])             # the two widest free bays
    # plan view: the row of the showers that fall between the detectors - clear of the heads of the towers and,
    # when the towers are low, high enough on the wall for their footprint to be whole
    y_free = min(0.5 * (max(t.top for t in ctx.towers.values()) + Y_GROUND), 0.5 * (VIEW_Y0 + VIEW_Y1) + 110.0)
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
    w = World(showers, rain, dets=geo.dets, seed=6, cache=f"dance_{geo.key}_{int(T0 * 1000)}_{int(y_free)}")
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


def _kicks(cues, ts, tau=0.12):
    """showdata.Cues.kick for an array of times: the decaying pulse of the kicks of the music at each of them."""
    ts = np.asarray(ts, np.float64)
    out = np.zeros(ts.shape)
    if not len(cues.kick_t):
        return out
    i = np.searchsorted(cues.kick_t, ts, side="right")
    for back in range(1, 7):
        j = np.maximum(i - back, 0)
        out += np.where(i - back >= 0, cues.kick_a[j] * np.exp(-(ts - cues.kick_t[j]) / tau), 0.0)
    return out


def _plate(f, box, x, age, t0=0.1, dur=0.25):
    """A black plate under a text that stands on a map: it opens from the side of the point (x) the text is
    tied to, while the text is made (age = seconds since that started; None = there). Nothing pops."""
    if age is None:
        g = 1.0
    else:
        g = float(B.ease(B.lin(age, t0, t0 + dur)))
        if g <= 0.0:
            return
    w = (box[2] - box[0]) * g
    if 0.5 * (box[0] + box[2]) >= x:
        f.occlude(box[0], box[1], box[0] + w, box[3])
    else:
        f.occlude(box[2] - w, box[1], box[2], box[3])


def _ground(w, f, cam, kind, view, gain, world=1.0, kick=0.0):
    """The ground of a view (World.draw_ground). With the wall rule the dot lattice of the perspective and plan
    views is made of dots that land on the brick instead of single pixels: one point per km (one in two each
    way), full level, radius 1.6 and more (the 4 km points larger); where the perspective brings the rows
    closer than a few px the dots shrink to nothing, so the far ground never fills. `gain` is what draw_ground
    gets; `world` the gain of the world alone (the level of the dots), `kick` swells them."""
    if not E.WALL or kind in ("side", "front"):
        return w.draw_ground(f, cam, kind, view, gain=gain)
    P = w.lattice
    one = (np.abs(P[:, 0]) % 1.0 < 1e-3) & (np.abs(P[:, 2]) % 1.0 < 1e-3)
    sx, sy, z, ok = cam.project(P[one])
    mj = w.lat_major[one]
    ok = ok & (sx > view[0] - 6) & (sx < view[2] + 6) & (sy > view[1] - 6) & (sy < view[3] + 6)
    px = np.full(len(sx), float(cam.scale), np.float32) if getattr(cam, "ortho", False) else cam.focal / np.maximum(z, 1e-3)
    u = np.clip((px - 7.0) / 9.0, 0.0, 1.0)                 # px per km on screen: gone under 7, whole from 16
    r = np.where(mj, 3.6, 2.4) * u * u * (3.0 - 2.0 * u) * (1.0 + 0.12 * kick)
    ok = ok & (r > 0.3)
    f.dots("w", sx[ok], sy[ok], r[ok], min(1.0, world))
    if kind == "top":
        m = mj & ok
        f.crosses("w", sx[m], sy[m], 6.0, 0.6 * gain)
    xs = np.linspace(-30, 30, 61, dtype=np.float32)
    Q = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
    lx, ly, lz, lok = cam.project(Q)
    m = lok[:-1] & lok[1:]
    f.segments("r", lx[:-1][m], ly[:-1][m], lx[1:][m], ly[1:][m], 0.5 * w.fog(lz[:-1][m]) * gain)


def _ring(cx, cy, r, n=48):
    """Vertices of a circle with a number of points that does not depend on its radius (a ring that grows)."""
    a = np.linspace(0.0, 2 * np.pi, n + 1)
    return cx + r * np.cos(a), cy + r * np.sin(a)


class Dance(Scene):
    name = "dance"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.world, self.geo = get_world(ctx)
        self._arr = {}
        self._stages = {"L": Stage(ctx, "L"), "R": Stage(ctx, "R"), "N": Stage(ctx, None)}
        self._elev = {}
        self._lab = {}
        self.town = None
        if _city is not None:
            try:
                self.town = _city.get()
            except Exception as err:                # no baked data: the plan keeps to the classic views
                print(f"dance: the city cannot be loaded ({err}): the city views are left out", file=sys.stderr)

    # ------------------------------------------------------------------ the plan of the sequence
    def kind(self, p, bar):
        """What bar `bar` of phrase p looks at. Without the model of the city, its views give their bar back to
        the classic view of that bar."""
        k = PLAN[p][bar] if p < len(PLAN) else KINDS[bar]
        return KINDS[bar] if (k in TOWN and self.town is None) else k

    def run(self, p, bar):
        """(first bar, number of bars) of the bars of phrase p that share the view of `bar`: one shot."""
        k = self.kind(p, bar)
        a = b = bar
        while a > 0 and self.kind(p, a - 1) == k:
            a -= 1
        while b < 3 and self.kind(p, b + 1) == k:
            b += 1
        return a, b - a + 1

    def site(self, p):
        """World (x, z) of the place the ground views of phrase p are about - where one stands to look up, where
        the wall is in the city: under the detector the shower falls on; 0.9 km from the core when it falls
        between the detectors."""
        aim = AIMS[p]
        if aim is not None:
            return self.geo.dets[aim][0], self.geo.dets[aim][2]
        e = self.world.events[p]
        ang = 0.6 + 2.4 * p
        return float(e["G"][0]) + 0.9 * math.cos(ang), float(e["G"][2]) + 0.9 * math.sin(ang)

    def _xf(self, p):
        """Where the model of the city stands in the world for phrase p: the corner on the site, the street
        grid squared with the axes (Vine Street up the wall in the plans, the wall facing +z)."""
        sx, sz = self.site(p)
        return self.town.xf(offset=(sx, 0.0, sz), yaw=self.town.GRID_YAW)

    def _model_xz(self, p, x, z):
        """Model coordinates (x, z) of the city for a point of the ground of the world, in phrase p."""
        key = ("inv", p)
        if key not in self._lab:
            xf = self._xf(p)
            self._lab[key] = (np.linalg.inv(np.asarray(xf.M, np.float64)), np.asarray(xf.offset, np.float64),
                              np.asarray(xf.pivot, np.float64))
        inv, off, piv = self._lab[key]
        m = inv @ (np.array([x, 0.0, z]) - off) + piv
        return float(m[0]), float(m[2])

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

    def _centre(self, st, y, p):
        """Screen x of the place in a view of the city (the corner, the wall): the middle of a free bay - never
        over a tower, whose live bloom would sit on it - and of the bay away from the particle column."""
        bays = self.elev(st)[0]
        if st.side == "L":
            b = bays[-1]
        elif st.side == "R":
            b = bays[0]
        else:
            return self._stage_x(p, st)
        return 0.5 * (b[0] + b[1])

    def zones(self, ctx, st, p, kind):
        """x-intervals taken by the cascade(s) of the current bar: (at the top of the view, over its height)."""
        geo = self.geo
        if kind in ("persp", "zenith"):
            x = self._stage_x(p, st)
            return [(x - 150.0, x + 150.0)], [zone(ctx, x, 330.0)]
        if kind == "side":
            bays, _, s_side = self.elev(st)
            xs = [0.5 * (b[0] + b[1]) for b in bays]
            return [(x - 150.0, x + 150.0) for x in xs], [zone(ctx, x, 3.4 * s_side, b) for x, b in zip(xs, bays)]
        if kind in TOWN:                          # a map fills the view: its texts stand on plates of their own
            return [], []
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
        kind = self.kind(p, bar)
        r0, rn = self.run(p, bar)
        ur = (bar - r0 + u) / rn                              # progress in the shot
        front = float(np.clip(w.front(p, t - e["t0"]), 0.0, e["h1"]))
        if kind == "persp":
            chase = (p, bar) in CHASE
            tx_, tz_ = gx, gz
            if chase:                                         # close on the front, coming down with it
                yaw = math.radians(30.0 + 47.0 * p + 22.0 * u)
                D = 12.5 - 2.5 * u
                ty = float(np.clip(front + 1.2, 2.4, e["h1"]))
                cy_ = ty + 3.0
                k_ax = ty / max(float(-e["d0"][1]), 0.5)      # (the point of the axis at that altitude: the
                tx_, tz_ = gx - float(e["d0"][0]) * k_ax, gz - float(e["d0"][2]) * k_ax      # shower is inclined)
            else:
                yaw = math.radians(-24.0 + 47.0 * p + 16.0 * u)
                D = 27.0 - 5.0 * u
                ty = 9.4 - 2.0 * u
                cy_ = ty - 0.4
            cam = Camera((tx_ + D * math.sin(yaw), cy_, tz_ + D * math.cos(yaw)), (tx_, ty, tz_), fov_deg=44.0,
                         screen_center=(self._stage_x(p, st), st.cy))
            cam.fov = 44.0
            cam.note = "F 44.0 DEG   CAM WITH THE FRONT" if chase else None
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
        if kind == "front":
            cam = OrthoCamera((0.0, 0.0, 60.0), (0.0, 0.0, 0.0), scale=geo.S, screen_center=(geo.x_mid, Y_GROUND))
            return kind, [(cam, view)]
        sx, sz = self.site(p)
        aim = AIMS[p]
        if kind == "zenith":                                  # on the ground, looking straight up: north up, east LEFT
            cam = Camera((sx, 0.0, sz), (sx, 1.0, sz), fov_deg=FOV_SKY, up=(0.0, 0.0, -1.0),
                         screen_center=(self._stage_x(p, st), st.cy))
            cam.fov = FOV_SKY
            cam.note = f"LOOKING UP FROM {L.NAMES[L.ORDER[aim]]}" if aim is not None else "LOOKING UP   CORE 0.9 KM AWAY"
            return kind, [(cam, view)]
        if kind == "dive":                 # behind the front, on the line of the shower through the place, coming
            d0 = e["d0"].astype(np.float64)                   # down with it: the cascade opens around the city
            dist = front / max(-d0[1], 0.5) + DIVE_BACK[0] + DIVE_BACK[1] * front
            pos = np.array([sx, 0.0, sz]) - d0 * dist
            yc = st.view[1] + 0.45 * (st.view[3] - st.view[1])
            cam = Camera(tuple(pos), (sx, 0.0, sz), fov_deg=50.0, up=(0.0, 0.0, -1.0),
                         screen_center=(self._centre(st, yc, p), yc))
            cam.fov, cam.alt = 50.0, float(pos[1])
            cam.note = f"ALT {pos[1]:05.2f} KM   LOOKING DOWN THE AXIS"
            return kind, [(cam, view)]
        if kind == "city":                                    # the street plan, the corner two fifths down the view
            yc = st.view[1] + 0.34 * (st.view[3] - st.view[1])
            cam = self.town.plan_camera((self._centre(st, yc, p), yc), S_CITY, at=(sx, sz))
            return kind, [(cam, view)]
        if kind == "skyline":                                 # a low orbit around the corner
            az = math.radians(SKY_AZ.get(p, -38.0) + 9.0 * (u - 0.5))
            R = 1.62 - 0.12 * u
            yc = st.view[1] + 0.5 * (st.view[3] - st.view[1])
            cam = Camera((sx + 0.1 + R * math.sin(az), 0.36, sz + 0.2 + R * math.cos(az)), (sx + 0.1, 0.05, sz + 0.2),
                         fov_deg=38.0, screen_center=(0.5 * L.W, yc))
            cam.fov = 38.0
            cam.note = "F 38.0 DEG   CAM ORBIT   CINCINNATI // OHIO"
            return kind, [(cam, view)]
        # the site: in front of the wall, a little above the heads, moving sideways along 9th Street
        wl = self.town.wall
        xf = self._xf(p)
        if wl is None:
            mid, nrm, along, hh = np.array([sx, 0.0, sz]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 0.013
        else:
            m3 = xf(np.array([[wl["mid"][0], 0.0, wl["mid"][1]], [wl["mid"][0] + wl["normal"][0], 0.0,
                                                                   wl["mid"][1] + wl["normal"][1]],
                              [wl["b"][0], 0.0, wl["b"][1]]], np.float32)).astype(np.float64)
            mid, nrm, hh = m3[0], m3[1] - m3[0], float(wl["height"])
            along = (m3[2] - m3[0]) / max(float(np.linalg.norm(m3[2] - m3[0])), 1e-9)
        yc = st.view[1] + 0.56 * (st.view[3] - st.view[1])
        pos = mid + nrm * (0.058 - 0.008 * u) + along * (0.036 * (u - 0.5)) + np.array([0.0, 0.0105, 0.0])
        cam = Camera(tuple(pos), (mid[0], 0.5 * hh + 0.001, mid[2]), fov_deg=44.0,
                     screen_center=(self._centre(st, yc, p), yc))
        cam.fov = 44.0
        cam.near = 0.002
        cam.note = "F 44.0 DEG   FROM ACROSS 9TH STREET"
        return kind, [(cam, view)]

    # ------------------------------------------------------------------ draw
    def _bar(self, ctx, p, bar, u, t):
        """(stage, kind of view, [(camera, clip)], info layout) of bar `bar` of phrase p."""
        e = self.world.events[p]
        st = self.stage(p)
        kind, cams = self.cameras(p, bar, u, t, st)
        aim = AIMS[p]
        on = L.NAMES[L.ORDER[aim]] if aim is not None else None
        z_top, z_full = self.zones(ctx, st, p, kind)
        name = NAMES[kind] + (" X / Z" if kind == "side" and len(cams) > 1 else "")
        n_ph = max(N_PHRASES, p + 1)
        lay = info_layout(ctx, st, f"VIEW {VIEW_NO[kind]:02d} // {name}", cams[0][0],
                          [(f"AIR_SHOWER // MUON BLOOM // SHOWER {p + 1:02d}/{n_ph:02d}", 0.85),
                           (f"E0 {e['E0'] * 3.2:.2f}E15 EV   ZENITH {e['zen']:04.1f} DEG", 0.6),
                           (f"FALLS ON {on}" if on else "FALLS BETWEEN THE DETECTORS", 0.6)],
                          [(f"AIR_SHOWER {p + 1:02d}/{n_ph:02d}", 0.85), (f"E0 {e['E0'] * 3.2:.2f}E15 EV", 0.6),
                           (f"ZENITH {e['zen']:04.1f} DEG", 0.6), (f"ON {on}" if on else "BETWEEN DETECTORS", 0.6)],
                          z_top, z_full, bars=((1.0, "1 KM"), (0.5, "500 M"), (0.2, "200 M")) if kind == "city" else None)
        return st, kind, cams, lay

    def _col_age(self, p, t):
        """Seconds since the particle column appeared where it is: at the start of the scene, or on the
        downbeat of the phrase that moved it to the other side of the wall."""
        col = self.stage(p).col
        q = p
        while q > 0 and self.stage(q - 1).col == col:
            q -= 1
        return t - (T0 + q * PHRASE) if q > 0 else t - T_SCENE - 0.1

    def draw(self, f, t, ctx, gain=1.0, hud_alpha=1.0, flash=True, bursts=True, hide=(), extended=False, hud_out=None):
        """`gain` scales the world, `hud_alpha` dims the HUD (a steady level: nothing fades in or out through
        it); `hide` = shower indices not drawn and `extended` = keep the grid running after phrase 12; `hud_out`
        = show time from which the instrument is taken apart, piece by piece. (`bursts` is accepted and does
        nothing: no barcode covers the picture any more.)"""
        w = self.world
        if not extended:
            t = min(t, T_END - 1e-3)
        p, bar, u, rp = grid(t)
        e = w.events[p]
        a = t - e["t0"]
        age, alive, env = w.state(t)
        for k in hide:
            alive[k] = False
        st, kind, cams, lay = self._bar(ctx, p, bar, u, t)
        view = st.view
        kick = min(1.5, ctx.cues.kick(t))
        level = min(1.0, p / (N_PHRASES - 1))
        pick = t < T0                                     # the pickup before the drop: almost nothing yet
        if pick:
            gain, hud_alpha = gain * 0.45, hud_alpha * 0.4

        # -- when things are made (seconds since ...); nothing here fades in ---------------------------------
        def leave(age_, k=0):                             # taken apart in turn, 0.5 s apart
            return age_ if hud_out is None else B.io(age_, hud_out + 0.5 * k + 0.45 - t, out=0.45)

        age0 = t - T_SCENE                                # ... the scene started: the furniture
        first = pick or (p == 0 and bar == 0)
        r0 = self.run(p, bar)[0]                          # ... the cut of this shot: what the view brings
        age_bar = age0 - 0.05 if first else rp - r0 * BAR
        age_id = age0 - 0.15 if p == 0 else rp            # ... the shower changed: the id lines
        if bar > 0 and self._bar(ctx, p, bar - 1, 1.0, t)[3]["b"] != lay["b"]:
            age_id = age_bar                              # (this view had to move them: made again where they are)
        age_bar, age_id = leave(age_bar), leave(age_id)
        age_s = leave(age0, 2)
        # level of what is written on the picture (rules, front line, marks, comb): it follows the world, except
        # when the world is being faded out - data do not fade, they are taken apart (hud_out)
        ov = gain if hud_out is None else (0.45 if pick else 1.0)

        if kind == "front" and lay["a"] is not None:          # room for the legend of the comb, under the view tag
            for k, txt in enumerate(LEGEND):
                box = tbox(lay["a"], lay["y0"] + 92, txt.format(n=0), L.T_SMALL)
                if _free(ctx, st, box, lay["zones_top"], lay["boxes"][1:]):
                    lay["legend"] = k
                    lay["boxes"].append(box)
                    break
        f.set_clip(*view)
        avoid = tower_boxes(ctx) + lay["boxes"]
        if kind in KINDS:
            for j, (cam, clip) in enumerate(cams):
                if j == 0:
                    f.set_clip(*view)
                    _ground(w, f, cam, kind, view, gain * (1.0 + 0.45 * kick) * (2.6 if kind == "persp" else 1.0), gain, kick)
                    if kind in ("side", "front"):
                        altitude_rules(f, ctx, st, cam, float(e["G"][0]), float(e["G"][2]), lay, gain=ov,
                                       front=w.front(p, a) if kind == "side" else 0.0, age=age_bar, wave=0.22)
                f.set_clip(*clip)
                g = gain * (1.0 + 0.16 * kick) * (1.7 if kind == "front" else 1.0)
                w.draw_cascades(f, cam, age, alive, env, gain=g)
                w.draw_hits(f, cam, age, alive, gain=gain)
                w.draw_splash(f, cam, age, gain=gain)
                if kind == "top":                         # what landed stays on the plan, a point each
                    self._marks(f, cam, p, a, gain, r_km=0.0)
                self._overlay(f, ctx, st, lay, cam, kind, p, a, clip, j, ov, age_bar)
                boxes = w.draw_interaction(f, cam, age, clip, ctx, tags=(j == 0), avoid=lay["boxes"])
                if kind == "front":
                    self._lateral(f, ctx, st, lay, cam, p, a, u, ov, age_bar)
                if kind in ("top", "front"):
                    self._crossings(f, cam, age, alive)
                w.draw_labels(f, cam, age, alive, clip, avoid=avoid + boxes,
                              limit=(6 + int(3 * level)) // len(cams) + 1)
        else:
            args = (f, ctx, t, st, lay, cams[0][0], p, a, u, age, alive, env, gain, ov, kick, age_bar, avoid)
            {"zenith": self._zenith, "dive": self._dive, "city": self._city_plan, "skyline": self._skyline,
             "site": self._site}[kind](*args)
        f.set_clip(*view)
        if kind != "top":
            self._strikes(f, t, ctx, st)
        if not pick and gain > 0.3:
            self._aim_mark(f, ctx, st, p, t, leave(rp))
            self._incoming(f, ctx, st, p, t, rp)
        f.set_clip()
        w.draw_column(f, p, a, st.col, alpha=hud_alpha, age=leave(self._col_age(p, t), 1))
        w.draw_strip(f, p, a, label=f"LONGITUDINAL_PROFILE // SHOWER {p + 1:02d}/{max(N_PHRASES, p + 1):02d} // 125.0 BPM",
                     alpha=hud_alpha, pulse=kick, age=age_s, title_age=min(age0 - 0.2 if p == 0 else rp, age_s))
        self._panels(f, t, p, bar, a, ctx, kick, hud_alpha,
                     (leave(age0 - 0.2, 3), leave(age0 - 0.3, 4), leave(age0 - 0.4, 5)), rp)
        if hud_alpha > 0.05:
            if kind in TOWN:                              # a map runs under them: the texts of the view get a plate
                self._plates(f, lay, age_bar, age_id)
            draw_info(f, ctx, lay, alpha=hud_alpha, age=age_bar, age_id=age_id)
        inv = flash and t >= T0 and 0.0 <= rp < 0.05
        return {"invert": bool(inv), "invert_rect": view}

    @staticmethod
    def _plates(f, lay, age_bar, age_id):
        """Black plates under the fixed texts of a view drawn over a map (they open with the texts)."""
        for k, bx in enumerate(lay["boxes"]):
            ag = age_id if (lay["b"] and k == (1 if lay["a"] is not None else 0)) else age_bar
            if ag is None or ag >= 0.0:
                f.occlude(bx[0] - 6.0, bx[1] - 4.0, bx[2] + 6.0, bx[3] + 4.0)

    def _overlay(self, f, ctx, st, lay, cam, kind, p, a, clip, j, gain, age=None):
        """What a view writes on its picture: front line and its tag, names of the elevations, marks of the
        plan, ground tag. age = seconds since the bar started (None = built): it is constructed on the cut -
        lines drawn by a pen, rings traced, tags and labels made - and never fades in."""
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
                with f.build(age, (view[0], y - 34.0, view[2], y + 8.0), flow="lr", wave=0.2, line=0.25, marks=False,
                             key=54):
                    f.segments("r", [view[0]], [y], [view[2]], [y], 0.9 * gain, width=L.LW)
                    put_right(f, ctx, st, lay, "r", y - 9, f"FRONT {front:06.3f} KM", size=L.T_LABEL, pad=5, alpha=al)
                f.set_clip(*clip)
            ax, ay, _, _ = cam.project(np.stack([e["G"], e["P1"]]).astype(np.float32))
            with f.build(age, clip, flow="bt", wave=0.15, marks=False, key=55 + j):
                f.segments("r", [ax[0]], [ay[0]], [ax[1]], [ay[1] - 60], E.wl(0.5 * gain), width=E.ww(1.0))
                if self.elev(st)[1] is not None:
                    put_text(f, ctx, "w", float(ax[0]) + 14, min(float(ay[0]) - 12, view[3] - 30),
                             "ELEVATION X" if j == 0 else "ELEVATION Z", size=L.T_SMALL, alpha=0.7 * al)
        elif kind == "top":
            gx, gy, _, _ = cam.project(e["G"][None].astype(np.float32))
            X, Y = float(gx[0]), float(gy[0])
            with f.build(age, view, flow="out", origin=(X, Y), wave=0.25, marks=False, key=57):
                f.segments("r", [X, X, X, X], [Y, Y, Y, Y], [view[0], view[2], X, X], [Y, Y, view[1], view[3]],
                           E.wl(0.45 * gain), width=E.ww(1.0))
                sc = cam.scale
                f.rings("w", [X] * 4, [Y] * 4, [sc * r for r in (1, 2, 4, 8)], E.wl(0.25 * gain), width=E.ww(1.0))
                for r in (1, 2, 4, 8):
                    if view[0] + 30 < X + sc * r + 7 < st.tx1 - 60:
                        put_text(f, ctx, "w", X + sc * r + 7, Y - 8, f"{r} KM", size=L.T_SMALL, alpha=0.6 * al)
                aim = AIMS[p]
                xr = X + 64 if aim is None else max(X + 64, ctx.towers[L.ORDER[aim]].x1 + 22)
                xl = X - 64 if aim is None else min(X - 64, ctx.towers[L.ORDER[aim]].x0 - 22)
                txt = f"CORE {float(e['G'][0]):+07.3f} {float(e['G'][2]):+07.3f}"
                for yt in (Y + 150, Y - 150):                 # under the core; above it when the core is low on the wall
                    done = False
                    for x, anchor in ((xr, "ls"), (xl, "rs")):
                        if (yt < view[3] - 16 and _free(ctx, st, tbox(x, yt, txt, L.T_SMALL, anchor, 4), (), lay["boxes"],
                                                         pad=4.0)):
                            f.tag("r", x, yt, txt, size=L.T_SMALL, pad=4, alpha=al, anchor=anchor)
                            done = True
                            break
                    if done:
                        break
        elif kind == "front":
            ax, ay, _, _ = cam.project(np.stack([e["G"], e["P1"]]).astype(np.float32))
            with f.build(age, view, flow="bt", wave=0.15, marks=False, key=58):
                f.segments("r", [ax[0]], [ay[0]], [ax[1]], [ay[1]], E.wl(0.4 * gain), width=E.ww(1.0))
                put_right(f, ctx, st, lay, "w", Y_GROUND - 9, "GROUND 00 KM", size=L.T_SMALL, pad=4, alpha=al)

    def _arrivals(self, k):
        """Everything of shower k that reached the ground: (x in km, age at arrival, is a muon, z in km)."""
        if k not in self._arr:
            w = self.world
            s0, s1 = w.ranges[k]
            end = w.SA[s0:s1] + 1
            m = w.V[end, 1] < 1e-3
            self._arr[k] = (w.V[end[m], 0].astype(np.float64), w.VT[end[m]].astype(np.float64),
                            K_RED[w.SK[s0:s1][m]] == 1, w.V[end[m], 2].astype(np.float64))
        return self._arr[k]

    def _lateral(self, f, ctx, st, lay, cam, p, a, u, gain, age=None):
        """The comb: what reached the ground, standing on the ground line (count per 125 m, muons in red), read
        by a scan line that crosses the wall in one bar. A tooth answers when the line passes over it: it
        springs up and settles, it lights (before the line it is only dim), the strongest ones give their count
        on a tag that is made and taken apart, and the line carries the total of what it has read. When the
        shower fell on a tower, the comb opens around it (the two halves start at the edges of the tower) so its
        peak is not hidden. age = seconds since the bar started: the comb rises in a wave that leaves the core
        (each tooth overshoots, then settles), the scan line is drawn and its read-out is made - nothing fades in."""
        view = st.view
        gx, gt, mu, _ = self._arrivals(p)
        S = cam.scale
        bw = 0.125
        core = float(self.world.events[p]["G"][0])
        aim = AIMS[p]
        shift = 0.0 if aim is None else (0.5 * ctx.towers[L.ORDER[aim]].w + 14.0)
        lo = math.floor(((view[0] - cam.cx) / S - core) / bw) * bw
        edges = np.arange(lo, (view[2] - cam.cx) / S - core + bw, bw)            # distance to the core (km)
        wt = np.clip((a - gt) / 0.15, 0.0, 1.0)               # an arrival joins its tooth in 0.15 s: no jump
        c_all, _ = np.histogram(gx - core, edges, weights=wt)
        c_mu, _ = np.histogram(gx[mu] - core, edges, weights=wt[mu])
        f_all, _ = np.histogram(gx - core, edges)             # ... and when everything has landed: the scale
        f_mu, _ = np.histogram(gx[mu] - core, edges)
        norm = max(int(f_all.max()), 1)
        mid = edges[:-1] + bw / 2
        far = np.abs(mid) / max(float(np.abs(mid).max()), 1e-6)
        grow = 1.0 if age is None else B.spring((age - 0.3 * far) / 0.3)
        xs = cam.cx + (core + edges[:-1]) * S + np.sign(mid) * shift
        w_ = bw * S
        # the scan line, and how long ago it passed each tooth (< 0: still to come)
        xc = view[0] + u * (view[2] - view[0])
        ap = (xc - (xs + 0.5 * w_)) / ((view[2] - view[0]) / BAR)
        read = ap >= 0.0
        apc = np.clip(ap, 0.0, 2.0)
        hop = np.where(read, np.sin(apc * (2 * np.pi / 0.26)) * np.exp(-apc / 0.13), 0.0)
        hot = np.where(read, np.exp(-apc / 0.22), 0.0)
        h = (330.0 * np.sqrt(c_all / norm) * (1.0 + 0.36 * hop) + 15.0 * hop * (c_all > 0.5)) * grow
        hm = (330.0 * np.sqrt(c_mu / norm) * (1.0 + 0.36 * hop) + 15.0 * hop * (c_mu > 0.5)) * grow
        k = (c_all > 0) & (h > 0.5)
        if E.WALL:          # no dim teeth: not read yet = its outline, read = full (the scan line fills it as it passes)
            g1 = min(1.0, gain)
            kr = k & read
            f.rects("w", xs[kr] + 2, Y_GROUND - h[kr], xs[kr] + w_ - 3, Y_GROUND - hm[kr] - 1, ((1.0 + 0.5 * hot) * g1)[kr])
            ko = k & ~read
            xa, xb, yt, yb = xs[ko] + 2.8, xs[ko] + w_ - 3.8, Y_GROUND - h[ko] + 0.8, Y_GROUND - hm[ko] - 1
            f.segments("w", np.r_[xa, xa, xb], np.r_[yb, yt, yt], np.r_[xa, xb, xb], np.r_[yt, yt, yb], g1, width=E.WALL_LINE)
        else:
            f.rects("w", xs[k] + 2, Y_GROUND - h[k], xs[k] + w_ - 3, Y_GROUND - hm[k] - 1,
                    ((0.13 + 0.49 * read + 0.9 * hot) * gain)[k])
        k = (c_mu > 0) & (hm > 0.5)
        f.rects("r", xs[k] + 2, Y_GROUND - hm[k], xs[k] + w_ - 3, Y_GROUND - 1, ((0.2 + 0.75 * read + 1.3 * hot) * gain)[k])
        k = read & (hot > 0.05) & (h > 0.5)                   # the teeth under the line: a white cap
        f.rects("w", xs[k] + 2, Y_GROUND - h[k] - 4, xs[k] + w_ - 3, Y_GROUND - h[k] - 1, (1.6 * hot * gain)[k])
        # the strongest teeth give their count as the line passes: a tag made on the passage, taken apart after
        for i in self._teeth(p, f_all, f_mu):
            if 0.0 <= ap[i] < 0.75:
                txt = f"{int(round(c_all[i])):03d}" if f_mu[i] == 0 else f"MU {int(round(c_mu[i])):02d}"
                x, y = float(xs[i] + 0.5 * w_), float(Y_GROUND - 330.0 * math.sqrt(f_all[i] / norm) - 30.0)
                if _free(ctx, st, tbox(x, y, txt, L.T_MICRO, "ms", 3), pad=4.0):
                    B.tag(f, "r" if f_mu[i] else "w", x, y, txt, B.io(float(ap[i]), 0.75 - float(ap[i]), out=0.15, span=0.2),
                          size=L.T_MICRO, pad=3, anchor="ms", cps=150.0, key=int(i) + 7 * p)
        i = int(np.argmin(np.abs(xs + w_ / 2 - xc)))
        with f.build(age, (xc - 6.0, Y_GROUND - 376.0, xc + 6.0, Y_GROUND + 2.0), flow="tb", wave=0.05, line=0.2,
                     marks=False, key=59):
            f.segments("r", [xc], [Y_GROUND - 372], [xc], [Y_GROUND], 0.9 * gain, width=L.LW)
        anchor = "ls" if xc < view[2] - 640 else "rs"
        txt = (f"R {mid[i]:+06.2f} KM  N {int(round(c_all[i])):03d}  MU {int(round(c_mu[i])):02d}  //  READ "
               f"{int(round(c_all[read].sum())):04d}  MU {int(round(c_mu[read].sum())):03d}")
        tx = xc + (8 if anchor == "ls" else -8)
        with f.build(age, tbox(tx, Y_GROUND - 356, txt, L.T_SMALL, anchor, 4), wave=0.05, marks=False, key=60):
            put_tag(f, ctx, "r", tx, Y_GROUND - 356, txt, size=L.T_SMALL, pad=4, anchor=anchor, alpha=min(1.0, gain))
        if "legend" in lay:                   # its legend, under the view tag
            txt = LEGEND[lay["legend"]].format(n=int(round(wt.sum())))
            with f.build(age, tbox(lay["a"], lay["y0"] + 92, txt, L.T_SMALL), wave=0.1, marks=False, key=61):
                f.text("w", lay["a"], lay["y0"] + 92, txt, size=L.T_SMALL, alpha=0.75 * min(1.0, gain))

    def _teeth(self, p, f_all, f_mu):
        """The teeth of the comb of phrase p that give their count when the scan line passes: the strongest,
        at least nine teeth apart (their tags must not touch)."""
        key = ("teeth", p, len(f_all))
        if key not in self._arr:
            score = f_mu * 1000.0 + f_all
            picked = []
            for i in np.argsort(-score):
                if score[i] <= 0 or len(picked) >= 9:
                    break
                if all(abs(int(i) - j) >= 9 for j in picked):
                    picked.append(int(i))
            self._arr[key] = picked
        return self._arr[key]

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
            rx, ry = _ring(x, y, 10.0 + 54.0 * (1 - (1 - u) ** 3))
            f.polyline("w", rx, ry, 0.9 * (1 - u) ** 1.5, width=L.LW)
            f.dots("w", [x], [y], 3.6 * (1 - u) + 1.0, 1.6 * (1 - u))

    def _strikes(self, f, t, ctx, st):
        """Every live hit of a detector is a muon: its track, straight down into the head of the tower (the
        latest one carries its energy, on the side of the track where there is room: a tag made on the hit
        and taken apart half a second later)."""
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
                            B.tag(f, "r", x, ym, txt, B.io(a, 0.5 - a, out=0.12, span=0.14), size=L.T_MICRO, pad=3,
                                  anchor=anchor, cps=120.0, key=int(th * 1000) % 9973)
                            break

    def _aim_mark(self, f, ctx, st, p, t, age=None):
        """Red corner brackets around the head of the tower the current shower falls on. age = seconds since
        the downbeat of the phrase: the brackets are thrown out from the head and its label is decoded."""
        aim = AIMS[p]
        if aim is None:
            return
        tw = ctx.towers[L.ORDER[aim]]
        cx, cy = tw.cx, tw.top + 0.5 * tw.det_h
        hw, hh, c = tw.w / 2 + 26, tw.det_h / 2 + 26, 22.0
        blink = 0.7 + 0.3 * math.sin(2 * math.pi * (t - T0) / BEAT)
        with f.build(age, (cx - hw - 220.0, cy - hh - 8.0, cx + hw + 220.0, cy + hh + 30.0), flow="out", origin=(cx, cy),
                     wave=0.15, marks=False, key=62):
            for sx in (-1, 1):
                for sy in (-1, 1):
                    x, y = cx + sx * hw, cy + sy * hh
                    f.segments("r", [x, x], [y, y], [x - sx * c, x], [y, y - sy * c], 0.95 * blink, width=L.LW_BOLD)
            for x, anchor, label in ((cx + hw + 12, "ls", f"SHOWER {p + 1:02d} >>"),
                                     (cx - hw - 12, "rs", f"<< SHOWER {p + 1:02d}")):
                if _free(ctx, st, tbox(x, cy + hh + 6, label, L.T_SMALL, anchor), pad=4.0):
                    f.text("r", x, cy + hh + 6, label, size=L.T_SMALL, alpha=0.95, anchor=anchor)
                    break

    def _incoming(self, f, ctx, st, p, t, rp):
        """What closes a phrase: the next primary is on its way. Half a bar before the downbeat its tag is
        made beside the head of the tower it aims at, counts down, and is taken apart on the downbeat."""
        left = PHRASE - rp
        if left > 2 * BEAT or p + 1 >= N_PHRASES or AIMS[p + 1] is None:
            return
        tw = ctx.towers[L.ORDER[AIMS[p + 1]]]
        y = tw.top + 0.5 * tw.det_h + 6.0
        txt = f"INCOMING // SHOWER {p + 2:02d} // T-{left:04.2f} S"
        for x, anchor in ((tw.x1 + 18.0, "ls"), (tw.x0 - 18.0, "rs")):
            if _free(ctx, st, tbox(x, y, txt, L.T_SMALL, anchor, 4), pad=4.0):
                B.tag(f, "r", x, y, txt, B.io(2 * BEAT - left, left, out=0.12, span=0.3), size=L.T_SMALL, pad=4,
                      anchor=anchor, cps=110.0, key=70 + p)
                break

    def _panels(self, f, t, p, bar, a, ctx, kick, alpha, ages=(None, None, None), rp=9.0):
        """Bottom band: the blocks flow into the free panels between the scopes and the towers.
        ages = seconds since (the counters, the barcode, the shower panel) appeared: each is constructed, none
        fades in; rp = seconds since the downbeat of the phrase (the shower number spins, then locks)."""
        if alpha <= 0.01:
            return
        w = self.world
        y0, y1 = ctx.slots["y0"], ctx.slots["y1"]
        place = flow(bottom_panels(ctx), PANEL_BLOCKS)
        if "count" in place:
            w.draw_counters(f, p, a, place["count"][0], place["count"][1], y0, alpha=alpha, age=ages[0])
        if "bar" in place:                    # (the kicks are written in the barcode as they go by)
            w.draw_barcode(f, t, place["bar"][0], place["bar"][1], y0, y1, alpha=alpha, age=ages[1],
                           pulse=lambda ts: 0.1 * np.minimum(1.5, _kicks(ctx.cues, ts)))
        if "shower" in place:
            x0, x1 = place["shower"]
            with f.build(ages[2], (x0 - 8, y0 - 24, x1 + 8, y1 + 8), wave=0.25, key=63):
                hud.panel_header(f, x0, x1, y0, "SHOWER", alpha=alpha)
                num = f"{p + 1:02d}"                   # the number of the shower spins on the downbeat, then locks
                f.text("r" if p >= N_PHRASES else "w", x0, y0 + 74,
                       (B.roll(num, rp, 0.35, key=p) if p > 0 else num) + f"/{N_PHRASES:02d}", size=44, alpha=alpha)
                bw = (x1 - x0 - 3 * 6) / 4
                for k in range(4):                    # the four bars of the phrase
                    bx = x0 + k * (bw + 6)
                    f.rect("w", bx, y0 + 92, bx + bw, y0 + 114, E.wl(0.7 * alpha), width=E.ww(1.0))
                    if k == bar:
                        f.rects("r", bx + 3, y0 + 95, bx + bw - 3, y0 + 111, 0.95 * alpha)
                    elif k < bar:
                        f.rects("w", bx + 3, y0 + 95, bx + bw - 3, y0 + 111, E.wl(0.55 * alpha))

    def _readout(self, f, ctx, st, lay, txt, age, key, taken=None, alpha=1.0):
        """A red read-out of the view (a tag): under the block of the view tag when there is room, else at the
        foot of the view, at the first anchor that is clear of the towers. Returns its box (or None)."""
        view = st.view
        taken = lay["boxes"] if taken is None else taken
        cands = []
        if lay["a"] is not None:
            cands.append((lay["a"], lay["y0"] + 86.0))
        cands += [(x, view[3] - 34.0) for x in anchors(ctx, st, "l")]
        for x, y in cands:
            box = tbox(x, y, txt, L.T_LABEL, pad=5)
            if _free(ctx, st, box, (), taken, pad=6.0):
                with f.build(age, box, wave=0.08, marks=False, key=key):
                    f.tag("r", x, y, txt, size=L.T_LABEL, pad=5, alpha=alpha)
                return box
        return None

    # ------------------------------------------------------------------ the sky from the ground
    def _zenith(self, f, ctx, t, st, lay, cam, p, a, u, age, alive, env, gain, ov, kick, age_bar, avoid):
        """Looking straight up from the ground: a chart of the sky, the first interaction overhead, the tracks
        of the cascade running out towards the horizon as the front comes down."""
        w = self.world
        view = st.view
        e = w.events[p]
        cx, cy, fo = cam.cx, cam.cy, cam.focal
        al = min(1.0, ov)
        # the chart: circles of equal zenith angle, a compass. North is up; looking up, east is on the LEFT.
        angs = (10, 20, 30, 45, 60)
        rr = [fo * math.tan(math.radians(v)) for v in angs]
        with f.build(age_bar, view, flow="out", origin=(cx, cy), wave=0.25, line=0.2, marks=False, key=64):
            f.rings("w", [cx] * len(rr), [cy] * len(rr), rr, E.wl(0.24 * ov), width=E.ww(1.0))
            az = np.radians(np.arange(0.0, 360.0, 30.0))
            card = (np.arange(12) % 3) == 0
            r1 = 1.2 * (view[2] - view[0])
            if E.WALL:                                        # the cardinal directions only, as full lines
                f.segments("w", cx + rr[0] * np.sin(az[card]), cy - rr[0] * np.cos(az[card]), cx + r1 * np.sin(az[card]),
                           cy - r1 * np.cos(az[card]), E.wl(0.2 * ov), width=E.WALL_LINE)
            else:
                f.segments("w", cx + rr[0] * np.sin(az), cy - rr[0] * np.cos(az), cx + r1 * np.sin(az),
                           cy - r1 * np.cos(az), np.where(card, 0.2, 0.09) * ov)
            tk = np.radians(np.arange(0.0, 360.0, 5.0))       # degrees of azimuth on the 30 degree circle
            ln = np.where(np.arange(72) % 6 == 0, 14.0, 6.0)
            f.segments("w", cx + rr[2] * np.sin(tk), cy - rr[2] * np.cos(tk), cx + (rr[2] + ln) * np.sin(tk),
                       cy - (rr[2] + ln) * np.cos(tk), E.wl(0.5 * ov), width=E.ww(1.0))
            for v, r in zip(angs, rr):
                put_text(f, ctx, "w", cx + 0.7071 * r + 6, cy - 0.7071 * r - 5, f"{v:02d} DEG", size=L.T_MICRO,
                         alpha=0.6 * al)
            for name, sx, sy, anchor in (("N", 0.0, -1.0, "ms"), ("S", 0.0, 1.0, "ms"), ("E", -1.0, 0.0, "rs"),
                                         ("W", 1.0, 0.0, "ls")):
                r = rr[1] if sx == 0.0 else rr[2] + 22.0
                x, y = cx + sx * (r + 8.0), cy + sy * r + (26.0 if sy > 0 else 6.0 if sy == 0 else -10.0)
                if view[0] + 20 < x < view[2] - 20 and view[1] + 30 < y < view[3] - 8:
                    put_text(f, ctx, "w", x, y, name, size=L.T_LABEL, alpha=0.85 * al, anchor=anchor)
            f.crosses("w", [cx], [cy], 12.0, 0.7 * ov)
            put_text(f, ctx, "w", cx + 16, cy + 26, "ZENITH", size=L.T_MICRO, alpha=0.6 * al)
        # where the shower comes from: its axis, a red mark on the sky (the point of the first interaction)
        px, py, _, ok = cam.project(e["P1"][None].astype(np.float32))
        if ok[0]:
            X, Y = float(px[0]), float(py[0])
            with f.build(age_bar - 0.1, (X - 30.0, Y - 30.0, X + 30.0, Y + 30.0), flow="out", wave=0.1, marks=False,
                         key=65):
                f.segments("r", [X - 24, X + 10, X, X], [Y, Y, Y - 24, Y + 10], [X - 10, X + 24, X, X],
                           [Y, Y, Y - 10, Y + 24], 0.9 * ov, width=L.LW)
        w.draw_cascades(f, cam, age, alive, env, gain=gain * (1.0 + 0.16 * kick))
        boxes = w.draw_interaction(f, cam, age, view, ctx, tags=True, avoid=lay["boxes"])
        w.draw_labels(f, cam, age, alive, view, avoid=avoid + boxes, limit=5)
        # the front, read from below: its altitude and the time it has left
        front = w.front(p, a)
        left = max(0.0, e["t_ground"] - a)
        txt = f"FRONT {front:06.3f} KM OVERHEAD   T-{left:05.3f} S" if left > 0 else "FRONT ON THE GROUND          "
        self._readout(f, ctx, st, lay, txt, age_bar - 0.15, 66, alpha=al)

    # ------------------------------------------------------------------ the city
    def _labels(self, p, kinds, rank):
        """Names of the model for phrase p (world positions), nearest to the site first."""
        key = (p, kinds, rank)
        if key not in self._lab:
            self._lab[key] = self.town.labels(kinds, rank=rank, xf=self._xf(p))
        return self._lab[key]

    def _names(self, f, ctx, st, cam, p, age, taken, kinds=("street", "landmark"), rank=1, limit=10, lift=0.0, tags=True,
               tall=False, d_min=0.0, key=80, wave=None, lod="basin"):
        """Names on a view of the city: each one on a leader out of its point, clear of the towers, of the
        fixed texts and of the others (`taken`: boxes, extended in place); the first `limit` that find room.
        tall=True: only the buildings whose height the map gives as surveyed, the tallest first (a height is
        written as a figure only then); d_min = km from the corner under which a name is left out (the place
        has its own tags). Made in turn from the cut - or, with `wave` (the seconds the city takes to be drawn,
        at level `lod`), each one when the pens that draw the streets get to it."""
        view = st.view
        n = 0
        labels = self._labels(p, kinds, rank)
        if tall:
            labels = sorted((lb for lb in labels if lb.h_src == "tag" and lb.h), key=lambda lb: -lb.h)
        for lb in labels:
            if n >= limit:
                break
            if lb.d < d_min:
                continue
            x, y, ok = self.town.screen(cam, lb.pos)
            if not ok or not (view[0] + 40 < x < view[2] - 40 and view[1] + 70 < y < view[3] - 30):
                continue
            txt = lb.short
            if lb.kind == "landmark" and lb.h_src == "tag" and lb.h:
                txt = f"{lb.short} {lb.h:.0f} M"
            land = lb.kind == "landmark"
            size = L.T_MICRO
            for dx, dy, anchor in ((16.0, -20.0 - lift, "ls"), (-16.0, -20.0 - lift, "rs"), (16.0, 26.0, "ls"),
                                   (-16.0, 26.0, "rs")):
                box = tbox(x + dx, y + dy, txt, size, anchor, 4)
                if (box[0] < view[0] + 10 or box[2] > view[2] - 10 or box[1] < view[1] + 60 or box[3] > view[3] - 8
                        or hidden(ctx, *box) or _hits(box, taken, 6.0)):
                    continue
                taken.append(box)
                a_n = None if age is None else age - (0.25 + 0.05 * n if wave is None else
                                                      min(self.town.arrival(lb.tau, wave, lod), wave) + 0.1)
                _plate(f, box, x, a_n, 0.06, 0.16)            # under the name: the map does not run through it
                with f.build(a_n, (min(x, box[0]) - 2, min(y, box[1]) - 2, max(x, box[2]) + 2, max(y, box[3]) + 2),
                             flow="out", origin=(x, y), wave=0.1, line=0.1, marks=False, key=key + n):
                    f.segments("w", [x], [y], [x + dx - (4 if dx > 0 else -4)], [y + dy - 4], E.wl(0.55), width=E.ww(1.0))
                    f.dots("w", [x], [y], 2.0, 1.0)
                    if land and tags:
                        f.tag("w", x + dx, y + dy, txt, size=size, pad=3, anchor=anchor)
                    else:
                        f.text("w", x + dx, y + dy, txt, size=size, alpha=0.9, anchor=anchor)
                n += 1
                break

    def _wall_tag(self, f, ctx, st, cam, p, age, taken, title="YOU ARE HERE", lines=("9TH ST X VINE ST", "CINCINNATI // OHIO"),
                  prefer=(1, -1), wall=False):
        """The red callout of the place, out of the corner of 9th Street and Vine Street (wall=True: out of
        the wall as the model draws it)."""
        sx, sz = self.site(p)
        pos = (sx, 0.0, sz)
        if wall:
            pos = next((lb.pos for lb in self._labels(p, ("site",), 0) if lb.short == "THE WALL"), pos)
        x, y, ok = self.town.screen(cam, pos)
        if not ok:
            return
        box = self._callout(f, ctx, st.view, x, y, title, list(lines), age, taken, prefer=prefer, dx=70.0, dy=86.0)
        if box:
            taken.append(box)

    @staticmethod
    def _callout(f, ctx, view, x, y, title, lines, age, avoid, prefer=(1, -1), dx=56.0, dy=50.0, size=L.T_TAG):
        """A red call-out over a map: where it finds room (shower.auto_callout), on a black plate that opens
        with it. Returns its box, or None."""
        kw = dict(red=True, avoid=avoid, prefer=prefer, dx=dx, dy=dy, size=size)
        with f.build(-1.0, view):                             # a dry run: nothing is drawn, the place is found
            box = auto_callout(f, ctx, view, x, y, title, lines, build=age, **kw)
        if box is None:
            return None
        _plate(f, box, x, age, 0.12, 0.25)
        auto_callout(f, ctx, view, x, y, title, lines, build=age, **kw)
        return box

    def _front(self, p, land):
        """What the front of shower p has uncovered of the city `land` seconds after it reached the ground:
        dict(reveal=radius in km, centre=the core in the model) for City.draw - the ring that runs out over
        the ground (World.draw_splash) draws the city as it goes."""
        e = self.world.events[p]
        return dict(reveal=0.3 + 11.0 * (1 - (1 - min(1.0, land / 2.6)) ** 2.2),
                    centre=self._model_xz(p, float(e["G"][0]), float(e["G"][2])))

    def _credit(self, f, ctx, st, lay, age):
        """Where the map comes from: the line its licence asks for, small, in a corner of the view."""
        txt = self.town.ATTRIBUTION
        y = st.view[3] - 12.0
        for x, anchor in ((st.tx1, "rs"), (st.tx0, "ls")):
            box = tbox(x, y, txt, L.T_MICRO, anchor, 3)
            if not hidden(ctx, *box) and not _hits(box, lay["boxes"]):
                f.occlude(box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2)
                with f.build(None if age is None else age - 0.4, box, wave=0.1, marks=False, key=79):
                    f.text("w", x, y, txt, size=L.T_MICRO, alpha=0.6, anchor=anchor)
                return box
        return None

    def _marks(self, f, cam, p, a, gain, r_km=0.03, fog=None):
        """What reached the ground, at the scale of a city: a point where each particle landed (muons red), and
        a ring that leaves the point when it lands (r_km: how far it runs)."""
        gx, gt, mu, gz = self._arrivals(p)
        m = gt <= a
        if not m.any():
            return 0, 0
        P = np.stack([gx[m], np.zeros(int(m.sum())), gz[m]], 1).astype(np.float32)
        sx, sy, z, ok = cam.project(P)
        al = (a - gt[m]).astype(np.float32)
        red = mu[m]
        fg = np.ones(len(sx), np.float32) if fog is None else np.clip(1.25 - z / fog, 0.3, 1.0)
        hot = np.exp(-al / 0.5)
        for sel, lay, r, i0 in ((ok & ~red, "w", 1.6, 0.5), (ok & red, "r", 2.6, 0.95)):
            if sel.any():
                f.dots(lay, sx[sel], sy[sel], r * (1.0 + 0.8 * hot[sel]), (i0 + 1.1 * hot[sel]) * fg[sel] * gain)
        new = ok & (al < 1.0) & (r_km > 0.0)
        if new.any():                                         # the rings: fixed number of vertices, they only grow
            un = al[new] / 1.0
            rad = r_km * (0.15 + 0.85 * (1 - (1 - un) ** 3))
            ang = np.linspace(0.0, 2 * np.pi, 25, dtype=np.float32)
            C = P[new]
            R = np.stack([C[:, 0:1] + rad[:, None] * np.cos(ang)[None], np.zeros((len(C), 25), np.float32),
                          C[:, 2:3] + rad[:, None] * np.sin(ang)[None]], -1)
            rx, ry, _, rok = cam.project(R.reshape(-1, 3))
            rx, ry, rok = rx.reshape(len(C), 25), ry.reshape(len(C), 25), rok.reshape(len(C), 25)
            inten = ((1 - un) ** 1.8 * fg[new] * gain)[:, None] * np.ones((1, 24), np.float32)
            okm = rok[:, :-1] & rok[:, 1:]
            for sel, lay, k in ((~red[new], "w", 0.6), (red[new], "r", 1.0)):
                if sel.any():
                    mm = okm[sel]
                    f.segments(lay, rx[sel][:, :-1][mm], ry[sel][:, :-1][mm], rx[sel][:, 1:][mm], ry[sel][:, 1:][mm],
                               k * inten[sel][mm], width=L.LW)
        return int(m.sum()), int((m & mu).sum())

    def _landed(self, p):
        """Up to three muons of shower p that land on something of the city: a building that has a name first,
        then the river, a street, a roof - well apart from each other and from the corner (which has its own
        tags). [(age of the shower at the landing, world x, z, what)]."""
        key = ("landed", p)
        if key not in self._lab:
            gx, gt, mu, gz = self._arrivals(p)
            ext = self.town.extent
            sx, sz = self.site(p)
            cands = []
            for i in np.nonzero(mu)[0]:
                mx, mz = self._model_xz(p, gx[i], gz[i])
                if not (ext[0] + 0.15 < mx < ext[2] - 0.15 and ext[1] + 0.15 < mz < ext[3] - 0.15):
                    continue
                if math.hypot(gx[i] - sx, gz[i] - sz) < 0.25:
                    continue
                w = self.town.at(mx, mz)
                if w["kind"] == "building":
                    what, score = (w["name"], 3.0) if w.get("name") else ("A ROOF", 0.5)
                elif w["kind"] in ("river", "street") and w.get("name"):
                    what, score = w["name"], 2.0 if w["kind"] == "river" else 1.0
                else:
                    continue
                what = str(what).upper()
                while len(what) > 28 and " " in what:         # (a long name loses its last words, not half a word)
                    what = what.rsplit(" ", 1)[0]
                cands.append((-score, float(gt[i]), float(gx[i]), float(gz[i]), what[:28]))
            out = []
            for _, tl, x, z, what in sorted(cands):
                if len(out) < 3 and all(math.hypot(x - o[1], z - o[2]) > 0.4 and what != o[3] for o in out):
                    out.append((tl, x, z, what))
            self._lab[key] = out
        return self._lab[key]

    def _landings(self, f, ctx, st, cam, p, a, taken, life=1.5):
        """A muon lands on the city: a tag out of its point says on what, made on the landing, taken apart
        `life` seconds later."""
        for tl, x, z, what in self._landed(p):
            ag = a - tl
            if not (0.0 <= ag < life):
                continue
            sx, sy, ok = self.town.screen(cam, (x, 0.0, z))
            if not ok:
                continue
            box = self._callout(f, ctx, st.view, sx, sy, "MU >> " + what, [], B.io(ag, life - ag, out=0.2, span=0.5),
                                taken, prefer=(-1, 1), dx=36.0, dy=40.0, size=L.T_SMALL)
            if box:
                taken.append(box)

    def _dive(self, f, ctx, t, st, lay, cam, p, a, u, age, alive, env, gain, ov, kick, age_bar, avoid):
        """Above the front, looking down, coming down with it: the lattice of the ground, downtown in the
        middle of it, growing; an altitude tape on the edge of the view."""
        w, town = self.world, self.town
        view = st.view
        e = w.events[p]
        al = min(1.0, ov)
        _ground(w, f, cam, "persp", view, gain * (1.0 + 0.45 * kick) * 1.6, gain, kick)
        # from high up the city is its streets, drawn out of the corner on the cut; the blocks and the buildings
        # are made when the camera comes under LOW km (the lines would only fill before)
        g_c = 0.9 * gain * float(np.clip(cam.focal / cam.alt / 300.0, 0.3, 1.0))
        g_w = min(gain, 1.0) if E.WALL else g_c               # (the wall rule: nothing is dimmed, see city.py)
        c_z = math.cos(math.radians(e["zen"]))
        f_low = (DIVE_LOW - DIVE_BACK[0] * c_z) / (1.0 + DIVE_BACK[1] * c_z)      # the front when the camera gets there
        a_low = a - (e["t1"] + max(0.0, e["h1"] - f_low) / (e["speed"] * c_z))
        kw = dict(xf=self._xf(p), lod="basin", wave=0.9, gain=g_w, t=t, view=view)
        town.draw(f, cam, layers=("river", "parks", "streets", "site"), age=age_bar,
                  site=dict(rings=(0.25, 0.5, 1.0), pole=0.0, cross=0.12, pulse=0.0, gain=g_c, white=g_w), **kw)
        town.draw(f, cam, layers=("blocks", "buildings"), age=a_low if age_bar is None else min(a_low, age_bar), **kw)
        w.draw_cascades(f, cam, age, alive, env, gain=gain * (1.0 + 0.16 * kick))
        w.draw_hits(f, cam, age, alive, gain=gain)
        boxes = w.draw_interaction(f, cam, age, view, ctx, tags=True, avoid=lay["boxes"])
        w.draw_labels(f, cam, age, alive, view, avoid=avoid + boxes, limit=4)
        taken = list(lay["boxes"]) + boxes
        self._wall_tag(f, ctx, st, cam, p, age_bar - 0.2, taken)
        # the altitude tape: a ruler that scrolls past a fixed mark, one tick per 100 m, a figure per km
        hc = cam.alt
        kpx = 46.0                                            # px per km of the tape
        box = None
        for xa in (st.tx1 - 156.0, st.tx0 + 8.0):             # on the right edge of the view, else on the left
            for yb in (view[3] - 70.0, min(t_.top for t_ in ctx.towers.values()) - 40.0):
                bx = (xa - 6.0, view[1] + 150.0, xa + 150.0, yb)
                if box is None and yb - bx[1] > 300.0 and _free(ctx, st, bx, (), lay["boxes"], pad=4.0):
                    box = bx
        if box is not None:
            x0, ym = box[0] + 6.0, 0.5 * (box[1] + box[3])
            f.occlude(box[0], box[1] - 4.0, box[2], box[3] + 4.0)
            with f.build(age_bar, box, flow="tb", wave=0.2, marks=False, key=67):
                f.segments("w", [x0], [box[1]], [x0], [box[3]], E.wl(0.7 * ov), width=E.ww(1.0))
                hs = np.arange(math.ceil((hc - (box[3] - ym) / kpx) * 10), math.floor((hc + (ym - box[1]) / kpx) * 10) + 1)
                hs = hs[hs >= 0]
                ys = ym - (hs / 10.0 - hc) * kpx
                ln = np.where(hs % 10 == 0, 22.0, np.where(hs % 5 == 0, 12.0, 6.0))
                if E.WALL:                                    # heavier ticks: the 500 m and the km ones only
                    ys, ln = ys[hs % 5 == 0], ln[hs % 5 == 0]
                f.segments("w", np.full(len(ys), x0), ys, x0 + ln, ys, E.wl(0.75 * ov), width=E.ww(1.0))
                for hk, yk in zip(hs, ys):
                    if hk % 10 == 0 and box[1] + 14 < yk < box[3] - 6:
                        f.text("w", x0 + 28, float(yk) + 5, f"{hk // 10:02d}", size=L.T_MICRO, alpha=0.7 * al)
                f.segments("r", [x0 - 4], [ym], [x0 + 40], [ym], 1.0 * ov, width=L.LW)
                f.tag("r", x0 + 46, ym - 14, f"{hc:05.2f} KM", size=L.T_SMALL, pad=4, alpha=al)
        self._credit(f, ctx, st, lay, age_bar)

    def _city_plan(self, f, ctx, t, st, lay, cam, p, a, u, age, alive, env, gain, ov, kick, age_bar, avoid):
        """The street plan around the corner. On the bar of the landing the front itself draws the map: the
        ring that runs out over the ground uncovers it; afterwards the map is made on the cut. What reached the
        ground is marked where it landed."""
        w, town = self.world, self.town
        view = st.view
        e = w.events[p]
        al = min(1.0, ov)
        xf = self._xf(p)
        land = a - e["t_ground"]                              # seconds since the front reached the ground
        landing = grid(t)[1] == 2 and land >= 0.0             # the bar of the landing
        site = dict(rings=(0.25, 0.5), pole=0.0, cross=0.1, pulse=0.6)
        if landing:                                           # the front on the ground uncovers the map as it runs out
            town.draw(f, cam, xf=xf, lod="basin", hatch=True, age=age_bar, gain=gain, t=t, view=view, site=site,
                      **self._front(p, land))
        else:
            town.draw(f, cam, xf=xf, lod="basin", hatch=True, age=age_bar, wave=0.8, gain=gain, t=t, view=view, site=site)
        w.draw_cascades(f, cam, age, alive, env, gain=0.8 * gain * (1.0 + 0.16 * kick))
        w.draw_splash(f, cam, age, gain=gain)
        n_all, n_mu = self._marks(f, cam, p, a, gain, r_km=0.09)
        # the core of the shower on the map, and how far from it
        gx, gy, _, _ = cam.project(e["G"][None].astype(np.float32))
        X, Y = float(gx[0]), float(gy[0])
        taken = list(lay["boxes"])
        with f.build(age_bar, view, flow="out", origin=(X, Y), wave=0.25, marks=False, key=57):
            f.segments("r", [X, X, X, X], [Y, Y, Y, Y], [view[0], view[2], X, X], [Y, Y, view[1], view[3]], E.wl(0.4 * ov),
                       width=E.ww(1.0))
            sc = cam.scale
            f.rings("w", [X] * 3, [Y] * 3, [sc * r for r in (0.5, 1.0, 2.0)], E.wl(0.22 * ov), width=E.ww(1.0))
            for r, name in ((0.5, "0.5 KM"), (1.0, "1 KM"), (2.0, "2 KM")):
                bx = tbox(X + sc * r + 7, Y - 8, name, L.T_MICRO)
                if bx[2] < st.tx1 and not hidden(ctx, *bx) and not _hits(bx, taken):
                    f.text("w", X + sc * r + 7, Y - 8, name, size=L.T_MICRO, alpha=0.7 * al)
        cb = self._credit(f, ctx, st, lay, age_bar)
        if cb:
            taken.append(cb)
        self._wall_tag(f, ctx, st, cam, p, age_bar - 0.15, taken)
        sx, sz = self.site(p)
        d_core = math.hypot(float(e["G"][0]) - sx, float(e["G"][2]) - sz)
        bx = self._readout(f, ctx, st, lay, f"AT GROUND {n_all:04d}   MU {n_mu:03d}   CORE {d_core:04.2f} KM FROM THE WALL",
                           age_bar - 0.2, 68, taken=taken, alpha=al)
        if bx:
            taken.append(bx)
        wv = None if landing else 0.8                         # the names come when the pens get there
        self._names(f, ctx, st, cam, p, age_bar, taken, kinds=("landmark",), rank=1, limit=4, tall=True, d_min=0.12,
                    wave=wv)
        self._names(f, ctx, st, cam, p, age_bar, taken, kinds=("bridge", "river", "place"), rank=1, limit=3, tags=False,
                    d_min=0.3, key=86, wave=wv)
        self._names(f, ctx, st, cam, p, age_bar, taken, kinds=("street",), rank=1, limit=5, tags=False, d_min=0.16, key=90,
                    wave=wv)
        self._landings(f, ctx, st, cam, p, a, taken)

    def _skyline(self, f, ctx, t, st, lay, cam, p, a, u, age, alive, env, gain, ov, kick, age_bar, avoid):
        """Downtown from the air, low: the towers, the corner in red, the tracks of the shower coming down
        through them, the front running out over the streets."""
        w, town = self.world, self.town
        view = st.view
        land = a - w.events[p]["t_ground"]
        g_w = min(gain, 1.0) if E.WALL else 0.5 * gain
        kw = dict(xf=self._xf(p), lod="district", gain=g_w, t=t, view=view, fog=(0.8, 3.2),
                  site=dict(rings=(0.1, 0.25), pole=0.12, cross=0.1, pulse=0.4, gain=0.5 * gain, white=g_w))
        if grid(t)[1] == 2 and land >= 0.0:                   # the bar of the landing: the front draws the city
            town.draw(f, cam, age=age_bar, **kw, **self._front(p, land))
        else:
            town.draw(f, cam, age=age_bar, wave=0.8, **kw)
        w.draw_cascades(f, cam, age, alive, env, gain=gain * (1.0 + 0.16 * kick))
        if land < 0.45:                                       # the front crosses the city, then it is out of the model
            w.draw_splash(f, cam, age, gain=gain * (1.0 - max(0.0, land - 0.2) / 0.25))
        self._marks(f, cam, p, a, gain, r_km=0.07, fog=3.2)
        taken = list(lay["boxes"])
        cb = self._credit(f, ctx, st, lay, age_bar)
        if cb:
            taken.append(cb)
        self._wall_tag(f, ctx, st, cam, p, age_bar - 0.15, taken)
        self._names(f, ctx, st, cam, p, age_bar, taken, kinds=("landmark",), rank=1, limit=4, lift=22.0, tall=True,
                    wave=None if (grid(t)[1] == 2 and land >= 0.0) else 0.8, lod="district")

    def _site(self, f, ctx, t, st, lay, cam, p, a, u, age, alive, env, gain, ov, kick, age_bar, avoid):
        """The corner: the wall, its three towers, the muons of the shower through the place. The model is the
        block as the map gives it; the muons are drawn at the density a shower of this energy has near its
        core, each one a line that comes down and stays a moment."""
        w, town = self.world, self.town
        view = st.view
        e = w.events[p]
        al = min(1.0, ov)
        xf = self._xf(p)
        g_w = min(gain, 1.0) if E.WALL else 0.7 * gain
        town.draw(f, cam, xf=xf, lod="block", age=age_bar, wave=0.6, gain=g_w, t=t, view=view,
                  fog=(0.06, 0.34) if E.WALL else (0.06, 0.5),      # (the far blocks would pile up on the horizon)
                  site=dict(rings=(0.05,), pole=0.0, cross=0.07, pulse=0.12, gain=0.7 * gain, white=g_w))
        wl = town.wall
        sx, sz = self.site(p)
        if wl is not None:
            mid = xf(np.array([[wl["mid"][0], 0.0, wl["mid"][1]]], np.float32))[0].astype(np.float64)
            nrm = xf(np.array([[wl["mid"][0] + wl["normal"][0], 0.0, wl["mid"][1] + wl["normal"][1]]],
                              np.float32))[0].astype(np.float64) - mid
            along = xf(np.array([[wl["b"][0], 0.0, wl["b"][1]]], np.float32))[0].astype(np.float64) - mid
            along /= max(float(np.linalg.norm(along)), 1e-9)
            tops = [xf(np.array([[ft[0], h, ft[1]]], np.float32))[0].astype(np.float64) for ft, h in wl["towers"]]
        else:
            mid, nrm, along, tops = np.array([sx, 0.0, sz]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), []
        # the muons of the shower through the place: they arrive with the front, over a second
        d0 = e["d0"].astype(np.float64)
        n = 30
        k = np.arange(n)
        t_arr = e["t_ground"] + 0.9 * hash01(k, p, 11) ** 1.6 - 0.04
        G = (mid[None] + along[None] * ((hash01(k, p, 12) - 0.5) * 0.07)[:, None]
             + nrm[None] * (0.004 + 0.04 * hash01(k, p, 13) - 0.012)[:, None])
        G[:, 1] = 0.0
        self._falls(f, cam, G, d0, a - t_arr, gain)
        # ... and the rain that never stops, on the eighth notes of the phrase (two muons each)
        j = np.repeat(np.arange(-2, 34), 2)
        kk = np.arange(len(j))
        Gr = (mid[None] + along[None] * ((hash01(kk, p, 21) - 0.5) * 0.09)[:, None]
              + nrm[None] * (0.05 * hash01(kk, p, 22) - 0.014)[:, None])
        Gr[:, 1] = 0.0
        zr, ar = 0.12 * hash01(kk, p, 23), 2 * np.pi * hash01(kk, p, 24)       # each from its own direction
        for q in range(len(j)):
            ag = a - (e["t1"] + j[q] * 0.5 * BEAT + 0.04 * (q % 2))
            if -0.5 < ag < 1.6:
                dq = np.array([math.sin(zr[q]) * math.cos(ar[q]), -math.cos(zr[q]), math.sin(zr[q]) * math.sin(ar[q])])
                self._falls(f, cam, Gr[q:q + 1], dq, np.array([ag]), 0.8 * gain)
        # the three that go through a detector (the tower the shower falls on): a white tick on its head
        aim = AIMS[p]
        if aim is not None and tops:
            for ev, det, ta, en in w.crossings:
                if ev != p:
                    continue
                self._falls(f, cam, tops[aim][None] + d0[None] * (tops[aim][1] / max(-d0[1], 1e-3)), d0,
                            np.array([a - ta - tops[aim][1] / max(-d0[1], 1e-3) / 0.6]), gain, bold=True)
                ac = a - ta
                if 0.0 <= ac < 0.8:
                    x, y, ok = town.screen(cam, tops[aim])
                    if ok:
                        un = ac / 0.8
                        rx, ry = _ring(x, y, 8.0 + 46.0 * (1 - (1 - un) ** 3))
                        f.polyline("w", rx, ry, 0.9 * (1 - un) ** 1.5, width=L.LW)
                        f.dots("w", [x], [y], 3.6 * (1 - un) + 1.0, 1.6 * (1 - un))
        # the live detectors: a hit on a real tower lights the head of the same tower in the drawing
        for j, key in enumerate(L.ORDER):
            if j >= len(tops) or not ctx.det.online(key, t):
                continue
            ha, he = ctx.det.last(key, t, echoes=True)
            x, y, ok = town.screen(cam, tops[j])
            if ok and ha < 1.2:
                f.dots("r", [x], [y], 5.0 + 5.0 * he, 1.8 * he * math.exp(-ha / 0.3))
        taken = list(lay["boxes"])
        cb = self._credit(f, ctx, st, lay, age_bar)
        if cb:
            taken.append(cb)
        # names: the detectors on their towers, the wall, the place of the audience
        for j, key in enumerate(L.ORDER):
            if j >= len(tops):
                break
            x, y, ok = town.screen(cam, tops[j])
            txt = L.NAMES[key]
            bx = tbox(x, y - 16, txt, L.T_MICRO, "ms", 3)
            if ok and view[0] + 30 < x < view[2] - 30 and not hidden(ctx, *bx) and not _hits(bx, taken):
                with f.build(age_bar - 0.3 - 0.06 * j, bx, wave=0.06, marks=False, key=74 + j):
                    f.tag("r" if j == aim else "w", x, y - 16, txt, size=L.T_MICRO, pad=3, anchor="ms")
                taken.append(bx)
        if wl is not None:
            self._wall_tag(f, ctx, st, cam, p, age_bar - 0.15, taken, title="THE WALL",
                           lines=(f"{wl['length'] * 1000.0:.1f} M WIDE", "3 MUON DETECTORS"), prefer=(-1, -1), wall=True)
        here = mid + nrm * 0.011
        x, y, ok = town.screen(cam, (here[0], 0.0, here[2]))
        if ok:
            f.crosses("r", [x], [y], 16.0, 0.95 * ov, width=L.LW)
            bx = self._callout(f, ctx, view, x, y, "YOU ARE HERE", ["9TH ST X VINE ST", "CINCINNATI // OHIO"],
                               age_bar - 0.35, taken, prefer=(1, 1), dx=80.0, dy=70.0)
            if bx:
                taken.append(bx)
        self._names(f, ctx, st, cam, p, age_bar, taken, kinds=("street",), rank=0, limit=3, tags=False)
        self._names(f, ctx, st, cam, p, age_bar, taken, kinds=("landmark",), rank=0, limit=2, lift=10.0)

    @staticmethod
    def _falls(f, cam, G, d0, age, gain, bold=False, drop=0.6, tail=0.22):
        """Muons through the site: each one comes down along d0 onto its point of G (age = seconds since it
        got there; it travels `drop` km a second on the way), leaves its line for a moment and a small ring on
        the ground."""
        age = np.asarray(age, np.float64)
        live = (age > -tail / drop) & (age < 1.6)
        if not live.any():
            return
        G, age = G[live], age[live]
        s_head = np.clip(-age * drop, 0.0, None)              # km still to go along the track (0 = landed)
        s_tail = s_head + tail
        A = G - d0[None] * s_tail[:, None]
        Bp = G - d0[None] * s_head[:, None]
        P = np.concatenate([A, Bp, G]).astype(np.float32)
        sx, sy, _, ok = cam.project(P)
        n = len(G)
        fade = np.where(age > 0, np.exp(-np.maximum(age, 0.0) / 0.5), 1.0)
        m = ok[:n] & ok[n:2 * n]
        if m.any():
            f.segments("r", sx[:n][m], sy[:n][m], sx[n:2 * n][m], sy[n:2 * n][m], 0.15 * fade[m] * gain,
                       (1.5 if bold else 1.0) * fade[m] * gain, width=L.LW_BOLD if bold else L.LW)
        fly = m & (age < 0)
        if fly.any():
            f.dots("r", sx[n:2 * n][fly], sy[n:2 * n][fly], 3.0, 1.6 * gain)
        hit = ok[2 * n:] & (age >= 0) & (age < 0.9)
        if hit.any():
            un = age[hit] / 0.9
            f.dots("w", sx[2 * n:][hit], sy[2 * n:][hit], 2.2, 1.4 * (1 - un) ** 2 * gain)
