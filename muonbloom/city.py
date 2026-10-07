"""CITY - downtown Cincinnati around 9th Street x Vine Street, where the wall of the show stands, as line work.

The model is real: it is baked from OpenStreetMap by tools/build_city.py into data/cincinnati.npz (2 600 street
runs, 6 070 building footprints, 7 480 extruded volumes, the Ohio River, its bridges). The show never needs
the network. The data is (C) OpenStreetMap contributors (ODbL): where the city is shown, set the line
`City.ATTRIBUTION` (MAP DATA (C) OPENSTREETMAP CONTRIBUTORS) somewhere in the view, small.

COORDINATES    kilometres, the conventions of the shower world (scenes/shower.py):
                   x = east,  y = up,  z = SOUTH (north is -z)
               so the plan views of the show (a camera looking down with `up=(0, 0, -1)`) have north up and east
               to the right, and an elevation seen from +z looks north. The origin is the middle of the crossing
               of Vine Street and 9th Street. The ground is the plane y = 0; the rivers lie RIVER_Y under it
               (the real pool is some 25 to 30 m under the street grid). The basin is drawn flat and the hills
               around it are not in the model.
EXTENT         City.extent = (x0, z0, x1, z1) = (-1.73, -1.40, 1.73, 2.15): from Over-the-Rhine (north) to the
               Kentucky bank (Covington, Newport) with the whole river, from I-75 (west) to I-471 (east).
THE GRID       the numbered streets run 78.9 degrees from north, Vine Street 348.9: the grid is turned 11
               degrees to the west. `yaw=City.GRID_YAW` squares it with the axes (Vine Street straight up).
               From centre line to centre line a block is about 142 m east-west (Race, Vine, Walnut, Main)
               and about 100 m north-south (8th, 9th, Court).

THE SITE       City.SITE = (0, 0, 0), the corner. City.wall (a dict, or None): the south face of 905 Vine
               Street, on the north-west corner, 27.7 m long, looking over 9th Street: `a`, `b` = its left and
               right ends for somebody who faces it ((x, z), model km), `mid`, `u` (from a to b), `normal`
               (towards the audience), `height` (km, 13.1 m to the parapet), `towers` (the three detector
               towers 1.8 m in front of it, on the sixths of the wall: foot (x, z), height: 3.45, 6.5 and
               3.45 m, a 400 mm detector box at each head: figures of the site drawing). This is where the
               map of the festival puts "muon : bloom", and the length of that face is the width of the wall
               in the site drawing, but nobody has written "this wall": see tools/build_city.py. Draw the
               corner without it with `draw_site(wall=False)`. The marker follows `age` only (it is made
               first), not `reveal`.

LAYERS         every layer has its own draw method; `draw()` calls them in this order
    river      banks, scan lines that drift with `t` (the Ohio flows west here), the piers of the bridges
    parks      dashed outlines of the named parks
    streets    centre lines by class (motorways, ramps, primary ... residential, pedestrian, lanes), railways
               dashed, bridges doubled
    blocks     the footprints of the buildings on the ground (+ `hatch`: scan lines through them, parallel to
               the numbered streets: the figure-ground of a plan)
    buildings  the volumes: corner edges that brighten towards the top, roof outlines, floor plates on what is
               tall. Lines that would fill thin out by themselves (far away, or seen from above): a tower
               never burns to a white bar. The towers that have a label are drawn stronger.
               (`landmarks`: those towers alone.)
    site       red: the crossing (the two streets over `cross` km), a pole over the corner, range rings, the
               wall with its three towers and their detectors, a ring that leaves the wall every PULSE_T s

LEVELS OF DETAIL   `lod=` picks what the layers hold and how far from the site they go. Segments when built,
               before the camera has cut anything (City.counts(lod)):
    "basin"    the whole box: roads down to residential, footprints over 250 m2, volumes over 20 m,
               floor plates every 16 m on the towers                               34 000 (+ 8 900 hatch)
    "district" 1.3 km around the site: every street, every footprint, volumes over 12 m (all of them within
               450 m), floor plates every 8 m                                      55 000 (+ 10 900 hatch)
    "block"    420 m around the site: everything, floor plates every 4 m           10 500 (+ 3 900 fine hatch)
    The model does not stop on a hard edge: it dims to nothing over the last 18 % of the radius, and over the
    last 150 m before the sides of the box. `radius=` shrinks a level further. `City.lod_for(px_per_km)`
    names the level that suits a scale. In an elevation (a camera looking along the ground) everything that
    lies on the ground falls on one line: leave it out (`layers=("buildings", "site")`).

BUILD          nothing pops, and nothing is wiped on: the city is drawn by what makes it. `age` = seconds since
               the model started to be made (None = built; a negative age draws nothing; give
               `build.io(age, left)` to take it apart the way it was made). Pens leave the corner and run
               through the street network (bright heads on the main streets): a street is drawn when the
               shortest way from the corner reaches it, in the direction the pen travels. From the street a
               pen goes round each footprint, the scan lines follow, and when the footprint is closed the
               building rises (RISE_T seconds, a small overshoot); the banks, the water and the piers come
               when the streets reach the river. The last pen arrives after `wave` seconds.
               `reveal=` (km) is the other way: a circle of that radius around `centre=(x, z)` (default: the
               corner) uncovers the model - for a view where something real sweeps over the city (the
               footprint of a shower). The circle cuts the lines where it is, the buildings rise behind it.

TRANSFORM      `City.xf(offset, scale, yaw, yscale, pivot)`: where the model stands in the world of the scene.
               world = offset + scale * turn(yaw) * (model - pivot), heights times yscale. yaw is in degrees,
               clockwise seen from above (like a compass bearing). Every draw method and `labels()` take `xf=`.

WHAT IS THERE  `City.at(x, z)` -> what stands at a point of the model: a building (its name if the map has one,
               its height and where that figure comes from, its levels), a named street, the river, or the
               ground. For what a scene writes where a muon lands.

LABELS         `City.labels(kinds, rank, radius, xf, per_name)` -> list of Label, nearest to the site first:
               name, short (fits a tag), kind (site, street, landmark, bridge, river, park, place), rank (0 =
               write it first), pos (world, with xf), d (km from the site), tau (km of streets from the corner:
               `City.arrival(tau, wave, lod)` = the age at which the pens get there, to build its tag
               then), brg (compass bearing of a street, 0..180), h (m), h_src ("tag" = the surveyed height
               written in the map: the only kind to print as a figure; "levels" / "default" are estimates),
               levels, info. The module draws no text: a scene places the tags itself
               (`City.screen(cam, pos)`), clear of the towers.

    from muonbloom import city
    c = city.get()                                             # loaded once per process (0.2 s)
    # a plan, north up, 372 px per km, the corner on (1560, 620): the whole box is 1290 x 1320 px
    cam = c.plan_camera((1560.0, 620.0), 372.0)
    c.draw(f, cam, lod="basin", hatch=True, age=t - t0, wave=2.4, t=t)
    # the same turned square with the canvas: Vine Street straight up, 1 km = 1080 px
    xf = c.xf(yaw=c.GRID_YAW)
    c.draw(f, c.plan_camera((1489.0, 760.0), 1080.0), xf=xf, lod="district", hatch=True, t=t)
    # a low perspective from Covington, over the river, looking at the towers and the corner behind them
    cam = Camera((-1.25, 0.26, 2.35), (0.25, 0.07, 0.35), fov_deg=34.0, screen_center=(1489.0, 760.0))
    c.draw(f, cam, lod="basin", fog=(1.2, 4.2, 0.2), age=t - t0, t=t, site=dict(pole=0.3))
    for lb in c.labels(("landmark",), rank=0):
        sx, sy, ok = c.screen(cam, lb.pos)                     # then the scene's own put_tag / auto_callout
    # the model set on the ground of another world (here 12 times larger, its wall on the point p)
    xf = c.xf(offset=p, scale=12.0, pivot=(c.wall["mid"][0], 0.0, c.wall["mid"][1]))

COST           pure numpy, no state, any frame alone and the same in any process. On the development laptop,
               logic only (what a draw-list recorder sees), full canvas, built / while the pens run (median,
               measured while other renders kept the machine busy):
                   basin, plan with hatch          2.0 / 4.1 ms     44 000 segments sent
                   district, plan with hatch       5.9 / 8.8 ms     44 000
                   basin, perspective              2.3 / 4.4 ms     30 000
                   district, perspective           4.9 / 7.7 ms     29 000
                   block, perspective with hatch   2.0 / 2.5 ms      7 500
               What the camera does not see is not sent to the frame (`view=` narrows it to a part of the
               canvas). Loading the file takes 0.2 s, the tables of a level 10 ms the first time.

FACTS (for what a scene writes on the wall; heights are the `height` tags of OpenStreetMap, which agree with
the published figures for the towers below)
    GREAT AMERICAN TOWER AT QUEEN CITY SQUARE 202.7 M (665 FT, 40 LEVELS, 2011, the tallest of the city)
    CAREW TOWER 175 M (574 FT, 49 LEVELS)      PNC TOWER 150.9 M (495 FT; its volume is mapped to 146 m)
    SCRIPPS CENTER 143 M (468 FT)      FIFTH THIRD CENTER 129 M (423 FT)
    KROGER BUILDING 98 M, 25 LEVELS (1014 VINE ST, 150 m north of the corner)
    DOWNTOWN MAIN LIBRARY: both sides of 9th Street, east of Vine (across Vine Street from the wall)
    The Ohio River is the state line: Cincinnati, Ohio on the north bank; Covington (west of the Licking
    River) and Newport (east of it), Kentucky, on the south bank.
    Bridges, west to east: Brent Spence (I-71 / I-75), Clay Wade Bailey, John A. Roebling Suspension Bridge,
    Taylor-Southgate, Purple People Bridge, Daniel Carter Beard (I-471).
    The corner: 39.10538 N, 84.51404 W.
"""
from __future__ import annotations

import json
import math
from collections import namedtuple

import numpy as np

from . import build as B
from . import engine as E
from . import layout as L
from .engine import OrthoCamera

DATA = L.DATA / "cincinnati.npz"
ATTRIBUTION = "MAP DATA (C) OPENSTREETMAP CONTRIBUTORS"
RISE = 0.11                     # km behind the circle over which a building rises (reveal)
RISE_T = 0.45                   # s a building takes to rise once its footprint is closed (age)
EDGE = 0.18                     # the last part of the radius dims to nothing
BOX_EDGE = 0.15                 # km before the sides of the box over which the model dims
PULSE_T = 2.6                   # s between two rings leaving the wall
NEAR = 0.003                    # km: the near plane of the perspective views (engine.Camera cuts at 50 m)
FLOW = 0.02                     # km / s: drift of the scan lines of the river (to the west)
PLATE_PX = 7.0                  # floor plates closer than this on screen dim, so that a tower never fills
WALL_PX = 5.0                   # ... and so do the corner edges of a building that is small on screen
CORNER = 28.0                   # degrees: a footprint turns at least this much where a corner edge stands
HEADS = 32                      # about how many pens of the streets show a bright head at the same time
# the pens of the streets run ahead; what stands along a street follows this many km of streets behind them
LAG = dict(parks=0.12, blocks=0.22, hatch=0.27, buildings=0.3)

# what each level of detail keeps: radius (km, None = the box), highest street class, smallest footprint (m2),
# lowest volume (m) [inside `near` km: all of them], metres between floor plates, hatch ("coarse" / "fine")
LOD = {
    "basin": dict(radius=None, cls=5, area=250.0, vol_h=20.0, near=0.0, floors=16.0, hatch="coarse"),
    "district": dict(radius=1.3, cls=8, area=0.0, vol_h=12.0, near=0.45, floors=8.0, hatch="coarse"),
    "block": dict(radius=0.42, cls=8, area=0.0, vol_h=0.0, near=9.0, floors=4.0, hatch="fine"),
}
LAYERS = ("river", "parks", "streets", "blocks", "buildings", "site")
# streets by class: intensity, width (motorway, ramp, primary, secondary, tertiary, residential, pedestrian,
# service lane, railway)
ST_I = np.array([0.62, 0.26, 0.66, 0.54, 0.44, 0.33, 0.2, 0.17, 0.3], np.float32)
ST_W = np.array([1.7, 1.0, 1.5, 1.3, 1.1, 1.0, 1.0, 1.0, 1.0], np.float32)
FLOOR_MIN_H, FLOOR_MIN_AREA = 30.0, 150.0          # m, m2: what gets floor plates

# The wall rule (engine.WALL): on the brick a grey line is lost and a pile of dim lines is a white block. So
# nothing here is grey: a line is there at full level and at least engine.WALL_LINE wide, or it is not there.
# What used to dim (lines that would fill, the distance, the edge of the model) now goes away over a narrow
# band (never in one frame), and each level of detail holds fewer lines, so that black stays between them.
W_PLATE_PX = (10.0, 18.0)       # floor plates: gone when they are this close on screen (px), all there from ...
W_EDGE_PX = (7.0, 14.0)         # ... the corner edges of one facade
W_SIZE_PX = (8.0, 18.0)         # ... the roof outline of a building this small on screen
W_FAR = (0.3, 0.6)              # the band of the old dimming (fog, edge of the model) over which a line leaves
# (part: a volume with a smaller footprint (m2) is left out - the small parts of a tower are what piles up;
# edges: at most this many corner edges per volume, on the towers that have a label only when `plain` is 0: the
# other towers are then stacks of floor plates under a roof, which stay apart where corner edges would merge)
W_LOD = {"basin": dict(vol_h=60.0, floors=48.0, area=500.0, part=600.0, edges=4, plain=0),
         "district": dict(vol_h=60.0, near=0.0, floors=32.0, area=300.0, part=600.0, edges=4, plain=0),
         "block": dict(vol_h=9.0, near=0.0, floors=12.0, part=150.0, edges=4, plain=1)}
W_PILE = 0.8                    # level of the lines of a volume without a label: where towers overlap, less glow
W_ST_W = np.array([2.4, 1.6, 2.0, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6], np.float32)       # streets: rank is weight
W_ST_MIN = 0.3                  # streets under this level are left out (ramps, lanes, footways, tunnels)
# ... and from far away the small streets leave first: a street is there when this many km are W_SIZE_PX on screen
# (0: always). Secondary from 50 - 110 px per km, tertiary 80 - 180, residential and railways 115 - 260.
W_ST_SZ = np.array([0.0, 0.07, 0.0, 0.16, 0.1, 0.07, 0.07, 0.07, 0.07], np.float32)
W_PARK_SZ, W_WATER_SZ = 0.05, 0.1


def _band(v, lo_hi):
    """0 under lo, 1 over hi, smooth between."""
    u = np.clip((np.asarray(v, np.float32) - lo_hi[0]) / (lo_hi[1] - lo_hi[0]), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)

Label = namedtuple("Label", "name short kind rank pos d tau brg h h_src levels info")


class Xf:
    """Where the model stands in the world of a scene (see City.xf)."""

    def __init__(self, offset=(0.0, 0.0, 0.0), scale=1.0, yaw=0.0, yscale=1.0, pivot=(0.0, 0.0, 0.0)):
        a = math.radians(yaw)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]) @ np.diag([1.0, yscale, 1.0])
        self.M = (float(scale) * R).astype(np.float32)
        self.offset = np.asarray(offset, np.float32)
        self.pivot = np.asarray(pivot, np.float32)
        self.scale, self.yscale = float(scale), float(yscale)
        self.identity = bool(np.allclose(self.M, np.eye(3)) and not self.offset.any() and not self.pivot.any())

    def __call__(self, P):
        P = np.asarray(P, np.float32)
        if self.identity:
            return P
        return (P - self.pivot) @ self.M.T + self.offset


_ID = Xf()


class _Tab:
    """The segments of one layer at one level of detail, sorted by their distance from the site.
    a, b (n, 3) model km, a = the end the pen reaches first; ta, tb = when the pens reach the two ends (km of
    streets from the corner); da, db = straight distances of the two ends from the site; i0, i1 = intensity
    at a and at b; w = width; head = a pen that shows its head.
    A table of buildings has owners instead: oc = centre of the building a segment belongs to, own = its
    distance from the site, ot = when its footprint is closed (the building then rises as one piece);
    sp > 0: km between a floor plate and the next; sp < 0: -sp = km between two corner edges on a facade;
    sz > 0 (wall rule): the size (km) of the building a roof line belongs to."""
    __slots__ = ("a", "b", "ta", "tb", "da", "db", "i0", "i1", "w", "sp", "sz", "own", "oc", "ot", "head", "key")

    def __init__(self, a, b, i0, i1=None, w=1.0, ta=None, tb=None, sp=None, oc=None, ot=None, head=None, box=None,
                 sz=None):
        a, b = np.array(a, np.float32).reshape(-1, 3), np.array(b, np.float32).reshape(-1, 3)
        n = len(a)
        full = lambda v, dt=np.float32: np.broadcast_to(np.asarray(v, dt), (n,)).copy()
        i0 = full(i0)
        i1 = i0.copy() if i1 is None else full(i1)
        if box is not None:                             # the model dims before the sides of the box
            for P, i in ((a, i0), (b, i1)):
                side = np.clip(np.minimum(np.minimum(P[:, 0] - box[0], box[2] - P[:, 0]),
                                          np.minimum(P[:, 2] - box[1], box[3] - P[:, 2])) / BOX_EDGE, 0.0, 1.0)
                i *= _band(side, W_FAR) if E.WALL else side     # (the wall rule: it leaves over a short way, no grey)
        if oc is None:
            ta, tb = full(ta), full(tb)
            sw = tb < ta                                # a = where the pen comes from
            a[sw], b[sw] = b[sw].copy(), a[sw].copy()
            i0[sw], i1[sw] = i1[sw].copy(), i0[sw].copy()
            ta, tb = np.minimum(ta, tb), np.maximum(ta, tb)
            da, db = np.hypot(a[:, 0], a[:, 2]), np.hypot(b[:, 0], b[:, 2])
            key = np.minimum(da, db)
        else:
            oc = np.asarray(oc, np.float32).reshape(-1, 2)
            key = np.hypot(oc[:, 0], oc[:, 1])
        o = np.argsort(key, kind="stable")
        self.a, self.b, self.i0, self.i1, self.w, self.key = a[o], b[o], i0[o], i1[o], full(w)[o], key[o]
        self.sp = None if sp is None else full(sp)[o]
        self.sz = None if sz is None else full(sz)[o]
        self.head = None if head is None else full(head, bool)[o]
        self.ta = self.tb = self.da = self.db = self.own = self.oc = self.ot = None
        if oc is None:
            self.ta, self.tb, self.da, self.db = ta[o], tb[o], da[o], db[o]
        else:
            self.own, self.oc, self.ot = self.key, oc[o], full(ot)[o]

    def __len__(self):
        return len(self.a)

    def upto(self, radius):
        """How many of the segments (or of the owners) lie within `radius` km of the site."""
        return len(self.a) if radius is None else int(np.searchsorted(self.key, radius))

    def last(self, n):
        """When the last pen of the first n segments arrives."""
        if n <= 0:
            return 0.0
        return float((self.tb if self.oc is None else self.ot)[:n].max())


def _g3(xz, y):
    """(n, 2) ground points -> (n, 3) at height y (scalar or (n,))."""
    xz = np.asarray(xz, np.float32).reshape(-1, 2)
    out = np.empty((len(xz), 3), np.float32)
    out[:, 0], out[:, 2] = xz[:, 0], xz[:, 1]
    out[:, 1] = y
    return out


def _dashes(P0, P1, T0, T1, on, off):
    """Dashes of `on` km every `on + off` km along the segments P0 -> P1 ((n, 2) each), with the times of their
    ends (T0, T1 = the times of the ends of the segments)."""
    d = P1 - P0
    ln = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
    n = np.maximum(1, np.ceil(ln / (on + off))).astype(int)
    seg = np.repeat(np.arange(len(ln)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    s0 = k * (on + off) / ln[seg]
    s1 = np.minimum(s0 + on / ln[seg], 1.0)
    dt = (T1 - T0)[seg]
    return P0[seg] + d[seg] * s0[:, None], P0[seg] + d[seg] * s1[:, None], T0[seg] + dt * s0, T0[seg] + dt * s1


def _project(cam, A, Bp, view):
    """Screen ends of world segments through an engine camera (a perspective one is cut at the near plane),
    their depth (None for an orthographic camera), and which of them touch `view` (None = all of them)."""
    if getattr(cam, "ortho", False):
        x0, y0, _, _ = cam.project(A)
        x1, y1, _, _ = cam.project(Bp)
        depth = keep = None
    else:
        pos = cam.pos.astype(np.float32)
        da, db = (A - pos) @ cam.R.T, (Bp - pos) @ cam.R.T
        za, zb = da[:, 2], db[:, 2]
        keep = (za > NEAR) | (zb > NEAR)
        cut = keep & ((za < NEAR) | (zb < NEAR))             # one end behind the camera: bring it to the near plane
        if cut.any():
            u = ((NEAR - za[cut]) / (zb[cut] - za[cut]))[:, None]
            c = da[cut] + (db[cut] - da[cut]) * u
            behind_a = (za[cut] < NEAR)[:, None]
            da[cut], db[cut] = np.where(behind_a, c, da[cut]), np.where(behind_a, db[cut], c)
        za, zb = np.maximum(da[:, 2], NEAR), np.maximum(db[:, 2], NEAR)
        x0, y0 = cam.cx + cam.focal * da[:, 0] / za, cam.cy - cam.focal * da[:, 1] / za
        x1, y1 = cam.cx + cam.focal * db[:, 0] / zb, cam.cy - cam.focal * db[:, 1] / zb
        depth = 0.5 * (za + zb)
    v = view or (-40.0, -40.0, L.W + 40.0, L.H + 40.0)
    out = (np.minimum(x0, x1) > v[2]) | (np.maximum(x0, x1) < v[0]) | (np.minimum(y0, y1) > v[3]) | (np.maximum(y0, y1) < v[1])
    keep = ~out if keep is None else keep & ~out
    return x0, y0, x1, y1, depth, (None if keep.all() else keep)


class _Front:
    """A build in progress. timed: the pens have covered `r` km of streets, of `reach` (they brake as they
    arrive: r = reach * p * (2 - p), p = age / wave). Else: a circle of radius r around `centre`."""

    def __init__(self, timed, r, reach=1.0, age=0.0, wave=1.0, centre=None):
        self.timed, self.r, self.reach, self.age, self.wave = timed, float(r), float(reach), age, wave
        self.centre = (0.0, 0.0) if centre is None else (float(centre[0]), float(centre[1]))

    def rise(self, d):
        """0..1 (with its overshoot) of a building: d = when its footprint is closed (timed), or its distance
        from the centre of the circle."""
        if self.timed:
            t_done = self.wave * (1.0 - np.sqrt(np.clip(1.0 - d / self.reach, 0.0, 1.0)))
            return B.spring((self.age - t_done) / RISE_T)
        return B.spring((self.r - d) / RISE)

    def laid(self, d):
        """0..1 of what is laid down behind the pens (the water)."""
        return np.clip((self.r - d) / RISE, 0.0, 1.0)


class City:
    ATTRIBUTION = ATTRIBUTION
    SITE = (0.0, 0.0, 0.0)

    def __init__(self, path=None):
        z = np.load(path or DATA)
        meta = json.loads(str(z["meta"]))
        self.meta = meta
        self.extent = tuple(meta["extent"])
        self.RIVER_Y = float(meta["river_y"])
        self.grid_bearing = float(meta["grid_bearing"])            # of the numbered streets, degrees from north
        self.GRID_YAW = 90.0 - self.grid_bearing
        a = math.radians(self.grid_bearing)
        self.axis_9th = np.array([math.sin(a), -math.cos(a)], np.float32)              # along 9th Street, to the east
        self.axis_vine = np.array([self.axis_9th[1], -self.axis_9th[0]], np.float32)   # along Vine Street, to the north
        self.r_max = float(max(math.hypot(x, y) for x in self.extent[0::2] for y in self.extent[1::2]))
        self._set_wall(meta.get("wall"))
        self._labels = sorted(meta["labels"], key=lambda v: v["d"])
        self._tabs, self._reach = {}, {}
        z = {k: z[k] for k in z.files if k != "meta"}
        self._prepare(z)
        # what `at` looks things up in
        off = z["b_off"]
        self._q = dict(pts=z["b_pts"], off=off, bld=z["b_ring_bld"], inner=z["b_ring_inner"],
                       lo=np.minimum.reduceat(z["b_pts"], off[:-1], axis=0),
                       hi=np.maximum.reduceat(z["b_pts"], off[:-1], axis=0), h=z["b_h"], src=z["b_src"],
                       levels=z["b_levels"], names={int(k): v for k, v in meta["building_names"].items()},
                       st=self._st_named(z), st_names=meta["street_names"], river=z["wh_river"])

    # ------------------------------------------------------------------ the data
    def _set_wall(self, w):
        self.wall = None
        if not w:
            return
        a, b = np.array(w["a"], np.float32), np.array(w["b"], np.float32)
        u = (b - a) / float(np.hypot(*(b - a)))
        nrm = np.array([-u[1], u[0]], np.float32)
        if nrm[1] < 0:                                             # the audience stands south of the wall
            nrm = -nrm
        towers = [(a + u * s / 1000.0 + nrm * w["tower_off_m"] / 1000.0, h / 1000.0) for s, h in w["towers_m"]]
        self.wall = dict(a=a, b=b, mid=0.5 * (a + b), u=u, normal=nrm, height=w["height_m"] / 1000.0,
                         length=float(np.hypot(*(b - a))), towers=towers, address=w["address"])

    def _prepare(self, z):
        def seg_of(pts, tau, off):                     # the segments of open polylines, and the polyline of each
            first = np.ones(len(pts), bool)
            first[off[:-1]] = False
            b = np.nonzero(first)[0]
            return pts[b - 1], pts[b], tau[b - 1], tau[b], np.repeat(np.arange(len(off) - 1), np.diff(off) - 1)

        def ring_of(off):                              # index of the next point around each closed ring
            nxt = np.arange(1, off[-1] + 1)
            nxt[off[1:] - 1] = off[:-1]
            return nxt, np.repeat(np.arange(len(off) - 1), np.diff(off))

        cat = np.concatenate
        # --- streets: railways dashed, bridges with the two edges of their deck
        p0, p1, t0, t1, run = seg_of(z["st_pts"], z["st_tau"], z["st_off"])
        cls, flag = z["st_cls"][run], z["st_flag"][run]
        inten = ST_I[cls] * np.where(flag & 2, 0.4, 1.0) * np.where(flag & 1, 1.35, 1.0)
        rail = cls == 8
        r0, r1, rt0, rt1 = _dashes(p0[rail], p1[rail], t0[rail], t1[rail], 0.014, 0.01)
        deck = (flag & 1 > 0) & (cls != 1) & ~rail
        d = p1[deck] - p0[deck]
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None] * 0.006
        self._st = dict(p0=cat([p0[~rail], r0, p0[deck] + nrm, p0[deck] - nrm]),
                        p1=cat([p1[~rail], r1, p1[deck] + nrm, p1[deck] - nrm]),
                        t0=cat([t0[~rail], rt0, t0[deck], t0[deck]]), t1=cat([t1[~rail], rt1, t1[deck], t1[deck]]),
                        cls=cat([cls[~rail], np.full(len(r0), 8), cls[deck], cls[deck]]),
                        i=cat([inten[~rail], np.full(len(r0), ST_I[8]), 0.5 * inten[deck], 0.5 * inten[deck]]),
                        head=cat([(cls[~rail] >= 2) & (cls[~rail] <= 4), np.zeros(len(r0) + 2 * int(deck.sum()), bool)]))
        # --- footprints and their hatch
        pts, tau = z["b_pts"], z["b_tau"]
        nxt, ring = ring_of(z["b_off"])
        x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
        ring_area = np.abs(np.add.reduceat(x * y[nxt] - y * x[nxt], z["b_off"][:-1])) * 0.5e6        # m2
        b_area = np.zeros(len(z["b_h"]))
        np.add.at(b_area, z["b_ring_bld"], np.where(z["b_ring_inner"], -ring_area, ring_area))
        self._fp = dict(p0=pts, p1=pts[nxt], t0=tau, t1=tau[nxt], area=b_area[z["b_ring_bld"][ring]])
        self._hatch = {"coarse": (z["hc"], z["hc_tau"], b_area[z["hc_b"]]), "fine": (z["hf"], z["hf_tau"], b_area[z["hf_b"]])}
        # --- volumes
        pts = z["v_pts"]
        nxt, vol = ring_of(z["v_off"])
        npt = np.diff(z["v_off"])
        cen = np.add.reduceat(pts, z["v_off"][:-1], axis=0) / npt[:, None]
        x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
        v_area = np.abs(np.add.reduceat(x * y[nxt] - y * x[nxt], z["v_off"][:-1])) * 0.5e6
        lm = np.zeros(len(z["b_h"]), bool)                             # the buildings that have a label
        for lb in self._labels:
            if lb["kind"] == "landmark" and lb["rank"] <= 1:
                lm[lb["bld"]] = True
        # a corner edge stands where the footprint really turns (a curved facade gets one every few points)
        prv = np.empty_like(nxt)
        prv[nxt] = np.arange(len(nxt))
        d1, d2 = pts - pts[prv], pts[nxt] - pts
        turn = np.abs(np.arctan2(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0], d1[:, 0] * d2[:, 0] + d1[:, 1] * d2[:, 1]))
        k_in = np.arange(len(pts)) - z["v_off"][:-1][vol]
        corner = (turn > math.radians(CORNER)) | (k_in % np.maximum(np.ceil(npt / 6.0), 1).astype(int)[vol] == 0)
        # how close the corner edges of one building stand on a facade: its widest outline over all its edges
        per = np.add.reduceat(np.hypot(d2[:, 0], d2[:, 1]), z["v_off"][:-1])
        owner = np.where(z["v_bld"] >= 0, z["v_bld"], len(z["b_h"]) + np.arange(len(npt)))
        n_edge = np.bincount(owner, weights=np.add.reduceat(corner.astype(float), z["v_off"][:-1]))
        p_max = np.zeros(len(n_edge))
        np.maximum.at(p_max, owner, per)
        self._vol = dict(pts=pts, nxt=nxt, vol=vol, y0=z["v_y0"] / 1000.0, y1=z["v_y1"] / 1000.0, cen=cen,
                         d=np.hypot(cen[:, 0], cen[:, 1]), area=v_area, corner=corner, tau=z["v_tau"],
                         gap=(p_max / (math.pi * np.maximum(n_edge, 1.0)))[owner], own=owner,
                         lm=np.where(z["v_bld"] >= 0, lm[np.maximum(z["v_bld"], 0)], False))
        # --- water, parks
        p0, p1, t0, t1, _ = seg_of(z["wb_pts"], z["wb_tau"], z["wb_off"])
        self._bank, self._wh, self._wh_tau = (p0, p1, t0, t1), z["wh"], z["wh_tau"]
        self._piers = (z["piers"], z["piers_tau"])
        nxt, _ = ring_of(z["pk_off"])
        self._park = _dashes(z["pk_pts"], z["pk_pts"][nxt], z["pk_tau"], z["pk_tau"][nxt], 0.008, 0.008)

    def _tab(self, layer, lod):
        """The table of a layer at a level of detail (made the first time it is asked for)."""
        t = self._tabs.get((layer, lod))
        if t is None:
            q = dict(LOD[lod], **W_LOD.get(lod, {})) if E.WALL else LOD[lod]
            t = self._tabs[(layer, lod)] = getattr(self, "_make_" + layer)(q)
        return t

    def _make_streets(self, q):
        s = self._st
        m = (s["cls"] <= q["cls"]) | (s["cls"] == 8)               # the railways at every level: the bridges
        if E.WALL:                                                 # fewer streets, all of them at full level
            m = m & (s["i"] >= W_ST_MIN)
        # which pens show their head: always the same ones (one main street segment out of `stride`), few enough
        # that about HEADS of them are at work at any moment of the build
        head, lo, hi = s["head"][m].copy(), np.minimum(s["t0"][m], s["t1"][m]), np.maximum(s["t0"][m], s["t1"][m])
        busy = max(int(((lo[head] < r) & (hi[head] > r)).sum()) for r in np.linspace(0.05, float(hi.max()), 80))
        idx = np.nonzero(head)[0]
        head[idx[np.arange(len(idx)) % max(1, int(math.ceil(busy / HEADS))) != 0]] = False
        if E.WALL:
            return _Tab(_g3(s["p0"][m], 0.0), _g3(s["p1"][m], 0.0), 1.0, w=W_ST_W[s["cls"][m]], ta=s["t0"][m],
                        tb=s["t1"][m], head=head, box=self.extent, sz=W_ST_SZ[s["cls"][m]])
        return _Tab(_g3(s["p0"][m], 0.0), _g3(s["p1"][m], 0.0), s["i"][m], w=ST_W[s["cls"][m]], ta=s["t0"][m],
                    tb=s["t1"][m], head=head, box=self.extent)

    def _make_blocks(self, q):
        s = self._fp
        m = s["area"] >= q["area"]
        return _Tab(_g3(s["p0"][m], 0.0), _g3(s["p1"][m], 0.0), 1.0 if E.WALL else 0.5, ta=s["t0"][m] + LAG["blocks"],
                    tb=s["t1"][m] + LAG["blocks"], box=self.extent, w=E.ww(1.0),
                    sz=np.sqrt(s["area"][m]) / 1000.0 if E.WALL else None)

    def _make_hatch(self, q):
        h, tau, area = self._hatch[q["hatch"]]
        m = area >= q["area"]
        return _Tab(_g3(h[m, 0:2], 0.0), _g3(h[m, 2:4], 0.0), 0.2 if q["hatch"] == "coarse" else 0.26,
                    ta=tau[m, 0] + LAG["hatch"], tb=tau[m, 1] + LAG["hatch"], box=self.extent)

    def _make_buildings(self, q, only=None):
        v = self._vol
        keep = ((v["y1"] * 1000.0 >= q["vol_h"]) | (v["d"] < q["near"])) if only is None else only
        if q["radius"] is not None:
            keep = keep & (v["d"] < q["radius"] + 0.05)
        corner = v["corner"]
        if E.WALL:          # fewer volumes (not the small parts), and a few corner edges each: black stays between
            keep = keep & (v["area"] >= q.get("part", 0.0) * np.where(v["lm"], 0.5, 1.0))
            first = np.searchsorted(v["vol"], np.arange(len(v["y1"])))
            cs = np.cumsum(corner)
            rank = cs - 1 - (cs[first] - corner[first])[v["vol"]]
            n_c = np.bincount(v["vol"], weights=corner, minlength=len(v["y1"]))
            corner = corner & (rank % np.maximum(np.ceil(n_c / q.get("edges", 6)), 1).astype(int)[v["vol"]] == 0)
            if not q.get("plain", 1):
                corner = corner & v["lm"][v["vol"]]
                # ... and two volumes per building at most: its tallest one and its widest one
                pick = np.zeros(len(keep), bool)
                for key in (v["y1"], v["area"]):
                    o = np.lexsort((key, v["own"]))
                    pick[o[np.r_[v["own"][o][1:] != v["own"][o][:-1], True]]] = True
                keep = keep & pick
        k = keep[v["vol"]]
        pts, nxt, vol = v["pts"][k], v["pts"][v["nxt"]][k], v["vol"][k]
        y0, y1, cen, tau = v["y0"][vol], v["y1"][vol], v["cen"][vol], v["tau"][vol] + LAG["buildings"]
        strong = np.where(v["lm"][vol], 1.3, 1.0) * (0.55 + 0.45 * np.clip(y1 * 1000.0 / 110.0, 0.0, 1.0))
        n = len(pts)
        # the corner edges (dim at the ground, bright at the top), the roof outline, the ledge of a part that
        # starts above the ground
        co, up = corner[k], y0 > 0.0005
        a = [_g3(pts[co], y0[co]), _g3(pts, y1), _g3(pts[up], y0[up])]
        b = [_g3(pts[co], y1[co]), _g3(nxt, y1), _g3(nxt[up], y0[up])]
        i0 = [0.14 * strong[co], 0.85 * strong, 0.4 * strong[up]]
        i1 = [0.62 * strong[co], 0.85 * strong, 0.4 * strong[up]]
        w = [np.full(int(co.sum()), 1.0), np.where(v["lm"][vol], 1.5, 1.1), np.full(int(up.sum()), 1.0)]
        sp = [-v["gap"][vol][co], np.zeros(n), np.zeros(int(up.sum()))]
        sz = None
        if E.WALL:          # every line at full level; the towers with a label are heavier, not brighter; a roof
            #                 goes with the size of its building on screen (the labelled towers always stay)
            lv = np.where(v["lm"][vol], 1.0, W_PILE)
            i0 = i1 = [lv[co], lv, lv[up]]
            wl = np.where(v["lm"][vol], 2.4, E.WALL_LINE)
            w = [wl[co], wl, np.full(int(up.sum()), E.WALL_LINE)]
            size = np.where(v["lm"][vol], 0.0, np.maximum(np.sqrt(v["area"][vol]) / 1000.0, y1 - y0))
            sz = [size[co], size, size[up]]
            sp = [np.zeros(int(co.sum())), np.zeros(n), np.zeros(int(up.sum()))]
        oc, ot = [cen[co], cen, cen[up]], [tau[co], tau, tau[up]]
        # floor plates on what is tall
        step = q["floors"] / 1000.0
        fl = (y1 - y0 >= FLOOR_MIN_H / 1000.0) & (v["area"][vol] >= FLOOR_MIN_AREA)
        nf = np.where(fl, np.maximum(np.ceil((y1 - y0) / step) - 1, 0), 0).astype(int)
        rep = np.repeat(np.arange(n), nf)
        yf = y0[rep] + (np.arange(nf.sum()) - np.repeat(np.cumsum(nf) - nf, nf) + 1) * step
        a.append(_g3(pts[rep], yf)), b.append(_g3(nxt[rep], yf))
        if E.WALL:
            i0, i1 = i0 + [lv[rep]], i1 + [lv[rep]]
            sz.append(np.zeros(len(rep)))
        else:
            i0.append(0.3 * strong[rep]), i1.append(0.3 * strong[rep])
        w.append(np.full(len(rep), E.ww(1.0))), sp.append(np.full(len(rep), step)), oc.append(cen[rep]), ot.append(tau[rep])
        cat = np.concatenate
        return _Tab(cat(a), cat(b), cat(i0), cat(i1), cat(w), sp=cat(sp), oc=cat(oc), ot=cat(ot), box=self.extent,
                    sz=None if sz is None else cat(sz))

    def _make_landmarks(self, q):
        """Only the towers that have a label: for a view that wants the skyline alone."""
        return self._make_buildings(dict(q, radius=None), only=self._vol["lm"])

    def _make_river(self, q):
        (p0, p1, t0, t1), y, (pr, pt) = self._bank, self.RIVER_Y, self._piers
        nb, npr = len(p0), len(pr)
        # a pier goes down from its bridge to the water when the pen has passed above
        return _Tab(np.concatenate([_g3(p0, y), _g3(pr, 0.0)]), np.concatenate([_g3(p1, y), _g3(pr, y)]),
                    np.r_[np.full(nb, E.wl(0.6)), np.full(npr, E.wl(0.55))],
                    np.r_[np.full(nb, E.wl(0.6)), np.full(npr, 1.0 if E.WALL else 0.3)],
                    w=np.r_[np.full(nb, E.ww(1.3)), np.full(npr, E.ww(1.0))], ta=np.r_[t0, pt], tb=np.r_[t1, pt - y],
                    box=self.extent)

    def _make_parks(self, q):
        p0, p1, t0, t1 = self._park
        return _Tab(_g3(p0, 0.0), _g3(p1, 0.0), 1.0 if E.WALL else 0.24, ta=t0 + LAG["parks"], tb=t1 + LAG["parks"],
                    box=self.extent, w=E.ww(1.0), sz=W_PARK_SZ if E.WALL else None)

    def counts(self, lod="district"):
        """Segments per layer at a level of detail (built, before the camera cuts anything; `hatch` is only
        drawn when asked for and is not in the total)."""
        r = LOD[lod]["radius"]
        out = {k: self._tab(k, lod).upto(r) for k in ("river", "parks", "streets", "blocks", "buildings")}
        out["total"] = sum(out.values())
        out["hatch"] = self._tab("hatch", lod).upto(r)
        return out

    @staticmethod
    def _st_named(z):
        """The segments of the streets that have a name: (p0, p1, index of the name)."""
        off, name = z["st_off"], z["st_name"]
        first = np.ones(len(z["st_pts"]), bool)
        first[off[:-1]] = False
        b = np.nonzero(first)[0]
        nm = np.repeat(name, np.diff(off) - 1)
        keep = (nm >= 0) & (np.repeat(z["st_cls"], np.diff(off) - 1) != 1)
        return z["st_pts"][b - 1][keep], z["st_pts"][b][keep], nm[keep]

    # ------------------------------------------------------------------ helpers for a scene
    def at(self, x, z, street=0.012):
        """What stands at the model point (x, z) - for what a scene writes where something lands:
            dict(kind="building", name (or None), h (m), h_src ("tag" / "levels" / "default"), levels (or None))
            dict(kind="river", name)                    OHIO RIVER or LICKING RIVER
            dict(kind="street", name)                   within `street` km of the centre line of a named street
            dict(kind="ground")"""
        q, found = self._q, None
        for r in np.nonzero((q["lo"][:, 0] <= x) & (q["hi"][:, 0] >= x) & (q["lo"][:, 1] <= z) & (q["hi"][:, 1] >= z))[0]:
            P = q["pts"][q["off"][r]: q["off"][r + 1]]
            x0, z0, x1, z1 = P[:, 0], P[:, 1], np.roll(P[:, 0], -1), np.roll(P[:, 1], -1)
            with np.errstate(divide="ignore", invalid="ignore"):
                hit = ((z0 > z) != (z1 > z)) & (x < x0 + (z - z0) * (x1 - x0) / (z1 - z0))
            if hit.sum() % 2:
                if q["inner"][r]:                       # a courtyard: not on the building
                    found = None
                    break
                found = int(q["bld"][r]) if found is None else found
        if found is not None:
            return dict(kind="building", name=q["names"].get(found), h=round(float(q["h"][found]), 1),
                        h_src=("tag", "levels", "default")[int(q["src"][found])], levels=int(q["levels"][found]) or None)
        h = self._wh
        wet = np.nonzero((np.abs(h[:, 1] - z) < 0.012) & (h[:, 0] <= x) & (h[:, 2] >= x))[0] if len(h) else ()
        if len(wet):
            return dict(kind="river", name=("OHIO RIVER", "LICKING RIVER")[int(q["river"][wet[0]])])
        p0, p1, nm = q["st"]
        d = p1 - p0
        u = np.clip(((x - p0[:, 0]) * d[:, 0] + (z - p0[:, 1]) * d[:, 1]) / np.maximum((d ** 2).sum(1), 1e-12), 0.0, 1.0)
        dist = np.hypot(p0[:, 0] + u * d[:, 0] - x, p0[:, 1] + u * d[:, 1] - z)
        j = int(np.argmin(dist))
        if dist[j] < street:
            return dict(kind="street", name=q["st_names"][int(nm[j])])
        return dict(kind="ground")

    @staticmethod
    def lod_for(px_per_km):
        """The level of detail that suits a scale on screen (pixels per kilometre of the model)."""
        return "basin" if px_per_km < 650.0 else "district" if px_per_km < 2600.0 else "block"

    @staticmethod
    def xf(offset=(0.0, 0.0, 0.0), scale=1.0, yaw=0.0, yscale=1.0, pivot=(0.0, 0.0, 0.0)):
        """world = offset + scale * turn(yaw) * (model - pivot), heights times yscale. yaw in degrees, clockwise
        seen from above (yaw=City.GRID_YAW squares the street grid with the axes: Vine Street towards -z)."""
        return Xf(offset, scale, yaw, yscale, pivot)

    @staticmethod
    def plan_camera(center, px_per_km, at=(0.0, 0.0)):
        """A camera looking straight down, north up: the world point (at[0], 0, at[1]) lands on `center` (px)."""
        return OrthoCamera((at[0], 10.0, at[1]), (at[0], 0.0, at[1]), px_per_km, up=(0.0, 0.0, -1.0), screen_center=center)

    @staticmethod
    def screen(cam, pos):
        """Screen position of a world point: (x, y, in front of the camera)."""
        sx, sy, _, ok = cam.project(np.asarray(pos, np.float32).reshape(1, 3))
        return float(sx[0]), float(sy[0]), bool(ok[0])

    def labels(self, kinds=None, rank=1, radius=None, xf=None, per_name=1):
        """The names of the model, nearest to the site first (see the top of the module).
        kinds = tuple of kinds (None = all), rank = the highest rank kept, radius = km from the site,
        per_name = how many anchors of the same street."""
        xf = xf or _ID
        out, seen = [], {}
        for lb in self._labels:
            if (kinds and lb["kind"] not in kinds) or lb["rank"] > rank or (radius is not None and lb["d"] > radius):
                continue
            n = seen.get((lb["kind"], lb["name"]), 0)
            if n >= per_name:
                continue
            seen[(lb["kind"], lb["name"])] = n + 1
            out.append(Label(lb["name"], lb["short"], lb["kind"], lb["rank"],
                             xf(np.array([lb["x"], lb["y"], lb["z"]], np.float32)), lb["d"], lb["tau"], lb.get("brg"),
                             lb.get("h"), lb.get("h_src"), lb.get("levels"), lb.get("info")))
        return out

    def reach(self, lod="district", radius=None):
        """km of streets the pens cover before the model of this level is complete."""
        r = LOD[lod]["radius"]
        radius = r if radius is None else (radius if r is None else min(radius, r))
        v = self._reach.get((lod, radius))
        if v is None:
            tabs = [self._tab(k, lod) for k in ("river", "parks", "streets", "blocks", "hatch", "buildings")]
            v = self._reach[(lod, radius)] = max(max(t.last(t.upto(radius)) for t in tabs), 1e-3)
        return v

    def arrival(self, tau, wave=2.0, lod="district", radius=None):
        """Seconds after the start of a build of `wave` seconds at which the pens have covered tau km of
        streets (Label.tau): when to make the tag of a label."""
        return wave * (1.0 - math.sqrt(max(0.0, 1.0 - min(tau / self.reach(lod, radius), 1.0))))

    # ------------------------------------------------------------------ drawing
    def _args(self, lod, xf, age, wave, reveal, centre, radius):
        """(xf, radius, front): front = None when everything is there, False when nothing is yet."""
        r = LOD[lod]["radius"]
        radius = r if radius is None else (radius if r is None else min(radius, r))
        xf = xf or _ID
        if reveal is not None:
            far = (radius if radius is not None else self.r_max) + (0.0 if centre is None else math.hypot(*centre))
            if reveal >= far + RISE:
                return xf, radius, None
            return xf, radius, (False if reveal <= 0.0 else _Front(False, reveal, centre=centre))
        if age is None or age >= wave + RISE_T:
            return xf, radius, None
        if age <= 0.0:
            return xf, radius, False
        reach = self.reach(lod, radius)
        p = min(age / max(wave, 1e-6), 1.0)
        return xf, radius, _Front(True, reach * p * (2.0 - p), reach, age, max(wave, 1e-6))

    def _emit(self, f, cam, tab, layer, xf, radius, gain, front, view, fog, heads=False):
        """Draw a table: what lies within the radius, the build, the transform, the camera, the intensity."""
        n = tab.upto(radius)
        if n <= 0 or gain <= 0.0 or front is False:
            return
        sel = slice(0, n)                               # what is drawn: a prefix (sorted by distance) or a list
        hx = None
        if front is None:
            A, Bp, i0, i1 = tab.a[sel], tab.b[sel], tab.i0[sel], tab.i1[sel]
        elif tab.oc is not None:                        # buildings: each rises as one piece
            if front.timed:
                g = front.rise(tab.ot[:n])
            else:
                g = front.rise(np.hypot(tab.oc[:n, 0] - front.centre[0], tab.oc[:n, 1] - front.centre[1]))
            sel = np.nonzero(g > 0.0)[0]
            if not len(sel):
                return
            g = g[sel]
            A, Bp, i0, i1 = tab.a[sel], tab.b[sel], tab.i0[sel], tab.i1[sel]
            A[:, 1] *= g
            Bp[:, 1] *= g
        else:                                           # lines: drawn by a pen, from the end it reaches first
            if front.timed:
                ta, tb = tab.ta[:n], tab.tb[:n]
                sel = np.nonzero(ta < front.r)[0]
                ta, tb = ta[sel], tb[sel]
                A, Bp, i0, i1 = tab.a[sel], tab.b[sel], tab.i0[sel], tab.i1[sel]
            else:
                c = front.centre
                ta = np.hypot(tab.a[:n, 0] - c[0], tab.a[:n, 2] - c[1])
                tb = np.hypot(tab.b[:n, 0] - c[0], tab.b[:n, 2] - c[1])
                sel = np.nonzero(np.minimum(ta, tb) < front.r)[0]
                ta, tb = ta[sel], tb[sel]
                sw = tb < ta
                A, Bp = np.where(sw[:, None], tab.b[sel], tab.a[sel]), np.where(sw[:, None], tab.a[sel], tab.b[sel])
                i0, i1 = np.where(sw, tab.i1[sel], tab.i0[sel]), np.where(sw, tab.i0[sel], tab.i1[sel])
                ta, tb = np.minimum(ta, tb), np.maximum(ta, tb)
            if not len(sel):
                return
            cut = tb > front.r
            if cut.any():
                u = ((front.r - ta[cut]) / np.maximum(tb[cut] - ta[cut], 1e-9)).astype(np.float32)
                Bp[cut] = A[cut] + (Bp[cut] - A[cut]) * u[:, None]
                i1[cut] = i0[cut] + (i1[cut] - i0[cut]) * u
                if heads and tab.head is not None:
                    hx = Bp[np.nonzero(cut & tab.head[sel])[0][: 4 * HEADS]]
        w = tab.w[sel]
        wall, far = E.WALL, None                        # far: what the wall rule turns into a short way out
        if radius is not None:                          # the model does not end on a hard edge
            edge = 1.0 - np.clip((np.hypot(A[:, 0], A[:, 2]) - (1.0 - EDGE) * radius) / (EDGE * radius), 0.0, 1.0)
            if wall:
                far = edge
            else:
                i0, i1 = i0 * edge, i1 * edge
        x0, y0, x1, y1, depth, keep = _project(cam, xf(A), xf(Bp), view)
        fac = None
        if tab.sp is not None or (wall and tab.sz is not None):
            px = (cam.scale if depth is None else cam.focal / depth) * xf.scale
        if tab.sp is not None:                          # lines that would fill thin out: a tower never burns white
            sp = tab.sp[sel]
            px = (cam.scale if depth is None else cam.focal / depth) * xf.scale
            sv = math.sqrt(max(0.0, 1.0 - float(cam.R[2][1]) ** 2))        # seen from above, the plates close up
            if wall:        # lines that would fill are not there (they leave over a few px of spacing): no dimming
                fac = np.where(sp > 0.0, _band(sp * xf.yscale * sv * px, W_PLATE_PX),
                               np.where(sp < 0.0, _band(-sp * px, W_EDGE_PX), 1.0))
            else:
                fac = np.where(sp > 0.0, np.clip((sp * xf.yscale * sv * px - 0.5) / (PLATE_PX - 0.5), 0.0, 1.0),
                               np.where(sp < 0.0, np.clip((-sp * px - 0.5) / (WALL_PX - 0.5), 0.12, 1.0), 1.0))
        if wall and tab.sz is not None:                 # what is too small on screen is not there
            sz = tab.sz[sel]
            fs = np.where(sz > 0.0, _band(sz * px, W_SIZE_PX), 1.0)
            fac = fs if fac is None else fac * fs
        if fog is not None and depth is not None:
            lo = fog[2] if len(fog) > 2 else 0.12
            fz = lo + (1.0 - lo) * np.clip((fog[1] - depth) / max(fog[1] - fog[0], 1e-6), 0.0, 1.0)
            if wall:
                far = fz if far is None else far * fz
            else:
                fac = fz if fac is None else fac * fz
        if wall:
            if far is not None:
                far = _band(far, W_FAR)
                fac = far if fac is None else fac * far
            gain, w = min(gain, 1.0), np.maximum(w, E.WALL_LINE)
            if fac is not None:                         # what is not there is not sent
                live = np.broadcast_to(fac > 0.004, i0.shape)
                keep = live if keep is None else keep & live
        if fac is not None:
            i0, i1 = i0 * fac, i1 * fac
        if keep is not None:
            if not keep.any():
                return
            x0, y0, x1, y1, i0, i1, w = x0[keep], y0[keep], x1[keep], y1[keep], i0[keep], i1[keep], w[keep]
        f.segments(layer, x0, y0, x1, y1, i0 * gain, i1 * gain, width=w)
        if hx is not None and len(hx):
            sx, sy, _, ok = cam.project(xf(hx))
            f.dots("w", sx[ok], sy[ok], 2.6, 1.5 * gain)

    def draw(self, f, cam, xf=None, lod="district", layers=LAYERS, age=None, wave=2.0, gain=1.0, t=0.0, view=None,
             fog=None, hatch=False, reveal=None, centre=None, radius=None, site=None):
        """The model through a camera: the layers of `layers`, in order (see the top of the module).
        view = (x0, y0, x1, y1) px: what lies outside is not sent to the frame (default: the canvas).
        fog = (near, far[, floor]) in world units from the camera: lines dim with the distance (perspective).
        hatch = True (or a gain): the scan lines through the footprints. site = arguments for draw_site."""
        kw = dict(xf=xf, lod=lod, age=age, wave=wave, gain=gain, view=view, fog=fog, reveal=reveal, centre=centre,
                  radius=radius)
        for name in layers:
            if name == "river":
                self.draw_river(f, cam, t=t, **kw)
            elif name == "parks":
                self.draw_parks(f, cam, **kw)
            elif name == "streets":
                self.draw_streets(f, cam, **kw)
            elif name == "blocks":
                self.draw_blocks(f, cam, hatch=hatch, **kw)
            elif name == "buildings":
                self.draw_buildings(f, cam, **kw)
            elif name == "landmarks":
                self.draw_buildings(f, cam, only_landmarks=True, **kw)
            elif name == "site":
                self.draw_site(f, cam, xf=xf, t=t, age=age, view=view, **dict(dict(gain=gain), **(site or {})))

    def draw_streets(self, f, cam, xf=None, lod="district", age=None, wave=2.0, gain=1.0, view=None, fog=None,
                     reveal=None, centre=None, radius=None, heads=True):
        xf, radius, front = self._args(lod, xf, age, wave, reveal, centre, radius)
        self._emit(f, cam, self._tab("streets", lod), "w", xf, radius, gain, front, view, fog, heads=heads)

    def draw_blocks(self, f, cam, xf=None, lod="district", age=None, wave=2.0, gain=1.0, view=None, fog=None,
                    reveal=None, centre=None, radius=None, hatch=False):
        """Footprints on the ground; hatch=True (or a gain) adds the scan lines through them."""
        xf, radius, front = self._args(lod, xf, age, wave, reveal, centre, radius)
        self._emit(f, cam, self._tab("blocks", lod), "w", xf, radius, gain, front, view, fog)
        if hatch and not E.WALL:                        # (scan lines at 0.2: a grey that the wall does not show)
            self._emit(f, cam, self._tab("hatch", lod), "w", xf, radius, gain * float(hatch), front, view, fog)

    def draw_buildings(self, f, cam, xf=None, lod="district", age=None, wave=2.0, gain=1.0, view=None, fog=None,
                       reveal=None, centre=None, radius=None, only_landmarks=False):
        """The volumes. only_landmarks: just the towers that have a label (the skyline alone)."""
        xf, radius, front = self._args(lod, xf, age, wave, reveal, centre, radius)
        self._emit(f, cam, self._tab("landmarks" if only_landmarks else "buildings", lod), "w", xf, radius, gain,
                   front, view, fog)

    def draw_parks(self, f, cam, xf=None, lod="district", age=None, wave=2.0, gain=1.0, view=None, fog=None,
                   reveal=None, centre=None, radius=None):
        xf, radius, front = self._args(lod, xf, age, wave, reveal, centre, radius)
        self._emit(f, cam, self._tab("parks", lod), "w", xf, radius, gain, front, view, fog)

    def draw_river(self, f, cam, xf=None, lod="district", age=None, wave=2.0, gain=1.0, t=0.0, view=None, fog=None,
                   reveal=None, centre=None, radius=None, flow=True):
        """Banks, piers, and the scan lines of the water: dashes that drift west with `t` (flow=False: still)."""
        xf, radius, front = self._args(lod, xf, age, wave, reveal, centre, radius)
        if front is False or gain <= 0.0:
            return
        self._emit(f, cam, self._tab("river", lod), "w", xf, radius, gain, front, view, fog)
        h, tau = self._wh, self._wh_tau
        if E.WALL:                                                # one scan line in three, at full level
            h, tau = h[::3], tau[::3]
        if not len(h):
            return
        on, per = 0.09, 0.15                                      # km: a dash, a period
        ph = ((FLOW * t if flow else 0.0) + 0.37 * per * (np.arange(len(h)) % 5)) % per
        nd = np.ceil((h[:, 2] - h[:, 0]) / per).astype(int) + 2
        seg = np.repeat(np.arange(len(h)), nd)
        k = np.arange(nd.sum()) - np.repeat(np.cumsum(nd) - nd, nd)
        xb = h[seg, 2] + per - ph[seg] - k * per                  # dashes come in at the east end and move west
        xa, xb = np.maximum(xb - on, h[seg, 0]), np.minimum(xb, h[seg, 2])
        m = xb > xa
        xa, xb, seg = xa[m], xb[m], seg[m]
        zz, xm = h[seg, 1], 0.5 * (xa + xb)
        e = self.extent
        i = np.clip(np.minimum(np.minimum(xm - e[0], e[2] - xm), np.minimum(zz - e[1], e[3] - zz)) / BOX_EDGE, 0.0, 1.0)
        if radius is not None:
            i = i * (1.0 - np.clip((np.hypot(xm, zz) - (1.0 - EDGE) * radius) / (EDGE * radius), 0.0, 1.0))
        i = _band(i, W_FAR) if E.WALL else 0.3 * i
        if front is not None:                                     # the water is laid down behind the pens
            if front.timed:
                u = (xm - h[seg, 0]) / np.maximum(h[seg, 2] - h[seg, 0], 1e-9)
                i = i * front.laid(tau[seg, 0] + (tau[seg, 1] - tau[seg, 0]) * u)
            else:
                i = i * front.laid(np.hypot(xm - front.centre[0], zz - front.centre[1]))
        live = i > 0.004
        if not live.any():
            return
        a, b = _g3(np.stack([xa[live], zz[live]], 1), self.RIVER_Y), _g3(np.stack([xb[live], zz[live]], 1), self.RIVER_Y)
        x0, y0, x1, y1, depth, keep = _project(cam, xf(a), xf(b), view)
        i = i[live].astype(np.float32)
        if fog is not None and depth is not None:
            lo = fog[2] if len(fog) > 2 else 0.12
            fz = lo + (1.0 - lo) * np.clip((fog[1] - depth) / max(fog[1] - fog[0], 1e-6), 0.0, 1.0)
            i = i * (_band(fz, W_FAR) if E.WALL else fz)
        if E.WALL:                                                # from far away the water is its banks
            i = i * _band(W_WATER_SZ * (cam.scale if depth is None else cam.focal / depth) * xf.scale, W_SIZE_PX)
        if keep is not None:
            x0, y0, x1, y1, i = x0[keep], y0[keep], x1[keep], y1[keep], i[keep]
        f.segments("w", x0, y0, x1, y1, i * (min(gain, 1.0) if E.WALL else gain), width=E.ww(1.0))

    def draw_site(self, f, cam, xf=None, t=0.0, age=None, gain=1.0, cross=0.07, rings=(0.1, 0.25, 0.5), pole=0.09,
                  wall=True, towers=True, pulse=0.3, view=None, white=None):
        """The site, in red: the two streets over `cross` km each way, a pole of `pole` km over the corner with a
        dot at its head, range rings on the ground (white, radii in km), the wall with its three towers (white)
        and their detectors (red points), a ring that leaves the wall every PULSE_T s and dies at `pulse` km.
        white = the gain of the white lines when it is not `gain` (the wall rule: the red keeps its level, the
        white is not dimmed with it).
        A part is left out with 0 / False / (). age: the marker is made first, in 0.7 s (None = built)."""
        if (age is not None and age <= 0.0) or gain <= 0.0:
            return
        xf = xf or _ID
        p = 1.0 if age is None else float(B.ease(age / 0.7))

        def lines(layer, a, b, i, width=1.6):
            x0, y0, x1, y1, _, keep = _project(cam, xf(np.asarray(a, np.float32).reshape(-1, 3)),
                                               xf(np.asarray(b, np.float32).reshape(-1, 3)), view)
            if keep is not None:
                x0, y0, x1, y1 = x0[keep], y0[keep], x1[keep], y1[keep]
            f.segments(layer, x0, y0, x1, y1, i * (gain if layer == "r" or white is None else white), width=width)

        def points(layer, P, r, i):
            sx, sy, _, ok = cam.project(xf(np.asarray(P, np.float32).reshape(-1, 3)))
            if view is not None:
                ok = ok & (sx >= view[0]) & (sx <= view[2]) & (sy >= view[1]) & (sy <= view[3])
            f.dots(layer, sx[ok], sy[ok], r, i * (gain if layer == "r" or white is None else white))

        if cross:
            e = np.array([self.axis_9th, -self.axis_9th, self.axis_vine, -self.axis_vine], np.float32) * cross * p
            lines("r", np.zeros((4, 3), np.float32), _g3(e, 0.0), 1.5, width=2.4)
            points("r", [0.0, 0.0, 0.0], 5.0, 1.7)
        if pole:
            lines("r", [0.0, 0.0, 0.0], [0.0, pole * p, 0.0], 1.3, width=2.0)
            points("r", [0.0, pole * p, 0.0], 4.2, 1.7)
            points("w", [0.0, pole * p, 0.0], 1.8, 1.2)
        nv = 128
        for k, r in enumerate(rings):                             # traced from the north, as the show traces a ring
            full = int(math.floor(nv * p + 1e-9))                 # (the vertices stay where they are: the end moves)
            a = -0.5 * math.pi + 2.0 * math.pi * np.r_[np.arange(full + 1) / nv, [p] if full < nv else []]
            ring = _g3(np.stack([r * np.cos(a), r * np.sin(a)], 1), 0.0)
            lines("w", ring[:-1], ring[1:], E.wl(max(0.34 - 0.06 * k, 0.16)), width=E.ww(1.0))
        w = self.wall
        if w is None or not wall:
            return
        a3, b3 = _g3(w["a"][None], 0.0)[0], _g3(w["b"][None], 0.0)[0]
        up = np.array([0.0, w["height"] * p, 0.0], np.float32)
        lines("r", [a3, a3 + up, b3 + up, b3], [a3 + up, b3 + up, b3, a3], 1.6, width=2.2)
        if towers:
            foot = _g3(np.array([q for q, _ in w["towers"]]), 0.0)
            head = foot.copy()
            head[:, 1] = np.array([h for _, h in w["towers"]]) * p
            lines("w", foot, head, 1.2, width=1.6)
            points("r", head, 2.6, 1.8)
        if pulse and p >= 1.0:
            ph = (t / PULSE_T) % 1.0
            a = np.linspace(0.0, 2.0 * math.pi, nv + 1)
            ring = _g3(w["mid"] + np.stack([np.cos(a), np.sin(a)], 1) * (pulse * ph), 0.0)
            lines("r", ring[:-1], ring[1:], 1.1 * (1.0 - ph) ** 1.6, width=1.4)

    def scale_bar(self, f, cam, at, length=0.1, ticks=10, xf=None, along=None, layer="w", inten=0.9, tick=0.006):
        """A ruler on the ground: from `at` = (x, z) (model km) over `length` km along `along` (unit (x, z),
        default: along 9th Street), with `ticks` divisions (every fifth is longer). Returns the screen positions
        of its two ends: the scene writes the figures."""
        xf = xf or _ID
        u = self.axis_9th if along is None else np.asarray(along, np.float32)
        nrm = np.array([-u[1], u[0]], np.float32)
        at = np.asarray(at, np.float32)
        base = at + u * np.linspace(0.0, length, ticks + 1)[:, None]
        ends = np.where((np.arange(ticks + 1) % 5 == 0)[:, None], 1.6, 1.0) * tick
        a = np.concatenate([_g3(at[None], 0.0), _g3(base, 0.0)])
        b = np.concatenate([_g3((at + u * length)[None], 0.0), _g3(base + nrm * ends, 0.0)])
        x0, y0, x1, y1, _, keep = _project(cam, xf(a), xf(b), None)
        if keep is not None:
            x0, y0, x1, y1 = x0[keep], y0[keep], x1[keep], y1[keep]
        f.segments(layer, x0, y0, x1, y1, inten, width=E.ww(1.0))
        sx, sy, _, _ = cam.project(xf(_g3(np.stack([at, at + u * length]), 0.0)))
        return (float(sx[0]), float(sy[0])), (float(sx[1]), float(sy[1]))


_CITY = []


def get():
    """The city, loaded once per process."""
    if not _CITY:
        _CITY.append(City())
    return _CITY[0]
