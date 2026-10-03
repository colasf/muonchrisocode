"""Bake data/cincinnati.npz: downtown Cincinnati around 9th Street x Vine Street, where the wall of the show is.

Source: OpenStreetMap, through the Overpass API (https://overpass-api.de). The data is (C) OpenStreetMap
contributors and is made available under the Open Database License (ODbL, https://www.openstreetmap.org/copyright):
whoever shows something made from it has to say so. The line to set where the city is shown is in the baked
file (meta["attribution"]) and in muonbloom/city.py (City.ATTRIBUTION): MAP DATA (C) OPENSTREETMAP CONTRIBUTORS.

The show never needs the network: this tool is run once, the baked file travels with the project.

  python tools/build_city.py                  bake data/cincinnati.npz (downloads the box once, kept in data/cache)
  python tools/build_city.py --refresh        download again (the map has changed)
  python tools/build_city.py --preview x.png  also draw a plan of what was baked

What is in the box (BBOX: 3.5 km east-west, 3.6 km north-south, from Over-the-Rhine to the Kentucky bank):

  streets     every road with a class (motorway ... residential, pedestrian, named service lanes) and the
              railways, as polylines, with their name, class, bridge / tunnel flags
  buildings   every building footprint (outer and inner rings) with a height:
                 0 = its `height` tag (or that of its parts), 1 = its `building:levels` tag times a storey
                 height (LEVEL_M), 2 = a default for its kind. Only 0 is a surveyed figure; the others are
                 estimates, good for a skyline. Their names and their numbers of levels, where the map has them.
  volumes     what is extruded: the `building:part` volumes (base and top heights: the set-backs of the towers)
              and the footprints of the buildings that have no parts
  hatch       scan lines through the footprints, parallel to the numbered streets (coarse: every HATCH_COARSE m
              over the whole box; fine: every HATCH_FINE m within FINE_R km of the site)
  water       the Ohio River and the Licking River, clipped to the box: banks, scan lines, and where the piers
              of the bridges stand
  parks       outlines of the named parks
  labels      street names (anchors along each street, nearest to the site first), landmarks (name, height),
              bridges, rivers, places, the site
  *_tau       when the model is being made, pens leave the corner and run along the streets: the `tau` of a
              point is the length (km) of the shortest way to it through the street network (Dijkstra from
              the node of the corner); off the streets - footprints, hatch, banks, parks - a pen goes on
              from the nearest street at DETOUR times the straight distance. A building rises when its
              footprint is closed (b_done, v_tau)
  site        the corner (origin of the coordinates) and the wall, see below

Coordinates of the baked file (those of muonbloom/city.py and of the shower world): kilometres, x = east,
z = SOUTH (north is -z: the plan views of the show look down with -z up the wall), y = up. The origin is the node
shared by Vine Street and 9th Street in OpenStreetMap (SITE). The ground is a plane (the basin really rises about
25 m from the river front to Central Parkway; the hills around it are not in the box).

The wall. The map of the festival (https://www.blinkcincinnati.com/map/muon-bloom, read on 2026-10-03) puts
"muon : bloom" at BLINK_MARKER: 28 m west and 18 m north of the corner, inside the footprint of 905 Vine Street
(OpenStreetMap way 340772332), the building on the north-west corner. The south face of that footprint, along
9th Street, is 27.7 m long - the width of the wall in the site drawing (27.737 m) - and stands 23 m back from
the centre line of 9th Street: room for the projectors of that drawing, 12.8 m in front of the wall. The baked
`wall` is that face. It is an inference from these three facts, not something the festival wrote down: to be
confirmed by somebody who has stood there. If the footprint is not found any more, no wall is baked and the
site is only the corner. The height of the wall, the three towers and their distance from it are the figures of
the site drawing (estimates), not of the map.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "cincinnati.npz"
RAW = ROOT / "data" / "cache" / "cincinnati_osm.json"

BBOX = (39.086, -84.534, 39.118, -84.494)          # south, west, north, east
SITE = (39.105381, -84.514042)                     # Vine Street x 9th Street (OSM node 197417418)
BLINK_MARKER = (39.1055431, -84.51436585)          # "muon : bloom" on the map of the festival
WALL_WAY = 340772332                               # 905 Vine Street: its long south face is the wall
WALL_W, WALL_H = 27.737, 13.1                      # m: site drawing; height to the parapet from the stage photo
TOWERS = ((4.623, 3.448), (13.868, 6.496), (23.114, 3.448))    # m along the wall from its left end, top of the box
TOWER_OFF = 1.8                                    # m in front of the wall (assumed in the site notes)
ATTRIBUTION = "MAP DATA (C) OPENSTREETMAP CONTRIBUTORS"

LEVEL_M = 3.9                                      # storey height when only the number of levels is known
LEVEL_M_LOW = 3.4                                  # ... for buildings under 10 levels (+ PARAPET_M)
PARAPET_M = 1.0
DEFAULT_H = {"house": 7.0, "residential": 9.0, "apartments": 12.0, "garage": 3.0, "garages": 3.0, "shed": 2.5,
             "roof": 4.0, "church": 16.0, "industrial": 8.0, "warehouse": 8.0, "retail": 6.0, "commercial": 9.0,
             "office": 12.0, "parking": 9.0, "hotel": 15.0, "school": 10.0, "stadium": 24.0}
DEFAULT_H_ANY = 8.0
MIN_AREA = 12.0                                    # m2: smaller footprints are left out
SIMPLIFY_RING, SIMPLIFY_LINE = 0.4, 0.8            # m, Douglas-Peucker
HATCH_COARSE, HATCH_FINE, FINE_R = 6.0, 1.5, 0.32  # m, m, km
HATCH_WATER = 22.0                                 # m between the scan lines of the rivers
RIVER_Y = -0.028                                   # km: the pool of the river, about 28 m under the street grid
SITE_NODE = 197417418                              # the node Vine Street and 9th Street share
DETOUR = 1.3                                       # off the streets a pen covers the straight distance times this
PIER_STEP = 0.075                                  # km between the piers drawn under a bridge
LANDMARK_H = 70.0                                  # m: a named building this tall is a landmark
# published heights that the map does not have (or has lower: the top part is not drawn). PNC Tower: 495 ft.
HEIGHTS = {"PNC Tower": 150.9}

# class of a street: the order of drawing importance
CLASSES = {"motorway": 0, "trunk": 0, "motorway_link": 1, "trunk_link": 1, "primary": 2, "primary_link": 2,
           "secondary": 3, "secondary_link": 3, "tertiary": 4, "tertiary_link": 4, "residential": 5,
           "unclassified": 5, "living_street": 5, "pedestrian": 6, "service": 7}
RAIL = 8
# buildings that are named on the wall even though they are not tall (the name as OpenStreetMap writes it)
NOTABLE = ("Kroger Building", "Downtown Main Library", "Cincinnati Music Hall", "Hamilton County Courthouse",
           "Cincinnati City Hall", "Aronoff Center", "Contemporary Arts Center", "Taft Theatre", "Findlay Market",
           "Great American Ball Park", "Paycor Stadium", "Heritage Bank Center", "TQL Stadium",
           "Duke Energy Convention Center", "Saint Peter in Chains", "Plum Street Temple", "Memorial Hall")
# places written on the map (name, latitude, longitude). Fountain Square: where OpenStreetMap has its fountain
PLACES = (("CINCINNATI // OHIO", 39.1003, -84.5165), ("COVINGTON // KENTUCKY", 39.0885, -84.5125),
          ("NEWPORT // KENTUCKY", 39.0905, -84.4975), ("OVER-THE-RHINE", 39.1120, -84.5160),
          ("FOUNTAIN SQUARE", 39.10162, -84.51246))

ENDPOINTS = ("https://lz4.overpass-api.de/api/interpreter", "https://overpass-api.de/api/interpreter",
             "https://z.overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
_B = "{},{},{},{}".format(*BBOX)
Q_MAIN = f"""[out:json][timeout:170][bbox:{_B}];
(
  way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|residential|unclassified|living_street|pedestrian)$"];
  way["highway"="service"]["name"];
  way["building"];
  relation["building"];
  way["building:part"];
  relation["building:part"];
  way["leisure"="park"]["name"];
  relation["leisure"="park"]["name"];
  way["railway"="rail"];
);
out body geom({_B});
"""
Q_WATER = f"""[out:json][timeout:120];
(
  way["natural"="water"]["water"="river"]({_B});
  relation["natural"="water"]["water"="river"]({_B});
);
out body geom;
"""


# ----------------------------------------------------------------------------
# download
# ----------------------------------------------------------------------------

def overpass(query):
    last = None
    for attempt in range(3 * len(ENDPOINTS)):
        url = ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": query}).encode(),
                                         headers={"User-Agent": "muonbloom-build-city/1.0", "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=200) as r:
                d = json.loads(r.read())
            print(f"  {url}: {len(d['elements'])} elements", flush=True)
            return d
        except Exception as e:                      # a busy server answers 429 / 504: try the next one
            last = e
            print(f"  {url}: {type(e).__name__} {str(e)[:80]}", flush=True)
            time.sleep(4.0)
    raise SystemExit(f"Overpass did not answer ({last}). Try again later, or keep the cached download.")


def fetch(refresh=False):
    if RAW.exists() and not refresh:
        return json.loads(RAW.read_text(encoding="utf-8"))
    print("downloading from OpenStreetMap (Overpass API)", flush=True)
    raw = {"main": overpass(Q_MAIN), "water": overpass(Q_WATER)}
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps(raw), encoding="utf-8")
    return raw


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------

def km_per_deg(lat):
    p = math.radians(lat)
    return ((111412.84 * math.cos(p) - 93.5 * math.cos(3 * p) + 0.118 * math.cos(5 * p)) / 1000.0,
            (111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p)) / 1000.0)


KX, KY = km_per_deg(SITE[0])
X0, X1 = (BBOX[1] - SITE[1]) * KX, (BBOX[3] - SITE[1]) * KX            # west, east (km)
Z0, Z1 = -(BBOX[2] - SITE[0]) * KY, -(BBOX[0] - SITE[0]) * KY          # north, south (km): z = south


def xz(lat, lon):
    return ((lon - SITE[1]) * KX, -(lat - SITE[0]) * KY)


def pieces(geom):
    """The runs of a way geometry inside the box (Overpass gives null for the nodes it has cut away)."""
    out, cur = [], []
    for p in geom:
        if p is None:
            if len(cur) > 1:
                out.append(cur)
            cur = []
        else:
            cur.append((p["lat"], p["lon"]))
    if len(cur) > 1:
        out.append(cur)
    return out


def assemble(ways):
    """Closed rings out of the member ways of a multipolygon, joined end to end."""
    segs = [list(w) for w in ways if len(w) >= 2]
    rings = []
    while segs:
        cur = segs.pop(0)
        while cur[0] != cur[-1]:
            for k, s in enumerate(segs):
                if s[0] == cur[-1]:
                    cur = cur + s[1:]
                elif s[-1] == cur[-1]:
                    cur = cur + s[-2::-1]
                elif s[-1] == cur[0]:
                    cur = s[:-1] + cur
                elif s[0] == cur[0]:
                    cur = s[:0:-1] + cur
                else:
                    continue
                segs.pop(k)
                break
            else:
                cur = None
                break
        if cur and len(cur) >= 4:
            rings.append(cur)
    return rings


def simplify(P, tol, closed=False):
    """Douglas-Peucker on an (n, 2) array (km), tolerance in km. A closed ring comes without its repeated point."""
    P = np.asarray(P, np.float64)
    if closed:
        if len(P) < 4:
            return P
        k = int(np.argmax(((P - P[0]) ** 2).sum(1)))            # split at the two points furthest apart
        a = simplify(np.vstack([P[: k + 1]]), tol)
        b = simplify(np.vstack([P[k:], P[:1]]), tol)
        return np.vstack([a[:-1], b[:-1]])
    return P[kept(P, tol)]


def kept(P, tol):
    """Douglas-Peucker: which points of an open polyline stay."""
    keep = np.zeros(len(P), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        d = P[j] - P[i]
        n = math.hypot(*d)
        q = P[i + 1: j] - P[i]
        dist = np.abs(q[:, 0] * d[1] - q[:, 1] * d[0]) / n if n > 1e-12 else np.hypot(q[:, 0], q[:, 1])
        m = int(np.argmax(dist))
        if dist[m] > tol:
            keep[i + 1 + m] = True
            stack += [(i, i + 1 + m), (i + 1 + m, j)]
    return keep


def dijkstra(graph, src):
    """Shortest way from src to every node of {node: [(node, length)]}."""
    import heapq
    dist, heap = {src: 0.0}, [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, ()):
            if d + w < dist.get(v, math.inf):
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return dist


def arrive(Q, pool, pool_tau, chunk=1500):
    """tau of the points Q (n, 2): a pen leaves the nearest point of the streets (pool, pool_tau) and covers
    the straight distance times DETOUR."""
    Q = np.asarray(Q, np.float32).reshape(-1, 2)
    px, pz, pt = pool[:, 0].astype(np.float32), pool[:, 1].astype(np.float32), pool_tau.astype(np.float32)
    out = np.empty(len(Q), np.float32)
    for a in range(0, len(Q), chunk):
        q = Q[a: a + chunk]
        out[a: a + chunk] = (pt[None, :] + DETOUR * np.hypot(q[:, 0:1] - px[None, :], q[:, 1:2] - pz[None, :])).min(1)
    return out


def area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def inside(pt, P):
    x, y = pt
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hit = ((y0 > y) != (y1 > y)) & (x < x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    return bool(hit.sum() % 2)


def clip_box(P, x0, z0, x1, z1):
    """Sutherland-Hodgman: a ring cut by the box (the pieces that run along the edges of the box stay)."""
    P = [tuple(p) for p in P]
    for axis, lim, keep_less in ((0, x0, False), (0, x1, True), (1, z0, False), (1, z1, True)):
        out = []
        for a, b in zip(P, P[1:] + P[:1]):
            ia = a[axis] <= lim if keep_less else a[axis] >= lim
            ib = b[axis] <= lim if keep_less else b[axis] >= lim
            if ia != ib:
                u = (lim - a[axis]) / (b[axis] - a[axis])
                cut = (a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]))
                out += [cut, b] if ib else [cut]
            elif ib:
                out.append(b)
        P = out
        if len(P) < 3:
            return np.zeros((0, 2))
    return np.array(P)


def hatch(rings, spacing, ang, eps=1e-7):
    """Scan lines through a polygon (even-odd over its rings), `spacing` km apart, along the direction `ang`
    (radians from +x towards +z); their positions are those of one raster over the whole map.
    -> (n, 4) x0, z0, x1, z1"""
    e = np.array([math.cos(ang), math.sin(ang)])
    nrm = np.array([-e[1], e[0]])
    u0, v0, u1, v1 = [], [], [], []
    for R in rings:
        U, V = R @ e, R @ nrm
        u0.append(U), v0.append(V), u1.append(np.roll(U, -1)), v1.append(np.roll(V, -1))
    u0, v0, u1, v1 = (np.concatenate(a) for a in (u0, v0, u1, v1))
    out = []
    for k in range(math.ceil(v0.min() / spacing), math.floor(v0.max() / spacing) + 1):
        v = k * spacing + eps
        m = (v0 <= v) != (v1 <= v)
        if m.sum() < 2:
            continue
        uc = np.sort(u0[m] + (v - v0[m]) * (u1[m] - u0[m]) / (v1[m] - v0[m]))
        for a, b in zip(uc[0::2], uc[1::2]):
            if b - a > 1e-4:
                out.append((*(a * e + v * nrm), *(b * e + v * nrm)))
    return np.array(out, np.float64).reshape(-1, 4)


# ----------------------------------------------------------------------------
# tags
# ----------------------------------------------------------------------------

def num(s):
    """A length tag in metres ('12', '12.5 m', '40 ft', "40'"), or None."""
    if s is None:
        return None
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(m|ft|feet|')?\s*$", str(s))
    if not m:
        return None
    v = float(m.group(1))
    return v * 0.3048 if m.group(2) in ("ft", "feet", "'") else v


def height_of(t, kind_key="building"):
    """(height m, source 0 / 1 / 2) of a building or of a part."""
    h = num(t.get("height"))
    if h:
        return h, 0
    lv = num(t.get("building:levels"))
    if lv:
        roof = num(t.get("roof:height")) or 0.0
        return (lv * LEVEL_M if lv >= 10 else lv * LEVEL_M_LOW + PARAPET_M) + roof, 1
    return DEFAULT_H.get(t.get(kind_key), DEFAULT_H_ANY), 2


def base_of(t):
    h = num(t.get("min_height"))
    if h is not None:
        return h
    lv = num(t.get("building:min_level"))
    return lv * LEVEL_M if lv else 0.0


_ASCII = {"\u2019": "'", "\u2018": "'", "\u2013": "-", "\u2014": "-", "\u00e9": "e", "&": "+"}
_ABBR = (("STREET", "ST"), ("AVENUE", "AVE"), ("PARKWAY", "PKWY"), ("BOULEVARD", "BLVD"), ("PLACE", "PL"),
         ("DRIVE", "DR"), ("EXPRESSWAY", "EXPY"), ("HIGHWAY", "HWY"), ("ALLEY", "ALY"), ("COURT", "CT"),
         ("SQUARE", "SQ"), ("ROAD", "RD"), ("LANE", "LN"), ("TERRACE", "TER"))


def ascii_up(s):
    s = "".join(_ASCII.get(c, c) for c in s).upper()
    return "".join(c for c in s if 32 <= ord(c) < 127).strip()


def street_label(name):
    """'West 9th Street' -> 'W 9TH ST': the way a street sign writes it."""
    w = ascii_up(name).split()
    if len(w) > 2 and w[0] in ("EAST", "WEST", "NORTH", "SOUTH"):
        w[0] = w[0][0]
    if len(w) > 1:
        for full, short in _ABBR:
            if w[-1] == full:
                w[-1] = short
    return " ".join(w)


# ----------------------------------------------------------------------------
# bake
# ----------------------------------------------------------------------------

def road_ref(ref):
    """'I 71;I 75' -> 'I-71 / I-75': the shield of a motorway."""
    return " / ".join(r.strip().replace("I ", "I-") for r in ascii_up(ref).split(";"))


SHORT = {"CATHEDRAL BASILICA OF SAINT PETER IN CHAINS": "ST PETER IN CHAINS",
         "DOWNTOWN MAIN LIBRARY NORTH BUILDING": "MAIN LIBRARY // NORTH",
         "DOWNTOWN MAIN LIBRARY SOUTH BUILDING": "MAIN LIBRARY // SOUTH",
         "JOHN G. AND PHYLLIS W. SMALE RIVERFRONT PARK": "SMALE RIVERFRONT PARK",
         "JOHN A. ROEBLING SUSPENSION BRIDGE": "ROEBLING SUSPENSION BRIDGE",
         "GREATER CINCINNATI FIREFIGHTERS MEMORIAL PARK": "FIREFIGHTERS MEMORIAL PARK",
         "HILTON CINCINNATI NETHERLAND PLAZA": "NETHERLAND PLAZA",
         "RENAISSANCE CINCINNATI DOWNTOWN HOTEL": "RENAISSANCE HOTEL"}


def short_name(name):
    """A name that fits a tag: the table above, else what stands before ' AT ' (if that is a name of its own)."""
    head = name.split(" AT ")[0]
    return SHORT.get(name, head if len(head.split()) > 1 else name)


def pack(polys):
    """List of (n, 2) arrays -> points (N, 2) float32, offsets (len + 1) int32."""
    off = np.zeros(len(polys) + 1, np.int32)
    off[1:] = np.cumsum([len(p) for p in polys])
    pts = np.vstack(polys).astype(np.float32) if polys else np.zeros((0, 2), np.float32)
    return pts, off


def bake(raw):
    els = raw["main"]["elements"]
    tol_r, tol_l = SIMPLIFY_RING / 1000.0, SIMPLIFY_LINE / 1000.0
    P = lambda run: np.array([xz(la, lo) for la, lo in run])
    site_d = lambda Q: float(np.hypot(Q[:, 0], Q[:, 1]).min())

    # ---- streets --------------------------------------------------------------
    runs, graph = [], {}
    for e in els:
        t = e.get("tags", {})
        if e["type"] != "way":
            continue
        if t.get("railway") == "rail":
            cls = RAIL
        elif t.get("highway") in CLASSES:
            cls = CLASSES[t["highway"]]
        else:
            continue
        name = None if cls == 1 else ("=" + t["ref"]) if cls == 0 and t.get("ref") else t.get("name")
        bname = t.get("bridge:name") or (t["name"] if "Bridge" in t.get("name", "") else None)
        flag = (1 if t.get("bridge") not in (None, "no") else 0) | (2 if t.get("tunnel") not in (None, "no") else 0)
        cur = []
        for nid, q in zip(e["nodes"] + [None], e["geometry"] + [None]):       # the runs of the way inside the box
            if q is None:
                if len(cur) > 1:
                    runs.append((cur, cls, flag, name, bname if flag & 1 else None))
                cur = []
            else:
                cur.append((nid, xz(q["lat"], q["lon"])))
    for cur, cls, *_ in runs:                       # the network the pens run through (the railways are not in it)
        if cls != RAIL:
            for (na, pa), (nb, pb) in zip(cur, cur[1:]):
                w = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                graph.setdefault(na, []).append((nb, w))
                graph.setdefault(nb, []).append((na, w))
    tau = dijkstra(graph, SITE_NODE)
    st, st_tau, st_cls, st_flag, st_name, st_bridge, names = [], [], [], [], [], [], {}
    for cur, cls, flag, name, bname in runs:
        Q = np.array([q for _, q in cur])
        k = kept(Q, tol_l)
        if k.sum() < 2:
            continue
        st.append(Q[k]), st_cls.append(cls), st_flag.append(flag), st_bridge.append(bname)
        st_tau.append(np.array([math.inf if cls == RAIL else tau.get(n, math.inf) for n, _ in cur])[k])
        st_name.append(names.setdefault(name, len(names)) if name else -1)
    order = sorted(range(len(st)), key=lambda k: site_d(st[k]))            # nearest to the site first
    st, st_tau, st_cls, st_flag, st_name, st_bridge = ([a[k] for k in order]
                                                       for a in (st, st_tau, st_cls, st_flag, st_name, st_bridge))
    name_list = [None] * len(names)
    for n, k in names.items():
        name_list[k] = n
    # what the network does not reach (a lane behind a footpath, the railways): from the nearest point it reaches
    allp, allt = np.vstack(st), np.concatenate(st_tau)
    ok = np.isfinite(allt)
    for k in range(len(st)):
        bad = ~np.isfinite(st_tau[k])
        if bad.any():
            st_tau[k][bad] = arrive(st[k][bad], allp[ok], allt[ok])
    # the streets as a cloud of points every 12 m, with their times: where the pens leave the streets from
    pool, pool_tau, pool_cls = [], [], []
    for Q, T, cls in zip(st, st_tau, st_cls):
        for q0, q1, t0, t1 in zip(Q[:-1], Q[1:], T[:-1], T[1:]):
            n = max(1, int(math.ceil(math.hypot(*(q1 - q0)) / 0.012)))
            u = (np.arange(n) / n)[:, None]
            pool.append(q0 + (q1 - q0) * u), pool_tau.append(t0 + (t1 - t0) * u[:, 0]), pool_cls.append(np.full(n, cls))
    pool, pool_tau, pool_cls = np.vstack(pool), np.concatenate(pool_tau), np.concatenate(pool_cls)
    ground = (pool_cls >= 2) & (pool_cls <= 7)       # a building is reached from a street, not from a motorway
    from_street = lambda Q: arrive(Q, pool[ground], pool_tau[ground])
    from_any = lambda Q: arrive(Q, pool, pool_tau)

    # ---- buildings and their parts ----------------------------------------------
    member_ways, outline_tags = set(), {}
    for e in els:
        t = e.get("tags", {})
        if e["type"] != "relation" or not ("building" in t or "building:part" in t):
            continue
        if t.get("type") == "multipolygon":
            member_ways |= {m["ref"] for m in e["members"] if m["type"] == "way"}
        else:                                       # type=building: what it says belongs to its outline
            for m in e["members"]:
                if m["type"] == "way" and m.get("role") == "outline":
                    outline_tags[m["ref"]] = t

    def rings_of(e):
        """[(ring (n, 2) km, inner)] of a way or of a multipolygon."""
        if e["type"] == "way":
            g = e["geometry"]
            if any(p is None for p in g) or len(g) < 4 or g[0] != g[-1]:
                return []
            return [(P([(p["lat"], p["lon"]) for p in g[:-1]]), False)]
        out = []
        if e.get("tags", {}).get("type") != "multipolygon":
            return out
        for role in ("outer", "inner"):
            ways = [[(p["lat"], p["lon"]) for p in m["geometry"]] for m in e["members"]
                    if m["type"] == "way" and (m.get("role") or "outer") == role and m.get("geometry")
                    and all(p is not None for p in m["geometry"])]
            out += [(P(r[:-1]), role == "inner") for r in assemble(ways)]
        return out

    blds, parts = [], []
    for e in els:
        t = e.get("tags", {})
        key = "building" if "building" in t else "building:part" if "building:part" in t else None
        if key is None or t.get(key) == "no":
            continue
        if e["type"] == "way" and e["id"] in member_ways and "name" not in t and key == "building":
            continue                                # only the outline of a multipolygon that is itself a building
        if e["type"] == "way" and e["id"] in outline_tags:
            rel = outline_tags[e["id"]]             # the name, the height, the levels the relation gives
            t = {**rel, **t, **({"name": rel["name"]} if rel.get("name") else {})}
        rs = [(simplify(R, tol_r, closed=True), inner) for R, inner in rings_of(e)]
        rs = [(R, inner) for R, inner in rs if len(R) >= 3]
        if not rs or abs(area(rs[0][0])) * 1e6 < MIN_AREA:
            continue
        h, src = height_of(t, key)
        if e["type"] == "way" and e["id"] == WALL_WAY and src == 2:
            h, src = WALL_H, 1                      # the map has no height for it: the wall of the show is its face
        rec = dict(id=e["id"], type=e["type"], rings=rs, h=h, src=src, y0=base_of(t), name=t.get("name"),
                   kind=t.get(key), levels=num(t.get("building:levels")))
        (blds if key == "building" else parts).append(rec)
    # the same building is often drawn twice: as a way (the outline) and as a multipolygon made of that way
    seen = {}
    for b in sorted(blds, key=lambda b: (b["type"] != "relation", b["src"])):
        c = b["rings"][0][0].mean(0)
        k = (round(c[0], 3), round(c[1], 3), round(abs(area(b["rings"][0][0])) * 1e4))
        if k in seen:
            if not seen[k]["name"]:
                seen[k]["name"] = b["name"]
            continue
        seen[k] = b
    blds = sorted(seen.values(), key=lambda b: site_d(b["rings"][0][0]))
    # which outline does a part belong to? (the outline is then not extruded: its parts are)
    cent = np.array([b["rings"][0][0].mean(0) for b in blds])
    rad = np.array([np.hypot(*(b["rings"][0][0] - c).T).max() for b, c in zip(blds, cent)])
    for b in blds:
        b["parted"], b["top"], b["top_src"] = False, b["h"], b["src"]
    for p in parts:
        c = p["rings"][0][0].mean(0)
        p["bld"] = -1
        for k in np.nonzero(np.hypot(*(cent - c).T) < rad + 1e-4)[0]:
            if inside(c, blds[k]["rings"][0][0]):
                p["bld"] = int(k)
                b = blds[k]
                b["parted"] = True
                # the parts know the height better than an outline that only has levels, or nothing
                if p["src"] < b["src"] or (p["src"] == b["src"] and b.get("_raised") and p["h"] > b["top"]):
                    if not b.get("_raised") or p["src"] < b["top_src"] or p["h"] > b["top"]:
                        b["top"], b["top_src"], b["_raised"] = p["h"], p["src"], True
                break
    parts = [p for p in parts if p["h"] > p["y0"] + 0.5]
    parts.sort(key=lambda p: site_d(p["rings"][0][0]))

    b_rings, b_ring_bld, b_ring_inner = [], [], []
    for k, b in enumerate(blds):
        for R, inner in b["rings"]:
            b_rings.append(R), b_ring_bld.append(k), b_ring_inner.append(inner)
    # volumes: the parts, and the outlines without parts
    vols = [dict(ring=R, y0=p["y0"], y1=p["h"], bld=p["bld"]) for p in parts for R, inner in p["rings"] if not inner]
    vols += [dict(ring=R, y0=0.0, y1=b["h"], bld=k) for k, b in enumerate(blds) if not b["parted"]
             for R, inner in b["rings"]]
    vols.sort(key=lambda v: site_d(v["ring"]))

    # ---- hatch -------------------------------------------------------------------
    # along the numbered streets: the direction of 9th Street at the site
    ninth = [st[k] for k in range(len(st)) if st_name[k] >= 0
             and name_list[st_name[k]] in ("West 9th Street", "East 9th Street") and site_d(st[k]) < 0.3]
    d9 = np.sum([q[-1] - q[0] if q[-1][0] > q[0][0] else q[0] - q[-1] for q in ninth], axis=0)
    grid_ang = math.atan2(d9[1], d9[0])                         # radians from east towards south
    hc, hc_b, hf, hf_b = [], [], [], []
    for k, b in enumerate(blds):
        rs = [R for R, _ in b["rings"]]
        h = hatch(rs, HATCH_COARSE / 1000.0, grid_ang)
        hc.append(h), hc_b.append(np.full(len(h), k))
        if site_d(rs[0]) < FINE_R:
            h = hatch(rs, HATCH_FINE / 1000.0, grid_ang)
            hf.append(h), hf_b.append(np.full(len(h), k))
    hc, hc_b, hf, hf_b = np.vstack(hc), np.concatenate(hc_b), np.vstack(hf), np.concatenate(hf_b)

    # ---- water ---------------------------------------------------------------------
    w_rings = []
    for e in raw["water"]["elements"]:
        if e["type"] == "way":
            g = [(p["lat"], p["lon"]) for p in e["geometry"]]
            rings = [g[:-1]] if g[0] == g[-1] else []
        else:
            rings = [r[:-1] for r in assemble([[(p["lat"], p["lon"]) for p in m["geometry"]] for m in e["members"]
                                               if m["type"] == "way" and m.get("role", "outer") == "outer"])]
        for r in rings:
            C = clip_box(P(r), X0, Z0, X1, Z1)
            if len(C) >= 3 and abs(area(C)) > 1e-3:
                w_rings.append(C)
    bank = []                                    # the banks: the edges of the water that are not edges of the box
    on_box = lambda p: min(abs(p[0] - X0), abs(p[0] - X1), abs(p[1] - Z0), abs(p[1] - Z1)) < 1e-6
    for C in w_rings:
        run = []
        for a, b in zip(C, np.roll(C, -1, axis=0)):
            if on_box(a) and on_box(b):
                if len(run) > 1:
                    bank.append(simplify(np.array(run), tol_l * 2))
                run = []
            else:
                run = run + [b] if run else [a, b]
        if len(run) > 1:
            bank.append(simplify(np.array(run), tol_l * 2))
    # the Licking River comes in from the south, east of the Roebling bridge: the ring that touches the south side
    licking = lambda C: C[:, 1].max() > Z1 - 1e-6 and C[:, 0].mean() > 0.6 and C[:, 1].mean() > 1.5
    hs = [hatch([C], HATCH_WATER / 1000.0, 0.0) for C in w_rings]
    wh = np.vstack(hs) if hs else np.zeros((0, 4))
    wh_river = np.concatenate([np.full(len(h), int(licking(C)), np.uint8) for h, C in zip(hs, w_rings)]
                              + [np.zeros(0, np.uint8)])
    in_water = lambda p: any(inside(p, C) for C in w_rings)
    piers = []
    for Q, flag in zip(st, st_flag):
        if not flag & 1:
            continue
        seg = np.hypot(*np.diff(Q, axis=0).T)
        cum = np.r_[0.0, np.cumsum(seg)]
        for s in np.arange(0.5 * PIER_STEP, cum[-1], PIER_STEP):
            k = int(np.searchsorted(cum, s)) - 1
            p = Q[k] + (Q[k + 1] - Q[k]) * (s - cum[k]) / max(seg[k], 1e-9)
            if in_water(p):
                piers.append(p)
    piers = np.array(piers).reshape(-1, 2)

    # ---- parks ----------------------------------------------------------------------
    parks, park_names = [], []
    for e in els:
        t = e.get("tags", {})
        if t.get("leisure") != "park" or not t.get("name"):
            continue
        for R, inner in rings_of(e):
            R = simplify(R, tol_r * 2, closed=True)
            if not inner and len(R) >= 3 and abs(area(R)) * 1e6 > 1500.0:
                parks.append(R), park_names.append(t["name"])

    # ---- the wall --------------------------------------------------------------------
    wall = None
    for e in els:
        if e["type"] == "way" and e["id"] == WALL_WAY:
            R = P([(p["lat"], p["lon"]) for p in e["geometry"][:-1]])
            ed = [(float(np.hypot(*(b - a))), a, b) for a, b in zip(R, np.roll(R, -1, axis=0))]
            ln, a, b = max(ed, key=lambda v: v[0])
            if a[0] > b[0]:                         # left end first, for somebody who faces the wall (west)
                a, b = b, a
            wall = dict(a=[float(a[0]), float(a[1])], b=[float(b[0]), float(b[1])], length_m=ln * 1000.0,
                        height_m=WALL_H, width_m=WALL_W, towers_m=[list(v) for v in TOWERS], tower_off_m=TOWER_OFF,
                        address="905 VINE ST", osm_way=WALL_WAY, marker=[float(v) for v in xz(*BLINK_MARKER)])

    # ---- labels ----------------------------------------------------------------------
    disp = [road_ref(n[1:]) if n.startswith("=") else street_label(n) for n in name_list]
    labels = [dict(name="9TH ST X VINE ST", short="9TH X VINE", kind="site", x=0.0, y=0.0, z=0.0, rank=0, d=0.0)]
    if wall:
        m = 0.5 * (np.array(wall["a"]) + np.array(wall["b"]))
        labels.append(dict(name="THE WALL", short="THE WALL", kind="site", x=float(m[0]), y=WALL_H / 1000.0,
                           z=float(m[1]), rank=0, d=float(np.hypot(*m)), info="905 VINE ST // 27.7 M WIDE"))

    def along(Q, step):
        """Points every `step` km along a polyline, with the compass bearing of the line there (0..180)."""
        seg = np.hypot(*np.diff(Q, axis=0).T)
        cum = np.r_[0.0, np.cumsum(seg)]
        out = []
        for s_ in np.arange(min(0.5 * step, 0.5 * cum[-1]), cum[-1], step):
            j = min(int(np.searchsorted(cum, s_, side="right")) - 1, len(seg) - 1)
            q = Q[j] + (Q[j + 1] - Q[j]) * (s_ - cum[j]) / max(seg[j], 1e-9)
            d = Q[j + 1] - Q[j]
            out.append((float(q[0]), float(q[1]), math.degrees(math.atan2(d[0], -d[1])) % 180.0))
        return out

    # streets: an anchor about every 180 m of every named street, the nearest to the site first
    for k, name in enumerate(name_list):
        runs = [j for j in range(len(st)) if st_name[j] == k]
        cls = min(st_cls[j] for j in runs)
        if sum(float(np.hypot(*np.diff(st[j], axis=0).T).sum()) for j in runs) < 0.06:
            continue
        anchors = []
        for j in runs:
            for a_ in along(st[j], 0.18):
                if not anchors or min(math.hypot(a_[0] - q[0], a_[1] - q[1]) for q in anchors) > 0.12:
                    anchors.append(a_)
        dmin = min(math.hypot(q[0], q[1]) for q in anchors)
        if name in ("Vine Street", "West 9th Street", "East 9th Street"):
            rank = 0
        elif dmin < 0.3 and cls <= 5:
            rank = 1
        elif dmin < 0.7 and cls <= 5:
            rank = 2
        else:
            rank = 3 if cls <= 4 else 4
        for q in sorted(anchors, key=lambda q: math.hypot(q[0], q[1])):
            labels.append(dict(name=disp[k], short=disp[k], kind="street", x=q[0], y=0.0, z=q[1], rank=rank,
                               d=math.hypot(q[0], q[1]), brg=round(q[2], 1), cls=cls))
    # bridges: where they are over the water, by the name of the bridge
    over = {}
    for j in range(len(st)):
        if st_bridge[j]:
            over.setdefault(st_bridge[j], []).extend(q for q in along(st[j], 0.04) if in_water(q[:2]))
    for name, pts in over.items():
        if len(pts) < 3:
            continue
        c = np.mean(np.array(pts)[:, :2], axis=0)
        name = ascii_up(name)
        labels.append(dict(name=name, short=short_name(name), kind="bridge", x=float(c[0]), y=0.0, z=float(c[1]),
                           rank=0 if "ROEBLING" in name else 1, d=float(np.hypot(*c)),
                           brg=round(float(np.median(np.array(pts)[:, 2])), 1)))
    # landmarks: the tall named buildings, and the ones a visitor knows
    used = set()
    for k, b in enumerate(blds):
        if not b["name"]:
            continue
        notable = next((n for n in NOTABLE if n.lower() in b["name"].lower()), None)
        top = b["top"]
        if b["name"] in HEIGHTS:
            top, b["top_src"] = HEIGHTS[b["name"]], 0
        if not ((top >= LANDMARK_H and b["top_src"] < 2) or notable):
            continue
        name = ascii_up(b["name"])
        if name in used:
            continue
        used.add(name)
        c = b["rings"][0][0].mean(0)
        near = notable in ("Kroger Building", "Downtown Main Library")
        rank = 0 if (top >= 140.0 and b["top_src"] == 0) or near else 1 if top >= 100.0 or notable else 2
        labels.append(dict(name=name, short=short_name(name), kind="landmark", x=float(c[0]), y=top / 1000.0,
                           z=float(c[1]), rank=rank, d=float(np.hypot(*c)), h=round(top, 1),
                           h_src=("tag", "levels", "default")[b["top_src"]], bld=k,
                           levels=int(b["levels"]) if b["levels"] else None))
    for name, R in zip(park_names, parks):
        c = R.mean(0)
        name = ascii_up(name)
        labels.append(dict(name=name, short=short_name(name), kind="park", x=float(c[0]), y=0.0, z=float(c[1]),
                           rank=1 if name in ("PIATT PARK", "WASHINGTON PARK") else 2, d=float(np.hypot(*c))))
    if w_rings:
        row = wh[(np.abs(wh[:, 0] + wh[:, 2]) < 1.2) & (wh[:, 2] - wh[:, 0] > 1.0)]
        zc = float(np.median(row[:, 1])) if len(row) else float(max(w_rings, key=lambda C: abs(area(C)))[:, 1].mean())
        labels.append(dict(name="OHIO RIVER", short="OHIO RIVER", kind="river", x=0.15, y=RIVER_Y, z=zc, rank=0,
                           d=abs(zc)))
        lick = [C for C in w_rings if licking(C)]
        if lick:
            c = lick[0].mean(0)
            labels.append(dict(name="LICKING RIVER", short="LICKING RIVER", kind="river", x=float(c[0]), y=RIVER_Y,
                               z=float(c[1]), rank=1, d=float(np.hypot(*c))))
    for name, la, lo in PLACES:
        x, z = xz(la, lo)
        labels.append(dict(name=name, short=name.split(" // ")[0], kind="place", x=x, y=0.0, z=z, rank=1,
                           d=math.hypot(x, z)))

    # ---- write -------------------------------------------------------------------------
    st_pts, st_off = pack(st)
    b_pts, b_off = pack(b_rings)
    v_pts, v_off = pack([v["ring"] for v in vols])
    wb_pts, wb_off = pack(bank)
    pk_pts, pk_off = pack(parks)
    # when the pens get there (see the top of the file)
    b_tau = from_street(b_pts)
    b_done = np.zeros(len(blds), np.float32)
    np.maximum.at(b_done, np.repeat(np.array(b_ring_bld), np.diff(b_off)), b_tau)
    v_bld = np.array([v["bld"] for v in vols], np.int32)
    v_own = np.maximum.reduceat(from_street(v_pts), v_off[:-1])
    v_tau = np.where(v_bld >= 0, b_done[np.maximum(v_bld, 0)], v_own).astype(np.float32)
    hc_tau = np.stack([from_street(hc[:, 0:2]), from_street(hc[:, 2:4])], 1)
    hf_tau = np.stack([from_street(hf[:, 0:2]), from_street(hf[:, 2:4])], 1)
    wb_tau = from_any(wb_pts)
    wh_tau = np.stack([from_any(wh[:, 0:2]), from_any(wh[:, 2:4])], 1) if len(wh) else np.zeros((0, 2), np.float32)
    piers_tau = from_any(piers) if len(piers) else np.zeros(0, np.float32)
    pk_tau = from_street(pk_pts) if len(pk_pts) else np.zeros(0, np.float32)
    for lb in labels:                               # ... and when a name can be written
        lb["tau"] = float(b_done[lb["bld"]]) if "bld" in lb else float(from_any(np.array([[lb["x"], lb["z"]]]))[0])
    src = np.array([b["src"] for b in blds], np.uint8)
    meta = dict(
        attribution=ATTRIBUTION, source="OpenStreetMap via the Overpass API (ODbL)",
        osm_base=raw["main"].get("osm3s", {}).get("timestamp_osm_base"), baked=time.strftime("%Y-%m-%d"),
        site=dict(lat=SITE[0], lon=SITE[1], node=197417418, name="VINE ST X 9TH ST"), bbox=list(BBOX),
        extent=[X0, Z0, X1, Z1], km_per_deg=[KX, KY], grid_bearing=round(90.0 + math.degrees(grid_ang), 2),
        river_y=RIVER_Y, wall=wall, street_names=disp, labels=labels,
        building_names={str(k): ascii_up(b["name"]) for k, b in enumerate(blds) if b["name"] and ascii_up(b["name"])},
        tau_max=float(max(np.concatenate(st_tau).max(), b_done.max(), v_tau.max())),
        counts=dict(streets=len(st), buildings=len(blds), parts=len(parts), volumes=len(vols),
                    height_tagged=int((src == 0).sum()), height_from_levels=int((src == 1).sum()),
                    height_default=int((src == 2).sum())))
    np.savez_compressed(
        OUT, meta=np.array(json.dumps(meta)),
        st_pts=st_pts, st_off=st_off, st_cls=np.array(st_cls, np.uint8), st_flag=np.array(st_flag, np.uint8),
        st_name=np.array(st_name, np.int16), st_tau=np.concatenate(st_tau).astype(np.float32),
        b_tau=b_tau, b_done=b_done, v_tau=v_tau, hc_tau=hc_tau, hf_tau=hf_tau, wb_tau=wb_tau, wh_tau=wh_tau,
        piers_tau=piers_tau, pk_tau=pk_tau,
        b_pts=b_pts, b_off=b_off, b_ring_bld=np.array(b_ring_bld, np.int32), b_ring_inner=np.array(b_ring_inner, bool),
        b_h=np.array([b["top"] for b in blds], np.float32), b_src=np.array([b["top_src"] for b in blds], np.uint8),
        b_levels=np.array([int(b["levels"] or 0) for b in blds], np.int16),
        b_parted=np.array([b["parted"] for b in blds], bool),
        v_pts=v_pts, v_off=v_off, v_y0=np.array([v["y0"] for v in vols], np.float32),
        v_y1=np.array([v["y1"] for v in vols], np.float32), v_bld=v_bld,
        hc=hc.astype(np.float32), hc_b=hc_b.astype(np.int32), hf=hf.astype(np.float32), hf_b=hf_b.astype(np.int32),
        wb_pts=wb_pts, wb_off=wb_off, wh=wh.astype(np.float32), wh_river=wh_river, piers=piers.astype(np.float32),
        pk_pts=pk_pts, pk_off=pk_off)
    return meta


def preview(path):
    """A plan of the baked file, drawn with PIL: what the bake kept, at a glance."""
    from PIL import Image, ImageDraw
    z = np.load(OUT)
    meta = json.loads(str(z["meta"]))
    x0, z0, x1, z1 = meta["extent"]
    S = 520.0
    W, H = int((x1 - x0) * S), int((z1 - z0) * S)
    im = Image.new("RGB", (W, H), (0, 0, 0))
    dr = ImageDraw.Draw(im)
    T = lambda p: [((float(a) - x0) * S, (float(b) - z0) * S) for a, b in p]
    for a, b in zip(z["wb_off"][:-1], z["wb_off"][1:]):
        dr.line(T(z["wb_pts"][a:b]), fill=(90, 90, 160), width=1)
    for s in z["wh"]:
        dr.line(T([s[:2], s[2:]]), fill=(40, 40, 90), width=1)
    for a, b in zip(z["b_off"][:-1], z["b_off"][1:]):
        dr.line(T(np.vstack([z["b_pts"][a:b], z["b_pts"][a:a + 1]])), fill=(120, 120, 120), width=1)
    for k, (a, b) in enumerate(zip(z["st_off"][:-1], z["st_off"][1:])):
        c = int(z["st_cls"][k])
        dr.line(T(z["st_pts"][a:b]), fill=(255, 255, 255) if c <= 3 else (170, 170, 170) if c <= 5 else (90, 90, 90),
                width=2 if c <= 2 else 1)
    if meta["wall"]:
        dr.line(T([meta["wall"]["a"], meta["wall"]["b"]]), fill=(255, 0, 0), width=3)
    c = T([(0.0, 0.0)])[0]
    dr.ellipse((c[0] - 9, c[1] - 9, c[0] + 9, c[1] + 9), outline=(255, 0, 0), width=2)
    im.save(path)
    print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="download the box again")
    ap.add_argument("--preview", default=None, help="write a plan of the baked data to this PNG")
    a = ap.parse_args()
    meta = bake(fetch(a.refresh))
    c = meta["counts"]
    print(f"{OUT}  {OUT.stat().st_size / 1e6:.2f} MB   OSM base {meta['osm_base']}")
    print(f"  streets {c['streets']}   buildings {c['buildings']} (height: {c['height_tagged']} tagged, "
          f"{c['height_from_levels']} from levels, {c['height_default']} by default)   parts {c['parts']}   "
          f"volumes {c['volumes']}")
    print(f"  grid: the numbered streets run {meta['grid_bearing']:.1f} deg from north; labels {len(meta['labels'])}; "
          f"wall {'%.1f m' % meta['wall']['length_m'] if meta['wall'] else 'NOT FOUND'}")
    if a.preview:
        preview(a.preview)


if __name__ == "__main__":
    main()
