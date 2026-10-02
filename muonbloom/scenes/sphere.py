"""SPHERE - muon scattering tomography of a lumpy body, live, on the groove.
Sheet 5.0 COSMIC GROOVE, 03:56 - 04:36 (the slot is read from showdata.SECTIONS).

The TouchDesigner plexus sphere (a noise-deformed sphere made of a tangle of thin lines, dotted
outline, "lollipop" halo, dot lattice) kept as the body, with the detail of the earlier study and
two rules:
  LIVE   every onset the three towers stream becomes one red muon track through the body. A leader
         runs from the detector up to a bus under the header and drops into the track; tracks that
         meet the hidden dense core kink at their point of closest approach (PoCA). The dots
         gather, the core is given away around the middle of the scene (ANOMALY) and the full
         picture stands for the last ten seconds. CENTRE tower = heavy slow tracks, LEFT / RIGHT =
         fine fast ones, echo repeats of a hit = thin companions that are gone in a few seconds;
         older tracks fade to ghosts so 90-odd tracks in 40 s stay readable. The PoCA dots stay.
  DRUMS  every kick of the music is answered in white: the net swells, a ripple runs up both
         sides of the halo, a dotted shock ring leaves the body through the opacity histogram,
         the lattice crosses brighten. No full-frame flash.

THE TOWERS STAND IN FRONT OF THE WALL and nobody knows yet where, how wide or how tall. Nothing
here has a fixed x: `Lay` reads the free columns from the context (ctx.cols, ctx.focus_bay,
ctx.slots, ctx.sub) and deals the blocks out by priority, scaling them to the room they get:
  1 the body + halo + its callouts        the focus bay (scale 0.55 .. 1.12)
  2 TOMOGRAM (top view of the slice through the core) + the opacity histogram unrolled
  3 MUOGRAPHY: target, exposure, live track log (the log drops fields when the column is narrow)
  4 DETECTORS: the three streams + the scattering-angle spectrum
  Every free column takes one block, or several side by side when it is wide (two towers close
  together leave one big bay); what does not fit anywhere is dropped, 4 first. Among equal
  solutions the reading order log | body | tomogram | detectors wins.
  bottom band: counters / stream barcode / RHO_CORE in the free panels, widest first
  top right:   KICK_GRID where the subtitle box sits when somebody speaks (nobody does here)
Leaders, the lattice and the timeline may pass behind a tower; text, numbers and pictures never.
The DISINTEGRATE scene (scenes/disintegrate.py) reuses the body, the layout and the drawing code.
"""
from __future__ import annotations

import math

import numpy as np

from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import CHAR_W, Camera, hash01, smoothstep, text_w
from ..show import Scene


def _span(look="sphere", default=(236.0, 276.0)):
    """Start / end of the slot the sheet gives this look."""
    secs = [x for x in sd.SECTIONS if x[4] == look]
    return (secs[0][2], secs[-1][3]) if secs else default


T0, T1 = _span()
FOUND_W = 4.5                     # weighted scattered tracks (primary 1, echo 0.25) it takes to call the core

# the body at scale 1 (design px); Lay scales it to the bay it gets
R_PX = 232.0                      # projected radius of the unit sphere
RING_R0 = 276.0                   # base circle of the opacity histogram
RING_L = (12.0, 90.0)             # stick length for opacity 0 .. 1
BODY_W = 2.0 * (RING_R0 + RING_L[1])
S_MIN, S_MAX = 0.55, 1.12
CAM_D, CAM_H = 7.4, 1.9
N_BINS = 144
HOT = 0.78                        # flagged bins
CORE = np.array([0.30, -0.14, 0.22])
CORE_R = 0.30
R_FAR = 1.6                       # tracks are drawn from / to this radius (world units)
MRAD = 350.0                      # drawn kink angle (rad) -> displayed scattering angle (mrad)
BUS_Y = {"C": 244.0, "L": 252.0, "R": 260.0}       # the leaders run just under the header band
KEY_SEED = {"L": 1, "C": 2, "R": 3}
TOMO_HALF = 1.25                  # half extent of the tomogram (m)
VOX = 0.125

# vertical layout (the bays are free from the header down to the bottom data band, whatever the towers are)
TOP, BOT = L.HEAD_Y + 12.0, 1196.0
Y_TITLE, Y_PANEL, Y_LOW, Y_BASE = 318.0, 300.0, 964.0, 1156.0
Y_CORNER = 1090.0                 # tag of the two callouts in the bottom corners of the body column
CORNER_W = 218.0
# what a block needs to exist
TOMO_MIN, TEXT_MIN, DETS_MIN = 330.0, 190.0, 320.0

_DIST = math.hypot(CAM_D, CAM_H)
_GLYPHS = "0123456789ABCDEF#%/*+-=<>"


def mwe(v):
    return 1.2 + 6.8 * np.asarray(v)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def perp(d):
    d = np.atleast_2d(d)
    a = np.where(np.abs(d[:, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    e1 = unit(np.cross(d, a))
    return e1, np.cross(d, e1)


def kink(d, theta, psi):
    e1, e2 = perp(d)
    side = e1[0] * math.cos(psi) + e2[0] * math.sin(psi)
    return unit(d * math.cos(theta) + side * math.sin(theta))


def reach(K, v, R):
    """Distance from the interior point K along the unit direction v to the sphere of radius R."""
    b = float(K @ v)
    return -b + math.sqrt(max(b * b - (float(K @ K) - R * R), 0.0))


def er(text, amount, key=0, frame=0):
    """Eroded text: a stable, growing subset of characters vanishes, the ones about to go flicker through
    random glyphs first. Empty once nothing readable is left, so the tag / label that carries it goes too."""
    if amount <= 0.0 or not text:
        return text
    if amount >= 0.94:
        return ""
    h = hash01(np.arange(len(text)), key)
    band = min(0.12, 0.6 * amount)
    out = []
    for k, (c, hv) in enumerate(zip(text, h)):
        if c == " " or hv >= amount + band:
            out.append(c)
        elif hv < amount:
            out.append(" ")
        else:
            out.append(_GLYPHS[int(hash01(k, key, frame) * len(_GLYPHS))])
    out = "".join(out)
    return out if out.strip() else ""


def fade(amount):
    """Alpha of the boxes and rules of a panel while its text erodes."""
    return float(1.0 - smoothstep(0.3, 0.85, amount))


def fit_fields(fields, width, size=L.T_MICRO):
    """Join as many leading fields as fit in `width` px (monospace): narrow columns drop the last ones."""
    n = int(width / (size * CHAR_W))
    out = ""
    for k, fld in enumerate(fields):
        nxt = fld if k == 0 else out + " " + fld
        if len(nxt) > n:
            break
        out = nxt
    return out


def fit_title(title, width, size=L.T_MICRO, pad=8.0):
    """Drop the trailing ' // ' parts of a title until it fits `width` px."""
    parts = title.split(" // ")
    while len(parts) > 1 and text_w(" // ".join(parts), size) + pad > width:
        parts.pop()
    return " // ".join(parts)


def fit_size(s, width, lo, hi):
    """Largest type size in lo..hi at which `s` fits in `width` px."""
    return float(np.clip(width / max(len(s) * CHAR_W, 1e-6), lo, hi))


# ----------------------------------------------------------------------------
# layout: everything is dealt out from the columns the towers leave free
# ----------------------------------------------------------------------------

def _box_dist(box, cx, cy):
    x0, y0, x1, y1 = box
    return math.hypot(cx - min(max(cx, x0), x1), cy - min(max(cy, y0), y1))


# what a block needs: (minimum, ideal, maximum) width in px. A bay that is wider than one block needs is
# shared: several blocks stand side by side in it, in the reading order of ORDER4.
BLOCKS = {"body": (438.0, 767.0, 855.0), "tomo": (TOMO_MIN, 700.0, 767.0), "info": (TEXT_MIN, 403.0, 440.0),
          "dets": (DETS_MIN, 403.0, 460.0)}
ORDER4 = ("info", "body", "tomo", "dets")
WEIGHT = {"tomo": 4.0, "info": 2.0, "dets": 1.0}        # what is kept first when there is not room for all
GUTTER = 40.0


def _share(col, names):
    """The blocks `names` side by side in one column -> {name: (x0, x1)}, or None when they do not fit.
    The body keeps its ideal size when it shares a column; a block alone in a column takes all of it
    (the tomogram and the detectors up to their maximum)."""
    x0, x1 = col
    w = x1 - x0
    names = [n for n in ORDER4 if n in names]
    if len(names) == 1:
        n = names[0]
        if n != "body" and w < BLOCKS[n][0]:
            return None
        if n == "tomo" and w > BLOCKS[n][2]:
            c = 0.5 * (x0 + x1)
            return {n: (c - 0.5 * BLOCKS[n][2], c + 0.5 * BLOCKS[n][2])}
        if n == "dets" and w > BLOCKS[n][2]:
            return {n: (x0, x0 + BLOCKS[n][2])}
        return {n: (x0, x1)}
    width = {n: (BLOCKS[n][1] if n == "body" else BLOCKS[n][0]) for n in names}
    extra = w - sum(width.values()) - GUTTER * (len(names) - 1)
    if extra < 0:
        return None
    for lim in (1, 2):                      # grow towards the ideals first, then towards the maxima
        want = {n: max(0.0, BLOCKS[n][lim] - width[n]) for n in names}
        tot = sum(want.values())
        if tot > 0 and extra > 0:
            give = min(extra, tot)
            for n in names:
                width[n] += give * want[n] / tot
            extra -= give
    gut = GUTTER + extra / (len(names) - 1)        # what is still left widens the gutters
    out, x = {}, x0
    for n in names:
        out[n] = (x, x + width[n])
        x += width[n] + gut
    return out


def _deal(cols, body_idx):
    """Deal the blocks out over the free columns: the body in column `body_idx`, then as many of the others
    as fit, the important ones first, each as close to its ideal width as possible; among equals the
    reading order log | body | tomogram | detectors wins. Small enough to try every assignment."""
    mid = lambda r: 0.5 * (r[0] + r[1])
    best = (-1.0, {"body": cols[body_idx]})
    options = list(range(len(cols))) + [None]
    for a_t in options:
        for a_i in options:
            for a_d in options:
                assign = {"body": body_idx, "tomo": a_t, "info": a_i, "dets": a_d}
                rects = {}
                for ci, col in enumerate(cols):
                    names = [n for n, c in assign.items() if c == ci]
                    if names:
                        r = _share(col, names)
                        if r is None:
                            break
                        rects.update(r)
                else:
                    score = 0.0
                    for n in ("tomo", "info", "dets"):
                        if n in rects:
                            score += 10.0 * WEIGHT[n] + min(rects[n][1] - rects[n][0], BLOCKS[n][1]) / BLOCKS[n][1]
                            score -= 0.25 if assign[n] == body_idx else 0.0     # rather leave the body alone
                    mb = mid(rects["body"])
                    score += 0.06 if "info" in rects and mid(rects["info"]) < mb else 0.0
                    score += 0.04 if "tomo" in rects and mid(rects["tomo"]) > mb else 0.0
                    score += 0.02 if "dets" in rects and mid(rects["dets"]) > mb else 0.0
                    if "tomo" in rects:
                        score -= 0.1 * abs(mid(rects["tomo"]) - mb) / L.W
                    if score > best[0] + 1e-9:
                        best = (score, rects)
    return best[1]


class Lay:
    """Where every block of the scene goes for one tower placement (see the module docstring)."""

    def __init__(self, ctx):
        width = lambda c: c[1] - c[0]
        mid = lambda c: 0.5 * (c[0] + c[1])
        cols = [tuple(map(float, c)) for c in ctx.cols]
        fa, fb = ctx.focus_bay
        if not cols:
            cols = [(max(fa + 28.0, L.COL_X0), min(fb - 28.0, L.COL_X1))]
        inside = [c for c in cols if fa <= mid(c) <= fb]
        home = max(inside, key=width) if inside else max(cols, key=width)
        rects = _deal(cols, cols.index(home))
        body = rects["body"]
        self.tomo, self.info, self.dets = rects.get("tomo"), rects.get("info"), rects.get("dets")
        self.spec = self.dets                       # the spectrum sits under the detectors
        # the lattice and the body are clipped to the bay, or to the body's share of it
        bay = next(((a, b) for a, b in ctx.bays if a <= mid(home) <= b), home)
        self.clip = (bay[0] + 6.0 if body[0] <= home[0] + 1 else body[0] - 18.0, TOP,
                     bay[1] - 6.0 if body[1] >= home[1] - 1 else body[1] + 18.0, BOT)
        # 1: the body, scaled to its column; the callouts in the corners must stay clear of the halo
        self.body = body
        self.cx, self.cy = mid(body), 0.5 * (TOP + BOT) + 8.0
        w = width(body)
        self.two_corners = w >= 2 * CORNER_W + 14.0
        corners = [(body[0], Y_PANEL - 18.0, body[0] + min(344.0, w), Y_PANEL + 6.0),            # VIEW tag
                   (body[0], Y_PANEL + 6.0, body[0] + min(226.0, w), Y_PANEL + 40.0),            # its line
                   (body[0], Y_CORNER - 24.0, body[0] + CORNER_W, BOT)]                          # bottom left
        if self.two_corners:
            corners.append((body[1] - CORNER_W, Y_CORNER - 24.0, body[1], BOT))                  # bottom right
        s = min(S_MAX, (w - 35.0) / BODY_W)
        while s > S_MIN and any(_box_dist(b, self.cx, self.cy) < 0.5 * BODY_W * s + 14.0 for b in corners):
            s *= 0.98
        self.s = s = max(s, S_MIN)
        self.R, self.r0 = R_PX * s, RING_R0 * s
        self.l0, self.l1 = RING_L[0] * s, RING_L[1] * s
        self.out = self.r0 + self.l1
        self.focal = self.R / math.tan(math.asin(1.0 / _DIST))
        self.fov = math.degrees(2 * math.atan((L.H / 2) / self.focal))
        # the tomogram square inside its column: readouts beside it when there is room, else under it
        self.tg = None
        if self.tomo:
            x0, x1 = self.tomo
            w = x1 - x0
            if w - 237.0 >= 430.0:
                sd_ = min(w - 237.0, 530.0)
                self.tg = dict(px0=x1 - 8.0 - sd_, side=sd_, mode="side")
            else:
                sd_ = min(w - 70.0, 430.0)
                self.tg = dict(px0=x0 + 62.0 + 0.5 * (w - 70.0 - sd_), side=sd_, mode="below")
            self.tg.update(py0=363.0, S=sd_ / (2 * TOMO_HALF))
        # bottom band: the free panels, widest first (ties keep their left-to-right order)
        panels = sorted((tuple(map(float, p)) for p in ctx.slots["panels"]), key=lambda p: -(p[1] - p[0]))
        need = (250.0, 180.0, 130.0)
        got = [p if p[1] - p[0] >= n else None for p, n in zip(panels, need)] + [None] * 3
        self.p_num, self.p_bar, self.p_one = got[:3]
        self.y0, self.y1 = float(ctx.slots["y0"]), float(ctx.slots["y1"])

    def camera(self, t, t0, span, phase=0.7):
        """One slow orbit around the body in `span` seconds."""
        yaw = 2 * math.pi * (t - t0) / span + phase
        return Camera((CAM_D * math.sin(yaw), CAM_H, CAM_D * math.cos(yaw)), (0.0, 0.0, 0.0), fov_deg=self.fov,
                      screen_center=(self.cx, self.cy)), yaw


# ----------------------------------------------------------------------------
# the body and its tracks
# ----------------------------------------------------------------------------

class Body:
    """The lumpy sphere: fibonacci vertices on a slowly breathing radius field, a surface net and the
    long struts through the interior that make it a plexus."""

    def __init__(self, seed=41, n=1500, amp=0.036):
        rng = np.random.default_rng(seed)
        m = 18
        self.d = unit(rng.normal(size=(m, 3)))
        self.fr = rng.uniform(3.2, 9.5, m)
        self.ph = rng.uniform(0, 2 * np.pi, m)
        a = rng.uniform(0.4, 1.0, m) / self.fr ** 0.5
        self.a = a * amp / math.sqrt((a ** 2).sum() / 2)
        self.w = rng.uniform(0.12, 0.4, m) * np.where(rng.random(m) < 0.5, -1.0, 1.0)
        # vertices
        k = np.arange(n) + 0.5
        y = 1 - 2 * k / n
        r = np.sqrt(1 - y * y)
        th = math.pi * (3 - math.sqrt(5)) * k
        u = np.stack([r * np.cos(th), y, r * np.sin(th)], 1)
        self.u = unit(u + rng.normal(0, 0.3 * math.sqrt(4 * math.pi / n), u.shape))
        self.n = n
        # surface net: every vertex to its 3 nearest, plus a few longer links
        P = self.surface(0.0).astype(np.float32)
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn = np.argsort(d2, axis=1)[:, :14]
        i = np.repeat(np.arange(n), 3)
        j = nn[:, :3].ravel()
        i2 = np.nonzero(rng.random(n) < 0.3)[0]
        j2 = nn[i2, rng.integers(3, 14, len(i2))]
        I, J = np.r_[i, i2], np.r_[j, j2]
        key = np.unique(np.minimum(I, J) * n + np.maximum(I, J))
        self.ea, self.eb = key // n, key % n
        # struts: chords through the interior between distant vertices (the plexus)
        a_, b_ = rng.integers(0, n, 900), rng.integers(0, n, 900)
        ang = np.arccos(np.clip((self.u[a_] * self.u[b_]).sum(1), -1, 1))
        ok = (ang > 0.75) & (ang < 2.7)
        self.sa, self.sb = a_[ok][:330], b_[ok][:330]
        self.s_var = rng.uniform(0.55, 1.0, len(self.sa)).astype(np.float32)
        self.e_var = rng.uniform(0.75, 1.0, len(self.ea)).astype(np.float32)

    def rho(self, u, tw):
        return 1.0 + (self.a * np.sin((u @ self.d.T) * self.fr + self.ph + self.w * tw)).sum(-1)

    def surface(self, tw, u=None):
        u = self.u if u is None else u
        return u * self.rho(u, tw)[..., None]

    def normals(self, tw, e=2e-3):
        u = self.u
        t1, t2 = perp(u)
        p = self.surface(tw)
        nrm = unit(np.cross(self.surface(tw, unit(u + e * t1)) - p, self.surface(tw, unit(u + e * t2)) - p))
        return nrm * np.sign((nrm * u).sum(1))[:, None]

    def slice(self, y, tw, n=200):
        """Contour of the body cut by the horizontal plane at height y."""
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        uy = np.full(n, y)
        for _ in range(5):
            uy = np.clip(uy, -0.999, 0.999)
            r = np.sqrt(1 - uy ** 2)
            u = np.stack([r * np.cos(a), uy, r * np.sin(a)], 1)
            uy = y / self.rho(u, tw)
        return self.surface(tw, u)


def build_tracks(det, t0, t1, core_from=None, core_at=None):
    """One muon track per detector onset in [t0, t1): path = far point, entry, PoCA, exit, far point.
    No track is aimed at the core before `core_from`; the primary hits of the first cluster at / after
    `core_at` always are (their echo trains then pile up on the core: that is when the anomaly shows)."""
    out = []
    forced = set()
    if core_at is not None:
        cand = []
        for key in sd.KEYS:
            tt, ee, ec = det.hits(key, t0, t1)
            cand += [(float(th), key, i) for i, (th, echo) in enumerate(zip(tt, ec)) if not echo and th >= core_at]
        if cand:
            first = min(cand)[0]
            forced = {(key, i) for th, key, i in cand if th < first + 1.6}
    for key in sd.KEYS:
        tt, ee, ec = det.hits(key, t0, t1)
        prev = None
        for i, (th, e, echo) in enumerate(zip(tt, ee, ec)):
            rng = np.random.default_rng([KEY_SEED[key], i, 77])
            heavy = key == "C"
            if echo and prev is not None:
                d = unit(prev["d"] + rng.normal(0, 0.045, 3))
                aim = prev["aim"] + rng.normal(0, 0.07, 3)
            else:
                zen = min(abs(rng.normal(0.0, 0.45)), 1.1)
                az = rng.uniform(0, 2 * np.pi)
                d = np.array([math.sin(zen) * math.cos(az), -math.cos(zen), math.sin(zen) * math.sin(az)])
                to_core = rng.random() < (0.8 if heavy else 0.4) and (core_from is None or th >= core_from)
                if to_core or (key, i) in forced:
                    aim = CORE + unit(rng.normal(size=3)) * CORE_R * rng.uniform(0.0, 0.75)
                else:
                    e1, e2 = perp(d)
                    r, a = 0.9 * math.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
                    aim = e1[0] * r * math.cos(a) + e2[0] * r * math.sin(a)
            s_c = float((CORE - aim) @ d)
            pc = aim + s_c * d
            dist = float(np.linalg.norm(CORE - pc))
            through = dist < CORE_R
            K = pc if through else aim + d * rng.uniform(-0.3, 0.3)
            if np.linalg.norm(K) > 0.93:
                K = K * 0.93 / np.linalg.norm(K)
            theta = (0.035 + 0.11 * rng.random() * (1.4 - dist / CORE_R)) if through else abs(rng.normal(0, 0.006))
            d2 = kink(d, theta, rng.uniform(0, 2 * np.pi))
            path = np.stack([K - d * reach(K, -d, R_FAR), K - d * reach(K, -d, 1.0), K,
                             K + d2 * reach(K, d2, 1.0), K + d2 * reach(K, d2, R_FAR)]).astype(np.float32)
            tr = dict(t=float(th), key=key, e=float(e), echo=bool(echo), heavy=heavy, d=d, aim=aim, K=K, path=path,
                      through=through, mrad=theta * MRAD, p=0.4 + 7.5 * float(e) ** 1.6,
                      charge="+" if rng.random() < 0.56 else "-", dur=0.9 if heavy else 0.3,
                      zen=math.degrees(math.acos(min(1.0, -d[1]))), azi=math.degrees(math.atan2(d[2], d[0])) % 360)
            out.append(tr)
            if not echo:
                prev = tr
    out.sort(key=lambda r: r["t"])
    for n, tr in enumerate(out):
        tr["id"] = n + 1
    return out


def sphere_tracks(det):
    """The tracks of the COSMIC GROOVE slot (the first onsets fall a few tenths before the cut: the scene
    opens on live data). DISINTEGRATE comes back with the same picture."""
    span = T1 - T0
    return build_tracks(det, T0 - 0.6, T1, core_from=T0 + 0.22 * span, core_at=T0 + 0.4 * span)


def persistence(past, t):
    """How much of each track's line is still drawn at t (id -> 0..1). The lines do not pile up: an echo is
    gone in a few seconds, a primary track fades to a ghost as newer ones arrive."""
    rank, n_prim = {}, 0
    for tr in past[::-1]:
        if not tr["echo"]:
            rank[tr["id"]] = n_prim
            n_prim += 1
    out = {}
    for tr in past:
        age = t - tr["t"]
        if tr["echo"]:
            out[tr["id"]] = math.exp(-max(0.0, age - 0.3) / 1.5) if age < 8.0 else 0.0
        else:
            k = rank[tr["id"]]
            hold = 1.0 if k < 5 else (0.5 if k < 11 else 0.2)
            out[tr["id"]] = hold * (0.3 + 0.7 * math.exp(-age / 5.0))
    return out


def ring_values(lay, cam, n_through, n_all, seed, jit):
    """Opacity per polar bin (screen angle, 0 = right, counter-clockwise). Flat while there is no data,
    then the peak towards the core grows with the tracks that crossed it."""
    cx, cy, _, _ = cam.project(CORE[None].astype(np.float32))
    phc = math.atan2(-(float(cy[0]) - lay.cy), float(cx[0]) - lay.cx)
    phi = 2 * np.pi * (np.arange(N_BINS) + 0.5) / N_BINS
    rng = np.random.default_rng(seed)
    k = np.arange(2, 9)
    amp = rng.uniform(0.4, 1.0, 7) / k ** 0.8
    ph = rng.uniform(0, 2 * np.pi, 7)
    base = (amp * np.sin(k * phi[:, None] + ph)).sum(-1) / amp.sum()
    peak = np.exp(-(np.angle(np.exp(1j * (phi - phc))) / 0.22) ** 2)
    data = min(1.0, n_all / 9.0)
    conf = min(1.0, n_through / 7.0)
    v = 0.1 + data * (0.24 + 0.2 * base) + 0.56 * peak * conf + (0.03 + 0.05 * data) * jit
    return phi, np.clip(v, 0.04, 1.0), phc


# ----------------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------------

class Sphere(Scene):
    name = "sphere"
    towers = "auto"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.lay = Lay(ctx)
        self.body = Body()
        span = T1 - T0
        self.tracks = sphere_tracks(ctx.det)
        self.tr_t = np.array([tr["t"] for tr in self.tracks])
        self.ring_jit = hash01(np.arange(N_BINS), 77) - 0.5
        rng = np.random.default_rng(5)
        self.scan_order = rng.random(len(self.body.ea))          # draw-on order of the net at the start
        self.t_first = float(self.tr_t[0]) if len(self.tr_t) else T0
        # when the core is called, and when the picture is complete (10 s before the end)
        thr = [tr for tr in self.tracks if tr["through"]]
        self.thr_t = np.array([tr["t"] for tr in thr])
        self.thr_w = np.cumsum([0.25 if tr["echo"] else 1.0 for tr in thr]) if thr else np.zeros(0)
        t_core = T0 + 0.4 * span
        ok = (self.thr_w >= max(FOUND_W, self._w(t_core) + 3.0)) & (self.thr_t >= t_core)
        self.t_found = float(self.thr_t[np.argmax(ok)]) if ok.any() else T0 + 0.5 * span
        self.w_found = self._w(self.t_found)
        self.w_full = max(self._w(T1 - 10.0), self.w_found + 2.0)

    # ------------------------------------------------------------------ state
    def _past(self, t):
        return [tr for tr in self.tracks if tr["t"] <= t]

    def _w(self, t):
        """Weighted count of the tracks that crossed the core so far."""
        i = int(np.searchsorted(self.thr_t, t, side="right"))
        return float(self.thr_w[i - 1]) if i else 0.0

    def _conf(self, t):
        """(found, confidence 0..1): the search, then the picture firming up until 10 s before the end."""
        w = self._w(t)
        if t < self.t_found:
            return False, 0.42 * min(1.0, w / max(self.w_found, 1e-6))
        return True, 0.5 + 0.5 * min(1.0, (w - self.w_found) / (self.w_full - self.w_found))

    @staticmethod
    def _kicks(ctx, t, phi):
        """What the drums do to the picture at t: (envelope 0..1.2, ripple per halo bin, recent kicks)."""
        env = min(1.2, ctx.cues.kick(t, tau=0.16))
        kt, ka = ctx.cues.kicks(t - 1.3, t + 1e-6)
        d = np.abs(np.angle(np.exp(1j * (phi + np.pi / 2))))          # angular distance from the bottom of the halo
        rip = np.zeros(len(phi))
        for th, a in zip(kt, ka):
            age = t - float(th)
            rip += min(float(a), 1.6) * np.exp(-((d - 5.4 * age) / 0.34) ** 2) * math.exp(-age / 0.7)
        return env, np.clip(0.5 * env + 0.8 * rip, 0.0, 1.5), [(t - float(th), min(float(a), 1.6)) for th, a in zip(kt, ka)]

    # ----------------------------------------------------------------- render
    def draw(self, f, t, ctx):
        lay = self.lay
        cam, yaw = lay.camera(t, T0, T1 - T0)
        past = self._past(t)
        thr = [tr for tr in past if tr["through"]]
        found, conf = self._conf(t)
        n_prim = sum(1 for tr in past if not tr["echo"])
        phi, v, phc = ring_values(lay, cam, 7.0 * conf, 9.0 * min(1.0, n_prim / 4.0), 9, self.ring_jit)
        env, pulse, kicks = self._kicks(ctx, t, phi)
        appear = float(smoothstep(T0 - 0.1, T0 + 1.5, t))
        f.set_clip(*lay.clip)
        self._lattice(f, kick=env)
        self._body(f, cam, t, appear, gain=1.0 + 0.4 * env, swell=0.028 * min(1.0, env))
        self._shock(f, kicks, appear)
        self._ring(f, phi, v, appear, pulse=pulse)
        f.set_clip()
        self._tracks(f, cam, t, past, ctx)
        self._core(f, cam, t, thr, conf=conf if found else 0.0)
        self._callouts(f, cam, t, past, thr, yaw, found, conf)
        self._tomogram(f, t, past, thr, yaw, conf=conf, found=found)
        self._profile(f, phi, np.clip(v + 0.1 * pulse * appear, 0.0, 1.0))
        self._left(f, t, past, thr, found)
        self._right(f, t, ctx, past)
        self._spectrum(f, t, past)
        self._strip(f, t, ctx)
        self._groove(f, t, ctx, env)
        self._bottom(f, t, ctx, past, thr, found)
        return {"burst_gain": 0.75, "burst_size": 0.6}

    # ------------------------------------------------------------------ world
    def _lattice(self, f, kick=0.0):
        lay = self.lay
        x0, y0, x1, y1 = lay.clip
        step = 46.0
        kx = np.arange(math.ceil((x0 + 14 - lay.cx) / step), math.floor((x1 - 14 - lay.cx) / step) + 1)
        ky = np.arange(math.ceil((y0 + 14 - lay.cy) / step), math.floor((y1 - 14 - lay.cy) / step) + 1)
        KX, KY = np.meshgrid(kx, ky)
        X, Y = lay.cx + KX * step, lay.cy + KY * step
        far = np.hypot(X - lay.cx, Y - lay.cy) > lay.out + 26
        major = (KX % 4 == 0) & (KY % 4 == 0)
        f.pixels("w", X[far], Y[far], np.where(major, 0.95, 0.5)[far])
        m = far & major
        f.crosses("w", X[m], Y[m], 5.0 + 3.0 * min(1.0, kick), 0.42 + 0.5 * kick)

    def _shock(self, f, kicks, appear=1.0):
        """Each kick sends a dotted ring from the surface of the body out through the opacity histogram."""
        lay = self.lay
        r0, r1 = lay.R * 1.05, lay.out + 44.0 * lay.s
        for age, amp in kicks:
            if not (0.0 <= age < 0.55):
                continue
            u = age / 0.55
            r = r0 + (r1 - r0) * u ** 0.7
            n = int(2 * np.pi * r / 11.0)
            a = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.4 * age
            f.dots("w", lay.cx + r * np.cos(a), lay.cy + r * np.sin(a), 1.5, 1.2 * min(1.0, amp) * (1.0 - u) ** 1.3 * appear)

    def _body(self, f, cam, t, appear, gain=1.0, swell=0.0):
        """The plexus: surface net (rim lit), struts through the interior, vertices. `swell` = breath on a kick."""
        b = self.body
        P = b.surface(t).astype(np.float32) * np.float32(1.0 + swell)
        N = b.normals(t).astype(np.float32)
        sx, sy, _, _ = cam.project(P)
        view = unit(cam.pos.astype(np.float32)[None] - P)
        nv = (N * view).sum(1)
        rim = (0.1 + 0.8 * (1 - np.abs(nv)) ** 2.2) * np.where(nv > 0, 1.0, 0.45)
        ea, eb = b.ea, b.eb
        ev = b.e_var * gain
        em = np.ones(len(ea), bool)
        if appear < 1.0:                    # scanned in from the top at the start of the scene
            ymid = 0.5 * (P[ea, 1] + P[eb, 1])
            em = em & ((1.0 - ymid) / 2.0 + 0.08 * self.scan_order < appear * 1.1)
        f.segments("w", sx[ea][em], sy[ea][em], sx[eb][em], sy[eb][em], (rim[ea] * ev)[em], (rim[eb] * ev)[em])
        sa, sb = b.sa, b.sb
        sm = np.ones(len(sa), bool)
        if appear < 1.0:
            sm = sm & (hash01(np.arange(len(sa)), 3) < max(0.0, appear * 1.4 - 0.4))
        si = (0.085 + 0.1 * np.maximum(rim[sa], rim[sb])) * b.s_var * gain
        f.segments("w", sx[sa][sm], sy[sa][sm], sx[sb][sm], sy[sb][sm], si[sm])
        vm = np.ones(len(P), bool)
        if appear < 1.0:
            vm = vm & ((1.0 - P[:, 1]) / 2.0 < appear * 1.1)
        f.pixels("w", sx[vm], sy[vm], (0.3 + 0.65 * rim[vm]) * gain)

    def _ring(self, f, phi, v, appear, alive=None, gain=1.0, pulse=None):
        """The lollipop halo = polar opacity histogram on a precise base circle. `pulse` (per bin, 0..1.5)
        is the answer to the drums: the sticks jump and their heads brighten as the ripple passes."""
        lay = self.lay
        cx, cy, r0 = lay.cx, lay.cy, lay.r0
        Ln = lay.l0 + (lay.l1 - lay.l0) * v
        glow = 1.0
        if pulse is not None:
            Ln = Ln + 22.0 * lay.s * pulse
            glow = 1.0 + 0.7 * pulse
        c, s = np.cos(phi), -np.sin(phi)
        on = np.ones(N_BINS, bool) if alive is None else alive
        if appear < 1.0:
            on = on & (np.arange(N_BINS) / N_BINS < appear)
        x0, y0 = cx + r0 * c, cy + r0 * s
        x1, y1 = cx + (r0 + Ln) * c, cy + (r0 + Ln) * s
        hot = v > HOT
        a, b = on & ~hot, on & hot
        gl = np.broadcast_to(np.asarray(glow, np.float64), (N_BINS,))
        hd = 2.2 + 1.1 * min(1.0, lay.s)              # the heads shrink a little with the body
        f.segments("w", x0[a], y0[a], x1[a], y1[a], 0.8 * gain * gl[a], width=L.LW)
        f.segments("r", x0[b], y0[b], x1[b], y1[b], 1.15 * gain, width=L.LW_BOLD)
        f.dots("w", x1[a], y1[a], hd, 1.35 * gain * gl[a])
        f.dots("r", x1[b], y1[b], hd + 0.7, 1.8 * gain)
        if pulse is not None:               # red sticks answer in white too: a white head rides on them
            f.dots("w", x1[b], y1[b], 1.8, 1.2 * np.minimum(pulse[b], 1.0))
        # base circle, arc by arc (so it can be taken apart bin by bin)
        da = 2 * np.pi / N_BINS
        for k in np.nonzero(on)[0]:
            aa = phi[k] - da / 2 + np.linspace(0, da, 4)
            f.polyline("w", cx + r0 * np.cos(aa), cy - r0 * np.sin(aa), 0.6 * gain, width=L.LW)
        # dotted scale circles at 1.2 / 4.6 / 8 MWE
        for frac, inten in ((0.0, 0.32), (0.5, 0.36), (1.0, 0.6)):
            r = r0 + lay.l0 + (lay.l1 - lay.l0) * frac
            n = int(2 * np.pi * r / 7)
            aa = np.linspace(0, 2 * np.pi, n, endpoint=False)
            keep = on[np.minimum((aa / da).astype(int), N_BINS - 1)]
            f.pixels("w", (cx + r * np.cos(aa))[keep], (cy - r * np.sin(aa))[keep], inten * gain * 1.6)
        # inward ticks every 10 deg, long ones every 45 deg
        aa = np.radians(np.arange(0, 360, 10))
        ln = np.where(np.arange(36) % 9 == 0, 15.0, np.where(np.arange(36) % 3 == 0, 9.0, 5.0)) * min(1.0, lay.s)
        keep = on[np.minimum((aa / da).astype(int), N_BINS - 1)]
        f.segments("w", (cx + r0 * np.cos(aa))[keep], (cy - r0 * np.sin(aa))[keep],
                   (cx + (r0 - ln) * np.cos(aa))[keep], (cy - (r0 - ln) * np.sin(aa))[keep], 0.8 * gain)
        if gain > 0.5 and on.all():
            # angle labels: on the axis when they stay inside the column, else only above / below the halo
            if cx + lay.out + 44.0 <= lay.body[1] and cx - lay.out - 44.0 >= lay.body[0]:
                for deg, anchor, dx in ((0, "lm", 12), (180, "rm", -12)):
                    f.text("w", cx + math.cos(math.radians(deg)) * (lay.out + 4) + dx, cy + 5, f"{deg:03d}",
                           size=L.T_MICRO, alpha=0.75, anchor=anchor)
            else:
                f.text("w", cx, cy - lay.out - 16, "090", size=L.T_MICRO, alpha=0.75, anchor="ms")
                if lay.body[1] - lay.body[0] >= 2 * CORNER_W + 70.0:
                    f.text("w", cx, cy + lay.out + 28, "270", size=L.T_MICRO, alpha=0.75, anchor="ms")
            k = int(np.argmax(v))
            if v[k] > HOT:                  # value of the peak beside its stick, when that stays inside the column
                r = r0 + Ln[k] + 20
                x = cx + r * c[k]
                if lay.body[0] + 50 <= x <= lay.body[1] - 50:
                    f.text("r", x, cy + r * s[k] + 5, f"{float(mwe(v[k])):.2f}", size=L.T_SMALL, alpha=0.95,
                           anchor="lm" if c[k] >= 0 else "rm")

    def _draw_track(self, f, cam, tr, age, ctx, persist=1.0, leader=True):
        """One muon: leader from its detector, then the red track entering the body, kinking, leaving."""
        px, py, _, ok = cam.project(tr["path"])
        if not ok.all():
            return None
        echo = tr["echo"]
        dur = tr["dur"] * (0.6 if echo else 1.0)
        prog = min(1.0, age / dur)
        fresh = math.exp(-max(0.0, age - dur) / 0.7)
        base = (0.1 if echo else 0.24) * persist
        lvl = base + (0.45 if echo else 1.0) * fresh * (0.6 + 0.6 * tr["e"])
        wd = (2.7 if tr["heavy"] else 1.5) * (0.6 if echo else 1.0)
        seg = np.hypot(np.diff(px), np.diff(py))
        cum = np.r_[0, np.cumsum(seg)]
        reach_ = prog * cum[-1]
        inside = (False, True, True, False)
        xe, ye = float(px[0]), float(py[0])
        for k in range(4):
            if cum[k] >= reach_:
                break
            u = min(1.0, (reach_ - cum[k]) / max(seg[k], 1e-6))
            xe, ye = px[k] + (px[k + 1] - px[k]) * u, py[k] + (py[k + 1] - py[k]) * u
            f.segments("r", [px[k]], [py[k]], [xe], [ye], lvl * (1.45 if inside[k] else 0.7),
                       width=wd if inside[k] else max(1.0, wd * 0.6))
        if not echo:
            for k in (0, 3):                # tracker planes: ticks across the track outside the body
                Ls = seg[k]
                if Ls < 30:
                    continue
                ux, uy = (px[k + 1] - px[k]) / Ls, (py[k + 1] - py[k]) / Ls
                d = np.arange(16.0, Ls - 8, 30.0)
                d = d[cum[k] + d <= reach_]
                hx, hy = px[k] + ux * d, py[k] + uy * d
                f.segments("r", hx - uy * 6, hy + ux * 6, hx + uy * 6, hy - ux * 6, 0.25 * persist + 0.7 * fresh)
        if prog < 1.0:
            f.dots("r", [xe], [ye], 4.2 if tr["heavy"] else 3.0, 1.9)
            f.dots("w", [xe], [ye], 1.5, 1.1)
        if reach_ >= cum[2]:                # the point of closest approach
            hot = tr["through"] and tr["mrad"] > 28
            f.dots("w", px[2:3], py[2:3], 2.2 if not echo else 1.6, 1.2 * (0.5 + 0.5 * persist))
            if hot:
                f.dots("r", px[2:3], py[2:3], 3.6 if not echo else 2.4, 1.5 * (0.45 + 0.55 * fresh) * persist + 0.3)
                if fresh > 0.05 and not echo:
                    f.rings("r", px[2:3], py[2:3], [10 + 34 * (1 - fresh)], 0.9 * fresh, width=L.LW)
        # leader: detector -> up to the bus -> over -> down into the track (it may pass behind a tower)
        if leader and not echo and age < 1.9:
            tw = ctx.towers[tr["key"]]
            ox, oy = tw.det
            yb = BUS_Y[tr["key"]]
            a = math.exp(-age / 0.55)
            draw = min(1.0, age / 0.12)
            xs = [ox, ox, float(px[0]), float(px[0])]
            ys = [oy - 14, yb, yb, float(py[0])]
            f.polyline("r", xs, ys, 1.1 * a * draw, width=L.LW)
            f.dots("r", [float(px[0])], [yb], 3.0, 1.4 * a)
        return px, py

    def _tracks(self, f, cam, t, past, ctx):
        """All the muons so far (their PoCA dots stay: they are the picture; the lines fade, see persistence)."""
        keep = persistence(past, t)
        for tr in past:
            self._draw_track(f, cam, tr, t - tr["t"], ctx, persist=keep[tr["id"]])

    def _core(self, f, cam, t, thr, gain=1.0, conf=None):
        """The dense core, as the scattering gives it away: a red circle that firms up with every PoCA.
        `conf` overrides the default confidence (3 scattered tracks to show, 7 to be sure)."""
        conf = ((min(1.0, len(thr) / 7.0) if len(thr) >= 3 else 0.0) if conf is None else conf) * gain
        # slice through the core height: front half bright, back half faint
        sl = self.body.slice(float(CORE[1]), t).astype(np.float32)
        lx, ly, lz, _ = cam.project(sl)
        li = np.where(lz < np.median(lz), 0.75, 0.2) * gain
        f.segments("w", lx, ly, np.roll(lx, -1), np.roll(ly, -1), li, np.roll(li, -1), width=L.LW)
        if conf <= 0.0:
            return
        cx, cy, cz, _ = cam.project(CORE[None].astype(np.float32))
        rc = CORE_R * cam.focal / float(cz[0])
        n = int(40 + 90 * conf)
        a = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.15 * t
        f.dots("r", float(cx[0]) + rc * np.cos(a), float(cy[0]) + rc * np.sin(a), 1.5, 0.5 + 0.7 * conf)
        if conf > 0.6:
            f.crosses("r", cx, cy, 10.0, 0.9 * conf, width=L.LW)

    # --------------------------------------------- callouts of the body column
    def _view_block(self, f, title, line, erode=0.0, fr=0):
        """Top left of the body column: the view tag (as much of the title as fits) and one line."""
        x0, x1 = self.lay.body
        title = fit_title(title, min(x1 - x0, 344.0), L.T_SMALL)
        f.tag("w", x0 + 4, Y_PANEL, er(title, erode, 91, fr), size=L.T_SMALL, pad=4, alpha=fade(erode))
        f.text("w", x0 + 4, Y_PANEL + 34, er(fit_fields(line, x1 - x0 - 8), erode, 92, fr), size=L.T_MICRO, alpha=0.75)

    def _corner(self, f, side, tag, lines, red=True, alpha=1.0, tag_size=L.T_TAG, tag_alpha=None):
        """Callout in a bottom corner of the body column (side -1 = left, +1 = right): an inverted tag with
        short lines stacked under it, never wider than CORNER_W so it stays outside the halo."""
        x0, x1 = self.lay.body
        if side > 0 and not self.lay.two_corners:
            return
        x, anchor = (x0 + 5, "ls") if side < 0 else (x1 - 5, "rs")
        if tag:
            f.tag("r" if red else "w", x, Y_CORNER, tag, size=tag_size, pad=5,
                  alpha=alpha if tag_alpha is None else tag_alpha, anchor=anchor)
        for k, (ln, layer) in enumerate(lines):
            f.text(layer, x + (-5 if side < 0 else 5), Y_CORNER + 30 + k * 22, ln, size=L.T_MICRO, alpha=0.9 * alpha,
                   anchor=anchor)

    def _callouts(self, f, cam, t, past, thr, yaw, found, conf):
        lay = self.lay
        self._view_block(f, "VIEW 01 // SCATTERING TOMOGRAPHY",
                         [f"ORBIT {math.degrees(yaw) % 360:05.1f} DEG", f" ELEV {math.degrees(math.atan2(CAM_H, CAM_D)):.1f}"])
        # bottom left: the anomaly, once enough tracks have crossed it
        if found:
            age = t - self.t_found
            cx, cy, _, _ = cam.project(CORE[None].astype(np.float32))
            ex = lay.body[0] + 5 + text_w("ANOMALY", L.T_TAG) + 12
            f.segments("w", [float(cx[0]), ex + 26], [float(cy[0]), Y_CORNER - 8], [ex + 26, ex], [Y_CORNER - 8] * 2, 0.6,
                       width=L.LW_HAIR)
            lines = ["RHO 11.3 G/CM3  Z~82", f"R {CORE_R:.2f} M  DEPTH {1 - np.linalg.norm(CORE):.2f} M",
                     f"POCA {len(thr):02d}  CONF {conf:.2f}"]
            self._corner(f, -1, "ANOMALY", [(hud.typed(ln, age, delay=0.15 * k), "w") for k, ln in enumerate(lines)])
        else:
            self._corner(f, -1, "SEARCHING", [(f"POCA {len(thr):02d}  CONF {conf:.2f}", "w"), ("NO ANOMALY YET", "w")],
                         red=False, alpha=0.6 + 0.4 * (int(t * 2) % 2))
        # bottom right: the muon that just came in
        prim = [tr for tr in past if not tr["echo"]]
        if prim:
            tr = prim[-1]
            age = t - tr["t"]
            if age < 6.0:
                al = 1.0 - float(smoothstep(5.0, 6.0, age))
                K = tr["K"]
                lines = [f"P {tr['p']:.3f} GEV/C", f"THETA {tr['mrad']:4.1f} MRAD", f"POCA {K[0]:+.2f} {K[1]:+.2f} {K[2]:+.2f}"]
                self._corner(f, 1, f"MU{tr['charge']} // {L.NAMES[tr['key']]} // #{tr['id']:03d}",
                             [(hud.typed(ln, age, cps=140.0, delay=0.05 + 0.08 * k), "r" if k == 1 and tr["through"] else "w")
                              for k, ln in enumerate(lines)], alpha=al, tag_size=L.T_SMALL)

    # ---------------------------------------------------------------- tomogram
    def _tomogram(self, f, t, past, thr, yaw, erode=0.0, alive=None, title="TOMOGRAM // TOP VIEW // SLICE THROUGH THE CORE",
                  erode_txt=None, conf=None, found=None):
        """Top view of the slice through the core, in the tomogram column. `erode` takes the picture apart
        (lattice, contour, voxels), `erode_txt` its lettering. `conf` / `found` override the default
        confidence (track count / 7, core shown from 3 tracks)."""
        lay = self.lay
        if not lay.tg:
            return
        x0, x1 = lay.tomo
        fr = int(t * 30)
        et = erode if erode_txt is None else erode_txt
        hud.panel_header(f, x0, x1, Y_PANEL, er(fit_title(title, x1 - x0 - 8), et, 3, fr), alpha=fade(et))
        S, Hh = lay.tg["S"], TOMO_HALF
        px0, py0 = lay.tg["px0"], lay.tg["py0"]
        px1, py1 = px0 + lay.tg["side"], py0 + lay.tg["side"]
        cx, cy = 0.5 * (px0 + px1), 0.5 * (py0 + py1)
        X = lambda x: cx + np.asarray(x) * S
        Y = lambda z: cy + np.asarray(z) * S
        g = 1.0 - erode
        # voxel lattice + axes in metres
        k = np.arange(-int(Hh / VOX), int(Hh / VOX) + 1)
        KX, KY = np.meshgrid(k, k)
        keep = np.ones(KX.shape, bool) if erode <= 0 else hash01(KX, KY, 5) > erode
        major = (KX % 4 == 0) & (KY % 4 == 0)
        f.pixels("w", X(KX * VOX)[keep], Y(KY * VOX)[keep], np.where(major, 0.95, 0.5)[keep])
        f.rect("w", px0, py0, px1, py1, 0.45 * g)
        hud.ruler(f, px0, px1, py1, -Hh, Hh, 0.125, 0.5 if S >= 150 else 1.0,
                  fmt=(lambda v: er(f"{v:+.1f}", et, 7, fr) if abs(v) < Hh - 0.1 else ""), down=True, inten=0.7 * g, lab_dy=30)
        hud.vruler(f, px0, py0, py1, -Hh, Hh, 0.125, 0.5, right=False, inten=0.7 * g)
        f.text("w", px0 - 20, py1 + 30, er("X / M", et, 8, fr), size=L.T_MICRO, alpha=0.6, anchor="rs")
        f.text("w", px0 - 20, py0 + 16, er("Z / M", et, 9, fr), size=L.T_MICRO, alpha=0.6, anchor="rs")
        # body contour in the slice plane
        sl = self.body.slice(float(CORE[1]), t)
        m = np.ones(len(sl), bool) if alive is None else alive
        xs, ys = X(sl[:, 0]), Y(sl[:, 2])
        xn, yn = np.roll(xs, -1), np.roll(ys, -1)
        f.segments("w", xs[m], ys[m], xn[m], yn[m], 0.85 * max(g, 0.4), width=L.LW)
        # camera direction of the 3D view
        ca = math.atan2(math.cos(yaw), math.sin(yaw))        # camera sits at (sin yaw, cos yaw) in x, z
        rr = 1.1
        f.segments("w", [float(X(rr * math.cos(ca)))], [float(Y(rr * math.sin(ca)))],
                   [float(X((rr + 0.1) * math.cos(ca)))], [float(Y((rr + 0.1) * math.sin(ca)))], 0.9 * g, width=L.LW_BOLD)
        lx = float(X((rr + 0.12) * math.cos(ca)))
        f.text("w", min(lx + 6, px1 - 34), float(Y((rr + 0.12) * math.sin(ca))) + 5, er("CAM", et, 10, fr), size=L.T_MICRO,
               alpha=0.7)
        # tracks of the last seconds, projected
        f.set_clip(px0, py0, px1, py1)
        for tr in past[-5:]:
            age = t - tr["t"]
            if age > 6.0 or erode > 0.3:
                continue
            a = math.exp(-age / 2.2) * (0.45 if tr["echo"] else 1.0)
            p = tr["path"]
            f.segments("r", X(p[1:3, 0]), Y(p[1:3, 2]), X(p[2:4, 0]), Y(p[2:4, 2]), 0.75 * a, width=L.LW)
        f.set_clip()
        # hot voxels + PoCA points
        cells = {}
        for tr in thr:
            if abs(tr["K"][1] - CORE[1]) < 0.3:
                key = (int(math.floor(tr["K"][0] / VOX)), int(math.floor(tr["K"][2] / VOX)))
                cells[key] = cells.get(key, 0) + 1
        vs = VOX * S
        for (ix, iz), n in cells.items():
            if erode > 0 and hash01(ix, iz, 11) < erode:
                continue
            a, b = float(X(ix * VOX)), float(Y(iz * VOX))
            f.rect("r", a + 2, b + 2, a + vs - 2, b + vs - 2, 0.85)
            if n >= 2:
                f.rects("r", a + 4, b + 4, a + vs - 4, b + vs - 4, 0.4 + 0.2 * min(n, 4))
        dot = 0.75 + 0.25 * min(1.0, S / 212.0)
        for tr in past:
            if erode > 0 and hash01(tr["id"], 13) < erode:
                continue
            hot = tr["through"] and tr["mrad"] > 28
            f.dots("r" if hot else "w", [float(X(tr["K"][0]))], [float(Y(tr["K"][2]))], (3.2 if hot else 2.0) * dot,
                   1.5 if hot else 0.9)
        # the estimate closing in on the core
        n = len(thr)
        readout = ["EST X  -.---", "EST Z  -.---", "SIGMA  -.---", "CONF   0.00"]
        if n >= 2 and erode < 0.9:
            Kp = np.array([tr["K"] for tr in thr])
            mx, mz = float(Kp[:, 0].mean()), float(Kp[:, 2].mean())
            sig = float(np.sqrt(Kp[:, [0, 2]].var(0).sum()) / math.sqrt(n)) + 0.02
            ex, ez = float(X(mx)), float(Y(mz))
            f.rings("r", [ex], [ez], [max(8.0, 2.2 * sig * S)], 0.9 * g, width=L.LW)
            f.set_clip(px0, py0, px1, py1)
            f.segments("r", [px0, ex], [ez, py0], [px1, ex], [ez, py1], 0.4 * g)
            f.set_clip()
            f.tag("r", min(ex + 6, px1 - 64), py0 - 8, er(f"{mx:+.3f}", et, 15, fr), size=L.T_MICRO, pad=3, alpha=fade(et))
            f.tag("r", px0 + 8, ez - 8, er(f"{mz:+.3f}", et, 16, fr), size=L.T_MICRO, pad=3, alpha=fade(et))
            readout = [f"EST X  {mx:+.3f}", f"EST Z  {mz:+.3f}", f"SIGMA  {sig:.3f}",
                       f"CONF   {(min(1.0, n / 7.0) if conf is None else conf):.2f}"]
        cf = min(1.0, n / 7.0) if conf is None else conf
        if (n >= 3 if found is None else found) and erode < 0.6:          # the true core, dotted
            a = np.linspace(0, 2 * np.pi, 60, endpoint=False)
            f.pixels("w", X(CORE[0] + CORE_R * np.cos(a)), Y(CORE[2] + CORE_R * np.sin(a)), 1.2 * cf * g)
        # readouts: beside the plot, or under it in two columns when the column is narrow
        lines = [f"VOXEL  {VOX:.3f} M", f"POCA   {len(past):03d}", f"CORE   {n:03d}"] + readout
        for k, ln in enumerate(lines):
            if lay.tg["mode"] == "side":
                x, y = x0, py0 + 42 + k * 24
            else:
                x, y = (x0 if k < 3 else x0 + max(150.0, 0.5 * (x1 - x0))), py1 + 62 + (k if k < 3 else k - 3) * 22
            f.text("r" if k >= 3 and n >= 2 else "w", x, y, er(ln, et, 20 + k, fr), size=L.T_MICRO, alpha=0.85)

    def _low_panel(self, f, col, title, erode=0.0, key=31, fr=0, cap=None):
        """Header of a panel in the lower part of a column (y = Y_LOW .. Y_BASE). Returns its x range."""
        x0, x1 = col
        if cap:
            x1 = min(x1, x0 + cap)
        hud.panel_header(f, x0, x1, Y_LOW, er(fit_title(title, x1 - x0 - 8), erode, key, fr), alpha=fade(erode))
        return x0, x1

    def _profile(self, f, phi, v, alive=None, erode=0.0, t=0.0):
        """The opacity histogram, unrolled under the tomogram: one bar per bin, flagged bins red."""
        lay = self.lay
        if not lay.tomo:
            return
        fr = int(t * 30)
        x0, x1 = self._low_panel(f, lay.tomo, f"OPACITY_PROFILE // MWE // {N_BINS} BINS // UNROLLED", erode, 31, fr)
        yb, hmax = Y_BASE, 150.0
        on = np.ones(N_BINS, bool) if alive is None else alive
        n = N_BINS
        if (x1 - x0 - 70) / n < 3.2:        # narrow column: pair the bins
            v, on, n = np.maximum(v[0::2], v[1::2]), on[0::2] | on[1::2], N_BINS // 2
        bw = (x1 - x0 - 70) / n
        xs = x0 + 60 + np.arange(n) * bw
        hot = v > HOT
        a, b = on & ~hot, on & hot
        f.rects("w", xs[a], yb - v[a] * hmax, xs[a] + max(2.0, bw - 2.2), yb, 0.92)
        f.rects("r", xs[b], yb - v[b] * hmax, xs[b] + max(2.0, bw - 2.2), yb, 1.0)
        f.segments("w", [x0 + 54], [yb - hmax - 6], [x0 + 54], [yb + 4], 0.6 * (1 - erode))
        for frac in (0.0, 0.5, 1.0):
            y = yb - frac * hmax
            f.segments("w", [x0 + 46], [y], [x0 + 54], [y], 0.8 * (1 - erode))
            f.text("w", x0, y + 5, er(f"{float(mwe(frac)):.1f}", erode, 33, fr), size=L.T_MICRO, alpha=0.7)
        yt = yb - HOT * hmax
        f.segments("r", [x0 + 54], [yt], [x1], [yt], 0.55 * (1 - erode))
        f.text("r", x0 + 62, yt - 8, er(f"FLAG > {float(mwe(HOT)):.2f} MWE", erode, 34, fr), size=L.T_MICRO,
               alpha=0.9)
        if hot.any() and x1 - x0 >= 420:    # above the tallest bar, clear of the header rule
            f.text("r", x1, yb - hmax - 12, er(f"PEAK {float(mwe(v.max())):.2f} MWE", erode, 30, fr), size=L.T_MICRO,
                   alpha=0.95, anchor="rs")
        step = 8 if x1 - x0 >= 520 else 4
        for k in range(0, step + 1):
            xx = x0 + 60 + k * (x1 - x0 - 70) / step
            f.segments("w", [xx], [yb + 2], [xx], [yb + 10], 0.8 * (1 - erode))
            if k < step:
                f.text("w", xx + 3, yb + 26, er(f"{k * 360 // step:03d}", erode, 35 + k, fr), size=L.T_MICRO, alpha=0.7)

    # ------------------------------------------------------- text columns
    def _info_block(self, f, title, rows, red=(), erode=0.0, fr=0, key=111):
        """Title tag + parameter rows at the top of the info column. Returns (x, width, y under the block)."""
        x0, x1 = self.lay.info
        w = x1 - x0
        size = fit_size(title, w - 30.0, 24.0, 46.0)
        f.tag("w", x0 + 10, Y_TITLE, er(title, erode, key, fr), size=size, pad=9, alpha=fade(erode))
        rs = L.T_SMALL if text_w("X" * max(len(r) for r in rows), L.T_SMALL) <= w - 4 else L.T_MICRO
        for k, ln in enumerate(rows):
            f.text("r" if k in red else "w", x0 + 4, Y_TITLE + 52 + k * 25.5, er(ln, erode, key + 1 + k, fr), size=rs,
                   alpha=0.9)
        return x0 + 4, w - 8, Y_TITLE + 52 + len(rows) * 25.5 + 30

    def _log_block(self, f, x, w, y, tag, head, lines, erode=0.0, fr=0, key=120):
        """A log under the info block: small tag, header line, then rows = (fields, layer, alpha). The fields
        that do not fit the column are dropped (time / detector / energy always stay)."""
        size = L.T_SMALL if w >= 412.0 else L.T_MICRO
        lead = 1.5 * size
        f.tag("w", x + 3, y, er(fit_title(tag, w - 6), erode, key, fr), size=L.T_MICRO, pad=3, alpha=fade(erode))
        f.text("w", x, y + 30, er(fit_fields(head, w, size), erode, key + 1, fr), size=size, alpha=0.55)
        for row, (fields, layer, alpha, typed_age) in enumerate(lines[: int((Y_BASE + 4 - (y + 54)) / lead) + 1]):
            s = fit_fields(fields, w, size)
            if typed_age is not None:
                s = hud.typed(s, typed_age, cps=260.0)
            f.text(layer, x, y + 54 + row * lead, er(s, erode, key + 10 + row, fr), size=size, alpha=alpha)

    def _left(self, f, t, past, thr, found):
        if not self.lay.info:
            return
        expo = max(0.0, t - self.t_first)
        rows_ = ["TARGET    SPHERE_01", "RADIUS    1.000 M", f"EXPOSURE  00:{int(expo // 60):02d}:{expo % 60:05.2f}",
                 f"TRACKS    {len(past):04d}", f"SCATTERED {len(thr):04d}", "RHO_BODY  2.65 G/CM3",
                 "RHO_CORE  11.3 G/CM3" if found else "RHO_CORE  --.- G/CM3"]
        x, w, y = self._info_block(f, "MUOGRAPHY", rows_, red=(6,) if found else ())
        lines = []
        for row, tr in enumerate(past[::-1][:24]):
            K = tr["K"]
            m = int(tr["t"] // 60)
            age = t - tr["t"]
            red = (tr["through"] and tr["mrad"] > 28) or age < 1.2
            fields = [f"{m:02d}:{tr['t'] - 60 * m:04.1f}", f"{tr['key']}{'e' if tr['echo'] else ' '}", f"{tr['e']:.2f}",
                      f"{tr['mrad']:5.1f}", f"{K[0]:+.2f}", f"{K[1]:+.2f}", f"{K[2]:+.2f}"]
            lines.append((fields, "r" if red else "w",
                          (0.95 if row < 2 or red else 0.62) * (0.7 if tr["echo"] and age > 1.2 else 1.0), age))
        if not past:
            lines = [(["WAITING FOR THE FIRST MUON" + ("_" if int(t * 2) % 2 else "")], "w", 0.8, None)]
        self._log_block(f, x, w, y, "TRACK_LOG // POCA // LIVE",
                        ["TIME   ", "D ", "E   ", " MRAD", "X    ", "Y    ", "Z"], lines)

    def _right(self, f, t, ctx, past, title="DETECTORS", erode=0.0):
        """The three streams: hit counts, last energy, the float each tower sends, its last hits as bars."""
        if not self.lay.dets:
            return
        x0, x1 = self.lay.dets
        x1 = min(x1, x0 + 460.0)
        fr = int(t * 30)
        y = Y_TITLE
        f.tag("r", x0 + 10, y, er(title, erode, 51, fr), size=fit_size(title, x1 - x0 - 30.0, 24.0, 46.0), pad=9,
              alpha=fade(erode))
        f.text("w", x0 + 4, y + 44, er("3 DETECTORS // 3 FLOATS // OSC", erode, 52, fr), size=L.T_MICRO, alpha=0.75)
        yy = y + 92
        for k, key in enumerate(L.ORDER):
            tt, ee, ec = ctx.det.hits(key, 0.0, t + 1e-6)
            age, e = ctx.det.last(key, t, echoes=True)
            hot = age < 0.7
            f.rects("w", x0, yy - 22, x1, yy - 19, 0.9 * fade(erode))
            f.tag("r" if hot else "w", x0 + 3, yy + 8, er(L.NAMES[key], erode, 53 + k, fr), size=L.T_LABEL, pad=4,
                  alpha=fade(erode))
            n = int((~ec).sum())
            val = float(ctx.det.value(key, t))
            lines = [f"HITS {n:03d}   ECHO {int(ec.sum()):03d}", f"LAST E {e:.3f}   T+{min(age, 99.9):04.1f}" if age < 90
                     else "LAST E -.---", f"/MUON/{key} {val:.3f}"]
            for j, ln in enumerate(lines):
                f.text("w", x0 + 110, yy - 2 + j * 22, er(ln, erode, 60 + 3 * k + j, fr), size=L.T_MICRO,
                       alpha=0.85 if j < 2 else 0.6)
            # the last hits of this tower as energy bars (newest right)
            m = 26
            bx = x0 + np.arange(m) * ((x1 - x0) / m)
            sel = slice(max(0, len(tt) - m), len(tt))
            es, cs = ee[sel], ec[sel]
            off = m - len(es)
            yb = yy + 112
            f.segments("w", [x0], [yb + 1], [x1], [yb + 1], 0.35 * (1 - erode))
            if len(es):
                keep = np.ones(len(es), bool) if erode <= 0 else hash01(np.arange(len(es)), k, 9) > erode
                xb = bx[off:]
                f.rects("r", xb[keep & ~cs], yb - 6 - 44 * es[keep & ~cs], xb[keep & ~cs] + 9, yb, 0.95)
                f.rects("w", xb[keep & cs], yb - 4 - 44 * es[keep & cs], xb[keep & cs] + 5, yb, 0.6)
            yy += 176

    def _spectrum(self, f, t, past, erode=0.0):
        """Scattering-angle spectrum of the live tracks (under the detectors)."""
        if not self.lay.spec:
            return
        fr = int(t * 30)
        x0, x1 = self._low_panel(f, self.lay.spec, "THETA_SCATTER // MRAD", erode, 71, fr, cap=460.0)
        mr = np.array([tr["mrad"] for tr in past]) if past else np.zeros(0)
        edges = np.linspace(0, 60, 25)
        cnt, _ = np.histogram(mr, edges)
        hgt = 150.0 * np.sqrt(cnt / max(cnt.max(), 1)) if len(mr) else np.zeros(24)
        bw = (x1 - x0 - 8) / 24
        xs = x0 + 4 + np.arange(24) * bw
        tail = edges[:-1] >= 28
        m = cnt > 0
        f.rects("w", xs[m & ~tail], Y_BASE - hgt[m & ~tail], xs[m & ~tail] + bw - 4, Y_BASE, 0.95)
        f.rects("r", xs[m & tail], Y_BASE - hgt[m & tail], xs[m & tail] + bw - 4, Y_BASE, 0.95)
        hud.ruler(f, x0 + 4, x0 + 4 + 24 * bw, Y_BASE + 2, 0, 60, 2.5, 10 if x1 - x0 >= 300 else 20,
                  fmt=lambda v: er(f"{v:.0f}", erode, 72, fr), inten=0.6 * (1 - erode), lab_dy=26)

    # --------------------------------------------------------- header band
    def _strip(self, f, t, ctx, title=None, t0=T0, t1=T1, erode=0.0, curve=None):
        """Score strip, top left (the header band is above every tower): loudness comb, one tick per onset."""
        fr = int(t * 30)
        if title is None:
            sec = sd.section_at(t)[1]
            title = f"TRACK_TIMELINE // {sec[0]} {sec[1]} // ONE TICK = ONE MUON"
        x0, y0, x1, y1, yb = hud.strip_base(f, title=er(title, erode, 81, fr), band=False,
                                            ticks=(t0, t1, 1.0, 10.0), alpha=float(1.0 - smoothstep(0.5, 0.96, erode)))
        X = lambda tt: x0 + (np.asarray(tt, np.float64) - t0) / (t1 - t0) * (x1 - x0)
        n = int((x1 - x0) / 5)
        lv = ctx.cues.loud_curve(t0, t1, n) if curve is None else curve(t0 + (np.arange(n) + 0.5) / n * (t1 - t0))
        tb = t0 + (np.arange(n) + 0.5) / n * (t1 - t0)
        xb = X(tb)
        keep = np.ones(n, bool) if erode <= 0 else hash01(np.arange(n), 83) > erode
        yc = y0 + 28
        past_ = (tb <= t) & keep
        f.rects("w", xb[past_], yc, xb[past_] + 2, yc + 2 + 26 * lv[past_] ** 1.4, 0.9)
        fut = (tb > t) & keep
        f.rects("w", xb[fut], yc, xb[fut] + 2, yc + 2 + 26 * lv[fut] ** 1.4, 0.22)
        for tv in np.arange(t0, t1 - 0.1, 10.0):
            f.text("w", float(X(tv)) + 5, y0 + 24, er(sd.tc(tv)[:5], erode, 86, fr), size=L.T_MICRO, alpha=0.6)
        lanes = {"L": y0 + 78, "C": y0 + 98, "R": y0 + 118}
        for key, yl in lanes.items():
            f.segments("w", [x0 + 22], [yl], [x1], [yl], 0.22 * (1 - erode))
            f.text("w", x0 + 5, yl + 1, er(key, erode, 84, fr), size=L.T_MICRO, alpha=0.75)
            tt, ee, ec = ctx.det.hits(key, t0, min(t, t1) + 1e-6)
            if len(tt):
                k2 = np.ones(len(tt), bool) if erode <= 0 else hash01(np.arange(len(tt)), 85) > erode
                xh = X(tt)
                f.rects("r", xh[~ec & k2], yl - 4 - 13 * ee[~ec & k2], xh[~ec & k2] + 3, yl, 1.0)
                f.rects("r", xh[ec & k2], yl - 2 - 9 * ee[ec & k2], xh[ec & k2] + 2, yl, 0.55)
        hud.strip_cursor(f, float(X(min(t, t1))), y0, y1 - 14, sd.tc(t))

    def _groove(self, f, t, ctx, env, span=8.0):
        """Top right (the subtitle box is absent in this slot): the kicks of the last seconds in white, the
        detector onsets under them in red - the drums and the muons on one grid."""
        x0, y0, x1, y1 = ctx.sub
        if hud.subtitle_box_alpha(t) > 0.01 or x1 - x0 < 420:      # somebody speaks after all / no room
            return
        sec = sd.section_at(t)[1]
        ix0, ix1 = x0 + 16, x1 - 16
        wide = x1 - x0 >= 760
        f.tag("w", ix0, y0 + 25, f"KICK_GRID // {sec[0]} {sec[1]} // LAST {span:.0f} S" if wide else "KICK_GRID",
              size=L.T_MICRO, pad=3)
        yb = y0 + 96
        X = lambda tt: ix0 + (np.asarray(tt, np.float64) - (t - span)) / span * (ix1 - ix0)
        f.segments("w", [ix0], [yb], [ix1], [yb], 0.8)
        hud.ruler(f, ix0, ix1, yb, t - span, t, 0.25, 1.0, down=True, inten=0.6)
        n = int((ix1 - ix0) / 4)
        tb = t - span + (np.arange(n) + 0.5) / n * span
        lv = ctx.cues.loud_curve(t - span, t, n)
        f.polyline("w", X(tb), yb - 4 - 46 * lv ** 1.3, 0.55, width=L.LW_HAIR)
        kt, ka = ctx.cues.kicks(t - span, t + 1e-6)
        if len(kt):
            xk = X(kt)
            f.rects("w", xk - 2, yb - 10 - 30 * np.minimum(ka, 1.5), xk + 3, yb, 0.95)
        for row, key in enumerate(L.ORDER):
            yl = yb + 24 + row * 20
            f.text("w", ix0, yl, key, size=L.T_MICRO, alpha=0.7)
            tt, ee, ec = ctx.det.hits(key, t - span, t + 1e-6)
            if len(tt):
                xh = X(tt)
                ok = xh >= ix0 + 22                  # keep the row label clear
                f.rects("r", xh[~ec & ok], yl - 4 - 10 * ee[~ec & ok], xh[~ec & ok] + 3, yl, 1.0)
                f.rects("r", xh[ec & ok], yl - 2 - 7 * ee[ec & ok], xh[ec & ok] + 2, yl, 0.55)
        f.segments("r", [ix1], [y0 + 44], [ix1], [y1 - 8], 1.1, width=L.LW)
        f.text("w", ix1 - 8, y0 + 25, f"KICK {min(env, 9.99):.2f}   LOUD {ctx.cues.loud(t):.2f}", size=L.T_MICRO, alpha=0.8,
               anchor="rs")
        f.rects("w", ix1 - 292, y0 + 32, ix1 - 292 + 284 * min(1.0, env), y0 + 37, 0.95)

    # --------------------------------------------------------- bottom band
    def _numbers_panel(self, f, title, cols, erode=0.0, fr=0, key=151):
        """Bottom band, widest free panel: up to three counters = (label, value, layer)."""
        lay = self.lay
        if not lay.p_num:
            return
        x0, x1 = lay.p_num
        x1 = min(x1, x0 + 620.0)
        hud.panel_header(f, x0, x1, lay.y0, er(title, erode, key, fr), alpha=fade(erode))
        cols = cols[: max(1, min(len(cols), int((x1 - x0) / 125.0)))]
        cw = (x1 - x0) / len(cols)
        for k, (lab, val, layer) in enumerate(cols):
            f.text("w", x0 + k * cw + 4, lay.y0 + 34, er(lab, erode, key + 1 + k, fr), size=L.T_MICRO, alpha=0.75)
            f.text(layer, x0 + k * cw + 2, lay.y0 + 96, er(val, erode * 0.5, key + 4 + k, fr),
                   size=fit_size(val, cw - 12.0, 26.0, 48.0), alpha=0.97)

    def _barcode_panel(self, f, t, ctx, erode=0.0, fr=0):
        """Bottom band: the three streams of the last 8 s as a barcode (quiet = sparse, hit = solid)."""
        lay = self.lay
        if not lay.p_bar:
            return
        x0, x1 = lay.p_bar
        hud.panel_header(f, x0, x1, lay.y0, er(fit_title("STREAM_BARCODE // L C R // LAST 8 S", x1 - x0 - 8), erode, 158, fr),
                         alpha=fade(erode))
        cols_n = int(np.clip((x1 - x0) / 3.2, 60, 150))
        dt = 8.0 / cols_n
        kk = math.floor((t - 8.0) / dt) + np.arange(cols_n)
        cw = (x1 - x0) / cols_n
        xs = x0 + np.arange(cols_n) * cw
        lane_h = (lay.y1 - lay.y0 - 12) / 3
        for ln, key in enumerate(L.ORDER):
            val = ctx.det.value(key, kk * dt + dt)
            on = hash01(kk, ln + 17) < 0.03 + 1.6 * val
            if erode > 0:
                on = on & (hash01(kk, ln, 159) > erode)
            yl = lay.y0 + 12 + ln * lane_h
            f.rects("w", xs[on], yl, xs[on] + cw, yl + lane_h - 4, 0.95)
        f.segments("r", [x1 - 1], [lay.y0 + 8], [x1 - 1], [lay.y1], 1.2 * fade(erode), width=L.LW)

    def _single_panel(self, f, title, value, layer="r", erode=0.0, fr=0):
        """Bottom band, smallest panel: one big figure."""
        lay = self.lay
        if not lay.p_one:
            return
        x0, x1 = lay.p_one
        hud.panel_header(f, x0, x1, lay.y0, er(title, erode, 160, fr), alpha=fade(erode))
        f.text(layer, x0, lay.y0 + 96, value, size=fit_size(value, x1 - x0 - 8.0, 26.0, 48.0), alpha=0.97)

    def _bottom(self, f, t, ctx, past, thr, found):
        expo = max(0.0, t - self.t_first)
        self._numbers_panel(f, "RECONSTRUCTED", [("TRACKS", f"{len(past):04d}", "w"),
                                                 ("SCATTERED", f"{len(thr):04d}", "r" if thr else "w"),
                                                 ("EXPOSURE", f"{int(expo // 60):02d}:{int(expo % 60):02d}", "w")])
        self._barcode_panel(f, t, ctx)
        self._single_panel(f, "RHO_CORE", "11.3" if found else "--.-", "r" if found else "w")
