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
         The hits are taken as they come, never read ahead (Sphere._update): the detectors are live on
         site. The core is called when enough tracks have crossed it, whenever that is.
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

NOTHING THAT SHOWS DATA FADES IN OR POPS IN. When the scene starts its furniture is constructed, one
block after the other (BLK, Sphere._blk): strip, kick grid, text columns, tomogram, profile, spectrum,
bottom panels - each inside a Frame.build block (see build.py). What appears later is made when it
appears: on the hit that calls the core the SEARCHING callout is taken apart, a leader is drawn from
the core, the ANOMALY callout is made, the red circle is traced and the density figures spin and lock;
the estimate of the tomogram is made when the tracks give one; the PEAK labels when bins get flagged;
the tag of every new muon when it arrives (taken apart six seconds later, or when the next one comes).
The body, the halo, the tracks and the shock rings are the image: they keep their own ramps.
Every block is drawn in two passes: its boxes and rules (Sphere._blk(..., erode=)) and its content
(Sphere._blk(...)) - the same here, but DISINTEGRATE takes the boxes and rules apart again as the text
erodes, and its tags break up with their text (etag) instead of fading.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import engine as E
from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import CHAR_W, Camera, font, hash01, smoothstep, text_w
from ..show import Scene


def _span(look="sphere", default=(236.0, 276.0)):
    """Start / end of the slot the sheet gives this look."""
    secs = [x for x in sd.SECTIONS if x[4] == look]
    return (secs[0][2], secs[-1][3]) if secs else default


T0, T1 = _span()
FOUND_W = 4.5                     # weighted scattered tracks (primary 1, echo 0.25) it takes to call the core
# The detectors are live on site: a hit is only known once it has happened, so nothing here may depend on the
# hits to come. What used to be measured on the whole scene in advance is a figure now (the ones the muon stem
# of the previews gave):
SEARCH_W = 13.5                   # the search: the confidence climbs to its ceiling over this many weighted tracks
FULL_W = 4.0                      # the picture is complete this many weighted tracks after the core was called ...
T_FULL = T1 - 10.0                # ... and 10 s before the end at the latest (it firms up by itself in the second before)
T_CALL = T0 + 0.6 * (T1 - T0)     # the core is called then at the latest, whatever the tracks have said
GHOSTS = 40                       # primary tracks that keep a line (the newest); the older ones keep their PoCA dot
MANY = 120                        # more tracks than this: the lines that are gone are no longer drawn one by one
HOT_DT = 1.0 / 30.0               # the flagged bins of the halo are looked for this often (Sphere._hot_scan)

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
SCAN_T, SCAN_A = 5.2, 0.87        # the scan line of the tomogram: seconds to go across and come back, reach (of the half width)
SCAN_WAKE = 0.45                  # s: what the scan line has just read glows that long behind it
DRIFT, DROP = 1.6, 0.5            # DISINTEGRATE: seconds a piece of the contour drifts / a voxel takes to drop out
SLIDE = 0.22                      # s: a new hit pushes the bars of its detector one place to the left in that time

# vertical layout (the bays are free from the header down to the bottom data band, whatever the towers are)
TOP, BOT = L.HEAD_Y + 12.0, 1196.0
Y_TITLE, Y_PANEL, Y_LOW, Y_BASE = 318.0, 300.0, 964.0, 1156.0
Y_CORNER = 1090.0                 # tag of the two callouts in the bottom corners of the body column
CORNER_W = 218.0
# what a block needs to exist
TOMO_MIN, TEXT_MIN, DETS_MIN = 330.0, 190.0, 320.0

_DIST = math.hypot(CAM_D, CAM_H)
# when the confidence passes 0.6 by itself, if the tracks have not taken it there (the cross of the core)
T_SURE = float(next(v for v in np.arange(T_FULL - 1.0, T_FULL + 0.02, 1.0 / 60.0)
                    if 0.5 + 0.5 * float(smoothstep(T_FULL - 1.0, T_FULL, v)) > 0.6))
_GLYPHS = "0123456789ABCDEF#%/*+-=<>"

# the blocks of furniture: lag = when a block starts to build, in seconds after the scene has started;
# wave / flow / bars / key = how it is constructed (build.Block); marks = registration brackets while it is made
BLK = {
    "strip": dict(lag=0.0, wave=0.5, flow="lr", bars="centre", key=81, marks=False),
    "groove": dict(lag=0.1, wave=0.5, flow="lr", key=141),
    "info": dict(lag=0.15, wave=0.4, flow="tb", key=111),
    "view": dict(lag=0.25, wave=0.2, key=91, marks=False),
    "tomo": dict(lag=0.3, wave=0.45, key=3),
    "dets": dict(lag=0.4, wave=0.5, flow="tb", key=51),
    "log": dict(lag=0.5, wave=0.5, flow="tb", key=120, marks=False),
    "corner": dict(lag=0.55, wave=0.2, key=96, marks=False),
    "num": dict(lag=0.55, wave=0.35, key=151),
    "profile": dict(lag=0.6, wave=0.5, flow="lr", key=31),
    "spec": dict(lag=0.7, wave=0.45, key=71),
    "bar": dict(lag=0.7, wave=0.4, flow="lr", bars="centre", key=158),
    "one": dict(lag=0.85, wave=0.3, key=160),
}
RING_LAG = 1.5                    # the labels of the halo: once it has been scanned in
TAG_GONE = 1.1                    # a tag has broken up a little before plain text is gone


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


def _er(text, amount, key, frame):
    """What is left of each character of an eroding text: the character, a random glyph (it is about to go:
    a new one at every frame), or None (gone)."""
    h = hash01(np.arange(len(text)), key)
    g = B.rnd(len(text), 7 * key + 3, frame)
    band = min(0.12, 0.6 * amount)
    out = []
    for k, (c, hv) in enumerate(zip(text, h)):
        if c == " " or hv >= amount + band:
            out.append(c)
        elif hv < amount:
            out.append(None)
        else:
            out.append(_GLYPHS[int(g[k] * len(_GLYPHS))])
    return out


def er(text, amount, key=0, frame=0):
    """Eroded text: a stable, growing subset of characters vanishes, the ones about to go flicker through
    random glyphs first. Empty once nothing readable is left, so the tag / label that carries it goes too."""
    if amount <= 0.0 or not text:
        return text
    if amount >= 0.94:
        return ""
    out = "".join(" " if c is None else c for c in _er(text, amount, key, frame))
    return out if out.strip() else ""


def er_runs(text, amount, key=0, frame=0):
    """er() for the text of an inverted tag, which has to break up with its text: the runs of characters
    that are left, as [(index of the first character, run)]. The characters that are gone are the gaps."""
    if not text or amount >= 0.94:
        return []
    if amount <= 0.0:
        return [(0, text)]
    runs, cur, start = [], "", 0
    for k, c in enumerate(_er(text, amount, key, frame) + [None]):
        if c is None:
            if cur.strip():
                runs.append((start + len(cur) - len(cur.lstrip()), cur.strip()))
            cur, start = "", k + 1
        else:
            cur += c
    return runs


def etag(f, layer, x, y, text, erode=0.0, key=0, fr=0, size=18, pad=5, anchor="ls", alpha=1.0, bold=False):
    """An inverted tag that can lose its text. While nothing is eroded it is Frame.tag; then it breaks up
    with its text: every run of characters that is left keeps its own piece of the box, the characters that
    are gone leave holes - instead of the tag standing there as a blank slab, or fading. All of it is gone
    a little before plain text is (TAG_GONE)."""
    if not text:
        return
    if erode <= 0.0:
        f.tag(layer, x, y, text, size=size, alpha=alpha, anchor=anchor, pad=pad, bold=bold)
        return
    fnt = font(max(6, int(round(size * f.s))), bold)        # the pieces stay where the whole tag had set them
    xl = x - fnt.getlength(text) / f.s * {"l": 0.0, "m": 0.5, "r": 1.0}[anchor[0]]
    # The pieces are drawn as they are (a "built" block of their own): the block around them may be in the
    # middle of being taken apart, and its wipe would leave them as blank boxes.
    with f.build(1e9, (xl, y - size, xl + 1.0, y), marks=False):
        for i, run in er_runs(text, min(1.0, erode * TAG_GONE), key, fr):
            f.tag(layer, xl + fnt.getlength(text[:i]) / f.s, y, run, size=size, alpha=alpha, anchor="l" + anchor[1],
                  pad=pad, bold=bold)


def header(f, x0, x1, y, title, erode=0.0, key=0, fr=0, layer="w"):
    """Thick rule with a small inverted title tag sitting on it (hud.panel_header), the tag breaking up with
    its text when it erodes."""
    f.rects("w", x0, y, x1, y + 5, 0.95)
    etag(f, layer, x0 + 4, y - 9, title, erode, key, fr, size=L.T_MICRO, pad=3)


def fade(amount):
    """How much of the boxes and rules of a panel is left while its text erodes (1 = whole, 0 = gone).
    It is not an alpha: nothing fades. DISINTEGRATE turns it into the age of a block that is taken apart
    (Disintegrate._age)."""
    return float(1.0 - smoothstep(0.3, 0.85, amount))


def draw_leader(f, layer, xs, ys, inten, prog, width=L.LW):
    """A leader being drawn (it does not fade in): the part `prog` (0..1) of the polyline xs, ys, from its
    first point. Returns how many of its points the pen has passed."""
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    n = len(xs)
    if prog <= 0.0:
        return 0
    if prog < 1.0:
        cum = np.r_[0.0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
        d = prog * cum[-1]
        k = int(np.clip(np.searchsorted(cum, d, side="right"), 1, len(xs) - 1))
        u = (d - cum[k - 1]) / max(cum[k] - cum[k - 1], 1e-6)
        xs = np.r_[xs[:k], xs[k - 1] + (xs[k] - xs[k - 1]) * u]
        ys = np.r_[ys[:k], ys[k - 1] + (ys[k] - ys[k - 1]) * u]
        n = k
    f.polyline(layer, xs, ys, inten, width=width)
    return n


def pop(age, dur=0.25):
    """Scale (0..1, scalar or array) of a mark that has just arrived, `age` seconds ago: a new tick, bar or
    dot of a live chart does not simply appear - it grows, a little too far, then settles."""
    return B.spring(np.asarray(age, np.float64) / dur)


def tick(value, age, width=3, dur=0.25, key=0, step=1):
    """A count that changed `age` seconds ago (by `step`): the digits that changed spin before they lock, the
    others do not move - the figure counts, and what it shows meanwhile is never far from the truth."""
    s = f"{value:0{width}d}"
    if age is None or not (0.0 <= age < dur):
        return s
    old = f"{max(value - step, 0):0{width}d}"
    h = B.rnd(len(s), key + 77, B.frame_no(age))
    return "".join(c if c == o else "0123456789"[int(h[k] * 10)] for k, (c, o) in enumerate(zip(s, old)))


def ring(f, layer, x, y, r, inten, n=48, width=L.LW):
    """A circle whose radius changes from frame to frame: always the same vertices, from its top (Frame.rings
    takes their number from the radius: a circle that grows would crawl). Traced inside a block being built."""
    a = -0.5 * np.pi + np.linspace(0.0, 2 * np.pi, n + 1)
    f.polyline(layer, x + r * np.cos(a), y + r * np.sin(a), inten, width=width)


def _path_dist(x, z, pts, prog=1.0):
    """Distance from the points (x, z) to the part `prog` (0..1 of its length) of the polyline pts (k, 2):
    how near a place of the tomogram is to a track, as far as the muon has come."""
    pts = np.asarray(pts, np.float64)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    left = prog * float(seg.sum())
    out = np.full(np.shape(x), 1e9)
    for k in range(len(seg)):
        if left <= 0.0 or seg[k] <= 1e-9:
            break
        u1 = min(1.0, left / seg[k])                    # how much of this piece is there
        ax, az = pts[k]
        dx, dz = (pts[k + 1] - pts[k]) * u1
        u = np.clip(((x - ax) * dx + (z - az) * dz) / max(dx * dx + dz * dz, 1e-12), 0.0, 1.0)
        out = np.minimum(out, np.hypot(x - ax - u * dx, z - az - u * dz))
        left -= seg[k]
    return out


def one_tag(times, t, life, swap=0.2, busy=0.35, t0=None):
    """A stream of items shares one tag: which item has it at t, and at what age (for Frame.build)?
    -> (index, age), or None when there is no tag. The tag is made when an item arrives and taken apart
    `life` seconds later. When the next item arrives while it is still up, it is first taken apart (in
    `swap` seconds), then made again for the new one - unless it was still being made (less than `busy`
    seconds old): then the new item simply takes it over and the build goes on. Nothing is anticipated,
    so the realtime app can do the same from the live stream. t0 = start of the scene: what arrived
    before it counts as arriving then."""
    times = np.asarray(times, np.float64)
    if t0 is not None:
        times = np.maximum(times, t0)
    k = int(np.searchsorted(times, t, side="right")) - 1
    if k < 0:
        return None
    start = float(times[0])                          # when the tag of the current item started to be made
    un = None                                        # (item, start of its tag, end of the taking apart)
    for j in range(1, k + 1):
        tj = float(times[j])
        if un is not None and tj < un[2]:            # the old tag is still being taken apart: the next one will be its
            continue
        if tj - start < busy:                        # still being made: taken over, the build goes on
            continue
        if tj - float(times[j - 1]) >= life:         # gone already: made anew
            start, un = tj, None
            continue
        un = (j - 1, start, tj + swap)               # up: taken apart first
        start = tj + swap
    if un is not None and t < un[2]:                 # ... which is what is happening now
        i, s0, end = un
        return i, min(B.io(t - s0, end - t, out=swap, span=0.6),
                      B.io(t - s0, life - (t - float(times[i])), out=0.45, span=0.7))
    age = t - float(times[k])
    if age >= life:
        return None
    return k, B.io(t - start, life - age, out=0.45, span=0.7)


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


def n_fit(fields, width, size=L.T_MICRO):
    """How many leading fields fit in `width` px (see fit_fields)."""
    n, used = int(width / (size * CHAR_W)), 0
    for k, fld in enumerate(fields):
        used += len(fld) + (1 if k else 0)
        if used > n:
            return k
    return len(fields)


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
        panels = sorted((tuple(map(float, p)) for p in ctx.slots["panels"]), key=lambda p: -round(p[1] - p[0]))
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

class Pose:
    """The shape of the body at one instant (see Body.rho): the clock of its travelling noise, how hard the
    music drives it, and the waves that are crossing it - one band per kick, climbing from the bottom with
    the ripple of the halo, and one ring per muon, spreading from where it went in."""
    __slots__ = ("t", "drive", "kicks", "hits")

    def __init__(self, t, drive=0.0, kicks=(), hits=()):
        self.t, self.drive = float(t), float(drive)
        self.kicks = kicks              # [(age s, amplitude)]
        self.hits = hits                # [(age s, strength, unit vector of the point of entry)]


def _attack(age, rise=0.03):
    """A wave does not start in one frame: it rises in a few hundredths of a second."""
    return 1.0 - math.exp(-max(age, 0.0) / rise)


K_SPEED, K_WIDTH, K_TAU = 5.4, 0.34, 0.7      # the ripple a kick sends up the halo: rad / s, rad, s
K_AMP, K_BAND = 0.04, 0.5                     # ... and the band it sends up the body with it: height (of the radius), width (rad)
H_SPEED, H_WIDTH, H_TAU = 2.6, 0.2, 0.75      # the ring of a muon
H_AMP, H_DENT = 0.05, 0.045
FOLLOW = 1.25                                 # the halo follows the outline of the body: px of stick per px of bulge
R_LIMIT = 0.12                                # the radius never leaves 1 +- this (soft limit)


class Body:
    """The lumpy sphere: fibonacci vertices on a radius field that never rests, a surface net and the
    long struts through the interior that make it a plexus.
    The radius field is a noise that travels over the surface (large lobes, lumps, a fine crinkle: plane
    waves of three scales, each on its way), swells with the music, and carries the waves of the kicks and
    of the muons (Pose). It is a closed form of the pose: any frame draws alone."""

    def __init__(self, seed=41, n=1500, amp=0.04):
        rng = np.random.default_rng(seed)
        m = 18
        self.d = unit(rng.normal(size=(m, 3)))
        self.fr = rng.uniform(3.2, 9.5, m)
        self.ph = rng.uniform(0, 2 * np.pi, m)
        a = rng.uniform(0.4, 1.0, m) / self.fr ** 0.5
        self.w = rng.uniform(0.12, 0.4, m) * np.where(rng.random(m) < 0.5, -1.0, 1.0)
        # (the vertices and the net below keep the draws of `rng`; the travelling noise has its own)
        r2 = np.random.default_rng(seed + 1000)
        lo, hi = 5, 6                                   # large lobes that change the outline, a fine crinkle
        self.d = np.r_[self.d, unit(r2.normal(size=(lo + hi, 3)))]
        self.fr = np.r_[self.fr, r2.uniform(1.5, 3.0, lo), r2.uniform(10.0, 14.0, hi)]
        self.ph = np.r_[self.ph, r2.uniform(0, 2 * np.pi, lo + hi)]
        a = np.r_[a, 0.85 * r2.uniform(0.7, 1.0, lo), 0.12 * r2.uniform(0.6, 1.0, hi)]
        self.a = a * amp / math.sqrt((a ** 2).sum() / 2)
        sgn = lambda k: np.where(r2.random(k) < 0.5, -1.0, 1.0)
        self.w = np.r_[self.w * 4.2, r2.uniform(0.7, 1.5, lo) * sgn(lo), r2.uniform(2.4, 4.2, hi) * sgn(hi)]
        self.gain = np.r_[np.full(m, 0.55), np.full(lo, 0.35), np.full(hi, 1.6)]     # what the music adds to each scale
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

    def _dev(self, u, p, grad=False):
        """What the pose adds to the radius 1 in the directions u, before the limit (and, if asked, the
        gradient of its noise part in space, (..., 3))."""
        arg = (u @ self.d.T) * self.fr + self.ph + self.w * p.t
        amp = self.a * (1.0 + self.gain * p.drive)
        r = (amp * np.sin(arg)).sum(-1)
        if p.kicks:                     # a band climbs from the bottom to the top
            th = np.arccos(np.clip(-u[..., 1], -1.0, 1.0))
            for age, a in p.kicks:
                r = r + (K_AMP * a * _attack(age) * math.exp(-age / K_TAU)) * np.exp(-((th - K_SPEED * age) / K_BAND) ** 2)
        for age, e, c in p.hits:        # a dent where the muon went in, and the ring it sends over the surface
            ang = np.arccos(np.clip(u @ c, -1.0, 1.0))
            r = r + e * _attack(age) * (H_AMP * math.exp(-age / H_TAU) * np.exp(-((ang - H_SPEED * age) / H_WIDTH) ** 2)
                                        - H_DENT * math.exp(-age / 0.22) * np.exp(-(ang / 0.3) ** 2))
        return (r, ((amp * self.fr) * np.cos(arg)) @ self.d) if grad else r

    def rho(self, u, tw):
        """Radius of the body in the directions u (..., 3). tw = a Pose, or a bare time (the noise alone)."""
        p = tw if isinstance(tw, Pose) else Pose(tw)
        return 1.0 + R_LIMIT * np.tanh(self._dev(u, p) / R_LIMIT)       # (however loud it gets, the body stays inside its halo)

    def shape(self, tw, u=None):
        """(radius, outward unit normal) in the directions u (the vertices by default), in one pass. The normal
        is that of the travelling noise - radius x direction minus the gradient of the radius along the
        surface; the waves of the kicks and of the muons are long and low: they are left out of it."""
        u = self.u if u is None else u
        p = tw if isinstance(tw, Pose) else Pose(tw)
        dev, g = self._dev(u, p, grad=True)
        th = np.tanh(dev / R_LIMIT)
        rho = 1.0 + R_LIMIT * th
        g = g * (1.0 - th * th)[:, None]
        return rho, unit(rho[:, None] * u - (g - (g * u).sum(-1, keepdims=True) * u))

    def surface(self, tw, u=None):
        u = self.u if u is None else u
        return u * self.rho(u, tw)[..., None]

    def outline(self, cam, phi, tw):
        """Radius of the outline of the body, as the camera sees it, in the screen directions phi (rad, 0 =
        right, counter-clockwise): the radius field on the limb of the unit sphere (to first order)."""
        r, up, fw = (cam.R[k].astype(np.float64) for k in range(3))
        s = 1.0 / float(np.linalg.norm(cam.pos))                # the limb leans towards the camera by asin(1 / distance)
        u = math.sqrt(1.0 - s * s) * (np.cos(phi)[:, None] * r + np.sin(phi)[:, None] * up) - s * fw
        return self.rho(u, tw)

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
        for _ in range(4):
            uy = np.clip(uy, -0.999, 0.999)
            r = np.sqrt(1 - uy ** 2)
            u = np.stack([r * np.cos(a), uy, r * np.sin(a)], 1)
            uy = y / self.rho(u, tw)
        return self.surface(tw, u)


def build_tracks(det, t0, t1, core_from=None, core_at=None, cache=None):
    """One muon track per detector onset in [t0, t1): path = far point, entry, PoCA, exit, far point.
    No track is aimed at the core before `core_from`; the primary hits of the first cluster at / after
    `core_at` always are (their echo trains then pile up on the core: that is when the anomaly shows).
    Nothing is read ahead: a track depends on its own hit and on the hits before it (its random numbers come
    from its tower and its rank among the hits of that tower since t0), so the tracks of the hits up to any
    moment are the first tracks of the whole scene, the same in every scene worker.
    cache = {} kept by the caller between two calls: a hit that is still the same gets the same track back
    (live, the scene asks again at every hit; a hit whose energy is still rising is made again)."""
    out = []
    used = {}
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
            ck = (key, i, float(th), float(e), bool(echo), (key, i) in forced, None if prev is None else prev["i"])
            tr = cache.get(ck) if cache is not None else None
            if tr is not None:
                used[ck] = tr
                out.append(tr)
                if not echo:
                    prev = tr
                continue
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
            tr = dict(t=float(th), key=key, i=i, e=float(e), echo=bool(echo), heavy=heavy, d=d, aim=aim, K=K, path=path,
                      through=through, mrad=theta * MRAD, p=0.4 + 7.5 * float(e) ** 1.6,
                      charge="+" if rng.random() < 0.56 else "-", dur=0.9 if heavy else 0.3,
                      zen=math.degrees(math.acos(min(1.0, -d[1]))), azi=math.degrees(math.atan2(d[2], d[0])) % 360)
            used[ck] = tr
            out.append(tr)
            if not echo:
                prev = tr
    if cache is not None:
        cache.clear()
        cache.update(used)
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
            hold = 1.0 if k < 5 else (0.5 if k < 11 else (0.2 if k < GHOSTS else 0.0))
            out[tr["id"]] = hold * (0.3 + 0.7 * math.exp(-age / 5.0))
    return out


def _ring_parts(lay, cam, seed):
    """What the opacity histogram is made of, whatever the data: (bin angles, its flat profile, the peak
    towards the core as this camera sees it, the angle of that peak)."""
    cx, cy, _, _ = cam.project(CORE[None].astype(np.float32))
    phc = math.atan2(-(float(cy[0]) - lay.cy), float(cx[0]) - lay.cx)
    phi = 2 * np.pi * (np.arange(N_BINS) + 0.5) / N_BINS
    rng = np.random.default_rng(seed)
    k = np.arange(2, 9)
    amp = rng.uniform(0.4, 1.0, 7) / k ** 0.8
    ph = rng.uniform(0, 2 * np.pi, 7)
    base = (amp * np.sin(k * phi[:, None] + ph)).sum(-1) / amp.sum()
    peak = np.exp(-(np.angle(np.exp(1j * (phi - phc))) / 0.22) ** 2)
    return phi, base, peak, phc


def ring_values(lay, cam, n_through, n_all, seed, jit):
    """Opacity per polar bin (screen angle, 0 = right, counter-clockwise). Flat while there is no data,
    then the peak towards the core grows with the tracks that crossed it."""
    phi, base, peak, phc = _ring_parts(lay, cam, seed)
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
    strip_grows = False     # the top-right slot is taken by the kick grid when nobody speaks (_groove)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.lay = Lay(ctx)
        self.body = Body()
        self.ring_jit = hash01(np.arange(N_BINS), 77) - 0.5
        rng = np.random.default_rng(5)
        self.scan_order = rng.random(len(self.body.ea))          # draw-on order of the net at the start
        self._init_motion(ctx, T0, T1)
        # the halo, 30 times a second over the scene: where its peak sits as the camera turns (_hot_scan)
        self._hot_ts = np.arange(T0, T1, HOT_DT)
        parts = [_ring_parts(self.lay, self.lay.camera(float(tt), T0, T1 - T0)[0], 9) for tt in self._hot_ts]
        self._hot_base = parts[0][1]
        self._hot_peak = np.array([q[2] for q in parts])
        self._cache, self._sig, self._est_done, self._hot_start = {}, None, [], {}
        self._update(T0 - 1.0, ctx)

    def _update(self, t, ctx):
        """The muons so far, and what the scene derives from them: the tracks, the moment the core is called,
        the cross at its centre, the muons that get a tag, the waves on the body, the estimate of the tomogram.
        Nothing is read ahead - a hit is only known once it has happened - so all of it is derived again when
        a hit comes in (or when its energy is still rising), from the hits up to now. A moment that has not
        come yet (t_found, t_cross) holds the latest time it can come."""
        end = min(float(np.nextafter(t, np.inf)), T1)
        t0, span = T0 - 0.6, T1 - T0
        sig = tuple(a.tobytes() for key in sd.KEYS for a in ctx.det.hits(key, t0, end))
        if sig == self._sig:
            return
        self._sig = sig
        was = getattr(self, "tr_t", np.zeros(0))
        self.tracks = build_tracks(ctx.det, t0, end, core_from=T0 + 0.22 * span, core_at=T0 + 0.4 * span, cache=self._cache)
        self.tr_t = np.array([tr["t"] for tr in self.tracks])
        self.t_first = float(self.tr_t[0]) if len(self.tr_t) else T1 + 1e6
        # when the core is called, and the cross at its centre comes on (confidence above 0.6)
        thr = [tr for tr in self.tracks if tr["through"]]
        self.thr_t = np.array([tr["t"] for tr in thr])
        self.thr_w = np.cumsum([0.25 if tr["echo"] else 1.0 for tr in thr]) if thr else np.zeros(0)
        self.t_found, self.w_found = self._call(self.thr_t, self.thr_w)
        self.t_cross = min([max(T_SURE, self.t_found)] +
                           [float(tt) for tt in self.thr_t if tt >= self.t_found and self._conf(float(tt))[1] > 0.6][:1])
        # the muons that get the tag of the bottom-right corner
        self.prim = [tr for tr in self.tracks if not tr["echo"]]
        self.prim_t = np.array([tr["t"] for tr in self.prim])
        # the waves the muons send over the body, and the estimate of the tomogram: where it stands after
        # every scattered track (it then moves there, it does not jump)
        self.waves = [(tr["t"], (0.3 if tr["echo"] else 1.0) * (0.55 + 0.6 * tr["e"]), unit(tr["path"][1].astype(np.float64)))
                      for tr in self.tracks]
        done = self._est_done                           # (one step per scattered track: only the new ones are worked out)
        m = 0
        while m < min(len(done), len(thr)) and done[m][0] is thr[m]:
            m += 1
        while 0 < m < len(thr) and thr[m - 1]["t"] == thr[m]["t"]:
            m -= 1
        del done[m:]
        for tr in thr[m:]:
            done.append((tr, self._est_steps([(tr["t"], [q for q in thr if q["t"] <= tr["t"]])])))
        self.est = (np.concatenate([st[0] for _, st in done]) if done else np.array([]),
                    np.concatenate([st[1] for _, st in done]) if done else np.array([]).reshape(-1, 3))
        # the flagged bins of the halo (_hot_age)
        self._hot_on = None
        if not (len(self.tr_t) >= len(was) and np.array_equal(self.tr_t[:len(was)], was)):
            self._hot_start = {}                        # (not the same past: the show was moved)

    @staticmethod
    def _call(thr_t, thr_w):
        """(when the core is called, the weighted count of scattered tracks then) from the scattered tracks so
        far: on the track that brings the count to FOUND_W and to 3 more than at 40 % of the scene - or at
        T_CALL, if the tracks have not said it by then."""
        def w(t):
            i = int(np.searchsorted(thr_t, t, side="right"))
            return float(thr_w[i - 1]) if i else 0.0
        t_core = T0 + 0.4 * (T1 - T0)
        ok = (thr_w >= max(FOUND_W, w(t_core) + 3.0)) & (thr_t >= t_core)
        t_found = min(float(thr_t[np.argmax(ok)]), T_CALL) if ok.any() else T_CALL
        return t_found, w(t_found)

    T_BUILD = T0                    # the furniture is constructed from here (DISINTEGRATE: from its own start)

    # ------------------------------------------------------------- what moves
    def _init_motion(self, ctx, t0, t1):
        """The clock of the travelling noise of the body. It runs with the music - fast when it is loud, slow
        in the quiet bars: the integral of the loudness, so it never jumps."""
        self._clock_t = np.arange(t0 - 2.0, t1 + 2.0, 0.02)
        rate = 0.45 + 1.1 * np.array([ctx.cues.loud(float(v), 0.4) for v in self._clock_t])
        self._clock = np.r_[0.0, np.cumsum(0.5 * (rate[1:] + rate[:-1]) * 0.02)]

    def _pose(self, t, ctx):
        """The shape of the body at t: noise clock, drive of the music, the bands of the last kicks, the rings
        of the last muons (Body.rho)."""
        kt, ka = ctx.cues.kicks(t - 1.5, t + 1e-6)
        kicks = [(t - float(th), min(1.5, 2.2 * float(a))) for th, a in zip(kt, ka)]
        soft = sum(a * _attack(age, 0.04) * math.exp(-age / 0.3) for age, a in kicks)
        drive = min(1.6, 0.9 * ctx.cues.loud(t, 0.4) + 0.35 * soft)
        hits = [(t - th, e, c) for th, e, c in self.waves if 0.0 <= t - th < 1.7][-10:]
        return Pose(float(np.interp(t, self._clock_t, self._clock)), drive, kicks, hits)

    def _slice(self):
        """Contour of the body in the plane of the core, at this frame (the 3D view and the tomogram share it)."""
        if self._sl is None:
            self._sl = self.body.slice(float(CORE[1]), self.pose)
        return self._sl

    @staticmethod
    def _est_steps(steps):
        """[(time, the scattered tracks that count from then on)] -> (times, values): the estimate of the core
        after every step = mean of the points of closest approach (x, z), its sigma, the number of tracks."""
        ts, vals = [], []
        for tk, trs in steps:
            if len(trs) < 2:
                continue
            Kp = np.array([tr["K"] for tr in trs])
            ts.append(float(tk))
            vals.append((float(Kp[:, 0].mean()), float(Kp[:, 2].mean()),
                         float(np.sqrt(Kp[:, [0, 2]].var(0).sum()) / math.sqrt(len(trs))) + 0.02))
        return np.array(ts), np.array(vals).reshape(-1, 3)

    def _estimate(self, t, loose=0.0, move=0.45):
        """(x, z, sigma) of the estimate at t, or None before there is one. A track that comes in (or decays)
        moves it: it travels to its new place in `move` seconds, with an ease. It also wanders inside its
        own uncertainty (0.5 sigma, more when `loose`): the less the tracks say, the less it holds still."""
        ts, vals = self.est
        k = int(np.searchsorted(ts, t, side="right"))
        if k == 0:
            return None
        j0 = max(1, int(np.searchsorted(ts, t - move, side="right")))       # the steps from j0 on are still on their way
        v = vals[min(j0, k) - 1].copy()
        for j in range(j0, k):
            v = v + (vals[j] - vals[j - 1]) * float(smoothstep(0.0, move, t - ts[j]))
        amp = (0.5 + loose) * v[2]
        wx = 0.6 * math.sin(1.9 * t + 0.4) + 0.4 * math.sin(4.3 * t + 2.0)
        wz = 0.6 * math.sin(2.3 * t + 1.7) + 0.4 * math.sin(3.7 * t + 0.3)
        return float(v[0] + amp * wx), float(v[1] + amp * wz), float(v[2])

    # ------------------------------------------------------------- build ages
    def _age0(self, name, t):
        """Age (for Frame.build) of a block of furniture: it is constructed when the scene starts, its `lag`
        after the cut. Negative = not there yet."""
        return t - self.T_BUILD - BLK[name]["lag"]

    def _age(self, name, t, erode=0.0):
        """Age of the boxes and rules of that block (tags, header rules, frames, axes). The same as _age0 in
        this scene; DISINTEGRATE takes them apart again as their text erodes."""
        return self._age0(name, t)

    def _blk(self, f, name, t, rect, erode=None, lag=0.0):
        """One of the blocks of furniture (BLK) being constructed: the Frame.build context for its drawing
        calls. erode=None: its content (lettering, bars, picture) - made when the scene starts. erode given
        (0 in this scene): its boxes and rules (tags, header rules, frames, axes, ticks) - made the same way,
        with the registration brackets; DISINTEGRATE takes them apart as the lettering erodes (see _age).
        lag = extra seconds before it starts."""
        b = BLK[name]
        kw = {k: v for k, v in b.items() if k not in ("lag", "marks")}
        a0 = self._age0(name, t) - lag
        if erode is None:
            return f.build(a0, rect, marks=False, **kw)
        return f.build(self._age(name, t, erode) - lag, rect, marks=b.get("marks", True) and a0 < 3.0, **kw)

    def _hot_scan(self, upto=None):
        """Has the opacity histogram flagged bins? - at every instant of the scene (self._hot_ts), with the
        muons known now (upto: with those up to that time only). Up to now it is what happened; after, what
        will happen if no other muon comes (the camera goes on turning, the picture firms up by itself)."""
        ts = self._hot_ts
        thr_t, thr_w, prim_t = self.thr_t, self.thr_w, self.prim_t
        if upto is not None:
            k = int(np.searchsorted(thr_t, upto, side="right"))
            thr_t, thr_w = thr_t[:k], thr_w[:k]
            prim_t = prim_t[:int(np.searchsorted(prim_t, upto, side="right"))]
        t_found, w_found = self._call(thr_t, thr_w)
        i = np.searchsorted(thr_t, ts, side="right")
        w = np.where(i > 0, np.r_[thr_w, 0.0][np.maximum(i - 1, 0)], 0.0)
        firm = np.minimum(1.0, (w - w_found) / FULL_W)
        conf = np.where(ts < t_found, 0.42 * np.minimum(1.0, w / max(SEARCH_W, 1e-6)),
                        0.5 + 0.5 * np.maximum(firm, smoothstep(T_FULL - 1.0, T_FULL, ts)))
        # (the same sums as ring_values, for every instant at once)
        conf = np.minimum(1.0, 7.0 * conf / 7.0)
        n_all = 9.0 * np.minimum(1.0, np.searchsorted(prim_t, ts, side="right") / 4.0)
        data = np.minimum(1.0, n_all / 9.0)[:, None]
        v = (0.1 + data * (0.24 + 0.2 * self._hot_base[None, :]) + 0.56 * self._hot_peak * conf[:, None]
             + (0.03 + 0.05 * data) * self.ring_jit[None, :])
        return (v > HOT).any(1)

    def _hot_age(self, t, shortest=0.4):
        """Age (for Frame.build) of the PEAK labels at t; negative when no bin is flagged. The labels are made
        when bins get flagged and taken apart just before they are let go; a flicker shorter than `shortest`
        gets no label. Both look a little ahead, which the live detectors do not allow: so they look at what
        will happen if no other muon comes (_hot_scan). A muon that comes in between can keep the labels up
        (they are then decoded again in place) or give a flicker its labels after all (they are made then)."""
        ts = self._hot_ts
        if self._hot_on is None:
            d = np.diff(np.r_[0, self._hot_scan().astype(int), 0])
            self._hot_on = list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))
        for i, j in self._hot_on:
            a, b = float(ts[i]), float(ts[j]) if j < len(ts) else 1e9
            if not (a <= t < b):
                continue
            start = self._hot_start.get(i)
            if start is None or start > t:
                # since when is it known that this one lasts? at its start, or at a muon that came just after
                start = None
                for tau in [a] + [float(v) for v in self.tr_t[(self.tr_t > a) & (self.tr_t <= min(t, a + shortest))]]:
                    on = self._hot_scan(upto=tau)
                    n = i
                    while n < len(on) and on[n]:
                        n += 1
                    if (n - i) * HOT_DT >= shortest:
                        start = self._hot_start[i] = tau
                        break
            return -1.0 if start is None else B.io(t - start, b - t, out=0.3, span=0.45)
        return -1.0

    # ------------------------------------------------------------------ state
    def _past(self, t):
        return [tr for tr in self.tracks if tr["t"] <= t]

    def _w(self, t):
        """Weighted count of the tracks that crossed the core so far."""
        i = int(np.searchsorted(self.thr_t, t, side="right"))
        return float(self.thr_w[i - 1]) if i else 0.0

    def _conf(self, t):
        """(found, confidence 0..1): the search, then the picture firming up with the tracks that cross the core
        (complete 10 s before the end at the latest)."""
        w = self._w(t)
        if t < self.t_found:
            return False, 0.42 * min(1.0, w / max(SEARCH_W, 1e-6))
        firm = min(1.0, (w - self.w_found) / FULL_W)
        return True, 0.5 + 0.5 * max(firm, float(smoothstep(T_FULL - 1.0, T_FULL, t)))

    @staticmethod
    def _kicks(ctx, t, phi):
        """What the drums do to the picture at t: (envelope 0..1.2, ripple per halo bin, recent kicks)."""
        kt, ka = ctx.cues.kicks(t - 1.3, t + 1e-6)
        # (the envelope of ctx.cues.kick, with an attack: the picture answers a kick in three frames, not in one)
        env = min(1.2, sum(float(a) * _attack(t - float(th), 0.022) * math.exp(-(t - float(th)) / 0.16) for th, a in zip(kt, ka)))
        d = np.abs(np.angle(np.exp(1j * (phi + np.pi / 2))))          # angular distance from the bottom of the halo
        rip = np.zeros(len(phi))
        for th, a in zip(kt, ka):
            age = t - float(th)
            rip += min(float(a), 1.6) * _attack(age, 0.022) * np.exp(-((d - K_SPEED * age) / K_WIDTH) ** 2) * math.exp(-age / K_TAU)
        return env, np.clip(0.5 * env + 0.8 * rip, 0.0, 1.5), [(t - float(th), min(float(a), 1.6)) for th, a in zip(kt, ka)]

    # ----------------------------------------------------------------- render
    def draw(self, f, t, ctx):
        lay = self.lay
        self._update(t, ctx)
        cam, yaw = lay.camera(t, T0, T1 - T0)
        self.pose, self._sl = self._pose(t, ctx), None
        past = self._past(t)
        thr = [tr for tr in past if tr["through"]]
        found, conf = self._conf(t)
        n_prim = sum(1 for tr in past if not tr["echo"])
        phi, v, phc = ring_values(lay, cam, 7.0 * conf, 9.0 * min(1.0, n_prim / 4.0), 9, self.ring_jit)
        env, pulse, kicks = self._kicks(ctx, t, phi)
        appear = float(smoothstep(T0 - 0.1, T0 + 1.5, t))
        hot_age = self._hot_age(t)
        # the halo follows the outline of the body: a bulge of the outline lengthens the sticks in front of it
        follow = FOLLOW * lay.R * (self.body.outline(cam, phi, self.pose) - 1.0)
        f.set_clip(*lay.clip)
        self._lattice(f, kick=env)
        self._body(f, cam, t, appear, gain=1.0 + 0.4 * env, swell=0.012 * min(1.0, env))
        self._shock(f, kicks, appear)
        self._ring(f, phi, v, appear, pulse=pulse, label_age=t - T0 - RING_LAG, peak_age=hot_age, follow=follow)
        f.set_clip()
        self._tracks(f, cam, t, past, ctx)
        self._core(f, cam, t, thr, conf=conf if found else 0.0, ring_age=t - self.t_found, cross_age=t - self.t_cross)
        self._callouts(f, cam, t, past, thr, yaw, found, conf)
        self._tomogram(f, t, past, thr, yaw, conf=conf, found=found,
                       est_age=t - thr[1]["t"] if len(thr) >= 2 else None, core_age=t - self.t_found,
                       count_age=t - past[-1]["t"] if past else None)
        self._profile(f, phi, np.clip(v + 0.1 * pulse * appear, 0.0, 1.0), t=t, peak_age=hot_age,
                      lift=0.45 * follow / (lay.l1 - lay.l0))
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
        if E.WALL:                  # a pixel does not land: one point in two each way, as a dot (the majors are crosses)
            m = far & ~major & (KX % 2 == 0) & (KY % 2 == 0)
            f.dots("w", X[m], Y[m], 1.6, 1.0)
        else:
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
            n = int(np.pi * (r0 + r1) / 11.0)           # (the same dots all the way: a count that follows the radius makes them crawl)
            a = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.4 * age
            f.dots("w", lay.cx + r * np.cos(a), lay.cy + r * np.sin(a), 2.0 if E.WALL else 1.5,
                   1.2 * min(1.0, amp) * (1.0 - u) ** 1.3 * appear)

    def _body(self, f, cam, t, appear, gain=1.0, swell=0.0):
        """The plexus: surface net (rim lit), struts through the interior, vertices. `swell` = breath on a kick."""
        b = self.body
        rho, N = b.shape(self.pose)
        P = (b.u * (rho * (1.0 + swell))[:, None]).astype(np.float32)
        N = N.astype(np.float32)
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
        if E.WALL:
            # the wall rule, the same body as DISINTEGRATE draws (disintegrate._eroding_wall): no grey net. One edge
            # in three and one node in three (by index: it never changes), at the weight that lands and at full
            # level where the surface turns away from the eye - the rim lights the outline -, no struts
            k3 = np.arange(len(ea)) % 3 == 0
            i0, i1 = E.wl(1.6 * rim[ea] * ev), E.wl(1.6 * rim[eb] * ev)
            em = em & k3
            f.segments("w", sx[ea][em], sy[ea][em], sx[eb][em], sy[eb][em], i0[em], i1[em], width=E.WALL_LINE)
        else:
            f.segments("w", sx[ea][em], sy[ea][em], sx[eb][em], sy[eb][em], (rim[ea] * ev)[em], (rim[eb] * ev)[em])
        sa, sb = b.sa, b.sb
        sm = np.ones(len(sa), bool)
        if appear < 1.0:
            sm = sm & (hash01(np.arange(len(sa)), 3) < max(0.0, appear * 1.4 - 0.4))
        si = (0.085 + 0.1 * np.maximum(rim[sa], rim[sb])) * b.s_var * gain
        if not E.WALL:
            f.segments("w", sx[sa][sm], sy[sa][sm], sx[sb][sm], sy[sb][sm], si[sm])
        vm = np.ones(len(P), bool)
        if appear < 1.0:
            vm = vm & ((1.0 - P[:, 1]) / 2.0 < appear * 1.1)
        if E.WALL:
            vm = vm & (np.arange(len(P)) % 3 == 0)
            f.dots("w", sx[vm], sy[vm], 1.6, E.wl(1.6 * rim[vm] * gain))
        else:
            f.pixels("w", sx[vm], sy[vm], (0.3 + 0.65 * rim[vm]) * gain)

    def _ring(self, f, phi, v, appear, alive=None, gain=1.0, pulse=None, label_age=None, peak_age=None, follow=None):
        """The lollipop halo = polar opacity histogram on a precise base circle. `pulse` (per bin, 0..1.5)
        is the answer to the drums: the sticks jump and their heads brighten as the ripple passes.
        `follow` (px per bin) = what the outline of the body adds to the sticks in front of it: the halo
        moves with the body.
        label_age / peak_age = ages (for Frame.build) of the angle labels and of the value of the peak: they
        are constructed, and taken apart when the age comes from build.io (None: there once the halo is whole)."""
        lay = self.lay
        cx, cy, r0 = lay.cx, lay.cy, lay.r0
        Ln = lay.l0 + (lay.l1 - lay.l0) * v
        if follow is not None:                  # (never shorter than a stub, never far over the top of the scale)
            Ln = np.clip(Ln + follow, 5.0 * lay.s, lay.l1 + 8.0 * lay.s)
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
        aa = phi[on][:, None] - da / 2 + np.linspace(0, da, 4)[None, :]
        ax, ay = cx + r0 * np.cos(aa), cy - r0 * np.sin(aa)
        f.segments("w", ax[:, :-1].ravel(), ay[:, :-1].ravel(), ax[:, 1:].ravel(), ay[:, 1:].ravel(), E.wl(0.6 * gain), width=L.LW)
        # dotted scale circles at 1.2 / 4.6 / 8 MWE
        for frac, inten in ((0.0, 0.32), (0.5, 0.36), (1.0, 0.6)):
            r = r0 + lay.l0 + (lay.l1 - lay.l0) * frac
            n = int(2 * np.pi * r / (14 if E.WALL else 7))
            aa = np.linspace(0, 2 * np.pi, n, endpoint=False)
            keep = on[np.minimum((aa / da).astype(int), N_BINS - 1)]
            if E.WALL:              # dots, twice as far apart, at the level of the wall
                f.dots("w", (cx + r * np.cos(aa))[keep], (cy - r * np.sin(aa))[keep], 1.5, E.wl(inten * gain * 1.6))
            else:
                f.pixels("w", (cx + r * np.cos(aa))[keep], (cy - r * np.sin(aa))[keep], inten * gain * 1.6)
        # inward ticks every 10 deg, long ones every 45 deg
        aa = np.radians(np.arange(0, 360, 10))
        ln = np.where(np.arange(36) % 9 == 0, 15.0, np.where(np.arange(36) % 3 == 0, 9.0, 5.0)) * min(1.0, lay.s)
        keep = on[np.minimum((aa / da).astype(int), N_BINS - 1)]
        f.segments("w", (cx + r0 * np.cos(aa))[keep], (cy - r0 * np.sin(aa))[keep],
                   (cx + (r0 - ln) * np.cos(aa))[keep], (cy - (r0 - ln) * np.sin(aa))[keep], E.wl(0.8 * gain),
                   width=E.ww(1.0))
        if gain > 0.5 and (on.all() if label_age is None else label_age >= 0.0):
            kw = dict(rect=(cx - lay.out - 60.0, cy - lay.out - 34.0, cx + lay.out + 60.0, cy + lay.out + 40.0),
                      flow="out", wave=0.25, marks=False, key=77)
            # angle labels: on the axis when they stay inside the column, else only above / below the halo
            with f.build(label_age, **kw):
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
                    with f.build(label_age if peak_age is None else peak_age, **kw):
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
                    ring(f, "r", float(px[2]), float(py[2]), 10 + 34 * (1 - fresh), 0.9 * fresh, n=32)
        # leader: detector -> up to the bus -> over -> down into the track (it may pass behind a tower)
        if leader and not echo and age < 1.9:
            tw = ctx.towers[tr["key"]]
            ox, oy = tw.det
            yb = BUS_Y[tr["key"]]
            a = math.exp(-age / 0.55)
            xs = [ox, ox, float(px[0]), float(px[0])]
            ys = [oy - 14, yb, yb, float(py[0])]
            if draw_leader(f, "r", xs, ys, 1.1 * a, age / 0.12) >= 3:       # drawn in 0.12 s, from the detector
                f.dots("r", [float(px[0])], [yb], 3.0, 1.4 * a)
        return px, py

    def _tracks(self, f, cam, t, past, ctx):
        """All the muons so far (their PoCA dots stay: they are the picture; the lines fade, see persistence)."""
        keep = persistence(past, t)
        if len(past) > MANY:        # (a busy detector) the lines that are gone: their PoCA dots, in one go
            old = [tr for tr in past if keep[tr["id"]] <= 0.0 and t - tr["t"] > 12.0]
            if old:
                gone = {tr["id"] for tr in old}
                past = [tr for tr in past if tr["id"] not in gone]
                px, py, _, ok = cam.project(np.array([tr["path"][2] for tr in old], np.float32))
                echo = np.array([tr["echo"] for tr in old])
                hot = ok & np.array([bool(tr["through"] and tr["mrad"] > 28) for tr in old])
                f.dots("w", px[ok], py[ok], np.where(echo, 1.6, 2.2)[ok], 0.6)
                f.dots("r", px[hot], py[hot], np.where(echo, 2.4, 3.6)[hot], 0.3)
        for tr in past:
            self._draw_track(f, cam, tr, t - tr["t"], ctx, persist=keep[tr["id"]])

    def _core(self, f, cam, t, thr, gain=1.0, conf=None, ring_age=None, cross_age=None):
        """The dense core, as the scattering gives it away: a red circle that firms up with every PoCA.
        `conf` overrides the default confidence (3 scattered tracks to show, 7 to be sure).
        ring_age = seconds since the circle was called (None = there): it does not pop in, it is traced from
        its top, both ways, in half a second - and un-traced when the age comes from build.io. cross_age = age
        (for Frame.build) of the cross at its centre (None = shown when the confidence is above 0.6)."""
        conf = ((min(1.0, len(thr) / 7.0) if len(thr) >= 3 else 0.0) if conf is None else conf) * gain
        # slice through the core height: front half bright, back half faint
        sl = self._slice().astype(np.float32)
        lx, ly, lz, _ = cam.project(sl)
        li = np.where(lz < np.median(lz), 0.75, 0.2) * gain
        f.segments("w", lx, ly, np.roll(lx, -1), np.roll(ly, -1), li, np.roll(li, -1), width=L.LW)
        if conf <= 0.0:
            return
        cx, cy, cz, _ = cam.project(CORE[None].astype(np.float32))
        rc = CORE_R * cam.focal / float(cz[0])
        n = int(40 + 90 * conf)
        a = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.15 * t
        if ring_age is not None and ring_age < 0.5:
            reach_ = np.pi * float(B.ease(max(ring_age, 0.0) / 0.5))
            a = a[np.abs(np.angle(np.exp(1j * (a + 0.5 * np.pi)))) < reach_]
        f.dots("r", float(cx[0]) + rc * np.cos(a), float(cy[0]) + rc * np.sin(a), 1.5, 0.5 + 0.7 * conf)
        if cross_age is None:
            if conf > 0.6:
                f.crosses("r", cx, cy, 10.0, 0.9 * conf, width=L.LW)
        else:
            x, y = float(cx[0]), float(cy[0])
            with f.build(cross_age, (x - 16.0, y - 16.0, x + 16.0, y + 16.0), wave=0.02, marks=False, key=66):
                f.crosses("r", cx, cy, 10.0, 0.9 * max(conf, 0.6), width=L.LW)

    # --------------------------------------------- callouts of the body column
    def _view_block(self, f, title, line, erode=0.0, fr=0, t=0.0):
        """Top left of the body column: the view tag (as much of the title as fits) and one line."""
        x0, x1 = self.lay.body
        title = fit_title(title, min(x1 - x0, 344.0), L.T_SMALL)
        rect = (x0, Y_PANEL - 24.0, x0 + min(x1 - x0, 344.0), Y_PANEL + 42.0)
        with self._blk(f, "view", t, rect, erode=erode):
            etag(f, "w", x0 + 4, Y_PANEL, title, erode, 91, fr, size=L.T_SMALL, pad=4)
        with self._blk(f, "view", t, rect):
            f.text("w", x0 + 4, Y_PANEL + 34, er(fit_fields(line, x1 - x0 - 8), erode, 92, fr), size=L.T_MICRO, alpha=0.75)

    def _corner(self, f, side, tag, lines, red=True, alpha=1.0, tag_size=L.T_TAG, age=None, tag_age=None, erode=0.0,
                key=95, fr=0):
        """Callout in a bottom corner of the body column (side -1 = left, +1 = right): an inverted tag with
        short lines stacked under it, never wider than CORNER_W so it stays outside the halo.
        age = seconds since it appeared, for Frame.build (None = there): it is constructed, and taken apart
        when the age comes from build.io - it never fades. tag_age = the same for its tag alone; `erode`
        breaks the tag up (the lines come eroded from the caller)."""
        x0, x1 = self.lay.body
        if side > 0 and not self.lay.two_corners:
            return
        x, anchor = (x0 + 5, "ls") if side < 0 else (x1 - 5, "rs")
        xa = x0 if side < 0 else x1 - CORNER_W
        kw = dict(rect=(xa, Y_CORNER - 26.0, xa + CORNER_W, Y_CORNER + 38.0 + 22.0 * len(lines)), wave=0.2, marks=False,
                  key=key + side)
        if tag:
            with f.build(age if tag_age is None else tag_age, **kw):
                etag(f, "r" if red else "w", x, Y_CORNER, tag, erode, key, fr, size=tag_size, pad=5, alpha=alpha,
                     anchor=anchor)
        with f.build(age, **kw):
            for k, (ln, layer) in enumerate(lines):
                f.text(layer, x + (-5 if side < 0 else 5), Y_CORNER + 30 + k * 22, ln, size=L.T_MICRO, alpha=0.9 * alpha,
                       anchor=anchor)

    def _callouts(self, f, cam, t, past, thr, yaw, found, conf):
        lay = self.lay
        self._view_block(f, "VIEW 01 // SCATTERING TOMOGRAPHY",
                         [f"ORBIT {math.degrees(yaw) % 360:05.1f} DEG", f" ELEV {math.degrees(math.atan2(CAM_H, CAM_D)):.1f}"],
                         t=t)
        # bottom left: the search, then the anomaly. On the hit that calls the core the SEARCHING callout is
        # taken apart while a leader is drawn from the core to the corner; then the ANOMALY tag is pushed out
        # and its lines are decoded.
        swap = 0.25
        if t < self.t_found + swap:
            self._corner(f, -1, "SEARCHING", [(f"POCA {len(thr):02d}  CONF {conf:.2f}", "w"), ("NO ANOMALY YET", "w")],
                         red=False, alpha=0.6 + 0.4 * (int(t * 2) % 2),
                         age=B.io(self._age0("corner", t), self.t_found + swap - t, out=swap, span=0.6))
        if found:
            age = t - self.t_found
            cx, cy, _, _ = cam.project(CORE[None].astype(np.float32))
            cx, cy = float(cx[0]), float(cy[0])
            ex, ey = lay.body[0] + 5 + text_w("ANOMALY", L.T_TAG) + 12, Y_CORNER - 8.0
            with f.build(age, (min(ex, cx) - 4.0, min(ey, cy) - 4.0, max(ex + 26.0, cx) + 4.0, max(ey, cy) + 4.0), flow="out",
                         origin=(cx, cy), wave=0.2, line=0.2, marks=False, key=90):
                f.segments("w", [cx, ex + 26], [cy, ey], [ex + 26, ex], [ey, ey], E.wl(0.6), width=E.ww(L.LW_HAIR))
            lines = ["RHO 11.3 G/CM3  Z~82", f"R {CORE_R:.2f} M  DEPTH {1 - np.linalg.norm(CORE):.2f} M",
                     f"POCA {len(thr):02d}  CONF {conf:.2f}"]
            self._corner(f, -1, "ANOMALY", [(ln, "w") for ln in lines], age=age - swap + 0.03)
        # bottom right: the muon that just came in - its tag is made when it arrives and taken apart six
        # seconds later (or as soon as the next one arrives)
        got = one_tag(self.prim_t, t, 6.0, t0=self.T_BUILD)
        if got:
            tr = self.prim[got[0]]
            K = tr["K"]
            lines = [f"P {tr['p']:.3f} GEV/C", f"THETA {tr['mrad']:4.1f} MRAD", f"POCA {K[0]:+.2f} {K[1]:+.2f} {K[2]:+.2f}"]
            self._corner(f, 1, f"MU{tr['charge']} // {L.NAMES[tr['key']]} // #{tr['id']:03d}",
                         [(ln, "r" if k == 1 and tr["through"] else "w") for k, ln in enumerate(lines)],
                         tag_size=L.T_SMALL, age=got[1])

    # ---------------------------------------------------------------- tomogram
    def _scan(self, t):
        """The scan line of the tomogram: where it is across the plot (-SCAN_A .. SCAN_A of the half width).
        It goes across and comes back (a sine: it eases at both ends and never jumps)."""
        return SCAN_A * math.sin(2 * math.pi * (t - self.T_BUILD) / SCAN_T)

    def _scan_age(self, t, xn):
        """Seconds since the scan line last passed the positions xn (same unit as _scan): what it has just
        read still glows behind it."""
        w = 2 * math.pi / SCAN_T
        ph = w * (t - self.T_BUILD)
        th = np.arcsin(np.clip(np.asarray(xn, np.float64) / SCAN_A, -1.0, 1.0))
        age = np.minimum(np.mod(ph - th, 2 * math.pi), np.mod(ph - (math.pi - th), 2 * math.pi)) / w
        return np.where(age > t - self.T_BUILD, 99.0, age)          # (it has not been there yet)

    def _tomogram(self, f, t, past, thr, yaw, erode=0.0, alive=None, title="TOMOGRAM // TOP VIEW // SLICE THROUGH THE CORE",
                  erode_txt=None, conf=None, found=None, est_age=None, core_age=None, lost=None, gone=(), loose=0.0,
                  count_age=None):
        """Top view of the slice through the core, in the tomogram column. `erode` takes the picture apart
        (lattice, contour, voxels), `erode_txt` its lettering. `conf` / `found` override the default
        confidence (track count / 7, core shown from 3 tracks).
        Nothing fades in: the plot is constructed when the scene starts (Sphere._blk), the estimate when the
        tracks give one (`est_age`), the dotted core when it is called (`core_age`) - ages for Frame.build,
        None = with the plot.
        It is alive: the contour is the slice of the body as it is at this instant (it moves with it); a scan
        line sweeps the slice, the lattice and the voxels it has just read glow behind it, the chord it cuts
        in the body is measured; the lattice lights along a track while the muon crosses; the estimate
        travels to its new place when a track moves it, and wanders inside its own uncertainty (`loose` adds
        to that).
        DISINTEGRATE: lost(h) = seconds since the erosion took the items whose hash is h (negative: still
        there) - a piece of the contour that breaks off drifts away, a voxel drops, a point shrinks, instead
        of being switched off; gone = [(x, z, seconds since)] of the tracks that have just decayed.
        count_age = seconds since the number of tracks changed (the figure spins before it locks)."""
        lay = self.lay
        if not lay.tg:
            return
        x0, x1 = lay.tomo
        fr = int(t * 30)
        et = erode if erode_txt is None else erode_txt
        S, Hh = lay.tg["S"], TOMO_HALF
        px0, py0 = lay.tg["px0"], lay.tg["py0"]
        side = lay.tg["side"]
        px1, py1 = px0 + side, py0 + side
        cx, cy = 0.5 * (px0 + px1), 0.5 * (py0 + py1)
        X = lambda x: cx + np.asarray(x) * S
        Y = lambda z: cy + np.asarray(z) * S
        since = (lambda h: np.full(np.shape(h), -1.0)) if lost is None else lost
        rect = (x0, Y_PANEL - 26.0, x1, py1 + (44.0 if lay.tg["mode"] == "side" else 140.0))
        a_box = self._age("tomo", t, et)
        ca = math.atan2(math.cos(yaw), math.sin(yaw))        # camera sits at (sin yaw, cos yaw) in x, z
        rr = 1.1
        # 1 - its boxes and rules: header, frame, rulers in metres, the camera direction of the 3D view
        with self._blk(f, "tomo", t, rect, erode=et):
            header(f, x0, x1, Y_PANEL, fit_title(title, x1 - x0 - 8), et, 3, fr)
            f.rect("w", px0, py0, px1, py1, E.wl(0.45), width=E.ww(1.0))
            hud.ruler(f, px0, px1, py1, -Hh, Hh, 0.125, 0.5 if S >= 150 else 1.0,
                      fmt=(lambda v: er(f"{v:+.1f}", et, 7, fr) if abs(v) < Hh - 0.1 else ""), down=True, inten=0.7, lab_dy=30)
            hud.vruler(f, px0, py0, py1, -Hh, Hh, 0.125, 0.5, right=False, inten=0.7)
            f.segments("w", [float(X(rr * math.cos(ca)))], [float(Y(rr * math.sin(ca)))],
                       [float(X((rr + 0.1) * math.cos(ca)))], [float(Y((rr + 0.1) * math.sin(ca)))], 0.9, width=L.LW_BOLD)
        # 2 - the picture: voxel lattice, contour of the body in the slice plane, scan line, tracks, voxels, PoCA points
        xs_n = self._scan(t)
        xs = cx + xs_n * 0.5 * side                          # the scan line
        chord = None                                         # (no reading where the line does not cut the contour twice)
        with self._blk(f, "tomo", t, rect):
            k = np.arange(-int(Hh / VOX), int(Hh / VOX) + 1)
            KX, KY = np.meshgrid(k, k)
            keep = np.ones(KX.shape, bool) if erode <= 0 else hash01(KX, KY, 5) > erode
            major = (KX % 4 == 0) & (KY % 4 == 0)
            # the lattice answers: it glows behind the scan line, and along a track while the muon crosses
            lit = np.where(major, 0.95, 0.5) + 1.1 * np.exp(-self._scan_age(t, KX * VOX / Hh) / SCAN_WAKE)
            if erode <= 0.3:
                for tr in past[-6:]:
                    age = t - tr["t"]
                    if 0.0 <= age < 2.4:
                        dur = tr["dur"] * (0.6 if tr["echo"] else 1.0)
                        dist = _path_dist(KX * VOX, KY * VOX, tr["path"][1:4][:, [0, 2]], min(1.0, age / dur))
                        lit = lit + (0.5 if tr["echo"] else 1.3) * math.exp(-age / 0.7) * np.exp(-(dist / 0.1) ** 2)
            if E.WALL:              # dots at full level: the lattice answers by their size, not by their level
                keep = keep & ((KX % 2 == 0) & (KY % 2 == 0))
                f.dots("w", X(KX * VOX)[keep], Y(KY * VOX)[keep],
                       (np.where(major, 2.2, 1.5) + 1.3 * np.clip(lit - np.where(major, 0.95, 0.5), 0.0, 1.3))[keep], 1.0)
            else:
                f.pixels("w", X(KX * VOX)[keep], Y(KY * VOX)[keep], np.minimum(lit, 1.8)[keep])
            f.text("w", px0 - 20, py1 + 30, er("X / M", et, 8, fr), size=L.T_MICRO, alpha=0.6, anchor="rs")
            f.text("w", px0 - 20, py0 + 16, er("Z / M", et, 9, fr), size=L.T_MICRO, alpha=0.6, anchor="rs")
            # the contour: the slice of the body as it is now. DISINTEGRATE: the pieces that break off drift away
            sl = self._slice()
            n_c = len(sl)
            xc, yc = X(sl[:, 0]), Y(sl[:, 2])
            xn, yn = np.roll(xc, -1), np.roll(yc, -1)
            if lost is None:
                m = np.ones(n_c, bool) if alive is None else alive
                ag = np.where(m, -1.0, 99.0)
            else:
                ag = lost(hash01(np.arange(n_c), 21))
                m = ag < 0.0
            if m.all():                                 # whole: one line, traced when the plot is made
                f.polyline("w", np.r_[xc, xc[0]], np.r_[yc, yc[0]], 0.85, width=L.LW)
            else:
                f.segments("w", xc[m], yc[m], xn[m], yn[m], 0.85, width=L.LW)
            f.set_clip(px0, py0, px1, py1)
            d = (ag >= 0.0) & (ag < DRIFT)
            if d.any():                                 # they turn, shrink and are gone
                i, a = np.nonzero(d)[0], ag[d]
                mx_, my_ = 0.5 * (xc[d] + xn[d]), 0.5 * (yc[d] + yn[d])
                out = np.arctan2(my_ - cy, mx_ - cx) + 1.7 * (hash01(i, 22) - 0.5)
                dist = (26.0 + 70.0 * hash01(i, 23)) * (1.0 - np.exp(-a / 0.7))
                mx_, my_ = mx_ + np.cos(out) * dist, my_ + np.sin(out) * dist
                ang = np.arctan2(yn[d] - yc[d], xn[d] - xc[d]) + 4.0 * (hash01(i, 24) - 0.5) * a
                hl = 0.5 * np.hypot(xn[d] - xc[d], yn[d] - yc[d]) * (1.0 - a / DRIFT)
                f.segments("w", mx_ - np.cos(ang) * hl, my_ - np.sin(ang) * hl, mx_ + np.cos(ang) * hl, my_ + np.sin(ang) * hl,
                           0.85 * (1.0 - a / DRIFT) ** 0.7, width=L.LW)
            # the scan line: bright inside the body (the chord it cuts), dim outside; DISINTEGRATE: it loses pieces too.
            # (It reads the slice itself, not what is left of its drawing: over a gap of a broken contour the
            # chord would come and go from one frame to the next.)
            hit = ((xc - xs) * (xn - xs) <= 0.0) & (xn != xc)
            ycross = yc[hit] + (yn[hit] - yc[hit]) * (xs - xc[hit]) / (xn[hit] - xc[hit])
            nd = 36
            ya = py0 + (py1 - py0) * np.arange(nd) / nd
            yb_ = ya + (py1 - py0) / nd
            inside = np.zeros(nd, bool)
            if len(ycross) >= 2:
                chord = float(ycross.max() - ycross.min()) / S
                inside = (0.5 * (ya + yb_) > ycross.min()) & (0.5 * (ya + yb_) < ycross.max())
            on = np.ones(nd, bool) if erode <= 0 else hash01(np.arange(nd), 26) > erode ** 2
            if E.WALL:              # outside the body: one dash in two instead of a dim line
                on = on & (inside | (np.arange(nd) % 2 == 0))
                f.segments("w", np.full(int(on.sum()), xs), ya[on], np.full(int(on.sum()), xs), yb_[on], 0.95, width=L.LW)
            else:
                f.segments("w", np.full(int(on.sum()), xs), ya[on], np.full(int(on.sum()), xs), yb_[on],
                           np.where(inside, 0.95, 0.3)[on], width=L.LW)
            if on.any() and len(ycross):
                f.dots("w", np.full(len(ycross), xs), ycross, 2.6, 1.3)
            f.set_clip()
            lx = float(X((rr + 0.12) * math.cos(ca)))
            f.text("w", min(lx + 6, px1 - 34), float(Y((rr + 0.12) * math.sin(ca))) + 5, er("CAM", et, 10, fr), size=L.T_MICRO,
                   alpha=0.7)
            if on.any():                                # ... and its mark on the top edge of the plot
                f.segments("w", [xs], [py0 - 9.0], [xs], [py0 - 2.0], 0.9, width=L.LW)
            # tracks of the last seconds, projected
            f.set_clip(px0, py0, px1, py1)
            for tr in past[-5:]:
                age = t - tr["t"]
                if age > 6.0 or erode > 0.3:
                    continue
                a = math.exp(-age / 2.2) * (0.45 if tr["echo"] else 1.0)
                p = tr["path"]                          # drawn while the muon crosses the body, like its 3D track
                draw_leader(f, "r", X(p[1:4, 0]), Y(p[1:4, 2]), 0.75 * a, age / (tr["dur"] * (0.6 if tr["echo"] else 1.0)))
            f.set_clip()
            # voxels: every cell that holds a point of closest approach is read out by the scan line (its
            # outline glows behind the line) and flashes when a new point falls in it; the cells of the core
            # are red. A new one does not simply appear: it pops (too large, then it settles)
            cells, core_cells = {}, {}
            for tr in past:
                key = (int(math.floor(tr["K"][0] / VOX)), int(math.floor(tr["K"][2] / VOX)))
                cells[key] = max(cells.get(key, -1e9), tr["t"])
                if tr["through"] and abs(tr["K"][1] - CORE[1]) < 0.3:
                    n_, first = core_cells.get(key, (0, tr["t"]))
                    core_cells[key] = (n_ + 1, first)
            vs = VOX * S
            if cells:
                ck = np.array(list(cells.keys()))
                cl = np.array(list(cells.values()))
                wake = np.exp(-self._scan_age(t, (ck[:, 0] + 0.5) * VOX / Hh) / SCAN_WAKE)
                iv = 0.9 * wake + 1.1 * np.exp(-np.maximum(t - cl, 0.0) / 0.6)
                iv = np.where(since(hash01(ck[:, 0], ck[:, 1], 11)) >= 0.0, 0.0, iv)
                sel = iv > 0.05
                if sel.any():
                    a_, b_ = X(ck[sel, 0] * VOX) + 0.5 * vs, Y(ck[sel, 1] * VOX) + 0.5 * vs
                    h = 0.5 * vs - 1.0
                    f.segments("w", np.r_[a_ - h, a_ + h, a_ + h, a_ - h], np.r_[b_ - h, b_ - h, b_ + h, b_ + h],
                               np.r_[a_ + h, a_ + h, a_ - h, a_ - h], np.r_[b_ - h, b_ + h, b_ + h, b_ - h],
                               np.tile(E.wl(iv[sel]), 4), width=E.ww(1.0))
            for (ix, iz), (n_, first) in core_cells.items():
                va = float(since(hash01(ix, iz, 11)))
                if va >= DROP:
                    continue
                a, b = float(X(ix * VOX)) + 0.5 * vs, float(Y(iz * VOX)) + 0.5 * vs
                h = (0.5 * vs - 2.0) * float(pop(t - first))
                if va >= 0.0:                           # it drops out: it falls and shrinks to nothing
                    h *= 1.0 - float(B.ease(va / DROP)) ** 2
                    b += 520.0 * va * va
                g = 1.0 + 0.5 * math.exp(-float(self._scan_age(t, (ix + 0.5) * VOX / Hh)) / SCAN_WAKE)
                f.rect("r", a - h, b - h, a + h, b + h, 0.85 * g)
                if n_ >= 2 and h > 3.0:
                    f.rects("r", a - h + 2, b - h + 2, a + h - 2, b + h - 2, (0.4 + 0.2 * min(n_, 4)) * g)
            dot = 0.75 + 0.25 * min(1.0, S / 212.0)
            if past:
                Kx = np.array([tr["K"][0] for tr in past])
                Kz = np.array([tr["K"][2] for tr in past])
                hot = np.array([bool(tr["through"] and tr["mrad"] > 28) for tr in past])
                g = pop(t - np.array([tr["t"] for tr in past]))
                da = since(hash01(np.array([tr["id"] for tr in past]), 13))
                g = g * np.where(da >= 0.0, np.clip(1.0 - da / 0.35, 0.0, 1.0), 1.0)       # (a point that goes shrinks)
                wk = np.exp(-self._scan_age(t, Kx / Hh) / SCAN_WAKE)
                f.dots("w", X(Kx[~hot]), Y(Kz[~hot]), 2.0 * dot * g[~hot], 0.9 * (1.0 + 0.9 * wk[~hot]))
                f.dots("r", X(Kx[hot]), Y(Kz[hot]), 3.2 * dot * g[hot], 1.5 * (1.0 + 0.4 * wk[hot]))
            for gx, gz, ga in gone:                     # a track that has just decayed: its point lets go in a ring
                if 0.0 <= ga < 0.5:
                    ring(f, "w", float(X(gx)), float(Y(gz)), 3.0 + 15.0 * float(B.ease(ga / 0.5)), 0.75 * (1.0 - ga / 0.5), n=20)
        # 3 - the estimate closing in on the core: made outwards from its centre when the tracks give one
        n = len(thr)
        readout = ["EST X  -.---", "EST Z  -.---", "SIGMA  -.---", "CONF   0.00"]
        est = self._estimate(t, loose) if n >= 2 and erode < 0.9 else None
        if est is not None:
            mx, mz, sig = est
            ex, ez = float(X(mx)), float(Y(mz))
            with f.build(a_box if est_age is None else min(a_box, est_age), (px0 - 4.0, py0 - 26.0, px1 + 4.0, py1 + 4.0),
                         flow="out", origin=(ex, ez), wave=0.3, marks=False, key=15):
                ring(f, "r", ex, ez, max(8.0, 2.2 * sig * S), 0.9, n=48)
                f.set_clip(px0, py0, px1, py1)
                f.segments("r", [ex, ex, ex, ex], [ez, ez, ez, ez], [px0, px1, ex, ex], [ez, ez, py0, py1], 0.4)
                f.set_clip()
                etag(f, "r", min(ex + 6, px1 - 64), py0 - 8, f"{mx:+.3f}", et, 15, fr, size=L.T_MICRO, pad=3)
                etag(f, "r", px0 + 8, ez - 8, f"{mz:+.3f}", et, 16, fr, size=L.T_MICRO, pad=3)
            readout = [f"EST X  {mx:+.3f}", f"EST Z  {mz:+.3f}", f"SIGMA  {sig:.3f}",
                       f"CONF   {(min(1.0, n / 7.0) if conf is None else conf):.2f}"]
            if est_age is not None and 0.0 <= est_age < 0.6:        # the figures spin before they lock
                readout = [B.roll(ln, est_age, 0.6, key=24 + j) for j, ln in enumerate(readout)]
        cf = min(1.0, n / 7.0) if conf is None else conf
        if (n >= 3 if found is None else found) and erode < 0.6:          # the true core, dotted: traced when it is called
            a = -0.5 * np.pi + np.linspace(0, 2 * np.pi, 60, endpoint=False)
            ac = a_box if core_age is None else min(a_box, core_age)
            a = a[: int(round(60 * float(B.ease(ac / 0.5))))] if ac < 0.5 else a
            if E.WALL:
                f.dots("w", X(CORE[0] + CORE_R * np.cos(a[::2])), Y(CORE[2] + CORE_R * np.sin(a[::2])), 1.7, E.wl(1.2 * cf))
            else:
                f.pixels("w", X(CORE[0] + CORE_R * np.cos(a)), Y(CORE[2] + CORE_R * np.sin(a)), 1.2 * cf)
        # readouts: beside the plot, or under it in two columns when the column is narrow
        # (the count ticks when a track comes in, or decays: the digit that changed spins before it locks)
        poca = "POCA   " + tick(len(past), count_age, key=23, step=-1 if lost is not None else 1)
        left = [f"VOXEL  {VOX:.3f} M", poca, f"CORE   {n:03d}", f"SCAN X {xs_n * Hh:+.2f}", "CHORD  -.-- M" if chord is None else f"CHORD  {chord:.2f} M"]
        with self._blk(f, "tomo", t, rect):
            if lay.tg["mode"] == "side":
                for k, ln in enumerate(left + readout):
                    f.text("r" if k >= 5 and est is not None else "w", x0, py0 + 42 + k * 24, er(ln, et, 20 + k, fr),
                           size=L.T_MICRO, alpha=0.85)
            else:
                xr = x0 + max(150.0, 0.5 * (x1 - x0))
                for k, ln in enumerate(left[:4]):
                    f.text("w", x0, py1 + 62 + k * 22, er(ln, et, 20 + k, fr), size=L.T_MICRO, alpha=0.85)
                for k, ln in enumerate(readout):
                    f.text("r" if est is not None else "w", xr, py1 + 62 + k * 22, er(ln, et, 25 + k, fr), size=L.T_MICRO,
                           alpha=0.85)

    def _low_panel(self, f, col, title, erode=0.0, key=31, fr=0, cap=None):
        """Header of a panel in the lower part of a column (y = Y_LOW .. Y_BASE). Returns its x range.
        (It is a box-and-rule: the caller draws it inside the boxes-and-rules block of its panel.)"""
        x0, x1 = col
        if cap:
            x1 = min(x1, x0 + cap)
        header(f, x0, x1, Y_LOW, fit_title(title, x1 - x0 - 8), erode, key, fr)
        return x0, x1

    def _profile(self, f, phi, v, alive=None, erode=0.0, t=0.0, peak_age=None, lift=None):
        """The opacity histogram, unrolled under the tomogram: one bar per bin, flagged bins red.
        peak_age = age (for Frame.build) of the PEAK label, negative / None = no flagged bin.
        lift (per bin, in units of v) = what the outline of the body adds to its sticks (Sphere._ring): the
        bars move with them. The flags stay those of v."""
        lay = self.lay
        if not lay.tomo:
            return
        fr = int(t * 30)
        x0, x1 = lay.tomo
        yb, hmax = Y_BASE, 150.0
        on = np.ones(N_BINS, bool) if alive is None else alive
        n = N_BINS
        vh = v if lift is None else np.clip(v + lift, 0.03, 1.0)
        if (x1 - x0 - 70) / n < 3.2:        # narrow column: pair the bins
            v, vh, on, n = np.maximum(v[0::2], v[1::2]), np.maximum(vh[0::2], vh[1::2]), on[0::2] | on[1::2], N_BINS // 2
        bw = (x1 - x0 - 70) / n
        xs = x0 + 60 + np.arange(n) * bw
        hot = v > HOT
        a, b = on & ~hot, on & hot
        yt = yb - HOT * hmax
        step = 8 if x1 - x0 >= 520 else 4
        rect = (x0, Y_LOW - 26.0, x1, Y_BASE + 34.0)
        with self._blk(f, "profile", t, rect, erode=erode):             # its boxes and rules
            self._low_panel(f, lay.tomo, f"OPACITY_PROFILE // MWE // {N_BINS} BINS // UNROLLED", erode, 31, fr)
            f.segments("w", [x0 + 54], [yb + 4], [x0 + 54], [yb - hmax - 6], E.wl(0.6), width=E.ww(1.0))
            for frac in (0.0, 0.5, 1.0):
                f.segments("w", [x0 + 54], [yb - frac * hmax], [x0 + 46], [yb - frac * hmax], E.wl(0.8), width=E.ww(1.0))
            f.segments("r", [x0 + 54], [yt], [x1], [yt], 0.55)
            for k in range(0, step + 1):
                xx = x0 + 60 + k * (x1 - x0 - 70) / step
                f.segments("w", [xx], [yb + 2], [xx], [yb + 10], E.wl(0.8), width=E.ww(1.0))
        with self._blk(f, "profile", t, rect):                          # its bars and lettering
            f.rects("w", xs[a], yb - vh[a] * hmax, xs[a] + max(2.0, bw - 2.2), yb, 0.92)
            f.rects("r", xs[b], yb - vh[b] * hmax, xs[b] + max(2.0, bw - 2.2), yb, 1.0)
            for frac in (0.0, 0.5, 1.0):
                f.text("w", x0, yb - frac * hmax + 5, er(f"{float(mwe(frac)):.1f}", erode, 33, fr), size=L.T_MICRO, alpha=0.7)
            f.text("r", x0 + 62, yt - 8, er(f"FLAG > {float(mwe(HOT)):.2f} MWE", erode, 34, fr), size=L.T_MICRO,
                   alpha=0.9)
            for k in range(0, step):
                xx = x0 + 60 + k * (x1 - x0 - 70) / step
                f.text("w", xx + 3, yb + 26, er(f"{k * 360 // step:03d}", erode, 35 + k, fr), size=L.T_MICRO, alpha=0.7)
        if peak_age is not None and peak_age >= 0.0 and x1 - x0 >= 420:    # above the tallest bar, clear of the header rule
            with f.build(peak_age, (x1 - 190.0, yb - hmax - 28.0, x1, yb - hmax - 4.0), wave=0.12, marks=False, key=30):
                f.text("r", x1, yb - hmax - 12, er(f"PEAK {float(mwe(v.max())):.2f} MWE", erode, 30, fr), size=L.T_MICRO,
                       alpha=0.95, anchor="rs")

    # ------------------------------------------------------- text columns
    def _info_block(self, f, title, rows, red=(), erode=0.0, fr=0, key=111, t=0.0):
        """Title tag + parameter rows at the top of the info column. Returns (x, width, y under the block)."""
        x0, x1 = self.lay.info
        w = x1 - x0
        size = fit_size(title, w - 30.0, 24.0, 46.0)
        rs = L.T_SMALL if text_w("X" * max(len(r) for r in rows), L.T_SMALL) <= w - 4 else L.T_MICRO
        y_end = Y_TITLE + 52 + len(rows) * 25.5
        rect = (x0, Y_TITLE - 50.0, x1, y_end - 10.0)
        with self._blk(f, "info", t, rect, erode=erode):
            etag(f, "w", x0 + 10, Y_TITLE, title, erode, key, fr, size=size, pad=9)
        with self._blk(f, "info", t, rect):
            for k, ln in enumerate(rows):
                f.text("r" if k in red else "w", x0 + 4, Y_TITLE + 52 + k * 25.5, er(ln, erode, key + 1 + k, fr), size=rs,
                       alpha=0.9)
        return x0 + 4, w - 8, y_end + 30

    def _log_block(self, f, x, w, y, tag, head, lines, erode=0.0, fr=0, key=120, t=0.0):
        """A log under the info block: small tag, header line, then rows = (fields, layer, alpha, age[, gone]).
        The fields that do not fit the column are dropped (time / detector / energy always stay). A row with
        an age is new: it is decoded out of noise as it comes in. gone (0..1) = erosion of its own: a row
        that leaves the log falls apart first."""
        size = L.T_SMALL if w >= 412.0 else L.T_MICRO
        lead = 1.5 * size
        rect = (x - 4.0, y - 22.0, x + w + 4.0, Y_BASE + 12.0)
        with self._blk(f, "log", t, rect, erode=erode):
            etag(f, "w", x + 3, y, fit_title(tag, w - 6), erode, key, fr, size=L.T_MICRO, pad=3)
        with self._blk(f, "log", t, rect):
            # (the header names the fields the rows show: a narrow column drops the same ones in both)
            nf = min([n_fit(ln[0], w, size) for ln in lines if len(ln[0]) == len(head)] + [len(head)])
            f.text("w", x, y + 30, er(fit_fields(head[:nf], w, size), erode, key + 1, fr), size=size, alpha=0.55)
            for row, (fields, layer, alpha, new_age, *gone) in enumerate(lines[: int((Y_BASE + 4 - (y + 54)) / lead) + 1]):
                s = fit_fields(fields, w, size)
                if new_age is not None and new_age < 1.0:
                    s = B.resolve(s, new_age, cps=260.0, key=key + row)
                f.text(layer, x, y + 54 + row * lead, er(s, max([erode] + gone), key + 10 + row, fr), size=size, alpha=alpha)

    def _left(self, f, t, past, thr, found):
        if not self.lay.info:
            return
        expo = max(0.0, t - self.t_first)
        rho = B.roll("RHO_CORE  11.3 G/CM3", t - self.t_found, 0.6, key=6) if found else "RHO_CORE  --.- G/CM3"
        rows_ = ["TARGET    SPHERE_01", "RADIUS    1.000 M", f"EXPOSURE  00:{int(expo // 60):02d}:{expo % 60:05.2f}",
                 f"TRACKS    {len(past):04d}", f"SCATTERED {len(thr):04d}", "RHO_BODY  2.65 G/CM3", rho]
        x, w, y = self._info_block(f, "MUOGRAPHY", rows_, red=(6,) if found else (), t=t)
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
                        ["TIME   ", "D ", "E   ", " MRAD", "X    ", "Y    ", "Z"], lines, t=t)

    def _right(self, f, t, ctx, past, title="DETECTORS", erode=0.0):
        """The three streams: hit counts, last energy, the float each tower sends, its last hits as bars.
        It is live: the counts spin when they change, the float has a level with a peak mark, a hit sends a
        red dash along the rule of its detector, and its bar grows at the right end of the row while the
        older ones slide one place to the left (they do not jump)."""
        if not self.lay.dets:
            return
        x0, x1 = self.lay.dets
        x1 = min(x1, x0 + 460.0)
        fr = int(t * 30)
        y = Y_TITLE
        rect = (x0, y - 50.0, x1, y + 92.0 + 2 * 176.0 + 122.0)
        xm0, xm1 = x0 + 110.0 + text_w("/MUON/C 0.000", L.T_MICRO) + 12.0, x1 - 2.0       # the level of the float
        meter = xm1 - xm0 >= 50.0
        # two passes over the same block: its boxes and rules, then its lettering and bars
        for rules in (True, False):
            with self._blk(f, "dets", t, rect, erode=erode if rules else None):
                if rules:
                    etag(f, "r", x0 + 10, y, title, erode, 51, fr, size=fit_size(title, x1 - x0 - 30.0, 24.0, 46.0), pad=9)
                else:
                    f.text("w", x0 + 4, y + 44, er("3 DETECTORS // 3 FLOATS // OSC", erode, 52, fr), size=L.T_MICRO, alpha=0.75)
                yy = y + 92
                slide = {"w": [], "r": []}      # the bars that move, of the three rows: drawn in one go (hud.bars)

                def bar(layer, xa, ya, xb_, yb_, inten):
                    slide[layer].append(np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, np.float64))
                                                              for v in (xa, ya, xb_, yb_, inten))))

                for k, key in enumerate(L.ORDER):
                    tt, ee, ec = ctx.det.hits(key, 0.0, t + 1e-6)
                    age, e = ctx.det.last(key, t, echoes=True)
                    yb = yy + 112
                    if rules:
                        f.rects("w", x0, yy - 22, x1, yy - 19, 0.9)
                        etag(f, "r" if age < 0.7 else "w", x0 + 3, yy + 8, L.NAMES[key], erode, 53 + k, fr, size=L.T_LABEL,
                             pad=4)
                        f.segments("w", [x0], [yb + 1], [x1], [yb + 1], E.wl(0.35), width=E.ww(1.0))
                        if meter:                       # the scale of the level: 0, half, 1
                            xt = xm0 + np.array([0.0, 0.5, 1.0]) * (xm1 - xm0)
                            f.segments("w", np.r_[xm0, xt], np.r_[yy + 42.0, np.full(3, yy + 42.0)],
                                       np.r_[xm1, xt], np.r_[yy + 42.0, np.full(3, yy + 46.0)], E.wl(0.4), width=E.ww(1.0))
                    else:
                        n, n_e = int((~ec).sum()), int(ec.sum())
                        val = float(ctx.det.value(key, t))
                        # the counts tick when they change (the digit that changed spins, then locks); the
                        # decimals of a new energy spin before they lock
                        hits_ = tick(n, t - float(tt[~ec][-1]) if n else None, key=160 + k)
                        echo_ = tick(n_e, t - float(tt[ec][-1]) if n_e else None, key=163 + k)
                        last_ = f"{e:.3f}"
                        if age < 0.25:
                            last_ = last_[:2] + B.roll(last_[2:], age, 0.25, key=166 + k)
                        lines = [f"HITS {hits_}   ECHO {echo_}",
                                 f"LAST E {last_}   T+{min(age, 99.9):04.1f}" if age < 90 else "LAST E -.---",
                                 f"/MUON/{key} {val:.3f}"]
                        for j, ln in enumerate(lines):
                            f.text("w", x0 + 110, yy - 2 + j * 22, er(ln, erode, 60 + 3 * k + j, fr), size=L.T_MICRO,
                                   alpha=0.85 if j < 2 else 0.6)
                        sl_ = slice(max(0, len(tt) - 34), len(tt))      # the hits that may still be in the row
                        ts_, es, cs = tt[sl_], ee[sl_], ec[sl_]
                        since_ = t - ts_
                        if meter and erode < 0.85:      # the float it streams, as a level; the red mark holds its last peaks
                            peak = max([val] + [float(v) for v in np.minimum(es, 1.0) * np.exp(-since_ / 1.1)])
                            bar("w", xm0, yy + 34, xm0 + val * (xm1 - xm0), yy + 40, 0.95)
                            xp = xm0 + min(1.0, peak) * (xm1 - xm0)
                            bar("r", xp - 1.5, yy + 30, xp + 1.5, yy + 44, 1.1)
                        for a_, c_ in zip(since_[-4:], cs[-4:]):        # a hit: a red dash runs along the rule of its detector
                            if a_ < 0.45 and erode < 0.85:
                                xh = x0 + (x1 - x0) * float(B.ease(a_ / 0.4))
                                bar("r", max(x0, xh - 64.0), yy - 17, xh, yy - 13, (0.6 if c_ else 1.3) * (1.0 - a_ / 0.45))
                        # the last hits of this tower as energy bars (newest right). A hit that comes in pushes
                        # the row one place to the left - it slides there - while its own bar grows
                        m = 26
                        sw = (x1 - x0) / m
                        if len(es):
                            push = smoothstep(0.0, SLIDE, since_)
                            xb = x0 + (m - 1 - (np.cumsum(push[::-1])[::-1] - push)) * sw
                            keep = xb + 9.0 > x0
                            if erode > 0:
                                keep = keep & (hash01(sl_.start + np.arange(len(es)), k, 9) > erode)
                            g = pop(since_)             # the bar of a hit that has just come in grows
                            a, b = keep & ~cs, keep & cs
                            bar("r", np.maximum(xb[a], x0), yb - (6 + 44 * es[a]) * g[a], np.maximum(xb[a] + 9, x0), yb, 0.95)
                            bar("w", np.maximum(xb[b], x0), yb - (4 + 44 * es[b]) * g[b], np.maximum(xb[b] + 5, x0), yb, 0.6)
                    yy += 176
                for layer, parts in slide.items():
                    if parts:
                        hud.bars(f, layer, *(np.concatenate([p[j] for p in parts]) for j in range(5)))

    def _spectrum(self, f, t, past, erode=0.0):
        """Scattering-angle spectrum of the live tracks (under the detectors)."""
        if not self.lay.spec:
            return
        fr = int(t * 30)
        x0, x1 = self.lay.spec
        x1 = min(x1, x0 + 460.0)
        mr = np.array([tr["mrad"] for tr in past]) if past else np.zeros(0)
        edges = np.linspace(0, 60, 25)
        cnt, _ = np.histogram(mr, edges)
        hgt = 150.0 * np.sqrt(cnt / max(cnt.max(), 1)) if len(mr) else np.zeros(24)
        bw = (x1 - x0 - 8) / 24
        xs = x0 + 4 + np.arange(24) * bw
        tail = edges[:-1] >= 28
        m = cnt > 0
        rect = (x0, Y_LOW - 26.0, x1, Y_BASE + 34.0)
        with self._blk(f, "spec", t, rect, erode=erode):                # its boxes and rules
            self._low_panel(f, self.lay.spec, "THETA_SCATTER // MRAD", erode, 71, fr, cap=460.0)
            hud.ruler(f, x0 + 4, x0 + 4 + 24 * bw, Y_BASE + 2, 0, 60, 2.5, 10 if x1 - x0 >= 300 else 20,
                      fmt=lambda v: er(f"{v:.0f}", erode, 72, fr), inten=0.6, lab_dy=26)
        with self._blk(f, "spec", t, rect):                             # its bars
            f.rects("w", xs[m & ~tail], Y_BASE - hgt[m & ~tail], xs[m & ~tail] + bw - 4, Y_BASE, 0.95)
            f.rects("r", xs[m & tail], Y_BASE - hgt[m & tail], xs[m & tail] + bw - 4, Y_BASE, 0.95)

    # --------------------------------------------------------- header band
    def _strip(self, f, t, ctx, title=None, t0=T0, t1=T1, erode=0.0, curve=None):
        """Score strip, top left (the header band is above every tower): loudness comb, one tick per onset.
        Constructed when the scene starts: its rules and ticks first (hud.strip_base), then its title, the
        comb, the lanes, the hits and the cursor from left to right."""
        fr = int(t * 30)
        if title is None:
            sec = sd.section_at(t)[1]
            title = f"TRACK_TIMELINE // {sec[0]} {sec[1]} // ONE TICK = ONE MUON"
        a_box = self._age("strip", t, erode)
        base = dict(title=None, band=False, ticks=(t0, t1, 1.0, 10.0))
        if a_box < 0.0:                         # not there (yet, or any more): only its geometry is needed
            with f.build(-1.0, L.STRIP):
                x0, y0, x1, y1, yb = hud.strip_base(f, **base)
        else:
            x0, y0, x1, y1, yb = hud.strip_base(f, age=a_box if a_box < 2.5 else None, **base)
        X = lambda tt: x0 + (np.asarray(tt, np.float64) - t0) / (t1 - t0) * (x1 - x0)
        n = int((x1 - x0) / 5)
        lv = ctx.cues.loud_curve(t0, t1, n) if curve is None else curve(t0 + (np.arange(n) + 0.5) / n * (t1 - t0))
        tb = t0 + (np.arange(n) + 0.5) / n * (t1 - t0)
        xb = X(tb)
        keep = np.ones(n, bool) if erode <= 0 else hash01(np.arange(n), 83) > erode
        yc = y0 + 28
        lanes = {"L": y0 + 78, "C": y0 + 98, "R": y0 + 118}
        with self._blk(f, "strip", t, L.STRIP, erode=erode, lag=0.2):   # its title and the rules of the three lanes
            etag(f, "w", x0, L.STRIP[1] + 25, title, erode, 81, fr, size=L.T_MICRO, pad=3)
            for key, yl in lanes.items():
                f.segments("w", [x0 + 22], [yl], [x1], [yl], E.wl(0.22), width=E.ww(1.0))
        with self._blk(f, "strip", t, L.STRIP, lag=0.25):               # the comb, the lettering, the hits, the cursor
            past_ = (tb <= t) & keep
            f.rects("w", xb[past_], yc, xb[past_] + 2, yc + 2 + 26 * lv[past_] ** 1.4, 0.9)
            fut = (tb > t) & keep
            if E.WALL:              # what is still to come: a stub at full level instead of a dim tooth
                f.rects("w", xb[fut], yc, xb[fut] + 2, yc + 3, 0.9)
            else:
                f.rects("w", xb[fut], yc, xb[fut] + 2, yc + 2 + 26 * lv[fut] ** 1.4, 0.22)
            for tv in np.arange(t0, t1 - 0.1, 10.0):
                f.text("w", float(X(tv)) + 5, y0 + 24, er(sd.tc(tv)[:5], erode, 86, fr), size=L.T_MICRO, alpha=0.6)
            for key, yl in lanes.items():
                f.text("w", x0 + 5, yl + 1, er(key, erode, 84, fr), size=L.T_MICRO, alpha=0.75)
                tt, ee, ec = ctx.det.hits(key, t0, min(t, t1) + 1e-6)
                if len(tt):
                    k2 = np.ones(len(tt), bool) if erode <= 0 else hash01(np.arange(len(tt)), 85) > erode
                    xh = X(tt)
                    g = pop(t - tt)                     # the tick of a hit that has just come in grows
                    a, b = ~ec & k2, ec & k2
                    f.rects("r", xh[a], yl - (4 + 13 * ee[a]) * g[a], xh[a] + 3, yl, 1.0)
                    f.rects("r", xh[b], yl - (2 + 9 * ee[b]) * g[b], xh[b] + 2, yl, 0.55)
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
        yb = y0 + 96
        X = lambda tt: ix0 + (np.asarray(tt, np.float64) - (t - span)) / span * (ix1 - ix0)
        with self._blk(f, "groove", t, (x0 + 8.0, y0 + 8.0, x1 - 8.0, y1 - 6.0), erode=0.0):
            f.tag("w", ix0, y0 + 25, f"KICK_GRID // {sec[0]} {sec[1]} // LAST {span:.0f} S" if wide else "KICK_GRID",
                  size=L.T_MICRO, pad=3)
            f.segments("w", [ix0], [yb], [ix1], [yb], 0.8)
            hud.ruler(f, ix0, ix1, yb, t - span, t, 0.25, 1.0, down=True, inten=0.6)
            n = int((ix1 - ix0) / 4)
            tb = t - span + (np.arange(n) + 0.5) / n * span
            lv = ctx.cues.loud_curve(t - span, t, n)
            f.polyline("w", X(tb), yb - 4 - 46 * lv ** 1.3, E.wl(0.55), width=E.ww(L.LW_HAIR))
            kt, ka = ctx.cues.kicks(t - span, t + 1e-6)
            if len(kt):                                 # (a kick / a hit that has just come in grows)
                xk = X(kt)                              # the grid scrolls: the bars slide (hud.bars), they leave under its left end
                hud.bars(f, "w", np.maximum(xk - 2, ix0), yb - (10 + 30 * np.minimum(ka, 1.5)) * pop(t - kt),
                         np.maximum(xk + 3, ix0), yb, 0.95)
            hx0, hy0, hx1, hy1, hi = [], [], [], [], []
            for row, key in enumerate(L.ORDER):
                yl = yb + 24 + row * 20
                f.text("w", ix0, yl, key, size=L.T_MICRO, alpha=0.7)
                tt, ee, ec = ctx.det.hits(key, t - span, t + 1e-6)
                if len(tt):
                    xh = X(tt)
                    g = pop(t - tt)
                    hx0.append(np.maximum(xh, ix0 + 22))             # (they leave before the row label)
                    hx1.append(np.maximum(xh + np.where(ec, 2.0, 3.0), ix0 + 22))
                    hy0.append(yl - np.where(ec, 2 + 7 * ee, 4 + 10 * ee) * g)
                    hy1.append(np.full(len(tt), yl))
                    hi.append(np.where(ec, 0.55, 1.0))
            if hx0:
                hud.bars(f, "r", *(np.concatenate(v) for v in (hx0, hy0, hx1, hy1, hi)))
            f.segments("r", [ix1], [y0 + 44], [ix1], [y1 - 8], 1.1, width=L.LW)
            f.text("w", ix1 - 8, y0 + 25, f"KICK {min(env, 9.99):.2f}   LOUD {ctx.cues.loud(t):.2f}", size=L.T_MICRO, alpha=0.8,
                   anchor="rs")
            f.rects("w", ix1 - 292, y0 + 32, ix1 - 292 + 284 * min(1.0, env), y0 + 37, 0.95)

    # --------------------------------------------------------- bottom band
    def _numbers_panel(self, f, title, cols, erode=0.0, fr=0, key=151, t=0.0):
        """Bottom band, widest free panel: up to three counters = (label, value, layer)."""
        lay = self.lay
        if not lay.p_num:
            return
        x0, x1 = lay.p_num
        x1 = min(x1, x0 + 620.0)
        rect = (x0 - 8.0, lay.y0 - 24.0, x1 + 8.0, lay.y1 + 8.0)
        with self._blk(f, "num", t, rect, erode=erode):
            header(f, x0, x1, lay.y0, title, erode, key, fr)
        cols = cols[: max(1, min(len(cols), int((x1 - x0) / 125.0)))]
        cw = (x1 - x0) / len(cols)
        size = min(fit_size(val, cw - 12.0, 26.0, 48.0) for _, val, _ in cols)       # one size for the row
        with self._blk(f, "num", t, rect):
            for k, (lab, val, layer) in enumerate(cols):
                f.text("w", x0 + k * cw + 4, lay.y0 + 34, er(lab, erode, key + 1 + k, fr), size=L.T_MICRO, alpha=0.75)
                f.text(layer, x0 + k * cw + 2, lay.y0 + 96, er(val, erode * 0.5, key + 4 + k, fr), size=size, alpha=0.97)

    def _barcode_panel(self, f, t, ctx, erode=0.0, fr=0):
        """Bottom band: the three streams of the last 8 s as a barcode (quiet = sparse, hit = solid)."""
        lay = self.lay
        if not lay.p_bar:
            return
        x0, x1 = lay.p_bar
        rect = (x0 - 8.0, lay.y0 - 24.0, x1 + 8.0, lay.y1 + 8.0)
        with self._blk(f, "bar", t, rect, erode=erode):
            header(f, x0, x1, lay.y0, fit_title("STREAM_BARCODE // L C R // LAST 8 S", x1 - x0 - 8), erode, 158, fr)
            f.segments("r", [x1 - 1], [lay.y0 + 8], [x1 - 1], [lay.y1], 1.2, width=L.LW)
        cols_n = int(np.clip((x1 - x0) / 3.2, 60, 150))
        kk, frac, dt = hud.barcode_keys(t, 8.0, cols_n)
        xl, xr = hud.barcode_cols(x0, x1, cols_n, frac)          # the bars slide, they do not jump a column
        lane_h = (lay.y1 - lay.y0 - 12) / 3
        with self._blk(f, "bar", t, rect):
            xa, xb, ya = [], [], []
            for ln, key in enumerate(L.ORDER):
                val = ctx.det.value(key, kk * dt)               # read at the start of its slot: it never changes after
                on = (hash01(kk, ln + 17) < 0.03 + 1.6 * val) & (xr > xl)
                if erode > 0:
                    on = on & (hash01(kk, ln, 159) > erode)
                xa.append(xl[on])
                xb.append(xr[on])
                ya.append(np.full(int(on.sum()), lay.y0 + 12 + ln * lane_h))
            xa, xb, ya = np.concatenate(xa), np.concatenate(xb), np.concatenate(ya)
            hud.bars(f, "w", xa, ya, xb, ya + lane_h - 4, 0.95)

    def _single_panel(self, f, title, value, layer="r", erode=0.0, fr=0, t=0.0):
        """Bottom band, smallest panel: one big figure."""
        lay = self.lay
        if not lay.p_one:
            return
        x0, x1 = lay.p_one
        rect = (x0 - 8.0, lay.y0 - 24.0, x1 + 8.0, lay.y1 + 8.0)
        with self._blk(f, "one", t, rect, erode=erode):
            header(f, x0, x1, lay.y0, title, erode, 160, fr)
        with self._blk(f, "one", t, rect):
            f.text(layer, x0, lay.y0 + 96, value, size=fit_size(value, x1 - x0 - 8.0, 26.0, 48.0), alpha=0.97)

    def _bottom(self, f, t, ctx, past, thr, found):
        expo = max(0.0, t - self.t_first)
        # the density of the core: the figure spins when the core is called, then locks. It has the smallest
        # panel of the band; when the towers leave only two panels it is the fourth counter of the first
        rho = ("RHO_CORE", B.roll("11.3", t - self.t_found, 0.6, key=9) if found else "--.-", "r" if found else "w")
        cols = [("TRACKS", f"{len(past):04d}", "w"), ("SCATTERED", f"{len(thr):04d}", "r" if thr else "w"),
                ("EXPOSURE", f"{int(expo // 60):02d}:{int(expo % 60):02d}", "w")]
        self._numbers_panel(f, "RECONSTRUCTED", cols if self.lay.p_one else cols + [rho], t=t)
        self._barcode_panel(f, t, ctx)
        self._single_panel(f, rho[0], rho[1], rho[2], t=t)
