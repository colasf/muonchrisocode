"""Wall proof: a still of the show as it should look on the real wall - four projectors, red brick, at night -
to judge whether the thin lines, the small type and the dim greys survive.

  python tools/wall_proof.py                            the three default stills (7:20.1, 2:58, 1:27)
  python tools/wall_proof.py 5:41.6 9:22                other show times
  python tools/wall_proof.py 2:58 --crops 1183,266 1201,509 1725,942    choose the close-ups (here the red loops
                                                        of the bloom and the white lines of the ground)
  python tools/wall_proof.py 2:58 --even                the same light on the whole wall (no fall-off at the top)
  python tools/wall_proof.py 1:27 --lift 1              with the LIFT of the engine's OUTPUT panel at 1 (the mid levels
                                                        raised before the picture leaves): proof_<time>_lift1_*

Times are show times (M:SS[.cc] or seconds); --crops is the top-left corner of each close-up (420 x 240) in wall
pixels (2978 x 1400). Without it the default stills have theirs, chosen by eye; at any other time the tool takes
the window with the most fine detail in the top band (upper projectors only), in the blend band and in the
bottom band (lower projectors only). Two files per still go to previews/wall and their paths are printed:

  proof_<time>_wall.jpg     the whole wall in the delivery raster (3000 x 1688): 5 lux of stray light, the
                            projectors half a pixel apart
  proof_<time>_sheet.jpg    the close-ups at 2x, each one as it is on the monitor and as it lands on the wall:
                            dark site (2 lux) perfectly aligned / 5 lux, half a pixel / 15 lux, one pixel

The model (the set-up is the one of the projector study; what is guessed is a constant below):
  1. each projector draws the part of the raster it covers on its own 1920 x 1200 pixels (the warp of the media
     server); a pixel gives value ^ 2.2 of light, as a square blurred by the lens;
  2. the four images are laid on the wall, cross-faded where they overlap; the alignment error moves the upper
     and the right-hand ones, which thickens the lines in the overlaps (and doubles them beyond a pixel) and
     only moves the picture elsewhere;
  3. the brick sends back reflectance x (projected light + stray light), the reflectance being read on the
     daytime photo; a fixed exposure and a gamma make the picture.
Not in the model: the black level of the projectors, their colour brightness (a single-chip DLP gives less red
than its white promises), lens fall-off and colour fringes, the relief of the brick (the joints are drawn, not
their shadows), pools of street light, glare and the adaptation of the eye (white light on red brick is shown
as a camera set for daylight sees it: salmon).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muonbloom import layout as L  # noqa: E402
from muonbloom import showdata as sd  # noqa: E402
from muonbloom.engine import font  # noqa: E402

# ---- the site, from the projector study ------------------------------------------------------------------------
WALL_PHOTO = Path(r"D:\muonchristo\Interactive_Stage_1_clean_v5 (1) (1).png")   # daytime photo, in the raster
RASTER_W, RASTER_H = 3000, 1688          # delivery raster: the width of the wall
GROUND = 1667                            # ... and the row of it the ground is on: in the template of the study (the
#                                          photo) the 50 ft of the building are rows 21 to 1666, black above and under
ORIGIN = (11, 272)                       # where the 2978 x 1400 picture sits in it (as in preview.py)
WALL_W, WALL_H = 27.737, 15.24           # m
PPM = RASTER_W / WALL_W                  # 108.16 px per metre: 1 px = 9.25 mm
PX, PY = 1920, 1200                      # one projector
THROW, RATIO = 12.80, 0.84               # lens to wall (m); throw ratio: 15.24 m wide on a plane square to the lens
TOWER_X = (7.007, 20.653)                # the two scaffold towers, along the wall from its left end (m)
LOW = (3.632, 0.0)                       # lens height (m) and tilt (degrees, upwards) of the lower projectors
UP = (5.512, 20.0)                       # ... and of the upper ones
WHITE_LUX = 97.0                         # full white: 9 foot-candles, on average over the wall
GAMMA = 2.2                              # light = (value / 255) ^ 2.2

# ---- the guesses: change them here -----------------------------------------------------------------------------
LENS_BLUR = 0.5                          # the lens: Gaussian sigma, in projector pixels
WARP_IN_LIGHT = False                    # the warp mixes pixel values, as a media server does (True: mixes light)
FALLOFF = True                           # a tilted projector spreads its light over larger pixels towards the top
BRICK = 0.22                             # median reflectance of the wall (luminous)
BRICK_MAX = 0.6                          # ... and the lightest it gets (joints, signs)
TEXTURE = 0.7                            # contrast of the brick read on the photo (an exponent). 1 = as photographed:
#                                          the joints come out at 0.45 - 0.6, more than mortar sends back next to
#                                          a brick at 0.2; 0.7 puts them at 0.3 - 0.4
STRAY_TINT = (1.10, 0.98, 0.82)          # stray light: neutral to warm white (linear RGB)
SKY = 0.16                               # night sky above the roof, in lux x reflectance: it only dresses the view
HEADROOM = 0.9                           # exposure: full white on a mid brick, dark site, lands there (1 = clips)
MONITOR = 0.25                           # pixel of the monitor the proof is read on (mm), for "the wall from ... m"
LIFT = 0.0                               # the LIFT of the engine's OUTPUT panel (--lift): v -> v (1 + lift) / (1 + lift v)
# (stray light in lux, alignment error in projector pixels, caption): the three columns of the sheet
VARIANTS = ((2.0, 0.0, "DARK SITE: 2 LUX, PERFECTLY ALIGNED"),
            (5.0, 0.5, "5 LUX, HALF A PIXEL OFF"),
            (15.0, 1.0, "15 LUX (STREET LIGHTS ON), ONE PIXEL OFF"))
WALL_VARIANT = 1                         # the one of the whole-wall picture
# where one pixel of error sends each projector, in its own pixels (x to the right, y down): the upper ones and
# the right-hand ones move, 120 degrees apart, so that two images side by side or one above the other are exactly
# one error apart (the lower right and the upper left, which only meet in the middle of the blend band: 1.7)
SHIFT = {"LL": (0.0, 0.0), "LR": (0.5, -0.866), "UL": (0.5, 0.866), "UR": (1.0, 0.0)}

SCALE = 2                                # samples per raster pixel on the wall: the close-ups are shown at 2x
CROP_W, CROP_H = 420, 240                # a close-up, in wall pixels (3.9 x 2.2 m)
# the stills made when no time is given, with their close-ups (top band, blend band, bottom band) chosen by eye on
# the show of 2026-10-03; at any other time the tool takes the window with the most fine detail in each band
DEFAULTS = (("7:20.1", ((1552, 247), (1546, 520), (976, 836))),     # glitch: the Standard Model plate twice, Two frames
            ("2:58", ((86, 40), (1340, 589), (2440, 1010))),        # bloom: the title block, the head of DET_C, DET_R
            ("1:27", ((24, 40), (60, 566), (2520, 836))))           # YOU: the header strip, the hit log, the right column

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
TINT = np.array(STRAY_TINT, np.float32) / float(np.dot(STRAY_TINT, LUMA))       # 1 lux of it has a luminance of 1
UMAX = 0.5 / RATIO                       # half the image across, in throws (0.5952)
VMAX = UMAX * PY / PX                    # ... and upwards (0.3720)


@dataclass(frozen=True)
class Proj:
    key: str            # "LL", "LR", "UL", "UR"
    name: str
    x: float            # its tower, along the wall (m)
    h: float            # lens height (m)
    tilt: float         # upwards, radians
    upper: bool
    right: bool


PROJECTORS = tuple(Proj("LU"[u] + "LR"[r], ("lower", "upper")[u] + (" left", " right")[r], TOWER_X[r],
                        (LOW, UP)[u][0], math.radians((LOW, UP)[u][1]), bool(u), bool(r))
                   for u in (0, 1) for r in (0, 1))
LOWER, UPPER = PROJECTORS[0], PROJECTORS[2]     # one of each level, for what depends on the height only


# ----------------------------------------------------------------------------
# geometry: a pixel (p, q) of a projector <-> a point of the wall (x along it, height), in metres
#   u = (p / PX - 0.5) / RATIO,  v = (0.5 - q / PY) * 2 VMAX       (in throws, v upwards)
#   s = THROW / (cos a - v sin a),  x = tower + s u,  height = lens + s (sin a + v cos a)
# a row of the panel lands on a level line of the wall: both resamplings are done rows first, then along the rows
# ----------------------------------------------------------------------------

def _ray(P, hgt):
    """For heights on the wall (m): the row of P that lands there (v) and its distance s along the lens axis
    (that row is s / RATIO wide on the wall)."""
    t = (np.asarray(hgt, np.float64) - P.h) / THROW
    v = (t - math.tan(P.tilt)) / (1.0 + t * math.tan(P.tilt))
    return v, THROW / (math.cos(P.tilt) - v * math.sin(P.tilt))


def _lands(P, v):
    """Height at which the row v of P lands."""
    s = THROW / (math.cos(P.tilt) - v * math.sin(P.tilt))
    return P.h + s * (math.sin(P.tilt) + v * math.cos(P.tilt))


def pixel_mm(P, hgt):
    """Size of a pixel of P on the wall at this height: (across, up), in mm."""
    v, s = _ray(P, hgt)
    t = (hgt - P.h) / THROW
    return 1e3 * s / (RATIO * PX), 1e3 * THROW * (1.0 + t * t) / (1.0 + v * v) / (RATIO * PX)


# the band lit by both levels: from the bottom of the upper images (5.42 m) to the top of the lower ones (8.39 m)
BLEND = (float(_lands(UPPER, -VMAX)), float(_lands(LOWER, VMAX)))
BLEND_ROWS = tuple(GROUND - h * PPM - ORIGIN[1] for h in BLEND[::-1])           # ... in rows of the picture: 488 - 809


def _upper(hgt):
    """Share of the upper projectors at these heights: 0 under the blend band, 1 above, linear across it."""
    return np.clip((np.asarray(hgt, np.float64) - BLEND[0]) / (BLEND[1] - BLEND[0]), 0.0, 1.0)


def _share(P, x, hgt):
    """Cross-fade weight of P at the points (x: m values, hgt: n values, metres) -> (n, m). Where two images
    overlap, the weights are linear across the overlap and sum to 1: the two levels across the blend band, the
    left and the right image of a level from the edge of one to the edge of the other (the upper images are
    keystoned: their overlap widens with the height)."""
    half = UMAX * _ray(P, hgt)[1]                               # half an image of this level, at each height
    a, b = TOWER_X[1] - half, TOWER_X[0] + half                 # the right image starts at a, the left one ends at b
    left = np.clip((b[:, None] - np.asarray(x)[None, :]) / np.maximum(b - a, 1e-6)[:, None], 0.0, 1.0)
    up = _upper(hgt)
    return (up if P.upper else 1.0 - up)[:, None] * (1.0 - left if P.right else left)


def _gain(P, hgt):
    """Light P puts on the wall at these heights, 1 = a projector square to the wall. Tilted, its pixels grow
    with the height and the same light is spread over more brick (1.2 at the foot of the upper images, 0.67 at
    the top of the picture)."""
    hgt = np.asarray(hgt, np.float64)
    if not FALLOFF:
        return np.ones(hgt.shape)
    v, s = _ray(P, hgt)
    t = (hgt - P.h) / THROW
    return THROW * (1.0 + v * v) / (s * (1.0 + t * t))


def _level(hgt):
    """Full white at these heights, 1 = a projector square to the wall."""
    return (1.0 - _upper(hgt)) * _gain(LOWER, hgt) + _upper(hgt) * _gain(UPPER, hgt)


def _mean_level():
    return float(_level(np.linspace(0.0, WALL_H, 3001)).mean())


def white_lux(hgt):
    """Full white on the wall at these heights, in lux: WHITE_LUX is the average over the wall."""
    return WHITE_LUX * _level(hgt) / _mean_level()


# ----------------------------------------------------------------------------
# resampling
# ----------------------------------------------------------------------------

def _tent(z):
    return np.maximum(0.0, 1.0 - np.abs(z))


RK = max(1, math.ceil(0.5 + 3.0 * LENS_BLUR))       # projector pixels mixed around a point of the wall, each way
_KD = np.linspace(0.0, RK + 1.0, 64 * (RK + 1) + 1)
_KV = np.array([0.5 * (math.erf((d + 0.5) / (LENS_BLUR * math.sqrt(2.0))) - math.erf((d - 0.5) / (LENS_BLUR * math.sqrt(2.0))))
                if LENS_BLUR > 0 else float(d < 0.5) for d in _KD])


def _spot(z):
    """A pixel of a projector on the wall, seen across: a square of light blurred by the lens (z = distance to
    its centre, in projector pixels)."""
    return np.interp(np.abs(z), _KD, _KV)


def _resample(src, rows, cols, kern, wr=1.0, wc=1.0, R=2):
    """`src` (h, w, c; sample k at k + 0.5) read at the row coordinates `rows` (n) and at the column coordinates
    `cols` (n, m: every output row has its own). `kern` weights a source sample by its distance, divided by
    wr / wc (a number or one per output row); 2 R samples are mixed each way and their weights sum to 1.
    Outside the source there is nothing (black)."""
    rows, cols = np.asarray(rows, np.float64), np.asarray(cols, np.float64)
    n, m = cols.shape
    wr, wc = (np.broadcast_to(np.asarray(w, np.float64), (n,)) for w in (wr, wc))
    out = np.zeros((n, m, src.shape[2]), np.float32)
    # the piece of the source that is needed, inside a black margin
    r0, r1 = (int(np.clip(v, 0, src.shape[0])) for v in (math.floor(rows.min()) - R - 1, math.ceil(rows.max()) + R + 1))
    c0, c1 = (int(np.clip(v, 0, src.shape[1])) for v in (math.floor(cols.min()) - R - 1, math.ceil(cols.max()) + R + 1))
    if r1 <= r0 or c1 <= c0:
        return out
    pad = np.zeros((r1 - r0 + 2 * R, c1 - c0 + 2 * R, src.shape[2]), np.float32)
    pad[R:-R, R:-R] = src[r0:r1, c0:c1]
    taps = np.arange(1 - R, R + 1)
    # rows first: every output row is a mix of 2 R rows of the source
    f = np.floor(rows - 0.5)
    w = kern(((rows - 0.5 - f)[:, None] - taps[None, :]) / wr[:, None])
    w = (w / np.maximum(w.sum(1, keepdims=True), 1e-12)).astype(np.float32)
    idx = np.clip(f.astype(np.int64)[:, None] + taps[None, :] - r0 + R, 0, pad.shape[0] - 1)
    mid = np.zeros((n, pad.shape[1], src.shape[2]), np.float32)
    for k in range(2 * R):
        mid += w[:, k, None, None] * pad[idx[:, k]]
    # then along each row, a block of rows at a time
    wid = pad.shape[1]
    for a in range(0, n, 256):
        b = min(n, a + 256)
        f = np.floor(cols[a:b] - 0.5)
        d = cols[a:b] - 0.5 - f
        f = f.astype(np.int64) - c0 + R + (np.arange(b - a) * wid)[:, None]
        ws = [kern((d - t) / wc[a:b, None]).astype(np.float32) for t in taps]
        tot = np.maximum(sum(ws), 1e-12)
        flat = mid[a:b].reshape(-1, src.shape[2])
        lo, hi = (np.arange(b - a) * wid)[:, None], (np.arange(b - a) * wid + wid - 1)[:, None]
        for t, wk in zip(taps, ws):
            out[a:b] += (wk / tot)[..., None] * flat[np.clip(f + t, lo, hi)]
    return out


def _grid(box, scale):
    """Sample points of a piece of the wall (box = x0, y0, x1, y1 in raster pixels), `scale` per pixel."""
    x0, y0, x1, y1 = box
    return (x0 + (np.arange(round((x1 - x0) * scale)) + 0.5) / scale,
            y0 + (np.arange(round((y1 - y0) * scale)) + 0.5) / scale)


def _zoom(a, box, scale=SCALE):
    """A piece of a raster-size map on that grid (bilinear)."""
    X, Y = _grid(box, scale)
    return _resample(a, Y, np.broadcast_to(X, (len(Y), len(X))), _tent, R=1)


# ----------------------------------------------------------------------------
# the projectors
# ----------------------------------------------------------------------------

def draw(raster, P):
    """What P sends: the raster under each of its pixels (PY x PX x 3), in light. The warp reads the raster
    with a bilinear filter, widened into an average where a projector pixel covers more than a raster pixel
    (the top of the upper images)."""
    v = (0.5 - (np.arange(PY) + 0.5) / PY) * 2.0 * VMAX
    s = THROW / (math.cos(P.tilt) - v * math.sin(P.tilt))
    Y = GROUND - _lands(P, v) * PPM
    u = ((np.arange(PX) + 0.5) / PX - 0.5) / RATIO
    X = (P.x + s[:, None] * u[None, :]) * PPM
    wy = np.maximum(1.0, np.abs(np.gradient(Y)))                # a projector pixel, in raster pixels
    wx = np.maximum(1.0, s * PPM / (RATIO * PX))
    if WARP_IN_LIGHT:
        return _resample(raster ** GAMMA, Y, X, _tent, wy, wx, 2)
    return _resample(raster, Y, X, _tent, wy, wx, 2) ** GAMMA


def wall_light(panels, box, err, scale=SCALE):
    """Projected light on a piece of the wall (box = x0, y0, x1, y1 in raster pixels, `scale` samples per pixel):
    1 = the average full white. Every pixel of a projector is a square of light blurred by the lens; `err` is
    the alignment error, in projector pixels."""
    X, Y = _grid(box, scale)
    x, hgt = X / PPM, (GROUND - Y) / PPM
    out = np.zeros((len(Y), len(X), 3), np.float32)
    mean = _mean_level()
    for P in PROJECTORS:
        dx, dy = (err * c for c in SHIFT[P.key])
        v, s = _ray(P, hgt)
        q = (0.5 - v / (2.0 * VMAX)) * PY - dy
        rr = np.nonzero((q > -RK) & (q < PY + RK))[0]           # the rows and the columns its image reaches
        if not len(rr):
            continue
        ra, rb = rr[0], rr[-1] + 1
        cc = np.nonzero(np.abs(x - P.x) < s[ra:rb].max() * (0.5 + (RK + abs(dx)) / PX) / RATIO)[0]
        if not len(cc):
            continue
        ca, cb = cc[0], cc[-1] + 1
        p = (0.5 + RATIO * (x[None, ca:cb] - P.x) / s[ra:rb, None]) * PX - dx
        w = _share(P, x[ca:cb], hgt[ra:rb]) * (_gain(P, hgt[ra:rb]) / mean)[:, None]
        out[ra:rb, ca:cb] += _resample(panels[P.key], q[ra:rb], p, _spot, R=RK) * w[..., None].astype(np.float32)
    return out


# ----------------------------------------------------------------------------
# the brick
# ----------------------------------------------------------------------------

def load_wall():
    """The wall, from the daytime photo: its reflectance (`albedo`, raster size, per channel), the night backdrop
    above the roof (`glow`), where the brick is (`brick`), the `exposure` of the pictures and the `spread` of the
    reflectance (10th and 90th percentile). The photo gives the texture and the colour, not the light: taken to
    linear light, it is scaled so that the brick has a median luminance of BRICK, and capped at BRICK_MAX."""
    rgb = np.asarray(Image.open(WALL_PHOTO).convert("RGB"), np.float32) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)
    lum = lin @ LUMA
    x0, y0 = ORIGIN
    face = (slice(y0 + 60, y0 + 1000), slice(x0 + 60, x0 + L.W - 60))          # brick only: no roof, cars or signs
    al = np.minimum(BRICK * np.maximum(lum / float(np.median(lum[face])), 1e-6) ** TEXTURE, BRICK_MAX)
    albedo = lin * (al / np.maximum(lum, 1e-6))[..., None]      # luminous reflectance `al`, the colour of the photo
    np.clip(albedo, 0.0, 0.9, out=albedo)
    # above the roof line there is no wall: the sky and the tower behind (bluish) stay as a dim night backdrop
    brick = (lin[..., 0] > 1.4 * lin[..., 2]) & (lum > 0.004)
    run = np.cumsum(brick, axis=0)
    run[8:] -= run[:-8].copy()
    roof = np.where((run >= 8).any(0), np.argmax(run >= 8, axis=0) - 7, RASTER_H)      # first brick of each column
    back = (np.arange(RASTER_H)[:, None] < roof[None, :]) & (lin[..., 0] < 1.1 * lin[..., 2])
    sky = lum[back & (lum > 0.004)]
    glow = np.where(back[..., None], lin * (SKY / float(np.median(sky))) if sky.size else 0.0, 0.0)
    albedo[back] = 0.0
    # exposure: the strongest channel of full white on a mid brick, in the dark-site case, lands at HEADROOM
    mid = float(np.median(albedo[face].reshape(-1, 3), axis=0).max())
    top = float(white_lux(np.linspace(0.0, WALL_H, 3001)).max())
    return dict(albedo=albedo, glow=glow.astype(np.float32), brick=brick & ~back,
                exposure=HEADROOM / (mid * (top + VARIANTS[0][0])), spread=np.percentile(al[face], (10, 90)))


def on_brick(light, wall, stray):
    """The picture of the wall (or of a piece of it: its maps cut with _zoom): projected light (1 = the average
    full white) and `stray` lux on the brick, a fixed exposure, a gamma."""
    lum = wall["albedo"] * (WHITE_LUX * light + stray * TINT) + wall["glow"]
    return (np.clip(lum * wall["exposure"], 0.0, 1.0) ** (1.0 / GAMMA) * 255.0 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------
# close-ups
# ----------------------------------------------------------------------------

def pick(img, brick):
    """Where to look closely: in each band (top: upper projectors only, blend band, bottom: lower ones only) the
    window that holds the most fine detail - thin lines and small type, the dim ones counted twice - on brick
    (not on the cars, the shelter or the signs of the photo)."""
    v = img.max(2).astype(np.float32)

    def thin(axis):                                             # bright, with dark 3 px away on both sides
        return (np.roll(v, 3, axis) < 0.5 * v) & (np.roll(v, -3, axis) < 0.5 * v)

    def windows(a):                                             # sum of `a` over the window at each top-left corner
        acc = np.zeros((L.H + 1, L.W + 1))
        acc[1:, 1:] = a.cumsum(0).cumsum(1)
        return acc[CROP_H:, CROP_W:] - acc[:-CROP_H, CROP_W:] - acc[CROP_H:, :-CROP_W] + acc[:-CROP_H, :-CROP_W]

    score = windows(((v > 24) & (thin(0) | thin(1))) * (1.0 + (v < 180)))
    x0, y0 = ORIGIN
    clear = windows(brick[y0: y0 + L.H, x0: x0 + L.W].astype(np.float64)) >= 0.97 * CROP_W * CROP_H
    b0, b1 = BLEND_ROWS
    out = []
    for y0, y1 in ((0, int(b0)), (math.ceil(b0), int(b1)), (math.ceil(b1), L.H)):
        s, ok = score[y0: y1 - CROP_H + 1], clear[y0: y1 - CROP_H + 1]
        j, i = np.unravel_index(int(np.argmax(np.where(ok, s, -1.0) if ok.any() else s)), s.shape)
        out.append((int(i), int(y0 + j)))
    return out


def where(x0, y0):
    """Caption of a close-up: its band, its place, the projectors that light it (their share of the light, on
    average over the close-up) and the size of their pixels at its centre."""
    X, Y = _grid((ORIGIN[0] + x0, ORIGIN[1] + y0, ORIGIN[0] + x0 + CROP_W, ORIGIN[1] + y0 + CROP_H), 0.1)
    x, hgt = X / PPM, (GROUND - Y) / PPM
    mid = float(hgt.mean())
    band = "TOP BAND" if mid > BLEND[1] else "BOTTOM BAND" if mid < BLEND[0] else "BLEND BAND"
    parts = []
    for P in PROJECTORS:
        w = float(_share(P, x, hgt).mean())
        if w > 0.02:
            parts.append("{} {:.0f} % (pixels {:.1f} x {:.1f} mm)".format(P.name, 100 * w, *pixel_mm(P, mid)))
    return (f"{band}   //   x {x0} - {x0 + CROP_W}, y {y0} - {y0 + CROP_H}   //   {CROP_W / PPM:.1f} x {CROP_H / PPM:.1f} m, "
            f"{mid:.1f} m above the ground   //   full white {float(white_lux(mid)):.0f} lux   //   " + " + ".join(parts))


# ----------------------------------------------------------------------------
# one still
# ----------------------------------------------------------------------------

_show = None
_wall = None


def _init(even=False, lift=0.0):
    global _show, _wall, FALLOFF, LIFT
    FALLOFF = FALLOFF and not even
    LIFT = lift
    from muonbloom.show import Show
    _show = Show()
    _wall = load_wall()


def _proof(job):
    """One still, (show time, close-ups or None): renders it, projects it, writes the two files."""
    t, crops = job
    ts = time.time()
    img = _show.render(t, L.W, L.H)
    if LIFT > 0.0:                                              # as the engine does it: on the strongest channel
        v = img / np.float32(255.0)
        img = (v * (1.0 + LIFT) / (1.0 + LIFT * v.max(-1, keepdims=True)) * 255.0 + 0.5).astype(np.uint8)
    x0, y0 = ORIGIN
    raster = np.zeros((RASTER_H, RASTER_W, 3), np.float32)
    raster[y0: y0 + L.H, x0: x0 + L.W] = img / np.float32(255.0)
    panels = {P.key: draw(raster, P) for P in PROJECTORS}
    d = ROOT / "previews" / "wall"
    d.mkdir(parents=True, exist_ok=True)
    tag = f"{int(t // 60):02d}m{t % 60:05.2f}s" + ("" if FALLOFF else "_even") + (f"_lift{LIFT:g}" if LIFT else "")
    title = (f"MUON : BLOOM   //   WALL PROOF   //   {sd.tc(t)}   {' '.join(sd.section_at(t)[1][:2])}"
             + (f"   //   LIFT {LIFT:g}" if LIFT else ""))
    sep = "   //   "
    mm, top = 1e3 / PPM, (GROUND - y0) / PPM                    # a pixel of the picture on the wall (mm), its top (m)
    white = f"full white {WHITE_LUX:.0f} lux on average" + (
        f" ({float(white_lux(1.0)):.0f} low on the wall, {float(white_lux(top)):.0f} at the top of the picture: the "
        "upper projectors are tilted)" if FALLOFF else " (the same everywhere)")

    # the whole wall, computed at 2x and averaged down to the raster
    lux, err, cap = VARIANTS[WALL_VARIANT]
    light = wall_light(panels, (0, 0, RASTER_W, RASTER_H), err)
    whole = Image.fromarray(on_brick(light.reshape(RASTER_H, SCALE, RASTER_W, SCALE, 3).mean((1, 3)), _wall, lux))
    note = sep.join([title, cap, white, f"1 pixel = {mm:.2f} mm: at 100 % and 60 cm, the wall from about "
                     f"{0.6 * mm / MONITOR:.0f} m"])
    ImageDraw.Draw(whole).text((16, RASTER_H - 19), note, font=font(13), fill=(150, 150, 150))
    p_wall = d / f"proof_{tag}_wall.jpg"
    whole.save(p_wall, quality=95, subsampling=0)

    # the sheet: one row per close-up, the monitor then the three conditions
    crops = crops or pick(img, _wall["brick"])
    tw, th, pad, head, lab = CROP_W * SCALE, CROP_H * SCALE, 14, 100, 30
    f0, f1, f2, f3 = font(30, True), font(20, True), font(18), font(17)
    foot = [sep.join([white, f"light = (value / 255) ^ {GAMMA}", "stray light: warm white, even over the wall",
                      "alignment error = the distance between two images side by side or one above the other"]),
            sep.join(["brick: reflectance {} (median), {:.2f} to {:.2f} for most of it, {} at most: the daytime photo in "
                      "linear light, its contrast ^ {}".format(BRICK, *_wall["spread"], BRICK_MAX, TEXTURE),
                      f"lens: Gaussian blur, sigma {LENS_BLUR} projector pixel",
                      "warp: bilinear, on the " + ("light" if WARP_IN_LIGHT else "pixel values"),
                      "overlaps cross-faded, linear"]),
            sep.join(["projector pixels: {:.1f} mm (lower), {:.1f} x {:.1f} mm at the foot of the upper images, {:.1f} x "
                      "{:.1f} mm at the top of the picture".format(pixel_mm(LOWER, 1.0)[0], *pixel_mm(UPPER, BLEND[0]),
                                                                   *pixel_mm(UPPER, top)),
                      f"{SCALE} x {SCALE} pixels here = 1 pixel of the picture = {mm:.2f} mm of wall: at 100 % and 60 cm, "
                      f"the wall from about {0.6 * mm / SCALE / MONITOR:.0f} m",
                      "one exposure for all: full white on a mid brick just under display white"])]
    sheet = Image.new("RGB", (4 * tw + 5 * pad, head + len(crops) * (th + lab + pad) + len(foot) * 26 + pad), (24, 24, 24))
    dr = ImageDraw.Draw(sheet)
    dr.text((pad + 2, 14), title, font=f0, fill=(255, 255, 255))
    for c, name in enumerate(["ON THE MONITOR"] + ["ON THE WALL   //   " + v[2] for v in VARIANTS]):
        dr.text((pad + c * (tw + pad) + 2, 62), name, font=f1, fill=(255, 214, 0))
    for r, (cx, cy) in enumerate(crops):
        y = head + r * (th + lab + pad)
        dr.text((pad + 2, y + 2), where(cx, cy), font=f2, fill=(230, 230, 230))
        box = (x0 + cx, y0 + cy, x0 + cx + CROP_W, y0 + cy + CROP_H)
        piece = dict(_wall, albedo=_zoom(_wall["albedo"], box), glow=_zoom(_wall["glow"], box))
        tiles = [img[cy: cy + CROP_H, cx: cx + CROP_W].repeat(SCALE, 0).repeat(SCALE, 1)]
        tiles += [on_brick(wall_light(panels, box, e), piece, s) for s, e, _ in VARIANTS]
        for c, tile in enumerate(tiles):
            sheet.paste(Image.fromarray(tile), (pad + c * (tw + pad), y + lab))
    for k, line in enumerate(foot):
        dr.text((pad + 2, sheet.size[1] - pad - 26 * (len(foot) - k) + 2), line, font=f3, fill=(150, 150, 150))
    p_sheet = d / f"proof_{tag}_sheet.jpg"
    sheet.save(p_sheet, quality=95, subsampling=0)
    return p_wall, p_sheet, sheet.size, crops, time.time() - ts


def parse_time(s):
    s = str(s)
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("times", nargs="*", help="show times (M:SS[.cc] or seconds)")
    ap.add_argument("--crops", nargs="+", default=None, metavar="X,Y",
                    help="top-left corner of each close-up, in wall pixels (one row of the sheet each)")
    ap.add_argument("--even", action="store_true", help="the same light on the whole wall (no fall-off at the top)")
    ap.add_argument("--lift", type=float, default=0.0, help="the LIFT of the engine's OUTPUT panel (0 = none)")
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 4)))
    a = ap.parse_args()
    if not WALL_PHOTO.exists():
        sys.exit(f"the photo of the wall is missing: {WALL_PHOTO}")
    chosen = [(parse_time(t), list(c)) for t, c in DEFAULTS]
    times = [parse_time(x) for x in a.times] or [t for t, _ in chosen]
    crops = None
    if a.crops:
        crops = [tuple(int(float(v)) for v in c.split(",")) for c in a.crops]
        crops = [(min(max(x, 0), L.W - CROP_W), min(max(y, 0), L.H - CROP_H)) for x, y in crops]
    jobs = [(t, crops or next((c for td, c in chosen if abs(td - t) < 1e-3), None)) for t in times]
    with Pool(min(a.workers, len(times)), initializer=_init, initargs=(a.even, max(0.0, a.lift))) as pool:
        for p_wall, p_sheet, size, used, dt in pool.imap(_proof, jobs):
            print(p_wall, (RASTER_W, RASTER_H), flush=True)
            print(p_sheet, size, "close-ups at " + " ".join(f"{x},{y}" for x, y in used), f"({dt:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
