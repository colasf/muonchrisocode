"""Shared timing, layout and Ikeda-style HUD pieces for the Muon Bloom scenes."""
from __future__ import annotations

import math

import numpy as np

from .engine import hash01

BPM = 128.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
PHRASE = 4 * BAR

STRIP = (40.0, 338.0, 2960.0, 470.0)          # top score strip
MAIN = (40.0, 490.0, 2960.0, 1462.0)          # main view
BOT = (40.0, 1482.0, 2960.0, 1612.0)          # bottom data band
WALL = (0, 318, 3000, 1688)                   # inverted flashes stay on the wall (no white sky)
BLOOM = (0.3, 0.26, 0.2, 0.16, 0.13, 0.1, 0.08, 0.06)

# the three detector towers in front of the wall (layout of ouptut/Scene2.png): x0, x1, top, bottom.
# CENTRE = hero (tallest), LEFT / RIGHT = accents
TOWERS = {"L": (531.0, 629.0, 951.0, 1611.0), "C": (1450.0, 1546.0, 653.0, 1611.0),
          "R": (2367.0, 2465.0, 951.0, 1611.0)}
# bottom-band panels that fit between the towers
GAPS = [(40.0, 500.0), (660.0, 1420.0), (1576.0, 2337.0), (2495.0, 2960.0)]
_GLYPHS = "0123456789ABCDEF#%/*+-=<>"


def timing(t, T):
    """Bar / phrase bookkeeping on the 128 BPM grid."""
    t = t % T
    n_bars = int(round(T / BAR))
    bar = int(t // BAR) % n_bars
    return dict(t=t, bar=bar, phrase=bar // 4, bip=bar % 4, u=(t - bar * BAR) / BAR,
                invert=(t % PHRASE) < 0.06, burst=t >= T - BEAT)


def finish(f, invert, palette=None, palette_mix=1.0):
    return f.finish(bloom_weights=BLOOM, bloom_gain=0.75, invert=invert, invert_rect=WALL, palette=palette,
                    palette_mix=palette_mix)


def show_time(s):
    """Seconds -> MM:SS.mmm show time code."""
    m = int(s // 60)
    return f"{m:02d}:{s - 60 * m:06.3f}"


def section_at(sections, t):
    """(index, progress 0..1) of the section holding t; sections = [(code, name, t0, t1), ...]."""
    for k, sec in enumerate(sections):
        if t < sec[3] or k == len(sections) - 1:
            return k, float(np.clip((t - sec[2]) / (sec[3] - sec[2]), 0.0, 1.0))


def erode(s, amount, key=0, frame=0):
    """Text falling apart: a stable, growing subset of characters vanishes (amount 0..1); the ones about to go
    flicker through random glyphs first. Keeps the length, so anchored text does not move."""
    if amount <= 0.0 or not s:
        return s
    if amount >= 1.0:
        return ""
    h = hash01(np.arange(len(s)), key)
    out = []
    for k, (c, hv) in enumerate(zip(s, h)):
        if c == " " or hv >= amount + 0.12:
            out.append(c)
        elif hv < amount:
            out.append(" ")
        else:
            out.append(_GLYPHS[int(hash01(k, key, frame) * len(_GLYPHS))])
    return "".join(out)


def tower_frame(f, key, inten=0.6, slab=48.0, layer="w"):
    """Outline of a detector tower with its scintillator slabs."""
    x0, x1, top, bot = TOWERS[key]
    if inten <= 0.004:
        return
    f.rect(layer, x0, top, x1, bot, inten)
    ys = np.arange(top + slab, bot - 1, slab)
    f.segments(layer, np.full_like(ys, x0), ys, np.full_like(ys, x0 + 7), ys, 0.8 * inten)
    f.segments(layer, np.full_like(ys, x1 - 7), ys, np.full_like(ys, x1), ys, 0.8 * inten)


def timeline_strip(f, t, T, sections, title, t_show=0.0, alpha=1.0):
    """Score strip of a linear scene: the whole scene on one ruler, its sections on the red band, a cursor.
    Returns (time -> x, red band y) so the scene can add its own combs and tags."""
    x0, y0, x1, y1 = STRIP

    def X(tt):
        return x0 + np.asarray(tt, np.float64) / T * (x1 - x0)

    f.segments("w", [x0, x0], [y0, y1], [x1, x1], [y0, y1], 0.9 * alpha)
    ruler(f, x0, x1, y0, 0.0, T, BEAT, 4 * BAR, down=True, inten=0.8 * alpha)
    ruler(f, x0, x1, y1, 0.0, T, BEAT, 4 * BAR, down=False, inten=0.8 * alpha)
    yb = (y0 + y1) / 2 - 6
    f.rects("r", x0, yb - 3, x1, yb + 3, alpha)
    for code, name, s0, s1 in sections:
        xs = float(X(s0))
        f.segments("w", [xs], [y0 - 6], [xs], [y1 + 6], 0.9 * alpha, width=1.4)
        now = s0 <= t < s1
        f.tag("r" if now else "w", xs + 5, yb + 6, f"{code} {name}", size=13, pad=3,
              alpha=alpha if s0 <= t else 0.45 * alpha)
        f.text("w", xs + 5, y1 - 22, show_time(t_show + s0), size=12, alpha=0.6 * alpha)
    xc = float(X(t))
    f.segments("r", [xc], [y0 - 4], [xc], [y1 + 4], 1.2 * alpha, width=1.6)
    f.tag("r", xc + 6, y1 + 22, f"T {t:07.3f} // {show_time(t_show + t)}", size=13, pad=3, alpha=alpha)
    f.tag("w", x0 + 4, y0 - 10, title, size=12, pad=3, alpha=alpha)
    return X, yb


def lerp_x(v, v0, v1, x0, x1):
    return x0 + (np.asarray(v, np.float64) - v0) / (v1 - v0) * (x1 - x0)


def ruler(f, x0, x1, y, v0, v1, minor, major, fmt=None, down=True, inten=0.8, size=12, layer="w", label_every=None,
          lab_dy=None, lo=None, hi=None):
    """Horizontal ruler: value v0 sits at x0, v1 at x1. Ticks every `minor`, long ticks every `major`."""
    vmin, vmax = min(v0, v1), max(v0, v1)
    if lo is not None:
        vmin = max(vmin, lo)
    if hi is not None:
        vmax = min(vmax, hi)
    k0, k1 = math.ceil(vmin / minor - 1e-9), math.floor(vmax / minor + 1e-9)
    if k1 < k0:
        return
    ks = np.arange(k0, k1 + 1)
    vals = ks * minor
    xs = lerp_x(vals, v0, v1, x0, x1)
    per = max(1, int(round(major / minor)))
    big = (ks % per) == 0
    half = (ks % max(1, per // 2)) == 0
    ln = np.where(big, 12.0, np.where(half, 7.0, 3.5))
    sgn = 1.0 if down else -1.0
    f.segments(layer, xs, np.full_like(xs, y), xs, y + sgn * ln, inten)
    if fmt:
        every = label_every or major
        pe = max(1, int(round(every / minor)))
        dy = lab_dy if lab_dy is not None else (sgn * 28 if down else -18)
        for v, x, k in zip(vals, xs, ks):
            if k % pe == 0:
                f.text(layer, float(x) + 4, y + dy, fmt(v), size=size, alpha=0.75)


def panel_header(f, x0, x1, y, title, layer="w"):
    f.rects("w", x0, y, x1, y + 5, 0.95)
    if title:
        f.tag(layer, x0 + 4, y - 10, title, size=12, pad=3)


def barcode_lanes(f, x0, x1, y0, y1, density, keys, lanes=3, seed=0):
    """Scrolling 'test pattern' barcode: one column per key, lit with the given density (0..1)."""
    n = len(keys)
    cw = (x1 - x0) / n
    lane_h = (y1 - y0) / lanes
    xs = x0 + np.arange(n) * cw
    for ln in range(lanes):
        on = hash01(keys, ln + 3 + seed) < np.asarray(density) * (1.0 - 0.18 * ln)
        f.rects("w", xs[on], y0 + ln * lane_h, xs[on] + cw, y0 + (ln + 1) * lane_h - 3, 0.95)


def barcode_burst(f, rect, t, T, lanes=8, seed=0):
    """The last beat of every loop: the view turns into a test-pattern barcode."""
    u = (t - (T - BEAT)) / BEAT
    fr = int(round(t * 30))
    x0, y0, x1, y1 = rect
    lh = (y1 - y0) / lanes
    span = x1 - x0
    for ln in range(lanes):
        k = np.arange(520)
        wdt = 1 + (hash01(k, ln + 31 + seed) * 9).astype(int)
        on = hash01(k, ln + 7 + seed, 5) < 0.4 + 0.45 * u
        xs = np.cumsum(wdt) - wdt
        xs = (xs + fr * (12 + 5 * ln) * (1 if ln % 2 else -1)) % span + x0
        f.rects("w", xs[on], y0 + ln * lh + 2, xs[on] + wdt[on], y0 + (ln + 1) * lh - 2, 1.0)
    f.rects("r", x0, (y0 + y1) / 2 - 3, x1, (y0 + y1) / 2 + 3, 1.0)


def big_number(f, x, y, label, value, size=64, layer="w", sub=None):
    """Large Ikeda-style readout: small tag above a big figure."""
    f.tag("w" if layer == "w" else "r", x, y - size - 14, label, size=13, pad=3)
    f.text(layer, x - 2, y, value, size=size, alpha=0.97)
    if sub:
        f.text("w", x, y + 24, sub, size=13, alpha=0.7)
