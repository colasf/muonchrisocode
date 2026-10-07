"""Shared furniture of the show: frame, subtitle box, edge meters, score strips, panels, tags.

The frame, the subtitle box (top right) and the edge ticks come from the TouchDesigner scenes;
the score strip, the data columns and the bottom-band panels come from the Ikeda-style studies.
Every scene is built from these so the whole performance reads as one instrument.
"""
from __future__ import annotations

import math

import numpy as np

from . import build as B
from . import engine as E
from . import layout as L
from . import showdata as sd
from .engine import CHAR_W, hash01, smoothstep, text_w

BLOOM = (0.3, 0.26, 0.2, 0.16, 0.13, 0.1, 0.08, 0.06)
_GLYPHS = "0123456789ABCDEF#%/*+-=<>"
SUB_CPS = 46.0                        # typing speed of the subtitles (chars / s)
NOGLOW_PAD = 6.0                      # the subtitle has no glow (Frame.noglow_rects): its box, and this much around it
SUB_OUT = 0.3                         # seconds a subtitle line takes to be un-typed when it leaves
HERO_MAX = 16                         # single-cue paragraphs this short are set as a tag (YOU, A bloom, ...)


def finish(f, invert=False, invert_rect=None, bloom_gain=0.75, exposure=1.0):
    return f.finish(bloom_weights=BLOOM, bloom_gain=bloom_gain, invert=invert,
                    invert_rect=invert_rect or (0, 0, L.W, L.H), exposure=exposure)


# ----------------------------------------------------------------------------
# frame + subtitles + edge meters (drawn by the show on top of every scene)
# ----------------------------------------------------------------------------

def frame(f, alpha=1.0):
    x0, y0, x1, y1 = L.FRAME
    f.rect("w", x0, y0, x1, y1, 0.95 * alpha, width=L.LW_FRAME)


def wrap(text, width):
    """Greedy word wrap into at most 2 balanced lines of `width` chars."""
    if len(text) <= width:
        return [text]
    words = text.split()
    best = None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        if len(a) <= width and len(b) <= width:
            # balanced lines, the first one rather the longer, and a break after punctuation if there is one
            score = abs(len(a) - len(b)) + (6 if len(a) < len(b) else 0) - (10 if a[-1] in ",.;:" else 0)
            if best is None or score < best[0]:
                best = (score, [a, b])
    if best:
        return best[1]
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur]


HERO = {"you", "a muon", "a bloom", "waiting", "almost nothing", "nothing"}     # the words set as a tag


def _is_hero(cue):
    return cue.text.lower().strip(". ") in HERO


BOX_HOLD = 1.2                        # s: a pause shorter than this leaves no time to close the box and open it again


def subtitle_box_alpha(t):
    """The box is there while somebody speaks (and across short pauses inside a paragraph).
    0..1 = how far it is open: it slides open from the right edge of the frame before the first word and
    closes again after the last one (it does not fade). Between two lines less than BOX_HOLD apart it stays
    open whatever the paragraph: it would only start to close and come back (and the score strip with it)."""
    cues = [c for c in sd.subtitles() if not _is_hero(c)]         # the single words are set as tags, no box
    a = 0.0
    for k, c in enumerate(cues):
        if c.t - 0.35 <= t < c.end + 0.7:
            a = max(a, float(smoothstep(c.t - 0.35, c.t - 0.1, t) * (1 - smoothstep(c.end + 0.2, c.end + 0.7, t))))
        if k + 1 < len(cues):
            n = cues[k + 1]
            gap = n.t - c.end
            if c.end <= t < n.t and ((n.group == c.group and gap < 4.0) or gap < BOX_HOLD):
                a = 1.0
    return a


def _hero_tag_w(cue, size=86):
    return len(cue.text.strip(". ")) * size * CHAR_W


def top_right_edge(t, rect=None):
    """Left edge (x) of whatever stands in the top-right corner at time t: the subtitle box while it slides
    open / closed, the tag of a single word (YOU, A MUON ...), or the frame edge when nobody speaks."""
    x0, _, x1, _ = rect or L.SUB
    edge = x1 - (x1 - x0) * float(B.ease(subtitle_box_alpha(t)))
    for c in sd.subtitles():
        if _is_hero(c) and c.t - 0.45 <= t < c.end + 0.55:
            a = float(smoothstep(c.t - 0.45, c.t - 0.05, t) * (1 - smoothstep(c.end + 0.05, c.end + 0.55, t)))
            edge = min(edge, x1 - (_hero_tag_w(c) + 86.0) * float(B.ease(a)))
    return edge


def strip_rect(t, rect=None):
    """The score strip of this frame: it takes the whole header band when the top-right corner is free and
    is scaled back as the subtitle box slides open (the show stores it in L.STRIP before a scene draws)."""
    x0, y0, _, y1 = L.STRIP0
    return (x0, y0, max(top_right_edge(t, rect), x0 + 400.0), y1)


def subtitle(f, t, rect=None):
    """Voice-over text in the box at the top right, typed on. Clears whatever is behind the box.
    The subtitle does not glow: plain white letters in a plain border (also the single words set as a tag).
    Nothing fades: the box opens from the right edge of the frame and closes back to it, the single words
    set as a tag are made (box pushed out, letters decoded) and taken apart, a line that leaves is un-typed."""
    box_a = subtitle_box_alpha(t)
    x0, y0, x1, y1 = rect or L.SUB
    cue, age = sd.subtitle_at(t)
    if box_a <= 0.01 and cue is None:
        return
    if cue is not None and _is_hero(cue):
        # a single word: no box, just the tag in the corner (so a bloom on the centre tower stays whole)
        s = cue.text.strip(". ").upper()
        size = 86
        w = _hero_tag_w(cue, size)
        f.occlude(x1 - 44 - w - 18, y0 + 14, x1 - 12, y0 + 156)
        f.noglow_rects.append((x1 - 44 - w - 18, y0 + 14, x1 - 12, y0 + 156))
        cps = 40.0
        B.tag(f, "w", x1 - 44 - w, y0 + 118, s, B.io(age, cue.end - t, out=0.25, span=(len(s) + 4) / cps), size=size,
              pad=14, bold=True, cps=cps, key=7)
        return
    if box_a <= 0.01:
        return
    xa = x1 - (x1 - x0) * float(B.ease(box_a))            # the plate slides open from the frame edge
    f.occlude(xa, y0, x1, y1)
    f.noglow_rects.append((xa - NOGLOW_PAD, y0 - NOGLOW_PAD, x1 + NOGLOW_PAD, y1 + NOGLOW_PAD))    # border included
    f.rect("w", xa, y0, x1, y1, 0.95, width=L.LW_FRAME)
    if box_a < 1.0:
        f.dots("w", [xa], [y1], 3.6, 1.7)
    if cue is None or box_a < 1.0:
        return
    size = L.T_SUB
    cols = int((x1 - x0 - 44) / (size * CHAR_W))
    lines = wrap(cue.text, cols)
    if len(lines) > 2:
        size = int(size * 2 / len(lines) * 0.98)
        lines = wrap(cue.text, int((x1 - x0 - 44) / (size * CHAR_W)))
    dur = cue.end - cue.t
    cps = max(SUB_CPS, len(cue.text) / max(0.35 * dur, 0.1))     # a short line is typed faster: it has to be read
    n = int(age * cps) + 1
    left = cue.end - t
    t_out = min(SUB_OUT, 0.15 * dur)                              # ... and un-typed faster
    out = dur > 1.0 and left < t_out                     # the line leaves the way it came: un-typed from its end
    if out:
        n = min(n, int(sum(len(ln) + 1 for ln in lines) * max(left, 0.0) / t_out))
    tx = x0 + 22
    typed = 0
    for k, ln in enumerate(lines):
        m = max(0, min(len(ln), n - typed))
        yb = y0 + 64 + k * 80 if len(lines) > 1 else y0 + 64
        f.text("w", tx, yb, ln[:m], size=size)
        if 0 < n - typed <= len(ln) + 1 and (out or int(t * 6) % 2 == 0):  # block cursor while typing / un-typing
            cx = tx + text_w(ln[:m], size) + 4
            f.rects("w", cx, yb - size * 0.62, cx + size * 0.5, yb + size * 0.1, 0.9)
        typed += len(ln) + 1


def edge_ticks(f, t, cues, alpha=1.0, y0=250.0, y1=1230.0, left=True, right=True, reveal=1.0):
    """Tick meters inside the left / right frame edges (the TouchDesigner rulers): 48 spectrum bands.
    reveal < 1: the meters are being built (they never fade in). A bright head runs down the left edge and up
    the right one; every tick shoots out long as the head passes, then falls back to its level."""
    sp = cues.spec(t)
    n = len(sp)
    j = hash01(np.arange(n), 5)
    ys = y0 + (np.arange(n) + 0.15 + 0.7 * j) / n * (y1 - y0)
    ln = 7.0 + 46.0 * sp ** 1.6
    dbl = hash01(np.arange(n), 9) < 0.22
    head = reveal * (n + 9.0)                   # index the head has reached (the last ticks still have to settle)
    for side, on in ((0, left), (1, right)):
        if not on:
            continue
        order = np.arange(n) if side == 0 else np.arange(n)[::-1]
        xe = L.FX0 if side == 0 else L.FX1
        sgn = 1.0 if side == 0 else -1.0
        le = ln[order]
        shown = np.ones(n, bool)
        if reveal < 1.0:
            rank = np.arange(n) if side == 0 else np.arange(n)[::-1]       # left: top to bottom, right: bottom to top
            shown = rank < head
            le = le + (74.0 - le) * np.exp(-np.maximum(head - rank, 0.0) / 2.2)
            if head < n:
                yh = y0 + (head if side == 0 else n - head) / n * (y1 - y0)
                f.dots("w", [xe], [yh], 3.6, 1.7)
        f.segments("w", np.full(shown.sum(), xe), ys[shown], xe + sgn * le[shown], ys[shown], 0.85 * alpha, width=L.LW)
        d = dbl[order] & shown
        f.segments("w", np.full(d.sum(), xe), ys[d] + 5.0, xe + sgn * le[d] * 0.6, ys[d] + 5.0, E.wl(0.6 * alpha), width=E.ww(1.0))


# ----------------------------------------------------------------------------
# generic pieces
# ----------------------------------------------------------------------------

def lerp_x(v, v0, v1, x0, x1):
    return x0 + (np.asarray(v, np.float64) - v0) / (v1 - v0) * (x1 - x0)


def ruler(f, x0, x1, y, v0, v1, minor, major, fmt=None, down=True, inten=0.8, size=L.T_MICRO, layer="w",
          label_every=None, lab_dy=None, lo=None, hi=None, reveal=None):
    """Horizontal ruler: value v0 sits at x0, v1 at x1. Ticks every `minor`, long ticks every `major`.
    reveal (0..1, None = built): the ticks arrive one after the other from x0 to x1, each one too long at
    first, then falling back to its length."""
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
    ln = np.where(big, 13.0, np.where(half, 8.0, 4.0))
    if reveal is not None and reveal < 1.0:
        if reveal <= 0.0:
            return
        pos = np.abs(xs - x0) / max(abs(x1 - x0), 1e-6)
        head = reveal * 1.25
        m = pos <= head
        ks, vals, xs = ks[m], vals[m], xs[m]
        ln = ln[m] * (1.0 + 1.8 * np.exp(-(head - pos[m]) / 0.07))
    sgn = 1.0 if down else -1.0
    f.segments(layer, xs, np.full_like(xs, y), xs, y + sgn * ln, E.wl(inten), width=E.ww(1.0))
    if fmt:
        every = label_every or major
        pe = max(1, int(round(every / minor)))
        dy = lab_dy if lab_dy is not None else (sgn * 30 if down else -19)
        for v, x, k in zip(vals, xs, ks):
            if k % pe == 0:
                f.text(layer, float(x) + 4, y + dy, fmt(v), size=size, alpha=0.75)


def vruler(f, x, y0, y1, v0, v1, minor, major, fmt=None, right=True, inten=0.8, size=L.T_MICRO, layer="w"):
    """Vertical ruler: value v0 at y0, v1 at y1."""
    vmin, vmax = min(v0, v1), max(v0, v1)
    k0, k1 = math.ceil(vmin / minor - 1e-9), math.floor(vmax / minor + 1e-9)
    if k1 < k0:
        return
    ks = np.arange(k0, k1 + 1)
    vals = ks * minor
    ys = y0 + (vals - v0) / (v1 - v0) * (y1 - y0)
    per = max(1, int(round(major / minor)))
    ln = np.where((ks % per) == 0, 13.0, np.where((ks % max(1, per // 2)) == 0, 8.0, 4.0))
    sgn = 1.0 if right else -1.0
    f.segments(layer, np.full_like(ys, x), ys, x + sgn * ln, ys, E.wl(inten), width=E.ww(1.0))
    if fmt:
        for v, y, k in zip(vals, ys, ks):
            if k % per == 0:
                f.text(layer, x + sgn * 18, float(y) + 5, fmt(v), size=size, alpha=0.75,
                       anchor="ls" if right else "rs")


def panel_header(f, x0, x1, y, title, layer="w", alpha=1.0, age=None):
    """Thick rule with a small inverted title tag sitting on it.
    age = seconds since the panel started to build (None = built): a pen draws a hairline across, the thick
    rule follows it, then the tag is made (box, then letters out of noise)."""
    if age is None:
        f.rects("w", x0, y, x1, y + 5, 0.95 * alpha)
        if title:
            f.tag(layer, x0 + 4, y - 9, title, size=L.T_MICRO, pad=3, alpha=alpha)
        return
    if age < 0.0:
        return
    p2 = float(B.ease(B.lin(age, 0.08, 0.34)))
    if p2 < 1.0:
        B.pen(f, "w", x0, y + 2.5, x1, y + 2.5, B.ease(B.lin(age, 0.0, 0.2)), 0.9 * alpha, width=1.3)
    f.rects("w", x0, y, x0 + (x1 - x0) * p2, y + 5, 0.95 * alpha)
    if title:
        B.tag(f, layer, x0 + 4, y - 9, title, age, t0=0.16, size=L.T_MICRO, pad=3, alpha=alpha, cps=90.0)


def big_number(f, x, y, label, value, size=64, layer="w", sub=None, alpha=1.0):
    """Large readout: small tag above a big figure."""
    f.tag("w" if layer == "w" else "r", x, y - size - 14, label, size=L.T_MICRO, pad=3, alpha=alpha)
    f.text(layer, x - 2, y, value, size=size, alpha=0.97 * alpha)
    if sub:
        f.text("w", x, y + 26, sub, size=L.T_MICRO, alpha=0.7 * alpha)


def rows(f, x, y, lines, size=L.T_SMALL, lead=1.5, alpha=0.9, layer="w", red=()):
    """Parameter rows (monospace table)."""
    for k, ln in enumerate(lines):
        f.text("r" if k in red else layer, x, y + k * size * lead, ln, size=size, alpha=alpha)
    return y + len(lines) * size * lead


def bars(f, layer, x0, y0, x1, y1, i):
    """Filled bars that slide smoothly. Frame.rects snaps to whole pixels: a bar that moves by a fraction of a
    pixel per frame would advance in uneven jumps and change width on the way. Here the two edge columns of
    a bar are lit by the fraction of the pixel it covers, so it glides at any speed (its top and bottom stay
    snapped: a barcode does not move up or down)."""
    x0, x1 = np.atleast_1d(np.asarray(x0, np.float64)), np.atleast_1d(np.asarray(x1, np.float64))
    shape = np.broadcast_shapes(x0.shape, x1.shape, np.shape(y0), np.shape(y1), np.shape(i))
    if len(shape) != 1 or shape[0] == 0:
        return
    x0, x1, y0, y1, i = (np.broadcast_to(np.asarray(v, np.float64), shape) for v in (x0, x1, y0, y1, i))
    k = f.s * f.vz
    off = f.vsx * f.s - f.vcx * k               # Frame.tx(x) = x * k + off
    a, b = x0 * k + off + 0.5, x1 * k + off + 0.5       # window coordinates: pixel j covers [j, j + 1)
    m = b > a
    if not m.any():
        return
    a, b, y0, y1, i = a[m], b[m], y0[m], y1[m], i[m]
    pa, pb = np.floor(a), np.floor(b)
    one = pa == pb                              # the whole bar inside one pixel column
    px0 = np.concatenate([pa, pa[~one] + 1.0, pb[~one]])
    px1 = np.concatenate([pa + 1.0, pb[~one], pb[~one] + 1.0])
    cov = np.concatenate([np.where(one, b - a, pa + 1.0 - a), np.ones(int((~one).sum())), (b - pb)[~one]])
    pick = np.concatenate([np.arange(len(a)), np.nonzero(~one)[0], np.nonzero(~one)[0]])
    f.rects(layer, (px0 - off) / k, y0[pick], (px1 - off) / k, y1[pick], i[pick] * cov)


def barcode_keys(t, span, n):
    """Time slots of a barcode that shows the last `span` seconds in n columns: (keys, frac, dt).
    keys = n + 1 slot numbers (slot k covers k * dt .. (k + 1) * dt; the newest one is being written at the
    right edge), frac = how far the pattern has slid towards the next slot (0..1), dt = seconds per slot.
    Give `frac` to barcode_lanes / barcode_cols: the bars then slide instead of jumping a column at a time."""
    dt = span / n
    u = (t - span) / dt
    kf = math.floor(u)
    return kf + np.arange(n + 1), u - kf, dt


def barcode_cols(x0, x1, n, frac):
    """Left and right x of the n + 1 columns of a sliding barcode (see barcode_keys), kept inside x0 .. x1:
    the oldest column leaves under the left edge while the newest one is written at the right edge."""
    cw = (x1 - x0) / n
    xa = x0 + (np.arange(n + 1) - frac) * cw
    return np.maximum(xa, x0), np.minimum(xa + cw, x1)


def barcode_lanes(f, x0, x1, y0, y1, density, keys, lanes=3, seed=0, layer="w", inten=0.95, frac=None):
    """Scrolling 'test pattern' barcode: one column per key, lit with the given density (0..1).
    frac (with the keys of barcode_keys: one more key than columns) = how far the pattern has slid since the
    last whole column: the bars glide to the left instead of jumping one column at a time."""
    n = len(keys)
    lane_h = (y1 - y0) / lanes
    if frac is None:
        cw = (x1 - x0) / n
        xs = x0 + np.arange(n) * cw
        for ln in range(lanes):
            on = hash01(keys, ln + 3 + seed) < np.asarray(density) * (1.0 - 0.18 * ln)
            f.rects(layer, xs[on], y0 + ln * lane_h, xs[on] + cw, y0 + (ln + 1) * lane_h - 3, inten)
        return
    xl, xr = barcode_cols(x0, x1, n - 1, frac)
    for ln in range(lanes):
        on = (hash01(keys, ln + 3 + seed) < np.asarray(density) * (1.0 - 0.18 * ln)) & (xr > xl)
        bars(f, layer, xl[on], y0 + ln * lane_h, xr[on], y0 + (ln + 1) * lane_h - 3, inten)


def barcode_burst(f, rect, t, u, lanes=8, seed=0):
    """A beat of pure test pattern (u = 0..1 progress): lanes of bars sliding against each other."""
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


def erode(s, amount, key=0, frame=0):
    """Text falling apart: a stable, growing subset of characters vanishes (amount 0..1); the ones about to
    go flicker through random glyphs first. Keeps the length, so anchored text does not move."""
    if amount <= 0.0 or not s:
        return s
    if amount >= 1.0:
        return ""
    h = hash01(np.arange(len(s)), key)
    band = 0.12 * min(1.0, amount / 0.1)          # the flickering fringe grows from nothing
    g = B.rnd(len(s), key, frame)                 # (hash01 would give the same glyph at every frame)
    out = []
    for k, (c, hv) in enumerate(zip(s, h)):
        if c == " " or hv >= amount + band:
            out.append(c)
        elif hv < amount:
            out.append(" ")
        else:
            out.append(_GLYPHS[int(g[k] * len(_GLYPHS))])
    return "".join(out)


def typed(s, age, cps=90.0, delay=0.0):
    """Type-on helper: the part of `s` written `age` seconds after its cue."""
    return s[: int(max(0.0, age - delay) * cps)]


def cross_grid(f, rect, step=96.0, inten=0.34, half=7.0, origin=None, dots=True, hole=None):
    """The lattice of small crosses behind the TouchDesigner scenes (plus a finer dot lattice)."""
    x0, y0, x1, y1 = rect
    ox, oy = origin or L.CENTER
    kx = np.arange(math.ceil((x0 + 10 - ox) / step), math.floor((x1 - 10 - ox) / step) + 1)
    ky = np.arange(math.ceil((y0 + 10 - oy) / step), math.floor((y1 - 10 - oy) / step) + 1)
    KX, KY = np.meshgrid(kx, ky)
    X, Y = (ox + KX * step).ravel(), (oy + KY * step).ravel()
    if hole is not None:
        m = np.hypot(X - hole[0], Y - hole[1]) > hole[2]
        X, Y = X[m], Y[m]
    f.crosses("w", X, Y, half, inten)
    if dots:
        s2 = step / 4
        kx = np.arange(math.ceil((x0 + 10 - ox) / s2), math.floor((x1 - 10 - ox) / s2) + 1)
        ky = np.arange(math.ceil((y0 + 10 - oy) / s2), math.floor((y1 - 10 - oy) / s2) + 1)
        KX, KY = np.meshgrid(kx, ky)
        m = ((KX % 4 != 0) | (KY % 4 != 0)).ravel()
        if E.WALL:                              # single pixels do not land: only the points half-way between the
            m = m & ((KX % 2 == 0) & (KY % 2 == 0)).ravel()      # crosses are kept, and drawn as dots
        X, Y = (ox + KX * s2).ravel()[m], (oy + KY * s2).ravel()[m]
        if hole is not None:
            m = np.hypot(X - hole[0], Y - hole[1]) > hole[2]
            X, Y = X[m], Y[m]
        if E.WALL:
            f.dots("w", X, Y, 1.7, E.wl(inten * 0.75))
            return
        f.pixels("w", X, Y, inten * 0.75)


def starfield(f, rect, n=260, seed=1, t=0.0, drift=(0.0, 0.0), inten=0.6):
    """Sparse dim points (the 'space' of the TouchDesigner scenes), optionally drifting."""
    x0, y0, x1, y1 = rect
    k = np.arange(n)
    w, h = x1 - x0, y1 - y0
    x = x0 + (hash01(k, seed) * w + drift[0] * t * (0.3 + hash01(k, seed + 2))) % w
    y = y0 + (hash01(k, seed + 1) * h + drift[1] * t * (0.3 + hash01(k, seed + 2))) % h
    b = hash01(k, seed + 3)
    if E.WALL:                                  # one star in two, at full level: its rank is its size
        m = k % 2 == 0
        f.dots("w", x[m], y[m], 1.6 + 1.4 * b[m] ** 2, E.wl(inten))
        return
    f.dots("w", x, y, 0.9 + 1.3 * b ** 4, inten * (0.25 + 0.75 * b ** 2))


def callout(f, x, y, dx, dy, title, lines=(), red=False, alpha=1.0, age=9.0, size=L.T_TAG, lsize=L.T_SMALL,
            side=None, build=None):
    """Leader from a point to an inverted tag + typed data lines. (dx, dy) = offset of the elbow.
    build = seconds since the callout appeared (see build.io for one that also leaves): it is then
    constructed outwards from its point - leader drawn by a pen, tag pushed out, lines decoded - instead of
    typed; negative = not there. Prefer it to an alpha: a callout never fades in."""
    if alpha <= 0.01:
        return
    side = side or (1 if dx >= 0 else -1)
    ex, ey = x + dx, y + dy
    hx = ex + side * 34
    if build is not None:
        w = max([text_w(title, size) + 12] + [text_w(ln, lsize) for ln in lines])
        xt = hx + side * (10 + w)
        rect = (min(x, xt) - 6, min(y, ey - size) - 6, max(x, xt) + 6, max(y, ey + size * 0.36 + len(lines) * lsize * 1.45 + 8) + 6)
        with f.build(build, rect, flow="out", origin=(x, y), wave=0.22, line=0.16, marks=False, key=int(x) + 3 * int(y)):
            callout(f, x, y, dx, dy, title, lines, red, alpha, 9.0, size, lsize, side)
        return
    f.segments("w", [x, ex], [y, ey], [ex, hx], [ey, ey], 0.75 * alpha, width=L.LW)
    f.dots("w", [x], [y], 2.6, 1.3 * alpha)
    anchor = "ls" if side > 0 else "rs"
    tx = hx + side * 10
    f.tag("r" if red else "w", tx, ey + size * 0.36, title, size=size, alpha=alpha, pad=5, anchor=anchor)
    for k, ln in enumerate(lines):
        f.text("w", tx, ey + size * 0.36 + (k + 1) * lsize * 1.45 + 8, typed(ln, age, delay=0.1 + 0.08 * k),
               size=lsize, alpha=0.9 * alpha, anchor=anchor)


CELL_BUILD = 1.2                      # seconds a counter cell takes to build


def counter_cell(f, label, value, alpha=1.0, red=False, rect=None, sub=None, age=None, from_right=False):
    """The framed cell at the bottom left (the 'Year' box of the TouchDesigner intro).
    age = seconds since its build started (None = built): a line is drawn across its middle (from the frame
    edge it sits on), the cell opens from that line like a plate, the label is decoded, the figure spins and
    locks, and the cell flashes once (inverted, two frames)."""
    x0, y0, x1, y1 = rect or L.CELL
    if age is not None and age < CELL_BUILD:
        if age < 0.0 or not B.open_box(f, (x0, y0, x1, y1), age, from_right=from_right, inten=0.95 * alpha):
            return
        f.text("w", x0 + 26, y0 + 34, B.decode(label, age, 80.0, 0.42, key=3), size=L.T_SMALL, alpha=0.75 * alpha)
        f.text("r" if red else "w", x0 + 26, y1 - 24, B.roll(value, age, 0.4, 0.55, key=5), size=38, alpha=alpha)
        if sub:
            f.text("w", x1 - 20, y0 + 34, B.decode(sub, age, 80.0, 0.8, key=7, pad=True), size=L.T_SMALL,
                   alpha=0.6 * alpha, anchor="rs")
        B.flash(f, (x0, y0, x1, y1), age, 0.98)
        return
    f.occlude(x0, y0, x1, y1)
    f.rect("w", x0, y0, x1, y1, 0.95 * alpha, width=L.LW_FRAME)
    f.text("w", x0 + 26, y0 + 34, label, size=L.T_SMALL, alpha=0.75 * alpha)
    f.text("r" if red else "w", x0 + 26, y1 - 24, value, size=38, alpha=alpha)
    if sub:
        f.text("w", x1 - 20, y0 + 34, sub, size=L.T_SMALL, alpha=0.6 * alpha, anchor="rs")


def through_you_cell(f, ctx, t, alpha=1.0, age=None):
    """The one number that runs through the whole show: muons through one spectator since 00:00
    (the voice: 'more than fifty thousand of them will flood through your skin')."""
    n = ctx.through_you(t)
    rect = (ctx.slots if t >= sd.T_REVEAL else ctx.slots_pre)["cell"]      # same slots as the scene's bottom band
    x0, _, x1, _ = rect
    w = x1 - x0                                   # the cell shrinks when the first tower stands close to the edge
    if w < 200:
        return
    label = "MUONS THROUGH YOU // SINCE 00:00" if w >= 465 else "MUONS THROUGH YOU"
    counter_cell(f, label, f"{n:,}".replace(",", " "), alpha=alpha, rect=rect,
                 sub=sd.tc(t)[:8] if w >= 330 else None, age=age)


def scope(f, rect, values, label, readout, alpha=1.0, hot=0.0):
    """Small oscilloscope panel (the 'Voltage' boxes of the bloom scene): red graticule, white trace."""
    x0, y0, x1, y1 = rect
    f.rect("w", x0, y0, x1, y1, E.wl(0.55 * alpha), width=E.ww(1.0))
    gx = np.arange(x0 + 20, x1 - 1, 20.0)
    gy = np.arange(y0 + 20, y1 - 1, 20.0)
    f.segments("r", gx, np.full_like(gx, y0), gx, np.full_like(gx, y1), 0.3 * alpha)
    f.segments("r", np.full_like(gy, x0), gy, np.full_like(gy, x1), gy, 0.3 * alpha)
    v = np.asarray(values, np.float64)
    xs = np.linspace(x0 + 2, x1 - 2, len(v))
    ys = y1 - 6 - np.clip(v, 0, 1) * (y1 - y0 - 14)
    f.polyline("w", xs, ys, (0.9 + 0.6 * hot) * alpha, width=L.LW)
    f.text("r" if hot > 0.3 else "w", x1, y0 - 9, readout, size=L.T_MICRO, alpha=0.9 * alpha, anchor="rs")
    f.text_vertical("w", x0 - 22, y1, label, size=L.T_MICRO, alpha=0.8 * alpha)


def strip_base(f, rect=None, title=None, alpha=1.0, band=True, ticks=None, age=None):
    """Outline of a score strip (sits left of the subtitle box). Returns (x0, y0, x1, y1, y_band).
    `ticks` = (v0, v1, minor, major) draws ruler ticks along the top and bottom inner edges.
    age = seconds since the strip started to build (None = built): the red band shoots across behind a
    bright head, the two rules follow it, their ticks arrive in a wave, the title tag is made."""
    x0, y0, x1, y1 = rect or L.STRIP
    ix0, ix1 = x0 + 16, x1 - 16
    iy0, iy1 = y0 + 34, y1 - 12
    yb = (iy0 + iy1) / 2
    if age is None:
        if band:
            f.rects("r", ix0, yb - 3, ix1, yb + 3, alpha)
        f.segments("w", [ix0, ix0], [iy0, iy1], [ix1, ix1], [iy0, iy1], 0.9 * alpha, width=E.ww(1.0))
        if ticks:
            v0, v1, minor, major = ticks
            ruler(f, ix0, ix1, iy0, v0, v1, minor, major, down=True, inten=0.8 * alpha)
            ruler(f, ix0, ix1, iy1, v0, v1, minor, major, down=False, inten=0.8 * alpha)
        if title:
            f.tag("w", ix0, y0 + 25, title, size=L.T_MICRO, pad=3, alpha=alpha)
        return ix0, iy0, ix1, iy1, yb
    pb = float(B.ease(B.lin(age, 0.0, 0.36)))
    xh = ix0 + (ix1 - ix0) * pb
    if band and pb > 0.0:
        f.rects("r", ix0, yb - 3, xh, yb + 3, alpha)
        if pb < 1.0:
            f.dots("w", [xh], [yb], 4.6, 1.8)
    pr = float(B.ease(B.lin(age, 0.1, 0.5)))                  # the two rules, a little behind the band
    for yy in (iy0, iy1):
        B.pen(f, "w", ix0, yy, ix1, yy, pr, 0.9 * alpha, width=E.ww(1.0))
    if ticks:
        v0, v1, minor, major = ticks
        rv = B.lin(age, 0.14, 0.7)
        ruler(f, ix0, ix1, iy0, v0, v1, minor, major, down=True, inten=0.8 * alpha, reveal=rv)
        ruler(f, ix0, ix1, iy1, v0, v1, minor, major, down=False, inten=0.8 * alpha, reveal=rv)
    if title:
        B.tag(f, "w", ix0, y0 + 25, title, age, t0=0.2, size=L.T_MICRO, pad=3, alpha=alpha, cps=110.0)
    return ix0, iy0, ix1, iy1, yb


def strip_cursor(f, x, y0, y1, label=None, alpha=1.0):
    f.segments("r", [x], [y0 - 4], [x], [y1 + 4], 1.2 * alpha, width=L.LW)
    if label:
        f.tag("r", x + 6, y1 - 4, label, size=L.T_MICRO, pad=3, alpha=alpha)


def show_strip(f, t, ctx, title, t0, t1, marks=(), alpha=1.0, age=None):
    """Score strip of a linear scene: its span on one ruler, the loudness of the music as a comb on top,
    the detector hits as a comb at the bottom, the cues as tags on the red band, a time cursor.
    age = seconds since the strip appeared (None = built): the band and the rules are drawn (strip_base),
    then the combs, the cue tags and the cursor are constructed from left to right."""
    x0, y0, x1, y1, yb = strip_base(f, title=title, alpha=alpha, ticks=(t0, t1, 1.0, 10.0), age=age)
    with f.build(None if age is None else age - 0.3, L.STRIP, flow="lr", wave=0.5, marks=False, bars="centre", key=11):
        return _show_strip_body(f, t, ctx, t0, t1, marks, alpha, x0, y0, x1, y1, yb)


def _show_strip_body(f, t, ctx, t0, t1, marks, alpha, x0, y0, x1, y1, yb):
    X = lambda tt: x0 + (np.asarray(tt, np.float64) - t0) / (t1 - t0) * (x1 - x0)
    n = int((x1 - x0) / 5)
    lv = ctx.cues.loud_curve(t0, t1, n)
    tb = t0 + (np.arange(n) + 0.5) / n * (t1 - t0)
    past = tb <= t
    xb = X(tb)
    if E.WALL:              # what is to come is not grey: only the tip of its bars, the outline of the profile
        yt = y0 + 2 + 40 * lv ** 1.5
        f.rects("w", xb, np.where(past, y0 + 1, np.maximum(yt - 3, y0 + 1)), xb + 2, yt, 0.95 * alpha)
    else:
        f.rects("w", xb, y0 + 1, xb + 2, y0 + 2 + 40 * lv ** 1.5, np.where(past, 0.95, 0.3) * alpha)
    for key in sd.KEYS:
        ht, he, _ = ctx.det.hits(key, t0, t1, echoes=False)
        if len(ht):
            xh = X(ht)
            f.rects("r", xh, y1 - 3 - 34 * he, xh + 2, y1 - 1, np.where(ht <= t, 0.95, 0.35) * alpha)
    placed = []
    for tv, word in marks:
        if not (t0 <= tv <= t1):
            continue
        xv = float(X(tv))
        row = sum(1 for p in placed if abs(p - xv) < 14 * len(word) + 30) % 2
        placed.append(xv)
        f.segments("w", [xv], [yb - 12], [xv], [yb + 12], 0.8 * alpha, width=E.ww(1.0))
        f.tag("r" if tv <= t < tv + 2.5 else "w", xv + 4, yb + 24 + 22 * row, word, size=L.T_MICRO, pad=3,
              alpha=(1.0 if tv <= t else 0.4) * alpha)
    strip_cursor(f, float(X(t)), y0, y1, f"{sd.tc(t)}", alpha)
    return X, yb
