"""OVERLOAD - the groove breaks, the instrument answers with everything it knows.   Sheet scene 7, 06:55 - 07:34.

  7.0  06:55 - 07:02.7  BREAK BEFORE DRUMS   "almost nothing." (07:00): the elevation of the DANCE scene,
                                              emptied. One muon, alone, falls for the whole break; it reaches
                                              the ground when the drums come back.
  7.1  07:02.7 - 07:27  DRUMS IN             on the kick the wall becomes a battery of HUD plates: diagrams,
                                              plots, equations and tables about the muon, quantum physics and
                                              relativity - the figures of a physics textbook redrawn as an
                                              instrument (white line work, red for the muon). Every bay between
                                              the towers is split into cells; each cell shows one plate. A plate
                                              appears in a cut, whole, on a hit of the music; a later hit takes
                                              it away in a flash of static (for three frames its cell is filled
                                              with the test-pattern bars of the MUON scene), the cell stays
                                              dark and the next hit cuts another plate into it. The small cells
                                              answer every hit, the middle ones the accents, the large ones the
                                              kicks. Every two bars the whole wall goes in one flash on the
                                              downbeat and is cut into new cells, which come in over the three
                                              hits that follow; on every bar a line reads the wall from top to
                                              bottom.
  7.3  07:27 - 07:34    TRANSITION           the cuts stop; on each of the last, weak kicks one plate is taken
                                              apart, the large ones last. What is left is single muons falling
                                              straight down. Almost black: OUTLAST next (muons falling 15 km).

The audio drives it: every cut and every flash IS a hit of the music (`music_hits`: the onsets and the kicks
of ctx.cues, moved onto the frame where they sound - the analysis finds a hit LAG = 30 ms before it is heard).
How much of the wall changes on a hit is how strong the hit is: one small cell on a weak one, a small, a
middle and a large one on an accented kick, more of them in the third phrase. Inside the plates the cursors,
the wave amplitudes, the tilt of the axes follow the bands and the kick envelope; the header of every plate
carries a six-band meter of the spectrum and its rule swells on the kicks. The schedule (which plate, in
which cell, from which hit to which hit) is built once from the cues and the tower placement with a seeded
generator: a frame is a pure function of t.

Everything written is real: constants with their measured digits (PDG / CODATA), formulas as they stand
in the books, curves computed from them (Bethe, Michel, Breit-Rabi, Schwarzschild null geodesics, the
GW150914 chirp ...). Space Mono has no Greek alphabet (only the micro sign and pi): the letters it lacks are
drawn as line work in the cell of a character (`fml`).

Nothing is built up while the drums play (review of 2026-10-03: "don't build up during the glitchy sequence
they can appear in a cut then desapear with a white flash"): the plates, the strip and the bottom panels are
there, whole, on the frame of their hit. Nothing fades either. Only in the transition are the plates, the
strip and the bottom panels taken apart ("at the end it fine to deconstruct them"): `Quick`, the build of
the show run backwards with a short wave - rules drawn back by a pen, curves untraced, text scrambled.
The flash is not a white field ("don't use white for the flash use the same effect that the background at
01:43:10"): it is the static behind the MUON card as it stands at that time (`muon.static`: rows of bars of
random width, a few of them red, a new pattern on every frame), in the cell of the plate that leaves. A fast
one is small (a hit takes at most a quarter of the wall, 7 % on average), a large one is rare (the whole wall
once every two bars, 3.84 s).

Layout: the cells are cut from ctx.cols (the bays between the towers), the bottom blocks flow into
ctx.slots, the lone muon of the break falls on ctx.focus. Nothing has a fixed x.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import engine as E
from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import OrthoCamera, font, hash01, smoothstep
from ..show import Scene
from .dance import BEAT, N_PHRASES, PANEL_BLOCKS, PHRASE, T0, Y_GROUND, Dance
from .muon import ROW_H, static
from .shower import (VIEW_Y0, VIEW_Y1, altitude_rules, auto_callout, bottom_panels, col_type, draw_info, flow,
                     info_layout, stage_for)

T_IN = 415.0
T_DRUMS = T0 + 13 * PHRASE          # 422.666: the kick that brings the drums back
T_TRANS = 447.0
T_OUT = 454.0
BAR = 4 * BEAT
T_LANES = T_TRANS + 3.5             # the last read-out is made
V_SHOCK = 2200.0                    # px / s: the ring that leaves the point where the muon of the break lands
LAG = 0.03                          # s: a hit sounds this long after the time the cues give it (measured on the mix)
FRAME = 1.0 / 60.0                  # the show plays at 60 frames a second: every cut is put on one of them
FLASH = 3 * FRAME                   # a plate that leaves leaves its cell full of static for three frames
MIN_LIFE = {"S": 0.2, "M": 0.3, "L": 0.55, "X": 1.0}       # s a plate is left on the wall at least, by cell size


def on_frame(t):
    """The frame nearest to t, as a time just before it: that frame is the first one of what starts there."""
    return round(t / FRAME) * FRAME - 1e-4


T_ON = on_frame(T_DRUMS + LAG)      # 422.700: the frame on which that kick is heard

# ----------------------------------------------------------------------------
# what the plates say: measured values (PDG 2024, CODATA 2022) and what follows from them
# ----------------------------------------------------------------------------
C = 299792458.0                     # m / s
M_MU, TAU_MU = 105.6583755, 2.1969811e-6            # MeV, s
M_E, M_PI = 0.51099895, 139.57039
E_SEA = 4000.0                      # MeV: the mean energy of a muon at sea level
GAM = E_SEA / M_MU                  # 37.86
BGAM = math.sqrt(GAM * GAM - 1.0)
BETA = BGAM / GAM                   # 0.99965
CTAU = C * TAU_MU                   # 658.64 m
RANGE = BGAM * CTAU                 # 24.9 km: how far it gets in a dilated mean life
P_EINSTEIN = math.exp(-15000.0 / RANGE)             # 0.548 of them survive 15 km
P_NEWTON = math.exp(-15000.0 / CTAU)                # 1.3e-10 without time dilation
A_MU = 0.001165920715               # (g - 2) / 2, world average 2025 (124 ppb)
G_MAGIC = math.sqrt(1.0 + 1.0 / A_MU)               # 29.3: the 'magic' gamma of the storage ring
E_MICHEL = (M_MU ** 2 + M_E ** 2) / (2 * M_MU)      # 52.83 MeV: the end of the decay electron spectrum
P_PI = (M_PI ** 2 - M_MU ** 2) / (2 * M_PI)         # 29.79 MeV/c: pion decay at rest
DM2 = 2.5e-3                        # eV2: the atmospheric neutrino mass splitting


def _bethe(bg):
    """Mean energy loss of a muon in air, MeV cm2 / g (Bethe, no density correction): 1.815 at its minimum."""
    bg = np.asarray(bg, np.float64)
    g = np.sqrt(1.0 + bg * bg)
    b2 = bg * bg / (g * g)
    tmax = 2 * M_E * bg * bg / (1 + 2 * g * M_E / M_MU + (M_E / M_MU) ** 2)
    return 0.307075 * 0.49919 / b2 * (0.5 * np.log(2 * M_E * bg * bg * tmax / 85.7e-6 ** 2) - b2)


def _highland(x, x0, p):
    """Width of the multiple-scattering angle (rad) of a muon of momentum p (MeV/c) behind x cm of a material
    of radiation length x0 cm."""
    b = p / math.hypot(p, M_MU)
    return 13.6 / (b * p) * math.sqrt(x / x0) * (1 + 0.038 * math.log(x / x0))


def _null_geodesic(b, n=420, r_in=40.0):
    """Path of light in the Schwarzschild field (units of r_s) for an impact parameter b: integrates
    u'' + u = 1.5 u^2 (u = 1 / r) from far away; stops at the horizon or when the ray has left again."""
    u, du, dphi = 1.0 / r_in, math.sqrt(max(1.0 / b ** 2 - (1.0 / r_in) ** 2 * (1 - 1.0 / r_in), 0.0)), 0.012
    phi = math.asin(min(1.0, b / r_in))
    pts = []
    for _ in range(n * 6):
        pts.append((phi, u))
        # Runge-Kutta 4 on (u, du)
        def acc(v):
            return 1.5 * v * v - v
        k1u, k1v = du, acc(u)
        k2u, k2v = du + 0.5 * dphi * k1v, acc(u + 0.5 * dphi * k1u)
        k3u, k3v = du + 0.5 * dphi * k2v, acc(u + 0.5 * dphi * k2u)
        k4u, k4v = du + dphi * k3v, acc(u + dphi * k3u)
        u += dphi / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        du += dphi / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        phi += dphi
        if u >= 1.0 or u <= 1.0 / (r_in * 1.02):
            break
    a = np.array(pts)
    r = 1.0 / a[:, 1]
    return np.stack([-r * np.cos(a[:, 0]), r * np.sin(a[:, 0])], 1)        # comes in from the left, above the axis


# ----------------------------------------------------------------------------
# line work
# ----------------------------------------------------------------------------

class _Wall:
    """The frame as a plate draws on it under the wall rule (engine.WALL): the line work of a plate was drawn in
    greys and hairlines, as on paper, and stray light on the brick eats both. Every white line, curve, ring and
    dot of a plate goes through here: level engine.wl (full from 0.6, never under 0.8), weight at least
    engine.WALL_LINE, a dot at least 2 px of radius. Red keeps its level; its hairlines get that weight too.
    What gets too heavy at full level is made of fewer lines in its plate (thin), not dimmer."""
    __slots__ = ("_f",)

    def __init__(self, f):
        self._f = f

    def __getattr__(self, k):
        return getattr(self._f, k)

    def segments(self, layer, x0, y0, x1, y1, i0, i1=None, width=1.0, spacing=0.5):
        if layer == "w":
            i0, i1 = E.wl(i0), (None if i1 is None else E.wl(i1))
        return self._f.segments(layer, x0, y0, x1, y1, i0, i1, max(width, E.WALL_LINE), spacing)

    def polyline(self, layer, xs, ys, i, width=1.0, closed=False, i_end=None):
        if layer == "w":
            i, i_end = E.wl(i), (None if i_end is None else E.wl(i_end))
        return self._f.polyline(layer, xs, ys, i, width=max(width, E.WALL_LINE), closed=closed, i_end=i_end)

    def rect(self, layer, x0, y0, x1, y1, i, width=1.0):
        if layer == "w":
            i, width = E.wl(i), max(width, E.WALL_LINE)
        return self._f.rect(layer, x0, y0, x1, y1, i, width=width)

    def rings(self, layer, cx, cy, radius, i, spacing=0.5, width=1.0):
        if layer == "w":
            i, width = E.wl(i), max(width, E.WALL_LINE)
        return self._f.rings(layer, cx, cy, radius, i, spacing=spacing, width=width)

    def dots(self, layer, x, y, r, i):
        if layer == "w":
            r, i = np.maximum(r, 2.0), E.wl(i)
        return self._f.dots(layer, x, y, r, i)


def thin(n):
    """One in n of a family of lines that is a tone on paper (a hatch, a grid, a fan): with the wall rule each
    line is full white, so the tone is made of fewer of them. 1 without the rule."""
    return n if E.WALL else 1


def fsz(n, width, hi):
    """Type size at which n characters fit in `width` (at most `hi`)."""
    return float(min(hi, width / (max(n, 1) * 0.615)))


def seg(f, layer, x0, y0, x1, y1, i=0.8, w=L.LW):
    f.segments(layer, [x0], [y0], [x1], [y1], i, width=w)


def arrow(f, layer, x0, y0, x1, y1, i=0.85, w=L.LW, head=11.0):
    d = math.hypot(x1 - x0, y1 - y0)
    if d < 1e-3:
        return
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    h = min(head, 0.6 * d)
    f.segments(layer, [x0, x1, x1], [y0, y1, y1],
               [x1, x1 - h * (0.9 * ux - 0.42 * uy), x1 - h * (0.9 * ux + 0.42 * uy)],
               [y1, y1 - h * (0.9 * uy + 0.42 * ux), y1 - h * (0.9 * uy - 0.42 * ux)], i, width=w)


def dashes(f, layer, x0, y0, x1, y1, i=0.55, dash=9.0, gap=7.0, w=1.0):
    d = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(round(d / (dash + gap))))
    k = np.arange(n) / n
    fr = dash / (dash + gap) / n
    f.segments(layer, x0 + (x1 - x0) * k, y0 + (y1 - y0) * k, x0 + (x1 - x0) * (k + fr), y0 + (y1 - y0) * (k + fr), i,
               width=w)


def wavy(f, layer, x0, y0, x1, y1, i=0.9, amp=6.0, wl=20.0, w=L.LW, ph=0.0):
    """A boson line: whole waves between two points."""
    d = math.hypot(x1 - x0, y1 - y0)
    if d < 1e-3:
        return
    n = max(2, int(round(d / wl)))
    s = np.linspace(0.0, 1.0, 10 * n + 1)
    o = amp * np.sin(2 * np.pi * n * s + ph) * np.minimum(1.0, 5.0 * n * np.minimum(s, 1 - s))
    f.polyline(layer, x0 + (x1 - x0) * s - (y1 - y0) / d * o, y0 + (y1 - y0) * s + (x1 - x0) / d * o, i, width=w)


def ring(f, layer, cx, cy, r, i=0.8, w=L.LW, n=72, a0=0.0, a1=2 * math.pi, ry=None):
    a = np.linspace(a0, a1, n + 1)
    f.polyline(layer, cx + r * np.cos(a), cy + (r if ry is None else ry) * np.sin(a), i, width=w)


def dring(f, layer, cx, cy, r, i=0.6, n=40, w=1.0, ph=0.0, ry=None):
    """A dashed circle."""
    a = ph + np.arange(n) / n * 2 * np.pi
    da = np.pi / n
    ry = r if ry is None else ry
    f.segments(layer, cx + r * np.cos(a), cy + ry * np.sin(a), cx + r * np.cos(a + da), cy + ry * np.sin(a + da), i, width=w)


def hatch(f, layer, cx, cy, r, i=0.45, step=8.0, w=1.0):
    """A disc filled with 45 degree hatching (the particles of the textbook figures)."""
    step = min(step * thin(2), max(step, 0.6 * r))            # (a small disc keeps three lines)
    c = np.arange(-int(r / step), int(r / step) + 1) * step * 1.41421
    hl = np.sqrt(np.maximum(r * r - c * c / 2.0, 0.0)) * 0.70711
    mx, my = cx + c / 2.0, cy - c / 2.0
    f.segments(layer, mx - hl, my - hl, mx + hl, my + hl, i, width=w)


def _arc(cx, cy, rx, ry, a0, a1, n):
    a = np.linspace(math.radians(a0), math.radians(a1), n)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def _pts(*p):
    return np.array(p, np.float64)


# The Greek letters Space Mono does not have, as strokes in the cell of a character (x 0 .. 0.61 em, y up from
# the baseline; x-height 0.495, capitals 0.70, descenders -0.20: the metrics of the font).
_GREEK = {
    "ν": [_pts((0.10, 0.495), (0.18, 0.48), (0.305, 0.0), (0.43, 0.22), (0.495, 0.40), (0.50, 0.495))],
    "γ": [_pts((0.07, 0.40), (0.12, 0.48), (0.20, 0.495), (0.26, 0.42), (0.33, 0.02)),
          _pts((0.53, 0.495), (0.33, 0.02), (0.28, -0.12), (0.27, -0.20), (0.32, -0.22), (0.36, -0.14), (0.33, 0.02))],
    "τ": [_pts((0.07, 0.41), (0.13, 0.48), (0.22, 0.495), (0.54, 0.495)),
          _pts((0.31, 0.495), (0.30, 0.12), (0.34, 0.02), (0.42, 0.0), (0.49, 0.04))],
    "α": [_pts((0.53, 0.495), (0.47, 0.25), (0.38, 0.07), (0.27, 0.0), (0.16, 0.03), (0.09, 0.14), (0.08, 0.28),
               (0.13, 0.42), (0.24, 0.495), (0.35, 0.46), (0.43, 0.33), (0.47, 0.14), (0.51, 0.03), (0.56, 0.0))],
    "β": [_pts((0.12, -0.20), (0.12, 0.56), (0.16, 0.66), (0.26, 0.70), (0.38, 0.67), (0.44, 0.58), (0.42, 0.46),
               (0.32, 0.38), (0.20, 0.37)),
          _pts((0.32, 0.38), (0.44, 0.33), (0.50, 0.22), (0.47, 0.09), (0.37, 0.01), (0.25, 0.0), (0.16, 0.05),
               (0.12, 0.14))],
    "θ": [_arc(0.305, 0.35, 0.19, 0.355, 0, 360, 19), _pts((0.115, 0.35), (0.495, 0.35))],
    "λ": [_pts((0.08, 0.70), (0.16, 0.70), (0.22, 0.62), (0.50, 0.0)), _pts((0.30, 0.44), (0.10, 0.0))],
    "ψ": [_pts((0.305, -0.20), (0.305, 0.70)),
          _pts((0.08, 0.495), (0.09, 0.26), (0.15, 0.10), (0.25, 0.03), (0.36, 0.03), (0.46, 0.10), (0.52, 0.26),
               (0.53, 0.495))],
    "φ": [_arc(0.305, 0.25, 0.21, 0.25, 0, 360, 19), _pts((0.305, -0.20), (0.305, 0.70))],
    "ρ": [_pts((0.13, -0.20), (0.13, 0.28), (0.17, 0.42), (0.27, 0.495), (0.38, 0.47), (0.47, 0.37), (0.49, 0.24),
               (0.44, 0.09), (0.33, 0.01), (0.22, 0.02), (0.13, 0.12))],
    "σ": [_pts((0.55, 0.495), (0.29, 0.495), (0.17, 0.44), (0.09, 0.31), (0.09, 0.17), (0.17, 0.05), (0.29, 0.0),
               (0.40, 0.05), (0.47, 0.17), (0.47, 0.31), (0.40, 0.44), (0.29, 0.495))],
    "ω": [_pts((0.15, 0.495), (0.08, 0.36), (0.07, 0.18), (0.12, 0.05), (0.19, 0.0), (0.26, 0.05), (0.30, 0.20),
               (0.305, 0.32)),
          _pts((0.305, 0.32), (0.31, 0.20), (0.35, 0.05), (0.42, 0.0), (0.49, 0.05), (0.54, 0.18), (0.53, 0.36),
               (0.46, 0.495))],
    "κ": [_pts((0.13, 0.0), (0.13, 0.495)), _pts((0.50, 0.495), (0.13, 0.20)), _pts((0.26, 0.30), (0.52, 0.0))],
    "Λ": [_pts((0.06, 0.0), (0.305, 0.70), (0.55, 0.0))],
    "Γ": [_pts((0.12, 0.0), (0.12, 0.70), (0.52, 0.70), (0.52, 0.58))],
    "Ω": [_pts((0.08, 0.0), (0.22, 0.0), (0.22, 0.06), (0.12, 0.20), (0.09, 0.38), (0.14, 0.56), (0.24, 0.67),
               (0.305, 0.70), (0.37, 0.67), (0.47, 0.56), (0.52, 0.38), (0.49, 0.20), (0.39, 0.06), (0.39, 0.0),
               (0.53, 0.0))],
    "Ψ": [_pts((0.305, 0.0), (0.305, 0.70)),
          _pts((0.07, 0.70), (0.08, 0.44), (0.15, 0.30), (0.25, 0.24), (0.36, 0.24), (0.46, 0.30), (0.53, 0.44),
               (0.54, 0.70))],
    "☉": [_arc(0.305, 0.30, 0.24, 0.24, 0, 360, 21), _arc(0.305, 0.30, 0.035, 0.035, 0, 360, 7)],
}

_ADV = {}


def adv(f, size):
    """Advance of one Space Mono cell at this size on this frame, in design px (the font is set on whole
    pixels: the line work that stands in for a letter has to sit in the same cells)."""
    px = max(6, int(round(size * f.s)))
    a = _ADV.get(px)
    if a is None:
        a = _ADV[px] = float(font(px).getlength("0"))
    return a / f.s


def greek(f, layer, x, y, ch, size, inten=0.95):
    """One letter as line work (short pieces: while a plate is made they are thrown out like the marks of a
    ruler, no pen runs along them)."""
    w = max(1.2, 0.085 * size)
    for st in _GREEK[ch]:
        xs, ys = x + st[:, 0] * size, y - st[:, 1] * size
        f.segments(layer, xs[:-1], ys[:-1], xs[1:], ys[1:], inten, width=w)


def _tokens(s):
    """Runs of a formula: (text, level), level 0 on the line, +1 raised (after ^), -1 lowered (after _); a
    group in braces, or one character, follows the sign."""
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in "^_" and i + 1 < n:
            lv = 1 if c == "^" else -1
            if s[i + 1] == "{":
                j = s.index("}", i + 2)
                out.append((s[i + 2: j], lv))
                i = j + 1
            else:
                out.append((s[i + 1], lv))
                i += 2
        else:
            j = i
            while j < n and s[j] not in "^_":
                j += 1
            out.append((s[i:j], 0))
            i = j
    return out


def fml_w(f, s, size):
    return sum(len(t.replace("̄", "")) * adv(f, size if lv == 0 else 0.62 * size) for t, lv in _tokens(s))


def fml(f, layer, x, y, s, size=17.0, alpha=0.95, anchor="l"):
    """A formula: Space Mono where it has the sign, line work for the Greek letters it lacks, raised and
    lowered indices (^ _), a bar over an antiparticle (a combining macron after the letter). Returns the x
    where it ends."""
    if anchor != "l":
        w = fml_w(f, s, size)
        x -= w if anchor == "r" else 0.5 * w
    for txt, lv in _tokens(s):
        sz = size if lv == 0 else 0.62 * size
        yy = y - 0.40 * size if lv > 0 else y + 0.16 * size if lv < 0 else y
        a = adv(f, sz)
        run, rx, k = [], x, 0
        for ch in txt:
            if ch == "̄":
                f.segments(layer, [x + (k - 1) * a + 0.06 * sz], [yy - 0.66 * sz], [x + k * a - 0.06 * sz], [yy - 0.66 * sz],
                           alpha, width=max(1.2, 0.085 * sz))
            elif ch in _GREEK:
                if run:
                    f.text(layer, rx, yy, "".join(run), size=sz, alpha=alpha)
                    run = []
                greek(f, layer, x + k * a, yy, ch, sz, alpha)
                k += 1
            else:
                if not run:
                    rx = x + k * a
                run.append(ch)
                k += 1
        if run:
            f.text(layer, rx, yy, "".join(run), size=sz, alpha=alpha)
        x += k * a
    return x


class Pl:
    """What a plate draws in: its rect on the wall (under its title), its age, the show time, the audio."""
    __slots__ = ("f", "x0", "y0", "x1", "y1", "w", "h", "cx", "cy", "u", "t", "a", "ts", "tl", "key")

    def __init__(self, f, rect, u, t, a, key=0):
        self.f = f
        self.x0, self.y0, self.x1, self.y1 = rect
        self.w, self.h = self.x1 - self.x0, self.y1 - self.y0
        self.cx, self.cy = 0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1)
        self.u, self.t, self.a, self.key = u, t, a, key
        area = self.w * self.h
        self.ts = 14.0 if area < 1.9e5 else 17.0 if area < 4.4e5 else 20.0
        self.tl = self.ts + 3.0

    def box(self, aspect, sx=1.0, sy=1.0, ax=0.5, ay=0.5):
        """The largest rect of an aspect (w / h) in the plate scaled by (sx, sy), placed at (ax, ay) of the room."""
        w = min(self.w * sx, self.h * sy * aspect)
        h = w / aspect
        x0, y0 = self.x0 + (self.w - w) * ax, self.y0 + (self.h - h) * ay
        return x0, y0, x0 + w, y0 + h

    def text(self, x, y, s, size=None, layer="w", alpha=0.8, anchor="ls"):
        self.f.text(layer, x, y, s, size=size or self.ts, alpha=alpha, anchor=anchor)

    def fml(self, x, y, s, size=None, layer="w", alpha=0.95, anchor="l"):
        return fml(self.f, layer, x, y, s, size=size or self.tl, alpha=alpha, anchor=anchor)


class Ax:
    """A plot: textbook axes (two lines with arrow heads, ticks, small figures) and curves in data units."""

    def __init__(self, P, xr, yr, logx=False, logy=False, ml=46.0, mr=18.0, mt=14.0, mb=30.0, rect=None):
        self.P, self.f, self.logx, self.logy = P, P.f, logx, logy
        x0, y0, x1, y1 = rect or (P.x0, P.y0, P.x1, P.y1)
        self.x0, self.y0, self.x1, self.y1 = x0 + ml, y0 + mt, x1 - mr, y1 - mb
        self.xr = (math.log10(xr[0]), math.log10(xr[1])) if logx else xr
        self.yr = (math.log10(yr[0]), math.log10(yr[1])) if logy else yr

    def X(self, v):
        v = np.log10(v) if self.logx else np.asarray(v, np.float64)
        return self.x0 + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * (self.x1 - self.x0)

    def Y(self, v):
        v = np.log10(v) if self.logy else np.asarray(v, np.float64)
        return self.y1 - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * (self.y1 - self.y0)

    def axes(self, xt=(), yt=(), xf=None, yf=None, xl=None, yl=None, i=0.8):
        f, ts = self.f, self.P.ts
        arrow(f, "w", self.x0, self.y1, self.x1 + 12, self.y1, i, head=9.0)
        arrow(f, "w", self.x0, self.y1, self.x0, self.y0 - 10, i, head=9.0)
        if len(xt):
            xs = self.X(np.asarray(xt, np.float64))
            f.segments("w", xs, np.full(len(xs), self.y1), xs, np.full(len(xs), self.y1 + 6.0), i)
            if xf:
                lim = self.x1 + 14 - fml_w(f, xl, ts * 0.86) - 3.0 * ts if xl else 1e9
                for v, x in zip(xt, xs):
                    if x <= lim:
                        f.text("w", float(x), self.y1 + 8 + 0.8 * ts, xf(v), size=ts * 0.86, alpha=0.6, anchor="ms")
        if len(yt):
            ys = self.Y(np.asarray(yt, np.float64))
            f.segments("w", np.full(len(ys), self.x0 - 6.0), ys, np.full(len(ys), self.x0), ys, i)
            if yf:
                for v, y in zip(yt, ys):
                    f.text("w", self.x0 - 9, float(y) + 0.3 * ts, yf(v), size=ts * 0.86, alpha=0.6, anchor="rs")
        if xl:                                  # the name of the axis takes the place of its last figures
            fml(f, "w", self.x1 + 14, self.y1 + 8 + 0.8 * ts, xl, size=ts * 0.86, alpha=0.85, anchor="r")
        if yl:
            fml(f, "w", self.x0 + 8, self.y0 - 2, yl, size=ts, alpha=0.8)

    def curve(self, layer, xs, ys, i=0.95, w=L.LW_BOLD):
        self.f.polyline(layer, self.X(xs), self.Y(ys), i, width=w)

    def mark(self, layer, x, y, r=4.0, i=1.5):
        self.f.dots(layer, [float(self.X(x))], [float(self.Y(y))], r, i)


# ----------------------------------------------------------------------------
# the plates: the muon
# ----------------------------------------------------------------------------

def _along(a, b, q):
    return a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q


def _fermion(f, layer, a, b, i=0.9, w=L.LW_BOLD, back=False):
    """A fermion line of a Feynman diagram, its arrow at mid-length (turned round for an antiparticle)."""
    seg(f, layer, a[0], a[1], b[0], b[1], i, w)
    d = max(math.hypot(b[0] - a[0], b[1] - a[1]), 1e-6)
    ux, uy = (b[0] - a[0]) / d, (b[1] - a[1]) / d
    m = _along(a, b, 0.44 if back else 0.56)
    if back:
        ux, uy = -ux, -uy
    h = 11.0
    f.segments(layer, [m[0], m[0]], [m[1], m[1]], [m[0] - h * (0.9 * ux - 0.45 * uy), m[0] - h * (0.9 * ux + 0.45 * uy)],
               [m[1] - h * (0.9 * uy + 0.45 * ux), m[1] - h * (0.9 * uy - 0.45 * ux)], i, width=w)


def _dcurve(f, layer, xs, ys, i=0.8, w=L.LW):
    """A dashed curve: every other piece of a polyline."""
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    f.segments(layer, xs[:-1:2], ys[:-1:2], xs[1::2], ys[1::2], i, width=w)


def _lines(P, x, y, lines, size=None, lead=1.5, alpha=0.85, red=()):
    """Data lines (formulas allowed), top to bottom."""
    size = size or P.ts
    for k, ln in enumerate(lines):
        fml(P.f, "r" if k in red else "w", x, y + k * size * lead, ln, size=size, alpha=alpha)


def p_identity(P):
    f = P.f
    s = min(0.86 * P.h, 0.26 * P.w)
    xb, yb = P.x0 + 2.0, P.cy + 0.25 * s
    f.text("r", xb, yb, "µ", size=s)
    f.text("r", xb + 0.60 * s, yb - 0.40 * s, "−", size=0.4 * s)
    rows = (("MASS", "105.658 375 5 MEV"), ("MEAN LIFE", "2.196 981 1 µS"), ("CHARGE", "−1 E"), ("SPIN", "½"),
            ("FAMILY", "LEPTON, GEN 2"), ("M / M(E)", "206.768 283"), ("C × LIFE", f"{CTAU:.2f} M"))
    x = xb + 0.92 * s
    ts = fsz(28, P.x1 - x, P.tl)
    n = int(min(len(rows), max(3, P.h // (1.75 * ts))))
    dy = min(P.h / n, 2.2 * ts)
    y = P.cy - 0.5 * dy * (n - 1) + 0.35 * ts
    for k in range(n):
        f.text("w", x, y + k * dy, rows[k][0], size=ts, alpha=0.6)
        f.text("w", x + 11 * adv(f, ts), y + k * dy, rows[k][1], size=ts, alpha=0.97)


_SM = (("u", "2.2 MEV"), ("c", "1.27 GEV"), ("t", "173 GEV"), ("g", "0"),
       ("d", "4.7 MEV"), ("s", "93 MEV"), ("b", "4.18 GEV"), ("γ", "0"),
       ("e", "0.511 MEV"), ("µ", "105.66 MEV"), ("τ", "1.777 GEV"), ("Z", "91.19 GEV"),
       ("ν_e", "< 1 EV"), ("ν_µ", "< 1 EV"), ("ν_τ", "< 1 EV"), ("W", "80.4 GEV"))


def p_model(P):
    """The seventeen particles: three generations of quarks and leptons, the four force carriers, the Higgs."""
    f = P.f
    c = min((P.w - 4) / 5.3, P.h / 4.0)
    x0, y0 = P.cx - 2.65 * c, P.cy - 2.0 * c
    hot = P.a["n_on"] % 17
    k = np.arange(16)
    xa, ya = x0 + (k % 4) * c + np.where(k % 4 == 3, 0.3 * c, 0.0) + 3.0, y0 + (k // 4) * c + 3.0
    xb, yb = xa + c - 6.0, ya + c - 6.0
    f.segments("w", np.r_[xa, xb, xb, xa], np.r_[ya, ya, yb, yb], np.r_[xb, xb, xa, xa], np.r_[ya, yb, yb, ya],
               np.tile(np.where(k == hot, 1.0, 0.42), 4), width=L.LW)
    for k, (sym, mass) in enumerate(_SM):
        i, j = k % 4, k // 4
        xa, ya = x0 + i * c + (0.3 * c if i == 3 else 0.0), y0 + j * c
        red = sym == "µ"
        lay = "r" if red else "w"
        if red:
            f.rect("r", xa + 3, ya + 3, xa + c - 3, ya + c - 3, 1.0, width=L.LW_BOLD)
        fml(f, lay, xa + 0.5 * c, ya + (0.56 if c > 62 else 0.66) * c, sym, size=0.42 * c, anchor="m")
        if c > 62:
            f.text(lay, xa + 0.5 * c, ya + c - 10, mass, size=fsz(10, c - 12, 14), alpha=0.7, anchor="ms")
    xa = x0 + 4.3 * c
    f.rect("w", xa + 3, y0 + 3, xa + c - 3, y0 + 4 * c - 3, 1.0 if hot == 16 else 0.42)
    f.text("w", xa + 0.5 * c, y0 + 2.15 * c, "H", size=0.42 * c, anchor="ms")
    if c > 62:
        f.text("w", xa + 0.5 * c, y0 + 4 * c - 10, "125 GEV", size=fsz(10, c - 12, 14), alpha=0.7, anchor="ms")


def p_decay(P):
    """Feynman diagram of the decay: the muon becomes its neutrino and a W, the W an electron and an antineutrino."""
    f = P.f
    x0, y0, x1, y1 = P.box(2.0, sx=0.84, sy=0.74, ax=0.0, ay=0.0)
    w, h = x1 - x0, y1 - y0
    A, V1 = (x0 + 0.03 * w, y0 + 0.50 * h), (x0 + 0.38 * w, y0 + 0.50 * h)
    N1 = (x0 + 0.96 * w, y0 + 0.08 * h)
    V2 = (x0 + 0.62 * w, y0 + 0.76 * h)
    E_, N2 = (x0 + 0.97 * w, y0 + 0.99 * h), (x0 + 0.97 * w, y0 + 0.50 * h)
    _fermion(f, "r", A, V1, 1.0)
    _fermion(f, "w", V1, N1, 0.8)
    wavy(f, "w", V1[0], V1[1], V2[0], V2[1], 0.9, amp=0.022 * w, wl=0.065 * w, ph=-9.0 * P.u)
    _fermion(f, "w", V2, E_, 0.95)
    _fermion(f, "w", V2, N2, 0.8, back=True)
    f.dots("w", [V1[0], V2[0]], [V1[1], V2[1]], 3.4, 1.4)
    ts = P.tl
    fml(f, "r", A[0], A[1] - 14, "µ^−", ts)
    fml(f, "w", N1[0] + 8, N1[1] + 6, "ν_µ", ts)
    fml(f, "w", 0.5 * (V1[0] + V2[0]) - 3.2 * adv(f, ts), 0.5 * (V1[1] + V2[1]) + 1.25 * ts, "W^−", ts)
    fml(f, "w", E_[0] + 8, E_[1] + 2, "e^−", ts)
    fml(f, "w", N2[0] + 8, N2[1] + 6, "ν̄_e", ts)
    q = (P.u * 1.25) % 1.0                        # what travels: the muon, then the W, then the three that leave
    if q < 0.36:
        m = _along(A, V1, q / 0.36)
        f.dots("r", [m[0]], [m[1]], 4.6, 1.8)
    else:
        m = _along(V1, N1, (q - 0.36) / 0.64)
        xs, ys = [m[0]], [m[1]]
        if q < 0.52:
            m = _along(V1, V2, (q - 0.36) / 0.16)
            xs.append(m[0])
            ys.append(m[1])
        else:
            for end in (E_, N2):
                m = _along(V2, end, (q - 0.52) / 0.48)
                xs.append(m[0])
                ys.append(m[1])
        f.dots("w", xs, ys, 3.4, 1.6)
    yb = y1 + 0.5 * (P.y1 - y1) + 0.4 * ts
    xe = fml(f, "w", P.x0 + 4, yb, "µ^− → e^− + ν̄_e + ν_µ", ts)
    if P.x1 - xe > 22 * adv(f, P.ts):
        fml(f, "w", P.x1 - 2, yb, "1/τ = G_F^2 m_µ^5 / 192π³", P.ts, alpha=0.8, anchor="r")


def p_pion(P):
    """Where the muon is born: a charged pion at rest breaks in two, back to back."""
    f = P.f
    cx, cy = P.x0 + 0.40 * P.w, P.y0 + 0.36 * P.h
    R = 0.11 * min(P.h, 0.6 * P.w) + 7.0
    q = ((P.t - T0) / BEAT % 2.0) / 2.0
    d = min(1.0, q / 0.6)
    ring(f, "w", cx, cy, R, 0.9, n=40)
    hatch(f, "w", cx, cy, R, 0.4, step=7.0)
    xl, xr = P.x0 + 0.05 * P.w, P.x0 + 0.90 * P.w
    arrow(f, "r", cx + R + 5, cy, xr, cy, 1.0, L.LW_BOLD)
    dashes(f, "w", cx - R - 5, cy, xl, cy, 0.7, w=L.LW)
    arrow(f, "w", xl + 14, cy, xl, cy, 0.7)
    f.dots("r", [cx + R + 5 + (xr - cx - R - 5) * d], [cy], 5.0, 1.8)
    f.dots("w", [cx - R - 5 - (cx - R - 5 - xl) * d], [cy], 3.0, 1.2)
    ts = P.tl
    fml(f, "w", cx, cy - R - 10, "π^+", ts, anchor="m")
    fml(f, "r", xr - 2.2 * adv(f, ts), cy - 14, "µ^+", ts)
    fml(f, "w", xl + 2, cy - 14, "ν_µ", ts)
    arrow(f, "r", xr - 0.16 * P.w, cy + 16, xr - 0.16 * P.w - 34, cy + 16, 0.9, head=8.0)      # its spin: against its flight
    f.text("w", xr - 0.16 * P.w + 8, cy + 21, "SPIN", size=P.ts * 0.86, alpha=0.6)
    y = cy + R + 1.9 * P.ts + 20
    _lines(P, P.x0 + 4, y, ("π^+ → µ^+ + ν_µ      99.9877 %", f"P = {P_PI:.2f} MEV/C   T(µ) = 4.12 MEV",
                            "LIFE OF THE PION  26.033 NS"), lead=1.55)


def p_lifetime(P):
    """The decay law, with a counting experiment on top of it (a fixed set of 'measured' points)."""
    f = P.f
    ax = Ax(P, (0.0, 10.0), (0.0, 1.08))
    ax.axes(xt=(0, 2, 4, 6, 8, 10), yt=(0, 0.5, 1.0), xf=lambda v: f"{v:g}", yf=lambda v: f"{v:g}", xl="t (µS)",
            yl="N/N₀")
    t = np.linspace(0.0, 10.0, 80)
    k = np.arange(36)
    tk = (k + 0.5) * 10.0 / 36
    n0 = 150.0 * np.exp(-tk / 2.1969811)
    z = (hash01(k, 3) + hash01(k, 5) + hash01(k, 7) - 1.5) * 2.0
    nk = np.maximum(n0 + z * np.sqrt(n0 + 1.0), 0.0) / 150.0
    er = np.sqrt(n0 + 1.0) / 150.0
    m = k < 4 + P.u * 90.0                         # the points come in with the counts
    f.segments("w", ax.X(tk[m]), ax.Y(nk[m] - er[m]), ax.X(tk[m]), ax.Y(nk[m] + er[m]), 0.6)
    f.dots("w", ax.X(tk[m]), ax.Y(nk[m]), 2.4, 1.2)
    ax.curve("r", t, np.exp(-t / 2.1969811))
    xt, y1e = float(ax.X(2.1969811)), float(ax.Y(math.exp(-1)))
    dashes(f, "w", xt, ax.y1, xt, y1e, 0.6)
    dashes(f, "w", ax.x0, y1e, xt, y1e, 0.6)
    fml(f, "w", xt + 4, ax.y1 - 6, "τ", P.tl)
    f.text("w", ax.x0 + 5, y1e - 5, "1/e", size=P.ts * 0.86, alpha=0.6)
    tc = ((P.t - T0) / BAR % 1.0) * 10.0
    seg(f, "r", float(ax.X(tc)), ax.y1, float(ax.X(tc)), float(ax.Y(math.exp(-tc / 2.1969811))), 0.7)
    ax.mark("r", tc, math.exp(-tc / 2.1969811))
    x = ax.x0 + 0.42 * (ax.x1 - ax.x0)
    fml(f, "w", x, ax.y0 + P.tl, "N = N₀·e^{−t/τ}", P.tl)
    fml(f, "w", x, ax.y0 + 2.6 * P.tl, "τ = 2.196 981 1 µS", P.ts, alpha=0.8)


def p_michel(P):
    """Energy of the electron a muon at rest decays into."""
    f = P.f
    ax = Ax(P, (0.0, 1.0), (0.0, 1.15))
    ax.axes(xt=(0, 0.25, 0.5, 0.75, 1.0), yt=(0, 0.5, 1.0), xf=lambda v: f"{v * E_MICHEL:.0f}", yf=lambda v: f"{v:g}",
            xl="E (MEV)", yl="dN/dE")
    k = np.arange(20)
    xc = (k + 0.5) / 20
    yh = xc * xc * (3 - 2 * xc) * (1 + 0.16 * (hash01(k, 21) - 0.5)) * float(B.ease(min(1.0, P.u / 0.3)))
    xs = np.repeat(np.arange(21) / 20, 2)[1:-1]
    f.polyline("w", ax.X(xs), ax.Y(np.repeat(yh, 2)), 0.55, width=L.LW)
    x = np.linspace(0.0, 1.0, 50)
    ax.curve("r", x, x * x * (3 - 2 * x))
    dashes(f, "w", ax.x1, ax.y1, ax.x1, float(ax.Y(1.0)), 0.6)
    f.text("w", ax.x1 - 4, float(ax.Y(1.0)) - 8, f"{E_MICHEL:.2f} MEV", size=P.ts, alpha=0.9, anchor="rs")
    xm = float(ax.X(0.7))
    seg(f, "w", xm, ax.y1, xm, ax.y1 - 14, 0.9)
    f.text("w", xm, ax.y1 - 18, "MEAN 36.98", size=P.ts * 0.86, alpha=0.6, anchor="ms")
    fml(f, "w", ax.x0 + 14, ax.y0 + P.tl + 4, "dN/dx = 2x²(3 − 2x)", fsz(19, 0.56 * (ax.x1 - ax.x0), P.tl))
    if P.h > 260:
        fml(f, "w", ax.x0 + 14, ax.y0 + 2.6 * P.tl + 4, "x = E / E_{MAX}", P.ts, alpha=0.7)


def p_asym(P):
    """Where the decay positron goes: more often along the spin of the muon. Parity is not conserved."""
    f = P.f
    wide = P.w > 1.5 * P.h
    R = 0.40 * min(P.h, P.w * (0.55 if wide else 0.9))
    cx, cy = (P.x0 + 0.30 * P.w if wide else P.cx), P.cy
    a0 = 1.3 * P.t                                 # the spin turns (in a field): the whole pattern turns with it
    th = np.linspace(0.0, 2 * np.pi, 97)
    r3 = R * (1 + np.cos(th) / 3.0) / (4.0 / 3.0)
    r1 = R * (1 + np.cos(th)) / 2.0
    f.polyline("w", cx + r3 * np.cos(th + a0), cy - r3 * np.sin(th + a0), 0.95, width=L.LW_BOLD)
    _dcurve(f, "w", cx + r1 * np.cos(th + a0), cy - r1 * np.sin(th + a0), 0.6)
    k = np.arange(18) * (2 * np.pi / 18)
    rk = R * (1 + np.cos(k) / 3.0) / (4.0 / 3.0)
    f.segments("w", np.full(18, cx), np.full(18, cy), cx + rk * np.cos(k + a0), cy - rk * np.sin(k + a0), 0.45)
    f.dots("w", cx + rk * np.cos(k + a0), cy - rk * np.sin(k + a0), 2.2, 1.0)
    arrow(f, "r", cx, cy, cx + 1.16 * R * math.cos(a0), cy - 1.16 * R * math.sin(a0), 1.0, L.LW_BOLD, head=14.0)
    f.dots("r", [cx], [cy], 4.4, 1.6)
    if wide:
        x = P.x0 + 0.62 * P.w
        _lines(P, x, P.cy - 1.6 * P.tl, ("N(θ) = 1 + a·cosθ", "a = 1/3   ALL ENERGIES", "a = 1     AT 52.8 MEV",
                                         "RED: SPIN OF THE µ^+"), size=fsz(22, P.x1 - x, P.tl), lead=1.6, red=(3,))
    else:
        fml(f, "w", P.x0 + 2, P.y0 + P.tl, "N(θ) = 1 + a·cosθ", P.ts)
        f.text("w", P.x0 + 2, P.y1 - 4, "a = 1/3", size=P.ts, alpha=0.7)


def p_ring(P):
    """The storage ring of the g-2 experiment: on every turn the spin gains 12.3 degrees on the momentum."""
    f = P.f
    wide = P.w > 1.45 * P.h
    R = 0.40 * min(P.h, P.w * (0.6 if wide else 0.95))
    cx, cy = (P.x0 + 0.06 * P.w + 1.12 * R if wide else P.cx), P.cy
    ring(f, "w", cx, cy, R, 0.9, L.LW_BOLD)
    dring(f, "w", cx, cy, R - 9, 0.5, n=36)
    k = np.arange(24) * (2 * np.pi / 24)
    f.segments("w", cx + (R + 6) * np.cos(k), cy + (R + 6) * np.sin(k), cx + (R + 15) * np.cos(k), cy + (R + 15) * np.sin(k), 0.6)
    phi = 2 * np.pi * (P.t - T_DRUMS) / BAR
    px, py = cx + R * math.cos(phi), cy - R * math.sin(phi)
    tx, ty = -math.sin(phi), -math.cos(phi)
    dl = A_MU * G_MAGIC * phi                      # how far the spin is ahead of the momentum
    sx, sy = tx * math.cos(dl) + ty * math.sin(dl), ty * math.cos(dl) - tx * math.sin(dl)
    arrow(f, "w", px, py, px + 0.34 * R * tx, py + 0.34 * R * ty, 0.9)
    arrow(f, "r", px, py, px + 0.34 * R * sx, py + 0.34 * R * sy, 1.0, L.LW_BOLD)
    f.dots("r", [px], [py], 5.0, 1.8)
    f.text("w", cx, cy - 0.3 * P.ts, "B 1.451 T", size=P.ts, alpha=0.8, anchor="ms")
    f.text("r", cx, cy + 1.2 * P.ts, f"{math.degrees(dl) % 360.0:05.1f}°", size=P.tl, anchor="ms")
    if wide:
        x = cx + R + 34
        _lines(P, x, P.cy - 2.4 * P.tl, ("RADIUS  7.112 M", "P  3.094 GEV/C", f"γ  {G_MAGIC:.1f}",
                                         f"LIFE × γ  {G_MAGIC * TAU_MU * 1e6:.1f} µS", "1 TURN  149.1 NS",
                                         "RED: SPIN  WHITE: MOMENTUM"), size=fsz(26, P.x1 - x, P.ts + 2), lead=1.6)


def p_wiggle(P):
    """What the ring counts: positrons against time. The decay, and on top of it the beat of spin against momentum."""
    f = P.f
    ax = Ax(P, (0.0, 60.0), (0.0, 1.55))
    ax.axes(xt=(0, 20, 40, 60), xf=lambda v: f"{v:g}", xl="t (µS)", yl="COUNTS")
    t = np.linspace(0.0, 60.0, 301)
    gt = G_MAGIC * TAU_MU * 1e6
    y = np.exp(-t / gt) * (1 + 0.4 * np.cos(2 * np.pi * t / 4.365))
    ax.curve("w", t, y, 0.95, L.LW)
    _dcurve(f, "r", ax.X(t[::5]), ax.Y(np.exp(-t[::5] / gt)), 0.8)
    tc = ((P.t - T0) / BAR % 1.0) * 60.0
    ax.mark("r", tc, math.exp(-tc / gt) * (1 + 0.4 * math.cos(2 * math.pi * tc / 4.365)), 4.6)
    xr = ax.x1 + 8
    fml(f, "w", xr, ax.y0 + 0.9 * P.ts, "N₀e^{−t/γτ}[1 + A·cos ω_a t]", P.ts, anchor="r")
    fml(f, "r", xr, ax.y0 + 2.5 * P.ts, "a_µ = ω_a·m/eB = 0.001 165 920 715", P.ts, alpha=1.0, anchor="r")
    f.text("w", xr, ax.y0 + 3.9 * P.ts, "ONE WIGGLE = 4.365 µS", size=P.ts * 0.86, alpha=0.6, anchor="rs")


def p_schwinger(P):
    """Why g is not exactly 2: the muon talks to the vacuum (the first loop, Schwinger's term)."""
    f = P.f
    x0, y0, x1, y1 = P.box(1.0, sx=0.42, ax=0.0)
    w, h = x1 - x0, y1 - y0
    V = (x0 + 0.5 * w, y0 + 0.42 * h)
    A, Bq = (x0 + 0.04 * w, y0 + 0.98 * h), (x0 + 0.96 * w, y0 + 0.98 * h)
    _fermion(f, "r", A, V, 1.0)
    _fermion(f, "r", V, Bq, 1.0)
    wavy(f, "w", V[0], V[1], V[0], y0 + 0.02 * h, 0.9, amp=0.035 * w, wl=0.11 * w, ph=-9.0 * P.u)
    m1, m2 = _along(A, V, 0.5), _along(V, Bq, 0.5)
    wavy(f, "w", m1[0], m1[1], m2[0], m2[1], 0.8, amp=0.035 * w, wl=0.11 * w, ph=7.0 * P.u)
    f.dots("w", [V[0], m1[0], m2[0]], [V[1], m1[1], m2[1]], 3.0, 1.3)
    fml(f, "w", V[0] + 10, y0 + 0.10 * h, "γ", P.tl)
    fml(f, "r", A[0] + 4, A[1] - 0.26 * h, "µ", P.tl)
    x = x1 + 0.05 * P.w
    ts = fsz(34, P.x1 - x, P.tl)
    _lines(P, x, P.cy - 2.3 * ts, ("g = 2·(1 + a)", "DIRAC 1928     a = 0", "SCHWINGER 1948 a = α/2π",
                                   "               = 0.001 161 41", "MEASURED 2025  a = 0.001 165 920 7"),
           size=ts, lead=1.55, red=(4,))


def _zenith_angles(n):
    """n zenith angles spread like the muons themselves (cos2 of the angle)."""
    th = np.linspace(-np.pi / 2, np.pi / 2, 721)
    cdf = (th + 0.5 * np.sin(2 * th)) / np.pi + 0.5
    return np.interp((np.arange(n) + 0.5) / n, cdf, th)


_ZEN = _zenith_angles(11)


def p_zenith(P):
    """Muons at sea level: most come from straight above."""
    f = P.f
    wide = P.w > 1.7 * P.h
    R = min(0.82 * P.h, (0.30 if wide else 0.46) * P.w)
    ox, oy = (P.x0 + 0.02 * P.w + R if wide else P.cx), P.y1 - 0.08 * P.h
    seg(f, "w", ox - 1.05 * R, oy, ox + 1.05 * R, oy, 0.8)
    th = np.linspace(-np.pi / 2, np.pi / 2, 61)
    r = R * np.cos(th) ** 2
    f.polyline("w", ox + r * np.sin(th), oy - r * np.cos(th), 0.95, width=L.LW_BOLD)
    a = np.pi + np.arange(24) / 24 * np.pi
    f.segments("w", ox + R * np.cos(a), oy + R * np.sin(a), ox + R * np.cos(a + np.pi / 48), oy + R * np.sin(a + np.pi / 48), 0.4)
    k = np.radians(np.arange(-80, 81, 10))
    rk = R * np.cos(k) ** 2
    f.segments("w", ox + rk * np.sin(k), oy - rk * np.cos(k), np.full(len(k), ox), np.full(len(k), oy), 0.4)
    q = (P.t * 0.9 + hash01(np.arange(len(_ZEN)), 31)) % 1.0
    d = 1.25 * R * (1 - q)
    f.dots("r", ox + d * np.sin(_ZEN), oy - d * np.cos(_ZEN), 3.2, 1.6)
    f.rects("r", ox - 16, oy - 5, ox + 16, oy + 1, 1.0)
    for dg in (30, 60):
        a = math.radians(dg)
        f.text("w", ox + 1.07 * R * math.sin(a), oy - 1.07 * R * math.cos(a), f"{dg}°", size=P.ts * 0.86, alpha=0.6)
    if wide:
        x = ox + 1.2 * R
        _lines(P, x, P.cy - 2.2 * P.tl, ("I(θ) = I₀·cos²θ", "1 / CM² / MINUTE", "70 / M² / S / SR  VERTICAL",
                                         "MEAN ENERGY  4 GEV", "µ^+ / µ^− = 1.28"), size=fsz(26, P.x1 - x, P.tl), lead=1.6)
    else:
        fml(f, "w", P.x0 + 2, P.y0 + P.tl, "I(θ) = I₀·cos²θ", P.ts)


def p_spectrum(P):
    """Momentum spectrum of the muons at sea level (vertical intensity; Bugaev / Reyna parametrisation)."""
    f = P.f
    ax = Ax(P, (1.0, 1000.0), (1e-10, 1e-2), logx=True, logy=True, ml=56.0)
    ax.axes(xt=(1, 10, 100, 1000), yt=(1e-10, 1e-8, 1e-6, 1e-4, 1e-2), xf=lambda v: f"{v:g}",
            yf=lambda v: f"1E{int(round(math.log10(v))):+d}", xl="p (GEV/C)", yl="I(p)")
    p = np.logspace(0.0, 3.0, 50)
    lg = np.log10(p)
    inten = 0.00253 * p ** -(0.2455 + 1.288 * lg - 0.2555 * lg ** 2 + 0.0209 * lg ** 3)
    ax.curve("r", p, inten)
    k = np.arange(1, 49, 4)
    yk = inten[k] * (1 + 0.5 * (hash01(k, 41) - 0.5))
    f.segments("w", ax.X(p[k]), ax.Y(yk * 0.6), ax.X(p[k]), ax.Y(yk * 1.6), 0.6)
    f.dots("w", ax.X(p[k]), ax.Y(yk), 2.4, 1.2)
    xm = float(ax.X(4.0))
    dashes(f, "w", xm, ax.y1, xm, ax.y0 + 10, 0.5)
    f.text("w", xm + 6, ax.y0 + 0.5 * (ax.y1 - ax.y0), "MEAN 4 GEV", size=P.ts * 0.86, alpha=0.7)
    x = ax.x0 + 0.40 * (ax.x1 - ax.x0)
    f.text("w", x, ax.y0 + P.ts, "/ CM² / S / SR / (GEV/C)", size=P.ts * 0.86, alpha=0.6)
    if P.h > 250:
        f.text("w", x, ax.y0 + 2.5 * P.ts, "STEEPER AND STEEPER: THE FAST", size=P.ts * 0.86, alpha=0.6)
        f.text("w", x, ax.y0 + 3.8 * P.ts, "PIONS HIT AIR BEFORE THEY DECAY", size=P.ts * 0.86, alpha=0.6)


def p_bethe(P):
    """How much energy a muon leaves in the air it crosses (Bethe)."""
    f = P.f
    ax = Ax(P, (0.3, 1000.0), (1.0, 20.0), logx=True, logy=True)
    ax.axes(xt=(1, 10, 100, 1000), yt=(1, 2, 5, 10, 20), xf=lambda v: f"{v:g}", yf=lambda v: f"{v:g}", xl="βγ = p/mc",
            yl="−dE/dx")
    bg = np.logspace(math.log10(0.3), 3.0, 70)
    ax.curve("w", bg, _bethe(bg), 0.95)
    ym = float(ax.Y(1.815))
    dashes(f, "r", ax.x0, ym, ax.x1, ym, 0.7)
    ax.mark("r", 3.31, 1.815, 4.6)
    f.text("r", ax.x1, ym + 1.3 * P.ts, "MINIMUM 1.815 MEV CM²/G", size=P.ts, anchor="rs")
    ax.mark("w", BGAM, float(_bethe(BGAM)), 3.4)
    f.text("w", float(ax.X(BGAM)), float(ax.Y(_bethe(BGAM))) - 10, "4 GEV", size=P.ts * 0.86, alpha=0.8, anchor="ms")
    fml(f, "w", float(ax.X(0.42)) + 6, float(ax.Y(8.0)), "1/β²", P.ts, alpha=0.7)
    f.text("w", ax.x1, ax.y0 + P.ts, "MUON IN AIR", size=P.ts, alpha=0.8, anchor="rs")
    f.text("w", ax.x1, ax.y0 + 2.4 * P.ts, "1030 G/CM² OF SKY = 2 GEV", size=P.ts * 0.86, alpha=0.6, anchor="rs")


def p_cherenkov(P):
    """A muon faster than light in water: the wavelets it leaves add up to a cone."""
    f = P.f
    cy = P.y0 + 0.44 * P.h
    K = 6
    d = min(0.105 * P.w, 0.40 * P.h / (K * 0.7519))
    q = (P.u * 0.8) % 1.0
    xp = P.x0 + 0.52 * P.w + (0.36 * P.w) * q + 0.5 * d
    seg(f, "r", P.x0 + 4, cy, xp, cy, 0.9, L.LW_BOLD)
    dashes(f, "w", xp, cy, P.x1 - 4, cy, 0.4)
    for j in range(1, K + 1):
        xc, r = xp - j * d, j * d * 0.7519
        if xc - r > P.x0:
            ring(f, "w", xc, cy, r, 0.30 + 0.5 / j, n=40)
    ps = math.asin(0.7519)                         # half-angle of the cone the wavelets build
    ln = K * d * math.cos(ps)
    for sg in (-1, 1):
        seg(f, "w", xp, cy, xp - ln * math.cos(ps), cy + sg * ln * math.sin(ps), 1.0, L.LW_BOLD)
    tc = math.acos(0.7519)                         # the light leaves at 41.2 degrees from the track
    bx, by = xp - 0.55 * ln * math.cos(ps), cy - 0.55 * ln * math.sin(ps)
    arrow(f, "w", bx, by, bx + 0.2 * P.h * math.cos(tc), by - 0.2 * P.h * math.sin(tc), 0.8)
    f.dots("r", [xp], [cy], 5.0, 1.8)
    y = P.y1 - 2.9 * P.ts
    _lines(P, P.x0 + 4, y, ("cosθ = 1/nβ   WATER n = 1.33   θ = 41.2°", "A MUON ABOVE 160 MEV LIGHTS UP WATER"),
           size=fsz(40, P.w - 8, P.ts + 1), lead=1.5)


def p_muonic(P):
    """Muonic hydrogen: the muon orbits 186 times closer than the electron and feels the size of the proton."""
    f = P.f
    R1 = 0.40 * min(P.h * 0.92, 0.46 * P.w)
    c1 = (P.x0 + 0.04 * P.w + R1, P.y0 + 0.46 * P.h)
    R2 = 0.92 * R1
    c2 = (P.x1 - 0.04 * P.w - R2, P.y0 + 0.46 * P.h)
    dring(f, "w", c1[0], c1[1], R1, 0.7, n=40)
    a = 1.3 * P.t
    f.dots("w", [c1[0] + R1 * math.cos(a)], [c1[1] - R1 * math.sin(a)], 3.4, 1.4)
    f.dots("r", [c1[0]], [c1[1]], 2.6, 1.6)
    ring(f, "w", c2[0], c2[1], R2, 0.6, n=56)
    al = math.asin(min(1.0, R2 / (c2[0] - c1[0])))
    for sg in (-1, 1):                              # the lens: from the nucleus to the enlarged view
        seg(f, "w", c1[0], c1[1], c2[0] - R2 * math.sin(al), c2[1] + sg * R2 * math.cos(al), 0.35)
    rp = 0.10 * R2
    ring(f, "w", c2[0], c2[1], rp, 0.9, n=24)
    hatch(f, "w", c2[0], c2[1], rp, 0.6, step=4.0)
    ring(f, "r", c2[0], c2[1], 0.62 * R2, 1.0, L.LW_BOLD, n=56)
    b = 8.0 * P.t
    f.dots("r", [c2[0] + 0.62 * R2 * math.cos(b)], [c2[1] - 0.62 * R2 * math.sin(b)], 4.4, 1.8)
    ts = P.ts
    fml(f, "w", c1[0], c1[1] + R1 + 1.4 * ts, "e^−  52 918 FM", ts, anchor="m")
    fml(f, "r", c2[0], c2[1] + R2 + 1.4 * ts, "µ^−  285 FM", ts, anchor="m")
    f.text("w", 0.5 * (c1[0] + c2[0]), c1[1] - R1 * 0.2, "× 186", size=P.tl, alpha=0.9, anchor="ms")
    if P.y1 - (c1[1] + R1 + 1.4 * ts) > 1.6 * ts:
        f.text("w", P.cx, P.y1 - 3, "E = −2.53 KEV   PROTON RADIUS 0.840 87 FM", size=fsz(42, P.w, ts), alpha=0.7, anchor="ms")


def p_muonium(P):
    """Muonium (a muon and an electron): its four ground levels in a magnetic field (Breit-Rabi)."""
    f = P.f
    ax = Ax(P, (0.0, 3.0), (-2.1, 2.1), mr=64.0)
    ax.axes(xt=(0, 1, 2, 3), yt=(-2, -1, 0, 1, 2), xf=lambda v: f"{v:g}", yf=lambda v: f"{v:g}", xl="B / 0.1585 T",
            yl="E / ∆W")
    x = np.linspace(0.0, 3.0, 40)
    lev = (0.25 + 0.4952 * x, -0.25 + 0.5 * np.sqrt(1 + x * x), 0.25 - 0.4952 * x, -0.25 - 0.5 * np.sqrt(1 + x * x))
    names = ("1,+1", "1, 0", "1,−1", "0, 0")
    xc = 1.5 + 1.35 * math.sin(2 * math.pi * (P.t - T0) / (2 * BAR)) * (0.6 + 0.4 * min(1.0, P.a["bass"] * 1.4))
    for e, nm in zip(lev, names):
        ax.curve("w", x, e, 0.9, L.LW)
        f.text("w", ax.x1 + 8, float(ax.Y(e[-1])) + 5, nm, size=P.ts * 0.86, alpha=0.7)
        ax.mark("r", xc, float(np.interp(xc, x, e)), 3.6)
    seg(f, "r", float(ax.X(xc)), ax.y0, float(ax.X(xc)), ax.y1, 0.5)
    f.text("w", ax.x0 + 12, ax.y0 + P.ts, "∆W / h = 4 463.302 MHZ", size=P.ts, alpha=0.9)
    f.text("w", ax.x1 + 8, ax.y0 + 2, "F, m", size=P.ts * 0.86, alpha=0.5)


def p_pyramid(P):
    """Muography of the pyramid of Khufu: more muons arrive where there is less stone (ScanPyramids, 2017).
    The section is a sketch: the chambers and the void are where the publications put them, not surveyed."""
    f = P.f
    s = min(0.94 * P.w / 230.3, 0.90 * P.h / 138.5)           # px per metre
    cx, yb = P.cx, min(P.y1 - 0.04 * P.h, P.cy + 0.56 * s * 138.5)
    X = lambda m: cx + s * np.asarray(m, np.float64)
    Y = lambda z: yb - s * np.asarray(z, np.float64)
    f.polyline("w", X([-115.15, 0.0, 115.15, -115.15]), Y([0.0, 138.5, 0.0, 0.0]), 0.95, width=L.LW_BOLD)
    f.polyline("w", X([-101.8, -76.0, -30.0, 12.0]), Y([17.0, 0.5, 21.5, 42.5]), 0.6)      # passages, grand gallery
    f.polyline("w", X([-30.0, -3.0]), Y([21.5, 21.5]), 0.6)
    f.rect("w", float(X(-3.0)), float(Y(27.0)), float(X(3.0)), float(Y(21.0)), 0.8)            # queen's chamber
    f.rect("w", float(X(12.0)), float(Y(49.0)), float(X(22.0)), float(Y(43.0)), 0.8)           # king's chamber
    u = np.linspace(0.0, 4.0, 41)
    _dcurve(f, "r", X(np.interp(u, np.arange(5), [-24.0, 3.0, 5.0, -22.0, -24.0])),
            Y(np.interp(u, np.arange(5), [47.0, 60.5, 66.0, 52.5, 47.0])), 1.0)                # the void: above the gallery
    sw = (P.t - T0) / BAR
    for k, a in enumerate(np.radians(np.linspace(-40.0, 40.0, 9))):       # muons from the sky to the detectors
        hit = abs(math.degrees(a) + 12.0) < 9.0       # these cross the void: there are more of them
        ln = (1 - 24.0 / 138.5) / (abs(math.sin(a)) / 115.15 + math.cos(a) / 138.5) + 16.0     # from just outside the face
        f.segments("r" if hit else "w", [float(X(ln * math.sin(a)))], [float(Y(24.0 + ln * math.cos(a)))], [float(X(0.0))],
                   [float(Y(24.0))], 0.9 if hit else 0.3)
        q = (sw + k / 9.0) % 1.0
        f.dots("r" if hit else "w", [float(X(ln * (1 - q) * math.sin(a)))], [float(Y(24.0 + ln * (1 - q) * math.cos(a)))],
               2.6, 1.5 if hit else 0.8)
    ts = P.ts * 0.9
    f.text("r", P.x0 + 2, float(Y(84.0)), "BIG VOID ≥ 30 M", size=ts)
    seg(f, "r", P.x0 + 2 + 7.5 * adv(f, ts), float(Y(84.0)) + 5, float(X(-20.0)), float(Y(57.0)), 0.7)
    f.text("w", float(X(26.0)), float(Y(44.0)), "KING", size=ts, alpha=0.7)
    f.text("w", float(X(7.0)), float(Y(20.0)), "QUEEN: DETECTORS", size=ts, alpha=0.7)
    f.text("w", float(X(-112.0)), float(Y(118.0)), "KHUFU  139 M", size=P.ts, alpha=0.8)


def p_scatter(P):
    """Multiple scattering: the denser the matter, the more a muon is deflected (how a scanner sees through)."""
    f = P.f
    xa, xb = P.x0 + 0.40 * P.w, P.x0 + 0.52 * P.w
    ya, yb = P.y0 + 0.04 * P.h, P.y0 + 0.70 * P.h
    f.rect("w", xa, ya, xb, yb, 0.9, width=L.LW_BOLD)
    k = np.arange(1, int((yb - ya + xb - xa) / 9.0))
    c = ya + k * 9.0                                  # hatching: lines x + y = const, cut by the slab
    xs0, xs1 = np.maximum(xa, xa + (c - yb)), np.minimum(xb, xa + (c - ya))
    m = (xs1 > xs0) & (k % thin(2) == 0)
    f.segments("w", xs0[m], (c - (xs0 - xa))[m], xs1[m], (c - (xs1 - xa))[m], 0.3)
    n = 9
    ev = P.a["n_on"] // 2
    y = ya + (np.arange(n) + 0.5) / n * (yb - ya)
    z = (hash01(np.arange(n), ev, 51) + hash01(np.arange(n), ev, 53) + hash01(np.arange(n), ev, 57) - 1.5) * 2.0
    f.segments("w", np.full(n, P.x0 + 4), y, np.full(n, xa), y, 0.7)
    ye = y + z * 0.075 * (P.x1 - 8 - xb)
    f.segments("r", np.full(n, xb), y, np.full(n, P.x1 - 8), ye, 0.9)
    f.dots("r", np.full(n, P.x1 - 8), ye, 2.4, 1.3)
    f.text("w", 0.5 * (xa + xb), ya + P.ts + 4, "x", size=P.tl, alpha=0.9, anchor="ms")
    ts = fsz(51, P.w - 8, P.ts + 1)
    _lines(P, P.x0 + 4, yb + 2.0 * ts, ("θ₀ = (13.6 MEV/βcp)·√(x/X₀)·[1 + 0.038 ln(x/X₀)]",
                                        f"10 CM, 3 GEV/C:  LEAD {1e3 * _highland(10, 0.5612, 3000):.0f} MRAD"
                                        f"   IRON {1e3 * _highland(10, 1.757, 3000):.1f} MRAD"), size=ts, lead=1.6)


def p_moon(P):
    """The Moon stops cosmic rays: seen in muons it is a hole in the sky, half a degree wide."""
    f = P.f
    wide = P.w > 1.6 * P.h
    r = 0.44 * min(P.h - 20, P.w * (0.6 if wide else 0.9))
    cx, cy = (P.x0 + r + 34 if wide else P.cx + 10), P.cy - 8
    f.rect("w", cx - r, cy - r, cx + r, cy + r, 0.7)
    tk = np.arange(-2, 3) * (r / 2.0)
    f.segments("w", cx + tk, np.full(5, cy + r), cx + tk, np.full(5, cy + r + 6), 0.7)
    f.segments("w", np.full(5, cx - r - 6), cy + tk, np.full(5, cx - r), cy + tk, 0.7)
    for v in (-2, 0, 2):
        f.text("w", cx + v * r / 2, cy + r + 8 + 0.8 * P.ts, f"{v:+d}°" if v else "0", size=P.ts * 0.86, alpha=0.6, anchor="ms")
    k = np.arange(520)
    x, y = (B.rnd(520, 61, 0) * 2 - 1) * 2.0, (B.rnd(520, 63, 0) * 2 - 1) * 2.0
    keep = (np.hypot(x, y) > 0.55) | (B.rnd(520, 67, 0) > 0.78)
    keep &= k < 30 + P.u * 900.0                    # the events come in
    f.dots("w", cx + x[keep] * r / 2, cy - y[keep] * r / 2, 1.5, 0.95)
    dring(f, "r", cx, cy, 0.26 * r / 2, 1.0, n=20, w=L.LW)
    dring(f, "w", cx, cy, 0.55 * r / 2, 0.5, n=30)
    f.segments("r", [cx - 9, cx], [cy, cy - 9], [cx + 9, cx], [cy, cy + 9], 0.9)
    f.text("w", cx, cy - r - 6, "N", size=P.ts, alpha=0.7, anchor="ms")
    f.text("w", cx + r + 6, cy + 5, "E", size=P.ts, alpha=0.7)
    if wide:
        x = cx + r + 40
        _lines(P, x, P.cy - 1.5 * P.tl, ("THE MOON  0.52°", "FEWER MUONS FROM THERE", "SEEN BY ICECUBE, MINOS",
                                         "IT CHECKS WHERE A DETECTOR LOOKS"), size=fsz(33, P.x1 - x, P.tl), lead=1.6, red=(0,))
    else:
        f.text("r", cx - r + 6, cy - r + P.ts + 6, "MOON 0.52°", size=P.ts)


def _make_tree():
    """A schematic air shower: segments (x0, y0, x1, y1, kind) in a unit box, kind 0 hadron, 1 electromagnetic,
    2 muon, 3 neutrino. One proton, two generations of pions; the neutral ones turn into light and electron
    pairs, the charged ones into muons that reach the ground."""
    rng = np.random.default_rng(9)
    out = [(0.5, 0.0, 0.5, 0.13, 0)]

    def em(x, y, dx, depth):
        if depth == 0 or y > 0.86:
            return
        for sg in (-1, 1):
            x2, y2 = x + sg * dx * rng.uniform(0.7, 1.2), min(y + rng.uniform(0.07, 0.11), 0.90)
            out.append((x, y, x2, y2, 1))
            em(x2, y2, dx * 0.62, depth - 1)

    def had(x, y, spread, gen):
        for k, off in enumerate((-1.0, -0.42, 0.36, 1.0)):
            x2, y2 = x + off * spread, y + rng.uniform(0.10, 0.15)
            if k == 1:                               # a neutral pion: two photons, then pairs
                out.append((x, y, x2, y2 - 0.04, 0))
                em(x2, y2 - 0.04, 0.05 * (1.5 if gen == 0 else 1.0), 4 if gen == 0 else 3)
            else:
                out.append((x, y, x2, y2, 0))
                out.append((x2, y2, x2 + off * 0.10 + rng.uniform(-0.02, 0.02), 1.0, 2))      # the muon: to the ground
                out.append((x2, y2, x2 - off * 0.05, y2 + 0.10, 3))                           # its neutrino
        if gen == 0:
            out.append((x, y, x + 0.03, y + 0.24, 0))                                         # the nucleon goes on
            had(x + 0.03, y + 0.24, spread * 0.55, 1)

    had(0.5, 0.13, 0.30, 0)
    return np.array(out, np.float64)


_TREE = _make_tree()


def p_tree(P):
    """One primary, a cascade: the pions decay into the muons that reach the ground."""
    f = P.f
    wide = P.w > 2.0 * P.h
    x0, y0, x1, y1 = P.box(1.25, ax=0.06 if wide else 0.5)
    w, h = x1 - x0, y1 - y0 - 12
    T = _TREE
    for kind, lay, inten, wd in ((0, "w", 0.9, L.LW_BOLD), (1, "w", 0.5, L.LW_HAIR), (2, "r", 0.95, L.LW)):
        m = T[:, 4] == kind
        f.segments(lay, x0 + T[m, 0] * w, y0 + T[m, 1] * h, x0 + T[m, 2] * w, y0 + T[m, 3] * h, inten, width=wd)
    for a in T[T[:, 4] == 3]:
        dashes(f, "w", x0 + a[0] * w, y0 + a[1] * h, x0 + a[2] * w, y0 + a[3] * h, 0.4, dash=5.0, gap=4.0)
    seg(f, "w", x0, y0 + h, x1, y0 + h, 0.9)
    mu = T[T[:, 4] == 2]
    q = ((P.t - T0) / BAR + hash01(np.arange(len(mu)), 71)) % 1.0
    f.dots("r", x0 + (mu[:, 0] + (mu[:, 2] - mu[:, 0]) * q) * w, y0 + (mu[:, 1] + (mu[:, 3] - mu[:, 1]) * q) * h, 2.8, 1.6)
    f.text("w", x0 + 0.5 * w + 8, y0 + 0.07 * h, "p", size=P.tl, alpha=0.95)
    f.text("w", x0 + 0.5 * w + 10, y0 + 0.13 * h + 4, "15 KM", size=P.ts * 0.86, alpha=0.6)
    f.text("w", x0 + 2, y0 + h - 5, "GROUND", size=P.ts * 0.86, alpha=0.6)
    if wide:
        x = x1 + 0.06 * P.w
        _lines(P, x, P.cy - 2.2 * P.tl, ("p + AIR → π^+ π^− π⁰ ...", "π⁰ → γγ → e^+e^− ...", "π^± → µ^± + ν",
                                         "A 1E15 EV PROTON:", "A MILLION PARTICLES"), size=fsz(26, P.x1 - x, P.tl),
               lead=1.6, red=(2,))
    else:
        fml(f, "r", x0 + 2, y0 + P.tl, "π → µ + ν", P.ts)


def p_crspec(P):
    """The primaries: flux against energy, twelve decades."""
    f = P.f
    ax = Ax(P, (1e9, 1e21), (1e-38, 1e2), logx=True, logy=True, ml=22.0)
    ax.axes(xt=(1e9, 1e12, 1e15, 1e18, 1e21), xf=lambda v: f"1E{int(round(math.log10(v)))}", xl="E (EV)", yl="FLUX")
    le = np.array([9.0, 15.48, 18.7, 19.7, 20.6])
    lf = np.cumsum([0.0, -2.7 * 6.48, -3.1 * 3.22, -2.6, -4.5])
    ax.curve("w", 10.0 ** le, 10.0 ** lf, 0.95)
    for e, lay, t1, t2 in ((15.48, "r", "KNEE 3E15 EV", "1 / M² / YEAR"), (18.7, "w", "ANKLE 5E18 EV", "1 / KM² / YEAR"),
                           (11.0, "w", "1E11 EV", "1 / M² / S")):
        y = float(np.interp(e, le, lf))
        ax.mark(lay, 10.0 ** e, 10.0 ** y, 4.2)
        f.text(lay, float(ax.X(10.0 ** e)) + 10, float(ax.Y(10.0 ** y)) - 0.9 * P.ts, t1, size=P.ts * 0.9, alpha=0.95)
        f.text("w", float(ax.X(10.0 ** e)) + 10, float(ax.Y(10.0 ** y)) + 0.2 * P.ts, t2, size=P.ts * 0.8, alpha=0.6)
    xl = float(ax.X(6.8e12))
    arrow(f, "w", xl, ax.y1 - 34, xl, ax.y1 - 4, 0.7, head=8.0)
    f.text("w", xl, ax.y1 - 40, "LHC BEAM", size=P.ts * 0.8, alpha=0.6, anchor="ms")
    fml(f, "w", float(ax.X(3e9)), float(ax.Y(1e-12)), "E^{−2.7}", P.ts, alpha=0.7)
    fml(f, "w", float(ax.X(1e16)), float(ax.Y(1e-31)), "E^{−3.1}", P.ts, alpha=0.7)


def p_cms(P):
    """A slice of CMS, the Compact Muon Solenoid: the muon is the track that goes through everything."""
    f = P.f
    wide = P.w > 1.5 * P.h
    R = 0.47 * min(P.h, P.w * (0.62 if wide else 1.0))
    cx, cy = (P.x0 + R + 6 if wide else P.cx), P.cy
    sc = R / 7.5                                      # px per metre
    for rr, i in ((1.2, 0.5), (1.8, 0.35), (2.9, 0.5)):
        ring(f, "w", cx, cy, rr * sc, i, n=60)
    ring(f, "w", cx, cy, 2.95 * sc, 0.9, L.LW_BOLD, n=64)
    ring(f, "w", cx, cy, 3.8 * sc, 0.9, L.LW_BOLD, n=64)
    for rr in (4.4, 5.3, 6.3, 7.3):
        dring(f, "w", cx, cy, rr * sc, 0.55, n=24)
    ev = P.a["n_k"] // 2
    r3 = B.rnd(3, 81, ev)
    pt = 3.0 + 17.0 * float(r3[0])
    ph = 2 * math.pi * float(r3[1])
    sgn = 1.0 if r3[2] > 0.5 else -1.0
    x = y = 0.0
    xs, ys = [0.0], [0.0]
    ds = 0.15
    for _ in range(int(7.4 / ds) + 8):               # the track: bent one way in the coil, the other way in the iron
        if math.hypot(x, y) > 7.4:
            break
        ph += sgn * ds * 0.3 * (3.8 if math.hypot(x, y) < 3.4 else -1.9) / pt
        x, y = x + ds * math.cos(ph), y + ds * math.sin(ph)
        xs.append(x)
        ys.append(y)
    f.polyline("r", cx + np.array(xs) * sc, cy - np.array(ys) * sc, 1.0, width=L.LW_BOLD)
    j = np.arange(6)
    a0 = (2 * np.pi * B.rnd(6, 87, ev))[:, None]
    rc = ((0.25 + 0.9 * B.rnd(6, 89, ev)) * np.where(B.rnd(6, 91, ev) > 0.5, 1.0, -1.0))[:, None]
    tt = np.linspace(0.0, 1.0, 14)[None, :] * (0.9 + 1.2 * B.rnd(6, 93, ev))[:, None] / np.abs(rc)
    px = rc * (np.sin(a0 + tt * np.sign(rc)) - np.sin(a0))          # the soft tracks: they curl up in the tracker
    py = -rc * (np.cos(a0 + tt * np.sign(rc)) - np.cos(a0))
    f.segments("w", cx + px[:, :-1].ravel() * sc, cy - py[:, :-1].ravel() * sc, cx + px[:, 1:].ravel() * sc,
               cy - py[:, 1:].ravel() * sc, 0.6)
    f.text("w", cx, cy + 3.4 * sc + 0.36 * P.ts, "3.8 T", size=P.ts * 0.9, alpha=0.8, anchor="ms")
    if wide:
        xx = cx + R + 28
        _lines(P, xx, P.cy - 2.4 * P.tl, ("COMPACT MUON SOLENOID", f"p_T  {pt:04.1f} GEV/C", "p_T = 0.3·B·R", "14 000 TONNES",
                                          "100 M UNDER GROUND", "RED: THE MUON"), size=fsz(22, P.x1 - xx, P.tl), lead=1.6,
               red=(1, 5))
    else:
        f.text("r", P.x0 + 2, P.y0 + P.ts, f"{pt:04.1f} GEV/C", size=P.ts)


def p_nuosc(P):
    """The neutrino that is born with the muon changes into another one on its way: it has a mass."""
    f = P.f
    ax = Ax(P, (10.0, 2500.0), (0.0, 1.12), logx=True)
    ax.axes(xt=(10, 100, 1000, 2500), yt=(0, 0.5, 1.0), xf=lambda v: f"{v:g}", yf=lambda v: f"{v:g}", xl="L/E (KM/GEV)",
            yl="P")
    le = np.logspace(1.0, math.log10(2500.0), 181)
    s2 = np.sin(1.267 * DM2 * le) ** 2
    ax.curve("w", le, 1 - s2, 0.95)
    _dcurve(f, "r", ax.X(le), ax.Y(s2), 0.9)
    lc = 10.0 ** (1.0 + 2.4 * (0.5 + 0.5 * math.sin(2 * math.pi * (P.t - T0) / (2 * BAR))))
    ax.mark("w", lc, 1 - math.sin(1.267 * DM2 * lc) ** 2, 3.6)
    ax.mark("r", lc, math.sin(1.267 * DM2 * lc) ** 2, 3.6)
    xm = float(ax.X(495.9))
    seg(f, "w", xm, ax.y1, xm, ax.y1 - 12, 0.9)
    f.text("w", xm, ax.y1 - 16, "496", size=P.ts * 0.86, alpha=0.7, anchor="ms")
    yt = ax.y1 - 0.42 * (ax.y1 - ax.y0)
    fml(f, "w", ax.x0 + 10, yt, "ν_µ → ν_µ", P.ts)
    fml(f, "r", ax.x0 + 10, yt + 1.5 * P.ts, "ν_µ → ν_τ", P.ts)
    f.text("w", ax.x0 + 10, yt + 3.0 * P.ts, "∆m² = 2.5E-3 EV²", size=P.ts * 0.86, alpha=0.6)


# ----------------------------------------------------------------------------
# the plates: special relativity
# ----------------------------------------------------------------------------

def p_gamma(P):
    """The Lorentz factor: how much time stretches, against speed."""
    f = P.f
    ax = Ax(P, (0.0, 1.0), (1.0, 100.0), logy=True)
    ax.axes(xt=(0, 0.2, 0.4, 0.6, 0.8, 1.0), yt=(1, 10, 100), xf=lambda v: f"{v:g}", yf=lambda v: f"{v:g}", xl="β = v/c",
            yl="γ")
    b = np.concatenate([np.linspace(0.0, 0.9, 30), 1.0 - 10.0 ** np.linspace(-1.0, -4.3, 60)])
    g = 1.0 / np.sqrt(1.0 - b * b)
    ax.curve("w", b[g <= 100.0], g[g <= 100.0], 0.95)
    gc = GAM ** float(B.ease(min(1.0, P.u / 0.5)))          # the cursor runs up the curve, then stays on the muon
    bc = math.sqrt(1.0 - 1.0 / (gc * gc))
    yc = float(ax.Y(gc))
    dashes(f, "r", ax.x0, yc, float(ax.X(bc)), yc, 0.7)
    ax.mark("r", bc, gc, 4.6)
    fml(f, "r", ax.x0 + 8, yc - 7, f"MUON, 4 GEV   γ = {gc:.1f}", P.ts)
    ax.mark("w", 0.866, 2.0, 3.2)
    fml(f, "w", float(ax.X(0.866)) - 8, float(ax.Y(2.0)) - 8, "β 0.866  γ 2", P.ts * 0.86, alpha=0.6, anchor="r")
    ym = float(ax.Y(5.0))
    fml(f, "w", ax.x0 + 14, ym, "γ = 1/√(1 − β²)", P.tl)
    fml(f, "w", ax.x0 + 14, ym + 1.5 * P.tl, f"β = {BETA:.6f}", P.ts, alpha=0.7)


def p_clock(P):
    """The light clock: seen moving, the light has further to go between two ticks, at the same speed."""
    f = P.f
    yt, yb = P.y0 + 0.10 * P.h, P.y0 + 0.62 * P.h
    q = ((P.t - T0) / BEAT) % 1.0                    # one tick per beat
    tri = 1.0 - abs(2.0 * q - 1.0)
    m = 0.045 * P.w
    xa = P.x0 + 0.11 * P.w
    f.segments("w", [xa - m, xa - m], [yt, yb], [xa + m, xa + m], [yt, yb], 0.95, width=L.LW_BOLD)
    dashes(f, "w", xa, yb, xa, yt, 0.5)
    f.dots("r", [xa], [yb + (yt - yb) * tri], 4.4, 1.8)
    f.text("w", xa + m + 8, 0.5 * (yt + yb) + 5, "L", size=P.tl, alpha=0.9)
    xl, xr = P.x0 + 0.34 * P.w, P.x0 + 0.92 * P.w
    xm = 0.5 * (xl + xr)
    f.segments("w", [xl - m, xr - m, xm - m], [yb, yb, yt], [xl + m, xr + m, xm + m], [yb, yb, yt], 0.4, width=L.LW)
    f.polyline("r", [xl, xm, xr], [yb, yt, yb], 0.9, width=L.LW)
    xc = xl + (xr - xl) * q
    f.segments("w", [xc - m, xc - m], [yt, yb], [xc + m, xc + m], [yt, yb], 0.95, width=L.LW_BOLD)
    f.dots("r", [xc], [yb + (yt - yb) * tri], 4.4, 1.8)
    arrow(f, "w", xm, yb + 16, xr, yb + 16, 0.7, head=8.0)
    arrow(f, "w", xm, yb + 16, xl, yb + 16, 0.7, head=8.0)
    fml(f, "w", xm, yb + 16 + 1.3 * P.ts, "v·∆t′", P.ts, anchor="m")
    f.text("w", xa, yb + 16 + 1.3 * P.ts, "AT REST", size=P.ts * 0.86, alpha=0.6, anchor="ms")
    y = P.y1 - 0.4 * P.tl
    xe = fml(f, "r", P.x0 + 4, y, "∆t′ = γ·∆t", P.tl * 1.15)
    if P.x1 - xe > 30 * adv(f, P.ts):
        fml(f, "w", P.x1 - 2, y, "(c∆t′/2)² = L² + (v∆t′/2)²", P.ts, alpha=0.8, anchor="r")


def p_frames(P):
    """Why the muon arrives: for the Earth its life is stretched, for the muon the atmosphere is squeezed."""
    f = P.f
    xa = P.x0 + 4.0
    lw = (P.w - 8.0) * 15000.0 / RANGE               # the distance to cross, at the scale where its reach fills the plate
    bh = min(0.11 * P.h, 30.0)
    ts = fsz(44, P.w - 8, P.ts + 1)
    rows = (("EARTH FRAME", "ATMOSPHERE 15 KM", f"IT CAN GO γβcτ = {RANGE / 1e3:.1f} KM",
             f"LIFE γτ = {GAM * TAU_MU * 1e6:.1f} µS"),
            ("MUON FRAME", f"ATMOSPHERE 15 KM / γ = {15000.0 / GAM:.0f} M", f"IT SEES βcτ = {BETA * CTAU:.0f} M GO BY",
             "LIFE τ = 2.197 µS"))
    q = ((P.t - T0) / BAR) % 1.0
    for r, (name, dist, reach, life) in enumerate(rows):
        y = P.y0 + (0.02 + 0.47 * r) * P.h + ts
        f.tag("w", xa, y, name, size=ts, pad=3)
        fml(f, "w", P.x1 - 4, y, life, ts, alpha=0.8, anchor="r")
        y += 10
        f.rect("w", xa, y, xa + lw, y + bh, 0.9)
        k = np.arange(1, 15)
        f.segments("w", xa + lw * k / 15, np.full(14, y), xa + lw * k / 15, np.full(14, y + 5.0), 0.6)
        f.dots("r", [xa + lw * q], [y], 4.0, 1.8)
        td = fsz(len(dist) + 2, lw, ts)
        fml(f, "w", xa + 8, y + 0.5 * bh + 0.42 * td, dist, td, alpha=0.9)
        y += bh + 7
        f.rect("r", xa, y, P.x1 - 4, y + bh, 1.0, width=L.LW_BOLD)
        fml(f, "r", xa + 8, y + bh + 1.3 * ts, reach, ts)


def p_minkowski(P):
    """A spacetime diagram: the axes of the moving frame close on the light cone like scissors."""
    f = P.f
    wide = P.w > 1.6 * P.h
    S = min(P.w * (0.62 if wide else 1.0), P.h)
    ox, oy = (P.x0 + 0.5 * S + 6 if wide else P.cx), P.y1 - 0.05 * S
    a, H = 0.48 * S, 0.90 * S
    arrow(f, "w", ox - a, oy, ox + a, oy, 0.8)
    arrow(f, "w", ox, oy, ox, oy - H, 0.8)
    for sg in (-1, 1):
        dashes(f, "w", ox, oy, ox + sg * a, oy - a, 0.5)
    beta = 0.60 + 0.27 * math.sin(2 * math.pi * (P.t - T0) / (2 * BAR)) * (0.55 + 0.45 * min(1.0, 1.4 * P.a["bass"]))
    gam = 1.0 / math.sqrt(1.0 - beta * beta)
    lc = min(H, a / beta)
    arrow(f, "r", ox, oy, ox + beta * lc, oy - lc, 1.0, L.LW_BOLD)                 # ct': the world line of the muon
    arrow(f, "r", ox, oy, ox + a, oy - beta * a, 0.9)                              # x'
    u0 = 0.17 * S
    x = np.linspace(-a, a, 41)
    for n in (1, 2, 3):                               # equal proper time: hyperbolae, and where the muon crosses them
        yh = np.sqrt((n * u0) ** 2 + x * x)
        m = yh <= H
        f.polyline("w", ox + x[m], oy - yh[m], 0.3)
        ex, ey = gam * beta * n * u0, gam * n * u0
        if ey <= lc:
            f.dots("r", [ox + ex], [oy - ey], 3.4, 1.6)
            xs = np.array([-a, a])
            ys = ey + beta * (xs - ex)                # its 'now': a line parallel to x'
            ok = (ys > 0) & (ys < H)
            if ok.all():
                dashes(f, "r", ox + xs[0], oy - ys[0], ox + xs[1], oy - ys[1], 0.45)
    ts = P.ts
    f.text("w", ox + a - 4, oy - 8, "x", size=ts, alpha=0.9, anchor="rs")
    f.text("w", ox + 8, oy - H + ts, "ct", size=ts, alpha=0.9)
    f.text("r", ox + a - 4, oy - beta * a - 8, "x′", size=ts, anchor="rs")
    f.text("r", ox + beta * lc + 8, oy - lc + ts, "ct′", size=ts)
    f.text("w", ox - a + 4, oy - a + ts + 6, "LIGHT", size=ts * 0.86, alpha=0.6)
    if wide:
        xx = ox + a + 30
        _lines(P, xx, P.cy - 1.6 * P.tl, (f"β = {beta:.2f}", f"γ = {gam:.2f}", "RED: THE FRAME OF THE MUON",
                                          "ITS CLOCK TICKS ON THE DOTS"), size=fsz(27, P.x1 - xx, P.tl), lead=1.6, red=(2,))
    else:
        fml(f, "w", P.x0 + 2, P.y0 + P.ts, f"β = {beta:.2f}  γ = {gam:.2f}", P.ts)


def p_survival(P):
    """How many muons are left after 15 km of air: with and without Einstein."""
    f = P.f
    ax = Ax(P, (0.0, 15.0), (1e-10, 4.0), logy=True, ml=58.0)
    ax.axes(xt=(0, 5, 10, 15), yt=(1e-10, 1e-5, 1.0), xf=lambda v: f"{v:g}",
            yf=lambda v: "1" if v == 1.0 else f"1E{int(round(math.log10(v)))}", xl="KM OF AIR", yl="N/N₀")
    d = np.linspace(0.0, 15.0, 30)
    ax.curve("w", d, np.exp(-d * 1e3 / CTAU), 0.9)
    ax.curve("r", d, np.exp(-d * 1e3 / RANGE))
    dc = ((P.t - T0) / BAR % 1.0) * 15.0
    ax.mark("r", dc, math.exp(-dc * 1e3 / RANGE), 4.4)
    ax.mark("w", dc, math.exp(-dc * 1e3 / CTAU), 3.4)
    f.text("r", ax.x1 - 4, float(ax.Y(P_EINSTEIN)) + 1.4 * P.ts, f"EINSTEIN  {P_EINSTEIN:.2f}", size=P.ts, anchor="rs")
    f.text("w", ax.x1 - 4, float(ax.Y(P_NEWTON)) - 8, "NEWTON  1.3E-10", size=P.ts, alpha=0.9, anchor="rs")
    fml(f, "w", ax.x0 + 12, float(ax.Y(1e-7)), "N = N₀·e^{−d/γβcτ}", P.ts)
    f.text("w", ax.x0 + 12, float(ax.Y(1e-7)) + 1.5 * P.ts, "MUON OF 4 GEV, NO ENERGY LOSS", size=P.ts * 0.86, alpha=0.6)


def p_triangle(P):
    """Energy, momentum and mass: a right triangle. The faster, the flatter."""
    f = P.f
    E = min(0.88 * P.h, 0.50 * P.w)
    ox, oy = P.x0 + 0.05 * P.w, P.y1 - 0.09 * P.h
    beta = 0.58 + 0.38 * math.sin(2 * math.pi * (P.t - T0) / (2 * BAR))
    gam = 1.0 / math.sqrt(1.0 - beta * beta)
    ax_, by = ox + beta * E, oy - E / gam
    f.polyline("w", [ox, ax_, ax_], [oy, oy, by], 0.9, width=L.LW_BOLD)
    seg(f, "r", ox, oy, ax_, by, 1.0, L.LW_BOLD)
    a = np.linspace(0.0, np.pi / 2, 25)
    _dcurve(f, "w", ox + E * np.cos(a), oy - E * np.sin(a), 0.4)
    f.segments("w", [ax_ - 10, ax_ - 10], [oy, oy - 10], [ax_ - 10, ax_], [oy - 10, oy - 10], 0.7)
    f.dots("r", [ax_], [by], 3.6, 1.6)
    ts = P.tl
    f.text("w", 0.5 * (ox + ax_), oy + 1.2 * ts, "pc", size=ts, alpha=0.9, anchor="ms")
    f.text("w", ax_ + 8, 0.5 * (oy + by) + 5, "mc²", size=ts, alpha=0.9)
    f.text("r", 0.5 * (ox + ax_) - 14, 0.5 * (oy + by) - 10, "E", size=ts, anchor="rs")
    x = ox + E + 3.6 * adv(f, ts) + 16
    if P.x1 - x > 200:
        _lines(P, x, P.cy - 2.0 * P.tl, ("E² = (pc)² + (mc²)²", f"β = {beta:.2f}   γ = {gam:.2f}", "mc² = 105.66 MEV",
                                         "AT 4 GEV: 97 % MOTION"), size=fsz(22, P.x1 - x, P.tl), lead=1.65)
    else:
        fml(f, "w", P.x1 - 2, P.y0 + P.ts, "E² = (pc)² + (mc²)²", P.ts, anchor="r")
        fml(f, "w", P.x1 - 2, P.y0 + 2.5 * P.ts, f"β = {beta:.2f}", P.ts, alpha=0.7, anchor="r")


# ----------------------------------------------------------------------------
# the plates: quantum mechanics
# ----------------------------------------------------------------------------

def p_packet(P):
    """A free particle as a wave packet: it moves, and it spreads."""
    f = P.f
    yc = P.y0 + 0.62 * P.h
    seg(f, "w", P.x0 + 4, yc, P.x1 - 4, yc, 0.4)
    q = ((P.t - T0) / (2 * BAR)) % 1.0
    s0 = 0.034
    sg = s0 * math.sqrt(1.0 + (3.2 * q) ** 2)
    xc = 0.10 + 0.80 * q
    x = np.linspace(0.0, 1.0, 240)
    env = np.exp(-((x - xc) ** 2) / (4 * sg * sg)) * math.sqrt(s0 / sg)
    A = 0.33 * P.h * (1.0 + 0.2 * min(1.0, P.a["kick"]))
    f.polyline("w", P.x0 + x * P.w, yc - A * env * np.cos(2 * np.pi * 30.0 * (x - 0.5 * xc)), 0.95, width=L.LW)
    m = env > 0.02
    _dcurve(f, "r", P.x0 + x[m] * P.w, yc - A * env[m], 0.9)
    sz = min(1.5 * P.tl, fsz(15, 0.6 * P.w, 60))
    fml(f, "w", P.x0 + 4, P.y0 + 1.05 * sz, "iħ ∂ψ/∂t = Ĥψ", sz)
    f.text("w", P.x0 + 4, P.y1 - 4, "RED: WHERE IT MAY BE FOUND", size=P.ts * 0.86, alpha=0.6)


def _slit_i(y):
    return np.cos(np.pi * 4.6 * y) ** 2 * np.sinc(1.25 * y) ** 2


def _slit_hits(n=420):
    """Where single particles land on the screen: drawn from the interference pattern (a fixed set)."""
    rng = np.random.default_rng(23)
    out = []
    while len(out) < n:
        y = rng.uniform(-1.0, 1.0)
        if rng.random() < _slit_i(y):
            out.append(y)
    return np.array(out), rng.random(n)


_SLIT = _slit_hits()


def p_slit(P):
    """Two slits: every particle lands in one place, all of them together draw the fringes of a wave."""
    f = P.f
    cy, hs = P.cy, 0.47 * P.h
    xs, xb, xc = P.x0 + 0.04 * P.w, P.x0 + 0.24 * P.w, P.x0 + 0.72 * P.w
    d2, g = 0.105 * P.h, 0.022 * P.h
    f.segments("w", [xb, xb, xb, xc], [cy - hs, cy - d2 + g, cy + d2 + g, cy - hs], [xb, xb, xb, xc],
               [cy - d2 - g, cy + d2 - g, cy + hs, cy + hs], 0.95, width=L.LW_BOLD)
    f.dots("r", [xs], [cy], 4.4, 1.8)
    lam = 0.05 * P.w
    ph = (P.u * 2.0) % 1.0
    a = np.radians(np.linspace(-38.0, 38.0, 13))
    for k in range(0, int((xb - xs) / lam), thin(2)):
        r = (k + ph) * lam
        if r < xb - xs - 2:
            f.polyline("w", xs + r * np.cos(a), cy + r * np.sin(a), 0.4)
    a = np.radians(np.linspace(-80.0, 80.0, 33))
    r = ((np.arange(0, int((xc - xb) / lam * 1.6) + 1, thin(2)) + ph) * lam)[:, None, None]
    px = xb + r * np.cos(a)[None, None, :] + 0.0 * np.zeros((1, 2, 1))
    py = np.array([cy - d2, cy + d2])[None, :, None] + r * np.sin(a)[None, None, :]
    ok = (px < xc - 3) & (np.abs(py - cy) < hs)
    m = ok[..., :-1] & ok[..., 1:]
    f.segments("w", px[..., :-1][m], py[..., :-1][m], px[..., 1:][m], py[..., 1:][m], 0.34)
    y = np.linspace(-1.0, 1.0, 140)
    f.polyline("w", xc + 8 + _slit_i(y) * 0.20 * P.w, cy + y * hs, 0.95, width=L.LW)
    n = int(min(len(_SLIT[0]), 14 + P.u * 240.0))
    f.dots("r", xc - 5 - 16.0 * _SLIT[1][:n], cy + _SLIT[0][:n] * hs, 1.6, 1.3)
    f.text("w", xb + 6, cy - hs + P.ts, "2 SLITS", size=P.ts * 0.86, alpha=0.6)
    f.text("r", xc - 6, cy + hs - 2, f"N {n:03d}", size=P.ts * 0.86, anchor="rs")


def p_modes(P):
    """A particle shut in a box: only whole half-waves fit, so only some energies exist."""
    f = P.f
    wide = P.w > 1.6 * P.h
    xa, xb = P.x0 + 0.10 * P.w, P.x0 + (0.62 if wide else 0.84) * P.w
    yb, yt = P.y1 - 0.03 * P.h, P.y0 + 0.02 * P.h
    f.polyline("w", [xa, xa, xb, xb], [yt, yb, yb, yt], 0.95, width=L.LW_BOLD)
    x = np.linspace(0.0, 1.0, 49)
    for n, lab in ((1, "E₁"), (2, "4E₁"), (3, "9E₁"), (4, "16E₁")):
        yl = yb - (n * n / 17.6) * (yb - yt) - 4
        dashes(f, "w", xa, yl, xb, yl, 0.35)
        psi = np.sin(n * np.pi * x) * math.cos(2 * math.pi * 0.3 * n * n * P.t)
        f.polyline("r" if n == 1 else "w", xa + x * (xb - xa), yl - 0.047 * P.h * (1.0 + 0.25 * (n == 1)) * psi, 0.95,
                   width=L.LW)
        f.text("w", xb + 8, yl + 5, f"n={n}", size=P.ts * 0.86, alpha=0.7)
        f.text("w", xa - 6, yl + 5, lab, size=P.ts * 0.86, alpha=0.7, anchor="rs")
    if wide:
        xx = xb + 8 + 5 * adv(f, P.ts)
        fml(f, "w", xx, P.cy, "E_n = n²h²/8mL²", fsz(15, P.x1 - xx, P.tl))


def p_twostate(P):
    """A system with two states (the ammonia molecule of the textbook): the probability goes back and forth."""
    f = P.f
    ax = Ax(P, (0.0, 1.5 * math.pi), (0.0, 1.14))
    names = {1: "π/4", 2: "π/2", 3: "3π/4", 4: "π", 5: "5π/4", 6: ""}
    ax.axes(xt=tuple(k * math.pi / 4 for k in range(1, 7)), yt=(0, 0.5, 1.0), xf=lambda v: names[int(round(v / (math.pi / 4)))],
            yf=lambda v: f"{v:g}", xl="t (ħ/A)", yl="P")
    t = np.linspace(0.0, 1.5 * math.pi, 91)
    ax.curve("w", t, np.cos(t) ** 2, 0.95)
    _dcurve(f, "r", ax.X(t), ax.Y(np.sin(t) ** 2), 0.95, w=L.LW_BOLD)
    tc = (((P.t - T0) / BEAT) % 4.0) / 4.0 * 1.5 * math.pi
    seg(f, "w", float(ax.X(tc)), ax.y1, float(ax.X(tc)), ax.y0, 0.35)
    ax.mark("w", tc, math.cos(tc) ** 2, 4.0)
    ax.mark("r", tc, math.sin(tc) ** 2, 4.0)
    f.text("w", float(ax.X(0.12)), float(ax.Y(1.0)) - 8, "P₁", size=P.tl, alpha=0.95)
    f.text("r", float(ax.X(math.pi / 2)) + 6, float(ax.Y(1.0)) - 8, "P₂", size=P.tl)


def p_stern(P):
    """Stern and Gerlach: a beam of spinning atoms in an uneven field splits in two, never into a smear."""
    f = P.f
    cy = P.y0 + 0.47 * P.h
    xo, xm0, xm1, xsc = P.x0 + 0.06 * P.w, P.x0 + 0.24 * P.w, P.x0 + 0.58 * P.w, P.x0 + 0.84 * P.w
    dy = 0.075 * P.h
    f.rect("w", xo - 0.035 * P.w, cy - 0.08 * P.h, xo + 0.035 * P.w, cy + 0.08 * P.h, 0.9)
    seg(f, "w", xo + 0.035 * P.w, cy, xm0, cy, 0.9)
    f.polyline("w", [xm0, xm0, 0.5 * (xm0 + xm1), xm1, xm1, xm0], [cy - 0.40 * P.h, cy - 0.22 * P.h, cy - 0.13 * P.h,
                                                                      cy - 0.22 * P.h, cy - 0.40 * P.h, cy - 0.40 * P.h], 0.8)
    f.rect("w", xm0, cy + 0.16 * P.h, xm1, cy + 0.40 * P.h, 0.8)
    f.text("w", 0.5 * (xm0 + xm1), cy - 0.27 * P.h, "N", size=P.tl, alpha=0.9, anchor="ms")
    f.text("w", 0.5 * (xm0 + xm1), cy + 0.31 * P.h, "S", size=P.tl, alpha=0.9, anchor="ms")
    slope = 2 * dy / (xm1 - xm0)

    def path(x, sg):
        u = np.clip((x - xm0) / (xm1 - xm0), 0.0, 1.0)
        return cy - sg * (dy * u * u + np.maximum(x - xm1, 0.0) * slope)

    x = np.linspace(xm0, xsc, 40)
    for sg in (1, -1):
        f.polyline("w", x, path(x, sg), 0.7)
    seg(f, "w", xsc, cy - 0.40 * P.h, xsc, cy + 0.40 * P.h, 0.95, L.LW_BOLD)
    k = np.arange(8)
    ph = P.t * 0.55 + k / 8.0
    q = ph % 1.0
    v = B.rnd(8, 95, 0) + np.floor(ph) * 0.61803              # up or down: drawn again at every passage
    sg = np.where(v - np.floor(v) > 0.5, 1.0, -1.0)
    xk = xo + q * (xsc - xo)
    f.dots("r", xk, np.where(xk < xm0, cy, path(xk, sg)), 3.2, 1.6)
    for sg, lab in ((1, "+ħ/2"), (-1, "−ħ/2")):
        ye = float(path(np.array([xsc]), sg)[0])
        f.rings("r", [xsc], [ye], [7.0], 1.0, width=L.LW)
        f.text("w", xsc + 12, ye + 5, lab, size=P.ts, alpha=0.9)
    f.text("w", xo, cy + 0.08 * P.h + 1.3 * P.ts, "OVEN", size=P.ts * 0.86, alpha=0.6, anchor="ms")
    fml(f, "w", xm1 + 8, cy - 0.30 * P.h, "∂B/∂z", P.ts, alpha=0.7)


def p_precess(P):
    """A spin in a magnetic field turns around it like a top. The muon does it at 135.5 MHz per tesla."""
    f = P.f
    wide = P.w > 1.5 * P.h
    R = 0.36 * min(P.h, P.w * (0.6 if wide else 1.0))
    cx, cy = (P.x0 + 0.06 * P.w + 1.15 * R if wide else P.cx), P.cy + 0.04 * P.h
    ring(f, "w", cx, cy, R, 0.5, n=64)
    ring(f, "w", cx, cy, R, 0.3, n=48, ry=0.3 * R)
    arrow(f, "w", cx, cy + 1.2 * R, cx, cy - 1.34 * R, 0.8)
    f.text("w", cx + 8, cy - 1.30 * R + P.ts, "B", size=P.tl, alpha=0.9)
    tilt, ph = math.radians(38.0), 2 * math.pi * 1.1 * P.t
    st, ct = R * math.sin(tilt), R * math.cos(tilt)
    dring(f, "w", cx, cy - ct, st, 0.6, n=28, ry=0.3 * st)
    a = ph - np.linspace(0.0, 1.7, 18)
    f.segments("r", cx + st * np.cos(a[:-1]), cy - ct + 0.3 * st * np.sin(a[:-1]), cx + st * np.cos(a[1:]),
               cy - ct + 0.3 * st * np.sin(a[1:]), np.linspace(1.0, 0.05, 17), width=L.LW_BOLD)
    arrow(f, "r", cx, cy, cx + st * math.cos(ph), cy - ct + 0.3 * st * math.sin(ph), 1.0, L.LW_BOLD, head=13.0)
    f.dots("r", [cx], [cy], 3.6, 1.5)
    if wide:
        x = cx + R + 30
        _lines(P, x, P.cy - 1.5 * P.tl, ("ω = γ_µ·B", "γ_µ/2π = 135.54 MHZ/T", "IN 0.1 T: 13.55 MHZ",
                                         "µSR: THE MUON AS A PROBE"), size=fsz(25, P.x1 - x, P.tl), lead=1.65)
    else:
        fml(f, "w", P.x0 + 2, P.y0 + P.ts, "ω = γ_µ·B", P.ts)


def _barrier(ka=4 * math.pi, qa=1.6):
    """Plane wave on a square barrier of width 1 (wave number ka outside, decay qa inside): the amplitudes
    (reflected, inside decaying, inside growing, transmitted), from the continuity of the wave and its slope."""
    k, q = 1j * ka, qa
    M = np.array([[1, -1, -1, 0], [-k, q, -q, 0], [0, math.exp(-q), math.exp(q), -1],
                  [0, -q * math.exp(-q), q * math.exp(q), -k]], complex)
    r, a, b, t = np.linalg.solve(M, np.array([-1, -k, 0, 0], complex))
    return ka, qa, r, a, b, t


_BAR = _barrier()


def p_tunnel(P):
    """Tunnelling: the wave dies away inside the wall, and what is left goes on behind it."""
    f = P.f
    yb = P.y0 + 0.86 * P.h
    xb0, wb = P.x0 + 0.46 * P.w, 0.13 * P.w
    V = 0.62 * P.h
    f.polyline("w", [P.x0 + 4, xb0, xb0, xb0 + wb, xb0 + wb, P.x1 - 4], [yb, yb, yb - V, yb - V, yb, yb], 0.95, width=L.LW_BOLD)
    k = np.arange(1, int((wb + V) / 10.0))
    c = k * 10.0
    xs0, xs1 = np.maximum(0.0, c - V), np.minimum(wb, c)
    m = (xs1 > xs0) & (k % thin(2) == 0)
    f.segments("w", xb0 + xs0[m], yb - (c - xs0)[m], xb0 + xs1[m], yb - (c - xs1)[m], 0.25)
    yE = yb - 0.60 * V
    dashes(f, "w", P.x0 + 4, yE, P.x1 - 4, yE, 0.4)
    ka, qa, r, a, b, t = _BAR
    x = np.linspace((P.x0 + 4 - xb0) / wb, (P.x1 - 4 - xb0) / wb, 220)
    xi = np.clip(x, 0.0, 1.0)
    psi = np.where(x < 0, np.exp(1j * ka * x) + r * np.exp(-1j * ka * x),
                   np.where(x <= 1, a * np.exp(-qa * xi) + b * np.exp(qa * xi), t * np.exp(1j * ka * (x - 1))))
    re = (psi * np.exp(-1j * 2 * math.pi * 1.5 * P.t)).real
    f.polyline("r", xb0 + x * wb, yE - 0.105 * P.h * re, 1.0, width=L.LW)
    f.text("w", xb0 + wb + 8, yb - V + P.ts, "V", size=P.tl, alpha=0.9)
    f.text("w", P.x0 + 6, yE - 0.24 * P.h, "E", size=P.tl, alpha=0.9)
    fml(f, "w", P.x1 - 4, P.y0 + P.tl, "T ≈ e^{−2κa}", P.tl, anchor="r")
    fml(f, "w", P.x1 - 4, P.y0 + 2.5 * P.tl, "κ = √(2m(V−E))/ħ", P.ts, alpha=0.8, anchor="r")


def p_hydrogen(P):
    """Where the electron of hydrogen is: not an orbit, a probability against the distance to the nucleus."""
    f = P.f
    ax = Ax(P, (0.0, 14.0), (0.0, 0.6))
    ax.axes(xt=(0, 2, 4, 6, 8, 10, 12, 14), yt=(0, 0.2, 0.4), xf=lambda v: f"{v:g}", yf=lambda v: f"{v:g}", xl="r / a₀",
            yl="r²|R|²")
    r = np.linspace(0.0, 14.0, 100)
    ax.curve("r", r, 4 * r * r * np.exp(-2 * r))
    ax.curve("w", r, r * r * (2 - r) ** 2 * np.exp(-r) / 8.0, 0.9, L.LW)
    _dcurve(f, "w", ax.X(r), ax.Y(r ** 4 * np.exp(-r) / 24.0), 0.9)
    f.text("r", float(ax.X(1.5)), float(ax.Y(0.52)), "1s", size=P.tl)
    f.text("w", float(ax.X(3.4)), float(ax.Y(0.215)), "2p", size=P.ts, alpha=0.9)
    f.text("w", float(ax.X(6.3)), float(ax.Y(0.185)), "2s", size=P.ts, alpha=0.9)
    dashes(f, "w", float(ax.X(1.0)), ax.y1, float(ax.X(1.0)), float(ax.Y(0.541)), 0.5)
    cx, cy, R = ax.x1 - 0.16 * (ax.x1 - ax.x0), ax.y0 + 0.30 * (ax.y1 - ax.y0), 0.20 * (ax.y1 - ax.y0)
    ring(f, "w", cx, cy, R, 0.6, n=40)
    hatch(f, "w", cx, cy, R, 0.35, step=6.0)
    f.dots("r", [cx], [cy], 2.6, 1.6)
    f.text("w", cx, cy + R + 1.3 * P.ts, "a₀ = 52.9 PM", size=P.ts * 0.86, alpha=0.7, anchor="ms")


_GAMMAS = (("γ⁰", ("1 0 0 0", "0 1 0 0", "0 0 −1 0", "0 0 0 −1")), ("γ¹", ("0 0 0 1", "0 0 1 0", "0 −1 0 0", "−1 0 0 0")),
           ("γ²", ("0 0 0 −i", "0 0 i 0", "0 i 0 0", "−i 0 0 0")), ("γ³", ("0 0 1 0", "0 0 0 −1", "−1 0 0 0", "0 1 0 0")))


def p_dirac(P):
    """The Dirac equation: quantum mechanics and relativity in one line. Spin, g = 2 and antimatter follow."""
    f = P.f
    wide = P.w > 1.5 * P.h and P.w > 600
    wt = P.w * (0.56 if wide else 1.0)
    sz = fsz(20, wt - 8, 0.34 * P.h)
    fml(f, "w", P.x0 + 4, P.y0 + 0.16 * P.h + 0.8 * sz, "(iħγ^µ∂_µ − mc)ψ = 0", sz)
    ts = fsz(34, wt - 8, P.tl)
    _lines(P, P.x0 + 4, P.y0 + 0.16 * P.h + 0.8 * sz + 2.2 * ts, ("DIRAC, 1928", "SPIN ½  //  g = 2  //  ANTIMATTER",
                                                                  "ALL THREE FOLLOW FROM THIS LINE"), size=ts, lead=1.6)
    if wide:
        name, rows = _GAMMAS[(P.a["n_k"] // 2) % 4]
        c = min(0.19 * P.h, 0.066 * P.w)
        x0, y0 = P.x1 - 4.3 * c, P.cy - 2.0 * c
        fml(f, "r", x0 - 0.5 * c, P.cy + 0.3 * c, name + " =", 0.5 * c, anchor="r")
        for sg, xx in ((1, x0), (-1, x0 + 4 * c)):
            f.polyline("w", [xx + sg * 10, xx, xx, xx + sg * 10], [y0, y0, y0 + 4 * c, y0 + 4 * c], 0.9, width=L.LW)
        for j, row in enumerate(rows):
            for i, v in enumerate(row.split()):
                f.text("w" if v == "0" else "r", x0 + (i + 0.5) * c, y0 + (j + 0.66) * c, v, size=0.42 * c,
                       alpha=0.45 if v == "0" else 1.0, anchor="ms")


def p_uncert(P):
    """Heisenberg: the narrower the position, the wider the momentum. Their product has a floor."""
    f = P.f
    sx = 0.16 * 2.0 ** (1.3 * math.sin(2 * math.pi * (P.t - T0) / (2 * BAR)))
    yb, A = P.y0 + 0.66 * P.h, 0.56 * P.h
    x = np.linspace(-1.0, 1.0, 70)
    for j, (sg, lay, lab) in enumerate(((sx, "r", "∆x"), (0.0256 / sx, "w", "∆p"))):
        xm, hw = P.x0 + (0.25 + 0.5 * j) * P.w, 0.22 * P.w
        seg(f, "w", xm - hw, yb, xm + hw, yb, 0.6)
        f.polyline(lay, xm + x * hw, yb - A * np.exp(-x * x / (2 * sg * sg)) * min(1.0, math.sqrt(0.16 / sg)), 0.95,
                   width=L.LW_BOLD)
        d = min(sg * 1.18 * hw, hw)
        arrow(f, "w", xm, yb + 14, xm + d, yb + 14, 0.7, head=7.0)
        arrow(f, "w", xm, yb + 14, xm - d, yb + 14, 0.7, head=7.0)
        f.text(lay, xm, yb + 14 + 1.4 * P.ts, lab, size=P.tl, anchor="ms")
    fml(f, "w", P.cx, P.y1 - 0.2 * P.tl, "∆x·∆p ≥ ħ/2", min(1.3 * P.tl, fsz(12, 0.5 * P.w, 40)), anchor="m")
    f.text("w", P.x1 - 2, P.y0 + P.ts, "ħ = 1.054 571 817E-34 J·S", size=fsz(26, 0.6 * P.w, P.ts), alpha=0.6, anchor="rs")


# ----------------------------------------------------------------------------
# the plates: general relativity
# ----------------------------------------------------------------------------

def p_field(P):
    """Einstein's field equations, over a grid that a mass has bent."""
    f = P.f
    sz = fsz(25, P.w - 8, 0.24 * P.h)
    fml(f, "w", P.cx, P.y0 + 0.08 * P.h + 0.8 * sz, "G_{µν} + Λg_{µν} = (8πG/c⁴)·T_{µν}", sz, anchor="m")
    ya, yb = P.y0 + 0.08 * P.h + 1.5 * sz, P.y1 - 1.6 * P.ts
    ym, hh = 0.5 * (ya + yb), 0.5 * (yb - ya)
    x = np.linspace(-1.0, 1.0, 50)
    pull = 0.34 + 0.10 * min(1.0, P.a["kick"])
    for k in range(7):
        yk = -1.0 + k / 3.0
        d = np.hypot(x * 2.6, yk * 1.6)
        f.polyline("w", P.cx + x * 0.48 * P.w, ym + hh * (yk - pull * yk * np.exp(-d * d)), 0.5)
    for k in range(13):
        xk = -1.0 + k / 6.0
        y = np.linspace(-1.0, 1.0, 21)
        d = np.hypot(xk * 2.6, y * 1.6)
        f.polyline("w", P.cx + (xk - pull * xk * np.exp(-d * d)) * 0.48 * P.w, ym + hh * y, 0.3)
    f.dots("r", [P.cx], [ym], 6.0, 1.7)
    f.text("w", P.x0 + 4, P.y1 - 3, "8πG/c⁴ = 2.077E-43 S²/M/KG", size=fsz(27, 0.55 * P.w, P.ts), alpha=0.7)
    f.text("w", P.x1 - 4, P.y1 - 3, "1915", size=P.ts, alpha=0.7, anchor="rs")


def p_schwarz(P):
    """The field of one mass (Schwarzschild), and the radius at which it would close on itself."""
    f = P.f
    sz = fsz(49, P.w - 8, 0.17 * P.h)
    fml(f, "w", P.cx, P.y0 + 0.05 * P.h + 0.9 * sz, "ds² = −(1 − r_s/r)c²dt² + (1 − r_s/r)^{−1}dr² + r²dΩ²", sz, anchor="m")
    y = P.y0 + 0.05 * P.h + 0.9 * sz + 0.14 * P.h
    R = min(0.155 * (P.y1 - y), 0.07 * P.w)
    cx, cy = P.x1 - 3.3 * R, 0.5 * (y + P.y1)
    ring(f, "r", cx, cy, R, 1.0, L.LW_BOLD, n=40)
    hatch(f, "r", cx, cy, R, 0.5, step=6.0)
    dring(f, "w", cx, cy, 1.5 * R, 0.7, n=28)
    dring(f, "w", cx, cy, 3.0 * R, 0.45, n=40)
    f.text("w", cx + 3.0 * R * 0.72, cy - 3.0 * R * 0.72 - 4, "3", size=P.ts * 0.86, alpha=0.6)
    f.text("w", cx + 1.5 * R * 0.72 + 2, cy - 1.5 * R * 0.72 - 2, "1.5", size=P.ts * 0.86, alpha=0.6)
    ts = fsz(22, cx - 3.3 * R - P.x0 - 12, P.tl)
    _lines(P, P.x0 + 4, 0.5 * (y + P.y1) - 1.9 * ts,
           ("r_s = 2GM/c²", "THE SUN    2.953 KM", "THE EARTH  8.87 MM", "YOU        1E-25 M"), size=ts, lead=1.6, red=(0,))


_FLAMM_R = np.array([1.0, 1.25, 1.6, 2.1, 2.8, 3.7, 4.9, 6.4])


def p_flamm(P):
    """Space around a mass, drawn as a surface (Flamm's paraboloid): a circle is longer round than across."""
    f = P.f
    rr = _FLAMM_R
    dep = 2.0 * math.sqrt(rr[-1] - 1.0) - 2.0 * np.sqrt(rr - 1.0)           # depth below the rim
    sc = min(0.47 * P.w / rr[-1], 0.92 * P.h / (2 * 0.40 * rr[-1] + 0.55 * dep[0]))
    cx, yr = P.cx, P.cy - 0.5 * sc * (0.55 * dep[0] + 0.40) + 0.5 * sc * 0.40 * rr[-1] - 0.03 * P.h
    rot = 0.35 * P.t
    a = np.linspace(0.0, 2 * np.pi, 49)
    wave = P.a["sk"] * 16.0                          # every kick sends a brighter ring outwards
    for k, r in enumerate(rr):
        i = 0.36 + 0.6 * math.exp(-((k - wave) ** 2) / 1.6)
        f.polyline("r" if k == 0 else "w", cx + sc * r * np.cos(a), yr + sc * (0.40 * r * np.sin(a) + 0.55 * dep[k]), i,
                   width=L.LW_BOLD if k == 0 else L.LW)
    rd = np.linspace(1.0, rr[-1], 12)
    dd = 2.0 * math.sqrt(rr[-1] - 1.0) - 2.0 * np.sqrt(rd - 1.0)
    for j in range(16):
        ph = rot + j * (2 * np.pi / 16)
        f.polyline("w", cx + sc * rd * math.cos(ph), yr + sc * (0.40 * rd * math.sin(ph) + 0.55 * dd), 0.3)
    po = 2.4 * P.t
    d3 = 2.0 * math.sqrt(rr[-1] - 1.0) - 2.0 * math.sqrt(2.0)
    tr = po - np.linspace(0.0, 1.3, 14)
    ox, oy = cx + sc * 3.0 * np.cos(tr), yr + sc * (1.2 * np.sin(tr) + 0.55 * d3)       # something in orbit at 3 r_s
    f.segments("r", ox[:-1], oy[:-1], ox[1:], oy[1:], np.linspace(0.9, 0.05, 13), width=L.LW)
    f.dots("r", [cx + sc * 3.0 * math.cos(po)], [yr + sc * (1.2 * math.sin(po) + 0.55 * d3)], 4.0, 1.8)
    fml(f, "w", P.x0 + 2, P.y1 - 4, "z = 2√(r_s(r − r_s))", fsz(22, 0.5 * P.w, P.ts), alpha=0.8)
    fml(f, "r", cx + sc * 1.3, yr + sc * (0.55 * dep[0] + 0.40) + P.ts, "r_s", P.ts)


def p_gps(P):
    """Gravity slows clocks: the one in orbit runs ahead of the one on the ground, every day, by a known amount."""
    f = P.f
    xa, xb = P.x0 + 4.0, P.x1 - 4.0
    ts = fsz(40, P.w - 8, P.ts + 1)
    for j, (name, per, lay) in enumerate((("CLOCK IN ORBIT // 20 200 KM", 20.0, "w"), ("CLOCK ON THE GROUND", 23.0, "r"))):
        y = P.y0 + (0.06 + 0.24 * j) * P.h + ts
        f.text(lay, xa, y, name, size=ts, alpha=0.9)
        y += 0.11 * P.h + 6
        seg(f, lay, xa, y, xb, y, 0.7)
        n = int((xb - xa) / per) + 2
        x = xb - ((46.0 * P.t) % per) - np.arange(n) * per
        x = x[x > xa]
        f.segments(lay, x, np.full(len(x), y), x, np.full(len(x), y - 0.10 * P.h), 0.95, width=L.LW)
    y = P.y0 + 0.60 * P.h
    big = min(0.20 * P.h, fsz(17, P.w - 8, 60))
    f.text("r", xa, y + big, "+38.6 µS / DAY", size=big)
    _lines(P, xa, y + big + 1.7 * ts, ("GRAVITY +45.8   SPEED −7.2   (DRAWN × 2E8)", "UNCORRECTED: 11.6 KM OF ERROR A DAY"),
           size=ts, lead=1.55)


def p_bending(P):
    """Light passing the Sun is bent: the star is seen where it is not (1919)."""
    f = P.f
    Rs = 0.15 * min(1.2 * P.h, 0.6 * P.w)
    sx, sy = P.x0 + 0.50 * P.w, P.y0 + 0.60 * P.h
    ring(f, "w", sx, sy, Rs, 0.9, n=48)
    hatch(f, "w", sx, sy, Rs, 0.35, step=8.0)
    cx_, cy_ = sx, sy - Rs - 8.0
    S = (P.x0 + 0.04 * P.w, cy_ - 0.05 * P.h)
    d1 = np.array([cx_ - S[0], cy_ - S[1]])
    d1 /= np.linalg.norm(d1)
    dl = math.radians(10.0)
    d2 = np.array([d1[0] * math.cos(dl) - d1[1] * math.sin(dl), d1[0] * math.sin(dl) + d1[1] * math.cos(dl)])
    E_ = np.array([cx_, cy_]) + d2 * (P.x0 + 0.94 * P.w - cx_) / d2[0]
    s = np.linspace(0.0, 1.0, 15)
    pa, pb = np.array([cx_, cy_]) - d1 * 0.13 * P.w, np.array([cx_, cy_]) + d2 * 0.13 * P.w
    bz = ((1 - s) ** 2)[:, None] * pa + (2 * s * (1 - s))[:, None] * np.array([cx_, cy_]) + (s * s)[:, None] * pb
    xs, ys = np.r_[S[0], bz[:, 0], E_[0]], np.r_[S[1], bz[:, 1], E_[1]]
    f.polyline("r", xs, ys, 1.0, width=L.LW)
    A = E_ - d2 * (E_[0] - S[0]) / d2[0]             # where the star seems to be
    dashes(f, "w", E_[0], E_[1], A[0], A[1], 0.6)
    f.crosses("w", [S[0]], [S[1]], 7.0, 1.0, width=L.LW)
    f.rings("w", [A[0]], [A[1]], [6.0], 0.9, width=L.LW)
    f.dots("w", [E_[0]], [E_[1]], 4.0, 1.4)
    cum = np.r_[0.0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
    q = ((P.t - T0) / BAR % 1.0) * cum[-1]
    f.dots("r", [float(np.interp(q, cum, xs))], [float(np.interp(q, cum, ys))], 4.0, 1.8)
    ts = P.ts * 0.9
    f.text("w", S[0], S[1] + 1.5 * ts, "STAR", size=ts, alpha=0.7)
    f.text("w", A[0], A[1] - 10, "SEEN HERE", size=ts, alpha=0.7)
    f.text("w", E_[0], E_[1] + 1.5 * ts, "EARTH", size=ts, alpha=0.7, anchor="rs")
    f.text("w", sx, sy + Rs + 1.4 * ts, "SUN", size=ts, alpha=0.7, anchor="ms")
    fml(f, "w", P.x0 + 4, P.y1 - 4, "α = 4GM/c²b = 1.75″", P.tl)
    f.text("w", P.x1 - 4, P.y1 - 4, "DRAWN × 20 000", size=ts, alpha=0.6, anchor="rs")


def p_lens(P):
    """A mass between a source and the eye: two bent paths, two images; a ring when the three are in line."""
    f = P.f
    cy = P.y0 + 0.52 * P.h
    xS, xL, xO = P.x0 + 0.06 * P.w, P.x0 + 0.46 * P.w, P.x0 + 0.82 * P.w
    tE = 0.10 * P.h
    beta = 0.9 * math.sin(2 * math.pi * (P.t - T0) / (2 * BAR))
    mag = (xO - xS) / (xO - xL)
    dashes(f, "w", xS - 4, cy, xO + 4, cy, 0.35)
    ys = cy - beta * tE * mag
    for th in (0.5 * (beta + math.sqrt(beta * beta + 4)), 0.5 * (beta - math.sqrt(beta * beta + 4))):
        yl = cy - th * tE
        f.polyline("w", [xS, xL, xO], [ys, yl, cy], 0.85, width=L.LW)
        ya = cy - th * tE * mag
        dashes(f, "r", xL, yl, xS, ya, 0.6)
        f.rings("r", [xS], [ya], [6.0], 1.0, width=L.LW)
    f.dots("w", [xS], [ys], 4.0, 1.5)
    ring(f, "r", xL, cy, 0.035 * P.h + 3, 1.0, n=24)
    hatch(f, "r", xL, cy, 0.035 * P.h + 3, 0.6, step=4.0)
    f.dots("w", [xO], [cy], 4.4, 1.5)
    ri = min(0.13 * P.h, 0.06 * P.w)
    ix, iy = P.x1 - ri * 1.9, P.y0 + 0.24 * P.h      # the sky as the eye sees it
    dring(f, "w", ix, iy, ri, 0.6, n=24)
    rt = math.sqrt(beta * beta + 4)
    f.dots("r", [ix, ix], [iy - 0.5 * (beta + rt) * ri, iy - 0.5 * (beta - rt) * ri], 3.4, 1.7)
    f.crosses("w", [ix], [iy - beta * ri], 5.0, 0.9)
    ts = P.ts * 0.86
    f.text("w", xS, cy + 0.42 * P.h, "SOURCE", size=ts, alpha=0.6, anchor="ms")
    f.text("w", xL, cy + 0.42 * P.h, "MASS", size=ts, alpha=0.6, anchor="ms")
    f.text("w", xO, cy + 0.42 * P.h, "EYE", size=ts, alpha=0.6, anchor="ms")
    f.text("w", ix, iy + ri * 2.1 + ts, "SEEN", size=ts, alpha=0.6, anchor="ms")
    fml(f, "w", P.x0 + 4, P.y0 + P.ts, "θ_E = √(4GM/c² · D_{LS}/D_L D_S)", fsz(32, 0.74 * P.w, P.ts), alpha=0.85)


def _chirp():
    """The wave of two black holes merging, as LIGO saw it on 14 September 2015: 35 Hz to 250 Hz in 0.2 s
    (frequency rising as the -3/8 power of the time left), then the ring-down."""
    t = np.linspace(-0.20, 0.035, 560)
    tau = np.maximum(-t, 1e-4)
    fr = np.minimum(35.0 * (0.2 / tau) ** 0.375, 250.0)
    ph = -2 * np.pi * 35.0 * 0.2 ** 0.375 * 1.6 * tau ** 0.625
    h = np.where(t < 0, (fr / 250.0) ** (2.0 / 3.0) * np.cos(ph),
                 np.exp(-np.maximum(t, 0.0) / 0.004) * np.cos(2 * np.pi * 250.0 * t))
    return t, h, fr


_CHIRP = _chirp()


def p_chirp(P):
    """GW150914: the first gravitational wave ever measured."""
    f = P.f
    wide = P.w > 1.7 * P.h
    T, Hh, Fr = _CHIRP
    xa, xb = P.x0 + 6.0, P.x0 + (0.74 if wide else 1.0) * P.w - 8.0
    yc, A = P.y0 + 0.36 * P.h, 0.30 * P.h
    X = xa + (T - T[0]) / (T[-1] - T[0]) * (xb - xa)
    f.polyline("w", X, yc - A * Hh, 0.95, width=L.LW)
    ya = yc + A + 8
    seg(f, "w", xa, ya, xb, ya, 0.6)
    for tv, lab in ((-0.2, "−0.2 S"), (-0.1, "−0.1"), (0.0, "0")):
        xx = xa + (tv - T[0]) / (T[-1] - T[0]) * (xb - xa)
        seg(f, "w", xx, ya, xx, ya + 6, 0.7)
        f.text("w", xx, ya + 8 + 0.8 * P.ts, lab, size=P.ts * 0.86, alpha=0.6, anchor="ls" if tv < -0.15 else "ms")
    q = ((P.t - T0) / BAR % 1.0) * (len(T) - 1)
    xc, hc, fc = float(np.interp(q, np.arange(len(T)), X)), float(np.interp(q, np.arange(len(T)), Hh)), float(
        np.interp(q, np.arange(len(T)), Fr))
    seg(f, "r", xc, yc - A - 4, xc, ya, 0.6)
    f.dots("r", [xc], [yc - A * hc], 4.2, 1.8)
    f.text("r", xa + 2, yc - A + 0.2 * P.ts, f"{fc:03.0f} HZ", size=P.tl)
    if wide:
        R0 = min(0.105 * P.w, 0.26 * P.h)
        rx_, ry_ = P.x1 - 1.45 * R0, yc
        a = np.arange(12) * (2 * np.pi / 12)
        f.dots("w", rx_ + R0 * (1 + 0.42 * hc) * np.cos(a), ry_ + R0 * (1 - 0.42 * hc) * np.sin(a), 3.0, 1.4)
        ring(f, "w", rx_, ry_, R0 * (1 + 0.42 * hc), 0.35, n=40, ry=R0 * (1 - 0.42 * hc))
    ts = fsz(47, P.w - 8, P.ts + 1)
    _lines(P, P.x0 + 4, P.y1 - 1.75 * ts,
           ("GW150914   36 + 29 → 62 ☉   3.0 ☉ RADIATED", "STRAIN 1.0E-21   FROM 1.3 BILLION LIGHT-YEARS"), size=ts, lead=1.5)


def p_perihelion(P):
    """An orbit in curved space does not close: its nearest point moves round (Mercury: 43 seconds of arc a century)."""
    f = P.f
    wide = P.w > 1.5 * P.h
    S = 0.47 * min(P.h, P.w * (0.62 if wide else 1.0))
    fx, fy = (P.x0 + S + 6 if wide else P.cx), P.cy
    e, dl = 0.55, 0.06
    p = S * (1 - e)
    n_t = (P.t - T_DRUMS) / BAR
    M = 2 * math.pi * (n_t % 1.0)
    E_ = M
    for _ in range(6):                                # Kepler's equation: the planet is quick near the Sun
        E_ -= (E_ - e * math.sin(E_) - M) / (1 - e * math.cos(E_))
    th = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E_ / 2), math.sqrt(1 - e) * math.cos(E_ / 2)) % (2 * math.pi)
    Th = (2 * math.pi * math.floor(n_t) + th) / (1 - dl)
    tt = Th - np.linspace(0.0, 4.2 * 2 * np.pi, 260)
    r = p / (1 + e * np.cos((1 - dl) * tt))
    x, y = fx + r * np.cos(tt), fy - r * np.sin(tt)
    f.segments("w", x[:-1], y[:-1], x[1:], y[1:], np.linspace(0.95, 0.08, 259), width=L.LW)
    f.dots("r", [x[0]], [y[0]], 4.4, 1.8)
    ring(f, "w", fx, fy, 7.0, 0.9, n=16)
    hatch(f, "w", fx, fy, 7.0, 0.6, step=3.5)
    k = math.floor((1 - dl) * Th / (2 * math.pi))
    ap = 2 * math.pi * k / (1 - dl)                   # the last passage nearest the Sun
    dashes(f, "r", fx, fy, fx + p / (1 + e) * math.cos(ap), fy - p / (1 + e) * math.sin(ap), 0.8)
    if wide:
        xx = fx + S + 26
        _lines(P, xx, P.cy - 1.5 * P.tl, ("MERCURY", "42.98″ PER CENTURY", "MORE THAN NEWTON GIVES", "DRAWN × 800 000"),
               size=fsz(22, P.x1 - xx, P.tl), lead=1.65, red=(1,))
    else:
        f.text("r", P.x0 + 2, P.y0 + P.ts, "42.98″ / CENTURY", size=P.ts)


def p_cones(P):
    """Light cones near a black hole (ingoing Eddington-Finkelstein time): they tip over; inside, every future
    points to the centre."""
    f = P.f
    xa, xb = P.x0 + 0.07 * P.w, P.x1 - 0.05 * P.w
    X = lambda r: xa + r / 3.3 * (xb - xa)
    yb, hc = P.y0 + 0.74 * P.h, 0.46 * P.h
    w0 = 0.36 * (xb - xa) / 3.3
    arrow(f, "w", xa, yb + 16, xb, yb + 16, 0.7)
    z = np.arange(0, 13)
    f.polyline("w", xa + 5.0 * (z % 2), yb + 10 - z * (hc + 30) / 12.0, 0.9)             # r = 0
    dashes(f, "r", X(1.0), yb + 16, X(1.0), yb - hc - 22, 0.9, w=L.LW)
    a = np.linspace(0.0, 2 * np.pi, 25)
    for r in (0.55, 1.0, 1.55, 2.2, 2.9):
        v = (r - 1.0) / (r + 1.0)                    # outgoing light; ingoing light always goes at -1
        lay = "r" if r <= 1.0 else "w"
        x0 = X(r)
        f.segments(lay, [x0, x0], [yb, yb], [x0 + v * w0, x0 - w0], [yb - hc, yb - hc], 0.95, width=L.LW)
        f.polyline(lay, x0 + 0.5 * (v - 1) * w0 + 0.5 * (v + 1) * w0 * np.cos(a), yb - hc + 5.0 * np.sin(a), 0.7)
        f.dots(lay, [x0], [yb], 2.6, 1.4)
    ts = P.ts * 0.9
    f.text("w", xa, yb + 16 + 1.4 * ts, "r = 0", size=ts, alpha=0.7)
    fml(f, "r", X(1.0), yb + 16 + 1.4 * ts, "r_s", ts, anchor="m")
    f.text("w", xb, yb + 16 + 1.4 * ts, "r", size=ts, alpha=0.7, anchor="rs")
    f.text("w", xa + 14, yb - hc - 12, "TIME ↑", size=ts, alpha=0.6)
    fml(f, "w", P.x1 - 2, P.y1 - 3, "dr/dt = (r − r_s)/(r + r_s)", fsz(27, 0.62 * P.w, P.ts), alpha=0.8, anchor="r")


_RAYS = [(b, _null_geodesic(b)[::3]) for b in (2.2, 2.5, 2.62, 3.0, 4.0, 5.5)]


def p_hole(P):
    """Paths of light around a black hole (exact, Schwarzschild): closer than 2.6 radii, it does not come back."""
    f = P.f
    sc = min(P.h / 12.4, P.w / 17.0)
    cx, cy = P.x0 + 0.56 * P.w, P.cy
    ring(f, "r", cx, cy, sc, 1.0, L.LW_BOLD, n=40)
    hatch(f, "r", cx, cy, sc, 0.5, step=5.0)
    dring(f, "w", cx, cy, 1.5 * sc, 0.8, n=26)
    dring(f, "w", cx, cy, 2.598 * sc, 0.4, n=44)
    q = (P.t - T0) / BAR
    for j, (b, pts) in enumerate(_RAYS):
        for sg in ((1,) if j % 2 else (1, -1)):
            x, y = cx + sc * pts[:, 0], cy - sg * sc * pts[:, 1]
            m = (x >= P.x0) & (x <= P.x1) & (y >= P.y0) & (y <= P.y1)
            if m.sum() < 3:
                continue
            i0 = int(np.argmax(m))
            i1 = i0 + int(np.argmin(m[i0:])) if not m[i0:].all() else len(m)
            f.polyline("w", x[i0:i1], y[i0:i1], 0.75 if b > 2.598 else 0.5)
            u, n = ((q * 0.5 + 0.17 * j) % 1.0) * (i1 - i0 - 1), np.arange(i1 - i0)
            f.dots("r", [float(np.interp(u, n, x[i0:i1]))], [float(np.interp(u, n, y[i0:i1]))], 3.0, 1.7)
    ts = P.ts * 0.9
    fml(f, "r", cx + sc * 1.05, cy + sc * 1.2 + ts, "r_s", ts)
    f.text("w", cx + 1.5 * sc * 0.75, cy - 1.5 * sc * 0.75 - 3, "1.5", size=ts * 0.9, alpha=0.7)
    f.text("w", cx + 2.6 * sc * 0.75, cy - 2.6 * sc * 0.75 - 3, "2.6", size=ts * 0.9, alpha=0.7)
    fml(f, "w", P.x0 + 2, P.y1 - 3, "M87*: 6.5E9 ☉   SHADOW 42 µAS", fsz(31, 0.7 * P.w, P.ts), alpha=0.8)


_CONST = ("c    299 792 458 M/S", "h    6.626 070 15E-34 J·S", "ħ    1.054 571 817E-34 J·S", "G    6.674 30E-11 M³/KG/S²",
          "e    1.602 176 634E-19 C", "k    1.380 649E-23 J/K", "α    1/137.035 999", "m_e  0.510 998 950 MEV",
          "m_µ  105.658 375 5 MEV", "m_p  938.272 088 MEV", "τ_µ  2.196 981 1 µS", "a_µ  0.001 165 920 715",
          "G_F  1.166 378 8E-5 /GEV²", "N_A  6.022 140 76E23 /MOL")


def p_const(P):
    """The constants everything here is made of. The red line moves on every onset of the music."""
    f = P.f
    ts = fsz(28, P.w - 8, P.tl)
    n = int(max(3, min(len(_CONST), P.h // (1.5 * ts))))
    hot = P.a["n_on"] % len(_CONST)
    first = (hot // n) * n
    for k in range(n):
        j = (first + k) % len(_CONST)
        fml(f, "r" if j == hot else "w", P.x0 + 4, P.y0 + (k + 0.9) * (P.h / n), _CONST[j], ts, alpha=1.0 if j == hot else 0.7)


def chrome(f, rect, code, title, cap, a=None, key=0):
    """What every plate has: a thick rule, its name in an inverted tag, its number, two corner marks, one line
    of caption - and the music: the rule swells on every kick, a small meter beside the number shows six bands
    of the spectrum (each plate its own). Returns the rect left for the drawing."""
    x0, y0, x1, y1 = rect
    f.rects("w", x0, y0, x1, y0 + 4.0 + (5.0 * min(1.0, a["kick"]) if a else 0.0), 0.95)
    f.tag("w", x0 + 5.0, y0 + 27.0, title, size=L.T_SMALL, pad=4)
    f.text("w", x1 - 3.0, y0 + 25.0, code, size=L.T_MICRO, alpha=0.6, anchor="rs")
    if a is not None:
        xm = x1 - 56.0 - 6 * 7.0
        v = np.clip(a["spec"][(key * 5 + np.arange(6) * 7) % len(a["spec"])], 0.06, 1.0) ** 1.15
        f.rects("w", xm + np.arange(6) * 7.0, y0 + 26.0 - 15.0 * v, xm + np.arange(6) * 7.0 + 4.0, y0 + 26.0, 0.8)
    f.segments("w", [x0, x0, x1, x1], [y1, y1, y1, y1], [x0 + 12, x0, x1 - 12, x1], [y1, y1 - 12, y1, y1 - 12], E.wl(0.7),
               width=E.ww(1.0))
    if cap:
        f.text("w", x0 + 16.0, y1 - 4.0, cap, size=fsz(len(cap), x1 - x0 - 32.0, L.T_MICRO), alpha=0.6)
    return (x0 + 8.0, y0 + 44.0, x1 - 8.0, y1 - (26.0 if cap else 8.0))


# (function, number, name, caption, the cell sizes it can be shown in: X L M S, least width)
PLATES = (
    (p_identity, "MU-01", "THE MUON", "FOUND IN COSMIC RAYS // ANDERSON, NEDDERMEYER 1936", "LMS", 300),
    (p_model, "MU-02", "STANDARD MODEL", "12 MATTER PARTICLES // 4 FORCE CARRIERS // THE HIGGS", "XLMS", 300),
    (p_decay, "MU-03", "MUON DECAY", "WEAK FORCE // THROUGH A VIRTUAL W BOSON", "LMS", 300),
    (p_pion, "MU-04", "BIRTH // PION DECAY", "15 KM UP: WHERE COSMIC MUONS ARE BORN", "LMS", 300),
    (p_lifetime, "MU-05", "DECAY LAW", "MEAN LIFE 2.197 µS // HALF-LIFE 1.523 µS", "XLMS", 300),
    (p_michel, "MU-06", "MICHEL SPECTRUM", "ENERGY OF THE ELECTRON FROM A MUON AT REST", "LMS", 300),
    (p_asym, "MU-07", "DECAY ASYMMETRY", "PARITY IS NOT CONSERVED // 1957", "LMS", 300),
    (p_ring, "MU-08", "g−2 STORAGE RING", "FERMILAB // THE SPIN GAINS 12.3° A TURN", "XLMS", 300),
    (p_wiggle, "MU-09", "g−2 WIGGLE", "POSITRONS AGAINST TIME // WORLD AVERAGE 2025", "LMS", 300),
    (p_schwinger, "MU-10", "THE ANOMALY", "THE VACUUM IS NOT EMPTY", "LM", 520),
    (p_zenith, "MU-11", "SEA LEVEL // ZENITH", "MOST COME FROM STRAIGHT ABOVE", "LMS", 300),
    (p_spectrum, "MU-12", "SEA LEVEL // MOMENTUM", "VERTICAL INTENSITY // BUGAEV, REYNA", "LMS", 300),
    (p_bethe, "MU-13", "ENERGY LOSS", "BETHE // dE/dx OF A MUON IN AIR", "LMS", 300),
    (p_cherenkov, "MU-14", "CHERENKOV LIGHT", "FASTER THAN LIGHT IN WATER", "LMS", 300),
    (p_muonic, "MU-15", "MUONIC HYDROGEN", "THE MUON MEASURES THE PROTON", "LMS", 300),
    (p_muonium, "MU-16", "MUONIUM // BREIT-RABI", "µ+ AND e−: AN ATOM WITHOUT A NUCLEUS", "LMS", 300),
    (p_pyramid, "MU-17", "MUOGRAPHY // KHUFU", "SCANPYRAMIDS, NATURE 2017 // SECTION SKETCHED", "XLMS", 300),
    (p_scatter, "MU-18", "MULTIPLE SCATTERING", "HIGHLAND // HOW MUONS SCAN CARGO", "LMS", 300),
    (p_moon, "MU-19", "SHADOW OF THE MOON", "A HOLE IN THE COSMIC RAYS", "LMS", 300),
    (p_tree, "MU-20", "AIR SHOWER", "ONE PROTON IN // MUONS OUT", "XLMS", 300),
    (p_crspec, "MU-21", "COSMIC-RAY SPECTRUM", "THE PRIMARIES // 12 DECADES OF ENERGY", "LMS", 300),
    (p_cms, "MU-22", "CMS // MUON TRACK", "CERN // THE MUON GOES THROUGH EVERYTHING", "XLMS", 300),
    (p_nuosc, "MU-23", "NEUTRINO OSCILLATION", "SUPER-KAMIOKANDE 1998 // THE NEUTRINO HAS A MASS", "LMS", 300),
    (p_gamma, "SR-01", "LORENTZ FACTOR", "HOW MUCH TIME STRETCHES, AGAINST SPEED", "LMS", 300),
    (p_clock, "SR-02", "LIGHT CLOCK", "MOVING CLOCKS RUN SLOW", "LMS", 300),
    (p_frames, "SR-03", "TWO FRAMES", "TIME STRETCHED OR SPACE SQUEEZED: BOTH AGREE", "LM", 400),
    (p_minkowski, "SR-04", "SPACETIME", "MINKOWSKI DIAGRAM // 1908", "XLMS", 300),
    (p_survival, "SR-05", "15 KM OF AIR", "WITHOUT EINSTEIN: 1 MUON IN 7.8 BILLION ARRIVES", "LMS", 300),
    (p_triangle, "SR-06", "ENERGY // MOMENTUM", "MASS IS ENERGY AT REST", "LMS", 300),
    (p_packet, "QM-01", "WAVE PACKET", "SCHRÖDINGER 1926 // A WAVE OF PROBABILITY", "LMS", 300),
    (p_slit, "QM-02", "TWO SLITS", "ONE PARTICLE AT A TIME: THE FRINGES STILL COME", "XLMS", 300),
    (p_modes, "QM-03", "PARTICLE IN A BOX", "CONFINEMENT QUANTISES ENERGY", "LMS", 300),
    (p_twostate, "QM-04", "TWO-STATE SYSTEM", "P₁ = cos²(At/ħ)  P₂ = sin²(At/ħ)  //  AMMONIA: 24 GHZ", "LMS", 300),
    (p_stern, "QM-05", "STERN-GERLACH", "SPIN IS QUANTISED // 1922", "LMS", 300),
    (p_precess, "QM-06", "SPIN PRECESSION", "A SPIN IN A FIELD TURNS LIKE A TOP", "XLMS", 300),
    (p_tunnel, "QM-07", "TUNNELLING", "THROUGH A WALL IT CANNOT CLIMB", "LMS", 300),
    (p_hydrogen, "QM-08", "HYDROGEN // 1s 2s 2p", "NOT AN ORBIT: A CLOUD OF PROBABILITY", "LMS", 300),
    (p_dirac, "QM-09", "DIRAC EQUATION", "THE EQUATION OF THE ELECTRON AND OF THE MUON", "LMS", 300),
    (p_uncert, "QM-10", "UNCERTAINTY", "HEISENBERG 1927 // SQUEEZE ONE, THE OTHER SPREADS", "LMS", 300),
    (p_field, "GR-01", "FIELD EQUATIONS", "MATTER TELLS SPACETIME HOW TO CURVE", "XLMS", 300),
    (p_schwarz, "GR-02", "SCHWARZSCHILD", "THE FIELD OF ONE MASS // 1916", "LM", 520),
    (p_flamm, "GR-03", "CURVED SPACE", "FLAMM'S PARABOLOID // 1916", "XLMS", 300),
    (p_gps, "GR-04", "CLOCKS AND GRAVITY", "GPS CORRECTS FOR EINSTEIN EVERY DAY", "LMS", 300),
    (p_bending, "GR-05", "LIGHT BENDING", "STARLIGHT BENT BY THE SUN // EDDINGTON 1919", "LMS", 300),
    (p_lens, "GR-06", "GRAVITATIONAL LENS", "ONE SOURCE: TWO IMAGES, OR A RING", "LMS", 300),
    (p_chirp, "GR-07", "GRAVITATIONAL WAVE", "LIGO // 14 SEPTEMBER 2015", "LMS", 300),
    (p_perihelion, "GR-08", "PERIHELION OF MERCURY", "THE ORBIT NEWTON COULD NOT CLOSE", "XLMS", 300),
    (p_cones, "GR-09", "LIGHT CONES // HORIZON", "INSIDE, EVERY FUTURE POINTS INWARDS", "LMS", 300),
    (p_hole, "GR-10", "BLACK HOLE", "PATHS OF LIGHT // SCHWARZSCHILD", "XLMS", 300),
    (p_const, "XX-00", "CONSTANTS", "CODATA 2022 // PDG 2024", "MS", 250),
)


# ----------------------------------------------------------------------------
# the wall cut into cells, and which plate each cell shows when
# ----------------------------------------------------------------------------

Y_TOP, Y_BOT, GAP = 246.0, 1186.0, 26.0
# ways to cut a bay into cells, by its width: rows of (height share, width shares), top to bottom
_NARROW = (((1, (1,)), (1, (1,)), (1, (1,))), ((1.25, (1,)), (1, (1,))), ((0.8, (1,)), (1.2, (1,)), (0.8, (1,))),
           ((1, (1,)), (1.3, (1,))))
_MID = (((1, (1,)), (1, (1,))), ((1, (1,)), (1, (1,)), (1, (1,))), ((1.2, (1,)), (1, (1, 1))), ((1, (1, 1)), (1.2, (1,))))
_WIDE = (((1.35, (1,)), (1, (1, 1))), ((1, (1, 1)), (1.35, (1,))), ((1, (1, 1)), (1, (1, 1))), ((1, (1, 1)), (2.2, (1,))),
         ((1, (1,)),), ((1, (1, 1)), (1, (1,)), (1, (1, 1))))
RANK = {"S": 0, "M": 1, "L": 2, "X": 3}


def _split(a, b, shares):
    """An interval cut in parts of the given proportions, GAP apart."""
    room = (b - a) - GAP * (len(shares) - 1)
    out, x = [], a
    for v in shares:
        out.append((x, x + room * v / sum(shares)))
        x += room * v / sum(shares) + GAP
    return out


def _cut(x0, x1, pick):
    """Cells of one bay (rects). `pick` chooses among the ways its width allows."""
    w = x1 - x0
    if w < 250.0:
        return []
    if w < 300.0:                                   # too narrow for a drawing: one column of figures
        return [(x0, Y_TOP, x1, Y_BOT)]
    if w >= 1000.0:                                 # a very wide bay is two bays
        (a0, a1), (b0, b1) = _split(x0, x1, ((1.45, 1.0), (1.0, 1.45), (1.0, 1.0))[pick % 3])
        return _cut(a0, a1, pick // 3 + 1) + _cut(b0, b1, pick // 3 + 2)
    ways = _NARROW if w < 520.0 else _MID if w < 720.0 else _WIDE
    ok = [r for r in ways if min((w - GAP * (len(c) - 1)) / len(c) for _, c in r) >= 300.0] or [ways[0]]
    rows = ok[pick % len(ok)]
    out = []
    for (y0, y1), (_, cols) in zip(_split(Y_TOP, Y_BOT, [r[0] for r in rows]), rows):
        out += [(xa, y0, xb, y1) for xa, xb in _split(x0, x1, cols)]
    return out


def _cls(r):
    a = (r[2] - r[0]) * (r[3] - r[1])
    return "X" if a >= 6.0e5 else "L" if a >= 2.8e5 else "M" if a >= 1.65e5 else "S"


def music_hits(cues, t0, t1):
    """The hits of the music between t0 and t1, each on the frame where it is heard: (times, strengths,
    kicks). Every onset of the cues is a hit; a kick is the onset it falls with (within two frames), or a hit
    of its own. Two hits less than five frames apart are one: a flash (FLASH) is over before the next one."""
    fr = {}
    m = (cues.onset_t >= t0 - LAG - 0.02) & (cues.onset_t < t1 - LAG)
    for t, a in zip(cues.onset_t[m], cues.onset_a[m]):
        n = int(round((float(t) + LAG) / FRAME))
        fr[n] = [max(fr.get(n, (0.0, 0.0))[0], float(a)), 0.0]
    m = (cues.kick_t >= t0 - LAG - 0.02) & (cues.kick_t < t1 - LAG)
    for t, a in zip(cues.kick_t[m], cues.kick_a[m]):
        n = int(round((float(t) + LAG) / FRAME))
        near = [v for v in fr if abs(v - n) <= 2]
        if near:
            v = min(near, key=lambda v: abs(v - n))
            fr[v][1] = max(fr[v][1], float(a))
        else:
            fr[n] = [0.0, float(a)]
    out = []
    for n in sorted(fr):
        if out and n - out[-1][0] < 5:
            out[-1][1], out[-1][2] = max(out[-1][1], fr[n][0]), max(out[-1][2], fr[n][1])
        else:
            out.append([n, fr[n][0], fr[n][1]])
    if len(out) < t1 - t0:                          # no cues to speak of: the eighth notes of the track, a kick on the beats
        out = [[int(round(v / FRAME)), 1.0 - 0.6 * (k % 2), 1.0 - k % 2]
               for k, v in enumerate(np.arange(t0 + LAG, t1, 0.5 * BEAT))]
    return ([o[0] * FRAME - 1e-4 for o in out], [max(o[1], 0.8 * o[2]) for o in out], [o[2] for o in out])


def schedule(ctx, land):
    """The whole sequence: a list of [t_in, t_out, rect, plate, class, taken apart].

    Every two bars the bays are cut into new cells, whose plates come in over the three hits after the
    downbeat (the first time: as the ring that leaves the landing point of the muon reaches them). Then every
    hit takes plates away - they flash - and brings a plate to every cell the hit before left dark. Which
    cells: the smallest of them are the high voice (every hit; two of them in the third phrase, one more on
    an accent), the middle ones answer the accents, the largest third the kicks; in a voice the plate that
    has been there longest goes. A hit never flashes more than a quarter of the wall, a plate stays MIN_LIFE.
    On the next downbeat everything that is lit goes in one flash. After T_TRANS nothing cuts any more: the
    plates that are there are taken apart, one or two on each of the last kicks, the large ones last."""
    cues = ctx.cues
    rng = np.random.default_rng(4217)
    ht, hs, hk = music_hits(cues, T_DRUMS, T_TRANS)
    if ht[0] > T_ON + 0.05:                         # the kick that brings the drums back is a hit, whatever the cues say
        ht.insert(0, T_ON); hs.insert(0, 1.0); hk.insert(0, 1.0)
    ht[0] = T_ON
    cols = [c for c in ctx.cols if c[1] - c[0] >= 250.0]
    fx = ctx.focus[0]
    n_lay = int(math.ceil((T_TRANS - T_DRUMS) / (2 * BAR) - 1e-6))
    edges = [0]                                     # the hit on the downbeat of every second bar (or the beat itself)
    for j in range(1, n_lay):
        g = T_DRUMS + LAG + j * 2 * BAR
        i = int(np.argmin(np.abs(np.asarray(ht) - g)))
        if abs(ht[i] - g) > 0.07:
            i = int(np.searchsorted(ht, on_frame(g)))
            ht.insert(i, on_frame(g)); hs.insert(i, 1.0); hk.insert(i, 0.0)
        edges.append(i)
    edges.append(len(ht))
    ht, hs, hk = np.asarray(ht, np.float64), np.asarray(hs), np.asarray(hk)
    deck = [int(v) for v in rng.permutation(len(PLATES))]
    by_fn = {pl[0].__name__: k for k, pl in enumerate(PLATES)}
    kept = set()                                    # plates that are not dealt
    if any(c[1] - c[0] < 300.0 for c in cols):      # a bay only wide enough for the figures: they are its own
        kept.add(by_fn["p_const"])
    want = [by_fn["p_frames"], by_fn["p_survival"]]  # the last large plates: why the muon gets here at all
    out, shown, prev_pick = [], set(), {}
    for j in range(n_lay):
        i0, i1 = edges[j], edges[j + 1]
        last = j == n_lay - 1
        q = 0 if j < 2 else 1 if j < 4 else 2          # the phrase: how much of the wall a hit changes
        rects = []
        for ci, (x0, x1) in enumerate(cols):
            cells = []
            for _ in range(40):
                cells = _cut(x0, x1, int(rng.integers(0, 3000)))
                sig = [tuple(np.round(c)) for c in cells]
                kinds = [_cls(c) for c in cells]
                if sig != prev_pick.get(ci) and (not last or "X" not in kinds) and (
                        not last or x1 - x0 < 720.0 or "L" in kinds):
                    break
            prev_pick[ci] = [tuple(np.round(c)) for c in cells]
            rects += cells
        n = len(rects)
        cls = [_cls(r) for r in rects]
        area = np.array([(r[2] - r[0]) * (r[3] - r[1]) for r in rects])
        order = np.argsort(area, kind="stable")
        voice = np.ones(n, int)                       # 0: every hit, 1: the accents, 2: the kicks
        voice[order[:max(1, int(round(0.4 * n)))]] = 0
        voice[order[n - max(1, int(round(0.3 * n))):]] = 2
        still = [r[2] - r[0] < 300.0 for r in rects]  # a column of figures has one plate: it is not taken away
        cx, cy = np.array([0.5 * (r[0] + r[2]) for r in rects]), np.array([0.5 * (r[1] + r[3]) for r in rects])
        if j == 0:                                    # births: as the ring that leaves the landing point reaches them
            dist = np.hypot(cx - land[0], cy - land[1])
            first = int(np.argmin(dist))
            dark = {k: int(np.searchsorted(ht, T_ON + (dist[k] - dist[first]) / V_SHOCK - 0.5 * FRAME)) for k in range(n)}
        else:                                         # ... then from the left, the right, the middle, the edges
            ox = (L.FX0, L.FX1, 0.5 * (L.FX0 + L.FX1), None)[(j - 1) % 4]
            d = np.abs(cx - ox) if ox is not None else -np.abs(cx - 0.5 * (L.FX0 + L.FX1))
            rank = np.argsort(np.argsort(d, kind="stable"), kind="stable")
            first = -1
            dark = {k: i0 + 1 + int(3 * rank[k] / n) for k in range(n)}       # cell -> the hit that brings its plate
        if last:
            kept |= set(want)                         # kept for the end
        stop = int(np.searchsorted(ht, T_TRANS - 0.3)) if last else i1       # the last flashes: then every cell is lit
        accent = np.argsort(np.argsort(hs[i0:i1], kind="stable"), kind="stable") / max(i1 - i0 - 1, 1)
        lit, prev = {}, {}

        def bring(k, h):
            """A plate cuts into cell k at h: the next one of the deck that fits it and is not on the wall."""
            r = rects[k]
            pid = None
            if j == 0 and k == first and k not in prev and cls[k] in PLATES[by_fn["p_identity"]][4]:
                pid = by_fn["p_identity"]             # where the lone muon landed: what it is
            if pid is None:
                for m, pp in enumerate(deck):
                    pl = PLATES[pp]
                    if (cls[k] in pl[4] and r[2] - r[0] >= pl[5] and pp not in kept and pp not in shown
                            and pp != prev.get(k)):
                        pid = pp
                        deck.append(deck.pop(m))
                        break
            if pid is None:
                pid = by_fn["p_const"]
            shown.add(pid)
            prev[k] = pid
            lit[k] = [h, T_OUT, r, pid, cls[k], False]
            out.append(lit[k])
            del dark[k]

        for i in range(i0, i1):
            h = float(ht[i])
            for k in [k for k in sorted(dark) if dark[k] <= i]:
                bring(k, h)
            if i >= stop:
                continue
            hold = {k: BAR - 0.05 if k == first and e[3] == by_fn["p_identity"] else MIN_LIFE[cls[k]]
                    for k, e in lit.items()}          # what the muon is stays for a bar
            ready = [k for k in sorted(lit, key=lambda k: (lit[k][0], k)) if h - lit[k][0] >= hold[k] and not still[k]]
            a = accent[i - i0]
            take, room = [], 0.25 * L.W * L.H
            for v, cnt in ((2, 1 if hk[i] > 0 else 0), (0, (1, 1 + (a >= 0.6), 2 + (a >= 0.8))[q]),
                           (1, 1 if a >= (0.5, 0.5, 0.3)[q] else 0)):
                for k in [k for k in ready if voice[k] == v][:cnt]:
                    if area[k] <= room:
                        take.append(k)
                        room -= area[k]
            for k in take:
                e = lit.pop(k)
                e[1] = h
                shown.discard(e[3])
                dark[k] = i + 1
        if not last:                                  # the downbeat of the next two bars: the wall goes in one flash
            for e in lit.values():
                e[1] = float(ht[i1])
                shown.discard(e[3])
            continue
        for k in sorted(dark):                        # (a cell the last hit left dark)
            bring(k, on_frame(min(float(ht[-1]) + 0.12, T_TRANS)))
        # the plates that are there when the cuts stop: taken apart
        fin = list(lit.values())
        wk = cues.kick_t[(cues.kick_t >= T_TRANS + 0.3 - LAG) & (cues.kick_t <= T_OUT - 1.2 - LAG)].astype(np.float64) + LAG
        if len(wk) < 3:
            wk = np.linspace(T_TRANS + 0.5, T_OUT - 1.3, 8)
        fin.sort(key=lambda e: (RANK[e[4]], -abs(0.5 * (e[2][0] + e[2][2]) - fx)))
        for m, e in enumerate(fin):
            e[1] = float(wk[min(len(wk) - 1, int(m * len(wk) / max(len(fin), 1)))]) + 0.35
            e[5] = True
        for e in sorted(fin, key=lambda e: (-RANK[e[4]], abs(0.5 * (e[2][0] + e[2][2]) - fx))):
            if want and e[4] in PLATES[want[0]][4] and e[2][2] - e[2][0] >= PLATES[want[0]][5]:
                e[3] = want.pop(0)
    out.sort(key=lambda e: e[0])
    return out


def audio(cues, t):
    """What the music is doing now: what the plates move with."""
    b = cues.bands(t, 0.08)
    return dict(kick=min(1.5, cues.kick(t, 0.14)), on=min(1.5, cues.onset(t, 0.08)), loud=cues.loud(t),
                bass=float(0.5 * (b[0] + b[1])), sk=cues.since_kick(t), spec=cues.spec(t),
                n_on=int(np.searchsorted(cues.onset_t, t, side="right")), n_k=int(np.searchsorted(cues.kick_t, t, side="right")))


BLOCKS = [("spec", 300.0, 560.0), ("count", 150.0, 200.0), ("rate", 200.0, 0.0, True)]     # the bottom band


class Quick(B.Block):
    """A plate being taken apart in the transition: the moves of the show (pens, marks thrown out, curves
    traced) run backwards with a short wave. Only the three characters at the pen are noise, and no figure
    spins: a figure that spins while its plate is still there would be a wrong figure on the wall."""
    HOLD = 0.5                  # everything is in place 0.45 s after the wave has passed: the frame then draws as usual

    def _key(self, xl, y):
        return (int(xl) * 7919 + int(y) * 104729 + self.key) & 0xFFFFF

    def text(self, s, xl, y, pad):
        return B.decode(s, float(self.la(xl, y)), cps=self.cps, key=self._key(xl, y), band=3, pad=pad)

    def tag(self, s, xl, y):
        a = float(self.la(xl, y))
        if a <= 0.0:
            return "", 0.0
        return B.tag_state(s, a, cps=self.cps, wipe=0.05, lead=8, key=self._key(xl, y))


def leaving(left, out, span=1.3):
    """Age for Frame.build of something that was cut in whole and is taken apart in its last `out` seconds
    (None: it is there, drawn as usual; negative: gone)."""
    return None if left >= out else B.io(span, left, out=out, span=span)


class Glitch(Scene):
    name = "glitch"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.dance = Dance(ctx)
        self.geo = self.dance.geo
        self.st0 = stage_for(ctx, ctx.focus[0])     # stage of the break: the lone muon falls on the focus
        rng = np.random.default_rng(77)
        n = 64
        self.lane_u = np.sort(rng.uniform(0.02, 0.98, n))       # position across the view
        self.lane_t = rng.uniform(T_TRANS - 0.5, T_OUT - 1.2, n)
        self.lane_v = rng.uniform(260.0, 620.0, n)
        self.lane_red = rng.random(n) < 0.3
        self.land = (ctx.focus[0] + 0.12 * (ctx.focus_bay[1] - ctx.focus_bay[0]), Y_GROUND)
        self.cells = schedule(ctx, self.land)
        self.c_in = np.array([c[0] for c in self.cells])
        # a plate that is taken apart is gone at its t_out; the others leave static for FLASH after it
        self.c_end = np.array([c[1] + (0.0 if c[5] else FLASH) for c in self.cells])
        self.cut_t = np.sort(self.c_in)
        self.place = flow(bottom_panels(ctx), BLOCKS)
        # the three phrases on the score strip: how many plates a second each one cuts
        self.marks = []
        for a, b in ((0, 4), (4, 8), (8, None)):
            ta, tb = T_DRUMS + a * BAR, (T_TRANS if b is None else T_DRUMS + b * BAR)
            n = int(np.searchsorted(self.cut_t, tb) - np.searchsorted(self.cut_t, ta))
            self.marks.append((ta, f"{n / (tb - ta):.0f} CUTS/S"))
        self.marks.append((T_TRANS, "RUN OUT"))

    # ------------------------------------------------------------------ draw
    def draw(self, f, t, ctx):
        if t < T_ON:
            return self._break(f, t, ctx)
        a = audio(ctx.cues, t - LAG)                # what is heard now
        self._strip(f, t, ctx)
        self._shock(f, t)
        self._plates(f, t, a)
        self._sweep(f, t)
        if t > T_TRANS - 0.6:
            view = (L.FX0 + 12.0, VIEW_Y0, L.FX1 - 12.0, VIEW_Y1)
            f.set_clip(*view)
            self._lanes(f, t, view)
            f.set_clip()
        self._bottom(f, t, ctx, a)
        return {"edge_alpha": 0.5 + 0.5 * float(1.0 - smoothstep(T_TRANS, T_TRANS + 4.5, t))}

    def _plates(self, f, t, a):
        """The plates on the wall at t. A plate is there, whole, from the frame of the hit that brings it; the
        hit that takes it away leaves static in its cell for FLASH. The last ones (the transition) are taken
        apart instead."""
        for k in np.flatnonzero((self.c_in <= t) & (t < self.c_end)):
            t_in, t_out, rect, pid, cls, taken = self.cells[k]
            if t >= t_out:
                self._static(f, t, rect, a, int(k))
                continue
            fn, code, title, cap = PLATES[pid][:4]
            with Quick(f, leaving(t_out - t, 0.35, span=0.4) if taken else None, rect, wave=0.1, line=0.2, cps=420.0,
                       marks=False, key=int(k) * 7 + pid):
                fn(Pl(_Wall(f) if E.WALL else f, chrome(f, rect, code, title, cap, a, int(k)), t - t_in, t, a, key=int(k)))

    @staticmethod
    def _static(f, t, rect, a, key):
        """The flash: the static behind the MUON card as it stands at 01:43:10, in one cell - the same bars
        (muon.static), as many of them to the metre, its bands of density drifting through, a few red ones - and
        a new pattern on every frame (FRAME), each cell its own."""
        x0, y0, x1, y1 = rect
        ry = y0 + (np.arange(int((y1 - y0) / ROW_H)) + 0.5) * ROW_H
        band = 0.55 + 0.45 * np.sin(ry * 0.013 + t * 9.0) * np.sin(ry * 0.0041 - t * 3.1)
        part = (x1 - x0) / (L.FX1 - L.FX0 - 8.0)      # static() deals its bars for the width of the wall
        tf = (round(t / FRAME) + 977 * key) / 30.0    # ... and a new pattern every 1 / 30 s: a cell takes its own run of them
        static(f, tf, rect, (0.26 + 0.1 * a["loud"]) * (0.55 + 0.75 * band ** 2) * part, seed=3 + 17 * key)
        static(f, tf, rect, 0.012 * part, seed=11 + 17 * key, layer="r")

    def _shock(self, f, t):
        """The kick that brings the drums back is the muon of the break reaching the ground: rings leave the
        point where it landed, and the plates cut in on the hits of the music as the first one reaches them."""
        a = t - T_ON
        if a >= 1.3:
            return
        f.set_clip(L.FX0, L.HEAD_Y, L.FX1, VIEW_Y1)
        ang = np.linspace(0.0, 2 * np.pi, 97)
        for k, lay in enumerate(("r", "w", "w")):
            r = V_SHOCK * (a - 0.07 * k)
            if r > 0.0:
                f.polyline(lay, self.land[0] + r * np.cos(ang), self.land[1] + r * np.sin(ang),
                           (1.2 if k == 0 else 0.9 if E.WALL else 0.5) * (1.0 - a / 1.3) ** 1.5, width=L.LW_BOLD if k == 0 else L.LW)
        f.dots("r", [self.land[0]], [self.land[1]], 4.2 + 14.0 * math.exp(-a / 0.12), 1.7 * (1.0 - a / 1.3))
        f.set_clip()

    def _sweep(self, f, t):
        """On every bar a line reads the wall from top to bottom (and stops with the cuts)."""
        ph = (t - T_ON) % BAR
        if ph < 0.24 and T_ON + BAR - 0.01 <= t < T_TRANS:
            y = Y_TOP + (Y_BOT - Y_TOP) * float(B.ease(ph / 0.24))
            f.segments("w", [L.FX0 + 30.0], [y], [L.FX1 - 30.0], [y], 0.8, width=L.LW)

    def _strip(self, f, t, ctx):
        """Score strip: the loudness of the music and the hits of the towers over the whole sequence, the
        three phrases, the time. There on the kick; taken apart at the end."""
        left = T_OUT - 0.5 - t
        if left <= 0.0:
            return
        n = int(np.searchsorted(self.cut_t, t, side="right"))
        hud.show_strip(f, t, ctx, f"OVERLOAD // MUON + QUANTUM + RELATIVITY // CUT ON EVERY HIT // PLATE {n:04d}",
                       T_DRUMS, T_OUT, marks=self.marks, age=leaving(left, 0.6, span=1.2))

    def _bottom(self, f, t, ctx, a):
        """Bottom band: the spectrum of the music (what drives the cuts), the count of plates, the cuts as a
        barcode - there on the kick, taken apart in the transition; then, alone, the count of the falling muons."""
        y0, y1 = ctx.slots["y0"], ctx.slots["y1"]
        pl = self.place
        box = lambda p: (p[0] - 8.0, y0 - 24.0, p[1] + 8.0, y1 + 8.0)
        if "spec" in pl:
            x0, x1 = pl["spec"]
            with f.build(leaving(T_LANES - 0.9 - t, 0.4), box(pl["spec"]), key=71):
                hud.panel_header(f, x0, x1, y0, "DRIVE // THE MUSIC, 30 HZ - 16 KHZ // ITS HITS CUT THE PLATES"
                                 if x1 - x0 > 520 else "DRIVE // THE MUSIC")
                sp = a["spec"]
                bw = (x1 - x0 - 150.0) / len(sp)
                xs = x0 + np.arange(len(sp)) * bw
                hb = (y1 - y0 - 30.0) * np.clip(sp, 0.02, 1.0) ** 1.15
                f.rects("w", xs, y1 - 4.0 - hb, xs + max(bw - 3.0, 2.0), y1 - 4.0, 0.9)
                f.rects("r", x0, y1 - 3.0, x1 - 150.0, y1, 1.0)
                xr = x1 - 134.0
                f.text("w", xr, y0 + 40.0, f"LOUD  {a['loud']:.2f}", size=L.T_MICRO, alpha=0.8)
                f.text("w", xr, y0 + 62.0, f"KICK  {min(a['kick'], 1.5):.2f}", size=L.T_MICRO, alpha=0.8)
                f.text("w", xr, y0 + 84.0, "125 BPM", size=L.T_MICRO, alpha=0.6)
                if a["sk"] < 0.09:                    # the lamp of the kick: a few frames
                    f.tag("r", xr, y0 + 112.0, "KICK", size=L.T_SMALL, pad=3)
        n = int(np.searchsorted(self.cut_t, t, side="right"))
        if "count" in pl:
            x0, x1 = pl["count"]
            with f.build(leaving(T_LANES - 0.7 - t, 0.4), box(pl["count"]), key=72):
                hud.panel_header(f, x0, x1, y0, "PLATES SHOWN")
                f.text("w", x0, y0 + 84.0, f"{n:04d}", size=fsz(4, x1 - x0, 58.0))
                rate = n - int(np.searchsorted(self.cut_t, t - 1.0, side="right"))
                f.text("w", x0 + 2.0, y0 + 112.0, f"{rate:02d} A SECOND", size=L.T_MICRO, alpha=0.75)
        if "rate" in pl:
            x0, x1 = pl["rate"]
            with f.build(leaving(T_LANES - 0.5 - t, 0.4), box(pl["rate"]), key=73):
                hud.panel_header(f, x0, x1, y0, "CUTS >> BARCODE // LAST 3 S" if x1 - x0 > 300 else "CUTS >> BARCODE")
                cols = int(np.clip((x1 - x0) / 3.6, 40, 160))
                kk, frac, dt = hud.barcode_keys(t, 3.0, cols)
                cnt = np.searchsorted(self.cut_t, kk * dt + dt) - np.searchsorted(self.cut_t, kk * dt)
                hud.barcode_lanes(f, x0, x1, y0 + 12.0, y1, 0.04 + 0.92 * np.tanh(cnt / 1.2), kk, lanes=3, seed=6, frac=frac)
                f.segments("r", [x1 - 1.0], [y0 + 8.0], [x1 - 1.0], [y1], 1.2, width=L.LW)
        # what is left: the muons that fall, counted
        where = pl.get("spec") or pl.get("rate") or pl.get("count")
        if where and t >= T_LANES:
            x0, x1 = where[0], min(where[1], where[0] + 560.0)
            with f.build(B.io(t - T_LANES, T_OUT - 0.1 - t, out=0.35), box((x0, x1)), key=74):
                hud.panel_header(f, x0, x1, y0, "LANES // MUONS FALLING")
                f.text("w", x0, y0 + 84.0, f"{int((t - self.lane_t > 0).sum()):02d}", size=58.0)
                if x1 - x0 > 330:
                    f.text("w", x0 + 96.0, y0 + 62.0, "15.000 KM TO GROUND", size=L.T_SMALL, alpha=0.8)
                    f.text("w", x0 + 96.0, y0 + 86.0, "ALMOST NOTHING IN THEIR WAY", size=L.T_MICRO, alpha=0.6)

    def _lanes(self, f, t, view):
        """What is left when the plates have gone: single muons, falling straight down (lines that may
        pass behind the towers)."""
        x0, y0, x1, y1 = view
        a = t - self.lane_t
        live = a > 0
        if not live.any():
            return
        gate = float(smoothstep(T_TRANS - 0.6, T_TRANS + 2.0, t))
        yh = y0 + a * self.lane_v
        out = float(1.0 - 0.6 * smoothstep(T_OUT - 2.0, T_OUT, t))
        lane_x = x0 + self.lane_u * (x1 - x0)
        for red in (False, True):
            m = live & (self.lane_red == red)
            if not m.any():
                continue
            lay = "r" if red else "w"
            x = lane_x[m]
            y = np.minimum(yh[m], y1 - 6)
            landed = yh[m] >= y1 - 6
            i1 = np.where(landed, 0.28, 0.6)
            i1 = (i1 if red else E.wl(i1)) * gate * out
            f.segments(lay, x, np.full_like(x, y0), x, y, 0.05 * gate * out, i1, width=L.LW)
            f.dots(lay, x[~landed], y[~landed], 2.6, 1.5 * gate * out)
            if red:
                f.dots("w", x[~landed], y[~landed], 1.0, 0.9 * gate * out)

    # ------------------------------------------------------------------ 7.0 almost nothing
    def _break(self, f, t, ctx):
        geo = self.geo
        w = self.dance.world
        st = self.st0
        view = st.view
        dim = 0.3
        cam = OrthoCamera((0.0, 0.0, 60.0), (0.0, 0.0, 0.0), scale=geo.S, screen_center=(geo.x_mid, Y_GROUND))
        # the only muon: it takes the whole break to come down on the focus, and lands when the drums come back
        p = float(np.clip((t - T_IN) / (T_ON - T_IN), 0.0, 1.0))
        ytop = view[1] + 4
        y = ytop + p * (Y_GROUND - ytop)
        slope = math.tan(math.radians(5.0))
        xg = self.land[0]
        x = xg - slope * (Y_GROUND - y)
        xt = xg - slope * (Y_GROUND - ytop)
        info = info_layout(ctx, st, "VIEW 04 // ORTHO_FRONT // HOLD", cam, zones_top=[(xt - 110.0, xt + 110.0)],
                           zones_full=[(xt - 260.0, xg + 260.0)])
        age0 = t - T_IN                         # what the break brings is constructed from its cut
        f.set_clip(*view)
        w.draw_ground(f, cam, "front", view, gain=dim)
        altitude_rules(f, ctx, st, cam, 0.0, 0.0, info, gain=dim, depth=False, age=age0 - 0.1)
        f.segments("r", [xg - slope * (Y_GROUND - ytop)], [ytop], [x], [y], 0.06, 0.75, width=L.LW)
        f.dots("r", [x], [y], 4.2, 1.7)
        f.dots("w", [x], [y], 1.6, 1.1)
        alt = (Y_GROUND - y) / geo.S
        said = sd.said("almost nothing", 420.0)
        auto_callout(f, ctx, view, x, y, "MU-", [f"ALT {alt:06.3f} KM", "E 3.871 GEV", "N 000001" if t < said else "1 OF 1"],
                     red=True, prefer=(1, -1), dx=64.0, dy=46.0, build=age0 - 0.2)
        with f.build(age0 - 0.3, (xg - 18.0, Y_GROUND - 18.0, xg + 18.0, Y_GROUND + 4.0), wave=0.05, marks=False, key=91):
            f.segments("r", [xg - 16, xg], [Y_GROUND, Y_GROUND - 16], [xg + 16, xg], [Y_GROUND, Y_GROUND + 2], 0.5, width=E.ww(1.0))
        f.set_clip()
        # what is left of the HUD: the furniture of the DANCE scene, still there, dimmed and emptied (it is not
        # rebuilt); what is new on it - title, cursor, the one row, the counts - is made on the cut
        a = 0.4
        x0, y0, x1, y1, yb = hud.strip_base(f, title=None, alpha=a)
        B.tag(f, "w", x0, y0 - 9, "LONGITUDINAL_PROFILE // --", age0, size=L.T_MICRO, pad=3, alpha=a, cps=110.0, key=83)
        xs = x0 + (16.0 - np.arange(0, 16.01, 1.0)) / 16.0 * (x1 - x0)
        f.segments("w", xs, np.full_like(xs, y0), xs, y0 + 12, E.wl(0.8 * a), width=E.ww(1.0))
        xc = x0 + (16.0 - min(alt, 16.0)) / 16.0 * (x1 - x0)
        with f.build(age0 - 0.1, (xc - 6.0, y0 - 8.0, xc + 190.0, y1 + 8.0), flow="tb", wave=0.1, marks=False, key=84):
            hud.strip_cursor(f, float(xc), y0, y1, f"ALT {alt:06.3f} KM", alpha=0.9)
        if st.col is not None:
            cx0, cy0, cx1, cy1 = st.col
            moved = self.dance.stage(N_PHRASES - 1).col != st.col       # the column was elsewhere: made here
            with f.build(age0 - 0.1 if moved else None, (cx0 - 6, cy0 - 8, cx1 + 6, cy1 + 6), flow="tb", wave=0.4,
                         key=46):
                f.rects("w", cx0, cy0, cx1, cy0 + 5, 0.95 * a)
                f.tag("w", cx0 + 4, cy0 + 32, "PARTICLE_STREAM", size=L.T_MICRO, pad=3, alpha=a)
                f.segments("w", [cx1, cx0], [cy0, cy1], [cx1, cx1], [cy1, cy1], E.wl(0.6 * a), width=E.ww(1.0))
            row = f"00000 MU-   003871.00 {(x - geo.x_mid) / geo.S:+06.2f} {alt:05.2f} +00.00"
            f.text("r", cx0 + 8, cy0 + 62, B.decode(row, age0 - 0.2, cps=120.0, key=85), size=col_type(st.col), alpha=0.95)
            f.text("w", cx0 + 8, cy1 - 10, B.roll("N 000001", age0, 0.4, 0.3, key=86), size=L.T_SMALL, alpha=0.9 * a)
        py0, py1 = ctx.slots["y0"], ctx.slots["y1"]
        place = flow(bottom_panels(ctx), PANEL_BLOCKS)
        if "count" in place:
            px0, px1 = place["count"]
            hud.panel_header(f, px0, px1, py0, "PARTICLES", alpha=a)
            rows = (("E+-", 0, "w"), ("GAMMA", 0, "w"), ("HADRON", 0, "w"), ("MU+-", 1, "r"))
            if px1 - px0 >= 430:
                cw = (px1 - px0) / 2
                for r, (lab, n, lay) in enumerate(rows):
                    xx, yy = px0 + (r % 2) * cw, py0 + 46 + (r // 2) * 44
                    f.tag(lay, xx + 4, yy, lab, size=L.T_SMALL, pad=3, alpha=a if lay == "w" else 1.0)
                    f.text(lay, xx + 104, yy + 2, B.roll(f"{n:06d}", age0, 0.45, 0.25 + 0.06 * r, key=87 + r), size=28,
                           alpha=a if lay == "w" else 1.0)
            else:
                for r, (lab, n, lay) in enumerate(rows):
                    yy = py0 + 38 + r * 27
                    f.tag(lay, px0 + 4, yy, lab, size=L.T_MICRO, pad=3, alpha=a if lay == "w" else 1.0)
                    f.text(lay, px0 + 92, yy + 2, B.roll(f"{n:06d}", age0, 0.45, 0.25 + 0.06 * r, key=87 + r), size=22,
                           alpha=a if lay == "w" else 1.0)
        if "bar" in place:
            px0, px1 = place["bar"]
            hud.panel_header(f, px0, px1, py0, "BIRTH_RATE >> BARCODE" if px1 - px0 > 260 else "BIRTH_RATE", alpha=a)
            f.segments("r", [px1 - 2], [py0 + 7], [px1 - 2], [py1], 1.2 * a, width=L.LW)
        draw_info(f, ctx, info, alpha=0.7, age=age0 - 0.05)
        return {"edge_alpha": 0.5}
