"""The three detector towers: their faces, their blooms, the strings between them, their scopes.

Everything here is driven by (a) the tower rectangles from layout / towers.json and (b) the detector
streams (showdata.Detectors): one float per tower, a hit = a rising edge, its peak = the energy.
"The more powerful the muon, brighter, louder, the reply."

The blooms and the strings move as in the TouchDesigner scene (scene1and2.mov): the loops of a bloom stay
nested on the detector and change size (bloom, _breath); the strings are rows of points displaced by one
smooth noise field, by nothing at a tower and the most in the middle of a span (strings). Everything is
continuous in t: a hit swells and brightens, it never switches a loop on or makes a line jump.
"""
from __future__ import annotations

import math

import numpy as np

from . import build as B
from . import hud
from . import layout as L
from . import showdata as sd
from .engine import hash01, smoothstep

N_LOOPS = {"L": 21, "C": 25, "R": 21}
S_MAX = {"L": 330.0, "C": 430.0, "R": 330.0}     # size of the outermost loop of a fully open bloom
S_MIN = 8.0
SOLID = 0.66                    # loops below this fraction are solid lines (the bright body), the rest are dotted
SPEED = 760.0                   # px / s of the burst of a bloom, and of the glow a hit sends along the strings
ATTACK = 0.12                   # s: a hit opens a bloom over this time, not in one frame
BREATH = 0.22                   # reference rate of the breathing of the loops (the `rate` of bloom() scales it)
GROW = 0.6                      # a loop of size s is 63 % grown when the burst has travelled GROW * s
REST = 0.8                      # rest size of the outermost loop, as a fraction of S_MAX: breathing takes it to about 1
SPREAD = 1.35                   # how the loops share the room: 1 = evenly, more = crowded on the detector


def loop_size(jf, n, s_max, s_min=None):
    """Rest size of loop number jf: small circles on the detector, wide loops outside."""
    s_min = S_MIN if s_min is None else s_min
    return s_min + (s_max - s_min) * (np.maximum(jf, 0.0) / n) ** SPREAD


def open_scale(open_):
    """Size factor of a bloom for a given openness (0..1.2): the stronger the muons, the larger the bloom.
    Proportional: a bloom that closes goes back into its detector, and a ruler beside it reads true."""
    return min(1.2, max(0.0, open_)) / 1.2


def bloom_height(key, open_, size=1.0):
    """Height (px) of the body of a bloom above its detector for a given openness - for scales and labels:
    the rest height of its largest solid loop (the loops breathe around their rest size, the dotted ones
    fly further out)."""
    return 2.0 * (1.0 - 0.1 * SOLID) * SOLID ** SPREAD * REST * S_MAX[key] * size * open_scale(open_)


def energy_env(det, key, t, tau=1.1, attack=0.0):
    """Slow envelope of the recent hits of a tower (0..~1.5): how 'open' its bloom is.
    attack (s) > 0: a hit raises it over about three times that long instead of in one frame (it still
    peaks at the energy of the hit), so that nothing driven by it jumps."""
    tt, ee, _ = det.hits(key, t - 6 * tau, t + 1e-6)
    if not len(tt):
        return 0.0
    a = t - tt
    w = np.exp(-a / tau)
    if attack > 0.0:
        a_pk = attack * math.log(1.0 + tau / attack)            # where one hit peaks
        w = w * (1.0 - np.exp(-a / attack)) / (math.exp(-a_pk / tau) * (1.0 - math.exp(-a_pk / attack)))
    return float((ee * w).sum())


# ----------------------------------------------------------------------------
# face
# ----------------------------------------------------------------------------

def face(f, tw, t, power=1.0, value=0.0, hit_age=99.0, hit_e=0.0, texture=True, dim=0.12, label=True):
    """Front of a tower: outline, detector module at the head, falling data texture when it is on."""
    if power <= 0.01:
        return
    p = min(1.0, power)
    on = power >= 0.99
    f.dim(tw.x0, tw.top, tw.x1, tw.bot, 1.0 - (1.0 - dim) * min(1.0, p * (1.0 if on else 1.6)))
    flash = hit_e * math.exp(-hit_age / 0.22) if hit_age < 2 else 0.0
    yd = tw.top + tw.det_h
    if on and texture:
        rh, n_c = 10.0, int(np.clip(round((tw.w - 14) / 9.0), 3, 9))      # (a narrow tower: fewer columns)
        cw = (tw.w - 14) / n_c
        scroll = t * 46.0
        ys = np.arange(yd + 8, tw.bot - 8, rh)
        r = np.floor((ys - scroll) / rh).astype(np.int64)
        dens = 0.3 + 0.35 * min(1.0, value * 2.2)
        sd_ = {"L": 11, "C": 22, "R": 33}[tw.key]
        R, C = np.meshgrid(r, np.arange(n_c), indexing="ij")
        lit = hash01(R, C, sd_) < dens * (0.6 + 0.8 * hash01(R, 7, sd_))
        Y = np.broadcast_to(ys[:, None], lit.shape)[lit]
        X = (tw.x0 + 7 + C * cw)[lit]
        f.rects("w", X, Y, X + cw - 1.5, Y + 4, 0.82)
        # the muon that was just caught runs down the tower as a red front
        if hit_age < 1.2:
            yf = yd + (tw.bot - yd) * min(1.0, hit_age / 0.5)
            fade = math.exp(-max(0.0, hit_age - 0.5) / 0.25)
            f.rects("r", tw.x0 + 4, yf - 18, tw.x1 - 4, yf, 0.9 * hit_e * fade)
    # detector module
    f.rect("w", tw.x0, tw.top, tw.x1, tw.bot, 0.95 * p, width=L.LW_FRAME)
    f.segments("w", [tw.x0], [yd], [tw.x1], [yd], 0.95 * p, width=L.LW_BOLD)
    n_s = 4 if tw.det_h >= 70.0 else 3                # the lines of the detector: fewer in a small box
    pad = min(22.0, 0.26 * tw.det_h)
    sy = tw.top + pad + np.arange(n_s) * (tw.det_h - 2 * pad) / (n_s - 1)
    inset = min(14.0, 0.18 * tw.w)
    f.segments("w", np.full(n_s, tw.x0 + inset), sy, np.full(n_s, tw.x1 - inset), sy, (0.55 + 0.4 * (not on)) * p,
               width=L.LW)
    if flash > 0.02:
        f.rects("r", tw.x0 + 4, tw.top + 4, tw.x1 - 4, yd - 3, 0.9 * flash)
        f.segments("r", np.full(n_s, tw.x0 + inset), sy, np.full(n_s, tw.x1 - inset), sy, 1.6 * flash, width=L.LW_BOLD)
    if not on:              # waiting: a bracket highlights the detector at the head of the tower
        b = 16.0
        blink = 0.55 + 0.45 * math.sin(t * 5.0)
        for sx, x in ((-1, tw.x0 - b), (1, tw.x1 + b)):
            f.segments("r", [x, x, x], [tw.top - b, tw.top - b, yd + b], [x, x - sx * 22, x - sx * 22],
                       [yd + b, tw.top - b, yd + b], 0.9 * p / 0.35 * blink * 0.6, width=L.LW)
    if label:                   # its name is decoded when the towers come on (the reveal)
        name = B.resolve(L.NAMES[tw.key], t - sd.T_REVEAL, 30.0, 0.25, pad=True, key=ord(tw.key))
        f.text("r" if flash > 0.2 else "w", tw.cx, tw.top - 14, name, size=L.T_MICRO,
               alpha=0.85 * p / max(p, 0.35) if on else 0.8, anchor="ms")


# ----------------------------------------------------------------------------
# bloom
# ----------------------------------------------------------------------------

_LOOP_H = {}


def _loop(s, j_frac, n, phi, t, seed, wide=1.0, j_noise=None, wander=0.36):
    """One loop of a bloom, tangent to the detector at phi = 0. Returns local (x, y), y up = negative.
    j_frac / n gives its shape (0 = a circle sitting on the detector, 1 = a wide petal); j_noise (default:
    j_frac) is its place in the family for the wandering of its outline: a loop that changes size keeps it.
    wander = how far the outline of an outer loop strays, as a fraction of its size."""
    q = np.clip(j_frac / n, 0.0, 1.0)                 # 0 = innermost, 1 = outermost
    wx = (1.0 + 0.45 * q ** 1.5) * wide
    wy = 1.0 - 0.1 * q
    p = 1.0 - 0.3 * q
    m = 0.06 + 0.74 * min(1.0, q / 0.6) ** 1.3        # 0 = circle sitting on the detector, 1 = petal (pointed)
    sn, cs = np.sin(phi), np.cos(phi)
    petal = np.sin(phi / 2) ** m / (0.77 ** m)
    x = s * wx * np.sign(sn) * np.abs(sn) ** p * petal
    y = -s * wy * (1.0 - cs)
    # the outer loops wander: smooth noise along the loop, pinned at the detector
    amp = wander * q ** 1.7
    k = np.arange(1, 7)[:, None]
    if seed not in _LOOP_H:                           # the draws of a family of loops: constants, made once
        _LOOP_H[seed] = (hash01(k, seed, 3) * 2 * np.pi,
                         (0.16 + 0.11 * hash01(k, seed, 5)) * np.where(k % 2 == 0, 1.0, -1.0), hash01(k, seed, 9) - 0.5)
    h0, hv, hj = _LOOP_H[seed]
    ph = h0 + t * hv + 0.55 * (j_frac if j_noise is None else j_noise) * hj
    nz = (np.sin(k * phi[None, :] + ph) / k ** 0.8).sum(0)
    pin = np.sin(phi / 2) ** 2
    cx, cy = 0.0, -s * wy
    dx, dy = x - cx, y - cy
    r = np.maximum(np.hypot(dx, dy), 1e-6)
    d = s * amp * nz * pin
    return x + dx / r * d, y + dy / r * d


def _breath(n, t, seed, rate=BREATH):
    """Size factor of each of the n loops of a bloom at time t (array). In the TouchDesigner scene the loops
    do not travel outward: every loop stays in the family and changes size on its own, so they cross each
    other all the time. The solid loops breathe slowly (a wave of 2 to 4 s, about a quarter of their size);
    the dotted ones pump: they go nearly all the way back to the detector and out again in about two
    seconds, through the solid ones. Sines with unrelated periods: smooth, and never the same twice in a
    show."""
    q = (np.arange(n) + 0.5) / n
    w = 2 * np.pi * (rate / BREATH) * t
    h = [B.rnd(n, 16 * seed + c, 0) for c in range(6)]         # (hash01 would give the three towers the same draw)
    slow = (0.2 * np.sin(w * (0.28 + 0.2 * h[0]) + 2 * np.pi * h[1])
            + 0.09 * np.sin(w * (0.55 + 0.3 * h[2]) + 2 * np.pi * h[3])
            + 0.08 * np.sin(w * (0.07 + 0.08 * h[4]) + 2 * np.pi * h[5]))
    pump = (0.54 + 0.46 * np.sin(w * (0.42 + 0.2 * h[0]) + 2 * np.pi * h[1])
            + 0.05 * np.sin(w * (0.9 + 0.4 * h[2]) + 2 * np.pi * h[3]))
    swell = 0.035 * math.sin(w * 0.137 + 0.01 * seed) + 0.02 * math.sin(w * 0.31 + 0.017 * seed)
    return np.where(q < SOLID, 1.0 + slow, pump) + swell


def _aspect(n, t, seed, rate=BREATH):
    """Width factor of each loop of a bloom at time t: in the TouchDesigner scene a bloom goes from tall eggs
    to wide bowls and back. One slow movement for the whole bloom (the loops stay nested), a little of its
    own for each loop."""
    w = 2 * np.pi * (rate / BREATH) * t
    own = np.sin(w * (0.19 + 0.12 * B.rnd(n, 16 * seed + 8, 0)) + 2 * np.pi * B.rnd(n, 16 * seed + 9, 0))
    return 0.93 + 0.13 * math.sin(w * 0.083 + 0.02 * seed) + 0.07 * math.sin(w * 0.21 + 0.013 * seed) + 0.05 * own


def loops(f, ox, oy, t, n, s_ref, s_open, seed, base, rate=BREATH, layer="r", reach=None, s_min=S_MIN, weight=1.0,
          m_min=(48, 60)):
    """The loops of a bloom, tangent to the point (ox, oy): the animation of the bloom on top of a tower, for
    any bloom of the show. n loops that stay in their place: the solid inner ones breathe, the dotted outer
    ones pump through them, the whole family goes from tall to wide and back (_breath, _aspect). Nothing is
    switched on or off: a loop only changes size.
    s_ref = rest size (px) of the outermost loop now, s_open = the same for the bloom fully open (it sets the
    number of dots, so that they do not change while it opens); base = level; s_min = size of the innermost
    loop; weight scales the line weights, the dots and their spacing (1 = a tower bloom); m_min = the least
    number of points of a solid and of a dotted loop."""
    q = (np.arange(n) + 0.5) / n
    s = loop_size(q * n, n, REST * s_ref, s_min) * _breath(n, t, seed, rate) * min(1.0, s_ref / (3.0 * s_min))
    wide = _aspect(n, t, seed, rate)
    if reach is not None:                                           # opening: they grow out behind the burst
        s = s * (1.0 - np.exp(-max(float(reach), 0.0) / (GROW * np.maximum(s, 1e-3))))
    s_full = loop_size(q * n, n, REST * s_open, s_min)              # ... of a fully open bloom: sets the dot counts
    for j in range(n):
        sj = float(s[j])
        if sj < 0.6:                                  # still inside the detector (under its red dot)
            continue
        # its shape follows its size: a circle near the detector, a wide petal far out
        jq = n * float(np.clip((sj - s_min) / max(REST * s_ref - s_min, 1.0), 0.0, 1.0)) ** (1.0 / SPREAD)
        if q[j] < SOLID:                              # solid inner loops
            m = int(np.clip(2 * math.pi * sj * 1.2 / 5.0, m_min[0], 800))
            phi = np.linspace(0, 2 * np.pi, m + 1)
            x, y = _loop(sj, jq, n, phi, t, seed, wide=float(wide[j]), j_noise=j)
            f.polyline(layer, ox + x, oy + y, base * (1.3 - 0.55 * q[j]), width=(3.4 - 1.6 * q[j]) * weight)
        else:                                         # dotted outer loops: always the same number of dots
            out = (q[j] - SOLID) / (1.0 - SOLID)
            m = int(np.clip(2 * math.pi * float(s_full[j]) * 1.2 / ((7.0 + 14.0 * (q[j] - SOLID)) * weight), m_min[1], 900))
            phi = (np.arange(m) + ((t * 0.4 * (1 if j % 2 else -1)) % 1.0)) / m * 2 * np.pi
            x, y = _loop(sj, jq, n, phi, t, seed, wide=float(wide[j]), j_noise=j, wander=0.27)
            crowd = min(1.0, 2 * math.pi * sj * 1.2 / m / (5.0 * weight))      # dots closer than their size: no brighter than a line
            f.dots(layer, ox + x, oy + y, (2.5 - 0.6 * q[j]) * weight,
                   1.5 * base * (1.3 - 0.55 * q[j]) * (1.0 - 0.55 * out ** 1.3) * crowd)


def bloom(f, tw, t, det, gain=1.0, size=1.0, clip=None, rate=BREATH, seed=None, layer="r", open_=None, reach=None):
    """The red bloom on top of a tower: a family of nested loops, all tangent to the detector (the
    TouchDesigner look). The loops stay in their place and breathe, each one on its own (see _breath; `rate`
    scales how fast). Nothing is ever switched on or off: a loop only changes size.
    `open_` (0..1.5) is how open the bloom is (its size is proportional to it: open_scale); by default it
    follows the recent hits of the tower. As it goes to 0 the loops go back into the detector.
    `reach` (px) is how far the burst has travelled since the bloom opened (SPEED * seconds): every loop
    grows out of the detector behind it, the small ones done first. None = grown."""
    key = tw.key
    seed = seed if seed is not None else {"L": 101, "C": 202, "R": 303}[key]
    env = energy_env(det, key, t, attack=ATTACK) if open_ is None else open_
    if env <= 0.0 or gain <= 0.0:
        return
    age, e_hit = det.last(key, t, echoes=True)
    n = N_LOOPS[key]
    ox, oy = tw.det
    s_ref = S_MAX[key] * size * open_scale(env)                     # rest size of the outermost loop
    # a hit is answered by the whole bloom at once: it brightens (and swells, through `open_`)
    hot = e_hit * math.exp(-age / 0.7) * (1.0 - math.exp(-age / 0.05)) if age < 5.0 else 0.0
    base = gain * (0.55 + 0.55 * min(1.0, env)) * (1.0 + 0.8 * hot)
    if clip:
        f.set_clip(*clip)
    loops(f, ox, oy, t, n, s_ref, S_MAX[key] * size, seed, base, rate=rate, layer=layer, reach=reach)
    f.dots(layer, [ox], [oy - 3], 10.0, 1.2 * gain * (0.5 + 0.5 * min(1.0, env)) * min(1.0, env / 0.05))
    f.dots("w", [ox], [oy - 3], 3.0, 1.0 * gain * min(1.0, env / 0.05))
    if clip:
        f.set_clip()


BURST_WIDE = 0.62               # the compact burst is a plume: narrower than tall, it stays over its own tower


def burst(f, tw, t, det, gain=1.0, clip=None, layer="r", size=1.0):
    """Compact reply of a tower inside another scene: a few loops thrown out of the detector by each hit,
    then gone. About +-120 px wide and 500 px tall for a full-energy hit at size 1."""
    tt, ee, ec = det.hits(tw.key, t - 2.6, t + 1e-6)
    if not len(tt):
        return
    ox, oy = tw.det
    if clip:
        f.set_clip(*clip)
    for th, e, echo in zip(tt, ee, ec):
        a = t - th
        g = gain * e * (0.45 if echo else 1.0) * math.exp(-a / 0.9) * float(1.0 - smoothstep(1.9, 2.6, a))
        if g < 0.004:
            continue
        n = 3 if echo else 6
        s_end = size * (18.0 + 170.0 * (0.4 + e))                   # where the outermost loop ends up
        for j in range(n):
            s = s_end * (1 - math.exp(-a / 0.45)) * (j + 1) / n     # every loop leaves the detector, and brakes
            if s < 0.6:
                continue
            if j < n - 2:
                m = int(np.clip(s * 1.6, 40, 400))
                x, y = _loop(s, j * 1.6, 12, np.linspace(0, 2 * np.pi, m + 1), t, 500 + j, wide=BURST_WIDE)
                f.polyline(layer, ox + x, oy + y, 1.1 * g, width=L.LW_BOLD - 0.3 * j)
            else:                                     # dotted: the same dots from the detector to the end
                m = int(np.clip(s_end * (j + 1) / n * 0.8, 20, 200))
                x, y = _loop(s, j * 1.6, 12, np.arange(m) / m * 2 * np.pi, t, 500 + j, wide=BURST_WIDE)
                f.dots(layer, ox + x, oy + y, 1.6, 1.3 * g * min(1.0, 2 * math.pi * s * 1.1 / m / 4.0))
        f.dots(layer, [ox], [oy - 3], 8.0, 1.4 * g)
    if clip:
        f.set_clip()


SLANT = {"L": 0.42, "C": -0.62, "R": -0.42}       # tan(zenith) of the arrival drawn on each tower


def incoming(f, tw, t, t_hit, slant=None, approach=6.0, label=True):
    """A muon on its way to a detector: drawn only for `approach` seconds before t_hit (and 0.35 s after).
    Dashed line of where it will go, red track so far, head with a pulsing ring, INCOMING tag with the ETA."""
    a = t - (t_hit - approach)
    if not (0.0 <= a < approach + 0.35):
        return
    slant = SLANT.get(tw.key, 0.4) if slant is None else slant
    ox, oy = tw.det
    x_top = ox - slant * (oy - L.FY0)
    u = min(1.0, a / approach) ** 1.7
    hx, hy = x_top + (ox - x_top) * u, L.FY0 + (oy - L.FY0) * u
    fade = 1.0 if a < approach else 1.0 - (a - approach) / 0.35
    f.set_clip(L.FX0, L.FY0, L.FX1, L.FY1)
    k = np.arange(0, 1.0, 0.04)
    f.segments("r", x_top + (ox - x_top) * k, L.FY0 + (oy - L.FY0) * k, x_top + (ox - x_top) * (k + 0.02),
               L.FY0 + (oy - L.FY0) * (k + 0.02), 0.45 * fade)
    f.segments("r", [x_top], [L.FY0], [hx], [hy], 1.2 * fade, width=L.LW_BOLD)
    if a < approach:
        f.dots("r", [hx], [hy], 6.0, 1.8)
        f.dots("w", [hx], [hy], 2.2, 1.2)
        f.rings("r", [hx], [hy], [14 + 10 * math.sin(t * 9.0) ** 2], 0.8, width=L.LW)
        if label:
            side = -1 if slant > 0 else 1
            hud.callout(f, hx, hy, side * 60, -30, "INCOMING", [f"ETA {t_hit - t:04.2f} S", f"TARGET {L.NAMES[tw.key]}"],
                        red=True, side=side, build=B.io(a, approach - a, out=0.25))
    f.set_clip()


# ----------------------------------------------------------------------------
# strings (the white flow lines between the towers)
# ----------------------------------------------------------------------------

STRING_AMP = (30.0, 46.0)       # px rms of the noise in the middle of a span: along the line, across it ...
STRING_STACK = 480.0            # ... for a stack of lines this tall; a tighter stack (short towers) moves less
STRING_FAN = 2.2                # at a frame edge the stack opens to at most this many times its height on the tower
STRING_REST = 0.6               # how far the lines of a tower reach towards a neighbour that is not on yet
STRING_GROW = 2.6               # s a line takes to grow out of its tower (the lowest line leaves 0.6 s after the top one)
STRING_TIP = 0.12               # length of the soft tip of a line that is growing (fraction of the span)
EDGE_OUT = 150.0                # px: at a frame edge the lines go on behind the frame (their end there is free)
_WAVES = {}


def _waves(b, seed, K):
    """The noise of span b: 2 K plane waves (K for the displacement along the line, K across it), each one
    (wave number along the line in rad / px, wave number over the stack of lines in rad, pulsation in
    rad / s, phase, amplitude). Constants: made once."""
    if (b, seed, K) not in _WAVES:
        k = np.arange(2 * K)
        h = [hash01(k, 16 * b + c, seed) for c in range(7)]
        lo = np.where(k < K, 210.0, 170.0)                              # (longer waves along the line: few folds)
        lam = lo * (720.0 / lo) ** h[0]                                 # wavelength along the line, px
        kx = 2 * np.pi / lam * np.where(h[1] < 0.5, -1.0, 1.0)
        kv = 2 * np.pi * 1.8 * (h[2] - 0.5)                             # up to 0.9 cycle across the stack
        om = 2 * np.pi * (0.42 + 0.58 * h[3]) * np.where(h[4] < 0.5, -1.0, 1.0)
        amp = (lam / 720.0) ** 0.6
        for c in (slice(0, K), slice(K, 2 * K)):
            amp[c] /= math.sqrt(float((amp[c] ** 2).sum()) / 2.0)       # unit rms for each component
        _WAVES[(b, seed, K)] = (kx, kv, om, 2 * np.pi * h[5], amp)
    return _WAVES[(b, seed, K)]


def _string_field(b, x, v, t, seed, K=12):
    """Smooth noise field of span b at time t, sampled where the points of its lines are: x (px along the
    line, the same for every line) and v (place of a line in the stack, 0..1). Returns (dx, dy), each
    (len(v), len(x)), unit rms. One field for all the lines: neighbours get nearly the same displacement."""
    kx, kv, om, ph, amp = _waves(b, seed, K)
    al = kx[:, None] * x[None, :]
    be = kv[:, None] * v[None, :] + (om * t + ph)[:, None]
    sa, ca = np.sin(al), np.cos(al)
    sb, cb = amp[:, None] * np.sin(be), amp[:, None] * np.cos(be)       # sin(al + be) = sa cb + ca sb
    return tuple(cb[c].T @ sa[c] + sb[c].T @ ca[c] for c in (slice(0, K), slice(K, 2 * K)))


def string_range(tw, n=27, y_span=(700.0, 1170.0)):
    """(top, bottom) y of the stack of n lines on a tower: along its shaft, from under the detector down to
    the bottom band. On a short tower the shaft has no room for them: the stack also takes the sides of
    the detector (the lines then leave the whole height of the tower that stands above the bottom band)."""
    hi = min(tw.bot - 120.0, y_span[1] + 10.0)
    lo = tw.top + tw.det_h + 14.0
    if hi - lo < 8.0 * n:
        lo = min(max(tw.top + 8.0, hi - 8.0 * n), hi - 10.0)
    return lo, hi


def strings(f, towers, t, det, n=27, gain=1.0, power=None, y_span=(700.0, 1170.0), seed=7, pts=220, y_clip=1196.0,
            kick=0.0, age=None, hold=None):
    """Lines strung between neighbouring towers (and out to the frame), made the way the TouchDesigner scene
    makes them: a line is a row of points, and a smooth noise is added to the position of every point - one
    field for all the lines of a span, so neighbours move together and fold together. The amplitude of the
    noise follows the distance from the ends: 0 on a tower, 1 in the middle of the span (and at a frame
    edge, where the line is free). Nothing else moves the lines: a hit only sends a glow along the lines of
    its tower, `kick` (the drum pulse of the music, 0..1.5) only makes them a little brighter.
    `age` = {key: seconds since that tower came on} (None: the lines are all there): the lines GROW out of
    their tower, the top one first, as far as STRING_REST of the way to a neighbour that is not on yet, and
    go on to meet its lines when it comes on.
    `power` = {key: 0..1} is the brightness of the lines of each tower (None = 1); without `age` it gates
    them as it always did (a line between a lit tower and a dark one dims along its way).
    `hold` = {key: {"l" or "r": (x, y_top, y_bottom)}} moves where the lines of one side of a tower are held
    (default: its edge, over string_range): a scene that puts a plate beside a tower holds them on it."""
    tws = [towers[k] for k in L.ORDER]
    posts = [None] + tws + [None]
    v = (np.arange(n) + 0.5) / n
    u = np.linspace(0.0, 1.0, pts)
    lift = 1.0 + 0.15 * min(1.0, max(0.0, kick))

    def attach(P, side, other=None):
        """Where the lines are held at one end of a span: (x, y of each line, room per line in px).
        P = None: the frame edge (other = the tower at the other end of the span): the lines are free there,
        the stack opens a little, from the bottom, and stays under the bloom of that tower."""
        if P is None:
            tall = min(y_span[1] - y_span[0], STRING_FAN * n * other[2])
            return (L.FX0 - EDGE_OUT if side == "r" else L.FX1 + EDGE_OUT), y_span[1] - tall * (1.0 - v), 99.0
        x, (lo, hi) = (P.x1 if side == "r" else P.x0), string_range(P, n, y_span)
        if hold and side in hold.get(P.key, {}):
            x, lo, hi = hold[P.key][side]
        return x, lo + (hi - lo) * v ** 0.95, (hi - lo) / n

    def grown(P, tt):
        """How much of its growth each line of tower P has done at time tt (0..1 per line)."""
        if age is None or P is None:
            return np.ones(n)
        return smoothstep(0.0, 1.0, (age.get(P.key, 99.0) - (t - tt) - 0.6 * v) / STRING_GROW).astype(np.float64)

    def reach(P, O, tt):
        """How far the lines of P have got towards O (fraction of the span, per line)."""
        g = grown(P, tt)
        return g * (STRING_REST + (1.0 - STRING_REST) * grown(O, tt)) if O is not None else g

    f.set_clip(L.FX0, L.FY0, L.FX1, y_clip)
    for b in range(len(posts) - 1):
        P0, P1 = posts[b], posts[b + 1]
        pw = [1.0 if power is None else float(power.get(P.key, 0.0)) for P in (P0, P1) if P is not None]
        if age is None and max(pw) < 0.02:
            continue
        if age is not None and max(age.get(P.key, 99.0) for P in (P0, P1) if P is not None) <= 0.0:
            continue
        if P0 is None:
            e1 = attach(P1, "l")
            e0 = attach(None, "r", e1)
        else:
            e0 = attach(P0, "r")
            e1 = attach(P1, "l") if P1 is not None else attach(None, "l", e0)
        (xa, ya, ra), (xb, yb, rb) = e0, e1
        if xb - xa < 60.0:
            continue
        x = xa + u * (xb - xa)
        su = u * u * (3 - 2 * u)
        y = ya[:, None] + (yb - ya)[:, None] * su[None, :]
        # amplitude of the noise: 0 on a tower, 1 in the middle of the span / at the free end
        if P0 and P1:
            env0 = np.sin(np.pi * u) ** 0.6
        else:
            env0 = np.sin(0.5 * np.pi * (1.0 - u if P1 else u)) ** 0.6
        env = env0
        # ... in proportion to the room the lines have: towers standing close, or a stack held on a short
        # tower, make smaller folds (the lines keep their place in the stack)
        tall = np.abs(ya[-1] - ya[0]) + (np.abs(yb[-1] - yb[0]) - np.abs(ya[-1] - ya[0])) * su
        env = env * min(1.0, (xb - xa) / 760.0) * np.minimum(1.0, tall / STRING_STACK)
        dx, dy = _string_field(b, x, v, t, seed)
        X = x[None, :] + STRING_AMP[0] * env[None, :] * dx
        Y = y + STRING_AMP[1] * env[None, :] * dy
        # what is there of each line (n, pts)
        heads = []
        if age is None:
            pa, pb = (pw[0], pw[-1]) if (P0 and P1) else (pw[0], pw[0])
            there = np.broadcast_to(pa + (pb - pa) * smoothstep(0.25, 0.75, u), (n, pts))
        else:
            there = np.zeros((n, pts))
            for P, O, d in ((P0, P1, u), (P1, P0, 1.0 - u)):
                if P is None:
                    continue
                r = reach(P, O, t)
                tip = r * (1.0 + STRING_TIP)                        # the soft tip leaves the span when the line is whole
                part = np.clip((tip[:, None] - d[None, :]) / STRING_TIP, 0.0, 1.0)
                there = np.maximum(there, part * (1.0 if power is None else float(power.get(P.key, 0.0))))
                speed = (r - reach(P, O, t - 0.03)) / 0.03          # spans per second: the head shows while it moves
                heads.append((tip if P is P0 else 1.0 - tip, np.clip(speed / 0.25, 0.0, 1.0)))
        # the glow a hit sends along the lines of its tower
        glow = np.zeros(pts)
        for P, x_edge in ((P0, xa), (P1, xb)):
            if P is None:
                continue
            tt, ee, ec = det.hits(P.key, t - 2.4, t + 1e-6)
            for th, e, echo in zip(tt, ee, ec):
                a = t - th
                glow += ((0.4 if echo else 1.0) * e * math.exp(-a / 0.9) * (1.0 - math.exp(-a / 0.06))
                         * np.exp(-((np.abs(x - x_edge) - SPEED * a) / 120.0) ** 2))
        line = gain * lift * (0.5 + 0.3 * hash01(np.arange(n), b, seed + 1))
        # brighter where they leave a tower - unless they are crowded there (a short tower): no white knot
        near = 0.5 * (min(1.0, ra / 10.0) * (1.0 - su) + min(1.0, rb / 10.0) * su)
        I = there * line[:, None] * ((0.75 + near * (1.0 - env0)) * (1.0 + 1.2 * glow))[None, :]
        f.segments("w", X[:, :-1].ravel(), Y[:, :-1].ravel(), X[:, 1:].ravel(), Y[:, 1:].ravel(),
                   I[:, :-1].ravel(), I[:, 1:].ravel(), width=L.LW)
        for ut, hi in heads:                                        # a line is drawn by its head
            m = (hi > 0.02) & (ut > 0.0) & (ut < 1.0)
            if not m.any():
                continue
            fi = ut[m] * (pts - 1)
            i0 = np.minimum(fi.astype(int), pts - 2)
            fr = fi - i0
            rows = np.nonzero(m)[0]
            hx = X[rows, i0] * (1 - fr) + X[rows, i0 + 1] * fr
            hy = Y[rows, i0] * (1 - fr) + Y[rows, i0 + 1] * fr
            f.dots("w", hx, hy, 2.2, 1.5 * gain * hi[m])
    f.set_clip()


# ----------------------------------------------------------------------------
# scopes + generic overlay
# ----------------------------------------------------------------------------

def scope_age(key, t, lag=0.0):
    """Build age of the scope of a tower: it is constructed when the detectors are revealed - it never fades
    (the outro takes it apart under the credits)."""
    return t - sd.T_REVEAL - lag


def scopes(f, ctx, t, alpha=1.0, span=3.0):
    """One small oscilloscope per tower, right of its base: the float it streams (what OSC will carry)."""
    for k, key in enumerate(L.ORDER):
        if key not in ctx.slots["scopes"]:
            continue
        rect = ctx.slots["scopes"][key]
        tt = t - span + np.linspace(0.0, span, 150)
        v = ctx.det.value(key, tt) if ctx.det.online(key, t) else 0.02 + 0.0 * tt
        age, e = ctx.det.last(key, t, echoes=True)
        hot = e * math.exp(-age / 0.4) if age < 3 and ctx.det.online(key, t) else 0.0
        with f.build(scope_age(key, t, 0.35 + 0.2 * k), (rect[0] - 26.0, rect[1] - 24.0, rect[2] + 4.0, rect[3] + 4.0),
                     wave=0.3, key=20 + k):
            hud.scope(f, rect, v, f"DETECTOR {k + 1}", f"/MUON/{key} {float(v[-1]):.2f}", alpha=alpha, hot=hot)


def dark(f, tw, outline=0.16):
    """An unlit tower. It stands in front of the wall for the whole show: whatever is drawn in its rectangle
    never reaches the wall behind it, so black it out (a faint outline keeps it readable in the previews)."""
    f.occlude(tw.x0, tw.top, tw.x1, tw.bot)
    if outline > 0:
        f.rect("w", tw.x0, tw.top, tw.x1, tw.bot, outline, width=L.LW)


def overlay(f, ctx, t, blooms=True, clip=None, dim=0.3, gain=1.0, texture=True, size=1.0, outline=0.16):
    """Towers on top of a scene that is not about them: faces + compact blooms on every hit. Before the
    detectors are revealed (or when one is powered down) the tower is dark."""
    clip = clip or (L.FX0, L.HEAD_Y + 6, L.FX1, L.FY1)
    for key in L.ORDER:
        tw = ctx.towers[key]
        p = ctx.det.power(key, t)
        if p <= 0.01:
            dark(f, tw, outline=outline)
            continue
        age, e = ctx.det.last(key, t, echoes=True)
        v = float(ctx.det.value(key, t)) if ctx.det.online(key, t) else 0.0
        face(f, tw, t, power=p, value=v, hit_age=age, hit_e=e, dim=dim, texture=texture)
        if blooms and ctx.det.online(key, t):
            burst(f, tw, t, ctx.det, gain=gain, clip=clip, size=size)
