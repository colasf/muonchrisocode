"""The three detector towers: their faces, their blooms, the strings between them, their scopes.

Everything here is driven by (a) the tower rectangles from layout / towers.json and (b) the detector
streams (showdata.Detectors): one float per tower, a hit = a rising edge, its peak = the energy.
"The more powerful the muon, brighter, louder, the reply."
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
SOLID = 0.56                    # loops below this fraction are solid lines (the bright bulb), the rest are dotted
SPEED = 760.0                   # px / s of the pulse a hit sends through the loops and the strings


def loop_size(jf, n, s_max):
    """Size of loop number jf (fractional while it grows): a tight bulb on the detector, wide loops outside."""
    return S_MIN + (s_max - S_MIN) * (np.maximum(jf, 0.0) / n) ** 1.9


def open_scale(open_):
    """Size factor of a bloom for a given openness (0..1.2): the stronger the muons, the larger the bloom."""
    return 0.35 + 0.65 * min(1.2, max(0.0, open_)) / 1.2


def bloom_height(key, open_, size=1.0):
    """Height (px) of the outermost loop above the detector for a given openness - for scales and labels."""
    return 2.0 * 0.9 * S_MAX[key] * size * open_scale(open_)


def energy_env(det, key, t, tau=1.1):
    """Slow envelope of the recent hits of a tower (0..~1.5): how 'open' its bloom is."""
    tt, ee, _ = det.hits(key, t - 6 * tau, t + 1e-6)
    if not len(tt):
        return 0.0
    return float((ee * np.exp(-(t - tt) / tau)).sum())


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
        rh, n_c = 10.0, 9
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
    n_s = 4
    sy = tw.top + 22 + np.arange(n_s) * (tw.det_h - 44) / (n_s - 1)
    f.segments("w", np.full(n_s, tw.x0 + 14), sy, np.full(n_s, tw.x1 - 14), sy, (0.55 + 0.4 * (not on)) * p,
               width=L.LW)
    if flash > 0.02:
        f.rects("r", tw.x0 + 4, tw.top + 4, tw.x1 - 4, yd - 3, 0.9 * flash)
        f.segments("r", np.full(n_s, tw.x0 + 14), sy, np.full(n_s, tw.x1 - 14), sy, 1.6 * flash, width=L.LW_BOLD)
    if not on:              # waiting: a bracket highlights the detector at the head of the tower
        b = 16.0
        blink = 0.55 + 0.45 * math.sin(t * 5.0)
        for sx, x in ((-1, tw.x0 - b), (1, tw.x1 + b)):
            f.segments("r", [x, x, x], [tw.top - b, tw.top - b, yd + b], [x, x - sx * 22, x - sx * 22],
                       [yd + b, tw.top - b, yd + b], 0.9 * p / 0.35 * blink * 0.6, width=L.LW)
    if label:                   # its name is decoded when the towers come on (the reveal, and again at T_BACK)
        name = B.resolve(L.NAMES[tw.key], t - (sd.T_BACK if t >= sd.T_BACK else sd.T_REVEAL), 30.0, 0.25, pad=True,
                         key=ord(tw.key))
        f.text("r" if flash > 0.2 else "w", tw.cx, tw.top - 14, name, size=L.T_MICRO,
               alpha=0.85 * p / max(p, 0.35) if on else 0.8, anchor="ms")


# ----------------------------------------------------------------------------
# bloom
# ----------------------------------------------------------------------------

def _loop(s, j_frac, n, phi, t, seed, wide=1.0):
    """One loop of a bloom, tangent to the detector at phi = 0. Returns local (x, y), y up = negative."""
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
    amp = 0.36 * q ** 1.7
    k = np.arange(1, 7)[:, None]
    ph = hash01(k, seed, 3) * 2 * np.pi + t * (0.16 + 0.11 * hash01(k, seed, 5)) * np.where(k % 2 == 0, 1.0, -1.0)
    ph = ph + 0.55 * j_frac * (hash01(k, seed, 9) - 0.5)
    nz = (np.sin(k * phi[None, :] + ph) / k ** 0.8).sum(0)
    pin = np.sin(phi / 2) ** 2
    cx, cy = 0.0, -s * wy
    dx, dy = x - cx, y - cy
    r = np.maximum(np.hypot(dx, dy), 1e-6)
    d = s * amp * nz * pin
    return x + dx / r * d, y + dy / r * d


def bloom(f, tw, t, det, gain=1.0, size=1.0, clip=None, rate=0.22, seed=None, layer="r", open_=None, reach=None):
    """The red bloom on top of a tower: nested loops growing out of the detector (the TouchDesigner look).
    `open_` (0..1.5) is how open it is; by default it follows the recent hits of the tower.
    `reach` (px) hides the loops larger than it: pass SPEED * (time since the hit) to make a bloom burst out."""
    key = tw.key
    seed = seed if seed is not None else {"L": 101, "C": 202, "R": 303}[key]
    env = energy_env(det, key, t) if open_ is None else open_
    age, e_hit = det.last(key, t, echoes=True)
    n = N_LOOPS[key]
    ox, oy = tw.det
    s_max = S_MAX[key] * size * open_scale(env)
    frac = (t * rate) % 1.0
    front = SPEED * age if age < 3.0 else 1e9
    if clip:
        f.set_clip(*clip)
    for j in range(n + 1):
        jf = j + frac
        q = jf / n
        if q >= 1.0:
            continue
        s = float(loop_size(jf, n, s_max))
        fade_in = min(1.0, jf / 1.2)
        fade_out = 1.0 - smoothstep(0.7, 1.0, q)
        pulse = e_hit * math.exp(-((s - front) / (0.3 * front + 26.0)) ** 2) * math.exp(-age / 1.4) if age < 3 else 0.0
        inten = gain * fade_in * fade_out * (0.55 + 0.55 * min(1.0, env)) * (1.3 - 0.55 * q) + 1.5 * pulse * gain
        if reach is not None:
            inten *= float(smoothstep(reach + 20.0, reach - 40.0, s))
        if inten < 0.02:
            continue
        per = 2 * math.pi * s * 1.2
        if q < SOLID:                                 # solid inner loops
            m = int(np.clip(per / 5.0, 48, 800))
            phi = np.linspace(0, 2 * np.pi, m + 1)
            x, y = _loop(s, jf, n, phi, t, seed)
            f.polyline(layer, ox + x, oy + y, inten, width=3.4 - 1.6 * q)
        else:                                         # dotted outer loops
            gap = 8.0 + 20.0 * (q - SOLID)
            m = int(np.clip(per / gap, 60, 900))
            phi = (np.arange(m) + ((t * 0.4 * (1 if j % 2 else -1)) % 1.0)) / m * 2 * np.pi
            x, y = _loop(s, jf, n, phi, t, seed)
            f.dots(layer, ox + x, oy + y, 2.3 - 0.7 * q, 1.5 * inten)
    f.dots(layer, [ox], [oy - 3], 10.0, 1.2 * gain * (0.5 + 0.5 * min(1.0, env)))
    f.dots("w", [ox], [oy - 3], 3.0, 1.0 * gain)
    if clip:
        f.set_clip()


BURST_WIDE = 0.62               # the compact burst is a plume: narrower than tall, it stays over its own tower


def burst(f, tw, t, det, gain=1.0, clip=None, layer="r", size=1.0):
    """Compact reply of a tower inside another scene: a few loops thrown out by each hit, then gone.
    About +-120 px wide and 500 px tall for a full-energy hit at size 1."""
    tt, ee, ec = det.hits(tw.key, t - 2.6, t + 1e-6)
    if not len(tt):
        return
    ox, oy = tw.det
    if clip:
        f.set_clip(*clip)
    for th, e, echo in zip(tt, ee, ec):
        a = t - th
        g = gain * e * (0.45 if echo else 1.0) * math.exp(-a / 0.9)
        if g < 0.03:
            continue
        n = 3 if echo else 6
        for j in range(n):
            s = size * (18.0 + 170.0 * (0.4 + e) * (1 - math.exp(-a / 0.45))) * (j + 1) / n
            m = int(np.clip(s * 1.6, 40, 400))
            phi = np.linspace(0, 2 * np.pi, m + 1)
            x, y = _loop(s, j * 1.6, 12, phi, t, 500 + j, wide=BURST_WIDE)
            if j < n - 2:
                f.polyline(layer, ox + x, oy + y, 1.1 * g, width=L.LW_BOLD - 0.3 * j)
            else:
                f.dots(layer, ox + x[::2], oy + y[::2], 1.6, 1.3 * g)
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

def strings(f, towers, t, det, n=27, gain=1.0, power=None, y_span=(700.0, 1170.0), seed=7, pts=220, y_clip=1196.0,
            kick=0.0):
    """Lines strung between neighbouring towers (and out to the frame). They breathe like a terrain;
    every hit plucks them: a wave leaves the tower and travels along its strings. `kick` (0..1.5, the drum
    pulse of the music) shivers the fine ripples."""
    tws = [towers[k] for k in L.ORDER]
    posts = [None] + tws + [None]
    f.set_clip(L.FX0, L.FY0, L.FX1, y_clip)

    def attach(P, v):
        if P is None:
            return y_span[0] + (y_span[1] - y_span[0]) * v
        a = P.top + P.det_h + 14.0
        return a + (min(P.bot - 120.0, y_span[1] + 10.0) - a) * v ** 0.95
    for b in range(len(posts) - 1):
        A, B = posts[b], posts[b + 1]
        xa = A.x1 if A else L.FX0
        xb = B.x0 if B else L.FX1
        pa = 1.0 if power is None else (power.get(A.key, 0.0) if A else power.get(B.key, 0.0))
        pb = 1.0 if power is None else (power.get(B.key, 0.0) if B else power.get(A.key, 0.0))
        if max(pa, pb) < 0.02:
            continue
        u = np.linspace(0.0, 1.0, pts)
        x = xa + u * (xb - xa)
        # displacement envelope: pinned on the towers, free on the frame edges
        if A and B:
            env = np.sin(np.pi * u) ** 0.7
        elif A is None:
            env = (0.35 + 0.65 * np.sin(np.pi * (1 - u) / 2) ** 0.8) * (1 - u ** 6)
        else:
            env = (0.35 + 0.65 * np.sin(np.pi * u / 2) ** 0.8) * (1 - (1 - u) ** 6)
        gate = pa + (pb - pa) * smoothstep(0.25, 0.75, u)
        for i in range(n):
            v = (i + 0.5) / n
            ya, yb = attach(A, v), attach(B, v)
            y = ya + (yb - ya) * (u * u * (3 - 2 * u))
            ph = hash01(b, np.arange(6), seed) * 2 * np.pi
            w1 = 44.0 * np.sin(2 * np.pi * (1.05 * u + 0.16 * v) + ph[0] + 0.21 * t)
            w2 = 24.0 * np.sin(2 * np.pi * (2.3 * u - 0.27 * v) + ph[1] - 0.33 * t)
            w3 = (10.0 + 26.0 * min(1.5, kick)) * np.sin(2 * np.pi * (5.1 * u + 0.5 * v) + ph[2] + 0.5 * t
                                                         + (3.0 * i if kick > 0.02 else 0.0))
            squeeze = (v - 0.5) * 150.0 * np.sin(2 * np.pi * (0.75 * u) + ph[3] + 0.13 * t) * np.sin(
                2 * np.pi * (1.9 * u) + ph[4] - 0.09 * t)
            fold = 24.0 * np.sin(2 * np.pi * (3.0 * u + 1.4 * v) + ph[5]) * np.sin(2 * np.pi * (0.5 * u) + 0.4 * t) ** 3
            d = (w1 + w2 + w3 + squeeze + fold) * env
            # plucks
            for P, x_edge, sgn in ((A, xa, 1.0), (B, xb, -1.0)):
                if P is None:
                    continue
                tt, ee, ec = det.hits(P.key, t - 2.4, t + 1e-6)
                for th, e, echo in zip(tt, ee, ec):
                    a = t - th
                    dist = (x - x_edge) * sgn
                    amp = (0.4 if echo else 1.0) * e * 85.0 * math.exp(-a / 0.9)
                    d = d + amp * np.exp(-((dist - SPEED * a) / 110.0) ** 2) * np.sin((dist - SPEED * a) / 26.0) * (
                        1 if i % 2 else -1) * np.minimum(1.0, dist / 60.0)
            inten = gain * gate * (0.34 + 0.3 * hash01(i, b, seed + 1)) * (0.75 + 0.5 * (1 - env))
            f.polyline("w", x, y + d, inten, width=L.LW)
    f.set_clip()


# ----------------------------------------------------------------------------
# scopes + generic overlay
# ----------------------------------------------------------------------------

def scope_age(key, t, lag=0.0):
    """Build age of the scope of a tower: it is constructed when the detectors are revealed (and again when
    they come back, T_BACK) and taken apart when its tower powers down - it never fades."""
    if t >= sd.T_BACK:
        return t - sd.T_BACK - lag
    return B.io(t - sd.T_REVEAL - lag, sd.T_OFF[key] + 0.45 - t, out=0.45)


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
