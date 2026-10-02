"""GLITCH - the groove breaks.   Sheet scene 7, 06:55 - 07:34.

  7.0  06:55 - 07:02.7  BREAK BEFORE DRUMS   "almost nothing." (07:00): the elevation of the DANCE scene,
                                              emptied. One muon, alone, falls for the whole break; it reaches
                                              the ground when the drums come back.
  7.1  07:02.7 - 07:27  DRUMS IN             the dance grid keeps running (phrases 14-17) and is corrupted:
                                              slices of the picture pushed sideways, blocks repeated like a
                                              beat-repeat, barcode bands, hex dumps, inverted bands, pixel-sorted
                                              smears, bands turning red. The amount follows the kicks of the
                                              music; slices move on the eighth notes, bright bands only change on
                                              the beat (2 Hz, never a full-frame flash).
  7.3  07:27 - 07:34    TRANSITION           the picture runs out: the smears turn vertical, the showers fade,
                                              only single lanes are left, falling. Almost black: OUTLAST next
                                              (muons falling 15 km).
White and red only. The corruption is a post-process on the light layers (before the bloom), so the
towers, the frame and the HUD break with the picture: the wall itself glitches.

Layout: everything comes from the DANCE stage (particle column and view derived from the towers); the
lone muon of the break falls on ctx.focus, in the bay that hosts the one-centre compositions.

The tearing is a post-process and stays what it was. The data, here as everywhere, never fade in: in the
break the furniture of the DANCE scene is still there (dimmed, emptied) and what is new - the view tag, the
rules, the lone muon's call-out, the cursor of the strip, the counts - is constructed; when the drums come
back the grid builds each bar as in DANCE; in the transition the instrument does not fade out with the
picture, it is taken apart piece by piece (T_HUD_OUT), and the last read-out (LANES) is constructed.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import OrthoCamera, hash01, smoothstep
from ..show import Scene
from .dance import BEAT, N_PHRASES, PANEL_BLOCKS, PHRASE, T0, Y_GROUND, Dance, grid
from .shower import (altitude_rules, auto_callout, bottom_panels, draw_info, flow, info_layout, put_tag, put_text,
                     stage_for, tbox)

T_IN = 415.0
T_DRUMS = T0 + 13 * PHRASE          # 422.666: the kick that brings the drums back
T_TRANS = 447.0
T_OUT = 454.0
T_HUD_OUT = T_TRANS + 1.0           # the instrument is taken apart from here, one piece every half second
STEP = BEAT / 2                     # the glitch re-rolls on the eighth notes (bright bands: on the beat)
_HEX = "0123456789ABCDEF"


def _amount(t, ctx):
    """How broken the picture is, 0..1: follows the kicks and onsets of the music, dies in the transition."""
    k = min(1.5, ctx.cues.kick(t, tau=0.2))
    o = min(1.5, ctx.cues.onset(t, tau=0.12))
    a = float(np.clip(0.3 + 0.42 * k + 0.2 * o, 0.0, 1.0))
    return a * float(1.0 - smoothstep(T_TRANS, T_TRANS + 5.0, t))


def make_post(t, amt, vertical, view):
    """Corruption of the light stack `base` (2, H, W) [white, red].
    What only *moves* the picture (slices, repeats) is re-rolled on the eighth notes; what adds light
    (smears, red bands) only on the beat, so large bright areas never change faster than 2 times a second."""
    step = int(math.floor((t - T0) / STEP))
    beat = int(math.floor((t - T0) / BEAT))
    sub = (t - T0) / STEP - step                      # progress inside the step: slices keep sliding a little

    def post(base, fr):
        s = fr.s
        H, W = base.shape[1:]
        ya, yb = int(view[1] * s), int(view[3] * s)
        xa, xb = max(0, int(view[0] * s)), min(W, int(view[2] * s))
        # 1 - slices pushed sideways
        rng = np.random.default_rng((11, step))
        for i in range(int(28 * amt + 0.5)):
            h = max(2, int(rng.choice([4, 6, 10, 16, 28, 48, 90, 150]) * s))
            full = rng.random() < 0.18 * amt           # the big hits also tear the header and the bottom band
            y0 = int(rng.integers(0 if full else ya, (H if full else yb) - h))
            dx = int(rng.normal(0.0, 1.0) * (30 + 520 * amt) * s * (0.85 + 0.3 * sub))
            if dx:
                base[:, y0:y0 + h] = np.roll(base[:, y0:y0 + h], dx, axis=2)
        # 2 - beat repeat: a block of the picture stamped again and again along its row
        rng = np.random.default_rng((12, step))
        for i in range(int(6 * amt + 0.5)):
            h = max(4, int(rng.integers(24, 240) * s))
            y0 = int(rng.integers(ya, yb - h))
            w = max(4, int(rng.choice([16, 32, 64, 128, 256, 512]) * s))
            x0 = int(rng.integers(0, W - w))
            reps = int(rng.integers(3, 16))
            sgn = 1 if rng.random() < 0.5 else -1
            tile = base[:, y0:y0 + h, x0:x0 + w].copy()
            for r in range(1, reps):
                x = x0 + sgn * r * w
                if x < 0 or x + w > W:
                    break
                base[:, y0:y0 + h, x:x + w] = tile * (0.92 ** r)
        # 3 - pixel-sorted smears: the bright pixels of a band drag to the side (or down, in the transition)
        rng = np.random.default_rng((13, beat))
        for i in range(int(4 * amt + 5 * vertical + 0.5)):
            if rng.random() < 0.2 + 0.8 * vertical:
                w = max(4, int(rng.integers(14, 150) * s))
                x0 = int(rng.integers(xa, max(xa + 1, xb - w)))
                band = base[:, ya:yb, x0:x0 + w].astype(np.float64)
                g = (0.990 ** (np.arange(yb - ya) / s))[None, :, None]
                src = np.where(band > 0.35, band, 0.0)
                base[:, ya:yb, x0:x0 + w] = np.maximum(band, 0.6 * np.maximum.accumulate(src / g, axis=1) * g)
            else:
                h = max(3, int(rng.integers(8, 80) * s))
                y0 = int(rng.integers(ya, yb - h))
                band = base[:, y0:y0 + h].astype(np.float64)
                flip = rng.random() < 0.5
                if flip:
                    band = band[:, :, ::-1]
                g = (0.9955 ** (np.arange(W) / s))[None, None, :]
                src = np.where(band > 0.4, band, 0.0)
                out = np.maximum(band, 0.75 * np.maximum.accumulate(src / g, axis=2) * g)
                base[:, y0:y0 + h] = out[:, :, ::-1] if flip else out
        # 4 - bands turning red (white -> red), others losing their red
        rng = np.random.default_rng((14, beat))
        for i in range(int(4 * amt + 0.5)):
            h = max(2, int(rng.integers(6, 110) * s))
            y0 = int(rng.integers(ya, yb - h))
            if rng.random() < 0.7:
                base[1, y0:y0 + h] += base[0, y0:y0 + h]
                base[0, y0:y0 + h] = 0.0
            else:
                base[0, y0:y0 + h] += base[1, y0:y0 + h]
                base[1, y0:y0 + h] = 0.0
        # 5 - vertical hold: a column of the picture slips down
        rng = np.random.default_rng((15, step))
        if rng.random() < 0.5 * amt + 0.6 * vertical:
            w = max(8, int(rng.integers(60, 500) * s))
            x0 = int(rng.integers(0, W - w))
            base[:, ya:yb, x0:x0 + w] = np.roll(base[:, ya:yb, x0:x0 + w], int(rng.integers(20, 400) * s), axis=1)
        return base

    return post


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

    # ------------------------------------------------------------------ draw
    def draw(self, f, t, ctx):
        if t < T_DRUMS:
            return self._break(f, t, ctx)
        fade = float(1.0 - smoothstep(T_TRANS, T_TRANS + 4.5, t))
        amt = _amount(t, ctx)
        vertical = float(smoothstep(T_TRANS - 2.0, T_TRANS + 1.5, t) * (1.0 - smoothstep(T_TRANS + 2.5, T_TRANS + 4.5, t)))
        self.dance.draw(f, t, ctx, gain=fade, hud_alpha=1.0, flash=False, bursts=fade > 0.5, hide=(12,), extended=True,
                        hud_out=T_HUD_OUT)
        st = self.dance.stage(grid(t)[0])
        view = st.view
        rng = np.random.default_rng((5, int(math.floor((t - T0) / BEAT))))
        f.set_clip(*view)
        if amt > 0.02:
            self._bands(f, t, amt, rng, view)
        if t > T_TRANS - 0.6:
            self._lanes(f, t, view)
        f.set_clip()
        if amt > 0.02 or vertical > 0.02:
            f.post.append(make_post(t, amt, vertical, view))
        self._tag(f, ctx, st, t, amt, fade)
        return {"edge_alpha": 0.4 + 0.6 * fade}

    def _bands(self, f, t, amt, rng, view):
        """Things thrown into the picture before it is torn: barcode bands, hex dumps, inverted bands."""
        x0, y0, x1, y1 = view
        wv = x1 - x0
        for i in range(int(rng.integers(0, int(2 + 3 * amt) + 1))):            # barcode bands
            h = float(rng.choice([24, 40, 64, 110, 180]))
            ya = float(rng.uniform(y0, y1 - h))
            xa = float(rng.uniform(x0, x1 - 0.2 * wv))
            xb = min(x1, xa + float(rng.uniform(0.16 * wv, 0.9 * wv)))
            f.set_clip(xa, ya, xb, ya + h)
            hud.barcode_burst(f, (xa, ya, xb, ya + h), t, amt, lanes=max(1, int(h // 36)), seed=int(rng.integers(0, 99)))
            f.set_clip(*view)
        for i in range(int(rng.integers(0, int(2 + 2 * amt) + 1))):            # hex dumps
            rows = int(rng.integers(2, 7))
            size = int(rng.choice([14, 17, 22]))
            ya = float(rng.uniform(y0 + 30, y1 - rows * size * 1.3 - 10))
            xa = float(rng.uniform(x0, x1 - 0.37 * wv))
            cols = int(rng.integers(40, 150))
            key = int(rng.integers(0, 9999))
            f.occlude(xa - 6, ya - size, min(x1, xa + cols * size * 0.61 + 6), ya + rows * size * 1.3)
            for r in range(rows):
                hx = (hash01(np.arange(cols), key, r) * 16).astype(int)
                line = "".join(_HEX[v] if (k % 5) != 4 else " " for k, v in enumerate(hx))
                f.text("r" if rng.random() < 0.2 else "w", xa, ya + r * size * 1.3, line, size=size, alpha=0.9)
        for i in range(int(rng.integers(0, int(1 + 3 * amt) + 1))):            # inverted bands
            h = float(rng.choice([18, 36, 70, 120, 200]))
            ya = float(rng.uniform(y0, y1 - h))
            if rng.random() < 0.5:
                f.invert_rects.append((x0, ya, x1, ya + h))
            else:
                xa = float(rng.uniform(x0, x1 - 0.12 * wv))
                f.invert_rects.append((xa, ya, min(x1, xa + float(rng.uniform(0.08 * wv, 0.57 * wv))), ya + h))

    def _lanes(self, f, t, view):
        """What is left when the picture has run out: single muons, falling straight down (lines that may
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
            i1 = np.where(landed, 0.28, 0.6) * gate * out
            f.segments(lay, x, np.full_like(x, y0), x, y, 0.05 * gate * out, i1, width=L.LW)
            f.dots(lay, x[~landed], y[~landed], 2.6, 1.5 * gate * out)
            if red:
                f.dots("w", x[~landed], y[~landed], 1.0, 0.9 * gate * out)

    def _tag(self, f, ctx, st, t, amt, fade):
        """The read-out of the scene: SIGNAL INTEGRITY is made when the drums come back and taken apart in the
        transition; LANES is then constructed in its place. Neither fades."""
        x0, y0 = st.focus_col[0] + 10, st.view[3] - 26
        txt = f"SIGNAL INTEGRITY {max(0.0, 1.0 - amt) * 100:05.1f} %"
        with f.build(B.io(t - T_DRUMS, T_TRANS + 3.3 - t, out=0.3, span=0.6), tbox(x0, y0, txt, L.T_LABEL, pad=5),
                     wave=0.05, marks=False, key=81):
            put_tag(f, ctx, "r", x0, y0, txt, size=L.T_LABEL, pad=5)
        txt = f"LANES {int((t - self.lane_t > 0).sum()):02d}   15.000 KM TO GROUND"
        with f.build(t - (T_TRANS + 3.5), tbox(x0, y0, txt, L.T_SMALL), wave=0.1, marks=False, key=82):
            put_text(f, ctx, "w", x0, y0, txt, size=L.T_SMALL, alpha=0.7)

    # ------------------------------------------------------------------ 7.0 almost nothing
    def _break(self, f, t, ctx):
        geo = self.geo
        w = self.dance.world
        st = self.st0
        view = st.view
        dim = 0.3
        cam = OrthoCamera((0.0, 0.0, 60.0), (0.0, 0.0, 0.0), scale=geo.S, screen_center=(geo.x_mid, Y_GROUND))
        # the only muon: it takes the whole break to come down on the focus, and lands when the drums come back
        p = float(np.clip((t - T_IN) / (T_DRUMS - T_IN), 0.0, 1.0))
        ytop = view[1] + 4
        y = ytop + p * (Y_GROUND - ytop)
        slope = math.tan(math.radians(5.0))
        xg = ctx.focus[0] + 0.12 * (ctx.focus_bay[1] - ctx.focus_bay[0])
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
            f.segments("r", [xg - 16, xg], [Y_GROUND, Y_GROUND - 16], [xg + 16, xg], [Y_GROUND, Y_GROUND + 2], 0.5)
        f.set_clip()
        # what is left of the HUD: the furniture of the DANCE scene, still there, dimmed and emptied (it is not
        # rebuilt); what is new on it - title, cursor, the one row, the counts - is made on the cut
        a = 0.4
        x0, y0, x1, y1, yb = hud.strip_base(f, title=None, alpha=a)
        B.tag(f, "w", x0, y0 - 9, "LONGITUDINAL_PROFILE // --", age0, size=L.T_MICRO, pad=3, alpha=a, cps=110.0, key=83)
        xs = x0 + (16.0 - np.arange(0, 16.01, 1.0)) / 16.0 * (x1 - x0)
        f.segments("w", xs, np.full_like(xs, y0), xs, y0 + 12, 0.8 * a)
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
                f.segments("w", [cx1, cx0], [cy0, cy1], [cx1, cx1], [cy1, cy1], 0.6 * a)
            row = f"00000 MU-   003871.00 {(x - geo.x_mid) / geo.S:+06.2f} {alt:05.2f} +00.00"
            f.text("r", cx0 + 8, cy0 + 62, B.decode(row, age0 - 0.2, cps=120.0, key=85), size=L.T_MICRO, alpha=0.95)
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
