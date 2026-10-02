"""BLOOM - the TouchDesigner tower scene: red blooms growing out of the detectors, white strings
between the towers, lattice of crosses, edge ticks, voltage scopes.   Sheet 2.2 + scene 3, 02:13 - 03:00.

  02:13  A BLOOM     the centre detector answers one muon: a single bloom bursts out and closes again
  02:27  DET_L ON    a muon is seen coming for 6 s, hits: the left bloom bursts open and stays, strings wake up
  02:40  DET_R ON
  02:50  DET_C ON    the hero: tallest tower, widest bloom. From here the three reply live (the same code
                     is the live look of the instrument: any later hit opens its bloom, plucks its strings,
                     and the drums of the music shiver them - usable as the ambient state between shows).

Added to the TouchDesigner look: a read-out on every detector (state, energy of the last hit, count),
an energy scale beside every bloom (how open it is = how strong the muons were), the incoming muon
with its ETA, a hit log, the activation sequence, per-tower energy histories, the drums as a barcode.
"""
from __future__ import annotations

import math

import numpy as np

from .. import hud, towers
from .. import layout as L
from .. import showdata as sd
from ..engine import smoothstep
from ..show import Scene

SIZE = {"L": 1.0, "C": 1.18, "R": 1.0}
BASE_OPEN = 0.85                # how open a bloom stays once its detector is on


class Bloom(Scene):
    name = "bloom"
    towers = "own"

    def draw(self, f, t, ctx):
        det = ctx.det
        live = t >= sd.T_ON["C"]                                  # 5.0: all three on
        on = {k: float(smoothstep(sd.T_ON[k], sd.T_ON[k] + 0.5, t)) for k in sd.KEYS}
        kick = ctx.cues.kick(t) if live else 0.0
        hud.cross_grid(f, (L.FX0, L.FY0, L.FX1, L.FY1), step=96.0, inten=0.26)
        if max(on.values()) > 0.01:
            towers.strings(f, ctx.towers, t, det, power=on, kick=kick)
        clip = (L.FX0 + 2, L.FY0 + 2, L.FX1 - 2, L.FY1 - 2)
        open_ = {}
        for key in L.ORDER:
            tw = ctx.towers[key]
            env = towers.energy_env(det, key, t, tau=1.6)
            open_[key] = env * (0.55 if live else 1.0) + BASE_OPEN * on[key]
            if open_[key] > 0.03:
                # a bloom bursts out of the detector at the speed of the pulse, then stays open once it is on
                since = t - sd.T_ON[key] if t >= sd.T_ON[key] else det.last(key, t)[0]
                towers.bloom(f, tw, t, det, size=SIZE[key], clip=clip, open_=open_[key],
                             gain=min(1.0, 0.25 + open_[key]) * (1.0 + 0.2 * min(1.0, kick)),
                             reach=towers.SPEED * since + 14.0)
        for key in L.ORDER:
            tw = ctx.towers[key]
            age, e = det.last(key, t, echoes=True)
            p = max(det.power(key, t), on[key])
            towers.face(f, tw, t, power=p if p < 0.99 else 1.0, value=float(det.value(key, t)) * on[key],
                        hit_age=age, hit_e=e, label=False, dim=0.06)
            if open_[key] > 0.03:
                self._scale(f, tw, open_[key])
            self._readout(f, tw, t, det)
            towers.incoming(f, tw, t, sd.T_ON[key])
        if live:
            self._left(f, t, ctx)
            if det.total(t) >= 8:               # the histories need a few hits to mean something
                self._right(f, t, ctx)
        self._bottom(f, t, ctx, live and t >= 181.0)
        return {}

    # ------------------------------------------------------------------ on the towers
    @staticmethod
    def _scale(f, tw, open_):
        """Energy scale left of a bloom: the height of the bloom reads as the energy it was given."""
        ox, oy = tw.det
        x = tw.x0 - 30.0
        y_of = lambda v: oy - towers.bloom_height(tw.key, v, SIZE[tw.key])
        if y_of(0.45) < L.HEAD_Y + 12.0:      # no room above this tower (the bloom leaves the view): no scale
            return
        y_lo, y_hi = y_of(0.0), max(y_of(1.2), L.HEAD_Y + 12.0)       # the header band stays free
        f.segments("w", [x], [y_lo], [x], [y_hi], 0.55)
        for k in range(0, 13):
            y = y_of(k / 10.0)
            if y < y_hi - 0.5:
                break
            major = k % 3 == 0
            f.segments("w", [x], [y], [x - (12 if major else 6)], [y], 0.7)
            if major:
                f.text("w", x - 16, y + 5, f"{k / 10:.1f}", size=L.T_MICRO, alpha=0.6, anchor="rs")
        y = max(y_of(open_), y_hi)
        f.segments("r", [x - 16], [y], [x + 16], [y], 1.2, width=L.LW_BOLD)
        f.tag("r", x + 22, y + 6, f"OPEN {min(open_, 1.2):.2f}", size=L.T_MICRO, pad=3)

    @staticmethod
    def _readout(f, tw, t, det):
        """Data tag beside the detector module."""
        key = tw.key
        age, e = det.last(key, t)
        x = tw.x1 + 18
        y = tw.top + 30
        hot = age < 0.6
        state = "ONLINE" if det.online(key, t) else ("BLOOM" if age < 5 else "WAITING")
        seen = age < 90.0
        lines = [state, f"E {e:.3f}" if seen else "E -.---", f"N {det.count(key, t):04d}"]
        if seen:
            lines.append(f"T+{min(age, 99.99):05.2f}")
        f.occlude(x - 6, y - 26, x + 132, y + 44 + (len(lines) - 1) * 24)
        f.tag("r" if hot else "w", x, y, L.NAMES[key], size=L.T_TAG, pad=5)
        for k, ln in enumerate(lines):
            f.text("r" if (k == 0 and hot) else "w", x, y + 34 + k * 24, ln, size=L.T_SMALL,
                   alpha=0.9 if k == 0 else 0.7)

    # ------------------------------------------------------------------ columns (5.0 only)
    @staticmethod
    def _rate(ctx, t, win=10.0):
        return sum(len(ctx.det.hits(k, t - win, t + 1e-6, echoes=False)[0]) for k in sd.KEYS) / win

    def _left(self, f, t, ctx):
        """Header band, left: title and the live figures."""
        _, sec, _ = sd.section_at(t)
        x, y = 96.0, L.FY0 + 46.0
        f.tag("w", x, y, f"{sec[0]} // {sec[1]} // LIVE DETECTORS", size=L.T_MICRO, pad=3)
        f.tag("w", x + 4, y + 86, "BLOOM", size=62, pad=10, bold=True)
        last = min(((ctx.det.last(k, t)[0], k) for k in sd.KEYS), key=lambda v: v[0])
        rows = ["DETECTORS  3 LIVE", f"DETECTED   {ctx.det.total(t):04d}", f"RATE       {self._rate(ctx, t):.1f} /S",
                f"LAST       {L.NAMES[last[1]]}  T+{min(last[0], 99.9):04.1f}"]
        hud.rows(f, x + 250, y + 36, rows, size=L.T_SMALL, red=(3,) if last[0] < 0.6 else ())

    def _right(self, f, t, ctx):
        """Header band, right (no voice-over in this section): the energies of the last hits of each tower."""
        x0, x1 = 2060.0, 2900.0
        y = L.FY0 + 46.0
        f.tag("w", x0, y, "ENERGY // LAST 16 HITS PER TOWER", size=L.T_MICRO, pad=3)
        w = (x1 - x0) / 3
        bw = 11.0
        for i, key in enumerate(L.ORDER):
            xa = x0 + i * w
            tt, ee, ec = ctx.det.hits(key, 0.0, t + 1e-6)
            tt, ee, ec = tt[-16:], ee[-16:], ec[-16:]
            age = (t - tt[-1]) if len(tt) else 99.0
            f.rects("w", xa, y + 16, xa + w - 24, y + 19, 0.9)
            f.tag("r" if age < 0.6 else "w", xa, y + 46, L.NAMES[key], size=L.T_SMALL, pad=4)
            f.text("w", xa + 88, y + 46, f"N {ctx.det.count(key, t):04d}", size=L.T_MICRO, alpha=0.7)
            bx = xa + np.arange(len(tt)) * (bw + 4)
            hgt = 6 + 58 * ee
            fresh = (t - tt) < 2.0
            for sel, lay in ((fresh, "r"), (~fresh, "w")):
                if sel.any():
                    f.rects(lay, bx[sel], y + 128 - hgt[sel], bx[sel] + bw, y + 128, np.where(ec[sel], 0.45, 0.95))

    # ------------------------------------------------------------------ bottom band
    def _bottom(self, f, t, ctx, live):
        panels = ctx.slots["panels"]
        y0, y1 = ctx.slots["y0"], ctx.slots["y1"]
        if not panels:
            return
        x0, x1 = panels[0]
        hud.panel_header(f, x0, x1, y0, "HIT_LOG")
        ev = []
        for key in sd.KEYS:
            tt, ee, _ = ctx.det.hits(key, 0.0, t + 1e-6, echoes=False)
            ev += [(float(a), key, float(b)) for a, b in zip(tt[-4:], ee[-4:])]
        ev.sort(reverse=True)
        for k, (th, key, e) in enumerate(ev[:4]):
            f.text("r" if k == 0 and t - th < 1.5 else "w", x0 + 4, y0 + 40 + k * 24,
                   f"{sd.tc(th)}  {L.NAMES[key]}  E {e:.3f}", size=L.T_SMALL, alpha=0.95 if k == 0 else 0.65)
        if len(panels) > 1:
            x0, x1 = panels[1]
            if live:        # the drums, as a barcode scrolling to the left
                hud.panel_header(f, x0, x1, y0, "DRUMS >> BARCODE // LAST 3 S")
                cols = 160
                dt = 3.0 / cols
                kk = math.floor((t - 3.0) / dt) + np.arange(cols)
                dens = np.array([0.05 + 0.9 * math.tanh(ctx.cues.kick(float(v) * dt, tau=0.05)) for v in kk])
                hud.barcode_lanes(f, x0, x1, y0 + 12, y1, dens, kk, lanes=3, seed=4)
            else:           # what comes next: the three detectors switching on
                hud.panel_header(f, x0, x1, y0, "SEQUENCE // DETECTORS ON")
                steps = [("A BLOOM", sd.T_BLOOM), ("DET_L", sd.T_ON["L"]), ("DET_R", sd.T_ON["R"]),
                         ("DET_C", sd.T_ON["C"])]
                w = (x1 - x0) / len(steps)
                for k, (name, ts) in enumerate(steps):
                    done = t >= ts
                    xx = x0 + k * w
                    f.rect("w", xx + 4, y0 + 22, xx + w - 8, y0 + 62, 0.7)
                    if done:
                        f.rects("r" if t - ts < 2.0 else "w", xx + 8, y0 + 26, xx + w - 12, y0 + 58, 0.9)
                    f.text("w", xx + 6, y0 + 88, name, size=L.T_SMALL, alpha=0.9 if done else 0.5)
                    f.text("w", xx + 6, y0 + 112, sd.tc(ts)[:5], size=L.T_MICRO, alpha=0.6)
        if len(panels) > 2:
            x0, x1 = panels[2]
            hud.panel_header(f, x0, x1, y0, "DETECTED")
            f.text("w", x0 + 2, y0 + 92, f"{ctx.det.total(t):04d}", size=54)
