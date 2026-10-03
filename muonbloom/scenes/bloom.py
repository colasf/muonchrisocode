"""BLOOM - the TouchDesigner tower scene: red blooms growing out of the detectors, white strings
between the towers, lattice of crosses, edge ticks, voltage scopes.   Sheet 2.2 + scene 3, 02:13 - 03:00.

  02:13  A BLOOM     the centre detector answers one muon: a single bloom grows out of it, stays open while
                     it is named, and goes back in
  02:27  DET_L ON    a muon is seen coming for 6 s, hits: the left bloom opens and stays, its white lines
                     grow out of the tower
  02:40  DET_R ON
  02:50  DET_C ON    the hero: tallest tower, widest bloom; its lines meet those of its neighbours. From
                     here the three reply live (the same code is the live look of the instrument: any later
                     hit swells its bloom and sends a glow along its lines - usable as the ambient state
                     between shows).

The motion is the one of the TouchDesigner scene (scene1and2.mov, 02:35). The loops of a bloom do not travel:
they stay nested on the detector and change size, the solid ones slowly, the dotted ones pumping in and out
through them (towers.bloom). The white lines are rows of points displaced by one smooth noise field, by
nothing at a tower and the most in the middle of a span (towers.strings). Nothing jumps and nothing is
switched on: a bloom grows out of its detector, a line grows out of its tower, drawn by its head.

The blooms and the lines are sized from the towers: the tallest tower is the hero, another bloom takes the
room above its tower; on a short tower the 27 lines cannot all leave the shaft under the detector, so they
also take the sides of the detector and, on the side of its read-out, leave from the edge of that plate.

Added to the TouchDesigner look: a read-out on every detector (state, energy of the last hit, count),
an energy scale beside every bloom (how open it is = how strong the muons were), the incoming muon
with its ETA, a hit log, the activation sequence, per-tower energy histories, the drums as a barcode.

NOTHING THAT SHOWS DATA FADES IN OR POPS IN (muonbloom/build.py). At the cut (02:13.3) the three read-outs
and the three bottom panels construct themselves; the energy scale of a tower is drawn upwards from its
detector when its bloom opens, and taken apart when the bloom is about to close; a read-out re-decodes
its state and spins its energy on every hit; the header columns are constructed when the three are live.
The blooms, the strings, the tower faces and the lattice are the image: they keep their own life.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import hud, towers
from .. import layout as L
from .. import showdata as sd
from ..engine import CHAR_W, smoothstep
from ..show import Scene

BASE_OPEN = 0.85                # how open a bloom stays once its detector is on
# height (px) of the outermost loop of a bloom at rest, once its detector is on - from the room above its tower:
HERO = (517.0, 620.0)           # the tallest tower, the hero: 0.9 of the room up to the frame, within these (on a
#                                 tall tower it leaves through the top of the frame, as in the TouchDesigner scene)
SIDE = (340.0, 520.0)           # another tower: 0.62 of the room under the header band, within these
HOLD, CLOSE = 2.8, 2.4          # s: the reply of a detector that is not on yet stays open, then goes back in
SCALE_ON = 0.1                  # the energy scale of a bloom is there while the bloom is more open than this
N_LINES = 27                    # white lines per span (towers.strings)


class Bloom(Scene):
    name = "bloom"
    towers = "own"

    def __init__(self, ctx):
        super().__init__(ctx)
        # the blooms take the room the towers leave them: the tallest tower is the hero, the bloom of another
        # tower opens up to the header band at the most
        self.hero = min(ctx.towers.values(), key=lambda tw: tw.top).key
        self.size = {}
        for key, tw in ctx.towers.items():
            if key == self.hero:
                rest = float(np.clip(0.9 * (tw.top - L.FY0), *HERO))
            else:
                rest = float(np.clip(0.62 * (tw.top - L.HEAD_Y - 30.0), *SIDE))
            self.size[key] = rest / (1.8 * towers.REST * towers.open_scale(BASE_OPEN)) / towers.S_MAX[key]
        # the read-out of a tower stands at the right of its detector, where the white lines would leave a
        # small tower. Either the lines of that side start under it, or (a short tower: no room under it) the
        # read-out is a plate and the lines are held on its edge
        self.hold, self.bar = {}, {}
        for key, tw in ctx.towers.items():
            _, y0, x1, y1 = self._plate(tw)
            lo, hi = towers.string_range(tw, N_LINES)
            if y1 <= lo - 4.0:
                continue
            if hi - (y1 + 8.0) >= 6.0 * N_LINES:
                self.hold[key] = {"r": (tw.x1, y1 + 8.0, hi)}
            else:
                self.hold[key] = {"r": (x1, max(lo, y0 + 4.0), min(hi, y1 - 4.0))}
                self.bar[key] = True

    @staticmethod
    def _plate(tw):
        """The block of the read-out of a tower (x0, y0, x1, y1)."""
        return (tw.x1 + 12.0, tw.top + 4.0, tw.x1 + 150.0, tw.top + 146.0)

    @staticmethod
    def _open_at(det, key, t):
        """How open the bloom of a tower is at time t (its scale is there above SCALE_ON). A hit opens it over
        a few frames (towers.ATTACK), never in one: the loops swell, they do not jump."""
        on = float(smoothstep(sd.T_ON[key], sd.T_ON[key] + 0.5, t))
        live = 1.0 - 0.45 * float(smoothstep(sd.T_ON["C"], sd.T_ON["C"] + 0.5, t))      # three on: each hit counts less
        env = towers.energy_env(det, key, t, tau=1.6, attack=towers.ATTACK) * live
        if on < 1.0:
            # not on yet ("A bloom"): the reply to a hit stays open while it is named, breathing, then its
            # loops go back into the detector
            tt, ee, _ = det.hits(key, t - HOLD - CLOSE, t + 1e-6)
            if len(tt):
                a = t - tt
                held = ee * (1.0 - np.exp(-a / (2.0 * towers.ATTACK))) * (1.0 - smoothstep(HOLD, HOLD + CLOSE, a))
                env = max(env, float(held.max()) * (1.0 - on))
        return env + BASE_OPEN * on

    @staticmethod
    def _since(det, key, t, gap=6.0):
        """Seconds since the bloom of a tower opened: since its detector came on, or since the first hit of
        the run of hits that keeps it open (a later hit swells an open bloom, it does not start it again)."""
        tt = det.hits(key, t - 60.0, t + 1e-6)[0]
        first = t
        for th in tt[::-1]:
            if first - th > gap:
                break
            first = float(th)
        return max(t - first, t - sd.T_ON[key] if t >= sd.T_ON[key] else 0.0)

    def _scale_age(self, det, key, t, step=1.0 / 30.0):
        """Build age of the energy scale of a tower: it is constructed when its bloom opens and taken apart
        during the half second before it closes (found by looking a little back and ahead in time: the hits
        are data, so this stays a pure function of t)."""
        if t - sd.T_ON[key] > 3.2:          # the bloom stays open once its detector is on
            return 99.0
        since = 2.6
        for k in range(1, 76):
            if self._open_at(det, key, t - k * step) <= SCALE_ON:
                since = (k - 1) * step
                break
        left = None
        for k in range(1, 16):
            if self._open_at(det, key, t + k * step) <= SCALE_ON:
                left = k * step
                break
        return B.io(since, left, out=0.45, span=0.9)

    def draw(self, f, t, ctx):
        det = ctx.det
        t0 = sd.scene_start(t)                                    # the cut to the towers
        live = t >= sd.T_ON["C"]                                  # 5.0: all three on
        on = {k: float(smoothstep(sd.T_ON[k], sd.T_ON[k] + 0.5, t)) for k in sd.KEYS}
        # the drums only make the image a little brighter, softly (a kick does not step it): they move nothing
        kt, ka = ctx.cues.kicks(t - 3.0, t + 1e-6)
        kick = min(1.0, float((ka * (1.0 - np.exp(-(t - kt) / 0.09)) * np.exp(-(t - kt) / 0.5)).sum()))
        kick *= float(smoothstep(sd.T_ON["C"], sd.T_ON["C"] + 2.0, t))
        hud.cross_grid(f, (L.FX0, L.FY0, L.FX1, L.FY1), step=96.0, inten=0.26)
        if t > min(sd.T_ON.values()):
            # the lines of a tower grow out of it when its detector comes on, and meet those of its neighbours
            towers.strings(f, ctx.towers, t, det, n=N_LINES, age={k: t - sd.T_ON[k] for k in sd.KEYS}, kick=kick,
                           hold=self.hold)
        clip = (L.FX0 + 2, L.FY0 + 2, L.FX1 - 2, L.FY1 - 2)
        open_ = {}
        for key in L.ORDER:
            tw = ctx.towers[key]
            open_[key] = self._open_at(det, key, t)
            if open_[key] > 0.002:
                # a bloom grows out of its detector behind a burst, breathes (and stays, once it is on); when
                # it closes its loops go back into the detector
                towers.bloom(f, tw, t, det, size=self.size[key], clip=clip, open_=open_[key],
                             gain=min(1.0, 0.55 + open_[key]) * (1.0 + 0.12 * kick),
                             reach=towers.SPEED * self._since(det, key, t))
        for i, key in enumerate(L.ORDER):
            tw = ctx.towers[key]
            age, e = det.last(key, t, echoes=True)
            p = max(det.power(key, t), on[key])
            towers.face(f, tw, t, power=p if p < 0.99 else 1.0, value=float(det.value(key, t)) * on[key],
                        hit_age=age, hit_e=e, label=False, dim=0.06)
            if open_[key] > SCALE_ON:
                self._scale(f, tw, open_[key], det, t)
            self._readout(f, tw, t, det, t - t0 - 0.12 * i, bar=key in self.bar)
            towers.incoming(f, tw, t, sd.T_ON[key])
        if live:
            self._left(f, t, ctx)
            t8 = self._t_hits(det, t, 8)        # the histories need a few hits to mean something
            if t8 is not None:
                self._right(f, t, ctx, t - max(sd.T_ON["C"], t8))
        self._bottom(f, t, ctx, live and t >= 181.0, t - t0)
        return {}

    @staticmethod
    def _t_hits(det, t, n):
        """Show time of the n-th detection (all towers, echoes apart), or None if it has not happened yet."""
        tt = np.sort(np.concatenate([det.hits(k, 0.0, t + 1e-6, echoes=False)[0] for k in sd.KEYS]))
        return float(tt[n - 1]) if len(tt) >= n else None

    # ------------------------------------------------------------------ on the towers
    def _scale(self, f, tw, open_, det, t):
        """Energy scale left of a bloom: the height of the bloom reads as the energy it was given.
        It is constructed when the bloom opens (see _scale_age): the axis is drawn upwards by a pen, its ticks
        are thrown out as the pen passes, the figures are decoded, the red marker and its tag are made."""
        ox, oy = tw.det
        x = tw.x0 - 30.0
        y_of = lambda v: oy - towers.bloom_height(tw.key, v, self.size[tw.key])
        # the scale of the hero may go up through the header band (nothing stands there above a tower: the
        # voice is silent in this scene, the header cards sit in the corners) as far as the frame
        y_top = L.FY0 + 18.0 if tw.key == self.hero else L.HEAD_Y + 12.0
        if y_of(0.45) < y_top:                # no room above this tower (the bloom leaves the view): no scale
            return
        y_lo, y_hi = y_of(0.0), max(y_of(1.2), y_top)
        with f.build(self._scale_age(det, tw.key, t), (x - 60.0, y_hi - 14.0, x + 124.0, y_lo + 10.0), flow="bt",
                     wave=0.35, line=0.35, marks=False, key=80 + L.ORDER.index(tw.key)):
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

    def _readout(self, f, tw, t, det, age_in, bar=False):
        """Data tag beside the detector module. age_in = seconds since it started to build (at the cut to the
        scene). Afterwards every hit is read out: the state is decoded again when it changes (BLOOM on a hit,
        WAITING five seconds later, ONLINE when the detector switches on), the energy spins and locks.
        bar: the read-out is a plate that holds the white lines of that side of the tower (a short tower):
        its right edge is drawn, the lines leave from it."""
        key = tw.key
        age, e = det.last(key, t)
        x = tw.x1 + 18
        y = tw.top + 30
        hot = age < 0.6
        online = det.online(key, t)
        state = "ONLINE" if online else ("BLOOM" if age < 5 else "WAITING")
        seen = age < 90.0
        a_state = t - sd.T_ON[key] if online else (age if age < 5 else (age - 5.0 if seen else 99.0))
        lines = [B.resolve(state, a_state, 60.0, key=1), B.roll(f"E {e:.3f}", age, 0.4, key=2) if seen else "E -.---",
                 f"N {det.count(key, t):04d}"]
        if seen:
            lines.append(f"T+{min(age, 99.99):05.2f}")
        px0, py0, px1, py1 = self._plate(tw)
        f.occlude(px0, py0, px1, py1 if bar else y + 44 + (len(lines) - 1) * 24)
        with f.build(age_in, (px0, py0, px1, py1), flow="tb", wave=0.3, marks=False, key=90 + L.ORDER.index(key)):
            if bar:             # drawn from the top when the detector comes on: every line leaves it as the pen passes
                B.pen(f, "w", px1, py0, px1, py1, float(smoothstep(0.0, 0.7, t - sd.T_ON[key])), 0.8)
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
        last = min(((ctx.det.last(k, t)[0], k) for k in sd.KEYS), key=lambda v: v[0])
        rows = ["DETECTORS  3 LIVE", f"DETECTED   {ctx.det.total(t):04d}", f"RATE       {self._rate(ctx, t):.1f} /S",
                f"LAST       {L.NAMES[last[1]]}  T+{min(last[0], 99.9):04.1f}"]
        # constructed when the third detector switches on (02:50.5): from here the three reply live
        with f.build(t - sd.T_ON["C"], (x - 8, y - 24, x + 250 + 330, y + 124), key=100, wave=0.45):
            f.tag("w", x, y, f"{sec[0]} // {sec[1]} // LIVE DETECTORS", size=L.T_MICRO, pad=3)
            f.tag("w", x + 4, y + 86, "BLOOM", size=62, pad=10, bold=True)
            hud.rows(f, x + 250, y + 36, rows, size=L.T_SMALL, red=(3,) if last[0] < 0.6 else ())

    def _right(self, f, t, ctx, age):
        """Header band, right (no voice-over in this section): the energies of the last hits of each tower.
        age = seconds since it became meaningful (the 8th detection): it is constructed then; afterwards the
        bar of every new hit rises (and overshoots) as it comes in."""
        x0, x1 = 2060.0, 2900.0
        y = L.FY0 + 46.0
        w = (x1 - x0) / 3
        bw = 11.0
        with f.build(age, (x0 - 8, y - 24, x1 + 8, y + 136), key=101, wave=0.45):
            f.tag("w", x0, y, "ENERGY // LAST 16 HITS PER TOWER", size=L.T_MICRO, pad=3)
            for i, key in enumerate(L.ORDER):
                xa = x0 + i * w
                tt, ee, ec = ctx.det.hits(key, 0.0, t + 1e-6)
                tt, ee, ec = tt[-16:], ee[-16:], ec[-16:]
                age_h = (t - tt[-1]) if len(tt) else 99.0
                f.rects("w", xa, y + 16, xa + w - 24, y + 19, 0.9)
                f.tag("r" if age_h < 0.6 else "w", xa, y + 46, L.NAMES[key], size=L.T_SMALL, pad=4)
                f.text("w", xa + 88, y + 46, f"N {ctx.det.count(key, t):04d}", size=L.T_MICRO, alpha=0.7)
                bx = xa + np.arange(len(tt)) * (bw + 4)
                hgt = (6 + 58 * ee) * B.spring((t - tt) / 0.25)
                fresh = (t - tt) < 2.0
                for sel, lay in ((fresh, "r"), (~fresh, "w")):
                    if sel.any():
                        f.rects(lay, bx[sel], y + 128 - hgt[sel], bx[sel] + bw, y + 128, np.where(ec[sel], 0.45, 0.95))

    # ------------------------------------------------------------------ bottom band
    def _bottom(self, f, t, ctx, live, age):
        """Bottom band. age = seconds since the cut to the scene: the three panels are constructed then, one
        after the other; a new line of the log is decoded as it comes in, a step of the sequence fills up when
        it happens."""
        panels = ctx.slots["panels"]
        y0, y1 = ctx.slots["y0"], ctx.slots["y1"]
        if not panels:
            return
        x0, x1 = panels[0]
        ev = []
        for key in sd.KEYS:
            tt, ee, _ = ctx.det.hits(key, 0.0, t + 1e-6, echoes=False)
            ev += [(float(a), key, float(b)) for a, b in zip(tt[-4:], ee[-4:])]
        ev.sort(reverse=True)
        with f.build(age - 0.1, (x0 - 8, y0 - 24, x1 + 8, y1 + 8), key=110):
            # (the towers leave no third panel for the count: it is read in the title of the log)
            hud.panel_header(f, x0, x1, y0, "HIT_LOG" if len(panels) > 2 else f"HIT_LOG // DETECTED {ctx.det.total(t):04d}")
            for k, (th, key, e) in enumerate(ev[:4]):
                line = f"{sd.tc(th)}  {L.NAMES[key]}  E {e:.3f}"
                if k == 0:                  # the detection that just came in
                    line = B.resolve(line, t - th, 110.0, key=111)
                f.text("r" if k == 0 and t - th < 1.5 else "w", x0 + 4, y0 + 40 + k * 24, line, size=L.T_SMALL,
                       alpha=0.95 if k == 0 else 0.65)
        if len(panels) > 1:
            x0, x1 = panels[1]
            if live:        # the drums, as a barcode scrolling to the left
                with f.build(t - 181.0, (x0 - 8, y0 - 24, x1 + 8, y1 + 8), key=112, flow="lr"):
                    hud.panel_header(f, x0, x1, y0, "DRUMS >> BARCODE // LAST 3 S")
                    kk, frac, dt = hud.barcode_keys(t, 3.0, 160)
                    dens = np.array([0.05 + 0.9 * math.tanh(ctx.cues.kick(float(v) * dt, tau=0.05)) for v in kk])
                    hud.barcode_lanes(f, x0, x1, y0 + 12, y1, dens, kk, lanes=3, seed=4, frac=frac)
            else:           # what comes next: the three detectors switching on
                steps = [("A BLOOM", sd.T_BLOOM), ("DET_L", sd.T_ON["L"]), ("DET_R", sd.T_ON["R"]),
                         ("DET_C", sd.T_ON["C"])]
                w = (x1 - x0) / len(steps)
                if w < 7 * CHAR_W * L.T_SMALL + 14.0:       # a narrow panel: shorter names, set smaller if need be
                    steps = [(name.replace("A BLOOM", "BLOOM").replace("DET_", ""), ts) for name, ts in steps]
                fs = min(float(L.T_SMALL), (w - 12.0) / (max(len(name) for name, _ in steps) * CHAR_W))
                with f.build(age - 0.25, (x0 - 8, y0 - 24, x1 + 8, y1 + 8), key=113):
                    hud.panel_header(f, x0, x1, y0, "SEQUENCE // DETECTORS ON" if x1 - x0 >= 230 else "SEQUENCE")
                    for k, (name, ts) in enumerate(steps):
                        xx = x0 + k * w
                        f.rect("w", xx + 4, y0 + 22, xx + w - 8, y0 + 62, 0.7)
                        f.text("w", xx + 6, y0 + 88, name, size=fs, alpha=0.9 if t >= ts else 0.5)
                        f.text("w", xx + 6, y0 + 112, sd.tc(ts)[:5], size=min(float(L.T_MICRO), fs), alpha=0.6)
                for k, (name, ts) in enumerate(steps):
                    xx = x0 + k * w
                    with f.build(min(t - ts, age - 0.5), (xx + 4, y0 + 22, xx + w - 8, y0 + 62), key=114 + k, wave=0.1,
                                 flow="lr", marks=False):
                        f.rects("r" if t - ts < 2.0 else "w", xx + 8, y0 + 26, xx + w - 12, y0 + 58, 0.9)
        if len(panels) > 2:
            x0, x1 = panels[2]
            with f.build(age - 0.4, (x0 - 8, y0 - 24, x1 + 8, y1 + 8), key=118):
                hud.panel_header(f, x0, x1, y0, "DETECTED")
                f.text("w", x0 + 2, y0 + 92, f"{ctx.det.total(t):04d}", size=54)
