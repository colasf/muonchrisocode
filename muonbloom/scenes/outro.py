"""OUTRO - the journey once more, then everything returns to the dot.   Sheet scene 11 + credits.

  11.0  10:52 - 12:20  NARRATIVE     "nothing chose when. everything returns to almost... nothing."
                                      the remains of the disintegration fade to one red dot. Out of it the
                                      whole journey is drawn as ONE line: a log time axis from the collapse
                                      of the star (left, -5 billion years) to NOW / YOU (right). The red dot
                                      travels it on the voice: billions of years in the first metres, the
                                      last heartbeat, the 50 microseconds of atmosphere, you. Two clocks for
                                      "time stretched around it", a heartbeat trace for "less than a
                                      heartbeat", the place and the moment for "in this moment, in this place".
  11.1  12:20 - 12:51  ACCELERANDO   "The universe does not stop": the journey repeats, faster, by the
                                      thousand; the axis coils into a whirl around the focus point of the wall.
                                      The towers stay live; the whirl passes behind them and continues.
  11.2  12:51 - 13:27  SWIRLING      (the drums of the V7 audio enter at 12:51.1 and leave at 13:27)
                                      the whirl is the streak field of the first minute, turning around the
                                      red dot, rings beating on the drums. 13:13.6: the fifty-thousandth muon
                                      through you. The credits start on it at 13:22.
  12.0  13:22 - 13:54  CREDITS       when the drums stop the detectors power down and nothing new arrives: the
                                      field drains into the dot. What is left is the lattice of crosses and one
                                      red dot = the first image of the show. It loops.

NOTHING THAT SHOWS DATA FADES (muonbloom/build.py): the strip, the stations of the axis and their glyphs,
the blocks under the axis, the bottom panels, the tags (NOW, YOU // NOW, the reading of the traveller, MORE
THAN FIFTY THOUSAND) and the credits are constructed when they come - leaders drawn by a pen, tags pushed
out, text decoded, figures spinning before they lock - and taken apart, backwards, when they leave. Only
the drawn things (dust, the line, the traveller, the figure, the whirl, the rings, the light of the towers)
keep their own ramps.

The three towers stand in front of the wall for the whole show and nobody knows yet where: so there is
no fixed x in this scene. Everything is laid out from ctx: text, numbers and panels live in ctx.cols
(the bays between the towers), the time axis skips the towers (no station ever falls behind one), the
whirl and the last red dot sit on ctx.focus. Only textures, trails, rings and the axis line pass behind
the towers. All of it is closed-form in show time, so any frame draws alone.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import hud, towers
from .. import layout as L
from .. import showdata as sd
from ..engine import CHAR_W, hash01, smoothstep
from ..show import Scene

T0, T_COIL, T_END = 652.0, 740.0, 802.0
AX_Y = 520.0                                       # the journey axis
LT0, LT1 = 17.4, -6.6                              # log10(seconds before now) at its two ends
YEAR = 3.156e7
T_STAR = 5.0e9 * YEAR                              # the star collapsed 5 billion years ago
T_ATM = 5.0e-5                                     # 15 km of atmosphere at the speed of light
TAU_MU = 2.197e-6
GAMMA_P, GAMMA_MU = 1.0e6, 30.0
B_SP, EPS = 3.2, 0.02                              # whirl: log spirals r = R (1 - u)
N_DOTS = 7000
T_AXIS = (668.0, 689.0)                            # the axis draws itself
T_PWR = 806.3                                      # the drums stop: the detectors power down (centre, right, left)
PWR_ORDER = {"C": 0.0, "R": 0.5, "L": 1.0}
T_DRAIN = (806.3, 812.2)                           # ... and nothing new arrives: the whirl drains into the dot
T_CREDITS_OUT = (827.0, 832.0)
PANEL_Y = AX_Y + 140.0                             # top of the blocks under the axis
BURST = 0.6                                        # size of the towers' live replies in this calm scene
T_STRIP = (664.0, 765.0)                           # the score strip: made / taken apart (before the whirl gets there)
T_ANN_OUT = (738.5, 741.8)                         # the annotations of the journey are taken apart, before the line coils
PANEL_OUT = dict(journey=739.3, clocks=739.6, heart=739.9, here=740.2)     # when each block under the axis is gone
T_NOW_TAG = (741.2, 767.0)                         # NOW rides beside the dot on its way to the centre of the whirl


def fit(size, n_chars, width):
    """Largest type size <= size that sets n_chars inside width."""
    return max(10.0, min(float(size), width / (max(n_chars, 1) * CHAR_W)))


def fmt_time(s):
    """Seconds -> short string in the unit a person would use."""
    s = abs(float(s))
    y = s / YEAR
    for lim, div, unit in ((1e9, 1e9, "GYR"), (1e6, 1e6, "MYR"), (1e3, 1e3, "KYR"), (1.0, 1.0, "YR")):
        if y >= lim:
            v = y / div
            return f"{v:.2f} {unit}" if v < 10 else f"{v:.1f} {unit}" if v < 100 else f"{v:.0f} {unit}"
    for lim, div, unit in ((86400.0, 86400.0, "D"), (3600.0, 3600.0, "H"), (60.0, 60.0, "MIN"), (1.0, 1.0, "S"),
                           (1e-3, 1e-3, "MS"), (0.0, 1e-6, "US")):
        if s >= lim:
            v = s / div
            return f"{v:.2f} {unit}" if v < 10 else f"{v:.1f} {unit}" if v < 100 else f"{v:.0f} {unit}"
    return "0"


def _runs(ts, vals, t_end, min_len=0.5):
    """Runs of one non-zero value in a sampled state: [(t_start, t_end, value)], the short ones dropped.
    (What rides with a moving point is made when it gets room, and taken apart before it loses it.)"""
    out, k, n = [], 0, len(vals)
    while k < n:
        j = k
        while j + 1 < n and vals[j + 1] == vals[k]:
            j += 1
        a, b = float(ts[k]), (float(ts[j + 1]) if j + 1 < n else float(t_end))
        if vals[k] and b - a >= min_len:
            out.append((a, b, vals[k]))
        k = j + 1
    return out


def _ecg(u):
    """One heartbeat, u = 0..1 of the beat."""
    g = lambda c, a, w: a * np.exp(-((u - c) / w) ** 2)
    return g(0.2, 0.12, 0.035) + g(0.335, -0.14, 0.014) + g(0.365, 1.0, 0.013) + g(0.395, -0.24, 0.014) + g(0.62, 0.3, 0.06)


def _body(y):
    """Front silhouette of a standing figure: y from 0 (feet) to 1 (top of the head) -> [(centre, half width)]."""
    out = []
    if y < 0.47:
        w = 0.026 + 0.03 * (y / 0.47)
        out += [(-0.052, w), (0.052, w)]
    elif y < 0.82:
        q = (y - 0.47) / 0.35
        out.append((0.0, 0.088 + 0.03 * math.sin(math.pi * min(1.0, q * 1.15)) ** 0.7 + 0.012 * q))
        if y < 0.8:
            out += [(-0.162 + 0.02 * q, 0.02), (0.162 - 0.02 * q, 0.02)]
    elif y < 0.865:
        out.append((0.0, 0.03))
    else:
        v = (y - 0.932) / 0.068
        if abs(v) < 1:
            out.append((0.0, 0.05 * math.sqrt(1 - v * v)))
    return out


class Outro(Scene):
    name = "outro"
    towers = "auto"

    def __init__(self, ctx):
        super().__init__(ctx)
        s = sd.said
        self.c = dict(
            chose=s("nothing chose when", 652.0), returns=s("everything returns", 655.5), nothing=s("nothing", 662.5),
            billions=s("Travelling for billions of years", 690.25), stretched=s("Travelling so fast", 693.25),
            heartbeat=s("Living, for less than a heartbeat", 705.5), reach=s("Just long enough to reach you", 709.0),
            endless=s("An endless, fleeting existence", 722.0), body=s("passing through your body", 725.0),
            moment=s("In this moment", 726.5), universe=s("The universe does not stop", 736.0),
            own=s("It keeps its own time", 739.73), continues=s("It continues", 748.37),
            always=s("As it always has", 751.23), part=s("You are part of this", 760.0),
            were=s("You always were", 762.43))
        c = self.c
        self.t_dep, self.t_arr = c["billions"], c["reach"] + 2.0
        # the drums of the swirl: the strongest low hit around 12:51
        kt, ka = ctx.cues.kicks(768.0, 775.0)
        self.t_hit = float(kt[int(np.argmax(ka))]) if len(kt) else 771.1
        # story markers on the axis: (seconds before now, title, value, preferred rows, priority)
        self.marks = [
            (T_STAR, "A STAR COLLAPSES", "-5.0 GYR", (2, 1, 0), 0),
            (4.54e9 * YEAR, "EARTH FORMS", "-4.54 GYR", (0, 1, 2), 4),
            (66e6 * YEAR, "DINOSAURS GONE", "-66 MYR", (1, 0, 2), 8),
            (3.0e5 * YEAR, "FIRST HUMANS", "-300 KYR", (0, 1, 2), 6),
            (4.5e3 * YEAR, "THE PYRAMIDS", "-4.5 KYR", (1, 0, 2), 7),
            (90 * YEAR, "MUON DISCOVERED", "1936", (0, 1, 2), 5),
            (None, "THIS SHOW BEGAN", None, (0, 1, 2), 3),
            (0.8, "YOUR LAST HEARTBEAT", "-0.8 S", (1, 0, 2), 2),
            (T_ATM, "ENTERS THE ATMOSPHERE", "-50 US // 15 KM UP", (2, 1, 0), 1),
        ]
        self.units = [(16.499, "1 GYR"), (13.499, "1 MYR"), (10.499, "1 KYR"), (7.499, "1 YR"), (4.937, "1 DAY"),
                      (3.556, "1 H"), (1.778, "1 MIN"), (0.0, "1 S"), (-3.0, "1 MS")]
        self._geometry(ctx)
        self._layout_marks()
        self._layout_panels()
        self._layout_credits()
        # the travellers of 11.1 / 11.2
        k = np.arange(N_DOTS)
        rng = np.random.default_rng(11)
        arm = rng.integers(0, 2, N_DOTS) * np.pi
        th = np.where(rng.random(N_DOTS) < 0.32, rng.uniform(0, 2 * np.pi, N_DOTS), arm + rng.normal(0, 0.5, N_DOTS))
        self.d_th = th * smoothstep(12, 260, k)                # the first ones follow the axis itself
        self.d_lane = rng.normal(0, 0.5, N_DOTS).clip(-1.4, 1.4)
        self.d_ph = rng.random(N_DOTS) * smoothstep(20, 400, k)
        self.d_b = 0.35 + 0.65 * rng.random(N_DOTS) ** 2
        self.d_tau = 0.1 + 0.1 * rng.random(N_DOTS)
        self.d_act = c["universe"] + (self.t_hit - c["universe"]) * (k / N_DOTS) ** (1 / 2.6)
        # what is left of the disintegration at 10:52
        n = 130
        self.rem = dict(x=rng.uniform(L.FX0 + 60, L.FX1 - 60, n), y=rng.uniform(L.HEAD_Y + 30, L.FY1 - 120, n),
                        gone=rng.uniform(T0 + 1.5, c["nothing"], n), r=0.9 + 1.6 * rng.random(n) ** 3,
                        b=0.3 + 0.7 * rng.random(n), vx=rng.normal(0, 5, n), vy=rng.uniform(2, 14, n),
                        seg=rng.random(n) < 0.25, ang=rng.uniform(0, np.pi, n), ln=rng.uniform(8, 34, n))
        self.rings = np.array([70.0, 112, 160, 232, 250, 334, 440, 468, 600, 760, 796, 980, 1180]) * self.r_out / 1560.0
        # build clocks -----------------------------------------------------------------------------
        self.t_rem_end = float(self.rem["gone"].max()) - 1.1             # the last of the remains is counted out
        x_star = float(self.X(math.log10(T_STAR)))
        self.t_fig = T_AXIS[0] + (self.x_now - 4.0 - x_star) / (self.x_now - x_star) * (T_AXIS[1] - T_AXIS[0])
        ts = np.arange(self.t_dep, self.t_arr, 1.0 / 60.0)                 # the reading that rides with the traveller:
        side, sides = 0, []                                                # (it keeps its side while it has room there)
        for v in ts:
            ok = self._read_sides(float(v))
            side = side if side in ok else (ok[0] if ok else 0)
            sides.append(side)
        self.read_runs = _runs(ts, sides, self.t_arr)
        ts = np.arange(T_NOW_TAG[0], T_NOW_TAG[1], 1.0 / 30.0)             # NOW, beside the dot on its way to the centre
        self.now_runs = _runs(ts, [int(self._now_fits(float(v))) for v in ts], T_NOW_TAG[1])

    # ------------------------------------------------------------------ geometry from the towers
    def _geometry(self, ctx):
        """Everything is placed from ctx: the text columns between the towers, the focus point."""
        cols = list(ctx.cols) or [(L.COL_X0, L.COL_X1)]
        self.cols = cols
        wide = [c for c in cols if c[1] - c[0] >= 200] or [max(cols, key=lambda c: c[1] - c[0])]
        self.col_now = wide[-1]                              # the column where the line ends: NOW, YOU
        self.ax0 = cols[0][0] + 48.0
        self.x_now = self.col_now[1] - 76.0
        # the time scale lives on the columns only: it skips the towers, so no station ever falls behind one
        S, Xb, s = [], [], 0.0
        for a, b in cols:
            a2, b2 = max(a, self.ax0), min(b, self.x_now)
            if b2 - a2 > 4.0:
                S += [s, s + (b2 - a2)]
                Xb += [a2, b2]
                s += (b2 - a2) + 1e-6
        self._S, self._Xb, self._total = np.array(S), np.array(Xb), s - 1e-6
        self.cx, self.cy = ctx.focus                         # centre of the whirl = the last red dot
        corners = ((L.FX0, L.FY0), (L.FX1, L.FY0), (L.FX0, L.FY1), (L.FX1, L.FY1))
        self.r_out = max(math.hypot(x - self.cx, y - self.cy) for x, y in corners) + 60.0

    def X(self, lt):
        s = (LT0 - np.asarray(lt, np.float64)) / (LT0 - LT1) * self._total
        return np.interp(s, self._S, self._Xb)

    def lt_at(self, x):
        return LT0 - float(np.interp(x, self._Xb, self._S)) / self._total * (LT0 - LT1)

    def _col_of(self, x):
        for c in self.cols:
            if c[0] - 1.0 <= x <= c[1] + 1.0:
                return c
        return None

    def _fits(self, xa, xb):
        """Is the x-interval inside one text column (clear of every tower)?"""
        c = self._col_of(xa)
        return c is not None and xb <= c[1] + 1.0

    def _layout_marks(self):
        """Row and side of every station label: inside its column, no two labels on each other, no leader
        through a label. A label that cannot be placed is dropped (its tick stays)."""
        self.mark_pos = {}
        placed = []                                           # (xa, xb, row, x)
        for i in sorted(range(len(self.marks)), key=lambda j: self.marks[j][4]):
            sec, title, val, rows, _ = self.marks[i]
            x = float(self.X(math.log10(sec if sec else 700.0)))
            w = max(len(title) * L.T_SMALL, len(val or "-00:00") * L.T_MICRO) * CHAR_W + 4.0
            best = None
            for row in rows:
                for side in (1, -1):
                    xa, xb = (x + 9.0, x + 9.0 + w) if side > 0 else (x - 9.0 - w, x - 9.0)
                    if not self._fits(xa - 2.0, xb + 2.0):
                        continue
                    if any(r == row and xa < pb + 16.0 and pa < xb + 16.0 for pa, pb, r, _ in placed):
                        continue
                    if any(r < row and pa - 2.0 < x < pb + 2.0 for pa, pb, r, _ in placed):
                        continue                              # its leader would strike through a lower label
                    if any(r > row and xa - 2.0 < px < xb + 2.0 for _, _, r, px in placed):
                        continue                              # a leader already placed would strike through it
                    best = (row, side, xa, xb)
                    break
                if best:
                    break
            if best:
                placed.append((best[2], best[3], best[0], x))
                self.mark_pos[i] = best[:2]

    def _layout_panels(self):
        """The four blocks under the axis flow into the columns left to right; the least important ones are
        dropped when the towers leave no room."""
        gap = 40.0
        blocks = [("journey", 330.0, 400.0, 3), ("clocks", 300.0, 740.0, 1), ("heart", 420.0, 740.0, 2),
                  ("here", 190.0, 400.0, 4)]
        inset = 24.0                      # extra room on the side of a tower: its live replies bloom there
        spans = []
        for a, b in self.cols:
            a2 = a + (inset if a > L.COL_X0 + 1.0 else 0.0)
            b2 = (self.x_now - 110.0) if (a, b) == self.col_now else b - (inset if b < L.COL_X1 - 1.0 else 0.0)
            spans.append((a2, b2))

        def one_each(bl):
            """A column of its own for every block, in reading order."""
            out, ci = {}, 0
            for name, mn, pref, _ in bl:
                while ci < len(spans) and spans[ci][1] - spans[ci][0] < mn:
                    ci += 1
                if ci >= len(spans):
                    return None
                out[name] = (spans[ci][0], min(spans[ci][1], spans[ci][0] + pref))
                ci += 1
            return out

        def flow(bl):
            out, ci, x = {}, 0, spans[0][0]
            per_col = {}
            for name, mn, pref, _ in bl:
                while ci < len(spans) and spans[ci][1] - x < mn:
                    ci += 1
                    x = spans[ci][0] if ci < len(spans) else 0.0
                if ci >= len(spans):
                    return None
                per_col.setdefault(ci, []).append([name, x, mn, pref])
                x += mn + gap
            for ci, items in per_col.items():                # share what is left of the column
                extra = spans[ci][1] - (items[-1][1] + items[-1][2])
                want = sum(p - m for _, _, m, p in items)
                shift = 0.0
                for it in items:
                    add = min(it[3] - it[2], extra * (it[3] - it[2]) / want) if want > 0 else 0.0
                    out[it[0]] = (it[1] + shift, it[1] + shift + it[2] + add)
                    shift += add
            return out

        while blocks:
            res = one_each(blocks) or flow(blocks)
            if res is not None:
                break
            blocks.remove(max(blocks, key=lambda b: b[3]))    # drop the least important and try again
        self.panel = res if blocks else {}

    def _layout_credits(self):
        """Two blocks, each in a column of its own, never the one that holds the red dot."""
        fx = self.cx
        wd = lambda c: c[1] - c[0]
        free = [c for c in self.cols if not (c[0] <= fx <= c[1]) and wd(c) >= 300.0]
        home = self._col_of(fx)
        if home is not None and len(free) < 2:                # the focus column itself, left and right of the dot
            free += [c for c in ((home[0], fx - 190.0), (fx + 190.0, home[1])) if wd(c) >= 300.0]
        A = B = None
        if free:
            top = max(wd(c) for c in free)
            A = min((c for c in free if wd(c) >= 0.85 * top), key=lambda c: c[0])     # the title: wide, and first
            rest = [c for c in free if c != A]
            right = [c for c in rest if c[0] > A[0]]
            B = max(right or rest, key=wd) if rest else None
        self.cred = dict(A=A, B=B)

    # ------------------------------------------------------------------ helpers
    def _phi(self, t):
        """Journeys completed by one traveller since 'the universe does not stop' (accelerando)."""
        return (66.0 / 25.0) * (math.exp((t - self.c["universe"]) / 66.0) - 1.0)

    def _phi_v(self, t):
        return (66.0 / 25.0) * (np.exp((np.asarray(t) - self.c["universe"]) / 66.0) - 1.0)

    def _rate(self, t):
        return math.exp((t - self.c["universe"]) / 66.0) / 25.0

    def _coil(self, t):
        return float(smoothstep(742.0, self.t_hit - 0.2, t))

    def _path(self, u, t, th_off=0.0, lane=0.0):
        """Point of the journey at phase u (0 = the star, 1 = now): the straight axis, coiling into a
        logarithmic spiral around the focus point during 11.1."""
        u = np.asarray(u, np.float64)
        cp = self._coil(t)
        m = np.clip(cp * 1.6 - (1.0 - u) * 0.6, 0.0, 1.0)
        m = m * m * (3 - 2 * m)
        w_line = 26.0 * float(smoothstep(self.c["universe"], 744.0, t))
        xl = self.ax0 + u * (self.x_now - self.ax0)
        yl = AX_Y + lane * w_line
        r = self.r_out * (1.0 - u)
        th = math.pi - B_SP * np.log((1.0 - u + EPS) / (1.0 + EPS)) + th_off
        xs, ys = self.cx + r * np.cos(th), self.cy + r * np.sin(th)
        return xl + (xs - xl) * m, yl + (ys - yl) * m, r, m

    def _dot_x(self, t):
        p = float(np.clip((t - self.t_dep) / (self.t_arr - self.t_dep), 0.0, 1.0))
        x_star = float(self.X(math.log10(T_STAR)))
        return x_star + (self.x_now - x_star) * p

    def _read_label(self, t):
        return f"T-{fmt_time(10 ** self.lt_at(self._dot_x(t)))}"

    def _read_sides(self, t):
        """Sides of the traveller on which its reading has room: -1 left, 1 right (none: a tower is in the way)."""
        xd = self._dot_x(t)
        w = len(self._read_label(t)) * L.T_TAG * CHAR_W + 16
        return [side for side in (-1, 1)
                if self._fits(*((xd - 16 - w, xd - 10) if side < 0 else (xd + 10, xd + 16 + w)))]

    def _now_fits(self, t):
        x = float(self._path(np.array([1.0]), t)[0][0])
        return self._fits(x + 14, x + 96)

    def _bottom_slots(self, ctx):
        """Bottom-band panels by need: (wide one for the numbers, another one for the second readout)."""
        panels = sorted(ctx.slots["panels"], key=lambda p: p[0])
        main = next((p for p in panels if p[1] - p[0] >= 240.0), None)
        rest = [p for p in panels if p is not main and p[1] - p[0] >= 150.0]
        return main, (rest[0] if rest else None), ctx.slots["y0"], ctx.slots["y1"]

    def _veils(self, t, ctx):
        """Soft-edged areas where the whirl is held back so the text on top stays readable:
        [(x0, y0, x1, y1, strength)]. Applied per particle (no hard rectangle cut into the field)."""
        c = self.c
        out = []
        main, second, y0, y1 = self._bottom_slots(ctx)
        if c["universe"] <= t < T_END + 3.0:
            a = float(smoothstep(c["universe"], c["universe"] + 1.5, t) * (1 - smoothstep(T_END - 1.5, T_END + 2.5, t)))
            out += [(p[0] - 6, y0 - 24, p[1] + 6, y1 + 4, 0.9 * a) for p in (main, second) if p]
        if c["universe"] <= t < T_PWR + 2.3:              # the three scopes stay readable under the whirl
            a = float(smoothstep(c["universe"], c["universe"] + 1.5, t) * (1 - smoothstep(T_PWR, T_PWR + 2.3, t)))
            out += [(r[0], r[1] - 16, r[2], r[3], 0.82 * a) for r in ctx.slots["scopes"].values()]
        if t >= T_END:
            a = t - T_END
            keep = float(1 - smoothstep(*T_CREDITS_OUT, t))
            if self.cred["A"]:
                x0, x1 = self.cred["A"]
                out.append((x0 - 20, 352.0, x1 + 20, 880.0, 0.93 * min(1.0, a / 1.5) * keep))
            if self.cred["B"] and a > 4.0:
                x0, x1 = self.cred["B"]
                out.append((x0 - 20, 352.0, x1 + 20, 830.0, 0.93 * min(1.0, (a - 4.0) / 1.5) * keep))
        return [v for v in out if v[4] > 0.01]

    @staticmethod
    def _veil_factor(px, py, veils, soft=110.0):
        k = np.ones_like(px)
        for x0, y0, x1, y1, s in veils:
            dx = np.maximum(np.maximum(x0 - px, px - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - py, py - y1), 0.0)
            k = k * (1.0 - s * (1.0 - smoothstep(0.0, soft, np.hypot(dx, dy))))
        return k

    # ------------------------------------------------------------------ draw
    def draw(self, f, t, ctx):
        c = self.c
        drain = float(smoothstep(*T_DRAIN, t))
        self._lattice(f, 0.72 + 0.28 * drain)
        if t < c["nothing"] + 1.0:
            self._remains(f, t)
        a_img = float(1 - smoothstep(738.5, 745.0, t))           # what is DRAWN of the journey dims into the whirl
        if t >= c["universe"]:            # the world stays under the header until the strip has gone
            top = L.HEAD_Y + 6 - (L.HEAD_Y + 4 - L.FY0) * float(smoothstep(765.0, self.t_hit, t))
            f.set_clip(L.FX0 + 2, top, L.FX1 - 2, L.FY1 - 2)
        if t >= 663.0:
            self._axis(f, t, ctx, drain)
        if 663.0 <= t < 745.0:
            self._stations(f, t, ctx, a_img)
            if T_AXIS[0] <= t < T_ANN_OUT[1]:
                self._panels(f, t, ctx)
        if t >= c["universe"]:
            self._travellers(f, t, ctx, drain)
        if t >= 764.0:
            self._rings(f, t, ctx, drain)
        self._now_dot(f, t, ctx, drain)
        f.set_clip()
        self._bottom(f, t, ctx)
        if t >= T_END:
            self._credits(f, t, ctx)
        age = B.io(t - T_STRIP[0], T_STRIP[1] - t, out=0.6, span=1.6)    # the strip is made, then taken apart
        if age >= 0.0:
            marks = [(c["nothing"], "NOTHING"), (c["billions"], "BILLIONS OF YEARS"), (c["reach"], "REACH YOU"),
                     (c["body"], "YOUR BODY"), (T_COIL, "ACCELERANDO"), (c["part"], "PART OF THIS"),
                     (self.t_hit, "SWIRL")]
            hud.show_strip(f, t, ctx, "OUTRO // SCENE 11 // THE JOURNEY, ONCE MORE", T0, T_END, marks, age=age)
        # the towers stay live to the end of the music; they power down under the credits, then stand dark
        opt = {"tower_dim": 0.34, "burst_size": BURST}
        if t >= T_PWR + 2.4:
            opt.update(towers="none", scopes=False)
        elif t >= T_PWR:
            self._power_down(f, t, ctx)
            opt.update(towers="own", scopes=False)
        return opt

    def _lattice(self, f, gain):
        """The lattice of the first image of the show (origin.py): 64 px, centred on the red dot, a bolder cross
        every fourth one. Quieter under the journey, at full strength again when everything has returned."""
        cx, cy = self.cx, self.cy
        x0, y0, x1, y1 = L.FRAME
        kx = np.arange(math.ceil((x0 + 14 - cx) / 64.0), math.floor((x1 - 14 - cx) / 64.0) + 1)
        ky = np.arange(math.ceil((y0 + 14 - cy) / 64.0), math.floor((y1 - 14 - cy) / 64.0) + 1)
        KX, KY = np.meshgrid(kx, ky)
        X, Y = (cx + KX * 64.0).ravel(), (cy + KY * 64.0).ravel()
        major = ((KX % 4 == 0) & (KY % 4 == 0)).ravel()
        f.crosses("w", X[~major], Y[~major], 7.0, 0.2 * gain)
        f.crosses("w", X[major], Y[major], 10.0, 0.42 * gain, width=1.3)

    # --- 11.0 ---------------------------------------------------------------------
    def _remains(self, f, t):
        """What the disintegration left: dust settling, gone one by one before 'nothing.'"""
        r = self.rem
        a = 1.0 - smoothstep(r["gone"] - 2.2, r["gone"], t)
        dt = t - T0
        x, y = r["x"] + r["vx"] * dt, r["y"] + r["vy"] * dt
        m = (a > 0.01) & ~r["seg"]
        f.dots("w", x[m], y[m], r["r"][m], (0.75 * r["b"] * a)[m])
        m = (a > 0.01) & r["seg"]
        dx, dy = np.cos(r["ang"][m]) * r["ln"][m], np.sin(r["ang"][m]) * r["ln"][m]
        f.segments("w", x[m], y[m], x[m] + dx, y[m] + dy, (0.5 * r["b"] * a)[m])
        n = int((a > 0.5).sum())
        if n:                             # their count: decoded at the cut, unwritten when the last one goes
            age = B.io(t - T0 - 0.2, self.t_rem_end - t, out=0.4, span=0.6)
            f.text("w", self.cols[0][0], 272, B.resolve(f"REMAINS {n:03d}", age, cps=40.0, key=5), size=L.T_SMALL,
                   alpha=0.7)

    def _axis(self, f, t, ctx, drain):
        """The line of the journey (it stays, and coils in 11.1), its decades, the part already travelled.
        The decades and the units are thrown out as the line passes them, their names decoded; they are taken
        off again, left to right, before the line coils."""
        c = self.c
        span = self.x_now - self.ax0
        q = float(np.clip((t - T_AXIS[0]) / (T_AXIS[1] - T_AXIS[0]), 0.0, 1.0))
        x_star = float(self.X(math.log10(T_STAR)))
        x_rev = x_star + q * (self.x_now - x_star)
        u_rev = (x_rev - self.ax0) / span
        cp = self._coil(t)
        swirl = float(smoothstep(self.t_hit - 1.0, self.t_hit + 6.0, t))
        if q > 0:
            s = np.linspace(0.0, 1.0, 520)
            u = (1 - (1 - s) ** 2.0) * u_rev if cp > 0 else s * u_rev
            px, py, _, _ = self._path(u, t)
            f.polyline("w", px, py, 0.85 * (1 - 0.7 * swirl) * (1 - drain), width=L.LW)
        if t >= T_ANN_OUT[1] or cp > 0.2:
            return

        def reach(x):                     # when the line gets to x ...
            return T_AXIS[0] + (np.asarray(x, np.float64) - x_star) / (self.x_now - x_star) * (T_AXIS[1] - T_AXIS[0])

        def leave(x):                     # ... and when what marks it is taken off again
            return T_ANN_OUT[0] + 1.5 + 1.4 * (np.asarray(x, np.float64) - self.ax0) / span

        def mark(x, ln):                  # a tick: thrown out long as the line passes, flicked off when it leaves
            a = t - reach(x)
            return (np.where(a >= 0.0, ln * (1.0 + 1.6 * np.exp(-np.maximum(a, 0.0) / 0.07)), 0.0)
                    * (1.0 - B.ease((t - leave(x)) / 0.15)))

        # decades + units (the scale lives in the columns)
        lts = np.arange(math.floor(LT0), math.ceil(LT1) - 1, -1.0)
        xs = self.X(lts)
        ln = mark(xs, 7.0)
        m = ln > 0.3
        f.segments("w", xs[m], np.full(m.sum(), AX_Y), xs[m], AX_Y + ln[m], 0.8)
        for k, (lt, lab) in enumerate(self.units):
            x = float(self.X(lt))
            if not self._fits(x + 2, x + 74) or x > self.x_now - 150:
                continue
            ln = float(mark(x, 16.0))
            if ln > 0.3:
                f.segments("w", [x], [AX_Y], [x], [AX_Y + ln], 0.9, width=L.LW)
            age = B.io(t - float(reach(x)) - 0.06, float(leave(x)) + 0.1 - t, out=0.2, span=0.4)
            f.text("w", x + 6, AX_Y + 34, B.resolve(lab, age, cps=60.0, key=k), size=L.T_MICRO, alpha=0.6)
        if self._fits(self.ax0, self.ax0 + 190):
            age = B.io(t - T_AXIS[0] - 0.5, T_ANN_OUT[0] + 1.7 - t, out=0.3, span=0.5)
            f.text("w", self.ax0, AX_Y + 62, B.decode("LOG TIME BEFORE NOW >", age, cps=60.0, key=2), size=L.T_MICRO,
                   alpha=0.5)
        # the part already travelled, bold; a flash runs along it on 'an endless, fleeting existence'. When the
        # annotations go it is released from its left end: the line is left thin, ready to coil
        xd = self._dot_x(t)
        if t >= self.t_dep and xd > x_star + 2:
            fl = math.exp(-max(0.0, t - c["endless"]) / 1.2) if t >= c["endless"] else 0.0
            xa = x_star + (xd - x_star) * float(B.ease(B.lin(t, T_ANN_OUT[0] + 1.7, T_ANN_OUT[0] + 3.1)))
            if xd - xa > 1.0:
                f.segments("w", [xa], [AX_Y], [xd], [AX_Y], 0.95 + 1.2 * fl, width=L.LW_BOLD + 1.5 * fl)

    def _stations(self, f, t, ctx, a_img):
        """Markers on the axis: what else happened along the way, the glyphs of the show, the traveller.
        Every marker is constructed when the line reaches it (its leader drawn up from the axis, its name
        decoded) and taken apart again, left to right, before the line coils. a_img only dims what is drawn
        (the traveller, the figure)."""
        x_star = float(self.X(math.log10(T_STAR)))
        xd = self._dot_x(t)
        dep = t >= self.t_dep

        def reach(x):                     # when the line gets to x ...
            return T_AXIS[0] + (x - x_star) / (self.x_now - x_star) * (T_AXIS[1] - T_AXIS[0])

        def leave(x):                     # ... and when its marker is gone again
            return T_ANN_OUT[0] + 0.5 + 1.6 * (x - self.ax0) / (self.x_now - self.ax0)

        star_top = AX_Y - 136.0
        for i, (sec, title, val, _, _) in enumerate(self.marks if t >= T_AXIS[0] else ()):
            if sec is None:
                sec = max(t, 1.0)
                m_, s_ = divmod(int(t), 60)
                val = f"-{m_:02d}:{s_:02d}"
            x = float(self.X(math.log10(sec)))
            if i in self.mark_pos and i == 0:
                star_top = AX_Y - 44 - self.mark_pos[i][0] * 50 - 22.0
            age = B.io(t - reach(x), leave(x) - t)
            if age < 0.0:
                continue
            passed = dep and xd >= x
            fl = math.exp(-max(0.0, (xd - x)) / 160.0) if passed and t < self.t_arr + 1 else 0.0
            al = 0.5 + 0.4 * passed + 0.5 * fl
            lay = "r" if fl > 0.3 else "w"
            if i not in self.mark_pos:
                with f.build(age, (x - 8, AX_Y - 26, x + 8, AX_Y + 6), flow="bt", wave=0.1, marks=False, key=40 + i):
                    f.dots(lay, [x], [AX_Y], 3.2, 1.2 * al)
                    f.segments(lay, [x], [AX_Y - 10], [x], [AX_Y - 22], 0.75 * al, width=L.LW)
                continue
            row, side = self.mark_pos[i]
            y1 = AX_Y - 44 - row * 50
            anchor = "ls" if side > 0 else "rs"
            tx = x + 9 * side
            xe = tx + side * max(len(title) * L.T_SMALL, len(val) * L.T_MICRO) * CHAR_W
            with f.build(age, (min(x, xe) - 6, y1 - 18, max(x, xe) + 6, AX_Y + 6), flow="bt", wave=0.25, line=0.2,
                         marks=False, key=40 + i):
                f.dots(lay, [x], [AX_Y], 3.2, 1.2 * al)
                f.segments(lay, [x], [AX_Y - 10], [x], [y1 + 8], 0.75 * al, width=L.LW)
                f.text(lay, tx, y1, title, size=L.T_SMALL, alpha=al, anchor=anchor)
                f.text("w", tx, y1 + 21, val, size=L.T_MICRO, alpha=0.7 * al, anchor=anchor)
        # its whole life, under the axis, just before now
        x_l = float(self.X(math.log10(TAU_MU)))
        if t >= T_AXIS[0] and self._fits(x_l - 160, x_l):
            al = 0.55 + 0.45 * (dep and xd >= x_l)
            with f.build(B.io(t - reach(x_l), leave(x_l) - t), (x_l - 166, AX_Y + 2, x_l + 6, AX_Y + 84), flow="tb",
                         wave=0.2, line=0.2, marks=False, key=51):
                f.segments("r", [x_l], [AX_Y + 4], [x_l], [AX_Y + 58], 0.8 * al, width=L.LW)
                f.text("r", x_l - 8, AX_Y + 56, "ITS WHOLE LIFE", size=L.T_SMALL, alpha=al, anchor="rs")
                f.text("w", x_l - 8, AX_Y + 77, "2.197 US", size=L.T_MICRO, alpha=0.7 * al, anchor="rs")
        # glyphs: drawn when the line gets to them, taken apart with the stations ------------------
        c0 = self.cols[0]
        if t >= T_AXIS[0] and c0[1] - c0[0] >= 230:      # the star: shells around a point, over the first column
            g = 0.6 + 0.4 * (not dep)
            gx, gy = min(x_star + 138.0, c0[1] - 70.0), 292.0
            rr = np.array([8.0, 16, 26, 37, 49, 60])
            with f.build(B.io(t - T_AXIS[0] - 0.5, T_ANN_OUT[0] + 1.1 - t),
                         (x_star - 8, gy - 68, gx + 68, max(star_top, gy + 68) + 4), flow="out",
                         origin=(x_star, star_top), wave=0.35, marks=False, key=52):
                f.rings("w", [gx] * 6, [gy] * 6, rr * (1 + 0.03 * math.sin(t * 1.3)), 0.6 * g)
                f.dots("r", [gx], [gy], 4.5, 1.3 * g)
                if gx - 66 > x_star + 8:
                    f.segments("w", [x_star, x_star], [star_top, gy], [x_star, gx - 66], [gy, gy], 0.45 * g)
        mid = [cc for cc in self.cols if cc != c0 and cc != self.col_now and cc[1] - cc[0] >= 330]
        if mid and t >= T_AXIS[0]:                        # the messenger: a point and the line it draws
            cc = max(mid, key=lambda v: v[1] - v[0])
            gx = float(np.clip(0.5 * (cc[0] + cc[1]), cc[0] + 80.0, cc[1] - 236.0))
            gy = 330.0
            g = 0.75
            ln = min(300.0, gx - cc[0] - 10.0)
            with f.build(B.io(t - reach(gx), T_ANN_OUT[0] + 1.4 - t),
                         (gx - ln - 6, gy - 30, gx + 232, gy + 0.147 * ln + 8), flow="lr", wave=0.35, marks=False,
                         key=53):
                f.segments("w", [gx - ln], [gy + 0.147 * ln], [gx], [gy], 0.0, 0.8 * g, width=L.LW)
                f.dots("w", [gx], [gy], 4.0, 1.4 * g)
                f.rings("r", [gx], [gy], [14.0], 0.7 * g)
                f.text("w", gx + 26, gy - 6, "PRIMARY // P+", size=L.T_SMALL, alpha=0.85 * g)
                f.text("w", gx + 26, gy + 18, "V 0.999 999 999 999 5 C", size=L.T_MICRO, alpha=0.65 * g)
        x_a = float(self.X(math.log10(T_ATM)))
        ca = self._col_of(x_a)
        if ca is not None and t >= T_AXIS[0]:             # 15 km of air: the cascade, above its station
            passed = dep and xd >= x_a
            g = 0.55 + 0.45 * passed
            half = max(0.0, min(75.0, x_a - ca[0] - 6.0, ca[1] - x_a - 6.0))
            ax_, ay_ = x_a + 4, 250.0
            k = np.arange(15)
            sp = (hash01(k, 71) - 0.5) * 2.0 * half
            y_split = ay_ + 22 + 30 * hash01(k, 72)
            with f.build(B.io(t - reach(x_a), T_ANN_OUT[0] + 1.9 - t),
                         (x_a - half - 6, ay_ - 24, x_a + half + 14, AX_Y - 144), flow="tb", wave=0.35, marks=False,
                         key=54):
                f.segments("r", [ax_], [ay_ - 20], [ax_], [ay_ + 24], 0.9 * g, width=L.LW)
                f.segments("w", np.full(15, ax_), y_split, ax_ + sp, np.full(15, AX_Y - 198.0), 0.55 * g)
                f.segments("r", [ax_], [ay_ + 24], [ax_ + 6], [AX_Y - 150], 0.8 * g, width=L.LW)
        if t >= self.t_fig:               # you, at the end of the line
            self._you(f, t, a_img)
        # the traveller and its clock reading
        fade_in = float(smoothstep(663.0, 665.0, t))
        pulse = 0.5 + 0.5 * math.sin(t * 3.0) ** 2
        f.dots("r", [xd], [AX_Y], 7.5 + 2.0 * pulse, 1.5 * fade_in * a_img)
        f.dots("w", [xd], [AX_Y], 2.4, 1.2 * fade_in * a_img)
        if dep and t < self.t_arr:
            f.rings("r", [xd], [AX_Y], [20.0 + 6 * pulse], 0.7, width=L.LW)
            for a, b, side in self.read_runs:             # its reading rides with it, on whichever side is free:
                if a <= t < b:                            # made when it gets room, taken apart before it loses it
                    age = B.io(t - a, b - t, out=0.2, span=0.4)
                    B.pen(f, "r", xd, AX_Y + 12, xd, AX_Y + 98, B.ease(B.lin(age, 0.0, 0.2)), 0.6, width=1.0, head=2.4)
                    B.tag(f, "r", xd + 16 * side, AX_Y + 104, self._read_label(t), age, t0=0.1, size=L.T_TAG, pad=5,
                          anchor="rs" if side < 0 else "ls", cps=70.0, key=8)     # above the headers of the blocks below
                    break
        elif t < T_AXIS[0] and self._fits(xd + 16, xd + 180):
            age = B.io(t - 664.5, T_AXIS[0] - t, out=0.4, span=1.2)      # unwritten before the line starts through it
            f.text("r", xd + 22, AX_Y + 7, B.decode("ALMOST NOTHING", age, cps=14.0, band=2, key=3), size=L.T_SMALL,
                   alpha=0.9)

    def _you(self, f, t, a_img):
        """The figure at the end of the axis (front view in slices, the journey line through the heart). It
        assembles slice by slice around the heart when the line gets there; its tag is made, then taken apart
        with the stations."""
        c = self.c
        Hh = 250.0
        x0 = self.x_now
        y_feet = AX_Y + 0.71 * Hh
        arr = float(smoothstep(self.t_arr - 0.6, self.t_arr + 0.2, t))
        g = a_img * (0.45 + 0.55 * arr)
        ys = np.arange(0.0, 1.0, 5.0 / Hh)
        xa, xb, yy = [], [], []
        for v in ys:
            for cen, hw in _body(float(v)):
                xa.append(x0 + (cen - hw) * Hh); xb.append(x0 + (cen + hw) * Hh); yy.append(y_feet - v * Hh)
        xa, xb, yy = np.array(xa), np.array(xb), np.array(yy)
        there = np.abs(yy - AX_Y) <= 190.0 * B.lin(t, self.t_fig, self.t_fig + 0.7)       # from the heart outwards
        heart = np.abs(yy - AX_Y) < 16
        hot = 0.0
        if t >= self.t_arr - 0.2:
            hot = math.exp(-max(0.0, t - self.t_arr) / 1.5)
        ab = t - c["body"]
        if ab >= 0:
            hot = max(hot, math.exp(-ab / 2.2))
        m = there & ~heart
        f.segments("w", xa[m], yy[m], xb[m], yy[m], 0.75 * g, width=L.LW)
        m = there & heart
        f.segments("w", xa[m], yy[m], xb[m], yy[m], 0.75 * g * (1 - hot), width=L.LW)
        if hot > 0.02:
            f.segments("r", xa[m], yy[m], xb[m], yy[m], 1.5 * hot * a_img, width=L.LW)
            f.rings("r", [x0 + 0.012 * Hh], [AX_Y], [10 + 60 * (1 - hot)], hot * a_img, width=L.LW)
        if 0 <= ab < 5.0:                     # 'passing through your body': from above, as it really does
            u = min(1.0, ab / 0.22)
            fade = float(1 - smoothstep(3.5, 5.0, ab))
            ya, yb_ = y_feet - Hh - 110, y_feet + 26
            f.segments("r", [x0 + 26], [ya], [x0 + 26 - 34 * u], [ya + (yb_ - ya) * u], 1.2 * fade * a_img, width=L.LW)
        now = t >= c["moment"]
        B.tag(f, "r" if (now or arr > 0.5) else "w", x0, y_feet + 44, "YOU // NOW",
              B.io(t - self.t_fig - 0.6, T_ANN_OUT[0] + 2.3 - t, out=0.3, span=0.45), size=L.T_TAG, pad=5,
              alpha=0.6 + 0.4 * arr, anchor="ms", cps=50.0, key=6)

    def _panels(self, f, t, ctx):
        """Under the axis, flowed into the columns: the journey, two clocks, a heartbeat, this place. Each block
        constructs itself on its cue and is taken apart - one after the other - before the line coils."""
        c = self.c
        y0 = PANEL_Y
        xd = self._dot_x(t)
        sec = 10 ** self.lt_at(xd)

        def age(t_in, name, out=0.45):
            return B.io(t - t_in, PANEL_OUT[name] - t, out=out)

        if "journey" in self.panel:
            x0, x1 = self.panel["journey"]
            lines = ["ORIGIN    CORE COLLAPSE", "DEPARTED  -5.0 GYR", "MESSENGER PROTON  P+", "GAMMA     1 000 000",
                     "PATH      5.0E9 LY", "LAST LEG  MUON  15 KM", "ARRIVAL   NOW"]
            with f.build(age(T_AXIS[0] + 0.2, "journey"), (x0 - 12, y0 - 10, x1 + 6, y0 + 96 + 6 * 27 + 12), flow="tb",
                         wave=0.45, cps=90.0, key=31):
                f.tag("w", x0, y0 + 36, "THE JOURNEY", size=fit(44, 11, x1 - x0 - 24), pad=8)
                for k, ln in enumerate(lines):
                    f.text("w", x0, y0 + 96 + k * 27, ln, size=L.T_SMALL, alpha=0.85)
        if "clocks" in self.panel:
            x0, x1 = self.panel["clocks"]
            w = x1 - x0
            wide = w >= 600
            muon = sec <= T_ATM
            our = (T_ATM - sec) if muon else (T_STAR - sec)
            its = our / (GAMMA_MU if muon else GAMMA_P)
            if t >= self.t_arr:
                our, its = T_ATM, T_ATM / GAMMA_MU
            size = fit(54, 8, (w * 0.5 - 24) if wide else (w - 8))
            xi, yi = (x0 + w * 0.5, y0) if wide else (x0, y0 + 100)
            yq = y0 + (0 if wide else 100)
            who = "MUON  GAMMA 30" if muon or t >= self.t_arr else "PROTON  GAMMA 1 000 000"
            line = f"{who}  //  ITS CLOCK RUNS GAMMA TIMES SLOWER" if w >= 545 else who
            with f.build(age(c["stretched"], "clocks"), (x0 - 10, y0 - 28, x1 + 10, yq + 166), key=32):
                hud.panel_header(f, x0, x1, y0, "TWO CLOCKS // TIME STRETCHED AROUND IT" if w >= 330 else "TWO CLOCKS")
                f.tag("w", x0 + 2, y0 + 52, "OUR TIME", size=L.T_MICRO, pad=3)
                f.text("w", x0, y0 + 116, fmt_time(our), size=size)
                f.tag("r", xi + 2, yi + 52, "ITS OWN TIME", size=L.T_MICRO, pad=3)
                f.text("r", xi, yi + 116, fmt_time(its), size=size)
                f.text("w", x0, yq + 152, line, size=L.T_MICRO, alpha=0.7)
            # its own time against its lifetime: just long enough
            by = yq + 196
            own = (T_ATM - min(sec, T_ATM)) / GAMMA_MU if t < self.t_arr else T_ATM / GAMMA_MU
            frac = float(np.clip(own / TAU_MU, 0.0, 1.0))
            cap = (f"THE MUON: ITS OWN TIME {own * 1e6:.2f} US  /  ITS LIFETIME 2.197 US" if w >= 495
                   else f"OWN TIME {own * 1e6:.2f} / 2.197 US")
            with f.build(age(c["reach"], "clocks"), (x0 - 6, by - 6, x1 + 6, by + 56), flow="lr", wave=0.3, marks=False,
                         key=33):
                f.rect("w", x0, by, x1, by + 22, 0.7)
                f.rects("r", x0 + 3, by + 3, x0 + 3 + (w - 6) * frac, by + 19, 0.95)
                f.text("w", x0, by + 48, cap, size=L.T_MICRO, alpha=0.8)
            B.tag(f, "r", x1, by + (52 if w >= 700 else 84), "JUST LONG ENOUGH",
                  B.io(t - self.t_arr, PANEL_OUT["clocks"] - t, out=0.3, span=0.5), size=L.T_SMALL, pad=4, anchor="rs",
                  cps=50.0, key=12, commit=t < T_ANN_OUT[0])
        if "heart" in self.panel:
            x0, x1 = self.panel["heart"]
            a = age(c["heartbeat"], "heart")
            with f.build(a, (x0 - 10, y0 - 28, x1 + 10, y0 + 250), key=34):
                hud.panel_header(f, x0, x1, y0, "ONE HEARTBEAT // 0.8 S")
                n = 360
                u = np.linspace(0, 1, n)
                v = _ecg(u)
                k = int(n * B.lin(a, 0.25, 1.35))             # the trace writes itself in one beat
                base = y0 + 150
                xr = x1 - 34.0                    # the last label of the ruler ("0.8") still ends inside the column
                f.segments("w", [x0], [base], [xr], [base], 0.25)
                if k > 1:
                    f.polyline("w", x0 + u[:k] * (xr - x0), base - v[:k] * 96, 0.95, width=L.LW)
                    if k < n:                     # the pen
                        f.dots("w", [x0 + u[k - 1] * (xr - x0)], [base - v[k - 1] * 96], 3.0, 1.7)
                hud.ruler(f, x0, xr, base + 30, 0.0, 0.8, 0.02, 0.1 if xr - x0 >= 520 else 0.2,
                          fmt=lambda vv: f"{vv:.1f}", inten=0.6, lab_dy=30)
                f.segments("r", [x0 + 1], [y0 + 40], [x0 + 1], [base + 30], 1.3, width=L.LW)
                f.tag("r", x0 + 12, y0 + 52, "ITS WHOLE LIFE 2.197 US", size=L.T_MICRO, pad=3)
                cap = ("LESS THAN ONE PIXEL OF THIS LINE  =  1 / 364 000 OF A HEARTBEAT" if x1 - x0 >= 545
                       else "ITS LIFE = 1 / 364 000 OF A HEARTBEAT")
                f.text("w", x0, y0 + 236, cap, size=L.T_MICRO, alpha=0.75)
        if "here" in self.panel:
            x0, x1 = self.panel["here"]
            lines = ["CINCINNATI, OHIO", "39.1031 N  84.5120 W", "BLINK 2026", f"SHOW TIME {sd.tc(t)[:8]}",
                     f"THROUGH YOU {ctx.through_you(t):,}".replace(",", " ")]
            wl = max(len(ln) for ln in lines) * L.T_SMALL * CHAR_W
            with f.build(age(c["moment"], "here"), (x0 - 12, y0 - 10, x0 + max(wl, 150.0) + 8, y0 + 96 + 4 * 27 + 12),
                         flow="tb", wave=0.4, cps=90.0, key=35):
                f.tag("r", x0, y0 + 36, "HERE", size=44, pad=8)
                for k, ln in enumerate(lines):
                    f.text("w", x0, y0 + 96 + k * 27, ln, size=L.T_SMALL, alpha=0.85)

    # --- 11.1 / 11.2 ------------------------------------------------------------------
    def _travellers(self, f, t, ctx, drain):
        """'The universe does not stop': the journey again and again, faster, until it is a whirl."""
        age = t - self.d_act
        idx = np.nonzero(age > 0)[0]
        if not len(idx):
            return
        u = (self._phi(t) - self._phi_v(self.d_act[idx]) + self.d_ph[idx]) % 1.0
        K = 5
        du = self._rate(t) * self.d_tau[idx]
        j = np.arange(K) / (K - 1)
        uu = np.clip(u[:, None] - j[None, :] * du[:, None], 0.0, 1.0)
        px, py, r, m = self._path(uu, t, self.d_th[idx][:, None], self.d_lane[idx][:, None])
        kick = min(1.5, ctx.cues.kick(t, 0.14))
        loud = ctx.cues.loud(t, 0.3)
        hit = math.exp(-(t - self.t_hit) / 0.5) if t >= self.t_hit else 0.0
        r_cut = self.r_out * (1.0 - drain) ** 1.25
        a = smoothstep(0.0, 1.2, age[idx]) * smoothstep(0.0, 0.03, u) * (1 - smoothstep(0.975, 1.0, u))
        if drain > 0:
            a = a * (1.0 - smoothstep(r_cut - 90.0, r_cut, r[:, 0]))
        credits = 1.0 - 0.45 * float(smoothstep(T_END, T_END + 2.0, t))      # quieter under the credits
        b = self.d_b[idx] * a * (0.55 + 0.45 * loud + 0.5 * kick + 1.2 * hit) * (0.7 + 0.5 * m[:, 0]) * credits
        veils = self._veils(t, ctx)
        if veils:
            b = b * self._veil_factor(px[:, 0], py[:, 0], veils)
        fall = (1.0 - j) ** 1.2
        i0 = b[:, None] * fall[None, :-1]
        i1 = b[:, None] * fall[None, 1:]
        f.segments("w", px[:, :-1].ravel(), py[:, :-1].ravel(), px[:, 1:].ravel(), py[:, 1:].ravel(), i0.ravel(),
                   i1.ravel(), width=L.LW)
        f.dots("w", px[:, 0], py[:, 0], 1.5 + 0.9 * self.d_b[idx], 1.25 * b)

    def _rings(self, f, t, ctx, drain):
        """The rings of the first minute, around the dot, beating on the drums; they fall in at the end."""
        g = float(smoothstep(764.0, self.t_hit, t)) * (1.0 - 0.4 * float(smoothstep(T_END, T_END + 2.0, t)))
        kick = min(1.5, ctx.cues.kick(t, 0.2))
        k = np.arange(len(self.rings))
        r = self.rings * (1 + 0.025 * np.sin(t * 0.9 + 1.7 * k) + 0.03 * kick * (1 + k % 3))
        r = r * (1.0 - drain) ** (1.0 + 0.25 * (k % 4))
        m = r > 6.0
        f.rings("w", np.full(m.sum(), self.cx), np.full(m.sum(), self.cy), r[m], 0.72 * g * (1 - 0.6 * drain),
                spacing=0.6, width=L.LW)
        kt, ka = ctx.cues.kicks(max(self.t_hit - 0.05, t - 2.5), t + 1e-6)     # every kick throws one more ring out
        for tk, a in zip(kt, ka):
            age = t - float(tk)
            rr = (40.0 + 520.0 * age) * (1.0 - drain)
            f.rings("w", [self.cx], [self.cy], [rr], 0.6 * min(float(a), 1.6) * g * math.exp(-age / 0.9) * (1 - drain),
                    spacing=0.7, width=L.LW)

    def _now_dot(self, f, t, ctx, drain):
        """NOW: the end of the axis. In 11.1 it travels to the focus point and becomes the centre of the whirl
        = the red dot of the first image of the show."""
        al = float(smoothstep(738.5, 745.0, t))       # before that it is drawn with the figure, on the axis
        if al <= 0.01:
            return
        x, y, _, m = self._path(np.array([1.0]), t)
        x, y = float(x[0]), float(y[0])
        kick = min(1.5, ctx.cues.kick(t, 0.2))
        pulse = 0.5 + 0.5 * math.sin(t * 2.2) ** 2
        big = float(m[0])
        # when everything has drained into it, it is the dot of the first image again (origin.py): a red disc
        # breathing with the low thumps, a thin white ring around it
        p0 = min(1.5, ctx.cues.kick(t, 0.28) * 1.6 + 0.5 * ctx.cues.onset(t, 0.3))
        r_first = 16.0 + 4.0 * math.sin(2 * math.pi * t / 2.9) + 13.0 * p0
        r = (9.0 + 2.0 * big + 2.5 * pulse + 3.0 * kick) * (1 - drain) + r_first * drain
        f.dots("r", [x], [y], r, (1.6 - 0.35 * drain) * al)
        f.dots("w", [x], [y], 2.8, 1.3 * al * (1 - drain))
        if big > 0.5 and drain < 1.0:
            f.rings("r", [x], [y], [28.0 + 8.0 * pulse + 10.0 * kick], 0.6 * al * big * (1 - drain), width=L.LW)
        if drain > 0.0:
            f.rings("w", [x], [y], [r + 3.0], 0.75 * drain, width=1.5)
        for a, b, _ in self.now_runs:     # its name rides with it: made when it has room, taken apart before a tower
            if a <= t < b:
                B.tag(f, "r", x + 22, y - 18, "NOW", B.io(t - a, b - t, out=0.2, span=0.3), size=L.T_TAG, pad=5, cps=40.0,
                      key=9)
                break

    def _power_down(self, f, t, ctx):
        """Under the credits, when the drums stop: the detectors power down one by one (centre, right, left).
        The light of a tower dims; its name falls apart and its scope is taken apart (they do not fade).
        After that the show keeps the towers as dark bands: they never leave the wall."""
        fr = int(t * 30)
        for k, key in enumerate(L.ORDER):
            tw = ctx.towers[key]
            fade = float(1 - smoothstep(T_PWR + PWR_ORDER[key], T_PWR + PWR_ORDER[key] + 1.3, t))
            if fade <= 0.01:
                towers.dark(f, tw)
            else:
                towers.face(f, tw, t, power=1.0, value=0.0, dim=0.0, label=False)
                f.dim(tw.x0 - 3, tw.top - 3, tw.x1 + 3, tw.bot + 3, fade)
                f.rect("w", tw.x0, tw.top, tw.x1, tw.bot, 0.16 * (1 - fade), width=L.LW)
                f.text("w", tw.cx, tw.top - 14, hud.erode(L.NAMES[key], 1.0 - fade, 60 + k, fr), size=L.T_MICRO,
                       alpha=0.85, anchor="ms")
            rect = ctx.slots["scopes"].get(key)
            age = B.io(99.0, T_PWR + PWR_ORDER[key] + 0.9 - t, out=0.5)
            if rect is None or age < 0.0:
                continue
            tt = t - 3.0 + np.linspace(0.0, 3.0, 150)
            on = ctx.det.online(key, t)
            v = ctx.det.value(key, tt) if on else 0.02 + 0.0 * tt
            h_age, e = ctx.det.last(key, t, echoes=True)
            hot = e * math.exp(-h_age / 0.4) if h_age < 3 and on else 0.0
            with f.build(age, (rect[0] - 26.0, rect[1] - 24.0, rect[2] + 4.0, rect[3] + 4.0), wave=0.3, key=20 + k):
                hud.scope(f, rect, v, f"DETECTOR {k + 1}", f"/MUON/{key} {float(v[-1]):.2f}", hot=hot)

    def _head(self, f, x0, x1, y, title, gone, key, fr):
        """Header of a bottom panel that is also taken apart: its rule retracts, its title falls apart and its
        tag closes (gone = 0..1). No fade."""
        if gone <= 0.0:
            hud.panel_header(f, x0, x1, y, title)
            return
        f.rects("w", x0, y, x0 + (x1 - x0) * (1.0 - gone), y + 5, 0.95)
        wipe = 1.0 - float(B.ease(B.lin(gone, 0.45, 0.9)))
        if wipe > 0.02:
            f.tag("w", x0 + 4, y - 9, hud.erode(title, min(1.0, 1.6 * gone), key, fr), size=L.T_MICRO, pad=3, ref=title,
                  wipe=wipe)

    def _bottom(self, f, t, ctx):
        """Bottom band: the reading of the traveller (11.0), then the count of arrivals (11.1 / 11.2). The panels
        are constructed, taken apart when the count takes their place, and fall apart under the credits."""
        c = self.c
        main, second, y0, y1 = self._bottom_slots(ctx)
        fr = int(t * 30)
        if main and T_AXIS[0] <= t < c["universe"]:
            left = c["universe"] - 0.1 - t
            xa0, xa1 = main
            xd = self._dot_x(t)
            sec = 10 ** self.lt_at(xd)
            val = f"-{fmt_time(sec)}" if t < self.t_arr else "NOW"
            with f.build(B.io(t - T_AXIS[0] - 0.3, left, out=0.45), (xa0 - 10, y0 - 26, xa1 + 10, L.FY1 - 4), key=36):
                hud.panel_header(f, xa0, xa1, y0, "TIME BEFORE NOW")
                f.text("r" if t >= self.t_dep else "w", xa0, y0 + 98, val, size=fit(58, 9, xa1 - xa0))
            if second:
                xb0, xb1 = second
                sp = "0.999 999 999 999 5" if (sec > T_ATM or t < self.t_dep) else "0.999 444 3"
                with f.build(B.io(t - T_AXIS[0] - 0.45, left, out=0.45), (xb0 - 10, y0 - 26, xb1 + 10, L.FY1 - 4),
                             key=37):
                    hud.panel_header(f, xb0, xb1, y0, "SPEED // FRACTION OF C" if xb1 - xb0 >= 200 else "SPEED / C")
                    f.text("w", xb0, y0 + 90, sp, size=fit(40, 19, xb1 - xb0))
                    if xb1 - xb0 >= 330:
                        f.text("w", xb0, y0 + 122, "PROTON, THEN FOR THE LAST 15 KM A MUON", size=L.T_MICRO, alpha=0.6)
        elif main and c["universe"] <= t < T_END + 3.0:
            gone = float(smoothstep(T_END - 1.5, T_END + 2.5, t))          # the credits take over: it falls apart
            age = t - c["universe"]
            xa0, xa1 = main
            w = xa1 - xa0
            arrivals = float(np.maximum(self._phi(t) - self._phi_v(self.d_act) + self.d_ph, 0.0).astype(int).sum())
            flying = int((t > self.d_act).sum())
            two = w >= 450
            size = fit(54, 6, (w * 0.5 - 30) if two else w)
            with f.build(age, (xa0 - 10, y0 - 26, xa1 + 10, L.FY1 - 4), key=38):
                self._head(f, xa0, xa1, y0, "IT CONTINUES // ARRIVALS", gone, 3, fr)
                f.text("w", xa0, y0 + 96, hud.erode(f"{int(arrivals):,}".replace(",", " "), gone, 4, fr), size=size)
                if two:
                    xm = xa0 + max(280.0, w * 0.52)
                    f.text("w", xm, y0 + 40, hud.erode("ON THEIR WAY", gone, 5, fr), size=L.T_MICRO, alpha=0.75)
                    f.text("w", xm, y0 + 96, hud.erode(f"{flying:,}".replace(",", " "), gone, 6, fr), size=size)
            if second:
                xb0, xb1 = second
                with f.build(age - 0.15, (xb0 - 10, y0 - 26, xb1 + 10, L.FY1 - 4), key=39):
                    self._head(f, xb0, xb1, y0, "ACCELERANDO >> BARCODE" if xb1 - xb0 >= 210 else "ACCELERANDO", gone,
                               7, fr)
                    cols = max(30, int((xb1 - xb0) / 3.7))
                    dt = 3.0 / 130
                    kk = math.floor((t - 3.0) / dt) + np.arange(cols)
                    lv = min(1.0, (t - c["universe"]) / (self.t_hit - c["universe"]))
                    dens = np.full(cols, (0.06 + 0.8 * lv ** 1.5) * (1.0 - gone) ** 2)
                    hud.barcode_lanes(f, xb0, xb1, y0 + 12, y1, dens, kk, lanes=3, seed=11)
        # the count the voice announced at 04:41: more than fifty thousand. The tag is made on that muon, and
        # flashes; the frame of the counter flashes red with it
        t50 = 50000 / ctx.RATE_YOU
        if t >= t50:
            x0, yc, x1, _ = ctx.slots["cell"]
            fl = math.exp(-(t - t50) / 1.5)
            if x1 - x0 >= 260:
                B.tag(f, "r", x0 + 2, yc - 12, "MORE THAN FIFTY THOUSAND", t - t50, size=L.T_SMALL, pad=4,
                      alpha=0.75 + 0.25 * fl, cps=50.0, key=13, commit=True)
            if fl > 0.05:
                f.rect("r", x0, yc, x1, L.FY1, 1.5 * fl, width=L.LW_FRAME)

    # --- credits ---------------------------------------------------------------------
    def _credits(self, f, t, ctx):
        """Over the last drums of the whirl, then on the lattice and the dot. Each block sits in a column of
        its own, clear of the towers and of the dot. Names are placeholders.
        Nothing fades: the title tag is pushed out and its letters decoded, every line is decoded in turn, the
        count spins before it locks; at the end the lines are taken apart in reverse order, the title last."""
        if t >= T_CREDITS_OUT[1]:
            return
        there = t < T_CREDITS_OUT[0]          # (no flash on the way out)

        def age(t_in, rank, span):            # rank = place in the taking apart (0 = the first to go)
            return B.io(t - t_in, T_CREDITS_OUT[0] + 0.5 + 0.28 * rank - t, out=0.5, span=span)

        if self.cred["A"]:
            x0, x1 = self.cred["A"]
            w = x1 - x0
            ts = fit(104, 12, w - 40)
            s = float(np.clip(fit(26, 39, w), 14, 26))          # the longest credit line is 38 characters
            head = hud.wrap("A LIVE PERFORMANCE FOR THREE COSMIC-RAY MUON DETECTORS", int(w / (s * CHAR_W)))
            lines = [(ln, 0.9) for ln in head] + [("BLINK // CINCINNATI // OCTOBER 2026", 0.9), ("", 0),
                                                  ("CONCEPT + MUSIC ............ NAME TBC", 0.75),
                                                  ("VISUALS .................... NAME TBC", 0.75),
                                                  ("DETECTORS .................. NAME TBC", 0.75),
                                                  ("VOICE ...................... NAME TBC", 0.75),
                                                  ("WITH THANKS TO ............. NAMES TBC", 0.75)]
            n = len(lines)
            B.tag(f, "w", x0 + ts * 0.15, 380 + ts * 0.86, "MUON : BLOOM", age(T_END, 7 + n, 1.3), size=ts, pad=ts * 0.15,
                  bold=True, cps=14.0, lead=3, wipe=0.3, key=14, commit=there)
            y = 380 + ts * 1.3 + s * 2.2
            for k, (ln, al) in enumerate(lines):
                if ln:
                    a = age(T_END + 1.2 + 0.35 * k, 6 + (n - 1 - k), len(ln) / 60.0 + 0.4)
                    f.text("w", x0, y + k * s * 1.55, B.resolve(ln, a, cps=60.0, key=20 + k), size=s, alpha=al)
        if self.cred["B"]:
            x0, x1 = self.cred["B"]
            w = x1 - x0
            tb = T_END + 4.0
            num = f"{ctx.through_you(T_END):,}".replace(",", " ")
            ns = fit(120, len(num), w - 8)
            s = float(np.clip(fit(26, 27, w), 14, 26))
            B.tag(f, "w", x0, 392, "WHILE YOU WATCHED", age(tb, 5, 0.8), size=L.T_TAG, pad=5, cps=30.0, key=15)
            yn = 410 + ns * 0.92
            f.text("r", x0 - 4, yn, B.resolve(num, age(tb + 0.5, 4, 1.2), cps=12.0, spin=0.6, key=16), size=ns)
            f.text("w", x0, yn + s * 1.8, B.resolve("MUONS WENT THROUGH YOU", age(tb + 1.2, 3, 0.85), cps=50.0, key=17),
                   size=s, alpha=0.9)
            caught = f"THE THREE TOWERS CAUGHT {ctx.det.total(T_END)}"
            f.text("w", x0, yn + s * 5.4, B.resolve(caught, age(tb + 2.0, 2, 1.0), cps=50.0, key=18), size=s, alpha=0.8)
            f.text("w", x0, yn + s * 6.95, B.resolve("YOU FELT NONE OF THEM", age(tb + 2.8, 1, 0.85), cps=50.0, key=19),
                   size=s, alpha=0.8)
            B.tag(f, "r", x0, yn + s * 10.4, "IT CONTINUES", age(tb + 4.0, 0, 0.9), size=min(40.0, s * 1.55), pad=8,
                  cps=20.0, key=19, commit=there)
