"""MUON - "...A muon".   Sheet 1.4, 01:41.0 - 01:44.0 ("transition into next section, static").

The track of the previous scene finally gets its name. Three hard seconds:
  01:41.00  the wall goes to static (test-pattern bars, a new pattern every frame)
  01:41.15  two plates open in it, like tubes switching on: the identity of the particle
              focus bay   MUON as one large inverted tag, its symbol, its place among the
                          twelve fermions (second generation, the heavy cousin of the electron)
              side bay    the data table and its decay
  01:43.45  the static closes over the plates again and hands over to the detectors (01:44.0)

LAYOUT. The towers stand in front of the wall (dark until 01:44): the static runs behind them,
the plates never do. They take the focus column and the widest other column from the tower
placement (see you.Plan); with a single usable bay the table moves under the name. Type sizes
follow the width of the plates.
"""
from __future__ import annotations

import numpy as np

from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import hash01, smoothstep, text_w
from ..show import Scene
from .you import Plan, fit_text, width

T0, T1 = 101.0, 104.0
T_OPEN, T_CLOSE = 101.15, 103.45
PLATE_Y = (262.0, 1170.0)                     # vertical extent of the plates
ROW_H = 14.0
MU, MINUS, ARROW = "µ", "−", "→"

# (label, value, short value for narrow plates)
ROWS = [("NAME", "MUON", "MUON"), ("SYMBOL", MU + MINUS, MU + MINUS),
        ("FAMILY", "LEPTON // 2ND GENERATION", "LEPTON // GEN 2"),
        ("MASS", "105.658 MEV/C2  = 207 ELECTRONS", "105.658 MEV/C2"), ("CHARGE", MINUS + "1 E", MINUS + "1 E"),
        ("SPIN", "1/2", "1/2"), ("LIFETIME", f"2.197 {MU}S  AT REST", f"2.197 {MU}S AT REST"),
        ("BORN", "15 KM UP // PION DECAY", "15 KM UP // PI DECAY"), ("SPEED", "0.9997 C", "0.9997 C"),
        ("DISCOVERED", "1936 // ANDERSON + NEDDERMEYER", "1936")]
FERMIONS = [["U", "C", "T"], ["D", "S", "B"], ["E", MU, "TAU"], ["NU E", "NU " + MU, "NU TAU"]]
KIND = ["QUARKS", "", "LEPTONS", ""]
LABEL_CH = 12.0                               # width of the label column of the table, in characters


def static(f, t, rect, density, seed=0, layer="w", row_h=ROW_H, inten=0.95):
    """Test-pattern static: rows of bars of random width, a new pattern every frame."""
    x0, y0, x1, y1 = rect
    fr = int(round(t * 30))
    n_rows = int((y1 - y0) / row_h)
    r = np.arange(n_rows)
    k = np.arange(170)
    R, K = np.meshgrid(r, k, indexing="ij")
    wdt = 3.0 + (hash01(R, K, seed + 1) ** 2.2) * 54.0
    xs = np.cumsum(wdt, 1) - wdt
    shift = hash01(R, fr, seed + 2) * 400.0
    xs = x0 + (xs + shift) % (x1 - x0)
    dens = np.broadcast_to(np.asarray(density, np.float64).reshape(-1, 1) if np.ndim(density) else density, R.shape)
    on = (hash01(R, K, fr + 7 * seed) < dens) & (xs + wdt < x1)
    Y = y0 + R * row_h
    f.rects(layer, xs[on], Y[on] + 1, (xs + wdt)[on], Y[on] + row_h - 2, inten)


def table_font(w, short):
    """Label size of the table so that its longest row fits a plate of width w."""
    longest = max(len(r[2] if short else r[1]) for r in ROWS)
    return 0.96 * (w - 80.0) / (0.61 * (LABEL_CH + 1.13 * longest))


class Muon(Scene):
    name = "muon"
    towers = "none"
    scopes = False

    def __init__(self, ctx):
        super().__init__(ctx)
        self.P = P = Plan(ctx)
        # two plates: the name in the focus column, the table in the best other column (whatever the YOU scene
        # used it for). One very wide bay and nothing else: a single plate, the name left, the table right.
        others = [c for c in (P.side, P.title, P.data) if c is not None and width(c) >= 300.0]
        self.name_col = P.fcol
        self.table_col = P.side if P.side is not None else (max(others, key=width) if others else None)
        self.plates = [P.fcol] + ([self.table_col] if self.table_col is not None else [])
        if self.table_col is None and width(P.fcol) >= 1300.0:
            split = P.fcol[0] + 0.54 * width(P.fcol)
            self.name_col, self.table_col = (P.fcol[0], split), (split, P.fcol[1])

    def draw(self, f, t, ctx):
        t = float(np.clip(t, T0, T1 - 1e-3))
        P = self.P
        y0, y1 = PLATE_Y
        cy = (y0 + y1) / 2
        # aperture of the plates: they open from a line, close back to a line
        ap = float(smoothstep(T_OPEN, T_OPEN + 0.22, t) * (1.0 - smoothstep(T_CLOSE, T_CLOSE + 0.3, t)))
        loud = ctx.cues.loud(t, 0.05)
        # static: dense at the cut, thin while the card is read, swelling with the noise of the music
        base = 0.46 if t < T_OPEN else 0.13 + 0.22 * smoothstep(102.2, 103.9, t) + 0.1 * loud
        if t >= T_CLOSE:
            base = max(base, 0.3 + 0.25 * smoothstep(T_CLOSE, T1, t))
        n_rows = int((L.FY1 - L.FY0 - 8) / ROW_H)
        ry = L.FY0 + 4 + (np.arange(n_rows) + 0.5) * ROW_H
        band = 0.55 + 0.45 * np.sin(ry * 0.013 + t * 9.0) * np.sin(ry * 0.0041 - t * 3.1)
        dens = base * (0.55 + 0.75 * band ** 2)
        wall = (L.FX0 + 4, L.FY0 + 4, L.FX1 - 4, L.FY1 - 4)
        static(f, t, wall, dens, seed=3)
        # a few red bars: the track is still in there somewhere
        static(f, t, wall, 0.012 + 0.03 * (t < T_OPEN), seed=11, layer="r")
        opt = {"edge_ticks": False, "cell": t >= T_OPEN}
        if ap <= 0.004:
            return opt
        hh = (y1 - y0) / 2 * ap
        for (x0, x1) in self.plates:
            f.occlude(x0, cy - hh, x1, cy + hh)
            f.rect("w", x0, cy - hh, x1, cy + hh, 0.95, width=L.LW_FRAME)
            if ap < 0.98:
                f.rects("w", x0, cy - 2, x1, cy + 2, 1.0 - ap)
        if ap < 0.98:
            return opt
        age = t - T_OPEN - 0.22
        fr = int(t * 30)
        gone = float(smoothstep(T_CLOSE - 0.5, T_CLOSE, t))             # the card starts to break up
        f.set_clip(self.name_col[0] + 4, y0 + 4, self.name_col[1] - 4, y1 - 4)
        y_free = self._name_plate(f, self.name_col, age, gone, fr, grid=self.table_col is not None)
        if self.table_col is None:
            self._table(f, self.name_col, y_free + 26, age, gone, fr, header=False)
        else:
            f.set_clip(self.table_col[0] + 4, y0 + 4, self.table_col[1] - 4, y1 - 4)
            self._table(f, self.table_col, y0, age, gone, fr, header=True)
        f.set_clip()
        return opt

    # ------------------------------------------------------------------ the name
    def _name_plate(self, f, col, age, gone, fr, grid=True):
        """IDENTIFIED / MUON / its symbol / the twelve fermions. Returns the y under what it drew."""
        x0, x1 = col
        y0, y1 = PLATE_Y
        w = x1 - x0
        xl = x0 + 40
        f.tag("w", xl, y0 + 62, "IDENTIFIED", size=L.T_TAG, pad=5)
        room = w - 80 - 180
        head = fit_text(["TRACK 0001 // 01:25.42 // THROUGH THE HEART", "TRACK 0001 // THROUGH THE HEART",
                         "TRACK 0001"], room, L.T_LABEL)
        if text_w(head, L.T_LABEL) <= room:
            f.text("w", xl + 180, y0 + 62, hud.erode(head, gone, 1, fr), size=L.T_LABEL, alpha=0.85)
        f.segments("w", [xl], [y0 + 86], [x1 - 40], [y0 + 86], 0.8, width=L.LW)
        # the name, as large as the plate allows
        s = float(np.clip((w - 80 - 52) / (4 * 0.61), 80.0, 330.0))
        base = y0 + 122 + 26 + 0.72 * s
        f.tag("w", xl + 26, base, hud.typed("MUON", age, cps=40), size=s, pad=26, bold=True)
        yb = base + 30
        m = max(56.0, 0.36 * s)
        f.text("r", xl, yb + 26 + 0.78 * m, hud.typed(MU + MINUS, age, cps=30, delay=0.12), size=m, alpha=1.0, bold=True)
        tx = xl + 2 * 0.61 * m + 34
        notes = ["ELEMENTARY // POINT-LIKE // NO PARTS", "THE ELECTRON'S HEAVY COUSIN"]
        ns = L.T_TAG if text_w(notes[0], L.T_TAG) <= x1 - 40 - tx else L.T_SMALL
        if text_w(notes[0], ns) > x1 - 40 - tx:
            notes = ["ELEMENTARY // NO PARTS", "A HEAVY ELECTRON"]
        for k, ln in enumerate(notes):
            f.text("w", tx, yb + 26 + 0.38 * m + k * (ns * 1.55), hud.erode(hud.typed(ln, age, cps=140, delay=0.15 + 0.1 * k),
                                                                         gone, 3 + k, fr), size=ns, alpha=0.9)
        y = yb + 26 + m + 34
        if not grid:
            return y
        # the twelve fermions, the muon lit
        ch = 50.0
        if y + 30 + 4 * (ch + 7) > y1 - 26:
            ch = (y1 - 26 - y - 30) / 4 - 7
        if ch < 30:
            return y
        lab_w = 150.0 if w >= 620 else 0.0
        cw = min(205.0, (w - 80 - lab_w) / 3)
        gx, gy = xl, y + 30
        f.tag("w", gx, gy - 10, fit_text(["FERMIONS // 12 // GENERATION  I  II  III", "FERMIONS // 12"], w - 80, L.T_MICRO),
              size=L.T_MICRO, pad=3)
        fs = min(28.0, ch * 0.56, (cw - 16) / (6 * 0.61))
        for r in range(4):
            for c in range(3):
                a = age - 0.2 - 0.025 * (r * 3 + c)
                if a < 0:
                    continue
                X, Y = gx + lab_w + c * cw, gy + 10 + r * (ch + 7)
                mu = (r, c) == (2, 1)
                f.rect("r" if mu else "w", X, Y, X + cw - 10, Y + ch, 0.95 if mu else 0.55, width=L.LW)
                if mu:
                    f.rects("r", X + 4, Y + 4, X + cw - 14, Y + ch - 4, 0.95)
                    f.text("w", X + (cw - 10) / 2, Y + ch * 0.5 + fs * 0.42, FERMIONS[r][c], size=fs * 1.3, anchor="ms",
                           bold=True)
                else:
                    f.text("w", X + (cw - 10) / 2, Y + ch * 0.5 + fs * 0.34, FERMIONS[r][c], size=fs, alpha=0.7,
                           anchor="ms")
            if KIND[r] and lab_w and age > 0.2:
                f.text("w", gx, gy + 10 + r * (ch + 7) + ch * 0.68, KIND[r], size=L.T_LABEL, alpha=0.75)
        return gy + 10 + 4 * (ch + 7)

    # ------------------------------------------------------------------ the data
    def _table(self, f, col, y_top, age, gone, fr, header=True):
        x0, x1 = col
        y1 = PLATE_Y[1]
        w = x1 - x0
        xl, xr = x0 + 40, x1 - 40
        if header:
            f.tag("w", xl, y_top + 62, "PARTICLE DATA", size=L.T_TAG, pad=5)
            if w >= 560:
                f.text("w", xr, y_top + 62, hud.erode("MUON BLOOM", gone, 2, fr), size=L.T_LABEL, alpha=0.7,
                       anchor="rs")
            f.segments("w", [xl], [y_top + 86], [xr], [y_top + 86], 0.8, width=L.LW)
            y_top += 96
        short = table_font(w, False) < 24.0
        avail = y1 - 26 - y_top
        rows = ROWS if avail >= 620 else ROWS[:7] if avail >= 330 else ROWS[:4]
        decay = avail >= 620
        ls = float(np.clip(table_font(w, short), 14.0, 30.0))
        ls = min(ls, (avail - (190 if decay else 20)) / (len(rows) * 2.07))
        vs = 1.13 * ls
        pitch = 2.07 * ls
        ty = y_top + 1.9 * ls
        vx = xl + LABEL_CH * 0.61 * ls
        for k, (lab, val, sval) in enumerate(rows):
            a = age - 0.05 - 0.035 * k
            if a < 0:
                continue
            y = ty + k * pitch
            f.text("w", xl, y, hud.erode(lab, gone, 10 + k, fr), size=ls, alpha=0.6)
            red = lab in ("NAME", "LIFETIME")
            f.text("r" if red else "w", vx, y, hud.erode(hud.typed(sval if short else val, a, cps=170), gone, 30 + k, fr),
                   size=vs, alpha=0.97)
            f.segments("w", [xl], [y + 0.6 * ls], [xr], [y + 0.6 * ls], 0.22)
        if not decay:
            return
        # decay: the reaction, then N / N0 against its own clock
        yd = ty + len(rows) * pitch + 0.5 * ls
        a = age - 0.45
        if a <= 0:
            return
        f.tag("w", xl, yd, "DECAY", size=L.T_TAG, pad=5)
        eq = fit_text([f"{MU}{MINUS}  {ARROW}  E{MINUS}  +  ANTI-NU E  +  NU {MU}", f"{MU}{MINUS} {ARROW} E{MINUS} + 2 NU"],
                      xr - xl - 110, vs)
        f.text("w", xl + 110, yd, hud.erode(hud.typed(eq, a, cps=130), gone, 50, fr), size=min(vs, 30.0), alpha=0.95)
        cyb = min(y1 - 44, yd + 118)
        hgt = cyb - yd - 34
        if hgt < 24:
            return
        xs = np.linspace(xl, xr, 160)
        u = (xs - xl) / (xr - xl) * 5.0
        n = int(min(len(xs), a * 420))
        f.polyline("r", xs[:n], cyb - hgt * np.exp(-u[:n]), 1.0, width=L.LW_BOLD)
        f.segments("w", [xl], [cyb], [xr], [cyb], 0.6)
        f.text("w", xr, cyb - hgt + 12, "N / N0", size=L.T_MICRO, alpha=0.7, anchor="rs")
        for k in range(6):
            xk = xl + k / 5.0 * (xr - xl)
            f.segments("w", [xk], [cyb], [xk], [cyb + 9], 0.8)
            if (xr - xl) / 5.0 >= 150.0 or k % 2 == 0:
                f.text("w", xk + (4 if k < 5 else -4), cyb + 26, f"{k * 2.197:.1f} {MU}S" if k else "0", size=L.T_MICRO,
                       alpha=0.7, anchor="ls" if k < 5 else "rs")
