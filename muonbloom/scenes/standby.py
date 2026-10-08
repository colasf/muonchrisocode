"""STANDBY - what the wall shows between two shows.   12:38.5 - 20:00 (the loop is 20 minutes)

The music has ended and the wall is black. Until the next show the instrument stays on and shows what its three
detectors catch: three generative scenes, one simple system each - no voice, no sound to follow, only the muons -
after a minute that says where to find us:

  13.0  FOLLOW        two codes to scan (Instagram: Tyrell, Christo Squier), large enough for brick
  13.1  FLUX          how many   the lattice of crosses is the metre grid of the wall; every square metre says how
                                 many muons went through it in its last second (1 /cm2/min = 167 a second, never
                                 twice the same). When a tower really catches one, its square metre is framed and a
                                 red wave leaves it: the figures it passes are taken again
  13.2  COINCIDENCE   where      every muon caught sends a circle over the wall from its detector, at walking
                                 pace. A dashed circle leaves the three towers together every few seconds: the
                                 time base. Where the circles of two towers cross, a red point rides the crossing
  13.3  RECORD        when       the last three quarters of a minute as a stack of lines. The line at the bottom is
                                 now: a muon raises it at its tower and the swell runs out along the wall; every
                                 second a copy of it is kept and moves up. A red point = a muon

Everything is a function of the show time and of the hits known at that time (nothing is read ahead: with live
detectors the picture is made of what the towers say; in the previews the hits after the muon stem are pretend
ones, showdata.Detectors). The header says where the loop is and when the next show starts.

NOTHING THAT SHOWS DATA FADES IN OR POPS IN: the frame, the towers, the scopes and the header are constructed
when the standby starts and taken apart before the loop goes back to 00:00 (a black wall, as the show expects);
the title and the picture of each scene are made at its cut and taken apart before the next one.

The wall rule holds here from the start: nothing grey, no type under 22 px of our own, lines of 1.8 px and more.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import hud, towers
from .. import layout as L
from .. import showdata as sd
from ..engine import CHAR_W, text_w
from ..show import Scene

LOOKS = ("follow", "flux", "coincidence", "record")
T0, T1 = sd.TRACK_END, sd.LOOP_END                  # the standby: from the end of the music to the end of the loop
SPAN = {s[4]: (s[2], s[3]) for s in sd.SECTIONS if s[4] in LOOKS}
OUT = 1.3                                           # s: a scene takes its picture apart before the next one
END = 1.7                                           # s before the loop ends: the last scene is gone, the furniture goes
VIEW_Y0, VIEW_Y1 = L.HEAD_Y + 8.0, 1184.0           # the picture lives between the header and the bottom band
BURST = 0.6                                         # size of the red reply of a tower (towers.burst) ...
BURST_EVERY = 0.3                                   # ... one in that many seconds when a tower fires all the time

M = 3000.0 / 27.737                                 # px per metre of wall (the delivery raster: 3000 px for 91 ft)
GROUND = 1667.0 - 272.0                             # canvas y of the ground
X_WALL = -11.0                                      # canvas x of the left end of the wall
RATE_M2 = 1.0e4 / 60.0                              # muons per second through one square metre (1 /cm2/min)


def mmss(s):
    s = max(0.0, s)
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


class Sparse:
    """The hits of the detectors, thinned when a tower fires all the time (a trigger level set in the noise):
    such a stream must not fill the wall, nor cost a frame its sixteen milliseconds. Every muon is there as long
    as the tower is calm; a hit that comes when the tower has already fired CALM times in the last WIN seconds
    is only kept if it is the first of its slot of `every` seconds. The rule only looks back: it never changes
    its mind afterwards, so live detectors can be read through it."""
    CALM, WIN = 6, 2.0

    def __init__(self, det, every):
        self.det, self.every = det, every

    def hits(self, key, t0, t1, echoes=True):
        tt, ee, ec = self.det.hits(key, t0 - self.every - self.WIN - 1e-3, t1, echoes)
        if self.every > 0.0 and len(tt) > self.CALM:
            busy = np.arange(len(tt)) - np.searchsorted(tt, tt - self.WIN) >= self.CALM
            slot = np.floor(tt / self.every)
            keep = ~busy | np.r_[True, slot[1:] != slot[:-1]]
            tt, ee, ec = tt[keep], ee[keep], ec[keep]
        m = tt >= t0
        return tt[m], ee[m], ec[m]


def rand(a, b, salt=0):
    """Pseudo-random floats in [0, 1) from two integer arrays (well mixed in both: engine.hash01 is not)."""
    with np.errstate(over="ignore"):
        x = (np.atleast_1d(np.asarray(a, np.int64)).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             ^ (np.atleast_1d(np.asarray(b, np.int64)).astype(np.uint64) + np.uint64(salt)) * np.uint64(0xC2B2AE3D27D4EB4F))
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xFF51AFD7ED558CCD)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xC4CEB9FE1A85EC53)
        x ^= x >> np.uint64(33)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


class Standby(Scene):
    """What the scenes of the standby share: the furniture and its life cycle."""
    look = "flux"
    title = "STANDBY"
    towers = "own"
    scopes = False
    edge_ticks = False
    frame = False
    strip_grows = False

    def __init__(self, ctx):
        super().__init__(ctx)
        self.t0, self.t1 = SPAN.get(self.look, (T0, T1))
        self.t_out = self.t1 if self.t1 < T1 - 1e-6 else T1 - END       # when what this scene made is gone
        self.no = LOOKS.index(self.look)

    # ------------------------------------------------------------------ life cycle
    @staticmethod
    def chrome(t, lag=0.0, out=0.7, span=1.2):
        """Build age of a piece of the furniture: made when the standby starts (`lag` seconds after the frame),
        taken apart before the loop ends - the last made is the first to go, the frame is the last."""
        return B.io(t - T0 - lag, T1 - 0.3 - 0.5 * lag - t, out=out, span=span)

    def age(self, t, lag=0.0, out=OUT, span=1.3):
        """Build age of a piece of this scene: made at its cut, taken apart before the next one."""
        return B.io(t - self.t0 - lag, self.t_out - t, out=out, span=span)

    def gone(self, t, dur=OUT):
        """0..1: how far the picture of this scene has been taken apart (1 when the next one starts)."""
        return float(B.ease(B.lin(t, self.t_out - dur, self.t_out - 0.1)))

    # ------------------------------------------------------------------ detectors
    @staticmethod
    def hits(ctx, key, ta, tb, every=0.0):
        """Hits of one tower in [ta, tb): (times, energies), echoes included. every > 0: thinned to one per slot
        of that many seconds when the tower fires all the time (see Sparse)."""
        return Sparse(ctx.det, every).hits(key, ta, tb)[:2]

    @staticmethod
    def detected(ctx, t):
        """Muons caught by the three towers since the standby started."""
        return sum(len(ctx.det.hits(k, T0, t + 1e-6, echoes=False)[0]) for k in sd.KEYS)

    @staticmethod
    def rate(ctx, t, win=20.0):
        return sum(len(ctx.det.hits(k, t - win, t + 1e-6, echoes=False)[0]) for k in sd.KEYS) / win

    # ------------------------------------------------------------------ furniture
    def furniture(self, f, t, ctx):
        """Frame, towers, scopes, header, log. Called by a scene after its picture (the towers stand in front of
        it). Returns the options of the frame."""
        f.set_clip()
        f.set_view()
        with f.build(self.chrome(t, 0.0, out=1.0, span=1.25), L.FRAME, wave=0.0, line=1.0, marks=False, key=1):
            hud.frame(f)
        self._towers(f, t, ctx)
        self._scopes(f, t, ctx)
        self._header(f, t, ctx)
        self._log(f, t, ctx)
        return {"cell_age": self.chrome(t, 0.9, out=0.6, span=hud.CELL_BUILD)}

    def _towers(self, f, t, ctx):
        clip = (L.FX0 + 2, L.HEAD_Y + 6, L.FX1 - 2, L.FY1 - 2)
        for i, key in enumerate(L.ORDER):
            tw = ctx.towers[key]
            f.occlude(tw.x0, tw.top, tw.x1, tw.bot)             # a tower stands in front of the wall
            a = self.chrome(t, 0.4 + 0.25 * i, out=0.7, span=1.1)
            age, e = ctx.det.last(key, t, echoes=True)
            with f.build(a, (tw.x0 - 4, tw.top - 30, tw.x1 + 4, tw.bot), flow="bt", wave=0.5, marks=False, key=30 + i):
                towers.face(f, tw, t, power=1.0, value=float(ctx.det.value(key, t)), hit_age=age, hit_e=e, dim=0.0)
            if a > 1.0:
                towers.burst(f, tw, t, Sparse(ctx.det, BURST_EVERY), clip=clip, size=BURST)

    def _scopes(self, f, t, ctx, span=3.0):
        for k, key in enumerate(L.ORDER):
            if key not in ctx.slots["scopes"]:
                continue
            rect = ctx.slots["scopes"][key]
            tt = t - span + np.linspace(0.0, span, 150)
            v = ctx.det.value(key, tt)
            age, e = ctx.det.last(key, t, echoes=True)
            hot = e * math.exp(-age / 0.4) if age < 3 else 0.0
            with f.build(self.chrome(t, 0.8 + 0.2 * k), (rect[0] - 26.0, rect[1] - 24.0, rect[2] + 4.0, rect[3] + 4.0),
                         wave=0.3, key=20 + k):
                hud.scope(f, rect, v, f"DETECTOR {k + 1}", f"/MUON/{key} {float(v[-1]):.2f}", hot=hot)

    def _header(self, f, t, ctx):
        """Header band: the name of this scene (left), where the loop is and when the next show starts (right).
        The name stands alone: a line above it and a legend beside it were taken out (user, 2026-10-08)."""
        x, y = 96.0, L.FY0 + 44.0
        with f.build(self.age(t, 0.1), (x - 8, y + 30, x + 28 + text_w(self.title, 62), y + 116), key=100 + self.no, wave=0.3):
            f.tag("w", x + 4, y + 92, self.title, size=62, pad=10, bold=True)
        # the loop: one rule for the twenty minutes, the show as a bar, the scenes of the standby, now
        xa, xb = 1500.0, 2360.0
        yb = L.FY0 + 112.0
        X = lambda s: xa + (xb - xa) * np.asarray(s, np.float64) / T1
        with f.build(self.chrome(t, 0.5), (xa - 10, L.FY0 + 20, L.FX1 - 30, L.HEAD_Y - 10), key=110, wave=0.5, flow="lr"):
            f.text("w", xa, yb - 24, "SHOW", size=L.T_TAG)
            f.segments("w", [xa], [yb], [xb], [yb], 0.95, width=L.LW_BOLD)
            mins = np.arange(0, int(T1 // 60) + 1)
            f.segments("w", X(mins * 60.0), np.full(len(mins), yb), X(mins * 60.0),
                       yb + np.where(mins % 5 == 0, 16.0, 8.0), 0.95, width=L.LW)
            for m in mins[::5]:
                f.text("w", float(X(m * 60.0)), yb + 42, f"{m:02d}", size=L.T_TAG, anchor="ms")
            f.rects("w", X(0.0), yb - 14, X(T0) - 3, yb - 4, 0.95)                       # the show
            for look in LOOKS:
                a, b = SPAN[look]
                on = look == self.look
                f.rects("r" if on else "w", X(a) + 2, yb - (14 if on else 9), X(b) - 3, yb - 4, 0.95)
            f.text("w", L.FX1 - 44, y + 4, "NEXT SHOW IN", size=L.T_TAG, anchor="rs")
            f.text("w", L.FX1 - 40, L.HEAD_Y - 34, mmss(T1 - t), size=104, anchor="rs", bold=True)
        if self.chrome(t, 1.0) > 0.3:
            xc = float(X(t))
            f.segments("r", [xc], [yb - 22], [xc], [yb + 20], 1.2, width=L.LW_BOLD)

    @staticmethod
    def _panel(f, x0, x1, y0, title):
        """Head of a panel of the bottom band: a thick rule, its title above it (22 px: it has to land)."""
        f.rects("w", x0, y0, x1, y0 + 5, 0.95)
        f.text("w", x0 + 2, y0 - 9, title[: max(int((x1 - x0) / (L.T_TAG * CHAR_W)), 4)], size=L.T_TAG)

    def _log(self, f, t, ctx):
        """Bottom band, first free panel: the last muons caught."""
        panels = ctx.slots["panels"]
        if not panels:
            return
        y0, y1 = ctx.slots["y0"], ctx.slots["y1"]
        x0, x1 = panels[0]
        ev = []
        for key in sd.KEYS:
            tt, ee, _ = ctx.det.hits(key, max(T0, t - 60.0), t + 1e-6, echoes=False)
            ev += [(float(a), key, float(b)) for a, b in zip(tt[-3:], ee[-3:])]
        ev.sort(reverse=True)
        with f.build(self.chrome(t, 1.1), (x0 - 8, y0 - 30, x1 + 8, y1 + 8), key=120):
            self._panel(f, x0, x1, y0, f"CAUGHT SINCE THE SHOW // {self.detected(ctx, t):04d}")
            for k, (th, key, e) in enumerate(ev[:3]):
                line = f"{sd.tc(th)[:8]}  {L.NAMES[key]}  E {e:.2f}"
                if k == 0:
                    line = B.resolve(line, t - th, 110.0, key=121)
                f.text("r" if k == 0 and t - th < 1.5 else "w", x0 + 4, y0 + 46 + k * 32, line, size=L.T_TAG)

    def _figure(self, f, t, ctx, title, value, unit=""):
        """Bottom band, second free panel: one figure of the scene."""
        panels = ctx.slots["panels"]
        if len(panels) < 2:
            return
        y0, y1 = ctx.slots["y0"], ctx.slots["y1"]
        x0, x1 = panels[1]
        with f.build(self.age(t, 0.5), (x0 - 8, y0 - 30, x1 + 8, y1 + 8), key=130 + self.no):
            self._panel(f, x0, x1, y0, title)
            fs = min(62.0, (x1 - x0 - 8.0) / (CHAR_W * max(len(value), 1)))          # (a narrow panel: a smaller figure)
            f.text("w", x0 + 2, y0 + 96, value, size=fs, bold=True)
            if unit and text_w(value, fs) + 26.0 + text_w(unit, L.T_TAG) <= x1 - x0:
                f.text("w", x0 + 2 + text_w(value, fs) + 22, y0 + 96, unit, size=L.T_TAG)


# ----------------------------------------------------------------------------
# 13.0  FOLLOW
# ----------------------------------------------------------------------------

# The two codes: QR version 3, error correction M, byte mode (29 x 29 modules: the largest modules the address
# allows; on the wall model they read better than a denser code with more correction). Made by tools/build_qr.py
# (segno) and read back with another library (zxing-cpp). 1 = a dark module.
QR = (
    dict(name="TYRELL", role="VISUALS", handle="@TYRELL.STUDIO", url="https://www.instagram.com/tyrell.studio/", rows=(
        "11111110110010001011101111111", "10000010110010001000101000001", "10111010011001000010101011101",
        "10111010110001111101001011101", "10111010000101111001101011101", "10000010001111001111101000001",
        "11111110101010101010101111111", "00000000101000001110000000000", "10110111011000100110001001011",
        "01010000011101001111111110001", "00001010100111001110001110110", "00101001010001111011011110001",
        "11101010011101011110000001100", "00001100000001111001001100111", "01110011100100101111010110111",
        "01001001111010101001011110010", "11100110010100110010010011010", "00001001101001011010110101110",
        "10011111111100010000110100100", "00101001000000101101011000100", "01110010001111011111111111100",
        "00000000111011001100100011111", "11111110111101000101101011010", "10000010101100000010100011001",
        "10111010011110100100111110111", "10111010101000111111110011001", "10111010111110000101010100101",
        "10000010010110111010100101010", "11111110111001000011100001010")),
    dict(name="CHRISTO SQUIER", role="CONCEPT + MUSIC", handle="@CHRISTOSQUIER", url="https://www.instagram.com/christosquier/", rows=(
        "11111110000110100110001111111", "10000010100000001110001000001", "10111010111100111001101011101",
        "10111010111111000101101011101", "10111010011000001111001011101", "10000010011110101100101000001",
        "11111110101010101010101111111", "00000000100010100000100000000", "10000010110111111101011001110",
        "01010000101000010111000110110", "11110111011001011000111000000", "00010000001110010000010101000",
        "01101111000100110011101100001", "10010001110001011111101010011", "01010010001101111100001101100",
        "11111000000000010001100110101", "10001011101010000100100101100", "10000001000110011011111110111",
        "11100110010101111011011001001", "10001000101100001001111110000", "10001011111010011100111110111",
        "00000000100011110110100011000", "11111110001011111011101011100", "10000010011001010011100010000",
        "10111010000101010001111111010", "10111010001010011001010101101", "10111010010010111110001111110",
        "10000010011010101010011101101", "11111110110100000101010111100")),
)
QUIET = 4                           # light modules around a code: what a reader needs to find it


def qr_runs(rows, quiet=QUIET):
    """The light of a code, as runs of light modules: (column from, column to, row), in modules, the quiet zone
    included. A code is read dark on light: on a wall the light is what is projected."""
    n = len(rows) + 2 * quiet
    m = np.zeros((n, n), bool)
    m[quiet:-quiet, quiet:-quiet] = np.array([[c == "1" for c in r] for r in rows])
    a, b, y = [], [], []
    for j in range(n):
        d = np.diff(np.r_[False, ~m[j], False].astype(np.int8))
        a += list(np.flatnonzero(d == 1))
        b += list(np.flatnonzero(d == -1))
        y += [j] * int((d == 1).sum())
    return np.array(a, np.float64), np.array(b, np.float64), np.array(y, np.float64), n


class Follow(Standby):
    """Where to find us: two codes to scan, large enough to be read on brick from across the lot."""
    name = look = "follow"
    title = "FOLLOW"
    MODULE = (12.0, 20.0)           # px: a module is at least two courses of brick, at most this
    LEVEL = 2.0                     # the light of a code is full white (it is what a phone reads)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.codes = [qr_runs(q["rows"]) for q in QR]
        n = self.codes[0][3]
        bays = sorted(sorted(ctx.bays, key=lambda ab: ab[0] - ab[1])[:2])           # the two widest, left to right
        room_y = VIEW_Y1 - VIEW_Y0 - 200.0
        if len(bays) == 2 and min(b - a for a, b in bays) >= n * self.MODULE[0] + 60.0:
            mod = float(np.clip(math.floor(min(min(b - a for a, b in bays) - 120.0, room_y) / n), *self.MODULE))
            cxs = [0.5 * (a + b) for a, b in bays]
        else:                                                                       # one wide bay: side by side in it
            a, b = bays[0]
            mod = float(np.clip(math.floor(min((b - a - 180.0) / 2.0, room_y) / n), *self.MODULE))
            cxs = [0.5 * (a + b) - 0.5 * n * mod - 40.0, 0.5 * (a + b) + 0.5 * n * mod + 40.0]
        self.mod, self.size = mod, n * mod
        y0 = round(VIEW_Y0 + 118.0 + 0.5 * (room_y - self.size))
        self.at = [(float(round(cx - 0.5 * self.size)), float(y0)) for cx in cxs]

    def draw(self, f, t, ctx):
        f.set_clip()
        for i, (q, (ra, rb, ry, n), (x0, y0)) in enumerate(zip(QR, self.codes, self.at)):
            sz, mod = self.size, self.mod
            age = self.age(t, 1.2 + 0.5 * i, out=0.9, span=1.6)
            rect = (x0, y0, x0 + sz, y0 + sz)
            if age >= 0.0:
                f.occlude(*rect)
                f.noglow_rects.append((x0 - 10, y0 - 10, x0 + sz + 10, y0 + sz + 10))     # a code is read by its edges
            with f.build(age, rect, flow="diag", wave=1.1, bars="centre", key=170 + i):
                f.rects("w", x0 + ra * mod, y0 + ry * mod, x0 + rb * mod, y0 + (ry + 1) * mod, self.LEVEL)
            with f.build(self.age(t, 0.8 + 0.5 * i), (x0, y0 - 110, x0 + sz, y0 + sz + 80), wave=0.5, marks=False, key=174 + i):
                # (a small code - towers standing close - gets smaller names, the same size for the two)
                ns = min(62.0, (sz - 40.0) / (CHAR_W * max(len(v["name"]) for v in QR)))
                hs = min(44.0, sz / (CHAR_W * max(len(v["handle"]) for v in QR)))
                f.tag("w", x0 + 10, y0 - 34, q["name"], size=ns, pad=10 * ns / 62.0, bold=True)
                f.text("w", x0, y0 + sz + 14 + hs, q["handle"], size=hs, bold=True)
                if text_w(q["handle"], hs) + text_w(q["role"], L.T_TAG) + 30.0 <= sz:
                    f.text("w", x0 + sz, y0 + sz + 14 + hs, q["role"], size=L.T_TAG, anchor="rs")
        return self.furniture(f, t, ctx)


# ----------------------------------------------------------------------------
# 13.1  FLUX
# ----------------------------------------------------------------------------

class Flux(Standby):
    """The wall as a counter: the lattice of crosses is its metre grid, every square metre says how many muons
    went through it in its last second (1 /cm2/min: 167 on average, never twice the same)."""
    name = look = "flux"
    title = "FLUX"
    SIZE = 34
    WAVE = (420.0, 120.0, 640.0)    # a muon caught shakes the count around its tower: speed of the wave (px / s),
    #                                 its depth, and how far it goes

    def __init__(self, ctx):
        super().__init__(ctx)
        js = np.arange(1, int((L.FX1 - X_WALL) / M) + 1)
        ks = np.arange(2, int((GROUND - VIEW_Y0 - 60.0) / M) + 1)
        self.gx = X_WALL + js * M                               # the metre lines of the wall, left to right
        self.gy = GROUND - ks * M                               # ... and from 2 m up
        self.mx, self.my = js, ks
        CX, CY = np.meshgrid(0.5 * (self.gx[:-1] + self.gx[1:]), 0.5 * (self.gy[:-1] + self.gy[1:]))
        cx, cy = CX.ravel(), CY.ravel()
        hw = 0.5 * text_w("000", self.SIZE) + 8.0
        ok = np.ones(len(cx), bool)
        for tw in ctx.towers.values():                          # no figure behind a tower, or on its name
            ok &= ~((cx + hw > tw.x0) & (cx - hw < tw.x1) & (cy + 18.0 > tw.top - 30.0))
        self.cx, self.cy = cx[ok], cy[ok]
        self.id = np.flatnonzero(ok)
        self.ph = rand(self.id, 0, 11)                          # when in the second each cell takes its figure
        self.det_cell = {}                                      # the square metre each detector stands in
        for key, tw in ctx.towers.items():
            ox, oy = tw.det
            jj, kk = int((ox - X_WALL) // M), int((GROUND - oy + 30.0) // M)
            self.det_cell[key] = (X_WALL + jj * M, GROUND - (kk + 1) * M, X_WALL + (jj + 1) * M, GROUND - kk * M)

    def figures(self, tau):
        """Count of each cell in its last second. A count of muons at 167 a second wanders by about 13 around it
        (the sum of four draws: close enough to the bell of a real count)."""
        k = np.floor(tau - self.ph)
        z = (sum(rand(self.id, k, 20 + c) for c in range(4)) - 2.0) * math.sqrt(3.0)
        return np.round(RATE_M2 + math.sqrt(RATE_M2) * z).astype(int)

    def draw(self, f, t, ctx):
        tau = t - self.t0 - 1.0
        f.set_clip(L.FX0, VIEW_Y0, L.FX1, VIEW_Y1)
        # the lattice: the metre grid of the wall, drawn row by row
        GX, GY = np.meshgrid(self.gx, self.gy)
        with f.build(self.age(t, 0.0), (L.FX0, VIEW_Y0, L.FX1, VIEW_Y1), flow="tb", wave=1.0, marks=False, key=140):
            f.crosses("w", GX.ravel(), GY.ravel(), 7.0, 0.95)
            for k, y in zip(self.my, self.gy):
                f.text("w", L.FX0 + 52, float(y) + 8, f"{k}", size=L.T_TAG, anchor="rs")
            for j, x in zip(self.mx, self.gx):
                if j % 5 == 0:
                    f.text("w", float(x), float(self.gy[-1]) - 26, f"{j} M", size=L.T_TAG, anchor="ms")
        # the figures
        n = self.figures(max(tau, 0.0))
        live = tau > 1.0 and self.t_out - t > OUT
        # a muon caught by a tower shakes the count around it: a wave leaves the tower, the figures it passes
        # are red and are taken again (their digits spin)
        red = np.zeros(len(self.cx), bool)
        if live:
            speed, depth, reach = self.WAVE
            for key in sd.KEYS:
                ox, oy = ctx.towers[key].det
                d = np.hypot(self.cx - ox, self.cy - oy)
                for th in self.hits(ctx, key, t - (reach + depth) / speed, t + 1e-6, every=0.5)[0]:
                    r = speed * (t - th)
                    red |= (d <= r) & (d > r - depth) & (d < reach)
        spin = (B.rnd(len(self.cx), 151, B.frame_no(t)) * 1000.0).astype(int)
        with f.build(self.age(t, 0.9, span=1.6), (L.FX0, VIEW_Y0, L.FX1, VIEW_Y1), flow="diag", wave=1.2, marks=False,
                     cps=40.0, key=141):
            for i in range(len(self.cx)):
                f.text("r" if red[i] else "w", float(self.cx[i]), float(self.cy[i]) + 12,
                       f"{spin[i] if red[i] else n[i]:03d}", size=self.SIZE, anchor="ms", bold=True)
        # ... and its own square metre is framed
        for key in sd.KEYS:
            a, e = ctx.det.last(key, t, echoes=True)
            if a < 1.6 and live:
                x0, y0, x1, y1 = self.det_cell[key]
                with f.build(B.io(a, 1.6 - a, out=0.3, span=0.5), (x0, y0, x1, y1), wave=0.1, line=0.2, marks=False, key=150):
                    f.rect("r", x0, y0, x1, y1, 1.2, width=L.LW_BOLD)
        f.set_clip()
        opt = self.furniture(f, t, ctx)
        wall_m2 = (L.W / M) * (GROUND / M)
        self._figure(f, t, ctx, f"THROUGH THE WHOLE WALL // SINCE {sd.mmss(self.t0)}",
                     sd.spaced(RATE_M2 * wall_m2 * max(tau, 0.0)))
        return opt


# ----------------------------------------------------------------------------
# 13.2  COINCIDENCE
# ----------------------------------------------------------------------------

class Coincidence(Standby):
    """Every muon caught sends a circle over the wall."""
    name = look = "coincidence"
    title = "COINCIDENCE"
    V = 150.0                       # px / s: the speed of a circle
    CLOCK = 6.0                     # s between two dashed circles (the time base)
    EVERY = 0.25                    # s: one circle per tower in that time when it fires all the time (Sparse)
    DASHES = 132                    # dashes of a dashed circle, whatever its size
    PTS = 240                       # points of a circle, whatever its size (a circle that grows keeps its points)
    CROSS = 14                      # the crossings of the last circles of a tower are marked: this many

    def __init__(self, ctx):
        super().__init__(ctx)
        self.c = {k: ctx.towers[k].det for k in sd.KEYS}
        corners = [(L.FX0, VIEW_Y0), (L.FX1, VIEW_Y0), (L.FX0, VIEW_Y1), (L.FX1, VIEW_Y1)]
        self.reach = {k: max(math.hypot(x - c[0], y - c[1]) for x, y in corners) for k, c in self.c.items()}
        self.a = np.linspace(0.0, 2 * np.pi, self.PTS + 1) - 0.5 * np.pi           # (a circle is traced from its top)

    def rings(self, ctx, key, t):
        """(radii, energies) of the circles of one tower that are still on the wall, the last one first."""
        ta = max(self.t0 + 0.6, t - self.reach[key] / self.V)
        tt, ee = self.hits(ctx, key, ta, t + 1e-6, every=self.EVERY)
        return self.V * (t - tt[::-1]), ee[::-1]

    def draw(self, f, t, ctx):
        left = 1.0 - self.gone(t)                       # the part of every circle that is still there at the end
        f.set_clip(L.FX0, VIEW_Y0, L.FX1, VIEW_Y1)
        R = {}
        n_ring = 0
        k_arc = max(1, int(round(self.PTS * left)))
        ca, sa = np.cos(self.a[: k_arc + 1]), np.sin(self.a[: k_arc + 1])
        for key in sd.KEYS:
            cx, cy = self.c[key]
            r, e = self.rings(ctx, key, t)
            R[key] = r[: self.CROSS]
            n_ring += len(r)
            if len(r) and left > 0.0:
                X, Y = cx + r[:, None] * ca[None, :], cy + r[:, None] * sa[None, :]
                wd = np.repeat(1.8 + 2.6 * e, k_arc)
                f.segments("w", X[:, :-1].ravel(), Y[:, :-1].ravel(), X[:, 1:].ravel(), Y[:, 1:].ravel(), 0.95, width=wd)
            # the time base: a dashed circle from the three towers together
            m0 = math.ceil((max(self.t0 + 0.6, t - self.reach[key] / self.V)) / self.CLOCK)
            nd = int(round(self.DASHES * left))
            for m in range(m0, int(t // self.CLOCK) + 1):
                rr = self.V * (t - m * self.CLOCK)
                if rr < 4.0 or nd < 1:
                    continue
                a = (np.arange(nd) + 0.02 * t * (1 if m % 2 else -1)) / self.DASHES * 2 * np.pi - 0.5 * np.pi
                da = 0.42 * 2 * np.pi / self.DASHES
                f.segments("w", cx + rr * np.cos(a), cy + rr * np.sin(a), cx + rr * np.cos(a + da), cy + rr * np.sin(a + da),
                           0.95, width=2.0)
        # where the circles of two towers cross
        n_x = 0
        for ka, kb in (("L", "C"), ("C", "R"), ("L", "R")):
            ra, rb = R[ka], R[kb]
            if not len(ra) or not len(rb):
                continue
            (xa, ya), (xb, yb) = self.c[ka], self.c[kb]
            d = math.hypot(xb - xa, yb - ya)
            ux, uy = (xb - xa) / d, (yb - ya) / d
            A, Bq = np.meshgrid(ra, rb, indexing="ij")
            ok = (A + Bq > d) & (np.abs(A - Bq) < d)
            if not ok.any():
                continue
            A, Bq = A[ok], Bq[ok]
            al = (A * A - Bq * Bq + d * d) / (2 * d)
            h = np.sqrt(np.maximum(A * A - al * al, 0.0))
            born = np.clip((A + Bq - d) / 60.0, 0.0, 1.0)       # two circles that just met: the point grows
            for sg in (1.0, -1.0):
                px, py = xa + al * ux - sg * h * uy, ya + al * uy + sg * h * ux
                m = (px > L.FX0 + 8) & (px < L.FX1 - 8) & (py > VIEW_Y0 + 8) & (py < VIEW_Y1 - 8)
                f.dots("r", px[m], py[m], 7.5 * born[m] ** 0.5 * left, 1.5)
                n_x += int(m.sum())
        f.set_clip()
        opt = self.furniture(f, t, ctx)
        self._figure(f, t, ctx, "CIRCLES ON THE WALL // CROSSINGS", f"{n_ring:02d} / {n_x:03d}")
        return opt


# ----------------------------------------------------------------------------
# 13.3  RECORD
# ----------------------------------------------------------------------------

class Record(Standby):
    """The last three quarters of a minute, line by line: now at the bottom."""
    name = look = "record"
    title = "RECORD"
    N = 45                          # lines kept
    S = 20.0                        # px between two of them
    U = 20.0                        # px / s: the kept lines move up (a line is kept every S / U seconds)
    Y_NOW = VIEW_Y1 - 22.0          # the line of now: it stays there, its copies leave it
    C = 60.0                        # px / s: a swell runs along the wall
    TAU = 5.0                       # s: ... and dies
    AMP = 84.0                      # px: height of the swell of a full-energy muon
    TOP = 260.0                     # px: the height a line never passes, however many muons (a soft limit)
    SIG = (22.0, 2.6)               # px: width of a swell when it starts, and what it gains a second
    EVERY = 0.2                     # s: one swell per tower in that time when it fires all the time (Sparse)
    DX = 4.0
    DRAW = (0.03, 1.0)              # s: the lines are drawn one after the other (lag per line), each in this long

    def __init__(self, ctx):
        super().__init__(ctx)
        self.x = np.arange(L.FX0, L.FX1 + 0.1, self.DX)
        self.dt = self.S / self.U
        self.xc = {k: ctx.towers[k].cx for k in sd.KEYS}
        self.ic = {k: int(np.argmin(np.abs(self.x - v))) for k, v in self.xc.items()}
        self._kept = {}
        half = int(math.ceil(4.0 * (self.SIG[0] + self.SIG[1] * 4.0 * self.TAU) / self.DX))
        self.win = np.arange(-half, half + 1)                   # the samples a swell is worked out on, around its top
        k = np.arange(10)
        self.nz = (2 * np.pi / (260.0 * (1400.0 / 260.0) ** rand(k, 0, 71)) * np.where(rand(k, 0, 72) < 0.5, -1, 1),
                   2 * np.pi * (0.012 + 0.03 * rand(k, 0, 73)), 2 * np.pi * rand(k, 0, 74))

    def ground(self, T):
        """The slow relief of a line (px, up = positive): a few plane waves in (x, time)."""
        kx, om, ph = self.nz
        return 5.0 * np.sin(kx[:, None] * self.x[None, :] + (om * T + ph)[:, None]).sum(0) / math.sqrt(len(kx) / 2.0)

    def line(self, ctx, T):
        """Height of the line that shows the wall at time T (px, up = positive): the swells of the muons caught
        before T, each one running out both ways from its tower and dying."""
        n = len(self.x)
        h = np.zeros(n)
        for key in sd.KEYS:
            tt, ee = self.hits(ctx, key, T - 4.0 * self.TAU, T + 1e-9, every=self.EVERY)
            if not len(tt):
                continue
            a = T - tt
            sig = self.SIG[0] + self.SIG[1] * a
            d = self.C * a
            amp = (self.AMP * (0.25 + 0.75 * ee) * (1.0 - np.exp(-a / 0.1)) * np.exp(-a / self.TAU)
                   / (1.0 + np.exp(-0.5 * (d / sig) ** 2)))             # (one swell while the two are still one)
            for way in (1.0, -1.0):
                # each swell is only worked out where it is: the samples within four of its widest sigmas
                c = self.xc[key] + way * d
                idx = np.round((c - self.x[0]) / self.DX).astype(int)[:, None] + self.win[None, :]
                ok = (idx >= 0) & (idx < n)
                v = amp[:, None] * np.exp(-0.5 * ((self.x[0] + idx * self.DX - c[:, None]) / sig[:, None]) ** 2)
                h += np.bincount(idx[ok], weights=v[ok], minlength=n)
        return self.TOP * np.tanh(h / self.TOP) + self.ground(T)

    def kept(self, ctx, k):
        """The line kept at time k * dt (cached: it does not change any more - unless the hits did, after a jump
        of the clock with live detectors)."""
        T = k * self.dt
        sig = tuple((len(tt), float(tt[-1]) if len(tt) else 0.0)
                    for tt in (ctx.det.hits(key, T - 4.0 * self.TAU - 2.0, T + 1e-9)[0] for key in sd.KEYS))
        got = self._kept.get(k)
        if got is None or got[0] != sig:
            got = self._kept[k] = (sig, self.line(ctx, T))
            if len(self._kept) > 6 * self.N:
                for old in sorted(self._kept)[: -3 * self.N]:
                    del self._kept[old]
        return got[1]

    def draw(self, f, t, ctx):
        x = self.x
        k0 = int(math.floor(t / self.dt + 1e-9))                    # the last line kept
        ks = k0 - np.arange(self.N)
        # front to back: the line of now, then its copies, the last kept first
        H = np.vstack([self.line(ctx, t)] + [self.kept(ctx, int(k)) for k in ks])
        Yb = np.r_[self.Y_NOW, self.Y_NOW - (t - ks * self.dt) * self.U]
        Y = Yb[:, None] - H
        hor = np.minimum.accumulate(Y, axis=0)                      # the highest point drawn so far, front to back
        vis = np.ones(Y.shape, bool)
        vis[1:] = Y[1:] < hor[:-1] - 0.75
        vis &= Y > VIEW_Y0 + 6.0
        # the lines are drawn from the left, the nearest first; at the end they leave to the right, the same way
        rank = np.arange(len(Yb))
        p_in = B.ease(np.clip((t - self.t0 - 0.2 - self.DRAW[0] * rank) / self.DRAW[1], 0.0, 1.0))
        p_out = B.ease(np.clip((t - (self.t_out - 0.15 - self.DRAW[0] * len(Yb) - 0.8) - self.DRAW[0] * rank) / 0.8, 0.0, 1.0))
        xh = L.FX0 + (L.FX1 - L.FX0) * p_in
        xt = L.FX0 + (L.FX1 - L.FX0) * p_out
        there = (x[None, :] <= xh[:, None]) & (x[None, :] >= xt[:, None]) & (p_out[:, None] < 1.0)
        f.set_clip(L.FX0, VIEW_Y0, L.FX1, VIEW_Y1)
        m = vis[:, :-1] & vis[:, 1:] & there[:, :-1] & there[:, 1:]
        f.segments("w", np.broadcast_to(x[:-1], m.shape)[m], Y[:, :-1][m], np.broadcast_to(x[1:], m.shape)[m], Y[:, 1:][m],
                   0.95, width=1.9)
        pen = (p_in > 0.0) & (p_in < 1.0)
        if pen.any():
            ih = np.clip(((xh[pen] - L.FX0) / self.DX).astype(int), 0, len(x) - 1)
            f.dots("w", x[ih], Y[pen, ih], 3.0, 1.7)
        # a red point where a muon was caught: on the line of now, then on the copy that was kept next
        for key in sd.KEYS:
            tt, ee = self.hits(ctx, key, (ks[-1] - 1) * self.dt, t + 1e-9, every=self.EVERY)
            if not len(tt):
                continue
            kh = np.ceil(tt / self.dt - 1e-9).astype(int)
            row = np.where(kh > k0, 0, 1 + k0 - kh)
            ok = row < len(Yb)
            row, tt, ee = row[ok], tt[ok], ee[ok]
            i = self.ic[key]
            ok = vis[row, i] & there[row, i]
            f.dots("r", np.full(int(ok.sum()), x[i]), Y[row[ok], i] - 1.0, (3.8 + 3.4 * ee[ok]) * B.spring((t - tt[ok]) / 0.22), 1.5)
        # the age of a line, every ten seconds
        with f.build(self.age(t, 1.2), (L.FX0, VIEW_Y0, L.FX0 + 200, VIEW_Y1), marks=False, key=160):
            for k, yb in zip(ks, Yb[1:]):
                if int(k) % int(round(10.0 / self.dt)) == 0 and VIEW_Y0 + 40 < yb < self.Y_NOW - 30:
                    f.occlude(L.FX0 + 8, yb - 27, L.FX0 + 112, yb - 3)
                    f.text("w", L.FX0 + 14, float(yb) - 8, f"-{mmss(t - k * self.dt)}", size=L.T_TAG)
        f.set_clip()
        opt = self.furniture(f, t, ctx)
        self._figure(f, t, ctx, "RATE // LAST 20 S", f"{self.rate(ctx, t):.2f}", "MUONS / S")
        return opt
