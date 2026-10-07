"""DETECTOR - the data explainer.   Sheet scene 2, 01:44 - 02:13.

The TouchDesigner idea is kept: the detector as a wireframe object turning in space, a new view on every
line of the voice. The object is the real thing, drawn from the photo of the board (ref/muon detector):
the PCB with its mounting holes, the white square with the SiPM on its carrier and the bias supply inside,
the scintillator block resting on the sensor (the photo shows the board without it), the amplifier section,
the Arduino Nano on its sockets, the radio module, the BNC, the POWER LED, the traces between them; numbered call-outs ride with the parts, and what the numbers are is drawn
under the card as a block diagram of the signal path (boxes and lines, as on the Apollo drawings; a plain
list when the column is narrow). Added to the idea: the three towers come out of the dark with their
detectors bracketed and WAITING, simulated muons cross the scintillator, their light is collected by the
SiPM, the pulse runs along the board (amplifier, converter, radio) and the reply is spelled out as a chain
of four read-outs:
ENERGY (the pulse) -> CODE (ADC word + the OSC float) -> LIGHT (the bloom) -> SOUND (the wave).

  01:44  These detectors. Live. Scientific apparatus...   VIEW 01 perspective, towers revealed
  01:49  Waiting                                          VIEW 02 edge-on, everything flat
  01:51  Every muon that finds them, becomes a conductor  VIEW 03 a muon crosses the block
  01:54  The more powerful the muon, brighter, louder...  VIEW 04 top: three muons, three energies
  02:00  Energy becomes code becomes light becomes sound  VIEW 05 the chain lights up word by word
  02:05  A visitor. A messenger. A muon...                 the board flies back onto the centre tower (one
                                                           eased move), one muon comes down to it -> 02:13

NOTHING THAT SHOWS DATA FADES IN OR POPS IN (muonbloom/build.py). At 01:44 the board is constructed part
after part (the bare board, then what is soldered on it, the SiPM, the scintillator last), then the
furniture of the scene, block after block (strip, view tag, call-outs and legend, card, reply chain, bottom
panels; the three scopes are built by towers.scopes). On every cut of the voice only what is new is made:
the name of the view and the call-outs. The read-outs of the chain are instruments: they run all the time
(noise floor, live float, level meters, the bloom breathing) and each one re-plots what it caught when a
muon reaches it - the pen writes the new trace over the last one; a mark travels down the BECOMES arrows.
The strip, the view tag and the call-outs are taken apart when the board flies back to its tower.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import engine as E
from .. import hud, towers
from .. import layout as L
from .. import showdata as sd
from ..engine import Camera, OrthoCamera, smoothstep
from ..show import Scene

# the wall rule (engine.WALL): a line of the furniture is never a hairline, and never grey
WH = E.ww(L.LW_HAIR)

T0, T1 = 104.0, 133.0
STAGES = ("ENERGY", "CODE", "LIGHT", "SOUND")
ADC_BITS = 10                       # the board reads its pulse with an Arduino Nano: a 10-bit successive-approximation ADC
ADC_MAX = (1 << ADC_BITS) - 1
T_BIT = 0.045                       # seconds a bit takes to settle on the CODE read-out

# The board, in board units: 1 unit = 5 mm, x to the right, y towards whoever looks at the photo, z down the
# photo. It is 20 x 20 units and its top face is at y = TB. The model handed to the cameras is
# (board - MODEL_C) * MODEL_K, so that it turns around the middle of what it shows.
TB = 0.32
MODEL_C, MODEL_K = np.array([0.0, 0.0, 0.9], np.float32), 0.95
SCINT = (-5.0, 5.0, -4.5, 5.5)      # the white square of the photo, 5 x 5 cm: where the scintillator goes (x0, x1, z0, z1)
SCINT_Y = TB + 0.9                  # the block is not on the photo: it rests on the SiPM carrier, above the parts of the square
SCINT_H = 2.0                       # ... and is 1 cm thick
SIPM = (0.3, 1.4)                   # the sensor, on its carrier in the middle of the square
NANO_X = -6.9                       # centre line of the Arduino Nano
BIAS_Z = -2.9                       # the bias supply: inside the square, at its top
P_PCB, P_RADIO, P_NANO, P_BIAS, P_AMP, P_BNC, P_POWER, P_SIPM, P_SCINT = range(9)
PART_T = (0.05, 0.5, 0.65, 0.8, 0.95, 1.1, 1.25, 1.4, 1.7)     # when each part starts to be made, s after the cut
T_CALL = 2.2                        # ... and the call-outs, one after the other
T_PATH = 3.1                        # ... and, last, the way of the signal is traced
# call-outs: name, part, point on the part, where its number floats (board units)
CALL = [
    ("SCINTILLATOR", P_SCINT, (3.6, SCINT_Y + SCINT_H, 4.6), (6.4, 7.0, 6.9)),
    ("SIPM", P_SIPM, (SIPM[0], TB + 0.8, SIPM[1]), (-0.9, 8.4, -0.9)),
    ("BIAS SUPPLY", P_BIAS, (-0.3, TB + 0.25, BIAS_Z), (-0.6, 6.6, -7.8)),
    ("AMPLIFIER", P_AMP, (6.9, TB + 0.35, 3.0), (10.9, 3.6, 2.0)),
    ("ADC // MCU", P_NANO, (NANO_X, TB + 1.9, 8.3), (-10.3, 5.0, 9.8)),
    ("RADIO 2.4 GHZ", P_RADIO, (-8.2, TB + 1.9, -5.5), (-10.7, 4.6, -8.2)),
    ("BNC", P_BNC, (6.5, TB + 2.9, 8.6), (10.2, 5.8, 11.0)),
    ("POWER LED", P_POWER, (0.2, TB + 0.7, 8.9), (-0.2, 3.4, 12.6)),
]
# the block diagram: the parts the signal goes through, from the top down, and what it is on the way to the next
FLOW = ((0, "PHOTONS"), (1, "CURRENT"), (3, "VOLTAGE"), (4, "NUMBER"), (5, ""))
R_CALL = 13.0                       # radius of a call-out number (px)
# what is printed on the board beside its chips (as on the photo): text, where (board units)
SILK = (("LT3461", (0.3, TB + 0.25, BIAS_Z - 0.35)), ("OPA2743", (7.4, TB + 0.35, 2.6)),
        ("NANO", (NANO_X + 1.8, TB + 1.9, 9.6)))


# ----------------------------------------------------------------------------
# the board as line work
# ----------------------------------------------------------------------------

def _arc(cx, cz, r, a0, a1, n):
    a = np.linspace(a0, a1, n + 1)
    return np.stack([cx + r * np.cos(a), cz + r * np.sin(a)], 1)


def _round_rect(x0, z0, x1, z1, r, n=4):
    """Outline of a rectangle with round corners, in the plane of the board: (4 * (n + 1), 2)."""
    h = 0.5 * math.pi
    return np.concatenate([_arc(x1 - r, z0 + r, r, -h, 0.0, n), _arc(x1 - r, z1 - r, r, 0.0, h, n),
                           _arc(x0 + r, z1 - r, r, h, 2 * h, n), _arc(x0 + r, z0 + r, r, 2 * h, 3 * h, n)])


class _Mesh:
    """Collects the segments of the model: two end points, the part they belong to, a level of detail
    (0 = outline, kept when the board is tiny; 1 = detail; 2 = fine detail) and how they are drawn."""

    def __init__(self):
        self.a, self.b, self.p, self.lod, self.i, self.w = [], [], [], [], [], []
        self.part = 0

    def seg(self, a, b, lod=1, i=0.7, w=L.LW):
        a = np.asarray(a, np.float32).reshape(-1, 3)
        b = np.asarray(b, np.float32).reshape(-1, 3)
        n = len(a)
        self.a.append(a)
        self.b.append(b)
        self.p.append(np.full(n, self.part, np.int16))
        self.lod.append(np.full(n, lod, np.int8))
        self.i.append(np.full(n, i, np.float32))
        self.w.append(np.full(n, w, np.float32))

    def poly(self, pts, closed=False, **kw):
        p = np.asarray(pts, np.float32).reshape(-1, 3)
        if closed:
            p = np.concatenate([p, p[:1]])
        self.seg(p[:-1], p[1:], **kw)

    def flat(self, xz, y, closed=True, **kw):
        """A polygon (or an open line) lying in the plane y = const."""
        xz = np.asarray(xz, np.float32).reshape(-1, 2)
        self.poly(np.stack([xz[:, 0], np.full(len(xz), y, np.float32), xz[:, 1]], 1), closed=closed, **kw)

    def rect(self, x0, z0, x1, z1, y, **kw):
        self.flat([(x0, z0), (x1, z0), (x1, z1), (x0, z1)], y, **kw)

    def box(self, x0, z0, x1, z1, y0, y1, top=None, **kw):
        self.rect(x0, z0, x1, z1, y1, **(dict(kw, **top) if top else kw))
        self.rect(x0, z0, x1, z1, y0, **kw)
        c = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
        self.seg([(x, y0, z) for x, z in c], [(x, y1, z) for x, z in c], **kw)

    def ring(self, cx, cz, r, y, n=12, **kw):
        self.flat(_arc(cx, cz, r, 0.0, 2 * math.pi, n)[:-1], y, **kw)


def _spread(ctx, col, y_bot, pad=28.0):
    """A column of ctx.cols, widened over the towers that stand lower than what is put in it: above the head
    of a short tower the wall is free up to the next bay (the towers of Site 3.1 are short)."""
    x0, x1 = col
    for tw in ctx.towers.values():
        if tw.top < y_bot + 30.0:
            continue
        if abs(tw.x0 - pad - x1) < 2.0:         # the tower closes the column on its right
            x1 = tw.x1
        elif abs(tw.x1 + pad - x0) < 2.0:       # ... or on its left
            x0 = tw.x0
    return x0, x1


def _model(p):
    return ((np.asarray(p, np.float32).reshape(-1, 3) - MODEL_C) * MODEL_K).astype(np.float32)


def _build_board():
    """The detector board after the photo of the real one: every group of components as an outlined
    footprint, the traces between them. Returns the segments in model space, sorted by part."""
    m = _Mesh()
    hair, thin, bold = L.LW_HAIR, L.LW, L.LW_BOLD
    main = dict(lod=0, i=0.88, w=thin)
    part = dict(lod=1, i=0.7, w=thin)
    side = dict(lod=1, i=0.5, w=hair)
    fine = dict(lod=2, i=0.5, w=hair)
    silk = dict(lod=2, i=0.4, w=hair)
    trace = dict(lod=2, i=0.32, w=hair)
    pi = math.pi

    # -- the bare board: outline, thickness, mounting holes, silk screen, the number written by hand, traces
    m.part = P_PCB
    out = _round_rect(-10.0, -10.0, 10.0, 10.0, 0.7)
    m.flat(out, TB, lod=0, i=0.92, w=bold)
    m.flat(out, 0.0, lod=1, i=0.45, w=hair)
    e = out[[0, 4, 5, 9, 10, 14, 15, 19]]
    m.seg(np.c_[e[:, 0], np.zeros(8), e[:, 1]], np.c_[e[:, 0], np.full(8, TB), e[:, 1]], **side)
    for hx, hz in ((-9.3, -9.4), (9.3, -9.4), (9.3, 9.4), (-9.3, 9.4)):
        m.ring(hx, hz, 0.32, TB, n=10, lod=1, i=0.8, w=thin)
        m.ring(hx, hz, 0.62, TB, n=12, **silk)
    m.rect(SCINT[0], SCINT[2], SCINT[1], SCINT[3], TB, lod=1, i=0.62, w=thin)   # the white square
    m.rect(-9.85, -9.05, -6.15, -2.3, TB, **silk)                              # around the radio
    m.rect(NANO_X - 2.05, 3.0, NANO_X + 2.05, 9.95, TB, **silk)                # around the Nano
    m.rect(4.95, 7.0, 8.05, 9.95, TB, **silk)                                  # around the BNC
    m.flat(_round_rect(5.5, 0.5, 8.9, 6.15, 0.45, n=3), TB, **silk)            # around the amplifier
    a = np.linspace(0.0, 2 * pi, 13)[:-1]                                       # "05", written on the board with a pen
    m.flat(np.c_[6.65 + 0.48 * np.cos(a), -5.85 + 0.8 * np.sin(a)], TB, **fine)
    m.flat([(7.95, -6.6), (7.35, -6.62), (7.25, -5.95), (7.62, -6.02), (7.95, -5.78), (7.92, -5.32), (7.55, -5.05),
            (7.18, -5.2)], TB, closed=False, **fine)
    m.flat([(6.0, -4.72), (8.25, -4.85)], TB, closed=False, **fine)
    m.flat([(6.15, -4.5), (8.1, -4.62)], TB, closed=False, **fine)
    for pts in (
            [(SIPM[0] + 0.85, SIPM[1]), (5.0, SIPM[1]), (6.1, SIPM[1] + 1.1), (6.35, 2.8)],         # SiPM -> amplifier
            [(6.9, 3.45), (6.9, 5.0), (6.1, 5.8), (-3.6, 5.8), (-4.5, 6.7), (NANO_X + 1.55, 6.7)],  # amplifier -> Nano
            [(7.5, 4.9), (7.5, 6.4), (6.9, 7.0), (6.9, 7.3)],                                       # amplifier -> BNC
            [(-0.3, BIAS_Z + 0.4), (-0.3, -1.4), (SIPM[0], -0.8), (SIPM[0], SIPM[1] - 1.7)],        # bias -> SiPM
            [(NANO_X + 1.55, 9.2), (-1.8, 9.2), (-1.5, 8.9), (-1.15, 8.9)], [(-0.45, 8.9), (-0.3, 8.9)],    # power
            [(0.2, 7.0), (0.2, 8.4)],
            [(-6.0, -9.4), (8.5, -9.4), (8.95, -8.95), (8.95, 0.3), (8.6, 0.65)],                   # supply rail
            [(0.6, -9.4), (0.6, BIAS_Z - 1.0)],
    ):
        m.flat(pts, TB, closed=False, **trace)
    cl = np.arange(-11.6, 11.6, 1.5)                                           # centre lines, as on a drawing: dash, dot
    cx, cz = 0.5 * (SCINT[0] + SCINT[1]), 0.5 * (SCINT[2] + SCINT[3])
    o = np.full(len(cl), TB, np.float32)
    for d0, d1 in ((0.0, 0.85), (1.1, 1.2)):
        m.seg(np.c_[cl + d0, o, np.full(len(cl), cz)], np.c_[cl + d1, o, np.full(len(cl), cz)], lod=2, i=0.3, w=hair)
        m.seg(np.c_[np.full(len(cl), cx), o, cl + d0], np.c_[np.full(len(cl), cx), o, cl + d1], lod=2, i=0.3, w=hair)
    for j in range(5):                                                         # the bus from the Nano to the radio
        zp, xv = 3.75 + 0.508 * j, NANO_X - 2.0 - 0.14 * j
        m.flat([(NANO_X - 1.55, zp), (xv, zp), (xv, -2.6)], TB, closed=False, **trace)

    # -- radio module (nRF24): on a 2 x 4 header, crystal can, chip, printed aerial
    m.part = P_RADIO
    ym = TB + 1.7
    m.box(-9.6, -3.5, -7.6, -2.6, TB, ym, **side)
    m.box(-9.7, -8.9, -6.3, -2.4, ym, ym + 0.2, top=main, **side)
    yt = ym + 0.2
    m.rect(-8.95, -5.4, -8.15, -4.6, yt, **fine)
    can = np.concatenate([_arc(-7.15, -6.2, 0.4, pi, 2 * pi, 4), _arc(-7.15, -5.0, 0.4, 0.0, pi, 4)])
    m.flat(can, yt, **fine)
    m.flat(can, yt + 0.6, lod=1, i=0.65, w=thin)
    m.seg([(-7.55, yt, -5.6), (-6.75, yt, -5.6)], [(-7.55, yt + 0.6, -5.6), (-6.75, yt + 0.6, -5.6)], **fine)
    m.flat([(-9.4, -7.4), (-9.4, -8.6), (-8.85, -8.6), (-8.85, -7.7), (-8.3, -7.7), (-8.3, -8.6), (-7.75, -8.6),
            (-7.75, -7.7), (-7.2, -7.7), (-7.2, -8.6), (-6.65, -8.6)], yt, closed=False, lod=2, i=0.62, w=hair)
    for i in range(4):
        for r in range(2):
            x, z = -9.36 + 0.508 * i, -3.3 + 0.45 * r
            m.rect(x - 0.1, z - 0.1, x + 0.1, z + 0.1, yt, **fine)

    # -- Arduino Nano: on two sockets of 15 pins, MCU, USB socket, reset button, ICSP header
    m.part = P_NANO
    yn = TB + 1.7
    nx = NANO_X
    m.box(nx - 1.8, 3.5, nx - 1.3, 11.1, TB, yn, **side)
    m.box(nx + 1.3, 3.5, nx + 1.8, 11.1, TB, yn, **side)
    m.box(nx - 1.8, 3.2, nx + 1.8, 12.2, yn, yn + 0.2, top=main, **side)
    yt = yn + 0.2
    zs = 3.75 + 0.508 * np.arange(15)
    for x in (nx - 1.55, nx + 1.55):
        m.seg(np.c_[np.full(15, x - 0.13), np.full(15, yt), zs], np.c_[np.full(15, x + 0.13), np.full(15, yt), zs],
              lod=2, i=0.6, w=hair)
    m.flat([(nx - 1.0, 8.3), (nx, 7.3), (nx + 1.0, 8.3), (nx, 9.3)], yt, lod=1, i=0.72, w=thin)
    m.flat([(nx - 0.5, 8.3), (nx, 7.8), (nx + 0.5, 8.3), (nx, 8.8)], yt, **fine)
    m.box(nx - 0.8, 11.0, nx + 0.8, 12.7, yt, yt + 0.8, top=dict(lod=1, i=0.75, w=thin), **side)
    m.rect(nx - 0.3, 6.0, nx + 0.3, 6.6, yt, **fine)
    m.ring(nx, 6.3, 0.17, yt, n=6, **fine)
    m.rect(nx - 0.5, 3.5, nx + 0.5, 4.6, yt, **fine)
    m.flat([(nx, 3.5), (nx, 4.6)], yt, closed=False, **fine)
    m.rect(nx + 0.55, 5.0, nx + 1.1, 5.35, yt, **fine)
    m.rect(nx - 1.1, 9.9, nx - 0.65, 10.5, yt, **fine)

    # -- bias supply for the SiPM (boost converter, its inductor, a few passives)
    m.part = P_BIAS
    bz = BIAS_Z
    m.box(-0.75, bz - 0.35, 0.15, bz + 0.35, TB, TB + 0.25, **part)
    m.box(-2.0, bz - 0.2, -1.0, bz + 0.8, TB, TB + 0.6, **part)
    m.ring(-1.5, bz + 0.3, 0.34, TB + 0.6, n=8, **fine)
    for x0, z0, x1, z1 in ((-2.85, 0.1, -2.3, 0.5), (0.35, -1.0, 0.9, -0.6), (0.65, 0.8, 1.2, 1.2), (0.35, -0.05, 0.85, 0.3)):
        m.rect(x0, bz + z0, x1, bz + z1, TB, **fine)

    # -- amplifier: an op-amp in SO-8, its resistors, capacitors and diodes, four test points
    m.part = P_AMP
    m.box(6.4, 2.6, 7.4, 3.4, TB, TB + 0.35, **part)
    lz = np.array([2.7, 2.9, 3.1, 3.3], np.float32)
    for xa, xb in ((6.15, 6.4), (7.4, 7.65)):
        m.seg(np.c_[np.full(4, xa), np.full(4, TB), lz], np.c_[np.full(4, xb), np.full(4, TB), lz], **fine)
    for cx, cz in ((6.3, 1.2), (7.6, 1.2), (6.2, 1.95), (7.7, 1.95), (8.25, 2.95), (6.2, 4.0), (7.6, 4.0), (6.3, 4.65),
                   (7.55, 4.65), (6.3, 5.3), (7.55, 5.3)):
        m.rect(cx - 0.35, cz - 0.17, cx + 0.35, cz + 0.17, TB, **fine)
    for cx, cz in ((6.95, 1.6), (8.35, 1.0), (5.95, 5.75), (6.95, 5.8)):
        m.ring(cx, cz, 0.2, TB, n=8, **fine)

    # -- BNC: the body on the board, the barrel over its edge
    m.part = P_BNC
    m.box(5.2, 7.3, 7.8, 10.0, TB, TB + 2.9, top=main, **part)
    cy, r = TB + 1.55, 0.95
    a = np.linspace(0.0, 2 * pi, 15)[:-1]
    for z, kw in ((10.0, side), (11.3, part), (12.6, main)):
        m.poly(np.c_[6.5 + r * np.cos(a), cy + r * np.sin(a), np.full(14, z)], closed=True, **kw)
    g = np.array([0.0, 0.5 * pi, pi, 1.5 * pi])
    m.seg(np.c_[6.5 + r * np.cos(g), cy + r * np.sin(g), np.full(4, 10.0)],
          np.c_[6.5 + r * np.cos(g), cy + r * np.sin(g), np.full(4, 12.6)], **part)
    m.poly(np.c_[6.5 + 0.42 * np.cos(a), cy + 0.42 * np.sin(a), np.full(14, 12.6)], closed=True, **fine)
    m.seg([(6.5 - r, cy, 11.85), (6.5 + r, cy, 11.85)], [(6.5 - r - 0.3, cy, 11.85), (6.5 + r + 0.3, cy, 11.85)], **fine)

    # -- power: a transistor (TO-92), the LED, its resistor and capacitor, a solder jumper
    m.part = P_POWER
    d = _arc(0.2, 6.45, 0.45, 0.0, pi, 6)
    m.flat(d, TB + 0.9, lod=1, i=0.7, w=thin)
    m.flat(d, TB, **fine)
    m.seg([(x, TB, z) for x, z in d[[0, 3, 6]]], [(x, TB + 0.9, z) for x, z in d[[0, 3, 6]]], **fine)
    m.ring(0.2, 8.9, 0.5, TB, n=10, **fine)
    m.ring(0.2, 8.9, 0.5, TB + 0.7, n=10, lod=1, i=0.78, w=thin)
    da = np.linspace(0.0, pi, 7)
    m.poly(np.c_[0.2 + 0.5 * np.cos(da), TB + 0.7 + 0.5 * np.sin(da), np.full(7, 8.9)], **fine)
    m.poly(np.c_[np.full(7, 0.2), TB + 0.7 + 0.5 * np.sin(da), 8.9 + 0.5 * np.cos(da)], **fine)
    m.rect(-1.15, 8.72, -0.45, 9.08, TB, **fine)
    m.rect(-0.55, 7.8, 0.0, 8.15, TB, **fine)
    m.ring(-3.5, 8.8, 0.16, TB, n=6, **fine)
    m.ring(-3.05, 8.8, 0.16, TB, n=6, **fine)

    # -- SiPM: the sensor on its carrier (a small board standing on two sockets) in the middle of the white
    #    square; the two wire loops beside it
    m.part = P_SIPM
    sx, sz = SIPM
    cx0, cx1, cz0, cz1 = sx - 0.85, sx + 0.85, sz - 1.7, sz + 1.7
    yc = TB + 0.7
    for za, zb in ((cz0, cz0 + 0.65), (cz1 - 0.65, cz1)):
        m.box(cx0, za, cx1, zb, TB, yc - 0.1, **fine)
    m.box(cx0, cz0, cx1, cz1, yc - 0.1, yc, top=main, **fine)
    m.box(sx - 0.6, sz - 0.6, sx + 0.6, sz + 0.6, yc, SCINT_Y, top=dict(lod=1, i=0.9, w=thin), **fine)
    for v in (-0.2, 0.2):                       # the sensor: 6 x 6 mm of cells, against the block
        m.flat([(sx + v, sz - 0.6), (sx + v, sz + 0.6)], SCINT_Y, closed=False, **fine)
        m.flat([(sx - 0.6, sz + v), (sx + 0.6, sz + v)], SCINT_Y, closed=False, **fine)
    u = np.linspace(0.0, 1.0, 9)
    m.flat(np.c_[-2.9 + 0.2 * u - 0.5 * np.sin(pi * u), -1.9 + 5.8 * u], TB, closed=False, lod=2, i=0.55, w=hair)
    m.flat(np.c_[2.6 + 0.55 * np.sin(pi * u), -2.1 + 5.6 * u], TB, closed=False, lod=2, i=0.55, w=hair)
    for hx, hz in ((-2.9, -1.9), (-2.7, 3.9), (2.6, -2.1), (2.6, 3.5)):
        m.ring(hx, hz, 0.14, TB, n=6, **fine)

    # -- scintillator: the block over the white square, resting on the sensor; a lattice of marks on its top face
    m.part = P_SCINT
    x0, x1, z0, z1 = SCINT
    yt = SCINT_Y + SCINT_H
    m.rect(x0, z0, x1, z1, yt, lod=0, i=0.95, w=bold)
    m.rect(x0, z0, x1, z1, SCINT_Y, lod=1, i=0.6, w=thin)
    c = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
    m.seg([(x, SCINT_Y, z) for x, z in c], [(x, yt, z) for x, z in c], lod=0, i=0.9, w=thin)
    gx = x0 + (np.arange(6) + 0.5) * (x1 - x0) / 6
    gz = z0 + (np.arange(6) + 0.5) * (z1 - z0) / 6
    X, Z = (v.ravel() for v in np.meshgrid(gx, gz))
    Y = np.full_like(X, yt)
    m.seg(np.c_[X - 0.22, Y, Z], np.c_[X + 0.22, Y, Z], lod=2, i=0.5, w=hair)
    m.seg(np.c_[X, Y, Z - 0.22], np.c_[X, Y, Z + 0.22], lod=2, i=0.5, w=hair)

    class Board:
        pass
    b = Board()
    b.A, b.B = _model(np.concatenate(m.a)), _model(np.concatenate(m.b))
    b.part, b.lod = np.concatenate(m.p), np.concatenate(m.lod)
    b.inten, b.width = np.concatenate(m.i), np.concatenate(m.w)
    if E.WALL:              # fewer lines, none of them grey: what was under 0.36 is not drawn
        b.inten = np.where(b.inten < 0.36, 0.0, E.wl(b.inten)).astype(np.float32)
    cut = np.searchsorted(b.part, np.arange(len(PART_T) + 1))                 # the parts were added in order
    b.slices = [slice(int(cut[k]), int(cut[k + 1])) for k in range(len(PART_T))]
    return b


def _signal_path():
    """The way of the pulse on the board: sensor -> amplifier -> converter (the MCU of the Nano) -> radio.
    Points in model space, their length along the path, and the lengths at which each leg ends."""
    yb, yn, yr = TB + 0.04, TB + 1.94, TB + 1.94
    nx, xv = NANO_X, NANO_X - 2.0 - 0.14 * 2           # the Nano, and the third trace of its bus to the radio
    sx, sz, yc = SIPM[0], SIPM[1], TB + 0.7
    pts = [(sx, yc, sz), (sx + 0.85, yc, sz), (sx + 0.85, yb, sz),
           (5.0, yb, sz), (6.1, yb, sz + 1.1), (6.35, yb, 2.8), (6.9, TB + 0.37, 3.0),                # 6: the amplifier
           (6.9, yb, 3.45), (6.9, yb, 5.0), (6.1, yb, 5.8), (-3.6, yb, 5.8), (-4.5, yb, 6.7), (nx + 1.55, yb, 6.7),
           (nx + 1.55, yn, 6.7), (nx, yn, 8.3),                                                       # 14: the MCU
           (nx - 1.55, yn, 4.766), (nx - 1.55, yb, 4.766), (xv, yb, 4.766), (xv, yb, -2.6), (xv, yr, -3.05),
           (-8.0, yr, -6.4), (-8.0, yr, -8.4)]                                                        # 21: the aerial
    P = _model(pts)
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return P, cum, (float(cum[6]), float(cum[14]), float(cum[-1]))


_UNIT = {}


def _circle(f, layer, cx, cy, r, inten, n=48, width=L.LW):
    """A circle with a fixed number of vertices (Frame.rings derives it from the radius: a ring that grows
    would crawl)."""
    if n not in _UNIT:
        a = np.linspace(0.0, 2 * np.pi, n + 1)
        _UNIT[n] = (np.cos(a), np.sin(a))
    ca, sa = _UNIT[n]
    f.polyline(layer, cx + r * ca, cy + r * sa, inten, width=width)


def _vbar(f, layer, x0, x1, y_top, y_bot, i):
    """A bar whose top moves smoothly. Frame.rects snaps to whole pixels: the row of pixels that holds the top
    is lit by the fraction of it the bar covers (what hud.bars does for a bar that slides sideways)."""
    k = f.s * f.vz
    off = f.vsy * f.s - f.vcy * k               # Frame.ty(y) = y * k + off
    a = y_top * k + off + 0.5                   # window coordinates: pixel row j covers [j, j + 1)
    b = y_bot * k + off + 0.5
    if b <= a:
        return
    pa = math.floor(a)
    if pa + 1.0 >= b:
        f.rects(layer, x0, (pa - off) / k, x1, (pa + 1.0 - off) / k, i * (b - a))
        return
    ym = (pa + 1.0 - off) / k
    f.rects(layer, [x0, x0], [(pa - off) / k, ym], [x1, x1], [ym, y_bot], [i * (pa + 1.0 - a), i])


def _upto(cum, xs, ys, s):
    """A polyline cut at the length s along it (cum = length at each of its points): (xs, ys) with the
    cut point interpolated, or None when s is past its end."""
    n = int(np.searchsorted(cum, s, side="right"))
    if n >= len(xs):
        return None
    fr = (s - cum[n - 1]) / max(cum[n] - cum[n - 1], 1e-6)
    return np.r_[xs[:n], xs[n - 1] + (xs[n] - xs[n - 1]) * fr], np.r_[ys[:n], ys[n - 1] + (ys[n] - ys[n - 1]) * fr]


def _pen_cut(xs, ys, p):
    """The part of a polyline a pen has drawn at progress p (0..1 of its samples), its last point
    interpolated: (xs, ys, index of the first sample not reached yet)."""
    c = p * (len(xs) - 1)
    m = min(int(c), len(xs) - 2)
    fr = c - m
    xe, ye = xs[m] + (xs[m + 1] - xs[m]) * fr, ys[m] + (ys[m + 1] - ys[m]) * fr
    return np.r_[xs[: m + 1], xe], np.r_[ys[: m + 1], ye], m + 1


class Detector(Scene):
    name = "detector"
    towers = "auto"

    def __init__(self, ctx):
        super().__init__(ctx)
        rng = np.random.default_rng(12)
        self.mesh = _build_board()
        self.path, self.path_cum, self.path_legs = _signal_path()
        self.call_a = _model([c[2] for c in CALL])
        self.call_f = _model([c[3] for c in CALL])
        self.silk_p = _model([c[1] for c in SILK])
        self.p_sipm = _model([(SIPM[0], SCINT_Y, SIPM[1])])
        self.p_aerial = self.path[-1:]
        blk = _model([(SCINT[0], SCINT_Y, SCINT[2]), (SCINT[1], SCINT_Y + SCINT_H, SCINT[3])])
        self.blk = (float(blk[0, 0]), float(blk[1, 0]), float(blk[0, 2]), float(blk[1, 2]))
        self.y_mid = float(0.5 * (blk[0, 1] + blk[1, 1]))
        self.flash = {p: np.nonzero((self.mesh.part == p) & (self.mesh.lod <= 1))[0]       # what lights up of a part
                      for p in (P_SIPM, P_AMP, P_BNC, P_NANO, P_RADIO)}
        # voice cues
        self.c_wait = sd.said("Waiting", 109.0)
        self.c_every = sd.said("Every muon that finds them", 111.4)
        self.c_more = sd.said("The more powerful the muon", 114.7)
        self.c_energy = sd.said("Energy becomes code", 120.0)
        self.c_visitor = sd.said("A visitor", 125.02)
        self.c_messenger = sd.said("A messenger", 127.32)
        self.c_muon = sd.said("A muon...", 130.27, nth=1)        # nth=1: "...A muon" at 01:41 is the first match
        # simulated muons crossing the scintillator: (time, energy 0..1, x, z on the board, stage delay)
        self.demo = [(self.c_every + 0.7, 0.50, -0.7, 2.0, 0.12), (self.c_more + 0.6, 0.25, 2.5, -3.25, 0.12),
                     (self.c_more + 2.2, 0.55, -2.75, 4.55, 0.12), (self.c_more + 3.8, 0.95, 0.85, 0.8, 0.12),
                     (self.c_energy + 0.15, 0.76, 1.6, 3.05, 1.02)]
        self.hits = [_model([(m[2], SCINT_Y + 0.5 * SCINT_H, m[3])])[0].astype(np.float64) for m in self.demo]
        self.dirs = []
        for k in range(len(self.demo)):
            th, ph = rng.uniform(0.05, 0.3), rng.uniform(0, 2 * np.pi)
            self.dirs.append(np.array([math.sin(th) * math.cos(ph), -math.cos(th), math.sin(th) * math.sin(ph)]))
        self.ph_ang = rng.uniform(0, 2 * np.pi, (len(self.demo), 46))
        self.ph_len = rng.uniform(0.3, 1.0, (len(self.demo), 46))
        self.roles = self._roles(ctx)
        # the views cut on these; the last one is not a cut: the board leaves view 05 in one move
        self.cuts = [T0, self.c_wait, self.c_every, self.c_more, self.c_energy]
        self.yaw5 = 118.0 + 10.0 * (self.c_visitor - self.c_energy)      # where view 05 has turned to on 'A visitor'
        self.t_go = self.c_visitor + 0.2                # the flight back to the tower: one eased move
        self.t_land = self.c_messenger + 1.9
        self.t_fly = self.t_go + 0.35                   # the view tag and the call-outs are gone by then
        tw = ctx.towers["C"]
        w1 = 2.0 * (700.0 / math.tan(math.radians(15.0))) * 10.0 * MODEL_K / (96.0 / self.roles["k"]) * 1.08
        self.s_end = float(np.clip(w1 / (0.84 * tw.w), 4.0, 14.0))      # how much smaller it is on the head of the tower
        self.legend = self._legend_at()            # the legend of the call-outs: where, and in which form
        self._tt = np.linspace(0.0, 1.0, 180)
        self._tw = np.linspace(0.0, 1.0, 420)
        self._phi = np.linspace(0.0, 2 * np.pi, 64)

    @staticmethod
    def _roles(ctx):
        """Which column does what, for any tower placement: the object view in the focus bay, the reply chain
        in the widest of the others, then the card (leftmost) and the three replies in what is left."""
        cols = list(ctx.cols)
        fb = ctx.focus_bay
        obj = max(cols, key=lambda c: min(c[1], fb[1]) - max(c[0], fb[0]))
        rest = [c for c in cols if c != obj]
        chain = max(rest, key=lambda c: c[1] - c[0]) if rest else None
        rest = sorted(c for c in rest if c != chain)
        card = next((c for c in rest if c[1] - c[0] >= 250), None)
        rest = [c for c in rest if c != card]
        reps = next((c for c in rest if c[1] - c[0] >= 330), None)
        if card:
            card = _spread(ctx, card, 940.0)        # the card, its log and the legend end above y 940, the replies above 900
        if reps:
            reps = _spread(ctx, reps, 900.0)
        return dict(obj=obj, chain=chain, card=card, reps=reps, k=min(1.2, (obj[1] - obj[0]) / 767.0))

    def _legend_at(self):
        """Where the legend of the call-outs goes, and in which form. Under the card and its log: a block
        diagram when the column is wide enough, else the list of the eight names (in two columns if it can).
        Without a card: the list, in two corners of the object view (if that is wide enough to leave them free)."""
        card = self.roles["card"]
        if card is not None:
            x, w = card[0] + 4.0, card[1] - card[0]
            y = L.HEAD_Y + 86 + 62 + 8 * 26 + 30 + (54 + 5 * 22 + 40 if w >= 380 else 0)
            if w >= 420:
                return dict(kind="diagram", x=x, y=y)
            rows = 4 if w >= 400 else 8
            return dict(kind="list", x=x, y=y, at=[(x + 212.0 * (k // rows), y + 36.0 + 27.0 * (k % rows))
                                                   for k in range(len(CALL))])
        b0, b1 = self.roles["obj"]
        if b1 - b0 < 560:
            return None
        return dict(kind="list", x=b0 + 30.0, y=320.0,
                    at=[(b0 + 30.0, (350.0 if k < 4 else 1009.0) + (k % 4) * 27.0) for k in range(len(CALL))])

    def _legend(self, f, t, hot):
        """What the numbers on the board are. Under the card it belongs to the card and stays to the end of
        the scene; in the corners of the object view it leaves with the view tag.
        As a block diagram, in the way of the drawings of the Apollo guidance: every part a box, the signal
        going down from one to the next, what it is on the way written beside the line; a muon lights the
        boxes in its turn and a mark runs down the lines with the pulse on the board."""
        lg = self.legend
        if lg is None:
            return
        x0, y0 = lg["x"], lg["y"]
        left = None if self.roles["card"] is not None else self.t_fly - t
        if lg["kind"] == "list":
            B.tag(f, "w", x0, y0, "PARTS", B.io(t - T0 - T_CALL + 0.2, left, out=0.35, span=0.5), size=L.T_MICRO, pad=3,
                  cps=90.0, key=89)
            for k, c in enumerate(CALL):
                lx, ly = lg["at"][k]
                a = B.io(t - T0 - T_CALL - 0.1 * k, left, out=0.35, span=0.6)
                lay = "r" if hot[k] else "w"
                B.tag(f, lay, lx, ly, f"{k + 1:02d}", a, size=L.T_MICRO, pad=3, cps=60.0, key=90 + k)
                f.text(lay, lx + 34, ly + 1, B.resolve(c[0], a - 0.1, 90.0, key=100 + k), size=L.T_SMALL, alpha=0.9)
            return
        bw, bh, pitch, xs = 204.0, 26.0, 44.0, x0 + 232.0        # main boxes, and where the side boxes start
        yr = lambda r: y0 + 26.0 + pitch * r                      # top of row r

        def box(k, x, y, w):
            lay = "r" if hot[k] else "w"
            f.rect(lay, x, y, x + w, y + bh, E.wl(0.75), width=WH)
            f.tag(lay, x + 5, y + 19, f"{k + 1:02d}", size=L.T_MICRO, pad=3)
            f.text(lay, x + 38, y + 19, CALL[k][0], size=L.T_SMALL, alpha=0.9)

        xa = x0 + 19.0
        with f.build(B.io(t - T0 - T_CALL + 0.2, left, out=0.35, span=1.3), (x0 - 6, y0 - 22, xs + 190, yr(4) + bh + 6),
                     flow="tb", wave=0.9, key=89):
            f.tag("w", x0, y0, "SIGNAL PATH", size=L.T_MICRO, pad=3)
            for r, (k, what) in enumerate(FLOW):
                box(k, x0, yr(r), bw)
                if r < len(FLOW) - 1:           # the line to the next box, what it carries
                    ya, yb = yr(r) + bh, yr(r + 1)
                    f.segments("w", [xa, xa - 5, xa + 5], [ya, yb - 7, yb - 7], [xa, xa, xa], [yb, yb, yb], E.wl(0.7), width=WH)
                    f.text("w", xa + 14, yb - 4, what, size=L.T_MICRO, alpha=0.55)
            for k, r, into in ((2, 1, True), (6, 2, False), (7, 0, None)):       # bias supply, BNC, power LED: beside
                box(k, xs, yr(r), 186.0)
                if into is not None:
                    ym, xl, xr = yr(r) + bh / 2, x0 + bw, xs
                    xt, sg = (xl, 1.0) if into else (xr, -1.0)                  # the tip of the arrow
                    f.segments("w", [xl, xt + sg * 7, xt + sg * 7], [ym, ym - 5, ym + 5], [xr, xt, xt], [ym, ym, ym], E.wl(0.7), width=WH)
            ym = yr(4) + bh / 2                  # ... and what the radio sends leaves the drawing
            f.segments("w", [x0 + bw, xs - 9, xs - 9], [ym, ym - 5, ym + 5], [xs - 2, xs - 2, xs - 2], [ym, ym, ym], E.wl(0.7), width=WH)
            f.text("r" if hot[5] else "w", xs + 6, ym + 6, "/MUON/C", size=L.T_SMALL, alpha=0.9)
        k, a = self._current(t)                  # the pulse of the last muon, running down the lines
        if k is None or a > 3.2:
            return
        t0, t1, t2, t3 = self._times(self.demo[k][4])
        for r, (ta, tb) in enumerate(((0.08, t0), (t0, t1), (t1, t2), (t2, t3))):
            if ta <= a <= tb:
                ya, yb = yr(r) + bh, yr(r + 1)
                yh = ya + (yb - ya) * (a - ta) / (tb - ta)
                f.segments("r", [xa], [ya], [xa], [yh], 0.3, 1.6, width=L.LW_BOLD)
                f.dots("r", [xa], [yh], 3.8, 1.8)

    # ------------------------------------------------------------------ views
    def _flight(self, t):
        """The move back to the tower: (progress 0..1 with an ease in and an ease out, yaw, elevation in degrees).
        The turning of view 05 is not cut: it slows down and comes to rest square to the wall as the board lands."""
        v = float(np.clip((t - self.t_go) / (self.t_land - self.t_go), 0.0, 1.0))
        u = v * v * v * (v * (6.0 * v - 15.0) + 10.0)
        span = self.t_land - self.c_visitor
        w = float(np.clip((t - self.c_visitor) / span, 0.0, 1.0))
        turn = 180.0 - self.yaw5 % 360.0
        n1 = 10.0 * span / turn if turn > 1.0 else 0.0
        if n1 >= 1.0:                           # 10 deg/s at the start (as in view 05), nothing at the end
            yaw = self.yaw5 + turn * (1.0 - (1.0 - w) ** n1)
        else:
            yaw = self.yaw5 + 5.0 * span * (1.0 - (1.0 - w) ** 2)
        return u, yaw, 34.0 - 4.0 * u

    def _camera(self, t, ctx):
        b0, b1 = self.roles["obj"]
        kz = self.roles["k"]                        # the object is scaled to the bay it gets
        sc = ((b0 + b1) / 2, 668.0)
        u = 0.0
        if t < self.c_wait:
            yaw, el, name = math.radians(28 + 9 * (t - T0)), math.radians(27), "01 // PERSPECTIVE"
        elif t < self.c_every:
            cam = OrthoCamera((0.0, 1.3, 60.0), (0.0, 1.3, 0.0), scale=31.0 * kz, screen_center=sc)
            return cam, "02 // ORTHO_SIDE", sc, 1.0
        elif t < self.c_more:
            yaw, el, name = math.radians(-42 + 9 * (t - self.c_every)), math.radians(21), "03 // PERSPECTIVE"
        elif t < self.c_energy:
            cam = OrthoCamera((0.0, 60.0, 1e-3), (0.0, 0.0, 0.0), scale=27.0 * min(kz, 0.96), up=(0.0, 0.0, -1.0),
                              screen_center=sc, roll_deg=45.0 + 3.0 * (t - self.c_more))   # (it has to fit the height too)
            return cam, "04 // ORTHO_TOP", sc, 1.0
        elif t < self.c_visitor:
            yaw, el, name = math.radians(118 + 10 * (t - self.c_energy)), math.radians(34), "05 // PERSPECTIVE"
        else:
            u, yaw, el = self._flight(t)
            yaw, el, name = math.radians(yaw), math.radians(el), "05 // PERSPECTIVE"
        D = 96.0 / kz
        shrink = 1.0
        if u > 0.0:                     # the board flies back to where it lives: the head of the centre tower
            tw = ctx.towers["C"]
            tx, ty = tw.cx, tw.top + tw.det_h * 0.5
            sc = (sc[0] + (tx - sc[0]) * u, sc[1] + (ty - sc[1]) * u - 70.0 * math.sin(math.pi * u))
            shrink = self.s_end ** u    # its size goes down by the same factor at every step: no rush at the start
            D *= shrink
        pos = (D * math.cos(el) * math.sin(yaw), D * math.sin(el) + 0.8, D * math.cos(el) * math.cos(yaw))
        return Camera(pos, (0.0, 0.8, 0.0), fov_deg=30.0, screen_center=sc), name, sc, shrink

    # ------------------------------------------------------------------ draw
    def draw(self, f, t, ctx):
        cam, vname, sc, shrink = self._camera(t, ctx)
        b0, b1 = self.roles["obj"]
        clip = (b0 - 16, L.HEAD_Y + 8, b1 + 16, L.VIEW[3]) if shrink == 1.0 else None
        hot = self._draw_object(f, cam, t, shrink, clip=clip, anchor=(b1 - 302, 1052.0))
        if t < self.t_fly:
            self._view_tag(f, t, vname, ctx, cam, hot)
        self._legend(f, t, hot)
        self._chain(f, t, ctx)
        self._left(f, t, ctx)
        self._right(f, t, ctx)
        self._bottom(f, t, ctx)
        self._visitor(f, t, ctx)
        marks = [(self.c_wait, "WAIT"), (self.c_every, "MUON"), (self.c_more, "REPLY"), (self.c_energy, "CHAIN"),
                 (self.c_visitor, "VISITOR"), (sd.T_BLOOM, "BLOOM")]
        age = B.io(t - T0, self.c_visitor - t, out=0.6)                # taken apart: the sky is clear for the visitor
        if age >= 0.0:
            hud.show_strip(f, t, ctx, "SCENE 02 // DETECTOR / DATA EXPLAINER", T0, 139.0, marks, age=age)
        # the edge meters were off during the static of 'A MUON': they are built again with the reveal
        return {"edge_kw": {"reveal": B.lin(t, T0, T0 + 0.7)}}

    def _draw_object(self, f, cam, t, shrink, clip=None, anchor=None):
        """The board, the muons that cross it and the pulse they send along it. Returns, for every call-out,
        whether its part is busy (its number then turns red)."""
        M = self.mesh
        if clip:
            f.set_clip(*clip)
        ax, ay, _, _ = cam.project(M.A)
        bx, by, _, _ = cam.project(M.B)
        # the detail goes as the board gets small: only its outlines land on the tower
        lodf = np.array([1.0, 1.0 - smoothstep(3.5, 8.0, shrink), 1.0 - smoothstep(1.5, 3.5, shrink)], np.float32)[M.lod]
        inten = M.inten * lodf
        width = M.width - (M.width - L.LW_HAIR) * float(smoothstep(2.0, 6.0, shrink))
        if E.WALL:
            width = np.maximum(width, E.WALL_LINE)
        for p, t_p in enumerate(PART_T):
            a = t - T0 - t_p
            if a < 0.0:
                continue
            sl = M.slices[p]
            if a < 2.6:                         # being made: part after part, the scintillator last
                rect = (float(min(ax[sl].min(), bx[sl].min())) - 10.0, float(min(ay[sl].min(), by[sl].min())) - 10.0,
                        float(max(ax[sl].max(), bx[sl].max())) + 10.0, float(max(ay[sl].max(), by[sl].max())) + 10.0)
                with f.build(a, rect, wave=0.35, line=0.3, key=20 + p):
                    f.segments("w", ax[sl], ay[sl], bx[sl], by[sl], inten[sl], width=width[sl])
            else:
                f.segments("w", ax[sl], ay[sl], bx[sl], by[sl], inten[sl], width=width[sl])
        a = t - T0 - T_PATH
        if 0.0 <= a < 2.3:                      # last of all, a pen runs the way the signal will take, and lets it go
            px, py, _, _ = cam.project(self.path)
            g = 0.95 * float(1.0 - smoothstep(1.4, 2.3, a))
            cut = _upto(self.path_cum, px, py, self.path_cum[-1] * float(B.ease(a / 1.2)))
            if cut is None:
                f.polyline("w", px, py, g, width=L.LW)
            else:
                f.polyline("w", cut[0], cut[1], g, width=L.LW)
                f.dots("w", cut[0][-1:], cut[1][-1:], 3.4, 1.7)
        hot = [False] * len(CALL)
        x0b, x1b, z0b, z1b = self.blk
        for k, (tm, e, _, _, dl) in enumerate(self.demo):
            a = t - tm
            if not (0 <= a < 3.2):
                continue
            d = self.dirs[k]
            hit = self.hits[k]
            fade = math.exp(-a / 1.1)
            prog = min(1.0, a / 0.16)
            A = hit - d * 34.0
            P = np.stack([A, A + d * 68.0 * prog]).astype(np.float32)
            px, py, _, ok = cam.project(P)
            if ok.all():
                f.segments("r", px[:1], py[:1], px[1:], py[1:], (0.7 + 0.8 * e) * fade, width=L.LW_BOLD)
                if prog < 1.0:
                    f.dots("r", px[1:], py[1:], 4.0, 1.8)
            if a < 0.08:
                continue
            ah = a - 0.08
            hot[0] = hot[0] or ah < 0.7
            # scintillation: light spreading in the block from the crossing point, collected by the SiPM
            n = int(12 + 34 * e)
            ang, ln = self.ph_ang[k, :n], self.ph_len[k, :n]
            r = (1.6 + 6.5 * ln) * (1 - math.exp(-ah / 0.12))
            ex = np.clip(hit[0] + np.cos(ang) * r, x0b, x1b)
            ez = np.clip(hit[2] + np.sin(ang) * r, z0b, z1b)
            Pa = np.repeat(hit[None], n, 0).astype(np.float32)
            Pb = np.stack([ex, np.full(n, self.y_mid), ez], 1).astype(np.float32)
            hx, hy, _, _ = cam.project(Pa)
            qx, qy, _, _ = cam.project(Pb)
            fl = math.exp(-ah / 0.5)
            f.segments("w", hx, hy, qx, qy, 0.9 * fl * (0.4 + e), 0.15 * fl)
            f.dots("w", qx, qy, 1.8, 1.3 * fl)
            sx, sy, _, _ = cam.project(self.p_sipm)             # the light that reaches the sensor
            f.segments("r", hx[:1], hy[:1], sx, sy, 0.9 * e * fl, width=L.LW)
            f.dots("r", sx, sy, 5.0, 1.6 * e * fl)
            u = min(1.0, ah / 1.2)
            _circle(f, "r", float(hx[0]), float(hy[0]), (10 + 120 * e * (1 - (1 - u) ** 3)) / shrink,
                    (1 - u) ** 1.5 * (0.6 + e))
            f.dots("w", hx[:1], hy[:1], 3.0, 1.6 * fl)
            if shrink < 1.5:
                self._pulse(f, cam, a, e, dl, hot)
            if shrink < 1.5 and a < 2.6 and anchor is not None:
                hud.callout(f, float(hx[0]), float(hy[0]), anchor[0] - float(hx[0]), anchor[1] - float(hy[0]),
                            f"MU  E {e:.2f}",
                            [f"DE {1.6 + 3.4 * e:.2f} MEV", f"{int(9000 * (1.6 + 3.4 * e)):,} PHOTONS".replace(",", " ")],
                            red=True, side=1, build=B.io(ah, 2.6 - a, out=0.4, span=0.9))
        if clip:
            f.set_clip()
        return hot

    @staticmethod
    def _times(dl):
        """When the pulse of a muon leaves the SiPM, reaches the amplifier, the converter (with the CODE stage of
        the chain) and the radio, in seconds after the muon."""
        t2 = max(0.34 + 0.26, dl + 0.04)
        return 0.14, 0.34, t2, t2 + 0.32

    def _pulse(self, f, cam, a, e, dl, hot):
        """What the sensor saw runs along the board: a red head on the traces, from the SiPM to the amplifier,
        to the converter of the Nano (it gets there when the chain says CODE), to the radio, which sends it."""
        M = self.mesh
        s1, s2, s3 = self.path_legs
        t0, t1, t2, t3 = self._times(dl)
        ax, ay, _, _ = cam.project(M.A)
        bx, by, _, _ = cam.project(M.B)
        for part, ta, kc in ((P_SIPM, 0.08, 1), (P_AMP, t1, 3), (P_BNC, t1 + 0.06, 6), (P_NANO, t2, 4), (P_RADIO, t3, 5)):
            if a >= ta:                         # the part it reaches lights up
                g = math.exp(-(a - ta) / 0.45)
                hot[kc] = hot[kc] or a - ta < 0.7
                if g > 0.03:
                    ix = self.flash[part]
                    f.segments("r", ax[ix], ay[ix], bx[ix], by[ix], 1.1 * g * (0.5 + e), width=L.LW)
        if a < t0:
            return
        s = float(np.interp(a, [t0, t1, t2, t3], [0.0, s1, s2, s3]))
        px, py, _, _ = cam.project(self.path)
        gone = math.exp(-max(0.0, a - t3) / 0.7)
        cut = _upto(self.path_cum, px, py, s)
        if cut is not None:                     # the way it has gone so far glows, to where the head is
            hx, hy = cut[0][-1], cut[1][-1]
            f.polyline("r", cut[0], cut[1], 0.5 + 0.4 * e, width=L.LW)
            tl = np.linspace(max(0.0, s - 4.0), s, 10)              # ... and a bright tail behind the head
            tx = np.interp(tl, self.path_cum, px)
            ty = np.interp(tl, self.path_cum, py)
            f.polyline("r", tx, ty, 0.1, width=L.LW_BOLD, i_end=1.5)
            f.dots("r", [hx], [hy], 4.2, 1.8)
            f.dots("w", [hx], [hy], 1.6, 1.2)
        elif gone > 0.03:
            f.polyline("r", px, py, (0.5 + 0.4 * e) * gone, width=L.LW)
        if a >= t3:                             # sent: three waves leave the aerial
            wx, wy, _, _ = cam.project(self.p_aerial)
            for j in range(3):
                aw = a - t3 - 0.12 * j
                if 0.0 <= aw < 0.8:
                    _circle(f, "r", float(wx[0]), float(wy[0]), 10.0 + 95.0 * aw, (1.0 - aw / 0.8) ** 1.5 * (0.5 + 0.6 * e),
                            n=36)

    def _view_tag(self, f, t, vname, ctx, cam, hot):
        """Name of the view and the numbered call-outs. Made at the cut to the scene (the call-outs once the
        parts are there); on every cut of the voice only what is new is made again: the name of the view and
        the call-outs, whose numbers ride with the parts. All of it is taken apart when the board starts to
        fly back to its tower."""
        b0, b1 = self.roles["obj"]
        x, y = b0 + 2, L.HEAD_Y + 52
        left = self.t_fly - t
        cut = max([c for c in self.cuts if c <= t], default=T0)
        a_all = B.io(t - T0 - 0.1, left, out=0.35, span=0.7)                        # what stays from view to view
        a_cut = B.io(t - cut, left, out=0.35, span=0.7)                             # new in this view
        B.tag(f, "w", x, y, f"VIEW {vname}", a_cut, size=L.T_LABEL, pad=4, cps=70.0, key=3)
        f.text("w", x, y + 32, B.resolve("DET_C // 1 OF 3 IDENTICAL // SIMULATED MUONS", a_all - 0.2, 120.0, key=5),
               size=L.T_MICRO, alpha=0.7)
        if not getattr(cam, "ortho", False) or vname.startswith("04"):      # what is printed beside the chips (not edge-on)
            px, py, _, _ = cam.project(self.silk_p)
            for k, (name, _) in enumerate(SILK):
                f.text("w", float(px[k]) + 7, float(py[k]) - 3, B.resolve(name, a_cut - 0.5 - 0.08 * k, 60.0, key=110 + k),
                       size=L.T_MICRO, alpha=0.6)
        px, py, _, _ = cam.project(self.call_a)
        qx, qy, _, _ = cam.project(self.call_f)
        qx, qy = self._flags(qx, qy, b0, b1)
        for k in range(len(CALL)):              # a point on the part, a leader, its number in a ring
            a = t - T0 - T_CALL - 0.1 * k if cut == T0 else t - cut - 0.05 - 0.05 * k
            a = B.io(a, left, out=0.3, span=0.5)
            if a < 0.0:
                continue
            x, y, X, Y = float(px[k]), float(py[k]), float(qx[k]), float(qy[k])
            lay = "r" if hot[k] else "w"
            with f.build(a, (min(x, X) - 20, min(y, Y) - 20, max(x, X) + 20, max(y, Y) + 20), flow="out", origin=(x, y),
                         wave=0.12, line=0.18, marks=False, key=80 + k):
                f.dots(lay, [x], [y], 2.6, 1.3)
                d = math.hypot(X - x, Y - y)
                if d > R_CALL + 2.0:
                    f.segments(lay, [x], [y], [X - (X - x) / d * R_CALL], [Y - (Y - y) / d * R_CALL], E.wl(0.65), width=WH)
                f.rings(lay, [X], [Y], [R_CALL], 0.9, width=L.LW)
                f.text(lay, X, Y + 5, f"{k + 1:02d}", size=L.T_MICRO, anchor="ms", alpha=0.95)

    @staticmethod
    def _flags(qx, qy, x0, x1):
        """Where the numbers of the call-outs go: inside the column of the view (never behind a tower), and
        apart from each other - two that come close on screen are pushed apart, the more the closer, so that
        nothing jumps while the board turns."""
        qx = np.clip(qx.astype(np.float64), x0 + 18.0, x1 - 18.0)
        qy = np.clip(qy.astype(np.float64), L.HEAD_Y + 110.0, L.VIEW[3] - 18.0)
        dmin = 2.0 * R_CALL + 6.0
        for _ in range(3):
            dx, dy = qx[:, None] - qx[None, :], qy[:, None] - qy[None, :]
            d = np.hypot(dx, dy)
            push = 0.5 * np.clip(dmin - d, 0.0, None)
            np.fill_diagonal(push, 0.0)
            d = np.maximum(d, 1e-3)
            qx, qy = qx + (push * dx / d).sum(1), qy + (push * dy / d).sum(1)
        return qx, qy

    # ------------------------------------------------------------------ the chain
    def _current(self, t):
        """The demo muon whose reply is on the read-outs: (index, age) or (None, 0)."""
        k = None
        for i, m in enumerate(self.demo):
            if m[0] <= t:
                k = i
        return (k, t - self.demo[k][0]) if k is not None else (None, 0.0)

    def _at(self, t, s):
        """The latest simulated muon that has reached stage s of the chain: (index, seconds since) or (None, -1)."""
        k = None
        for i, m in enumerate(self.demo):
            if m[0] + s * m[4] <= t:
                k = i
        return (k, t - self.demo[k][0] - s * self.demo[k][4]) if k is not None else (None, -1.0)

    def _env(self, t, s, attack, tau):
        """Level of stage s: every muon that reached it adds its energy, with an attack and a decay."""
        v = 0.0
        for tm, e, _, _, dl in self.demo:
            a = t - tm - s * dl
            if a > 0.0:
                v += e * (1.0 - math.exp(-a / attack)) * math.exp(-a / tau)
        return v

    def _chain(self, f, t, ctx):
        if self.roles["chain"] is None:
            return
        b0, b1 = self.roles["chain"]
        x0, x1 = b0 + 16, b1 - 12
        y_top, y_bot = L.HEAD_Y + 34, L.VIEW[3] - 6
        rh = (y_bot - y_top) / 4
        k0, age0 = self._current(t)
        dl0 = self.demo[k0][4] if k0 is not None else 1.0
        draw = (self._stage_energy, self._stage_code, self._stage_light, self._stage_sound)
        for s, name in enumerate(STAGES):
            y0 = y_top + s * rh
            y1 = y0 + rh - 22
            k, a = self._at(t, s)
            hot = k is not None and a < 1.3
            wait = k is None or k < k0          # the muon on its way has not reached this stage yet
            p = (age0 - s * dl0) / dl0 if k0 is not None else -1.0      # the mark on the arrow to the next stage
            busy = s < 3 and 0.0 <= p and age0 - (s + 1) * dl0 < 0.5
            # the four stages are constructed one after the other at the start of the scene; after that they
            # are instruments: they run, and re-plot what they caught when a muon reaches them
            with f.build(t - (T0 + 0.6) - 0.2 * s, (x0 - 8, y0 - 10, x1 + 8, y1 + 8), key=30 + s, wave=0.35):
                f.rects("w", x0, y0, x1, y0 + 4, 0.9)
                f.tag("r" if hot else "w", x0, y0 + 36, f"{s + 1:02d} {name}", size=L.T_TAG, pad=5)
                status = ("WAITING" if int(t * 2) % 2 else "WAITING _") if wait else f"T+{min(a, 99.99):05.2f} S"
                f.text("r" if hot else "w", x0, y0 + 62, status, size=L.T_MICRO, alpha=0.6 if wait else 0.9)
                if s < 3:           # 'becomes' arrow to the next stage
                    xa = x0 + 60
                    f.segments("w", [xa, xa - 7, xa + 7], [y0 + 74, y1 + 6, y1 + 6], [xa, xa, xa],
                               [y1 + 16, y1 + 16, y1 + 16], E.wl(0.7), width=WH)
                    f.text("r" if busy else "w", xa + 16, (y0 + 74 + y1 + 16) / 2 + 6, "BECOMES", size=L.T_MICRO,
                           alpha=1.0 if busy else 0.6)
                draw[s](f, t, x0 + 190, x1 - 6, y0 + 22, y1, k, a)
            if s < 3 and k0 is not None and p >= 0.0:           # what the muon gave travels down the arrow
                xa, ya, yb = x0 + 60, y0 + 74, y1 + 16
                if p <= 1.0:
                    yh = ya + (yb - ya) * float(smoothstep(0.0, 1.0, p))
                    f.segments("r", [xa], [max(ya, yh - 34.0)], [xa], [yh], 0.15, 1.6, width=L.LW_BOLD)
                    f.dots("r", [xa], [yh], 4.4, 1.8)
                    f.dots("w", [xa], [yh], 1.6, 1.2)
                else:
                    g = math.exp(-(age0 - (s + 1) * dl0) / 0.4)
                    if g > 0.03:
                        f.segments("r", [xa, xa - 7, xa + 7], [ya, yb - 10, yb - 10], [xa, xa, xa], [yb, yb, yb], 1.3 * g,
                                   width=L.LW)

    def _stage_energy(self, f, t, x0, x1, y0, y1, k, a):
        """The pulse of the photo-sensor on a scope: the input runs under the threshold; a muon triggers it and
        its pulse is written by a pen (fast rise, slow decay), then read by a cursor."""
        H, W = y1 - y0 - 14, x1 - x0 - 16
        yb = y1 - 4
        f.segments("w", [x0, x0], [y0, y1], [x0, x1], [y1, y1], E.wl(0.6), width=WH)
        tk = x0 + 8 + np.arange(11) * W / 10.0
        f.segments("w", tk, np.full(11, y1), tk, np.full(11, y1 + 5.0), E.wl(0.5), width=WH)
        vk = yb - np.arange(1, 5) * 0.25 * H
        f.segments("w", np.full(4, x0 - 5.0), vk, np.full(4, x0), vk, E.wl(0.5), width=WH)
        yt = yb - 0.12 * H                              # threshold: its dashes march
        da = np.arange(x0 + 8 - 16 + (t * 10.0) % 16.0, x1 - 8, 16.0)
        s0, s1 = np.maximum(da, x0 + 8), np.minimum(da + 8, x1 - 8)
        on = s1 > s0
        f.segments("r", s0[on], np.full(int(on.sum()), yt), s1[on], np.full(int(on.sum()), yt), 0.9)
        f.text("r", x1 - 4, yt - 8, "THRESHOLD", size=L.T_MICRO, anchor="rs", alpha=0.9)
        xs = np.linspace(x0 + 8, x1 - 8, 150)           # the input, live: the noise floor runs to the left
        u = (xs - x0) * 0.055 + t * 4.0
        nz = 0.5 + 0.22 * np.sin(u) + 0.14 * np.sin(2.3 * u + 1.3) + 0.09 * np.sin(5.1 * u + 0.4) + 0.05 * np.sin(9.7 * u + 2.1)
        f.polyline("w", xs, yb - 0.05 * H * nz, E.wl(0.42), width=WH)
        if k is None:
            return
        tt = self._tt
        e = self.demo[k][1]
        xs = x0 + 8 + tt * W
        shape = (np.exp(-tt / 0.22) - np.exp(-tt / 0.018)) / 0.76
        ys = yb - np.clip(e * shape, 0, 1) * H
        br = 0.55 + 0.45 * math.exp(-a / 0.8)
        p = min(1.0, a / 0.45)
        if p < 1.0:                                     # the pen writes the new pulse over the last one
            cx, cy, m = _pen_cut(xs, ys, p)
            f.polyline("w", cx, cy, 1.2 * br, width=L.LW_BOLD)
            f.dots("w", cx[-1:], cy[-1:], 3.2, 1.7)
            if k > 0:
                old = yb - np.clip(self.demo[k - 1][1] * shape, 0, 1) * H
                bo = 0.55 + 0.45 * math.exp(-(t - self.demo[k - 1][0]) / 0.8)
                f.polyline("w", xs[m:], old[m:], 1.2 * bo, width=L.LW_BOLD)
        else:
            f.polyline("w", xs, ys, 1.2 * br, width=L.LW_BOLD)
        x_pen = x0 + 8 + p * W
        # where it crosses the threshold, up and down, and the time it stays over it
        tc, tf = -0.018 * math.log(max(1.0 - 0.0912 / e, 1e-3)), -0.22 * math.log(0.0912 / e)
        xc, xf = x0 + 8 + tc * W, x0 + 8 + tf * W
        f.segments("r", [xc], [yt], [min(x_pen, xf)], [yt], 1.3, width=L.LW)
        f.rings("r", [xc], [yt], [5.0 * float(B.spring(a / 0.2))], 1.3, width=L.LW)
        if x_pen >= xf:
            f.rings("r", [xf], [yt], [5.0 * float(B.spring((a - 0.45 * tf) / 0.2))], 1.3, width=L.LW)
        f.segments("r", [xc - 5, xc + 5, xc - 5], [y0 + 2, y0 + 2, y0 + 2], [xc, xc, xc + 5], [y0 + 9, y0 + 9, y0 + 2], 1.1)
        # the peak, held: a dashed line the pen draws to the right
        xp, yp = x0 + 8 + 0.0491 * W, yb - min(1.0, 0.9665 * e) * H
        dx = np.arange(xp + 12, xp + 12 + (x1 - 8 - xp - 12) * B.ease(B.lin(a, 0.03, 0.4)), 12.0)
        f.segments("w", dx, np.full_like(dx, yp), np.minimum(dx + 6, x1 - 8), np.full_like(dx, yp), E.wl(0.55), width=WH)
        f.segments("w", [xp - 6, xp], [yp, yp - 6], [xp + 6, xp], [yp, yp + 6], 1.1 * float(B.spring(a / 0.2)), width=WH)
        if a >= 0.6:                                    # the cursor reads the pulse, back and forth
            g = min(1.0, (a - 0.6) / 0.25)
            tcur = 0.02 + 0.96 * (0.5 - 0.5 * math.cos(2 * math.pi * (a - 0.6) / 3.4))
            v = e * (math.exp(-tcur / 0.22) - math.exp(-tcur / 0.018)) / 0.76
            xcur, ycur = x0 + 8 + tcur * W, yb - min(max(v, 0.0), 1.0) * H
            f.segments("w", [xcur], [ycur], [xcur], [ycur + (y1 - ycur) * g], E.wl(0.55), width=WH)
            f.dots("w", [xcur], [ycur], 3.0, 1.5 * g)
            if g >= 1.0:
                f.text("w", xcur, y1 + 17, f"{v * 3.3:.2f} V", size=L.T_MICRO, anchor="ms", alpha=0.85)
        with f.build(a, (x0 - 6, y0 - 6, x1 + 6, y0 + 26), key=40, wave=0.25, line=0.22, marks=False):
            if W >= 420:
                f.text("r", xc + 12, y0 + 12, "TRIG", size=L.T_MICRO, alpha=0.9)
            f.text("w", x1 - 4, y0 + 16, f"PEAK {e * 3.3:.2f} V   DE {1.6 + 3.4 * e:.2f} MEV", size=L.T_SMALL,
                   anchor="rs", alpha=0.95)

    def _stage_code(self, f, t, x0, x1, y0, y1, k, a):
        """The converter of the Nano: ten bits tried one after the other, the heaviest first (each one is set,
        compared, kept or cleared), then the float that goes out over OSC, live."""
        nb = ADC_BITS
        cw = min(46.0, (x1 - x0 - 10) / nb)
        bx = x0 + 6 + np.arange(nb) * cw
        ya, yb = y0 + 8, y0 + 8 + cw - 8
        f.segments("w", np.r_[bx, bx + cw - 8, bx + cw - 8, bx], np.r_[np.full(2 * nb, ya), np.full(2 * nb, yb)],
                   np.r_[bx + cw - 8, bx + cw - 8, bx, bx], np.r_[np.full(nb, ya), np.full(2 * nb, yb), np.full(nb, ya)], E.wl(0.6),
                   width=WH)
        if cw >= 36:
            for i in range(nb):
                f.text("w", bx[i] + (cw - 8) / 2, yb + 17, f"{1 << (nb - 1 - i)}", size=L.T_MICRO, anchor="ms", alpha=0.45)
        e = self.demo[k][1] if k is not None else 0.0
        val = int(e * ADC_MAX)
        ib = a / T_BIT if k is not None else nb + 9.0
        done = int(min(max(ib, 0.0), nb))               # bits settled so far
        acc, trials = 0, []
        for i in range(nb):                             # what a successive-approximation converter does
            trial = acc + (1 << (nb - 1 - i))
            trials.append(trial)
            if trial <= val:
                acc = trial
            if i == done - 1:
                shown = acc
        if done == 0:
            shown = 0
        br = 0.55 + 0.45 * math.exp(-a / 0.8) if k is not None else 0.0
        old = int(self.demo[k - 1][1] * ADC_MAX) if k is not None and k > 0 else 0
        for i in range(nb):
            w = 1 << (nb - 1 - i)
            if i < done:
                lv = 0.95 * br + 0.1 if val & w else 0.0
            elif i == done and k is not None:
                lv = 0.5                                # the bit being tried
            else:
                lv = 0.3 if old & w and k is not None else 0.0      # the last word, until this one overwrites it
            if lv and E.WALL and not i < done:          # tried: the lower half; the last word: a small square
                xm_, ym_ = bx[i] + (cw - 8) / 2, (ya + yb) / 2
                if lv == 0.5:
                    f.rects("w", bx[i] + 3, ym_, bx[i] + cw - 11, yb - 3, 1.0)
                else:
                    f.rects("w", xm_ - 4, ym_ - 4, xm_ + 4, ym_ + 4, 1.0)
            elif lv:
                f.rects("w", bx[i] + 3, ya + 3, bx[i] + cw - 11, yb - 3, E.wl(lv))
        if k is not None and ib < nb + 2.0:             # the bracket under the bit being tried
            xb = x0 + 6 + min(ib, nb - 1.0) * cw
            wb = (cw - 8) * (1.0 - max(0.0, ib - nb) / 2.0)
            f.segments("r", [xb, xb, xb + wb], [yb + 2, yb + 6, yb + 6], [xb, xb + wb, xb + wb], [yb + 6, yb + 6, yb + 2], 1.2)
        ty = yb + 42
        word = f"ADC {shown:04d}/{ADC_MAX}   0x{shown:03X}" if k is not None else f"ADC ----/{ADC_MAX}   0x---"
        f.text("w", x0 + 6, ty, word, size=L.T_SMALL, alpha=0.9 if k is not None else 0.5)
        # the float the tower streams: the pulse, decaying, on its noise floor
        v = 0.018 + 0.012 * math.sin(5.3 * t + 3.4) * math.sin(0.9 * t)
        for tm, em, _, _, dl in self.demo:
            am = t - tm - dl - nb * T_BIT
            if am > 0.0:
                v += em * math.exp(-am / sd.Detectors.TAU) * (1.0 - math.exp(-am / 0.02))
        v = min(v, 1.0)
        sent = k is not None and nb * T_BIT <= a < nb * T_BIT + 1.3
        osc = f"OSC  /MUON/C  ,F  {v:.3f}"
        if k is not None and a >= nb * T_BIT:           # the word is sent: the line is typed again, then it runs
            osc = B.decode(osc, a - nb * T_BIT, cps=80.0, key=41)
        f.text("r" if sent else "w", x0 + 6, ty + 30, osc, size=L.T_LABEL, alpha=0.95)
        wl = min(270.0, x1 - x0 - 12)
        f.segments("w", [x0 + 6, x0 + 6, x0 + 6 + wl / 2, x0 + 6 + wl], [ty + 48, ty + 44, ty + 44, ty + 44],
                   [x0 + 6 + wl, x0 + 6, x0 + 6 + wl / 2, x0 + 6 + wl], [ty + 48, ty + 52, ty + 52, ty + 52], E.wl(0.5), width=WH)
        hud.bars(f, "r" if sent else "w", x0 + 6, ty + 38, x0 + 6 + wl * v, ty + 45, 0.95)
        # the approximations, as a staircase closing in on what came in
        gx0, gx1, gy0, gy1 = x0 + 318, x1 - 4, yb + 26, y1 - 2
        if gx1 - gx0 < 150:
            return
        f.segments("w", [gx0, gx0], [gy0, gy1], [gx0, gx1], [gy1, gy1], E.wl(0.5), width=WH)
        f.text("w", gx0 + 8, gy0 + 10, f"SAR // {nb} BIT", size=L.T_MICRO, alpha=0.6)
        if k is None:
            return
        sw = (gx1 - gx0 - 16) / nb
        gy = lambda q: gy1 - 4 - (np.asarray(q, np.float64) / ADC_MAX) * (gy1 - gy0 - 24)
        yin = float(gy(val))
        dx = np.arange(gx0 + 4, gx1 - 6, 10.0)
        f.segments("r", dx, np.full_like(dx, yin), dx + 5, np.full_like(dx, yin), 0.8)
        f.text("r", gx1 - 2, yin - 6, "IN", size=L.T_MICRO, anchor="rs", alpha=0.9)
        sx = gx0 + 8 + np.repeat(np.arange(nb + 1), 2)[1:-1] * sw
        sy = np.repeat(gy(trials), 2)
        pr = min(1.0, max(ib, 0.0) / nb)
        if pr < 1.0:
            cx, cy, m = _pen_cut(sx, sy, pr)
            f.polyline("w", cx, cy, 1.0, width=L.LW)
            f.dots("w", cx[-1:], cy[-1:], 2.8, 1.6)
            if k > 0:                                   # the last staircase, until the pen gets there
                oacc, otr = 0, []
                for i in range(nb):
                    trial = oacc + (1 << (nb - 1 - i))
                    otr.append(trial)
                    if trial <= old:
                        oacc = trial
                oy = np.repeat(gy(otr), 2)
                f.polyline("w", sx[m:], oy[m:], E.wl(0.5), width=L.LW)
        else:
            f.polyline("w", sx, sy, E.wl(0.5 + 0.5 * math.exp(-(a - nb * T_BIT) / 0.8)), width=L.LW)

    def _small_bloom(self, f, ox, oy, open_, t, seed, n=7, rate=0.45, step=13.0, gain=1.0, clip=None):
        """A bloom in small, with the animation of the bloom on top of a tower (towers.loops): its loops stay in
        their place, the solid inner ones breathe, the dotted outer ones pump through them. open_ sets its size;
        rate, how fast it lives (0.45 = as a tower bloom)."""
        if clip:
            f.set_clip(*clip)
        reach = 1.25 * (6.0 + step * (n + 0.5))                 # rest size of its outer loop when open_ = 1
        k = max(0.35 + 1.1 * open_, 0.0) / 1.45
        towers.loops(f, ox, oy, t, n + 4, reach * k / towers.REST, reach / towers.REST, seed, gain,
                     rate=towers.BREATH * rate / 0.45, s_min=3.0, weight=0.7, m_min=(24, 14))
        f.dots("r", [ox], [oy - 2], 4.6, 0.9 * gain)
        if clip:
            f.set_clip()

    def _stage_light(self, f, t, x0, x1, y0, y1, k, a):
        """What the bloom is told. The bloom follows the level: it opens when the muon gets here, and closes
        again in 1.6 s; the brightness bar is that level, with a mark held on its peak."""
        lv = min(1.0, self._env(t, 2, 0.12, 1.6))
        e = self.demo[k][1] if k is not None else 0.0
        breath = 1.0 + 0.05 * math.sin(1.7 * t)
        self._small_bloom(f, x0 + 110, y1 - 6, (0.1 + lv) * breath, t, 900, gain=0.5 + 0.7 * lv,
                          clip=(x0, y0 - 6, x1, y1 + 4))
        bx0, bx1 = x0 + 235, x1 - 6
        if bx1 - bx0 < 60:
            return
        wb = bx1 - bx0 - 6
        chars = int((bx1 - bx0) / (L.T_SMALL * 0.61))
        f.text("w", bx0, y0 + 26, "BRIGHTNESS", size=L.T_MICRO, alpha=0.75)
        f.rect("w", bx0, y0 + 36, bx1, y0 + 58, E.wl(0.6), width=WH)
        tk = bx0 + 3 + wb * np.arange(5) / 4.0
        f.segments("w", tk, np.full(5, y0 + 58.0), tk, np.full(5, y0 + 64.0), E.wl(0.5), width=WH)
        hud.bars(f, "r", bx0 + 3, y0 + 39, bx0 + 3 + wb * lv, y0 + 55, 0.95)
        if k is not None and e > lv:                    # the peak, held, then let go
            pk = lv + (e - lv) * float(1.0 - smoothstep(1.4, 2.6, a))
            xk = bx0 + 3 + wb * min(1.0, pk)
            f.segments("w", [xk], [y0 + 33], [xk], [y0 + 61], 0.95, width=L.LW)
        radius = f"BLOOM RADIUS {int(120 + 420 * lv):3d} PX" if chars >= 23 else f"RADIUS {int(120 + 420 * lv):3d} PX"
        f.text("w", bx0, y0 + 92, radius[:chars], size=L.T_SMALL, alpha=0.9)
        if k is None:
            return
        with f.build(a, (bx0 - 6, y0 + 98, bx1 + 6, y1 + 6), key=42, wave=0.25, line=0.22, marks=False):
            lines = [f"OPEN {e:.2f}   DECAY 1.6 S"] if chars >= 23 else [f"OPEN {e:.2f}", "DECAY 1.6 S"]
            for j, ln in enumerate(lines):
                f.text("w", bx0, y0 + 117 + j * 25, ln[:chars], size=L.T_SMALL, alpha=0.7)

    def _stage_sound(self, f, t, x0, x1, y0, y1, k, a):
        """The note it triggers, louder with the energy: written as it sounds (a play head crosses it), then it
        keeps running while it dies down; a level meter with its peak held."""
        ym = (y0 + y1) / 2 + 4
        mx = x1 - 18.0                                  # the level meter, at the right end
        if not E.WALL:      # (the axis under the live trace: with the wall rule the trace itself is the line)
            f.segments("w", [x0], [ym], [mx - 12], [ym], 0.3)
        f.rect("w", mx, y0 + 26, x1, y1, E.wl(0.5), width=WH)
        mk = y1 - 3 - (y1 - y0 - 32) * np.arange(5) / 4.0
        f.segments("w", np.full(5, mx - 6.0), mk, np.full(5, mx), mk, E.wl(0.5), width=WH)
        tt = self._tw
        xs = x0 + 6 + tt * (mx - 18 - x0 - 6)
        u = xs[::3] * 0.09 + t * 6.0                    # the output, live: its noise floor runs to the left
        f.polyline("w", xs[::3], ym - 1.6 * (np.sin(u) + 0.7 * np.sin(2.7 * u + 1.1) + 0.5 * np.sin(6.3 * u + 0.6)), E.wl(0.4),
                   width=WH)
        if k is None:
            return
        hh = (y1 - y0) * 0.44
        shape = np.exp(-tt / 0.3) * (1 - np.exp(-tt / 0.006))

        def wave(km, ak):                               # the wave keeps running under its envelope, and dies down
            em = self.demo[km][1]
            return em * (0.2 + 0.8 * math.exp(-ak / 1.3)) * shape * np.sin(2 * np.pi * (26 + 20 * em) * tt - 11.0 * ak)

        e = self.demo[k][1]
        ys = ym - wave(k, a) * hh
        p = min(1.0, a / 0.5)
        br = 0.55 + 0.45 * math.exp(-a / 0.8)
        if p < 1.0:
            cx, cy, m = _pen_cut(xs, ys, p)
            f.polyline("w", cx, cy, 1.0 * br, width=L.LW)
            if k > 0:
                ao = t - self.demo[k - 1][0] - 3 * self.demo[k - 1][4]
                f.polyline("w", xs[m:], (ym - wave(k - 1, ao) * hh)[m:], E.wl(0.55), width=L.LW)
        else:
            f.polyline("w", xs, ys, 1.0 * br, width=L.LW)
        if a < 1.0:                                     # the play head: the plot is one second long
            hp = hh * float(1.0 - smoothstep(0.85, 1.0, a))
            xh = xs[0] + a * (xs[-1] - xs[0])
            f.segments("r", [xh], [ym - hp], [xh], [ym + hp], 1.0, width=L.LW)
        lvl = e * (1.0 - math.exp(-a / 0.02)) * math.exp(-a / 0.45)
        span = y1 - y0 - 32
        _vbar(f, "w", mx + 3, x1 - 3, y1 - 3 - span * lvl, y1 - 3, 0.9)
        pk = lvl + (e - lvl) * float(1.0 - smoothstep(0.9, 1.9, a)) * (1.0 - math.exp(-a / 0.02))
        f.segments("r" if a < 0.9 else "w", [mx - 4], [y1 - 3 - span * pk], [x1 + 4], [y1 - 3 - span * pk], 0.95, width=L.LW)
        with f.build(a, (x0 - 6, y0 - 6, mx - 6, y0 + 26), key=43, wave=0.25, line=0.22, marks=False):
            db = 20 * math.log10(max(e, 1e-3))
            f.text("w", mx - 12, y0 + 16, f"GAIN {db:+05.1f} DB   VEL {int(e * 127):03d}", size=L.T_SMALL,
                   anchor="rs", alpha=0.95)

    # ------------------------------------------------------------------ columns
    def _left(self, f, t, ctx):
        if self.roles["card"] is None:
            return
        c0, c1 = self.roles["card"]
        x = c0 + 4.0
        y = L.HEAD_Y + 86
        wide = c1 - c0 >= 380
        ts = int(min(62, (c1 - c0 - 30) / (9 * 0.61)))
        B.tag(f, "w", x, y, "DETECTORS", t - T0, t0=0.1, size=ts, pad=10, bold=True, cps=45.0, key=50, commit=True)
        waiting = t < sd.T_BLOOM
        rows = ["COUNT     3  (L / C / R)", "TYPE      PLASTIC SCINTILLATOR", "READOUT   SIPM x 2 // COINCIDENCE",
                "SIGNAL    1 FLOAT PER TOWER", "LINK      OSC  /MUON/L /C /R", "RATE      ~0.2 MUONS /S EACH",
                f"STATE     {'WAITING' if waiting else 'LIVE'}"]
        chars = int((c1 - c0 - 4) / (L.T_SMALL * 0.61))
        # the rows are decoded one after the other, top to bottom (they used to pop in, one every 0.3 s)
        with f.build(t - T0 - 0.35, (x - 8, y + 38, c1 + 4, y + 62 + 6 * 25.5 + 12), flow="tb", wave=1.4, cps=90.0, key=51):
            hud.rows(f, x, y + 62, [r[:chars] for r in rows], size=L.T_SMALL, red=(6,))
        if t >= self.c_every and wide:
            yy = y + 62 + 8 * 26 + 30
            a_log = t - self.c_every            # the log is made when the first simulated muon is announced
            B.tag(f, "w", x, yy, "REPLY_LOG // SIMULATION", a_log, size=L.T_MICRO, pad=3, cps=90.0, key=52)
            f.text("w", x, yy + 30, B.resolve("T        E     ADC   BLOOM  GAIN", a_log, 110.0, 0.15, key=53),
                   size=L.T_MICRO, alpha=0.55)
            done = [m for m in self.demo if m[0] <= t][::-1]
            for r, (tm, e, _, _, _) in enumerate(done):
                line = f"{sd.tc(tm)[:8]} {e:.2f}  {int(e * ADC_MAX):04d}  {int(120 + 420 * e):3d}   {20 * math.log10(e):+05.1f}"
                if r == 0:                  # the reply that just came in is decoded on top of the log
                    line = B.resolve(line, t - tm, 110.0, 0.1, key=54)
                f.text("r" if r == 0 and t - tm < 1.5 else "w", x, yy + 54 + r * 22, line, size=L.T_MICRO,
                       alpha=0.95 if r == 0 else 0.65)

    def _right(self, f, t, ctx):
        """Right outer bay: the three replies side by side - the stronger the muon, the larger the bloom and
        the louder the note. Once a muon has come it is replayed in its turn, every 1.6 s: its bloom opens,
        its meter jumps and falls back, a mark is held on its peak."""
        if self.roles["reps"] is None or t < self.c_more - 0.2:
            return
        x0, x1 = self.roles["reps"]
        x0 += 2.0
        y0 = L.HEAD_Y + 60
        ms = self.demo[1:4]
        w = (x1 - x0) / 3
        z = float(np.clip(w / 134.0, 0.8, 1.3))  # the block is drawn larger when its column is wide
        base = y0 + 60 + 270 * z
        mh = 150.0 * z                          # height of a loudness meter
        t_first, step = ms[0][0], ms[1][0] - ms[0][0]
        ages = []                               # seconds since each one last fired (it fires again every third step)
        for i in range(3):
            d = t - t_first - step * i
            ages.append(d - 3 * step * math.floor(d / (3 * step)) if d >= 0.0 else -1.0)
        # the frame of the comparison is constructed on "The more powerful the muon"; the blooms and the levels
        # are image: they come with their muon and live by themselves
        with f.build(t - (self.c_more - 0.2), (x0 - 8, y0 - 24, x1 + 4, base + mh + 94), key=55, wave=0.45):
            f.tag("w", x0, y0, "MORE ENERGY = BRIGHTER = LOUDER", size=L.T_MICRO, pad=3)
            f.segments("w", [x0], [base], [x1], [base], E.wl(0.6), width=WH)
            for i, (tm, e, _, _, _) in enumerate(ms):
                f.text("r" if 0.0 <= ages[i] < 0.6 else "w", x0 + (i + 0.5) * w, base + 28, f"E {e:.2f}", size=L.T_SMALL,
                       anchor="ms", alpha=0.9 if t >= tm else 0.35)
            f.text("w", x0, base + mh + 82, "LOUDNESS", size=L.T_MICRO, alpha=0.6)
            for i in range(3):                  # the three meters, empty until their muon comes
                bx, yb = x0 + (i + 0.5) * w - 16, base + 47 + mh
                f.rect("w", bx, base + 44, bx + 32, yb + 3, E.wl(0.5), width=WH)
                tk = yb - mh * np.arange(5) / 4.0
                f.segments("w", np.full(5, bx - 6.0), tk, np.full(5, bx), tk, E.wl(0.5), width=WH)
        m = (t - t_first) / step                # the mark under the one being replayed slides to the next
        if m >= 0.0:
            mi = math.floor(m)
            col = (mi % 3) + ((mi + 1) % 3 - mi % 3) * float(smoothstep(0.7, 1.0, m - mi))
            xm = x0 + (col + 0.5) * w
            hud.bars(f, "r", xm - 24, base + 34, xm + 24, base + 38, 1.0)
        for i, (tm, e, _, _, _) in enumerate(ms):
            a = ages[i]
            cx = x0 + (i + 0.5) * w
            if a < 0:
                continue
            born = 1 - math.exp(-(t - tm) / 0.3)
            lvb = (1.0 - math.exp(-a / 0.1)) * math.exp(-a / 1.5)       # the bloom of this muon: image, it lives by itself
            size = (0.3 + 1.25 * e) * (0.62 + 0.38 * lvb) * born
            self._small_bloom(f, cx, base - 4, (size - 0.35) / 1.1, t, 950 + i, n=5, rate=0.3 + 0.6 * e, step=11.5 * z,
                              gain=(0.55 + 0.65 * lvb) * born)
            lvl = e * (1.0 - math.exp(-a / 0.05)) * math.exp(-a / 0.7)
            pk = (lvl + (e - lvl) * float(1.0 - smoothstep(1.0, 2.2, a))) * (1.0 - math.exp(-a / 0.05))
            bx, yb = cx - 16, base + 47 + mh
            _vbar(f, "w", bx + 3, bx + 29, yb - mh * lvl, yb, 0.9)
            f.segments("r" if a < 1.0 else "w", [bx - 4], [yb - mh * pk], [bx + 36], [yb - mh * pk], 0.95, width=L.LW)

    def _bottom(self, f, t, ctx):
        panels = ctx.slots["panels"]
        y0, y1 = ctx.slots["y0"], ctx.slots["y1"]
        k, age = self._current(t)
        # the three panels are constructed at the cut, between the scopes of their towers (towers.scopes)
        ap = [t - T0 - lag for lag in (0.45, 0.65, 0.85)]
        if len(panels) > 0:
            x0, x1 = panels[0]
            flash = k is not None and age < 0.9
            incoming = t >= self.c_visitor
            word = "MUON" if flash else "INCOMING" if incoming else "WAITING"
            with f.build(ap[0], (x0 - 8, y0 - 24, x1 + 8, y1 + 8), key=60):
                hud.panel_header(f, x0, x1, y0, "STATE")
                if x1 - x0 >= 500 and not incoming:
                    f.text("w", x1, y0 + 44, f"T_WAIT {max(0.0, t - T0) if k is None else age:06.2f} S", size=L.T_SMALL,
                           anchor="rs", alpha=0.85)
                    f.text("w", x1, y0 + 70, "EXPECTED 1 EVERY ~5 S", size=L.T_MICRO, anchor="rs", alpha=0.6)
            # the state word is made again when a muon is caught (MUON) and when the visitor is announced
            # (INCOMING); WAITING simply resumes its blinking
            a_word = age if flash else t - self.c_visitor if incoming else 99.0
            if flash or incoming or int(t * 1.6) % 2 == 0 or t < self.c_wait:
                B.tag(f, "r" if flash or incoming else "w", x0 + 4, y0 + 84, word, min(ap[0] - 0.2, a_word), size=50, pad=8,
                      cps=55.0, key=61)
            if x1 - x0 >= 500 and incoming:
                with f.build(t - self.c_visitor, (x1 - 230, y0 + 24, x1 + 4, y0 + 78), key=62, wave=0.15, marks=False):
                    f.text("r", x1, y0 + 44, f"ETA {max(0.0, sd.T_BLOOM - t):05.2f} S", size=L.T_SMALL, anchor="rs")
                    f.text("w", x1, y0 + 70, "TARGET DET_C", size=L.T_MICRO, anchor="rs", alpha=0.7)
        count = panels[2] if len(panels) > 2 else None
        if len(panels) == 2 and panels[1][1] - panels[1][0] >= 520:      # no third panel: the count takes the end of the second
            count = (panels[1][1] - 110.0, panels[1][1])
            panels = [panels[0], (panels[1][0], count[0] - 26.0)]
        if len(panels) > 1:
            x0, x1 = panels[1]
            w = (x1 - x0) / 4
            with f.build(ap[1], (x0 - 8, y0 - 24, x1 + 8, y1 + 8), key=63):
                hud.panel_header(f, x0, x1, y0, "CHAIN // ENERGY > CODE > LIGHT > SOUND")
                for s, name in enumerate(STAGES):
                    xx = x0 + s * w
                    f.rect("w", xx + 4, y0 + 22, xx + w - 10, y0 + 70, E.wl(0.7), width=WH)
                    f.text("w", xx + 6, y0 + 98, name if w >= 96 else name[:1], size=L.T_SMALL,
                           alpha=0.9 if self._at(t, s)[0] is not None else 0.45)
            for s, name in enumerate(STAGES):          # a stage fills up when the muon reaches it: red, then white
                ks, a = self._at(t, s)
                xx = x0 + s * w
                if ks is not None:
                    with f.build(a, (xx + 4, y0 + 22, xx + w - 10, y0 + 70), key=64 + s, wave=0.1, flow="lr", marks=False):
                        lv = E.wl(0.5 + 0.5 * math.exp(-a / 0.8))
                        red = float(1.0 - smoothstep(1.0, 1.6, a))
                        if red > 0.0:
                            f.rects("r", xx + 8, y0 + 26, xx + w - 14, y0 + 66, lv * red)
                        if red < 1.0:
                            f.rects("w", xx + 8, y0 + 26, xx + w - 14, y0 + 66, lv * (1.0 - red))
        if count:
            x0, x1 = count
            with f.build(ap[2], (x0 - 8, y0 - 24, x1 + 8, y1 + 8), key=68):
                hud.panel_header(f, x0, x1, y0, "SIMULATED")
                f.text("w", x0 + 2, y0 + 92, f"{sum(1 for m in self.demo if m[0] <= t):02d}", size=54)

    # ------------------------------------------------------------------ the visitor
    def _visitor(self, f, t, ctx):
        """One muon on its way to the centre detector while the voice names it. It arrives on 'A bloom'."""
        if t < self.c_visitor:
            return
        tw = ctx.towers["C"]
        ox, oy = tw.det
        ang = math.radians(33.0)
        x_top = ox - math.tan(ang) * (oy - L.FY0)
        u = float(np.clip((t - self.c_visitor) / (sd.T_BLOOM - self.c_visitor), 0.0, 1.0)) ** 1.6
        hx, hy = x_top + (ox - x_top) * u, L.FY0 + (oy - L.FY0) * u
        f.set_clip(L.FX0, L.FY0, L.FX1, L.FY1)
        k = np.arange(0, 1.0, 0.04)                          # dashed line of where it will go
        with f.build(t - self.c_visitor, (min(x_top, ox) - 10, L.FY0, max(x_top, ox) + 10, oy), flow="tb", wave=0.5,
                     marks=False, key=70):
            f.segments("r", x_top + (ox - x_top) * k, L.FY0 + (oy - L.FY0) * k, x_top + (ox - x_top) * (k + 0.02),
                       L.FY0 + (oy - L.FY0) * (k + 0.02), 0.45)
        f.segments("r", [x_top], [L.FY0], [hx], [hy], 1.2, width=L.LW_BOLD)
        f.dots("r", [hx], [hy], 6.0, 1.8)
        f.dots("w", [hx], [hy], 2.2, 1.2)
        f.rings("r", [hx], [hy], [14 + 10 * math.sin(t * 9.0) ** 2], 0.8, width=L.LW)
        f.set_clip()
        for tc, word, uu in ((self.c_visitor, "A VISITOR", 0.2), (self.c_messenger, "A MESSENGER", 0.48),
                             (self.c_muon, "A MUON", 0.76)):
            if t < tc:
                continue
            px, py = x_top + (ox - x_top) * uu, L.FY0 + (oy - L.FY0) * uu
            now = t < (self.c_messenger if word == "A VISITOR" else self.c_muon if word == "A MESSENGER" else 1e9)
            hud.callout(f, px, py, -70, 26 if uu < 0.3 else 34, word, red=now, alpha=1.0 if now else 0.55,
                        size=L.T_TAG, side=-1, build=t - tc)
