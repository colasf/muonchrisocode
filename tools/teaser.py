"""Teaser for MUON : BLOOM, cut on the sound of the show itself (30 fps). One edit, two formats:

  instagram   1080 x 1350 (4:5)       the default
  linkedin    1920 x 1080 (16:9)      the same cuts on the same sound; every shot framed again for the wide frame

The sound is the mix of the show from 06:54.5: the big muon of the centre detector, the last hits of the drums,
the break of scene 7 and the drums coming back. What is seen follows what is heard (times measured on the
sound, not taken from the cue list alone: its times are about 30 ms early, and it also lists hits of the
silence):

  06:54.5  the muon, the last hits     six shots of the show, one per hit (the first frame is the star)
  06:55.4  last kick, then a bass note black: the red dot, its two lines, the lattice of crosses; the date is typed
  06:59.0  silence                     the GPS position is typed; "almost nothing." (the voice, 07:00.0)
  07:01.6  a muon, two soft hits       a red ring leaves the dot on each
  07:02.7  the drums                   fast cuts: a new shot on every hit, 2 - 3 frames each
  07:08.3  the music stops for 0.14 s  black
  07:08.5  the downbeat                the Tyrell logo, white on red, on the diagonal; the sound stops there
                                       (linkedin: the name CHRISTO SQUIER here, the same way, and
  07:09.9  the kick one bar later      the logo)

How the extracts are framed (SHOTS):
  - one axis for the whole teaser: the subject of every shot stands on the vertical centre line, where the
    crosshair of the first sequence is; what is symmetric (rings, star, sphere, card) is centred on the dot
    itself, what stands on the ground (a bloom on its tower, the column of light) has its head on the upper
    third and its ground in the lower third;
  - the subject is whole, with room around it; the header strip and the bottom band of the wall stay out;
  - no cut text: a text or a tag is drawn only when it lies whole inside the frame and clear of its edge
    (ShotFrame); the frame, the edge ticks, the counter cell, the scopes and the subtitles are not drawn.

The wide frame (WIDE) keeps the axis and the rules above, and uses its width as the wall does: the wall is laid
out as two bays around the centre tower, so a shot is either one subject on the dot (what stands beside it is
kept whole or masked, never sliced by the edge) or the two bays with the centre tower on the axis.

  python tools/teaser.py sheet [linkedin] [name@time ...]   every shot as framed -> previews/teaser/shots[_linkedin]
  python tools/teaser.py video [linkedin] [name]            the teaser -> previews/teaser/
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math
import struct
import subprocess
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "muonbloom").is_dir() else Path(r"D:\muonchristo\claude")
sys.path.insert(0, str(ROOT))

import preview  # noqa: E402  (the audio files, ffmpeg)
from muonbloom import layout as L  # noqa: E402
from muonbloom.engine import Frame, font  # noqa: E402

OUT = ROOT / "previews" / "teaser"
LOGO = Path(r"D:\muonchristo\logo\Tyrell_Logo_DEF.eps")

FPS = 30
W, H = 1080, 1350                                # the frame: set by use() (FORMATS)
DATE = ("2026.10.08", "2026.10.11")              # BLINK 2026
GPS = ("39.105381 N", "84.514042 W")             # 9th St x Vine St (tools/build_city.py: SITE)
WHITE, RED, GREY = (255, 255, 255), (255, 11, 9), (150, 150, 150)       # the red of the show

# ---- the sound (show time, seconds) ---------------------------------------------------------------------------
A0 = 414.475                                     # frame 0: the big muon of the centre detector
LAG = 0.03                                       # data/cues.npz is about 30 ms early against the sound
T_BLACK = 415.38                                 # the last kick before the break
T_GPS = 419.0
T_VOICE = 420.005                                # "almost nothing."
T_RINGS = (421.555, 421.96, 422.165)             # a muon, then two soft hits
T_DROP = 422.685                                 # the drums
T_GAP = 428.31                                   # the music stops ...
T_DOWN, T_BAR = 428.455, 429.895                 # ... and comes back on the downbeat; the kick one bar later
T_NAME, T_LOGO = None, T_DOWN                    # the logo on the downbeat; in a format with the name (use()):
#                                                  the name on the downbeat, the logo one bar later
NAME = "CHRISTO SQUIER"
HOLD_LOGO = 1.4                                  # s of logo (the sound is out after 0.3 s)
MAX_HOLD = 3                                     # frames: in the drums a shot never stays longer
RING_SPEED = 1300.0                              # px / s

# ---- the shots ---------------------------------------------------------------------------------------------------
# name: (show time, x, y, scale, options). The point (x, y) of the 2978 x 1400 frame of the show is put on the
# centre of the teaser frame, at `scale` teaser pixels per pixel of the show. Options: erase = boxes of the show
# frame painted black (the header strip, the bottom band, a tower), text = False for no text at all.
HEAD = (0, 0, 2978, 235)                         # the score strip
FOOT = (0, 1190, 2978, 1400)                     # the bottom band: rules, counters, barcodes
LOW = (0, 1140, 2978, 1400)                      # ... and the glow of its rule, where nothing else is that low
FX, FY = 989, 705                                # layout.Context.focus: the centre of the one-centre scenes
CX = 1489                                        # the axis of the centre tower
MARGIN = 30                                      # px of the teaser: a text closer to the edge is left out (use())

SHOTS = {
    # the opening, all on the focus: the red dot stays where the dot of the cards is
    "rings": (13.0, FX, FY, 1.5), "spiral": (26.0, FX, FY, 1.5), "star": (32.6, FX, FY, 1.6),
    "rays": (42.5, FX, FY, 1.41), "messenger": (63.5, 993, 610, 1.75),
    "cone": (70.2, 981, 733, 1.66), "cone_red": (70.2, 2207, 743, 2.5),
    "figure": (80.7, 987, 741, 1.32, {"erase": ((0, 1165, 2978, 1400),)}), "heart": (86.0, 1990, 730, 1.42),
    "hex": (91.8, 1005, 715, 1.6), "muon": (102.2, 987, 716, 1.08),
    "plate": (118.0, 991, 654, 1.3, {"erase": (HEAD,)}), "detector": (126.0, 1037, 658, 1.58),
    "blooms": (178.0, CX, 650, 1.25), "galaxy": (210.5, CX, 713, 1.4),
    "sphere": (256.0, 989, 728, 1.3, {"erase": (HEAD, LOW)}),
    "flood": (286.0, 990, 725, 1.3, {"erase": (HEAD, (0, 1150, 2978, 1400))}),
    "count": (290.0, 1992, 665, 1.08, {"erase": (LOW, (0, 0, 1530, 1400), (2452, 0, 2978, 1400))}),
    "shower": (341.6, 1010, 707, 1.4), "city": (358.0, CX, 802, 1.72, {"text": False}),
    "grid": (504.0, 1938, 679, 1.3, {"erase": ((0, 0, 1530, 1400),)}), "rise": (550.0, 1490, 551, 1.5), "column": (562.0, CX, 630, 1.25),
    "decay": (576.0, 989, 728, 1.3, {"erase": (HEAD, LOW)}),
    "whirl_a": (686.5, FX, FY, 1.5), "whirl_b": (704.5, FX, FY, 1.65),
}
# the opening: (hit, shot). One shot per hit, held until the next one.
HOOK = ((A0, "star"), (414.66, "spiral"), (414.78, "flood"), (414.90, "rise"), (415.02, "blooms"),
        (415.22, "galaxy"))
FIRST = "muon"                                   # the shot on the first hit of the drums

# ---- the same shots in the wide frame (16:9) --------------------------------------------------------------------
# name: (x, y, scale, options): the window of the shot in the wide frame. Four more options:
#   t    = another moment of the scene than the one of SHOTS, where the wide window shows what the tall one left
#          out: a figure still spinning (the card of the muon), a label half written (the city);
#   mask = boxes of the show frame blacked out BEFORE the glow (what stands beside the subject and would be
#          sliced by the edge of the frame: a column of text, a neighbouring panel), so no light is left of them;
#   hide = boxes whose texts are not drawn and where no plate is laid: the picture runs through;
#   quiet = towers whose bloom is not drawn (the tower is masked: its loops would hang in the air beside it).
# The wall is two bays around the centre tower (x 519 .. 1459 / 1519 .. 2459). A shot is one bay with its subject
# on the dot; or both bays, the centre tower on the axis; or the picture from edge to edge, cut where the wall
# itself ends it (the rule of the score strip at y 221, the bottom band at y 1190): no flat cut inside the frame.
TOP = (0, 0, 2978, 221)                                      # the score strip, to its rule
BAY_L = ((0, 0, 532, 1400), (1446, 0, 2978, 1400))           # all but the bay left of the centre tower
BAY_R = ((0, 0, 1532, 1400), (2446, 0, 2978, 1400))          # all but the bay right of it
ENDS = ((0, 0, 532, 1400), (2446, 0, 2978, 1400))            # what stands beyond the outer towers
COLS = ((0, 0, 440, 1400), (2540, 0, 2978, 1400))            # the columns of text at both ends of the wall
BAND = ((0, 1190, 455, 1400), (523, 1190, 1455, 1400), (1523, 1190, 2455, 1400), (2523, 1190, 2978, 1400))
#                                                              the bottom band, the three towers left standing
BALL = (BAY_L[0], (1435, 0, 2978, 1400), (0, 1180, 2978, 1400), (1385, 420, 1436, 700))
#                                                              the view of the sphere: its bay, less the bloom of
#                                                              the centre tower that reaches into it

WIDE = {
    # the opening, on the focus as in the tall frame; HERE / RIGHT NOW stands on the axis (its two lines of
    # coordinates are those of the city, not of the corner the card gives: left out)
    "rings": (FX, FY, 1.12, {"hide": ((1560, 575, 1900, 646),)}), "spiral": (FX, FY, 1.2),
    "star": (FX, FY, 1.5), "rays": (FX, FY, 2.5),
    "messenger": (993, 610, 1.0, {"mask": (TOP,)}),
    "cone": (989, 735, 1.65, {"mask": BAY_L}), "cone_red": (1990, 735, 1.9),            # the two cones side by side
    "figure": (CX, 710, 0.96, {"mask": (TOP, FOOT)}), "heart": (1990, 730, 1.25, {"mask": BAY_R}),
    "hex": (989.5, 584.5, 2.04),                                                       # the bay from edge to edge
    "muon": (CX, 712, 0.96, {"t": 102.6}),                                             # both cards, every figure locked
    "plate": (991, 690, 1.15, {"mask": (TOP, (0, 0, 545, 1400), (1436, 0, 2978, 1400))}),
    "detector": (1037, 658, 1.7, {"mask": (BAY_L[0], (1548, 0, 2978, 1400))}),
    "blooms": (CX, 700, 0.72, {"hide": ((0, 0, 700, 235),), "mask": BAND}),            # the three towers
    "galaxy": (CX, 705, 1.115),                                                        # the view, strip to band
    "sphere": (989, 726.5, 1.117, {"mask": BALL}),
    "flood": (990, 745, 1.15, {"mask": BAY_L + (FOOT,)}),
    "count": (CX, 700, 0.96, {"mask": (TOP,) + BAND + ENDS}),
    "shower": (989, 815, 1.65, {"mask": COLS[:1]}),                                    # from tower to tower
    "city": (CX, 705, 1.115, {"t": 359.0}),                                            # the labels are written
    "grid": (CX, 700, 0.96, {"mask": (TOP,) + BAND + ENDS, "quiet": "LR"}),
    "rise": (CX, 539, 1.32),                                                           # the red line on the upper third
    "column": (CX, 698, 0.86, {"mask": COLS + BAND}),                                  # the three columns of light
    "decay": (989, 726.5, 1.117, {"mask": BALL}),
    "whirl_a": (FX, FY, 1.2), "whirl_b": (CX, 666, 1.03),
}

# ---- the formats -------------------------------------------------------------------------------------------------
# size; shots = the table of the framings; grid = the pitch of the lattice of the cards; text = (x, the two
# baselines, size) of the typed lines; logo = (its length on the diagonal, what is centred: the artboard, or
# the ink - the outline is not centred on its artboard, and on the flat diagonal of the wide frame it shows);
# margin = how far a text stays from the edge; dark = False: an unlit tower is not drawn (in the show it is a
# black band over the picture: it masks the scaffold tower that stands there - in the wide frame it would cut
# the picture in two).
# The wide card: the same lattice and the same type; the dot and its two lines keep the centre, the typed lines
# stand in the lower right quarter, one column off the axis, on the second and third row under it.
FORMATS = {
    "instagram": dict(size=(1080, 1350), shots=SHOTS, grid=120, text=(180, (915, 1035), 112),
                      logo=(1150.0, "artboard"), margin=30),
    "linkedin": dict(size=(1920, 1080), shots=WIDE, grid=120, text=(1080, (780, 900), 112),
                     logo=(1700.0, "ink"), margin=40, dark=False, name=True),
}
FORMAT = "instagram"


def fr(t):
    """Show time -> frame of the teaser."""
    return int(round((t - A0) * FPS))


# ----------------------------------------------------------------------------
# a shot: the show rendered large, a window of it kept
# ----------------------------------------------------------------------------

class ShotFrame(Frame):
    """The frame of the show, knowing which window of it the teaser keeps (pixels of this render): a text or a
    tag is drawn only when it lies whole inside the window, clear of its edge and of the boxes painted black.
    What would be cut, or nearly touch the edge, is left out (and noted in `cut`)."""
    window = None
    margin = 0.0
    text_on = True
    holes = ()                                   # the boxes no text may reach into (pixels of this render)
    masks = ()                                   # the boxes blacked out before the glow
    bare = ()                                    # the boxes where texts are hidden: no plate is laid there either
    quiet = ()                                   # the towers whose bloom is not drawn
    cut = []

    def _whole(self, s, l, t, r, b):
        w = self.window
        if w is None:
            return True
        if r <= w[0] or l >= w[2] or b <= w[1] or t >= w[3]:      # outside: not seen
            return False
        m = self.margin
        if (self.text_on and l >= w[0] + m and t >= w[1] + m and r <= w[2] - m and b <= w[3] - m
                and not any(l < h[2] and r > h[0] and t < h[3] and b > h[1] for h in self.holes)):
            return True
        ShotFrame.cut.append(s)
        return False

    def _font(self, size, bold=False):
        return font(max(6, int(round(size * self.s))), bold)

    def _text(self, layer, x, y, s, size=22, alpha=1.0, anchor="ls", bold=False):
        if self.window is not None and s and alpha > 0.004:
            box = self._text_layer(layer)[1].textbbox((float(self.tx(x)), float(self.ty(y))), s,
                                                      font=self._font(size, bold), anchor=anchor)
            if not self._whole(s, *box):
                return
        super()._text(layer, x, y, s, size, alpha, anchor, bold)

    def _tag(self, layer, x, y, s, size=18, alpha=1.0, anchor="ls", pad=5, bold=False, ref=None, wipe=1.0):
        if self.window is not None and (s or ref) and alpha > 0.004 and wipe > 0.0:
            l, t, r, b = self._text_layer(layer)[1].textbbox((float(self.tx(x)), float(self.ty(y))), ref or s,
                                                             font=self._font(size, bold), anchor=anchor)
            p = pad * self.s
            if not self._whole(ref or s, l - p, t - p, r + p, b + p):       # the callers lay out from the box
                return tuple(v / self.s for v in (l - p, t - p, l - p + (r - l + 2 * p) * min(wipe, 1.0), b + p))
        return super()._tag(layer, x, y, s, size, alpha, anchor, pad, bold, ref, wipe)

    def text_vertical(self, layer, x, y, s, size=22, alpha=1.0):
        if self.window is not None and s:
            l, t, r, b = self._font(size).getbbox(s)
            X, Y = float(self.tx(x)), float(self.ty(y))
            if not self._whole(s, X, Y - (r - l + 2), X + (b - t + 2), Y):
                return
        super().text_vertical(layer, x, y, s, size, alpha)

    def occlude(self, x0, y0, x1, y1):
        if not self.bare or self.muted:
            return super().occlude(x0, y0, x1, y1)
        self.flush()                             # what is drawn in a bare box so far comes back after the plate
        kept = [(b, [a.reshape(self.H, self.W)[b[1]:b[3], b[0]:b[2]].copy() for a in self.acc.values()],
                 [img.crop(b) for img in self._txt.values()]) for b in self.bare]
        super().occlude(x0, y0, x1, y1)
        for b, accs, imgs in kept:
            for a, c in zip(self.acc.values(), accs):
                a.reshape(self.H, self.W)[b[1]:b[3], b[0]:b[2]] = c
            for img, c in zip(self._txt.values(), imgs):
                img.paste(c, b[:2])

    def finish(self, *a, **k):
        if self.masks:                           # no light is left in a mask: nothing glows out of it
            self.flush()
            for x0, y0, x1, y1 in self.masks:
                for acc in self.acc.values():
                    acc.reshape(self.H, self.W)[y0:y1, x0:x1] = 0.0
                for img in self._txt.values():
                    ImageDraw.Draw(img).rectangle((x0, y0, x1 - 1, y1 - 1), fill=0)
        return super().finish(*a, **k)


_show = None


def use(name):
    """Set the format: the frame, the layout of the cards, the framings of the shots."""
    global FORMAT, W, H, DOT, GRID, TEXT_X, TEXT_Y, TEXT_SIZE, MARGIN, T_NAME, T_LOGO
    fmt = FORMATS[name]
    assert list(fmt["shots"]) == list(SHOTS), "the formats share one edit: the same shots in the same order"
    FORMAT, (W, H), GRID, MARGIN = name, fmt["size"], fmt["grid"], fmt["margin"]
    DOT = (W // 2, H // 2)
    TEXT_X, TEXT_Y, TEXT_SIZE = fmt["text"]
    T_NAME, T_LOGO = (T_DOWN, T_BAR) if fmt.get("name") else (None, T_DOWN)


def _init(fmt="instagram"):
    """One Show per process, drawing the scene and its towers only."""
    global _show
    from muonbloom import hud, show, towers
    use(fmt)
    hud.frame = hud.edge_ticks = hud.through_you_cell = towers.scopes = lambda *a, **k: None
    if not FORMATS[fmt].get("dark", True):
        towers.dark = lambda *a, **k: None
    burst = towers.burst
    towers.burst = lambda f, tw, *a, **k: None if tw.key in ShotFrame.quiet else burst(f, tw, *a, **k)
    show.Frame = ShotFrame
    _show = show.Show()


def shot_spec(name):
    """(show time, x, y, scale, options) of a shot in the format in use."""
    t, x, y, s, *opt = SHOTS[name]
    if FORMATS[FORMAT]["shots"] is not SHOTS:
        x, y, s, *opt = FORMATS[FORMAT]["shots"][name]
    opt = opt[0] if opt else {}
    return opt.get("t", t), x, y, s, opt


def _boxes(boxes, kx, ky):
    """Boxes of the show frame -> pixels of a render."""
    return [(max(int(a * kx), 0), max(int(b * ky), 0), int(math.ceil(c * kx)), int(math.ceil(d * ky)))
            for a, b, c, d in boxes]


def render_shot(name, t, out_w=None, out_h=None):
    """The shot at show time t, out_w x out_h; and the texts the frame left out."""
    out_w, out_h = out_w or W, out_h or H
    _, x, y, s, opt = shot_spec(name)
    k = s * out_h / H
    rw, rh = int(round(L.W * k)) // 2 * 2, int(round(L.H * k)) // 2 * 2
    kx, ky = rw / L.W, rh / L.H
    x0, y0 = int(round(x * kx - out_w / 2)), int(round(y * ky - out_h / 2))
    ShotFrame.window, ShotFrame.margin = (x0, y0, x0 + out_w, y0 + out_h), MARGIN * out_h / H
    ShotFrame.text_on, ShotFrame.cut, ShotFrame.quiet = opt.get("text", True), [], opt.get("quiet", ())
    holes = _boxes(opt.get("erase", ()), kx, ky)                   # painted black afterwards
    ShotFrame.masks = _boxes(opt.get("mask", ()), kx, ky)
    ShotFrame.bare = [(a, b, min(c, rw), min(d, rh)) for a, b, c, d in _boxes(opt.get("hide", ()), kx, ky)]
    ShotFrame.holes = holes + ShotFrame.masks + ShotFrame.bare      # a text that reaches into one is left out whole
    img = _show.render(t, rw, rh, subtitles=False)
    if holes:
        img = np.array(img)
        for hx0, hy0, hx1, hy1 in holes:
            img[hy0:hy1, hx0:hx1] = 0
    out = np.zeros((out_h, out_w, 3), np.uint8)                   # black where the window leaves the wall
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + out_w, rw), min(y0 + out_h, rh)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out, list(ShotFrame.cut)


# ----------------------------------------------------------------------------
# the black card: dot, lines, lattice, date, position, rings
# ----------------------------------------------------------------------------

S = 2                                            # the cards are drawn twice the size and reduced: smooth edges
DOT = (W // 2, H // 2)                           # the centre: where the subject of every shot is put
GRID = 120                                       # the lattice; the text stands on it          (all four: use())
TEXT_X, TEXT_Y, TEXT_SIZE = 180, (915, 1035), 112


def _box(d, x0, y0, x1, y1, fill):
    d.rectangle([S * x0, S * y0, S * x1 - 1, S * y1 - 1], fill=fill)


def card(f):
    if f >= fr(T_LOGO):
        return logo_card()
    if T_NAME is not None and f >= fr(T_NAME):
        return name_card()
    if False:
        return logo_card()
    if f >= fr(T_GAP):
        return np.zeros((H, W, 3), np.uint8)
    v = f / FPS
    v0 = fr(T_BLACK) / FPS
    img = Image.new("RGB", (S * W, S * H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    x, y = DOT
    # the lattice of small crosses of the opening, written row by row, each row from left to right
    cols = [cx for cx in range(x % GRID, W, GRID) if 10 <= cx <= W - 10]      # no cross is cut by the edge
    rows = [cy for cy in range(y % GRID, H, GRID) if 10 <= cy <= H - 10]
    for r, cy in enumerate(rows):
        k = float(np.clip((v - v0 - 0.1 - r * 0.05) / 0.22, 0, 1))
        for c, cx in enumerate(cols):
            if (c + 1) / len(cols) <= k + 1e-6 and (cx, cy) != DOT:
                _box(d, cx - 9, cy - 1, cx + 10, cy + 1, GREY)
                _box(d, cx - 1, cy - 9, cx + 1, cy + 10, GREY)
    # the position: two lines drawn out of the dot
    k = float(np.clip((v - v0 - 0.05) / 0.5, 0, 1))
    k = k * k * (3 - 2 * k)
    if k > 0:
        _box(d, x - k * (x - 60), y - 1, x + k * (W - 60 - x), y + 1, WHITE)
        _box(d, x - 1, y - k * (y - 60), x + 1, y + k * (H - 60 - y), WHITE)
    # the text: typed character by character, a red block as cursor; a line hides the furniture behind it
    if v < fr(T_GPS) / FPS:
        lines = [(s, v - v0 - 0.85 - 0.85 * k) for k, s in enumerate(DATE)]
    else:
        lines = [(GPS[0], v - fr(T_GPS) / FPS), (GPS[1], v - fr(T_VOICE) / FPS)]
    fnt = font(S * TEXT_SIZE, True)
    cw = fnt.getlength("0") / S
    typed = []
    for (s, age), ty in zip(lines, TEXT_Y):
        n = int(np.clip(age * 15.0, 0, len(s)))
        if n:
            _box(d, TEXT_X - 16, ty - TEXT_SIZE * 0.86, TEXT_X + n * cw + 16, ty + TEXT_SIZE * 0.3, (0, 0, 0))
        typed.append((s, n, ty, age))
    # a ring leaves the dot on each hit before the drums (it passes under the text)
    for t in T_RINGS:
        r = RING_SPEED * (f - math.floor((t - A0) * FPS) + 1) / FPS      # born on the frame of the hit
        if 0 < r < max(1100.0, math.hypot(W, H) / 2 + 6):
            d.ellipse([S * (x - r), S * (y - r), S * (x + r), S * (y + r)], outline=RED, width=S * 3)
    d.ellipse([S * (x - 13), S * (y - 13), S * (x + 13) - 1, S * (y + 13) - 1], fill=RED)
    for s, n, ty, age in typed:
        if n:
            d.text((S * TEXT_X, S * ty), s[:n], font=fnt, fill=WHITE, anchor="ls")
        if 0 <= age < len(s) / 15.0 + 0.25:
            cx = TEXT_X + n * cw
            _box(d, cx + 4, ty - TEXT_SIZE * 0.72, cx + cw - 4, ty + 2, RED)
    return np.asarray(img.resize((W, H), Image.BOX))


# ----------------------------------------------------------------------------
# the logo
# ----------------------------------------------------------------------------

def _logo_paths():
    """The filled outlines of the logo, read from the Illustrator EPS (385.4 x 120.4 pt, y down)."""
    b = LOGO.read_bytes()
    o, n = struct.unpack("<2I", b[4:12])
    t = b[o:o + n].decode("latin1")
    page = t[t.index("clp", t.index("%%EndPageSetup")) + 3:t.index("%ADOBeginClientInjection: EndPageContent")]
    paths, cur, st = [], [], []
    for tok in page.split():
        try:
            st.append(float(tok))
            continue
        except ValueError:
            pass
        if tok in ("mo", "li"):
            cur.append((st[-2], st[-1]))
        elif tok == "cv":
            p0, (x1, y1, x2, y2, x3, y3) = cur[-1], st[-6:]
            for u in np.linspace(0, 1, 13)[1:]:
                a, b_, c, e = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u * u * (1 - u), u ** 3
                cur.append((a * p0[0] + b_ * x1 + c * x2 + e * x3, a * p0[1] + b_ * y1 + c * y2 + e * y3))
        elif tok == "cp":
            paths.append(cur)
            cur = []
        st = []
    return paths


_logo_cache = {}


def logo_card():
    """White on red, set on the diagonal of the frame (reading upwards), as long as the format says."""
    length, centred = FORMATS[FORMAT]["logo"]
    if FORMAT not in _logo_cache:
        ss = 4
        th = math.atan2(H, W)
        co, si = math.cos(th), math.sin(th)
        k = length * ss / 385.4
        mask = Image.new("L", (W * ss, H * ss), 0)
        d = ImageDraw.Draw(mask)
        polys = []
        for path in _logo_paths():
            pts = []
            for px, py in path:
                dx, dy = (px - 385.4 / 2) * k, (py - 120.362 / 2) * k
                pts.append((W * ss / 2 + dx * co + dy * si, H * ss / 2 - dx * si + dy * co))
            polys.append(pts)
        if centred == "ink":
            xs, ys = [q[0] for pts in polys for q in pts], [q[1] for pts in polys for q in pts]
            ox, oy = (W * ss - min(xs) - max(xs)) / 2, (H * ss - min(ys) - max(ys)) / 2
            polys = [[(x + ox, y + oy) for x, y in pts] for pts in polys]
        for pts in polys:
            d.polygon(pts, fill=255)
        m = np.asarray(mask.resize((W, H), Image.LANCZOS), np.float32)[..., None] / 255.0
        red, white = np.array(RED, np.float32), np.array(WHITE, np.float32)
        _logo_cache[FORMAT] = np.clip(red + (white - red) * m + 0.5, 0, 255).astype(np.uint8)
    return _logo_cache[FORMAT]


def name_card():
    """The name, white on red, on the same diagonal as the logo that follows, a little shorter."""
    if ("name", FORMAT) not in _logo_cache:
        ss = 2
        length = 0.9 * FORMATS[FORMAT]["logo"][0] * ss
        fnt = font(200, True)
        size = int(200 * length / fnt.getlength(NAME))
        fnt = font(size, True)
        l, t, r, b = fnt.getbbox(NAME)
        txt = Image.new("L", (r - l + 8, b - t + 8), 0)
        ImageDraw.Draw(txt).text((4 - l, 4 - t), NAME, font=fnt, fill=255)
        txt = txt.rotate(math.degrees(math.atan2(H, W)), expand=True, resample=Image.BICUBIC)
        mask = Image.new("L", (W * ss, H * ss), 0)
        mask.paste(txt, ((W * ss - txt.size[0]) // 2, (H * ss - txt.size[1]) // 2))
        m = np.asarray(mask.resize((W, H), Image.LANCZOS), np.float32)[..., None] / 255.0
        red, white = np.array(RED, np.float32), np.array(WHITE, np.float32)
        _logo_cache[("name", FORMAT)] = np.clip(red + (white - red) * m + 0.5, 0, 255).astype(np.uint8)
    return _logo_cache[("name", FORMAT)]


# ----------------------------------------------------------------------------
# the cuts
# ----------------------------------------------------------------------------

def cut_frames():
    """The frames of the cuts in the drums: every hit of the music, and the longer gaps split."""
    c = np.load(ROOT / "data" / "cues.npz")
    f0, f1 = fr(T_DROP), fr(T_GAP)
    out = [f0]
    for t in np.sort(np.concatenate([c["kick_t"], c["onset_t"]])) + LAG:
        f = fr(t)
        if f - out[-1] >= 2 and f <= f1 - 2:
            out.append(f)
    fine = []
    for a, b in zip(out, out[1:] + [f1]):
        n = -(-(b - a) // MAX_HOLD)
        fine += [a + (b - a) * k // n for k in range(n)]
    return fine


def plan():
    """One entry per frame: None (a card) or (show time, shot). A shot always starts at its own show time: it
    was framed for that moment."""
    n = fr(T_LOGO) + int(round(HOLD_LOGO * FPS))
    jobs = [None] * n

    def put(f0, f1, name):
        t0 = shot_spec(name)[0]
        for f in range(f0, f1):
            jobs[f] = (t0 + (f - f0) / FPS, name)

    hook = [fr(t) for t, _ in HOOK] + [fr(T_BLACK)]
    for (_, name), a, b in zip(HOOK, hook, hook[1:]):
        put(a, b, name)
    cuts = cut_frames()
    rng = np.random.default_rng(7)
    order, names = [FIRST], list(SHOTS)
    while len(order) < len(cuts):
        p = [names[i] for i in rng.permutation(len(names))]
        if p[0] == order[-1]:
            p = p[1:] + p[:1]
        order += p
    for k, a in enumerate(cuts):
        put(a, cuts[k + 1] if k + 1 < len(cuts) else fr(T_GAP), order[k])
    return jobs, cuts


def _render(job):
    f, j = job
    try:
        if j is None:
            return card(f), None, []
        img, cut = render_shot(j[1], j[0])
        return img, None, cut
    except Exception:
        return None, traceback.format_exc(), []


def sheet(workers, names=None):
    names = names or list(SHOTS)                 # name@time: the same framing at another moment
    at = {n: (float(n.split("@")[1]) if "@" in n else shot_spec(n)[0]) for n in names}
    with Pool(min(workers, len(names)), initializer=_init, initargs=(FORMAT,)) as pool:
        res = pool.map(_render, [(0, (at[n], n.split("@")[0])) for n in names])
    d0 = OUT / ("shots" if FORMAT == "instagram" else f"shots_{FORMAT}")
    d0.mkdir(parents=True, exist_ok=True)
    tw, th = W // 2, H // 2
    for s in range(0, len(names), 6):
        out = Image.new("RGB", (3 * tw + 8, 2 * (th + 24)), (40, 40, 40))
        d = ImageDraw.Draw(out)
        for k, (name, (img, err, cut)) in enumerate(zip(names[s:s + 6], res[s:s + 6])):
            if err:
                print(name, err)
                continue
            Image.fromarray(img).save(d0 / f"{name.replace('@', '_')}.png")
            base = name.split("@")[0]
            x, y = (k % 3) * (tw + 4), (k // 3) * (th + 24)
            out.paste(Image.fromarray(img).resize((tw, th), Image.LANCZOS), (x, y))
            d.text((x + 4, y + th + 4), f"{name}  {at[name]}  x{shot_spec(base)[3]}  texts left out: {len(cut)}",
                   font=font(14), fill=(255, 200, 0))
            print(f"{name}: left out {len(cut)}: " + " | ".join(str(c)[:28] for c in cut[:10]))
        p = d0 / f"sheet_{'try_' if '@' in ''.join(names) else ''}{s // 6}.jpg"
        out.save(p, quality=90)
        print(p)


def video(workers, name):
    jobs, cuts = plan()
    n = len(jobs)
    dur = n / FPS
    print(f"{n} frames, {dur:.2f} s, {len(HOOK)} + {len(cuts)} cuts", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    mp4 = OUT / name
    out_at, out_d = fr(T_LOGO) / FPS + 0.07, 0.22    # the downbeat is heard, then the sound is taken out
    if FORMATS[FORMAT].get("name"):              # ... with the name: the music plays under the logo to the end
        out_at, out_d = dur - 0.6, 0.6
    # each stem alone reaches full scale: their sum goes 4 dB over it on the big muon, so it is limited (-1 dB)
    graph = ("[0:v]scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p,"
             "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709[v];"
             "[1:a][2:a]amix=inputs=2:normalize=0,alimiter=limit=0.89:attack=3:release=80:level=false:latency=true,"
             f"afade=t=in:d=0.008,afade=t=out:st={out_at:.3f}:d={out_d:.2f}[a]")
    cmd = [preview.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-ss", f"{A0}", "-t", f"{dur}", "-i", str(preview.MUSIC),
           "-ss", f"{A0}", "-t", f"{dur}", "-i", str(preview.MUON), "-filter_complex", graph,
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-profile:v", "high",
           "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", "-t", f"{dur}", str(mp4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last = np.zeros((H, W, 3), np.uint8)
    failed, left_out = 0, 0
    with Pool(workers, initializer=_init, initargs=(FORMAT,)) as pool:
        for i, (img, err, cut) in enumerate(pool.imap(_render, list(enumerate(jobs)), chunksize=2)):
            if err:
                failed += 1
                print(f"ERROR frame {i}: {jobs[i]}\n{err}", flush=True)
                img = last
            last = img
            left_out += len(cut)
            proc.stdin.write(img.tobytes())
            if i % 60 == 0:
                print(f"  frame {i}/{n}", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"{failed} failed frames, {left_out} texts left out at the edges -> {mp4}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sheet"
    args = sys.argv[2:]
    use(next((a for a in args if a in FORMATS), "instagram"))
    args = [a for a in args if a not in FORMATS]
    if mode == "sheet":
        sheet(20, args)
    else:
        video(20, args[0] if args else f"muonbloom_teaser_{'' if FORMAT == 'instagram' else FORMAT + '_'}{W}x{H}.mp4")
