"""The test card of the show: one still in the delivery raster, to put on the wall before anything else.
It answers two questions on site: is the signal that reaches the projectors the one the engine sends, and
what do our lines, our type and our greys become on the brick?

  python tools/test_card.py                          the card for the raster of the projector study (3000 x 1688,
                                                     the 2978 x 1400 picture of the show at 11, 272)
  python tools/test_card.py --raster 3000x1688 --at 11,272

It writes  previews/testcard/muonbloom_testcard_<W>x<H>.png   (a still for anybody: the media server can load it)
           engine/out/testcard_<W>x<H>.bgra                   (the same, raw: what the engine shows as TEST CARD)

What is on it:
  the edges     a white line on the outermost pixels of the raster, a mark and its coordinates in every corner,
                pixel ticks along the top and the right; the ground line (the raster is a little taller than
                the wall: in the template of the projector study the 50 ft of the building are rows 21 to 1666,
                so the top and the bottom of the raster are not on the wall), metres along it from the left end
                of the wall, metres above it up the left, a circle that has to be round
  three rows    TOP (the upper projectors alone), MIDDLE (where upper and lower projectors blend), BOTTOM (the
                lower projectors alone). In every bay of every row, LINES + TYPE: our four line weights at three
                levels, red lines, our type sizes - drawn by the renderer of the show, with its glow
  gratings      one-pixel and two-pixel lines and a one-pixel checkerboard, exact: an even grey with no bands
                means the picture is carried pixel for pixel; bands mean it is scaled somewhere on the way
  4:4:4         the same in red and black, red and blue, green and magenta: on a 4:2:2 or 4:2:0 path the red
                lines become a dull field (thin red lines of the show would soften the same way)
  levels        steps 0 .. 255; 2 4 8 12 16 24 on black and 223 .. 251 on white (they merge when the range is
                clipped to 16 - 235 on the way); ramps; the levels of the show as its renderer gives them
  what is ours  the outline of the picture of the show in the raster, the towers of data/towers.json, and the
                blend zones as the projector study predicts them (dashed: a prediction, to compare with the wall)

The gratings, the steps and the ramps are written pixel by pixel; the rest of the picture area is drawn with
engine.Frame like any scene, so it has the bloom and the tone curve of the show.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muonbloom import hud  # noqa: E402
from muonbloom import layout as L  # noqa: E402
from muonbloom.engine import Frame, font  # noqa: E402

WALL_W_M, WALL_H_M = 91 * 0.3048, 50 * 0.3048       # the building of the projector study: the raster covers its width
# the projectors of that study (4Wall, v5): two towers 12.80 m from the wall; on each one a level projector, its
# lens 3.632 m above the ground, and one tilted up by 20 degrees, at 5.512 m; throw ratio 0.84, 1920 x 1200
THROW, RATIO, TOWERS_M, LENS_LOW, LENS_UP, TILT = 12.80, 0.84, (7.007, 20.653), 3.632, 5.512, math.radians(20.0)
ROWS = (("TOP", 170.0), ("MIDDLE", 498.0), ("BOTTOM", 850.0))   # y of the three rows of samples (picture)
PATCH_W, PATCH_H = 330.0, 285.0
STRIP_Y = 1165.0                        # the last strip: ramps, the levels of the show


def face_rows(W, H):
    """Rows of the raster the wall face of the study starts and ends on (its top, the ground): the template
    4Wall delivers (3000 x 1688) has it on rows 21 to 1666; another raster: centred, to the same scale."""
    if (W, H) == (3000, 1688):
        return 21, 1667
    face = WALL_H_M * W / WALL_W_M
    return int(round(0.5 * (H - face))), int(round(0.5 * (H + face)))


def blends():
    """Where two projectors of the study light the same brick, in metres: the band between the foot of the upper
    images and the top of the lower ones (heights), the stretch of wall the two lower images share (along the
    wall), and for the upper ones, which widen with the height, that stretch at a few heights (h, x0, x1)."""
    hu, hv = 0.5 / RATIO, 0.5 / RATIO * 1200 / 1920

    def up(v):                              # a row of an upper image: the height it lands at, half its width there
        s = THROW / (math.cos(TILT) - v * math.sin(TILT))
        return LENS_UP + s * (math.sin(TILT) + v * math.cos(TILT)), hu * s

    seam_up = [(h, TOWERS_M[1] - half, TOWERS_M[0] + half) for h, half in (up(v) for v in np.linspace(-hv, hv, 9))]
    return (up(-hv)[0], LENS_LOW + THROW * hv), (TOWERS_M[1] - hu * THROW, TOWERS_M[0] + hu * THROW), seam_up


def patch(f, x, y, name):
    """LINES + TYPE: what the show is made of, as its renderer draws it. Four line weights (rows) at three
    levels (columns), each horizontal, vertical and diagonal; red lines; the type sizes."""
    f.tag("w", x, y + 18, f"LINES + TYPE // {name}", size=L.T_MICRO, pad=3)
    levels = (1.0, 0.6, 0.42)
    for c, lv in enumerate(levels):
        f.text("w", x + 40 + c * 96, y + 40, f"LEVEL {lv:g}", size=12, alpha=0.6)
    for r, w in enumerate((L.LW_HAIR, L.LW, L.LW_BOLD, L.LW_FRAME)):
        yy = y + 58 + r * 22
        f.text("w", x, yy + 5, f"{w:.1f}", size=12, alpha=0.6)
        for c, lv in enumerate(levels):
            xx = x + 40 + c * 96
            f.segments("w", [xx, xx + 58, xx + 66], [yy, yy - 9, yy + 9], [xx + 48, xx + 58, xx + 84], [yy, yy + 9, yy - 9], lv, width=w)
    yy = y + 58 + 4 * 22
    f.text("r", x, yy + 5, "RED", size=12, alpha=0.9)
    for c, w in enumerate((L.LW_HAIR, L.LW, L.LW_BOLD)):
        xx = x + 40 + c * 96
        f.segments("r", [xx, xx + 58, xx + 66], [yy, yy - 9, yy + 9], [xx + 48, xx + 58, xx + 84], [yy, yy + 9, yy - 9], 1.0, width=w)
    ty = y + 178
    for size, s in ((L.T_MICRO, "14 MUON BLOOM 0123456789 µ"), (L.T_SMALL, "17 MUON BLOOM 0123456789"), (L.T_TAG, "22 MUON BLOOM 0123 µ")):
        f.text("w", x, ty, s, size=size, alpha=0.95)
        ty += 6 + 1.25 * size
    f.text("w", x, ty - 4, "14 AT LEVEL 0.6 // 105.658 MEV", size=L.T_MICRO, alpha=0.6)
    f.text("r", x, ty + 20, "17 RED // 2.197 µS", size=L.T_SMALL, alpha=1.0)


def show_levels(f, x, y):
    """The levels the show draws with, as filled blocks: what is left of each one on the wall?"""
    f.tag("w", x, y + 18, "LEVELS OF THE SHOW", size=L.T_MICRO, pad=3)
    for k, lv in enumerate((1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.42, 0.3, 0.2)):
        f.rects("w", x + k * 56, y + 34, x + k * 56 + 46, y + 74, lv)
        f.text("w", x + k * 56, y + 92, f"{lv:g}", size=12, alpha=0.7)
    for k, lv in enumerate((1.0, 0.6, 0.42)):
        f.rects("r", x + k * 56, y + 104, x + k * 56 + 46, y + 130, lv)
    f.text("r", x + 3 * 56, y + 122, "RED 1 / 0.6 / 0.42", size=12, alpha=0.9)


def picture(towers):
    """The part of the card the renderer of the show draws: 2978 x 1400, as any frame of the show."""
    f = Frame(L.W, L.H)
    cols = L.columns(towers)
    for name, y in ROWS:
        for k, (x0, x1) in enumerate(cols):
            if x1 - x0 >= PATCH_W:
                patch(f, x0, y, f"{name} {k + 1}")
    wide = [c for c in cols if c[1] - c[0] >= PATCH_W + 30.0 + 524.0]
    if len(wide) > 1:
        show_levels(f, wide[1][0], STRIP_Y)
    return hud.finish(f), cols, wide


def card(W, H, at, towers):
    """The whole raster: (H, W, 3) uint8, and the text that goes on it (drawn last, with PIL)."""
    px_m = W / WALL_W_M
    top, ground = face_rows(W, H)
    ox, oy = at
    im = np.zeros((H, W, 3), np.uint8)
    look, cols, wide = picture(towers)
    im[oy: oy + L.H, ox: ox + L.W] = look[: max(0, min(L.H, H - oy)), : max(0, min(L.W, W - ox))]
    text = []                               # (x, y, string, size, colour, anchor, bold)

    def say(x, y, s, size=14, col=(200, 200, 200), anchor="ls", bold=False):
        text.append((float(x), float(y), s, int(size), col, anchor, bold))

    def box(x0, y0, x1, y1, col):
        im[max(int(y0), 0): max(int(y1), 0), max(int(x0), 0): max(int(x1), 0)] = col

    def dashed(x0, y0, x1, y1, col=90, dash=12):
        n = int(max(abs(x1 - x0), abs(y1 - y0)))
        k = np.arange(n + 1)
        on = (k // dash) % 2 == 0
        xs = np.clip(np.round(x0 + (x1 - x0) * k / max(n, 1)).astype(int), 0, W - 1)[on]
        ys = np.clip(np.round(y0 + (y1 - y0) * k / max(n, 1)).astype(int), 0, H - 1)[on]
        im[ys, xs] = col

    # ---- what is ours in the raster: the circle, the picture, the towers, the blends (under everything else)
    cx, cy, R = W // 2, H // 2, min(640, H // 2 - 60)
    yy, xx = np.ogrid[:H, :W]
    d = np.hypot(xx - (cx - 0.5), yy - (cy - 0.5))
    ring = np.abs(d - R) < 1.0
    ring[oy: oy + L.H, ox: ox + L.W] &= look[: H - oy, : W - ox].max(-1) < 8        # the circle passes behind what is drawn
    im[ring] = 110
    box(cx - 60, cy - 1, cx + 60, cy + 1, 150)
    box(cx - 1, cy - 60, cx + 1, cy + 60, 150)
    for a, b in ((ox, ox + L.W - 1), ):                                             # the picture of the show
        im[oy, a: b + 1] = im[oy + L.H - 1, a: b + 1] = 120
        im[oy: oy + L.H, a] = im[oy: oy + L.H, b] = 120
    say(ox + 8, oy - 8, f"PICTURE OF THE SHOW {L.W} x {L.H} AT {ox}, {oy}", 14, (150, 150, 150))
    for key in L.ORDER:
        t = towers[key]
        x0, x1, y0, y1 = int(t.x0) + ox, int(t.x1) + ox, int(t.top) + oy, min(int(t.bot) + oy, H - 1)
        im[y0, x0: x1 + 1] = im[y1, x0: x1 + 1] = 150
        im[y0: y1 + 1, x0] = im[y0: y1 + 1, x1] = 150
        im[int(t.top + t.det_h) + oy, x0: x1 + 1] = 150
        say(0.5 * (x0 + x1), y0 - 10, L.NAMES[key], 14, (150, 150, 150), "ms")
    band, seam_low, seam_up = blends()
    Y = lambda h: ground - h * px_m          # a height above the ground -> a row of the raster
    for h in band:
        dashed(ox, Y(h), ox + L.W - 1, Y(h))
    for x in seam_low:
        dashed(x * px_m, Y(band[1]), x * px_m, ground)
    for (ha, a0, a1), (hb, b0, b1) in zip(seam_up[:-1], seam_up[1:]):
        if Y(ha) > oy:                      # (not above the picture of the show)
            dashed(a0 * px_m, Y(ha), b0 * px_m, max(Y(hb), oy))
            dashed(a1 * px_m, Y(ha), b1 * px_m, max(Y(hb), oy))
    say(ox + L.W - 70, Y(band[1]) - 8, "BLEND OF THE UPPER AND LOWER PROJECTORS, AS PREDICTED", 12, (120, 120, 120), "rs")
    say(ox + L.W - 70, Y(band[0]) - 8, f"... DOWN TO HERE ({band[0]:.1f} TO {band[1]:.1f} M ABOVE THE GROUND)", 12, (120, 120, 120), "rs")

    # ---- the blocks written pixel by pixel, right of the samples of the two wide bays
    def gratings(x, y, colour):
        s = 116
        names = ("1 PX LINES", "1 PX LINES", "1 PX CHECKER", "2 PX LINES") if not colour else (
            "RED / BLACK", "RED / BLACK", "RED / BLUE", "GREEN / MAGENTA")
        a = np.array((255, 0, 0) if colour else (255, 255, 255), np.uint8)
        for k in range(4):
            x0 = int(x) + k * (s + 20)
            blk = np.zeros((s, s, 3), np.uint8)
            jj, ii = np.mgrid[:s, :s]
            if k == 0:
                blk[ii % 2 == 0] = a
            elif k == 1:
                blk[jj % 2 == 0] = a
            elif not colour:
                blk[((ii + jj) % 2 == 0) if k == 2 else ((ii // 2) % 2 == 0)] = a
            else:
                blk[:] = (0, 0, 255) if k == 2 else (255, 0, 255)
                blk[ii % 2 == 0] = a if k == 2 else (0, 255, 0)
            im[int(y): int(y) + s, x0: x0 + s] = blk
            say(x0, y + s + 16, names[k], 12, (170, 170, 170))
        say(x, y - 10, "4:4:4 // THE RED LINES STAY RED LINES" if colour else "PIXEL FOR PIXEL // AN EVEN GREY, NO BANDS", 14,
            (255, 255, 255))

    def levels(x, y):
        x = int(x)
        say(x, y - 10, "FULL RANGE // EVERY STEP CAN BE TOLD FROM THE NEXT", 14, (255, 255, 255))
        for k, v in enumerate(list(range(0, 256, 16)) + [255]):
            box(x + k * 30, y, x + k * 30 + 30, y + 50, v)
            if k % 2 == 0:
                say(x + k * 30 + 2, y + 66, str(v), 12, (170, 170, 170))
        box(x - 1, y - 1, x + 511, y, 90)
        box(x - 1, y + 50, x + 511, y + 51, 90)
        y2 = y + 96
        box(x, y2, x + 250, y2 + 50, 0)
        im[int(y2) - 1, x: x + 250] = im[int(y2) + 50, x: x + 250] = 90
        for k, v in enumerate((2, 4, 8, 12, 16, 24)):
            box(x + 8 + k * 40, y2 + 6, x + 38 + k * 40, y2 + 44, v)
            say(x + 8 + k * 40, y2 + 66, str(v), 12, (170, 170, 170))
        say(x, y2 + 84, "ON BLACK", 12, (120, 120, 120))
        box(x + 262, y2, x + 512, y2 + 50, 255)
        for k, v in enumerate((251, 247, 243, 235, 223)):
            box(x + 270 + k * 48, y2 + 6, x + 306 + k * 48, y2 + 44, v)
            say(x + 270 + k * 48, y2 + 66, str(v), 12, (170, 170, 170))
        say(x + 262, y2 + 84, "ON WHITE", 12, (120, 120, 120))

    def ramps(x, y):
        x = int(x)
        say(x, y + 18, "RAMPS 0 - 255, TWO PIXELS A LEVEL // SMOOTH, NO STEPS LARGER THAN THE OTHERS", 14, (255, 255, 255))
        g = np.repeat(np.arange(256, dtype=np.uint8), 2)
        im[int(y) + 34: int(y) + 64, x: x + 512] = g[None, :, None]
        im[int(y) + 72: int(y) + 102, x: x + 512, 0] = g[None, :]
        for k, c in enumerate(((255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 0, 0), (0, 0, 255))):
            box(x + k * 73, y + 110, x + k * 73 + 73, y + 140, c)

    if wide:
        for n, (_, y) in enumerate(ROWS):
            gratings(ox + wide[0][0] + PATCH_W + 30, oy + y + 60, False)
            if len(wide) > 1:
                if n < 2:
                    gratings(ox + wide[1][0] + PATCH_W + 30, oy + y + 60, True)
                else:
                    levels(ox + wide[1][0] + PATCH_W + 30, oy + y + 30)
        ramps(ox + wide[0][0], oy + STRIP_Y)
        # the title and what to look at, at the head of the two wide bays (the top row is the top of the wall:
        # nothing above the picture of the show is sure to be on it)
        x = ox + wide[0][0]
        say(x, oy + 84, "MUON : BLOOM // TEST CARD", 44, (255, 255, 255), bold=True)
        say(x, oy + 118, f"RASTER {W} x {H} // PICTURE {L.W} x {L.H} AT {ox}, {oy} // 60 HZ", 17, (220, 220, 220))
        say(x, oy + 142, f"RGB 4:4:4, FULL RANGE 0 - 255 // ONE PIXEL = {1000.0 / px_m:.2f} MM ON THE WALL", 17, (220, 220, 220))
        say(x, oy + 166, time.strftime("CARD MADE %Y-%m-%d %H:%M // TOWERS OF data/towers.json"), 14, (150, 150, 150))
        if len(wide) > 1:
            x = ox + wide[1][0]
            for k, s in enumerate(("1  EDGES: THE WHITE LINE ENDS WHERE THE WALL ENDS, LEFT AND RIGHT; THE GROUND LINE IS ON",
                                   "   THE GROUND; THE CIRCLE IS ROUND; THE METRE MARKS ARE METRES",
                                   "2  PIXEL FOR PIXEL: THE GRATINGS ARE AN EVEN GREY (BANDS = SCALED SOMEWHERE)",
                                   "3  4:4:4: THE RED GRATINGS ARE RED LINES, NOT A DULL FIELD",
                                   "4  FULL RANGE: 4 8 12 16 SEPARATE ON BLACK, 235 TO 251 ON WHITE",
                                   "5  LINES + TYPE: COMPARE TOP, MIDDLE (THE BLEND) AND BOTTOM, BAY BY BAY")):
                say(x, oy + 62 + k * 21, s, 14, (220, 220, 220))

    # ---- the edges of the raster: on top of everything
    im[0, :] = im[H - 1, :] = 255
    im[:, 0] = im[:, W - 1] = 255
    for (x, y, sx, sy, an) in ((0, 0, 1, 1, "la"), (W - 1, 0, -1, 1, "ra"), (0, H - 1, 1, -1, "ld"), (W - 1, H - 1, -1, -1, "rd")):
        xa, xb = sorted((x, x + sx * 70))
        ya, yb = sorted((y, y + sy * 70))
        box(xa, y if sy > 0 else y - 2, xb + 1, (y + 3) if sy > 0 else y + 1, 255)
        box(x if sx > 0 else x - 2, ya, (x + 3) if sx > 0 else x + 1, yb + 1, 255)
        say(x + sx * 14, y + sy * 12, f"{x}, {y}", 17, (255, 255, 255), an)
    for v in range(100, W, 100):                                                    # pixels along the top
        box(v, 1, v + 1, 14 if v % 500 else 26, 200)
        if v % 500 == 0 and 200 < v < W - 200:
            say(v + 5, 26, str(v), 12, (170, 170, 170), "la")
    for v in range(100, H, 100):                                                    # ... and down the right
        box(W - (14 if v % 500 else 26), v, W - 1, v + 1, 200)
        if v % 500 == 0:
            say(W - 30, v - 3, str(v), 12, (170, 170, 170), "rs")
    # the ground, as the template of the study has it: metres along it, and above it up the left edge
    im[ground, 1: W - 1] = np.where((np.arange(1, W - 1) // 12) % 2 == 0, 200, im[ground, 1: W - 1, 0])[:, None]
    say(120, ground - 22, f"GROUND (ROW {ground} OF THE RASTER, AS IN THE TEMPLATE OF THE PROJECTOR STUDY): WHAT IS UNDER THIS LINE IS NOT ON THE WALL",
        12, (170, 170, 170))
    for m in range(1, int(WALL_W_M) + 1):
        x = int(round(m * px_m))
        if 90 < x < W - 90:
            box(x, ground - (20 if m % 5 == 0 else 10), x + 1, ground, 200)
            if m % 5 == 0:
                say(x + 5, ground - 8, f"{m} M", 12, (170, 170, 170))
    for m in range(1, int(ground / px_m) + 1):
        y = int(round(ground - m * px_m))
        if 60 < y:
            box(1, y, 26 if m % 5 == 0 else 14, y + 1, 200)
            if m % 5 == 0:
                say(30, y + 5, f"{m} M", 12, (170, 170, 170))
    box(1, top, 40, top + 1, 200)
    say(96, top + 5, "50 FT ABOVE THE GROUND: THE TOP OF THE BUILDING IN THE TEMPLATE", 12, (170, 170, 170))

    img = Image.fromarray(im)
    d = ImageDraw.Draw(img)
    for x, y, s, size, col, anchor, bold in text:
        d.text((x, y), s, font=font(size, bold), fill=col, anchor=anchor)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raster", default="3000x1688", help="size of the delivery raster, WxH")
    ap.add_argument("--at", default="11,272", help="where the picture of the show sits in it, X,Y")
    ap.add_argument("--raw", default=None, help="the raw BGRA file (default: engine/out/testcard_<W>x<H>.bgra)")
    ap.add_argument("--png", default=None, help="the PNG (default: previews/testcard/muonbloom_testcard_<W>x<H>.png)")
    a = ap.parse_args()
    W, H = (int(v) for v in a.raster.lower().split("x"))
    at = tuple(int(v) for v in a.at.split(","))
    if at[0] < 0 or at[1] < 0 or at[0] + L.W > W or at[1] + L.H > H:
        sys.exit(f"the picture of the show ({L.W} x {L.H}) does not fit in a {W} x {H} raster at {at[0]}, {at[1]}")
    im = card(W, H, at, L.load_towers())
    png = Path(a.png) if a.png else ROOT / "previews" / "testcard" / f"muonbloom_testcard_{W}x{H}.png"
    raw = Path(a.raw) if a.raw else ROOT / "engine" / "out" / f"testcard_{W}x{H}.bgra"
    for p in (png, raw):
        p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(im).save(png)
    tmp = raw.with_name(raw.name + ".tmp")          # written aside, then moved: the engine never reads half a card
    bgra = np.dstack([im[..., ::-1], np.full(im.shape[:2], 255, np.uint8)])
    tmp.write_bytes(bgra.tobytes())
    tmp.replace(raw)
    print(png)
    print(raw, f"{W} x {H} BGRA, {raw.stat().st_size} bytes")


if __name__ == "__main__":
    main()
