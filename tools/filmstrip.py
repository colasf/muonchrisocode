"""Film strip of a piece of the wall: consecutive frames of one region, tiled on a sheet - to check an
animation (a build-up, a transition) frame by frame without playing a video.

  python tools/filmstrip.py 11.9 13.4 --crop 620,1180,2340,1354             every frame (30 fps) of that region
  python tools/filmstrip.py 3.1 4.1 --crop 28,46,900,240 --step 2 --cols 3  one frame out of two, three columns
  python tools/filmstrip.py 10.5 12.0                                       the whole wall (small)

Times are show times (M:SS[.cc] or seconds); --crop is x0,y0,x1,y1 in wall pixels (2978 x 1400).
The sheet goes to previews/strips and its path is printed.
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muonbloom import layout as L  # noqa: E402
from muonbloom.engine import font  # noqa: E402

_show = None


def _init():
    global _show
    from muonbloom.show import Show
    _show = Show()


def _frame(job):
    t, W, H, box = job
    img = Image.fromarray(_show.render(t, W, H))
    return t, (img.crop(box) if box else img)


def parse_time(s):
    s = str(s)
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("t0")
    ap.add_argument("t1")
    ap.add_argument("--crop", default=None, help="x0,y0,x1,y1 in wall pixels")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--step", type=int, default=1, help="keep one frame out of N")
    ap.add_argument("--scale", type=float, default=None, help="render scale (default: fits the sheet width)")
    ap.add_argument("--cols", type=int, default=1)
    ap.add_argument("--width", type=int, default=1900, help="width of the sheet in pixels")
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 4)))
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    t0, t1 = parse_time(a.t0), parse_time(a.t1)
    crop = [float(v) for v in a.crop.split(",")] if a.crop else [0.0, 0.0, float(L.W), float(L.H)]
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    tile_w = (a.width - 8 * (a.cols + 1)) // a.cols
    scale = a.scale or min(1.0, tile_w / cw)
    W, H = int(L.W * scale) // 2 * 2, int(L.H * scale) // 2 * 2
    s = W / L.W
    box = tuple(int(round(v * s)) for v in crop) if a.crop else None
    n = int(round((t1 - t0) * a.fps))
    times = [t0 + k / a.fps for k in range(0, n + 1, a.step)]
    with Pool(min(a.workers, len(times)), initializer=_init) as pool:
        frames = pool.map(_frame, [(t, W, H, box) for t in times])
    tw = min(tile_w, frames[0][1].size[0])
    th = int(round(frames[0][1].size[1] * tw / frames[0][1].size[0]))
    lab, pad = 20, 8
    rows = (len(frames) + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (a.cols * tw + (a.cols + 1) * pad, rows * (th + lab) + pad), (40, 40, 40))
    dr = ImageDraw.Draw(sheet)
    f1 = font(15, True)
    for k, (t, im) in enumerate(frames):                    # column-major: time runs down, then to the next column
        c, r = k // rows, k % rows
        x, y = pad + c * (tw + pad), pad + r * (th + lab)
        sheet.paste(im.resize((tw, th), Image.LANCZOS) if im.size[0] != tw else im, (x, y + lab - 2))
        dr.text((x + 2, y), f"{int(t // 60):02d}:{t % 60:06.3f}", font=f1, fill=(255, 214, 0))
    d = ROOT / "previews" / "strips"
    d.mkdir(parents=True, exist_ok=True)
    name = a.name or f"strip_{t0:07.3f}_{t1:07.3f}" + (f"_{int(crop[0])}_{int(crop[1])}" if a.crop else "")
    p = d / f"{name}.png"
    sheet.save(p)
    print(p, sheet.size, f"{len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()
