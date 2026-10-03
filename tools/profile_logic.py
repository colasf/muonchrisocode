"""How long does the SCENE LOGIC take per frame, without any rasterisation?

  python tools/profile_logic.py              the whole show, one frame every 0.25 s
  python tools/profile_logic.py --step 0.1   finer
  python tools/profile_logic.py --look dance glitch      only these looks (names of showdata.SECTIONS)

The frame used here only counts what the scenes ask it to draw (no splats, no PIL, no bloom): what is left
is the Python / numpy time a scene needs to decide what to draw - the part a GPU renderer does not replace -
and the size of the draw list a renderer has to swallow. It hooks the raw primitives of engine.Frame, i.e.
exactly the calls a draw-list recorder has to capture (see engine/BRIEF.md).

Run it on the show machine: single-thread speed is what matters.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muonbloom import engine, hud  # noqa: E402
from muonbloom import layout as L  # noqa: E402
from muonbloom import show as show_mod  # noqa: E402
from muonbloom import showdata as sd  # noqa: E402
from muonbloom.engine import text_w  # noqa: E402

COUNT = {}


class NullFrame(engine.Frame):
    """engine.Frame without a pixel buffer: every raw primitive only counts its items."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.s = W / engine.DESIGN_W
        self.acc = {}
        self._pend = {k: [] for k in engine.LAYERS}
        self._diff, self._txt, self._txt_draw, self._occl = {}, {}, {}, []
        self.post, self.invert_rects, self._bld = [], [], None
        self.set_view()
        self.set_clip()

    def _n(self, key, n):
        COUNT[key] = COUNT.get(key, 0) + int(n)

    def flush(self):
        pass

    def points(self, layer, x, y, w):
        self._n("points", np.size(x))

    def scale_rect(self, *a, **k):
        self._n("occlude", 1)

    def occlude(self, *a, **k):
        self._n("occlude", 1)

    def dim(self, *a, **k):
        self._n("occlude", 1)

    def _segments(self, layer, x0, y0, x1, y1, i0, i1=None, width=1.0, spacing=0.5):
        self._n("segments", max(np.size(x0), np.size(x1), np.size(y0), np.size(y1)))

    def _dots(self, layer, x, y, r, i):
        self._n("dots", np.size(x))

    def pixels(self, layer, x, y, i, snap=False):
        self._n("pixels", np.size(x))

    def _rects(self, layer, x0, y0, x1, y1, i):
        self._n("rects", max(np.size(x0), np.size(x1), np.size(y0), np.size(y1)))

    def _text(self, layer, x, y, s, size=22, alpha=1.0, anchor="ls", bold=False):
        self._n("glyphs", len(s))

    def _tag(self, layer, x, y, s, size=18, alpha=1.0, anchor="ls", pad=5, bold=False, ref=None, wipe=1.0):
        r = ref or s
        if not r:
            return None
        self._n("glyphs", len(s))
        self._n("rects", 1)
        w = text_w(r, size)
        xl = x - (0.0 if anchor[0] == "l" else w if anchor[0] == "r" else 0.5 * w)
        return (xl - pad, y - size - pad, xl + w * min(wipe, 1.0) + pad, y + 0.3 * size + pad)

    def text_vertical(self, layer, x, y, s, size=22, alpha=1.0):
        self._n("glyphs", len(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=float, default=0.25, help="seconds between the frames that are timed")
    ap.add_argument("--look", nargs="*", default=None, help="only these looks")
    a = ap.parse_args()
    show_mod.Frame = NullFrame
    hud.finish = lambda f, **kw: None
    sh = show_mod.Show()
    looks = []
    for code, name, t0, t1, look in sd.SECTIONS:
        if not looks or looks[-1][0] != look:
            looks.append([look, t0, t1])
        else:
            looks[-1][2] = t1
    if a.look:
        looks = [k for k in looks if k[0] in a.look]
    print(f"{'look':<13} {'frames':>6} {'median':>8} {'p95':>8} {'max':>8}   {'segments':>9} {'dots':>8} {'pixels':>8} "
          f"{'rects':>7} {'glyphs':>7}   (ms per frame; items = largest frame of the scene)")
    allms = []
    for look, t0, t1 in looks:
        sh.render(t0 + 0.5, L.W, L.H)                     # warm up: build the scene object and its caches
        sh.render(t0 + 1.0, L.W, L.H)
        ms, peak = [], {}
        for t in np.arange(t0 + 0.017, t1, a.step):
            COUNT.clear()
            c = time.perf_counter()
            sh.render(float(t), L.W, L.H)
            ms.append((time.perf_counter() - c) * 1000.0)
            for k, v in COUNT.items():
                peak[k] = max(peak.get(k, 0), v)
        ms = np.array(ms)
        allms.append(ms)
        print(f"{look:<13} {len(ms):>6} {np.median(ms):>8.1f} {np.percentile(ms, 95):>8.1f} {ms.max():>8.1f}   "
              f"{peak.get('segments', 0):>9} {peak.get('dots', 0):>8} {peak.get('pixels', 0):>8} "
              f"{peak.get('rects', 0):>7} {peak.get('glyphs', 0):>7}")
    allms = np.concatenate(allms)
    print(f"\nwhole show: median {np.median(allms):.1f} ms, p95 {np.percentile(allms, 95):.1f} ms, max {allms.max():.1f} ms; "
          f"frames over 16.7 ms: {100 * (allms > 16.7).mean():.0f} %, over 33.3 ms: {100 * (allms > 33.3).mean():.0f} %")


if __name__ == "__main__":
    main()
