"""Is a recorded frame the frame?

  python engine/tools/check_drawlist.py                 one probe per look (a third of the way in), half size
  python engine/tools/check_drawlist.py --sweep 7       one frame every 7 s over the whole show
  python engine/tools/check_drawlist.py --times 6:58 7:10 --scale 1
  python engine/tools/check_drawlist.py --text          only the text model against PIL (fast)

For every time: the reference picture (engine.Frame, as the previews) against the same frame recorded
(drawlist.DrawList), packed, read back and rasterised from the blob with the reference code
(drawlist.replay). The geometry goes through the same rasteriser, so it must be identical; the text is set
from single glyphs instead of by PIL and the glitch parameters travel as float32: a few pixels may differ by
one or two levels (more than 2, or a mean above 0.01, is reported).
On the glitch frames the operations of drawlist.post_ops are also checked against glitch.make_post itself.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_time(s):
    s = str(s)
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


_show = None


def _one(job):
    global _show
    t, W, H = job
    from muonbloom import drawlist as D
    from muonbloom.show import Show
    try:
        if _show is None:
            _show = Show()
        ref = _show.render(t, W, H)
        c = time.perf_counter()
        dl = D.record(_show, t, W, H)
        blob = dl.pack()
        rec_ms = (time.perf_counter() - c) * 1e3
        b = D.Blob(blob)
        img = D.replay(b)
        d = np.abs(ref.astype(np.int16) - img.astype(np.int16))
        post = ""
        if dl.post:                                 # the operations against the closure they were read from
            rng = np.random.default_rng(1)
            base = (rng.random((2, H, W), dtype=np.float32) ** 6 * 3.0).astype(np.float32)
            a = base.copy()
            for fn in dl.post:
                a = fn(a, dl)
            ops = []
            for fn in dl.post:
                o = D.post_ops(fn, dl)
                if o is None:
                    post = "UNKNOWN POST"
                    break
                ops += o
            else:
                bb = D.apply_post(base.copy(), ops)
                post = f"post {len(ops)} ops " + ("identical" if np.array_equal(a, bb) else f"DIFFERENT (max {np.abs(a - bb).max():.3g})")
        return t, int(d.max()), float(d.mean()), int((d.max(-1) > 1).sum()), rec_ms, b.counts(), b.flags, post, None
    except Exception:
        return t, 0, 0.0, 0, 0.0, {}, 0, "", traceback.format_exc()


def check_text():
    """A string set by PIL against the same string set glyph by glyph (drawlist.Font), and the boxes."""
    from PIL import Image, ImageDraw
    from muonbloom import drawlist as D
    rng = np.random.default_rng(3)
    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789   .,:;/+-=<>#%*()[]_|'\"!?&@")
    chars += ["µ", "−", "→"]
    worst, nbox, npos = 0, 0, 0
    for px in (6, 10, 14, 17, 20, 22, 28, 38, 60, 86):
        for bold in (False, True):
            fnt = D.get_font(px, bold)
            for trial in range(40):
                s = "".join(rng.choice(chars, int(rng.integers(1, 30))))
                anchor = ("ls", "ms", "rs", "lm", "rm", "la", "md", "lt", "rb")[trial % 9]
                X, Y = float(rng.uniform(300, 500)), float(rng.uniform(150, 250))
                im = Image.new("L", (2600, 400), 0)
                d = ImageDraw.Draw(im)
                d.text((X, Y), s, font=fnt.pil, fill=255, anchor=anchor)
                ref = np.asarray(im, np.float64)
                mine = np.zeros_like(ref)
                xi, yi = D.DrawList._origin(fnt, X, Y, s, anchor)
                x = xi
                for ch in s:
                    adv, ox, oy, w, h, bits = fnt.glyph(ch)[:6]
                    if w and h:
                        D._paint(mine, x + ox, yi + oy, np.frombuffer(bits, np.uint8).reshape(h, w), 1.0, 1.0)
                    x += adv
                e = np.abs(ref - np.round(mine)).max()
                if e > 16:
                    npos += 1
                worst = max(worst, e)
                l, t, r, b, adv = fnt.bbox(s)
                sh = 0 if anchor[0] == "l" else adv if anchor[0] == "r" else (adv + 1) // 2
                dy = fnt.dy(anchor[1], s)
                if (l - sh, t + dy, r - sh, b + dy) != tuple(fnt.pil.getbbox(s, anchor=anchor)):
                    nbox += 1
    print(f"text model: worst difference {worst:.0f} / 255 where glyphs touch, {npos} strings misplaced, {nbox} boxes wrong")
    return npos == 0 and nbox == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", type=float, default=0.0)
    ap.add_argument("--times", nargs="*", default=[])
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--text", action="store_true")
    a = ap.parse_args()
    ok = check_text()
    if a.text:
        sys.exit(0 if ok else 1)
    from muonbloom import layout as L
    from muonbloom import showdata as sd
    W, H = int(L.W * a.scale) // 2 * 2, int(L.H * a.scale) // 2 * 2
    if a.times:
        times = [parse_time(v) for v in a.times]
    elif a.sweep > 0:
        times = [float(t) for t in np.arange(0.5, sd.TRACK_END - 0.5, a.sweep)]
    else:
        times, seen = [], set()
        for code, name, t0, t1, look in sd.SECTIONS:
            a0, a1 = sd.look_span(look, 0.5 * (t0 + t1))
            if (look, a0) not in seen:
                seen.add((look, a0))
                times.append(a0 + (a1 - a0) / 3.0)
        times += [418.0, 430.3, 449.0]              # the glitch: break, drums (post-process), transition
    print(f"{len(times)} frames at {W} x {H}")
    print(f"{'time':>10} {'look':<13} {'max':>4} {'mean':>7} {'px>1':>6}  {'rec ms':>6}  {'segs':>7} {'dots':>6} {'splats':>7} "
          f"{'rects':>6} {'chars':>6} {'KB':>6}")
    bad = 0
    with Pool(min(a.workers, len(times))) as pool:
        for t, dmax, dmean, n1, ms, cnt, flags, post, err in pool.imap(_one, [(t, W, H) for t in times], chunksize=1):
            _, sec, _ = sd.section_at(t)
            if err:
                bad += 1
                print(f"{sd.tc(t)} {sec[4]:<13} ERROR\n{err}", flush=True)
                continue
            note = ("  " + post if post else "") + ("  FLAGS %d" % flags if flags else "")
            if dmax > 2 or dmean > 0.01 or flags or "DIFFERENT" in post or "UNKNOWN" in post:
                bad += 1
                note += "   <-- CHECK"
            if a.sweep <= 0 or note:
                print(f"{sd.tc(t)} {sec[4]:<13} {dmax:>4} {dmean:>7.4f} {n1:>6}  {ms:>6.1f}  {cnt['segments']:>7} {cnt['dots']:>6} "
                      f"{cnt['splats']:>7} {cnt['rects']:>6} {cnt['chars']:>6} {cnt['bytes'] // 1024:>6}{note}", flush=True)
    print(f"{len(times)} frames, {bad} to check" + ("" if ok else "; TEXT MODEL WRONG"))
    sys.exit(1 if bad or not ok else 0)


if __name__ == "__main__":
    main()
