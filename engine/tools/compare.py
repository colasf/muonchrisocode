"""The engine's picture against the Python reference, frame by frame.

  python engine/tools/compare.py 0:20 2:29 5:53                 these show times, full size
  python engine/tools/compare.py --probes                       one frame per look (a third of the way in) + the glitch
  python engine/tools/compare.py --sweep 20 --scale 0.5         one frame every 20 s over the show, half size
  python engine/tools/compare.py 3:18 --save                    also write engine / reference / difference PNGs
  python engine/tools/compare.py 3:18 --repeat 50               and time the engine on each frame (ms per frame)

For every time: the frame is recorded (muonbloom/drawlist.py), drawn by muonengine.exe from the draw list,
and rasterised by the reference code from the same draw list (drawlist.replay, which check_drawlist.py
checks against the previews). Both pictures are compared in floats, before the dither.

The reference is replayed with its lines sampled four times finer than in the previews (replay(fine=True)):
the engine integrates a line exactly, so that is the picture it has to give, and the comparison can be
tight. A frame FAILS, and the exit code is 1, when

    mean        the mean difference is over MEAN_MAX levels (0..255, over the three colour channels), or
    > 2         more than PCT2_MAX per cent of the pixels are further than 2 levels from the reference, or
    > 8         more than PCT8_MAX per cent are further than 8, or
    notes       the engine drew it without something it asked for (MISSING GLYPHS, FLAGS).

Files go to engine/out/compare.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXE = ROOT / "engine" / "build" / "muonengine.exe"
OUT = ROOT / "engine" / "out" / "compare"
MEAN_MAX, PCT2_MAX, PCT8_MAX = 0.02, 0.05, 0.005


def parse_time(s):
    s = str(s)
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


def probe_times():
    from muonbloom import showdata as sd
    times, seen = [], set()
    for code, name, t0, t1, look in sd.SECTIONS:
        a0, a1 = sd.look_span(look, 0.5 * (t0 + t1))
        if (look, a0) not in seen:
            seen.add((look, a0))
            times.append(a0 + (a1 - a0) / 3.0)
    # + the streak field (lines whose splats show), the collapse (discs too large for the table), the glitch
    # (post-process) and the credits
    return times + [25.0, 30.0, 41.0, 418.0, 430.3, 449.0, 830.6]


_show = None


def _record(job):
    """Record one frame and rasterise it with the reference code (worker process)."""
    global _show
    k, t, W, H = job
    from muonbloom import drawlist as D
    from muonbloom.show import Show
    if _show is None:
        _show = Show()
    blob = bytes(D.record(_show, t, W, H).pack(t=t, frame=k))
    c = time.perf_counter()
    ref = D.replay(blob, as_float=True, fine=True)
    np.save(OUT / f"ref{k:04d}.npy", ref)
    return k, blob, time.perf_counter() - c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("times", nargs="*")
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--sweep", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save", action="store_true", help="write PNGs: engine, reference, difference x 8")
    ap.add_argument("--repeat", type=int, default=1, help="draw each frame this many times and time it")
    a = ap.parse_args()
    from muonbloom import layout as L
    from muonbloom import showdata as sd
    W, H = int(L.W * a.scale) // 2 * 2, int(L.H * a.scale) // 2 * 2
    times = [parse_time(v) for v in a.times]
    if a.probes:
        times += probe_times()
    if a.sweep > 0:
        times += [float(t) for t in np.arange(0.5, sd.TRACK_END - 0.5, a.sweep)]
    if not times:
        ap.error("no time given")
    if not EXE.exists():
        sys.exit(f"{EXE} is missing: run engine\\build.bat")
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.rgbaf"):
        f.unlink()

    jobs = [(k, t, W, H) for k, t in enumerate(times)]
    blobs = [None] * len(jobs)
    with Pool(min(a.workers, len(jobs))) as pool:
        for k, blob, dt in pool.imap_unordered(_record, jobs, chunksize=1):
            blobs[k] = blob
    path = OUT / "frames.mbdl"
    with open(path, "wb") as fh:
        for b in blobs:
            fh.write(b)
    res = subprocess.run([str(EXE), "render", str(path), str(OUT / "eng"), "--float", "--repeat", str(a.repeat)],
                         capture_output=True, text=True)
    if res.returncode:
        sys.exit(f"muonengine failed:\n{res.stdout}\n{res.stderr}")
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("frame ")]
    print(res.stdout.splitlines()[0])
    print(f"{len(times)} frames at {W} x {H}")
    print(f"{'time':>10} {'look':<13} {'max':>6} {'mean':>7} {'> 2':>7} {'> 8':>7}  {'ms':>6}  notes")
    worst, bad = 0.0, 0
    for k, t in enumerate(times):
        eng = np.fromfile(OUT / f"eng{k:04d}.rgbaf", np.float32).reshape(H, W, 4)[..., :3]
        ref = np.load(OUT / f"ref{k:04d}.npy")
        d = np.abs(eng - ref) * 255.0
        dm = d.max(-1)
        _, sec, _ = sd.section_at(t)
        ms = float(lines[k].split(" ms")[0].split()[-1])
        note = " ".join(w for w in ("MISSING GLYPHS", "FLAGS") if w in lines[k])
        worst = max(worst, float(d.mean()))
        p2, p8 = 100 * (dm > 2).mean(), 100 * (dm > 8).mean()
        if note or d.mean() > MEAN_MAX or p2 > PCT2_MAX or p8 > PCT8_MAX:
            bad += 1
            note += "   <-- FAILS"
        print(f"{sd.tc(t)} {sec[4]:<13} {d.max():>6.1f} {d.mean():>7.3f} {p2:>6.3f}% {p8:>6.3f}%  {ms:>6.2f}  {note}", flush=True)
        if a.save:
            from PIL import Image
            name = f"{int(t // 60):02d}m{t % 60:05.2f}s_{sec[4]}"
            Image.fromarray((np.clip(eng, 0, 1) * 255 + 0.5).astype(np.uint8)).save(OUT / f"{name}_engine.png")
            Image.fromarray((np.clip(ref, 0, 1) * 255 + 0.5).astype(np.uint8)).save(OUT / f"{name}_reference.png")
            Image.fromarray(np.clip(d * 8, 0, 255).astype(np.uint8)).save(OUT / f"{name}_diff_x8.png")
    print(f"worst mean difference: {worst:.3f} levels; {len(times)} frames, {bad} FAIL" + ("" if bad else ": the engine draws the reference"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
