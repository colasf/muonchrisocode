"""Scene worker of the realtime engine: runs the Python scenes and hands the draw lists to muonengine.exe.

Started by the engine, several at a time (the scenes are pure functions of the show time, so the frames of
the next few sixtieths of a second are computed in parallel, a few frames ahead of the clock):

    python engine/worker.py --shm NAME --size BYTES --bufs N [--det NAME --det-mode live|both]

The draw lists are written in a block of shared memory (N buffers of BYTES each); the orders come on stdin
and the answers go to stdout, one line each:

    engine -> worker    F <job> <time> <buffer>     record the frame at this show time into this buffer
                        W <time>                    warm up: build the scene of this time (nothing is kept)
                        Q                           quit
    worker -> engine    H <start>:<end>:<look> ...      ready (the modules are imported); the looks of the show
                        D <job> <bytes> <ms>        done: the blob is in the buffer
                        E <job> <message>           the scene raised: the engine keeps its last good frame
                        W <ms>                      warmed up
                        W <ms> E <message>          the scene of that time cannot be built (it raised)

Anything a scene prints goes to stderr (the console of the engine).
--det: the detector streams come from the engine (OSC) instead of the muon stem, see engine/detectors.py.
"""
from __future__ import annotations

import argparse
import gc
import mmap
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _oneline(e):
    return f"{type(e).__name__}: {' '.join(str(e).split())[:300]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shm", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--bufs", type=int, default=2)
    ap.add_argument("--det", default="")
    ap.add_argument("--det-mode", default="live", choices=("live", "both"))
    a = ap.parse_args()

    out = sys.stdout.buffer
    sys.stdout = sys.stderr                         # prints of the scenes must not break the protocol

    def send(msg):
        out.write(msg.encode("ascii", "replace") + b"\n")
        out.flush()

    shm = mmap.mmap(-1, a.size * a.bufs, tagname=a.shm)
    mv = memoryview(shm)
    from muonbloom import drawlist as D
    from muonbloom import engine as E
    from muonbloom import showdata as sd
    from muonbloom.show import Show
    fpath = getattr(E.font(22), "path", None)
    if not (isinstance(fpath, str) and "SpaceMono" in fpath):       # PIL's built-in font: not the look of the show
        print("WARNING: Space Mono was not found (muonbloom/engine.py: _FONT_DIRS): the text uses PIL's default font",
              file=sys.stderr, flush=True)
    ctx = sd.Context()
    live = None
    if a.det:
        sys.path.insert(0, str(ROOT / "engine"))
        from detectors import LiveDetectors
        live = ctx.det = LiveDetectors(ctx.cues, a.det, a.det_mode)
        live.poll()                                 # what the detectors said before this worker was started
    show = Show(ctx)
    known = {}                                      # the glyphs the renderer already has from this worker
    gc.collect()
    gc.freeze()                                     # the modules and the data of the show are there for good
    looks = []
    for code, name, t0, t1, look in sd.SECTIONS:
        if looks and looks[-1][2] == look:
            looks[-1][1] = t1
        else:
            looks.append([t0, t1, look])
    send("H " + " ".join(f"{t0}:{t1}:{look}" for t0, t1, look in looks))
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        cmd = line.split()
        if not cmd:
            continue
        if cmd[0] == "F":
            job, t, buf = cmd[1], float(cmd[2]), int(cmd[3])
            try:
                c = time.perf_counter()
                if live is not None:
                    live.poll()
                dl = D.record(show, t)
                n = dl.pack(known, t=t, frame=int(job) & 0xFFFFFFFF, out=mv[buf * a.size: (buf + 1) * a.size])
                send(f"D {job} {n} {(time.perf_counter() - c) * 1e3:.2f}")
            except Exception as e:
                known = {}                          # a glyph of this frame may not have left: send them all again
                traceback.print_exc()
                send(f"E {job} {_oneline(e)}")
        elif cmd[0] == "W":
            c = time.perf_counter()
            failed = ""
            try:
                D.record(show, float(cmd[1]))
                gc.collect()
                gc.freeze()
            except Exception as e:
                traceback.print_exc()
                failed = f" E {_oneline(e)}"
            send(f"W {(time.perf_counter() - c) * 1e3:.1f}{failed}")
        elif cmd[0] == "Q":
            break


if __name__ == "__main__":
    main()
