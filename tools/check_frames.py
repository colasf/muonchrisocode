"""Sanity checks on the scenes, at low resolution.

  python tools/check_frames.py                 every look: is a frame identical when rendered by two
                                               different processes (needed for multi-process video and
                                               for the port), and how much does it move in 1/30 s?
  python tools/check_frames.py --sweep 10      render one frame every 10 s over the whole show and report
                                               errors (exceptions) only
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCALE = 0.25


def _one(job):
    t, = job
    from muonbloom import layout as L
    from muonbloom.show import render_frame
    W, H = int(L.W * SCALE) // 2 * 2, int(L.H * SCALE) // 2 * 2
    try:
        t0 = time.time()
        img = render_frame(t, W, H)
        return t, img, time.time() - t0, None
    except Exception:
        return t, None, 0.0, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    from muonbloom import showdata as sd
    if a.sweep > 0:
        times = list(np.arange(0.5, sd.LOOP_END - 0.5, a.sweep))
        bad = 0
        with Pool(a.workers) as pool:
            for t, img, dt, err in pool.imap(_one, [(float(t),) for t in times], chunksize=1):
                if err:
                    bad += 1
                    _, sec, _ = sd.section_at(t)
                    print(f"ERROR at {sd.tc(t)} ({sec[0]} {sec[4]}):\n{err}", flush=True)
        print(f"sweep: {len(times)} frames, {bad} errors")
        return
    # one probe per run of sections sharing a look, a third of the way in, plus t + 1/30 s
    probes = []
    seen = set()
    for code, name, t0, t1, look in sd.SECTIONS:
        a0, a1 = sd.look_span(look, 0.5 * (t0 + t1))
        if (look, a0) in seen:
            continue
        seen.add((look, a0))
        probes.append((look, a0 + (a1 - a0) / 3.0))
    jobs = []
    for look, t in probes:
        jobs += [(t,), (t,), (t + 1 / 30.0,)]
    with Pool(a.workers, maxtasksperchild=1) as pool:           # a fresh process for every frame
        res = pool.map(_one, jobs, chunksize=1)
    print(f"{'look':14s} {'time':>10s}  identical  move/frame  sec/frame")
    for k, (look, t) in enumerate(probes):
        r1, r2, r3 = res[3 * k: 3 * k + 3]
        if r1[3] or r2[3] or r3[3]:
            print(f"{look:14s} {sd.tc(t)}  ERROR\n{r1[3] or r2[3] or r3[3]}")
            continue
        same = bool(np.array_equal(r1[1], r2[1]))
        diff = float(np.abs(r1[1].astype(np.int16) - r2[1].astype(np.int16)).mean())
        move = float(np.abs(r1[1].astype(np.int16) - r3[1].astype(np.int16)).mean())
        print(f"{look:14s} {sd.tc(t)}  {'yes' if same else f'NO ({diff:.2f})':9s}  {move:10.2f}  {r1[2]:9.2f}")


if __name__ == "__main__":
    main()
