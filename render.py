"""Render the Muon Bloom FX scenes.

Examples
  python render.py chamber --still 4 9.5 15          # full-res PNG stills
  python render.py chamber --sheet                   # contact sheet across the loop
  python render.py chamber --scale 0.5 --video       # half-res H.264 preview
  python render.py all --video --hap                 # 3000x1688 H.264 + HAP Q (.mov) for MadMapper
  python render.py finale --sheet                    # scenes 9-11: rise, disintegrate, outro

Scenes loop seamlessly: the last frame flows into the first one. Except the finale
(rise, disintegrate, outro): linear, the length of their sections on the sheet.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import importlib
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "renders"
SCENES = {
    "chamber": ("muonfx.chamber", "Chamber"),
    "shower": ("muonfx.shower", "Shower"),
    "outlast": ("muonfx.outlast", "Outlast"),
    "you": ("muonfx.you", "You"),
    "sphere": ("muonfx.sphere", "Sphere"),
    "rise": ("muonfx.rise", "Rise"),
    "disintegrate": ("muonfx.disintegrate", "Disintegrate"),
    "outro": ("muonfx.outro", "Outro"),
    "chamber_v1": ("muonfx.chamber_v1", "ChamberV1"),
    "shower_v1": ("muonfx.shower_v1", "ShowerV1"),
}
DESIGN = (3000, 1688)
TD_FFMPEG = Path(r"C:\Program Files\Derivative\TouchDesigner.2025.33230\bin\ffmpeg.exe")


def make_scene(name):
    mod, cls = SCENES[name]
    return getattr(importlib.import_module(mod), cls)()


def _even(v):
    return int(v) // 4 * 4


_scene = None
_size = None


def _init(name, size):
    global _scene, _size
    sys.path.insert(0, str(ROOT))
    _scene = make_scene(name)
    _size = size


def _render(t):
    return _scene.render(t, *_size).tobytes()


def x264_ffmpeg():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def render_video(name, scale, fps, workers, hap, duration=None):
    W, H = _even(DESIGN[0] * scale), _even(DESIGN[1] * scale)
    scene = make_scene(name)
    T = duration or scene.T
    n = int(round(T * fps))
    times = [i / fps for i in range(n)]
    OUT.mkdir(exist_ok=True)
    tag = f"{name}_{W}x{H}_{fps}fps"
    raw = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    procs = []
    mp4 = OUT / f"{tag}.mp4"
    procs.append(subprocess.Popen(
        [x264_ffmpeg(), "-y", "-loglevel", "error", *raw, "-c:v", "libx264", "-preset", "slow", "-crf", "15",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE))
    outs = [mp4]
    if hap:
        mov = OUT / f"{tag}_hapq.mov"
        exe = str(TD_FFMPEG) if TD_FFMPEG.exists() else x264_ffmpeg()
        procs.append(subprocess.Popen(
            [exe, "-y", "-loglevel", "error", *raw, "-c:v", "hap", "-format", "hap_q", "-chunks", "8", str(mov)],
            stdin=subprocess.PIPE))
        outs.append(mov)
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(name, (W, H))) as pool:
        for i, buf in enumerate(pool.imap(_render, times, chunksize=2)):
            for p in procs:
                p.stdin.write(buf)
            if i % max(1, n // 20) == 0:
                el = time.time() - t0
                print(f"  {name}: frame {i + 1}/{n}  {el:6.1f}s", flush=True)
    for p in procs:
        p.stdin.close()
        p.wait()
    print(f"{name}: {n} frames in {time.time() - t0:.1f}s -> " + ", ".join(str(o) for o in outs), flush=True)


def render_stills(name, ts, scale):
    W, H = _even(DESIGN[0] * scale), _even(DESIGN[1] * scale)
    scene = make_scene(name)
    d = OUT / "stills"
    d.mkdir(parents=True, exist_ok=True)
    for t in ts:
        t0 = time.time()
        img = scene.render(t, W, H)
        p = d / f"{name}_{W}x{H}_t{t:05.2f}.png"
        Image.fromarray(img).save(p)
        print(f"{p}  ({time.time() - t0:.2f}s)", flush=True)


def render_sheet(name, count, scale):
    W, H = _even(DESIGN[0] * scale), _even(DESIGN[1] * scale)
    scene = make_scene(name)
    cols = 2
    rows = (count + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W + (cols - 1) * 6, rows * H + (rows - 1) * 6), (60, 60, 60))
    for k in range(count):
        t = scene.T * (k + 0.5) / count
        img = Image.fromarray(scene.render(t, W, H))
        sheet.paste(img, ((k % cols) * (W + 6), (k // cols) * (H + 6)))
    d = OUT / "stills"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}_sheet.jpg"
    sheet.save(p, quality=90)
    print(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", choices=[*SCENES, "all", "new", "finale"])
    ap.add_argument("--still", type=float, nargs="*", help="times (s) to render as PNG")
    ap.add_argument("--sheet", action="store_true", help="contact sheet of 8 moments across the loop")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--hap", action="store_true", help="also write a HAP Q .mov (MadMapper / TouchDesigner)")
    ap.add_argument("--scale", type=float, default=1.0, help="1.0 = 3000x1688, 0.5 = half res preview")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=None, help="override the loop length (s)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    a = ap.parse_args()
    names = {"all": ["chamber", "shower", "outlast", "you"], "new": ["outlast", "you"],
             "finale": ["rise", "disintegrate", "outro"]}.get(a.scene, [a.scene])
    for name in names:
        if a.still:
            render_stills(name, a.still, a.scale)
        if a.sheet:
            render_sheet(name, 8, a.scale if a.scale != 1.0 else 0.33)
        if a.video:
            render_video(name, a.scale, a.fps, a.workers, a.hap, a.duration)


if __name__ == "__main__":
    main()
