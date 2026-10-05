"""What the ENGINE shows at a show time, as a picture to look at - to check a change of a scene.

  python tools/engine_frame.py 00:15:58                     the frame at that time code (MM:SS:FF, FF = frames)
  python tools/engine_frame.py 0:21 0:23.2 27               several times (M:SS[.cc] or seconds work too)
  python tools/engine_frame.py 00:15:58 --crop 989,255,2389,1055      also that part of the wall at full size
  python tools/engine_frame.py 00:15:58 --lift 0            without the lift of the OUTPUT panel

The frames are recorded from the scenes as they are on disk (muonbloom/drawlist.py) and drawn by
engine/build/muonengine.exe itself (`render`): the picture of the engine window, not the Python preview. The
lift is the one of the OUTPUT panel (engine/output.json), so the levels are those the user sees.
It can run while the engine plays (it does not touch it). The paths of the pictures are printed:
engine/out/look/MM-SS-FF_scene.jpg (the whole wall, 2000 px wide) and ..._crop.png (the crop, 1 : 1).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXE = ROOT / "engine" / "build" / "muonengine.exe"
OUT = ROOT / "engine" / "out" / "look"
FPS = 60.0


def parse_time(s):
    """MM:SS:FF (frames), M:SS[.cc] or seconds."""
    p = str(s).split(":")
    if len(p) == 3:
        return int(p[0]) * 60 + int(p[1]) + int(p[2]) / FPS
    if len(p) == 2:
        return int(p[0]) * 60 + float(p[1])
    return float(s)


def timecode(t):
    f = int(t * FPS + 1e-6)
    return f"{f // 60 // 60:02d}-{f // 60 % 60:02d}-{f % 60:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("times", nargs="+")
    ap.add_argument("--crop", default=None, help="x0,y0,x1,y1 in wall pixels (2978 x 1400): also written at full size")
    ap.add_argument("--lift", type=float, default=None, help="lift of the output (default: the one of engine/output.json)")
    a = ap.parse_args()
    if not EXE.exists():
        sys.exit(f"{EXE} is missing: run engine\\build.bat")
    lift = a.lift
    if lift is None:
        try:
            lift = float(json.loads((ROOT / "engine" / "output.json").read_text(encoding="utf-8")).get("lift", 0.0))
        except (OSError, ValueError):
            lift = 0.0
    from muonbloom import drawlist as D
    from muonbloom import layout as L
    from muonbloom import showdata as sd
    from muonbloom.show import Show
    times = [parse_time(x) for x in a.times]
    show = Show()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("eng*.rgba"):
        f.unlink()
    path = OUT / "frames.mbdl"
    D.write_frames(path, [bytes(D.record(show, t).pack(t=t, frame=k)) for k, t in enumerate(times)])
    res = subprocess.run([str(EXE), "render", str(path), str(OUT / "eng"), "--lift", f"{lift:g}"], capture_output=True, text=True)
    if res.returncode:
        sys.exit(f"muonengine failed:\n{res.stdout}\n{res.stderr}")
    print(f"drawn by the engine, lift {lift:g}:")
    for k, t in enumerate(times):
        raw = OUT / f"eng{k:04d}.rgba"
        img = Image.fromarray(np.fromfile(raw, np.uint8).reshape(L.H, L.W, 4)[..., 2::-1])       # (the file is B, G, R, A)
        raw.unlink()
        name = f"{timecode(t)}_{sd.section_at(t)[1][4]}"
        whole = OUT / f"{name}.jpg"
        img.resize((2000, round(2000 * L.H / L.W)), Image.LANCZOS).save(whole, quality=92)
        print(f"  {whole.relative_to(ROOT).as_posix()}")
        if a.crop:
            x0, y0, x1, y1 = (int(float(v)) for v in a.crop.split(","))
            part = OUT / f"{name}_crop.png"
            img.crop((x0, y0, x1, y1)).save(part)
            print(f"  {part.relative_to(ROOT).as_posix()}   (wall pixels {x0},{y0} - {x1},{y1}, 1 : 1)")
    path.unlink()


if __name__ == "__main__":
    main()
