"""Does the raster leave the engine pixel for pixel?

  python engine/tools/check_output.py                    the frames of the last compare.py run (engine/out/compare)
  python engine/tools/check_output.py 2:58 7:20.1        record these show times first

What `muonengine live` sends to the display of its OUTPUT panel is composed by Renderer::output: the delivery
raster (3000 x 1688) at the top left of the display, black but for the picture of the show at 11, 272 of it -
or the whole test card. `muonengine render --display WxH` runs that same code on a target of the size of a
display and writes what it got; this checks, byte for byte:

  a 3840 x 2160 display       the picture is in its place untouched, every other pixel is black
  ... the raster elsewhere    the same with --raster-at
  ... with the test card      the raster is the card (tools/test_card.py), untouched
  a 1920 x 1200 display       smaller than the raster: scaled to fit, and said so (not pixel for pixel)

Exit code 1 if anything differs.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXE = ROOT / "engine" / "build" / "muonengine.exe"
OUT = ROOT / "engine" / "out" / "compare"
PW, PH = 2978, 1400                     # the picture of the show
RW, RH, PX, PY = 3000, 1688, 11, 272    # the raster, and the picture in it


def render(*args):
    res = subprocess.run([str(EXE), "render", str(OUT / "frames.mbdl"), str(OUT / "out"), *args], capture_output=True, text=True)
    if res.returncode:
        sys.exit(f"muonengine failed:\n{res.stdout}\n{res.stderr}")
    return res.stdout


def main():
    if not EXE.exists():
        sys.exit(f"{EXE} is missing: run engine\\build.bat")
    if sys.argv[1:]:
        from muonbloom import drawlist as D
        from muonbloom.show import Show
        from compare import parse_time
        OUT.mkdir(parents=True, exist_ok=True)
        show = Show()
        with open(OUT / "frames.mbdl", "wb") as fh:
            for k, t in enumerate(parse_time(v) for v in sys.argv[1:]):
                fh.write(bytes(D.record(show, t, PW, PH).pack(t=t, frame=k)))
    elif not (OUT / "frames.mbdl").exists():
        sys.exit("no frames: give show times, or run engine/tools/compare.py first")
    card = ROOT / "engine" / "out" / f"testcard_{RW}x{RH}.bgra"
    if not card.exists():
        subprocess.run([sys.executable, str(ROOT / "tools" / "test_card.py")], check=True, capture_output=True)
    cardpx = np.fromfile(card, np.uint8).reshape(RH, RW, 4)
    bad = 0

    def frames(suffix, w, h):
        files = sorted(OUT.glob(f"out*.{suffix}"))
        return [np.fromfile(f, np.uint8).reshape(h, w, 4) for f in files]

    for name, args, (ox, oy) in (("3840 x 2160, the raster at 0, 0", ["--display", "3840x2160"], (0, 0)),
                                 ("3840 x 2160, the raster at 420, 236", ["--display", "3840x2160", "--raster-at", "420,236"], (420, 236))):
        said = render(*args)
        pics, disp = frames("rgba", PW, PH), frames("display", 3840, 2160)
        ok = "pixel for pixel" in said and len(pics) == len(disp) > 0
        for p, d in zip(pics, disp):
            want = np.zeros_like(d)
            want[..., 3] = 255
            want[oy + PY: oy + PY + PH, ox + PX: ox + PX + PW] = p
            ok &= bool(np.array_equal(d[..., :3], want[..., :3]))
        bad += not ok
        print(f"{name}: {len(disp)} frames, {'the picture is in its place untouched, black around it' if ok else 'DIFFERENT   <-- CHECK'}")

    said = render("--display", "3840x2160", "--card-file", str(card))
    disp = frames("display", 3840, 2160)
    ok = "pixel for pixel" in said and len(disp) > 0
    for d in disp:
        want = np.zeros_like(d)
        want[:RH, :RW] = cardpx
        ok &= bool(np.array_equal(d[..., :3], want[..., :3]))
    bad += not ok
    print(f"3840 x 2160 with the test card: {len(disp)} frames, {'the raster is the card, untouched' if ok else 'DIFFERENT   <-- CHECK'}")

    said = render("--display", "1920x1200")
    disp = frames("display", 1920, 1200)
    k = min(1920 / RW, 1200 / RH)
    y0 = int(0.5 * (1200 - RH * k) + PY * k)
    ok = "scaled to fit" in said and len(disp) > 0 and all(d[: y0 - 2, :, :3].max() == 0 for d in disp)
    lit = [float((d[..., :3].max(-1) > 16).mean()) for d in disp]
    ok &= any(v > 0.0 for v in lit)                 # (a frame of the show may be all black: not all of them)
    bad += not ok
    print(f"1920 x 1200: {len(disp)} frames, {'scaled to fit, and said so' if ok else 'NOT AS EXPECTED   <-- CHECK'} "
          f"({100 * np.mean(lit):.1f} % of the display lit)")
    for f in list(OUT.glob("out*.rgba")) + list(OUT.glob("out*.display")):
        f.unlink()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
