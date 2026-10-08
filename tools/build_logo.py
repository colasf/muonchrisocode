"""The Tyrell logo of the standby (scene 13.3 VISUALS BY, muonbloom/scenes/standby.py): bake its outlines.

  python tools/build_logo.py [logo.eps]         writes data/tyrell_logo.npz

The Illustrator EPS is read by hand (no Ghostscript on the machine): nine closed outlines made of `mo / li / cv /
cp`, filled even-odd. The file keeps them as points (the curves cut in 12 chords), in the units of the EPS (points,
y down, 385.4 x 120.4): x, y = the points of all the outlines one after the other, n = how many each one has.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGO = Path(r"D:\muonchristo\logo\Tyrell_Logo_DEF.eps")


def outlines(path):
    b = Path(path).read_bytes()
    o, n = struct.unpack("<2I", b[4:12])
    t = b[o:o + n].decode("latin1")
    page = t[t.index("clp", t.index("%%EndPageSetup")) + 3:t.index("%ADOBeginClientInjection: EndPageContent")]
    paths, cur, st = [], [], []
    for tok in page.split():
        try:
            st.append(float(tok))
            continue
        except ValueError:
            pass
        if tok in ("mo", "li"):
            cur.append((st[-2], st[-1]))
        elif tok == "cv":
            p0, (x1, y1, x2, y2, x3, y3) = cur[-1], st[-6:]
            for u in np.linspace(0, 1, 13)[1:]:
                a, b_, c, e = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u * u * (1 - u), u ** 3
                cur.append((a * p0[0] + b_ * x1 + c * x2 + e * x3, a * p0[1] + b_ * y1 + c * y2 + e * y3))
        elif tok == "cp":
            paths.append(cur)
            cur = []
        st = []
    return paths


def main():
    paths = outlines(sys.argv[1] if len(sys.argv) > 1 else LOGO)
    pts = np.array([p for q in paths for p in q], np.float64)
    out = ROOT / "data" / "tyrell_logo.npz"
    np.savez_compressed(out, x=pts[:, 0], y=pts[:, 1], n=np.array([len(q) for q in paths]))
    print(f"{out}: {len(paths)} outlines, {len(pts)} points, {pts[:, 0].min():.1f} .. {pts[:, 0].max():.1f} x "
          f"{pts[:, 1].min():.1f} .. {pts[:, 1].max():.1f}")


if __name__ == "__main__":
    main()
