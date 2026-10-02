"""Copy stills to JPEG in a scratch folder for a quick look (keeps the PNGs free of viewer locks).

  python tools/peek.py previews/stills/02m09*.png            -> prints the JPEG paths
  python tools/peek.py --crop 600,200,1500,900 file.png      -> a 1:1 crop (x0,y0,x1,y1 in wall pixels)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--crop", default=None)
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--out", default=os.environ.get("PEEK_DIR", str(Path(tempfile.gettempdir()) / "muonbloom_peek")))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [f for pat in a.files for f in sorted(glob.glob(pat))] or a.files
    for f in files:
        im = Image.open(f).convert("RGB")
        tag = ""
        if a.crop:
            x0, y0, x1, y1 = (int(v) for v in a.crop.split(","))
            im = im.crop((x0, y0, x1, y1))
            tag = f"_crop{x0}_{y0}"
        elif im.size[0] > a.width:
            im = im.resize((a.width, int(im.size[1] * a.width / im.size[0])), Image.LANCZOS)
        q = out / (Path(f).stem + tag + ".jpg")
        im.save(q, quality=90)
        print(q)


if __name__ == "__main__":
    sys.exit(main())
