"""The QR codes of the standby (scene 13.0 FOLLOW, muonbloom/scenes/standby.py): make their modules.

  python tools/build_qr.py                      the two addresses of the show: prints the rows to paste in standby.QR
  python tools/build_qr.py https://...          the rows of another address
  python tools/build_qr.py --check              read the codes of standby.py back and say what they hold

The show itself needs nothing but numpy and Pillow: the modules of the codes are written in standby.py. This
tool is only run when an address changes, and needs two libraries the show does not use:

  pip install segno zxing-cpp                   (segno makes a code, zxing-cpp reads it back: two different hands)

Version and correction: the smallest version that holds the address with correction M (for the two Instagram
addresses: version 3, 29 x 29 modules). On the wall model (tools/wall_proof.py) the larger modules of that
code read better than a denser code with more correction, from 5 to 60 lux of stray light.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

URLS = ("https://www.instagram.com/tyrell.studio/", "https://www.instagram.com/christosquier/")


def need(name, pip):
    try:
        return __import__(name)
    except ImportError:
        sys.exit(f"{name} is missing: pip install {pip}")


def read(rows, quiet=4, px=8):
    """What a reader finds in a code given as rows of 0 / 1 (1 = dark)."""
    zx = need("zxingcpp", "zxing-cpp")
    m = np.pad(np.array([[c == "1" for c in r] for r in rows], np.uint8), quiet)
    img = (np.kron(1 - m, np.ones((px, px), np.uint8)) * 255).astype(np.uint8)
    return [r.text for r in zx.read_barcodes(img)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*", help="addresses (default: the two of the show)")
    ap.add_argument("--error", default="m", choices=("l", "m", "q", "h"), help="error correction (default m)")
    ap.add_argument("--check", action="store_true", help="read back the codes written in standby.py")
    a = ap.parse_args()
    if a.check:
        import importlib
        from muonbloom import showdata  # noqa: F401  (standby reads the timeline when it is imported)
        st = importlib.import_module("muonbloom.scenes.standby")
        bad = 0
        for q in st.QR:
            got = read(q["rows"], st.QUIET)
            ok = got == [q["url"]]
            bad += not ok
            print(f"{q['name']:16s} {len(q['rows'])} x {len(q['rows'][0])} modules   {'OK ' if ok else 'WRONG'}  {got}")
        sys.exit(1 if bad else 0)
    segno = need("segno", "segno")
    for url in a.urls or URLS:
        qr = segno.make(url, error=a.error, micro=False, boost_error=False)
        rows = ["".join(str(int(v)) for v in r) for r in qr.matrix]
        if read(rows) != [url]:
            sys.exit(f"the code made for {url} does not read back")
        print(f"# {url}   version {qr.designator}, {len(rows)} x {len(rows)} modules, read back")
        print("    rows=(")
        for k in range(0, len(rows), 3):
            print("        " + ", ".join(f'"{r}"' for r in rows[k: k + 3]) + ("," if k + 3 < len(rows) else ")),"))


if __name__ == "__main__":
    main()
