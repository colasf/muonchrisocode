"""Send one OSC message to the engine (transport orders, a detector value).

  python engine/tools/osc_send.py /muonbloom/start 1         play from the beginning (on the passage from 0 to 1)
  python engine/tools/osc_send.py /muonbloom/play
  python engine/tools/osc_send.py /muonbloom/pause
  python engine/tools/osc_send.py /muonbloom/seek 180        show time in seconds
  python engine/tools/osc_send.py /muonbloom/reload          start the scene workers again
  python engine/tools/osc_send.py /muon/C 0.8                a detector value (engine started with --detectors live)
"""
from __future__ import annotations

import argparse
import socket
import struct


def message(addr, values):
    def pad(b):
        b += b"\0"
        return b + b"\0" * (-len(b) % 4)
    return pad(addr.encode()) + pad(("," + "f" * len(values)).encode()) + b"".join(struct.pack(">f", v) for v in values)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("address")
    ap.add_argument("values", nargs="*", type=float)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    a = ap.parse_args()
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(message(a.address, a.values), (a.host, a.port))


if __name__ == "__main__":
    main()
